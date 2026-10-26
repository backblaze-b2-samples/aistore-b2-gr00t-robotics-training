# SPDX-FileCopyrightText: Copyright (c) 2026 Backblaze, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Signals stop trainer workers before checkpoints are published; the newest is withheld."""

import json
import os
import signal
import sys
import time
from types import SimpleNamespace

import pytest
from test_artifacts import Store

from aistore_b2_gr00t import finetune
from aistore_b2_gr00t.artifacts import MANIFEST

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX training process groups")


def training_args(tmp_path):
    return SimpleNamespace(gr00t_dir=tmp_path, sync_interval=0.02)


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
def test_signal_stops_trainer_even_when_it_exits_zero(signum, tmp_path, monkeypatch):
    ready = tmp_path / "ready"
    stopped = tmp_path / "stopped"
    trainer = """
import pathlib, signal, sys, time
ready, stopped = map(pathlib.Path, sys.argv[1:])
def stop(signum, frame):
    stopped.write_text(str(signum))
    raise SystemExit(0)
signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
ready.write_text('ready')
while True:
    time.sleep(0.02)
"""
    calls = []
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}

    class Syncer:
        sent = False

        def sync(self, training_done):
            calls.append(training_done)
            if ready.exists() and not self.sent:
                self.sent = True
                os.kill(os.getpid(), signum)

        def sync_rest(self):
            pytest.fail("interrupted training must not publish run-level outputs")

    monkeypatch.setattr(finetune, "_SHUTDOWN_TIMEOUT_S", 0.5)
    cmd = [sys.executable, "-c", trainer, str(ready), str(stopped)]
    rc = finetune.run_training(cmd, training_args(tmp_path), os.environ.copy(), Syncer())

    assert rc == 128 + signum
    assert stopped.read_text() == str(int(signum))
    assert calls and not any(calls)
    assert {sig: signal.getsignal(sig) for sig in previous} == previous


@pytest.mark.parametrize("phase", ["periodic", "exited-during-periodic", "final"])
def test_interruption_during_upload_never_publishes_newest(phase, tmp_path, monkeypatch):
    for step in (100, 200):
        checkpoint = tmp_path / f"checkpoint-{step}"
        checkpoint.mkdir()
        (checkpoint / "model.bin").write_bytes(b"model")

    class InterruptingStore(Store):
        sent = False

        def upload_file(self, filename, bucket, key, **kwargs):
            target = "checkpoint-200" if phase == "final" else "checkpoint-100"
            if target in key and not self.sent:
                self.sent = True
                if phase == "exited-during-periodic":
                    time.sleep(0.1)
                os.kill(os.getpid(), signal.SIGTERM)
                time.sleep(0.1)
            super().upload_file(filename, bucket, key, **kwargs)

    store = InterruptingStore()
    monkeypatch.setattr(finetune, "ais_s3_client", lambda cfg: store)
    monkeypatch.setattr(finetune, "_SHUTDOWN_TIMEOUT_S", 0.3)
    syncer = finetune.CheckpointSync(SimpleNamespace(b2_bucket="b"), tmp_path, "run")
    trainer = {
        "periodic": "import time; time.sleep(10)",
        "exited-during-periodic": "import time; time.sleep(0.05)",
        "final": "pass",
    }[phase]
    rc = finetune.run_training(
        [sys.executable, "-c", trainer], training_args(tmp_path), os.environ.copy(), syncer
    )

    assert rc == 128 + signal.SIGTERM
    assert json.loads(store.data[f"checkpoints/run/checkpoint-100/{MANIFEST}"])["status"] == (
        "complete"
    )
    newest = store.data.get(f"checkpoints/run/checkpoint-200/{MANIFEST}")
    assert newest is None or json.loads(newest)["status"] == "pending"
    assert f"checkpoints/run/{MANIFEST}" not in store.data


def test_stubborn_process_group_stops_while_upload_threads_unwind(tmp_path, monkeypatch):
    ready, heartbeat = tmp_path / "ready", tmp_path / "heartbeat"
    worker = """
import pathlib, signal, sys, time
heartbeat = pathlib.Path(sys.argv[1])
signal.signal(signal.SIGTERM, signal.SIG_IGN)
signal.signal(signal.SIGINT, signal.SIG_IGN)
for step in range(10000):
    heartbeat.write_text(str(step))
    time.sleep(0.02)
"""
    trainer = """
import pathlib, signal, subprocess, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
signal.signal(signal.SIGINT, signal.SIG_IGN)
child = subprocess.Popen([sys.executable, '-c', sys.argv[1], sys.argv[2]])
while not pathlib.Path(sys.argv[2]).exists():
    time.sleep(0.01)
pathlib.Path(sys.argv[3]).write_text('ready')
child.wait()
"""
    for step in (100, 200):
        checkpoint = tmp_path / f"checkpoint-{step}"
        checkpoint.mkdir()
        (checkpoint / "model.bin").write_bytes(b"model")

    observed = []

    class SlowStore(Store):
        sent = False

        def upload_file(self, filename, bucket, key, **kwargs):
            if not self.sent:
                self.sent = True
                deadline = time.monotonic() + 5
                while not ready.exists():
                    if time.monotonic() > deadline:
                        raise TimeoutError("trainer did not start")
                    time.sleep(0.01)
                os.kill(os.getpid(), signal.SIGTERM)
                time.sleep(0.5)
                observed.append(heartbeat.read_text())
                time.sleep(0.1)
                observed.append(heartbeat.read_text())
            super().upload_file(filename, bucket, key, **kwargs)

    store = SlowStore()
    monkeypatch.setattr(finetune, "ais_s3_client", lambda cfg: store)
    monkeypatch.setattr(finetune, "_SHUTDOWN_TIMEOUT_S", 0.2)
    syncer = finetune.CheckpointSync(SimpleNamespace(b2_bucket="b"), tmp_path, "run")
    cmd = [sys.executable, "-c", trainer, worker, str(heartbeat), str(ready)]
    rc = finetune.run_training(cmd, training_args(tmp_path), os.environ.copy(), syncer)

    assert rc == 128 + signal.SIGTERM
    assert len(observed) == 2 and observed[0] == observed[1]
    final = heartbeat.read_text()
    time.sleep(0.1)
    assert heartbeat.read_text() == final
    assert "checkpoint-200" not in syncer.done


def test_clean_launcher_with_active_worker_is_not_a_clean_training_run(tmp_path, monkeypatch):
    for step in (100, 200):
        (tmp_path / f"checkpoint-{step}").mkdir()
    (tmp_path / "checkpoint-100" / "model.bin").write_bytes(b"complete")
    newest = tmp_path / "checkpoint-200" / "model.bin"
    worker = """
import pathlib, sys, time
model = pathlib.Path(sys.argv[1])
while True:
    model.write_bytes(b'unfinished')
    time.sleep(0.02)
"""
    launcher = """
import pathlib, subprocess, sys, time
subprocess.Popen([sys.executable, '-c', sys.argv[1], sys.argv[2]])
while not pathlib.Path(sys.argv[2]).exists():
    time.sleep(0.01)
"""
    store = Store()
    monkeypatch.setattr(finetune, "ais_s3_client", lambda cfg: store)
    monkeypatch.setattr(finetune, "_EXIT_GRACE_S", 0.1)
    monkeypatch.setattr(finetune, "_SHUTDOWN_TIMEOUT_S", 0.3)
    syncer = finetune.CheckpointSync(SimpleNamespace(b2_bucket="b"), tmp_path, "run")
    rc = finetune.run_training(
        [sys.executable, "-c", launcher, worker, str(newest)],
        training_args(tmp_path),
        os.environ.copy(),
        syncer,
    )

    assert rc == 1
    assert syncer.done == {"checkpoint-100"}
    assert f"checkpoints/run/checkpoint-200/{MANIFEST}" not in store.data
    assert f"checkpoints/run/{MANIFEST}" not in store.data


def test_helper_exiting_shortly_after_clean_launcher_publishes_final_checkpoint(
    tmp_path, monkeypatch
):
    # e.g. a wandb service that tears down a moment after torchrun returns 0.
    for step in (100, 200):
        (tmp_path / f"checkpoint-{step}").mkdir()
        (tmp_path / f"checkpoint-{step}" / "model.bin").write_bytes(b"complete")
    launcher = """
import subprocess, sys
subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(0.3)'])
"""
    store = Store()
    monkeypatch.setattr(finetune, "ais_s3_client", lambda cfg: store)
    monkeypatch.setattr(finetune, "_EXIT_GRACE_S", 5.0)
    monkeypatch.setattr(finetune, "_SHUTDOWN_TIMEOUT_S", 0.3)
    syncer = finetune.CheckpointSync(SimpleNamespace(b2_bucket="b"), tmp_path, "run")
    rc = finetune.run_training(
        [sys.executable, "-c", launcher], training_args(tmp_path), os.environ.copy(), syncer
    )

    assert rc == 0
    assert syncer.done == {"checkpoint-100", "checkpoint-200"}
    assert f"checkpoints/run/checkpoint-200/{MANIFEST}" in store.data


@pytest.mark.parametrize("failure", ["launch", "publication"])
def test_signal_handlers_restore_after_failures(failure, tmp_path):
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}

    class BrokenSync:
        def sync(self, training_done):
            raise ConnectionError("publication failed")

        def sync_rest(self):
            pytest.fail("failed checkpoint sync must prevent run-level publication")

    cmd = [str(tmp_path / "missing")] if failure == "launch" else [sys.executable, "-c", "pass"]
    args = training_args(tmp_path)
    args.sync_interval = 1
    expected = FileNotFoundError if failure == "launch" else ConnectionError
    with pytest.raises(expected):
        finetune.run_training(cmd, args, os.environ.copy(), BrokenSync())
    assert {sig: signal.getsignal(sig) for sig in previous} == previous


def test_shutdown_error_keeps_interruption_status_and_prevents_publication(tmp_path, monkeypatch):
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}

    class Trainer:
        pid = 123

        def poll(self):
            return None

        def wait(self, timeout):
            os.kill(os.getpid(), signal.SIGTERM)

    class Syncer:
        def sync(self, training_done):
            pytest.fail("a group that could not stop must not publish")

        def sync_rest(self):
            pytest.fail("a group that could not stop must not publish")

    def inaccessible_group(*args):
        raise PermissionError("cannot signal trainer")

    monkeypatch.setattr(finetune.subprocess, "Popen", lambda *a, **kw: Trainer())
    monkeypatch.setattr(finetune.os, "killpg", inaccessible_group)
    monkeypatch.setattr(
        finetune.threading,
        "Timer",
        lambda *a: SimpleNamespace(start=lambda: None, cancel=lambda: None),
    )
    monkeypatch.setattr(finetune, "_SHUTDOWN_TIMEOUT_S", 0)
    rc = finetune.run_training([], training_args(tmp_path), os.environ.copy(), Syncer())
    assert rc == 128 + signal.SIGTERM
    assert {sig: signal.getsignal(sig) for sig in previous} == previous


def test_unexpected_error_stops_the_trainer_group(tmp_path, monkeypatch):
    started = []

    def crash(proc, interval, syncer):
        started.append(proc)
        raise RuntimeError("unexpected")

    monkeypatch.setattr(finetune, "_wait_with_periodic_sync", crash)
    monkeypatch.setattr(finetune, "_EXIT_GRACE_S", 0.1)
    monkeypatch.setattr(finetune, "_SHUTDOWN_TIMEOUT_S", 0.5)
    cmd = [sys.executable, "-c", "import time; time.sleep(30)"]
    with pytest.raises(RuntimeError, match="unexpected"):
        finetune.run_training(cmd, training_args(tmp_path), os.environ.copy(), SimpleNamespace())
    assert started[0].poll() is not None
    with pytest.raises(ProcessLookupError):
        os.killpg(started[0].pid, 0)

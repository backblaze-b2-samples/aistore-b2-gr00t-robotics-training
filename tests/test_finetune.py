# SPDX-FileCopyrightText: Copyright (c) 2026 Backblaze, Inc.
# SPDX-License-Identifier: Apache-2.0
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from aistore_b2_gr00t import finetune
from aistore_b2_gr00t.artifacts import MANIFEST, read_manifest

STAGED = SimpleNamespace(objects=1, bytes=1, seconds=0, manifest_sha256="d" * 64)


def test_build_command_matches_upstream_entrypoint(tmp_path):
    args = finetune.parse_args(
        ["--gr00t-dir", str(tmp_path), "--stage-dir", str(tmp_path / "stage"), "--run-name", "r1"]
    )
    cmd = finetune.build_command(args)
    assert cmd[:4] == ["uv", "run", "--project", str(tmp_path.resolve())]
    assert cmd[4:7] == ["--locked", "python", "gr00t/experiment/launch_finetune.py"]
    flags = dict(zip(cmd[7::2], cmd[8::2], strict=True))
    assert flags["--max-steps"] == "2000"
    assert flags["--base-model-path"] == "nvidia/GR00T-N1.7-3B"
    assert flags["--dataset-path"] == str(finetune.dataset_dir(args))
    assert finetune.dataset_dir(args).name == "r1"
    assert flags["--embodiment-tag"] == "NEW_EMBODIMENT"
    assert flags["--modality-config-path"] == "examples/SO100/so100_config.py"
    assert flags["--output-dir"] == str(Path("outputs/r1").resolve())


def test_multi_gpu_uses_torchrun_and_passes_extra_args(tmp_path):
    args = finetune.parse_args(
        [
            "--gr00t-dir", str(tmp_path), "--num-gpus", "4", "--embodiment-tag", "UNITREE_G1",
            "--modality-config-path", "",
            "--dataset-prefix", "datasets/unitree_g1.LMPnPAppleToPlateDC/",
            "--", "--use-wandb", "--tune-visual",
        ]
    )  # fmt: skip
    cmd = finetune.build_command(args)
    assert cmd[4:8] == ["--locked", "torchrun", "--standalone", "--nproc_per_node=4"]
    assert not any(arg.startswith("--master_port") for arg in cmd)
    assert "--modality-config-path" not in cmd
    assert cmd[-2:] == ["--use-wandb", "--tune-visual"]
    assert cmd[cmd.index("--dataset-path") + 1] == str(finetune.dataset_dir(args))


@pytest.mark.parametrize(
    ("inherited", "expected"),
    [(None, "0"), ("2,3", "2"), (" 5 ", "5"), ("GPU-abc,GPU-def", "GPU-abc")],
)
def test_single_gpu_run_sees_exactly_one_granted_device(inherited, expected):
    environ = {"B2_KEY_ID": "k", "B2_APP_KEY": "s", "VIRTUAL_ENV": "/v", "PATH": "/bin"}
    if inherited is not None:
        environ["CUDA_VISIBLE_DEVICES"] = inherited
    env = finetune.training_env(environ, num_gpus=1)
    assert env == {"PATH": "/bin", "CUDA_VISIBLE_DEVICES": expected}


def test_multi_gpu_run_keeps_inherited_devices():
    env = finetune.training_env({"CUDA_VISIBLE_DEVICES": "2,3"}, num_gpus=2)
    assert env["CUDA_VISIBLE_DEVICES"] == "2,3"


def test_symlinked_stage_and_output_roots_resolve_to_local_disk(tmp_path):
    # Cloud GPU hosts often link data/ and outputs/ to local NVMe.
    nvme = tmp_path / "nvme"
    (nvme / "stage").mkdir(parents=True)
    (tmp_path / "stage").symlink_to(nvme / "stage", target_is_directory=True)
    (tmp_path / "out").symlink_to(nvme, target_is_directory=True)
    args = finetune.parse_args(
        ["--stage-dir", str(tmp_path / "stage"), "--output-dir", str(tmp_path / "out" / "r1")]
    )
    assert args.stage_dir == (nvme / "stage").resolve()
    assert args.output_dir == (nvme / "r1").resolve()
    assert finetune.dataset_dir(args).is_relative_to(args.stage_dir)


def test_dry_run_prints_plan_without_credentials(monkeypatch, capsys, tmp_path, fake_env):
    monkeypatch.delenv("B2_KEY_ID")
    monkeypatch.delenv("B2_APP_KEY")
    rc = finetune.main(["--dry-run", "--gr00t-dir", str(tmp_path), "--run-name", "r1"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "gr00t/experiment/launch_finetune.py" in out
    assert "s3://gr00t-unit-test/checkpoints/r1/" in out
    assert "B2 GETs" not in capsys.readouterr().err  # dry runs never query AIS


def test_checkpoint_prefix_scopes_sync_destination(monkeypatch, capsys, tmp_path, fake_env):
    rc = finetune.main(
        ["--dry-run", "--gr00t-dir", str(tmp_path), "--run-name", "r1",
         "--checkpoint-prefix", "it/x/checkpoints/"]
    )  # fmt: skip
    assert rc == 0
    assert "s3://gr00t-unit-test/it/x/checkpoints/r1/" in capsys.readouterr().out
    monkeypatch.setattr(finetune, "ais_s3_client", lambda cfg: None)
    syncer = finetune.CheckpointSync(SimpleNamespace(b2_bucket="b"), tmp_path, "r1", "it/x/")
    assert syncer.prefix == "it/x/r1"


@pytest.mark.parametrize("prefix", ["", "/", "a/../b", "a//b", "./a", "a\\b"])
def test_checkpoint_prefix_rejects_path_aliases(prefix):
    with pytest.raises(SystemExit):
        finetune.parse_args(["--checkpoint-prefix", prefix])


def test_training_backend_gets_are_reported(monkeypatch, capsys, tmp_path, fake_env):
    """Prints the AIS backend GET delta measured over training."""
    snapshots = iter([(10, 1000), (10, 1000)])
    monkeypatch.setattr(finetune, "ais_backend_gets", lambda cfg: next(snapshots))
    monkeypatch.setattr(finetune, "stage", lambda *a: STAGED)
    monkeypatch.setattr(finetune, "run_training", lambda *a: 0)
    (tmp_path / "gr00t/experiment").mkdir(parents=True)
    (tmp_path / finetune.LAUNCH_SCRIPT).write_text("")
    rc = finetune.main(["--gr00t-dir", str(tmp_path), "--output-dir", str(tmp_path / "out")])
    assert rc == 0
    assert "B2 GETs via AIS during training: 0 (0.0 MB)" in capsys.readouterr().err


def test_unreadable_stats_never_block_training(monkeypatch, capsys, tmp_path, fake_env):
    def unreadable(cfg):
        raise ConnectionError("stats endpoint down")

    trained = []
    monkeypatch.setattr(finetune, "ais_backend_gets", unreadable)
    monkeypatch.setattr(finetune, "stage", lambda *a: STAGED)
    monkeypatch.setattr(finetune, "run_training", lambda *a: trained.append(True) or 0)
    (tmp_path / "gr00t/experiment").mkdir(parents=True)
    (tmp_path / finetune.LAUNCH_SCRIPT).write_text("")
    rc = finetune.main(["--gr00t-dir", str(tmp_path), "--output-dir", str(tmp_path / "out")])
    assert rc == 0
    assert trained == [True]
    err = capsys.readouterr().err
    assert "[stats] could not read AIS backend counters: stats endpoint down" in err
    assert "B2 GETs" not in err


def test_completed_checkpoints_holds_back_newest_until_done(tmp_path):
    for step in (1000, 500, 1500):
        (tmp_path / f"checkpoint-{step}").mkdir()
    (tmp_path / "checkpoint-tmp").mkdir()  # not a step dir
    (tmp_path / "trainer_state.json").write_text("{}")

    running = finetune.completed_checkpoints(tmp_path, training_done=False)
    done = finetune.completed_checkpoints(tmp_path, training_done=True)
    assert [p.name for p in running] == ["checkpoint-500", "checkpoint-1000"]
    assert [p.name for p in done] == ["checkpoint-500", "checkpoint-1000", "checkpoint-1500"]
    assert finetune.completed_checkpoints(tmp_path / "missing", training_done=True) == []


def test_periodic_sync_failure_does_not_stop_training(monkeypatch, tmp_path, fake_env):
    """An upload error mid-run is logged and retried; the trainer runs to completion."""
    calls: list[bool] = []

    class FlakySync:
        def __init__(self, *args):
            pass

        def sync(self, training_done):
            calls.append(training_done)
            if len(calls) == 1:
                raise ConnectionError("AIS unreachable")

        def sync_rest(self):
            calls.append("rest")

    (tmp_path / "gr00t/experiment").mkdir(parents=True)
    (tmp_path / finetune.LAUNCH_SCRIPT).write_text("")
    monkeypatch.setattr(finetune, "stage", lambda *a: STAGED)
    monkeypatch.setattr(finetune, "ais_backend_gets", lambda cfg: (0, 0))
    monkeypatch.setattr(finetune, "CheckpointSync", FlakySync)
    trainer = [sys.executable, "-c", "import time; time.sleep(1); raise SystemExit(0)"]
    monkeypatch.setattr(finetune, "build_command", lambda args: trainer)

    rc = finetune.main(
        [
            "--gr00t-dir",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--sync-interval",
            "0.2",
        ]
    )
    assert rc == 0
    assert calls[0] is False and calls[-2:] == [True, "rest"]


def test_final_sync_error_does_not_mask_trainer_failure(monkeypatch, tmp_path, fake_env):
    class BrokenSync:
        def __init__(self, *args):
            pass

        def sync(self, training_done):
            raise ConnectionError("AIS unreachable")

    (tmp_path / "gr00t/experiment").mkdir(parents=True)
    (tmp_path / finetune.LAUNCH_SCRIPT).write_text("")
    monkeypatch.setattr(finetune, "stage", lambda *a: STAGED)
    monkeypatch.setattr(finetune, "ais_backend_gets", lambda cfg: (0, 0))
    monkeypatch.setattr(finetune, "CheckpointSync", BrokenSync)
    monkeypatch.setattr(
        finetune, "build_command", lambda args: [sys.executable, "-c", "raise SystemExit(3)"]
    )

    assert (
        finetune.main(
            [
                "--gr00t-dir",
                str(tmp_path),
                "--output-dir",
                str(tmp_path / "out"),
                "--sync-interval",
                "5",
            ]
        )
        == 3
    )


def test_full_dataset_identity_and_run_name_isolate_staging():
    a = finetune.parse_args(["--dataset-prefix", "datasets/a/task", "--run-name", "r1"])
    b = finetune.parse_args(["--dataset-prefix", "datasets/b/task", "--run-name", "r1"])
    c = finetune.parse_args(["--dataset-prefix", "datasets/a/task", "--run-name", "r2"])
    assert len({finetune.dataset_dir(x) for x in (a, b, c)}) == 3


@pytest.mark.parametrize("name", ["../outside", "/absolute", "a/b", "."])
def test_run_name_cannot_escape_output_directory(name):
    with pytest.raises(SystemExit):
        finetune.parse_args(["--run-name", name])


@pytest.mark.parametrize(
    "option",
    [
        "--experiment-name",
        "--experiment-name=nested",
        "--experiment_name=nested",
        "--experiment=nested",
        "--exper=nested",
        "--dataset_path=outside",
        "--output=outside",
        "--base_model_path=other",
        "--num_gpus=4",
    ],
)
def test_passthrough_cannot_redirect_owned_paths_or_launcher(option):
    with pytest.raises(SystemExit):
        finetune.parse_args(["--", option])


def test_wrapper_options_do_not_accept_abbreviations():
    with pytest.raises(SystemExit):
        finetune.parse_args(["--output", "outside"])


@pytest.mark.parametrize("interval", ["0", "-1", "nan", "inf", "-inf"])
def test_sync_interval_is_finite_and_positive(interval):
    with pytest.raises(SystemExit):
        finetune.parse_args([f"--sync-interval={interval}"])


def test_final_sync_does_not_reupload_completed_checkpoints(monkeypatch, tmp_path):
    from test_artifacts import Store

    class Gateway(Store):
        def __init__(self):
            super().__init__()
            self.uploads = []

        def head_object(self, Bucket, Key):
            head = super().head_object(Bucket, Key)
            head["Metadata"] = {}
            return head

        def upload_file(self, filename, bucket, key, **kwargs):
            self.uploads.append(key)
            super().upload_file(filename, bucket, key, **kwargs)

    checkpoint = tmp_path / "checkpoint-100"
    checkpoint.mkdir()
    (checkpoint / "model.bin").write_bytes(b"model")
    (tmp_path / "experiment_cfg.yaml").write_text("run: test\n")
    store = Gateway()
    monkeypatch.setattr(finetune, "ais_s3_client", lambda cfg: store)
    syncer = finetune.CheckpointSync(SimpleNamespace(b2_bucket="b"), tmp_path, "run")

    syncer.sync(training_done=True)
    syncer.sync_rest()

    assert store.uploads == [
        "checkpoints/run/checkpoint-100/model.bin",
        "checkpoints/run/experiment_cfg.yaml",
    ]
    assert set(read_manifest(store, "b", "checkpoints/run")["files"]) == {"experiment_cfg.yaml"}
    assert set(read_manifest(store, "b", "checkpoints/run/checkpoint-100")["files"]) == {
        "model.bin"
    }


def test_checkpoint_only_output_has_no_run_root_publication(monkeypatch, tmp_path):
    from test_artifacts import Store

    checkpoint = tmp_path / "checkpoint-100"
    checkpoint.mkdir()
    (checkpoint / "model.bin").write_bytes(b"model")
    store = Store()
    monkeypatch.setattr(finetune, "ais_s3_client", lambda cfg: store)
    syncer = finetune.CheckpointSync(SimpleNamespace(b2_bucket="b"), tmp_path, "run")
    syncer.sync(training_done=True)
    syncer.sync_rest()
    assert "checkpoints/run/" + MANIFEST not in store.data
    assert "checkpoints/run/checkpoint-100/" + MANIFEST in store.data


def test_failed_checkpoint_must_complete_before_run_root_publication(monkeypatch, tmp_path):
    from test_artifacts import Store

    checkpoint = tmp_path / "checkpoint-100"
    checkpoint.mkdir()
    (checkpoint / "model.bin").write_bytes(b"model")
    (tmp_path / "config.json").write_text("{}")
    store = Store()
    store.fail = True
    monkeypatch.setattr(finetune, "ais_s3_client", lambda cfg: store)
    syncer = finetune.CheckpointSync(SimpleNamespace(b2_bucket="b"), tmp_path, "run")
    with pytest.raises(ConnectionError):
        syncer.sync(training_done=True)
    with pytest.raises(RuntimeError, match="still need publication"):
        syncer.sync_rest()
    assert not syncer.done
    assert "checkpoints/run/" + MANIFEST not in store.data

    store.fail = False
    syncer.sync(training_done=True)
    syncer.sync_rest()
    assert syncer.done == {"checkpoint-100"}
    assert set(read_manifest(store, "b", "checkpoints/run")["files"]) == {"config.json"}


def test_main_passes_dataset_manifest_sha_to_sync(monkeypatch, tmp_path, fake_env):
    seen = []

    class RecordingSync:
        def __init__(self, cfg, output_dir, run_name, prefix, input_manifest_sha256):
            seen.append(input_manifest_sha256)

    (tmp_path / "gr00t/experiment").mkdir(parents=True)
    (tmp_path / finetune.LAUNCH_SCRIPT).write_text("")
    monkeypatch.setattr(finetune, "stage", lambda *a: STAGED)
    monkeypatch.setattr(finetune, "ais_backend_gets", lambda cfg: (0, 0))
    monkeypatch.setattr(finetune, "CheckpointSync", RecordingSync)
    monkeypatch.setattr(finetune, "run_training", lambda *a: 0)
    finetune.main(["--gr00t-dir", str(tmp_path), "--output-dir", str(tmp_path / "out")])
    assert seen == [STAGED.manifest_sha256]


@pytest.mark.parametrize("input_sha", ["a" * 64, None])
def test_every_published_manifest_records_input_dataset_manifest(monkeypatch, tmp_path, input_sha):
    from test_artifacts import Store

    checkpoint = tmp_path / "checkpoint-100"
    checkpoint.mkdir()
    (checkpoint / "model.bin").write_bytes(b"model")
    (tmp_path / "config.json").write_text("{}")
    store = Store()
    monkeypatch.setattr(finetune, "ais_s3_client", lambda cfg: store)
    syncer = finetune.CheckpointSync(
        SimpleNamespace(b2_bucket="b"), tmp_path, "run", input_manifest_sha256=input_sha
    )
    syncer.sync(training_done=True)
    syncer.sync_rest()
    for prefix in ("checkpoints/run/checkpoint-100", "checkpoints/run"):
        provenance = read_manifest(store, "b", prefix)["provenance"]
        assert provenance["input_manifest_sha256"] == input_sha
        assert provenance["base_model"] == finetune.BASE_MODEL

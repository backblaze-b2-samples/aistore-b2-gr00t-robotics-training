# SPDX-FileCopyrightText: Copyright (c) 2026 Backblaze, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage a dataset from B2 via AIStore, run Isaac GR00T fine-tuning, sync checkpoints to B2.

  python -m aistore_b2_gr00t.finetune --dataset-prefix datasets/cube_to_bowl_5 --dry-run

Arguments after `--` are passed to GR00T's launch_finetune.py unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from aistore_b2_gr00t.artifacts import MANIFEST, reject_symlinks
from aistore_b2_gr00t.config import Config
from aistore_b2_gr00t.storage import ais_backend_gets, ais_s3_client, stage, upload_dir

# Upstream entrypoint at the pinned tag (NVIDIA/Isaac-GR00T n1.7-release).
LAUNCH_SCRIPT = "gr00t/experiment/launch_finetune.py"
BASE_MODEL = "nvidia/GR00T-N1.7-3B"
_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")
_SHUTDOWN_TIMEOUT_S = 30.0
_EXIT_GRACE_S = 5.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    argv = sys.argv[1:] if argv is None else argv
    extra: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1 :]

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter, allow_abbrev=False
    )
    ap.add_argument("--dataset-prefix", default="datasets/cube_to_bowl_5")
    ap.add_argument(
        "--run-name",
        default=datetime.now(UTC).strftime("gr00t-%Y%m%d-%H%M%S-") + uuid4().hex[:12],
    )
    ap.add_argument(
        "--checkpoint-prefix",
        default="checkpoints",
        help="B2 key prefix; checkpoints sync to <prefix>/<run-name>/",
    )
    ap.add_argument("--stage-dir", type=Path, default=Path("data/stage"), help="node-local disk")
    ap.add_argument("--output-dir", type=Path, help="default: outputs/<run-name>")
    ap.add_argument("--stage-method", choices=["getbatch", "get"], default="getbatch")
    ap.add_argument(
        "--gr00t-dir",
        type=Path,
        default=Path(os.environ.get("GR00T_DIR", "third_party/Isaac-GR00T")),
    )
    ap.add_argument("--embodiment-tag", default="NEW_EMBODIMENT")
    ap.add_argument(
        "--modality-config-path",
        default="examples/SO100/so100_config.py",
        help="relative to --gr00t-dir; pass '' for pre-registered tags (e.g. UNITREE_G1)",
    )
    ap.add_argument("--num-gpus", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument(
        "--sync-interval", type=float, default=60.0, help="seconds between checkpoint syncs"
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="print the plan and GR00T command, run nothing"
    )
    args = ap.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", args.run_name):
        ap.error("--run-name must be a single identifier (letters, digits, underscores, hyphens)")
    if (
        args.num_gpus < 1
        or args.max_steps < 1
        or not math.isfinite(args.sync_interval)
        or args.sync_interval <= 0
    ):
        ap.error("--num-gpus, --max-steps and --sync-interval must be positive")
    args.checkpoint_prefix = args.checkpoint_prefix.strip("/")
    if "\\" in args.checkpoint_prefix or any(
        part in ("", ".", "..") for part in args.checkpoint_prefix.split("/")
    ):
        ap.error("--checkpoint-prefix must be a key prefix without path aliases")
    reserved = (
        "--dataset-path",
        "--output-dir",
        "--base-model-path",
        "--num-gpus",
        "--experiment-name",
    )
    for arg in extra:
        option = arg.split("=", 1)[0].replace("_", "-")
        if option.startswith("--") and any(flag.startswith(option) for flag in reserved):
            ap.error(
                "dataset, output, base model, GPU and experiment flags are owned by the wrapper"
            )
    args.extra = extra
    # Resolve user-chosen roots once so they may be symlinks (e.g. to local NVMe); symlinks
    # inside staged or uploaded trees are still refused.
    args.stage_dir = args.stage_dir.resolve()
    args.output_dir = (args.output_dir or Path("outputs") / args.run_name).resolve()
    return args


def dataset_dir(args: argparse.Namespace) -> Path:
    reject_symlinks(args.stage_dir)
    prefix = args.dataset_prefix.strip("/")
    identity = hashlib.sha256(prefix.encode()).hexdigest()[:16]
    return args.stage_dir / f"dataset-{identity}" / args.run_name


def build_command(args: argparse.Namespace) -> list[str]:
    """The exact GR00T invocation; run with cwd=--gr00t-dir inside GR00T's own uv env."""
    launcher = (
        ["python"]
        if args.num_gpus == 1
        # --standalone picks a free rendezvous port, so concurrent jobs on one host don't collide.
        else ["torchrun", "--standalone", f"--nproc_per_node={args.num_gpus}"]
    )
    cmd = [
        "uv", "run", "--project", str(args.gr00t_dir.resolve()), "--locked",
        *launcher, LAUNCH_SCRIPT,
        "--base-model-path", BASE_MODEL,
        "--dataset-path", str(dataset_dir(args)),
        "--embodiment-tag", args.embodiment_tag,
        "--num-gpus", str(args.num_gpus),
        "--output-dir", str(args.output_dir.resolve()),
        "--max-steps", str(args.max_steps),
    ]  # fmt: skip
    if args.modality_config_path:
        cmd += ["--modality-config-path", args.modality_config_path]
    return cmd + args.extra


def training_env(environ: Mapping[str, str], num_gpus: int) -> dict[str, str]:
    """Trainer environment: no B2 credentials, and exactly one visible GPU for 1-GPU runs."""
    # Drop VIRTUAL_ENV too so `uv run --project` uses GR00T's own venv without warnings.
    hidden = ("B2_KEY_ID", "B2_APP_KEY", "VIRTUAL_ENV")
    env = {k: v for k, v in environ.items() if k not in hidden}
    if num_gpus == 1:
        # HF Trainer wraps the model in DataParallel when it sees more than one GPU (upstream
        # examples/finetune.sh pins device 0). Keep the first device a scheduler granted.
        granted = env.get("CUDA_VISIBLE_DEVICES", "0")
        env["CUDA_VISIBLE_DEVICES"] = granted.split(",")[0].strip()
    return env


def completed_checkpoints(output_dir: Path, training_done: bool) -> list[Path]:
    """`checkpoint-N` dirs that are safe to upload, oldest first.

    The newest is held back until a newer one appears or the run exits cleanly, because
    transformers 4.57.3 (pinned by GR00T n1.7) writes checkpoint-N in place, not via
    a temp dir and rename, so it may still be mid-write.
    """
    if not output_dir.is_dir():
        return []
    ckpts = sorted(
        (p for p in output_dir.iterdir() if p.is_dir() and _CKPT_RE.match(p.name)),
        key=lambda p: int(_CKPT_RE.match(p.name).group(1)),
    )
    return ckpts if training_done else ckpts[:-1]


class CheckpointSync:
    """Uploads finished checkpoints to s3://<bucket>/<prefix>/<run>/ through AIS."""

    def __init__(
        self,
        cfg: Config,
        output_dir: Path,
        run_name: str,
        prefix: str = "checkpoints",
        input_manifest_sha256: str | None = None,
    ) -> None:
        self.s3 = ais_s3_client(cfg)
        self.bucket = cfg.b2_bucket
        self.output_dir = output_dir
        self.prefix = f"{prefix.strip('/')}/{run_name}"
        # Lineage: every published manifest names the dataset manifest it was trained on.
        self.provenance = {"base_model": BASE_MODEL, "input_manifest_sha256": input_manifest_sha256}
        self.done: set[str] = set()

    def sync(self, training_done: bool) -> None:
        for ckpt in completed_checkpoints(self.output_dir, training_done):
            if ckpt.name in self.done:
                continue
            try:
                n, _ = upload_dir(
                    self.s3,
                    ckpt,
                    self.bucket,
                    f"{self.prefix}/{ckpt.name}",
                    publish=True,
                    provenance={**self.provenance, "checkpoint": ckpt.name},
                )
            except FileNotFoundError:
                # The trainer's save_total_limit rotated it away mid-upload; skip it.
                print(f"[sync] {ckpt.name} vanished during upload, skipped", file=sys.stderr)
                continue
            self.done.add(ckpt.name)
            print(f"[sync] {ckpt.name}: {n} files -> s3://{self.bucket}/{self.prefix}/{ckpt.name}/")

    def sync_rest(self) -> None:
        """Publish run-level files; completed checkpoints keep their own manifests."""
        remaining = {
            p.name for p in completed_checkpoints(self.output_dir, training_done=True)
        } - self.done
        if remaining:
            raise RuntimeError(
                f"checkpoints still need publication: {', '.join(sorted(remaining))}"
            )
        exclude = tuple(sorted(self.done))
        reject_symlinks(self.output_dir)
        has_files = False
        for path in self.output_dir.rglob("*"):
            if path.relative_to(self.output_dir).parts[0] in self.done:
                continue
            reject_symlinks(path)
            if path.is_file() and path.name != MANIFEST:
                has_files = True
                break
        if not has_files:
            return
        n, _ = upload_dir(
            self.s3,
            self.output_dir,
            self.bucket,
            self.prefix,
            publish=True,
            provenance={**self.provenance, "training_exit_code": 0},
            exclude=exclude,
        )
        print(f"[sync] final: {n} files -> s3://{self.bucket}/{self.prefix}/")


class _TrainingInterrupted(BaseException):
    """Unwind transfers without treating an interruption as a retryable upload error."""


def _signal_group(proc: subprocess.Popen, signum: int) -> bool:
    proc.poll()
    try:
        os.killpg(proc.pid, signum)
    except PermissionError:
        # macOS can report EPERM for an exited, unreaped group leader.
        proc.poll()
        try:
            os.killpg(proc.pid, signum)
        except ProcessLookupError:
            return False
    except ProcessLookupError:
        return False
    return True


def _wait_group(proc: subprocess.Popen, timeout: float) -> bool:
    """Poll until the process group is empty; return whether it emptied within `timeout`."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            if not _signal_group(proc, 0):
                return True
        except PermissionError:
            # Reparented zombie workers can briefly keep a macOS group registered.
            pass
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


def _stop_trainer(proc: subprocess.Popen, *, signal_sent: bool = False) -> bool:
    """Stop the owned process group; return whether it emptied on its own."""
    # Short-lived helpers (wandb, Ray, dataloader workers) may trail a clean launcher exit;
    # a group that empties on its own within the grace period exited cleanly.
    if _wait_group(proc, 0 if signal_sent else _EXIT_GRACE_S):
        return True
    if not signal_sent:
        with suppress(PermissionError):
            _signal_group(proc, signal.SIGTERM)
    # Wait on the whole group, not the leader: workers may outlive a fast-exiting launcher.
    if not _wait_group(proc, _SHUTDOWN_TIMEOUT_S):
        _signal_group(proc, signal.SIGKILL)
    proc.wait(timeout=5)
    return False


def _forward(proc: subprocess.Popen, signum: int) -> None:
    """Signal the group from a handler or timer, where raising would mask the interruption."""
    try:
        _signal_group(proc, signum)
    except OSError as exc:
        print(f"[train] signal forwarding failed: {exc}", file=sys.stderr)


def _wait_with_periodic_sync(
    proc: subprocess.Popen, interval: float, syncer: CheckpointSync
) -> None:
    """Wait for the launcher, uploading superseded checkpoints every `interval` seconds."""
    while True:
        try:
            proc.wait(timeout=interval)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            syncer.sync(training_done=False)
        except Exception as exc:
            print(f"[sync] failed, retrying next tick: {exc}", file=sys.stderr)


def _publish(syncer: CheckpointSync, rc: int) -> None:
    """Final upload: everything after a clean run, otherwise only superseded checkpoints."""
    try:
        syncer.sync(training_done=rc == 0)
        if rc == 0:
            syncer.sync_rest()
    except Exception as exc:
        if rc == 0:
            raise
        print(f"[sync] final sync failed: {exc}", file=sys.stderr)


def run_training(
    cmd: list[str], args: argparse.Namespace, env: dict, syncer: CheckpointSync
) -> int:
    """Run a POSIX training process group and publish only after its workers stop.

    The first SIGINT/SIGTERM is forwarded to the group and raised as `_TrainingInterrupted`
    to abort any in-flight upload; a second one SIGKILLs the group.
    """
    proc = None
    interrupted = None
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    # Armed by the first signal: unwinding upload threads can block the main thread past
    # the shutdown deadline, so the group is killed from a timer thread.
    deadline = threading.Timer(_SHUTDOWN_TIMEOUT_S, lambda: _forward(proc, signal.SIGKILL))
    deadline.daemon = True

    def handle_signal(signum, _frame):
        nonlocal interrupted
        first = interrupted is None
        interrupted = interrupted or signum
        if proc is None:
            return  # Popen is still starting the group; checked right after it returns.
        _forward(proc, signum if first else signal.SIGKILL)
        if first:
            deadline.start()
            raise _TrainingInterrupted

    try:
        for sig in previous:
            signal.signal(sig, handle_signal)
        try:
            proc = subprocess.Popen(cmd, cwd=args.gr00t_dir, env=env, start_new_session=True)
            if interrupted is not None:
                _forward(proc, interrupted)
                raise _TrainingInterrupted
            _wait_with_periodic_sync(proc, args.sync_interval, syncer)
            rc = proc.returncode
            emptied = _stop_trainer(proc)  # always runs: reaps helpers, stops lingering workers
            if not emptied and rc == 0:
                print(
                    "[train] launcher exited with active workers; output is incomplete",
                    file=sys.stderr,
                )
                rc = 1
            _publish(syncer, rc)
            return rc
        except _TrainingInterrupted:
            pass
        except BaseException:
            if proc is not None:
                _stop_trainer(proc)  # never leave an orphaned GPU job behind an unexpected error
            raise

        # Interrupted. Later signals only escalate to SIGKILL and never raise, so this path
        # runs to completion: stop the group, then publish the checkpoints it superseded.
        try:
            _stop_trainer(proc, signal_sent=True)
        except Exception as exc:
            # Workers may still be writing; publishing now could ship a torn checkpoint.
            print(f"[train] shutdown failed: {exc}", file=sys.stderr)
            return 128 + interrupted
        finally:
            deadline.cancel()  # the group is reaped; a late timer must not SIGKILL a reused id
        _publish(syncer, 128 + interrupted)
        return 128 + interrupted
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = Config.from_env(require_b2_credentials=False)
    cmd = build_command(args)
    env = training_env(os.environ, args.num_gpus)

    print(
        f"stage:  s3://{cfg.b2_bucket}/{args.dataset_prefix} -> {dataset_dir(args)} "
        f"(via AIS {cfg.ais_endpoint}, {args.stage_method})"
    )
    print(f"train:  (cd {args.gr00t_dir} && {shlex.join(cmd)})")
    print(
        f"sync:   {args.output_dir} -> "
        f"s3://{cfg.b2_bucket}/{args.checkpoint_prefix}/{args.run_name}/"
    )
    if args.dry_run:
        return 0

    if not (args.gr00t_dir / LAUNCH_SCRIPT).is_file():
        print(
            f"error: {args.gr00t_dir / LAUNCH_SCRIPT} not found (run 'make setup')", file=sys.stderr
        )
        return 1

    reject_symlinks(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    result = stage(cfg, args.dataset_prefix, dataset_dir(args), args.stage_method)
    print(f"staged {result.objects} objects, {result.bytes / 1e6:.1f} MB in {result.seconds:.1f}s")

    syncer = CheckpointSync(
        cfg, args.output_dir, args.run_name, args.checkpoint_prefix, result.manifest_sha256
    )
    # Training reads only the staged copy; any B2 GET from here on would be a cache bypass.
    gets_before = _backend_gets(cfg)
    rc = run_training(cmd, args, env, syncer)
    _report_training_gets(cfg, gets_before)
    return rc


def _backend_gets(cfg: Config) -> tuple[int, int] | None:
    """AIS backend GET counters; diagnostic only, so a failed read never blocks publication."""
    try:
        return ais_backend_gets(cfg)
    except Exception as exc:
        print(f"[stats] could not read AIS backend counters: {exc}", file=sys.stderr)
        return None


def _report_training_gets(cfg: Config, before: tuple[int, int] | None) -> None:
    """Print the AIS -> B2 GET delta over training."""
    if before is None or (after := _backend_gets(cfg)) is None:
        return
    n, size = after[0] - before[0], after[1] - before[1]
    print(f"[stats] B2 GETs via AIS during training: {n} ({size / 1e6:.1f} MB)", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())

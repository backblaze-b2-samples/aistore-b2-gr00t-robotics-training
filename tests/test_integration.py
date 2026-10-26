# SPDX-FileCopyrightText: Copyright (c) 2026 Backblaze, Inc.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end checks against AIStore fronting a real B2 bucket.

Run with `make test-integration`: it loads .env (B2_IT=1 and a key scoped to a dedicated
test bucket), starts a throwaway AIS cluster on ports 51180/51181, and removes its cache
afterwards. Uses GR00T's demo_data/cube_to_bowl_5 from DEMO_DATA_DIR.

Every object is written under a unique `it/<timestamp>-<hex>/` prefix and is never
deleted by the suite; a B2 lifecycle rule on `it/` expires them.
"""

import hashlib
import json
import os
import secrets
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import pytest
from botocore.exceptions import ClientError
from conftest import ROOT, load_module

from aistore_b2_gr00t.artifacts import MANIFEST, read_manifest, verify_remote
from aistore_b2_gr00t.config import Config
from aistore_b2_gr00t.finetune import CheckpointSync
from aistore_b2_gr00t.storage import (
    BACKEND_GET_COUNT,
    ais_backend_gets,
    ais_bucket,
    ais_client,
    ais_s3_client,
    b2_client,
    cached_keys,
    evict,
    list_objects,
    stage,
)

pytestmark = pytest.mark.integration
RUN = f"it/{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{secrets.token_hex(4)}/"
# Trailing slash: `cube_to_bowl_5` must not match a sibling such as `cube_to_bowl_50`.
PREFIX = f"{RUN}datasets/cube_to_bowl_5/"
CKPT_PREFIX = f"{RUN}checkpoints"
IT_AIS_PORT = 51180


def sha256_tree(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != MANIFEST
    }


def gets_since(cfg: Config, before: tuple[int, int]) -> tuple[int, int]:
    """(count, bytes) of AIS -> B2 GETs since `before`.

    Tests run serially on a dedicated cluster, so the counters move only with the test's
    own reads.

    Only GET counters are asserted: S3 CopyObject and bucket copies (tcb) do not record
    backend GET stats, so never extend these checks to copies.
    """
    after = ais_backend_gets(cfg)
    return after[0] - before[0], after[1] - before[1]


@pytest.fixture(scope="module")
def cfg() -> Config:
    # Hard stops: real B2 only on explicit opt-in, and never the user's `make up` cluster
    # (the attach script rewrites cluster-wide features).
    if os.environ.get("B2_IT") != "1":
        pytest.fail("set B2_IT=1 in .env to run integration tests against real B2")
    cfg = Config.from_env()
    if urlparse(cfg.ais_endpoint).port != IT_AIS_PORT:
        pytest.fail(f"AIS_ENDPOINT must be the test cluster on port {IT_AIS_PORT}")
    return cfg


@pytest.fixture(scope="module")
def demo_dir() -> Path:
    path = Path(os.environ.get("DEMO_DATA_DIR", ""))
    if not (path / "meta" / "info.json").is_file():
        pytest.fail("DEMO_DATA_DIR not set to GR00T demo_data/cube_to_bowl_5 (run `make setup`)")
    return path


@pytest.fixture(scope="module")
def attached(cfg: Config) -> None:
    try:
        b2_client(cfg).head_bucket(Bucket=cfg.b2_bucket)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in ("403", "404", "AccessDenied", "NoSuchBucket"):
            pytest.fail(
                f"B2 bucket {cfg.b2_bucket!r} not accessible ({code}); use a dedicated test "
                "bucket and a key scoped to it with listBuckets, listFiles, readFiles, writeFiles"
            )
        raise
    for _ in range(2):  # second run proves the script is idempotent
        subprocess.run([str(ROOT / "scripts/attach_b2_bucket.sh")], check=True)


@pytest.fixture(scope="module")
def seeded(cfg: Config, attached: None, demo_dir: Path) -> list[str]:
    seed = load_module("scripts/seed_b2.py")
    assert seed.main(["--local-dir", str(demo_dir), "--prefix", PREFIX]) == 0
    assert seed.main(["--local-dir", str(demo_dir), "--prefix", PREFIX]) == 0  # idempotent
    return [
        o.key
        for o in list_objects(b2_client(cfg), cfg.b2_bucket, PREFIX)
        if not o.key.endswith("/" + MANIFEST)
    ]


def cached(cfg: Config) -> set[str]:
    return cached_keys(ais_bucket(ais_client(cfg), cfg), PREFIX)


def test_cold_get_via_ais_reads_b2_once(cfg, seeded):
    key = next(k for k in seeded if k.endswith(".mp4"))
    evict(ais_bucket(ais_client(cfg), cfg), [key])
    assert key not in cached(cfg)

    before = ais_backend_gets(cfg)
    via_ais = ais_s3_client(cfg).get_object(Bucket=cfg.b2_bucket, Key=key)["Body"].read()
    assert gets_since(cfg, before) == (1, len(via_ais))
    direct = b2_client(cfg).get_object(Bucket=cfg.b2_bucket, Key=key)["Body"].read()
    assert via_ais == direct
    assert key in cached(cfg)  # the next reader is served from AIS
    # The counter name is image-specific; fail loudly if an image bump renames it.
    perf = ais_client(cfg).cluster().get_performance()
    assert any(BACKEND_GET_COUNT in node["tracker"] for node in perf.values())


@pytest.mark.parametrize("method", ["getbatch", "get"])
def test_stage_cold_reads_b2_then_warm_skips_it(cfg, seeded, demo_dir, tmp_path, method):
    manifest = PREFIX + MANIFEST
    manifest_size = b2_client(cfg).head_object(Bucket=cfg.b2_bucket, Key=manifest)["ContentLength"]
    evict(ais_bucket(ais_client(cfg), cfg), [*seeded, manifest])
    expected = sha256_tree(demo_dir)

    before = ais_backend_gets(cfg)
    cold = stage(cfg, PREFIX, tmp_path / "cold", method)
    # One backend GET per object plus the publication manifest (its re-read is warm).
    assert gets_since(cfg, before) == (cold.objects + 1, cold.bytes + manifest_size)
    assert cold.objects == len(expected)
    assert sha256_tree(tmp_path / "cold") == expected
    assert set(seeded) <= cached(cfg)

    before = ais_backend_gets(cfg)
    warm = stage(cfg, PREFIX, tmp_path / "warm", method)
    assert gets_since(cfg, before) == (0, 0)  # repeat reads never touch B2
    assert sha256_tree(tmp_path / "warm") == expected
    assert warm.bytes == cold.bytes


def test_checkpoint_sync_writes_through_to_backend(cfg, attached, tmp_path):
    out = tmp_path / "outputs"
    for step in (500, 1000):
        ckpt = out / f"checkpoint-{step}"
        ckpt.mkdir(parents=True)
        (ckpt / "trainer_state.json").write_text(f'{{"global_step": {step}}}')
    # 12 MiB forces boto3's multipart path (8 MiB threshold) through AIS to B2.
    (out / "checkpoint-1000" / "model.safetensors").write_bytes(os.urandom(12 * 1024 * 1024))
    (out / "experiment_cfg.yaml").write_text("run: it\n")

    run = f"it-{os.getpid()}"
    root = f"{CKPT_PREFIX}/{run}"
    syncer = CheckpointSync(cfg, out, run, CKPT_PREFIX)
    direct = b2_client(cfg)

    def backend_keys() -> set[str]:
        return {
            o.key
            for o in list_objects(direct, cfg.b2_bucket, root)
            if not o.key.endswith("/" + MANIFEST)
        }

    syncer.sync(training_done=False)  # newest checkpoint held back while "training"
    assert backend_keys() == {f"{root}/checkpoint-500/trainer_state.json"}

    syncer.sync(training_done=True)
    syncer.sync_rest()
    assert backend_keys() == {
        f"{root}/checkpoint-500/trainer_state.json",
        f"{root}/checkpoint-1000/trainer_state.json",
        f"{root}/checkpoint-1000/model.safetensors",
        f"{root}/experiment_cfg.yaml",
    }
    key = f"{root}/checkpoint-1000/model.safetensors"
    body = direct.get_object(Bucket=cfg.b2_bucket, Key=key)["Body"].read()
    assert body == (out / "checkpoint-1000" / "model.safetensors").read_bytes()
    assert verify_remote(direct, cfg.b2_bucket, f"{root}/checkpoint-1000") == 2
    assert verify_remote(direct, cfg.b2_bucket, root) == 1
    assert set(read_manifest(direct, cfg.b2_bucket, root)["files"]) == {"experiment_cfg.yaml"}


@pytest.mark.parametrize("method", ["getbatch", "get"])
def test_stage_benchmark_runs_end_to_end(cfg, seeded, tmp_path, method):
    bench = load_module("benchmarks/stage_time.py")
    args = ["--prefix", PREFIX, "--repeats", "1", "--stagers", "2", "--method", method]
    assert bench.main([*args, "--out-dir", str(tmp_path), "--scratch", str(tmp_path)]) == 0
    (report,) = tmp_path.glob("stage_time-*.json")
    results = {r["mode"]: r for r in json.loads(report.read_text())["results"]}
    assert list(results) == ["b2-direct", "ais-cold", "ais-warm"]
    assert len({r["bytes_per_repeat"] for r in results.values()}) == 1  # all staged everything
    assert results["b2-direct"]["b2_gets"] == 2 * len(seeded)  # client-counted, per stager
    # Concurrent cold stagers may race on a key, so only the lower bound is exact.
    assert results["ais-cold"]["b2_gets"] >= len(seeded)
    assert results["ais-warm"]["b2_gets"] == 0


FAKE_LAUNCHER = """
import argparse, json, pathlib, time
ap = argparse.ArgumentParser()
ap.add_argument("--dataset-path")
ap.add_argument("--output-dir")
args, _ = ap.parse_known_args()
assert (pathlib.Path(args.dataset_path) / "meta" / "modality.json").is_file()
out = pathlib.Path(args.output_dir)
for step in (1, 2):
    (out / f"checkpoint-{step}").mkdir(parents=True)
    (out / f"checkpoint-{step}" / "trainer_state.json").write_text(json.dumps({"step": step}))
    time.sleep(1.5)
"""


def test_finetune_main_stages_trains_and_syncs(cfg, seeded, tmp_path, capsys):
    """Full `finetune.main` flow with a fake GR00T launch_finetune.py."""
    from aistore_b2_gr00t import finetune

    gr00t = tmp_path / "Isaac-GR00T"
    (gr00t / "gr00t/experiment").mkdir(parents=True)
    (gr00t / finetune.LAUNCH_SCRIPT).write_text(FAKE_LAUNCHER)
    (gr00t / "pyproject.toml").write_text(
        '[project]\nname = "fake-gr00t"\nversion = "0"\nrequires-python = ">=3.10"\n'
    )
    subprocess.run(["uv", "lock", "--project", str(gr00t), "--offline"], check=True)
    run = f"it-main-{os.getpid()}"
    root = f"{CKPT_PREFIX}/{run}/"
    rc = finetune.main(
        [
            "--gr00t-dir", str(gr00t), "--stage-dir", str(tmp_path / "stage"),
            "--output-dir", str(tmp_path / "out"), "--run-name", run, "--sync-interval", "1",
            "--dataset-prefix", PREFIX, "--checkpoint-prefix", CKPT_PREFIX,
        ]
    )  # fmt: skip
    assert rc == 0
    # Checkpoint PUTs write through and stay cached, so training causes no B2 reads.
    assert "B2 GETs via AIS during training: 0 " in capsys.readouterr().err
    listed = {o.key for o in list_objects(b2_client(cfg), cfg.b2_bucket, root)}
    assert {k for k in listed if not k.endswith("/" + MANIFEST)} == {
        f"{root}checkpoint-1/trainer_state.json",
        f"{root}checkpoint-2/trainer_state.json",
    }
    assert root + MANIFEST not in listed

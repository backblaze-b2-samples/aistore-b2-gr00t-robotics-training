# SPDX-FileCopyrightText: Copyright (c) 2026 Backblaze, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for scripts/seed_b2.py and benchmarks/stage_time.py helpers."""

import json
from types import SimpleNamespace

import pytest
from conftest import load_module
from test_storage import FakeS3

from aistore_b2_gr00t import storage
from aistore_b2_gr00t.artifacts import MANIFEST

seed = load_module("scripts/seed_b2.py")
bench = load_module("benchmarks/stage_time.py")


def test_find_lfs_pointers(tmp_path):
    (tmp_path / "videos").mkdir()
    (tmp_path / "videos" / "ep0.mp4").write_bytes(
        b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 2414613\n"
    )
    (tmp_path / "info.json").write_text('{"codebase_version": "v2.1"}')
    assert [p.name for p in seed.find_lfs_pointers(tmp_path)] == ["ep0.mp4"]


def test_seed_requires_pinned_revision_for_hf(fake_env):
    with pytest.raises(SystemExit):
        seed.main(["--hf-repo", "nvidia/x", "--prefix", "datasets/"])


def test_seed_refuses_lfs_pointer_tree(tmp_path, fake_env, capsys):
    (tmp_path / "a.parquet").write_bytes(b"version https://git-lfs.github.com/spec/v1\n")
    assert seed.main(["--local-dir", str(tmp_path), "--prefix", "datasets/x"]) == 1
    assert "git lfs pull" in capsys.readouterr().err


@pytest.mark.parametrize(
    "values,pct,expected",
    [([], 50, 0.0), ([3.0, 1.0, 2.0], 50, 2.0), (list(range(1, 101)), 99, 99)],
)
def test_percentile(values, pct, expected):
    assert bench.percentile(values, pct) == expected


@pytest.mark.parametrize(
    "subdir", ["/outside", "../outside", "task/../other", "task//meta", ".", "task\\meta"]
)
def test_seed_rejects_unsafe_source_subdirectory_before_download(fake_env, monkeypatch, subdir):
    monkeypatch.setattr(seed, "download_hf", lambda *a: pytest.fail("unexpected download"))
    with pytest.raises(SystemExit):
        seed.main(
            [
                "--hf-repo",
                "nvidia/x",
                "--revision",
                "a" * 40,
                "--allow-pattern",
                "task/**",
                "--source-subdir",
                subdir,
                "--prefix",
                "datasets/task",
            ]
        )


def test_seed_source_subdirectory_cannot_follow_symlink(tmp_path, fake_env, monkeypatch):
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "source"
    root.mkdir()
    (root / "task").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(seed, "b2_client", lambda cfg: pytest.fail("unexpected client"))
    with pytest.raises(ValueError, match="symlink"):
        seed.main(
            [
                "--local-dir",
                str(root),
                "--source-subdir",
                "task",
                "--prefix",
                "datasets/task",
            ]
        )


def test_hf_subdirectory_publishes_and_gates_exact_training_prefix(tmp_path, fake_env, monkeypatch):
    snapshot = tmp_path / "snapshot"
    task = snapshot / "task"
    (task / "meta").mkdir(parents=True)
    (task / "meta" / "info.json").write_bytes(b"{}")
    (task / "data").mkdir()
    episode = task / "data" / "episode.parquet"
    episode.write_bytes(b"new")
    (task / ".cache").mkdir()
    (task / ".cache" / "download-metadata").write_bytes(b"ignore")
    store = FakeS3({})
    monkeypatch.setattr(seed, "download_hf", lambda *args: snapshot)
    monkeypatch.setattr(seed, "b2_client", lambda cfg: store)
    monkeypatch.setattr(storage, "ais_s3_client", lambda cfg: store)
    argv = [
        "--hf-repo",
        "nvidia/x",
        "--revision",
        "a" * 40,
        "--allow-pattern",
        "task/**",
        "--source-subdir",
        "task",
        "--prefix",
        "datasets/task",
        "--workers",
        "1",
    ]
    assert seed.main(argv) == 0
    marker = "datasets/task/" + MANIFEST
    manifest = json.loads(store.objects[marker])
    assert manifest["status"] == "complete"
    assert set(manifest["files"]) == {"meta/info.json", "data/episode.parquet"}
    assert manifest["provenance"] == {
        "source": "nvidia/x",
        "revision": "a" * 40,
        "source_subdir": "task",
    }
    cfg = SimpleNamespace(b2_bucket=fake_env["B2_BUCKET"])
    storage.stage(cfg, "datasets/task", tmp_path / "valid", "get")
    store.objects["datasets/task/data/episode.parquet"] = b"old"
    with pytest.raises(OSError, match="checksum mismatch"):
        storage.stage(cfg, "datasets/task", tmp_path / "corrupt", "get")
    assert not (tmp_path / "corrupt").exists()

    episode.write_bytes(b"newer")

    def fail_upload(*args, **kwargs):
        raise ConnectionError("backend unavailable")

    monkeypatch.setattr(store, "upload_file", fail_upload)
    with pytest.raises(ConnectionError):
        seed.main(argv)
    assert json.loads(store.objects[marker])["status"] == "pending"
    with pytest.raises(ValueError, match="incomplete"):
        storage.stage(cfg, "datasets/task", tmp_path / "pending", "get")
    assert not (tmp_path / "pending").exists()


def test_benchmark_resolves_only_created_temporary_roots(tmp_path, monkeypatch):
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    created = real / "created"
    created.mkdir()
    seen = []

    class Temporary:
        def __init__(self, **kwargs):
            assert kwargs["dir"] is None

        def __enter__(self):
            return str(alias / "created")

        def __exit__(self, *args):
            return False

    def stage_once(cfg, mode, prefix, dest, method, workers):
        storage.reject_symlinks(dest)
        seen.append(dest)
        return 3, [0.01]

    monkeypatch.setattr(bench.tempfile, "TemporaryDirectory", Temporary)
    monkeypatch.setattr(bench, "stage_once", stage_once)
    monkeypatch.setattr(bench, "ais_backend_gets", lambda cfg: (0, 0))
    args = SimpleNamespace(
        scratch=None, prefix="ds", method="getbatch", workers=None, repeats=1, stagers=1
    )
    report = bench.run_mode(None, args, "ais-warm", ["ds/a"])
    assert seen == [created, created / "stager-0"]
    assert report["workers"] == 4 and report["prefetch_workers"] == 16


@pytest.mark.parametrize(
    "mode,method,expected_direct",
    [
        ("b2-direct", "get", True),
        ("ais-cold", "getbatch", False),
        ("ais-warm", "getbatch", False),
    ],
)
def test_benchmark_stages_through_storage_stage(
    tmp_path, monkeypatch, mode, method, expected_direct
):
    def stage(cfg, prefix, dest, selected_method, workers, *, direct):
        assert selected_method == method and workers is None
        assert direct is expected_direct
        return SimpleNamespace(bytes=3, latencies=[0.1])

    monkeypatch.setattr(bench, "stage", stage)
    assert bench.stage_once(None, mode, "ds", tmp_path, "getbatch", None) == (3, [0.1])


@pytest.mark.parametrize(
    "mode,expected", [("b2-direct", (4, 6)), ("ais-cold", (5, 7)), ("ais-warm", (0, 0))]
)
def test_benchmark_reports_measured_b2_gets(monkeypatch, mode, expected):
    """AIS modes diff backend counters around the timed repeats; b2-direct counts its own GETs.

    b2-direct bypasses AIS, so it records one latency and one data GET per object."""
    counters = {"ais-cold": iter([(10, 100), (15, 107)]), "ais-warm": iter([(15, 107)] * 2)}
    monkeypatch.setattr(bench, "ais_backend_gets", lambda cfg: next(counters[mode]))
    monkeypatch.setattr(bench, "evict", lambda bucket, keys: None)
    monkeypatch.setattr(bench, "ais_client", lambda cfg: None)
    monkeypatch.setattr(bench, "ais_bucket", lambda client, cfg: None)
    monkeypatch.setattr(bench, "stage_once", lambda *a: (3, [0.1, 0.1]))
    args = SimpleNamespace(
        scratch=None, prefix="ds", method="get", workers=None, repeats=2, stagers=1
    )
    report = bench.run_mode(None, args, mode, ["ds/a", "ds/b"])
    assert (report["b2_gets"], report["b2_get_bytes"]) == expected
    assert "expected_b2_gets" not in report


@pytest.mark.parametrize("flag", ["--repeats", "--stagers", "--workers"])
def test_benchmark_rejects_non_positive_counts(flag, capsys):
    with pytest.raises(SystemExit):
        bench.main([flag, "0"])
    assert "must be positive" in capsys.readouterr().err


def test_benchmark_writes_reports_to_out_dir(tmp_path, monkeypatch):
    cfg = SimpleNamespace(b2_bucket="b", b2_endpoint="https://b2", ais_endpoint="http://ais")
    monkeypatch.setattr(bench, "Config", SimpleNamespace(from_env=lambda: cfg))
    monkeypatch.setattr(bench, "b2_client", lambda cfg: None)
    monkeypatch.setattr(bench, "dataset_listing", lambda *a: ([SimpleNamespace(key="ds/a")], None))
    monkeypatch.setattr(bench, "tcp_rtt_ms", lambda url: 1.0)
    row = {
        "mode": "ais-warm",
        "method": "get",
        "workers": 16,
        "prefetch_workers": 0,
        "stagers": 1,
        "objects": 1,
        "bytes_per_repeat": 3,
        "wall_s_median": 1.0,
        "aggregate_mb_s": 0.0,
        "request_latency_ms_p50": 1.0,
        "request_latency_ms_p99": 1.0,
        "b2_gets": 0,
        "b2_get_bytes": 0,
    }
    monkeypatch.setattr(bench, "run_mode", lambda *a: row)
    out = tmp_path / "reports"
    assert bench.main(["--modes", "ais-warm", "--out-dir", str(out)]) == 0
    (report,) = out.glob("stage_time-*.json")
    assert json.loads(report.read_text())["results"] == [row]
    assert len(list(out.glob("stage_time-*.md"))) == 1

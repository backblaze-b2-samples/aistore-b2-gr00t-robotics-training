# SPDX-FileCopyrightText: Copyright (c) 2026 Backblaze, Inc.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import io
import json
from types import SimpleNamespace

import pytest
from botocore.exceptions import ClientError

from aistore_b2_gr00t import storage
from aistore_b2_gr00t.artifacts import MANIFEST, manifest_sha256
from aistore_b2_gr00t.config import Config
from aistore_b2_gr00t.storage import ObjectInfo


class FakeS3:
    """Minimal boto3 S3 client fake for the helpers under test."""

    def __init__(self, objects: dict[str, bytes]):
        self.objects = dict(objects)
        self.uploads: list[str] = []
        self.metadata = {}

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        pages = [
            {"Contents": [{"Key": k, "Size": len(v)} for k, v in sorted(self.objects.items())]}
        ]
        return SimpleNamespace(
            paginate=lambda Bucket, Prefix: [
                {"Contents": [c for c in p["Contents"] if c["Key"].startswith(Prefix)]}
                for p in pages
            ]
        )

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {"Body": io.BytesIO(self.objects[Key])}

    def head_object(self, Bucket, Key):
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
        return {"ContentLength": len(self.objects[Key]), "Metadata": self.metadata.get(Key, {})}

    def put_object(self, Bucket, Key, Body, **kwargs):
        self.objects[Key] = Body

    def upload_file(self, filename, bucket, key, ExtraArgs=None):
        with open(filename, "rb") as f:
            self.objects[key] = f.read()
        self.uploads.append(key)
        self.metadata[key] = ExtraArgs["Metadata"]


class FakeBatch:
    def __init__(self, items):
        self.items = items

    def get(self):
        return iter(self.items)


class FakeAIS:
    """Returns canned GetBatch results, optionally dropping entries to mimic a short stream."""

    def __init__(self, objects: dict[str, bytes], drop: set[str] = frozenset()):
        self.objects, self.drop = objects, drop

    def batch(self, names, bucket, cont_on_err):
        assert cont_on_err is False
        return FakeBatch(
            [(SimpleNamespace(obj_name=n), self.objects[n]) for n in names if n not in self.drop]
        )


DATA = {
    "ds/meta/info.json": b"{}",
    "ds/data/chunk-000/episode_000000.parquet": b"p" * 10,
    "ds/videos/chunk-000/front/episode_000000.mp4": b"v" * 20,
    "ds/dir-marker/": b"",
    "other/x": b"x",
}


def infos(prefix="ds/"):
    return storage.list_objects(FakeS3(DATA), "b", prefix)


def test_list_objects_filters_prefix_and_dir_markers():
    keys = [o.key for o in infos()]
    assert "ds/dir-marker/" not in keys and "other/x" not in keys
    assert len(keys) == 3


@pytest.mark.parametrize(
    "key,expected",
    [("ds/a/b.mp4", "a/b.mp4"), ("ds/meta/info.json", "meta/info.json")],
)
def test_local_path_maps_relative_to_prefix(tmp_path, key, expected):
    assert storage.local_path(tmp_path, key, "ds/") == tmp_path / expected


@pytest.mark.parametrize("key", ["ds/../escape", "ds/a/../../b", "ds/", "../../etc/passwd"])
def test_local_path_rejects_traversal(tmp_path, key):
    with pytest.raises(ValueError):
        storage.local_path(tmp_path, key, "ds/")


def test_download_objects_writes_identical_files(tmp_path):
    lat = storage.download_objects(FakeS3(DATA), "b", infos(), tmp_path, "ds/", workers=4)
    assert len(lat) == 3
    assert (tmp_path / "videos/chunk-000/front/episode_000000.mp4").read_bytes() == b"v" * 20
    assert not list(tmp_path.rglob("*.part"))


def test_download_detects_size_mismatch(tmp_path):
    stale = [ObjectInfo("ds/meta/info.json", 999)]
    with pytest.raises(OSError, match="size mismatch"):
        storage.download_objects(FakeS3(DATA), "b", stale, tmp_path, "ds/")


def test_getbatch_download_batches_and_writes(tmp_path):
    lat = storage.getbatch_download(
        FakeAIS(DATA), None, infos(), tmp_path, "ds/", workers=2, batch_size=2
    )
    assert len(lat) == 2  # 3 objects in batches of 2
    assert (tmp_path / "data/chunk-000/episode_000000.parquet").read_bytes() == b"p" * 10


def test_getbatch_short_stream_is_an_error(tmp_path):
    # A streaming GetBatch over uncached objects can end early without an HTTP error.
    ais = FakeAIS(DATA, drop={"ds/meta/info.json"})
    with pytest.raises(OSError):
        storage.getbatch_download(ais, None, infos(), tmp_path, "ds/", batch_size=10)


def test_upload_dir_prefixes_keys_and_skips_existing(tmp_path):
    (tmp_path / "checkpoint-500").mkdir()
    (tmp_path / "checkpoint-500" / "model.safetensors").write_bytes(b"w" * 8)
    (tmp_path / "config.json").write_bytes(b"{}")
    s3 = FakeS3({})
    assert storage.upload_dir(s3, tmp_path, "b", "checkpoints/run/") == (2, 0)
    assert sorted(s3.uploads) == [
        "checkpoints/run/checkpoint-500/model.safetensors",
        "checkpoints/run/config.json",
    ]
    assert storage.upload_dir(s3, tmp_path, "b", "checkpoints/run") == (0, 2)
    (tmp_path / "config.json").write_bytes(b'{"changed": 1}')
    assert storage.upload_dir(s3, tmp_path, "b", "checkpoints/run") == (1, 1)


class FakeBucket:
    """Cached set grows by `per_job` keys per prefetch job (mimics racing nodes)."""

    def __init__(self, per_job: int):
        self.cached: set[str] = set()
        self.per_job = per_job
        self.requested: list[list[str]] = []

    def list_objects_iter(self, prefix, props, flags):
        return [SimpleNamespace(name=k) for k in self.cached if k.startswith(prefix)]

    def objects(self, obj_names):
        def prefetch(num_workers, continue_on_error):
            assert continue_on_error is True
            self.requested.append(obj_names)
            self.cached.update(obj_names[: self.per_job])
            return "job-id"

        return SimpleNamespace(prefetch=prefetch)


@pytest.fixture
def no_job_wait(monkeypatch):
    monkeypatch.setattr(
        storage, "_wait", lambda bucket, job_id: SimpleNamespace(success=False, error="skip")
    )


def test_prefetch_retries_missing_keys_once_then_fails_clearly(no_job_wait):
    keys = [f"ds/{i}" for i in range(3)]
    bucket = FakeBucket(per_job=2)  # a racing node leaves one object uncached per job
    storage.prefetch(bucket, keys)
    assert [len(r) for r in bucket.requested] == [3, 1]

    with pytest.raises(RuntimeError, match="not cached after prefetch"):
        storage.prefetch(FakeBucket(per_job=0), ["ds/a"])


def test_list_directory_prefix_does_not_include_siblings():
    objects = {"ds/a": b"a", "ds-old/b": b"b"}
    assert [o.key for o in storage.list_objects(FakeS3(objects), "b", "ds")] == ["ds/a"]


@pytest.mark.parametrize("key", ["x/y", "ds//a", "ds/./a", "ds/a\\b", "/absolute"])
def test_local_path_rejects_aliases_and_outside_prefix(tmp_path, key):
    with pytest.raises(ValueError):
        storage.local_path(tmp_path, key, "ds")


def test_symlink_parent_cannot_redirect_download(tmp_path):
    outside, stage = tmp_path / "outside", tmp_path / "stage"
    outside.mkdir()
    stage.mkdir()
    (stage / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        storage.download_objects(
            FakeS3({"ds/link/x": b"x"}), "b", [ObjectInfo("ds/link/x", 1)], stage, "ds"
        )
    assert not (outside / "x").exists()


def test_stage_rejects_stale_directory_without_modifying_it(tmp_path):
    (tmp_path / "old.parquet").write_bytes(b"old")
    with pytest.raises(ValueError, match="absent or empty"):
        storage.stage(None, "ds", tmp_path, "get")
    assert (tmp_path / "old.parquet").read_bytes() == b"old"


def test_failed_stage_never_publishes_partial_directory(tmp_path, monkeypatch):
    data = {"ds/a": b"a", "ds/b": b"b"}
    s3 = FakeS3(data)
    monkeypatch.setattr(storage, "ais_s3_client", lambda cfg: s3)
    cfg = SimpleNamespace(b2_bucket="b")
    original = s3.get_object

    def fail(Bucket, Key):
        if Key == "ds/b":
            raise ConnectionError("connection lost")
        return original(Bucket, Key)

    s3.get_object = fail
    dest = tmp_path / "dest"
    with pytest.raises(ConnectionError):
        storage.stage(cfg, "ds", dest, "get")
    assert not dest.exists()
    assert not list(tmp_path.glob(".dest.stage-*"))
    s3.get_object = original
    storage.stage(cfg, "ds", dest, "get")
    assert sorted(p.name for p in dest.iterdir()) == [MANIFEST, "a", "b"]


def test_manifest_excludes_orphans_and_detects_same_size_corruption(tmp_path, monkeypatch):
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "files": {"a": {"size": 3, "sha256": hashlib.sha256(b"new").hexdigest()}},
    }
    s3 = FakeS3(
        {"ds/a": b"old", "ds/orphan": b"x", "ds/" + MANIFEST: json.dumps(manifest).encode()}
    )
    monkeypatch.setattr(storage, "ais_s3_client", lambda cfg: s3)
    cfg = SimpleNamespace(b2_bucket="b")
    dest = tmp_path / "dest"
    with pytest.raises(OSError, match="checksum mismatch"):
        storage.stage(cfg, "ds", dest, "get")
    assert not dest.exists()
    s3.objects["ds/a"] = b"new"
    result = storage.stage(cfg, "ds", dest, "get")
    assert result.objects == 1
    assert not (dest / "orphan").exists()


def test_pending_dataset_manifest_blocks_staging(tmp_path, monkeypatch):
    manifest = {"schema_version": 1, "status": "pending", "files": {}}
    s3 = FakeS3({"ds/" + MANIFEST: json.dumps(manifest).encode()})
    monkeypatch.setattr(storage, "ais_s3_client", lambda cfg: s3)
    with pytest.raises(ValueError, match="incomplete"):
        storage.stage(SimpleNamespace(b2_bucket="b"), "ds", tmp_path / "dest", "get")


@pytest.mark.parametrize(
    "workers,expected_batch,expected_prefetch", [(None, 4, 16), (1, 1, 1), (8, 8, 8)]
)
def test_stage_getbatch_workers_default_and_override(
    tmp_path, monkeypatch, workers, expected_batch, expected_prefetch
):
    s3 = FakeS3({"ds/a": b"a"})
    monkeypatch.setattr(storage, "ais_s3_client", lambda cfg: s3)
    monkeypatch.setattr(storage, "ais_client", lambda cfg: FakeAIS(s3.objects))
    monkeypatch.setattr(storage, "ais_bucket", lambda client, cfg: None)
    calls = {}

    def prefetch(bucket, keys, num_workers):
        calls["prefetch"] = num_workers

    timed_map = storage._timed_map

    def capture_map(fn, items, workers):
        calls["transfer"] = workers
        return timed_map(fn, items, workers)

    monkeypatch.setattr(storage, "prefetch", prefetch)
    monkeypatch.setattr(storage, "_timed_map", capture_map)
    storage.stage(SimpleNamespace(b2_bucket="b"), "ds", tmp_path / "dest", workers=workers)
    assert calls == {"prefetch": expected_prefetch, "transfer": expected_batch}
    assert (tmp_path / "dest" / "a").read_bytes() == b"a"


@pytest.mark.parametrize("direct", [False, True])
def test_direct_and_ais_get_staging_verify_identically(tmp_path, monkeypatch, direct):
    s3 = FakeS3({"ds/a": b"new"})
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "files": {"a": {"size": 3, "sha256": hashlib.sha256(b"new").hexdigest()}},
    }
    s3.objects["ds/" + MANIFEST] = json.dumps(manifest).encode()
    monkeypatch.setattr(storage, "ais_s3_client", lambda cfg: s3)
    monkeypatch.setattr(storage, "b2_client", lambda cfg: s3)
    original = s3.get_object
    reads = []

    def capture_read(Bucket, Key):
        reads.append(Key)
        return original(Bucket, Key)

    s3.get_object = capture_read
    result = storage.stage(
        SimpleNamespace(b2_bucket="b"), "ds", tmp_path / "dest", "get", direct=direct
    )
    local = json.loads((tmp_path / "dest" / MANIFEST).read_text())
    assert result.objects == 1 and result.bytes == 3
    assert local["files"] == manifest["files"]
    assert local["provenance"]["remote_manifest_verified"] is True
    assert result.manifest_sha256 == manifest_sha256(manifest)
    assert reads == ["ds/" + MANIFEST, "ds/a", "ds/" + MANIFEST]


def test_direct_staging_rejects_getbatch_before_client_creation(tmp_path):
    with pytest.raises(ValueError, match="direct staging requires get"):
        storage.stage(None, "ds", tmp_path / "dest", direct=True)


def test_sibling_prefix_is_not_staged(tmp_path, monkeypatch):
    """`ds5` must never pick up `ds50/...` (the prefix is a directory, not a string match)."""
    s3 = FakeS3({"ds5/a": b"a", "ds50/b": b"b"})
    monkeypatch.setattr(storage, "ais_s3_client", lambda cfg: s3)
    result = storage.stage(SimpleNamespace(b2_bucket="b"), "ds5", tmp_path / "dest", "get")
    assert result.objects == 1
    assert result.manifest_sha256 is None  # unpublished datasets have no lineage identity
    assert {p.name for p in (tmp_path / "dest").iterdir()} == {"a", MANIFEST}


# Shape of Cluster.get_performance(): {target_id: NodeStatus JSON}; AIS omits zero counters.
PERF = {
    "t1": {"tracker": {"aws.get.n": 3, "aws.get.size": 300, "get.n": 9}},
    "t2": {"tracker": {"aws.get.n": 2, "aws.get.size": 50}},
    "t3": {"tracker": {"get.n": 1}},
}


def test_backend_get_stats_sums_aws_counters_across_targets():
    assert storage.backend_get_stats(PERF) == (5, 350)


@pytest.mark.parametrize("perf", [{}, {"t1": {"status": "online"}}])
def test_backend_get_stats_rejects_unexpected_payload(perf):
    with pytest.raises(ValueError, match="AIS performance payload"):
        storage.backend_get_stats(perf)


def test_ais_backend_gets_reads_cluster_performance(monkeypatch):
    cluster = SimpleNamespace(get_performance=lambda: PERF)
    monkeypatch.setattr(storage, "ais_client", lambda cfg: SimpleNamespace(cluster=lambda: cluster))
    assert storage.ais_backend_gets(None) == (5, 350)


def test_b2_client_requires_credentials(fake_env, monkeypatch):
    monkeypatch.delenv("B2_APP_KEY")
    cfg = Config.from_env(require_b2_credentials=False)
    monkeypatch.setattr(storage.boto3, "client", lambda *a, **k: pytest.fail("AWS chain"))
    with pytest.raises(ValueError, match="B2_APP_KEY"):
        storage.b2_client(cfg)


def test_ais_bucket_uses_canonical_aws_provider():
    calls = []
    client = SimpleNamespace(bucket=lambda name, provider: calls.append((name, provider)))
    storage.ais_bucket(client, SimpleNamespace(b2_bucket="b"))
    assert calls == [("b", "aws")]

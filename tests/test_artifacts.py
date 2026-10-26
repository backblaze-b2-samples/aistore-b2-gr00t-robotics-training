# SPDX-FileCopyrightText: Copyright (c) 2026 Backblaze, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Publication must survive retries without declaring partial or changing files complete."""

import hashlib
import io
import json
import os
from pathlib import Path

import pytest
from botocore.exceptions import ClientError

from aistore_b2_gr00t import artifacts
from aistore_b2_gr00t.artifacts import (
    MANIFEST,
    manifest_sha256,
    read_manifest,
    upload_tree,
    verify_remote,
)


class Store:
    def __init__(self):
        self.data = {}
        self.metadata = {}
        self.fail = False
        self.on_upload = None  # called with the local path after each upload

    def head_object(self, Bucket, Key):
        if Key not in self.data:
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
        return {"ContentLength": len(self.data[Key]), "Metadata": self.metadata.get(Key, {})}

    def put_object(self, Bucket, Key, Body, **kwargs):
        self.data[Key] = Body

    def get_object(self, Bucket, Key):
        if Key not in self.data:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {"Body": io.BytesIO(self.data[Key])}

    def upload_file(self, filename, bucket, key, ExtraArgs):
        if self.fail:
            raise ConnectionError("backend unavailable")
        self.data[key] = Path(filename).read_bytes()
        self.metadata[key] = ExtraArgs["Metadata"]
        if self.on_upload:
            self.on_upload(Path(filename))

    def status(self):
        return json.loads(self.data["run/" + MANIFEST])["status"]


def test_failed_republication_invalidates_old_marker_and_retry_completes(tmp_path):
    path = tmp_path / "model.bin"
    path.write_bytes(b"old")
    store = Store()
    upload_tree(store, tmp_path, "b", "run", publish=True)
    assert store.status() == "complete"
    path.write_bytes(b"new")  # equal size, different content
    store.fail = True
    with pytest.raises(ConnectionError):
        upload_tree(store, tmp_path, "b", "run", publish=True)
    assert store.status() == "pending"
    store.fail = False
    assert upload_tree(store, tmp_path, "b", "run", publish=True) == (1, 0)
    assert store.data["run/model.bin"] == b"new"
    assert store.status() == "complete"
    assert upload_tree(store, tmp_path, "b", "run", publish=True) == (0, 1)


def _touch(path):
    """Advance mtime without changing size: a same-size rewrite as seen by stat."""
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))


def _replace_keeping_mtime(path):
    """Swap in same-size content and restore mtime, as `cp -p` or `rsync -t` would."""
    st = path.stat()
    tmp = path.with_name("replacement")
    tmp.write_bytes(b"new")
    os.utime(tmp, ns=(st.st_atime_ns, st.st_mtime_ns))
    tmp.replace(path)


@pytest.mark.parametrize(
    "change",
    [
        lambda p: p.write_bytes(b"changed after upload"),  # resized
        _touch,
        _replace_keeping_mtime,
        lambda p: p.unlink(),  # rotated away
        lambda p: (p.parent / "new.bin").write_bytes(b"x"),
        lambda p: (p.parent / "sub").mkdir(exist_ok=True) or (p.parent / "sub/new.bin").touch(),
    ],
    ids=["resized", "touched", "mtime-preserving-replace", "removed", "added", "added-nested"],
)
def test_changed_tree_is_not_marked_complete(tmp_path, change):
    (tmp_path / "model.bin").write_bytes(b"old")
    store = Store()
    store.on_upload = change
    with pytest.raises(OSError, match="changed during upload") as exc:
        upload_tree(store, tmp_path, "b", "run", publish=True)
    assert store.status() == "pending"
    assert ".bin" in str(exc.value) and "left pending; retry" in str(exc.value)


def test_changes_outside_the_artifact_are_not_false_positives(tmp_path):
    (tmp_path / "model.bin").write_bytes(b"old")
    (tmp_path / "checkpoint-1").mkdir()
    (tmp_path / "checkpoint-1" / "shard.bin").write_bytes(b"done")

    def change(_):
        (tmp_path / "checkpoint-1" / "shard.bin").write_bytes(b"rewritten")
        (tmp_path / "checkpoint-1" / "extra.bin").write_bytes(b"x")
        (tmp_path / MANIFEST).write_bytes(b"{}")  # local manifests are never artifact files

    store = Store()
    store.on_upload = change
    upload_tree(store, tmp_path, "b", "run", publish=True, exclude=("checkpoint-1",))
    assert store.status() == "complete"
    assert read_manifest(store, "b", "run")["files"].keys() == {"model.bin"}


def test_upload_hashes_each_file_once(tmp_path, monkeypatch):
    (tmp_path / "a.bin").write_bytes(b"a")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.bin").write_bytes(b"b")
    hashed = []
    real = artifacts.file_sha256
    monkeypatch.setattr(artifacts, "file_sha256", lambda p: hashed.append(p) or real(p))
    upload_tree(Store(), tmp_path, "b", "run", publish=True)
    assert sorted(hashed) == [tmp_path / "a.bin", tmp_path / "sub" / "b.bin"]


def test_manifest_sha256_matches_published_bytes(tmp_path):
    (tmp_path / "model.bin").write_bytes(b"new")
    store = Store()
    upload_tree(store, tmp_path, "b", "run", publish=True, provenance={"k": "v"})
    published = hashlib.sha256(store.data["run/" + MANIFEST]).hexdigest()
    assert manifest_sha256(read_manifest(store, "b", "run")) == published


def test_upload_refuses_symlinked_files(tmp_path):
    outside = tmp_path / "secret"
    outside.write_bytes(b"secret")
    root = tmp_path / "artifact"
    root.mkdir()
    (root / "link").symlink_to(outside)
    store = Store()
    with pytest.raises(ValueError, match="symlink"):
        upload_tree(store, root, "b", "run", publish=True)
    assert store.data == {}


def test_consumer_detects_corruption_and_rejects_pending(tmp_path):
    (tmp_path / "model.bin").write_bytes(b"new")
    store = Store()
    upload_tree(store, tmp_path, "b", "run", publish=True)
    assert verify_remote(store, "b", "run") == 1
    store.data["run/model.bin"] = b"old"
    with pytest.raises(OSError, match="integrity check"):
        verify_remote(store, "b", "run")
    store.data["run/" + MANIFEST] = json.dumps({"status": "pending"}).encode()
    with pytest.raises(ValueError, match="incomplete"):
        verify_remote(store, "b", "run")


def test_empty_artifact_is_not_published(tmp_path):
    store = Store()
    with pytest.raises(ValueError, match="empty artifact"):
        upload_tree(store, tmp_path, "b", "run", publish=True)
    assert store.data == {}


def test_complete_manifest_supports_gateways_without_hash_metadata(tmp_path):
    path = tmp_path / "model.bin"
    path.write_bytes(b"old")
    store = Store()
    upload_tree(store, tmp_path, "b", "run", publish=True)
    store.metadata.clear()  # Some gateways omit user metadata on HEAD.
    assert upload_tree(store, tmp_path, "b", "run", publish=True) == (0, 1)
    path.write_bytes(b"new")
    assert upload_tree(store, tmp_path, "b", "run", publish=True) == (1, 0)
    assert verify_remote(store, "b", "run") == 1


GOOD_ENTRY = {"size": 1, "sha256": "a" * 64}


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": 2},
        {"status": "pending"},
        {"files": {}},
        {"files": {"../escape.bin": GOOD_ENTRY}},
        {"files": {"a/./b.bin": GOOD_ENTRY}},
        {"files": {"a//b.bin": GOOD_ENTRY}},
        {"files": {"/abs.bin": GOOD_ENTRY}},
        {"files": {"a\\b.bin": GOOD_ENTRY}},
        {"files": {MANIFEST: GOOD_ENTRY}},
        {"files": {"m.bin": {"size": -1, "sha256": "a" * 64}}},
        {"files": {"m.bin": {"size": True, "sha256": "a" * 64}}},
        {"files": {"m.bin": {"size": 1, "sha256": "a" * 63}}},
        {"files": {"m.bin": {"size": 1, "sha256": "A" * 64}}},
    ],
)
def test_read_manifest_rejects_unsafe_or_incomplete_records(change):
    store = Store()
    manifest = {"schema_version": 1, "status": "complete", "files": {"m.bin": GOOD_ENTRY}}
    store.data["run/" + MANIFEST] = json.dumps({**manifest, **change}).encode()
    with pytest.raises(ValueError):
        read_manifest(store, "b", "run")

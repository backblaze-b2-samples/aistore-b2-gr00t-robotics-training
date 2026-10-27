# SPDX-FileCopyrightText: Copyright (c) 2026 Backblaze, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SHA-256 content identity and the directory publication contract.

Consumers accept only a manifest with status=complete and verify its file hashes.
One writer owns a prefix. Use immutable dataset revisions and unique run prefixes.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from botocore.exceptions import ClientError

MANIFEST = "_SUCCESS.json"


def read_manifest(s3: Any, bucket: str, prefix: str) -> dict[str, Any]:
    """Read and validate a publication record. Pending and unsafe records fail closed."""
    key = f"{prefix.strip('/')}/{MANIFEST}" if prefix.strip("/") else MANIFEST
    body = s3.get_object(Bucket=bucket, Key=key)["Body"]
    try:
        manifest = json.loads(body.read())
    finally:
        body.close()
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != 1
        or manifest.get("status") != "complete"
        or not manifest.get("files")
        or not isinstance(manifest["files"], dict)
    ):
        raise ValueError(f"artifact publication is incomplete or invalid: {key}")
    for rel, info in manifest["files"].items():
        if (
            not isinstance(rel, str)
            or "\\" in rel
            or any(p in ("", ".", "..") for p in rel.split("/"))
            or rel == MANIFEST
            or not isinstance(info, dict)
            or type(info.get("size")) is not int
            or info["size"] < 0
            or not isinstance(info.get("sha256"), str)
            or len(info["sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in info["sha256"])
        ):
            raise ValueError(f"invalid artifact manifest entry: {rel!r}")
    return manifest


def _manifest_bytes(manifest: dict[str, Any]) -> bytes:
    return json.dumps(manifest, sort_keys=True).encode()


def manifest_sha256(manifest: dict[str, Any]) -> str:
    """SHA-256 of the manifest re-serialized with sorted keys.

    Equals `sha256sum _SUCCESS.json` for manifests `upload_tree` wrote; a manifest written
    by another tool still gets a stable identity, but not its raw object hash.
    """
    return hashlib.sha256(_manifest_bytes(manifest)).hexdigest()


def verify_remote(
    s3: Any,
    bucket: str,
    prefix: str,
    *,
    progress: Callable[[int, int, int, int], None] | None = None,
) -> int:
    """Verify every object, then recheck the manifest.

    Optional progress receives (files_verified, total_files, bytes_read, total_bytes).
    Bytes include the current file; files count only after size and SHA-256 match.
    """
    manifest = read_manifest(s3, bucket, prefix)
    total_files = len(manifest["files"])
    total_bytes = sum(info["size"] for info in manifest["files"].values())
    files_verified = bytes_read = 0
    if progress is not None:
        progress(files_verified, total_files, bytes_read, total_bytes)
    prefix = prefix.strip("/")
    for rel, expected in manifest["files"].items():
        key = f"{prefix}/{rel}" if prefix else rel
        body = s3.get_object(Bucket=bucket, Key=key)["Body"]
        digest, size = hashlib.sha256(), 0
        try:
            for block in iter(lambda body=body: body.read(1024 * 1024), b""):
                size += len(block)
                digest.update(block)
                bytes_read += len(block)
                if progress is not None:
                    progress(files_verified, total_files, bytes_read, total_bytes)
        finally:
            body.close()
        if size != expected["size"] or digest.hexdigest() != expected["sha256"]:
            raise OSError(f"artifact integrity check failed: {key}")
        files_verified += 1
        if progress is not None and files_verified < total_files:
            progress(files_verified, total_files, bytes_read, total_bytes)
    if read_manifest(s3, bucket, prefix) != manifest:
        raise OSError("artifact publication changed during verification")
    if progress is not None:
        progress(files_verified, total_files, bytes_read, total_bytes)
    return total_files


def reject_symlinks(path: Path) -> None:
    """Reject existing symlinks in a path, including its ancestors."""
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError(f"refusing symlink path: {path}")


def file_sha256(path: Path) -> str:
    reject_symlinks(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _walk_files(root: Path, exclude: tuple[str, ...]) -> Iterator[tuple[str, Path]]:
    """Yield (posix relpath, path) for every publishable file under root, sorted.

    Shared by the hashing pass and the post-upload change check so both see the same
    file set.
    """
    reject_symlinks(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if rel.parts[0] in exclude:
            continue
        reject_symlinks(path)
        if not path.is_file() or path.name == MANIFEST:
            continue
        if "\\" in rel.as_posix():
            raise ValueError(f"refusing nonportable artifact path: {rel}")
        yield rel.as_posix(), path


def _fingerprint(path: Path) -> tuple[int, int, int, int]:
    """Cheap change fingerprint that reads no file bytes.

    ctime and inode catch replacements that restore mtime (`cp -p`, `rsync -t`, rename).
    """
    st = path.stat()
    return st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino


def _stat_tree(root: Path, exclude: tuple[str, ...]) -> dict[str, tuple[int, int, int, int]]:
    return {rel: _fingerprint(path) for rel, path in _walk_files(root, exclude)}


def _hash_tree(
    root: Path, exclude: tuple[str, ...]
) -> tuple[dict[str, dict[str, Any]], dict[str, tuple[int, int, int, int]]]:
    """Hash every file once; also return the fingerprint each hash was taken under."""
    files, stats = {}, {}
    for rel, path in _walk_files(root, exclude):
        before = _fingerprint(path)
        digest = file_sha256(path)
        if _fingerprint(path) != before:
            raise OSError(f"file changed while hashing: {path}")
        files[rel] = {"size": before[0], "sha256": digest}
        stats[rel] = before
    return files, stats


def tree_manifest(root: Path, exclude: tuple[str, ...] = ()) -> dict[str, dict[str, Any]]:
    return _hash_tree(root, exclude)[0]


def upload_tree(
    s3: Any,
    root: Path,
    bucket: str,
    prefix: str,
    *,
    exclude: tuple[str, ...] = (),
    workers: int = 8,
    skip_existing: bool = True,
    transfer_config: Any = None,
    publish: bool = False,
    provenance: dict[str, Any] | None = None,
) -> tuple[int, int]:
    """Upload stable files with SHA-256 metadata; optionally publish a manifest last.

    Equal sizes alone never justify a skip. Files without matching hash metadata or a
    complete prior manifest are uploaded again. A pending manifest invalidates any
    previous publication before data changes; failure leaves it pending.

    Files are hashed once. Change during upload is detected by re-walking the tree and
    comparing paths and stat fingerprints (size, mtime, ctime, inode), not bytes; an
    in-place rewrite that leaves all four unchanged goes unnoticed, as in the hashing pass.
    """
    if workers < 1:
        raise ValueError("workers must be positive")
    prefix = prefix.strip("/")
    files, stats = _hash_tree(root, exclude)
    if publish and not files:
        raise ValueError(f"refusing to publish an empty artifact: {root}")

    def key(rel: str) -> str:
        return f"{prefix}/{rel}" if prefix else rel

    manifest_key = key(MANIFEST)
    previous = {}
    if publish and skip_existing:
        try:
            previous = read_manifest(s3, bucket, prefix)["files"]
        except ClientError as exc:
            if exc.response["Error"]["Code"] not in ("404", "NoSuchKey", "NotFound"):
                raise
        except ValueError:
            # A pending/invalid publication never establishes prior content identity.
            pass

    def put_manifest(status: str) -> None:
        body = {
            "schema_version": 1,
            "status": status,
            "files": files if status == "complete" else {},
            "provenance": provenance or {},
        }
        s3.put_object(
            Bucket=bucket,
            Key=manifest_key,
            Body=_manifest_bytes(body),
            ContentType="application/json",
        )

    if publish:
        put_manifest("pending")

    def upload(rel: str) -> bool:
        expected = files[rel]
        if skip_existing:
            try:
                head = s3.head_object(Bucket=bucket, Key=key(rel))
            except ClientError as exc:
                if exc.response["Error"]["Code"] not in ("404", "NoSuchKey", "NotFound"):
                    raise
            else:
                if head["ContentLength"] == expected["size"] and (
                    head.get("Metadata", {}).get("sha256") == expected["sha256"]
                    or previous.get(rel) == expected
                ):
                    return False
        kwargs = {"ExtraArgs": {"Metadata": {"sha256": expected["sha256"]}}}
        if transfer_config is not None:
            kwargs["Config"] = transfer_config
        s3.upload_file(str(root / rel), bucket, key(rel), **kwargs)
        return True

    with ThreadPoolExecutor(max_workers=workers) as pool:
        uploaded = sum(pool.map(upload, files))
    # Detect rotation, mutation, and new files before declaring the directory complete.
    current = _stat_tree(root, exclude)
    if current != stats:
        changed = sorted(
            rel for rel in stats.keys() | current.keys() if stats.get(rel) != current.get(rel)
        )
        more = f" and {len(changed) - 3} more" if len(changed) > 3 else ""
        pending = f"; {MANIFEST} left pending" if publish else ""
        raise OSError(
            f"artifact changed during upload: {root} ({', '.join(changed[:3])}{more}){pending}; "
            "retry once the directory is stable"
        )
    if publish:
        put_manifest("complete")
    return uploaded, len(files) - uploaded

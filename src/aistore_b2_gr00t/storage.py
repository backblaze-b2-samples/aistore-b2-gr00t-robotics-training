# SPDX-FileCopyrightText: Copyright (c) 2026 Backblaze, Inc.
# SPDX-License-Identifier: Apache-2.0
"""S3/AIStore clients plus the stage (B2 -> node-local disk) and sync (disk -> B2) helpers.

Two data paths reach the same B2 bucket:
  * direct:  boto3 -> B2 S3 endpoint (baseline: one WAN round trip and B2 GET per object per node)
  * via AIS: AIStore `s3://$B2_BUCKET` caches objects on first read and writes PUTs
             through to B2, so repeat jobs and additional nodes read from the cluster.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import boto3
from aistore.sdk import Client
from aistore.sdk.bucket import Bucket
from aistore.sdk.job import Job
from aistore.sdk.list_object_flag import ListObjectFlag
from aistore.sdk.wait_result import WaitResult
from botocore.client import BaseClient as S3Client
from botocore.config import Config as BotoConfig

from aistore_b2_gr00t.artifacts import (
    MANIFEST,
    manifest_sha256,
    read_manifest,
    reject_symlinks,
    tree_manifest,
    upload_tree,
)
from aistore_b2_gr00t.config import Config

StageMethod = Literal["getbatch", "get"]

# Placeholder credentials for AIS: botocore insists on signing, AIS (no AuthN) ignores it.
AIS_ANON_USER = "ais-anonymous"

# Above the default 16 workers so threaded GETs don't queue on connections.
_MAX_POOL = 64
# Upper bound for one AIS prefetch/evict job. The SDK polls every timeout/64 s (~28 s).
_JOB_TIMEOUT_S = 30 * 60


@dataclass(frozen=True)
class ObjectInfo:
    key: str
    size: int
    sha256: str | None = None


@dataclass
class StageResult:
    objects: int
    bytes: int
    seconds: float
    # Per-request latencies: one per object for "get", one per batch for "getbatch".
    latencies: list[float] = field(default_factory=list)
    # Identity of the dataset's published manifest, for lineage; None for unpublished datasets.
    manifest_sha256: str | None = None


def b2_client(cfg: Config) -> S3Client:
    """boto3 client straight to B2 (seeding and the benchmark baseline)."""
    if not cfg.has_b2_credentials:
        raise ValueError("B2_KEY_ID and B2_APP_KEY are required for direct B2 access")
    return boto3.client(
        "s3",
        endpoint_url=cfg.b2_endpoint,
        region_name=cfg.b2_region,
        aws_access_key_id=cfg.b2_key_id,
        aws_secret_access_key=cfg.b2_app_key,
        config=BotoConfig(max_pool_connections=_MAX_POOL, retries={"mode": "adaptive"}),
    )


def ais_s3_client(cfg: Config) -> S3Client:
    """boto3 client to AIStore's S3 API.

    AIS (without AuthN) ignores client credentials and signs backend requests with the
    B2 key it holds, so placeholder values are used here. Path-style addressing is required
    because the bucket lives under `/s3/<bucket>`; the cluster runs with the
    `S3-Reverse-Proxy` feature so unpatched boto3 never sees an HTTP redirect.
    """
    return boto3.client(
        "s3",
        endpoint_url=cfg.ais_s3_endpoint,
        region_name=cfg.b2_region,
        aws_access_key_id=AIS_ANON_USER,
        aws_secret_access_key=AIS_ANON_USER,
        config=BotoConfig(
            max_pool_connections=_MAX_POOL,
            retries={"mode": "standard"},
            s3={"addressing_style": "path"},
        ),
    )


def ais_client(cfg: Config) -> Client:
    return Client(cfg.ais_endpoint, max_pool_size=_MAX_POOL)


def ais_bucket(client: Client, cfg: Config) -> Bucket:
    """The AIS view of the B2 bucket (attached by scripts/attach_b2_bucket.sh)."""
    return client.bucket(cfg.b2_bucket, provider="aws")


# AIS target counters for GETs sent to the `aws` (B2) backend: cold GETs and prefetch.
# S3 CopyObject does not record backend GET stats.
BACKEND_GET_COUNT = "aws.get.n"
BACKEND_GET_SIZE = "aws.get.size"


def backend_get_stats(perf: Mapping[str, Mapping]) -> tuple[int, int]:
    """Total (GET count, bytes) AIS fetched from B2, summed over a get_performance() payload.

    Counters are cumulative per target and AIS omits zero values, so a missing counter is 0.
    No targets, or a target without a `tracker`, is payload drift, not "no reads", and raises.
    """
    if not perf or any(not isinstance(node.get("tracker"), Mapping) for node in perf.values()):
        raise ValueError("unexpected AIS performance payload: no targets or no 'tracker' stats")
    trackers = [node["tracker"] for node in perf.values()]
    return (
        sum(t.get(BACKEND_GET_COUNT, 0) for t in trackers),
        sum(t.get(BACKEND_GET_SIZE, 0) for t in trackers),
    )


def ais_backend_gets(cfg: Config) -> tuple[int, int]:
    """Snapshot of backend GET (count, bytes); diff two snapshots to measure B2 reads."""
    return backend_get_stats(ais_client(cfg).cluster().get_performance())


def list_objects(s3: S3Client, bucket: str, prefix: str) -> list[ObjectInfo]:
    """List every object under `prefix` (zero-byte "directory" markers are skipped)."""
    out: list[ObjectInfo] = []
    prefix = prefix.strip("/") + "/" if prefix.strip("/") else ""
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if not obj["Key"].endswith("/"):
                out.append(ObjectInfo(obj["Key"], obj["Size"]))
    return out


def dataset_listing(s3: S3Client, bucket: str, prefix: str) -> tuple[list[ObjectInfo], dict | None]:
    """Resolve the same authoritative dataset for staging and benchmark baselines."""
    prefix = prefix.strip("/") + "/" if prefix.strip("/") else ""
    objects = list_objects(s3, bucket, prefix)
    if not any(o.key == prefix + MANIFEST for o in objects):
        return objects, None
    manifest = read_manifest(s3, bucket, prefix)
    available = {o.key: o.size for o in objects}
    selected = []
    for rel, info in manifest["files"].items():
        key = prefix + rel
        if available.get(key) != info["size"]:
            raise ValueError(f"missing or wrong-size dataset object: {key}")
        selected.append(ObjectInfo(key, info["size"], info["sha256"]))
    return selected, manifest


def local_path(dest: Path, key: str, prefix: str) -> Path:
    """Map an object key to a file under `dest`, refusing keys that escape it."""
    prefix = prefix.strip("/") + "/" if prefix.strip("/") else ""
    if not key.startswith(prefix):
        raise ValueError(f"object key {key!r} is outside prefix {prefix!r}")
    rel = key[len(prefix) :]
    parts = rel.split("/")
    if not rel or any(p in ("..", ".", "") for p in parts) or "\\" in rel:
        raise ValueError(f"refusing unsafe object key {key!r}")
    path = dest.joinpath(*parts)
    reject_symlinks(path)
    return path


def _write_atomic(path: Path, data: bytes) -> None:
    # Rename-into-place so an interrupted stage never leaves a truncated file behind.
    reject_symlinks(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _timed_map(fn: Callable[[object], None], items: Iterable[object], workers: int) -> list[float]:
    """Run `fn` over `items` on a thread pool; return per-call latencies (seconds)."""

    def run(item: object) -> float:
        t0 = time.perf_counter()
        fn(item)
        return time.perf_counter() - t0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(run, items))


def download_objects(
    s3: S3Client,
    bucket: str,
    objects: Sequence[ObjectInfo],
    dest: Path,
    prefix: str,
    workers: int = 16,
) -> list[float]:
    """Parallel GET of `objects` into `dest`, from B2 directly or through AIS.

    Through AIS, a cache miss triggers a cold GET from B2."""

    def fetch(obj: ObjectInfo) -> None:
        path = local_path(dest, obj.key, prefix)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".part")
        body = None
        try:
            body = s3.get_object(Bucket=bucket, Key=obj.key)["Body"]
            size, digest = 0, hashlib.sha256()
            with os.fdopen(fd, "wb") as stream:
                fd = -1
                for block in iter(lambda: body.read(1024 * 1024), b""):
                    size += len(block)
                    digest.update(block)
                    stream.write(block)
            if size != obj.size:
                raise OSError(f"size mismatch for {obj.key}: expected {obj.size}, got {size}")
            if obj.sha256 and digest.hexdigest() != obj.sha256:
                raise OSError(f"checksum mismatch for {obj.key}; use an immutable dataset prefix")
            os.replace(name, path)
        finally:
            if fd != -1:
                os.close(fd)
            if body is not None:
                body.close()
            Path(name).unlink(missing_ok=True)

    return _timed_map(fetch, objects, workers)


def _wait(bucket: Bucket, job_id: str) -> WaitResult:
    return Job(bucket.client, job_id).wait(timeout=_JOB_TIMEOUT_S, verbose=False)


def cached_keys(bucket: Bucket, prefix: str) -> set[str]:
    """Keys under `prefix` currently stored in the AIS cluster (no backend round trip)."""
    entries = bucket.list_objects_iter(prefix=prefix, props="name", flags=[ListObjectFlag.CACHED])
    return {e.name for e in entries}


def _cached_subset(bucket: Bucket, keys: Sequence[str]) -> set[str]:
    """Which of `keys` are cached, via one listing of their common prefix."""
    if not keys:
        return set()
    # S3 keys use character prefixes, including partial filename matches.
    prefix = os.path.commonprefix(list(keys))  # noqa: RUF071
    return set(keys) & cached_keys(bucket, prefix)


def prefetch(bucket: Bucket, keys: Sequence[str], num_workers: int = 16) -> None:
    """Have the AIS cluster pull `keys` from B2, returning once all of them are cached.

    When several nodes stage the same prefix at once, their prefetch jobs race and AIS
    skips objects another job is already fetching. Completion is therefore checked against
    the cached listing rather than the job result; still-missing keys get one retry.

    `num_workers` is raised from the AIS default (one reader per mountpath, i.e. one in
    the compose deployment) so WAN round trips to B2 overlap.
    """
    for _ in range(2):
        missing = sorted(set(keys) - _cached_subset(bucket, keys))
        if not missing:
            return
        job_id = bucket.objects(obj_names=missing).prefetch(
            num_workers=num_workers, continue_on_error=True
        )
        _wait(bucket, job_id)
    if missing := sorted(set(keys) - _cached_subset(bucket, keys)):
        raise RuntimeError(f"{len(missing)} object(s) not cached after prefetch: {missing[:5]}")


def evict(bucket: Bucket, keys: Sequence[str]) -> None:
    """Drop `keys` from the AIS cache; B2 is untouched. Used to measure cold staging.

    AIS fails an evict job on objects that are not cached, so only cached keys are sent.
    """
    if cached := sorted(_cached_subset(bucket, keys)):
        result = _wait(bucket, bucket.objects(obj_names=cached).evict())
        if not result.success:
            raise RuntimeError(f"AIS evict job failed: {result.error or 'aborted'}")


def getbatch_download(
    client: Client,
    bucket: Bucket,
    objects: Sequence[ObjectInfo],
    dest: Path,
    prefix: str,
    workers: int = 4,
    batch_size: int = 32,
) -> list[float]:
    """Stage cached objects with AIS GetBatch: one ordered TAR stream per `batch_size` objects.

    GetBatch reads in-cluster data only (no cold GET), so call `prefetch` first. The
    streaming response carries no per-item status, so every batch is checked for count,
    order, and size; a missing object surfaces as an error rather than a short dataset.
    """

    def fetch(batch: Sequence[ObjectInfo]) -> None:
        req = client.batch([o.key for o in batch], bucket, cont_on_err=False)
        got = 0
        stream = iter(req.get())
        for obj in batch:
            try:
                meta, data = next(stream)
            except StopIteration:
                break
            if meta.obj_name != obj.key or len(data) != obj.size:
                raise OSError(
                    f"GetBatch mismatch: expected {obj.key} ({obj.size} B), "
                    f"got {meta.obj_name} ({len(data)} B)"
                )
            if obj.sha256 and hashlib.sha256(data).hexdigest() != obj.sha256:
                raise OSError(f"checksum mismatch for {obj.key}; use an immutable dataset prefix")
            _write_atomic(local_path(dest, obj.key, prefix), data)
            got += 1
        if got != len(batch):
            raise OSError(
                f"GetBatch returned {got}/{len(batch)} objects starting at {batch[0].key}; "
                "objects must be cached in AIS; run prefetch first"
            )
        if next(stream, None) is not None:
            raise OSError("GetBatch returned more objects than requested")

    batches = [objects[i : i + batch_size] for i in range(0, len(objects), batch_size)]
    return _timed_map(fetch, batches, workers)


def stage(
    cfg: Config,
    prefix: str,
    dest: Path,
    method: StageMethod = "getbatch",
    workers: int | None = None,
    *,
    direct: bool = False,
) -> StageResult:
    """Verify a dataset in a private temporary tree, then rename it into an empty `dest`.

    A remote publication manifest is authoritative when present. Datasets without
    one still get size checks and a local content manifest. `direct=True`
    uses B2 GETs with the same verification and publication checks as AIS staging.
    Default concurrency is 16 for GET/prefetch and 4 for GetBatch's buffered objects;
    an explicit `workers` value overrides both defaults.
    """
    if workers is not None and workers < 1:
        raise ValueError("workers must be positive")
    if method not in ("getbatch", "get") or (direct and method != "get"):
        raise ValueError("direct staging requires get; AIS staging requires get or getbatch")
    prefix = prefix.strip("/") + "/" if prefix.strip("/") else ""
    reject_symlinks(dest)
    if dest.exists() and (not dest.is_dir() or any(dest.iterdir())):
        raise ValueError(f"stage destination must be absent or empty: {dest}")
    s3 = b2_client(cfg) if direct else ais_s3_client(cfg)
    t0 = time.perf_counter()
    objects, remote_manifest = dataset_listing(s3, cfg.b2_bucket, prefix)
    if not objects:
        raise FileNotFoundError(f"no objects under s3://{cfg.b2_bucket}/{prefix}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(dir=dest.parent, prefix=f".{dest.name}.stage-"))
    try:
        if method == "getbatch":
            client = ais_client(cfg)
            bucket = ais_bucket(client, cfg)
            prefetch(bucket, [o.key for o in objects], num_workers=workers or 16)
            latencies = getbatch_download(
                client, bucket, objects, temporary, prefix, workers=workers or 4
            )
        else:
            latencies = download_objects(
                s3, cfg.b2_bucket, objects, temporary, prefix, workers or 16
            )
        files = tree_manifest(temporary)
        expected = {o.key[len(prefix) :]: o for o in objects}
        if set(files) != set(expected) or any(
            files[rel]["size"] != obj.size or (obj.sha256 and files[rel]["sha256"] != obj.sha256)
            for rel, obj in expected.items()
        ):
            raise OSError("staged tree does not match the dataset listing")
        manifest = {
            "schema_version": 1,
            "status": "complete",
            "files": files,
            "provenance": {
                "bucket": cfg.b2_bucket,
                "prefix": prefix,
                "remote_manifest_verified": remote_manifest is not None,
            },
        }
        _write_atomic(temporary / MANIFEST, json.dumps(manifest, sort_keys=True).encode())
        if (
            remote_manifest is not None
            and read_manifest(s3, cfg.b2_bucket, prefix) != remote_manifest
        ):
            raise OSError("dataset publication changed during staging")
        os.replace(temporary, dest)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return StageResult(
        objects=len(objects),
        bytes=sum(o.size for o in objects),
        seconds=time.perf_counter() - t0,
        latencies=latencies,
        manifest_sha256=None if remote_manifest is None else manifest_sha256(remote_manifest),
    )


def upload_dir(
    s3: S3Client,
    src: Path,
    bucket: str,
    prefix: str,
    workers: int = 8,
    skip_existing: bool = True,
    *,
    publish: bool = False,
    provenance: dict | None = None,
    exclude: tuple[str, ...] = (),
) -> tuple[int, int]:
    """Upload every file under `src` to `<prefix>/<relative path>`; return (uploaded, unchanged).

    Through AIS this is write-through: each PUT lands in B2 before it returns. boto3's
    managed transfer switches to multipart for large files (checkpoint shards).
    `skip_existing` requires both equal size and matching SHA-256. `publish` writes a
    pending manifest first and a complete manifest only after all files succeed.
    """
    return upload_tree(
        s3,
        src,
        bucket,
        prefix,
        workers=workers,
        skip_existing=skip_existing,
        publish=publish,
        provenance=provenance,
        exclude=exclude,
    )

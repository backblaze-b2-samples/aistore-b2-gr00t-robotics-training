# SPDX-FileCopyrightText: Copyright (c) 2026 Backblaze, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Time staging a dataset prefix to local disk: B2-direct vs AIS-cold vs AIS-warm.

`--stagers N` runs N concurrent stagers (each into its own directory) to approximate
N training nodes starting at once. Results go to --out-dir as JSON and a Markdown table.

  uv run python benchmarks/stage_time.py --prefix datasets/cube_to_bowl_5 --stagers 4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import socket
import statistics
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urlparse

from aistore_b2_gr00t.config import Config
from aistore_b2_gr00t.storage import (
    ais_backend_gets,
    ais_bucket,
    ais_client,
    b2_client,
    dataset_listing,
    evict,
    stage,
)

MODES = ("b2-direct", "ais-cold", "ais-warm")


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile; adequate for request-latency summaries."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(pct / 100 * len(ordered)) - 1))]


def tcp_rtt_ms(url: str, samples: int = 5) -> float | None:
    """Median TCP connect time to the URL's host:port (a cheap RTT proxy)."""
    parsed = urlparse(url)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    times = []
    for _ in range(samples):
        t0 = time.perf_counter()
        try:
            socket.create_connection((parsed.hostname, port), timeout=5).close()
        except OSError:
            return None
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def stage_once(cfg: Config, mode: str, prefix: str, dest: Path, method: str, workers: int | None):
    """One stager: returns (bytes, latencies)."""
    direct = mode == "b2-direct"
    result = stage(cfg, prefix, dest, "get" if direct else method, workers, direct=direct)
    return result.bytes, result.latencies


def run_mode(cfg: Config, args: argparse.Namespace, mode: str, keys: list[str]) -> dict:
    walls, latencies, total_bytes, all_bytes = [], [], 0, 0
    direct = mode == "b2-direct"
    if mode == "ais-warm":
        # Untimed fill so ais-warm is warm regardless of which modes ran before it.
        with tempfile.TemporaryDirectory(dir=args.scratch) as tmp:
            stage_once(cfg, mode, args.prefix, Path(tmp).resolve(), args.method, args.workers)
    # AIS modes: backend counters bracket the timed repeats (the warm fill is excluded).
    gets_before = None if direct else ais_backend_gets(cfg)
    for _ in range(args.repeats):
        if mode == "ais-cold":
            evict(ais_bucket(ais_client(cfg), cfg), keys)
        with tempfile.TemporaryDirectory(dir=args.scratch) as tmp:
            dests = [Path(tmp).resolve() / f"stager-{i}" for i in range(args.stagers)]
            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=args.stagers) as pool:
                outs = list(
                    pool.map(
                        lambda d: stage_once(cfg, mode, args.prefix, d, args.method, args.workers),
                        dests,
                    )
                )
            walls.append(time.perf_counter() - t0)
        total_bytes = sum(b for b, _ in outs)
        all_bytes += total_bytes
        latencies += [x for _, lat in outs for x in lat]
    if direct:
        # Bypasses AIS: count this client's data GETs (one latency per object).
        b2_gets, b2_bytes = len(latencies), all_bytes
    else:
        after = ais_backend_gets(cfg)
        b2_gets, b2_bytes = after[0] - gets_before[0], after[1] - gets_before[1]

    wall = statistics.median(walls)
    return {
        "mode": mode,
        "method": "get" if mode == "b2-direct" else args.method,
        "workers": args.workers or (16 if mode == "b2-direct" or args.method == "get" else 4),
        "prefetch_workers": (args.workers or 16)
        if mode != "b2-direct" and args.method == "getbatch"
        else 0,
        "stagers": args.stagers,
        "repeats": args.repeats,
        "objects": len(keys),
        "bytes_per_repeat": total_bytes,
        "wall_s_median": wall,
        "wall_s_all": walls,
        "aggregate_mb_s": total_bytes / 1e6 / wall if wall else 0.0,
        "request_latency_ms_p50": percentile(latencies, 50) * 1000,
        "request_latency_ms_p99": percentile(latencies, 99) * 1000,
        # Totals over all timed repeats. AIS modes: measured backend counters (aws.get.n /
        # aws.get.size), including cold manifest reads. b2-direct: client data GETs only.
        "b2_gets": b2_gets,
        "b2_get_bytes": b2_bytes,
    }


def to_markdown(report: dict) -> str:
    head = (
        "| mode | method | workers | prefetch workers | stagers | objects | MB/repeat "
        "| wall s (median) | agg MB/s "
        "| p50 ms | p99 ms | B2 GETs | B2 GET MB |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    rows = "".join(
        f"| {r['mode']} | {r['method']} | {r['workers']} | {r['prefetch_workers']} "
        f"| {r['stagers']} | {r['objects']} "
        f"| {r['bytes_per_repeat'] / 1e6:.1f} | {r['wall_s_median']:.2f} "
        f"| {r['aggregate_mb_s']:.1f} | {r['request_latency_ms_p50']:.1f} "
        f"| {r['request_latency_ms_p99']:.1f} | {r['b2_gets']} "
        f"| {r['b2_get_bytes'] / 1e6:.1f} |\n"
        for r in report["results"]
    )
    env = report["environment"]
    return (
        f"# Stage benchmark: `{report['prefix']}`\n\n{head}{rows}\n"
        f"Host: {env['platform']}, {env['cpus']} CPUs; RTT to B2 {env['rtt_ms_b2']} ms, "
        f"to AIS {env['rtt_ms_ais']} ms.\n"
        "p50/p99 are per request: one object for `get`, one batch for `getbatch`. "
        "B2 GETs are totals over all timed repeats: measured from AIS backend counters "
        "(`aws.get.n`, `aws.get.size`) for AIS modes, including cold manifest reads; "
        "b2-direct counts its own data GETs (manifest reads excluded). Listings and HEADs "
        "are not counted.\n"
        "Wall time includes listing, transfer, content verification, and local publication "
        "for every mode.\n"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "--prefix", default="datasets/cube_to_bowl_5", help="dataset prefix (default: %(default)s)"
    )
    ap.add_argument(
        "--modes",
        default=",".join(MODES),
        help="comma-separated subset of %(default)s (default: all)",
    )
    ap.add_argument(
        "--method",
        choices=["getbatch", "get"],
        default="getbatch",
        help="staging method for AIS modes (default: %(default)s)",
    )
    ap.add_argument(
        "--stagers",
        type=int,
        default=1,
        help="concurrent stagers, one per simulated node (default: %(default)s)",
    )
    ap.add_argument(
        "--repeats", type=int, default=3, help="timed repeats per mode (default: %(default)s)"
    )
    ap.add_argument(
        "--workers",
        type=int,
        help="parallel requests per stager (default: GET/prefetch 16, GetBatch 4)",
    )
    ap.add_argument(
        "--scratch", type=Path, help="local disk for staged copies (default: system temp dir)"
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results"),
        help="directory for the JSON and Markdown reports (default: %(default)s)",
    )
    args = ap.parse_args(argv)
    if args.scratch is not None:
        args.scratch = args.scratch.resolve()  # may be a symlink to local disk
    if min(args.stagers, args.repeats) < 1 or (args.workers is not None and args.workers < 1):
        ap.error("--stagers, --repeats and --workers must be positive")

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    if bad := set(modes) - set(MODES):
        ap.error(f"unknown modes: {sorted(bad)}")

    cfg = Config.from_env()
    objects, _ = dataset_listing(b2_client(cfg), cfg.b2_bucket, args.prefix)
    keys = [o.key for o in objects]
    if not keys:
        ap.error(f"no objects under {args.prefix!r}; run `make seed` first")

    results = []
    for mode in modes:
        print(f"[{mode}] {len(keys)} objects x {args.stagers} stager(s) x {args.repeats} repeat(s)")
        results.append(run_mode(cfg, args, mode, keys))

    report = {
        "prefix": args.prefix,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": {
            "platform": platform.platform(),
            "cpus": os.cpu_count(),
            "python": platform.python_version(),
            "aistore_sdk": version("aistore"),
            "b2_endpoint": cfg.b2_endpoint,
            "ais_endpoint": cfg.ais_endpoint,
            "rtt_ms_b2": tcp_rtt_ms(cfg.b2_endpoint),
            "rtt_ms_ais": tcp_rtt_ms(cfg.ais_endpoint),
        },
        "results": results,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / f"stage_time-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    stem.with_suffix(".json").write_text(json.dumps(report, indent=2))
    md = to_markdown(report)
    stem.with_suffix(".md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())

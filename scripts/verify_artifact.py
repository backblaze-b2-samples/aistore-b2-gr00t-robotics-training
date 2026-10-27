# SPDX-FileCopyrightText: Copyright (c) 2026 Backblaze, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Verify a completed artifact through AIS, or directly against B2 with --direct."""

import argparse
import sys
import time

from aistore_b2_gr00t.artifacts import verify_remote
from aistore_b2_gr00t.config import Config
from aistore_b2_gr00t.storage import ais_s3_client, b2_client


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--direct", action="store_true", help="verify B2's durable copy")
    args = parser.parse_args()
    cfg = Config.from_env(require_b2_credentials=args.direct)
    client = b2_client(cfg) if args.direct else ais_s3_client(cfg)
    last_update = float("-inf")

    def report_progress(files: int, total_files: int, received: int, total_bytes: int) -> None:
        nonlocal last_update
        now = time.monotonic()
        if files == total_files or now - last_update >= 5:
            print(
                f"verifying: {files}/{total_files} files, {received:,}/{total_bytes:,} bytes read",
                file=sys.stderr,
                flush=True,
            )
            last_update = now

    count = verify_remote(client, cfg.b2_bucket, args.prefix, progress=report_progress)
    print(f"verified {count} files in s3://{cfg.b2_bucket}/{args.prefix}")


if __name__ == "__main__":
    main()

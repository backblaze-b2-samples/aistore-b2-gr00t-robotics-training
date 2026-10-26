# SPDX-FileCopyrightText: Copyright (c) 2026 Backblaze, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Upload a LeRobot-format dataset directly to B2, bypassing AIS.

Sources:
  --local-dir   a directory on disk, e.g. Isaac-GR00T's demo_data/cube_to_bowl_5
  --hf-repo     a Hugging Face dataset at a pinned --revision, narrowed by --allow-pattern

Idempotent: objects with the same SHA-256 metadata are skipped. A manifest is published last.

  uv run python scripts/seed_b2.py --local-dir third_party/Isaac-GR00T/demo_data/cube_to_bowl_5 \\
      --prefix datasets/cube_to_bowl_5
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
from pathlib import Path

from aistore_b2_gr00t.artifacts import reject_symlinks
from aistore_b2_gr00t.config import Config
from aistore_b2_gr00t.storage import b2_client, upload_dir

_LFS_POINTER = b"version https://git-lfs.github.com/spec/v1"


def find_lfs_pointers(root: Path) -> list[Path]:
    """Files that are un-fetched Git LFS pointers (a clone without `git lfs pull`)."""
    hits = []
    for path in root.rglob("*"):
        if path.is_file() and path.stat().st_size < 1024:
            with path.open("rb") as f:
                if f.read(len(_LFS_POINTER)) == _LFS_POINTER:
                    hits.append(path)
    return hits


def download_hf(
    repo: str, revision: str, patterns: list[str], token: str | None, dest: Path
) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=repo,
            repo_type="dataset",
            revision=revision,
            allow_patterns=patterns,
            token=token,
            local_dir=dest,
        )
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--local-dir", type=Path, help="local dataset directory to upload")
    src.add_argument("--hf-repo", help="Hugging Face dataset repo id")
    ap.add_argument("--revision", help="HF commit SHA to pin (required with --hf-repo)")
    ap.add_argument(
        "--allow-pattern",
        action="append",
        default=[],
        help="HF allow_patterns glob (repeatable), e.g. 'unitree_g1.LMPnPAppleToPlateDC/**'",
    )
    ap.add_argument("--prefix", required=True, help="destination key prefix in B2")
    ap.add_argument(
        "--source-subdir",
        help="dataset directory inside the source, e.g. unitree_g1.LMPnPAppleToPlateDC",
    )
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    if args.local_dir is not None:
        args.local_dir = args.local_dir.resolve()  # the root may be a symlink; its contents may not

    if args.hf_repo and not (args.revision and args.allow_pattern):
        ap.error("--hf-repo needs --revision (a pinned commit) and at least one --allow-pattern")
    if args.hf_repo and not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        ap.error("--revision must be a full 40-character commit SHA")
    if args.source_subdir is not None and (
        "\\" in args.source_subdir
        or any(part in ("", ".", "..") for part in args.source_subdir.split("/"))
    ):
        ap.error("--source-subdir must be a relative directory without path aliases")

    cfg = Config.from_env()
    with tempfile.TemporaryDirectory(prefix="seed-gr00t-") as tmp:
        root = args.local_dir or download_hf(
            args.hf_repo, args.revision, args.allow_pattern, cfg.hf_token, Path(tmp).resolve()
        )
        if args.source_subdir is not None:
            selected = root / args.source_subdir
            reject_symlinks(selected)
            if not selected.resolve().is_relative_to(root.resolve()):
                ap.error("--source-subdir must stay inside the source directory")
            root = selected
        if not root.is_dir():
            ap.error(f"{root} is not a directory")

        if pointers := find_lfs_pointers(root):
            print(
                f"error: {len(pointers)} Git LFS pointer files under {root} (e.g. {pointers[0]}).\n"
                "Fetch the real files first: git lfs pull --include '<path>/**'",
                file=sys.stderr,
            )
            return 1

        uploaded, unchanged = upload_dir(
            b2_client(cfg),
            root,
            cfg.b2_bucket,
            args.prefix,
            args.workers,
            publish=True,
            exclude=(".cache",),
            provenance={
                "source": args.hf_repo or "local",
                "revision": args.revision,
                "source_subdir": args.source_subdir,
            },
        )
        dest = f"s3://{cfg.b2_bucket}/{args.prefix.rstrip('/')}/"
        print(f"{dest}: {uploaded} uploaded, {unchanged} unchanged")
    return 0


if __name__ == "__main__":
    sys.exit(main())

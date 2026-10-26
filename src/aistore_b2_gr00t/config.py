# SPDX-FileCopyrightText: Copyright (c) 2026 Backblaze, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Environment-driven configuration shared by the scripts, benchmarks, and tests."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from urllib.parse import urlparse

DEFAULT_AIS_ENDPOINT = "http://127.0.0.1:51080"

# B2 bucket names: 6-63 chars, letters, digits, and hyphens (no dots, unlike S3).
_BUCKET_RE = re.compile(r"^[A-Za-z0-9-]{6,63}$")


class ConfigError(ValueError):
    """Raised when required settings are missing or malformed."""


@dataclass(frozen=True)
class Config:
    b2_endpoint: str
    b2_region: str
    b2_bucket: str
    b2_key_id: str = field(repr=False)
    b2_app_key: str = field(repr=False)
    ais_endpoint: str = DEFAULT_AIS_ENDPOINT
    hf_token: str | None = field(default=None, repr=False)

    @classmethod
    def from_env(
        cls, env: Mapping[str, str] | None = None, *, require_b2_credentials: bool = True
    ) -> Config:
        """Build a validated Config from environment variables.

        `require_b2_credentials=False` lets paths that go through AIS (and `--dry-run`) run
        without the B2 key.
        """
        env = os.environ if env is None else env

        def get(name: str) -> str:
            return env.get(name, "").strip()

        required = ["B2_ENDPOINT", "B2_REGION", "B2_BUCKET"]
        if require_b2_credentials:
            required += ["B2_KEY_ID", "B2_APP_KEY"]
        missing = [name for name in required if not get(name)]
        if missing:
            raise ConfigError(f"missing required env vars: {', '.join(missing)} (see .env.example)")

        cfg = cls(
            b2_endpoint=get("B2_ENDPOINT").rstrip("/"),
            b2_region=get("B2_REGION"),
            b2_bucket=get("B2_BUCKET"),
            b2_key_id=get("B2_KEY_ID"),
            b2_app_key=get("B2_APP_KEY"),
            ais_endpoint=(get("AIS_ENDPOINT") or DEFAULT_AIS_ENDPOINT).rstrip("/"),
            hf_token=get("HF_TOKEN") or None,
        )
        cfg._validate()
        return cfg

    @property
    def ais_s3_endpoint(self) -> str:
        """AIStore's S3-compatible API lives under /s3 on any gateway."""
        return f"{self.ais_endpoint}/s3"

    @property
    def has_b2_credentials(self) -> bool:
        return bool(self.b2_key_id and self.b2_app_key)

    def _validate(self) -> None:
        for name, url in (("B2_ENDPOINT", self.b2_endpoint), ("AIS_ENDPOINT", self.ais_endpoint)):
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https") or not parsed.netloc:
                raise ConfigError(f"{name} must be an http(s) URL, got {url!r}")
        if not _BUCKET_RE.match(self.b2_bucket):
            raise ConfigError(
                f"B2_BUCKET {self.b2_bucket!r} is not a valid B2 bucket name "
                "(6-63 letters, digits, or hyphens)"
            )

# SPDX-FileCopyrightText: Copyright (c) 2026 Backblaze, Inc.
# SPDX-License-Identifier: Apache-2.0
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Throwaway values; nothing here reaches a real service.
FAKE_ENV = {
    "B2_ENDPOINT": "https://s3.us-west-004.backblazeb2.com",
    "B2_REGION": "us-west-004",
    "B2_BUCKET": "gr00t-unit-test",
    "B2_KEY_ID": "unit-key-id",
    "B2_APP_KEY": "unit-app-key",
}


def load_module(relpath: str):
    """Import a file from scripts/ or benchmarks/ (they are not part of the package)."""
    path = ROOT / relpath
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fake_env(monkeypatch):
    for k, v in FAKE_ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("AIS_ENDPOINT", raising=False)
    return FAKE_ENV

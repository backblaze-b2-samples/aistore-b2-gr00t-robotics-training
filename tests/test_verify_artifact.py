# SPDX-FileCopyrightText: Copyright (c) 2026 Backblaze, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Verification progress must remain visible without claiming success on corruption."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_artifacts import Store

from aistore_b2_gr00t.artifacts import upload_tree


@pytest.fixture
def verifier(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/verify_artifact.py"
    spec = importlib.util.spec_from_file_location("verify_artifact_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module.Config, "from_env", lambda **kwargs: SimpleNamespace(b2_bucket="test-bucket")
    )
    return module


@pytest.mark.parametrize("direct", [False, True])
def test_cli_reports_streaming_progress_on_stderr(verifier, monkeypatch, tmp_path, capsys, direct):
    payload = b"x" * (2 * 1024 * 1024 + 7)
    (tmp_path / "model.bin").write_bytes(payload)
    store = Store()
    upload_tree(store, tmp_path, "test-bucket", "run", publish=True)
    selected, unused = ("b2_client", "ais_s3_client") if direct else ("ais_s3_client", "b2_client")
    monkeypatch.setattr(verifier, selected, lambda cfg: store)
    monkeypatch.setattr(verifier, unused, lambda cfg: pytest.fail("unexpected storage client"))
    ticks = iter([0, 1, 6, 7, 8])
    monkeypatch.setattr(verifier, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
    monkeypatch.setattr(
        "sys.argv", ["verify_artifact", "--prefix", "run", *(["--direct"] if direct else [])]
    )

    verifier.main()

    output = capsys.readouterr()
    assert output.out == "verified 1 files in s3://test-bucket/run\n"
    assert output.err.splitlines() == [
        "verifying: 0/1 files, 0/2,097,159 bytes read",
        "verifying: 0/1 files, 2,097,152/2,097,159 bytes read",
        "verifying: 1/1 files, 2,097,159/2,097,159 bytes read",
    ]


def test_cli_does_not_print_success_for_corruption(verifier, monkeypatch, tmp_path, capsys):
    (tmp_path / "model.bin").write_bytes(b"new")
    store = Store()
    upload_tree(store, tmp_path, "test-bucket", "run", publish=True)
    store.data["run/model.bin"] = b"old"
    monkeypatch.setattr(verifier, "b2_client", lambda cfg: store)
    monkeypatch.setattr("sys.argv", ["verify_artifact", "--prefix", "run", "--direct"])

    with pytest.raises(OSError, match="integrity check"):
        verifier.main()

    output = capsys.readouterr()
    assert output.out == ""
    assert "verifying: 0/1 files" in output.err
    assert "verifying: 1/1 files" not in output.err

# SPDX-FileCopyrightText: Copyright (c) 2026 Backblaze, Inc.
# SPDX-License-Identifier: Apache-2.0
import pytest
from conftest import FAKE_ENV

from aistore_b2_gr00t.config import DEFAULT_AIS_ENDPOINT, Config, ConfigError


def test_from_env_defaults_and_normalization():
    cfg = Config.from_env({**FAKE_ENV, "B2_ENDPOINT": FAKE_ENV["B2_ENDPOINT"] + "/"})
    assert cfg.b2_endpoint == "https://s3.us-west-004.backblazeb2.com"
    assert cfg.ais_endpoint == DEFAULT_AIS_ENDPOINT
    assert cfg.ais_s3_endpoint == f"{DEFAULT_AIS_ENDPOINT}/s3"
    assert cfg.hf_token is None


def test_repr_excludes_secrets():
    cfg = Config.from_env({**FAKE_ENV, "HF_TOKEN": "hf_unit_token"})
    text = repr(cfg)
    for secret in ("unit-key-id", "unit-app-key", "hf_unit_token"):
        assert secret not in text


def test_missing_vars_are_listed():
    with pytest.raises(ConfigError, match="B2_KEY_ID, B2_APP_KEY"):
        Config.from_env({k: v for k, v in FAKE_ENV.items() if not k.endswith(("_ID", "APP_KEY"))})


def test_credentials_optional_when_not_required():
    env = {k: v for k, v in FAKE_ENV.items() if k not in ("B2_KEY_ID", "B2_APP_KEY")}
    assert Config.from_env(env, require_b2_credentials=False).b2_key_id == ""


@pytest.mark.parametrize(
    "override",
    [
        {"B2_ENDPOINT": "s3.us-west-004.backblazeb2.com"},
        {"AIS_ENDPOINT": "ftp://ais:51080"},
        {"B2_BUCKET": "bad.bucket.name"},
        {"B2_BUCKET": "short"},
    ],
)
def test_rejects_malformed_values(override):
    with pytest.raises(ConfigError):
        Config.from_env({**FAKE_ENV, **override})

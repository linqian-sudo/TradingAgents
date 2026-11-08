"""Optional request_timeout forwarded only when a run sets it."""

from __future__ import annotations

import pytest

import tradingagents.default_config as default_config_module
from tradingagents.llm_clients.factory import _coerce_request_timeout, build_llm_kwargs


def _config_with_env(monkeypatch, **overrides):
    for key in list(default_config_module._ENV_OVERRIDES):
        monkeypatch.delenv(key, raising=False)
    for key, val in overrides.items():
        monkeypatch.setenv(key, val)
    return default_config_module.build_default_config()


@pytest.mark.unit
@pytest.mark.parametrize("value,expected", [(1, 1), (120, 120), ("45", 45), (1.5, 1.5), ("2.5", 2.5)])
def test_coerce_accepts_positive_numbers(value, expected):
    assert _coerce_request_timeout(value) == expected


@pytest.mark.unit
@pytest.mark.parametrize("bad", [0, -1, "0", True, False, "abc", None])
def test_coerce_rejects_bad_timeouts(bad):
    with pytest.raises(ValueError):
        _coerce_request_timeout(bad)


@pytest.mark.unit
def test_timeout_not_forwarded_when_unset():
    kwargs = build_llm_kwargs({"llm_provider": "openai", "request_timeout": None})
    assert "timeout" not in kwargs


@pytest.mark.unit
def test_timeout_env_is_forwarded(monkeypatch):
    config = _config_with_env(monkeypatch, TRADINGAGENTS_REQUEST_TIMEOUT="90")
    assert config["request_timeout"] == "90"
    kwargs = build_llm_kwargs(config)
    assert kwargs["timeout"] == 90

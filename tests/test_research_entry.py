"""Research entry: mapping, unconfigured model, and no fabricated prices."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from research_entry.cli import main
from research_entry.data_check import check_history
from research_entry.mapping import map_symbol
from research_entry.model_gate import assess_model
from research_entry.runner import execute_run
from tradingagents.dataflows.errors import VendorUnavailableError

_PROVIDER_KEYS = (
    "OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "ANTHROPIC_API_KEY",
    "XAI_API_KEY",
    "DEEPSEEK_API_KEY",
    "DASHSCOPE_API_KEY",
    "DASHSCOPE_CN_API_KEY",
    "ZHIPU_API_KEY",
    "ZHIPU_CN_API_KEY",
    "MINIMAX_API_KEY",
    "MINIMAX_CN_API_KEY",
    "OPENROUTER_API_KEY",
    "MISTRAL_API_KEY",
    "MOONSHOT_API_KEY",
    "GROQ_API_KEY",
    "NVIDIA_API_KEY",
    "OPENAI_COMPATIBLE_API_KEY",
    "AZURE_OPENAI_API_KEY",
)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("raw", "market", "canonical", "changed"),
    [
        ("0700.HK", "HK", "0700.HK", False),
        ("700.HK", "HK", "0700.HK", True),
        ("00700.HK", "HK", "0700.HK", True),
        ("1810.HK", "HK", "1810.HK", False),
        ("601398.SS", "SS", "601398.SS", False),
        ("601398.SH", "SS", "601398.SS", True),
        ("002779.SZ", "SZ", "002779.SZ", False),
        ("688825.SS", "SS", "688825.SS", False),
        ("688825.SH", "SH", "688825.SS", True),
    ],
)
def test_reference_symbols_map_without_claiming_existence(raw, market, canonical, changed):
    mapped = map_symbol(raw, market)
    assert mapped["canonical"] == canonical
    assert mapped["changed"] is changed
    assert mapped["market_matches"] is True
    assert mapped["exists"] is None
    assert mapped["quote"] is None
    assert mapped["ok"] is True


@pytest.mark.unit
def test_market_mismatch_does_not_invent_a_quote():
    mapped = map_symbol("601398.SS", "HK")
    assert mapped["ok"] is False
    assert mapped["market_matches"] is False
    assert mapped["quote"] is None
    assert mapped["exists"] is None


@pytest.mark.unit
def test_model_missing_provider_is_unconfigured():
    gate = assess_model(
        provider=None, quick_model=None, deep_model=None, backend_url=None, environ={},
    )
    assert gate["status"] == "model_unconfigured"
    assert gate["called"] is False
    assert "openai" not in (gate["provider"] or "")


@pytest.mark.unit
def test_hosted_provider_without_key_is_unconfigured():
    gate = assess_model(
        provider="deepseek",
        quick_model="deepseek-chat",
        deep_model="deepseek-chat",
        backend_url=None,
        environ={},
    )
    assert gate["status"] == "model_unconfigured"
    assert gate["credential_env"] == "DEEPSEEK_API_KEY"
    assert gate["credential_present"] is False
    assert gate["called"] is False


@pytest.mark.unit
def test_ollama_needs_model_names_and_no_key():
    missing = assess_model(provider="ollama", quick_model=None, deep_model=None, backend_url=None, environ={})
    assert missing["status"] == "model_unconfigured"
    ready = assess_model(
        provider="ollama",
        quick_model="qwen2.5",
        deep_model="qwen2.5",
        backend_url=None,
        environ={},
    )
    assert ready["status"] == "configured"
    assert ready["credential_present"] is False
    assert ready["called"] is False


@pytest.mark.unit
def test_compatible_endpoint_requires_backend_url():
    gate = assess_model(
        provider="openai_compatible",
        quick_model="local",
        deep_model="local",
        backend_url=None,
        environ={},
    )
    assert gate["status"] == "model_unconfigured"
    assert "backend_url" in gate["missing"]


@pytest.mark.unit
def test_empty_history_does_not_invent_prices(monkeypatch):
    monkeypatch.setattr("research_entry.data_check.yf_retry", lambda func, **kwargs: None)
    result = check_history("0700.HK", "2026-11-06")
    assert result["status"] == "no_data"
    assert result["rows"] == 0
    assert result["last_bar"] is None
    assert result["research_claim"] is False


@pytest.mark.unit
def test_rows_after_as_of_are_dropped_and_missing_close_stays_null(monkeypatch):
    frame = pd.DataFrame(
        {
            "Open": [10.0, 11.0],
            "High": [12.0, 12.5],
            "Low": [9.0, 10.5],
            "Volume": [100, None],
        },
        index=pd.to_datetime(["2026-11-06", "2026-11-07"]),
    )
    monkeypatch.setattr("research_entry.data_check.yf_retry", lambda func, **kwargs: frame)
    result = check_history("0700.HK", "2026-11-06")
    assert result["status"] == "ok"
    assert result["rows"] == 1
    assert result["last_bar"]["date"] == "2026-11-06"
    assert result["last_bar"]["open"] == 10.0
    assert result["last_bar"]["close"] is None
    assert result["last_bar"]["volume"] == 100


@pytest.mark.unit
def test_vendor_failure_is_reported_as_unavailable(monkeypatch):
    def explode(func, **kwargs):
        raise VendorUnavailableError("Yahoo Finance rate limited after 3 retries")

    monkeypatch.setattr("research_entry.data_check.yf_retry", explode)
    result = check_history("0700.HK", "2026-11-06")
    assert result["status"] == "unavailable"
    assert result["last_bar"] is None
    assert "rate limited" in result["error"]


@pytest.mark.unit
def test_run_without_model_does_not_construct_graph(monkeypatch, tmp_path):
    class Boom:
        def __init__(self, *args, **kwargs):
            raise AssertionError("model client should not be constructed")

    monkeypatch.setattr("tradingagents.graph.trading_graph.TradingAgentsGraph", Boom)
    code = execute_run(
        tmp_path,
        "0700.HK",
        "HK",
        "2026-11-06",
        provider=None,
        quick_model=None,
        deep_model=None,
        backend_url=None,
        analysts="market",
        limit_overrides={},
        environ={},
    )
    assert code == 2
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["model"]["status"] == "model_unconfigured"
    assert status["model"]["called"] is False
    assert status["research"]["end_to_end"] is False
    assert status["research"]["decision"] is None
    assert status["safety"]["orders_sent"] is False
    assert status["safety"]["portfolio_read"] is False
    summary = (tmp_path / "summary.zh.md").read_text(encoding="utf-8")
    assert "未配置" in summary or "未选择" in summary
    assert "端到端研判：否" in summary


@pytest.mark.unit
def test_configured_run_keeps_limits_and_refuses_portfolio(monkeypatch, tmp_path):
    captured = {}

    class FakeGraph:
        def __init__(self, selected_analysts, debug, config):
            captured["analysts"] = selected_analysts
            captured["config"] = config

        def propagate(self, symbol, trade_date, asset_type="stock", portfolio=None):
            captured["symbol"] = symbol
            captured["portfolio"] = portfolio
            captured["asset_type"] = asset_type
            return {"market_report": "来自测试桩"}, "Hold"

        def save_reports(self, final_state, ticker, save_path=None, html=True):
            captured["save_path"] = save_path
            return save_path

    monkeypatch.setattr("tradingagents.graph.trading_graph.TradingAgentsGraph", FakeGraph)
    code = execute_run(
        tmp_path,
        "0700.HK",
        "HK",
        "2026-11-06",
        provider="openai",
        quick_model="test-quick",
        deep_model="test-deep",
        backend_url=None,
        analysts="market",
        limit_overrides={},
        environ={"OPENAI_API_KEY": "test-key-not-real"},
    )
    assert code == 0
    assert captured["portfolio"] is None
    assert captured["symbol"] == "0700.HK"
    assert captured["config"]["checkpoint_enabled"] is False
    assert captured["config"]["output_language"] == "Chinese"
    assert captured["config"]["max_tool_rounds"] == 8
    assert captured["config"]["max_tokens"] == 4096
    assert captured["config"]["request_timeout"] == 120
    assert captured["config"]["data_vendors"]["fundamental_data"] == "yfinance"
    assert str(tmp_path) in captured["config"]["memory_log_path"]
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["research"]["end_to_end"] is True
    assert status["research"]["decision"] == "Hold"
    assert status["model"]["called"] is True
    assert "test-key-not-real" not in (tmp_path / "status.json").read_text(encoding="utf-8")


@pytest.mark.unit
def test_cli_help_and_unconfigured_run(monkeypatch, tmp_path):
    for name in _PROVIDER_KEYS:
        monkeypatch.setenv(name, "")
    for name in (
        "TRADINGAGENTS_LLM_PROVIDER",
        "TRADINGAGENTS_QUICK_THINK_LLM",
        "TRADINGAGENTS_DEEP_THINK_LLM",
        "TRADINGAGENTS_LLM_BACKEND_URL",
    ):
        monkeypatch.setenv(name, "")
    with pytest.raises(SystemExit) as help_exit:
        main(["--help"])
    assert help_exit.value.code == 0
    with pytest.raises(SystemExit) as run_exit:
        main([
            "run",
            "--symbol", "1810.HK",
            "--market", "HK",
            "--date", "2026-11-06",
            "--output", str(tmp_path),
        ])
    assert run_exit.value.code == 2
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["symbol"]["canonical"] == "1810.HK"
    assert status["symbol"]["exists"] is None
    assert status["data"]["status"] == "not_requested"

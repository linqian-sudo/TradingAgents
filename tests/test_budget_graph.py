"""A scripted multi-role graph must pay the ledger on every model call."""

from __future__ import annotations

import copy

import pandas as pd
import pytest

from research_entry.budget import (
    RESEARCH_INPUT_CAP,
    RESEARCH_OUTPUT_CAP,
    TRUNCATION_BANNER,
    Ledger,
    build_plan,
    conservative_prior_reservation_hold,
    prepare_for_send,
)
from research_entry.budget_gate import BudgetedChatModel, consume_transport_permit
from research_entry.runner import _run_reserved_research
from tests.test_graph_end_to_end import TRADE_DATE, ScriptedModel
from tradingagents.agents import context
from tradingagents.dataflows import router
from tradingagents.dataflows.vendors.yahoo import market as yahoo_market, snapshot
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph import trading_graph


def _offline(monkeypatch):
    called: set[str] = set()
    for method, vendors in router.VENDOR_METHODS.items():
        for vendor in vendors:
            monkeypatch.setitem(
                vendors, vendor, lambda *a, _m=method, **k: called.add(_m) or f"{_m} data",
            )
    prices = pd.DataFrame({
        "Date": pd.bdate_range(end=TRADE_DATE, periods=60),
        "Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.5, "Volume": 1_000_000,
    })
    monkeypatch.setattr(snapshot, "load_ohlcv", lambda *a, **k: called.add("ohlcv") or prices.copy())
    monkeypatch.setattr(yahoo_market.yf, "Ticker", lambda s: type("T", (), {"info": {"longName": "Tencent"}})())
    context._identity.cache_clear()
    return called


@pytest.mark.unit
def test_tencent_graph_charges_every_model_call(tmp_path, monkeypatch):
    called = _offline(monkeypatch)
    scripted = ScriptedModel(structured=True)
    plan = build_plan(
        quick_model="deepseek-flash",
        deep_model="deepseek-flash",
        analysts=["market", "news", "fundamentals"],
        tool_rounds=1,
        debate_rounds=1,
        risk_rounds=1,
        input_cap=RESEARCH_INPUT_CAP,
        output_cap=RESEARCH_OUTPUT_CAP,
        retries=0,
    )
    prior = str(conservative_prior_reservation_hold())
    ledger = Ledger(tmp_path / "budget-ledger.json", "9.5", prior)
    ledger.reserve(plan, "tencent-research")

    def factory(config, tier, **_kwargs):
        wrapped = BudgetedChatModel(
            inner=scripted,
            ledger=ledger,
            input_cap=plan["input_cap"],
            output_cap=plan["output_cap"],
            expected_model="deepseek-flash",
        )

        class _Client:
            def get_llm(self):
                return wrapped

        return _Client()

    monkeypatch.setattr(trading_graph, "create_tier_client", factory)
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg.update(
        llm_provider="deepseek",
        quick_think_llm="deepseek-flash",
        deep_think_llm="deepseek-flash",
        max_debate_rounds=1,
        max_risk_discuss_rounds=1,
        max_tool_rounds=1,
        max_tokens=RESEARCH_OUTPUT_CAP,
        llm_max_retries=0,
        checkpoint_enabled=False,
        results_dir=str(tmp_path / "results"),
        data_cache_dir=str(tmp_path / "cache"),
        memory_log_path=str(tmp_path / "memory.md"),
    )
    graph = trading_graph.TradingAgentsGraph(
        selected_analysts=["market", "news", "fundamentals"],
        debug=False,
        config=cfg,
    )
    _state, decision = graph.propagate("0700.HK", TRADE_DATE)

    snapshot = ledger.snapshot()
    used = [slot for slot in snapshot["slots"] if slot["status"] != "reserved"]
    assert decision
    assert len(scripted.calls) == len(used)
    assert len(used) <= len(snapshot["slots"])
    assert len(used) > 8
    for key in ("market_report", "news_report", "fundamentals_report", "investment_plan",
                "trader_investment_plan", "final_trade_decision"):
        assert _state[key].strip(), key
    assert "get_stock_data" in called
    assert "get_news" in called
    assert "get_fundamentals" in called
    with pytest.raises(Exception, match="传输层没有发送前预留"):
        consume_transport_permit()


@pytest.mark.unit
def test_oversized_tool_text_keeps_the_edges_and_states_the_cut():
    header = "# Stock data for 0700.HK from 2026-10-01 to 2026-11-06\n# Total records: 4000\n"
    body = "".join(f"2026-01-01,{index},1,1,1,1\n" for index in range(4000))
    tail = "2026-11-06,419.8,428.6,416.2,419.8,29605485\n"
    messages = [{"role": "tool", "content": header + body + tail}]
    sent, bound, truncated = prepare_for_send(messages, None, 2500)
    text = sent[0]["content"]
    assert truncated is True
    assert "0700.HK" in text
    assert "Total records: 4000" in text
    assert "2026-11-06,419.8" in text
    assert TRUNCATION_BANNER.strip() in text
    assert bound <= 2500
    with pytest.raises(Exception, match="拒绝发送"):
        prepare_for_send([{"role": "user", "content": "x" * 5000}], None, 100)


@pytest.mark.unit
def test_run_entry_reserves_then_uses_the_gated_graph(tmp_path, monkeypatch):
    called = _offline(monkeypatch)
    scripted = ScriptedModel(structured=True)

    def factory(config, tier, **_kwargs):
        class _Client:
            def get_llm(self):
                return scripted

        return _Client()

    monkeypatch.setattr(trading_graph, "create_tier_client", factory)
    plan = build_plan(
        quick_model="deepseek-flash",
        deep_model="deepseek-flash",
        analysts=["market", "news", "fundamentals"],
        tool_rounds=1,
        debate_rounds=1,
        risk_rounds=1,
        input_cap=RESEARCH_INPUT_CAP,
        output_cap=RESEARCH_OUTPUT_CAP,
        retries=0,
    )
    code = _run_reserved_research(
        tmp_path,
        {"canonical": "0700.HK", "input": "0700.HK", "market": "HK"},
        {"provider": "deepseek", "quick_model": "deepseek-flash", "deep_model": "deepseek-flash", "called": False},
        {
            "max_debate_rounds": 1,
            "max_risk_discuss_rounds": 1,
            "max_tool_rounds": 1,
            "max_tokens": RESEARCH_OUTPUT_CAP,
            "llm_max_retries": 0,
            "request_timeout": 120,
        },
        ["market", "news", "fundamentals"],
        TRADE_DATE,
        None,
        "9.5",
        str(conservative_prior_reservation_hold()),
        plan,
    )
    assert code == 0
    assert called
    assert len(scripted.calls) > 8

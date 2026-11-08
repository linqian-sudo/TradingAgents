"""Child process for SIGTERM and SIGKILL persistence tests. No network and no key."""

from __future__ import annotations

import os
import sys

import pandas as pd

from research_entry.cli import main
from tests.test_graph_end_to_end import TRADE_DATE, ScriptedModel
from tradingagents.agents import context
from tradingagents.dataflows import router
from tradingagents.dataflows.vendors.yahoo import market as yahoo_market, snapshot
from tradingagents.graph import trading_graph


def run() -> None:
    output, ready = sys.argv[1], sys.argv[2]
    os.environ["TRADINGAGENTS_TEST_PAUSE_FILE"] = ready
    for method, vendors in router.VENDOR_METHODS.items():
        for vendor in list(vendors):
            vendors[vendor] = (lambda *a, _m=method, **k: f"{_m} as of 2026-11-06T16:00:00Z")
    prices = pd.DataFrame({
        "Date": pd.bdate_range(end=TRADE_DATE, periods=60),
        "Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.5, "Volume": 1_000_000,
    })
    snapshot.load_ohlcv = lambda *a, **k: prices.copy()
    yahoo_market.yf.Ticker = lambda s: type("T", (), {"info": {"longName": "Tencent"}})()
    context._identity.cache_clear()
    model = ScriptedModel(structured=True)

    def factory(config, tier, **_kwargs):
        class _Client:
            def get_llm(self):
                return model

        return _Client()

    trading_graph.create_tier_client = factory
    os.environ["DEEPSEEK_API_KEY"] = "test-not-real"
    main([
        "run", "--symbol", "0700.HK", "--market", "HK", "--date", "2026-11-06",
        "--provider", "deepseek", "--quick-model", "deepseek-flash", "--deep-model", "deepseek-flash",
        "--analysts", "market,news,fundamentals",
        "--max-tool-rounds", "1", "--max-debate-rounds", "1", "--max-risk-rounds", "1",
        "--max-tokens", "1024", "--max-retries", "0",
        "--budget-cny", "9.5", "--prior-hold-cny", "0.008448", "--input-token-cap", "24000",
        "--budget-mode", "research", "--output", output,
    ])


if __name__ == "__main__":
    run()

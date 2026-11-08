"""The console command, not an internal helper, must reach the mocked graph."""

from __future__ import annotations

import json

import pytest

from research_entry.cli import main
from tests.test_budget_graph import _offline
from tests.test_graph_end_to_end import ScriptedModel
from tradingagents.graph import trading_graph

_RUN = [
    "run",
    "--symbol", "0700.HK",
    "--market", "HK",
    "--date", "2026-11-06",
    "--provider", "deepseek",
    "--quick-model", "deepseek-flash",
    "--deep-model", "deepseek-flash",
    "--analysts", "market,news,fundamentals",
    "--max-tool-rounds", "1",
    "--max-debate-rounds", "1",
    "--max-risk-rounds", "1",
    "--max-tokens", "1024",
    "--max-retries", "0",
    "--budget-cny", "9.5",
    "--prior-hold-cny", "0.008448",
    "--input-token-cap", "24000",
    "--budget-mode", "research",
]


def _argv(tmp_path):
    return [*_RUN, "--output", str(tmp_path)]


@pytest.mark.unit
def test_cli_run_passes_ledger_prior_and_finishes_the_mock_graph(tmp_path, monkeypatch):
    called = _offline(monkeypatch)
    scripted = ScriptedModel(structured=True)

    def factory(config, tier, **_kwargs):
        class _Client:
            def get_llm(self):
                return scripted

        return _Client()

    monkeypatch.setattr(trading_graph, "create_tier_client", factory)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-not-real")
    with pytest.raises(SystemExit) as exc:
        main(_argv(tmp_path))
    assert exc.value.code == 0
    text = (tmp_path / "status.json").read_text(encoding="utf-8")
    status = json.loads(text)
    assert status["model"]["called"] is True
    assert status["model"]["status"] == "research_attempted"
    assert status["research"]["end_to_end"] is False
    assert status["research"]["decision"]
    assert status["extra"]["worst_case_reserved_cny"]
    assert status["extra"]["ledger"]["prior_hold_cny"].startswith("0.008448")
    assert status["extra"]["ledger"]["budget_cny"].startswith("9.5")
    assert len(scripted.calls) > 8
    assert "get_stock_data" in called
    assert "test-not-real" not in text


@pytest.mark.unit
def test_cli_startup_exception_still_writes_status(tmp_path, monkeypatch):
    def broken(*_args, **_kwargs):
        raise TypeError("ledger_prior boom")

    monkeypatch.setattr("research_entry.runner._run_reserved_research", broken)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-not-real")
    with pytest.raises(SystemExit) as exc:
        main(_argv(tmp_path))
    assert exc.value.code == 1
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["research"]["status"] == "startup_failed"
    assert status["research"]["end_to_end"] is False
    assert status["research"]["decision"] is None
    assert status["model"]["called"] is False
    assert "TypeError" in status["research"]["reason"]
    assert "ledger_prior boom" in status["extra"]["startup_error"]

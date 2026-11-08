"""Frozen-ledger inheritance, interrupt checkpoints, and vendor provenance."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from decimal import Decimal
from pathlib import Path

import pytest

from research_entry.budget import BudgetRefusal, inherit_frozen_ledger, sha256_file
from research_entry.cli import main
from tests.test_budget_graph import _offline
from tests.test_graph_end_to_end import ScriptedModel
from tradingagents.graph import trading_graph

_RUN = [
    "run", "--symbol", "0700.HK", "--market", "HK", "--date", "2026-11-06",
    "--provider", "deepseek", "--quick-model", "deepseek-flash", "--deep-model", "deepseek-flash",
    "--analysts", "market,news,fundamentals",
    "--max-tool-rounds", "1", "--max-debate-rounds", "1", "--max-risk-rounds", "1",
    "--max-tokens", "4096", "--max-retries", "0",
    "--budget-cny", "9.5", "--input-token-cap", "24000", "--budget-mode", "research",
]


def _frozen(path: Path, *, frozen: bool = True) -> None:
    path.write_text(json.dumps({
        "run_id": "previous-tencent",
        "frozen": frozen,
        "freeze_reason": "usage_missing",
        "budget_cny": "9.5",
        "prior_hold_cny": "1.959598",
        "reserved_cny": "0.955264",
        "slots": [
            {"status": "settled", "charged_cny": "0.156340", "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
            {"status": "usage_missing", "charged_cny": "0.056192", "freeze_reason": "usage_missing"},
        ],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_understated_prior_and_unfrozen_ledger_are_refused(tmp_path):
    old = tmp_path / "old" / "budget-ledger.json"
    old.parent.mkdir()
    _frozen(old)
    lock = old.parent / "budget-ledger.lock"
    lock.write_text("held\n", encoding="utf-8")
    before = old.read_bytes()
    lock_before = lock.read_bytes()
    with pytest.raises(BudgetRefusal, match="不一致"):
        inherit_frozen_ledger(
            old, new_ledger_path=tmp_path / "new" / "budget-ledger.json",
            budget_cny="9.5", new_reserved_cny="1.130752", stated_prior="1.959598",
        )
    plain = tmp_path / "plain" / "budget-ledger.json"
    plain.parent.mkdir()
    _frozen(plain, frozen=False)
    plain_before = plain.read_bytes()
    with pytest.raises(BudgetRefusal, match="未冻结"):
        inherit_frozen_ledger(
            plain, new_ledger_path=tmp_path / "new" / "budget-ledger.json",
            budget_cny="9.5", new_reserved_cny="1.130752",
        )
    with pytest.raises(BudgetRefusal, match="不能写在已冻结"):
        inherit_frozen_ledger(
            old, new_ledger_path=old, budget_cny="9.5", new_reserved_cny="1.130752",
        )
    assert old.read_bytes() == before
    assert lock.read_bytes() == lock_before
    assert plain.read_bytes() == plain_before


def test_cli_inherits_full_hold_without_rewriting_the_frozen_ledger(tmp_path, monkeypatch):
    _offline(monkeypatch)
    old_dir = tmp_path / "old"
    old_dir.mkdir()
    old = old_dir / "budget-ledger.json"
    _frozen(old)
    (old_dir / "budget-ledger.lock").write_text("held\n", encoding="utf-8")
    before = sha256_file(old)
    scripted = ScriptedModel(structured=True)

    def factory(config, tier, **_kwargs):
        class _Client:
            def get_llm(self):
                return scripted

        return _Client()

    monkeypatch.setattr(trading_graph, "create_tier_client", factory)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-not-real")
    new = tmp_path / "next"
    with pytest.raises(SystemExit) as exc:
        main([*_RUN, "--output", str(new), "--inherit-frozen-ledger", str(old)])
    assert exc.value.code == 0
    manifest = json.loads((new / "inheritance-manifest.json").read_text(encoding="utf-8"))
    assert Decimal(manifest["inherited_hold_cny"]) == Decimal("2.914862")
    assert Decimal(manifest["known_usage_cny_not_added"]) == Decimal("0.156340")
    assert Decimal(manifest["new_reserved_cny"]) == Decimal("1.130752")
    assert Decimal(manifest["cumulative_conservative_cny"]) == Decimal("4.045614")
    assert manifest["unknown_released"] is False
    assert manifest["old_file_modified"] is False
    assert manifest["old_ledger_sha256"] == before
    assert manifest["lock_files"][0]["sha256"] == sha256_file(old_dir / "budget-ledger.lock")
    assert sha256_file(old) == before
    assert json.loads(old.read_text(encoding="utf-8"))["frozen"] is True
    status = json.loads((new / "status.json").read_text(encoding="utf-8"))
    assert status["extra"]["ledger"]["prior_hold_cny"].startswith("2.914862")
    assert status["extra"]["ledger"]["frozen"] is False
    assert "test-not-real" not in (new / "status.json").read_text(encoding="utf-8")


def test_vendor_times_are_quoted_on_success_and_absent_on_failure(tmp_path, monkeypatch):
    from tradingagents.dataflows import router
    from tradingagents.dataflows.errors import NoMarketDataError

    def vendor_for(method):
        def respond(*_args, **_kwargs):
            if method == "get_news":
                raise NoMarketDataError("0700.HK", "0700.HK", "no rows")
            if method == "get_global_news":
                raise RuntimeError("source down")
            return f"{method} as of 2026-11-06T16:00:00Z close 320"

        return respond

    for method, vendors in router.VENDOR_METHODS.items():
        for vendor in list(vendors):
            monkeypatch.setitem(vendors, vendor, vendor_for(method))
    _offline_prices(monkeypatch)
    scripted = ScriptedModel(structured=True)

    def factory(config, tier, **_kwargs):
        class _Client:
            def get_llm(self):
                return scripted

        return _Client()

    monkeypatch.setattr(trading_graph, "create_tier_client", factory)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-not-real")
    with pytest.raises(SystemExit) as exc:
        main([*_RUN, "--max-tokens", "1024", "--prior-hold-cny", "0.008448", "--output", str(tmp_path)])
    assert exc.value.code in {0, 1, 2}
    rows = [
        json.loads(line)
        for line in (tmp_path / "data-sources.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    available = [row for row in rows if row["outcome"] == "available"]
    unavailable = [row for row in rows if row["outcome"] == "unavailable"]
    errors = [row for row in rows if row["outcome"] == "error"]
    assert available
    assert any("2026-11-06T16:00:00Z" in row["source_times_quoted"] for row in available)
    assert unavailable or errors
    for row in unavailable + errors:
        assert row["available"] is False
        assert row["source_times_quoted"] == []
        assert row["observed_at"]


def _offline_prices(monkeypatch):
    import pandas as pd

    from tests.test_graph_end_to_end import TRADE_DATE
    from tradingagents.agents import context
    from tradingagents.dataflows.vendors.yahoo import market as yahoo_market, snapshot

    prices = pd.DataFrame({
        "Date": pd.bdate_range(end=TRADE_DATE, periods=60),
        "Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.5, "Volume": 1_000_000,
    })
    monkeypatch.setattr(snapshot, "load_ohlcv", lambda *a, **k: prices.copy())
    monkeypatch.setattr(yahoo_market.yf, "Ticker", lambda s: type("T", (), {"info": {"longName": "Tencent"}})())
    context._identity.cache_clear()


def test_model_exception_keeps_partial_roles_and_status(tmp_path, monkeypatch):
    _offline(monkeypatch)
    scripted = ScriptedModel(structured=True, fail_at=3)

    def factory(config, tier, **_kwargs):
        class _Client:
            def get_llm(self):
                return scripted

        return _Client()

    monkeypatch.setattr(trading_graph, "create_tier_client", factory)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-not-real")
    with pytest.raises(SystemExit) as exc:
        main([*_RUN, "--max-tokens", "1024", "--prior-hold-cny", "0.008448", "--output", str(tmp_path)])
    assert exc.value.code == 1
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["research"]["end_to_end"] is False
    assert status["research"]["decision"] is None
    assert list((tmp_path / "roles").glob("*.md"))
    assert (tmp_path / "checkpoint" / "responses.jsonl").exists()


def _wait_ready(path: Path, proc: subprocess.Popen) -> None:
    deadline = time.time() + 30
    while time.time() < deadline:
        if path.exists():
            return
        if proc.poll() is not None:
            raise AssertionError(f"child exited {proc.returncode} before ready")
        time.sleep(0.05)
    raise AssertionError("child did not reach the second model call")


def test_sigterm_writes_status_and_sigkill_leaves_only_the_checkpoint(tmp_path):
    child = Path(__file__).resolve().parents[0] / "support" / "interrupt_child.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])

    term_out = tmp_path / "term"
    term_ready = tmp_path / "term-ready"
    term = subprocess.Popen([sys.executable, str(child), str(term_out), str(term_ready)], env=env)
    try:
        _wait_ready(term_ready, term)
        term.send_signal(signal.SIGTERM)
        assert term.wait(timeout=20) == 3
    finally:
        if term.poll() is None:
            term.kill()
    status = json.loads((term_out / "status.json").read_text(encoding="utf-8"))
    assert status["research"]["status"] == "research_interrupted"
    assert status["research"]["decision"] is None
    assert (term_out / "checkpoint" / "responses.jsonl").exists()
    assert "SIGKILL 不能保证" in status["extra"]["sigkill_note"]

    kill_out = tmp_path / "kill"
    kill_ready = tmp_path / "kill-ready"
    killed = subprocess.Popen([sys.executable, str(child), str(kill_out), str(kill_ready)], env=env)
    try:
        _wait_ready(kill_ready, killed)
        killed.send_signal(signal.SIGKILL)
        assert killed.wait(timeout=20) != 0
    finally:
        if killed.poll() is None:
            killed.kill()
    checkpoint = (kill_out / "checkpoint" / "responses.jsonl").read_text(encoding="utf-8")
    assert checkpoint.strip()
    final = kill_out / "status.json"
    if final.exists():
        body = json.loads(final.read_text(encoding="utf-8"))
        assert body["research"].get("status") != "research_attempted"

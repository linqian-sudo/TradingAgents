"""Mocked budget reservation. No model endpoint is contacted."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from research_entry.budget import (
    BudgetRefusal,
    Ledger,
    PriceCard,
    assess_affordability,
    build_plan,
    build_probe_plan,
    call_cost,
    conservative_probe_hold,
    model_price,
    official_price_card,
    request_token_upper_bound,
    require_endpoint,
    research_call_counts,
    text_token_upper_bound,
)
from research_entry.budget_gate import assert_controlled, install_budget_gate
from research_entry.runner import execute_run


class _Root:
    def __init__(self):
        self.max_retries = 0


class _Client:
    def __init__(self, behavior):
        self.calls = []
        self._behavior = behavior

        def create(**kwargs):
            self.calls.append(kwargs)
            return self._behavior(kwargs)

        self.create = create


class _LLM:
    def __init__(self, behavior):
        self.max_retries = 0
        self.model_name = "deepseek-flash"
        self.root_client = _Root()
        self.client = _Client(behavior)
        self.async_client = None


def _ok(_kwargs):
    return {"usage": {"prompt_tokens": 4, "completion_tokens": 2}}


def _ledger(tmp_path, budget="1", prior=None):
    hold = prior if prior is not None else str(conservative_probe_hold())
    return Ledger(tmp_path / "budget-ledger.json", budget, hold)


def _probe():
    return build_probe_plan()


@pytest.mark.unit
@pytest.mark.parametrize(
    "text",
    ["a", "中文", "😀", "é", "e\u0301", "\u0000", "\u202e", "𐍈", "a😀中\n"],
)
def test_unicode_bound_never_uses_code_point_length(text):
    bound = text_token_upper_bound(text)
    assert bound == len(text.encode("utf-8"))
    assert bound >= len(text)
    if text == "😀":
        assert len(text) == 1
        assert bound == 4


@pytest.mark.unit
def test_image_and_float_payloads_are_refused():
    with pytest.raises(BudgetRefusal):
        request_token_upper_bound([{"type": "image_url", "image_url": {"url": "http://example"}}])
    with pytest.raises(BudgetRefusal):
        request_token_upper_bound([{"role": "user", "content": 1.25}])


@pytest.mark.unit
def test_legacy_models_and_foreign_endpoints_are_refused():
    card = official_price_card()
    for name in ("deepseek-v4-flash", "deepseek-chat", "deepseek-reasoner", "gpt-6-sol"):
        with pytest.raises(BudgetRefusal):
            model_price(card, name)
    with pytest.raises(BudgetRefusal):
        require_endpoint("https://api.deepseek.com/anthropic")
    require_endpoint("https://api.deepseek.com")
    require_endpoint(None)


@pytest.mark.unit
def test_peak_hold_is_double_the_off_peak_estimate():
    hold = conservative_probe_hold()
    assert hold == Decimal("0.00003")
    # Parent's 0.000015 figure is off-peak cache-miss, not the reservation.
    assert hold == Decimal("0.000015") * 2


@pytest.mark.unit
def test_expired_price_and_exact_budget_boundary(tmp_path):
    expired = datetime(2026, 11, 16, tzinfo=UTC)
    with pytest.raises(BudgetRefusal):
        build_probe_plan(now=expired)
    probe = _probe()
    prior = conservative_probe_hold()
    exact = prior + Decimal(probe["reserved_cny"])
    Ledger(tmp_path / "exact.json", str(exact), str(prior)).reserve(probe, "edge", now=datetime(2026, 11, 8, tzinfo=UTC))
    with pytest.raises(BudgetRefusal):
        Ledger(tmp_path / "short.json", str(exact - Decimal("0.000000000001")), str(prior)).reserve(probe, "edge")


@pytest.mark.unit
def test_send_reserves_before_transport_and_blocks_a_second_call(tmp_path):
    ledger = _ledger(tmp_path, prior="0")
    ledger.reserve(_probe(), "probe")
    llm = _LLM(_ok)
    install_budget_gate(llm, ledger)
    llm.client.create(model="deepseek-flash", messages=[{"role": "user", "content": "ok"}], max_tokens=32)
    assert len(llm.client.calls) == 1
    assert llm.client.calls[0]["extra_body"]["thinking"]["type"] == "disabled"
    with pytest.raises(BudgetRefusal):
        llm.client.create(model="deepseek-flash", messages=[{"role": "user", "content": "ok"}], max_tokens=32)
    assert len(llm.client.calls) == 1
    assert ledger.snapshot()["slots"][0]["released_for_more_calls"] is False


@pytest.mark.unit
def test_parallel_checkout_cannot_exceed_one_slot(tmp_path):
    import threading

    ledger = _ledger(tmp_path, prior="0")
    ledger.reserve(_probe(), "probe")
    llm = _LLM(_ok)
    install_budget_gate(llm, ledger)
    barrier = threading.Barrier(2)
    outcomes = []

    def worker():
        barrier.wait()
        try:
            llm.client.create(model="deepseek-flash", messages=[{"role": "user", "content": "ok"}], max_tokens=32)
            outcomes.append("sent")
        except BudgetRefusal:
            outcomes.append("refused")

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(outcomes) == ["refused", "sent"]
    assert len(llm.client.calls) == 1


@pytest.mark.unit
def test_exception_retry_and_missing_usage_keep_the_reservation(tmp_path):
    def explode(_kwargs):
        raise RuntimeError("transport failed before a billable body")

    ledger = _ledger(tmp_path, prior="0")
    ledger.reserve(_probe(), "probe")
    llm = _LLM(explode)
    install_budget_gate(llm, ledger)
    with pytest.raises(RuntimeError):
        llm.client.create(model="deepseek-flash", messages=[{"role": "user", "content": "ok"}], max_tokens=32)
    slot = ledger.snapshot()["slots"][0]
    assert slot["status"] == "usage_missing"
    assert Decimal(slot["charged_cny"]) == Decimal(_probe()["reserved_cny"])
    with pytest.raises(BudgetRefusal):
        llm.client.create(model="deepseek-flash", messages=[{"role": "user", "content": "ok"}], max_tokens=32)
    assert len(llm.client.calls) == 1

    missing = _ledger(tmp_path / "missing", prior="0")
    missing.reserve(_probe(), "probe")
    quiet = _LLM(lambda _kwargs: {"usage": {}})
    install_budget_gate(quiet, missing)
    quiet.client.create(model="deepseek-flash", messages=[{"role": "user", "content": "ok"}], max_tokens=32)
    assert missing.snapshot()["slots"][0]["status"] == "usage_missing"

    llm.max_retries = 2
    with pytest.raises(BudgetRefusal):
        llm.client.create(model="deepseek-flash", messages=[{"role": "user", "content": "ok"}], max_tokens=32)
    assert len(llm.client.calls) == 1


@pytest.mark.unit
def test_resume_does_not_charge_twice_and_does_not_resend(tmp_path):
    path = tmp_path / "budget-ledger.json"
    probe = _probe()
    first = Ledger(path, "1", "0")
    first.reserve(probe, "same")
    again = Ledger(path, "1", "0")
    again.reserve(probe, "same")
    assert again.snapshot()["reserved_cny"] == first.snapshot()["reserved_cny"]
    with pytest.raises(BudgetRefusal):
        again.reserve(build_probe_plan(input_cap=128, output_cap=8), "same")
    index = again.begin_send(
        model="deepseek-flash",
        messages=[{"role": "user", "content": "ok"}],
        max_tokens=32,
        thinking="disabled",
        effort="none",
        stream=False,
    )
    reloaded = Ledger(path, "1", "0")
    with pytest.raises(BudgetRefusal):
        reloaded.begin_send(
            model="deepseek-flash",
            messages=[{"role": "user", "content": "ok"}],
            max_tokens=32,
            thinking="disabled",
            effort="none",
            stream=False,
        )
    assert reloaded.snapshot()["slots"][index]["status"] == "sending"


@pytest.mark.unit
def test_uncontrolled_path_and_price_change_are_refused(tmp_path):
    bare = _LLM(_ok)
    with pytest.raises(BudgetRefusal):
        assert_controlled(bare)
    ledger = _ledger(tmp_path, prior="0")
    ledger.reserve(_probe(), "probe")
    llm = _LLM(_ok)
    install_budget_gate(llm, ledger)
    assert_controlled(llm)
    card = official_price_card()
    dearer = PriceCard(
        source=card.source,
        retrieved_at=card.retrieved_at,
        expires_at=card.expires_at,
        currency=card.currency,
        models={
            "deepseek-flash": card.models["deepseek-flash"],
            "deepseek-v4-pro": card.models["deepseek-v4-pro"].__class__(
                "deepseek-v4-pro",
                "changed",
                Decimal("20"),
                Decimal("40"),
            ),
        },
    )
    ledger.card = dearer
    with pytest.raises(BudgetRefusal):
        llm.client.create(model="deepseek-flash", messages=[{"role": "user", "content": "ok"}], max_tokens=32)
    assert llm.client.calls == []


@pytest.mark.unit
def test_overrun_freezes_later_sends(tmp_path):
    def too_many(_kwargs):
        return {"usage": {"prompt_tokens": 10_000, "completion_tokens": 1}}

    ledger = _ledger(tmp_path, prior="0")
    ledger.reserve(_probe(), "probe")
    llm = _LLM(too_many)
    install_budget_gate(llm, ledger)
    llm.client.create(model="deepseek-flash", messages=[{"role": "user", "content": "ok"}], max_tokens=32)
    assert ledger.snapshot()["frozen"] is True
    with pytest.raises(BudgetRefusal):
        ledger.begin_send(
            model="deepseek-flash",
            messages=[{"role": "user", "content": "ok"}],
            max_tokens=32,
            thinking="disabled",
            effort="none",
            stream=False,
        )


@pytest.mark.unit
def test_full_research_is_not_guaranteed_inside_one_yuan():
    card = official_price_card()
    plan = build_plan(
        quick_model="deepseek-flash",
        deep_model="deepseek-v4-pro",
        analysts=["market"],
        tool_rounds=8,
        debate_rounds=1,
        risk_rounds=1,
        input_cap=8192,
        output_cap=4096,
        retries=0,
        card=card,
    )
    afford = assess_affordability("1", str(conservative_probe_hold()), plan)
    assert plan["completion_guaranteed"] is False
    assert Decimal(plan["one_unbounded_context_call_cny"]) > Decimal("1")
    assert afford["send_allowed"] is False
    with pytest.raises(BudgetRefusal):
        research_call_counts(["social"], tool_rounds=1, debate_rounds=1, risk_rounds=1)
    assert call_cost(card, "deepseek-v4-pro", 1, 0) > call_cost(card, "deepseek-flash", 1, 0)


@pytest.mark.unit
def test_research_run_refuses_before_the_graph(monkeypatch, tmp_path):
    def explode(*_args, **_kwargs):
        raise AssertionError("graph should not be constructed")

    monkeypatch.setattr("tradingagents.graph.trading_graph.TradingAgentsGraph", explode)
    code = execute_run(
        tmp_path,
        "0700.HK",
        "HK",
        "2026-11-06",
        provider="deepseek",
        quick_model="deepseek-flash",
        deep_model="deepseek-v4-pro",
        backend_url=None,
        analysts="market",
        limit_overrides={"llm_max_retries": 0},
        environ={"DEEPSEEK_API_KEY": "test-not-real"},
        budget_cny="1",
        budget_mode="research",
    )
    assert code == 2
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["model"]["called"] is False
    assert status["research"]["end_to_end"] is False
    assert status["extra"]["full_research_reliable_within_budget"] is False
    assert "test-not-real" not in (tmp_path / "status.json").read_text(encoding="utf-8")


@pytest.mark.unit
def test_probe_mode_sends_once_through_the_gate(monkeypatch, tmp_path):
    class Fake:
        def __init__(self, **_kwargs):
            self.max_retries = 0
            self.root_client = self
            self.client = self
            self.async_client = None
            self.calls = []

            def create(**kwargs):
                self.calls.append(kwargs)
                return {"usage": {"prompt_tokens": 2, "completion_tokens": 1}}

            self.create = create

    monkeypatch.setattr("tradingagents.llm_clients.openai_client.DeepSeekChatOpenAI", Fake)
    code = execute_run(
        tmp_path,
        "0700.HK",
        "HK",
        "2026-11-06",
        provider="deepseek",
        quick_model="deepseek-flash",
        deep_model="deepseek-flash",
        backend_url=None,
        analysts="market",
        limit_overrides={"llm_max_retries": 0},
        environ={"DEEPSEEK_API_KEY": "test-not-real"},
        budget_cny="1",
        budget_mode="probe",
    )
    assert code == 0
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["model"]["called"] is True
    assert status["research"]["end_to_end"] is False
    assert status["research"]["decision"] is None
    assert "test-not-real" not in (tmp_path / "status.json").read_text(encoding="utf-8")


@pytest.mark.unit
def test_rounding_never_drops_a_fraction():
    card = official_price_card()
    custom = PriceCard(
        source=card.source,
        retrieved_at=card.retrieved_at,
        expires_at=card.expires_at,
        currency="CNY",
        models={
            "deepseek-flash": card.models["deepseek-flash"].__class__(
                "deepseek-flash",
                card.models["deepseek-flash"].version,
                Decimal("0.0000000000001"),
                Decimal("0"),
            ),
        },
    )
    assert call_cost(custom, "deepseek-flash", 1, 0) == Decimal("0.000000000001")

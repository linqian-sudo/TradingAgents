"""Missing usage, structured fallback, and an interrupted slot stop the ledger."""

from __future__ import annotations

import json
from decimal import Decimal

import pytest
from langchain_core.messages import HumanMessage

from research_entry.budget import Ledger, build_plan
from research_entry.budget_gate import install_chat_gate
from research_entry.cli import main
from tradingagents.agents.schemas import ResearchPlan
from tradingagents.agents.structured import invoke_structured_or_freetext
from tradingagents.graph import trading_graph
from tradingagents.llm_clients.openai_client import DeepSeekChatOpenAI

_RUN = [
    "run", "--symbol", "0700.HK", "--market", "HK", "--date", "2026-11-06",
    "--provider", "deepseek", "--quick-model", "deepseek-flash", "--deep-model", "deepseek-flash",
    "--analysts", "market,news,fundamentals",
    "--max-tool-rounds", "1", "--max-debate-rounds", "1", "--max-risk-rounds", "1",
    "--max-tokens", "1024", "--max-retries", "0",
    "--budget-cny", "9.5", "--prior-hold-cny", "0.008448", "--input-token-cap", "24000",
    "--budget-mode", "research",
]


class _Raw:
    def __init__(self, body):
        self._body = body
        self.headers = {}

    def parse(self):
        return self._body


def _completion(content, usage, finish="stop", tool_calls=None):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    body = {
        "id": "mock",
        "model": "deepseek-flash",
        "choices": [{"index": 0, "finish_reason": finish, "message": message}],
    }
    if usage is not None:
        body["usage"] = usage
    return _Raw(body)


def test_missing_usage_freezes_before_another_http_call(tmp_path):
    calls = []

    def fake_create(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return _completion("ok", {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6})
        return _completion("later", None)

    llm = DeepSeekChatOpenAI(
        model="deepseek-flash", api_key="not-a-real-key",
        base_url="https://api.deepseek.com", max_tokens=1024, max_retries=0,
    )
    llm.client.create = fake_create
    ledger = Ledger(tmp_path / "ledger.json", "9.5", "0.008448")
    plan = build_plan(
        quick_model="deepseek-flash", deep_model="deepseek-flash", analysts=["market"],
        tool_rounds=1, debate_rounds=1, risk_rounds=1, input_cap=24000, output_cap=1024, retries=0,
    )
    ledger.reserve(plan, "missing")
    gated = install_chat_gate(llm, ledger, input_cap=24000, output_cap=1024, expected_model="deepseek-flash")
    gated.invoke([HumanMessage(content="ok")])
    gated.invoke([HumanMessage(content="ok")])
    with pytest.raises(Exception, match="冻结"):
        gated.invoke([HumanMessage(content="ok")])
    assert len(calls) == 2
    assert calls[0]["max_tokens"] == 1024
    assert calls[0]["extra_body"]["thinking"]["type"] == "disabled"
    slots = [slot for slot in ledger.snapshot()["slots"] if slot["status"] != "reserved"]
    assert slots[1]["status"] == "usage_missing"
    assert ledger.snapshot()["frozen"] is True


def test_structured_parse_failure_does_not_send_free_text(tmp_path):
    calls = []

    def fake_create(**kwargs):
        calls.append(kwargs)
        return _completion("not a schema", {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12})

    llm = DeepSeekChatOpenAI(
        model="deepseek-flash", api_key="not-a-real-key",
        base_url="https://api.deepseek.com", max_tokens=1024, max_retries=0,
    )
    llm.client.create = fake_create
    ledger = Ledger(tmp_path / "ledger.json", "9.5", "0.008448")
    plan = build_plan(
        quick_model="deepseek-flash", deep_model="deepseek-flash", analysts=["market"],
        tool_rounds=1, debate_rounds=1, risk_rounds=1, input_cap=24000, output_cap=1024, retries=0,
    )
    ledger.reserve(plan, "structured")
    gated = install_chat_gate(llm, ledger, input_cap=24000, output_cap=1024, expected_model="deepseek-flash")
    text = invoke_structured_or_freetext(
        gated.with_structured_output(ResearchPlan),
        gated,
        "Write the plan",
        lambda result: result.rationale,
        "Research Manager",
    )
    assert "没有第二次模型请求" in text
    assert len(calls) == 1


def test_interrupted_sending_slot_freezes_on_reload(tmp_path):
    path = tmp_path / "ledger.json"
    ledger = Ledger(path, "9.5", "0.008448")
    plan = build_plan(
        quick_model="deepseek-flash", deep_model="deepseek-flash", analysts=["market"],
        tool_rounds=1, debate_rounds=1, risk_rounds=1, input_cap=24000, output_cap=1024, retries=0,
    )
    ledger.reserve(plan, "dead")
    ledger.begin_send(
        model="deepseek-flash",
        messages=[{"role": "user", "content": "ok"}],
        max_tokens=1024, thinking="disabled", effort="none", stream=False,
    )
    reloaded = Ledger(path, "9.5", "0.008448")
    snapshot = reloaded.snapshot()
    assert snapshot["frozen"] is True
    assert snapshot["slots"][0]["status"] == "usage_missing"
    assert Decimal(snapshot["slots"][0]["charged_cny"]) > 0
    with pytest.raises(Exception, match="冻结"):
        reloaded.begin_send(
            model="deepseek-flash",
            messages=[{"role": "user", "content": "ok"}],
            max_tokens=1024, thinking="disabled", effort="none", stream=False,
        )


def test_cli_stops_when_a_real_client_response_has_no_usage(tmp_path, monkeypatch):
    calls = []

    def fake_create(**kwargs):
        calls.append(kwargs)
        usage = {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30} if len(calls) < 3 else None
        finish = "length" if len(calls) == 1 else "stop"
        return _completion(
            "Report.\n\n**Rating**: Overweight\n\nFINAL TRANSACTION PROPOSAL: **BUY**",
            usage,
            finish=finish,
        )

    def factory(config, tier, **_kwargs):
        llm = DeepSeekChatOpenAI(
            model="deepseek-flash", api_key="not-a-real-key",
            base_url="https://api.deepseek.com", max_tokens=1024, max_retries=0,
        )
        llm.client.create = fake_create

        class _Client:
            def get_llm(self):
                return llm

        return _Client()

    monkeypatch.setattr(trading_graph, "create_tier_client", factory)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "not-a-real-key")
    with pytest.raises(SystemExit) as exc:
        main([*_RUN, "--output", str(tmp_path)])
    assert exc.value.code == 2
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["research"]["end_to_end"] is False
    assert status["research"]["decision"] is None
    assert status["model"]["called"] is True
    assert any(slot["status"] == "usage_missing" for slot in status["extra"]["ledger"]["slots"])
    assert status["extra"]["ledger"]["frozen"] is True
    assert len(calls) >= 1
    assert all(call["max_tokens"] == 1024 for call in calls)
    assert all(call["extra_body"]["thinking"]["type"] == "disabled" for call in calls)
    assert (tmp_path / "budget-journal.jsonl").exists()
    assert "not-a-real-key" not in (tmp_path / "status.json").read_text(encoding="utf-8")

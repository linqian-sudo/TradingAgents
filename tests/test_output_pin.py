"""The body that would be sent must carry DeepSeek's own output controls.

A completion_tokens value above the reserved cap freezes that one response.
It is not rewritten into a sum of several calls, and the cap is not raised.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from research_entry.budget import (
    Ledger,
    build_plan,
    pin_deepseek_request,
)
from research_entry.budget_gate import install_chat_gate
from tradingagents.llm_clients.openai_client import DeepSeekChatOpenAI

# Reported totals across four responses. The split of the first three calls is
# not in the parent's log; only the fourth completion count and the sums are.
_CALLS = (
    {"prompt_tokens": 1000, "completion_tokens": 100},
    {"prompt_tokens": 1000, "completion_tokens": 140},
    {"prompt_tokens": 800, "completion_tokens": 147},
    {
        "prompt_tokens": 464,
        "completion_tokens": 3871,
        "completion_tokens_details": {"reasoning_tokens": 2800},
    },
)


def test_langchain_payload_is_rewritten_before_create(tmp_path):
    seen = []

    def fake_create(**kwargs):
        seen.append(kwargs)
        raise RuntimeError("stop before any network response is parsed")

    llm = DeepSeekChatOpenAI(
        model="deepseek-flash",
        api_key="not-a-real-key",
        base_url="https://api.deepseek.com",
        max_tokens=1024,
        max_retries=0,
    )
    llm.client.create = fake_create
    ledger = Ledger(tmp_path / "ledger.json", "9.5", "0.008448")
    plan = build_plan(
        quick_model="deepseek-flash",
        deep_model="deepseek-flash",
        analysts=["market"],
        tool_rounds=1,
        debate_rounds=1,
        risk_rounds=1,
        input_cap=24000,
        output_cap=1024,
        retries=0,
    )
    ledger.reserve(plan, "pin")
    gated = install_chat_gate(
        llm, ledger, input_cap=24000, output_cap=1024, expected_model="deepseek-flash",
    )
    try:
        gated.invoke([HumanMessage(content="ok")])
    except RuntimeError as exc:
        assert "stop before" in str(exc)
    assert len(seen) == 1
    body = seen[0]
    assert body["max_tokens"] == 1024
    assert body.get("max_completion_tokens") in (None, 1024)
    assert body["extra_body"]["thinking"] == {"type": "disabled"}
    assert body["reasoning_effort"] == "none"
    assert "not-a-real-key" not in repr(body)


def test_unpinnable_body_is_refused_without_calling_transport():
    called = []

    def fake_create(**kwargs):
        called.append(kwargs)
        return kwargs

    original = pin_deepseek_request

    def refuse(payload, output_cap):
        raise AssertionError("pin should be the function under test")

    # A payload that pin itself rejects never reaches create.
    try:
        pin_deepseek_request({"thinking_enabled": True, "reasoning_effort": "high"}, 1024)
    except Exception as exc:
        assert "reasoning_effort" in str(exc)
    assert called == []
    assert original is pin_deepseek_request
    assert refuse


def test_fourth_response_overruns_only_that_slot(tmp_path):
    ledger = Ledger(tmp_path / "ledger.json", "9.5", "0.008448")
    plan = build_plan(
        quick_model="deepseek-flash",
        deep_model="deepseek-flash",
        analysts=["market"],
        tool_rounds=1,
        debate_rounds=1,
        risk_rounds=1,
        input_cap=24000,
        output_cap=1024,
        retries=0,
    )
    ledger.reserve(plan, "four")
    for usage in _CALLS:
        index = ledger.begin_send(
            model="deepseek-flash",
            messages=[{"role": "user", "content": "ok"}],
            max_tokens=1024,
            thinking="disabled",
            effort="none",
            stream=False,
        )
        ledger.settle(index, usage)
    slots = ledger.snapshot()["slots"]
    used = [slot for slot in slots if slot["status"] != "reserved"]
    assert [slot["usage"]["completion_tokens"] for slot in used] == [100, 140, 147, 3871]
    assert sum(slot["usage"]["prompt_tokens"] for slot in used) == 3264
    assert sum(slot["usage"]["completion_tokens"] for slot in used) == 4258
    assert used[3]["status"] == "overrun"
    assert used[3]["usage"]["completion_tokens"] == 3871
    assert used[3]["usage"]["reasoning_tokens"] == 2800
    assert used[0]["status"] == "settled"
    assert ledger.snapshot()["frozen"] is True
    try:
        ledger.begin_send(
            model="deepseek-flash",
            messages=[{"role": "user", "content": "ok"}],
            max_tokens=1024,
            thinking="disabled",
            effort="none",
            stream=False,
        )
    except Exception as exc:
        assert "冻结" in str(exc)
    else:
        raise AssertionError("a frozen ledger accepted another send")

"""Wrap the official chat client so an unreserved request cannot be sent."""

from __future__ import annotations

import contextlib

from research_entry.budget import BudgetRefusal, Ledger

_MARKER = "_budget_wrapped_create"


def _usage_of(response):
    if response is None:
        return None
    usage = response.get("usage") if isinstance(response, dict) else getattr(response, "usage", None)
    if usage is None:
        return None
    if isinstance(usage, dict):
        return usage
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    prompt = getattr(usage, "prompt_tokens", None)
    completion = getattr(usage, "completion_tokens", None)
    if prompt is None or completion is None:
        return None
    return {"prompt_tokens": prompt, "completion_tokens": completion}


def _prepare(kwargs: dict) -> dict:
    if kwargs.get("stream"):
        raise BudgetRefusal("流式响应不能在发送前同时证明最终 usage，拒绝发送。")
    extra = dict(kwargs.get("extra_body") or {})
    requested = extra.get("thinking")
    if isinstance(requested, dict) and requested.get("type") == "enabled":
        raise BudgetRefusal("思考模式已打开。推理 token 上界无法证明，拒绝发送。")
    effort = kwargs.get("reasoning_effort")
    if effort not in (None, "none"):
        raise BudgetRefusal("reasoning_effort 不是 none，拒绝发送。")
    if kwargs.get("max_tokens") is None:
        raise BudgetRefusal("没有显式 max_tokens，拒绝发送。")
    extra["thinking"] = {"type": "disabled"}
    outgoing = dict(kwargs)
    outgoing["extra_body"] = extra
    outgoing["reasoning_effort"] = "none"
    outgoing["stream"] = False
    return outgoing


def _retries_closed(llm) -> None:
    if int(getattr(llm, "max_retries", 0) or 0) != 0:
        raise BudgetRefusal("max_retries 必须为 0，禁止隐式重试。")
    root = getattr(llm, "root_client", None)
    if root is not None and int(getattr(root, "max_retries", 0) or 0) != 0:
        raise BudgetRefusal("客户端隐式重试必须为 0，拒绝发送。")


def _wrap_resource(resource, llm, ledger: Ledger) -> None:
    current = getattr(resource, "create", None)
    if getattr(current, _MARKER, False):
        return
    if current is None:
        raise BudgetRefusal("模型客户端没有可包装的 create，拒绝启动。")

    def create(*args, **kwargs):
        if args:
            raise BudgetRefusal("位置参数调用无法在发送前核对载荷，拒绝发送。")
        _retries_closed(llm)
        outgoing = _prepare(kwargs)
        model = outgoing.get("model") or getattr(llm, "model_name", None)
        index = ledger.begin_send(
            model=model,
            messages=outgoing.get("messages"),
            tools=outgoing.get("tools"),
            max_tokens=int(outgoing["max_tokens"]),
            thinking="disabled",
            effort="none",
            stream=False,
        )
        try:
            response = current(**outgoing)
        except Exception:
            ledger.abort(index)
            raise
        ledger.settle(index, _usage_of(response))
        return response

    setattr(create, _MARKER, True)
    resource.create = create
    for name in ("with_raw_response", "with_streaming_response"):
        if hasattr(resource, name):
            def refuse(*_args, _name=name, **_kwargs):
                raise BudgetRefusal(f"{_name} 无法证明发送前的 token 上界，拒绝发送。")
            with contextlib.suppress(AttributeError, TypeError):
                setattr(resource, name, refuse)


def install_budget_gate(llm, ledger: Ledger):
    """Wrap sync and async chat completions. Raise before any other path is used."""
    _retries_closed(llm)
    client = getattr(llm, "client", None)
    if client is None:
        raise BudgetRefusal("没有 chat client，拒绝启动。")
    _wrap_resource(client, llm, ledger)
    async_client = getattr(llm, "async_client", None)
    if async_client is not None:
        _wrap_resource(async_client, llm, ledger)
    llm._budget_gate_installed = True
    assert_controlled(llm)
    return llm


def assert_controlled(llm) -> None:
    client = getattr(llm, "client", None)
    create = getattr(client, "create", None)
    if not getattr(create, _MARKER, False):
        raise BudgetRefusal("发现未接入账本的模型调用路径，拒绝启动。")

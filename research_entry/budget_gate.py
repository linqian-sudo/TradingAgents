"""Wrap every chat call so an unreserved request cannot be sent."""

from __future__ import annotations

import contextlib
import threading

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import ConfigDict, Field

from research_entry.budget import (
    BudgetRefusal,
    Ledger,
    pin_deepseek_request,
    prepare_for_send,
    tool_schema,
)

_PERMIT = threading.local()

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


def _install_payload_pin(llm, output_cap: int) -> None:
    original = getattr(llm, "_get_request_payload", None)
    if original is None or getattr(original, "_budget_payload_pin", False):
        return

    def _get_request_payload(input_, *, stop=None, **kwargs):
        return pin_deepseek_request(original(input_, stop=stop, **kwargs), output_cap)

    _get_request_payload._budget_payload_pin = True
    llm._get_request_payload = _get_request_payload


def guard_transport(llm, output_cap: int) -> None:
    """Rewrite the body, then reject a create that has no reservation."""
    client = getattr(llm, "client", None)
    if client is None or not hasattr(client, "create"):
        return
    current = client.create
    if getattr(current, "_budget_transport_guard", False):
        return

    def guarded(*args, **kwargs):
        if args:
            raise BudgetRefusal("位置参数调用无法核对输出上限，拒绝发送。")
        pinned = pin_deepseek_request(kwargs, output_cap)
        consume_transport_permit()
        return current(**pinned)

    guarded._budget_transport_guard = True
    client.create = guarded


def install_chat_gate(llm, ledger: Ledger, *, input_cap: int, output_cap: int, expected_model: str):
    """Return the model the graph must use. Direct creates without a permit fail."""
    _install_payload_pin(llm, output_cap)
    guard_transport(llm, output_cap)
    return BudgetedChatModel(
        inner=llm,
        ledger=ledger,
        input_cap=input_cap,
        output_cap=output_cap,
        expected_model=expected_model,
    )


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


def _open_permit() -> None:
    _PERMIT.depth = getattr(_PERMIT, "depth", 0) + 1


def _close_permit() -> None:
    _PERMIT.depth = 0


def consume_transport_permit() -> None:
    """A raw HTTP create must sit inside a reservation made on this thread."""
    depth = getattr(_PERMIT, "depth", 0)
    if depth <= 0:
        raise BudgetRefusal("传输层没有发送前预留，拒绝发送。")
    _PERMIT.depth = depth - 1


def _usage_from_result(result):
    llm_output = getattr(result, "llm_output", None) or {}
    usage = llm_output.get("token_usage") or llm_output.get("usage")
    return usage if isinstance(usage, dict) else None


class _ChargedRunnable(Runnable):
    """One structured or delegated invoke, charged before the inner call."""

    def __init__(self, inner, ledger: Ledger, input_cap: int, output_cap: int, expected_model: str):
        self.inner = inner
        self.ledger = ledger
        self.input_cap = input_cap
        self.output_cap = output_cap
        self.expected_model = expected_model

    def invoke(self, input, config=None, **kwargs):
        prepared, _bound, truncated = prepare_for_send(input, None, self.input_cap)
        index = self.ledger.begin_send(
            model=self.expected_model,
            messages=prepared if isinstance(prepared, list) else [prepared],
            tools=None,
            max_tokens=self.output_cap,
            thinking="disabled",
            effort="none",
            stream=False,
            truncated=truncated,
        )
        _open_permit()
        try:
            if isinstance(self.inner, RunnableLambda):
                result = self.inner.invoke(input, config, **kwargs)
            else:
                result = self.inner.invoke(prepared, config, **kwargs)
        except Exception:
            self.ledger.abort(index)
            raise
        else:
            self.ledger.settle(index, _usage_from_result(result))
            return result
        finally:
            _close_permit()


class BudgetedChatModel(BaseChatModel):
    """LangChain model whose every generate or structured call pays the ledger first."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    inner: object = Field(exclude=True)
    ledger: object = Field(exclude=True)
    input_cap: int
    output_cap: int
    expected_model: str
    bound_tools: tuple = ()

    @property
    def _llm_type(self) -> str:
        return "budgeted-chat"

    def bind_tools(self, tools, **kwargs):
        bound = self.inner.bind_tools(tools, **kwargs)
        return BudgetedChatModel(
            inner=bound,
            ledger=self.ledger,
            input_cap=self.input_cap,
            output_cap=self.output_cap,
            expected_model=self.expected_model,
            bound_tools=tuple(tools),
        )

    def with_structured_output(self, schema, **kwargs):
        bound = self.inner.with_structured_output(schema, **kwargs)
        return _ChargedRunnable(bound, self.ledger, self.input_cap, self.output_cap, self.expected_model)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        prepared, _bound, truncated = prepare_for_send(messages, self.bound_tools, self.input_cap)
        max_tokens = int(kwargs.get("max_tokens") or self.output_cap)
        if max_tokens > self.output_cap:
            raise BudgetRefusal(f"max_tokens {max_tokens} 超过输出上界 {self.output_cap}，拒绝发送。")
        schema = tool_schema(self.bound_tools)
        index = self.ledger.begin_send(
            model=self.expected_model,
            messages=prepared,
            tools=schema,
            max_tokens=max_tokens,
            thinking="disabled",
            effort="none",
            stream=False,
            truncated=truncated,
        )
        _open_permit()
        call_kwargs = dict(kwargs)
        call_kwargs["max_tokens"] = max_tokens
        try:
            result = self.inner._generate(prepared, stop=stop, run_manager=run_manager, **call_kwargs)
        except Exception:
            self.ledger.abort(index)
            raise
        else:
            self.ledger.settle(index, _usage_from_result(result))
            return result
        finally:
            _close_permit()


def assert_controlled(llm) -> None:
    client = getattr(llm, "client", None)
    create = getattr(client, "create", None)
    if not getattr(create, _MARKER, False):
        raise BudgetRefusal("发现未接入账本的模型调用路径，拒绝启动。")

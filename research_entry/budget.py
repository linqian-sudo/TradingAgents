"""Run-level DeepSeek budget reservation.

Prices are the official CNY card retrieved on 2026-11-08. The card is not a
promise that DeepSeek will keep billing that way. When the card is expired,
the model is unknown, or a request has no proven token ceiling, nothing is sent.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import ROUND_UP, Decimal
from pathlib import Path

PRICE_SOURCE = "https://api-docs.deepseek.com/zh-cn/quick_start/pricing"
PRICE_RETRIEVED_AT = "2026-11-08T09:00:00Z"
PRICE_EXPIRES_AT = "2026-11-15T00:00:00Z"
OFFICIAL_CHAT_URLS = frozenset({
    "https://api.deepseek.com",
    "https://api.deepseek.com/",
    "https://api.deepseek.com/v1",
    "https://api.deepseek.com/v1/",
})
MILLION = Decimal("1000000")
QUANTUM = Decimal("0.000000000001")
TEMPLATE_OVERHEAD_TOKENS = 128
CONTEXT_TOKEN_CEILING = 1_000_000
OUTPUT_TOKEN_CEILING = 384 * 1024
# The parent's basic call reported 7 prompt tokens and 2 completion tokens.
# Peak cache-miss usage of those counts is 0.000030 CNY. The call was not
# made through this ledger, so the held amount is a 4096/32 reservation.
PROBE_PROMPT_TOKENS = 7
PROBE_COMPLETION_TOKENS = 2
PRIOR_RESERVATION_INPUT_TOKENS = 4096
PRIOR_RESERVATION_OUTPUT_TOKENS = 32
# A full research reserves this measured ceiling per call, never the 1M window.
RESEARCH_INPUT_CAP = 24_000
RESEARCH_OUTPUT_CAP = 1024
RESEARCH_TOOL_ROUNDS = 1
RESEARCH_ANALYSTS = ("market", "news", "fundamentals")
TRUNCATION_BANNER = (
    "\n[已截断] 本段超过本次请求的输入上界。已保留开头的标识、记录数、区间和结尾的最近记录；"
    "中间明细未送入模型。不得把这段写成完整原始数据。\n"
)

_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


class BudgetRefusal(Exception):
    """A request must not be sent."""


def _money(value: Decimal) -> Decimal:
    return value.quantize(QUANTUM, rounding=ROUND_UP)


def _parse_money(value: str | Decimal) -> Decimal:
    if isinstance(value, Decimal):
        amount = value
    else:
        try:
            amount = Decimal(str(value).strip())
        except Exception as exc:
            raise BudgetRefusal(f"金额不是十进制数：{value!r}") from exc
    if amount.is_nan() or amount < 0:
        raise BudgetRefusal(f"金额必须是非负小数，收到 {value!r}")
    return _money(amount)


def _stamp(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


@dataclass(frozen=True)
class ModelPrice:
    model: str
    version: str
    input_miss_peak_per_million: Decimal
    output_peak_per_million: Decimal

    def input_price(self) -> Decimal:
        return self.input_miss_peak_per_million / MILLION

    def output_price(self) -> Decimal:
        return self.output_peak_per_million / MILLION


@dataclass(frozen=True)
class PriceCard:
    source: str
    retrieved_at: str
    expires_at: str
    currency: str
    models: dict[str, ModelPrice]

    def digest(self) -> str:
        payload = {
            "source": self.source,
            "retrieved_at": self.retrieved_at,
            "expires_at": self.expires_at,
            "currency": self.currency,
            "models": {
                name: {
                    "version": item.version,
                    "input_miss_peak_per_million": str(item.input_miss_peak_per_million),
                    "output_peak_per_million": str(item.output_peak_per_million),
                }
                for name, item in sorted(self.models.items())
            },
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
        return hashlib.sha256(raw).hexdigest()


def official_price_card() -> PriceCard:
    """CNY prices from the official Chinese pricing page, peak and cache-miss only."""
    return PriceCard(
        source=PRICE_SOURCE,
        retrieved_at=PRICE_RETRIEVED_AT,
        expires_at=PRICE_EXPIRES_AT,
        currency="CNY",
        models={
            "deepseek-flash": ModelPrice(
                "deepseek-flash",
                "DeepSeek-V4.1-Flash-1017",
                Decimal("2"),
                Decimal("8"),
            ),
            "deepseek-v4-pro": ModelPrice(
                "deepseek-v4-pro",
                "DeepSeek-V4-Pro-0813",
                Decimal("9.0"),
                Decimal("27.0"),
            ),
        },
    )


def require_fresh(card: PriceCard, now: datetime | None = None) -> None:
    current = now or datetime.now(UTC)
    if current.tzinfo is None:
        current = current.replace(tzinfo=UTC)
    if current >= _stamp(card.expires_at):
        raise BudgetRefusal(
            f"价格卡已于 {card.expires_at} 失效。不能沿用旧价发送请求。"
        )


def require_endpoint(backend_url: str | None) -> None:
    if backend_url is None or not str(backend_url).strip():
        return
    if str(backend_url).strip() not in OFFICIAL_CHAT_URLS:
        raise BudgetRefusal(
            "只接受官方 Chat Completions 地址 https://api.deepseek.com。"
            "Anthropic 格式、代理和其他兼容地址无法证明计费，拒绝发送。"
        )


def model_price(card: PriceCard, model: str) -> ModelPrice:
    name = (model or "").strip()
    if name in {"deepseek-v4-flash", "deepseek-v4-flash-vision-exp", "deepseek-chat", "deepseek-reasoner"}:
        raise BudgetRefusal(f"不沿用已退役或旧别名 {name}。请使用 deepseek-flash 或 deepseek-v4-pro。")
    item = card.models.get(name)
    if item is None:
        raise BudgetRefusal(f"价格卡没有模型 {name or '空'}，拒绝发送。")
    return item


def call_cost(card: PriceCard, model: str, input_tokens: int, output_tokens: int) -> Decimal:
    if input_tokens < 0 or output_tokens < 0:
        raise BudgetRefusal("token 数不能为负")
    item = model_price(card, model)
    amount = (Decimal(input_tokens) * item.input_price()) + (Decimal(output_tokens) * item.output_price())
    return _money(amount)


def conservative_probe_hold(card: PriceCard | None = None) -> Decimal:
    """Peak cache-miss price of the reported 7/2 usage. Not the ledger hold."""
    return call_cost(card or official_price_card(), "deepseek-flash", PROBE_PROMPT_TOKENS, PROBE_COMPLETION_TOKENS)


def conservative_prior_reservation_hold(card: PriceCard | None = None) -> Decimal:
    """Hold a 4096-input / 32-output reservation for the unbounded basic call."""
    return call_cost(
        card or official_price_card(),
        "deepseek-flash",
        PRIOR_RESERVATION_INPUT_TOKENS,
        PRIOR_RESERVATION_OUTPUT_TOKENS,
    )


def text_token_upper_bound(text: str) -> int:
    """UTF-8 byte length. One byte is the ceiling used for text tokens.

    A code-point count underestimates emoji and other non-ASCII characters.
    This bound does not. It is an upper bound only if the server tokenizer
    never spends more than one token on one UTF-8 byte, plus the separate
    template overhead applied by ``request_token_upper_bound``.
    """
    if not isinstance(text, str):
        raise BudgetRefusal(f"非字符串没有可证明的 token 上界：{type(text).__name__}")
    return len(text.encode("utf-8"))


def _walk_text(value, where: str):
    if value is None:
        return
    if isinstance(value, str):
        yield value
        return
    if isinstance(value, bool):
        yield "true" if value else "false"
        return
    if isinstance(value, int):
        yield str(value)
        return
    if isinstance(value, float):
        raise BudgetRefusal(f"{where} 含浮点数，序列化后的 token 上界不稳定，拒绝发送。")
    if isinstance(value, dict):
        kind = str(value.get("type") or "")
        if kind in {"image_url", "image", "input_audio", "audio", "file", "input_image"}:
            raise BudgetRefusal(f"{where} 含非文本内容（{kind}），拒绝发送。")
        for key, item in value.items():
            yield from _walk_text(item, f"{where}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _walk_text(item, f"{where}[{index}]")
        return
    raise BudgetRefusal(f"{where} 的类型 {type(value).__name__} 没有可证明的 token 上界。")


def request_token_upper_bound(messages, tools=None) -> int:
    texts = list(_walk_text(messages, "messages"))
    total = sum(text_token_upper_bound(item) for item in texts)
    message_count = len(messages) if isinstance(messages, list) else 1
    total += TEMPLATE_OVERHEAD_TOKENS * max(message_count, 1)
    if tools:
        if not isinstance(tools, list):
            raise BudgetRefusal("tools 必须是列表，否则无法证明上界。")
        rendered = json.dumps(tools, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        total += text_token_upper_bound(rendered)
        total += TEMPLATE_OVERHEAD_TOKENS * len(tools)
    return total


ASSUMPTIONS = (
    "价格卡是 2026-11-08 官方中文页的人民币高峰价，缓存一律按未命中。过期、改价或未知模型拒绝发送。这不能约束 DeepSeek 日后的扣费政策。",
    "只覆盖官方 OpenAI Chat Completions（https://api.deepseek.com）。流式、Responses、Anthropic 格式、FIM、Search/Fetch 和旧模型名都拒绝。",
    "思考模式必须在即将发出的请求体里写成 thinking.type=disabled，输出上限必须写成官方字段 max_tokens。只在账本里记下 1024 并不等于供应商收到了这个上限。",
    "completion_tokens 按单次响应记账，不把多次相加后再当成一次。reasoning_tokens 只作明细，不重复加到 completion_tokens 上。",
    "请求发出后如果供应商返回的 completion_tokens 仍大于 max_tokens，费用无法撤回。账本会冻结并停止后续请求。这不是对 9.5 元账单的绝对保证。",
    "输入上界是载荷文本的 UTF-8 字节数，再加每条消息和每个工具 128 token 的模板余量。非文本内容拒绝。该字节上界依赖“一个 UTF-8 字节至多一个 token”。",
    "SDK 隐式重试必须为 0。计划外调用、usage 缺失都不退预留，也不再追加调用。",
    "完整研究按每次请求的输入上界和 max_tokens 预留，不按 100 万上下文预留。上界装不进剩余预算，或压缩后仍超过上界，就在发送前拒绝。",
    "工具文本超限时保留开头标识和结尾最近记录，并写入截断声明。这不能把被截掉的明细说成已经分析过。",
    "价格于 2026-11-08 再次核对官网，数字未变。价格卡仍在 2026-11-15 00:00 UTC 失效。失效后拒绝发送。",
)


def message_payload(messages) -> list:
    """Plain messages whose text is exactly what the token bound measures."""
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    rendered = []
    for message in messages or []:
        if isinstance(message, str):
            rendered.append({"role": "user", "content": message})
            continue
        if isinstance(message, dict):
            rendered.append(message)
            continue
        content = getattr(message, "content", "")
        item = {"role": str(getattr(message, "type", "unknown")), "content": content}
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            item["tool_calls"] = tool_calls
        rendered.append(item)
    return rendered


def tool_schema(tools) -> list | None:
    if not tools:
        return None
    rendered = []
    for tool in tools:
        if isinstance(tool, dict):
            rendered.append(tool)
            continue
        rendered.append({
            "name": str(getattr(tool, "name", "")),
            "description": str(getattr(tool, "description", "") or ""),
        })
    return rendered


def _clone_message(message):
    if isinstance(message, (str, dict)):
        return message if isinstance(message, str) else dict(message)
    if hasattr(message, "model_copy"):
        return message.model_copy()
    raise BudgetRefusal(f"无法复制消息 {type(message).__name__}，拒绝发送。")


def _content_of(message) -> str | None:
    if isinstance(message, str):
        return message
    content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
    return content if isinstance(content, str) else None


def _set_content(message, content: str):
    if isinstance(message, dict):
        message["content"] = content
        return message
    if hasattr(message, "content"):
        message.content = content
        return message
    raise BudgetRefusal("消息内容不能改写，拒绝发送。")


def _shrink_text(text: str, head_lines: int, tail_lines: int) -> str:
    lines = text.splitlines(keepends=True)
    if len(lines) > head_lines + tail_lines + 1:
        body = "".join(lines[:head_lines]) + TRUNCATION_BANNER + "".join(lines[-tail_lines:])
    else:
        keep = max(200, len(text) // 2)
        body = text[:keep] + TRUNCATION_BANNER + text[-keep:]
    if TRUNCATION_BANNER not in body:
        body = text[:200] + TRUNCATION_BANNER
    return body


def pin_deepseek_request(payload: dict, output_cap: int) -> dict:
    """Force the documented output cap onto the body that is about to be sent.

    LangChain renames ``max_tokens`` to ``max_completion_tokens``. DeepSeek's
    published chat schema names the limit ``max_tokens`` and enables thinking
    when that field and ``thinking`` are both omitted. The ledger's 1024 is
    not a cap unless those documented fields are actually present.
    """
    if not isinstance(payload, dict):
        raise BudgetRefusal("请求载荷不是对象，拒绝发送。")
    if output_cap < 1 or output_cap > OUTPUT_TOKEN_CEILING:
        raise BudgetRefusal(f"输出上界 {output_cap} 不在可发送范围内，拒绝发送。")
    pinned = dict(payload)
    extra = dict(pinned.get("extra_body") or {})
    thinking = extra.get("thinking")
    if isinstance(thinking, dict) and thinking.get("type") == "enabled":
        raise BudgetRefusal("思考模式已打开，拒绝发送。")
    if pinned.get("reasoning_effort") not in (None, "none"):
        raise BudgetRefusal("reasoning_effort 不是 none，拒绝发送。")
    extra["thinking"] = {"type": "disabled"}
    pinned["extra_body"] = extra
    pinned["reasoning_effort"] = "none"
    pinned["max_tokens"] = output_cap
    if "max_completion_tokens" in pinned:
        pinned["max_completion_tokens"] = output_cap
    if pinned.get("max_tokens") != output_cap or pinned["extra_body"]["thinking"]["type"] != "disabled":
        raise BudgetRefusal("无法把 max_tokens 和关闭思考写入请求，拒绝发送。")
    return pinned


def prepare_for_send(messages, tools, input_cap: int):
    """Fit a request under ``input_cap`` or refuse before anything is sent.

    Tool and dialogue text keeps its head and its latest lines. The returned
    bound is the conservative token ceiling of the payload that would be sent.
    """
    if input_cap < 1:
        raise BudgetRefusal("输入上界必须为正。")
    if input_cap >= CONTEXT_TOKEN_CEILING:
        raise BudgetRefusal("拒绝按 100 万上下文预留。请设置更小且可测量的输入上界。")
    was_string = isinstance(messages, str)
    cloned = [_clone_message(message) for message in ([messages] if was_string else list(messages or []))]
    truncated = False
    head, tail = 40, 12
    for _ in range(24):
        payload = message_payload(cloned)
        schema = tool_schema(tools)
        bound = request_token_upper_bound(payload, schema)
        if bound <= input_cap:
            sent = cloned[0] if was_string else cloned
            return sent, bound, truncated
        longest = None
        longest_size = 0
        for message in cloned:
            content = _content_of(message)
            if not content:
                continue
            size = text_token_upper_bound(content)
            if size > longest_size:
                longest = message
                longest_size = size
        if longest is None or longest_size < 400:
            break
        _set_content(longest, _shrink_text(_content_of(longest), head, tail))
        truncated = True
        head = max(8, head // 2)
        tail = max(4, tail // 2)
    raise BudgetRefusal(
        f"保留关键事实后，请求输入上界仍超过 {input_cap} token，拒绝发送。"
    )


def research_call_counts(analysts: list[str], *, tool_rounds: int, debate_rounds: int, risk_rounds: int) -> dict:
    """Worst-case model calls for one research run, including structured fallback."""
    if "social" in analysts:
        raise BudgetRefusal("情绪分析员会调用账本之外的 Jev 接口，预算运行拒绝该分析员。")
    quick = 0
    for name in analysts:
        if name not in {"market", "news", "fundamentals"}:
            raise BudgetRefusal(f"分析员 {name} 不在可预留的调用清单里。")
        quick += tool_rounds + 1
    quick += 2 * debate_rounds
    quick += 2  # trader structured call plus free-text fallback
    quick += 3 * risk_rounds
    deep = 2 + 2  # research manager and portfolio manager, each with fallback
    return {
        "quick": quick,
        "deep": deep,
        "reflections": 0,
        "note": "记忆日志必须为空。已有待结算决策会再调用模型，本清单不包含，因此拒绝带历史记忆的运行。",
    }


def _slots(model: str, count: int, input_cap: int, output_cap: int, role: str) -> list[dict]:
    return [
        {
            "model": model,
            "input_cap": input_cap,
            "output_cap": output_cap,
            "role": role,
            "status": "reserved",
        }
        for _ in range(count)
    ]


def build_plan(
    *,
    quick_model: str,
    deep_model: str,
    analysts: list[str],
    tool_rounds: int,
    debate_rounds: int,
    risk_rounds: int,
    input_cap: int,
    output_cap: int,
    retries: int,
    card: PriceCard | None = None,
    now: datetime | None = None,
) -> dict:
    card = card or official_price_card()
    require_fresh(card, now)
    if retries != 0:
        raise BudgetRefusal("预算运行禁止隐式重试。max_retries 必须为 0。")
    if input_cap < 1 or output_cap < 1:
        raise BudgetRefusal("输入和输出上限都必须为正。")
    if output_cap > OUTPUT_TOKEN_CEILING:
        raise BudgetRefusal(f"输出上限超过官方最大输出 {OUTPUT_TOKEN_CEILING}。")
    counts = research_call_counts(
        analysts, tool_rounds=tool_rounds, debate_rounds=debate_rounds, risk_rounds=risk_rounds,
    )
    slots = _slots(quick_model, counts["quick"], input_cap, output_cap, "quick")
    slots += _slots(deep_model, counts["deep"], input_cap, output_cap, "deep")
    total = Decimal("0")
    for slot in slots:
        total += call_cost(card, slot["model"], input_cap, output_cap)
    total = _money(total)
    unbounded_one = call_cost(card, quick_model, CONTEXT_TOKEN_CEILING, output_cap)
    attempt_allowed = input_cap < CONTEXT_TOKEN_CEILING
    blocker = (
        "最坏费用按每次请求的输入上界和输出上限预留，不按 100 万上下文预留。"
        "某次请求在保留关键事实后仍超过输入上界时，发送前拒绝，报告标明未完成。"
        "不保证平台账单与本次一定跑完。"
    )
    if not attempt_allowed:
        blocker = (
            f"输入上界不能使用官方 100 万上下文。那样单次高峰未命中预留为 {unbounded_one} 元。"
        )
    return {
        "kind": "research",
        "counts": counts,
        "analysts": list(analysts),
        "tool_rounds": tool_rounds,
        "debate_rounds": debate_rounds,
        "risk_rounds": risk_rounds,
        "quick_model": quick_model,
        "deep_model": deep_model,
        "input_cap": input_cap,
        "output_cap": output_cap,
        "retries": 0,
        "slots": len(slots),
        "reserved_cny": str(total),
        "one_call_cny": str(call_cost(card, quick_model, input_cap, output_cap)),
        "one_unbounded_context_call_cny": str(unbounded_one),
        "attempt_allowed": attempt_allowed,
        "completion_guaranteed": False,
        "completion_blocker": blocker,
    }


def build_probe_plan(
    *,
    model: str = "deepseek-flash",
    input_cap: int = 512,
    output_cap: int = 32,
    card: PriceCard | None = None,
    now: datetime | None = None,
) -> dict:
    card = card or official_price_card()
    require_fresh(card, now)
    amount = call_cost(card, model, input_cap, output_cap)
    return {
        "kind": "probe",
        "model": model,
        "input_cap": input_cap,
        "output_cap": output_cap,
        "slots": 1,
        "retries": 0,
        "reserved_cny": str(amount),
        "attempt_allowed": True,
        "completion_guaranteed": True,
        "completion_note": "这只是一次有上界的对话请求，不是多代理研究结论。",
    }


def _lock_for(path: Path) -> threading.Lock:
    key = str(path.resolve())
    with _LOCKS_GUARD:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _LOCKS[key] = lock
        return lock


class Ledger:
    """Persistent reservation. Reloading the same plan does not charge it again."""

    def __init__(self, path: Path, budget_cny: str, prior_hold_cny: str, card: PriceCard | None = None):
        self.path = Path(path)
        self.card = card or official_price_card()
        self.budget = _parse_money(budget_cny)
        self.prior_hold = _parse_money(prior_hold_cny)
        if self.prior_hold > self.budget:
            raise BudgetRefusal("先前占用已经超过总预算。")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = _lock_for(self.path)

    def _read(self) -> dict:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _write(self, payload: dict) -> None:
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, self.path)

    def reserve(self, plan: dict, run_id: str, *, now: datetime | None = None) -> dict:
        require_fresh(self.card, now)
        digest = hashlib.sha256(
            json.dumps(plan, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()
        if plan.get("kind") == "probe":
            reserved = call_cost(self.card, plan["model"], int(plan["input_cap"]), int(plan["output_cap"]))
            slot_rows = _slots(plan["model"], 1, int(plan["input_cap"]), int(plan["output_cap"]), "probe")
        elif plan.get("kind") == "research":
            if not plan.get("attempt_allowed"):
                raise BudgetRefusal(plan.get("completion_blocker") or "研究方案不能发送。")
            rebuilt = build_plan(
                quick_model=plan["quick_model"],
                deep_model=plan["deep_model"],
                analysts=list(plan["analysts"]),
                tool_rounds=int(plan["tool_rounds"]),
                debate_rounds=int(plan["debate_rounds"]),
                risk_rounds=int(plan["risk_rounds"]),
                input_cap=int(plan["input_cap"]),
                output_cap=int(plan["output_cap"]),
                retries=0,
                card=self.card,
                now=now,
            )
            reserved = _parse_money(rebuilt["reserved_cny"])
            slot_rows = _slots(
                rebuilt["quick_model"], rebuilt["counts"]["quick"],
                int(rebuilt["input_cap"]), int(rebuilt["output_cap"]), "quick",
            )
            slot_rows += _slots(
                rebuilt["deep_model"], rebuilt["counts"]["deep"],
                int(rebuilt["input_cap"]), int(rebuilt["output_cap"]), "deep",
            )
        else:
            raise BudgetRefusal("未知预算方案，拒绝预留。")
        if _parse_money(plan["reserved_cny"]) < reserved:
            raise BudgetRefusal("方案声明的金额低于按价格卡重算的上界，拒绝预留。")
        with self._thread:
            state = self._read()
            if state.get("run_id") == run_id:
                if state.get("plan_digest") != digest:
                    raise BudgetRefusal("恢复时方案与已入账的预留不一致，拒绝再次扣款或更换方案。")
                return state
            if state.get("run_id"):
                raise BudgetRefusal("账本已有其他运行，拒绝叠加预留。")
            if self.prior_hold + reserved > self.budget:
                raise BudgetRefusal(
                    f"预留 {reserved} 元加上先前占用 {self.prior_hold} 元，超过总预算 {self.budget} 元。"
                )
            state = {
                "run_id": run_id,
                "plan_digest": digest,
                "plan_kind": plan["kind"],
                "card_digest": self.card.digest(),
                "budget_cny": str(self.budget),
                "prior_hold_cny": str(self.prior_hold),
                "reserved_cny": str(reserved),
                "frozen": False,
                "slots": [],
            }
            state["slots"] = slot_rows
            if len(state["slots"]) != int(plan["slots"]):
                raise BudgetRefusal("预留槽位数量与方案不一致。")
            self._write(state)
            return state

    def begin_send(self, *, model: str, messages, tools=None, max_tokens: int, thinking: str, effort: str, stream: bool, card: PriceCard | None = None, truncated: bool = False) -> int:
        card = card or self.card
        if stream:
            raise BudgetRefusal("流式响应不能在发送前同时证明最终 usage，拒绝发送。")
        if thinking != "disabled":
            raise BudgetRefusal("思考模式未关闭，拒绝发送。")
        if effort != "none":
            raise BudgetRefusal("reasoning_effort 必须为 none，拒绝发送。")
        if card.digest() != self._require_state()["card_digest"]:
            raise BudgetRefusal("价格卡与预留时不一致，拒绝发送。")
        require_fresh(card)
        bound = request_token_upper_bound(message_payload(messages), tools)
        with self._thread:
            state = self._read()
            if state.get("frozen"):
                raise BudgetRefusal("账本已冻结，拒绝继续发送。")
            if state.get("card_digest") != card.digest():
                raise BudgetRefusal("价格卡与预留时不一致，拒绝发送。")
            chosen = None
            for index, slot in enumerate(state.get("slots") or []):
                if slot["status"] != "reserved":
                    continue
                if slot["model"] != model:
                    continue
                if bound > slot["input_cap"] or max_tokens > slot["output_cap"] or max_tokens < 1:
                    raise BudgetRefusal(
                        f"请求上界 input={bound} output={max_tokens} 超过槽位 "
                        f"{slot['input_cap']}/{slot['output_cap']}，拒绝发送。"
                    )
                slot["status"] = "sending"
                slot["input_bound"] = bound
                slot["max_tokens"] = max_tokens
                slot["truncated"] = bool(truncated)
                chosen = index
                break
            if chosen is None:
                raise BudgetRefusal("没有剩余的预留槽位，拒绝计划外或重复发送。")
            self._write(state)
            return chosen

    def abort(self, slot_index: int) -> None:
        """The request was not proven successful. Keep the reservation."""
        with self._thread:
            state = self._read()
            slot = state["slots"][slot_index]
            if slot["status"] == "sending":
                slot["status"] = "usage_missing"
                slot["charged_cny"] = str(call_cost(
                    self.card, slot["model"], slot["input_cap"], slot["output_cap"],
                ))
            self._write(state)

    def settle(self, slot_index: int, usage: dict | None) -> None:
        with self._thread:
            state = self._read()
            slot = state["slots"][slot_index]
            reserved = call_cost(self.card, slot["model"], slot["input_cap"], slot["output_cap"])
            if not usage or "prompt_tokens" not in usage or "completion_tokens" not in usage:
                slot["status"] = "usage_missing"
                slot["charged_cny"] = str(reserved)
                self._write(state)
                return
            prompt = int(usage["prompt_tokens"])
            completion = int(usage["completion_tokens"])
            if prompt < 0 or completion < 0:
                slot["status"] = "usage_missing"
                slot["charged_cny"] = str(reserved)
                self._write(state)
                return
            details = usage.get("completion_tokens_details") or {}
            reasoning = details.get("reasoning_tokens")
            if isinstance(reasoning, bool) or (reasoning is not None and not isinstance(reasoning, int)):
                reasoning = None
            actual = call_cost(self.card, slot["model"], prompt, completion)
            # completion_tokens is the billed size we were given. reasoning_tokens
            # is recorded beside it and is not added again.
            slot["usage"] = {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "reasoning_tokens": reasoning,
            }
            if actual > reserved or prompt > slot["input_cap"] or completion > slot["output_cap"]:
                slot["status"] = "overrun"
                slot["charged_cny"] = str(actual)
                state["frozen"] = True
            else:
                slot["status"] = "settled"
                slot["charged_cny"] = str(actual)
                slot["released_for_more_calls"] = False
            self._write(state)

    def _require_state(self) -> dict:
        state = self._read()
        if not state.get("run_id"):
            raise BudgetRefusal("还没有运行级预留，拒绝发送。")
        return state

    def snapshot(self) -> dict:
        with self._thread:
            return self._read()


def assess_affordability(budget_cny: str, prior_hold_cny: str, plan: dict) -> dict:
    budget = _parse_money(budget_cny)
    prior = _parse_money(prior_hold_cny)
    reserved = _parse_money(plan["reserved_cny"])
    remaining = _money(budget - prior)
    fits = prior + reserved <= budget
    return {
        "budget_cny": str(budget),
        "prior_hold_cny": str(prior),
        "remaining_cny": str(remaining),
        "reserved_cny": str(reserved),
        "fits_reservation": fits,
        "send_allowed": bool(fits and plan.get("attempt_allowed")),
    }

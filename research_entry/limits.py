"""Caps applied only when an operator explicitly runs a model.

These are bounds, not a choice of paid plan. Upstream defaults stay in
place for anyone who uses the stock CLI. The research entry sets tighter
tool-round, token and timeout caps unless the operator overrides them.
"""

from __future__ import annotations

DEFAULT_LIMITS = {
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
    "max_tool_rounds": 8,
    "max_tokens": 4096,
    "request_timeout": 120,
    "llm_max_retries": 2,
}

ALLOWED_ANALYSTS = ("market", "social", "news", "fundamentals")


def parse_analysts(raw: str | None) -> list[str]:
    if raw is None or not str(raw).strip():
        return ["market"]
    chosen = []
    for item in str(raw).split(","):
        name = item.strip().lower()
        if not name:
            continue
        if name not in ALLOWED_ANALYSTS:
            raise ValueError(
                f"未知分析员 {name}。可选：{', '.join(ALLOWED_ANALYSTS)}。"
            )
        if name not in chosen:
            chosen.append(name)
    if not chosen:
        raise ValueError("至少指定一个分析员。")
    return chosen


def resolve_limits(overrides: dict | None = None) -> dict:
    limits = dict(DEFAULT_LIMITS)
    for key, value in (overrides or {}).items():
        if value is None:
            continue
        limits[key] = value
    if limits["max_tool_rounds"] < 1:
        raise ValueError("max_tool_rounds 必须 >= 1")
    if 2 * int(limits["max_tool_rounds"]) + 2 >= 100:
        raise ValueError("max_tool_rounds 过大，超过上游递归上限可容纳的范围。")
    if int(limits["max_tokens"]) <= 0:
        raise ValueError("max_tokens 必须 > 0")
    if float(limits["request_timeout"]) <= 0:
        raise ValueError("request_timeout 必须 > 0")
    if int(limits["llm_max_retries"]) < 0:
        raise ValueError("llm_max_retries 必须 >= 0")
    return limits

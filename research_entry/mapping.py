"""Syntactic symbol checks. No network, no quotes, no existence claims."""

from __future__ import annotations

from tradingagents.dataflows.symbols import normalize_symbol, safe_ticker_component

# Declared market -> suffix of the canonical Yahoo symbol.
# SH is the operator's name for Shanghai; Yahoo's suffix is .SS.
MARKET_SUFFIX = {
    "HK": ".HK",
    "SS": ".SS",
    "SH": ".SS",
    "SZ": ".SZ",
    "US": "",
}


def suffix_of(canonical: str) -> str:
    if "." not in canonical:
        return ""
    return "." + canonical.rsplit(".", 1)[1]


def map_symbol(raw: str, market: str | None = None) -> dict:
    """Normalize one symbol and compare its suffix with the declared market.

    ``exists`` and ``quote`` are always null. A successful mapping is not
    evidence that the instrument is listed or that a price is available.
    """
    if not isinstance(raw, str) or not raw.strip():
        return {
            "input": raw,
            "ok": False,
            "error": "代码为空",
            "canonical": None,
            "exists": None,
            "quote": None,
            "notes": [],
        }

    canonical = normalize_symbol(raw)
    try:
        safe_ticker_component(canonical)
        path_safe = True
        path_error = None
    except ValueError as exc:
        path_safe = False
        path_error = str(exc)

    market_norm = market.strip().upper() if isinstance(market, str) and market.strip() else None
    notes: list[str] = []
    if market_norm == "SH":
        notes.append("市场参数 SH 表示上交所；Yahoo 规范后缀是 .SS，输入 .SH 会改写为 .SS。")
    if canonical[:3] == "688" and suffix_of(canonical) == ".SS":
        notes.append(
            "688 前缀在公开资料中常被称作上交所科创板。"
            "这里只核对后缀归一，不证明该代码存在，也不提供行情。"
        )
    if market_norm is None:
        match = None
        expected = None
    elif market_norm not in MARKET_SUFFIX:
        match = False
        expected = None
        notes.append(f"未识别的市场参数 {market_norm}。支持 HK、SS、SH、SZ、US。")
    else:
        expected = MARKET_SUFFIX[market_norm]
        match = suffix_of(canonical) == expected
        if not match:
            notes.append(
                f"规范化代码 {canonical} 的后缀与声明市场 {market_norm} 不一致。"
                "未改写为其他代码，也未查询行情。"
            )

    return {
        "input": raw.strip(),
        "canonical": canonical,
        "changed": canonical != raw.strip().upper(),
        "market": market_norm,
        "expected_suffix": expected,
        "actual_suffix": suffix_of(canonical) if path_safe or canonical else None,
        "market_matches": match,
        "path_safe": path_safe,
        "path_error": path_error,
        "exists": None,
        "quote": None,
        "notes": notes,
        "ok": path_safe and match is not False,
    }

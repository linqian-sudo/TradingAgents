"""Read-only public daily history through the project's Yahoo client.

Uses yfinance and the existing rate-limit retry. It does not switch to
another vendor when Yahoo fails, and it does not fill in a price the
response did not contain.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import yfinance as yf

from tradingagents.dataflows.errors import VendorUnavailableError
from tradingagents.dataflows.symbols import normalize_symbol
from tradingagents.dataflows.vendors.yahoo.common import YAHOO_HOST, yf_retry

_PRICE_COLUMNS = ("Open", "High", "Low", "Close", "Volume")


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _number(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return number


def _bar_date(index_value) -> str | None:
    try:
        stamp = pd.Timestamp(index_value)
    except (ValueError, TypeError):
        return None
    if stamp.tzinfo is not None:
        stamp = stamp.tz_localize(None)
    if pd.isna(stamp):
        return None
    return stamp.strftime("%Y-%m-%d")


def _empty_result(symbol: str, canonical: str, as_of: str, start: str, end: str) -> dict:
    return {
        "source": "yfinance",
        "source_host": YAHOO_HOST,
        "endpoint_note": (
            "只读调用项目内 yfinance 的日线 history。"
            "遇到 HTTP 429 时只使用既有退避重试，不更换数据源，不绕过源站限制。"
        ),
        "symbol_input": symbol,
        "canonical": canonical,
        "interval": "1d",
        "auto_adjust": True,
        "window_start": start,
        "window_end_exclusive": end,
        "as_of": as_of,
        "fetched_at": _utc_now(),
        "rows": 0,
        "first_date": None,
        "last_date": None,
        "last_bar": None,
        "columns": [],
        "status": "not_run",
        "error": None,
        "error_type": None,
        "cause_type": None,
        "cause": None,
        "research_claim": False,
        "coverage_note": "港股与 A 股的公开文档支持不保证实时行情，也不保证财报字段完整。无返回则记为无数据，不编造。",
    }


def _slice_through(frame: pd.DataFrame, as_of: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return frame
    dates = []
    keep = []
    for label in frame.index:
        day = _bar_date(label)
        dates.append(day)
        keep.append(day is not None and day <= as_of)
    if not any(keep):
        return frame.iloc[0:0]
    return frame.loc[pd.Series(keep, index=frame.index)]


def check_history(symbol: str, as_of: str, *, lookback_days: int = 31) -> dict:
    """Fetch daily bars from ``as_of - lookback_days`` through ``as_of`` inclusive."""
    canonical = normalize_symbol(symbol)
    as_of_dt = datetime.strptime(as_of, "%Y-%m-%d")
    start = (as_of_dt - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    end = (as_of_dt + timedelta(days=1)).strftime("%Y-%m-%d")
    result = _empty_result(symbol, canonical, as_of, start, end)

    try:
        frame = yf_retry(lambda: yf.Ticker(canonical).history(
            start=start,
            end=end,
            interval="1d",
            auto_adjust=True,
            actions=False,
        ))
    except VendorUnavailableError as exc:
        cause = exc.__cause__
        result["status"] = "unavailable"
        result["error"] = str(exc)
        result["error_type"] = type(exc).__name__
        if cause is not None:
            result["cause_type"] = type(cause).__name__
            result["cause"] = str(cause)
        return result

    if frame is None or getattr(frame, "empty", True):
        result["status"] = "no_data"
        result["error"] = "源站未返回该窗口内的日线。未编造价格，也未据此判断代码是否上市。"
        return result

    frame = _slice_through(frame, as_of)
    if frame is None or frame.empty:
        result["status"] = "no_data"
        result["error"] = "返回中没有不晚于分析日的日线。未编造价格。"
        return result

    columns = [str(column) for column in frame.columns]
    first = _bar_date(frame.index[0])
    last = _bar_date(frame.index[-1])
    row = frame.iloc[-1]
    last_bar = {"date": last}
    for column in _PRICE_COLUMNS:
        if column in frame.columns:
            last_bar[column.lower()] = _number(row[column])
        else:
            last_bar[column.lower()] = None

    result.update({
        "status": "ok",
        "rows": int(len(frame)),
        "first_date": first,
        "last_date": last,
        "last_bar": last_bar,
        "columns": columns,
        "error": None,
    })
    return result

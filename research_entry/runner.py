"""Command implementations. Network and model calls stay behind explicit commands."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from research_entry.data_check import check_history
from research_entry.limits import parse_analysts, resolve_limits
from research_entry.mapping import map_symbol
from research_entry.model_gate import assess_model, credential_catalog
from research_entry.provenance import REFERENCE_SYMBOLS
from research_entry.status_report import build_status, not_requested_data, write_status


def _parse_date(value: str) -> str:
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"分析日必须是 YYYY-MM-DD，收到 {value!r}") from exc
    if parsed.strftime("%Y-%m-%d") != value:
        raise ValueError(f"分析日必须是 YYYY-MM-DD，收到 {value!r}")
    if value > datetime.now().strftime("%Y-%m-%d"):
        raise ValueError(f"分析日不能晚于本机今天：{value}")
    return value


def _research_blocked(reason: str) -> dict:
    return {
        "status": "not_run",
        "end_to_end": False,
        "decision": None,
        "reason": reason,
    }


def execute_map(output: Path, symbol: str | None, market: str | None, reference: bool) -> int:
    rows = []
    if reference or not symbol:
        for item, item_market in REFERENCE_SYMBOLS:
            rows.append(map_symbol(item, item_market))
    if symbol:
        rows.append(map_symbol(symbol, market))
    # De-duplicate when --reference already contains the same input.
    seen = set()
    unique = []
    for row in rows:
        key = (row.get("input"), row.get("market"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    model = assess_model(provider=None, quick_model=None, deep_model=None, backend_url=None, environ={})
    model["reason"] = "符号核对不需要模型，也未读取凭证。"
    research = _research_blocked("只做了符号规范化核对，没有研判。")
    status = build_status(
        command="map",
        symbol=unique[0] if len(unique) == 1 else None,
        model=model,
        data=not_requested_data(),
        research=research,
        limits={},
        extra={"mappings": unique, "existence_checked": False},
    )
    write_status(output, status)
    (output / "mapping.json").write_text(
        json.dumps({"mappings": unique, "existence_checked": False}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    bad = [row for row in unique if not row.get("ok")]
    return 0 if not bad else 1


def execute_preflight(output: Path, environ: dict) -> int:
    model = assess_model(provider=None, quick_model=None, deep_model=None, backend_url=None, environ=environ)
    model["reason"] = "预检不选择模型、不读取凭证值、不发起调用。安装后的模型状态以 run 的显式参数为准。"
    research = _research_blocked("预检完成。模型未配置，不能把本次结果称为端到端研究成功。")
    status = build_status(
        command="preflight",
        symbol=None,
        model=model,
        data=not_requested_data(),
        research=research,
        limits=resolve_limits({}),
        extra={"credential_catalog": credential_catalog()},
    )
    write_status(output, status)
    return 0


def execute_data_check(output: Path, symbol: str, market: str, as_of: str, lookback_days: int) -> int:
    as_of = _parse_date(as_of)
    mapped = map_symbol(symbol, market)
    model = assess_model(provider=None, quick_model=None, deep_model=None, backend_url=None, environ={})
    model["reason"] = "行情连通性测试不调用模型。"
    if not mapped.get("ok"):
        research = _research_blocked("符号与声明市场不一致，未请求行情，避免把错误代码的空结果说成该股票无行情。")
        status = build_status(
            command="data-check",
            symbol=mapped,
            model=model,
            data=not_requested_data(as_of),
            research=research,
            limits={},
        )
        write_status(output, status)
        return 1
    data = check_history(mapped["canonical"], as_of, lookback_days=lookback_days)
    if data["status"] == "ok":
        research = _research_blocked(
            "只确认公开日线可读。这不是研究报告，不是实时行情，也不是完整财报。"
        )
        code = 0
    elif data["status"] == "no_data":
        research = _research_blocked("源站没有返回可用日线。未编造行情，因此也不能称为研究成功。")
        code = 4
    else:
        research = _research_blocked("源站未能完成这次只读请求。原样记录错误，未改用其他数据接口。")
        code = 3
    status = build_status(
        command="data-check",
        symbol=mapped,
        model=model,
        data=data,
        research=research,
        limits={},
    )
    write_status(output, status)
    (output / "data.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return code


def _research_config(output: Path, provider: str, quick: str, deep: str, backend_url: str | None, limits: dict) -> dict:
    from tradingagents.default_config import build_default_config

    config = build_default_config()
    root = output / "run"
    config.update({
        "llm_provider": provider,
        "quick_think_llm": quick,
        "deep_think_llm": deep,
        "backend_url": (backend_url or "").strip() or None,
        "quick_think_provider": None,
        "deep_think_provider": None,
        "output_language": "Chinese",
        "checkpoint_enabled": False,
        "max_debate_rounds": int(limits["max_debate_rounds"]),
        "max_risk_discuss_rounds": int(limits["max_risk_discuss_rounds"]),
        "max_tool_rounds": int(limits["max_tool_rounds"]),
        "max_tokens": int(limits["max_tokens"]),
        "request_timeout": limits["request_timeout"],
        "llm_max_retries": int(limits["llm_max_retries"]),
        "results_dir": str(root / "results"),
        "data_cache_dir": str(root / "cache"),
        "memory_log_path": str(root / "memory" / "trading_memory.md"),
        "data_vendors": {
            "core_stock_apis": "yfinance",
            "technical_indicators": "yfinance",
            "fundamental_data": "yfinance",
            "news_data": "yfinance",
            "macro_data": "fred",
            "prediction_markets": "polymarket",
        },
    })
    return config


def execute_run(
    output: Path,
    symbol: str,
    market: str,
    as_of: str,
    *,
    provider: str | None,
    quick_model: str | None,
    deep_model: str | None,
    backend_url: str | None,
    analysts: str | None,
    limit_overrides: dict,
    environ: dict,
) -> int:
    as_of = _parse_date(as_of)
    mapped = map_symbol(symbol, market)
    limits = resolve_limits(limit_overrides)
    chosen = parse_analysts(analysts)
    model = assess_model(
        provider=provider,
        quick_model=quick_model,
        deep_model=deep_model,
        backend_url=backend_url,
        environ=environ,
    )
    if not mapped.get("ok"):
        research = _research_blocked("符号未通过市场核对，未调用模型。")
        status = build_status(
            command="run",
            symbol=mapped,
            model=model,
            data=not_requested_data(as_of),
            research=research,
            limits=limits,
            extra={"analysts": chosen},
        )
        write_status(output, status)
        return 1
    if model["status"] != "configured":
        research = {
            "status": "model_unconfigured",
            "end_to_end": False,
            "decision": None,
            "reason": model["reason"],
        }
        status = build_status(
            command="run",
            symbol=mapped,
            model=model,
            data=not_requested_data(as_of),
            research=research,
            limits=limits,
            extra={"analysts": chosen, "credential_catalog": credential_catalog()},
        )
        write_status(output, status)
        return 2

    # Imported only after the gate, so a missing key never constructs a client.
    import os

    os.environ["TRADINGAGENTS_NO_UPDATE_CHECK"] = "1"
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    config = _research_config(
        output, model["provider"], model["quick_model"], model["deep_model"], backend_url, limits,
    )
    try:
        graph = TradingAgentsGraph(selected_analysts=chosen, debug=False, config=config)
    except Exception as exc:
        research = {
            "status": "failed",
            "end_to_end": False,
            "decision": None,
            "reason": f"模型客户端未能建立：{type(exc).__name__}: {exc}",
        }
        status = build_status(
            command="run",
            symbol=mapped,
            model=model,
            data=not_requested_data(as_of),
            research=research,
            limits=limits,
            extra={"analysts": chosen},
        )
        write_status(output, status)
        return 1
    try:
        final_state, decision = graph.propagate(
            mapped["canonical"], as_of, asset_type="stock", portfolio=None,
        )
    except Exception as exc:
        model["called"] = True
        research = {
            "status": "failed",
            "end_to_end": False,
            "decision": None,
            "reason": f"模型或数据步骤失败：{type(exc).__name__}: {exc}",
        }
        status = build_status(
            command="run",
            symbol=mapped,
            model=model,
            data=not_requested_data(as_of),
            research=research,
            limits=limits,
            extra={"analysts": chosen},
        )
        write_status(output, status)
        return 1

    model["called"] = True
    report_dir = graph.save_reports(final_state, mapped["canonical"], save_path=output / "reports")
    decision_text = decision if isinstance(decision, str) else str(decision)
    research = {
        "status": "completed",
        "end_to_end": True,
        "decision": decision_text,
        "reason": "模型已调用并返回。决策文本来自本次运行，不额外补充未返回的数字。",
        "report_dir": str(report_dir),
    }
    status = build_status(
        command="run",
        symbol=mapped,
        model=model,
        data=not_requested_data(as_of),
        research=research,
        limits=limits,
        extra={"analysts": chosen},
    )
    write_status(output, status)
    return 0

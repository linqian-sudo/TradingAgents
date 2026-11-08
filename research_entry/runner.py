"""Command implementations. Network and model calls stay behind explicit commands."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from research_entry.budget import (
    ASSUMPTIONS,
    BudgetRefusal,
    assess_affordability,
    build_plan,
    build_probe_plan,
    conservative_prior_reservation_hold,
    official_price_card,
    require_endpoint,
)
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


def _budget_status(output: Path, *, command: str, symbol, model: dict, limits: dict, extra: dict, research: dict) -> int:
    status = build_status(
        command=command,
        symbol=symbol,
        model=model,
        data=not_requested_data(symbol.get("as_of") if symbol else None),
        research=research,
        limits=limits,
        extra=extra,
    )
    write_status(output, status)
    code = 0 if research.get("status") == "budget_assessed" else 2
    return code


def execute_budget(
    output: Path,
    *,
    analysts: list[str],
    limits: dict,
    quick_model: str,
    deep_model: str,
    budget_cny: str,
    prior_hold_cny: str | None,
    input_token_cap: int,
    backend_url: str | None,
) -> int:
    """Offline reservation math. Does not construct a model client."""
    card = official_price_card()
    prior = prior_hold_cny if prior_hold_cny not in (None, "") else str(conservative_prior_reservation_hold(card))
    try:
        require_endpoint(backend_url)
        plan = build_plan(
            quick_model=quick_model,
            deep_model=deep_model,
            analysts=analysts,
            tool_rounds=int(limits["max_tool_rounds"]),
            debate_rounds=int(limits["max_debate_rounds"]),
            risk_rounds=int(limits["max_risk_discuss_rounds"]),
            input_cap=input_token_cap,
            output_cap=int(limits["max_tokens"]),
            retries=int(limits["llm_max_retries"]),
            card=card,
        )
        probe = build_probe_plan(card=card)
        afford = assess_affordability(budget_cny, prior, plan)
        probe_afford = assess_affordability(budget_cny, prior, probe)
        flash_only = build_plan(
            quick_model="deepseek-flash",
            deep_model="deepseek-flash",
            analysts=analysts,
            tool_rounds=int(limits["max_tool_rounds"]),
            debate_rounds=int(limits["max_debate_rounds"]),
            risk_rounds=int(limits["max_risk_discuss_rounds"]),
            input_cap=input_token_cap,
            output_cap=int(limits["max_tokens"]),
            retries=0,
            card=card,
        )
        flash_only_afford = assess_affordability(budget_cny, prior, flash_only)
    except BudgetRefusal as exc:
        plan = {"completion_guaranteed": False, "completion_blocker": str(exc)}
        probe = None
        afford = {"send_allowed": False, "fits_reservation": False}
        probe_afford = {"send_allowed": False}
        flash_only = None
        flash_only_afford = None
        refusal = str(exc)
    else:
        refusal = None
    research = {
        "status": "budget_assessed",
        "end_to_end": False,
        "decision": None,
        "reason": (
            "最坏预留没有超过剩余预算，可以按输入上界尝试多角色研究。"
            "某次请求无法压进上界时会在发送前拒绝，报告标明未完成。"
            "不保证平台账单。"
            if (afford or {}).get("send_allowed")
            else "最坏预留超过剩余预算，或输入上界不可用，拒绝发送。"
        ),
    }
    model = {
        "status": "not_called",
        "provider": "deepseek",
        "quick_model": quick_model,
        "deep_model": deep_model,
        "called": False,
        "reason": "预算核算没有构造模型客户端，也没有读取 API Key。",
    }
    extra = {
        "assumptions": list(ASSUMPTIONS),
        "price_source": card.source,
        "price_retrieved_at": card.retrieved_at,
        "price_expires_at": card.expires_at,
        "price_digest": card.digest(),
        "research_plan": plan,
        "research_affordability": afford,
        "minimum_probe": probe,
        "minimum_probe_affordability": probe_afford,
        "flash_only_plan": flash_only,
        "flash_only_affordability": flash_only_afford,
        "refusal": refusal,
        "full_research_reliable_within_budget": False,
    }
    return _budget_status(
        output, command="budget", symbol=None, model=model, limits=limits, extra=extra, research=research,
    )


def _run_reserved_research(output, mapped, model, limits, chosen, as_of, card, budget_cny, prior, plan) -> int:
    """Run the multi-role graph only after the worst-case plan is reserved."""
    from research_entry.budget import Ledger
    from research_entry.budget_gate import install_chat_gate
    from tradingagents.graph import trading_graph
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    ledger = Ledger(output / "budget-ledger.json", budget_cny, prior, card)
    ledger.reserve(plan, "tencent-research")
    original = trading_graph.create_tier_client

    def gated(config, tier, **kwargs):
        client = original(config, tier, **kwargs)
        llm = install_chat_gate(
            client.get_llm(),
            ledger,
            input_cap=int(plan["input_cap"]),
            output_cap=int(plan["output_cap"]),
            expected_model=plan["quick_model"] if tier == "quick" else plan["deep_model"],
        )

        def get_llm():
            return llm

        client.get_llm = get_llm
        return client

    trading_graph.create_tier_client = gated
    model_out = dict(model)
    try:
        graph = TradingAgentsGraph(
            selected_analysts=chosen,
            debug=False,
            config=_research_config(
                output, "deepseek", plan["quick_model"], plan["deep_model"], None, limits,
            ),
        )
        _state, decision = graph.propagate(mapped["canonical"], as_of, asset_type="stock", portfolio=None)
    except BudgetRefusal as exc:
        model_out["called"] = True
        model_out["status"] = "research_incomplete"
        model_out["reason"] = str(exc)
        research = {
            "status": "research_incomplete",
            "end_to_end": False,
            "decision": None,
            "reason": str(exc),
        }
        code = 2
    except Exception as exc:
        model_out["called"] = True
        model_out["status"] = "research_incomplete"
        model_out["reason"] = f"{type(exc).__name__}: {exc}"
        research = {
            "status": "research_incomplete",
            "end_to_end": False,
            "decision": None,
            "reason": model_out["reason"],
        }
        code = 1
    else:
        model_out["called"] = True
        model_out["status"] = "research_attempted"
        model_out["reason"] = "多角色流程已在预算闸门内跑完。截断或缺失必须看账本，不能把结果当成无保留的完整研判。"
        research = {
            "status": "research_attempted",
            "end_to_end": False,
            "decision": decision if isinstance(decision, str) else str(decision),
            "reason": model_out["reason"],
        }
        code = 0
    finally:
        trading_graph.create_tier_client = original
    snapshot = ledger.snapshot()
    status = build_status(
        command="run",
        symbol=mapped,
        model=model_out,
        data=not_requested_data(as_of),
        research=research,
        limits=limits,
        extra={
            "assumptions": list(ASSUMPTIONS),
            "research_plan": plan,
            "ledger": snapshot,
            "truncated_slots": [
                slot for slot in snapshot.get("slots", []) if slot.get("truncated")
            ],
            "full_research_reliable_within_budget": False,
            "worst_case_reserved_cny": plan["reserved_cny"],
        },
    )
    write_status(output, status)
    return code


def _send_reserved_probe(ledger, probe: dict):
    """One already-reserved chat completion. Tests replace this or the client."""
    import os

    from research_entry.budget_gate import install_budget_gate
    from tradingagents.llm_clients.openai_client import DeepSeekChatOpenAI

    llm = DeepSeekChatOpenAI(
        model=probe["model"],
        api_key=os.environ.get("DEEPSEEK_API_KEY") or "",
        max_retries=0,
        base_url="https://api.deepseek.com",
    )
    install_budget_gate(llm, ledger)
    return llm.client.create(
        model=probe["model"],
        messages=[{"role": "user", "content": "ok"}],
        max_tokens=int(probe["output_cap"]),
    )


def _refuse_unbounded_research(
    output: Path,
    *,
    mapped: dict,
    model: dict,
    limits: dict,
    chosen: list[str],
    as_of: str,
    budget_cny: str | None,
    prior_hold_cny: str | None,
    input_token_cap: int | None,
    backend_url: str | None,
    budget_mode: str,
) -> int:
    """DeepSeek research is not sent until a complete ceiling is proven."""
    card = official_price_card()
    prior = prior_hold_cny if prior_hold_cny not in (None, "") else str(conservative_prior_reservation_hold(card))
    cap = int(input_token_cap or 8192)
    refusal = None
    plan = None
    afford = None
    probe = None
    probe_afford = None
    try:
        require_endpoint(backend_url)
        if (model.get("provider") or "") != "deepseek":
            raise BudgetRefusal("带预算的运行只接受价格卡中的 DeepSeek 模型，拒绝发送。")
        if budget_cny in (None, ""):
            raise BudgetRefusal("未设置单次总预算。DeepSeek 运行拒绝发送。")
        if int(limits["llm_max_retries"]) != 0:
            raise BudgetRefusal("预算运行禁止隐式重试。请把重试设为 0。")
        plan = build_plan(
            quick_model=model["quick_model"],
            deep_model=model["deep_model"],
            analysts=chosen,
            tool_rounds=int(limits["max_tool_rounds"]),
            debate_rounds=int(limits["max_debate_rounds"]),
            risk_rounds=int(limits["max_risk_discuss_rounds"]),
            input_cap=cap,
            output_cap=int(limits["max_tokens"]),
            retries=0,
            card=card,
        )
        probe = build_probe_plan(card=card)
        afford = assess_affordability(budget_cny, prior, plan)
        probe_afford = assess_affordability(budget_cny, prior, probe)
        if budget_mode == "probe":
            if not probe_afford["send_allowed"]:
                raise BudgetRefusal("单次探测的保守预留超过剩余预算，拒绝发送。")
            from research_entry.budget import Ledger

            ledger = Ledger(output / "budget-ledger.json", budget_cny, prior, card)
            ledger.reserve(probe, "minimum-probe")
            _send_reserved_probe(ledger, probe)
            model_out = dict(model)
            model_out["called"] = True
            model_out["status"] = "probe_sent"
            model_out["reason"] = "只发送了一次已预留的探测请求，不是完整研究。"
            research = {
                "status": "probe_sent",
                "end_to_end": False,
                "decision": None,
                "reason": "探测请求不能当作研判结论。",
            }
            status = build_status(
                command="run",
                symbol=mapped,
                model=model_out,
                data=not_requested_data(as_of),
                research=research,
                limits=limits,
                extra={"ledger": ledger.snapshot(), "minimum_probe": probe, "assumptions": list(ASSUMPTIONS)},
            )
            write_status(output, status)
            return 0
        if not afford["send_allowed"]:
            detail = (
                f"最坏预留 {plan['reserved_cny']} 元，剩余 {afford['remaining_cny']} 元。"
                f"{plan['completion_blocker']}"
            )
            raise BudgetRefusal(detail)
        return _run_reserved_research(
            output, mapped, model, limits, chosen, as_of, card, budget_cny, prior, plan, ledger_prior=prior,
        )
    except BudgetRefusal as exc:
        refusal = str(exc)
    model = dict(model)
    model["called"] = False
    model["status"] = "budget_refused"
    model["reason"] = refusal
    research = {
        "status": "budget_refused",
        "end_to_end": False,
        "decision": None,
        "reason": refusal,
    }
    status = build_status(
        command="run",
        symbol=mapped,
        model=model,
        data=not_requested_data(as_of),
        research=research,
        limits=limits,
        extra={
            "assumptions": list(ASSUMPTIONS),
            "price_source": card.source,
            "price_expires_at": card.expires_at,
            "research_plan": plan,
            "research_affordability": afford,
            "minimum_probe": probe,
            "minimum_probe_affordability": probe_afford,
            "full_research_reliable_within_budget": False,
            "budget_mode": budget_mode,
        },
    )
    write_status(output, status)
    return 2


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
    budget_cny: str | None = None,
    prior_hold_cny: str | None = None,
    input_token_cap: int | None = None,
    budget_mode: str = "research",
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

    provider_name = (model.get("provider") or "").lower()
    if provider_name == "deepseek" or budget_cny not in (None, ""):
        return _refuse_unbounded_research(
            output,
            mapped=mapped,
            model=model,
            limits=limits,
            chosen=chosen,
            as_of=as_of,
            budget_cny=budget_cny,
            prior_hold_cny=prior_hold_cny,
            input_token_cap=input_token_cap,
            backend_url=backend_url,
            budget_mode=budget_mode,
        )

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

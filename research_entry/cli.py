"""Console entry: tradingagents-research.

No interactive prompts. Missing model configuration is a normal, non-crash
result with exit code 2.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from research_entry.budget import conservative_probe_hold
from research_entry.limits import DEFAULT_LIMITS
from research_entry.runner import (
    execute_budget,
    execute_data_check,
    execute_map,
    execute_preflight,
    execute_run,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tradingagents-research",
        description=(
            "TradingAgents 研究入口。只做隔离研究调用："
            "不连券商、不下单、不读持仓、不发飞书、不启定时任务、不监听公网。"
            "未配置模型时友好退出，不会假装研究已经完成。"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    map_cmd = sub.add_parser("map", help="核对符号规范化（不访问网络，不产生行情）")
    map_cmd.add_argument("--symbol", help="单个代码，例如 0700.HK 或 601398.SH")
    map_cmd.add_argument("--market", help="HK、SS、SH、SZ 或 US")
    map_cmd.add_argument(
        "--reference",
        action="store_true",
        help="同时核对 0700.HK、1810.HK、601398.SS、002779.SZ、688825.SS",
    )
    map_cmd.add_argument("--output", required=True, help="状态与中文概要的输出目录")

    preflight = sub.add_parser("preflight", help="安装自检：不读凭证值，不访问模型")
    preflight.add_argument("--output", required=True)

    data = sub.add_parser("data-check", help="用 yfinance 做只读公开日线连通性测试")
    data.add_argument("--symbol", required=True)
    data.add_argument("--market", required=True)
    data.add_argument("--date", required=True, help="分析日 YYYY-MM-DD，只保留不晚于该日的K线")
    data.add_argument("--lookback-days", type=int, default=31)
    data.add_argument("--output", required=True)

    run = sub.add_parser("run", help="在模型已显式配置时运行研究；否则写明未配置并退出")
    run.add_argument("--symbol", required=True)
    run.add_argument("--market", required=True)
    run.add_argument("--date", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--provider", help="必须显式指定，例如 ollama、openai_compatible、deepseek")
    run.add_argument("--quick-model")
    run.add_argument("--deep-model")
    run.add_argument("--backend-url", help="openai_compatible 必填；也可由环境变量提供")
    run.add_argument("--analysts", default="market", help="逗号分隔，默认只有 market")
    run.add_argument("--max-debate-rounds", type=int, default=DEFAULT_LIMITS["max_debate_rounds"])
    run.add_argument("--max-risk-rounds", type=int, default=DEFAULT_LIMITS["max_risk_discuss_rounds"])
    run.add_argument("--max-tool-rounds", type=int, default=DEFAULT_LIMITS["max_tool_rounds"])
    run.add_argument("--max-tokens", type=int, default=DEFAULT_LIMITS["max_tokens"])
    run.add_argument("--timeout", type=float, default=DEFAULT_LIMITS["request_timeout"])
    run.add_argument("--max-retries", type=int, default=DEFAULT_LIMITS["llm_max_retries"])
    run.add_argument("--budget-cny", help="本次总预算，人民币，十进制字符串")
    run.add_argument("--prior-hold-cny", help="先前占用。默认按基础探测的高峰未命中预留")
    run.add_argument("--input-token-cap", type=int, default=8192)
    run.add_argument("--budget-mode", choices=("research", "probe"), default="research")

    budget = sub.add_parser("budget", help="只核算预算，不构造模型客户端")
    budget.add_argument("--output", required=True)
    budget.add_argument("--quick-model", default="deepseek-flash")
    budget.add_argument("--deep-model", default="deepseek-v4-pro")
    budget.add_argument("--analysts", default="market")
    budget.add_argument("--budget-cny", default="1")
    budget.add_argument("--prior-hold-cny", default=None)
    budget.add_argument("--input-token-cap", type=int, default=8192)
    budget.add_argument("--max-debate-rounds", type=int, default=DEFAULT_LIMITS["max_debate_rounds"])
    budget.add_argument("--max-risk-rounds", type=int, default=DEFAULT_LIMITS["max_risk_discuss_rounds"])
    budget.add_argument("--max-tool-rounds", type=int, default=DEFAULT_LIMITS["max_tool_rounds"])
    budget.add_argument("--max-tokens", type=int, default=DEFAULT_LIMITS["max_tokens"])
    budget.add_argument("--max-retries", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    output = Path(args.output)
    try:
        if args.command == "map":
            if not args.symbol and not args.reference:
                parser.error("map 需要 --symbol 或 --reference")
            code = execute_map(output, args.symbol, args.market, args.reference)
        elif args.command == "preflight":
            code = execute_preflight(output, dict(os.environ))
        elif args.command == "data-check":
            code = execute_data_check(output, args.symbol, args.market, args.date, args.lookback_days)
        elif args.command == "run":
            backend = args.backend_url or os.environ.get("TRADINGAGENTS_LLM_BACKEND_URL")
            code = execute_run(
                output,
                args.symbol,
                args.market,
                args.date,
                provider=args.provider or os.environ.get("TRADINGAGENTS_LLM_PROVIDER"),
                quick_model=args.quick_model or os.environ.get("TRADINGAGENTS_QUICK_THINK_LLM"),
                deep_model=args.deep_model or os.environ.get("TRADINGAGENTS_DEEP_THINK_LLM"),
                backend_url=backend,
                analysts=args.analysts,
                limit_overrides={
                    "max_debate_rounds": args.max_debate_rounds,
                    "max_risk_discuss_rounds": args.max_risk_rounds,
                    "max_tool_rounds": args.max_tool_rounds,
                    "max_tokens": args.max_tokens,
                    "request_timeout": args.timeout,
                    "llm_max_retries": args.max_retries,
                },
                environ=dict(os.environ),
                budget_cny=args.budget_cny or os.environ.get("TRADINGAGENTS_BUDGET_CNY"),
                prior_hold_cny=args.prior_hold_cny or os.environ.get("TRADINGAGENTS_PRIOR_HOLD_CNY"),
                input_token_cap=args.input_token_cap,
                budget_mode=args.budget_mode,
            )
        elif args.command == "budget":
            from research_entry.limits import parse_analysts, resolve_limits
            code = execute_budget(
                output,
                analysts=parse_analysts(args.analysts),
                limits=resolve_limits({
                    "max_debate_rounds": args.max_debate_rounds,
                    "max_risk_discuss_rounds": args.max_risk_rounds,
                    "max_tool_rounds": args.max_tool_rounds,
                    "max_tokens": args.max_tokens,
                    "llm_max_retries": args.max_retries,
                }),
                quick_model=args.quick_model,
                deep_model=args.deep_model,
                budget_cny=args.budget_cny,
                prior_hold_cny=args.prior_hold_cny or str(conservative_probe_hold()),
                input_token_cap=args.input_token_cap,
                backend_url=None,
            )
        else:
            parser.error(f"未知命令 {args.command}")
            return
    except ValueError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    raise SystemExit(code)


if __name__ == "__main__":
    main()

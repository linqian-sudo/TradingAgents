"""Write status.json and a Chinese summary from the same fields.

The summary copies numbers that are already in the status object. It does
not add prices, ratings, or company facts that the status does not contain.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from research_entry.provenance import (
    FORK_REPO,
    LICENSE_NAME,
    SCHEMA,
    UPSTREAM_COMMIT,
    UPSTREAM_REPO,
    UPSTREAM_VERSION,
)


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _zh_flag(value) -> str:
    if value is True:
        return "是"
    if value is False:
        return "否"
    return "未核对"


def _bar_lines(bar: dict | None) -> list[str]:
    if not bar:
        return ["最近一根K线：无。未编造开高低收或成交量。"]
    order = ("date", "open", "high", "low", "close", "volume")
    parts = []
    for key in order:
        if key in bar:
            parts.append(f"{key}={bar[key] if bar[key] is not None else '无'}")
    return ["最近一根K线（仅来自本次返回）：" + "，".join(parts)]


def render_summary(status: dict) -> str:
    model = status.get("model") or {}
    data = status.get("data") or {}
    research = status.get("research") or {}
    symbol = status.get("symbol") or {}
    lines = [
        "# TradingAgents 研究入口状态",
        "",
        f"- 生成时间（UTC）：{status.get('generated_at')}",
        f"- 命令：{status.get('command')}",
        f"- 上游：TradingAgents {status.get('package', {}).get('version')}，commit {status.get('package', {}).get('upstream_commit')}，许可证 {status.get('package', {}).get('license')}",
        "- 验证位置：仅当前执行环境。本文件不表示已安装到父级共享 Linux 或操作者的个人电脑。",
        "",
        "## 模型",
        f"- 状态：{model.get('status')}",
        f"- 是否已调用模型：{_zh_flag(bool(model.get('called')))}",
        f"- 说明：{model.get('reason')}",
        "",
        "## 研究结论",
        f"- 状态：{research.get('status')}",
        f"- 端到端研判：{_zh_flag(bool(research.get('end_to_end')))}",
        f"- 说明：{research.get('reason')}",
        "",
        "## 本次代码",
    ]
    if not symbol:
        lines.append("- 本命令未指定单一个股。")
    else:
        lines.extend([
            f"- 输入：{symbol.get('input')}",
            f"- 规范化：{symbol.get('canonical')}",
            f"- 声明市场：{symbol.get('market')}",
            f"- 后缀是否一致：{_zh_flag(symbol.get('market_matches'))}",
            "- 映射结果不包含行情，也不证明代码已经上市。",
        ])
        for note in symbol.get("notes") or []:
            lines.append(f"- 备注：{note}")
    mappings = (status.get("extra") or {}).get("mappings")
    if mappings:
        lines.append("- 对照表（exists 与 quote 固定为空，表示未核验上市与行情）：")
        for row in mappings:
            lines.append(
                f"  - {row.get('input')} / 市场 {row.get('market')} -> {row.get('canonical')}"
                f"（改写={_zh_flag(row.get('changed'))}，市场匹配={_zh_flag(row.get('market_matches'))}，exists=空，quote=空）"
            )
    lines.extend([
        "",
        "## 公开历史数据",
        f"- 状态：{data.get('status')}",
        f"- 来源：{data.get('source')}（{data.get('source_host')}）",
        f"- 拉取时间（UTC）：{data.get('fetched_at')}",
        f"- 窗口：{data.get('window_start')} 至分析日 {data.get('as_of')}（请求 end 为 {data.get('window_end_exclusive')}，yfinance 的 end 不含当日）",
        f"- 返回行数：{data.get('rows')}",
        f"- 首日：{data.get('first_date')}",
        f"- 末日：{data.get('last_date')}",
    ])
    lines.extend(_bar_lines(data.get("last_bar")))
    if data.get("error"):
        lines.append(f"- 错误：{data.get('error')}")
        if data.get("cause"):
            lines.append(f"- 原因类型：{data.get('cause_type')}；原因：{data.get('cause')}")
    lines.extend([
        f"- 覆盖说明：{data.get('coverage_note')}",
        "",
        "## 安全边界",
        "- 未连接券商，未下单，未读取持仓或资产。",
        "- 未发送飞书，未启用异动监控或定时任务，未监听公网端口。",
        "- 未读取或打印 API Key、token 或既有凭证。",
        "",
        "这不是投资建议。无数据处保持空白，不用估计值填补。",
        "",
    ])
    return "\n".join(lines)


def build_status(*, command: str, symbol: dict | None, model: dict, data: dict, research: dict, limits: dict, extra: dict | None = None) -> dict:
    import tradingagents

    version = getattr(tradingagents, "__version__", UPSTREAM_VERSION)
    status = {
        "schema": SCHEMA,
        "generated_at": _utc_now(),
        "verified_on": "this process only; not a parent-host or laptop install",
        "package": {
            "distribution": "tradingagents",
            "version": version,
            "research_entry": "0.1.0",
            "upstream_repo": UPSTREAM_REPO,
            "upstream_commit": UPSTREAM_COMMIT,
            "fork_repo": FORK_REPO,
            "license": LICENSE_NAME,
        },
        "command": command,
        "symbol": symbol,
        "model": model,
        "data": data,
        "research": research,
        "limits": limits,
        "safety": {
            "broker_connected": False,
            "orders_sent": False,
            "portfolio_read": False,
            "feishu_sent": False,
            "scheduler_enabled": False,
            "public_port_bound": False,
            "credentials_printed": False,
            "paid_model_call": bool(model.get("called")),
        },
    }
    if extra:
        status["extra"] = extra
    return status


def write_status(output_dir: Path, status: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "status.json"
    path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = output_dir / "summary.zh.md"
    summary.write_text(render_summary(status), encoding="utf-8")
    return path


def not_requested_data(as_of: str | None = None) -> dict:
    return {
        "source": None,
        "source_host": None,
        "endpoint_note": "本命令未请求行情。",
        "symbol_input": None,
        "canonical": None,
        "interval": None,
        "window_start": None,
        "window_end_exclusive": None,
        "as_of": as_of,
        "fetched_at": None,
        "rows": None,
        "first_date": None,
        "last_date": None,
        "last_bar": None,
        "columns": [],
        "status": "not_requested",
        "error": None,
        "cause_type": None,
        "cause": None,
        "research_claim": False,
        "coverage_note": "港股与 A 股的公开文档支持不保证实时行情，也不保证财报字段完整。",
    }

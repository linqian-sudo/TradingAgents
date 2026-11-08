"""Decide whether a model call is allowed. Never reads or prints secret values."""

from __future__ import annotations

from tradingagents.llm_clients.api_key_env import PROVIDER_API_KEY_ENV

# These authenticate outside a single API key. This entry does not start
# a browser login and does not consult the AWS credential chain.
SIGN_IN_PROVIDERS = frozenset({"chatgpt"})
CREDENTIAL_CHAIN_PROVIDERS = frozenset({"bedrock"})

# Operator-selectable paths that can avoid a hosted per-token bill.
# Cost is still not promised: local compute and relay pricing are outside
# this package, and nothing here was timed against a live model.
LOCAL_OR_COMPATIBLE = (
    {
        "provider": "ollama",
        "credential_env": None,
        "endpoint_env": "OLLAMA_BASE_URL",
        "default_endpoint": "http://localhost:11434/v1",
        "note": "本地 Ollama。未设置 OLLAMA_BASE_URL 时上游默认本机 11434。不需要 API Key。本安装未启动 Ollama，也未拉取模型。",
    },
    {
        "provider": "openai_compatible",
        "credential_env": "OPENAI_COMPATIBLE_API_KEY",
        "endpoint_env": "TRADINGAGENTS_LLM_BACKEND_URL",
        "default_endpoint": None,
        "note": "OpenAI 兼容接口（vLLM、LM Studio、llama.cpp 或其他兼容端点）。必须提供 backend URL。Key 可省略，仅在端点要求时设置。",
    },
)


def _present(environ: dict, name: str | None) -> bool:
    if not name:
        return False
    value = environ.get(name)
    return isinstance(value, str) and bool(value.strip())


def credential_catalog() -> dict:
    """Names of credential variables. Values are never included."""
    hosted = []
    for provider, env_name in PROVIDER_API_KEY_ENV.items():
        if provider in SIGN_IN_PROVIDERS or provider in CREDENTIAL_CHAIN_PROVIDERS:
            continue
        if provider in {item["provider"] for item in LOCAL_OR_COMPATIBLE}:
            continue
        hosted.append({"provider": provider, "credential_env": env_name, "required": True})
    return {
        "minimum_for_a_hosted_provider": "只需要所选 provider 对应的一个 API Key 变量，加上显式的 quick/deep 模型名。未选择 provider 时不会回落到默认付费模型。",
        "not_required_for_the_yahoo_history_check": [
            "任何模型 API Key",
            "FRED_API_KEY",
            "ALPHA_VANTAGE_API_KEY",
        ],
        "local_or_compatible": list(LOCAL_OR_COMPATIBLE),
        "hosted_provider_env": hosted,
        "refused_here": [
            {
                "provider": "chatgpt",
                "reason": "需要交互登录。本入口不代为登录，也不读取已保存的登录令牌。",
            },
            {
                "provider": "bedrock",
                "reason": "走 AWS 凭证链。本入口不读取、不使用该凭证链。",
            },
        ],
        "cost": "安装与本次测试未调用模型，因此没有测得单次费用。选定服务商、模型、轮数和输出长度之前，不能承诺单次价格。",
    }


def assess_model(
    *,
    provider: str | None,
    quick_model: str | None,
    deep_model: str | None,
    backend_url: str | None,
    environ: dict,
) -> dict:
    """Return a structured gate. ``configured`` is the only state that may call a model."""
    catalog_note = "未打印任何凭证值。"
    base = {
        "status": "model_unconfigured",
        "provider": None,
        "quick_model": (quick_model or "").strip() or None,
        "deep_model": (deep_model or "").strip() or None,
        "backend_url_set": bool((backend_url or "").strip()),
        "credential_env": None,
        "credential_present": False,
        "called": False,
        "missing": [],
        "reason": "",
        "note": catalog_note,
    }
    chosen = (provider or "").strip().lower()
    if not chosen:
        base["missing"] = ["provider", "quick_model", "deep_model"]
        base["reason"] = (
            "未选择模型服务。本入口不会使用上游默认的 openai / gpt-6 组合去发起调用。"
            "请显式指定 provider 与两个模型名；没有密钥或本地端点时到此停止。"
        )
        return base

    base["provider"] = chosen
    if chosen not in PROVIDER_API_KEY_ENV:
        base["missing"] = ["provider"]
        base["reason"] = f"不支持的 provider：{chosen}。未发起调用。"
        return base
    if chosen in SIGN_IN_PROVIDERS:
        base["reason"] = "chatgpt 需要交互登录。本阶段不登录、不读取令牌、不调用模型。"
        return base
    if chosen in CREDENTIAL_CHAIN_PROVIDERS:
        base["reason"] = "bedrock 依赖 AWS 凭证链。本入口不读取、不使用该凭证，也不调用模型。"
        return base

    missing = []
    if not base["quick_model"]:
        missing.append("quick_model")
    if not base["deep_model"]:
        missing.append("deep_model")

    key_env = PROVIDER_API_KEY_ENV[chosen]
    base["credential_env"] = key_env
    base["credential_present"] = _present(environ, key_env)

    if chosen == "openai_compatible" and not base["backend_url_set"]:
        missing.append("backend_url")
    if chosen not in {"ollama", "openai_compatible"} and not base["credential_present"]:
        missing.append(key_env or "api_key")

    if missing:
        base["missing"] = missing
        base["reason"] = "模型服务未配置完整：" + "、".join(missing) + "。未构造模型客户端，未发起调用。"
        return base

    base["status"] = "configured"
    base["missing"] = []
    base["reason"] = "参数与所需凭证名称已齐。只有显式执行 run 且通过此检查后才会构造客户端。"
    if chosen == "ollama":
        base["reason"] += " 本地服务是否在监听、模型是否已拉取，本次检查不代为连接。"
    return base

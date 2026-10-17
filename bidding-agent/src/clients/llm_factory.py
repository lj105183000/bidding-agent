"""LLM 客户端工厂：按 provider 名实例化对应客户端。

设计：**注册表驱动**，不是 if-elif 链
-----------------------------------
本模块原先是一个 `if provider == "deepseek" ... elif "zhipu" ...` 的长链，
每接一个 provider 就要在两个地方各加一个分支（凭据解析 + 客户端构造），
漏改一处就出现"凭据拿到了但构造不出来"或反之。

现在每个 provider 一条 `ProviderSpec` 声明式登记，新增 provider 只加一行。

★ 泛化而非特例：Ollama / vLLM 也走同一套
---------------------------------------
它们都是 OpenAI 兼容协议，只是凭据来源不同。故 `ProviderSpec` 把
"从哪读凭据"与"怎么造客户端"分开——前者是配置问题，后者是协议问题。
原先的 if-elif 把两者混在一起，导致每加一个 provider 都要重写一遍凭据回落逻辑。

★ 凭据回落口径（沿用并明确化）
----------------------------
    通用 LLM_API_KEY 优先（显式覆盖），否则回落到 provider 专属变量

历史事故：原实现只读通用的 `LLM_API_KEY`，完全不看 provider 专属变量，
于是 `.env` 里明明配了 `DEEPSEEK_API_KEY` 也被判为"未配置"，整条链路降级成
"直接返回知识库原文"——用户看到的是"AI 不说话了"，而不是任何报错。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from src.clients.base_client import BaseLLMClient, OpenAICompatClient, UnavailableLLMClient
from src.clients.deepseek_client import DeepSeekClient
from src.clients.zai_client import ZaiClient
from src.config import settings
from src.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ProviderSpec:
    """一个 provider 的登记信息。

    把"凭据从哪来"与"客户端怎么造"分开声明：
        `credentials`  读取该 provider 的 (api_key, base_url, model)，已是纯配置问题
        `factory`      用凭据造客户端，已是纯协议问题

    这样新增 provider 只要给一条 spec，不必再动 `get_llm_client` 的控制流。
    """

    name: str
    credentials: Callable[[], tuple[str, str | None, str]]
    factory: Callable[[str, str, str | None, float], BaseLLMClient]
    #: 该 provider 的凭据缺失时，提示用户该配哪些环境变量
    env_hint: str
    #: ★ **厂商**名。默认与 name 相同，只有"同一家的多个名字"才需显式指定。
    #:
    #: 为什么需要这个字段（实测缺陷）
    #: -------------------------
    #: zhipu 与 zai 是智谱的两个名字，但它们在注册表里是**两条独立记录**。
    #: 于是失败冷却按 name 记时会各存一份：
    #:     _cooldown['zhipu']  与  _cooldown['zai']  互不相干
    #: 拉黑一个不会拉黑另一个 → 候选链里写 zhipu,zai 会**两次撞同一个坏凭据**
    #: （实测：parse_chain("", "zhipu,zai") 返回 ["zhipu","zai"] 都保留）。
    #:
    #: 有了 vendor，冷却与候选链都能按**厂商**折叠。
    vendor: str = ""


def _deepseek_credentials() -> tuple[str, str | None, str]:
    return (
        settings.llm_api_key or settings.deepseek_api_key,
        settings.llm_base_url or settings.deepseek_base_url,
        settings.llm_model if settings.llm_api_key else settings.deepseek_model,
    )


def _zhipu_credentials() -> tuple[str, str | None, str]:
    from src.clients.zai_client import ZHIPU_BASE_URL

    return (
        settings.llm_api_key or settings.zhipu_api_key,
        settings.llm_base_url or ZHIPU_BASE_URL,
        settings.llm_model if settings.llm_api_key else settings.zhipu_model,
    )


def _ollama_credentials() -> tuple[str, str | None, str]:
    return (
        settings.llm_api_key or settings.ollama_api_key,
        settings.llm_base_url or settings.ollama_base_url,
        settings.llm_model if settings.llm_api_key else settings.ollama_model,
    )


def _vllm_credentials() -> tuple[str, str | None, str]:
    return (
        settings.llm_api_key or settings.vllm_api_key,
        settings.llm_base_url or settings.vllm_base_url,
        settings.llm_model if settings.llm_api_key else settings.vllm_model,
    )


def _openai_credentials() -> tuple[str, str | None, str]:
    return (settings.llm_api_key, settings.llm_base_url, settings.llm_model)


def _make_deepseek(api_key: str, model: str, base_url: str | None, temperature: float):
    return DeepSeekClient(api_key=api_key, model=model, base_url=base_url, temperature=temperature)


def _make_zai(api_key: str, model: str, base_url: str | None, temperature: float):
    return ZaiClient(api_key=api_key, model=model, base_url=base_url, temperature=temperature)


def _make_openai_compat(provider: str):
    """构造通用 OpenAI 兼容客户端（Ollama / vLLM / OpenAI 等）。"""

    def factory(api_key: str, model: str, base_url: str | None, temperature: float):
        return OpenAICompatClient(
            api_key=api_key, model=model, base_url=base_url,
            temperature=temperature, provider=provider,
        )

    return factory


#: provider 注册表。新增 provider 只需在此加一行。
def vendor_of(name: str) -> str:
    """provider 名 → **厂商**名。

    别名（zai → zhipu）折叠到同一厂商；未登记的 provider 原样返回
    （不猜、不抛异常 —— 调用方常在异常处理路径上用它）。
    """
    spec = PROVIDERS.get((name or "").strip().lower())
    if spec is None:
        return (name or "").strip().lower()
    return (getattr(spec, "vendor", "") or spec.name or name).strip().lower()


PROVIDERS: dict[str, ProviderSpec] = {
    "deepseek": ProviderSpec(
        name="deepseek",
        credentials=_deepseek_credentials,
        factory=_make_deepseek,
        env_hint="DEEPSEEK_API_KEY / DEEPSEEK_MODEL / DEEPSEEK_BASE_URL",
    ),
    # "zai" 是智谱 SDK 的叫法，与配置里的 "zhipu" 同义。
    # 两个名字都收：用户在 .env 里写哪个都能用，避免"配了却不生效"的困惑。
    "zhipu": ProviderSpec(
        name="zhipu",
        credentials=_zhipu_credentials,
        factory=_make_zai,
        env_hint="ZHIPU_API_KEY / ZHIPU_MODEL",
    ),
    "zai": ProviderSpec(
        name="zai",
        # 与 zhipu 是同一家：冷却与候选链按 vendor 折叠，不会试两遍
        vendor="zhipu",
        credentials=_zhipu_credentials,
        factory=_make_zai,
        env_hint="ZHIPU_API_KEY / ZHIPU_MODEL",
    ),
    "ollama": ProviderSpec(
        name="ollama",
        credentials=_ollama_credentials,
        factory=_make_openai_compat("ollama"),
        env_hint="OLLAMA_BASE_URL / OLLAMA_MODEL",
    ),
    "vllm": ProviderSpec(
        name="vllm",
        credentials=_vllm_credentials,
        factory=_make_openai_compat("vllm"),
        env_hint="VLLM_BASE_URL / VLLM_MODEL",
    ),
    "openai": ProviderSpec(
        name="openai",
        credentials=_openai_credentials,
        factory=_make_openai_compat("openai"),
        env_hint="LLM_API_KEY / LLM_MODEL",
    ),
    # 别名：这些厂商都提供 OpenAI 兼容端点，直接走通用客户端
    "qwen": ProviderSpec(
        name="qwen",
        credentials=_openai_credentials,
        factory=_make_openai_compat("qwen"),
        env_hint="LLM_API_KEY / LLM_BASE_URL / LLM_MODEL",
    ),
    "moonshot": ProviderSpec(
        name="moonshot",
        credentials=_openai_credentials,
        factory=_make_openai_compat("moonshot"),
        env_hint="LLM_API_KEY / LLM_BASE_URL / LLM_MODEL",
    ),
}


def get_llm_client(provider: str) -> BaseLLMClient:
    """按 provider 名返回客户端实例。

    **永不返回 None、永不抛异常**——配置缺失时返回 `UnavailableLLMClient`，
    让 `pipeline` 的 `self.llm.available()` 判断能正常工作并降级为
    "直接返回知识库原文"。抛异常会让"没配密钥"这种纯配置问题变成服务不可用。

    `provider` 大小写与首尾空格都会归一（`.env` 是手写的，`" DeepSeek "`
    这种写法不该导致回落）。
    """
    name = (provider or "").strip().lower()
    if not name:
        return UnavailableLLMClient("未指定 provider（检查 LLM_PROVIDER）")

    spec = PROVIDERS.get(name)
    if spec is None:
        return UnavailableLLMClient(
            f"不支持的 LLM provider: {provider!r}"
            f"（可用：{', '.join(sorted(PROVIDERS))}）"
        )

    api_key, base_url, model = spec.credentials()

    if not api_key:
        return UnavailableLLMClient(
            f"未找到 provider='{name}' 的 API Key"
            f"（可设通用 LLM_API_KEY，或设专属变量：{spec.env_hint}）"
        )
    if not model:
        return UnavailableLLMClient(
            f"provider='{name}' 未配置模型名（{spec.env_hint}）"
        )

    try:
        return spec.factory(api_key, model, base_url, settings.llm_temperature)
    except Exception as exc:
        # 构造期异常（如 base_url 非法）也降级而不是抛——与"未配置"同等对待，
        # 因为对上层来说两者的处理方式相同：不可用 → 返回知识库原文
        logger.exception("构造 %s 客户端失败", name)
        return UnavailableLLMClient(f"构造 provider='{name}' 客户端失败：{exc}")


def list_providers() -> list[str]:
    """返回已登记的 provider 名（供 `/api/models` 之类的接口展示）。"""
    return sorted(PROVIDERS)


__all__ = ["PROVIDERS", "ProviderSpec", "get_llm_client", "list_providers"]

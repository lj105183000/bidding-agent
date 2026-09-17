from src.clients.base import BaseLLMClient
from src.clients.deepseek import DeepSeekClient
from src.clients.openai_compatible import OpenAICompatibleClient
from src.clients.zhipu import ZhipuClient
from src.config import settings


def get_llm_client(provider: str | None = None) -> BaseLLMClient:
    """根据 provider 和项目配置创建 LLM 客户端。"""
    selected_provider = provider or settings.llm_provider

    if selected_provider == "deepseek":
        return DeepSeekClient(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            model=settings.deepseek_model,
        )

    if selected_provider == "zhipu":
        return ZhipuClient(
            api_key=settings.zhipu_api_key,
            model=settings.zhipu_model,
        )

    if selected_provider == "vllm":
        return OpenAICompatibleClient(
            provider="vllm",
            api_key="EMPTY",
            base_url=settings.vllm_base_url,
            model=settings.vllm_model,
        )

    if selected_provider == "ollama":
        return OpenAICompatibleClient(
            provider="ollama",
            api_key="ollama",
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
        )

    raise ValueError(f"不支持的 LLM provider: {selected_provider}")
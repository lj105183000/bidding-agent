"""LLM 客户端工厂：根据配置动态实例化大模型客户端。

架构约定（与 pipeline.py 对齐）：
1. 工厂函数 get_llm_client(provider) 返回客户端实例。
2. 客户端必须实现 available() 方法，返回布尔值。
3. 客户端必须实现 chat_stream(messages)，返回生成器，逐块吐出文本。
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from openai import OpenAI

from src.config import settings
from src.logging_config import get_logger

logger = get_logger(__name__)


class BaseLLMClient:
    """LLM 客户端基类，定义接口契约。"""

    def available(self) -> bool:
        """返回当前客户端是否可用（用于 pipeline 的降级判断）。"""
        raise NotImplementedError

    def chat_stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        """流式生成回答，逐块返回字符串。"""
        raise NotImplementedError


class UnavailableLLMClient(BaseLLMClient):
    """当配置缺失时使用的空客户端，确保 pipeline 能平滑降级到检索直返。"""

    def __init__(self, reason: str):
        self.reason = reason
        logger.warning(f"LLM 客户端不可用，将降级为知识库原文返回。原因: {reason}")

    def available(self) -> bool:
        return False

    def chat_stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        # 理论上 pipeline 在 available() 为 False 时不会调用到这里
        yield ""


class OpenAICompatibleClient(BaseLLMClient):
    """基于 openai SDK 的客户端，兼容 OpenAI、DeepSeek、通义千问、Kimi、本地 Ollama 等。"""

    def __init__(self, api_key: str, base_url: str | None, model: str, temperature: float = 0.1):
        self.model = model
        self.temperature = temperature
        try:
            # openai>=1.0.0 的标准初始化方式
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        except Exception as exc:
            logger.error(f"初始化 OpenAI 客户端失败: {exc}")
            self.client = None

    def available(self) -> bool:
        return self.client is not None and bool(self.model)

    def chat_stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        """调用 LLM 并流式返回文本。"""
        if not self.available():
            logger.error("OpenAI 客户端不可用，无法发起请求")
            return

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                stream=True,
            )
            for chunk in response:
                # 提取流式文本块，跳过空块
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        yield delta.content
        except Exception as exc:
            logger.error(f"LLM 流式调用失败: {exc}")
            raise  # 向上抛出，让 pipeline 捕获并执行降级逻辑


# ==========================================
# 工厂函数（pipeline.py 的调用入口）
# ==========================================

def get_llm_client(provider: str) -> BaseLLMClient | None:
    """根据 provider 和 settings 配置返回 LLM 客户端实例。

    如果配置缺失或 provider 不支持，返回 UnavailableLLMClient（或 None）。
    这样 pipeline.py 中的 `self.llm is not None and self.llm.available()` 能正常工作。
    """
    api_key = settings.llm_api_key
    base_url = settings.llm_base_url
    model = settings.llm_model

    if not api_key:
        return UnavailableLLMClient("未在 .env 中配置 LLM_API_KEY")

    # 统一使用 OpenAI 兼容协议，支持绝大多数国内外大模型
    if provider.lower() in ["openai", "deepseek", "qwen", "moonshot", "ollama", "zhipu"]:
        # 注意：如果用的是 zai-sdk（智谱），官方也提供 OpenAI 兼容接口
        # 这里统一走 OpenAI 客户端，极大简化代码
        return OpenAICompatibleClient(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=settings.llm_temperature,
        )
    
    # 如果未来要接入其他特殊 SDK，可以在这里添加 elif 分支
    logger.error(f"不支持的 LLM Provider: {provider}")
    return UnavailableLLMClient(f"不支持的 LLM Provider: {provider}")
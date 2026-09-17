from abc import ABC, abstractmethod
from collections.abc import AsyncIterator


class BaseLLMClient(ABC):
    """所有 LLM 客户端必须实现的统一接口。"""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """返回当前使用的模型名称。"""
        raise NotImplementedError

    @property
    @abstractmethod
    def provider(self) -> str:
        """返回当前 LLM 服务提供商名称。"""
        raise NotImplementedError

    @abstractmethod
    async def chat(self, messages: list[dict[str, str]]) -> str:
        """执行普通对话并返回完整回答。"""
        raise NotImplementedError

    @abstractmethod
    async def chat_stream(
        self, messages: list[dict[str, str]]
    ) -> AsyncIterator[str]:
        """执行流式对话并逐段返回回答。"""
        raise NotImplementedError

    @abstractmethod
    async def chat_stream_thinking(
        self, messages: list[dict[str, str]]
    ) -> AsyncIterator[dict[str, str]]:
        """执行带思考过程的流式对话。"""
        raise NotImplementedError

    @abstractmethod
    async def chat_raw(self, messages: list[dict[str, str]]) -> object:
        """返回底层 LLM API 的原始响应。"""
        raise NotImplementedError
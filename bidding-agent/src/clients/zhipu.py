from collections.abc import AsyncIterator
from functools import partial

import asyncio
from zai import ZhipuAiClient

from src.clients.base import BaseLLMClient
from src.config import settings


class ZhipuClient(BaseLLMClient):
    """智谱 GLM 文本模型客户端。"""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self._api_key = api_key or settings.zhipu_api_key
        self._model = model or settings.zhipu_model
        self._client = ZhipuAiClient(api_key=self._api_key)

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "zhipu"

    async def chat(self, messages: list[dict[str, str]]) -> str:
        response = await asyncio.to_thread(
            partial(
                self._client.chat.completions.create,
                model=self._model,
                messages=messages,
            )
        )
        return response.choices[0].message.content or ""

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
    ) -> AsyncIterator[str]:
        stream = await asyncio.to_thread(
            partial(
                self._client.chat.completions.create,
                model=self._model,
                messages=messages,
                stream=True,
            )
        )

        for chunk in stream:
            if not chunk.choices:
                continue

            content = chunk.choices[0].delta.content
            if content:
                yield content

    async def chat_stream_thinking(
        self,
        messages: list[dict[str, str]],
    ) -> AsyncIterator[dict[str, str]]:
        stream = await asyncio.to_thread(
            partial(
                self._client.chat.completions.create,
                model=self._model,
                messages=messages,
                stream=True,
            )
        )

        for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            reasoning = getattr(delta, "reasoning_content", None)
            content = delta.content

            if reasoning:
                yield {"type": "thinking", "content": reasoning}

            if content:
                yield {"type": "content", "content": content}

    async def chat_raw(self, messages: list[dict[str, str]]) -> object:
        return await asyncio.to_thread(
            partial(
                self._client.chat.completions.create,
                model=self._model,
                messages=messages,
            )
        )
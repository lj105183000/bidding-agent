from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from src.config import settings


class OpenRouterClient:
    """OpenRouter LLM 客户端。"""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        model: str | None = None,
    ):
        self._client = AsyncOpenAI(
            api_key=api_key or settings.openrouter_api_key,
            base_url=base_url,
            trust_env=False,
        )
        self._model = model or settings.openrouter_model

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "openrouter"

    async def chat(self, messages: list[dict[str, str]]) -> str:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
        )
        return response.choices[0].message.content or ""

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
    ) -> AsyncIterator[str]:
        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            stream=True,
        )

        async for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content

    async def chat_stream_thinking(
        self,
        messages: list[dict[str, str]],
    ) -> AsyncIterator[dict[str, str]]:
        async for content in self.chat_stream(messages):
            yield {"type": "content", "content": content}

    async def chat_raw(self, messages: list[dict[str, str]]) -> object:
        return await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
        )
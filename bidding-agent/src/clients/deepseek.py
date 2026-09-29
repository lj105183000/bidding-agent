from openai import AsyncOpenAI

from src.clients.base import BaseLLMClient


class DeepSeekClient(BaseLLMClient):
    """DeepSeek LLM 客户端。"""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
    ):
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            trust_env=False,
        )
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "deepseek"

    async def chat(self, messages: list[dict[str, str]]) -> str:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
        )

        return response.choices[0].message.content or ""

    async def chat_stream(self, messages):
        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            stream=True,
        )

        async for chunk in stream:
            if not chunk.choices:
                continue

            content = chunk.choices[0].delta.content

            if content:
                yield content

    async def chat_stream_thinking(self, messages):
        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            stream=True,
        )

        async for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            reasoning = getattr(delta, "reasoning_content", None)
            content = delta.content

            if reasoning:
                yield {"type": "thinking", "content": reasoning}

            if content:
                yield {"type": "content", "content": content}
                
    async def chat_raw(self, messages):
        return await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
        )
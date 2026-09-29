import asyncio

from zai import ZhipuAiClient

from src.config import settings


class VisionClient:
    """智谱视觉模型客户端。"""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self._api_key = api_key or settings.zhipu_api_key
        self._model = model or settings.zhipu_vision_model
        self._client = ZhipuAiClient(api_key=self._api_key)

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "zhipu"

    async def analyze(
        self,
        image_base64: str,
        prompt: str,
    ) -> str:
        response = await asyncio.to_thread(
            self._client.chat.completions.create,
            model=self._model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": (
                                    f"data:image/jpeg;base64,"
                                    f"{image_base64}"
                                ),
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
        )

        return response.choices[0].message.content or ""
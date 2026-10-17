"""DeepSeek 客户端（原生 HTTP，OpenAI 兼容协议）。

为什么单独成文件而不是继续走 openai SDK
--------------------------------------
见 `base_client` 模块头的说明。这里补充 DeepSeek 特有的两点：

1. **reasoning_content**：思考模式下思维链放在 `delta.reasoning_content`，
   与正文分开。`chat_stream_thinking` 依赖它，而 `chat_stream` 必须**丢弃**它
   ——否则用户会在回答里看到大段思考过程。
   （`base_client.extract_delta` 已把该字段摊平，本文件只决定怎么用。）

   ⚠️ 旧文档说"换推理模型时把模型名设成 deepseek-reasoner"——**已失效**：
   DeepSeek 发布 V4 后取消了独立推理模型名，现在只有 `deepseek-flash` 与
   `deepseek-v4-pro`，两者都由请求参数 `{"thinking": {"type": "enabled"}}`
   控制思考模式（见 `BaseLLMClient.set_thinking`）。
   旧名 `deepseek-chat` / `deepseek-reasoner` 将于 2026-07-24 15:59 UTC 退役。

2. **base_url 已含 `/v1`**：`.env` 里的 `DEEPSEEK_BASE_URL` 默认
   `https://api.deepseek.com/v1`，故端点拼接是 `{base_url}/chat/completions`。
   不要在这里再补 `/v1`——会拼成 `/v1/v1/chat/completions` 而 404。
"""

from __future__ import annotations

from src.clients.base_client import OpenAICompatClient


class DeepSeekClient(OpenAICompatClient):
    """DeepSeek Chat / Responses API 客户端。

    模型名由 `settings.deepseek_model` 提供（默认 `deepseek-flash`）。

    **思考模式不是换模型名，而是开请求参数**：调用 `set_thinking(True)`
    后 `_payload` 会发送 `{"thinking": {"type": "enabled"}}`，思维链自动走
    `chat_stream_thinking` / `last_reasoning`。
    """

    provider = "deepseek"

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        base_url: str | None = None,
        temperature: float = 0.1,
        timeout: float = 300.0,
    ) -> None:
        super().__init__(
            api_key=api_key,
            model=model,
            base_url=base_url or "https://api.deepseek.com/v1",
            temperature=temperature,
            timeout=timeout,
        )


__all__ = ["DeepSeekClient"]

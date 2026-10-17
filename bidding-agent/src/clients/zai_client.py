"""智谱 AI 客户端（原生 HTTP，OpenAI 兼容协议）。

模型：`GLM-4.7-FlashX`（`settings.zhipu_model` 默认 `glm-4.7-flashx`），
用于纯文本对话。

★ 为什么不用 zai SDK
------------------
理由与 `base_client` 模块头相同（依赖更少、出站策略统一、行为可控）。
这里补充智谱特有的三点：

1. **端点**：`https://open.bigmodel.cn/api/paas/v4/chat/completions`
   （与 OpenAI 同形，故能复用 `OpenAICompatClient` 的整套实现）

2. **不接受 `parallel_tool_calls`**：该字段是 OpenAI 的扩展，
   智谱的兼容层不认，传了会报参数错误。故本类把它关掉
   （`supports_parallel_tool_calls = False`）——这正是基类留这个开关的原因。

3. **鉴权头仍是 `Bearer`**：智谱 v4 接口用 `Authorization: Bearer <api_key>`，
   与 OpenAI 一致；**不要**改成它文档里另一处提到的 JWT 形式——
   那是旧版 v3 接口的用法，v4 的 OpenAI 兼容层用 Bearer。

视觉模型（`glm-4.6v-flashx`）不在这里，见 `vision.py`：它的消息体是
多模态结构（`content` 为数组、带 data URL），与纯文本对话差异较大，
分成两个客户端比塞进一个类更清楚。
"""

from __future__ import annotations

from src.clients.base_client import OpenAICompatClient

#: 智谱 v4 的 OpenAI 兼容根地址（**不含** /chat/completions）
ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"


class ZaiClient(OpenAICompatClient):
    """智谱 GLM 文本对话客户端。

    类名 `ZaiClient` 与文件名 `zai_client.py` 对齐（zai 是智谱的 Python SDK 名，
    此处沿用该叫法以便与 `zhipu` 视觉客户端区分）。
    `provider` 仍报 `"zhipu"`——那是配置里的 provider 名（`LLM_PROVIDER=zhipu`），
    改成 `"zai"` 会让 `get_llm_client("zhipu")` 的匹配失效。
    """

    provider = "zhipu"
    # 智谱兼容层不认这个 OpenAI 扩展字段，传了会报参数错误
    supports_parallel_tool_calls = False

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
            base_url=base_url or ZHIPU_BASE_URL,
            temperature=temperature,
            timeout=timeout,
        )


__all__ = ["ZHIPU_BASE_URL", "ZaiClient"]

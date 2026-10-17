"""LLM / 视觉客户端集合。

模块职责划分
-----------
    base_client.py      接口契约（BaseLLMClient）+ 空实现 + 共享协议工具
    llm_factory.py      按 provider 名实例化客户端（注册表驱动）
    deepseek_client.py  DeepSeek（原生 HTTP，OpenAI 兼容协议）
    zai_client.py       智谱 GLM 文本对话（原生 HTTP，OpenAI 兼容协议）
    vision.py           智谱视觉模型（多模态消息体，单独一个客户端更清楚）

关于导出
-------
改造前本文件是空的，调用方一律写全路径（`from src.clients.llm_factory import ...`）。
这里显式导出后，新代码可写 `from src.clients import get_llm_client`。

两种写法都保留是有意的：显式路径能让人一眼看出某个 client 来自哪个模块，
排查"到底用的哪个 provider 实现"时少一次跳转。
"""

from src.clients.base_client import (
    BaseLLMClient,
    OpenAICompatClient,
    UnavailableLLMClient,
)
from src.clients.deepseek_client import DeepSeekClient
from src.clients.llm_factory import PROVIDERS, ProviderSpec, get_llm_client, list_providers
from src.clients.zai_client import ZHIPU_BASE_URL, ZaiClient

__all__ = [
    "PROVIDERS",
    "ZHIPU_BASE_URL",
    "BaseLLMClient",
    "DeepSeekClient",
    "OpenAICompatClient",
    "ProviderSpec",
    "UnavailableLLMClient",
    "ZaiClient",
    "get_llm_client",
    "list_providers",
]

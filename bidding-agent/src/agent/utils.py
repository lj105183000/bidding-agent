"""SSE 与流式输出工具函数。

帧格式契约见 docs/开发文档.md §6.1：每帧一行 JSON（ensure_ascii=False 保留中文），
以 `data: ` 前缀，帧间空行分隔；首帧是 2KB padding 注释用于强制代理立即 flush。
"""

import json
import time

from src.agent.constants import MAX_HISTORY_ROUNDS, STREAM_CHUNK_CHARS, STREAM_CHUNK_DELAY


def _sse(event: dict) -> str:
    """构造单帧 SSE 文本。ensure_ascii=False 保留中文，前端无需二次解码。"""
    return "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"


# 首帧 padding：`:` 开头是 SSE 注释行，约 2KB 的内容把代理/缓冲区撑开，保证后续帧实时到达
SSE_PADDING = ":" + " " * 2045 + "\n\n"

# 流结束标记（OpenAI 风格；前端解析器对非 JSON 行直接跳过）
SSE_DONE_MARK = "data: [DONE]\n\n"


def _pace_stream_chunks(text: str):
    """把长文本切成 STREAM_CHUNK_CHARS 小帧并附带帧间延时。

    用于非流式文本转流式（约束重试、RAG 降级拼接）的场景，
    保证前端渲染节奏与真实流式一致。
    """
    for i in range(0, len(text), STREAM_CHUNK_CHARS):
        yield text[i:i + STREAM_CHUNK_CHARS]
        time.sleep(STREAM_CHUNK_DELAY)


def _truncate_history(history: list) -> list:
    """只保留最近 MAX_HISTORY_ROUNDS 轮（user + assistant 成对）。

    截断后对齐到 user 开头，避免出现 assistant 打头的角色序列
    （部分模型对首条非 user 消息的对话处理不稳定）。
    """
    msgs = [
        m for m in (history or [])
        if isinstance(m, dict)
        and m.get("role") in ("user", "assistant")
        and m.get("content")
    ]
    trimmed = msgs[-(MAX_HISTORY_ROUNDS * 2):]
    while trimmed and trimmed[0].get("role") != "user":
        trimmed.pop(0)
    return trimmed

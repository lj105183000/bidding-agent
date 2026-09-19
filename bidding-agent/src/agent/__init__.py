"""Agent 包公共重导出。

契约（勿破坏）：`from src.agent import ...` 路径被测试与 eval 依赖，
调整内部结构时保持这里的导出面稳定。
"""

from src.agent.constants import (
    MAX_HISTORY_ROUNDS,
    MAX_TOKENS_DEEP,
    MAX_TOOL_ROUNDS,
    STREAM_CHUNK_CHARS,
    STREAM_CHUNK_DELAY,
    STREAM_FLUSH_CHARS,
    TOOL_TIMEOUT_SECONDS,
)
from src.agent.core import AgentError, BiddingAgent, LLMUnavailableError
from src.agent.prompts import SYSTEM_PROMPT

__all__ = [
    "AgentError",
    "BiddingAgent",
    "LLMUnavailableError",
    "MAX_HISTORY_ROUNDS",
    "MAX_TOKENS_DEEP",
    "MAX_TOOL_ROUNDS",
    "STREAM_CHUNK_CHARS",
    "STREAM_CHUNK_DELAY",
    "STREAM_FLUSH_CHARS",
    "SYSTEM_PROMPT",
    "TOOL_TIMEOUT_SECONDS",
]

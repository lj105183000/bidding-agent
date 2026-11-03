"""LangGraph 多 Agent 图的状态定义。

Phase 1 Chat 不接 checkpointer，故 state 里允许出现 `runtime_meta` 这类
进程内对象；但 **specialist_results / route / merged** 都是普通 dict，
方便后续接 checkpointer 或做 JSON 断言。
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class MultiAgentState(TypedDict, total=False):
    question: str
    history: list
    session_id: str
    provider: str
    web_enabled: bool
    deep_thinking_enabled: bool
    route: list[dict]
    task: dict
    specialist_results: Annotated[list[dict], operator.add]
    merged: dict
    answer: str
    partial: bool
    deadline_exceeded: bool
    fallback: str
    orchestration: dict
    errors: Annotated[list[str], operator.add]
    runtime_meta: Any
    #: 主请求的 LLMTraceRecorder（线程安全）；单独放 state，避免嵌套 dict
    #: 在 LangGraph 节点间被复制后丢失。
    trace_recorder: Any

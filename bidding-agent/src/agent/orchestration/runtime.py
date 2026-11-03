"""多 Agent 运行时：决定是否走图、桥接 LangGraph custom stream 到 SSE 事件。"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.agent.orchestration.graph import get_multi_agent_graph
from src.agent.orchestration.mode import multi_agent_mode
from src.agent.orchestration.router import (
    AgentTask,
    route_question,
    should_use_multi_agent,
)

logger = logging.getLogger(__name__)


class LegacyFallback(Exception):
    """图在产出任何 token 前决定/失败于走旧 ReAct；由 core 捕获并回退。"""


@dataclass(frozen=True)
class PreparedRun:
    mode: str
    tasks: tuple[AgentTask, ...]


def prepare_multi_agent(
    question: str,
    *,
    web_enabled: bool,
    skills=None,
    mode: str | None = None,
) -> PreparedRun | None:
    """返回 None 表示走旧 ReAct；否则返回确定性路由结果。"""
    selected_mode = mode or multi_agent_mode()
    if selected_mode == "off":
        return None
    tasks = route_question(question, web_enabled=web_enabled, skills=skills)
    if not should_use_multi_agent(question, selected_mode, tasks):
        return None
    return PreparedRun(mode=selected_mode, tasks=tuple(tasks))


def run_multi_agent_stream(
    agent,
    prepared: PreparedRun,
    *,
    question: str,
    history: list | None,
    session_id: str,
    provider: str,
    web_enabled: bool,
    deep_thinking_enabled: bool,
    meta: dict,
):
    """迭代图并产出 `(type, payload)`；token 只来自 synthesize 节点。"""
    meta["_multi_agent_mode"] = prepared.mode
    meta.setdefault("orchestration", {
        "enabled": True,
        "mode": prepared.mode,
        "router": "rules",
        "route": [task.agent for task in prepared.tasks],
        "partial": False,
        "deadline_exceeded": False,
        "fallback": "none",
    })
    state = {
        "question": question,
        "history": list(history or []),
        "session_id": session_id,
        "provider": provider,
        "web_enabled": bool(web_enabled),
        "deep_thinking_enabled": bool(deep_thinking_enabled),
        "route": [task.to_dict() for task in prepared.tasks],
        "specialist_results": [],
        "errors": [],
        "runtime_meta": meta,
    }
    graph = get_multi_agent_graph(agent)

    custom_stream_ok = True
    try:
        for event in graph.stream(state, stream_mode="custom"):
            if not isinstance(event, dict):
                continue
            event_type = str(event.get("type") or "")
            if event_type not in ("status", "token", "thinking", "reset"):
                continue
            payload = None if event_type == "reset" else event.get("content")
            yield event_type, payload
    except (TypeError, ValueError) as exc:
        # 旧版 LangGraph 不支持 custom stream：退化为 invoke 后一次性回放。
        custom_stream_ok = False
        logger.warning("LangGraph custom stream 不可用，回退 post_graph: %s", exc)
        graph.invoke(state)

    if not custom_stream_ok and meta.get("answer_text"):
        yield "status", "正在生成回答..."
        yield "token", meta["answer_text"]

    if meta.get("_multi_agent_needs_legacy") and not meta.get("streamed"):
        raise LegacyFallback

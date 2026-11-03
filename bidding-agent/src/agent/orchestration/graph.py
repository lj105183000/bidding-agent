"""LangGraph 多 Agent 图（Phase 1 Chat）。

图只负责路由/并行/合并/综合；specialist 与最终生成仍复用现有原生实现。
所有 LangGraph import 都在 `build_multi_agent_graph` 内，缺依赖不影响启动。
"""

from __future__ import annotations

import logging
import threading
import time

from src.agent.orchestration.merge import merge_results
from src.agent.orchestration.prompts import SYNTHESIS_SYSTEM
from src.agent.orchestration.router import AgentTask
from src.agent.orchestration.specialists import run_specialist_with_timeout
from src.agent.orchestration.state import MultiAgentState

logger = logging.getLogger(__name__)

_GRAPH_LOCK = threading.Lock()


def _emit(event: dict) -> None:
    """把节点内事件写到 LangGraph custom stream；非流式上下文静默跳过。"""
    try:
        from langgraph.config import get_stream_writer

        writer = get_stream_writer()
    except Exception:  # noqa: BLE001 - 非流式/旧版本没有 writer
        return
    try:
        writer(event)
    except Exception:  # noqa: BLE001 - 观测事件不能搞挂节点
        logger.debug("写入 custom stream 失败", exc_info=True)


def _legacy(meta: dict) -> None:
    meta["_multi_agent_needs_legacy"] = True
    meta.setdefault("orchestration", {})["fallback"] = "legacy"


def build_multi_agent_graph(agent):
    """构建并编译图；同一个 agent 复用一份编译结果。"""
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Send

    def route_node(state: MultiAgentState) -> dict:
        meta = state.get("runtime_meta") or {}
        t0 = time.monotonic()
        agent._mark_phase(meta, "问题路由", t0)
        meta["_dispatch_t0"] = time.monotonic()
        _emit({"type": "status", "content": "正在并行检索..."})
        return {}

    def route_edge(state: MultiAgentState):
        route = state.get("route") or []
        if not route:
            return "legacy"
        # ★ Send 的 payload 会**替换**该分支的 state，而不是与父 state 合并。
        # 只传 {"task": ...} 会让 question/provider/runtime_meta/trace_recorder
        # 全部丢失（实测 specialist 拿不到问题与记录器）。
        base = {key: value for key, value in state.items() if key != "task"}
        return [
            Send("run_specialist", {**base, "task": task})
            for task in route
        ]

    def legacy_node(state: MultiAgentState) -> dict:
        _legacy(state.get("runtime_meta") or {})
        return {"fallback": "legacy"}

    def specialist_node(state: MultiAgentState) -> dict:
        task = state.get("task") or {}
        meta = state.get("runtime_meta") or {}
        if not task:
            return {
                "specialist_results": [],
                "errors": ["missing specialist task"],
            }
        result = run_specialist_with_timeout(
            agent,
            AgentTask(
                agent=str(task.get("agent") or ""),
                index=int(task.get("index") or 0),
                task=str(task.get("task") or ""),
                tools=tuple(task.get("tools") or ()),
            ),
            question=str(state.get("question") or ""),
            provider=str(state.get("provider") or ""),
            web_enabled=bool(state.get("web_enabled")),
            session_id=str(state.get("session_id") or ""),
            trace_recorder=state.get("trace_recorder"),
        )
        return {"specialist_results": [result]}

    def merge_node(state: MultiAgentState) -> dict:
        meta = state.get("runtime_meta") or {}
        if meta.get("_multi_merged"):
            return {}
        meta["_multi_merged"] = True
        t0 = time.monotonic()
        dispatch_t0 = float(meta.pop("_dispatch_t0", t0) or t0)
        agent._mark_phase(meta, "并行检索", dispatch_t0)
        merged = merge_results(state.get("specialist_results") or [])
        agent._mark_phase(meta, "合并结果", t0)

        for key in (
            "sources", "web_sources", "tool_called", "tool_name",
            "tool_timings", "tool_deduped", "charts", "rag_timings",
            "succeeded_tools", "tool_output_text",
        ):
            meta[key] = merged.get(key)
        # `_build_done` 与评测读 `tool_call_names`；合并结果内部叫 tool_names。
        meta["tool_call_names"] = list(merged.get("tool_names") or [])
        meta["agent_trace"] = list(merged.get("agent_trace") or [])
        meta["agent_timings"] = list(merged.get("agent_timings") or [])
        meta["orchestration"] = {
            "enabled": True,
            "mode": str(meta.get("_multi_agent_mode") or ""),
            "router": "rules",
            "route": [
                str((task or {}).get("agent") or "")
                for task in (state.get("route") or [])
            ],
            "partial": bool(merged.get("partial")),
            "deadline_exceeded": False,
            "fallback": "none",
        }
        return {"merged": merged, "partial": bool(merged.get("partial"))}

    def synthesize_node(state: MultiAgentState) -> dict:
        meta = state.get("runtime_meta") or {}
        results = sorted(
            [r for r in (state.get("specialist_results") or []) if isinstance(r, dict)],
            key=lambda r: int(r.get("index") or 0),
        )
        findings = [
            r for r in results
            if str(r.get("answer") or "").strip()
        ]
        if not findings:
            _legacy(meta)
            return {"fallback": "legacy"}

        blocks = []
        for result in findings:
            sources = result.get("sources") or []
            urls = [
                str(s.get("url") or "")
                for s in sources[:3]
                if isinstance(s, dict) and s.get("url")
            ]
            blocks.append(
                f"【{result.get('agent')}】\n{result.get('answer')}"
                + (f"\n来源：{', '.join(urls)}" if urls else "")
            )
        messages = [
            {"role": "system", "content": SYNTHESIS_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"用户问题：{state.get('question') or ''}\n\n"
                    "专家发现：\n\n" + "\n\n".join(blocks) + "\n\n"
                    "请综合以上发现给出最终回答。"
                ),
            },
        ]
        meta["last_stream_messages"] = messages

        factory = getattr(agent, "_synthesis_client_factory", None)
        if callable(factory):
            client = factory(str(state.get("provider") or ""))
        else:
            client = agent._get_llm_client(str(state.get("provider") or ""))
        deep = bool(state.get("deep_thinking_enabled"))
        if deep:
            client = agent._thinking_model(client, str(state.get("provider") or ""))

        answer_parts: list[str] = []
        try:
            for event_type, payload in agent._generate_final(
                messages, client, meta, deep
            ):
                if event_type == "token":
                    answer_parts.append(str(payload))
                    _emit({"type": "token", "content": payload})
                elif event_type == "reset":
                    answer_parts.clear()
                    _emit({"type": "reset"})
                elif event_type in ("status", "thinking"):
                    _emit({"type": event_type, "content": payload})
        except Exception as exc:  # noqa: BLE001
            logger.warning("综合生成失败: %s", exc)
            fallback = next(
                (str(r.get("answer") or "") for r in findings if r.get("answer")),
                "",
            )
            if not fallback:
                _legacy(meta)
                return {"fallback": "legacy"}
            _emit({"type": "status", "content": "正在生成回答..."})
            _emit({"type": "token", "content": fallback})
            meta["streamed"] = True
            meta["answer_text"] = fallback
            meta.setdefault("orchestration", {})["fallback"] = "fragment"
            return {"answer": fallback}

        answer = "".join(answer_parts)
        meta["answer_text"] = answer
        return {"answer": answer}

    builder = StateGraph(MultiAgentState)
    builder.add_node("plan", route_node)
    builder.add_node("run_specialist", specialist_node)
    builder.add_node("merge", merge_node)
    builder.add_node("synthesize", synthesize_node)
    builder.add_node("legacy", legacy_node)
    builder.add_edge(START, "plan")
    builder.add_conditional_edges("plan", route_edge, ["run_specialist", "legacy"])
    builder.add_edge("run_specialist", "merge")
    builder.add_edge("merge", "synthesize")
    builder.add_edge("synthesize", END)
    builder.add_edge("legacy", END)
    return builder.compile()


def get_multi_agent_graph(agent):
    graph = getattr(agent, "_multi_agent_graph", None)
    if graph is not None:
        return graph
    with _GRAPH_LOCK:
        graph = getattr(agent, "_multi_agent_graph", None)
        if graph is None:
            graph = build_multi_agent_graph(agent)
            agent._multi_agent_graph = graph
    return graph

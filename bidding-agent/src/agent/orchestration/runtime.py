"""多 Agent 运行时：决定是否走图、桥接 LangGraph custom stream 到 SSE 事件。"""

from __future__ import annotations

import contextvars
import logging
import queue
import threading
import time
from dataclasses import dataclass

from src.agent.constants import (
    CLARIFY_MIN_QUESTION_CHARS,
    MULTI_AGENT_DEADLINE_S,
    MULTI_AGENT_IDLE_TIMEOUT_S,
    MULTI_AGENT_LLM_CONCURRENCY,
    MULTI_AGENT_MAX_LLM_CALLS,
    MULTI_AGENT_ROUTER_TIMEOUT_S,
    MULTI_AGENT_TOOL_CONCURRENCY,
)
from src.agent.llm_budget import CallBudget
from src.agent.orchestration.graph import get_multi_agent_graph
from src.agent.orchestration.llm_router import route_with_llm
from src.agent.orchestration.mode import multi_agent_mode, router_mode
from src.agent.orchestration.router import (
    DECISION_ACTIVATED,
    DECISION_OFF,
    AgentTask,
    multi_agent_decision,
    route_question,
)

logger = logging.getLogger(__name__)


class LegacyFallback(Exception):
    """图在产出任何 token 前决定/失败于走旧 ReAct；由 core 捕获并回退。"""


@dataclass(frozen=True)
class PreparedRun:
    mode: str
    tasks: tuple[AgentTask, ...]
    #: 实际使用的路由：rules / llm / llm_fallback（llm_fallback 表示尝试过 LLM）。
    router: str = "rules"
    router_reason: str = ""
    router_attempted: bool = False
    router_elapsed_ms: int = 0
    #: 多 Agent 阶段的总截止时刻（monotonic）；包含 LLM 路由耗时。
    deadline: float | None = None


def _auto_should_try_llm(question: str, rule_tasks: list[AgentTask]) -> bool:
    """`auto` 只在规则明显不够用时才付一次 LLM 调用。

    条件是：规则只给了一个专家、问题长度够、且结构上像复合问题。
    规则已经给出 ≥2 个专家时说明关键词足够明确，没必要再问 LLM。
    """
    if len(rule_tasks) != 1:
        return False
    if len(str(question or "").strip()) < CLARIFY_MIN_QUESTION_CHARS:
        return False
    try:
        from src.agent.planning import is_compound_question

        return bool(is_compound_question(question))
    except Exception:  # noqa: BLE001 - 规划辅助失败不影响路由
        return False


def prepare_multi_agent(
    question: str,
    *,
    web_enabled: bool,
    skills=None,
    mode: str | None = None,
    decision: dict | None = None,
    client=None,
    trace_recorder=None,
) -> PreparedRun | None:
    """返回 None 表示走旧 ReAct；否则返回路由结果。

    `decision` 是可选的 out-param：调用方传入一个 dict，本函数会填入
    mode/route/router/activated/decision 等字段，供 done 帧与监控区分
    "off" / "auto 被拒" / "已走图"。不传时行为与旧版本完全一致。

    `client` 仅用于 Phase 1b LLM 路由；为 None 时 LLM 路由会回落规则。
    """
    selected_mode = mode or multi_agent_mode()
    deadline = time.monotonic() + MULTI_AGENT_DEADLINE_S
    if decision is not None:
        decision.setdefault("mode", selected_mode)
        decision.setdefault("route", [])
        decision.setdefault("activated", False)
        decision.setdefault("fallback", "none")
        decision.setdefault("partial", False)
        decision.setdefault("router", "off")
        decision.setdefault("router_reason", "")
        decision.setdefault("router_attempted", False)
        decision.setdefault("router_elapsed_ms", 0)
    if selected_mode == "off":
        if decision is not None:
            decision["decision"] = DECISION_OFF
        return None

    rule_tasks = route_question(question, web_enabled=web_enabled, skills=skills)
    if decision is not None:
        decision["route"] = [task.agent for task in rule_tasks]

    router = router_mode()
    tasks: list[AgentTask] = rule_tasks
    router_used = "rules"
    router_reason = ""
    router_attempted = False
    router_elapsed_ms = 0

    use_llm = router == "llm" or (
        router == "auto" and _auto_should_try_llm(question, rule_tasks)
    )
    if use_llm:
        result = route_with_llm(
            question,
            web_enabled=web_enabled,
            client=client,
            trace_recorder=trace_recorder,
            timeout_s=MULTI_AGENT_ROUTER_TIMEOUT_S,
            deadline=deadline,
        )
        router_attempted = result.attempted
        router_elapsed_ms = result.elapsed_ms
        if result.tasks:
            tasks = list(result.tasks)
            router_used = "llm"
            router_reason = result.reason or "llm"
        else:
            # llm_fallback 表示"真的尝试过 LLM"；client_unavailable/deadline
            # 在调用前就失败，仍按规则路由记录，避免虚报 LLM 成本。
            router_used = "llm_fallback" if result.attempted else "rules"
            router_reason = result.error or "fallback"
    else:
        if router == "off":
            router_reason = "router_off"
        elif len(rule_tasks) >= 2:
            router_reason = "auto_rules_multi_expert"
        else:
            router_reason = "auto_single_non_compound"

    if decision is not None:
        decision.update(
            router=router_used,
            router_reason=router_reason,
            router_attempted=router_attempted,
            router_elapsed_ms=router_elapsed_ms,
        )
        decision["route"] = [task.agent for task in tasks]

    verdict = multi_agent_decision(question, selected_mode, tasks)
    if decision is not None:
        decision["decision"] = verdict
    if verdict != DECISION_ACTIVATED:
        return None
    if decision is not None:
        decision["activated"] = True
    return PreparedRun(
        mode=selected_mode,
        tasks=tuple(tasks),
        router=router_used,
        router_reason=router_reason,
        router_attempted=router_attempted,
        router_elapsed_ms=router_elapsed_ms,
        deadline=deadline,
    )


def _mark_deadline(meta: dict, reason: str) -> None:
    """标记多 Agent 阶段超时；可见答案已流出时不回退旧 ReAct。"""
    meta["_multi_agent_deadline_exceeded"] = True
    meta["_multi_agent_timeout_reason"] = reason
    meta["_multi_agent_abandoned"] = True
    event = meta.get("_multi_agent_cancel")
    if event is not None:
        try:
            event.set()
        except Exception:  # noqa: BLE001 - 观测/清理不能影响收尾
            pass
    orch = meta.setdefault("orchestration", {})
    orch["deadline_exceeded"] = True
    orch["partial"] = True
    if not meta.get("answer_text"):
        orch["fallback"] = "legacy"
        meta["_multi_agent_needs_legacy"] = True


def _iter_graph_events(graph, state: dict, *, meta: dict, deadline: float):
    """在线程里跑 graph.stream，主线程按 deadline/idle 消费事件。

    线程无法强杀；超时/客户端断开时设置 cancel/abandoned，让节点在下一个
    检查点退出。已在 provider 调用中的请求可能继续到 socket 超时。
    """
    q: queue.Queue = queue.Queue()

    def pump() -> None:
        try:
            for event in graph.stream(state, stream_mode="custom"):
                q.put(("event", event))
        except BaseException as exc:  # noqa: BLE001 - 交给消费线程处理
            q.put(("error", exc))
        finally:
            q.put(("end", None))

    # 复制请求上下文（request_id 等）到图线程；否则节点日志会丢关联 ID。
    ctx = contextvars.copy_context()
    threading.Thread(
        target=ctx.run, args=(pump,), name="multi-agent-graph", daemon=True
    ).start()
    while True:
        # 即使队列一直有事件（长流），也不能让总 deadline 失效。
        if time.monotonic() >= deadline:
            _mark_deadline(meta, "total")
            return
        try:
            kind, item = q.get_nowait()
        except queue.Empty:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _mark_deadline(meta, "total")
                return
            wait = min(remaining, MULTI_AGENT_IDLE_TIMEOUT_S)
            try:
                kind, item = q.get(timeout=max(0.01, wait))
            except queue.Empty:
                _mark_deadline(meta, "idle")
                return
        if kind == "end":
            return
        if kind == "error":
            raise item
        yield item


def _finalize_orchestration(meta: dict) -> None:
    """把 stream 期间才确定的预算/超时状态补进 orchestration（done 帧用）。"""
    orch = meta.setdefault("orchestration", {})
    if not isinstance(orch, dict):
        return
    orch.setdefault("router", meta.get("_multi_agent_router") or "rules")
    orch.setdefault(
        "router_reason", meta.get("_multi_agent_router_reason") or ""
    )
    budget = meta.get("_llm_budget")
    if budget is not None:
        try:
            orch.update(budget.snapshot())
        except Exception:  # noqa: BLE001 - 观测不能影响收尾
            pass
    if meta.get("_multi_agent_deadline_exceeded"):
        orch["deadline_exceeded"] = True
        orch["partial"] = True
    if meta.get("_multi_agent_timeout_reason"):
        orch["timeout_reason"] = str(meta["_multi_agent_timeout_reason"])


def _invoke_graph_bounded(graph, state: dict, *, meta: dict, deadline: float):
    """旧版 LangGraph 不支持 custom stream 时的有界 invoke。"""
    q: queue.Queue = queue.Queue()

    def pump() -> None:
        try:
            q.put(("ok", graph.invoke(state)))
        except BaseException as exc:  # noqa: BLE001
            q.put(("error", exc))

    ctx = contextvars.copy_context()
    threading.Thread(
        target=ctx.run, args=(pump,), name="multi-agent-invoke", daemon=True
    ).start()
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        _mark_deadline(meta, "total")
        return None
    try:
        kind, item = q.get(timeout=max(0.01, remaining))
    except queue.Empty:
        _mark_deadline(meta, "total")
        return None
    if kind == "error":
        raise item
    return item


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
    meta["_multi_agent_router"] = prepared.router
    meta["_multi_agent_router_reason"] = prepared.router_reason
    meta["_multi_agent_deadline"] = prepared.deadline or (
        time.monotonic() + MULTI_AGENT_DEADLINE_S
    )
    meta["_multi_agent_cancel"] = threading.Event()
    meta["_llm_budget"] = CallBudget(MULTI_AGENT_MAX_LLM_CALLS)
    meta["_llm_semaphore"] = threading.BoundedSemaphore(MULTI_AGENT_LLM_CONCURRENCY)
    meta["_tool_semaphore"] = threading.BoundedSemaphore(MULTI_AGENT_TOOL_CONCURRENCY)
    meta.setdefault("orchestration", {
        "enabled": True,
        "mode": prepared.mode,
        "router": prepared.router,
        "router_reason": prepared.router_reason,
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
        "trace_recorder": meta.get("llm_trace"),
    }
    graph = get_multi_agent_graph(agent)
    deadline = float(meta["_multi_agent_deadline"])

    custom_stream_ok = True
    yielded_any = False
    finished = False
    try:
        for event in _iter_graph_events(
            graph, state, meta=meta, deadline=deadline
        ):
            if not isinstance(event, dict):
                continue
            event_type = str(event.get("type") or "")
            if event_type not in ("status", "token", "thinking", "reset"):
                continue
            payload = None if event_type == "reset" else event.get("content")
            yielded_any = True
            yield event_type, payload
        finished = True
    except (TypeError, ValueError) as exc:
        # 旧版 LangGraph 不支持 custom stream：退化为有界 invoke 后一次性回放。
        if yielded_any:
            raise
        custom_stream_ok = False
        logger.warning("LangGraph custom stream 不可用，回退 post_graph: %s", exc)
        _invoke_graph_bounded(graph, state, meta=meta, deadline=deadline)
        finished = True
    finally:
        # 消费者提前关闭（客户端断开/调用方停止迭代）时通知图节点退出。
        if not finished and not meta.get("_multi_agent_deadline_exceeded"):
            meta["_multi_agent_abandoned"] = True
            event = meta.get("_multi_agent_cancel")
            if event is not None:
                try:
                    event.set()
                except Exception:  # noqa: BLE001 - 清理不能影响收尾
                    pass

    _finalize_orchestration(meta)

    if not custom_stream_ok and meta.get("_multi_agent_answer"):
        yield "status", "正在生成回答..."
        yield "token", meta["_multi_agent_answer"]

    # ★ 用"是否已有可见正文"判断能否回退，而不是 `streamed`：
    #   reset 之后 streamed 可能仍为 True，会把一次本可回退的失败误判为"已答"。
    if meta.get("_multi_agent_needs_legacy") and not meta.get("answer_text"):
        raise LegacyFallback

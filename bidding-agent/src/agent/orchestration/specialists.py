"""Specialist 执行：复用现有 `_run_tool_loop`，不产出最终用户答案。

每个 specialist 有独立 sub-meta、独立客户端（clone）与显式工具白名单；
这里只负责「跑工具循环 + 收敛结构化结果」，token 生成留给 synthesizer。
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError

from src.agent.constants import (
    MAX_SPECIALIST_ROUNDS,
    MULTI_AGENT_PER_AGENT_OUTPUT_CHARS,
    MULTI_AGENT_SPECIALIST_TIMEOUT_S,
)
from src.agent.orchestration.prompts import SPECIALIST_PROMPTS, SPECIALIST_SYSTEM
from src.agent.orchestration.router import AgentTask

logger = logging.getLogger(__name__)


def _last_assistant_text(messages: list, fallback: str = "") -> str:
    for msg in reversed(messages or []):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "assistant":
            text = str(msg.get("content") or "").strip()
            if text:
                return text
    return fallback


def _empty_result(task: AgentTask, status: str, error: str = "") -> dict:
    return {
        "agent": task.agent,
        "index": task.index,
        "status": status,
        "answer": "",
        "sources": [],
        "web_sources": [],
        "tool_called": False,
        "tool_name": "",
        "tool_names": [],
        "tool_timings": [],
        "tool_deduped": 0,
        "charts": [],
        "rag_timings": [],
        "succeeded_tools": [],
        "tool_output_text": [],
        "rounds": 0,
        "hit_cap": False,
        "provider": "",
        "provider_switches": 0,
        "phase_times": [],
        "elapsed_ms": 0,
        "error": error[:200],
    }


def run_specialist(
    agent,
    task: AgentTask,
    *,
    question: str,
    provider: str = "",
    web_enabled: bool = False,
    session_id: str = "",
) -> dict:
    """执行一个 specialist；任何异常都收敛为结构化结果，不向上抛。"""
    t0 = time.monotonic()
    try:
        factory = getattr(agent, "_specialist_client_factory", None)
        if callable(factory):
            client = factory(task.agent, provider)
        else:
            client = agent._get_llm_client(provider)
            if hasattr(client, "clone"):
                client = client.clone()
        meta = {
            "sources": [],
            "web_sources": [],
            "tool_called": False,
            "tool_name": "",
            "phase_times": [],
            "t_start": t0,
            "question": question,
            "retrieval_hint": question,
            "web_search_enabled": bool(web_enabled),
            "clarify_allowed": False,
            "planning_enabled": False,
            "session_id": session_id,
        }
        messages = [
            {
                "role": "system",
                "content": (
                    SPECIALIST_SYSTEM
                    + "\n\n"
                    + SPECIALIST_PROMPTS.get(task.agent, "")
                ),
            },
            {
                "role": "user",
                "content": (
                    f"用户问题：{question}\n\n"
                    f"你的子任务：只处理 {task.agent} 领域，调用允许的工具并给出结论。"
                ),
            },
        ]
        gen = agent._run_tool_loop(
            messages,
            client,
            meta,
            allowed_tools=task.tools,
            max_rounds=MAX_SPECIALIST_ROUNDS,
            agent_id=task.agent,
        )
        result = None
        while True:
            try:
                next(gen)
            except StopIteration as stop:
                result = stop.value
                break
        if result is None:
            return _empty_result(task, "error", "工具循环未返回结果")
        if result.clarification is not None:
            return _empty_result(task, "clarify")

        answer = _last_assistant_text(result.messages)
        if result.hit_cap or not answer:
            answer = "\n\n".join(result.tool_output_text or [])
        answer = answer[:MULTI_AGENT_PER_AGENT_OUTPUT_CHARS]
        status = "ok"
        if result.hit_cap:
            status = "partial"
        if not answer:
            status = "empty"

        return {
            "agent": task.agent,
            "index": task.index,
            "status": status,
            "answer": answer,
            "sources": list(result.sources),
            "web_sources": list(result.web_sources),
            "tool_called": bool(result.tool_called),
            "tool_name": str(result.tool_name or ""),
            "tool_names": list(result.tool_names),
            "tool_timings": list(result.tool_timings),
            "tool_deduped": int(meta.get("tool_deduped") or 0),
            "charts": list(result.charts),
            "rag_timings": list(result.rag_timings),
            "succeeded_tools": list(result.succeeded_tools),
            "tool_output_text": list(result.tool_output_text),
            "rounds": int(result.rounds),
            "hit_cap": bool(result.hit_cap),
            "provider": str(
                getattr(client, "last_used_provider", "")
                or getattr(client, "provider", "")
                or ""
            ),
            "provider_switches": int(getattr(client, "switch_count", 0) or 0),
            "phase_times": list(meta.get("phase_times") or []),
            "elapsed_ms": int((time.monotonic() - t0) * 1000),
            "error": "",
        }
    except Exception as exc:  # noqa: BLE001 - 单个 specialist 失败不拖垮整图
        logger.warning("specialist %s 执行失败: %s", task.agent, exc)
        result = _empty_result(task, "error", str(exc))
        result["elapsed_ms"] = int((time.monotonic() - t0) * 1000)
        return result


def run_specialist_with_timeout(
    agent,
    task: AgentTask,
    *,
    question: str,
    provider: str = "",
    web_enabled: bool = False,
    session_id: str = "",
    timeout: float = MULTI_AGENT_SPECIALIST_TIMEOUT_S,
) -> dict:
    """带墙钟超时的 specialist；超时返回 partial 结果，不阻塞其他专家。"""
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"specialist-{task.agent}")
    future = pool.submit(
        run_specialist,
        agent,
        task,
        question=question,
        provider=provider,
        web_enabled=web_enabled,
        session_id=session_id,
    )
    try:
        return future.result(timeout=timeout)
    except FutureTimeoutError:
        logger.warning("specialist %s 超时（%.1fs）", task.agent, timeout)
        result = _empty_result(task, "timeout", f"超时 {timeout:.0f}s")
        result["elapsed_ms"] = int(timeout * 1000)
        return result
    except Exception as exc:  # noqa: BLE001
        logger.warning("specialist %s 线程异常: %s", task.agent, exc)
        return _empty_result(task, "error", str(exc))
    finally:
        pool.shutdown(wait=False)

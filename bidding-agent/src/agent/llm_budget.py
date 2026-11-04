"""多 Agent 请求级的 LLM 调用预算与并发槽。

为什么需要独立对象而不是读 `llm_trace.call_count`：
`LLMTraceRecorder.summary()` 是线程安全的**读**，但它只统计**已完成**的调用。
两个 specialist 可以同时读到 `call_count < limit` 然后各自发出调用，最终超限。
预算必须是"先占位、再调用"的原子操作，故用锁保护的计数器。

所有 helper 在 meta 没有对应键时都是 no-op —— 旧 ReAct / off 路径完全不受影响。
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager


class CallBudget:
    """线程安全的调用预算；`try_acquire()` 成功才允许发出一次 LLM 调用。"""

    def __init__(self, limit: int) -> None:
        self._limit = max(0, int(limit))
        self._used = 0
        self._denied = 0
        self._lock = threading.Lock()
        self.exhausted = threading.Event()

    def try_acquire(self) -> bool:
        with self._lock:
            if self._used >= self._limit:
                self._denied += 1
                self.exhausted.set()
                return False
            self._used += 1
            return True

    def snapshot(self) -> dict:
        with self._lock:
            used, denied = self._used, self._denied
        return {
            "llm_calls_used": used,
            "llm_call_limit": self._limit,
            "llm_calls_denied": denied,
            "budget_exhausted": used >= self._limit,
        }


def try_acquire_llm_call(meta: dict) -> bool:
    """没有预算时返回 True（旧路径）；有预算时原子占位。"""
    budget = meta.get("_llm_budget") if isinstance(meta, dict) else None
    if budget is None:
        return True
    return bool(budget.try_acquire())


@contextmanager
def llm_slot(meta: dict):
    """可选的 LLM 并发槽；没有 `_llm_semaphore` 时是 no-op。"""
    sem = meta.get("_llm_semaphore") if isinstance(meta, dict) else None
    if sem is None:
        yield
        return
    sem.acquire()
    try:
        yield
    finally:
        sem.release()


def deadline_passed(meta: dict) -> bool:
    deadline = meta.get("_multi_agent_deadline") if isinstance(meta, dict) else None
    return deadline is not None and time.monotonic() >= float(deadline)


def cancel_requested(meta: dict) -> bool:
    event = meta.get("_multi_agent_cancel") if isinstance(meta, dict) else None
    return bool(event is not None and event.is_set())


__all__ = [
    "CallBudget",
    "cancel_requested",
    "deadline_passed",
    "llm_slot",
    "try_acquire_llm_call",
]

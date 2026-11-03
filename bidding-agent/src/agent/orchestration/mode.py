"""多 Agent 编排开关与 LangGraph 能力探测。

设计原则：**默认 off，缺依赖也不影响启动**。所有 LangGraph import 都在
真正构建图时发生，本模块只做 `find_spec` 探测。
"""

from __future__ import annotations

import importlib.util
import os

from src.agent.constants import AGENT_MULTI_DEFAULT


def langgraph_available() -> bool:
    """LangGraph 是否可导入；`find_spec` 不会真正加载模块。"""
    try:
        return importlib.util.find_spec("langgraph") is not None
    except Exception:  # noqa: BLE001 - 探测失败按不可用处理
        return False


def _norm(value: str | None) -> str:
    return (value or "").strip().lower()


def multi_agent_mode() -> str:
    """`off` / `auto` / `on`；非法值或缺依赖一律回落 off。"""
    raw = _norm(os.getenv("AGENT_MULTI_AGENT"))
    if raw not in ("off", "auto", "on"):
        raw = AGENT_MULTI_DEFAULT
    if raw == "off":
        return "off"
    return raw if langgraph_available() else "off"


def router_mode() -> str:
    """`off` / `auto` / `llm`；Phase 1a 只实现确定性路由。"""
    raw = _norm(os.getenv("AGENT_MULTI_ROUTER", "off"))
    return raw if raw in ("off", "auto", "llm") else "off"

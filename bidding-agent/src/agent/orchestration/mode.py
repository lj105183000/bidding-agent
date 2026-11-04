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


def multi_agent_mode_detail() -> dict[str, str]:
    """返回模式解析的完整结果：`configured` / `effective` / `reason`。

    ★ 为什么需要拆出 detail
    ----------------------
    灰度时最危险的情况是：环境里明明配了 `AGENT_MULTI_AGENT=auto`，
    但 LangGraph 缺失或环境值写错，**实际仍走 off**。只看 done 帧里
    `orchestration=null` 无法区分"配置为 off"与"配置为 auto 但没生效"。
    这里把配置值、生效值和原因一并返回，供 done 帧与监控记录。

    `effective` 与旧 `multi_agent_mode()` 完全一致（非法值/缺依赖回落 off），
    不改变任何现有行为。
    """
    raw = _norm(os.getenv("AGENT_MULTI_AGENT"))
    if raw in ("off", "auto", "on"):
        configured = raw
        reason = "off" if raw == "off" else "active"
    else:
        configured = AGENT_MULTI_DEFAULT
        # 未设置与显式写错要区分：前者是正常默认，后者是配置事故。
        reason = "off" if not raw else "invalid_env"

    if configured == "off":
        return {"configured": configured, "effective": "off", "reason": reason}
    if not langgraph_available():
        return {
            "configured": configured,
            "effective": "off",
            "reason": "langgraph_unavailable",
        }
    return {"configured": configured, "effective": configured, "reason": "active"}


def multi_agent_mode() -> str:
    """`off` / `auto` / `on`；非法值或缺依赖一律回落 off。"""
    return multi_agent_mode_detail()["effective"]


def router_mode() -> str:
    """`off` / `auto` / `llm`；Phase 1a 只实现确定性路由。"""
    raw = _norm(os.getenv("AGENT_MULTI_ROUTER", "off"))
    return raw if raw in ("off", "auto", "llm") else "off"

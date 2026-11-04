"""多 Agent 编排包（Phase 1 Chat）。

注意：不在包初始化时 import graph/runtime —— 那会把 LangGraph 变成启动硬依赖。
core 只在 `AGENT_MULTI_AGENT != off` 时按需 import runtime。
"""

from src.agent.orchestration.merge import merge_results
from src.agent.orchestration.mode import (
    langgraph_available,
    multi_agent_mode,
    multi_agent_mode_detail,
)
from src.agent.orchestration.router import (
    DECISION_ACTIVATED,
    DECISION_ERROR,
    DECISION_NO_TASKS,
    DECISION_OFF,
    DECISION_SHORT,
    DECISION_SINGLE_EXPERT,
    AgentTask,
    multi_agent_decision,
    route_question,
    should_use_multi_agent,
)

__all__ = [
    "AgentTask",
    "DECISION_ACTIVATED",
    "DECISION_ERROR",
    "DECISION_NO_TASKS",
    "DECISION_OFF",
    "DECISION_SHORT",
    "DECISION_SINGLE_EXPERT",
    "langgraph_available",
    "merge_results",
    "multi_agent_decision",
    "multi_agent_mode",
    "multi_agent_mode_detail",
    "route_question",
    "should_use_multi_agent",
]

"""多 Agent 编排包（Phase 1 Chat）。

注意：不在包初始化时 import graph/runtime —— 那会把 LangGraph 变成启动硬依赖。
core 只在 `AGENT_MULTI_AGENT != off` 时按需 import runtime。
"""

from src.agent.orchestration.merge import merge_results
from src.agent.orchestration.mode import langgraph_available, multi_agent_mode
from src.agent.orchestration.router import (
    AgentTask,
    route_question,
    should_use_multi_agent,
)

__all__ = [
    "AgentTask",
    "langgraph_available",
    "merge_results",
    "multi_agent_mode",
    "route_question",
    "should_use_multi_agent",
]

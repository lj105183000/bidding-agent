"""确定性 specialist 路由。

默认**零额外 LLM 调用**：关键词 + 技能 + `is_compound_question`。
`AGENT_MULTI_ROUTER=llm` 的 LLM 路由留到 Phase 1b，本模块只提供规则。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from src.agent.constants import (
    CLARIFY_MIN_QUESTION_CHARS,
    MAX_SPECIALISTS,
    MAX_SPECIALIST_ROUNDS,
)

#: specialist -> 工具白名单。工具名必须与 `TOOL_EXECUTORS` 一致。
SPECIALIST_TOOLS: dict[str, tuple[str, ...]] = {
    "retrieval": ("search_knowledge_base", "search_knowledge_graph"),
    "data": ("query_database",),
    "web": ("search_web", "search_web_cn", "search_exa"),
    "compliance": ("check_tender_compliance",),
}

#: 技能 -> specialist。技能命中只作为路由信号，不替代关键词。
SKILL_TO_AGENTS: dict[str, tuple[str, ...]] = {
    "bid-analysis": ("data", "retrieval"),
    "competition-landscape": ("data", "retrieval"),
    "legal-update-monitor": ("web", "retrieval"),
    "regulation-citation-audit": ("compliance", "retrieval"),
    "debug-answer": ("retrieval",),
}

COMPLIANCE_KEYWORDS = (
    "合规", "违规", "条款", "法规", "法律", "第几条", "依据", "禁止",
    "合法性", "是否符合", "法定",
)
DATA_KEYWORDS = (
    "金额", "数量", "趋势", "排行", "排名", "分布", "占比", "同比",
    "平均", "中位数", "统计", "频次", "多少家", "top", "前几", "对比",
)
GRAPH_KEYWORDS = (
    "关系", "关联", "图谱", "供应商", "采购人", "中标", "竞争",
    "谁在做", "合作", "参与",
)
WEB_KEYWORDS = (
    "最新", "今年", "新闻", "政策", "实时", "近期", "发布", "网上", "外部",
)


@dataclass(frozen=True)
class AgentTask:
    agent: str
    index: int
    task: str
    tools: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def _hit(text: str, keywords: tuple[str, ...]) -> bool:
    return any(word in text for word in keywords)


def _skill_agents(question: str, skills) -> list[str]:
    if not skills:
        return []
    try:
        from src.agent.skills import _match_skills

        matched = _match_skills(question, skills)
    except Exception:  # noqa: BLE001 - 技能是增强信号，失败不影响路由
        return []
    out: list[str] = []
    for skill in matched:
        name = str((skill or {}).get("name") or "")
        for agent in SKILL_TO_AGENTS.get(name, ()):
            if agent not in out:
                out.append(agent)
    return out


def route_question(
    question: str,
    *,
    web_enabled: bool = False,
    skills=None,
) -> list[AgentTask]:
    """按问题特征产出有序 specialist 列表；至少返回 retrieval。"""
    text = str(question or "")
    intents: list[str] = []

    def add(agent: str) -> None:
        if agent == "web" and not web_enabled:
            return
        if agent not in intents:
            intents.append(agent)

    if _hit(text, COMPLIANCE_KEYWORDS):
        add("compliance")
    if _hit(text, DATA_KEYWORDS):
        add("data")
    if _hit(text, GRAPH_KEYWORDS):
        add("retrieval")
    if web_enabled and _hit(text, WEB_KEYWORDS):
        add("web")
    for agent in _skill_agents(text, skills):
        add(agent)

    if not intents:
        intents = ["retrieval"]
    # 复合问题至少带一个通用检索；否则 specialist 可能只看到局部数据。
    try:
        from src.agent.planning import is_compound_question

        compound = is_compound_question(text)
    except Exception:  # noqa: BLE001
        compound = False
    if compound and "retrieval" not in intents:
        intents.append("retrieval")

    selected = intents[:MAX_SPECIALISTS]
    return [
        AgentTask(
            agent=agent,
            index=index,
            task=text,
            tools=SPECIALIST_TOOLS[agent],
        )
        for index, agent in enumerate(selected)
        if agent in SPECIALIST_TOOLS
    ]


#: 多 Agent 决策结果。与 done 帧的 `multi_agent.decision` 一一对应。
DECISION_OFF = "off"
DECISION_ACTIVATED = "activated"
DECISION_SHORT = "declined_short"
DECISION_SINGLE_EXPERT = "declined_single_expert"
DECISION_NO_TASKS = "declined_no_tasks"
DECISION_ERROR = "declined_error"


def multi_agent_decision(
    question: str,
    mode: str,
    tasks: list[AgentTask],
) -> str:
    """返回"为什么不走图/为什么走图"的确定性原因。

    这是 `should_use_multi_agent` 的唯一判据来源：先判 mode，再判 tasks，
    再判短问题，最后按 on/auto 分流。这样 done 帧记录的原因与实际行为
    不会各写一套逻辑。

    ★ 为什么 auto 不再看 `is_compound_question`
    ----------------------------------------
    实测 #5「公开招标和邀请招标有什么区别？」被判为复合，但路由只有
    `retrieval` 一个专家：走图没有并行收益，反而多一次路由/综合生成，
    A/B 里 latency 8s → 33.8s（含一次偶发 fallback），质量无提升。
    单专家题交给旧 ReAct 更稳、更便宜；`on` 仍无条件走图（评测/A-B 用）。
    """
    if mode not in ("auto", "on"):
        return DECISION_OFF
    if not tasks:
        return DECISION_NO_TASKS
    # 过短/疑似需要澄清的问题交给旧 ReAct，沿用现有 clarify 逻辑。
    if len(str(question or "").strip()) < CLARIFY_MIN_QUESTION_CHARS:
        return DECISION_SHORT
    if mode == "on":
        return DECISION_ACTIVATED
    # auto：只有至少两个 specialist 才值得并行；单专家一律回退旧路径。
    return DECISION_ACTIVATED if len(tasks) >= 2 else DECISION_SINGLE_EXPERT


def should_use_multi_agent(
    question: str,
    mode: str,
    tasks: list[AgentTask],
) -> bool:
    """决定是否真的走图；判据见 `multi_agent_decision`。"""
    return multi_agent_decision(question, mode, tasks) == DECISION_ACTIVATED


__all__ = [
    "AgentTask",
    "DECISION_ACTIVATED",
    "DECISION_ERROR",
    "DECISION_NO_TASKS",
    "DECISION_OFF",
    "DECISION_SHORT",
    "DECISION_SINGLE_EXPERT",
    "MAX_SPECIALIST_ROUNDS",
    "SPECIALIST_TOOLS",
    "multi_agent_decision",
    "route_question",
    "should_use_multi_agent",
]

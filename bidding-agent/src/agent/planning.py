"""Plan-and-Solve：生成前的"信息需求"规划。

★ 为什么需要它（实测依据，见 docs/28）
------------------------------------
ReAct 是**反应式**的：每一步只根据已有结果决定下一步。实测在复合问题上会打转：

    单源·知识库        1 次调用   1 种工具
    复合·三个数据源     5 次       4 种
    多跳·图谱+业务库    6 次       2 种   ← query_database 被调 4 次（参数略不同）
    达到轮次上限 4 的题：2/5

`query_database` 调 4 次是典型的"没有先想清需要什么信息，只能试探"。
规划轮把"这道题需要哪些信息、各自该用哪个工具"先定下来，让它照着走。

★ 为什么是"先规划、不执行"（而不是把工具调用也一起规划掉）
------------------------------------------------------
一并规划掉能省更多调用，但**规划轮本身可能错**：它看不到任何检索结果，
对"库里有没有"完全没有概念。若照着一个错的计划执行，连 ReAct 的纠错能力
也一起丢了。故规划只产出**指引**，仍由 ReAct 逐轮决定调什么——
计划对了少走弯路，计划错了 ReAct 还能纠正。

★ 成本
-----
规划轮是一次非流式 LLM 调用（实测 2~5 秒）。对单源简单问题是净亏，
故默认**关闭**，由 `AGENT_PLANNING` 环境变量控制，并需 A/B 数据支持才考虑设默认。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from src.logging_config import get_logger

logger = get_logger(__name__)

#: 单次规划最多接受几条信息需求。
#:
#: 上限 4 的理由：MAX_TOOL_ROUNDS 也是 4，需求多于轮次上限时后面几条必然执行不到，
#: 列出来只会让模型在"该先做哪个"上纠结。宁可让规划聚焦。
MAX_PLAN_STEPS = 4

#: 复合问题的判定阈值：**不同类疑问词**出现 ≥2 类，或显式并列/比较词出现。
#:
#: ★ 为什么要按复杂度分流（`AGENT_PLANNING=auto`）
#: -------------------------------------------
#: 规划轮是一次额外的非流式 LLM 调用（实测 2~5 秒）。实测依据（docs/28）：
#:     · 单源问题只需 1~2 次工具调用、本来就不打转 → 多付这几秒是净亏
#:     · 复合问题（5~6 次调用、2/5 撞轮次上限）才有收益
#: 故"一律开"与"一律关"都不对：要按问题本身决定。
#:
#: 判据只认**结构特征**，不做语义理解（那需要又一次 LLM 调用，与省时间矛盾）：
#:     · 不同类疑问词 ≥2："投标保证金是多少" + "什么时候退" → 两件事
#:     · 显式并列/比较："A 和 B 分别……""对比一下……"
#:     · 多个问号（≥2）
#: 这些特征在**一句话问两件事**时几乎必然出现，而单源问题很少同时命中。
_COMPOUND_MARKERS = (
    "以及", "并且", "同时", "另外", "还有", "分别", "各自",
    "对比", "比较", "区别", "差异", "哪个更", "哪一个",
)
_QUESTION_WORD_GROUPS = (
    ("多少", "上限", "比例", "标准", "限额"),      # 数值/标准类
    ("怎么", "如何", "流程", "步骤", "办理"),      # 做法类
    ("多久", "什么时候", "几天", "期限", "时限"),  # 时间类
    ("能不能", "能否", "可不可以", "是否", "允许"),  # 可行性类
    ("哪些", "有什么", "包括"),                    # 列举类
    ("谁", "哪个部门", "主体"),                    # 主体类
)


def is_compound_question(question: str) -> bool:
    """问题是否**复合**（一次问了不止一件事）→ 决定要不要跑规划轮。

    刻意保守：宁可漏判（退回不规划，与现状一致），也不要误判
    （给简单问题多花 2~5 秒，且规划出的多步会让简单问题变啰嗦）。
    """
    text = (question or "").strip()
    if not text:
        return False

    # ① 问号数量：≥2 个问号基本就是两件事
    if text.count("？") + text.count("?") >= 2:
        return True

    # ② 显式并列/比较词
    if any(marker in text for marker in _COMPOUND_MARKERS):
        return True

    # ③ 不同类疑问词 ≥2 类：覆盖"一句话问两件事但只用一个问号"的写法
    groups_hit = sum(1 for group in _QUESTION_WORD_GROUPS
                     if any(word in text for word in group))
    return groups_hit >= 2


PLANNER_PROMPT = """你是招投标采购领域的检索规划助手。

请分析用户问题，列出**回答它需要哪些信息**，以及每条信息该用哪个工具获取。

可用工具：
- search_knowledge_base：招投标问答知识库（政策法规、办理流程、常见问题）
- search_knowledge_graph：企业与标的物、采购单位、供应商之间的关系
- query_database：结构化业务数据（中标金额、交易频次、地区分布、时间趋势）
- search_web_cn：中文联网搜索（国内招标公告、政府采购网、地方政策通知，带发布时间）
- search_web：英文联网搜索
- search_exa：联网语义搜索

输出要求（严格遵守）：
1. 只输出一个 JSON 对象，不要任何解释文字、不要 markdown 代码块。
2. 格式：{"needs": [{"info": "需要什么信息", "tool": "工具名", "query": "建议的检索词"}]}
3. 最多 4 条。问题简单时 1 条就够，不要为凑数而拆分。
4. 时效性问题（最新公告、法规更新）必须用联网工具。
5. 若问题只需一种信息，就只列一条。

用户问题："""


@dataclass
class PlanStep:
    """一条信息需求。"""

    info: str
    tool: str
    query: str = ""


@dataclass
class Plan:
    """规划结果。"""

    steps: list[PlanStep] = field(default_factory=list)
    #: 规划是否成功产出（解析失败时为 False，但仍可带空 steps）
    ok: bool = False
    #: 失败/降级原因，写入日志与 meta 便于排查
    reason: str = ""

    def tools(self) -> list[str]:
        """计划里用到的工具名（去重保序）。"""
        out: list[str] = []
        for step in self.steps:
            if step.tool and step.tool not in out:
                out.append(step.tool)
        return out

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "steps": [
                {"info": s.info, "tool": s.tool, "query": s.query} for s in self.steps
            ],
        }


def _extract_json(text: str) -> dict | None:
    """从模型输出里抠出 JSON 对象。

    ★ 为什么要容错：即便提示词明确要求"只输出 JSON"，实测模型仍会
    包上 markdown 代码块、或在前后加一句"好的，分析如下"。
    不做容错就等于这个功能随机失效——而失效是静默的（计划为空，
    ReAct 退化成原样），很难发现。
    """
    if not text:
        return None
    candidate = text.strip()
    # 去掉 ```json ... ``` 围栏
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, re.S)
    if fence:
        candidate = fence.group(1)
    else:
        # 取第一个 { 到最后一个 } 之间——容忍前后的解释文字
        start, end = candidate.find("{"), candidate.rfind("}")
        if start >= 0 and end > start:
            candidate = candidate[start : end + 1]
    try:
        parsed = json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def parse_plan(text: str, allowed_tools: set[str] | None = None) -> Plan:
    """把模型输出解析成 `Plan`。**永不抛异常**。

    `allowed_tools` 非空时过滤掉不在名单里的工具——计划里出现未暴露的工具
    （如用户没开联网却规划了 search_web）会让模型去调一个不存在的工具，
    白浪费一轮。这与工具白名单门禁是同一道防线，只是提前到规划阶段。
    """
    payload = _extract_json(text)
    if payload is None:
        return Plan(ok=False, reason="输出不是合法 JSON")

    raw_steps = payload.get("needs")
    if not isinstance(raw_steps, list) or not raw_steps:
        return Plan(ok=False, reason="needs 为空或不是列表")

    steps: list[PlanStep] = []
    dropped: list[str] = []
    for item in raw_steps[:MAX_PLAN_STEPS]:
        if not isinstance(item, dict):
            continue
        info = str(item.get("info") or "").strip()
        tool = str(item.get("tool") or "").strip()
        query = str(item.get("query") or "").strip()
        if not info:
            continue
        if allowed_tools is not None and tool and tool not in allowed_tools:
            dropped.append(tool)
            continue
        steps.append(PlanStep(info=info, tool=tool, query=query))

    if not steps:
        return Plan(ok=False, reason="没有可用的信息需求（可能工具全被过滤）")

    reason = f"已过滤未暴露的工具：{sorted(set(dropped))}" if dropped else ""
    return Plan(steps=steps, ok=True, reason=reason)


def build_plan_note(plan: Plan) -> str:
    """把计划渲染成注入 system 的指引文本。

    措辞要点：**用"建议"而不是"必须"**。计划看不到任何检索结果，
    对"库里有没有"没有概念；用命令式会压制 ReAct 的纠错能力——
    那正是它相对 Plan-and-Solve 的优势。
    """
    if not plan.ok or not plan.steps:
        return ""
    lines = ["【建议的检索计划】先按下面的信息需求取数，再回答："]
    for i, step in enumerate(plan.steps, 1):
        piece = f"{i}. {step.info}"
        if step.tool:
            piece += f"（建议工具：{step.tool}"
            if step.query:
                piece += f"，检索词：{step.query}"
            piece += "）"
        lines.append(piece)
    lines.append(
        "以上是建议而非强制：若检索结果显示方向不对，或某条信息实际取不到，"
        "请按实际情况调整，不要为了对齐计划而硬凑。"
    )
    return "\n".join(lines)


__all__ = [
    "MAX_PLAN_STEPS",
    "PLANNER_PROMPT",
    "Plan",
    "PlanStep",
    "build_plan_note",
    "parse_plan",
]

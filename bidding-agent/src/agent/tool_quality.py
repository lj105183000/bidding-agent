"""工具结果质量校验：检查执行结果是否可用，不达标时给出换工具建议或告知用户。

★ 为什么需要这一层
----------------
工具"执行成功"不等于"结果有用"。现有 `_execute_tool` 的 `ok` 只表示
**没有抛异常、没超时**：

    知识库检索返回 0 条            → ok=True，模型拿到"未找到相关内容"就懵了
    检索到 5 条但余弦只有 0.31     → ok=True，实际全是噪音（无关问题实测 0.284~0.421）
    工具未暴露（如关闭联网）        → ok=False，但模型不知道该换哪个

前两种会让模型**基于空洞或噪音继续作答**，退化成一个看起来很确定的错误答案。
第三种会让模型**重试同一个工具**白耗轮次。

★ 状态为什么要分这么细
--------------------
不同状态的**正确应对完全不同**，混在一起就只会得到"一切正常"的假象：

    ok            正常用
    empty         查到了、确实没有 → 换数据源或如实说"没有"
    weak          查到了但不相关   → 换数据源或换问法（**不能**当依据用）
    unavailable   服务/后端不可用   → 换数据源
    misconfigured 缺密钥          → 换数据源（换问法没用）

把 `empty` 与 `unavailable` 混为一谈是常见错误：前者换问法可能有用，
后者换问法一定没用。

★ `score` 与 `relevance` 不能混用（这是本项目已验证的结论）
-------------------------------------------------------
    relevance  与查询的稠密余弦相似度。实测相关问 0.67~0.71、无关问 0.29~0.34，
               **可以**判断"这条到底相不相关"。
    score      RRF 融合分。**任何查询都会有第一名**，无关问题的分数甚至可能
               高于相关问题，故**不能**用来判断相关性。

联网结果的 `score_kind` 还要区分口径：`cosine`（本地补算，与知识库同量纲可比）
与 `engine`（引擎自带，口径不同）。阈值只对 `cosine` 有意义。

见 `src/rag/vector_store.py` 的 `hybrid_search` docstring 与 `docs/39`。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ResultStatus(Enum):
    """工具结果的质量状态。"""

    OK = "ok"
    EMPTY = "empty"                  # 查到了、确实没有
    WEAK = "weak"                    # 有结果但相关性不足
    UNAVAILABLE = "unavailable"      # 服务/后端不可用
    MISCONFIGURED = "misconfigured"  # 缺密钥/未就绪
    REJECTED = "rejected"            # 工具本轮不可用（白名单外）


#: 相关性下限。低于此值的结果"存在但不可作为依据"。
#:
#: 取值依据（`pipeline.search` 的实测分布，见其 docstring）：
#:     完全无关问题 top1      0.284 ~ 0.421
#:     同领域但库内缺失       0.585 ~ 0.598
#:     评测题（都应有答案）   0.892 ~ 0.987
#: 库自身的 `MIN_RELEVANCE` 取 0.5（挡掉完全无关）。
#: 这里取 **0.45** 略低于它——检索层已经用 0.5 过滤过一轮，
#: 校验层再卡同样的线只会重复；0.45 的作用是抓"刚好擦过检索层阈值"的那批，
#: 以及**联网结果**（不走检索层过滤）。
DEFAULT_MIN_RELEVANCE = 0.45

#: 结果条数下限。1 条是"能看"，但结论孤证；0 条才是真的空。
MIN_SOURCES = 1

#: 判定"空结果"的文案特征。
_EMPTY_PATTERNS = (
    re.compile(r"未找到"),
    re.compile(r"没有搜到"),
    re.compile(r"没有返回结果"),
    re.compile(r"暂未找到"),
)

#: 判定"缺配置/未就绪"的文案特征（换问法无用，只能换数据源）。
_MISCONFIGURED_PATTERNS = (
    re.compile(r"未配置"),
    re.compile(r"暂未就绪"),
    re.compile(r"尚未就绪"),
)

#: 判定"服务不可用"的文案特征。
_UNAVAILABLE_PATTERNS = (
    re.compile(r"暂时不可用"),
    re.compile(r"暂不可用"),
    re.compile(r"执行失败"),
    re.compile(r"执行超时"),
    re.compile(r"请稍后重试"),
    re.compile(r"不可用，请稍后"),
)

#: 工具 → 可替代的数据源（**按优先级排序**）。
#:
#: ★ 为什么需要替代建议
#: -----------------
#: 只告诉模型"这个结果不行"，它会**重试同一个工具**（实测过：打转 4~5 次）。
#: 给出具体的替代工具名，它才知道该换哪个。
#:
#: ★ 替代关系只列"信息类型相近"的
#: ---------------------------
#: 知识库（问答对）/ 知识图谱（实体关系）/ 业务库（结构化中标记录）/
#: 联网（时效性信息）——这四类互相可替代，因为都能回答"这个事实是什么"。
#: 而 `check_tender_compliance`（合规检查）是**加工**而非检索，
#: 不能替代数据源，故不在表里（它失败时应换数据源取依据，由模型自行判断）。
ALTERNATIVES: dict[str, tuple[str, ...]] = {
    "search_knowledge_base": (
        "search_knowledge_graph",
        "query_database",
        "search_web_cn",
        "search_web",
    ),
    "search_knowledge_graph": (
        "search_knowledge_base",
        "query_database",
    ),
    "query_database": (
        "search_knowledge_base",
        "search_knowledge_graph",
    ),
    "search_web": (
        "search_web_cn",
        "search_knowledge_base",
        "search_exa",
    ),
    "search_web_cn": (
        "search_web",
        "search_knowledge_base",
        "search_exa",
    ),
    "search_exa": (
        "search_web",
        "search_web_cn",
        "search_knowledge_base",
    ),
    "check_tender_compliance": (
        "search_knowledge_base",
        "search_web_cn",
    ),
}


@dataclass
class QualityVerdict:
    """一次工具结果的质量判定。"""

    status: ResultStatus
    source_count: int = 0
    best_relevance: float | None = None
    reasons: list[str] = field(default_factory=list)
    #: 建议改用的工具名（**已确认在本轮允许集合内**）；空表示没有可替代的
    suggested: tuple[str, ...] = ()

    @property
    def usable(self) -> bool:
        """结果能否作为作答依据。

        `WEAK` 也算可用——有总比没有强，只是**要告诉模型它弱**，
        让它要么补充检索、要么在回答里说明依据不足。
        直接丢弃会让"勉强能答"变成"答不出来"，那是更差的取舍。
        """
        return self.status in (ResultStatus.OK, ResultStatus.WEAK)

    @property
    def on_topic_unknown(self) -> bool:
        """有结果但**判断不了**它切不切题。

        ★ 为什么必须与 `WEAK` 分开（诚实性）
        ---------------------------------
        `WEAK` 的含义是"**测出来**低于阈值"（相关 0.31 < 0.45），这是有证据的。
        而"没有可用的相关性分数"（如联网结果的 `score_kind == "engine"`，
        口径与余弦不同不可横比）是**测不了**——把它也说成 `WEAK`
        等于**把"不知道"表述成"不好"**，是编造结论。

        两者的正确提示也不同：
            已知弱 → "可作参考但不能作唯一依据，建议换更具体的说法"
            未知   → "无法确认是否切题，请自行判断或补充检索"
        """
        return self.status is ResultStatus.WEAK and self.best_relevance is None


def _first_match(text: str, patterns) -> bool:
    return any(p.search(text) for p in patterns)


def _has_usable_sources(sources: list, min_relevance: float) -> tuple[bool, float | None]:
    """结果里有没有"相关性达标"的条目。

    只对**同量纲**的分数做阈值判断：
        relevance 字段（稠密余弦）        → 可比，判阈值
        score_kind == "cosine" 的 score   → 本地补算，与知识库同量纲，判阈值
        score_kind == "engine" 的 score   → 引擎自带口径，**不判**（横比会误判）
    """
    best: float | None = None
    for item in sources or []:
        if not isinstance(item, dict):
            continue
        value = item.get("relevance")
        if not isinstance(value, (int, float)):
            if item.get("score_kind") == "cosine":
                value = item.get("score")
            else:
                value = None
        if isinstance(value, (int, float)):
            best = value if best is None else max(best, float(value))
    return (best is not None and best >= min_relevance), best


def assess_tool_result(
    name: str,
    text: str,
    sources: list | None,
    ok: bool,
    allowed: set | None = None,
    min_relevance: float = DEFAULT_MIN_RELEVANCE,
) -> QualityVerdict:
    """判定一次工具结果的质量。

    `allowed` 非空时，替代建议只从允许集合里挑——
    **这是必须的**：关闭联网时建议"改用 search_web"，模型会去调它，
    然后被白名单拒绝，白白浪费一个轮次（实测过这种"建议了却不能做"的来回）。
    """
    verdict = QualityVerdict(status=ResultStatus.OK)
    text = text or ""
    sources = sources if isinstance(sources, list) else []
    verdict.source_count = len(sources)
    allowed_set = set(allowed) if allowed else None

    # --- 状态判定（顺序重要：先看硬失败，再看内容） ---

    if not ok:
        # 白名单拒绝与真正的失败要分开：前者是"本轮不允许"，后者是"调了但坏了"
        if "本轮不可用" in text:
            verdict.status = ResultStatus.REJECTED
            verdict.reasons.append("该工具本轮未暴露（可能因关闭联网或配置未启用）")
        elif _first_match(text, _MISCONFIGURED_PATTERNS):
            verdict.status = ResultStatus.MISCONFIGURED
            verdict.reasons.append("工具缺配置或未就绪")
        else:
            verdict.status = ResultStatus.UNAVAILABLE
            verdict.reasons.append("工具执行失败或超时")
    elif _first_match(text, _MISCONFIGURED_PATTERNS):
        # ok=True 但文案是"未配置"——部分工具用成功返回表达"没密钥"
        verdict.status = ResultStatus.MISCONFIGURED
        verdict.reasons.append("工具缺配置或未就绪")
    elif _first_match(text, _UNAVAILABLE_PATTERNS):
        verdict.status = ResultStatus.UNAVAILABLE
        verdict.reasons.append("后端服务不可用")
    elif len(sources) < MIN_SOURCES:
        # 有文案但没来源：可能是"未找到"（empty），也可能是纯文本结果
        if _first_match(text, _EMPTY_PATTERNS):
            verdict.status = ResultStatus.EMPTY
            verdict.reasons.append("未检索到任何结果")
        elif len(text.strip()) < 10:
            verdict.status = ResultStatus.EMPTY
            verdict.reasons.append("返回内容过短，视为无有效结果")
        # 否则：无 sources 但有实质文本（如合规检查的风险清单）→ 保持 OK

    # --- 相关性判定（仅在有来源时） ---

    if verdict.status is ResultStatus.OK and sources:
        good, best = _has_usable_sources(sources, min_relevance)
        verdict.best_relevance = best
        if not good:
            verdict.status = ResultStatus.WEAK
            if best is None:
                verdict.reasons.append("结果没有可用的相关性分数，无法确认切题")
            else:
                verdict.reasons.append(
                    f"最高相关性 {best:.3f} 低于阈值 {min_relevance:.2f}，可能不切题"
                )

    # --- 替代建议 ---

    if not verdict.usable:
        candidates = ALTERNATIVES.get(name, ())
        if allowed_set is not None:
            candidates = tuple(c for c in candidates if c in allowed_set)
        # 不把自己列为替代
        verdict.suggested = tuple(c for c in candidates if c != name)

    return verdict


def format_spin_hint(repeat_count: int, tool_name: str, suggested: tuple[str, ...]) -> str:
    """对"同一个工具被反复调用"给出停止提示。

    ★ 为什么单靠结果质量判定不够（实测发现的缺口）
    -------------------------------------------
    实测一个问题（"电子招投标系统中投标文件的数字证书加密算法应采用哪一种
    国家标准？"）触发了 **6 次搜索，全是 `search_knowledge_base`**，
    而每次返回的来源**主题相关、相关性也不低**（都是电子招投标/数字证书），
    故质量判定是 `ok`、不产生任何提示——**模型就一直在同一个坑里打转**。

    真正的问题不是"结果质量差"，而是"**结果切题但不回答问题**"：
    知识库讲的是电子招投标的一般规定，而用户问的是一个具体的加密算法标准。
    这种"同主题不同颗粒度"的落差，任何基于相关性分数的判据都测不出来
    （分数确实不低）。

    能测出来的是**行为模式**：同一个工具、同一个问题被反复调用 N 次而
    结论没有推进。这个信号是确定的、零成本的，且直接指向正确的应对
    ——停止检索、换数据源、或如实说明"库内没有这个颗粒度的信息"。
    """
    names = "、".join(suggested[:3]) if suggested else ""
    switch_line = (
        f"请改用其他数据源（如 {names}），或" if names
        else "请"
    )
    return (
        f"\n\n【检索停滞提示】{tool_name} 已被调用 {repeat_count} 次而结论没有推进。"
        f"**不要再用相同或相近的问法调用它**——重复调用不会得到新信息。"
        f"{switch_line}直接依据已有信息作答，并**如实说明**"
        "「本地知识库中没有该颗粒度的具体规定」，不要用同主题的泛泛规定来充数。"
    )


def format_hint(verdict: QualityVerdict, tool_name: str) -> str:
    """把判定转成**给模型看**的提示文本（追加在工具结果后）。

    ★ 为什么拼进工具结果而不是单独发一条消息
    ------------------------------------
    工具结果就是模型下一轮的输入。把提示拼在同一处，
    模型才会**在同一次决策里**看到"这个结果不行、可以换 X"。
    单独发消息要等下一轮，而工具结果与提示分离时模型容易只看结果。

    ★ 有替代方案时**不告诉用户**
    -------------------------
    用户不需要知道"第一个数据源空了、系统换了一个"——那是内部编排。
    只有在**没有可替代工具**时才提示用户，因为那时模型只能如实说明依据不足。
    """
    if verdict.usable:
        if verdict.status is ResultStatus.WEAK:
            if verdict.on_topic_unknown:
                # ★ "测不了"不能说成"测出来不好"（诚实性）
                return (
                    "\n\n【数据质量提示】该结果**没有可用于判断切题程度的相关性分数**"
                    "（引擎自带的分数口径与本地余弦不同，不可横比）。"
                    "请自行判断是否切题；若不确定，建议换更具体的说法再检索一次，"
                    "或在回答中说明该部分依据的可信度未经校验。"
                )
            why = "；".join(verdict.reasons) or "相关性偏低"
            return (
                f"\n\n【数据质量提示】{why}。"
                "该结果**可作为参考但不能作为唯一依据**："
                "如仍需该主题，请换更具体的说法重试，或在回答中明确说明依据不足。"
            )
        return ""

    why = "；".join(verdict.reasons) or "结果不可用"
    if verdict.suggested:
        names = "、".join(verdict.suggested[:3])
        return (
            f"\n\n【数据质量提示】{why}。"
            f"**请改用其他数据源**（如 {names}）重新检索，不要重复调用 {tool_name}。"
        )
    return (
        f"\n\n【数据质量提示】{why}，且本轮没有其他可替代的数据源。"
        "请依据已有信息作答，并**如实说明该部分依据不足**，不要凭推测编造。"
    )


def auto_switch_hint(verdict: QualityVerdict, tool_name: str) -> str:
    """给**用户**看的说明（仅在无替代工具时需要提示）。

    与 `format_hint` 的分工：那个给模型看（含内部工具名，用户不关心），
    这个给用户看（不含工具名，只说"这部分依据不足"）。
    """
    if verdict.suggested or verdict.usable:
        return ""
    return "部分数据源暂不可用，相关结论的依据可能不完整。"

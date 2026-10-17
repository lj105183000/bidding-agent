"""评估指标注册表与报告模型。

★ 为什么要有"注册表"而不是各脚本各打各的
------------------------------------
现有三条评测线（`agent_eval` / `retrieval_eval` / `ragas_eval`）各自 print，
结果是**没法比较、没法追踪、没法门禁**：

    · 想回答"这次改动有没有让某个指标退化"→ 只能人眼比对两段终端输出
    · 想回答"上周到现在趋势如何"→ 没有任何历史
    · 想在 CI 里自动拦住退化 → 无从下手

注册表把"指标叫什么、什么方向算好、门槛多少"集中定义一次，
统一运行器与门禁都读它。散落在各脚本里迟早漂移。

★ 三态语义（关键设计）
------------------
每个指标的结果是**三态**，不是布尔：

    PASS    达到门槛
    FAIL    没达到
    NOT_RUN 这轮没测（如没配 RAGAS 依赖、或没跑那条线）

把 NOT_RUN 当成 FAIL 会让"没测"变成"退化"，误报退化；
把它当成 PASS 则会让真正的退化被掩盖。
**必须与 PASS/FAIL 并列成第三种状态**，报告里也如实显示"未测"。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum

#: 报告格式版本。字段语义变化时递增，便于趋势文件按版本解析。
SCHEMA_VERSION = 1


class MetricStatus(str, Enum):
    """指标状态。继承 str 便于 JSON 序列化与直接比较。"""

    PASS = "pass"
    FAIL = "fail"
    NOT_RUN = "not_run"


class Direction(str, Enum):
    """指标方向：越高越好 / 越低越好。

    门禁必须知道方向才能判退化——把"延迟升高"当成"变好"是最容易犯的错。
    """

    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"


@dataclass(frozen=True)
class MetricSpec:
    """一个指标的定义。"""

    name: str
    #: 中文展示名（报告用）
    label: str
    #: 指标方向
    direction: Direction
    #: 门槛。含义随方向：HIB 时是"最低要求"，LIB 时是"最高容忍"。
    #: None 表示**只记录不判定**（观测指标，如答案平均长度）——
    #: 有些量有意义但不该设阈值（设了只会频繁误报）。
    threshold: float | None = None
    #: 单位，用于展示
    unit: str = ""
    #: 所属评测线
    suite: str = ""
    #: 为什么是这个门槛（留痕，便于日后重标定）
    rationale: str = ""
    #: 这个门槛是在**几次实测**上标定的。
    #:
    #: ★ 为什么要记（实测教训）：新增的检索排序指标（首位切题率 0.9444、
    #: 首个切题位次 1.06）当时**只跑了一遍**就定了门槛 0.80 / 1.6 ——
    #: 而一次观测不足以区分"正常波动"与"退化"：门槛可能过松（挡不住退化）
    #: 也可能过紧（天天误报）。记下样本数，报告里就能提示"这个门槛还是临时的"，
    #: 标定工具（`eval/calibrate.py`）也会要求至少 5 次同条件观测才给建议值。
    #: 0 表示"未记录"（本项目大部分门槛是从若干次探针里取的，没有严格留痕）。
    calibrated_on_runs: int = 0

    def judge(self, value: float | None) -> MetricStatus:
        """按方向与门槛判定状态。`value=None` 一律视为未测。"""
        if value is None:
            return MetricStatus.NOT_RUN
        if self.threshold is None:
            # 只记录的观测指标：只要有值就算 PASS（无判定标准）
            return MetricStatus.PASS
        if self.direction is Direction.HIGHER_IS_BETTER:
            return MetricStatus.PASS if value >= self.threshold else MetricStatus.FAIL
        return MetricStatus.PASS if value <= self.threshold else MetricStatus.FAIL


@dataclass
class MetricResult:
    """一个指标的实测结果。"""

    name: str
    value: float | None
    status: MetricStatus = MetricStatus.NOT_RUN
    #: 门槛（从 spec 抄一份进报告，便于离线阅读时不查注册表）
    threshold: float | None = None
    #: 备注：未测原因、或失败时的说明
    note: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d


# ===========================================================================
# 指标注册表
# ===========================================================================

#: 门槛取值依据一律留痕。**这些数字会随项目演进需要重标定**——
#: 留痕是为了日后能判断"当初为什么定这个数"。
METRIC_SPECS: dict[str, MetricSpec] = {
    # ---- 检索线：排序质量（补 precision 不看顺序的盲区）----
    "retrieval.top1_relevant_rate": MetricSpec(
        name="retrieval.top1_relevant_rate",
        label="首位切题率",
        direction=Direction.HIGHER_IS_BETTER,
        threshold=0.80,
        unit="ratio",
        suite="retrieval",
        calibrated_on_runs=1,
        rationale=(
            "RAG 的上下文窗口有限，**第一条切不切题**直接决定答案质量，"
            "而 `retrieval.precision` 完全不看顺序（把最相关的排到第 5 位，"
            "precision 一模一样）。门槛 0.80：允许 1/5 的主题首条不切题。"
            "★ **门槛是临时的**：只在一次实测（0.9444）上定过，"
            "而 0.80 比那次实测低 0.14 —— 挡退化能力有限。"
            "待同配置跑满 5 次后用 `python -m eval.calibrate` 重标定。"
        ),
    ),
    "retrieval.first_relevant_rank_mean": MetricSpec(
        name="retrieval.first_relevant_rank_mean",
        label="首个切题位次均值",
        direction=Direction.LOWER_IS_BETTER,
        threshold=1.6,
        unit="位",
        suite="retrieval",
        calibrated_on_runs=1,
        rationale=(
            "第一个切题文档平均排在第几位。门槛 1.6 = 大体上前两条内出现。"
            "★ 与首位切题率互补：后者是 0/1 判定，这个看**位次分布** —— "
            "两个指标都过而它们同时变差时，说明排序在整体劣化。"
            "★ 只在**前 5 条内找到**切题文档的主题上取均值；"
            "前 5 条全不切题的主题不进分母（它们由 `"
            "retrieval.no_relevant_in_topk_rate` 与 note 同时报出）。"
            "★ **门槛是临时的**：只在一次实测（1.06）上定过，"
            "待同配置跑满 5 次后用 `python -m eval.calibrate` 重标定。"
        ),
    ),
    "retrieval.no_relevant_in_topk_rate": MetricSpec(
        name="retrieval.no_relevant_in_topk_rate",
        label="前5条全不切题比例",
        direction=Direction.LOWER_IS_BETTER,
        threshold=0.20,
        unit="ratio",
        suite="retrieval",
        calibrated_on_runs=1,
        rationale=(
            "★ **补一个分母陷阱**：`retrieval.first_relevant_rank_mean` 的分母是"
            "「前 5 条内**有**切题文档的主题」，于是「前 5 条全不切题」的主题"
            "**完全不拉高那个均值** —— 100 个主题里 90 个一条都没切题时，"
            "只要剩下 10 个的首位都命中，位次均值照样是 1.0，看起来完美。"
            "把这种情形单列成比率，与 `top1_relevant_rate` 互补"
            "（一个是「第一条对不对」、一个是「五条里有没有」）。"
            "门槛 0.20：允许 1/5 的主题在前 5 条内完全找不到切题文档。"
            "★ 与首位切题率一样，**门槛是临时的**（在首次实测上定），"
            "待同配置跑满 5 次后用 `python -m eval.calibrate` 重标定。"
        ),
    ),
    # ---- Agent 线：工具选择 ----
    "agent.tool_accuracy": MetricSpec(
        name="agent.tool_accuracy",
        label="工具选择准确率",
        direction=Direction.HIGHER_IS_BETTER,
        threshold=0.90,
        unit="ratio",
        suite="agent",
        rationale=(
            "口径是「最后一个执行完的工具与期望精确相等」。定 0.90 而非 1.0："
            "多轮 ReAct 后工具名可能与第一直觉不同（文档 §6.4 已说明），"
            "个别用例本就存在合理的第二种选法。低于 0.90 说明检索路由系统性出错。"
        ),
    ),
    "agent.latency_p95": MetricSpec(
        name="agent.latency_p95",
        label="延迟 P95",
        direction=Direction.LOWER_IS_BETTER,
        threshold=60000.0,
        unit="ms",
        suite="agent",
        rationale=(
            "实测单源题 5~32 秒、复合题 7~30 秒；联网兜底时可达 40 秒。"
            "P95 定 60 秒是「用户还能等」的上限。超过说明打转或联网超时未收敛。"
        ),
    ),
    # ---- Agent 线：过程指标（本轮新增，补"准确率全绿但产品有问题"的盲区）----
    "agent.tool_calls_mean": MetricSpec(
        name="agent.tool_calls_mean",
        label="平均工具调用次数",
        direction=Direction.LOWER_IS_BETTER,
        threshold=3.5,
        unit="次",
        suite="agent",
        rationale=(
            "实测均值 3.2 次（探针 5 题：1/2/2/5/6）。门槛 3.5 略高于实测，"
            "留出波动余量。**超过 3.5 说明普遍在试探而非直奔目标**。"
        ),
    ),
    "agent.hit_rounds_cap_rate": MetricSpec(
        name="agent.hit_rounds_cap_rate",
        label="撞轮次上限比例",
        direction=Direction.LOWER_IS_BETTER,
        threshold=0.20,
        unit="ratio",
        suite="agent",
        rationale=(
            "★ **口径已修正**（此前算错）：本指标此前用"
            "`tool_calls >= MAX_TOOL_ROUNDS` 判定，即拿**工具调用次数**与**轮次上限**比。"
            "而一轮可并行调多个工具——实测模型会在同一轮发两个**不同 query** 的"
            "`search_knowledge_base`（两路召回），于是 1 轮被记成 2 次调用。"
            "后果是**假性未达标**：历史趋势里本指标从 0 涨到 0.2353 并让门禁 FAIL，"
            "而真实 `max(rounds)` 实测全部 ≤ 2，从未触达 4 轮上限。"
            "现在按 `tool_timings[].round` 的**最大值**判定。"
            "门槛 0.20 保留：撞上限仍是打转的直接信号，只是此前统计的分子是错的。"
        ),
    ),
    "agent.rounds_mean": MetricSpec(
        name="agent.rounds_mean",
        label="平均轮次",
        direction=Direction.LOWER_IS_BETTER,
        threshold=2.5,
        unit="轮",
        suite="agent",
        rationale=(
            "★ 新增指标，补上「调用次数」与「轮次」混用造成的盲区。"
            "实测 17 题 `rounds` 分布为 {1, 2}，均值约 1.2；"
            "门槛 2.5 意味着「平均不超过 2.5 轮」，即允许个别题走两轮补充检索。"
            "**与 `tool_calls_mean` 区分看**：调用次数高但轮次低是正常的"
            "（一轮内多路召回）；两者同高才是打转。"
        ),
    ),
    "agent.redundant_calls_mean": MetricSpec(
        name="agent.redundant_calls_mean",
        label="平均重复调用次数",
        direction=Direction.LOWER_IS_BETTER,
        threshold=1.0,
        unit="次",
        suite="agent",
        rationale=(
            "重复调用 = 同一工具被调多次（参数略不同）。实测多跳题 `query_database` "
            "被调 4 次。门槛 1.0 允许偶发的一次重试，超过说明**查询构造能力不足**、"
            "在靠次数试错。"
        ),
    ),
    "agent.verification_critical_ok_rate": MetricSpec(
        name="agent.verification_critical_ok_rate",
        label="无严重编造比例",
        direction=Direction.HIGHER_IS_BETTER,
        threshold=0.95,
        unit="ratio",
        suite="agent",
        rationale=(
            "严重编造 = 答案里出现检索结果中**没有的条款号或数字**。"
            "这是招投标场景真正的高风险项：用户会照条款号与百分比核对合规性，"
            "错一个数字就可能误导决策。实测出现过 3 题编造 `10%`。"
            "门槛 0.95 而非 1.0：单题偶发一次仍可接受，但**系统性出现就是硬伤**。"
        ),
    ),
    "agent.verification_ok_rate": MetricSpec(
        name="agent.verification_ok_rate",
        label="答案核对通过率",
        direction=Direction.HIGHER_IS_BETTER,
        threshold=0.90,
        unit="ratio",
        suite="agent",
        rationale=(
            "此项含**非白名单**法规名未在检索结果中（低风险）。"
            "常识性骨架法律（《招标投标法》等）已归入 `common_laws` 不计入——"
            "实测语料里根本没有它们（检索 top5 命中 0/5），要求模型不提并不合理。"
            "**门槛 0.90**：排除常识法规后，实测稳定 1.0000，"
            "故可以卡得紧；一旦掉下来说明模型开始编造**非骨架**的法规名，"
            "那才是真正需要关注的可疑信号。"
            "（曾一度定 0.70，是在常识法规尚未排除时的临时放水，现已收紧。）"
        ),
    ),
    "agent.false_unavailable_rate": MetricSpec(
        name="agent.false_unavailable_rate",
        label="编造工具不可用比例",
        direction=Direction.LOWER_IS_BETTER,
        threshold=0.0,
        unit="ratio",
        suite="agent",
        rationale=(
            "实测出现过「联网检索工具本轮不可用」而工具成功调了 4 次。"
            "**门槛 0**：这是明确的错误陈述，用户会以为系统坏了，一次都不该有。"
        ),
    ),
    # ---- RAG 线 ----
    "rag.faithfulness": MetricSpec(
        name="rag.faithfulness",
        label="忠实度",
        direction=Direction.HIGHER_IS_BETTER,
        threshold=0.75,
        unit="score",
        suite="rag",
        rationale=(
            "RAGAS Faithfulness（答案是否忠于检索文档）。"
            "实测基线：默认提示词 + top5 + relevance 过滤 = 0.791；"
            "`prompt-strict` 0.881 但相关性掉到 0.779（数据否决，见 docs/22）。"
            "门槛 0.75 定在实测基线之下，用于**挡退化**而非追求高分。"
        ),
    ),
    "rag.answer_relevancy": MetricSpec(
        name="rag.answer_relevancy",
        label="答案相关性",
        direction=Direction.HIGHER_IS_BETTER,
        threshold=0.85,
        unit="score",
        suite="rag",
        rationale=(
            "RAGAS AnswerRelevancy（是否回答了用户问题）。实测 0.891。"
            "与忠实度是**此消彼长**关系（提示词收紧会一升一降），"
            "故两条都要设门槛，防止「优化」其中一条而牺牲另一条。"
        ),
    ),
    "rag.context_recall": MetricSpec(
        name="rag.context_recall",
        label="上下文召回率",
        direction=Direction.HIGHER_IS_BETTER,
        threshold=0.95,
        unit="score",
        suite="rag",
        rationale=(
            "实测 1.000，且 A/B 显示 top3/top5/top8 全为 1.000——"
            "**检索已达天花板**，提升空间在语料粒度而非检索参数（见 docs/23）。"
            "门槛 0.95 用于守住这个天花板。"
        ),
    ),
    "retrieval.precision": MetricSpec(
        name="retrieval.precision",
        label="检索切题率",
        direction=Direction.HIGHER_IS_BETTER,
        threshold=0.85,
        unit="ratio",
        suite="retrieval",
        rationale=(
            "召回结果里**真正回答了该主题**的比例（top-5）。"
            "与 `rag.context_recall` 互补：后者衡量「参考信息是否在上下文中」，"
            "因评测集与知识库高度重叠而**测不出「召回了相近但不对的条目」**；"
            "本项用精确主题词 + 人工切题判据，刻意不用语义相似度"
            "（相似度会把「履约保证金」也算成「投标保证金」的切题，而那正是要抓的问题）。"
            "实测：补语料前 38%，补完后 92%。"
            "门槛 0.85 定在实测值之下、远高于补语料前的 38%——"
            "既能挡住退化，又留出正常波动余量。"
        ),
    ),
    # ---- 难例集（真实问法）：排序类改动**唯一有分辨力**的地方 ----
    #
    # 为什么必须单列一套：评测集（18 主题）与语料高度重叠，首位切题率长期
    # 0.9444、位次 1.06，**贴着天花板**，只剩 1 题的余量。用它验证排序改动，
    # 好方案与坏方案都测不出差别。难例集用真实问法（不含语料原话），
    # 30 题实测 0.6667 / 1.40 / 0.1667 —— 这才是线上体验的真实水平。
    "retrieval.hard_top1_relevant_rate": MetricSpec(
        name="retrieval.hard_top1_relevant_rate",
        label="难例首位切题率",
        direction=Direction.HIGHER_IS_BETTER,
        threshold=0.60,
        unit="ratio",
        suite="retrieval",
        calibrated_on_runs=1,
        rationale=(
            "**真实问法**（同义改写/间接描述/术语替换/口语化/跨条目）下首条是否切题。"
            "30 题实测 **0.6667（20/30）**，比同口径的评测集（0.9444）低 28 个百分点。"
            "门槛 0.60：30 题下 1 题 = 3.3 个百分点，故留约 2 题的余量，"
            "只挡**系统性**退化（连掉两三题才会红）。"
            "★ 门槛是临时的：只在一次实测上标定过。实测两个定向修正"
            "（dense#1 保送、sentence 权重 0.7→0.85）各只挽回 1 题 = **噪声**，"
            "说明当前规模测不出小改进；题目扩到 100 道以上后用 `python -m eval.calibrate` 重标定。"
        ),
    ),
    "retrieval.hard_first_relevant_rank_mean": MetricSpec(
        name="retrieval.hard_first_relevant_rank_mean",
        label="难例首个切题位次",
        direction=Direction.LOWER_IS_BETTER,
        threshold=1.90,
        unit="位",
        suite="retrieval",
        calibrated_on_runs=1,
        rationale=(
            "难例集里第一个切题文档的平均位次。30 题实测 **1.40**"
            "（同口径的评测集 1.06）。门槛 1.90：高于实测约 0.5 位，允许波动；"
            "★ 与首位切题率一起看 —— 只优化「首条」而不动整体位次，"
            "说明改动只是碰巧命中了那几道题。"
        ),
    ),
    "retrieval.hard_no_relevant_in_topk_rate": MetricSpec(
        name="retrieval.hard_no_relevant_in_topk_rate",
        label="难例前5条全不切题比例",
        direction=Direction.LOWER_IS_BETTER,
        threshold=0.23,
        unit="ratio",
        suite="retrieval",
        calibrated_on_runs=1,
        rationale=(
            "难例里**召回彻底失手**的比例（前 5 条一条都不切题）。30 题实测 "
            "**0.1667（5/30）**。这条比排序更重要：排序差只是答案质量下降，"
            "召回失手则等于「库里明明有、用户却拿不到」。"
            "门槛 0.23 = 允许 7/30（留 2 题余量）。"
            "★ 5 题已逐题归因（`logs/test_hard_failure_causes.py`）："
            "3 题属**排序**（支持文档在池内、融合后位次 6/10/12，可调深度或融合救回）、"
            "1 题属**语料粒度**（知识埋在讲别的事的条目里，位次 49，只能改语料）、"
            "1 题属**未匹配**（两路都没命中）。即调检索能解决约 60%，其余要动语料。"
        ),
    ),
    # ---- 观测指标（只记录不判定）----
    "agent.answer_chars_mean": MetricSpec(
        name="agent.answer_chars_mean",
        label="平均答案长度",
        direction=Direction.HIGHER_IS_BETTER,
        threshold=None,
        unit="字",
        suite="agent",
        rationale=(
            "**故意不设门槛**：答案长短本身不代表好坏（详尽与啰嗦是两回事）。"
            "记录它是为了在退化时提供线索——若某个改动让答案突然变短，"
            "往往是内容被截断或降级了。"
        ),
    ),
}


def get_spec(name: str) -> MetricSpec | None:
    """按名取指标定义。未知指标返回 None（调用方据此记 NOT_RUN 并说明）。"""
    return METRIC_SPECS.get(name)


def make_result(name: str, value: float | None, note: str = "") -> MetricResult:
    """按注册表判定并构造结果。

    ★ 两道防御（都因实测崩溃而加）
    ---------------------------
    ① 未知指标名 → NOT_RUN 而不是静默通过（否则拼错的指标名看起来"达标"）
    ② **非数值 value → NOT_RUN**（实测踩过：`summarize()` 返回嵌套 dict，
       直接传进来会在 `value >= threshold` 处 TypeError，把整个评估打挂）。
       评估工具本身不该因为上游数据结构变化而崩——那会让人无法得到任何结论。
    """
    spec = get_spec(name)
    numeric = value if isinstance(value, (int, float)) and not isinstance(value, bool) else None
    if numeric is None:
        return MetricResult(
            name=name, value=None, status=MetricStatus.NOT_RUN, threshold=None,
            note=note or (f"值不是数值（{type(value).__name__}），记为未测" if value is not None else ""),
        )
    if spec is None:
        return MetricResult(
            name=name, value=float(numeric), status=MetricStatus.NOT_RUN,
            threshold=None, note=note or f"未注册的指标名：{name}",
        )
    return MetricResult(
        name=name,
        value=float(numeric),
        status=spec.judge(float(numeric)),
        threshold=spec.threshold,
        note=note,
    )


# ===========================================================================
# 报告模型
# ===========================================================================


@dataclass
class SuiteReport:
    """一条评测线的结果。"""

    suite: str
    #: 是否真正执行了。False 时 results 为空，原因写 note。
    ran: bool = False
    note: str = ""
    results: list[MetricResult] = field(default_factory=list)
    #: 该线的规模（用例数），便于判断"分数变化是否只是样本变了"
    case_count: int = 0

    def to_dict(self) -> dict:
        return {
            "suite": self.suite,
            "ran": self.ran,
            "note": self.note,
            "case_count": self.case_count,
            "verdict": self.verdict,
            "results": [r.to_dict() for r in self.results],
        }

    # -------- 分线结论 --------
    #
    # ★ 为什么必须补（此前**不存在**）
    #
    # 之前只有 `EvalReport.verdict` ——**一个总的 pass/fail**。
    # 于是门禁只能说"评测没过"，说不出**是哪条线没过**。
    # 而 RAG（检索/生成）与 Agent（编排）的修法完全不同 ——
    # 分不清哪条线，人就得去翻整份报告。
    #
    # 我最初把 `EvalReport.verdict` 误读成 `SuiteReport` 的，
    # 直到实测报 `AttributeError: 'SuiteReport' object has no attribute 'verdict'`
    # 才发现。故这里真正补上。

    @property
    def failed(self) -> list[MetricResult]:
        """本线未达标的指标。"""
        return [r for r in self.results if r.status is MetricStatus.FAIL]

    @property
    def not_run(self) -> list[MetricResult]:
        """本线没能测的指标（如 RAGAS 依赖缺失）。**不等于失败**。"""
        return [r for r in self.results if r.status is MetricStatus.NOT_RUN]

    @property
    def verdict(self) -> str:
        """本线结论：`pass` / `fail` / `not_run`。

        `not_run` 单独一档而不并入 `pass`：没跑不等于没问题
        （与 `EvalReport.verdict` 同一口径 —— `not_run` 不导致失败，
        但必须能看出来）。
        """
        if not self.ran:
            return "not_run"
        if self.failed:
            return "fail"
        # 跑了但一条指标都没测出来（如 RAGAS 全 NOT_RUN）→ 也是 not_run
        if self.results and all(
            r.status is MetricStatus.NOT_RUN for r in self.results
        ):
            return "not_run"
        return "pass"


@dataclass
class EvalReport:
    """一次完整评估的报告。这是趋势文件与门禁的输入。"""

    schema_version: int
    #: 运行标识（时间戳，UTC ISO8601），趋势文件按它排序
    run_id: str
    #: 被测版本：git 短哈希 + 脏标记；不在 git 仓库时为 "unknown"
    version: str = "unknown"
    #: 关键配置快照——**没有它，两次分数不可比**
    config: dict = field(default_factory=dict)
    suites: list[SuiteReport] = field(default_factory=list)
    #: 数据集登记与重叠核算（见 `eval/datasets.py`）。
    #:
    #: ★ 为什么结论旁边必须跟这一节：三条线用的是**三份不同**的数据
    #: （Agent 18 题 / 检索 18 主题 / RAG 25 题，Agent 与 RAG 的交集只有 12 条），
    #: 跨集分数**不可直接比**。不写出来，"RAG 0.79 比 Agent 1.00 差"这种
    #: 错误解读一定会出现。
    datasets: dict = field(default_factory=dict)

    @property
    def all_results(self) -> list[MetricResult]:
        return [r for s in self.suites for r in s.results]

    @property
    def failed(self) -> list[MetricResult]:
        return [r for r in self.all_results if r.status is MetricStatus.FAIL]

    @property
    def not_run(self) -> list[MetricResult]:
        return [r for r in self.all_results if r.status is MetricStatus.NOT_RUN]

    @property
    def verdict(self) -> str:
        """总体结论。

        `not_run` **不**导致失败——没测不等于退化。
        但报告里会单列，避免"看起来全绿"掩盖"有东西没跑"。
        """
        return "fail" if self.failed else "pass"

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "version": self.version,
            "config": self.config,
            "verdict": self.verdict,
            "failed_count": len(self.failed),
            "not_run_count": len(self.not_run),
            "suites": [s.to_dict() for s in self.suites],
            "datasets": self.datasets,
        }


def _fmt_value(value: float, spec: MetricSpec | None) -> str:
    """按指标量纲选小数位。

    ★ 为什么不能一律用 `:g`
    ---------------------
    `:g` 默认 6 位有效数字，但会把 0.9 显示成 `0.9`、把 1.0 显示成 `1`。
    实测踩过：报告里"工具选择准确率"显示成 `1ratio`、"平均调用次数"显示成
    `2次`（真值 2.0），**小数被截掉后看起来像整数**，
    读报告的人会以为指标是离散计数而不是比例。

    比例类保留 4 位（够看清 0.6667 这样的值），其余保留 2 位。
    """
    if spec is not None and spec.unit == "ratio":
        return f"{value:.4f}"
    return f"{value:.2f}"


def render_text(report: EvalReport) -> str:
    """把报告渲染成终端可读的文本。"""
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append(f"评测报告  run_id={report.run_id}")
    lines.append(f"版本 {report.version}")
    if report.config:
        cfg = "  ".join(f"{k}={v}" for k, v in sorted(report.config.items()))
        lines.append(f"配置 {cfg}")
    lines.append("=" * 78)

    for suite in report.suites:
        lines.append("")
        lines.append(f"--- {suite.suite} 线（{suite.case_count} 用例）---")
        if not suite.ran:
            lines.append(f"  ⏭  未执行：{suite.note}")
            continue
        if suite.note:
            lines.append(f"  注：{suite.note}")
        for r in suite.results:
            spec = get_spec(r.name)
            label = spec.label if spec else r.name
            if r.status is MetricStatus.NOT_RUN:
                icon, shown = "⏭ ", "未测"
            elif r.status is MetricStatus.PASS:
                icon, shown = "✅", "达标"
            else:
                icon, shown = "❌", "未达标"
            value = "—" if r.value is None else f"{_fmt_value(r.value, spec)}{spec.unit if spec else ''}"
            thr = ""
            if r.threshold is not None and spec is not None:
                op = "≥" if spec.direction is Direction.HIGHER_IS_BETTER else "≤"
                thr = f"（门槛 {op}{r.threshold:g}）"
            lines.append(f"  {icon} {label:<16} {value:>12}  {shown}{thr}")
            if r.note:
                lines.append(f"       ↳ {r.note}")
            # ★ 门槛只在少数几次实测上标定时，就地提示（见 MetricSpec.calibrated_on_runs）。
            #   放在这里而不是报告末尾：读分数的人正看着这一行，
            #   提示离数字越近越可能被读到。
            if spec is not None and 0 < spec.calibrated_on_runs < 5 and r.threshold is not None:
                lines.append(
                    f"       ⚠️ 门槛 {r.threshold:g} 仅在 {spec.calibrated_on_runs} 次实测上标定过，"
                    "属临时值（`python -m eval.calibrate` 看何时可重标定）"
                )

    lines.append("")
    lines.append("=" * 78)
    if report.verdict == "pass":
        lines.append(f"总体：✅ 通过（{len(report.not_run)} 项未测）")
    else:
        lines.append(f"总体：❌ 未通过（{len(report.failed)} 项未达标）")
        for r in report.failed:
            spec = get_spec(r.name)
            lines.append(f"  · {spec.label if spec else r.name} = {r.value:g}")
    # 数据集一节放在**结论之后**：它解释的是"这些数字是在什么上测出来的"，
    # 读完结论的人正需要这个限定条件。
    if report.datasets:
        lines.append("")
        from eval.dataset_registry import render as _render_datasets

        lines.append(_render_datasets(report.datasets))
    lines.append("=" * 78)
    return "\n".join(lines)


__all__ = [
    "METRIC_SPECS",
    "SCHEMA_VERSION",
    "Direction",
    "EvalReport",
    "MetricResult",
    "MetricSpec",
    "MetricStatus",
    "SuiteReport",
    "get_spec",
    "make_result",
    "render_text",
]

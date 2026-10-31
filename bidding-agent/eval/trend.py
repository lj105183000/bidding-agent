"""趋势存储与退化门禁。

★ 它解决什么问题
-------------
没有历史的情况下，"这次改动好不好"只能人眼比对两段终端输出——
而实测已经吃过这个亏：`docs/22` 记着一次"提示词收紧"看起来让忠实度涨了 0.111，
实际上相关性掉了 0.102 且 4 道题崩到 0.362。**单看一条指标会得出错误结论。**

故本模块做两件事：
    1. **存档**：把每次报告追加进 `eval/trend.jsonl`（一行一次运行，便于 diff）
    2. **门禁**：与最近一次比，按**指标方向**判退化，退化则返回非零退出码

★ 门禁的三条纪律
--------------
    ① **只比同一套指标**：新增指标不从历史推断（历史没测过它，比了就是假退化）
    ② **方向必须对**：延迟"升高"是退化、准确率"降低"是退化。
       搞反方向会让门禁把变好判成变坏——这是最容易犯又最难发现的错。
    ③ **容差必留**：LLM 有随机性，±2% 的正常波动不该报退化。
       不留容差的门禁会因为噪音频繁报警，久而久之被无视。

★ 为什么用 jsonl 而不是 json 数组
-----------------------------
jsonl 追加是 O(1)、且每行独立可解析——某一行写坏了不会毁掉整个历史文件。
趋势文件是**长期资产**，不能因为一次写盘中退出就全丢。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from eval.report import (
    SCHEMA_VERSION,
    Direction,
    EvalReport,
    get_spec,
)

TREND_PATH = Path(__file__).resolve().parent / "trend.jsonl"

#: 历史保留上限。超限时保留最新的一批——趋势文件不该无限增长。
MAX_HISTORY = 200


def report_to_record(report: EvalReport) -> dict:
    """把报告压成趋势文件的一行。

    只留**指标名 + 值 + 状态**，不留逐题明细：
    趋势文件要长期保存，塞满明细会让它迅速膨胀；明细另有 --json 落盘。

    `case_count` 必须留：**样本量不同的两次运行不可比**。
    实测踩过——用 `--limit 3` 与 18 题的基线比，报出 3 项"退化"，
    实际只是样本从 18 变成 3。

    ★ `suites` 也必须留（这是本书面纪律①此前**没有真正落实**的地方）
    --------------------------------------------------------
    门禁纪律①说"只比同一套指标"，但原实现只按 `schema_version` 过滤基线
    ——**没按套件过滤**。而"跑了哪些线"是个变量：

        实测历史：`20261013T015044Z ~ T041459Z` 只有 `agent`；
                 `T054351Z` 起变成 `agent + retrieval`

    于是单独跑一条线时会出现两类**假信号**：

        跑 `--suites agent`      → 基线是混合那次 → retrieval 指标
                                   `prev is None` → 被记成**"新增指标"**
        （反向）跑 `--suites retrieval` → 基线里没有 retrieval →
                                   `missing` 被塞进一堆 agent 指标

    `case_counts` 的键**碰巧**能反推套件，但那是**间接**的（样本量为 0
    的线不在里面），且语义是"用例数"不是"跑了哪条线"。
    故显式留 `suites` —— 让基线选择有**直接**依据。
    """
    return {
        "schema_version": report.schema_version,
        "run_id": report.run_id,
        "version": report.version,
        "config": report.config,
        "verdict": report.verdict,
        # ★ 只记**真正跑过**的线（`ran`）。没跑的不算 —— 否则
        #   "跳过"会被当成"跑了且没问题"，两类运行就混为一谈了。
        "suites": sorted(s.suite for s in report.suites if s.ran),
        # ★ `suites` 是实测还是回填推断的（见 `migrate_suites`）。
        #   两者都可用于选基线，但**可信度不同**：推断值在"某条线跑了
        #   但用例数为 0"时会少一条线。门禁据此提示，而不是把推断值当实测值。
        "suites_source": "observed",
        "case_counts": {s.suite: s.case_count for s in report.suites if s.ran},
        "metrics": {
            r.name: {"value": r.value, "status": r.status.value}
            for r in report.all_results
        },
        # ★ 各线的注记也要留（此前**丢了**）
        # ----------------------------------
        # 注记里有"有效 22/25（error 3，计入分母）""裁判模型 glm-5.3-flash｜
        # 裁判与生成同源"这类**解读分数可信度的前提**。趋势文件只存分数与
        # config 的话，三个月后看到 `rag.faithfulness=0.77` 根本不知道它是在
        # 多少次有效样本、哪个裁判下得到的 —— 而门禁的"退化"判定正是拿它比的。
        "suite_notes": {
            s.suite: s.note for s in report.suites if s.ran and s.note
        },
    }


def append_report(report: EvalReport, path: Path | None = None) -> None:
    """追加一次运行到趋势文件。超限时截断保留最新的。"""
    target = path or TREND_PATH
    lines: list[str] = []
    if target.exists():
        lines = [ln for ln in target.read_text(encoding="utf-8").splitlines() if ln.strip()]
    lines.append(json.dumps(report_to_record(report), ensure_ascii=False))
    if len(lines) > MAX_HISTORY:
        lines = lines[-MAX_HISTORY:]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_history(path: Path | None = None) -> list[dict]:
    """读取趋势文件。**单行解析失败只跳过该行**，不影响其余历史。

    这是选 jsonl 的直接好处：一行写坏不至于让整个趋势文件不可用。
    """
    target = path or TREND_PATH
    if not target.exists():
        return []
    records: list[dict] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            records.append(rec)
    return records


@dataclass
class Regression:
    """一条退化。"""

    name: str
    label: str
    previous: float
    current: float
    #: 相对变化（带符号，负=下降）。方向判定用它配合 spec.direction。
    delta: float
    threshold: float | None

    def describe(self) -> str:
        spec = get_spec(self.name)
        unit = spec.unit if spec else ""
        return (
            f"{self.label}：{self.previous:g}{unit} → {self.current:g}{unit}"
            f"（{self.delta:+.4g}）"
        )


@dataclass
class Comparison:
    """与历史一次比较的结果。"""

    baseline_run_id: str = ""
    baseline_version: str = ""
    #: ★ "没有基线可比"的**具体原因**（套件集合不一致 / 旧记录无 suites 字段）。
    #:
    #: 为什么要有这个字段：`missing` 被塞满时，读者会以为"指标退化了"或
    #: "指标没了"。实际常见原因是"**这次跑的线跟历史不一样**" ——
    #: 那是**没法比**，不是**变差了**。把原因说清，才能避免误判。
    no_baseline_reason: str = ""
    #: 每条线的**独立结论**（`suite → "pass" / "fail" / "not_run"`）。
    #:
    #: ★ 为什么需要它：全局只有一个 `regressed` 布尔值时，CI 只能说
    #: "评测没过"，说不出**是哪条线没过** —— 而 RAG 与 Agent 的修法完全不同
    #: （前者查检索/生成，后者查编排）。分开报才能直接指向该看的代码。
    suite_verdicts: dict[str, str] = field(default_factory=dict)
    #: 本次每条线**各自的最近一次基线**（`suite → run_id`；`""` = 该线还没有基线）。
    #:
    #: ★ 为什么按线记：基线的套件集合必须一致才能逐指标比，于是"只跑 rag"时
    #: rag 的历史记录未必存在 —— 全局一句"没有基线"说不清**是哪条线没有**。
    #: 而"rag 还没跑过基线"与"agent 有基线"是完全不同的处理：
    #: 前者要先跑一次不带 `--gate` 的存档，后者可以直接比。
    suite_baselines: dict[str, str] = field(default_factory=dict)
    #: 选中的基线的 `suites` 是实测（`observed`）还是回填推断（`inferred`）。
    baseline_suites_source: str = ""
    #: 基线记录里**疑似漏记**的线（推断的已知边界，见 `migrate_suites`）。
    baseline_inferred_gap: list[str] = field(default_factory=list)
    #: 因"一条实测值都没有"而被跳过的候选基线（run_id 列表）。
    #:
    #: ★ 为什么要有它：那种记录当基线会让门禁**静默失效**（所有指标都变成
    #: "新增指标"、永远不判退化）。跳过是对的，但必须写出来 ——
    #: 否则读报告的人会以为"比过了、没问题"。
    baseline_skipped_all_not_run: list[str] = field(default_factory=list)
    #: 选中的基线**本身**也没有实测值（所有候选都如此，无更好选择）。
    #: 此时对比结果不具判定力，`render()` 会明说。
    baseline_all_not_run: bool = False
    regressions: list[Regression] = field(default_factory=list)
    improvements: list[Regression] = field(default_factory=list)
    #: 本次没测、但历史测过的指标（如实列出，避免"看起来没退化"）
    missing: list[str] = field(default_factory=list)
    #: 新增指标（历史没有），不参与比较
    new_metrics: list[str] = field(default_factory=list)
    #: 样本量不一致的线（本次 vs 基线）。非空时**不据分数判退化**。
    case_count_mismatch: list[str] = field(default_factory=list)

    @property
    def regressed(self) -> bool:
        """是否应判定为退化。

        ★ 样本量不一致时**一律不判退化**：那种差异来自样本变了，
        不是代码退化。误报退化的代价是让人不再信任门禁。
        """
        return bool(self.regressions) and not self.case_count_mismatch

    @property
    def regressed_suites(self) -> list[str]:
        """出现退化的线（按指标归属反推），排序去重。

        用途：分线 CI（"哪条线挂了就单独红"）。`Comparison.regressed` 只是
        一个全局布尔值，拆 CI job 时需要"是哪条线"。
        """
        return sorted({suite_of(r.name) for r in self.regressions})

    @property
    def improved_suites(self) -> list[str]:
        """出现提升的线（仅用于展示，不参与判定）。"""
        return sorted({suite_of(r.name) for r in self.improvements})

    def render(self) -> str:
        lines = ["=" * 78, f"门禁对比（基线 {self.baseline_run_id} / {self.baseline_version}）", "=" * 78]
        # ★ 按线分组的结论放在最前。
        #
        # 为什么：全局只有一个结论时，读者知道"没过"但不知道**该去查哪条线** ——
        # 而 RAG（检索/生成）与 Agent（编排）的修法完全不同。
        # 分组结论直接把范围缩小到一条线。
        if self.suite_verdicts:
            icon = {"pass": "✅", "fail": "❌", "not_run": "⏭"}
            parts = [
                f"{icon.get(v, '?')} {k}（{v}）"
                for k, v in sorted(self.suite_verdicts.items())
            ]
            lines.append("分线结论：" + "  ".join(parts))
        # ★ 分线基线：回答"缺少哪条线的基线"，而不是笼统的"没有基线"。
        #   有基线时也照打 —— 读的人要知道这次比的是哪一次。
        if self.suite_baselines:
            parts = [
                f"{k} {'↔ ' + v if v else '⏳ 无基线'}"
                for k, v in sorted(self.suite_baselines.items())
            ]
            lines.append("分线基线：" + "  ".join(parts))
        # ★ 先讲"能不能比"。没有基线时下面那些"新增/未测"都是**噪音**，
        #   必须在它们之前说清原因，否则读者会以为指标出问题了。
        if self.no_baseline_reason:
            lines.append(f"⏭  没有可比基线：{self.no_baseline_reason}")
            lines.append("    （这不是退化 —— 是两次运行的**套件集合不同**，无法逐指标对比）")
            # ★ 给出**下一步该做什么**：只说"没有基线"，使用者仍要自己琢磨
            #   怎么造一个。缺基线的线在这里逐条点名并给出命令。
            missing_baseline = [s for s, rid in sorted(self.suite_baselines.items()) if not rid]
            if missing_baseline:
                lines.append(
                    "    建立基线："
                    + "；".join(
                        f"`python eval/run_all.py --suites {s} --save`" for s in missing_baseline
                    )
                    + "（存档一次，之后 --gate 才有得比）"
                )
            lines.append("=" * 78)
            return "\n".join(lines)
        if self.baseline_suites_source == "inferred":
            lines.append(
                "⚠️  该基线的 `suites` 是**回填推断**的，不是实测记录："
                "若当时某条线跑了但用例数为 0，推断会漏掉它"
                + (
                    f"（记录里标出的疑似漏项：{self.baseline_inferred_gap}）"
                    if self.baseline_inferred_gap
                    else ""
                )
            )
        if self.baseline_skipped_all_not_run:
            lines.append(
                "⚠️  已跳过这些**全为未测**的历史记录（它们当基线会让门禁静默失效）："
                + ", ".join(self.baseline_skipped_all_not_run)
            )
        if self.regressions:
            lines.append(f"❌ 退化 {len(self.regressions)} 项：")
            for r in self.regressions:
                lines.append(f"  · [{suite_of(r.name)}] {r.describe()}")
        else:
            lines.append("✅ 无退化")
        if self.improvements:
            lines.append(f"提升 {len(self.improvements)} 项：")
            for r in self.improvements:
                lines.append(f"  · [{suite_of(r.name)}] {r.describe()}")
        if self.missing:
            lines.append(f"⏭  本次未测（历史有值，不参与判定）：{', '.join(self.missing)}")
        if self.new_metrics:
            lines.append(f"＋ 新增指标（无从比较）：{', '.join(self.new_metrics)}")
        if self.case_count_mismatch:
            lines.append(
                f"⚠️  样本量不一致，分数差异**不作为退化判据**：{'；'.join(self.case_count_mismatch)}"
            )
            lines.append("    （要判退化请用与基线相同的用例数重跑）")
        # 找不到任何有实测值的基线时，最后一条提示读的人"这次比了等于没比"
        if self.baseline_all_not_run:
            lines.append(
                "⚠️  该基线**没有任何实测值**（当次运行全线未测）——本次对比不具备判定力"
            )
        lines.append("=" * 78)
        return "\n".join(lines)


def _has_value(entry: object) -> bool:
    """趋势记录里的一个指标项是否有实测值（`value` 非 None）。"""
    if not isinstance(entry, dict):
        return False
    return entry.get("value") is not None


def suite_of(metric_name: str) -> str:
    """指标名 → 所属评测线。

    优先用注册表里的 `spec.suite`（权威），取不到时退回**指标名前缀**
    （`agent.tool_accuracy` → `agent`），故未注册的指标也能归线。
    """
    spec = get_spec(metric_name)
    if spec is not None and spec.suite:
        return spec.suite
    return metric_name.split(".", 1)[0]


def latest_baseline_for(history: list[dict], suite: str, schema_version: int) -> str:
    """历史里**最近一次包含该线**的 run_id（没有则空串）。

    ★ 与基线选择**刻意用不同口径**：基线要求"套件集合完全一致"
    （否则逐指标比会产出假信号），而这里只问"这条线历史上跑过没有"。
    两者的用途不同 —— 前者用于对比，后者用于告诉使用者"缺哪条线的基线"。
    """
    for rec in reversed(history):
        if rec.get("schema_version") != schema_version:
            continue
        if not isinstance(rec.get("metrics"), dict):
            continue
        suites = rec.get("suites")
        if isinstance(suites, list) and suite in suites:
            return str(rec.get("run_id") or "")
    return ""


def all_suites(history: list[dict]) -> list[str]:
    """历史里出现过的全部评测线（用于生成 CI 矩阵、以及"有哪些线"的提示）。"""
    seen: set[str] = set()
    for rec in history:
        suites = rec.get("suites")
        if isinstance(suites, list):
            seen.update(str(s) for s in suites)
        counts = rec.get("case_counts")
        if isinstance(counts, dict):
            seen.update(str(k) for k in counts)
    return sorted(seen)


def decide_exit_code(
    report: EvalReport,
    comparison: Comparison | None = None,
    *,
    fail_on: list[str] | None = None,
    allow_fail: list[str] | None = None,
) -> tuple[int, dict]:
    """算退出码与决策明细（**纯函数**，便于单测覆盖分线红的情形）。

    退出码约定（沿用既有语义）：

        0  通过
        1  有指标未达门槛
        2  相对基线出现退化（比 1 更严重：**变差了**）

    `fail_on` / `allow_fail` 把"哪些线算数"从全局收窄到分线，
    这是"分线 CI"能成立的前提：

        `--fail-on agent`     → 只有 agent 线的未达标/退化会让 CI 红，
                                retrieval/rag 的问题照常打印但**不改退出码**
        `--allow-fail rag`    → 除 rag 外的线都算数（rag 是允许抖动的探索线时用）

    两者**互斥**（同时给会得到自相矛盾的口径），且只认已知的线名。
    被排除的线若确实未达标/退化，会在返回的 `ignored` 里点名 ——
    **静默吞掉失败是不可接受的**，必须让人看见"它红了，但我们选择不拦"。
    """
    ran = sorted(s.suite for s in report.suites if s.ran)
    if fail_on and allow_fail:
        raise ValueError("--fail-on 与 --allow-fail 不能同时使用（口径会自相矛盾）")
    known = set(ran) | set(comparison.suite_verdicts if comparison else {}) | set(all_suites([]))
    for name, values in (("--fail-on", fail_on), ("--allow-fail", allow_fail)):
        unknown = [
            v for v in (values or [])
            if v not in known
            and v not in {"agent", "retrieval", "retrieval_offline", "rag"}
        ]
        if unknown:
            raise ValueError(f"{name} 里有未知的评测线：{unknown}")

    if fail_on:
        selected = sorted(set(fail_on))
    elif allow_fail:
        selected = sorted(set(ran) - set(allow_fail))
    else:
        selected = ran

    metric_fail_suites = sorted({suite_of(r.name) for r in report.failed})
    regressed_suites = comparison.regressed_suites if (comparison and comparison.regressed) else []
    # 样本量不一致时 `regressed` 为 False，但 regressions 仍列着 —— 那种情况下
    # 不算退化（既有纪律），故这里的 regressed_suites 只在 regressed 为真时取。

    in_scope_fail = sorted(set(metric_fail_suites) & set(selected))
    in_scope_regress = sorted(set(regressed_suites) & set(selected))
    ignored: list[str] = []
    for suite in sorted(set(metric_fail_suites) | set(regressed_suites)):
        if suite not in selected:
            ignored.append(suite)

    if in_scope_regress:
        code = 2
    elif in_scope_fail:
        code = 1
    else:
        code = 0

    detail = {
        "exit_code": code,
        "ran_suites": ran,
        "selected_suites": selected,
        "metric_failed_suites": metric_fail_suites,
        "regressed_suites": regressed_suites,
        "in_scope_failed_suites": in_scope_fail,
        "in_scope_regressed_suites": in_scope_regress,
        "ignored_suites": ignored,
        "fail_on": fail_on or [],
        "allow_fail": allow_fail or [],
        "case_count_mismatch": list(comparison.case_count_mismatch) if comparison else [],
        "no_baseline_reason": comparison.no_baseline_reason if comparison else "",
    }
    return code, detail


def render_exit_decision(detail: dict) -> str:
    """把决策明细渲染成一行结论（CI 日志里最需要看到的那一行）。"""
    code = detail.get("exit_code", 0)
    reason = {0: "通过", 1: "有指标未达门槛", 2: "相对基线退化"}.get(code, "未知")
    parts = [
        f"退出码 {code}（{reason}）",
        f"本次跑：{','.join(detail.get('ran_suites') or []) or '无'}",
        f"参与判定：{','.join(detail.get('selected_suites') or []) or '无'}",
    ]
    if detail.get("fail_on"):
        parts.append(f"--fail-on={','.join(detail['fail_on'])}")
    if detail.get("allow_fail"):
        parts.append(f"--allow-fail={','.join(detail['allow_fail'])}")
    if detail.get("in_scope_failed_suites"):
        parts.append(f"未达标：{','.join(detail['in_scope_failed_suites'])}")
    if detail.get("in_scope_regressed_suites"):
        parts.append(f"退化：{','.join(detail['in_scope_regressed_suites'])}")
    if detail.get("ignored_suites"):
        parts.append(
            f"⚠️ 已忽略（不在判定范围内）：{','.join(detail['ignored_suites'])}"
        )
    return "  ｜  ".join(parts)


def compare_with_previous(
    report: EvalReport,
    history: list[dict],
    tolerance: float = 0.02,
) -> Comparison:
    """与历史里**最近一次同 schema 版本、同套件集合**的运行比较。

    ★ 为什么必须按套件集合过滤（这是"分别评测 RAG 与 Agent"的前提）
    ---------------------------------------------------------
    原实现只按 `schema_version` 过滤，取 `candidates[-1]` 当基线。
    但"跑了哪些线"是个变量 —— 实测历史里前 6 次只有 `agent`，
    之后变成 `agent + retrieval`。于是**单独跑一条线时必然误报**：

        跑 `--suites retrieval --gate`
          → 基线取到"只有 agent"的那次
          → 本次每条 retrieval 指标 `prev is None`
          → 全部被记成**"新增指标"**（而不是"没法比"）
        跑 `--suites agent --gate`（基线是混合那次）
          → retrieval 指标本次没测 → 塞进 `missing`
          → 报告里出现一堆"缺失"，而它只是**没跑那条线而已**

    两类假信号都会让门禁不可信。故现在要求**套件集合完全一致**才作基线。

    ★ 没有 `suites` 字段的旧记录**不参与基线选择**
    ----------------------------------------
    它们是本字段引入前写的，无法判断跑了哪些线。
    宁可"没有基线可比"（明确告知），也不要拿一个**不知道包含什么**的
    基线去比 —— 后者会产出看似有理的假结论。
    """
    cmp = Comparison()
    cur_suites = sorted(s.suite for s in report.suites if s.ran)

    # ★ 分线基线状态：先算出来，无论后面有没有共同基线，都能告诉使用者
    #   "哪条线有基线、哪条线还没有"（这正是 `--suites rag --gate` 首次运行
    #   最需要知道的信息：不是"没有历史"，而是"rag 这条线还没跑过"）。
    cmp.suite_baselines = {
        suite: latest_baseline_for(history, suite, SCHEMA_VERSION) for suite in cur_suites
    }

    candidates = [
        r for r in history
        if r.get("schema_version") == SCHEMA_VERSION
        and isinstance(r.get("metrics"), dict)
        # ★ 旧记录没有 suites：不猜，直接不作为基线
        and isinstance(r.get("suites"), list)
        and sorted(r["suites"]) == cur_suites
    ]

    # ★ 跳过"一条实测值都没有"的记录（实测踩到）
    # ----------------------------------------
    # 场景：某次运行检索全部异常（例如 import 撞名），四条 retrieval 指标
    # 全是 `value: null`。这条记录**照样会进趋势文件**，然后成为下一次
    # `--suites retrieval --gate` 的基线 —— 于是每个指标 `prev_value is None`
    # → 全被记成"新增指标" → **门禁静默失效（永远绿）**，而报告上看不出异常。
    #
    # 故：优先用"至少测出一项"的记录当基线；全都如此时退回最新那条，
    # 并在 `render()` 里明说 —— 静默选择一个没有信息量的基线是不可接受的。
    usable = [r for r in candidates if any(_has_value(m) for m in (r.get("metrics") or {}).values())]
    skipped = [str(r.get("run_id") or "") for r in candidates if r not in usable]
    if usable:
        candidates = usable
    cmp.baseline_skipped_all_not_run = skipped
    cmp.baseline_all_not_run = not usable

    if not candidates:
        cmp.missing = [r.name for r in report.all_results]
        # ★ 如实说明"为什么没有基线"。三种原因要说清是哪一种 ——
        #   笼统说"没有历史"会让人以为趋势文件坏了。
        same_schema = [
            r for r in history
            if r.get("schema_version") == SCHEMA_VERSION
            and isinstance(r.get("metrics"), dict)
        ]
        if not same_schema:
            cmp.no_baseline_reason = "趋势文件里没有同 schema 版本的历史记录"
        elif not any(isinstance(r.get("suites"), list) for r in same_schema):
            cmp.no_baseline_reason = (
                f"没有套件集合为 {cur_suites} 的历史记录可比；"
                "现有记录都是 `suites` 字段引入前写的，无法判断跑了哪些线"
            )
        else:
            seen = sorted({tuple(sorted(r["suites"])) for r in same_schema
                           if isinstance(r.get("suites"), list)})
            cmp.no_baseline_reason = (
                f"没有套件集合为 {cur_suites} 的历史记录可比；"
                f"历史里出现过的是 {[list(s) for s in seen]}"
            )
        return cmp

    baseline = candidates[-1]
    cmp.baseline_run_id = str(baseline.get("run_id") or "")
    cmp.baseline_version = str(baseline.get("version") or "")
    # ★ 记录基线的 `suites` 是实测还是回填推断：推断值在"某线跑了但用例数为 0"
    #   时会漏一条线，读报告的人需要知道这个基线是哪种。
    cmp.baseline_suites_source = str(
        baseline.get("suites_source")
        or ("inferred" if baseline.get("suites_inferred") else "observed")
    )
    gap = baseline.get("suites_inferred_gap")
    cmp.baseline_inferred_gap = [str(g) for g in gap] if isinstance(gap, list) else []
    prev_metrics: dict = baseline.get("metrics") or {}
    prev_counts: dict = baseline.get("case_counts") or {}

    # ★ 样本量校验：`--limit 3` 与 18 题的基线比会报出一堆"退化"，
    # 而那些差异只来自样本变了。实测踩过（RAGAS 里的同类问题见 docs/21）。
    # 样本量不一致时**保留对比结果供参考，但标记出来让调用方不据此判失败**。
    cur_counts = {s.suite: s.case_count for s in report.suites if s.ran}
    for suite_name, cur_n in cur_counts.items():
        prev_n = prev_counts.get(suite_name)
        if prev_n is not None and prev_n != cur_n:
            cmp.case_count_mismatch.append(f"{suite_name} {prev_n} → {cur_n} 题")

    for result in report.all_results:
        spec = get_spec(result.name)
        label = spec.label if spec else result.name
        prev = prev_metrics.get(result.name)

        if prev is None:
            cmp.new_metrics.append(result.name)
            continue
        prev_value = prev.get("value") if isinstance(prev, dict) else None
        if prev_value is None:
            cmp.new_metrics.append(result.name)
            continue
        if result.value is None:
            # 本次没测但历史有值：如实列出。**不算退化**——
            # 没测不等于变坏，把它算退化会让"跳过 RAGAS"变成门禁失败。
            cmp.missing.append(label)
            continue

        try:
            prev_f = float(prev_value)
            cur_f = float(result.value)
        except (TypeError, ValueError):
            continue

        # 只记录、不参与相对退化判定（如 token/成本，抖动远大于 2% 容差）。
        # 放在 new/missing 判定之后：它们仍会作为"新指标/本次未测"出现，
        # 只是不会被归入退化或进步。
        if spec is not None and not spec.track_regression:
            continue

        delta = cur_f - prev_f
        # 相对容差：小数值指标（如 0.79 → 0.78）用绝对容差会过于宽松，
        # 故用相对比例。prev_f 为 0 时退化为绝对比较。
        allowed = abs(prev_f) * tolerance if prev_f else tolerance
        if abs(delta) <= allowed:
            continue

        # ★ 方向判定：延迟升高是退化、准确率升高是进步
        is_worse = (
            delta < 0
            if (spec is not None and spec.direction is Direction.HIGHER_IS_BETTER)
            else delta > 0
        )
        item = Regression(
            name=result.name, label=label,
            previous=prev_f, current=cur_f, delta=delta,
            threshold=result.threshold,
        )
        (cmp.regressions if is_worse else cmp.improvements).append(item)

    return cmp


__all__ = [
    "MAX_HISTORY",
    "TREND_PATH",
    "Comparison",
    "Regression",
    "all_suites",
    "append_report",
    "compare_with_previous",
    "decide_exit_code",
    "latest_baseline_for",
    "load_history",
    "render_exit_decision",
    "report_to_record",
    "suite_of",
]

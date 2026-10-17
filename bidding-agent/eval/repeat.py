"""重复采样与中位聚合（`--repeat N`）。

★ 要解决的缺陷
==========
门禁容差是 2%，而**实测 agent 线指标在同代码同配置下可波动 2~3 倍**：

    agent.latency_p95          11172 ~ 19405 ms   → 74%
    agent.redundant_calls_mean 0.5294 ~ 0.8235    → 56%
    agent.hit_rounds_cap_rate  0.0588 ~ 0.2353    → 300%

故**单次运行 + 2% 容差**必然产出假阳性 —— 不是偶发，是结构性的。
原体系**没有**任何重复采样机制（`run_all` 里搜 `repeat` 零命中），
趋势文件里的"重复"是人手动跑的。

★ 为什么取**中位**而不是均值
====================
LLM 的抖动是**偏态**的（偶尔一次跑飞、分数被拉到很低），
均值会被离群值拖动；中位抗离群。

★ 为什么 `NOT_RUN` 不参与中位
========================
`NOT_RUN` 是"没测到"，不是"测出 0"。把它当 0 参与中位，
会把"三次里一次依赖缺失"变成"分数被拉低" —— 那是**用缺失冒充低分**。
故：**全 NOT_RUN → NOT_RUN；否则只在有效值里取中位**。

★ 为什么状态要**重算**而不是沿用
==========================
中位数可能与任何一次的单次值都不同（如 3 次的 0.7/0.8/0.9 → 中位 0.8）。
沿用某次的 `status` 会与 `value` 不一致，
故用 `MetricSpec.judge(median)` 重新判定。
"""

from __future__ import annotations

import statistics
from typing import Iterable

from eval.report import (
    MetricResult,
    MetricStatus,
    SuiteReport,
    get_spec,
)


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def merge_suite_runs(runs: list[SuiteReport], *, note_suffix: str = "") -> SuiteReport:
    """把同一评测线的 N 次运行合并成一次（**指标取中位**）。

    `runs` 为空时返回一个 `ran=False` 的空报告（不抛异常）。
    """
    if not runs:
        return SuiteReport(suite="?", ran=False, note="没有可合并的运行")

    suite_name = runs[0].suite
    merged = SuiteReport(suite=suite_name)
    merged.ran = any(r.ran for r in runs)
    # 用例数取**众数/首值**：同一个套件各次运行的用例数应当相同，
    # 若不同（如中途改了 --limit），取最小值最保守。
    counts = [r.case_count for r in runs if r.ran]
    merged.case_count = min(counts) if counts else 0

    # 按指标名收集各次的值
    values: dict[str, list[float]] = {}
    statuses: dict[str, list[MetricStatus]] = {}
    notes: dict[str, list[str]] = {}
    order: list[str] = []
    for r in runs:
        for res in r.results:
            if res.name not in values:
                values[res.name] = []
                statuses[res.name] = []
                notes[res.name] = []
                order.append(res.name)
            statuses[res.name].append(res.status)
            if res.note:
                notes[res.name].append(res.note)
            if res.status is not MetricStatus.NOT_RUN and isinstance(res.value, (int, float)):
                values[res.name].append(float(res.value))

    for name in order:
        vals = values[name]
        spec = get_spec(name)
        if not vals:
            # 全 NOT_RUN：保持 NOT_RUN，不冒充 0
            merged.results.append(
                MetricResult(
                    name=name,
                    value=None,
                    status=MetricStatus.NOT_RUN,
                    threshold=(spec.threshold if spec else None),
                    note=f"{len(runs)} 次运行均为未测",
                )
            )
            continue
        med = _median(vals)
        status = spec.judge(med) if spec else MetricStatus.PASS
        # note 说明"这是 N 次的中位"，并带上各次实值 —— 否则看不出波动
        spread = (
            f"{len(runs)} 次中位={med:.4g}"
            f"（实值 {', '.join(f'{v:.4g}' for v in vals)}）"
        )
        if len(vals) < len(runs):
            spread += f"；{len(runs) - len(vals)} 次未测未计入"
        if note_suffix:
            spread += f"；{note_suffix}"
        merged.results.append(
            MetricResult(
                name=name,
                value=med,
                status=status,
                threshold=(spec.threshold if spec else None),
                note=spread,
            )
        )

    first_note = next((r.note for r in runs if r.note), "")
    merged.note = f"{first_note}｜{len(runs)} 次取中位".strip("｜") if runs else ""
    return merged


def runs_of(suite_names: Iterable[str], report_runs: list[dict[str, SuiteReport]]) -> dict[str, SuiteReport]:
    """把 `[{suite: SuiteReport}]` × N 转成 `{suite: 合并后的 SuiteReport}`。"""
    out: dict[str, SuiteReport] = {}
    for name in suite_names:
        rs = [d[name] for d in report_runs if name in d]
        if rs:
            out[name] = merge_suite_runs(rs)
    return out

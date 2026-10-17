"""离线评测结果 → 前端可用的快照。

★ 为什么需要它（而不是让前端直接读文件）
============================
浏览器读不了磁盘。前端要展示离线评测（`eval/run_all.py` 的产物），
必须经后端——故这里是"**读落盘快照 + 归一成前端契约**"的一层。

★ 数据来源（三个文件，各管一段）
========================
    `eval/trend.jsonl`         每次运行的**聚合指标**（含 verdict / config / case_counts）
                               —— 这是主数据源，11 次历史运行都在里面
    `eval/judge_agreement.json` 裁判一致性（test-retest）的表格与失败计数
    `eval/eval_snapshot.json`   可选：`run_all.py --json` 导出的**完整报告**
                               （含每个 suite 的逐指标明细）

★ 一个必须如实标注的局限
==================
**没有逐用例明细。** `EvalReport` 的结构是
`EvalReport → suites[] → results[]（MetricResult）`，全是**聚合值**；
逐用例数据在 `run_all.py` 内部用过就丢了（`per_case_metrics` 只用于 `aggregate`）。

故前端**做不出"逐用例表格"**。要做得先改 `run_all.py` 把逐用例结果也落盘 ——
那是另一项改动，本次未做。快照里显式给出 `capabilities.per_case = False`，
让前端据此**隐藏**那张表，而不是画一张空表。
"""

from __future__ import annotations

import json
import logging
import pathlib
from typing import Any

logger = logging.getLogger(__name__)

#: 评测目录（相对本文件：src/eval_offline/ → 仓库根 → eval/）
EVAL_DIR = pathlib.Path(__file__).resolve().parents[2] / "eval"
TREND = EVAL_DIR / "trend.jsonl"
AGREEMENT = EVAL_DIR / "judge_agreement.json"
SNAPSHOT = EVAL_DIR / "eval_snapshot.json"

#: 指标名前缀 → 中文分组名（前端按这个分组展示）
SUITE_LABELS = {
    "agent": "Agent 编排",
    "retrieval": "检索质量",
    "rag": "RAG 生成",
    "judge": "裁判一致性",
}

#: 指标名 → 中文标签。**只覆盖已知的**，未知的原样显示（不猜语义）。
METRIC_LABELS = {
    "agent.tool_accuracy": "工具选择准确率",
    "agent.latency_p95": "延迟 P95（ms）",
    "agent.tool_calls_mean": "平均工具调用次数",
    "agent.hit_rounds_cap_rate": "触达轮次上限比例",
    "agent.redundant_calls_mean": "平均冗余调用",
    "agent.verification_ok_rate": "答案核对通过率",
    "agent.verification_critical_ok_rate": "关键核对通过率",
    "agent.false_unavailable_rate": "误报不可用比例",
    "agent.answer_chars_mean": "平均答案字数",
    "retrieval.precision": "检索精确率",
    "retrieval.recall": "检索召回率",
    "retrieval.mrr": "MRR",
    "rag.faithfulness": "忠实度",
    "rag.answer_relevancy": "答案相关性",
    "rag.context_precision": "上下文精确率",
    "rag.context_recall": "上下文召回率",
}

#: 哪些指标是**比例**（0~1），前端要用百分比而不是原值渲染。
#:
#: ★ 为什么必须显式列出而不是"值 ≤ 1 就按比例渲染"
#: --------------------------------------------
#: `agent.redundant_calls_mean` 的 0.82 是"平均 0.82 次"，不是 82%；
#: `agent.answer_chars_mean` 的 1.0 更不是 100%。用值域猜必然出错，
#: 而错法是静默的（数字看起来仍然合理）。故**白名单**，不在表里就按原值渲染。
RATIO_METRICS = frozenset({
    "agent.tool_accuracy",
    "agent.hit_rounds_cap_rate",
    "agent.verification_ok_rate",
    "agent.verification_critical_ok_rate",
    "agent.false_unavailable_rate",
    "retrieval.precision",
    "retrieval.recall",
    "retrieval.mrr",
    "rag.faithfulness",
    "rag.answer_relevancy",
    "rag.context_precision",
    "rag.context_recall",
})


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue          # 坏行跳过，不让整份历史失败
            if isinstance(obj, dict):
                out.append(obj)
    except OSError:
        return []
    return out


def _read_json(path: pathlib.Path) -> dict | None:
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        return obj if isinstance(obj, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _suite_of(metric_key: str) -> str:
    return metric_key.split(".", 1)[0] if "." in metric_key else "other"


def build_snapshot() -> dict[str, Any]:
    """组装前端契约。**永不抛异常**（端点直接返回它的结果）。"""
    try:
        return _build()
    except Exception:  # noqa: BLE001
        logger.warning("组装离线评测快照失败", exc_info=True)
        return {"error": "组装离线评测快照失败", "runs": [], "metrics": []}


def _build() -> dict[str, Any]:
    runs = _read_jsonl(TREND)

    # ---- 指标趋势：把 11 次运行按指标名转置 ----
    #
    # 前端要画"指标随时间"的折线，需要 `{metric: [{run_id, value}, ...]}`；
    # 而落盘是"每次运行一行、内含全部指标"（更适合 diff 与门禁）。
    # 转置放在后端：两处各转一次容易出现口径差异。
    series: dict[str, list[dict]] = {}
    latest_metrics: dict[str, dict] = {}
    for run in runs:
        rid = str(run.get("run_id") or "")
        for key, cell in (run.get("metrics") or {}).items():
            if not isinstance(cell, dict):
                continue
            v = cell.get("value")
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                continue
            series.setdefault(key, []).append({"run_id": rid, "value": float(v)})
            latest_metrics[key] = {
                "metric": key,
                "label": METRIC_LABELS.get(key, key),
                "suite": _suite_of(key),
                "value": float(v),
                "status": cell.get("status") or "",
                "is_ratio": key in RATIO_METRICS,
            }

    # 按 suite 排序，前端可直接分组渲染
    metrics = sorted(
        latest_metrics.values(), key=lambda m: (m["suite"], m["metric"])
    )

    last = runs[-1] if runs else {}
    agreement = _read_json(AGREEMENT) or {}

    return {
        "schema_version": 1,
        "run_count": len(runs),
        "latest": {
            "run_id": last.get("run_id", ""),
            "version": last.get("version", ""),
            "verdict": last.get("verdict", ""),
            "config": last.get("config") or {},
            "case_counts": last.get("case_counts") or {},
        },
        "metrics": metrics,
        "series": series,
        "suite_labels": SUITE_LABELS,
        "judge_agreement": {
            "judges": agreement.get("judges") or [],
            "cases": agreement.get("cases"),
            "repeats": agreement.get("repeats"),
            "failures": agreement.get("failures") or {},
            "table": agreement.get("table") or {},
        },
        # ★ 能力声明：让前端**据此隐藏**做不到的东西，而不是画空表
        "capabilities": {
            # 逐用例明细没有落盘（见模块 docstring）
            "per_case": False,
            "per_case_reason": (
                "EvalReport 只保存聚合后的 MetricResult；逐用例结果在 run_all.py "
                "内部用完即弃。要展示需先让它把逐用例也落盘。"
            ),
            "trend": bool(runs),
            "judge_agreement": bool(agreement),
        },
        "sources": {
            "trend": str(TREND.name),
            "judge_agreement": str(AGREEMENT.name) if agreement else "",
        },
    }

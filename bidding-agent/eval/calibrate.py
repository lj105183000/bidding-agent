"""门槛标定：用趋势历史回答"这个门槛定得对不对、够不够样本"。

★ 为什么必须有这个工具（实测的教训）
==============================
新增的检索排序指标（`retrieval.top1_relevant_rate` = 0.9444、
`retrieval.first_relevant_rank_mean` = 1.06）当时**只跑了一遍**就定了门槛
0.80 / 1.6 —— 而这两个数离"理想值 1.0 / 1.0"很近，意味着：

    · 门槛 0.80 比实测最低值还低 0.14，**几乎不可能拦住退化**
    · 而如果下次因抖动掉到 0.83，门禁仍然判"达标"，人也不会去看

问题的根源不是"门槛取错了"，而是**一次实测不足以标定门槛**：
LLM 有随机性，单次分数是"真值 + 噪声"，用一次观测去卡上下限，
要么过松（挡不住退化）要么过紧（天天误报）。

★ 标定规则（写在代码里，避免"凭感觉调参"）
====================================
    HIB（越高越好）：门槛 = 观测均值 − 2×标准差，且**不高于观测最小值**
    LIB（越低越好）：门槛 = 观测均值 + 2×标准差，且**不低于观测最大值**
    样本不足（< MIN_RUNS_FOR_THRESHOLD）**不给建议值**，只报"还差几次"

"−2σ"是控制图里的常用下控制限：正常波动有约 95% 落在 μ±2σ 内，
故它天然满足"正常波动不报警、真退化才报警"。样本少时 σ 不可信，
**宁可不给数字**——给出一个基于 2 次观测的门槛，比不给更危险。

★ 为什么用趋势文件而不是重跑评测
==========================
门槛要在**可比条件下**的多次观测上标定：换模型、换 k、换提示词之后的分数
不是同一个分布。趋势文件里的每次运行都带 `config` 快照，
故本工具默认**按 config 分组**，只在同配置的观测上标定 —— 否则会把
"换了模型导致的整体抬升"误当成"正常波动范围"。
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

from eval.report import METRIC_SPECS, Direction, get_spec
from eval.trend import TREND_PATH, load_history

#: 少于这么多次观测就不给建议门槛。5 是"能看出大致波动范围"的最低要求：
#: 3 次算不出稳定的 σ，4 次仍可能被一次异常值主导。
MIN_RUNS_FOR_THRESHOLD = 5

#: 控制限倍数（μ ± k·σ）。2 对应约 95% 的正常波动区间。
CONTROL_LIMIT_K = 2.0


def collect_values(history: list[dict], name: str, *, same_conditions: bool = True) -> list[float]:
    """从历史里取某指标的观测值（跳过 None 与非法值）。

    ★ 只取**可比条件**下的观测（这是本工具最容易出错的地方）
    ------------------------------------------------
    两组过滤缺一不可：

        · **同 config**：换了模型 / top_k / 提示词之后的分数属于另一个分布，
          混进来会把"正常波动范围"算大（甚至把整体抬升当成波动）。
        · **同 case_counts**：`--limit 3` 与 17 题的分数不可比。
          实测趋势文件里既有 17 题运行也有 3 题运行；不过滤就会拿
          "3 题时的波动"去标定"17 题时的门槛"，而 3 题的准确率天然更极端。

    过滤基准是**最近一次含该指标的记录**：标定是为了判断"当前这套配置下
    门槛定得对不对"，故以当下的条件为准。
    """
    spec = get_spec(name)
    suite = spec.suite if spec and spec.suite else name.split(".", 1)[0]

    def cfg_key(rec: dict) -> str:
        return json.dumps(rec.get("config") or {}, sort_keys=True, ensure_ascii=False)

    records = [
        r for r in history
        if isinstance(r.get("metrics"), dict)
        and (name in r["metrics"])
        # 只认**跑了该线**的记录：没跑的记录里该指标缺失，本就取不到值
        and (not isinstance(r.get("suites"), list) or suite in r["suites"])
    ]
    if same_conditions and records:
        latest_cfg = cfg_key(records[-1])
        latest_n = (records[-1].get("case_counts") or {}).get(suite)
        kept: list[dict] = []
        for rec in records:
            if cfg_key(rec) != latest_cfg:
                continue
            n = (rec.get("case_counts") or {}).get(suite)
            if latest_n is not None and n is not None and n != latest_n:
                continue
            kept.append(rec)
        records = kept

    out: list[float] = []
    for rec in records:
        entry = (rec.get("metrics") or {}).get(name)
        value = entry.get("value") if isinstance(entry, dict) else None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out.append(float(value))
    return out


def describe(values: list[float]) -> dict:
    """观测分布。样本 < 2 时标准差无意义，返回 None 而不是 0。"""
    if not values:
        return {"n": 0, "min": None, "max": None, "mean": None, "median": None, "stdev": None}
    return {
        "n": len(values),
        "min": min(values),
        "max": max(values),
        "mean": round(statistics.fmean(values), 6),
        "median": round(statistics.median(values), 6),
        "stdev": round(statistics.stdev(values), 6) if len(values) >= 2 else None,
    }


def suggest(
    values: list[float],
    direction: Direction,
    current_threshold: float | None,
) -> tuple[float | None, str]:
    """按控制限给建议门槛，返回 `(建议值或 None, 说明)`。"""
    stats = describe(values)
    n = stats["n"]
    if current_threshold is None:
        return None, "该指标是观测指标（无门槛），不需要标定"
    if n < MIN_RUNS_FOR_THRESHOLD:
        return None, (
            f"样本不足：仅有 {n} 次同配置观测，需 {MIN_RUNS_FOR_THRESHOLD} 次才能给出建议值"
            f"（还差 {MIN_RUNS_FOR_THRESHOLD - n} 次）。"
            "★ 用一两次观测定门槛要么过松（挡不住退化）要么过紧（天天误报）"
        )
    stdev = stats["stdev"] or 0.0
    mean = stats["mean"]
    lo = round(mean - CONTROL_LIMIT_K * stdev, 4)
    hi = round(mean + CONTROL_LIMIT_K * stdev, 4)
    if direction is Direction.HIGHER_IS_BETTER:
        # 不高于观测最小值：否则门槛比历史最差还松，等于没有门槛
        value = min(lo, stats["min"])
        return value, f"μ−{CONTROL_LIMIT_K:g}σ = {lo}，再取不超过观测最小值 {stats['min']}"
    value = max(hi, stats["max"])
    return value, f"μ+{CONTROL_LIMIT_K:g}σ = {hi}，再取不低于观测最大值 {stats['max']}"


def calibration(history: list[dict] | None = None,
                names: list[str] | None = None) -> dict:
    """逐指标标定结论。返回结构化结果（供报告与测试断言）。"""
    history = load_history() if history is None else history
    names = names or sorted(METRIC_SPECS)
    items: list[dict] = []
    for name in names:
        spec = get_spec(name)
        if spec is None:
            continue
        values = collect_values(history, name)
        stats = describe(values)
        suggested, why = suggest(values, spec.direction, spec.threshold)
        band = _verdict(stats, spec.threshold, spec.direction)
        items.append(
            {
                "name": name,
                "label": spec.label,
                "suite": spec.suite,
                "direction": spec.direction.value,
                "threshold": spec.threshold,
                "observed": stats,
                "suggested": suggested,
                "suggestion_reason": why,
                "verdict": band,
            }
        )
    return {
        "history_runs": len(history),
        "min_runs": MIN_RUNS_FOR_THRESHOLD,
        "items": items,
    }


def _verdict(stats: dict, threshold: float | None, direction: Direction) -> str:
    """门槛相对观测区间的松紧判断（只看能判的情形）。"""
    if threshold is None:
        return "no_threshold"
    if stats["n"] == 0:
        return "no_data"
    if stats["n"] < MIN_RUNS_FOR_THRESHOLD:
        return "insufficient_samples"
    if direction is Direction.HIGHER_IS_BETTER:
        if threshold < stats["min"]:
            return "too_loose"      # 比历史最差还低：挡住退化的能力有限
        if threshold > stats["max"]:
            return "too_tight"      # 比历史最好还高：正常波动就会报警
        return "ok"
    if threshold > stats["max"]:
        return "too_loose"
    if threshold < stats["min"]:
        return "too_tight"
    return "ok"


_VERDICT_TEXT = {
    "no_threshold": "观测指标（无门槛）",
    "no_data": "趋势里没有该指标的观测",
    "insufficient_samples": "样本不足，暂不判定",
    "too_loose": "偏松（低于历史最差，挡退化能力有限）",
    "too_tight": "偏紧（高于历史最好，正常波动会误报）",
    "ok": "合理（落在观测区间内）",
}


def render(data: dict | None = None) -> str:
    """渲染成终端可读的一段。"""
    data = data or calibration()
    lines = ["-" * 78, "门槛标定（按趋势历史）", "-" * 78]
    lines.append(
        f"  历史运行 {data['history_runs']} 次；同配置观测数 ≥ {data['min_runs']} 才给建议值"
    )
    for item in data["items"]:
        obs = item["observed"]
        if obs["n"]:
            shown = (
                f"n={obs['n']}  min={obs['min']:g}  中位={obs['median']:g}  "
                f"max={obs['max']:g}  σ={obs['stdev'] if obs['stdev'] is not None else '—'}"
            )
        else:
            shown = "n=0"
        thr = "—" if item["threshold"] is None else f"{item['threshold']:g}"
        lines.append(f"  · {item['label']}（{item['name']}）")
        lines.append(f"      门槛 {thr}｜{shown}")
        lines.append(f"      {_VERDICT_TEXT.get(item['verdict'], item['verdict'])}")
        if item["suggested"] is not None:
            lines.append(f"      建议门槛 {item['suggested']:g}（{item['suggestion_reason']}）")
        elif item["suggestion_reason"]:
            lines.append(f"      {item['suggestion_reason']}")
    lines.append("-" * 78)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    path = TREND_PATH
    if "--trend" in args:
        idx = args.index("--trend")
        if idx + 1 < len(args):
            path = Path(args[idx + 1])
    data = calibration(load_history(path))
    if "--json" in args:
        idx = args.index("--json")
        target = args[idx + 1] if idx + 1 < len(args) else ""
        payload = json.dumps(data, ensure_ascii=False, indent=2)
        if target:
            Path(target).write_text(payload, encoding="utf-8")
            print(f"已写入 {target}")
        else:
            print(payload)
    else:
        print(render(data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

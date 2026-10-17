"""一次性迁移：给 `suites` 字段引入前的历史记录回填该字段。

★ 为什么需要它
==========
`compare_with_previous` 现在按**套件集合**过滤基线（这是"分别评测 RAG 与
Agent"的前提，见 `docs/77`）。但 `suites` 是本字段引入前不存在的，
于是 11 条旧记录**全部不作基线** —— 分线门禁"现在没有可比的历史"。

★ 为什么可以推断（已核实，不是猜）
========================
`report_to_record` 的推导式是：

    "case_counts": {s.suite: s.case_count for s in report.suites if s.ran}

**键与"运行的套件集合"用的是同一个 `if s.ran` 过滤** ——
故 `case_counts` 的键集合**就是**当时的套件集合。

实测校验（`logs/prove_backfill.py`）：11 条旧记录里，
每条的 `metrics` 前缀集合都**被 `case_counts` 键集合覆盖** ——
说明推断保守且自洽（没有"算出指标但没记 case_count"的情况）。

★ 这个推断的**已知边界**（必须如实标注，不能假装是实测值）
==============================================
若某条线跑了但**用例数为 0**（`--limit 0` 或用例文件为空），
它不会出现在 `case_counts` 里 → 只靠 `case_counts` 推断会**少一条线**。

★ 第二版：把"少一条线"这个情形**具体识别出来**（而不是只挂一个标记）
=====================================================
只用 `case_counts` 推断时，"少一条线"是完全隐形的 —— 标记 `suites_inferred`
只说明"这是推断值"，说不出**可能漏了谁**。但漏掉的线其实会留下另一条痕迹：

    `run_*_suite` 即使 0 用例也会 `make_result(..., None)` 追加指标
    （值为 None = 未测，但**键存在**）

故 `metrics` 的**键前缀集合**能反推出"这条线跑过"。于是：

    推断集合 = case_counts 的键 ∪ metrics 的键前缀
    疑似漏项 = metrics 的键前缀 − case_counts 的键      ← 就是"跑了但 0 用例"

回填时把 `suites_inferred_gap` 一并写上，门禁读到该基线时会提示
"记录里标出的疑似漏项：[...]" —— 让人知道这个基线可能缺哪条线，
而不是笼统地"这是推断的，自己小心"。

故回填的每条记录都加 `suites_inferred: true` 标记 ——
让需要严格口径的调用方能区分「实测记录的」与「推断的」。
这是本项目一贯的取舍：**宁可标注局限，也不让推断冒充事实**。

用法::

    python -m eval.migrate_suites            # 预览（不写盘）
    python -m eval.migrate_suites --apply    # 实际写入（自动备份）
"""

from __future__ import annotations

import json
import pathlib
import shutil
import sys

from eval.trend import TREND_PATH


def metric_suite_prefixes(record: dict) -> set[str]:
    """从 `metrics` 的键前缀反推"跑过哪些线"。

    `agent.tool_accuracy` → `agent`。指标名一律是 `线.指标` 的形状
    （见 `eval/report.METRIC_SPECS`），故前缀就是线名。
    """
    metrics = record.get("metrics")
    if not isinstance(metrics, dict):
        return set()
    out: set[str] = set()
    for name in metrics:
        text = str(name)
        if "." in text:
            prefix = text.split(".", 1)[0]
            if prefix:
                out.add(prefix)
    return out


def infer_gap(record: dict) -> list[str]:
    """**疑似漏记**的线：`metrics` 里有它、`case_counts` 里没有它。

    这正是"该线跑了但用例数为 0"的形态（0 用例不进 `case_counts`，
    但 `run_*_suite` 仍会追加值为 None 的指标）。返回排序后的线名。
    """
    cc = record.get("case_counts")
    counted = {str(k) for k in cc} if isinstance(cc, dict) else set()
    return sorted(metric_suite_prefixes(record) - counted)


def infer_suites(record: dict) -> list[str] | None:
    """推断套件集合：`case_counts` 的键 ∪ `metrics` 的键前缀。

    返回 None 的情形：两个来源都取不到 —— 这些记录无法判断跑了哪些线，
    应保持"不作基线"。

    ★ 为什么是**并集**而不是只用 `case_counts`
    ----------------------------------------
    只用 `case_counts` 会漏掉"跑了但 0 用例"的线（见模块文档）。
    `metrics` 的键前缀能补上这一类，且**不会引入没跑过的线**：
    只有 `run_*_suite` 执行过才会追加结果。
    """
    keys: set[str] = set()
    cc = record.get("case_counts")
    if isinstance(cc, dict):
        keys.update(str(k) for k in cc)
    keys |= metric_suite_prefixes(record)
    return sorted(keys) if keys else None


def plan(records: list[dict]) -> list[tuple[int, str, list[str] | None]]:
    """算出要改哪些行。返回 `[(行号, run_id, 推断出的 suites)]`。"""
    out: list[tuple[int, str, list[str] | None]] = []
    for i, rec in enumerate(records):
        if isinstance(rec.get("suites"), list):
            continue                      # 已有：不动
        out.append((i, str(rec.get("run_id") or ""), infer_suites(rec)))
    return out


def migrate(path: pathlib.Path | None = None, apply: bool = False) -> dict:
    """回填 `suites`。返回统计信息（供测试断言，不打印）。"""
    target = path or TREND_PATH
    if not target.exists():
        return {"error": f"趋势文件不存在：{target}"}

    lines = [ln for ln in target.read_text(encoding="utf-8").splitlines() if ln.strip()]
    records: list[dict] = []
    bad_lines = 0
    for ln in lines:
        try:
            obj = json.loads(ln)
        except json.JSONDecodeError:
            bad_lines += 1
            continue
        if isinstance(obj, dict):
            records.append(obj)

    todo = plan(records)
    filled = [t for t in todo if t[2] is not None]
    skipped = [t for t in todo if t[2] is None]

    stat = {
        "total": len(records),
        "already": sum(1 for r in records if isinstance(r.get("suites"), list)),
        "filled": len(filled),
        "skipped_uninferable": len(skipped),
        "skipped_ids": [t[1] for t in skipped],
        "bad_lines": bad_lines,
        #: 带"疑似漏项"的记录数，以及这些漏项的并集（供报告与测试断言）
        "gap_records": sum(1 for idx, _rid, _s in filled if infer_gap(records[idx])),
        "gap_suites": sorted({g for idx, _rid, _s in filled for g in infer_gap(records[idx])}),
        "applied": False,
    }

    if not apply:
        return stat

    for idx, _rid, suites in filled:
        records[idx]["suites"] = suites
        # ★ 标记"这是推断的"，不冒充实测值。`compare_with_previous` 会照用，
        #   但读文件的人/工具能分辨。
        records[idx]["suites_inferred"] = True
        # ★ 显式写出推断来源：后续新增记录写 `observed`，两者一眼可分。
        records[idx]["suites_source"] = "inferred"
        # ★ 把"可能漏了哪条线"写进记录（只在确有漏项时写，避免噪音）
        gap = infer_gap(records[idx])
        if gap:
            records[idx]["suites_inferred_gap"] = gap

    # 先备份再写：趋势文件是长期资产，写坏就没了
    backup = target.with_suffix(target.suffix + ".pre-suites-migration")
    shutil.copy2(target, backup)

    target.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )
    stat["applied"] = True
    stat["backup"] = str(backup)
    return stat


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    apply = "--apply" in args

    stat = migrate(apply=apply)
    if "error" in stat:
        print(f"  {stat['error']}")
        return 1

    print(f"  趋势文件记录数：{stat['total']}"
          + (f"（{stat['bad_lines']} 行解析失败已跳过）" if stat["bad_lines"] else ""))
    print(f"  已有 suites（新记录）：{stat['already']}")
    print(f"  可回填（case_counts ∪ metrics 键前缀）：{stat['filled']}")
    if stat["gap_records"]:
        print(f"  ⚠️  其中 {stat['gap_records']} 条记录存在**疑似漏项**（跑了但用例数为 0 的线）：")
        print(f"      {stat['gap_suites']}")
        print("      这些线已按 metrics 键前缀补进 suites，并写入 suites_inferred_gap")
    if stat["skipped_uninferable"]:
        print(f"  无法推断（保持不作基线）：{stat['skipped_uninferable']}")
        for rid in stat["skipped_ids"]:
            print(f"      · {rid}")
    if stat["applied"]:
        print(f"  ✅ 已写入。备份：{stat['backup']}")
        print("     回填的记录带 `suites_inferred: true` 与 `suites_source: \"inferred\"`"
              "（推断值，非实测）")
    else:
        print("  （预览模式，未写盘。加 --apply 实际执行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

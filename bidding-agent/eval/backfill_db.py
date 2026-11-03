"""把 `eval/trend.jsonl` 的历史回填到 PostgreSQL `eval_runs`。

★ 为什么单独一个脚本
----------------
回填是一次性/运维动作，不该塞进 `run_all.py` 的每次运行。它只读趋势文件、
只写数据库，失败不会影响门禁；`--dry-run` 甚至不导入 psycopg，方便在
没有数据库的机器上先看会处理多少条。

★ 幂等与覆盖边界
------------
默认 `ON CONFLICT DO NOTHING`，重复执行不会产生重复行。
`--overwrite` 只更新 `source='backfill'` 的行——实时 CLI/API 记录
永远不会被历史文件覆盖，否则一次误操作就能改掉看板上的真实数字。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 与 run_all.py 一致：脚本直接运行时，把项目根加入导入路径。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_TREND = Path(__file__).resolve().parent / "trend.jsonl"


def build_records(trend_path: Path) -> list[dict]:
    """读取趋势文件并归一成 `eval_runs` 行；重复 run_id 保留最后一条。"""
    from eval.trend import load_history
    from src.eval_runs.records import normalize_trend_record

    unique: dict[str, dict] = {}
    for raw in load_history(trend_path):
        run_id = str(raw.get("run_id") or "").strip()
        if not run_id:
            continue
        unique[run_id] = normalize_trend_record(raw, source="backfill")
    return list(unique.values())


def main() -> int:
    # 与 run_all.py 一致：Windows GBK 控制台下中文会乱码，统一 reconfigure 成 UTF-8。
    from eval.common import setup_stdio

    setup_stdio()

    parser = argparse.ArgumentParser(
        description="把 eval/trend.jsonl 回填到 PostgreSQL（幂等）"
    )
    parser.add_argument(
        "--trend",
        default="",
        help="趋势文件路径（默认 eval/trend.jsonl）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只统计将处理的记录数，不连数据库",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="更新已存在的 source='backfill' 行（不影响 CLI/API 实时记录）",
    )
    args = parser.parse_args()

    trend_path = Path(args.trend) if args.trend else DEFAULT_TREND
    if not trend_path.exists():
        print(f"趋势文件不存在：{trend_path}")
        return 1

    records = build_records(trend_path)
    if args.dry_run:
        print(
            f"dry-run：趋势文件 {trend_path} 共 {len(records)} 条可回填记录（未连接数据库）"
        )
        return 0

    try:
        from src.eval_runs.store import save_records

        counts = save_records(records, overwrite_backfill=args.overwrite)
    except Exception as exc:  # noqa: BLE001 - CLI 需要打印可读原因
        print(f"回填失败（{type(exc).__name__}: {exc}）")
        return 1

    print(
        "回填完成："
        f"新增 {counts['inserted']}、"
        f"已存在跳过 {counts['skipped']}、"
        f"覆盖 {counts['overwritten']}、"
        f"非 backfill 受保护 {counts['protected']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

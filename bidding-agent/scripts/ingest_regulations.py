"""法规语料入库脚本：语料统计 → 写入 `bid_regulations` → 入库后校验。

用法（在项目根目录执行）：

    .venv\\Scripts\\python.exe scripts\\ingest_regulations.py
    .venv\\Scripts\\python.exe scripts\\ingest_regulations.py --recreate
    .venv\\Scripts\\python.exe scripts\\ingest_regulations.py --dry-run

与 `main.py ingest` 的区别：
  - `main.py ingest` 导入的是 **问答对**（`bid_qa` 集合 + BM25 词表），全量重建语义；
  - 本脚本导入的是 **法规条文**（`bid_regulations` 集合），point id 由
    `law|article` 决定，**默认就是幂等增量写入**，重复跑不会产生重复点。
    因此 `--recreate` 只在语料结构变化时（例如向量维度调整）才需要。

脚本**不触碰 `bid_qa`**：它只读写 `bid_regulations`。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 允许 `python scripts/ingest_regulations.py` 直接运行：把项目根挂到 sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants.regulations import corpus_stats  # noqa: E402
from src.rag.regulation_store import (
    REGULATION_COLLECTION,
    RegulationError,
    regulation_stats,
    upsert_regulations,
    verify_regulations,
)


def print_corpus_stats(stats: dict) -> None:
    print("=" * 68)
    print("① 语料统计（模块内置，尚未连库）")
    print("=" * 68)
    print(f"  法规数：{stats['laws']}    条文数：{stats['articles']}")

    print("  ── 按效力层级 ──")
    for level, count in stats["by_level"].items():
        print(f"    {level:<6} {count:>4} 条")

    print("  ── 按法规 ──")
    for law, count in stats["by_law"].items():
        print(f"    {count:>4} 条  {law}")

    top = list(stats["topics"].items())[:15]
    print(f"  ── 主题分布（共 {len(stats['topics'])} 个主题，列前 15）──")
    for topic, count in top:
        print(f"    {count:>4} 条  {topic}")


def print_verification(result: dict, expected: int) -> None:
    print("=" * 68)
    print("③ 入库后校验")
    print("=" * 68)
    print(f"  集合：{result['collection']}")
    print(f"  库内条数：{result['points']}    语料条数：{result['corpus_articles']}")
    print(f"  库内法规数：{result['laws']}    语料法规数：{result['corpus_laws']}")
    print(f"  与语料一致：{'是' if result['in_sync'] else '否 ⚠️'}")

    print("  ── 库里按法规分布 ──")
    for law, count in result["by_law"].items():
        print(f"    {count:>4} 条  {law}")

    print("  ── 随机抽 3 条 ──")
    for idx, item in enumerate(result.get("sample", []), 1):
        preview = item["text"].replace("\n", " ")[:60]
        print(f"    [{idx}] {item['law']} {item['article']}")
        print(f"        {preview}")
        print(f"        topics={item['topics']}")

    if result["points"] != expected:
        print(f"  ⚠️ 警告：库内条数 {result['points']} != 语料条数 {expected}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="把法规条文语料写入 Qdrant 的 bid_regulations 集合")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="先删除再重建集合（point id 稳定，日常增补语料不需要）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印语料统计，不连 Qdrant、不向量化",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="额外打印一份机器可读的 JSON 摘要",
    )
    args = parser.parse_args(argv)

    stats = corpus_stats()
    print_corpus_stats(stats)

    if args.dry_run:
        print("\n[dry-run] 已跳过入库。")
        if args.json:
            print(json.dumps(stats, ensure_ascii=False, indent=2))
        return 0

    if stats["articles"] < 120:
        print(f"\n⚠️ 语料只有 {stats['articles']} 条，低于 120 条的底线，请先补充语料。")

    print()
    print("=" * 68)
    print(f"② 入库（集合 {REGULATION_COLLECTION}，recreate={args.recreate}）")
    print("=" * 68)

    try:
        written = upsert_regulations(recreate=args.recreate)
    except RegulationError as exc:
        print(f"\n❌ 入库失败：{exc}")
        print("   排查提示：Qdrant 是否已启动（QDRANT_URL）、嵌入模型是否已本地缓存。")
        return 1

    print(f"  写入完成：{written} 条")

    print()
    try:
        result = verify_regulations()
    except RegulationError as exc:
        print(f"❌ 入库后校验失败：{exc}")
        return 1

    print_verification(result, stats["articles"])

    if args.json:
        print()
        print(json.dumps({"corpus": stats, "stored": result}, ensure_ascii=False, indent=2))

    print()
    if result["in_sync"]:
        print("✅ 法规库就绪。")
        return 0
    print("⚠️ 法规库已写入，但库内条数与语料条数不一致，请检查是否有并发写入或部分失败。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

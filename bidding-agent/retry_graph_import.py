"""带重试的知识图谱导入包装器。

为什么需要
----------
Aura 实例在境外（AWS 美区），从国内访问的 TCP 连通成功率实测约 50%
（连测 6 次成功 3 次，失败时直接超时 12 秒）。
而 `to_neo4j.py` 要分批写入 8730 行，中间任何一次网络抖动都会让整次导入中断。

本包装器在**外层**做重试：
  - 导入脚本内部用 MERGE，天然幂等，重跑不会产生重复数据（前提是没加 --clear）
  - 因此首次用 --clear 建立干净图谱后，后续重试一律走增量 MERGE
  - 每轮结束检查节点数，达标即停，避免无意义的重复写入

用法
----
    python retry_graph_import.py              # 首次：--clear + 最多 5 轮重试
    python retry_graph_import.py --no-clear   # 已有数据，只做增量补写
    python retry_graph_import.py --max-rounds 8
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

PROJECT_ROOT = Path(__file__).resolve().parent
IMPORT_SCRIPT = PROJECT_ROOT / "src" / "database" / "to_neo4j.py"
load_dotenv(PROJECT_ROOT / ".env")

# 期望的节点数下界：CSV 有 8730 行交易 + 2528 行频次，全部导入后节点数必然远超这个值


def count_nodes(timeout: int = 25) -> int | None:
    """连库数节点；网络失败返回 None（不抛异常，交给调用方决定重试）。"""
    uri = (os.getenv("NEO4J_URI") or "").strip()
    user = (os.getenv("NEO4J_USERNAME") or "").strip()
    pwd = (os.getenv("NEO4J_PASSWORD") or "").rstrip("\r\n")
    db = (os.getenv("NEO4J_DATABASE") or "").strip() or None
    try:
        driver = GraphDatabase.driver(uri, auth=(user, pwd), connection_timeout=timeout)
        driver.verify_connectivity()
        with driver.session(database=db) as session:
            value = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        driver.close()
        return value
    except Exception:
        return None


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description="带重试的 Neo4j 知识图谱导入")
    parser.add_argument("--max-rounds", type=int, default=5, help="最多重试轮数，默认 5")
    parser.add_argument("--no-clear", action="store_true", help="不在首轮清空图谱（增量补写模式）")
    parser.add_argument("--target", type=int, default=8000, help="节点数达标即停止，默认 8000")
    args = parser.parse_args()

    print("=" * 70)
    print("带重试的知识图谱导入")
    print(f"  URI     : {os.getenv('NEO4J_URI')}")
    print(f"  数据库  : {os.getenv('NEO4J_DATABASE')}")
    print(f"  最多轮数: {args.max_rounds}，达标阈值: {args.target} 个节点")
    print("=" * 70)

    for round_no in range(1, args.max_rounds + 1):
        print()
        print(f"########## 第 {round_no}/{args.max_rounds} 轮 ##########")

        before = count_nodes()
        print(f"  导入前节点数: {before if before is not None else '(网络失败，无法读取)'}")

        # 首轮按需 --clear；后续轮次一律增量，避免把已写入的数据删掉重来
        cmd = [sys.executable, "-u", str(IMPORT_SCRIPT)]
        if round_no == 1 and not args.no_clear:
            cmd.append("--clear")
            print("  模式: --clear（清空后重建）")
        else:
            print("  模式: 增量 MERGE（幂等，不会产生重复数据）")

        started = time.perf_counter()
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        elapsed = time.perf_counter() - started
        print(f"  脚本退出码: {result.returncode}，耗时 {elapsed:.0f}s")

        after = count_nodes()
        print(f"  导入后节点数: {after if after is not None else '(网络失败，无法读取)'}")

        if result.returncode == 0 and after is not None and after >= args.target:
            print()
            print("=" * 70)
            print(f"成功！知识图谱已导入，共 {after} 个节点（用了 {round_no} 轮）")
            print("=" * 70)
            return 0

        if after is None:
            print("  → 读不到节点数（网络抖动），准备重试")
        elif after < args.target:
            print(f"  → 节点数 {after} 未达阈值 {args.target}，准备重试补写")

        if round_no < args.max_rounds:
            wait = min(5 * round_no, 20)
            print(f"  等待 {wait}s 后重试...")
            time.sleep(wait)

    print()
    print("=" * 70)
    final = count_nodes()
    print(f"达到最大轮数仍未成功。当前节点数: {final}")
    print("建议：改用本机 Neo4j（NEO4J_URI=neo4j://127.0.0.1:7687）以避免跨境网络问题。")
    print("=" * 70)
    return 1


if __name__ == "__main__":
    sys.exit(main())

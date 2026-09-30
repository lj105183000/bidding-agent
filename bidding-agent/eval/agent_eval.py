"""Agent 端到端评测：工具选择准确率 + 延迟。

口径（docs/开发文档.md §6.4/§10）：走非流式 chat()、默认参数（无联网/深度思考）；
tool_name 是"最后一个执行完的工具"单字符串，与 expected_tool 精确相等才算对。
多轮 ReAct 后工具名可能与第一直觉不同——用例标注时就按这个口径来。

用法::

    python eval/agent_eval.py                # 全量
    python eval/agent_eval.py --limit 5      # 快速验证
    python eval/agent_eval.py --verbose      # 附带每题答案全文
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.common import load_cases, mean, run_cases, setup_stdio
from src.agent import BiddingAgent
from src.logging_config import setup_logging

CASES_PATH = Path(__file__).resolve().parent / "test_cases.json"

# main() 里构造；测试直接替换本模块属性（文件内局部 fake 的常规打法）
agent: BiddingAgent | None = None


def eval_one(case: dict) -> dict:
    result = agent.chat(case["question"])
    expected = case["expected_tool"]
    actual = result.get("tool_name", "")
    return {
        "ok": True,
        "match": actual == expected,
        "expected": expected,
        "actual": actual,
        "elapsed_ms": result.get("elapsed_ms", 0),
        "answer": result.get("answer", ""),
    }


def _p95(values: list[int]) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]


def main() -> int:
    global agent
    setup_stdio()
    parser = argparse.ArgumentParser(description="Agent 工具选择准确率 + 延迟评测")
    parser.add_argument("--limit", type=int, default=0, help="只用前 N 条做快速验证")
    parser.add_argument("--verbose", action="store_true", help="打印每题答案全文")
    args = parser.parse_args()

    setup_logging()
    agent = BiddingAgent()
    cases = load_cases(CASES_PATH, required=("question", "expected_tool"))
    if args.limit:
        cases = cases[: args.limit]

    print(f"用例 {len(cases)} 条 | 非流式 chat()，默认参数")
    started = time.perf_counter()
    results = run_cases(cases, eval_one)
    matched = sum(1 for r in results if r.get("match"))
    latencies = [r["elapsed_ms"] for r in results if r.get("ok")]

    print()
    print(f"工具选择准确率：{matched}/{len(results)} = {matched / len(results):.1%}")
    # 用 common.mean 而非裸除法：极端情况下全部题目失败时 latencies 为空，
    # 裸除会 ZeroDivisionError 把整轮评测结果毁在汇总这一步
    print(f"延迟：平均 {mean(latencies):.0f}ms | P95 {_p95(latencies)}ms | 总耗时 {time.perf_counter() - started:.1f}s")

    print("\n===== 每题明细 =====")
    for case, r in zip(cases, results):
        if not r.get("ok"):
            print(f"[失败] {case['question']} -> {r.get('error')}")
        elif r["match"]:
            print(f"[√] {case['question']} -> {r['actual']} ({r['elapsed_ms']}ms)")
        else:
            print(f"[×] {case['question']} -> 期望 {r['expected']}，实际 {r['actual']}")
        if args.verbose and r.get("ok"):
            print(f"    答案：{r['answer'][:120]}{'…' if len(r['answer']) > 120 else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

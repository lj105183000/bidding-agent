"""检索切题率评测：召回的结果里有多少真正回答了问题。

★ 补的是评估体系的哪块缺口
------------------------
`rag.context_recall` 实测 1.000（RAGAS 口径），看起来检索完美。
但那个指标衡量的是"**期望答案里的信息是否出现在上下文中**"——
评测集的 reference 与知识库高度重叠（`docs/23` 记过：25 题里 8 题近乎原题），
所以它测不出"召回了同主题但答非所问的条目"。

实测暴露（本轮）：
    问「投标保证金」（法定 2% / 80 万上限）
    召回「履约保证金」（10% 上限）—— 都是保证金，答案却不同
    模型于是补上参数记忆的 2% / 80万元 / 第二十六条 → 触发编造告警

故本脚本补一个**独立于参考文本**的口径：用精确主题词检索，
人工标注"切题判据"，算 top-k 里真正切题的比例。

★ 判据怎么定的
------------
每个主题给一组关键词（如「投标保证金」必须含"投标保证金"），
命中任一即算切题。刻意用**关键词**而不是语义相似度：
语义相似度会把"履约保证金"也算成切题（它确实语义相近），
而那正是要抓的问题——**用户要的是那一个，不是相近的那一个**。

用法::

    python eval/retrieval_precision.py
    python eval/retrieval_precision.py --top-k 5 --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.common import setup_stdio  # noqa: E402

#: (检索词, 切题判据关键词组, 该主题在语料中的期望规模)
#:
#: 期望规模用于区分两类问题：
#:     切题率低但语料里有 → **检索精度问题**（该调检索）
#:    切题率低且语料里也没有 → **语料覆盖问题**（该补语料）
#: ★ 检索词要用**法条里的规范用语**，不要用口语笼统词。
#:
#: 实测踩过：用「废标」检索，召回里混进「**废**旧钢材拍卖」
#: （向量与 BM25 都会因字面重叠而召回），且判据词表没含「拒收」，
#: 于是切题率算成 0%——但换用规范用语「否决投标」时**榜首就是正确条目**。
#:
#: 这不是检索坏了、也不是语料缺，而是**测量口径的错**：
#:     · 口语词「废标」对应的法定表述是「否决投标」「投标无效」
#:     · 判据词表必须覆盖该主题的**全部**法定表述
#: 故检索词改用「否决投标」，判据含否决/无效/拒收三种表述。
TOPICS: list[tuple[str, tuple[str, ...]]] = [
    ("投标保证金", ("投标保证金",)),
    ("履约保证金", ("履约保证金",)),
    ("中标候选人公示", ("中标候选人",)),
    ("评标委员会", ("评标委员会",)),
    ("资格预审", ("资格预审", "资格审查")),
    ("招标公告", ("招标公告", "招标信息", "公开招标")),
    ("中标通知书", ("中标通知书", "中标结果通知")),
    ("异议", ("异议",)),
    ("投诉", ("投诉",)),
    ("否决投标", ("否决投标", "投标无效", "拒收", "废标")),
    ("联合体投标", ("联合体",)),
    ("分包", ("分包", "转包")),
    ("招标代理", ("代理机构", "招标代理", "采购代理")),
    ("政府采购", ("政府采购",)),
    ("评标方法", ("评标方法", "综合评分", "最低价", "评标标准")),
    ("招标文件澄清", ("澄清", "修改")),
    ("开标", ("开标",)),
    ("合同签订", ("合同",)),
]

#: 切题率低于此值算"该主题检索不达标"
WEAK_THRESHOLD = 0.6


def main() -> int:
    setup_stdio()
    parser = argparse.ArgumentParser(description="检索切题率评测")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    from src.rag.pipeline import RAGPipeline

    pipe = RAGPipeline()
    rows: list[dict] = []
    total_recall = total_relevant = 0

    print(f"检索切题率（top-{args.top_k}）")
    print(f"  {'检索词':<16}{'召回':>5}{'切题':>5}{'切题率':>8}  首条切题条目")
    print("  " + "-" * 84)

    for query, keys in TOPICS:
        try:
            docs = pipe.search(query, top_k=args.top_k) or []
        except Exception as exc:
            print(f"  {query:<16}  检索失败：{type(exc).__name__}")
            rows.append({"query": query, "recall": 0, "relevant": 0,
                         "rate": 0.0, "error": f"{type(exc).__name__}: {exc}"})
            continue

        relevant: list[str] = []
        for d in docs:
            blob = f"{d.get('question') or ''}{d.get('answer') or ''}"
            if any(k in blob for k in keys):
                relevant.append(str(d.get("question") or ""))

        total_recall += len(docs)
        total_relevant += len(relevant)
        rate = len(relevant) / len(docs) if docs else 0.0
        print(f"  {query:<16}{len(docs):>5}{len(relevant):>5}{rate:>7.0%}  "
              f"{(relevant[0][:36] if relevant else '（无切题条目）')}")
        rows.append({
            "query": query, "recall": len(docs), "relevant": len(relevant),
            "rate": round(rate, 4), "first_relevant": relevant[0] if relevant else "",
        })

    print("  " + "-" * 84)
    overall = total_relevant / total_recall if total_recall else 0.0
    print(f"  {'合计':<16}{total_recall:>5}{total_relevant:>5}{overall:>7.0%}")

    weak = [r["query"] for r in rows if r["recall"] and r["rate"] < WEAK_THRESHOLD]
    empty = [r["query"] for r in rows if not r["recall"]]
    print()
    print(f"  切题率 < {WEAK_THRESHOLD:.0%} 的主题（{len(weak)} 个）：{weak}")
    if empty:
        print(f"  **零召回**的主题（{len(empty)} 个）：{empty}")
    print()
    print("  判读：")
    print("    · 切题率低但该主题确有语料 → **检索精度问题**")
    print("    · 零召回或切题条目极少     → **语料覆盖问题**（该补语料）")
    print("  两者都会让模型用参数记忆补答案，进而触发条款号/数字编造。")

    if args.json:
        Path(args.json).write_text(
            json.dumps({"top_k": args.top_k, "overall_rate": round(overall, 4),
                        "weak": weak, "empty": empty, "rows": rows},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n  明细已写入 {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

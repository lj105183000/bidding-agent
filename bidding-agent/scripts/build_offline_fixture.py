"""生成离线检索评测的冻结数据（人工刷新用，不进入 CI）。

★ 为什么需要它
------------
`retrieval_offline` 线要在无 Qdrant、无 LLM、无 `data/` 的 CI 环境里跑，
故必须把语料与查询**冻结**成仓库内的一份 JSON。刷新数据是人工动作，
所以脚本单独放在 `scripts/`，不参与线上导入。

★ 为什么显式传路径
----------------
`load_qa()` 不带路径时会优先扫 `data/processed/`，本机实测会命中
「招标采购标的物信息提取训练数据_2_标的物.xlsx」（8109 行），
不是问答语料。这个陷阱是静默的：条数看着更多，但内容完全错位。

★ qrels 口径
-----------
沿用 `hard_topics` 的关键词判据：文档的「问题+答案」里出现该题任一
关键词即算相关。它不完美（会漏掉语义相关但用词不同的文档），
但**与现有 retrieval 线同源**，且固定后可用于回归比较。
本脚本只产 draft；刷新后应人工抽查 `relevant_ids` 是否离谱。
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.common import setup_stdio  # noqa: E402
from eval.hard_topics import as_pairs as hard_pairs  # noqa: E402
from eval.retrieval_precision import TOPICS  # noqa: E402
from src.rag.ingest import load_qa  # noqa: E402

OUT = ROOT / "eval" / "fixtures" / "retrieval_offline.json"
QA = ROOT / "data" / "qa.xlsx"
SUPPLEMENT = ROOT / "data" / "qa_supplement.xlsx"

#: 从 `retrieval_precision.TOPICS` 里选 10 个宽窄适中的主题。
#: 过滤标准：`question+answer` 命中 3~50 篇。太宽的（如「政府采购」152 篇）
#: 会让 recall@5 的分母失去意义；太窄的（1 篇）则没有区分度。
TOPIC_PICKS: tuple[str, ...] = (
    "投标保证金", "履约保证金", "中标候选人公示", "中标通知书", "异议",
    "投诉", "否决投标", "联合体投标", "分包", "开标",
)


def _sha12(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def _load_corpus() -> list[dict]:
    base = load_qa(QA)
    supp = load_qa(SUPPLEMENT)
    merged: dict[str, dict] = {}
    for entry in base:
        merged[entry["question"]] = {**entry, "source": QA.name}
    for entry in supp:
        merged[entry["question"]] = {**entry, "source": SUPPLEMENT.name}
    return [
        {"id": f"d{i:04d}", **entry}
        for i, entry in enumerate(merged.values(), start=1)
    ]


def _queries(corpus: list[dict]) -> list[dict]:
    pairs = [(q, keys, "hard") for q, keys in hard_pairs()]
    by_topic = {q: keys for q, keys in TOPICS}
    for name in TOPIC_PICKS:
        if name not in by_topic:
            raise SystemExit(f"TOPICS 里没有主题：{name}")
        pairs.append((name, by_topic[name], "topic"))

    queries: list[dict] = []
    for i, (query, keys, category) in enumerate(pairs, start=1):
        relevant = [
            doc["id"]
            for doc in corpus
            if any(key in (doc["question"] + doc["answer"]) for key in keys)
        ]
        if not relevant:
            raise SystemExit(f"查询没有任何相关文档：{query} keys={keys}")
        queries.append({
            "id": f"q{i:02d}",
            "query": query,
            "keys": list(keys),
            "relevant_ids": relevant,
            "category": category,
        })
    return queries


def _validate(fixture: dict) -> None:
    corpus, queries = fixture["corpus"], fixture["queries"]
    doc_ids = [d["id"] for d in corpus]
    questions = [d["question"] for d in corpus]
    if len(doc_ids) != len(set(doc_ids)):
        raise SystemExit("corpus id 重复")
    if len(questions) != len(set(questions)):
        raise SystemExit("corpus question 重复")
    if len(queries) != 40:
        raise SystemExit(f"查询数必须为 40，实际 {len(queries)}")
    seen_q = set()
    for q in queries:
        if q["id"] in seen_q:
            raise SystemExit(f"query id 重复：{q['id']}")
        seen_q.add(q["id"])
        if not q["relevant_ids"]:
            raise SystemExit(f"query 没有相关文档：{q['id']}")
        missing = [d for d in q["relevant_ids"] if d not in set(doc_ids)]
        if missing:
            raise SystemExit(f"{q['id']} 有悬挂 relevant_id：{missing[:3]}")


def main() -> int:
    setup_stdio()
    corpus = _load_corpus()
    fixture = {
        "schema_version": 1,
        "name": "retrieval_offline_v1",
        "source": {
            "qa_sha256_12": _sha12(QA),
            "qa_supplement_sha256_12": _sha12(SUPPLEMENT),
            "builder": "scripts/build_offline_fixture.py",
        },
        "corpus": corpus,
        "queries": _queries(corpus),
    }
    _validate(fixture)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(fixture, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(f"已写入 {OUT}（{len(corpus)} 文档 / {len(fixture['queries'])} 查询）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

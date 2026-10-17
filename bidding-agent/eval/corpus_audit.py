"""语料真值审计：把「难例集的判据」与「库里到底有没有据」对齐。

★ 为什么必须有这个工具（实测踩到，且不是一次性的）
==========================================
难例集里"前 5 条全不切题"有两种**修法相反**的成因：

    A. 库里**有**那条，检索没捞上来   → 真·召回失手（改检索）
    B. 库里**根本没有**这条知识       → 知识库缺口（补语料）

第一版只看了检索输出，把两种混为一谈，于是差点得出"召回率有问题"的
错误结论去调检索参数。全量扫一遍语料（集合只有几百个点，一次 scroll 就够）
才把成因分开 —— 14 题里 1 题属于 B。

★ 为什么做成常驻工具而不是一次性脚本
==============================
    · **语料会变**：补进文档后，B 类题应自动变成 A 类（或直接通过）；
      半年后没人记得当初为什么把某题标成"库内无据"
    · **新增题目要过闸**：往难例集里加题时，必须先确认"库内有据"，
      否则加的是一道**永远测不出检索能力**的题
    · 手工核对十几个关键词命中数很容易漏，机器扫 500+ 条是几秒的事

用法::

    python -m eval.corpus_audit                  # 审计难例集，列出每题的命中数
    python -m eval.corpus_audit --json out.json  # 落盘，便于对比语料更新前后
    python -m eval.corpus_audit --query "关键词"  # 临时查某个词在库里的分布

输出里每题会标 `✅ 库内有据` / `❌ 库内无据`，并给出命中篇数与示例问题 ——
**核对的是"库里有没有"，与检索结果无关**（这正是它的价值：独立于被测系统）。
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent


@dataclass
class Doc:
    """语料里的一条，只保留判据需要的最小字段。"""

    doc_id: object
    question: str
    answer: str

    @property
    def blob(self) -> str:
        return f"{self.question}{self.answer}"


def scan_corpus(store=None) -> list[Doc]:
    """全量拉取语料（scroll 分页）。失败抛异常 —— 审计拿不到真值就该明显失败。

    ★ 为什么不复用 `RAGPipeline.search`：那正是**被测对象**。
    用检索接口来确认"库里有据"是循环论证（检索没捞上来 → 误判成库里没有）。
    必须绕开检索，直接读存储。集合规模是几百条，全量扫的成本可以接受。
    """
    if store is None:
        from src.rag.pipeline import rag_pipeline

        store = rag_pipeline.store
    client = store.client
    docs: list[Doc] = []
    offset = None
    while True:
        batch, offset = client.scroll(
            collection_name=store.collection,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in batch:
            payload = point.payload or {}
            docs.append(
                Doc(
                    doc_id=point.id,
                    question=str(payload.get("question") or ""),
                    answer=str(payload.get("answer") or ""),
                )
            )
        if offset is None:
            break
    return docs


def audit(items, docs: list[Doc] | None = None) -> dict:
    """审计 `[(query, keys)]`：每题的判据在语料里命中多少篇。

    返回 `{"corpus_size", "items": [{query, keys, hits, examples, has_evidence}]}`。
    """
    docs = scan_corpus() if docs is None else docs
    out = []
    for query, keys in items:
        hits = [d for d in docs if any(k in d.blob for k in keys)]
        out.append(
            {
                "query": query,
                "keys": list(keys),
                "hits": len(hits),
                "examples": [d.question[:60] for d in hits[:3]],
                "has_evidence": bool(hits),
            }
        )
    return {
        "corpus_size": len(docs),
        "items": out,
        "with_evidence": sum(1 for i in out if i["has_evidence"]),
        "corpus_gaps": [i["query"] for i in out if not i["has_evidence"]],
    }


def render(result: dict) -> str:
    lines = [
        "=" * 78,
        f"语料真值审计（语料 {result['corpus_size']} 条）",
        "=" * 78,
    ]
    for item in result["items"]:
        flag = "✅ 库内有据" if item["has_evidence"] else "❌ 库内无据（知识库缺口）"
        lines.append(f"{flag}  命中 {item['hits']:>3} 篇｜{item['query']}")
        lines.append(f"          判据 {item['keys']}")
        for example in item["examples"]:
            lines.append(f"          · {example}")
    lines.append("-" * 78)
    lines.append(
        f"有据 {result['with_evidence']}/{len(result['items'])}；"
        f"库内无据 {len(result['corpus_gaps'])} 题"
    )
    if result["corpus_gaps"]:
        lines.append("  → 这些题**不该算检索失败**，应标进 hard_topics.KNOWN_CORPUS_GAPS")
    lines.append("=" * 78)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

    try:  # 独立跑时自己读 .env（应用启动时由 api/server.py 读）
        from dotenv import load_dotenv

        from src.config import PROJECT_ROOT

        load_dotenv(PROJECT_ROOT / ".env")
    except Exception:  # pragma: no cover
        pass

    if "--query" in args:
        keyword = args[args.index("--query") + 1]
        docs = scan_corpus()
        hits = [d for d in docs if keyword in d.blob]
        print(f"关键词 {keyword!r}：命中 {len(hits)}/{len(docs)} 篇")
        for d in hits[:10]:
            print(f"  · {d.question[:70]}")
        return 0

    sys.path.insert(0, str(EVAL_DIR))
    from hard_topics import HARD_TOPICS  # type: ignore[import-not-found]

    result = audit([(q, keys) for _label, q, keys in HARD_TOPICS])
    if "--json" in args:
        idx = args.index("--json")
        target = args[idx + 1] if idx + 1 < len(args) else ""
        payload = json.dumps(result, ensure_ascii=False, indent=2)
        if target:
            Path(target).write_text(payload, encoding="utf-8")
            print(f"已写入 {target}")
        else:
            print(payload)
    else:
        print(render(result))
    return 0


__all__ = ["Doc", "audit", "render", "scan_corpus"]


if __name__ == "__main__":
    raise SystemExit(main())

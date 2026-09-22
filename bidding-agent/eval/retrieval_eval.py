"""检索质量评测：对比无精排与各精排策略的 top-1 / top-3 命中率。

**为什么需要这个脚本**：精排到底有没有用、"查询该和哪个字段配对打分"这类问题，
靠肉眼看几条结果是判断不了的——必须先混合检索，再叠加不同的二次排序，用同一批
查询跑出可比数字。

**在 75 条样例语料上的实测结论（2026-09，见 docs/技术栈.md 踩坑记录）**：

============  ==================  =======  =======
查询风格       策略                 Top-1    MRR
============  ==================  =======  =======
keyword       不精排               98.7%    0.993
keyword       精排（问题字段）       98.7%    0.993
keyword       精排（答案字段）       78.7%    0.863   ← 有害
paraphrase    不精排               98.7%    0.993
paraphrase    精排（问题字段）       98.7%    0.993
paraphrase    精排（答案字段）       89.3%    0.937   ← 有害
============  ==================  =======  =======

两点结论：① 混合检索基线已接近天花板（Top-5 100%），精排**没有提升空间**，
故 `RERANK_ENABLED` 默认关闭；② **配对字段选错会显著变差**——用答案配对相当于
拿短问句去比长说明文，CrossEncoder 会偏向文体相似的错误候选。

**口径与局限**：知识库没有独立标注的 query 集，这里的查询由问题字段机械改写而来
（见 `--query-style`）。`exact` 几乎等于把答案原文当查询，命中率必然很高、没有区分度；
`keyword` 才是接近真实用户的那种"几个词丢进去"的问法，**策略差异主要看这一档**。
结论只用于横向对比不同策略，不代表线上绝对指标。

默认在**内存集合**上跑：同一批向量、同一套 RRF 代码，省掉数百次云端往返（分钟级
降到秒级），也因此可以放进 CI。要校验线上索引本身时用 `--use-cloud`。

用法::

    python eval/retrieval_eval.py                          # 全部策略 × 两种查询风格
    python eval/retrieval_eval.py --query-style keyword
    python eval/retrieval_eval.py --use-cloud --mode rerank_question
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client import QdrantClient, models  # noqa: E402

from src.config import settings  # noqa: E402
from src.logging_config import setup_logging  # noqa: E402
from src.rag.constants import DEFAULT_TOP_K, PREFETCH_LIMIT  # noqa: E402
from src.rag.embedder import BM25Encoder, _tokenize, embedder, reranker  # noqa: E402
from src.rag.ingest import load_qa  # noqa: E402
from src.rag.vector_store import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME, VectorStore  # noqa: E402

# 候选池：精排只能重排它看到的这些，池子太小会限制上限
CANDIDATE_K = 20

# 精排策略 → (说明, 取哪个字段与查询配对)
RERANK_MODES = {
    "none": ("不精排（混合检索原始 RRF 顺序）", None),
    "rerank_answer": ("精排：查询 × 答案", lambda c: c["answer"]),
    "rerank_question": ("精排：查询 × 知识库问题", lambda c: c["question"]),
    "rerank_both": ("精排：查询 × (问题+答案)", lambda c: f"{c['question']}\n{c['answer']}"),
}

# 提取关键词时丢弃的高频虚词：它们不承载检索意图，留着只会稀释关键词信号
STOPWORDS = {
    "什么", "怎么", "怎样", "如何", "哪些", "哪个", "可以", "能否", "是否", "应该",
    "需要", "要求", "关于", "对于", "以及", "或者", "还是", "我们", "他们", "这个",
    "那个", "一个", "进行", "情况", "问题", "请问", "如果", "那么", "就是", "时候",
}


PARAPHRASE_CACHE = Path(__file__).resolve().parent / "paraphrase_cache.json"

_PARAPHRASE_PROMPT = """你是招投标采购领域的从业者。请把下面每个知识库问题改写成**用户会真正问出口的话**。

要求：
1. 不要照抄原问题的措辞，换一种日常、口语的说法；
2. 保持原问题的意图与关键信息（主体、情形、动作）不变；
3. 每条只输出一行，格式严格为 `编号|改写后的问题`，不要任何额外说明。

问题列表：
{items}"""


def build_paraphrases(entries: list[dict], refresh: bool = False) -> list[str]:
    """用 LLM 把知识库问题改写成口语化提问，作为更接近真实分布的查询集。

    这是评估精排价值的**关键**：原文查询的基线已接近天花板（99%+），没有提升空间，
    只有换成"措辞不同但意图相同"的查询，才能看出二次排序是否真的把正确答案顶上来。

    结果缓存到 `eval/paraphrase_cache.json`：LLM 有随机性，缓存才能让前后两次对比
    建立在同一批查询上，否则指标波动会掩盖策略差异。
    """
    if PARAPHRASE_CACHE.exists() and not refresh:
        cached = json.loads(PARAPHRASE_CACHE.read_text(encoding="utf-8"))
        if len(cached) == len(entries):
            return cached

    from src.clients.llm_factory import get_llm_client

    client = get_llm_client()
    if not client.available():
        raise SystemExit("改写查询需要 LLM 凭据；或改用 --query-style exact/keyword")

    questions = [e["question"] for e in entries]
    paraphrases: list[str] = []
    batch_size = 5
    for start in range(0, len(questions), batch_size):
        batch = questions[start : start + batch_size]
        items = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(batch))
        reply = client.chat(
            [{"role": "user", "content": _PARAPHRASE_PROMPT.format(items=items)}]
        )
        parsed: dict[int, str] = {}
        for line in reply.splitlines():
            if "|" not in line:
                continue
            head, _, text = line.partition("|")
            digits = "".join(ch for ch in head if ch.isdigit())
            if digits and text.strip():
                parsed[int(digits)] = text.strip()
        # 解析失败的条目退回原问题：宁可少一条改写，也不要让整轮评测中断
        paraphrases.extend(parsed.get(i + 1, q) for i, q in enumerate(batch))
        print(f"  改写进度 {min(start + batch_size, len(questions))}/{len(questions)}", end="\r")

    print()
    PARAPHRASE_CACHE.write_text(json.dumps(paraphrases, ensure_ascii=False, indent=1), encoding="utf-8")
    return paraphrases


_paraphrase_cache_mem: list[str] | None = None
_refresh_paraphrases = False


def build_queries(entries: list[dict], style: str) -> list[str]:
    """按风格把知识库问题改写成查询。"""
    global _paraphrase_cache_mem
    queries = []
    for entry in entries:
        question = entry["question"]
        if style == "exact":
            queries.append(question)
        elif style == "keyword":
            tokens = [t for t in _tokenize(question) if t not in STOPWORDS]
            # 关键词全被过滤掉时退回原问题，避免产生空查询
            queries.append(" ".join(tokens) if tokens else question)
        elif style == "paraphrase":
            # 整批只生成一次：不同策略必须跑在同一批查询上
            if _paraphrase_cache_mem is None:
                _paraphrase_cache_mem = build_paraphrases(entries, refresh=_refresh_paraphrases)
            queries = _paraphrase_cache_mem[: len(entries)]
            break
        else:
            raise ValueError(f"未知的查询风格: {style}")
    return queries


def build_local_index(entries: list[dict]) -> tuple[VectorStore, BM25Encoder]:
    """用同一批数据在内存里重建索引：向量、分词、RRF 与线上完全同源，
    只是省掉了网络往返。"""
    questions = [e["question"] for e in entries]
    bm25 = BM25Encoder().fit(questions)
    dense_vectors = embedder.encode_documents(questions)
    sparse_vectors = bm25.encode_documents(questions)

    store = VectorStore(
        url=":memory:",
        collection="eval_local",
        dim=len(dense_vectors[0]),
        client=QdrantClient(":memory:"),
    )
    store.ensure_collection()
    store.upsert(
        [
            models.PointStruct(
                id=i,
                vector={DENSE_VECTOR_NAME: dense_vectors[i], SPARSE_VECTOR_NAME: sparse_vectors[i]},
                payload={"question": questions[i], "answer": entries[i]["answer"]},
            )
            for i in range(len(questions))
        ]
    )
    return store, bm25


def build_cloud_index(entries: list[dict]) -> tuple[VectorStore, BM25Encoder]:
    """连线上 Qdrant；词表仍从本地 vocab.json 读，保证查询侧编码口径一致。"""
    return (
        VectorStore(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            collection=settings.qdrant_collection,
            dim=settings.qdrant_vector_size,
            distance=settings.qdrant_distance,
            timeout=settings.qdrant_timeout,
        ),
        BM25Encoder.load(),
    )


class Retriever:
    """封装一次评测所需的检索依赖；候选池按 (style, query) 记忆化。

    记忆化不是为了省时间那么简单——若每个策略各自重新检索，各策略看到的候选池
    可能因索引/网络抖动而不同，对比就失去意义。
    """

    def __init__(self, entries: list[dict], use_cloud: bool = False) -> None:
        self.entries = entries
        self.store, self.bm25 = build_cloud_index(entries) if use_cloud else build_local_index(entries)
        self._cache: dict[tuple[str, str], list[dict]] = {}

    def candidates(self, style: str, query: str) -> list[dict]:
        key = (style, query)
        if key not in self._cache:
            dense = embedder.encode_query(query)
            sparse = self.bm25.encode_query(query)
            self._cache[key] = self.store.hybrid_search(
                dense, sparse, limit=CANDIDATE_K, prefetch_limit=PREFETCH_LIMIT
            )
        return self._cache[key]


def apply_mode(mode: str, query: str, candidates: list[dict]) -> list[dict]:
    key_fn = RERANK_MODES[mode][1]
    if key_fn is None:
        return candidates
    pairs = [(query, key_fn(c) or c["question"]) for c in candidates]
    scores = reranker.model.predict(pairs)
    order = sorted(range(len(candidates)), key=lambda i: float(scores[i]), reverse=True)
    return [candidates[i] for i in order]


def score(retriever: Retriever, mode: str, style: str, top_k: int, limit: int) -> dict:
    entries = retriever.entries[:limit] if limit else retriever.entries
    queries = build_queries(entries, style)

    hits_top1 = hits_topk = 0
    ranks: list[int] = []
    started = time.perf_counter()

    for entry, query in zip(entries, queries):
        gold = entry["question"]
        ordered = apply_mode(mode, query, retriever.candidates(style, query))
        rank = next((i + 1 for i, c in enumerate(ordered) if c["question"] == gold), None)
        if rank is None:
            continue
        ranks.append(rank)
        hits_top1 += rank == 1
        hits_topk += rank <= top_k

    total = len(entries)
    return {
        "n": total,
        "top1": hits_top1 / total,
        "topk": hits_topk / total,
        "mrr": sum(1 / r for r in ranks) / total if ranks else 0.0,
        "seconds": time.perf_counter() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG 检索质量评测")
    parser.add_argument("--mode", choices=list(RERANK_MODES), action="append")
    parser.add_argument("--query-style", choices=["exact", "keyword", "paraphrase"], action="append")
    parser.add_argument(
        "--refresh-paraphrases",
        action="store_true",
        help="重新生成口语化查询（默认复用 eval/paraphrase_cache.json）",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--limit", type=int, default=0, help="只用前 N 条做快速验证")
    parser.add_argument("--use-cloud", action="store_true", help="改用线上 Qdrant（慢，但校验真实索引）")
    args = parser.parse_args()

    global _refresh_paraphrases
    _refresh_paraphrases = args.refresh_paraphrases

    setup_logging()
    modes = args.mode or list(RERANK_MODES)
    styles = args.query_style or ["keyword", "exact"]

    entries = load_qa()
    print(f"知识库 {len(entries)} 条 | 索引来源={'线上 Qdrant' if args.use_cloud else '内存重建'}")

    retriever = Retriever(entries, use_cloud=args.use_cloud)
    if any(m != "none" for m in modes):
        reranker.model  # noqa: B018 - 触发懒加载，让加载耗时落在计时之外
        print(f"精排模型已加载（生产开关 RERANK_ENABLED={int(settings.rerank_enabled)}）")
    print()

    header = f"{'查询风格':<12}{'策略':<20}{'Top-1':>7}{'Top-' + str(args.top_k):>8}{'MRR':>8}{'耗时':>9}"
    print(header)
    print("-" * 64)
    for style in styles:
        for mode in modes:
            result = score(retriever, mode, style, args.top_k, args.limit)
            print(
                f"{style:<12}{mode:<20}{result['top1']:>7.1%}{result['topk']:>8.1%}"
                f"{result['mrr']:>8.3f}{result['seconds']:>8.1f}s"
            )
        print()

    print("提示：exact 风格几乎等于原文查询，命中率高且无区分度；策略差异主要看 keyword 档。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

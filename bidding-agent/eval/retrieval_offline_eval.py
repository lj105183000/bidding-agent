"""离线检索评测：冻结语料 + 显式 qrels，不连 Qdrant、不调 LLM。

★ 为什么需要一条独立线
-------------------
现有 `retrieval` 线用 `RAGPipeline()` 连线上 Qdrant，`rag` 线还要裁判 LLM，
两者都无法在 PR 的干净 CI 里跑。本线把语料与相关标注**冻结**成
`eval/fixtures/retrieval_offline.json`，只在内存里重建索引，因此：

    · 不读 `data/`（该目录被 .gitignore 忽略，CI checkout 后不存在）
    · 不连 Qdrant、不调 LLM、不需要任何密钥
    · 唯一外部依赖是 BGE 嵌入模型（CI 用 HF 缓存）

★ 为什么用 qrels 而不是 `ranking_metrics()`
----------------------------------------
`ranking_metrics()` 靠关键词在 `question+answer` 里匹配判相关，适合现有
「同一批数据跑不同精排策略」的横向对比；但它无法回答「相关文档到底有几条」，
也就算不出真正的 recall@k。离线固定集的价值是**可审计的相关集**：
每题显式列出 `relevant_ids`，指标只依赖这份标注。

本模块的 qrels 由 `scripts/build_offline_fixture.py` 用关键词初筛生成，
口径与 `hard_topics` 同源；刷新 fixture 后应人工抽查。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from eval.report import SuiteReport, make_result

DEFAULT_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "retrieval_offline.json"

#: 指标名前缀。与现有 `retrieval.*` 明确分开，避免趋势文件里混线。
METRIC_PREFIX = "retrieval_offline"

#: qrels 指标的默认 top-k。
DEFAULT_TOP_K = 5


@dataclass(frozen=True)
class OfflineQuery:
    id: str
    query: str
    relevant_ids: tuple[str, ...]
    category: str = ""


@dataclass(frozen=True)
class OfflineFixture:
    schema_version: int
    name: str
    corpus: tuple[dict, ...]
    queries: tuple[OfflineQuery, ...]
    source: dict


def _fail(message: str) -> ValueError:
    """统一个可读错误：fixture 是人工维护的，报错必须指出具体哪一条。"""
    return ValueError(f"离线检索数据不合法：{message}")


def load_fixture(path: Path | None = None) -> OfflineFixture:
    """读取并校验 fixture。任何结构问题都抛 `ValueError`（带可读原因）。"""
    target = Path(path) if path is not None else DEFAULT_FIXTURE
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise _fail(f"文件不存在：{target}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise _fail(f"读取失败：{target}（{type(exc).__name__}）") from exc

    if not isinstance(raw, dict):
        raise _fail("顶层必须是对象")
    if raw.get("schema_version") != 1:
        raise _fail(f"schema_version 必须为 1，实际 {raw.get('schema_version')!r}")

    corpus_raw = raw.get("corpus")
    queries_raw = raw.get("queries")
    if not isinstance(corpus_raw, list) or not corpus_raw:
        raise _fail("corpus 必须是非空数组")
    if not isinstance(queries_raw, list) or not queries_raw:
        raise _fail("queries 必须是非空数组")

    corpus: list[dict] = []
    doc_ids: set[str] = set()
    questions: set[str] = set()
    for i, doc in enumerate(corpus_raw):
        if not isinstance(doc, dict):
            raise _fail(f"corpus[{i}] 不是对象")
        doc_id = str(doc.get("id") or "").strip()
        question = str(doc.get("question") or "").strip()
        answer = str(doc.get("answer") or "").strip()
        if not doc_id or not question or not answer:
            raise _fail(f"corpus[{i}] 缺 id/question/answer")
        if doc_id in doc_ids:
            raise _fail(f"corpus id 重复：{doc_id}")
        if question in questions:
            raise _fail(f"corpus question 重复：{question[:30]}")
        doc_ids.add(doc_id)
        questions.add(question)
        corpus.append({"id": doc_id, "question": question, "answer": answer, **{
            k: v for k, v in doc.items() if k not in {"id", "question", "answer"}
        }})

    queries: list[OfflineQuery] = []
    query_ids: set[str] = set()
    query_texts: set[str] = set()
    for i, item in enumerate(queries_raw):
        if not isinstance(item, dict):
            raise _fail(f"queries[{i}] 不是对象")
        qid = str(item.get("id") or "").strip()
        text = str(item.get("query") or "").strip()
        relevant = item.get("relevant_ids")
        if not qid or not text:
            raise _fail(f"queries[{i}] 缺 id/query")
        if not isinstance(relevant, list) or not relevant:
            raise _fail(f"{qid} 的 relevant_ids 必须是非空数组")
        if qid in query_ids:
            raise _fail(f"query id 重复：{qid}")
        if text in query_texts:
            raise _fail(f"query 文本重复：{text[:30]}")
        missing = [str(d) for d in relevant if str(d) not in doc_ids]
        if missing:
            raise _fail(f"{qid} 引用了不存在的文档：{missing[:3]}")
        query_ids.add(qid)
        query_texts.add(text)
        queries.append(OfflineQuery(
            id=qid,
            query=text,
            relevant_ids=tuple(dict.fromkeys(str(d) for d in relevant)),
            category=str(item.get("category") or ""),
        ))

    return OfflineFixture(
        schema_version=1,
        name=str(raw.get("name") or target.stem),
        corpus=tuple(corpus),
        queries=tuple(queries),
        source=raw.get("source") if isinstance(raw.get("source"), dict) else {},
    )


def qrels_metrics(
    ranked_ids: Sequence[Sequence[str]],
    queries: Sequence[OfflineQuery],
    *,
    top_k: int = DEFAULT_TOP_K,
) -> dict:
    """按显式 qrels 算检索指标。

    ★ 分母纪律：所有指标的分母都是**全部 query**。某题检索抛异常时调用方传空
    列表，它按未命中计 0 分，而不是从分母里消失——否则系统越坏分数越好看。
    """
    n = len(queries)
    if n == 0:
        return {
            "cases": 0,
            "top1_relevant_rate": None,
            "recall_at_k": None,
            "precision_at_k": None,
            "mrr": None,
            "no_relevant_in_topk_rate": None,
        }

    k = max(1, int(top_k))
    top1_hits = 0
    recall_sum = 0.0
    precision_sum = 0.0
    mrr_sum = 0.0
    no_relevant = 0

    for i, query in enumerate(queries):
        ranked = list(ranked_ids[i]) if i < len(ranked_ids) else []
        relevant = set(query.relevant_ids)
        top = ranked[:k]

        if top and top[0] in relevant:
            top1_hits += 1

        hit = relevant.intersection(top)
        recall_sum += len(hit) / len(relevant) if relevant else 0.0
        precision_sum += len(hit) / k
        if not hit:
            no_relevant += 1

        first_rank = next(
            (rank for rank, doc_id in enumerate(ranked, start=1) if doc_id in relevant),
            None,
        )
        if first_rank is not None:
            mrr_sum += 1.0 / first_rank

    return {
        "cases": n,
        "top1_relevant_rate": round(top1_hits / n, 4),
        "recall_at_k": round(recall_sum / n, 4),
        "precision_at_k": round(precision_sum / n, 4),
        "mrr": round(mrr_sum / n, 4),
        "no_relevant_in_topk_rate": round(no_relevant / n, 4),
    }


def _default_retriever_factory(entries: list[dict]):
    """延迟导入重型依赖：只有真正跑离线线时才加载 qdrant/embedder。

    `run_all.py --list-suites` 等早退路径不应因为新增一条线而变慢。
    """
    from eval.retrieval_eval import Retriever

    return Retriever(entries, use_cloud=False)


def run_offline_suite(
    limit: int = 0,
    *,
    fixture_path: Path | None = None,
    retriever_factory: Callable[[list[dict]], object] | None = None,
) -> SuiteReport:
    """跑 `retrieval_offline` 线并返回 `SuiteReport`。**永不抛异常**。"""
    suite = SuiteReport(suite=METRIC_PREFIX)
    try:
        fixture = load_fixture(fixture_path)
    except Exception as exc:  # noqa: BLE001 - eval 工具不该因数据问题崩掉
        suite.note = f"离线检索数据加载失败：{type(exc).__name__}: {exc}"
        return suite

    queries = list(fixture.queries)
    if limit:
        queries = queries[:limit]
    entries = [
        {"question": doc["question"], "answer": doc["answer"]}
        for doc in fixture.corpus
    ]
    doc_id_by_question = {doc["question"]: doc["id"] for doc in fixture.corpus}

    try:
        retriever = (retriever_factory or _default_retriever_factory)(entries)
    except Exception as exc:  # noqa: BLE001 - 缺模型/索引失败记 NOT_RUN
        suite.note = f"离线检索器初始化失败：{type(exc).__name__}: {exc}"
        return suite

    ranked_ids: list[list[str]] = []
    errors = 0
    for query in queries:
        try:
            docs = retriever.candidates("exact", query.query)
            ranked_ids.append([
                doc_id_by_question[doc["question"]]
                for doc in docs
                if isinstance(doc, dict) and doc.get("question") in doc_id_by_question
            ])
        except Exception:  # noqa: BLE001 - 单题失败计入分母，不中断整轮
            errors += 1
            ranked_ids.append([])

    # top_k 在这里固定为 5：指标名（`recall_at_5` 等）与 MetricSpec 都按 5 注册，
    # 允许调用方改 k 会让名字与口径悄悄不一致。需要别的 k 时请先改注册表。
    metrics = qrels_metrics(ranked_ids, queries, top_k=DEFAULT_TOP_K)
    suite.case_count = len(queries)
    suite.ran = True
    if errors:
        suite.note = f"{errors}/{len(queries)} 题检索异常（按未命中计 0 分）"
    suite.results.extend([
        make_result(
            f"{METRIC_PREFIX}.top1_relevant_rate",
            metrics["top1_relevant_rate"],
            note=f"fixture={fixture.name}",
        ),
        make_result(
            f"{METRIC_PREFIX}.recall_at_{DEFAULT_TOP_K}",
            metrics["recall_at_k"],
            note=f"相关集大小中位数见 fixture；分母为全部 {len(queries)} 题",
        ),
        make_result(
            f"{METRIC_PREFIX}.precision_at_{DEFAULT_TOP_K}",
            metrics["precision_at_k"],
        ),
        make_result(
            f"{METRIC_PREFIX}.mrr",
            metrics["mrr"],
        ),
        make_result(
            f"{METRIC_PREFIX}.no_relevant_in_top{DEFAULT_TOP_K}_rate",
            metrics["no_relevant_in_topk_rate"],
        ),
    ])
    return suite


__all__ = [
    "DEFAULT_FIXTURE",
    "DEFAULT_TOP_K",
    "METRIC_PREFIX",
    "OfflineFixture",
    "OfflineQuery",
    "load_fixture",
    "qrels_metrics",
    "run_offline_suite",
]

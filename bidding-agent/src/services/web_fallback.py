"""统一联网搜索 + 本地知识库充足度判断（自动兜底的数据层）。

为什么需要这一层
--------------
项目里已有两路搜索客户端（Tavily 的 `src/web_search.py`、智谱的 `src/web_search_zhipu.py`），
但它们的返回**形状不同**：

    Tavily  返回 `score`（0~1 相关度）——可直接用
    智谱    不返回分数——无法按相关性排序，也无法判断"搜到的东西相不相关"

而"前端显示每个结果的相关性评分"要求**两路都有可比的分数**。故本模块做两件事：

    1. 把两路结果归一成同一形状，**智谱的结果用余弦相似度补算分数**
       （与法规检索同一套做法：向量相似度可直接与阈值比较，而 RRF 融合分不能）
    2. 提供"本地知识库是否够用"的判断，供 Agent 自动决定要不要联网

★ 判断"本地无答案"的依据是**稠密余弦相似度**，不是 RRF 融合分
-----------------------------------------------------------
实测（见 `rag/vector_store.py` 的说明）：RRF 分只反映候选内排名，
**无关问题的融合分能高于相关问题**。用它判断"知识库够不够用"会误判：
把一个知识库明明能答的问题判成"答不了"从而多余地联网。

故用 `relevance`（与查询的稠密余弦相似度）。阈值沿用 75 条语料下标定的
`vector_store.MIN_RELEVANCE = 0.5`。

⚠️ 这个阈值与语料规模强相关
------------------------
无关问题的最高相似度会随候选增多而上升，故语料显著扩充后必须重新标定。
`local_coverage()` 会把实际得分与阈值一并返回，便于观察与重新校准。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from src.logging_config import get_logger

logger = get_logger(__name__)

# 与 vector_store.MIN_RELEVANCE 保持一致。这里显式重复一次而不是 import，
# 是因为本模块要能在 RAG 未就绪（Qdrant 不可用）时独立工作，
# 不应因为 import 失败而整体不可用。
DEFAULT_MIN_RELEVANCE = 0.5

# 联网结果的默认条数。5 条是实测的平衡点：
# 少于 3 条常常凑不出完整信息，多于 8 条会把上下文塞满噪声。
DEFAULT_WEB_TOP_K = 5

# 智谱结果补算分数时的截断长度。
# 太短会让不同网页的向量趋同（都在说同类话题），太长则慢且无增益。
SCORE_TEXT_CHARS = 400


class WebSearchUnavailable(RuntimeError):
    """联网搜索不可用（未配密钥 / 网络失败 / 库未就绪）。

    与"搜不到结果"区分：前者是**能力问题**（要提示用户配密钥），
    后者是**数据问题**（换个说法再搜）。混为一谈会让用户以为系统坏了。
    """


@dataclass
class SearchHit:
    """一条联网结果（两路搜索归一后的形状）。"""

    title: str
    url: str
    content: str
    score: float | None = None       # 0~1 相关度
    published: str = ""              # 发布时间（智谱给，Tavily 不给）
    source: str = ""                 # "tavily" | "zhipu"
    score_kind: str = ""             # "engine"（引擎自带）| "cosine"（本地补算）

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "content": self.content,
            "score": self.score,
            "published": self.published,
            "source": self.source,
            "score_kind": self.score_kind,
        }


@dataclass
class LocalCoverage:
    """本地知识库对某问题的覆盖程度。"""

    enough: bool                     # 是否足够回答（best_relevance >= 阈值）
    best_relevance: float | None     # 最高相关度（None 表示知识库不可用/为空）
    threshold: float
    hits_count: int = 0
    reason: str = ""                 # 判定说明（面向用户）
    sources: list[dict[str, Any]] = field(default_factory=list)

    # 第二层（LLM 判读）的结果。`judged=False` 表示**没做** LLM 判读，
    # 此时 `enough` 只反映相似度门槛，可信度较低。
    judged: bool = False
    judge_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "enough": self.enough,
            "best_relevance": self.best_relevance,
            "threshold": self.threshold,
            "hits_count": self.hits_count,
            "reason": self.reason,
            "judged": self.judged,
            "judge_reason": self.judge_reason,
        }


# LLM 判读用的提示词。
#
# ★ 为什么必须让模型判读"能不能回答"，而不是只看相似度
# --------------------------------------------------
# 实测反例（用户给的例子）：
#     问题："中国大陆和香港的招投标法规的区别是什么？"
#     本地库全是**大陆**招投标问答，没有香港法规
#     但"招投标法规"这个主题让余弦相似度达到 0.55，过了 0.5 门槛
#     → 判定为"本地够用"，于是用大陆法规回答一个需要香港法规的问题
#
# 根因：余弦相似度衡量的是**主题相关**，不是**能否回答**。
# "都是招投标"和"能回答这个具体问题"是两件事，前者远比后者容易满足。
#
# 故相似度只作**粗筛**（过滤掉明显无关的），过筛后再让模型做一次
# 针对性的判读。多花一次 LLM 调用，换来"该联网时真的联网"。
_JUDGE_PROMPT = """你要判断"以下资料能否回答用户的问题"。

严格要求：
- 只有当资料**直接包含**回答问题所需的关键信息时，才回答"能"。
- 以下情况一律回答"不能"：
  · 资料只涉及同一主题，但没有回答问题所问的具体方面
    （例：问"中国与香港法规的区别"，资料只有中国法规 → 不能）
  · 资料缺少问题中明确要求的对象、地区、时间或比较项
  · 需要靠推测或外部知识才能补全答案
- 拿不准时回答"不能"——让系统去联网查证，比用不足的资料作答更安全。

只输出一行 JSON，不要任何其他文字：
{"can_answer": true 或 false, "reason": "不超过 40 字的理由"}"""


def _judge_by_llm(
    question: str,
    sources: Sequence[dict[str, Any]],
    *,
    llm: Any | None = None,
    max_chars: int = 1800,
) -> tuple[bool | None, str]:
    """让 LLM 判读本地资料能否回答问题。

    返回 `(can_answer, reason)`；判读不可用时返回 `(None, 原因)`——
    调用方据此决定是否退回"只看相似度"。

    设计上**判读失败不阻断流程**：多一次调用是增强，不该成为新的失败点。
    """
    try:
        if llm is None:
            from src.clients.llm_factory import get_llm_client
            from src.config import settings

            llm = get_llm_client((settings.llm_provider or "deepseek").strip())
        if llm is None or not getattr(llm, "available", lambda: False)():
            return None, "LLM 不可用，无法判读"
    except Exception as exc:
        return None, f"LLM 不可用（{type(exc).__name__}）"

    blocks: list[str] = []
    used = 0
    for i, s in enumerate(sources, 1):
        q = str(s.get("question") or "").strip()
        a = str(s.get("answer") or "").strip()
        piece = f"【资料{i}】\n问：{q}\n答：{a}"
        if used + len(piece) > max_chars:
            piece = piece[: max(0, max_chars - used)]
        if not piece:
            break
        blocks.append(piece)
        used += len(piece)
        if used >= max_chars:
            break

    if not blocks:
        return None, "没有可判读的资料"

    user = f"用户问题：{question}\n\n===== 资料 =====\n" + "\n\n".join(blocks)

    try:
        raw = llm.chat_raw(
            [
                {"role": "system", "content": _JUDGE_PROMPT},
                {"role": "user", "content": user},
            ]
        )
    except Exception as exc:
        logger.info("覆盖判读调用失败：%s", exc)
        return None, f"判读调用失败（{type(exc).__name__}）"

    text = (raw or "").strip()
    if not text:
        return None, "判读返回空"

    # 解析：优先整体 JSON，再退回截取大括号（与项目其他处同一策略）
    import json as _json
    import re as _re

    obj = None
    body = text
    if "```" in body:
        parts = body.split("```")
        if len(parts) >= 3:
            body = parts[1].lstrip()
            if body.lower().startswith("json"):
                body = body[4:]
    try:
        obj = _json.loads(body.strip())
    except Exception:
        m = _re.search(r"\{.*\}", body, _re.S)
        if m:
            try:
                obj = _json.loads(m.group(0))
            except Exception:
                obj = None
    if not isinstance(obj, dict):
        # 模型没按 JSON 输出时做关键词兜底：判读的价值太高，不该因格式丢掉
        low = text.lower()
        if "不能" in text or "false" in low or "无法回答" in text:
            return False, "判读认为资料不足以回答（非结构化输出）"
        if "能" in text or "true" in low:
            return True, "判读认为资料可以回答（非结构化输出）"
        return None, "判读输出无法解析"

    can = obj.get("can_answer")
    reason = str(obj.get("reason") or "").strip()[:80]
    if isinstance(can, bool):
        return can, reason
    if isinstance(can, str):
        low = can.strip().lower()
        if low in ("true", "yes", "能", "可以"):
            return True, reason
        if low in ("false", "no", "不能", "无法"):
            return False, reason
    return None, "判读结果字段非法"


# ---------------------------------------------------------------------------
# 本地知识库充足度
# ---------------------------------------------------------------------------


def local_coverage(
    question: str,
    *,
    threshold: float = DEFAULT_MIN_RELEVANCE,
    top_k: int = 3,
    pipeline: Any | None = None,
    judge: bool = True,
    llm: Any | None = None,
) -> LocalCoverage:
    """判断本地知识库能否回答该问题。**两层判定**：

        第一层 相似度粗筛（便宜）：最高余弦相似度 < 阈值 → 直接判"不够"
        第二层 LLM 判读（较贵但准）：过筛后让模型确认"资料能否回答"

    ★ 为什么需要第二层：相似度只能判断**主题相关**，不能判断**能否回答**
    ------------------------------------------------------------------
    实测反例——问"中国大陆和香港的招投标法规的区别是什么？"：
    本地库全是大陆招投标问答、没有香港法规，但"招投标法规"这个主题
    让相似度达 0.55，过了 0.5 门槛 → 被误判为"本地够用"，
    于是用大陆法规去回答一个需要香港法规的问题。

    而"都是招投标"远比"能回答这个具体问题"容易满足。故相似度只作粗筛。

    `judge=False` 或判读不可用时退回纯相似度判定，并在 `judged=False`
    里如实标记——调用方据此知道这个结论的可信度较低。

    **本函数不抛异常**：知识库不可用（Qdrant 没起 / 集合为空）时返回
    `enough=False` 并说明原因，让自动兜底把请求转给联网搜索。
    **"知识库没东西"更应该联网，而不是报错。**
    """
    try:
        if pipeline is None:
            from src.rag.pipeline import rag_pipeline

            pipeline = rag_pipeline
    except Exception as exc:
        logger.info("RAG 流水线不可用，视为本地无覆盖：%s", exc)
        return LocalCoverage(
            enough=False,
            best_relevance=None,
            threshold=threshold,
            reason=f"本地知识库不可用（{type(exc).__name__}），已改用联网搜索",
        )

    try:
        sources = pipeline.search(question, top_k=top_k) or []
    except Exception as exc:
        logger.info("本地检索失败，视为本地无覆盖：%s", exc)
        return LocalCoverage(
            enough=False,
            best_relevance=None,
            threshold=threshold,
            reason=f"本地检索失败（{type(exc).__name__}），已改用联网搜索",
        )

    best: float | None = None
    for s in sources:
        rel = s.get("relevance")
        if isinstance(rel, (int, float)):
            best = float(rel) if best is None else max(best, float(rel))

    if not sources:
        return LocalCoverage(
            enough=False,
            best_relevance=None,
            threshold=threshold,
            hits_count=0,
            reason="本地知识库没有检索到任何内容，已改用联网搜索",
        )

    if best is None:
        # 相关性字段缺失（旧版向量库或降级路径）。保守起见判为"不够"，
        # 因为**不能证明够**时联网的风险小于"凭不足的依据作答"。
        return LocalCoverage(
            enough=False,
            best_relevance=None,
            threshold=threshold,
            hits_count=len(sources),
            reason="本地检索结果没有相关性分数，无法确认覆盖，已改用联网搜索",
            sources=sources,
        )

    # 第一层：相似度粗筛
    if best < threshold:
        return LocalCoverage(
            enough=False,
            best_relevance=best,
            threshold=threshold,
            hits_count=len(sources),
            reason=(
                f"本地知识库最高匹配度仅 {best:.2f}（低于阈值 {threshold:.2f}），"
                "采信度不足，已自动改用联网搜索"
            ),
            sources=sources,
        )

    # 第二层：过筛后让模型确认"资料能否回答"
    if not judge:
        return LocalCoverage(
            enough=True,
            best_relevance=best,
            threshold=threshold,
            hits_count=len(sources),
            reason=f"本地知识库匹配度 {best:.2f} ≥ 阈值 {threshold:.2f}，用本地知识回答",
            sources=sources,
        )

    can, judge_reason = _judge_by_llm(question, sources, llm=llm)
    if can is True:
        return LocalCoverage(
            enough=True,
            best_relevance=best,
            threshold=threshold,
            hits_count=len(sources),
            reason=f"本地匹配度 {best:.2f}，且确认资料可直接回答",
            sources=sources,
            judged=True,
            judge_reason=judge_reason,
        )
    if can is False:
        return LocalCoverage(
            enough=False,
            best_relevance=best,
            threshold=threshold,
            hits_count=len(sources),
            reason=(
                f"本地有相关内容（匹配度 {best:.2f}）但**不足以回答该问题**"
                f"{('：' + judge_reason) if judge_reason else ''}，已自动改用联网搜索"
            ),
            sources=sources,
            judged=True,
            judge_reason=judge_reason,
        )

    # 判读不可用：退回纯相似度结论，并如实标记未判读
    return LocalCoverage(
        enough=True,
        best_relevance=best,
        threshold=threshold,
        hits_count=len(sources),
        reason=(
            f"本地知识库匹配度 {best:.2f} ≥ 阈值 {threshold:.2f}，用本地知识回答"
            f"（未能做覆盖判读：{judge_reason}）"
        ),
        sources=sources,
        judged=False,
        judge_reason=judge_reason,
    )


# ---------------------------------------------------------------------------
# 联网搜索（两路归一）
# ---------------------------------------------------------------------------


def _score_by_cosine(query: str, texts: Sequence[str]) -> list[float | None]:
    """用嵌入模型给一批文本算与查询的余弦相似度。

    为什么需要：智谱那一路**不返回相关度分数**，前端就无法显示评分，
    也无法判断结果是否相关。用与本地检索同一套嵌入模型补算，
    好处是**分数与本地知识库的 relevance 同量纲**，可以直接比较
    （"这条联网结果比本地最匹配的还相关"是用户会关心的判断）。

    ★ 查询与文档必须用**不同的编码函数**（BGE 是非对称模型）
    ------------------------------------------------------
        `embedder.encode_query(q)`     加检索指令前缀（查询侧）
        `embedder.encode_documents(ds)` **不加**前缀（文档侧）+ 支持批量

    第一版两侧都用 `encode_query`，等于给文档也加了查询前缀，语义错位。
    实测两种做法的分数差 0.04~0.07（本场景下不致命，但方向是错的）。
    正确做法同时带来批量编码的性能收益。

    失败返回全 None——不因为"算不出分数"就让整次搜索失败。
    """
    try:
        from src.rag.embedder import embedder

        qv = embedder.encode_query(query)
        if not qv:
            return [None] * len(texts)
        na = sum(a * a for a in qv) ** 0.5
        if not na:
            return [None] * len(texts)

        # 文档侧批量编码（不加查询前缀）
        prepared = [(t or "")[:SCORE_TEXT_CHARS] for t in texts]
        try:
            vecs = embedder.encode_documents(prepared)
        except Exception:
            logger.info("文档侧批量编码失败，退回逐条 encode_query", exc_info=True)
            vecs = []
            for t in prepared:
                try:
                    vecs.append(embedder.encode_query(t))
                except Exception:
                    vecs.append(None)

        out: list[float | None] = []
        for v in vecs:
            if not v:
                out.append(None)
                continue
            # 与 rag/vector_store._cosine 同一算法（各自独立实现，避免跨模块私有依赖）
            dot = sum(a * b for a, b in zip(qv, v))
            nb = sum(b * b for b in v) ** 0.5
            out.append(round(dot / (na * nb), 4) if nb else None)

        # 健全性检查：若"有分数的全都很低"，多半是编码/取文本出了问题
        # （实测曾因读错字段名导致整批补算被跳过、或分数全是 0.0001）。
        # 这类失败**不报错、不影响功能**，只能靠日志发现，故这里主动告警。
        valid = [s for s in out if s is not None]
        if valid and max(valid) < 0.1:
            logger.warning(
                "联网结果补算的相关度异常偏低（最高 %.4f，共 %d 条）——"
                "可能取错了正文字段或编码方式不对，请检查 _score_by_cosine 的入参",
                max(valid), len(valid),
            )
        return out
    except Exception as exc:
        logger.info("联网结果补算相关度失败（不影响搜索本身）：%s", exc)
        return [None] * len(texts)


def _from_tavily(query: str, max_results: int) -> tuple[list[SearchHit], str]:
    """Tavily 搜索。返回 `(结果, AI 摘要)`。

    ★ 用**模块级单例**，不要 `WebSearchClient()` 现 new
    -----------------------------------------------
    原来写的是 `WebSearchClient().search(...)`。而缓存 `_cache` 是**实例属性**，
    每次 new 一个实例等于每次都是空缓存 —— 这个数据源的缓存在这条路径上
    **完全没生效**（实测：`/api/web/search` 同一查询连打两次都是 14 秒，
    `/api/cache` 显示 `web_search_tavily size=0`）。

    而 Agent 的工具路径（`rag_tools.search_web`）用的是模块级单例
    `web_search_client`，缓存是有效的。同一个类、两条路径、一个有缓存一个没有，
    差别只在"是否复用实例"——这类 bug 不会报错，只表现为"偶尔很慢"。

    单例只多持有一个 dict 与一把锁（构造函数不做任何 I/O），复用无副作用。
    """
    from src.web_search import WebSearchError, is_configured, web_search_client

    if not is_configured():
        raise WebSearchUnavailable("未配置 TAVILY_API_KEY，联网搜索不可用")
    try:
        payload = web_search_client.search(query, max_results=max_results)
    except WebSearchError as exc:
        raise WebSearchUnavailable(str(exc)) from exc

    hits = []
    for item in payload.get("results") or []:
        raw_score = item.get("score")
        score = float(raw_score) if isinstance(raw_score, (int, float)) else None
        hits.append(
            SearchHit(
                title=str(item.get("title") or ""),
                url=str(item.get("url") or ""),
                content=str(item.get("content") or ""),
                score=score,
                source="tavily",
                score_kind="engine" if score is not None else "",
            )
        )
    return hits, str(payload.get("answer") or "")


def _from_zhipu(query: str, max_results: int) -> list[SearchHit]:
    """智谱中文搜索。

    智谱的 `_normalize_items` 已把结果归一成与 Tavily **同形**的字段集
    （`title/url/content/score/publish_date/media`），故这里只需搬运，
    额外保留 `publish_date`（智谱独有，是判断信息时效性的直接依据——
    这正是引入第二路搜索的核心理由）。

    智谱的 `score` 实测常为 None，由调用方用余弦相似度补算。

    ★ 同样改用模块级单例（理由见 `_from_tavily` 的说明）：
    `ZhipuSearchClient()` 现 new 会让实例级缓存永远为空。
    """
    try:
        from src.web_search_zhipu import ZhipuSearchError, zhipu_search_client
    except ImportError as exc:
        raise WebSearchUnavailable("智谱搜索模块未就绪") from exc

    try:
        payload = zhipu_search_client.search(query, max_results=max_results)
    except ZhipuSearchError as exc:
        raise WebSearchUnavailable(str(exc)) from exc

    hits = []
    for item in payload.get("results") or []:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        url = str(item.get("url") or "").strip()
        if not title and not url:
            continue
        raw_score = item.get("score")
        score = float(raw_score) if isinstance(raw_score, (int, float)) else None
        hits.append(
            SearchHit(
                title=title or url,
                url=url,
                content=str(item.get("content") or ""),
                score=score,
                published=str(item.get("publish_date") or ""),
                source="zhipu",
                score_kind="engine" if score is not None else "",
            )
        )
    return hits


def search_web(
    query: str,
    *,
    max_results: int = DEFAULT_WEB_TOP_K,
    providers: Iterable[str] = ("tavily", "zhipu"),
    min_score: float | None = None,
) -> dict[str, Any]:
    """联网搜索（多路归一）。返回：

        results        按相关度降序的结果列表（每条带 score）
        answer         引擎给的 AI 摘要（Tavily 有，智谱无）
        providers_used 实际成功的数据源
        errors         各数据源的失败原因（**部分失败不影响整体**）
        min_score      过滤阈值（低于它的结果被丢弃，`filtered` 记录丢了几条）

    ★ 部分失败不整体失败：两路搜索是互补的（Tavily 英文强、智谱中文站点强），
    一路挂了另一路的结果仍然有用。只有**全部失败**才抛 `WebSearchUnavailable`。
    """
    query = (query or "").strip()
    if not query:
        return {"results": [], "answer": "", "providers_used": [], "errors": [], "filtered": 0}

    all_hits: list[SearchHit] = []
    errors: list[dict[str, str]] = []
    used: list[str] = []
    answer = ""

    for name in providers:
        try:
            if name == "tavily":
                hits, ans = _from_tavily(query, max_results)
                answer = answer or ans
            elif name == "zhipu":
                hits = _from_zhipu(query, max_results)
            else:
                continue
            used.append(name)
            all_hits.extend(hits)
        except WebSearchUnavailable as exc:
            errors.append({"provider": name, "reason": str(exc)})
            logger.info("联网搜索数据源 %s 不可用：%s", name, exc)
        except Exception as exc:  # pragma: no cover - 兜底
            errors.append({"provider": name, "reason": f"{type(exc).__name__}"})
            logger.warning("联网搜索数据源 %s 异常", name, exc_info=True)

    if not used:
        raise WebSearchUnavailable(
            "；".join(f"{e['provider']}：{e['reason']}" for e in errors) or "没有可用的搜索数据源"
        )

    # 智谱结果补算分数（Tavily 自带引擎分，保留原值以便与引擎口径一致）
    needs = [h for h in all_hits if h.score is None and h.content]
    if needs:
        scores = _score_by_cosine(query, [h.content for h in needs])
        for h, s in zip(needs, scores):
            if s is not None:
                h.score = s
                h.score_kind = "cosine"

    # 按分数降序；无分数的排在最后（而不是丢弃——它可能仍有用，只是无法评估）
    all_hits.sort(key=lambda h: (h.score is None, -(h.score or 0)))

    filtered = 0
    if min_score is not None:
        kept = [h for h in all_hits if h.score is None or h.score >= min_score]
        filtered = len(all_hits) - len(kept)
        all_hits = kept

    # 同一 URL 只留分数最高的一条（两路可能搜到同一页面）
    deduped: list[SearchHit] = []
    seen_urls: set[str] = set()
    for h in all_hits:
        key = h.url or h.title
        if key in seen_urls:
            continue
        seen_urls.add(key)
        deduped.append(h)

    return {
        "results": [h.to_dict() for h in deduped],
        "answer": answer,
        "providers_used": used,
        "errors": errors,
        "filtered": filtered,
    }


def is_available() -> dict[str, Any]:
    """各数据源的可用性（供健康检查与前端提示）。"""
    status: dict[str, Any] = {"tavily": False, "zhipu": False, "any": False}
    try:
        from src.web_search import is_configured as tavily_ok

        status["tavily"] = bool(tavily_ok())
    except Exception:
        status["tavily"] = False
    try:
        from src import web_search_zhipu

        status["zhipu"] = bool(
            getattr(web_search_zhipu, "is_configured", lambda: True)()
        )
    except Exception:
        status["zhipu"] = False
    status["any"] = bool(status["tavily"] or status["zhipu"])
    return status


__all__ = [
    "DEFAULT_MIN_RELEVANCE",
    "DEFAULT_WEB_TOP_K",
    "LocalCoverage",
    "SearchHit",
    "WebSearchUnavailable",
    "is_available",
    "local_coverage",
    "search_web",
]

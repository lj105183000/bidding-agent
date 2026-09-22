"""RAG 流水线：混合检索 → （可选精排）→ 生成回答。

**两种回答模式共存**（本项目的一处刻意设计）：

1. **LLM 生成**——把检索到的问答对作为知识库参考交给大模型组织语言；
2. **检索直返**——无 LLM 凭据、无检索结果或调用失败时，直接把最匹配的
   问答原文返回给用户。

降级是**运行时**发生的：某次 LLM 调用超时/报错，正在流式输出的半截回答会被
`reset` 事件清掉再重新输出原文，而不是把残缺内容留在屏幕上。

架构约束（见 docs/开发文档.md §8.1）：`chat_events` 是唯一的编排入口，SSE 与
非流式两条出口都消费它——新增逻辑只改这一处。
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Iterator

from src.clients.llm_factory import get_llm_client
from src.config import settings
from src.logging_config import get_logger
from src.rag.classify_question import fuse_rrf, get_params
from src.rag.constants import (
    DEFAULT_TOP_K,
    MAX_HISTORY_ROUNDS,
    PREFETCH_LIMIT,
    RERANK_CANDIDATE_K,
    STREAM_CHUNK_CHARS,
    STREAM_CHUNK_DELAY,
    TOOL_NAME_KB,
)
from src.rag.vector_store import VectorStore

logger = get_logger(__name__)

SYSTEM_PROMPT = """你是招投标采购领域的资深专家助手，服务于采购人、代理机构与供应商。

请依据「知识库参考」回答用户问题，并遵守：
1. 以知识库参考为主要依据，不要编造参考中没有的法规名称、条款号、数字或事实。
2. 参考不足以回答时，明确说明"知识库中未找到直接依据"，再给出一般性说明，不要硬答。
3. 用中文回答，条理清晰；涉及法规时写明依据的法规名称。
4. 不要在正文里写"参考1/参考2"这类编号引用——来源由系统在回答下方单独展示。"""

_NO_RESULT_TEXT = (
    "未在知识库中找到与「{question}」相关的内容。\n\n"
    "建议：换用更具体的关键词（如法规名称、业务动作），"
    "或确认知识库已导入（`python main.py ingest`）。"
)

# 检索缓存：同一问题在进程内重复出现时（多轮追问、前端重试）省掉一次
# 编码 + 向量检索。注意重新 ingest 后缓存不会自动失效——重启服务即可。
_CACHE_MAXSIZE = 256


class RAGPipeline:
    """检索与生成的门面。依赖均可注入，便于测试替换。"""

    def __init__(self, *, store=None, embedder=None, reranker=None, llm=None, bm25=None) -> None:
        self._store = store
        self._embedder = embedder
        self._reranker = reranker
        self._llm = llm
        self._bm25 = bm25
        self._bm25_lock = threading.Lock()
        self._bm25_error: str | None = None
        # 检索缓存挂在实例上而非模块级：注入依赖的实例（测试、多实例场景）
        # 不会互相污染，也不会读到一个用别的依赖算出来的结果
        self._cache: OrderedDict[tuple, tuple] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._cache_hits = 0
        self._cache_misses = 0

    # ---- 依赖的懒加载 ----

    @property
    def store(self) -> VectorStore:
        if self._store is None:
            self._store = VectorStore(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
                collection=settings.qdrant_collection,
                dim=settings.qdrant_vector_size,
                distance=settings.qdrant_distance,
                timeout=settings.qdrant_timeout,
            )
        return self._store

    @property
    def embedder(self):
        if self._embedder is None:
            from src.rag.embedder import embedder as default_embedder

            self._embedder = default_embedder
        return self._embedder

    @property
    def reranker(self):
        if self._reranker is None:
            from src.rag.embedder import reranker as default_reranker

            self._reranker = default_reranker
        return self._reranker

    @property
    def bm25(self):
        """词表只在首次检索时加载一次；缺失时记下原因，由 /api/health 暴露。"""
        if self._bm25 is None:
            with self._bm25_lock:
                if self._bm25 is None:
                    from src.rag.embedder import BM25Encoder

                    try:
                        self._bm25 = BM25Encoder.load()
                        self._bm25_error = None
                    except FileNotFoundError as exc:
                        self._bm25_error = str(exc)
                        raise
        return self._bm25

    @property
    def llm(self):
        if self._llm is None:
            self._llm = get_llm_client(settings.llm_provider)
        return self._llm

    @property
    def ready(self) -> bool:
        """流水线是否可用（词表已就绪）。Qdrant 连通性由 /api/health 单独探测。"""
        try:
            self.bm25  # noqa: B018 - 触发懒加载以判定就绪
            return True
        except FileNotFoundError:
            return False

    @property
    def ready_error(self) -> str | None:
        return self._bm25_error

    # ---- 检索 ----

    def search_cached(self, question: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
        """带 LRU 缓存的检索。同一问题在进程内重复出现（多轮追问、前端重试）时
        省掉一次编码 + 向量检索。注意**重新 ingest 后缓存不会自动失效**——
        调用 `clear_search_cache()` 或重启服务即可（检索结果与数据一致性
        由调用方负责，缓存层不做失效探测）。
        """
        key = (question, top_k)
        with self._cache_lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache_hits += 1
                return list(self._cache[key])

        result = tuple(self.search(question, top_k))  # 异常不写入缓存

        with self._cache_lock:
            self._cache[key] = result
            self._cache_misses += 1
            while len(self._cache) > _CACHE_MAXSIZE:
                self._cache.popitem(last=False)
        return list(result)

    def clear_cache(self) -> None:
        with self._cache_lock:
            self._cache.clear()

    def cache_info(self) -> dict:
        with self._cache_lock:
            return {
                "hits": self._cache_hits,
                "misses": self._cache_misses,
                "size": len(self._cache),
                "maxsize": _CACHE_MAXSIZE,
            }

    def search(self, question: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
        """混合检索，返回 `[{question, answer, score}]`（契约字段，勿改名）。

        **问题分类驱动的参数选择**：先对问题分类（关键词 / 自然语言 / 法条引用），
        按类别取不同的 RRF k 与召回深度——不同问法对 dense 与 sparse 两路的倚重不同，
        分类后调参比"一套参数走天下"的召回质量更稳。
        """
        dense = self.embedder.encode_query(question)
        sparse = self.bm25.encode_query(question)

        # 分类 → 检索参数（rrf_k / prefetch_limit / top_k）
        params = get_params(question)
        rrf_k = params["rrf_k"]
        prefetch = params["prefetch_limit"]
        # 调用方显式传了 top_k 时尊重调用方（评测、测试可能要不同条数）
        effective_top_k = top_k

        if not settings.rerank_enabled:
            return self._hybrid_retrieve(
                dense, sparse, rrf_k, prefetch, effective_top_k
            )

        # 精排需要更大的候选池才有意义：先多召回，再由 CrossEncoder 收敛到 top_k
        candidates = self._hybrid_retrieve(
            dense, sparse, rrf_k, prefetch, max(effective_top_k, RERANK_CANDIDATE_K)
        )
        try:
            return self.reranker.rerank(question, candidates, effective_top_k)
        except Exception as exc:
            # 精排是**增益项**而非必需项：模型下载失败、显存不足、文件损坏都不该让
            # 整个检索失败——退回混合检索顺序，质量略降但用户仍拿得到答案
            logger.warning("精排失败，退回混合检索顺序: %s", exc)
            return candidates[:effective_top_k]

    def _hybrid_retrieve(
        self,
        dense: list[float],
        sparse,
        rrf_k: int,
        prefetch_limit: int,
        top_k: int,
    ) -> list[dict]:
        """执行混合检索。

        k == 60（Qdrant 内置默认值）时走服务端 RRF，一次网络往返完成融合；
        k ≠ 60 时改为两路独立查询 + 应用层 `fuse_rrf`——多一次网络往返但 k 可控。
        """
        # Qdrant 内置 RRF 的 k 固定为 60（见 qdrant-client 源码 Fusion.RRF），
        # k=60 时直接用内置融合，省掉一次网络往返
        _QDRANT_DEFAULT_RRF_K = 60
        if rrf_k == _QDRANT_DEFAULT_RRF_K:
            return self.store.hybrid_search(
                dense, sparse, limit=top_k, prefetch_limit=prefetch_limit
            )
        # 非 60：两路独立查询后应用层融合
        dense_results = self.store.dense_search(dense, limit=prefetch_limit)
        sparse_results = self.store.sparse_search(sparse, limit=prefetch_limit)
        return fuse_rrf(dense_results, sparse_results, k=rrf_k, limit=top_k)

    # ---- 生成 ----

    def _format_context(self, sources: list[dict]) -> str:
        blocks = [
            f"【参考{i}】问：{s['question']}\n答：{s['answer']}"
            for i, s in enumerate(sources, 1)
        ]
        return "\n\n".join(blocks)

    def build_messages(
        self, question: str, sources: list[dict], history: list[dict] | None = None
    ) -> list[dict]:
        """组装 LLM 消息：system + 截断后的历史 + 带参考的当前问题。"""
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

        if history:
            # 只保留最近 N 轮：更早的对话对当前问题的边际价值低，
            # 却会挤占上下文窗口并拉高每次调用的成本
            recent = history[-(MAX_HISTORY_ROUNDS * 2) :]
            for item in recent:
                role = item.get("role")
                content = (item.get("content") or "").strip()
                if role in {"user", "assistant"} and content:
                    messages.append({"role": role, "content": content})

        if sources:
            user_content = (
                f"知识库参考：\n{self._format_context(sources)}\n\n"
                f"用户问题：{question}"
            )
        else:
            user_content = f"（知识库无相关参考）\n\n用户问题：{question}"
        messages.append({"role": "user", "content": user_content})
        return messages

    @staticmethod
    def fallback_answer(sources: list[dict], question: str) -> str:
        """降级回答：命中则返回最匹配的问答原文，否则给出无结果提示。"""
        if sources:
            return sources[0]["answer"]
        return _NO_RESULT_TEXT.format(question=question)

    # ---- 编排入口 ----

    def chat_events(
        self,
        question: str,
        history: list[dict] | None = None,
        top_k: int = DEFAULT_TOP_K,
        paced: bool = True,
    ) -> Iterator[dict]:
        """产出协议事件（status / token / reset / done / error），供 SSE 与
        非流式两条出口消费。

        `paced=False` 关闭流式小帧的帧间延时——非流式调用与测试都不需要它。
        """
        started = time.perf_counter()
        phase_times: list[list] = []

        yield {"type": "status", "content": "正在检索知识库..."}
        search_started = time.perf_counter()
        try:
            sources = self.search_cached(question, top_k)
        except Exception as exc:
            # 检索失败是致命路径：没有来源就无从生成，直接以 error 收尾。
            # 完整异常只进日志，用户侧只看到可操作的提示（§13.2）
            logger.exception("知识库检索失败: %s", exc)
            yield {
                "type": "error",
                "content": "知识库检索失败，请稍后重试；若持续失败请联系管理员。",
            }
            return

        phase_times.append(["检索知识库", int((time.perf_counter() - search_started) * 1000)])

        use_llm = self.llm is not None and self.llm.available() and bool(sources)
        if use_llm:
            yield {"type": "status", "content": "正在生成回答..."}
        elif sources:
            yield {"type": "status", "content": "未配置大模型，直接返回知识库原文"}
        else:
            yield {"type": "status", "content": "知识库中没有找到相关内容"}

        generate_started = time.perf_counter()
        emitted = False
        if use_llm:
            messages = self.build_messages(question, sources, history)
            try:
                for chunk in self.llm.chat_stream(messages):
                    if chunk:
                        emitted = True
                        yield {"type": "token", "content": chunk}
            except Exception as exc:
                logger.warning("LLM 生成失败，降级为检索原文: %s", exc)
                # 已经流出的半截内容必须清掉，否则会与降级文本拼在一起
                yield {"type": "reset"}
                yield {"type": "status", "content": "大模型调用失败，已切换为知识库原文"}
                # reset 已经把前端已渲染的内容清空了，所以这里的 emitted 必须
                # 归零——否则会跳过下面的降级输出，用户看到一个空气泡
                emitted = False

        if not emitted:
            text = self.fallback_answer(sources, question)
            yield from self._stream_text(text, paced=paced)

        phase_times.append(["生成回答", int((time.perf_counter() - generate_started) * 1000)])

        yield {
            "type": "done",
            "sources": sources,
            "web_sources": [],
            "tool_called": bool(sources),
            "tool_name": TOOL_NAME_KB if sources else "",
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
            "phase_times": phase_times,
        }

    def _stream_text(self, text: str, paced: bool = True) -> Iterator[dict]:
        """把整段文本切成小帧输出，让前端渲染节奏平滑。"""
        for start in range(0, len(text), STREAM_CHUNK_CHARS):
            yield {"type": "token", "content": text[start : start + STREAM_CHUNK_CHARS]}
            if paced and STREAM_CHUNK_DELAY:
                time.sleep(STREAM_CHUNK_DELAY)

    def chat(
        self, question: str, history: list[dict] | None = None, top_k: int = DEFAULT_TOP_K
    ) -> dict:
        """非流式出口：消费同一份事件流，收敛为结果字典（评测脚本用）。"""
        answer_parts: list[str] = []
        result: dict = {"answer": "", "sources": [], "tool_called": False, "tool_name": ""}
        for event in self.chat_events(question, history, top_k, paced=False):
            if event["type"] == "token":
                answer_parts.append(event["content"])
            elif event["type"] == "reset":
                answer_parts.clear()
            elif event["type"] == "done":
                result.update(
                    {
                        "sources": event["sources"],
                        "tool_called": event["tool_called"],
                        "tool_name": event["tool_name"],
                        "elapsed_ms": event["elapsed_ms"],
                        "phase_times": event["phase_times"],
                    }
                )
            elif event["type"] == "error":
                result["error"] = event["content"]
        result["answer"] = "".join(answer_parts).strip()
        return result


# ---- 模块级单例 ----

rag_pipeline = RAGPipeline()


def clear_search_cache() -> None:
    """清空默认流水线的检索缓存（重新 ingest 或数据变更后调用）。"""
    rag_pipeline.clear_cache()


def cache_info() -> dict:
    return rag_pipeline.cache_info()

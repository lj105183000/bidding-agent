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

import os
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
    VOCAB_FILE,
)
from src.rag.vector_store import MIN_RELEVANCE, VectorStore

logger = get_logger(__name__)

SYSTEM_PROMPT = """你是招投标采购领域的资深专家助手，服务于采购人、代理机构与供应商。

请依据「知识库参考」回答用户问题，并遵守：
1. 以知识库参考为主要依据，不要编造参考中没有的法规名称、条款号、数字或事实。
2. 参考不足以回答时，明确说明"知识库中未找到直接依据"，再给出一般性说明，不要硬答。
3. 用中文回答，条理清晰；涉及法规时写明依据的法规名称。
4. 不要在正文里写"参考1/参考2"这类编号引用——来源由系统在回答下方单独展示。
5. **条款号与具体数字只在参考里确实出现时才写**。若你依据的是常识性法规
   （如《招标投标法》）但参考中没有该条款原文，写明法规名称即可，
   **不要补具体条款号**（如"第四十五条"）或参考中没有的百分比、金额——
   改用"依据该法相关规定"这类不含编号的表述。凭记忆写的条款号极易出错，
   而用户会据此核对合规性，错一个数字就可能误导决策。"""

# 更严的变体：把"不许超纲"写成硬约束。
#
# ★ 为什么需要它（实测数据驱动，见 docs/21-RAGAS评测改造与数据驱动优化.md）
# ------------------------------------------------------------------
# 用 ragas 评测 25 题发现一个反直觉现象：
#
#     库内缺失的 5 题  → 平均忠实度 0.872
#     几乎原题的 8 题  → 平均忠实度 0.494   ← 照着库里的题问，反而更低
#
# 排查后确认原因**不在检索**（召回率三组配置全是 1.000，无漏召回），
# 而在生成侧：知识库答案多是"一句话法条"（实测第 12 题的 context 仅 36 字），
# 而原提示词要求"条理清晰"，模型于是补出解释、"要点提示"与推断——
# 这些内容**在参考里没有**，Faithfulness 因此判低分。
#
# 具体反例（第 12 题，context 只说"3日内公示、公示期不少于3日"）：
#     模型补出 →"招标人可根据项目情况适当延长，但不得缩短"   ← 参考里完全没有
#
# 故这个变体明确禁止三类越界：
#     ① "补充说明/要点提示"式的自撰解读
#     ② 从参考推断出的新结论
#     ③ 参考没提的法规名称或条款号
#
# 变体形式（而不是直接替换默认值）是**刻意的**：改产品行为必须有数据支撑，
# 所以先让 A/B 跑出对比，再决定是否设为默认。用 `RAG_PROMPT_VARIANT` 选择。
SYSTEM_PROMPT_STRICT = """你是招投标采购领域的资深专家助手，服务于采购人、代理机构与供应商。

请**只依据**「知识库参考」回答问题，并遵守：
1. 回答中的每一条事实、数字、时限、比例、法规名称与条款号，都必须能在参考里原文找到。
2. **不要写参考中没有的内容**，包括但不限于：
   - "补充说明""要点提示""注意事项"这类自撰的解读段落；
   - 根据参考推断出的新结论（例如参考只说"不得少于3日"，就不要加"可适当延长"）；
   - 参考未提及的法规名称、条款号或主管部门。
3. 参考内容较短时，**直接给出参考中的结论即可**，不要为了篇幅而扩写。
4. 参考不足以回答时，明确说明"知识库中未提供该信息"，然后停止，不要用常识补充。
5. 用中文回答；不要在正文里写"参考1/参考2"这类编号引用——来源由系统在回答下方单独展示。"""

# 平衡变体：**禁止超纲，但要求完整作答**。
#
# ★ 为什么需要它（A/B 实测数据，见 docs/22-生成提示词优化.md）
# --------------------------------------------------------
# `strict` 把忠实度从 0.763 提到 0.870，但相关性从 0.890 掉到 0.793。
# 逐题看重灾个案：
#
#     [6] "单一来源采购文件需要写明所需货物的品牌和型号吗？"
#         相关性 0.937 → 0.362    回答 217 字 → 40 字
#
# 40 字不是"更简洁"，而是**回答不再完整**——它只摘了库内一句话，
# 没有真正回应用户问的那个点。原因在 strict 的第 3 条
# （"参考内容较短时直接给出结论即可，不要为了篇幅而扩写"）被模型过度执行了。
#
# 实测还排除了"短答案被度量惩罚"这一解释：strict 下答案长度与相关性的
# 皮尔逊相关系数仅 +0.133，即相关性下降**不是**长度造成的假象。
#
# 故这一版把两条约束分开写清楚：
#     禁止的是"参考里没有的内容"（超纲），
#     要求的 是"把参考里有的内容讲完整"（完整）。
# 二者不矛盾——之前的问题是把"完整"和"扩写"混为一谈了。
SYSTEM_PROMPT_BALANCED = """你是招投标采购领域的资深专家助手，服务于采购人、代理机构与供应商。

请只依据「知识库参考」回答用户问题，并遵守：

【必须做的】
1. 完整回应用户问的点。若参考里有相关内容，就把它讲清楚，不要只丢一句话。
2. 参考包含多个要点时，逐条列出，使回答自成一体、可直接使用。
3. 用中文回答，条理清晰。

【不许做的】
4. 不写参考里没有的内容，包括：
   - "补充说明""要点提示"这类自撰的解读段落；
   - 从参考推断出的新结论（参考只说"不得少于3日"，就不要加"可适当延长"）；
   - 参考未提及的法规名称、条款号或主管部门。
5. 参考只覆盖了问题的一部分时，**答完覆盖到的部分，并明确指出哪一部分参考未涉及**，
   不要用常识把缺口补上，也不要因此连覆盖到的部分都省略。

6. 不要在正文里写"参考1/参考2"这类编号引用——来源由系统在回答下方单独展示。"""

# 极简变体：用来验证"越简洁越忠实"的边界，作为 A/B 的下界参照
SYSTEM_PROMPT_MINIMAL = """你是招投标采购领域的专家助手。

只依据「知识库参考」回答，遵守：
1. 只用参考里出现过的内容作答，一个字都不要补充。
2. 参考里没有的，直接说"知识库中未提供该信息"。
3. 不要写"补充说明""要点提示"这类段落，不要举例，不要推断。
4. 简洁作答，用中文。"""

# 可选提示词变体。键名用于 `RAG_PROMPT_VARIANT` 环境变量与评测 A/B 的配置名。
_PROMPT_VARIANTS: dict[str, str] = {
    "default": SYSTEM_PROMPT,
    "balanced": SYSTEM_PROMPT_BALANCED,
    "strict": SYSTEM_PROMPT_STRICT,
    "minimal": SYSTEM_PROMPT_MINIMAL,
}


def _parallel_search_enabled() -> bool:
    """双路检索是否并发。默认**开**；`RAG_PARALLEL_SEARCH=0/false/no/off` 关闭。

    留这个开关是为了**可对照**：并发是否真的更快取决于部署形态
    （本地 Qdrant 时往返只有几毫秒，并发的收益会被线程开销吃掉；
    跨区云端时往返几百毫秒，收益接近一倍）。排查"为什么反而慢了"时，
    能一键退回串行，不用改代码重发版本。

    与 `_resolve_prompt_variant` 同一口径：**非法值不抛异常**，按默认处理——
    拼错一个环境变量不该让问答服务起不来。
    """
    raw = (os.getenv("RAG_PARALLEL_SEARCH") or "").strip().lower()
    if not raw:
        return True
    return raw not in ("0", "false", "no", "off", "disable", "disabled")


def _resolve_prompt_variant() -> str:
    """读取 `RAG_PROMPT_VARIANT`，非法值回落到 default。

    ★ 非法值只告警、不抛异常（与 `RERANK_ENABLED` 的容忍口径一致）：
    拼错一个环境变量就让整个问答服务起不来，是把配置错误的代价放大到了
    不可用级别——那比"用了默认提示词"严重得多。
    """
    raw = (os.getenv("RAG_PROMPT_VARIANT") or "").strip().lower()
    if not raw:
        return "default"
    if raw not in _PROMPT_VARIANTS:
        logger.warning(
            "RAG_PROMPT_VARIANT=%r 不是已知变体（可用：%s），回落到 default",
            raw, ", ".join(_PROMPT_VARIANTS),
        )
        return "default"
    return raw


# 启动时解析一次并记日志：提示词变体会显著改变回答风格，
# 排查线上表现时必须能从日志确认当时用的是哪一个。
PROMPT_VARIANT = _resolve_prompt_variant()
ACTIVE_SYSTEM_PROMPT = _PROMPT_VARIANTS[PROMPT_VARIANT]
if PROMPT_VARIANT != "default":
    logger.info("RAG 提示词变体：%s（非默认，用于 A/B 或验证）", PROMPT_VARIANT)

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
        # 语料点数缓存。首次 `_corpus_fingerprint()` 时查一次，
        # 之后 O(1)（每次查 Qdrant 要 208ms，见该方法说明）。
        self._corpus_count: int | None = None
        self._cache_hits = 0
        # 注：曾经在这里放过 `self.last_timing`（最近一次检索的耗时分解）。
        # 那是**单例共享字段**，不同请求会互相覆盖，已改为 ContextVar
        # （见 `src/rag/timing.py` 的 `_last_marks` 与 `take_marks()`）。
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
        """带 LRU 缓存的检索。同一问题在进程内重复出现（多轮追问、前端重试、
        Agent 反复用同一 query 调工具）时省掉一次编码 + 向量检索 + 融合。

        ★ 缓存键带**语料指纹**，重新 ingest 后自动失效（实测缺陷修复）
        ----------------------------------------------------------
        原先键是 `(question, top_k)`，而 docstring 只说"重新 ingest 后缓存不会
        自动失效，调用方负责 clear"。这个约定很脆弱——实测吃过亏：
        给知识库补语料后，**长跑的 API 仍在返回补充前的旧结果**，
        直到重启服务；而没有任何地方会提示这一点。

        现在键里加上 `(点数, 词表长度)` 作为语料指纹：
            · 重新 ingest 会改变点数或词表 → 键变化 → 旧缓存自然不被命中
            · 指纹查询是**纯内存读**（`store.count()` 与 BM25 词表长度都在本地），
              不像点数要从 Qdrant 往返；失败时降级为 `None`，不影响缓存工作

        指纹只回答"语料变过吗"，不回答"变的内容与这个问题相关吗"——
        后者做不到（要重算才知道），故这是**失效探测的合理下界**：
        宁可在语料变化后多算几次，也不要一直返回旧数据。
        """
        key = (question, top_k, self._corpus_fingerprint())
        with self._cache_lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache_hits += 1
                # ★ 标注缓存命中：命中与未命中的延迟差**两个数量级**
                # （实测 未命中 30222ms 含首次加载嵌入模型 / 命中 204ms）。
                # 不标注的话，用户看到"检索 0.2s"或"检索 30s"会以为是随机抖动。
                #
                # ★ 这里**直接产出一个 `RagTiming`**，而不是去改某个已存在的对象
                # ------------------------------------------------------
                # 命中分支是**提前 return** 的，根本不经过 `search()`，
                # 故 `finish()` 从未执行、线程本地快照也是空的 ——
                # 表现就是"缓存命中时 `rag_timing` 为 null"（实测确认）。
                # 而缓存命中恰恰是**最需要被标注**的场景（差两个数量级）。
                #
                # 不依赖任何上一步留下的状态：需要一个 `cache_hit=True,
                # total_ms=0, steps={}` 的记录，就地造一个最直接。
                # 同一轮内两次命中会各得一条，`summarize()` 会把
                # `count` 与 `cache_hits` 分别累加。
                try:
                    from src.rag.timing import RagTiming, _last_marks

                    t = RagTiming(
                        total_ms=0,
                        steps={},
                        cache_hit=True,
                        result_count=len(self._cache[key]),
                    )
                    _last_marks.marks = t.marks()
                except Exception:  # noqa: BLE001 - 计时不影响检索
                    pass
                return list(self._cache[key])

        result = tuple(self.search(question, top_k))  # 异常不写入缓存

        with self._cache_lock:
            self._cache[key] = result
            self._cache_misses += 1
            while len(self._cache) > _CACHE_MAXSIZE:
                self._cache.popitem(last=False)
        return list(result)

    def _corpus_fingerprint(self) -> tuple | None:
        """语料指纹：`(点数, 词表长度)`。取不到返回 None（缓存照常工作）。

        ★ 为什么点数要**惰性缓存**而不是每次查（实测数据）
        ---------------------------------------------
        第一版每次调 `store.count()`（Qdrant `exact=True` 计数往返），
        实测**单次 208ms**。而缓存命中本该接近 0——

        实测对比：
            未命中（真检索）  30222ms（含首次加载嵌入模型）
            命中（带指纹查询） 204ms    ← 几乎全是指纹的开销

        即"为省 IO 而做的优化"自己变成了 IO。故改为：

            点数      首次调用时查一次并缓存，之后 O(1)；
                      `invalidate_corpus_fingerprint()` 可在 ingest 后刷新
            词表长度  纯内存读（BM25 词表长度），零成本

        ingest 后若没调刷新，点数会停在旧值 —— 但**词表长度仍会变**
        （新增语料通常带来新词），故不是完全失效。两路互补：
        调用方主动刷新最可靠，忘了刷新仍有词表这道兜底。
        """
        words = None
        # 用 `bm25` 属性（惰性加载词表）而不是裸 `_bm25` 字段：
        # 首次检索前 `_bm25` 还是 None，用它会让词表信息恒为 None，
        # 于是指纹退化成只有点数（少一道失效探测）。
        #
        # ⚠️ 读 `vocab` / `n_docs` 属性，**不要调 `_weights()`**——
        # 后者签名是 `_weights(tokens, doc_length)`，需要实际分词结果，
        # 用它当"词表长度"会 TypeError（实测踩过）。
        # `vocab`（dict）与 `n_docs`（int）都是加载后的现成字段，零成本。
        try:
            bm25 = self.bm25
        except Exception:
            bm25 = None
        if bm25 is not None:
            try:
                vocab = getattr(bm25, "vocab", None)
                n_docs = getattr(bm25, "n_docs", None)
                words = (len(vocab) if vocab is not None else None, n_docs)
            except Exception:
                words = None

        if self._corpus_count is None:
            try:
                self._corpus_count = self.store.count()
            except Exception:
                # 查不到就保持 None：下次再试，不影响缓存工作
                pass

        if self._corpus_count is None and words is None:
            return None
        return (self._corpus_count, words, self._vocab_stamp())

    def _vocab_stamp(self) -> int | None:
        """词表文件的 mtime（纳秒）。**这是"改内容"唯一能检测到的信号**。

        ★ 为什么必须加它（实测确认的缺陷）
        -----------------------------
        原来的指纹是 `(点数, (词表长度, n_docs))`。而 QA 语料的 point id 是
        **顺序整数**（`ingest.py`: `id=idx`），且写入走 `upsert`：

            · **改某条问答的答案**（条数不变）→ 点数、词表长度、n_docs **全不变**
              → 缓存键不变 → **继续返回修改前的旧答案**
            · **两条问答换序**（条数不变）→ 同上，而且更隐蔽：
              id 指向了**不同内容**，缓存却认为"语料没变"

        （`regulation_store` 那条路用 `law|article` 的 UUID5 做 id，
        改内容和换序都不改 id，同样检测不到 —— 两个集合都有这个盲区。）

        ★ 为什么用 vocab 文件的 mtime 而不是内容哈希
        -------------------------------------
        内容哈希要在**每次缓存查询**上读全部语料（565 条）—— 那正是我们要避免的 IO。
        而 `ingest` 每次都会重写 `data/vocab.json`（`bm25.save()`），
        所以它的 mtime **等价于"语料被重新导入过"**，成本是一次 `stat`（微秒级）。

        ★ 代价与边界（如实说明）
        -------------------
        这是个**保守**信号：mtime 变 ≠ 内容一定变（重跑一次 ingest 而语料没改，
        也会让缓存整体失效一次）。代价是"多算几次检索"——
        与"一直返回旧数据"相比，这个方向是安全的。
        反过来，**若有人绕过 ingest 直接改 Qdrant**，mtime 不会变 ——
        那条路检测不到，只能靠 `invalidate_corpus_fingerprint()` 显式刷新。
        """
        try:
            return VOCAB_FILE.stat().st_mtime_ns
        except OSError:
            # 文件不存在（没跑过 ingest）或读不到 → 退化为 None，不阻塞缓存
            return None

    def invalidate_corpus_fingerprint(self) -> None:
        """让语料指纹在下次检索时重算。**ingest 之后应调用**。

        ★ 为什么还要显式调用（指纹里已经有 mtime 了）
        ----------------------------------------
        指纹的第三个分量是 vocab 文件的 mtime，**每次检索都重新 stat**
        （`_vocab_stamp` 不做缓存），所以 ingest 后 mtime 变了会被自动发现 ——
        这一点上显式调用不是必需的。

        但 `_corpus_count` 是**惰性缓存**的：同一进程里 ingest 后点数变了，
        若不清它，指纹的第一个分量会停在旧值。mtime 变了仍会让整个键不同
        （所以不会返回旧数据），但两个分量指向"不同代"的语料并不干净。
        故 ingest 之后调用它，让三者一致。
        """
        self._corpus_count = None

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

    def search(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        *,
        min_relevance: float | None = None,
    ) -> list[dict]:
        """混合检索，返回 `[{question, answer, score, relevance}]`（契约字段，勿改名）。

        **问题分类驱动的参数选择**：先对问题分类（关键词 / 自然语言 / 法条引用），
        按类别取不同的 RRF k 与召回深度——不同问法对 dense 与 sparse 两路的倚重不同，
        分类后调参比"一套参数走天下"的召回质量更稳。

        ★ 相关性过滤（`min_relevance`，默认 `MIN_RELEVANCE`）
        --------------------------------------------------
        RRF 融合**保证任何查询都有 top_k 条结果**——无关问题（实测"冰岛的首都是哪里"
        最高余弦相似度仅 0.284）也会拿到 5 条。不过滤就等于把纯噪音喂给生成端，
        而"知识库没有答案"这个判断会退化成**依赖大模型自己识破噪音**。

        为什么必须有这一层（实测数据）：
            完全无关问题 top1 相似度   0.284 ~ 0.421
            同领域但库内缺失           0.585 ~ 0.598
            评测题（都应有答案）       0.892 ~ 0.987   ← 与上面完全不重叠

        阈值 0.5 能把"完全无关"挡掉且**不误伤任何一道有答案的题**；
        但挡不住"同领域但库里确实没有"的那类（0.585/0.598 高于阈值）——
        那种仍要靠生成端如实说明，两者互补而非替代。

        `relevance` 为 `None`（取向量失败）时**不过滤**：未知不等于不相关，
        基础设施出问题时宁可返回结果，也不要误判成"无答案"。
        这与 `/api/ask` 的既有口径一致（见 `api/server.py` 的说明）。

        `min_relevance=None` 用 `MIN_RELEVANCE`；显式传 `0.0` 关闭过滤
        （语料自检等需要看全量结果的场景）。

        ★ 耗时打点（`src/rag/timing.py`）
        ------------------------------
        本方法此前**完全没有打点**，于是"RAG 慢"只有一个总数（上层 `phase_times`
        的"检索"），看不出慢在哪一步。五步的优化方向完全不同：
            查询向量化（嵌入模型首次要加载，实测 10s 量级）
            稀疏编码（纯内存，<1ms）
            混合检索（Qdrant 往返，主要延迟来源）
            精排（CrossEncoder 模型加载 + 逐条打分）
            相关性过滤（纯计算）
        计时通过 ContextVar 旁路传出（见该模块文件头），**不改本方法签名**。
        """
        from src.rag.timing import begin, finish, step

        _timing = begin()
        _t_start = time.perf_counter()

        with step(_timing, "encode_query"):
            dense = self.embedder.encode_query(question)
        with step(_timing, "encode_sparse"):
            sparse = self.bm25.encode_query(question)

        # 分类 → 检索参数（rrf_k / 双路权重 / prefetch_limit / top_k）
        params = get_params(question)
        rrf_k = params["rrf_k"]
        prefetch = params["prefetch_limit"]
        # 权重与 k 一起透传给融合层（服务端与应用层两条路径都要收到）
        dense_weight = params.get("dense_weight", 1.0)
        sparse_weight = params.get("sparse_weight", 1.0)
        # 调用方显式传了 top_k 时尊重调用方（评测、测试可能要不同条数）
        effective_top_k = top_k

        # ★ 这两个标志必须在分支**之前**初始化
        # 放在 else 分支里的话，"未启用精排"路径不会定义它们，
        # 收尾的 `finish(...)` 会 NameError —— 而那条路径是**默认路径**。
        _reranked = False
        _rerank_failed = False

        if not settings.rerank_enabled:
            with step(_timing, "hybrid_retrieve"):
                records = self._hybrid_retrieve(
                    dense, sparse, rrf_k, prefetch, effective_top_k,
                    dense_weight, sparse_weight,
                )
        else:
            # 精排需要更大的候选池才有意义：先多召回，再由 CrossEncoder 收敛到 top_k
            with step(_timing, "hybrid_retrieve"):
                candidates = self._hybrid_retrieve(
                    dense, sparse, rrf_k, prefetch,
                    max(effective_top_k, RERANK_CANDIDATE_K),
                    dense_weight, sparse_weight,
                )
            try:
                with step(_timing, "rerank"):
                    records = self.reranker.rerank(question, candidates, effective_top_k)
                _reranked = True
                _rerank_failed = False
            except Exception as exc:
                # 精排是**增益项**而非必需项：模型下载失败、显存不足、文件损坏都不该让
                # 整个检索失败——退回混合检索顺序，质量略降但用户仍拿得到答案
                logger.warning("精排失败，退回混合检索顺序: %s", exc)
                records = candidates[:effective_top_k]
                _reranked = False
                _rerank_failed = True

        with step(_timing, "relevance_filter"):
            result = self._apply_relevance_filter(records, min_relevance)

        # ★ 收尾写总耗时与上下文。放在 `return` 前而不是 `finally`：
        # 异常路径会让整个检索失败（上层会转成 error），此时计时没有消费方，
        # 但**失败那一步的耗时仍留在 `_timing.steps` 里**，调试时可以从
        # `current()` 取到——这是 `step()` 异常时照样记账的意义。
        try:
            finish(
                _timing, _t_start,
                result_count=len(result),
                question_category=str(params.get("category") or ""),
                reranked=_reranked,
                rerank_failed=_rerank_failed,
            )
            # ★ 兜底通道：把结果**也**挂到实例上
            #
            # 为什么 ContextVar 不够（实测踩到的关键点）
            # ------------------------------------
            # ★ 计时快照改由 `finish()` 存到**本线程的 ContextVar**
            # （见 `src/rag/timing.py` 的 `finish()` / `take_marks()`）。
            #
            # 这里**不再**写实例属性 `self.last_timing`。第一版写的是它，
            # 但那是**单例上的共享字段**：不同请求之间会互相覆盖 ——
            # 请求 B 的检索盖掉 A 的值，A 读回时就拿到 B 的计时
            # （张冠李戴、且没有任何报错）。"同名工具顺序执行"只能保证
            # **单个请求内部**不冲突，挡不住**请求之间**。
            #
            # 改为 ContextVar 后，写入（`finish()`）与读取（`take_marks()`）
            # 都在**同一个检索线程**内，天然按请求隔离。
        except Exception:  # noqa: BLE001 - 计时不影响检索
            logger.debug("RAG 计时收尾失败", exc_info=True)
        return result

    @staticmethod
    def _apply_relevance_filter(
        records: list[dict], min_relevance: float | None
    ) -> list[dict]:
        """按 `relevance` 阈值裁掉不相关结果。见 `search` 的说明。"""
        threshold = MIN_RELEVANCE if min_relevance is None else min_relevance
        if threshold <= 0:
            return records  # 显式关闭过滤
        kept: list[dict] = []
        for record in records:
            relevance = record.get("relevance")
            # None / 非数值 = 取向量失败（未知），不是"不相关" → 保留
            if not isinstance(relevance, (int, float)):
                kept.append(record)
            elif relevance >= threshold:
                kept.append(record)
        if len(kept) != len(records):
            logger.debug(
                "相关性过滤：%d 条中保留 %d 条（阈值 %.2f）",
                len(records), len(kept), threshold,
            )
        return kept

    def _two_path_search(
        self, dense: list[float], sparse, limit: int
    ) -> tuple[list[dict], list[dict]]:
        """跑 dense / sparse 两路检索，返回 `(dense_results, sparse_results)`。

        ★ 为什么并发（实测：一次 `search()` 要 **3 次** Qdrant 往返）
        ------------------------------------------------------
        这条链路上每次检索的往返是：

            ① `dense_search`      ┐ 互不依赖，却按顺序发
            ② `sparse_search`     ┘
            ③ `_fetch_vectors`    取向量算 relevance（依赖①的结果 id）

        而 Qdrant 在**远端**（本项目实测连的是 Qdrant Cloud，跨区往返），
        所以这三步的耗时几乎全是网络等待，不是本地计算：实测
        `hybrid_retrieve` 占 `search()` 墙钟的 **~99%**（encode_query 热态 15~17ms、
        encode_sparse 热态 0ms、relevance_filter 0ms，其余全在这一步）。

        ①② 之间**没有任何依赖**（一个要 dense 向量、一个要 sparse 向量），
        串行发等于把两次网络等待叠加。并发后这一步的墙钟接近"单次往返 +
        融合计算"。③ 依赖①的 id，无法一起并发——本方法只解决 ①②。

        ★ 为什么**不**改成 Qdrant 服务端 RRF（一次往返完成 ①②③）
        ----------------------------------------------------
        服务端 RRF 与应用层 `fuse_rrf` 的公式不同（实测分数比 ≈1.6、
        同一条记录两边排序不同，见 `_hybrid_retrieve` 的文档），
        换过去会让"问题分类 → rrf_k"这套已标定的参数失去意义。
        并发是**不改排序语义**的那一半优化。

        ★ 失败语义与串行**完全一致**
        -------------------------
        任一路抛异常 → 本方法抛出（先拿到的那个异常），调用方看到的仍是
        "检索失败"。串行时 dense 抛了就不会再跑 sparse，这里则两路都已发出，
        但**对外结论一样**（异常照样上抛，返回值为空）。
        用 `with` 而非手工 shutdown：确保异常路径上两个线程都已收尾，
        不会留下悬空线程把日志上下文带乱。

        可用 `RAG_PARALLEL_SEARCH=0` 关掉（退回串行，用于排查环境差异）。
        """
        if not _parallel_search_enabled():
            return (
                self.store.dense_search(dense, limit=limit),
                self.store.sparse_search(sparse, limit=limit),
            )

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="rag-search") as pool:
            dense_future = pool.submit(self.store.dense_search, dense, limit=limit)
            sparse_future = pool.submit(self.store.sparse_search, sparse, limit=limit)
            # 两个结果都要取：即便 dense 先抛，也要等 sparse 收尾（`with` 会 join），
            # 否则线程会在连接上继续跑，异常栈与日志对不上。
            dense_error = sparse_error = None
            dense_results: list[dict] = []
            sparse_results: list[dict] = []
            try:
                dense_results = dense_future.result()
            except Exception as exc:  # noqa: BLE001
                dense_error = exc
            try:
                sparse_results = sparse_future.result()
            except Exception as exc:  # noqa: BLE001
                sparse_error = exc
        if dense_error is not None:
            raise dense_error
        if sparse_error is not None:
            raise sparse_error
        return dense_results, sparse_results

    def _hybrid_retrieve(
        self,
        dense: list[float],
        sparse,
        rrf_k: int,
        prefetch_limit: int,
        top_k: int,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
    ) -> list[dict]:
        """执行混合检索。

        k == 60（Qdrant 内置默认值）时走服务端**加权** RRF；
        k ≠ 60 时改为两路独立查询 + 应用层 `fuse_rrf`——多一次网络往返但 k 可控。

        ★ **两条融合路径的公式不一致，故统一走应用层**（实测发现，重要）
        ------------------------------------------------------------
        原实现按 `k == 60` 分流：k=60 走 Qdrant 服务端 RRF，其余走应用层 `fuse_rrf`。
        实测发现**两条路径公式不同**：

            同一候选集、同为均权 (0.5,0.5)：
                服务端 分 = 0.02416518   应用层 分 = 0.01557765
                服务端 分 = 0.02371674   应用层 分 = 0.01389127
                （比值稳定 ≈ 1.6，且**同一条记录两边排序不同**）

        即服务端 RRF 的基准 k 不是 60。后果是**同一问题只因分类不同就用不同公式排序**：
        分类器给 `sentence` 类 k=60（走服务端）、`keyword` 类 k=20（走应用层）。

        实测延迟代价为**零**（各 8 次取中位：服务端 435.2ms / 应用层 434.7ms，1.00x），
        因为每次搜索的固定成本远大于一次本地往返。故统一走应用层——
        换来公式可控、结果可复现、调参有意义（服务端公式还可能随版本变）。

        ★ 两条路径都必须给出 `relevance`（与查询的稠密余弦相似度）
        ------------------------------------------------------
        `score` 是 RRF 融合分，只反映"候选内排名"，**任何查询都会有第一名**，
        无法用来判断"这条到底相不相关"（实测无关问题的 RRF 分甚至能高于相关问题）。
        `relevance` 才是可判相关的指标（实测相关问 0.67~0.71、无关问 0.29~0.34）。
        应用层路径在此处补算 relevance。
        """
        # 两路独立检索 + 应用层加权融合。
        #
        # 不再按 k 分流到服务端：见 docstring——两条路径公式不同，
        # 会让"分类结果"意外决定排序公式，而统一的应用层延迟代价为零。
        #
        # ★ 两路**并发**发出（见 `_two_path_search`）：它们互不依赖，
        #   而实测这一步占了 `search()` 墙钟的 ~99%（远端 Qdrant 的网络等待）。
        dense_results, sparse_results = self._two_path_search(dense, sparse, prefetch_limit)
        merged = fuse_rrf(
            dense_results,
            sparse_results,
            k=rrf_k,
            limit=top_k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
        )

        # 补算 relevance（需要记录带 id，两个 search 都已保留）。
        #
        # 用 getattr 探测而非直接调用：`attach_relevance` 是 VectorStore 的方法，
        # 而测试与调用方会注入**替身 store**（只需实现 search 系列即可工作）。
        # 直接调用会让所有替身在运行时报 AttributeError——实测已因此挂掉
        # 4 个既有测试。relevance 是**增强信息**，拿不到不应影响检索本身，
        # 故这里降级为"不补算"，而不是让整个检索失败。
        attach = getattr(self.store, "attach_relevance", None)
        if callable(attach):
            try:
                return attach(merged, dense)
            except Exception as exc:
                logger.warning("补算 relevance 失败（检索结果仍返回）: %s", exc)
        return merged

    # ---- 生成 ----

    @staticmethod
    def _retrieval_query(question: str, history: list[dict] | None) -> str:
        """构造检索式：多轮对话时把上一轮用户问题拼进来。

        ★ 为什么需要（实测数据，见 docs/25）
        ----------------------------------
        真实使用里用户的第二句几乎总是省略式的：

            用户：投标保证金的退还时限是多久？
            用户：那履约保证金呢？          ← 检索式只有这 7 个字

        `search_cached` 原先直接用 `question` 检索，**不做指代消解**，
        而"那履约保证金呢"里的"那"没有指代对象，检索质量明显下降。实测：

            只用追问当检索式      平均最高相似度 0.627（5 题中 1 题检索不到）
            拼接上一轮用户问题     平均最高相似度 0.868（5 题全中）
            人工改写成完整问题     平均最高相似度 0.908（上界）

        拼接式已拿到上界的 96% 收益，且**不增加任何 LLM 调用**——
        相比之下，"用 LLM 做查询改写"要多一次几百毫秒的调用，
        在本项目的语料规模下不值得。

        ★ 为什么只拼上一轮、且只拼用户问题
        ---------------------------------
        · 只拼**上一轮**：更早的轮次与当前问题的指代关系很弱，
          拼进来反而引入无关词。实测本轮场景下一轮足够。
        · 只拼**用户问题**（不拼助手回答）：助手回答通常很长
          （实测平均 480 字），拼进来会把检索式稀释成"什么都有点像"。

        ★ 为什么不改 `question` 本身
        ---------------------------
        检索式与"用户问的问题"是两件事。生成阶段仍必须用原始问题，
        否则模型会把上一轮的问题也当成待答内容，答出多余的东西。
        故本函数只在**检索**路径上使用。
        """
        if not history:
            return question
        for item in reversed(history):
            if not isinstance(item, dict) or item.get("role") != "user":
                continue
            previous = (item.get("content") or "").strip()
            # 上一轮问题为空或与当前问题相同（重复提问）时不必拼
            if previous and previous != question.strip():
                return f"{previous} {question}"
            break
        return question

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
        messages: list[dict] = [{"role": "system", "content": ACTIVE_SYSTEM_PROMPT}]

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
            # 检索式与"用户问的问题"是两件事：多轮时把上一轮问题拼进来做指代消解
            # （见 `_retrieval_query` 的实测数据）。生成阶段仍用原始 `question`。
            sources = self.search_cached(self._retrieval_query(question, history), top_k)
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

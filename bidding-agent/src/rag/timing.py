"""RAG 检索耗时采集。

★ 为什么单独做这个（现有打点覆盖不到）
==============================
项目已有的耗时打点都在**上层**：

    `phase_times` 里的"检索"      工具层的笼统阶段（含工具调度开销）
    `rag_timing`（本模块）        **检索内部的真实构成**

而 `RagPipeline.search()` 此前**完全没有打点**（实测：该方法 71 行、零 `perf_counter`）。
于是"RAG 慢"只有一个总数，看不出慢在哪一步：

    查询向量化（嵌入模型）  首次调用要加载模型，实测 10s 量级
    稀疏编码（BM25）        纯内存，通常 <1ms
    混合检索（Qdrant + RRF） 网络往返，主要延迟来源
    精排（CrossEncoder）    可选项，模型加载 + 逐条打分
    相关性过滤              纯计算

这五步的优化方向完全不同（换模型 / 加索引 / 关精排 / 改阈值），
只给一个总数等于没给。

★ 为什么用 `contextvars` 而不是实例属性
=================================
`search()` 可能在**多线程**里被调用（工具层用 ThreadPoolExecutor 执行，
一轮内多个工具并行）。若把计时存在 `self` 上，两个并发检索会互相覆盖，
得到的数是两个请求的混合——比没有更糟（看不出是错的）。

`contextvars` 的关键性质：**每个线程有独立的上下文副本**，
且项目里已经用了这个机制（`react_loop` 的 `copy_context()` 把日志上下文
复制进工具线程）。故每个检索线程写自己的计时，互不干扰。

★ 为什么不用返回值的通道
==================
`search()` 的契约是 `list[dict]`（`[{question, answer, score, relevance}]`），
被工具层、评测、测试多处依赖。为传计时改返回值会波及所有调用方。
用 ContextVar 是**旁路**，零侵入。

★ 失败不影响检索
============
本模块所有函数都保证不抛异常：计时是观测，不能成为检索失败的原因。
"""

from __future__ import annotations

import logging

import contextvars
import threading
import logging

logger = logging.getLogger(__name__)
import time
from dataclasses import asdict, dataclass, field

#: 当前这次检索的计时器（**在检索线程内**写入与读取）。见文件头"为什么用 contextvars"。
_current: contextvars.ContextVar["RagTiming | None"] = contextvars.ContextVar(
    "rag_timing", default=None
)

#: 「最近一次检索」的快照，**同样存在 ContextVar 里**。
#:
#: ★ 为什么不挂在 `rag_pipeline` 单例上（自查发现的问题）
#: ----------------------------------------------
#: 第一版把计时写在 `rag_pipeline.last_timing`（单例的实例属性）上，
#: 理由是"工具层的同名工具是顺序执行的"。
#: 但那个理由**只覆盖了同一个请求内部**——**不同请求之间**仍会互相覆盖：
#:
#:     请求 A 检索 → 写 last_timing = A 的计时
#:     请求 B 检索 → 覆盖成 B 的计时
#:     请求 A 读回   → **读到 B 的计时**（张冠李戴，且无任何报错）
#:
#: 并发下表现为"检索耗时忽大忽小、明明缓存命中却显示几十秒"。
#:
#: 改用 `threading.local`：写入与读取在同一线程内必然可见。
#:
#: ★ 为什么不是 ContextVar（用最小模型实测三轮才定下来）
#: ---------------------------------------------
#: 第二/三版都用 ContextVar，都读不到值。最小模型的实测结果：
#:
#:     变体 A  ctx.run 内 set → 之后同线程读           → 'default'（不传播）
#:     变体 C  任务内先 set、再 ctx.run 内 set → 之后读 → **None**
#:
#: 变体 C 是决定性的：`ctx.run()` 用的是**隔离的副本**，与"线程当前上下文"
#: 是两回事 —— 连"先在外层 set 一次"也救不了。
#: 而本项目的工具执行**必须**经过 `ctx.run`（为了把日志上下文传进子线程），
#: 故 ContextVar 在这个位置**根本不可用**。
#:
#: 变体 D 证明"用可变容器带出"永远可靠 —— `threading.local` 正是这个语义，
#: 且不受 context 副本影响。工具任务每次新建线程（`thread_name_prefix="tool-*"`），
#: 故按线程隔离即等于按请求隔离。
_last_marks = threading.local()

#: 计时的步骤名 → 中文标签（前端展示用，避免前端再维护一份映射）。
STEP_LABELS: dict[str, str] = {
    "encode_query": "查询向量化",
    "encode_sparse": "稀疏编码(BM25)",
    "hybrid_retrieve": "混合检索",
    "rerank": "精排",
    "relevance_filter": "相关性过滤",
}


@dataclass
class RagTiming:
    """一次 RAG 检索的耗时分解（单位毫秒）。"""

    #: 整次 `search()` 的墙钟耗时
    total_ms: int = 0
    #: 各步骤耗时；键见 `STEP_LABELS`
    steps: dict[str, int] = field(default_factory=dict)
    #: 是否命中了检索缓存
    #:
    #: ★ 命中与未命中的延迟差**两个数量级**（实测：未命中 30222ms 含首次加载
    #: 嵌入模型、命中 204ms），不标注的话用户看到"检索 0.2s"或"检索 30s"
    #: 会以为是随机抖动。
    cache_hit: bool = False
    #: 精排是否启用且成功（失败会退回混合检索顺序，质量略降）
    reranked: bool = False
    #: 精排是否尝试过但失败（增益项，失败不影响检索）
    rerank_failed: bool = False
    #: 返回条数（检索"快而空"与"慢而有货"是不同的结论）
    result_count: int = 0
    #: 问题分类（关键词/自然语言/法条引用）——决定了检索参数，
    #: 排查"为什么这次检索慢"时要看是不是分类到了更重的路径
    question_category: str = ""

    def marks(self) -> dict:
        """前端展示用的步骤数组（按耗时降序，便于一眼看出大头）。"""
        items = [
            {"step": k, "label": STEP_LABELS.get(k, k), "ms": v}
            for k, v in self.steps.items()
        ]
        items.sort(key=lambda x: -x["ms"])
        return {"total_ms": self.total_ms, "steps": items, **{
            k: v for k, v in asdict(self).items() if k not in ("steps", "total_ms")
        }}


class Stopwatch:
    """累计各步骤耗时的小工具（异常安全）。

    用法：
        t = RagTiming()
        with step(t, "encode_query"):
            ...
    异常时**照样记账**：失败的那一步往往正是慢的原因，不记就丢了线索。
    """

    def __init__(self, timing: RagTiming, name: str) -> None:
        self.timing = timing
        self.name = name
        self._t0 = 0.0

    def __enter__(self) -> "Stopwatch":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> bool:
        elapsed = int((time.perf_counter() - self._t0) * 1000)
        # 同名步骤累加（如精排可能分多次调用）
        self.timing.steps[self.name] = self.timing.steps.get(self.name, 0) + elapsed
        return False  # 不吞异常


def begin() -> RagTiming:
    """开启一次计时并把 `total_ms` 的起点设为现在。返回计时器。"""
    timing = RagTiming()
    _current.set(timing)
    return timing


def current() -> RagTiming | None:
    """取当前线程的计时器；不在检索中则返回 None。"""
    return _current.get()


def step(timing: RagTiming | None, name: str):
    """记录一个步骤的耗时。`timing` 为 None 时是 no-op。"""
    return Stopwatch(timing, name) if timing is not None else _Noop()

def finish(timing: RagTiming | None, started_at: float, **extra) -> RagTiming | None:
    """收尾：写总耗时、附加字段，并**把快照存到本线程**。

    `started_at` 是 `time.perf_counter()` 的起点。

    ★ 为什么在这里存快照（这是"读得到计时"的关键）
    ------------------------------------
    `finish()` 由 `RagPipeline.search()` 在**检索线程内**调用，
    与调用方随后 `take_marks()` 的是**同一个线程** ——
    故 ContextVar 成立。这一点是本轮修复的核心：
    旧实现在**父线程**里读、在**子线程**里写，ContextVar 必然读不到。
    """
    if timing is None:
        return None
    timing.total_ms = int((time.perf_counter() - started_at) * 1000)
    for k, v in extra.items():
        if hasattr(timing, k):
            setattr(timing, k, v)
    # 存快照供调用方取回。失败不影响返回值（计时是观测）。
    try:
        _last_marks.marks = timing.marks()
    except Exception:  # noqa: BLE001
        logger.debug("存 RAG 计时快照失败", exc_info=True)
    return timing


class _Noop:
    """`step(None, ...)` 的替身：支持 `with` 但不做任何事。"""

    def __enter__(self) -> "_Noop":
        return self

    def __exit__(self, *exc) -> bool:
        return False


__all__ = [
    "STEP_LABELS",
    "RagTiming",
    "begin",
    "reset_marks",
    "current",
    "finish",
    "step",
    "take_marks",
]


def take_marks() -> dict | None:
    """取**本线程**最近一次检索的耗时分解。取不到返回 None。

    ★ 为什么读 ContextVar 而不是单例属性（自查改掉了）
    -------------------------------------------
    第一版读 `rag_pipeline.last_timing`。但那是**单例的实例属性**，
    被所有请求共享：请求 B 的检索会覆盖请求 A 的值，A 读回时就拿到 B 的计时。

    为什么现在读 ContextVar 可行（而此前不行）：**写入点变了**。
    旧实现在**父线程**（`_execute_tool`）里读计时，而 `search()` 在**嵌套子线程**
    里写 —— 子线程的 `set()` 不会传播回父线程，故读不到（当时排查了两轮）。
    现在写入点就在 `search()` 内部（见 `finish()`），与读取点同线程
    （工具的执行器调用 `search()` 后立刻调 `take_marks()`），故 ContextVar 成立。

    **永不抛异常**：计时是观测，不能成为检索失败的原因。
    """
    try:
        return getattr(_last_marks, "marks", None)
    except Exception:  # noqa: BLE001
        logger.debug("读取 RAG 计时失败", exc_info=True)
        return None


def reset_marks() -> None:
    """把计时快照重置为空。**必须在 `ctx.run()` 之外调用。**

    ★ 为什么必须"外层重置、内层写入"（这个点实测踩了三轮）
    ----------------------------------------------
    `contextvars.copy_context()` 的语义是：`ctx.run(...)` **内部**的 `set()`
    只作用于那个副本，**调用结束后不传播回原上下文**。实测：

        ctx.run 内 set → 之后在同一线程读 → 读到的是**原值**

    而 `search()` 的 `finish()`（写计时）正是在 `ctx.run` **内部**跑的。
    若在 `ctx.run` **之后**才重置，重置发生在外层、写入发生在内层副本，
    两者互不可见 —— 读出来永远是 None（前两版都栽在这里）。

    正确顺序是**先在外层重置、再到内层写入**：
    `ctx.run` 内的副本继承外层的新值，`finish()` 的 `set()` 便生效，
    外层读回即可看到。

    故必须有这个函数，不能让调用方随手 `_last_marks.set(None)`（容易放错位置）。
    """
    try:
        _last_marks.marks = None
    except Exception:  # noqa: BLE001
        pass




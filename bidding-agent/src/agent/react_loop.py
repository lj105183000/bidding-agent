"""ReAct 多轮工具循环（ReActMixin）。

事件流架构：mixin 方法均为 generator，产出事件元组——
  "status"|"token"|"thinking", 文本 / "reset", None / "done"|"error", dict
core._chat_events 统一调度，chat / chat_stream 两条路径共用同一流程。

工具执行契约（冻结，见 分配说明.md §5）：TOOL_EXECUTORS[name] 接收一个
arguments dict，返回 (格式化文本, sources 列表)；任何失败都返回结构化
错误文本而非抛异常，Agent 永不因后端宕机崩溃。

注意：本模块与 src.agent.core 各持有一份 TOOL_EXECUTORS 模块级绑定
（此处用于文本兜底过滤），测试替换时必须同时替换两处。
"""

import contextvars
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import TYPE_CHECKING, Any
from collections.abc import Iterator

from src.agent.constants import (
    BASE_TOOL_NAMES,
    CLARIFY_TOOL_NAME,
    MAX_CLARIFY_OPTION_CHARS,
    MAX_CLARIFY_OPTIONS,
    MAX_CLARIFY_QUESTION_CHARS,
    MAX_CLARIFY_QUESTIONS,
    MAX_CHARTS,
    MAX_CLARIFY_REASON_CHARS,
    MAX_LOCAL_SOURCES,
    MAX_TOOL_ROUNDS,
    MAX_WEB_SOURCES,
    MIN_CLARIFY_QUESTION_CHARS,
    TOOL_TIMEOUT_SECONDS,
    WEB_TOOL_NAMES,
)
from src.agent.tool_quality import (
    ALTERNATIVES,
    ResultStatus,
    assess_tool_result,
    format_hint,
    format_spin_hint,
)

try:
    # 工具层（成员 C）尚未合入时以空集运行，测试经 monkeypatch 注入
    from src.tools.rag_tools import TOOL_EXECUTORS
except ImportError:  # pragma: no cover - 阶段一工具层未落地
    TOOL_EXECUTORS = {}

try:
    # 并行度复用工具层的既有常量：`docs/组件工作机制.md §13` 与 `tools/base.py`
    # 都已按 4 线程描述，避免两处各写一个数字后跑偏
    from src.tools.base import PARALLEL_MAX_WORKERS as TOOL_PARALLEL_WORKERS
except ImportError:  # pragma: no cover - 工具层未合入
    TOOL_PARALLEL_WORKERS = 4

logger = logging.getLogger(__name__)

#: 同一个工具被调用多少次后判定为检索停滞。
#:
#: 取 3：第 1 次是正常检索，第 2 次可能是换问法补充，
#: **第 3 次还调同一个工具**基本可以确定是在打转（实测出现过 6 次）。
SPIN_REPEAT_THRESHOLD = 3

# 约束重试提示：决策轮检出畸形工具文本时，要求模型重新输出
_TOOL_CONSTRAINT = (
    "你刚才的输出中混入了工具调用格式的文本。"
    "请重新输出：要么只输出工具调用，要么直接给出回答正文，不要混合。"
)
# 最终生成约束：追加在消息末尾，防止模型复读工具调用过程或输出函数格式
_FINAL_CONSTRAINT = (
    "请基于以上工具结果直接输出最终中文回答，"
    "不要输出任何函数调用格式，不要复述你的分析过程。"
)


# ---------------------------------------------------------------------------
# 「给用户看的质量说明」旁路（局限 6 的修复）
# ---------------------------------------------------------------------------
# 背景：tool_quality.auto_switch_hint 早就实现了，但**零调用点**
#       —— 它只是「就绪的能力」，用户从未真的收到过那句话。
#
# 为什么用模块级 list 而不是实例属性：
#   _validate_result 是 @staticmethod（没有 self）。改成实例方法会动到
#   调用方与既有测试；模块级旁路改动面最小。
#
# 线程安全：
#   并发工具批次的每个任务在自己线程里 append，CPython 的 list.append 是原子的。
#   唯一风险是**跨请求混入** —— 故每次问答开始时 reset（见 core._chat_events）。
_USER_HINTS: list[str] = []


def reset_user_hints() -> None:
    """清空累积的用户提示。**每次问答开始时调用**，避免跨请求混入。"""
    _USER_HINTS.clear()


def take_user_hints() -> list[str]:
    """取出并清空。返回副本，调用方可安全持有。"""
    out = list(_USER_HINTS)
    _USER_HINTS.clear()
    return out


def _note_user_hint(verdict, name: str) -> None:
    """把「给用户看的质量说明」记到旁路，供 done 帧取用。

    去重：同一轮里多个工具都质量低时那句话会重复，对用户只是噪音。
    吞异常：这是**观测**，不能因为它的任何问题影响问答
           （与 _trace、monitoring.safe_record 同一原则）。
    """
    try:
        from src.agent.tool_quality import auto_switch_hint

        msg = auto_switch_hint(verdict, name)
    except Exception:  # noqa: BLE001 - 观测不能影响问答
        return
    if msg and msg not in _USER_HINTS:
        _USER_HINTS.append(msg)


class ReActMixin:
    """ReAct 多轮工具循环：决策（chat_raw）→ 解析调用 → 执行 → 结果回填，
    直到模型不再调用工具或达到 MAX_TOOL_ROUNDS 上限，转入最终流式生成。"""

    if TYPE_CHECKING:
        # 仅供 IDE/类型检查解析：这些方法分别来自 ToolDefenseMixin / GenerationMixin，
        # 由 BiddingAgent 组合各 mixin 后才真正存在，运行时此处不产生任何属性
        def _detect_tool_text(self, text: str) -> bool: ...
        def _parse_tool_text(self, text: str) -> list: ...
        def _generate_final(self, messages: list, client, meta: dict,
                            deep_thinking: bool = False) -> Iterator[tuple]: ...

    # --- 思维链回传（思考模式 + tools 的硬要求） ---

    def _take_reasoning(self, client) -> str:
        """取走本轮决策产出的思维链（并清空属性）。

        ★ 为什么必须"取走"而不是"读一下"
        ---------------------------
        该属性由 `chat_raw` **每轮覆盖**。若只读不清，某轮没产出思维链时
        （普通模式，属性为空串）会保留上一轮的值，于是把**旧的思维链**
        错配到这一轮的消息上——回传给 API 的上下文与实际不符。

        故读完即清，语义等同于"消费一次"。
        """
        value = str(getattr(client, "last_reasoning", "") or "")
        if value:
            try:
                client.last_reasoning = ""
            except Exception:  # noqa: BLE001 - 只读属性/替身对象都不该让问答失败
                pass
        return value

    def _assistant_msg(self, content: str, reasoning: str = "") -> dict:
        """构造 assistant 消息；有思维链时一并带上。

        ★ 为什么必须回传 `reasoning_content`
        --------------------------------
        DeepSeek「思考模式」文档明确：**请求携带 `tools` 时，历史轮次的
        `reasoning_content` 都应回传给 API；若未正确回传，API 会返回 400。**

        本项目的 ReAct 决策轮**每轮都带 `tools`**，故这条是硬约束——
        不回传不是"效果差一点"，而是整个问答直接失败。

        不产出思维链时（普通模式、或非推理模型）不添加该键：
        多传一个空串虽被官方说明为"会被忽略"，但没有必要，
        且会让请求体与普通模式产生无谓差异。
        """
        msg: dict[str, Any] = {"role": "assistant", "content": content}
        if reasoning:
            msg["reasoning_content"] = reasoning
        return msg

    # --- 工具执行 ---

    @staticmethod
    def _allowed_tool_names(meta: dict) -> set:
        """本轮允许执行的工具名集合。

        ★ 为什么需要执行侧的白名单（实测踩坑，这是**安全边界**而非便利性）
        --------------------------------------------------------------
        `_resolve_tool_schemas` 只控制**把哪些工具的定义发给模型**，
        而 `_execute_tool` 用的是全局注册表 `TOOL_EXECUTORS.get(name)` ——
        于是只要模型"凭记忆"输出一个未暴露的工具名，调用照样成功。

        实测：用户在"关闭联网"（web_search_mode=off）下提问，
        模型仍自行调用了 `search_web_cn`，返回 **30 条联网来源**。
        用户明确说了不联网，系统却联网了——这既是行为违背，也可能泄露内网查询。

        `meta["web_search_enabled"]` 由 `core._chat_events` 写入（含自动兜底的结果），
        故这里以它为唯一依据。缺省视为"不暴露联网工具"（保守）。
        """
        names = set(BASE_TOOL_NAMES)
        if meta.get("web_search_enabled"):
            names |= set(WEB_TOOL_NAMES)
        return names

    @staticmethod
    def _take_rag_timing() -> dict | None:
        """取本线程最近一次 RAG 检索的耗时分解；没有则返回 None。

        ★ 为什么要包一层 try/except
        ----------------------
        计时是**观测**，不能因为它的任何问题影响问答。`src/rag/timing.py`
        内部已保证不抛异常，这里是双保险（与 `_trace`、`monitoring.safe_record`
        同一原则）。
        """
        try:
            from src.rag.timing import current

            t = current()
            return t.marks() if t is not None else None
        except Exception:  # noqa: BLE001
            logger.debug("读取 RAG 计时失败（不影响问答）", exc_info=True)
            return None


    def _execute_tool(self, name: str, arguments: dict,
                      allowed: set | None = None, question: str = ""):
        """执行单个工具。返回 (文本, sources, ok)；失败返回结构化错误文本。

        ⚠️ 这是**串行兼容入口**：它把观测值（RAG 耗时分解、图表数据）写回
        `self.last_tool_rag_timing` / `self.last_tool_charts` 两个**实例属性**。
        那条通道只在"一次只跑一个工具"时成立 —— 并行时两个工具会互相覆盖。
        并行路径请用 `_execute_tool_ex`（观测值随结果一起返回）。
        """
        text, sources, ok, obs = self._execute_tool_ex(
            name, arguments, allowed, question
        )
        self.last_tool_rag_timing = obs.get("rag")
        self.last_tool_charts = obs.get("charts") or []
        return text, sources, ok

    def _execute_tool_ex(self, name: str, arguments: dict,
                         allowed: set | None = None, question: str = ""):
        """执行单个工具，**观测值随结果返回**：`(文本, sources, ok, obs)`。

        `obs = {"rag": dict|None, "charts": list, "ms": int}`。
        与 `_execute_tool` 的唯一区别就是观测通道：并行执行时实例属性会被
        别的线程覆盖，必须走返回值。

        `allowed` 非空时做白名单校验：不在名单里的工具**拒绝执行**。
        传入 None 表示不校验（仅供不关心工具暴露范围的直接调用/测试）。

        ★ `question` 必须传下去（实测缺陷）
        ---------------------------------
        所有执行器的签名都是 `(arguments, question="")`，但这里原先只传
        `arguments`——于是工具内部的 `question` 恒为空串，
        联网结果**从未按用户问题重排或补算相关度**
        （`_rerank_web_results` 在 question 为空时直接早退）。

        后果：前端拿到的联网来源没有可用的相关性评分
        （Tavily 只剩引擎分、智谱恒为 None，界面显示"无分数"），
        用户无法判断该不该采信——而"每条结果带相关性评分"正是本功能的明确要求。
        """
        t_begin = time.monotonic()
        if allowed is not None and name not in allowed:
            # 明确告知模型该工具本轮不可用，让它改用其他工具或直接回答，
            # 而不是"失败后重试同一个工具"浪费轮次
            logger.info("拒绝执行未暴露的工具 %s（本轮允许：%s）", name, sorted(allowed))
            return (
                f"工具 {name} 本轮不可用。请改用其他工具或直接依据已有信息回答。",
                [], False, {"rag": None, "charts": [], "ms": 0},
            )

        executor = TOOL_EXECUTORS.get(name)
        if executor is None:
            return (f"工具 {name} 暂不可用（未注册或后端未就绪）。", [], False,
                    {"rag": None, "charts": [], "ms": 0})

        # ★ 必须在 try **之前**初始化：超时/异常分支不会赋值，
        # 而函数末尾的 return 要用到它（第一版放在 try 内，超时路径会 NameError）
        _rag_from_thread = None
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"tool-{name}")
        try:
            # ★ 把日志上下文复制进子线程（实测缺口修复）
            # -----------------------------------------
            # `contextvars` **不会跨线程自动复制**：新线程拿到的是空上下文。
            # 于是工具执行期间的日志没有 `request_id`——而**工具阶段恰恰是最耗时的**
            # （嵌入模型加载、向量检索、Neo4j 查询都在这里）。
            # 实测：`src.rag.embedder` 的"加载嵌入模型"那行就没有 request_id。
            #
            # `copy_context()` 在**当前**线程拍快照，再在子线程里 `run()` 执行，
            # 于是子线程里的日志也带上了本请求的标识。
            # 用 `run()` 而不是给线程设属性：contextvars 的语义要求
            # 在快照上下文中执行，直接设值不会生效。
            ctx = contextvars.copy_context()

            def _run_with_timing():
                """在**子线程内**执行工具，并取走 RAG 计时。

                ★ 为什么必须在这里取（实测踩到的关键点）
                ----------------------------------
                `copy_context()` 是把父线程的上下文**复制到**子线程——
                子线程里 `ContextVar.set()` 的改动**不会传播回父线程**。
                计时器由 `rag_pipeline.search()` 在子线程里 `begin()`，
                故父线程读 `current()` 永远是 None —— 第一版就是这样，
                表现为"面板一直不出现"，而数据其实都记下了。

                故在子线程里执行完就把值取出来，随结果一起返回。
                这也顺带避免了"父线程读到别处残留值"的可能。

                返回 `(result, rag_timing_or_None)`。
                """
                # ★ 重置必须放在 `ctx.run()` **外面**（实测踩到的关键点）
                #
                # `copy_context()` 的语义是：`ctx.run(...)` 内部的 `set()`
                # **只作用于那个副本**，调用结束后**不传播回来**。实测：
                #     在 ctx.run 内读 → 'set-inside'
                #     ctx.run 之后读   → 原值（改动只留在副本里）
                #
                # 故 `finish()` 在 `ctx.run` 内写的计时，外层读不到。
                # 把重置放在外面，写的就是**本线程（外层）的上下文** ——
                # 而 `finish()` 的 `set()` 作用于"当前 context"
                # （即 `ctx.run` 内那个副本，它继承了外层的新值），
                # 于是外层的这次重置对它是可见的，且读完仍是本线程私有。
                from src.rag.timing import reset_marks

                reset_marks()
                out = ctx.run(executor, arguments or {}, question)
                # ★ 从**本线程**读回计时（写入点是 `search()` 内部的 `finish()`）
                #
                # 前两版都读不到，共排查三轮，记下来免得再走：
                #   ① 读 `current()`：`begin()` 在 ctx.run 内调用，
                #      其 set() 不传播出来 → 恒为 None。
                #   ② 读单例 `rag_pipeline.last_timing`：读得到，但那是
                #      **共享字段**，不同请求互相覆盖（A 的检索被 B 盖掉，
                #      A 读到 B 的计时，张冠李戴且无报错）。
                #   ③ 本方案：**重置在外、写入在内**，按线程隔离且不共享。
                from src.rag.timing import take_marks

                return out, take_marks()

            future = pool.submit(_run_with_timing)
            result, _rag_from_thread = future.result(timeout=TOOL_TIMEOUT_SECONDS)
        except FutureTimeoutError:
            logger.warning("工具 %s 执行超时（%ss）", name, TOOL_TIMEOUT_SECONDS)
            return ("工具执行超时，请稍后重试或缩小查询范围。", [], False,
                    {"rag": None, "charts": [],
                     "ms": int((time.monotonic() - t_begin) * 1000)})
        except Exception as exc:
            logger.warning("工具 %s 执行失败: %s", name, exc)
            return ("工具执行失败，已跳过该数据源，请结合其他信息回答。", [], False,
                    {"rag": None, "charts": [],
                     "ms": int((time.monotonic() - t_begin) * 1000)})
        finally:
            # 不等待超时任务：该线程随后自行结束，不阻塞回答流程
            pool.shutdown(wait=False)

        # ★ 工具返回值支持**两种长度**（2 元组 / 3 元组）
        #
        # 2 元组 `(文本, sources)` 是**原有契约**，仍然完全支持 ——
        # 8 个工具里只有 3 个产图表数据，其余没必要为它多返回一个空列表。
        #
        # 3 元组 `(文本, sources, charts)` 是**新增的可选第三项**：
        # 工具把手里的结构化数据直接交出来，替代"从格式化文本里正则解析"
        # （见 `src/agent/chart_data.py` 里为什么必须这么改）。
        #
        # 为什么用"可选第三项"而不是改所有工具：改动面小、且**不会破坏
        # 已有测试替身**（它们返回 2 元组，这里照样认）。
        charts: list = []
        if isinstance(result, tuple) and len(result) == 3:
            text, sources, charts = result
            if not isinstance(charts, list):
                charts = []
        elif isinstance(result, tuple) and len(result) == 2:
            text, sources = result
        else:
            # 执行器未按契约返回二元组：降级为纯文本，不丢弃内容
            text, sources = result, []
        if not isinstance(sources, list):
            sources = []
        # ★ 观测值**随结果返回**而不是写实例属性
        #
        # 为什么不写 `self.last_tool_rag_timing` / `self.last_tool_charts`：
        # 那两个属性是"父线程读得到、又不改冻结契约"的通道，但**只在串行时成立**
        # ——并行跑 3 个工具时，谁最后写完就是谁的值，调用方拿到的计时会张冠李戴
        # （正是当初否掉单例字段 `rag_pipeline.last_timing` 的同一个理由）。
        # 串行入口 `_execute_tool` 仍会把它们写回去，供既有调用方与测试使用。
        return str(text), sources, True, {
            "rag": _rag_from_thread,
            "charts": charts,
            "ms": int((time.monotonic() - t_begin) * 1000),
        }

    @staticmethod
    def _call_key(name: str, arguments: Any) -> str:
        """一次工具调用的**规范化标识**，用于同一轮内的重复调用去重。

        参数排序后序列化：`{"a":1,"b":2}` 与 `{"b":2,"a":1}` 是同一个调用
        （模型每次给的键顺序可能不同，不去规范化会漏掉一半的重复）。
        `default=str` 兜住不可序列化的值（不让去重本身成为失败点）。
        """
        try:
            payload = json.dumps(arguments or {}, sort_keys=True, ensure_ascii=False,
                                 default=str)
        except Exception:  # noqa: BLE001 - 序列化失败就退化为"不去重"
            payload = repr(arguments)
        return f"{name}|{payload}"

    def _execute_tool_batch(self, calls: list, allowed: set | None,
                            question: str) -> list[tuple]:
        """一轮内的多个工具调用：**同轮并行 + 同参去重**。

        返回与 `calls` **等长同序**的列表：`[(text, sources, ok, obs, deduped), ...]`。
        去重命中的那些调用 `deduped=True`，其 text/sources 复用首次执行的结果。

        ★ 为什么并行（实测依据）
        --------------------
        实测 4/4 道复合题都出现"一轮里 2~3 个工具"，而阶段耗时**恰好等于
        各工具耗时之和**：2999/3000、2467/2468、2842/2843、702/702 ms。
        即串行执行，一轮的等待是叠加的。并行后只取决于最慢的一个，
        同一批题可省 **1.1~1.4 秒**（占端到端 6~9%）。

        而 `src/tools/base.py` 的 `ToolRunner`（4 线程/共享截止时刻/异常隔离）
        以及 5 处注释、设计文档 `组件工作机制.md:519` 都**已经按"并行"描述**，
        只有 `react_loop` 这一处还是 `for` 循环逐个等待。

        ★ 为什么复用 `_execute_tool_ex` 而不是接 `ToolRunner`
        ------------------------------------------------
        ToolRunner 的 `ToolResult` 是**冻结的三字段契约**（源码注明"勿随意增删
        字段"），没有位置放本项目的"RAG 耗时分解"与"图表透传"两份观测值；
        而取这两份观测值的代码（工具子线程内 `take_marks()`）是整个模块里
        最容易错的一段（注释里记着三次失败尝试）。接 ToolRunner 就得把它
        搬进 ToolRunner 的 worker —— 收益相同、风险大得多。
        故这里**把 `_execute_tool_ex` 当作并行单元**：它的超时、异常隔离、
        上下文复制、计时取用逻辑一行不动，只在外面套一层并发批次。

        ★ 关于去重（**实测更正**：上游两条解析路径都已去重）
        ------------------------------------------------
        这一层最初是当作"修掉重复调用的浪费"写的，**核实后不成立**：
        `tool_defense._parse_tool_text` 的 **JSON 分支**与 **XML/DSML 分支**
        （`_parse_xml_tool_calls` 末尾）**各自都有**"同名同参去重"
        （两处注释都写着"模型偶发重复输出同一调用，重复执行会放大后端压力"）。

        故线上正常情况下这里**无事可做**（`meta["tool_deduped"]` 恒为 0）。
        保留它是因为 `_execute_tool_batch` 是模块内的公开入口：将来若有新调用方
        （新的编排路径、直接构造调用的工具循环）绕过 `_parse_tool_text`，
        重复调用不会被执行两遍。成本只是每条调用一次 JSON 序列化（微秒级）。

        `meta["tool_deduped"]` 因此是一个**告警位**：它非 0 说明上游解析
        漏掉了重复调用，是信号而不是常态。
        """
        items = list(calls or ())
        if not items:
            return []

        # 1) 去重：同 (name, 规范化参数) 只保留一个待执行项
        unique: list[tuple[str, Any]] = []
        key_to_slot: dict[str, int] = {}
        slots: list[int] = []
        deduped: list[bool] = []
        for call in items:
            name = str((call or {}).get("name") or "")
            arguments = (call or {}).get("arguments") or {}
            key = self._call_key(name, arguments)
            if key in key_to_slot:
                slots.append(key_to_slot[key])
                deduped.append(True)
                continue
            key_to_slot[key] = len(unique)
            slots.append(len(unique))
            deduped.append(False)
            unique.append((name, arguments))

        # 2) 执行待执行项（单条直接跑，省掉线程与上下文的开销）
        results: list[tuple | None] = [None] * len(unique)
        if len(unique) == 1 or TOOL_PARALLEL_WORKERS <= 1:
            for index, (name, arguments) in enumerate(unique):
                results[index] = self._execute_tool_ex(name, arguments, allowed, question)
        else:
            workers = min(TOOL_PARALLEL_WORKERS, len(unique))
            with ThreadPoolExecutor(max_workers=workers,
                                    thread_name_prefix="tool-batch") as pool:
                futures = []
                for name, arguments in unique:
                    # ★ 每个任务一份**独立**的 context 副本，两个原因缺一不可：
                    #   ① 线程不继承 contextvars → 不复制就丢 request_id
                    #      （工具阶段恰恰最耗时，日志里没有 request_id 最难查）
                    #   ② 同一个 Context 不能被两个线程同时进入
                    #      （`ctx.run` 会抛 "cannot enter context: already entered"）
                    ctx = contextvars.copy_context()
                    futures.append(
                        pool.submit(ctx.run, self._execute_tool_ex,
                                    name, arguments, allowed, question)
                    )
                for index, future in enumerate(futures):
                    try:
                        results[index] = future.result()
                    except Exception as exc:  # noqa: BLE001 - _execute_tool_ex 已吞异常，这是最后一道保险
                        logger.exception("工具批量执行出现未兜住异常: %s", exc)
                        results[index] = (
                            "工具执行失败，已跳过该数据源，请结合其他信息回答。",
                            [], False, {"rag": None, "charts": [], "ms": 0},
                        )

        # 3) 按**原始顺序**展开（重复项复用首次结果），保证回填给模型的消息顺序稳定
        out: list[tuple] = []
        for slot, is_dup in zip(slots, deduped):
            text, sources, ok, obs = results[slot]  # type: ignore[misc]
            out.append((text, sources, ok, obs, is_dup))
        return out

    @staticmethod
    def _validate_result(name: str, text: str, ok: bool,
                         sources: list | None = None,
                         allowed: set | None = None) -> str:
        """工具结果的两层校验：**质量判定**（新增）+ 数据来源标签。

        ★ 第一层：质量判定（`tool_quality.assess_tool_result`）
        ---------------------------------------------------
        原先只有下面的来源标签，即**只标注、不检查**：工具返回 0 条也照样
        把"未找到相关内容"喂给模型，模型要么懵、要么拿噪音硬答。

        现在判定结果质量（空 / 不切题 / 服务不可用 / 缺密钥 / 被白名单拒），
        并把结论**拼进工具结果文本**——那是模型下一轮的输入，
        拼在同一处它才会在**同一次决策里**看到"该换哪个工具"。
        单独发消息要等下一轮，模型容易只看结果不看提示。

        ★ 替代建议只提"本轮允许"的工具
        ----------------------------
        `allowed` 是执行侧白名单。关闭联网时若建议"改用 search_web"，
        模型会去调、然后被拒绝，白耗一个轮次（实测过这种来回）。

        `sources` 与 `allowed` 都给了默认值：既有调用方（测试）只传三个参数，
        此时退化为"只加来源标签"，行为与改动前一致。
        """
        if not ok:
            # 失败也要给质量提示：模型需要知道该换工具还是该放弃这一路
            verdict = assess_tool_result(name, text, sources, ok=False, allowed=allowed)
            _note_user_hint(verdict, name)
            return text + format_hint(verdict, name)

        if name == "search_knowledge_graph":
            text = text + "\n\n【数据来源：知识图谱】"
        elif name == "query_database":
            text = text + "\n\n【数据来源：业务数据库】"
        elif name in WEB_TOOL_NAMES:
            text = text + "\n\n【数据来源：联网搜索，请注意信息时效】"

        verdict = assess_tool_result(name, text, sources, ok=True, allowed=allowed)
        if verdict.usable and verdict.status is not ResultStatus.OK:
            logger.info(
                "工具 %s 结果质量偏低：%s（来源 %d 条，最高相关性 %s）",
                name, "；".join(verdict.reasons) or verdict.status.value,
                verdict.source_count, verdict.best_relevance,
            )
        _note_user_hint(verdict, name)
        return text + format_hint(verdict, name)

    @staticmethod
    def _spin_hint(name: str, meta: dict, allowed: set | None) -> str:
        """同一个工具被反复调用时的停滞提示。

        ★ 为什么需要它（实测缺口）
        ------------------------
        实测一个问题触发 **6 次 `search_knowledge_base`**，而每次返回的来源
        **主题相关、相关性不低**（0.5+），故质量判定是 `ok`、不产生提示——
        模型就在同一个坑里反复检索。

        问题不在"结果质量差"，而在"**结果切题但不回答问题**"：
        库里讲的是该领域的一般规定，用户问的是一个具体颗粒度的问题。
        这种落差**测不出来**（分数确实不低），能测出来的是**行为模式**：
        同一个工具被反复调用而结论没推进。

        阈值取 3：第 1 次是正常检索，第 2 次可能是换问法补充，
        **第 3 次还调同一个工具**基本可以确定是在打转。
        """
        counts = meta.setdefault("_tool_call_counts", {})
        counts[name] = counts.get(name, 0) + 1
        if counts[name] < SPIN_REPEAT_THRESHOLD:
            return ""
        candidates = ALTERNATIVES.get(name, ())
        if allowed is not None:
            candidates = tuple(c for c in candidates if c in allowed)
        suggested = tuple(c for c in candidates if c != name)
        logger.info(
            "检测到检索停滞：%s 已调用 %d 次，建议改用 %s",
            name, counts[name], suggested or "（无可用替代）",
        )
        return format_spin_hint(counts[name], name, suggested)

    @staticmethod
    def _merge_sources(existing: list, new: list, cap: int | None = None) -> list:
        """跨轮来源合并去重：按 `(question, answer, url)` 组合键判断重复。

        `cap` 非空时限制返回条数，**保留分数最高的那批**。

        ★ 为什么要限条数（实测）
        ----------------------
        多轮工具调用会累加来源：实测一次提问拿到 **24~36 条联网来源**
        （3 轮检索 × 每轮 8~12 条）。前端把 30 多个卡片全渲染出来，
        用户根本读不完，真正相关的前几条反而被淹没。

        限条数时按分数**降序保留**而不是截断前 N 条——后者会把后面轮次
        检索到的、可能更相关的结果直接丢掉。无分数的排在最后。
        """
        seen = {
            (s.get("question"), s.get("answer"), s.get("url"))
            for s in existing
            if isinstance(s, dict)
        }
        merged = list(existing)
        for source in new or []:
            if not isinstance(source, dict):
                continue
            key = (source.get("question"), source.get("answer"), source.get("url"))
            if key not in seen:
                seen.add(key)
                merged.append(source)

        if cap is None or len(merged) <= cap:
            return merged

        def score_of(s: dict) -> float:
            v = s.get("score")
            return float(v) if isinstance(v, (int, float)) else -1.0

        merged.sort(key=score_of, reverse=True)
        return merged[:cap]

    def _record_tool_meta(self, name: str, sources: list, meta: dict):
        """工具执行成功后回填 done 帧元数据。tool_name 恒为最后一个执行完成的工具（单字符串）。"""
        meta["tool_called"] = True
        meta["tool_name"] = name
        # 记录**成功执行过的工具集合**（不只是最后一个）。
        #
        # 用途一：生成前注入"本轮工具执行事实"——实测模型会编造
        #   「联网检索工具本轮不可用」而它其实成功返回了 20 条结果。
        #   把事实摆到它面前是最低成本的预防（见 verification.build_grounding_note）。
        # 用途二：done 帧回填核对结果（verification.verify_answer）。
        #
        # 为什么记集合而不是沿用 tool_name：tool_name 是"最后一个成功的"，
        # 多轮里前面成功过的工具会被覆盖掉，核对时就漏判。
        succeeded = meta.setdefault("succeeded_tools", [])
        if name not in succeeded:
            succeeded.append(name)

        # 记录**每一次**调用（含重复）的完整序列，供评测算"打转"指标。
        #
        # 为什么需要它：`tool_name` 单字符串 + `succeeded_tools` 去重集合
        # **都算不出重复调用次数**。而实测打转正是重复调用：
        # 多跳题里 `query_database` 被调 4 次（参数略不同）、2/5 撞轮次上限。
        # 缺了这个序列，评测就只能看到"工具选对了"（准确率 100%），
        # 完全看不到用户多等了 20 秒。
        meta.setdefault("tool_call_names", []).append(name)

        if name in WEB_TOOL_NAMES:
            meta["web_sources"] = self._merge_sources(
                meta["web_sources"], sources, cap=MAX_WEB_SOURCES
            )
        else:
            meta["sources"] = self._merge_sources(
                meta["sources"], sources, cap=MAX_LOCAL_SOURCES
            )

    # --- 最终生成准备 ---

    @staticmethod
    def _clean_for_final(messages: list) -> list:
        """构造最终生成消息：system 必须在最前（messages[0]），末尾追加约束消息。

        约束消息的位置是测试断言点：最后一条必须是 user 角色的约束指令。
        """
        cleaned = [messages[0]] + [
            m for m in messages[1:] if m.get("role") != "system"
        ]
        cleaned.append({"role": "user", "content": _FINAL_CONSTRAINT})
        return cleaned

    def _fallback_execute(self, leaked_text: str, meta: dict, question: str | None = None):
        """文本兜底执行：从泄漏文本解析工具调用并执行（流式轮检出的调用走这里）。

        只执行第一个可解析调用，避免模型一次泄漏多个调用时放大后端压力。
        返回 (工具结果文本, calls)。

        `question` 用于把用户原问题传给工具——联网工具据此重排结果并补算相关度。
        """
        if question is None:
            # 检索提示：多轮时带上上一轮用户问题（_execute_tool 把它作为第二个参数
            # 交给执行器）。用途见 core._retrieval_hint 的说明——
            # 模型自己改写的 query 不保证稳定，带上文可少走弯路。
            question = str(meta.get("retrieval_hint") or meta.get("question") or "")
        calls = self._parse_tool_text(leaked_text)
        if not calls:
            return "", []
        name = calls[0]["name"]
        allowed = self._allowed_tool_names(meta)
        text, sources, ok = self._execute_tool(
            name, calls[0]["arguments"], allowed, question
        )
        text = self._validate_result(name, text, ok, sources, allowed)
        text += self._spin_hint(name, meta, allowed)
        if ok:
            self._record_tool_meta(name, sources, meta)
        # 兜底路径也要累积，否则泄漏兜底后的回答没有可核对的工具文本
        meta.setdefault("tool_output_text", []).append(text)
        return f"[工具 {name} 结果]\n{text}", calls

    def _resolve_tool_schemas(self, web_search_enabled: bool = False,
                              clarify_allowed: bool = False) -> list:
        """产出本轮可暴露给大模型的工具定义（OpenAI function-calling 格式）。

        为什么需要：项目原先只靠 SYSTEM_PROMPT 的文字描述告诉模型"有哪些工具"，
        但决策轮的 API 调用**从未传 tools 参数**——模型看不到工具 schema 就无法
        返回 tool_calls，于是永远走"凭自身知识直接回答"，知识库检索与来源引用
        全部落空（前端来源卡片恒为空）。

        联网搜索工具（search_web / search_exa）只在用户显式打开开关时暴露：
        既是省 token，也避免模型在用户没要求联网时擅自发起外部请求。

        ★ `clarify_allowed` 控制是否暴露追问工具
        ------------------------------------
        只在**首轮且无历史**时为真（判定在 `core._chat_events` 组装 meta 时做）。
        不暴露时模型连这个工具的定义都看不到，**结构上**不可能调用它——
        这比"暴露但在拦截层忽略"更稳：后者依赖模型不去调，而模型的行为不可控。
        """
        names = list(BASE_TOOL_NAMES) + (list(WEB_TOOL_NAMES) if web_search_enabled else [])
        if not clarify_allowed and CLARIFY_TOOL_NAME in names:
            # 追问工具只服务于首轮澄清（见 core._chat_events 的 clarify_allowed）
            names = [n for n in names if n != CLARIFY_TOOL_NAME]
        try:
            from src.tools.rag_tools import get_tool_schemas
        except ImportError:  # pragma: no cover - 工具层未落地时按无工具运行
            return []
        try:
            return get_tool_schemas(names)
        except Exception as exc:  # pragma: no cover - schema 生成异常不应阻断回答
            logger.warning("工具 schema 生成失败，本轮按无工具运行: %s", exc)
            return []

    # --- 追问（首轮澄清） ---

    @staticmethod
    def _normalize_options(raw) -> list[str]:
        """规整候选项：去空白、去重、限长、限数量。

        ★ 为什么要去重：模型常给出"是/否/是"这类重复项（实测），
        渲染成按钮后用户看到两个一样的选项会以为是 bug。
        """
        if not isinstance(raw, list):
            return []
        out: list[str] = []
        for item in raw:
            text = str(item).strip() if isinstance(item, (str, int, float)) else ""
            if not text or len(text) > MAX_CLARIFY_OPTION_CHARS:
                continue
            if text in out:
                continue
            out.append(text)
            if len(out) >= MAX_CLARIFY_OPTIONS:
                break
        return out

    @classmethod
    def _extract_clarification(cls, calls: list) -> dict | None:
        """从本轮的调用里取出追问请求；没有则返回 None。

        ★ 为什么在这里解析而不是在执行器里
        -----------------------------
        `_execute_tool` 的契约是 `(文本, sources, ok)` 三元组，**没有回传结构化
        数据的通道**。若把问题与选项序列化进文本再由上层解析回来，
        就多了一对"格式化→解析"的代码，两处一旦漂移就会出现
        "追问内容显示错乱"这类难查的问题。
        直接在调用点读 `arguments` 最直接，且执行器仍保留（见
        `rag_tools.ask_clarification`）用于兜底与注册表一致性。

        ★ 校验失败的策略：返回 None（= 不追问，继续正常回答）
        ------------------------------------------------
        宁可不问也不能问出畸形问题（空问题、纯符号）。
        追问出现在用户面前的门槛应当比"检索"高：一次糟糕的追问
        比不追问更伤害体验，而"不追问"退化到的正是改动前的行为，
        是安全的下限。
        """
        for call in calls or []:
            if not isinstance(call, dict) or call.get("name") != CLARIFY_TOOL_NAME:
                continue
            args = call.get("arguments")
            if not isinstance(args, dict):
                logger.warning("追问调用缺少 arguments，按不追问处理")
                return None
            raw = args.get("questions")
            if not isinstance(raw, list):
                logger.warning("追问调用缺少 questions 数组，按不追问处理")
                return None

            questions: list[dict] = []
            for item in raw:
                if isinstance(item, str):
                    text, options = item.strip(), []
                elif isinstance(item, dict):
                    text = str(item.get("question") or "").strip()
                    options = cls._normalize_options(item.get("options"))
                else:
                    continue
                # 过滤过短/过长的"问题"：过短无信息量，过长多半是模型把
                # 整段解释塞了进来（实测出现过一次 200+ 字的"问题"）
                if not text or len(text) < MIN_CLARIFY_QUESTION_CHARS:
                    continue
                if len(text) > MAX_CLARIFY_QUESTION_CHARS:
                    text = text[:MAX_CLARIFY_QUESTION_CHARS].rstrip() + "…"
                questions.append({"question": text, "options": options})
                if len(questions) >= MAX_CLARIFY_QUESTIONS:
                    break

            if not questions:
                logger.warning("追问调用未产出可用问题，按不追问处理")
                return None

            reason = str(args.get("reason") or "").strip()
            if len(reason) > MAX_CLARIFY_REASON_CHARS:
                reason = reason[:MAX_CLARIFY_REASON_CHARS].rstrip() + "…"
            return {
                "questions": questions,
                # 前端据此显示"为什么要问"，让追问不显得突兀
                "reason": reason,
            }
        return None

    # --- 主循环 ---

    def _run_planner(self, question: str, client, meta: dict, tool_schemas: list) -> str:
        """跑一次规划轮，把结果写进 meta，返回要注入的指引文本（失败返回空串）。

        ★ 规划是**增益项**：任何失败都不能影响问答。
        解析失败、模型返回垃圾、超时——一律记日志并按"无计划"继续，
        让 ReAct 照常跑。规划本身不该成为新的失败点。
        """
        from src.agent.planning import PLANNER_PROMPT, build_plan_note, parse_plan

        allowed = {t["function"]["name"] for t in tool_schemas} if tool_schemas else None
        try:
            raw = client.chat_raw([
                {"role": "user", "content": PLANNER_PROMPT + question},
            ])
        except Exception:
            logger.warning("规划轮调用失败，按无计划继续", exc_info=True)
            meta["plan"] = {"ok": False, "reason": "规划轮调用失败", "steps": []}
            return ""

        plan = parse_plan(raw, allowed_tools=allowed)
        meta["plan"] = plan.to_dict()
        if not plan.ok:
            logger.info("规划未产出可用计划（%s），按无计划继续", plan.reason)
            return ""

        logger.info(
            "规划产出 %d 条信息需求：%s", len(plan.steps), plan.tools()
        )
        if plan.reason:
            logger.info("规划提示：%s", plan.reason)
        return build_plan_note(plan)

    @staticmethod
    def _append_plan_note(conversation: list, note: str) -> list:
        """把计划指引追加到 system 消息末尾，返回新的 messages。

        追加到首条 system 而不是另起一条：部分厂商只认首条 system。
        与 `generation._inject_grounding` 同一做法。
        """
        messages = list(conversation)
        if messages and messages[0].get("role") == "system":
            messages[0] = {
                "role": "system",
                "content": str(messages[0].get("content") or "") + "\n\n" + note,
            }
        else:
            messages.insert(0, {"role": "system", "content": note})
        return messages

    def _chat_stream_tools(self, messages: list, client, meta: dict, deep_thinking: bool = False):
        """ReAct 多轮循环主体。产出 status 事件，最终转入 _generate_final 产出 token 事件。

        每轮：chat_raw 决策 → 解析调用 → 无调用则进入最终生成 / 有调用则执行并回填结果。
        决策轮检出畸形工具文本时做一次约束重试；仍无法解析则视为无调用。
        """
        conversation = list(messages)
        # 检索提示：多轮时带上上一轮用户问题，经 `_execute_tool` 作为第二个参数
        # 交给执行器。用途见 `core._retrieval_hint` 的说明——模型自己改写的
        # query 不保证稳定（实测有跑偏到上一话题、三轮才修正的情况），
        # 带上文可少走弯路。没有历史时它就是当轮原问题。
        question = str(meta.get("retrieval_hint") or meta.get("question") or "")
        # 按 meta 里记录的开关决定暴露哪些工具（core._chat_events 已把开关放进 meta）
        tool_schemas = self._resolve_tool_schemas(
            bool(meta.get("web_search_enabled")),
            # 仅在首轮且无历史时把追问工具暴露给模型（判定在
            # `core._chat_events` 组装 meta 时完成，见 `clarify_allowed`）
            clarify_allowed=bool(meta.get("clarify_allowed")),
        )
        if tool_schemas:
            logger.info(
                "本轮暴露工具 %d 个: %s",
                len(tool_schemas),
                [t["function"]["name"] for t in tool_schemas],
            )

        if meta.get("planning_enabled"):
            yield "status", "正在规划检索步骤..."
            note = self._run_planner(question, client, meta, tool_schemas)
            if note:
                conversation = self._append_plan_note(conversation, note)

        for round_no in range(1, MAX_TOOL_ROUNDS + 1):
            # 状态序列对齐开发文档 §6.3：
            # 检索与搜索 → 分析检索结果 → 补充检索（第N轮）→ ... → 生成回答
            if round_no == 1:
                yield "status", "正在检索与搜索..."
            else:
                yield "status", "正在分析检索结果..."

            t0 = time.monotonic()
            decision = client.chat_raw(conversation, tools=tool_schemas)
            # ★ 立刻取走本轮思维链（该属性每轮被覆盖，晚取就丢了）
            round_reasoning = self._take_reasoning(client)
            self._mark_phase(
                meta, "首轮分析" if round_no == 1 else "分析检索结果", t0
            )

            calls = self._parse_tool_text(decision)
            if not calls and self._detect_tool_text(decision):
                # 疑似工具调用但解析失败（畸形 JSON）：约束重试一次
                logger.warning("决策轮输出疑似工具格式但无法解析，约束重试")
                conversation.append(
                    self._assistant_msg(decision, round_reasoning)
                )
                conversation.append({"role": "user", "content": _TOOL_CONSTRAINT})
                decision = client.chat_raw(conversation, tools=tool_schemas)
                round_reasoning = self._take_reasoning(client)
                if self._detect_tool_text(decision):
                    # 约束重试仍输出工具格式：放弃本轮调用，按无调用处理
                    logger.warning("约束重试后仍输出工具格式文本，按无调用处理")
                    decision = ""
                calls = self._parse_tool_text(decision) if decision else []
            if not calls:
                conversation.append(
                    self._assistant_msg(decision or "", round_reasoning)
                )
                break

            # ★ 追问拦截：必须在**执行工具之前**
            #
            # 追问不是"检索动作"，而是"停下来问用户"。若按常规路径执行：
            #   `_execute_tool` 返回一段文本 → 回填 conversation →
            #   模型拿着"【需要用户澄清】…"继续决策 → 最终仍然生成一段回答。
            # 结果就是用户既看到一段勉强作答、又看到几个问题，而答案是基于
            # **缺失信息**硬编出来的——正是本功能要避免的"直接检索"。
            #
            # 故一旦模型调用追问工具，立即产出 `clarify` 事件并**结束本轮生成**，
            # 不检索、不生成。用户回答后由前端作为新一轮请求发回，那时
            # `clarify_allowed` 为假（已有历史），模型只能正常检索作答。
            clarify = self._extract_clarification(calls)
            if clarify is not None:
                logger.info(
                    "第 %d 轮判定信息不足，向用户追问 %d 条",
                    round_no, len(clarify.get("questions") or []),
                )
                yield "clarify", clarify
                return

            if round_no > 1:
                yield "status", f"正在补充检索（第{round_no - 1}轮）..."
            t1 = time.monotonic()
            allowed = self._allowed_tool_names(meta)
            # ★ 一轮内的多个工具**并行**执行，且同参数重复调用只跑一次
            #   （见 `_execute_tool_batch`：实测串行时阶段耗时 = 各工具之和）
            batch = self._execute_tool_batch(calls, allowed, question)
            results = []
            for call, (text, sources, ok, obs, was_dup) in zip(calls, batch):
                name = call["name"]
                # ★ 每个工具的耗时现在由执行侧在**它自己的线程里**测量并随结果返回。
                #   不能像原来那样在父线程"包住调用"计时：并行时那样量到的是
                #   "这一批跑完用多久"，每个工具都会记成同一个数。
                _timing = {
                    "round": round_no,
                    "name": name,
                    "ms": int(obs.get("ms") or 0),
                    "ok": bool(ok),
                    "sources": len(sources or []),
                    "chars": len(text or ""),
                }
                if was_dup:
                    _timing["deduped"] = True
                # 把 RAG 检索耗时附在该工具的条目上——用户看到"检索知识库 1.2s"
                # 时能直接在同一处展开看这 1.2s 花在哪一步，不必去别处找
                _rag = obs.get("rag")
                if _rag is not None:
                    _timing["rag"] = _rag
                    meta.setdefault("rag_timings", []).append(_rag)
                meta.setdefault("tool_timings", []).append(_timing)
                # 顺手上报一条 Langfuse tool observation（未配置时 no-op）。
                # 工具是并行执行的，故 `push_tool` 必须线程安全——见
                # `LangfuseTagger` 的锁。这里仍在**父线程按原顺序**上报，
                # 顺序与轮次因此保持确定，不随并发完成次序抖动。
                _rec2 = meta.get("llm_trace")
                if _rec2 is not None and hasattr(_rec2, "push_tool"):
                    _rec2.push_tool(_timing)
                text = self._validate_result(name, text, ok, sources, allowed)
                # 打转计数按**模型请求的次数**算（含被去重的那次）：模型确实又调了一次
                # 同一个工具，这正是"停滞"信号；若因为我们去重就不计数，
                # 评测里的"重复调用"指标会假性变好，掩盖真实行为。
                text += self._spin_hint(name, meta, allowed)
                if ok:
                    self._record_tool_meta(name, sources, meta)
                    if was_dup:
                        # 去重也要记进调用序列（理由同上：指标要反映模型的真实行为）
                        meta.setdefault("tool_call_names", []).append(name)
                        meta["tool_deduped"] = int(meta.get("tool_deduped") or 0) + 1
                        # 明确告诉模型"这一轮你已经问过同样的问题了"，
                        # 比让它把同样的结果再读一遍更有用（减少下一轮再重复）
                        text += "\n（同一轮内重复调用且参数相同，已复用上一次结果）"
                    # ★ 从工具结果里提取可图表数据（前端据此画图）
                    #
                    # 放在 `_validate_result` **之后**：图表必须画"模型真正看到的
                    # 那份数据"，与引用核对同一原则（`tool_output_text` 也是取
                    # 加工后的文本）。若取原始结果，可能出现"图里有、回答里没有"
                    # 的不一致。
                    #
                    # 只在 `ok` 时提取：失败结果是错误文案，不含数据。
                    # ★ 图表数据：**优先用工具透传的结构化数据**，
                    # 没有再退回解析工具文本（见 `chart_data.py` 的说明）。
                    #
                    # 透传的好处：数字取自格式化**之前**的原始行，
                    # 不受 `_fmt_amount` / `_truncate` 影响，也不怕展示格式改版。
                    # 兜底保留的原因：联网类工具（`search_web*`）的结果本就是
                    # 自然语言，没有结构化数据可交；其余工具尚未逐个改造。
                    _charts = obs.get("charts") or []
                    if not _charts:
                        from src.agent.charts import extract_charts

                        _charts = extract_charts(text, name)
                    # `setdefault` 必须在这里（不能只在下面 append）：
                    # 上面那行 `meta["charts"]` 在键不存在时会 KeyError。
                    _bucket = meta.setdefault("charts", [])
                    for chart in _charts:
                        if len(_bucket) >= MAX_CHARTS:
                            break
                        _bucket.append(chart)
                results.append(f"[工具 {name} 结果]\n{text}")
                # 累积**模型实际看到过的**工具结果文本，供生成后核对引用。
                # 必须是"模型看到的那份"（即 _validate_result 加工后的 text），
                # 而不是原始检索结果——否则核对会以为模型看到了它其实没看到的内容。
                seen = meta.setdefault("tool_output_text", [])
                seen.append(text)
            self._mark_phase(
                meta,
                "检索与搜索" if round_no == 1 else f"补充检索（第{round_no - 1}轮）",
                t1,
            )

            conversation.append(self._assistant_msg(decision, round_reasoning))
            conversation.append({
                "role": "user",
                "content": (
                    "工具结果：\n" + "\n\n".join(results)
                    + "\n请继续：若信息足够请直接回答，若仍需查询请再次调用工具。"
                ),
            })
        else:
            # 达到轮次上限：不再决策，直接带着已有工具结果进入最终生成
            logger.info("ReAct 循环达到 %d 轮上限，进入最终生成", MAX_TOOL_ROUNDS)

        final_messages = self._clean_for_final(conversation)
        meta["last_stream_messages"] = final_messages
        yield from self._generate_final(final_messages, client, meta, deep_thinking)
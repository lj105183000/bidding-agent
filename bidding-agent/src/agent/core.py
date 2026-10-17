"""BiddingAgent 主体：生命周期、_chat_events 编排入口、chat / chat_stream。

两条输出路径共用同一事件流（开发文档 §8.1）：
  _chat_events  产出六类事件元组（status/token/thinking/reset/done/error）
  chat_stream   序列化为 SSE 帧（api/server.py 的 /api/chat/stream 使用）
  chat          收集为 dict（非流式接口与 eval/agent_eval.py 使用）
"""

import logging
import os
import time

from src.langfuse_tracing import build_recorder
from src.monitoring_trace import LLMTraceRecorder, attach_recorder, detach_recorder
from src.rag.timing_summary import summarize as rag_timing_summary
from src.agent.constants import (
    CLARIFY_MIN_QUESTION_CHARS,
    THINKING_MODEL_ENV,
    llm_client_lock,
    resolve_thinking_model,
    thinking_enabled_for,
)
from src.agent.generation import GenerationMixin
from src.agent.react_loop import ReActMixin
from src.agent.compaction import compaction_enabled
from src.agent.skills import _load_skills, _match_skills
from src.agent.tool_defense import ToolDefenseMixin
from src.agent.utils import (
    _pace_stream_chunks,
    _sse,
    _truncate_history,
    SSE_DONE_MARK,
    SSE_PADDING,
)
from src.agent.prompts import SYSTEM_PROMPT

try:
    # 工具层（成员 C）尚未合入时以空集运行；此绑定用于阶段二接 ToolRunner（runner 构造）
    from src.tools.rag_tools import TOOL_EXECUTORS
except ImportError:  # pragma: no cover - 阶段一工具层未落地
    TOOL_EXECUTORS = {}

try:
    # LLM 客户端工厂（成员 D）尚未合入时不可用，测试经注入 fake client 自测
    from src.clients.llm_factory import get_llm_client
    from src.clients.fallback import build_fallback_client
except ImportError:  # pragma: no cover - 阶段一客户端未落地
    get_llm_client = None
    build_fallback_client = None

logger = logging.getLogger(__name__)

DEFAULT_PROVIDER = "deepseek"

def _take_user_hints_for_done() -> list[str]:
    """取「给用户看的质量说明」。**永不抛异常**。
    单独包一层有两个理由：
      ① 这是观测数据，任何问题都不该让 done 帧产不出来；
      ② 便于测试 monkeypatch（不必 import react_loop 的真实实现）。
    """
    try:
        from src.agent.react_loop import take_user_hints
        return take_user_hints()
    except Exception:  # noqa: BLE001 - 观测不能影响问答
        return []




class AgentError(Exception):
    """Agent 内部错误基类。完整异常只进日志，用户可见文案由 _chat_events 统一给出。"""


class LLMUnavailableError(AgentError):
    """LLM 客户端不可用（工厂未就绪或 provider 缺少配置）。"""


def _deep_thinking_configured(provider: str = "") -> bool:
    """该 provider 是否配置了"深度思考的模型侧能力"。

    ★ 两条路取"或"（2026-04 的 API 变更让第二条成为主路）
    ------------------------------------------------
        ① 显式指定推理模型名（`*_THINKING_MODEL`）
           —— 用于确有独立推理模型的部署，如本地 vLLM 上单独部署的 R1
        ② 开思考模式参数（`*_THINKING_ENABLED`）
           —— **DeepSeek 官方方式**：现在只有 `deepseek-flash` /
              `deepseek-v4-pro`，两者都由请求参数控制思考模式，
              已无独立推理模型名

    两条都为假时返回 False，调用方据此回退到提示词注入
    （往 system 提示注入三步分析指令）。

    ★ 为什么抽成模块级函数而不是类方法
    -----------------------------
    `generation` 层也要用同一口径判断"本轮有没有真思维链"。放在模块级
    便于两边 import 同一个实现，避免各写一遍再写歪（这正是原实现的问题之一）。
    """
    return bool(resolve_thinking_model(provider)) or thinking_enabled_for(provider)


class BiddingAgent(ReActMixin, GenerationMixin, ToolDefenseMixin):
    """编排核心。llm_client / rag_pipeline 可注入（测试与阶段二联调前自测）。"""

    def __init__(self, llm_client=None, rag_pipeline=None):
        self._llm_client = llm_client
        self._rag_pipeline = rag_pipeline
        self._provider_clients = {}  # provider -> client 缓存，受 llm_client_lock 保护
        self.skills = _load_skills()
        # 健康检查的 agent_ready 依据：初始化走到这里即视为就绪
        self.ready = True
        logger.info("BiddingAgent 初始化完成，加载技能 %d 个", len(self.skills))

    # --- 组件获取（懒加载 + 缓存） ---

    def _get_llm_client(self, provider: str = ""):
        """按 provider 取 LLM 客户端，**带多 provider 自动切换**。

        注入的 `llm_client` 优先（测试/特殊部署），此时不做切换——
        注入方要的是确定性替身，包一层兜底会让"测试里明明该失败"变成静默成功。

        否则返回 `FallbackLLMClient`：候选链是
        `[显式指定的 provider] + LLM_FALLBACK_PROVIDERS`（默认 deepseek→zhipu）。

        ★ 为什么按**链**缓存而不是按 provider 缓存
        ---------------------------------------
        链是配置的函数（`provider + 环境变量`），同一链复用同一个客户端对象，
        才能把 `last_used_provider` / `switch_count` 这类状态保留下来供展示。
        按单个 provider 缓存会把"用了哪家"的信息丢掉。

        ★ 为什么显式指定也要包兜底
        ------------------------
        调用方传 `provider="zhipu"` 而智谱没配密钥时，**不该整个问答失败**——
        链会退到 deepseek。链把"指定"排在第一位，故意图仍被尊重；
        真正生效的是谁由 `last_used_provider` 如实报告（不静默）。
        """
        if self._llm_client is not None:
            return self._llm_client
        # ★ 没显式指定时回落 `LLM_PROVIDER`（原实现有这一步，重写时漏掉了）
        #
        # 漏掉的后果实测过：把 `LLM_FALLBACK_PROVIDERS` 设成空（关闭自动切换）
        # 又不传 provider 时，候选链变成**空**，整个问答直接失败并报
        # "没有可用的 LLM provider"——而 `LLM_PROVIDER` 明明配好了。
        # 也就是说那次重写让 `LLM_PROVIDER` 这个配置项**完全失效**，
        # 且失败形态是"静默没有 LLM"而不是"用配置里的那家"。
        resolved = (provider or "").strip() or os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER)
        key = resolved.strip().lower()
        with llm_client_lock:
            client = self._provider_clients.get(key)
            if client is None:
                if get_llm_client is None or build_fallback_client is None:
                    raise LLMUnavailableError("LLM 客户端工厂未就绪")
                client = self._build_llm_chain(resolved)
                self._provider_clients[key] = client
            return client

    def _build_llm_chain(self, provider: str = ""):
        """按配置拼出带自动切换的客户端链。

        ★ 这里**不**应用思考模型（原来应用了，是个真实缺陷）
        ---------------------------------------------
        原实现在 `resolve()` 里无条件调用 `_thinking_model`：

            def resolve(name):
                client = get_llm_client(name)
                return self._thinking_model(client, name)   # ← 建链时就切了

        而链是**按 provider 缓存**的（`_provider_clients`）。后果是：

            只要配了 `DEEPSEEK_THINKING_MODEL`，链在**首次构建时**就被永久
            换成推理模型 —— 此后**不开"深度思考"开关也走推理模型**。
            用户会看到"平时回答也变慢了、变贵了"，而开关看起来毫无作用。

        正确做法：链只按普通模型构建；思考模型在**请求级**按需切换
        （`_chat_events` 里 `deep_thinking_enabled` 为真时才切，
          且切的是浅拷贝，见 `_thinking_model`）。

        ★ resolver 必须**缓存实例**（实测缺口的修复）
        --------------------------------------
        `FallbackLLMClient` 会多次对同一 provider 调 resolver：

            · `_available_pair()` 每次调用都解析一遍
            · `trace_recorder` 的 setter 会遍历链上所有 provider
            · 真实失败切换时还要再解析

        而 `get_llm_client()` **每次都新建实例**（实测 `a is b` → False）。
        不缓存的后果是个隐蔽的静默失效：

            记录器挂在第 N 次解析出的实例上 → 该实例随即被丢弃
            → 真正发请求的第 N+1 个实例上没有记录器
            → **`done` 帧里 `llm_trace.call_count` 恒为 0**

        看起来像"这轮没调用 LLM"，而日志里明明有调用记录。
        故这里按 provider 缓存，保证"同一个 provider 拿到同一个客户端"。
        """
        cache: dict[str, object] = {}

        def resolve(name: str):
            key = (name or "").strip().lower()
            client = cache.get(key)
            if client is None:
                client = get_llm_client(name)
                cache[key] = client
            return client

        return build_fallback_client(provider, resolve)

    def _get_rag_pipeline(self):
        """取 RAG 流水线单例（成员 B）。未合入时返回 None，RAG 降级路径据此报错。"""
        if self._rag_pipeline is not None:
            return self._rag_pipeline
        if getattr(self, "_rag_probed", False):  # 已探测过且不可用，避免每次请求重复 import
            return None
        self._rag_probed = True
        try:
            from src.rag.pipeline import rag_pipeline
        except ImportError:
            logger.info("RAG 流水线尚未合入，降级路径不可用")
            return None
        self._rag_pipeline = rag_pipeline
        return self._rag_pipeline

    def _thinking_model(self, client, provider: str = ""):
        """深度思考：在**浅拷贝**的客户端上开启思考模式，返回该副本。

        ★ 为什么从"换模型名"改成"开开关"（2026-04-24 的 API 变更）
        ---------------------------------------------------
        原设计是切到一个独立的推理模型名（`DEEPSEEK_THINKING_MODEL`，
        如 `deepseek-reasoner`）。但 DeepSeek 发布 V4 后**取消了独立推理模型名**：

            现在只有 deepseek-flash / deepseek-v4-pro，两个模型都由
            **请求参数** `{"thinking": {"type": "enabled"}}` 控制思考模式。
            旧名 deepseek-reasoner 将于 2026-07-24 15:59 UTC 彻底退役
            （官方「模型 & 价格」与「思考模式」文档）。

        故判定依据从**模型名**变成**开关状态**。迁移公告特别指出最常见的
        bug 就是"忘记加 thinking 参数"——结果是**推理质量下降但没有任何报错**。

        ★ 三步
        ----
            ① provider 没配"深度思考开关"或开关为关 → 原样返回
            ② 客户端支持 `set_thinking` → 浅拷贝后开启，返回副本
            ③ 客户端不支持（第三方注入实现）→ 原样返回，由生成层回退提示词注入

        ★ 为什么仍然浅拷贝
        --------------
        客户端实例在 `_provider_clients` 里**按链缓存、跨请求共享**。原地开
        思考模式会让"这一轮开了深度思考"污染"另一轮并发的普通请求"：
        后者会莫名其妙变慢变贵。浅拷贝后副本进本次请求，原件留在缓存里。

        浅拷贝足够：客户端状态只有配置字段；而 `_lock` / `_cooldown` 这类
        可变状态**故意共享**（见 `FallbackLLMClient`），这样冷却与切换计数
        跨副本一致。
        """
        enabled = _deep_thinking_configured(provider)
        if not enabled:
            return client

        setter = getattr(client, "set_thinking", None)
        model_name = resolve_thinking_model(provider)
        if not callable(setter) and not model_name:
            logger.info(
                "provider=%s 的客户端不支持思考模式开关，深度思考走提示词注入", provider
            )
            return client

        import copy as _copy

        try:
            switched = _copy.copy(client)
            # ★ 两条路都要走，它们是**互补**的而不是二选一：
            #     ① 换模型名 —— 确有独立推理模型的部署（如本地 vLLM 上的 R1）
            #     ② 开思考参数 —— DeepSeek V4 之后的官方方式
            #   只做 ② 会让 ① 静默失效：用户配了 *_THINKING_MODEL 却没有效果，
            #   且没有任何报错（实测由 test_switch_thinking_model_still_works 抓到）。
            if model_name:
                switcher = getattr(switched, "switch_thinking_model", None)
                if callable(switcher):
                    switcher(model_name)
            if callable(setter):
                switched.set_thinking(True)
        except Exception:  # noqa: BLE001
            # 开启失败不该让整轮问答失败：退化为提示词注入即可
            logger.warning("开启思考模式失败，回退提示词注入", exc_info=True)
            return client

        logger.info(
            "深度思考已开启（provider=%s%s）",
            provider or "默认",
            f"，模型={model_name}" if model_name else "，思考参数=on",
        )
        return switched

    # --- 上下文构建 ---

    def _build_context(self, question: str, history: list = None,
                       web_search_enabled: bool = False,
                       fallback_note: str = "",
                       memory_enabled: bool = False,
                       meta: dict | None = None,
                       session_id: str = "",
                       client=None) -> list:
        """构建首轮消息：system（角色 + 技能注入）+ 截断后的历史 + 当前问题。

        返回列表首条必须是 system 消息（_clean_for_final 依赖 messages[0]）。

        `fallback_note` 是自动兜底时插入的说明。为什么要显式告诉模型
        「本地知识库没有答案」：否则模型看到检索结果为空或相关度低，
        倾向自己"编一个合理答案"而不是去联网。明确告知缺什么，
        它才会去调用搜索工具。
        """
        system = SYSTEM_PROMPT
        matched = _match_skills(question, self.skills)
        if matched:
            blocks = "\n\n".join(f"### 技能：{s['name']}\n{s['body']}" for s in matched)
            system += "\n\n## 已启用技能（按其步骤执行）\n" + blocks

        # 历史压缩：把被 `_truncate_history` 丢掉的那部分内容压成摘要注入。
        #
        # 为什么在 system 而不是当历史消息：它是**更早对话的浓缩**，
        # 放到 system 能明确区分"摘要（可能失真）"与"原始消息（可信）"，
        # 也避免打乱 user/assistant 交替的角色序列（部分模型对此不稳定）。
        #
        # ⚠️ 必须**调用** compaction_enabled()。
        # 曾误写成 `if compaction_enabled:`（函数对象恒为真），
        # 后果是 `AGENT_COMPACT=0` **完全失效**、永远压缩。
        # 这类错不会报错、也不会改变行为（恰好总是压缩），
        # 只让开关变成假的——所以开关必须有测试覆盖。
        if compaction_enabled():
            note = self._compact_history(history, session_id, client, meta)
            if note:
                system += "\n\n" + note

        # 会话记忆：把历史**被采纳**的相似问答作为参考注入。
        # 为什么在 system 而不是当工具结果：它是背景经验，不是本轮检索所得；
        # 放 system 能让模型区分"权威语料"（工具结果）与"历史经验"（此处）。
        if memory_enabled:
            note = self._recall_memory(question, meta)
            if note:
                system += "\n\n" + note

        if fallback_note:
            system += "\n\n" + fallback_note
        elif web_search_enabled:
            system += "\n\n已开启联网搜索：search_web / search_exa 工具可用，时效性问题优先联网。"
        messages = [{"role": "system", "content": system}]
        messages.extend(_truncate_history(history))
        messages.append({"role": "user", "content": question})
        return messages

    @staticmethod
    def _compact_history(history: list, session_id: str, client, meta: dict | None) -> str:
        """压缩更早的历史并返回注入文本。**永不抛异常**。

        压缩是增益项：它失败最多退化成现有的硬截断（丢掉更早内容），
        不该让整轮问答失败——那比"记不住前文"严重得多。
        """
        if not history:
            return ""
        try:
            from src.agent.compaction import build_compaction_note, compact_history

            result = compact_history(history, client, session_id=session_id)
            if meta is not None:
                meta["compaction"] = result.to_dict()
            return build_compaction_note(result)
        except Exception:
            logger.warning("历史压缩失败（不影响问答）", exc_info=True)
            return ""

    @staticmethod
    def _recall_memory(question: str, meta: dict | None = None) -> str:
        """取会话记忆并渲染成注入文本。**永不抛异常**，失败即返回空串。

        记忆是增益项：数据库不可用、编码失败都只该降级为"无记忆"，
        不能影响问答本身。
        """
        try:
            from src.agent.memory import build_memory_note, session_memory

            examples = session_memory.recall(question)
            if meta is not None:
                meta["memory_examples"] = [e.to_dict() for e in examples]
            return build_memory_note(examples)
        except Exception:
            logger.warning("会话记忆召回失败（不影响问答）", exc_info=True)
            return ""

    # --- 编排入口 ---

    @staticmethod
    def _retrieval_hint(question: str, history: list | None) -> str:
        """构造"带对话历史的检索提示"，供知识库工具做指代消解。

        ★ 为什么 Agent 路径需要它（实测）
        ------------------------------
        Agent 的决策 LLM **会自己改写追问**，实测有效：

            追问「公示期多长？」
              → 模型给的 query: '依法必须进行招标的项目 招标人应于何时公示中标候选人 公示期多长'

        所以 Agent 路径并非"完全没有指代消解"。但模型改写**不保证稳定**——
        实测另一例它先给出 '投标保证金退还时限是多久'（跑偏到上一个话题），
        用了三轮工具调用才自我修正到 '履约保证金退还时限规定'。

        把上一轮问题拼进检索提示，可以：
            · 模型改写得好 → 检索式里多几个相关词，无害
            · 模型改写跑偏 → 上一轮问题兜住上下文，少走几轮弯路

        代价：检索式变长，可能引入上一轮的无关词。故实测收益（0.627 → 0.868）
        明显大于这点噪音。

        ★ 拼在**后面**而不是前面
        ----------------------
        模型改写的 query 是针对当前问题的，语义上更该主导检索；
        上一轮问题作为上下文补充。故 `改写query + 上一轮问题` 的顺序
        比反过来更合理（与 `pipeline._retrieval_query` 的
        `上一轮问题 + 当前问题` 不同——那里"当前问题"本身就是用户的原始追问，
        没有模型改写这一层）。
        """
        if not history:
            return question
        for item in reversed(history):
            if not isinstance(item, dict) or item.get("role") != "user":
                continue
            previous = (item.get("content") or "").strip()
            if previous and previous != (question or "").strip():
                return f"{question}（上文：{previous}）"
            break
        return question

    @staticmethod
    def _planning_enabled(question: str = "") -> bool:
        """是否启用 Plan-and-Solve 规划轮。

        `AGENT_PLANNING` 环境变量控制：
            · 未设 / `0` / `false` / `no` / `off` → **关闭**（默认）
            · `auto`                             → **按问题复杂度分流**：
                                                   复合问题才规划（见下）
            · `1` / `true` / `yes` / `on`         → 总是规划

        ★ 为什么默认关闭（实测依据，见 docs/28）
        --------------------------------------
        规划轮是一次额外的非流式 LLM 调用（实测 2~5 秒）。而实测显示：
        单源问题只需 1~2 次工具调用、本来就不打转，多付这几秒是净亏。
        只有复合问题（5~6 次调用、2/5 撞轮次上限）才有收益。

        故先做成开关，用 A/B 拿数据再决定是否设默认——
        与提示词变体（`RAG_PROMPT_VARIANT`）、精排开关（`RERANK_ENABLED`）
        同一套路：**改行为先有数据**。

        ★ `auto` 模式解决的是"一律开/一律关都不对"
        --------------------------------------
        默认关闭意味着**复合问题也拿不到规划**（而那正是它唯一有收益的场景）；
        一律打开又会让简单问题白等 2~5 秒。`auto` 用问题本身的结构特征
        （`planning.is_compound_question`，纯函数、带单测）做分流：
        复合问题规划、单源问题不规划。

        ⚠️ `auto` **尚未设为默认**：它同样需要 A/B 数据支撑
        （规划是否真的减少了轮次/工具调用，而不是只增加了耗时）。
        在拿到数据前，默认仍是关闭 —— 与"改行为先有数据"一致。

        取值：非法值一律当关闭，不抛异常（拼错一个环境变量不该让服务起不来）。
        """
        raw = (os.getenv("AGENT_PLANNING") or "").strip().lower()
        if raw == "auto":
            from src.agent.planning import is_compound_question

            compound = is_compound_question(question)
            logger.info(
                "规划分流（auto）：问题判定为%s → %s规划轮",
                "复合" if compound else "单一",
                "启用" if compound else "跳过",
            )
            return compound
        return raw in ("1", "true", "yes", "on")

    @staticmethod
    def _memory_enabled() -> bool:
        """是否启用会话记忆（把历史被采纳的问答作为参考注入）。

        `AGENT_MEMORY` 环境变量控制，**默认开启**。

        与 `AGENT_PLANNING`（默认关闭）相反，这里默认开是因为两者成本结构不同：
            规划轮  每次问答多一次 LLM 调用（2~5 秒固定成本）
            会话记忆 只在**首次**编码历史问答时花 CPU，且命中不了相似历史时
                    直接返回空（注入 0 字符），没有固定延迟成本

        代价是首次问答多几百毫秒编码时间（历史条数少时很快），之后走进程内缓存。
        """
        raw = (os.getenv("AGENT_MEMORY") or "").strip().lower()
        # 显式关闭才关：默认开启
        return raw not in ("0", "false", "no", "off")

    def _chat_events(  # noqa: PLR0913 - 编排入口，参数多于一般函数属预期
        self, question: str, history: list = None,
        web_search_enabled: bool = False, provider: str = "",
        deep_thinking_enabled: bool = False,
        auto_web_fallback: bool = False,
        on_status=None,
        session_id: str = "",
    ):
        """编排入口。产出事件元组：

        ("status"|"token"|"thinking", 文本)  ("reset", None)
        ("clarify", {questions: [{question, options}], reason})
        ("done", {sources, web_sources, tool_called, tool_name, elapsed_ms,
                  phase_times, web_fallback})
        ("error", {"content": 用户可见文案})

        Agent 主路径失败时自动降级 RAG；降级也失败产出 error，不产出 done。

        ★ 开头必须清空「给用户看的质量说明」旁路（局限 6）
        --------------------------------------------
        `react_loop._USER_HINTS` 是**模块级**累积区（因为 `_validate_result`
        是 staticmethod，没有 self）。不清空的话，上一轮问答的提示会
        串到这一轮 —— 用户会看到与本问题无关的"依据可能不完整"。
        

        ★ `clarify`（首轮追问）的语义
        ------------------------
        模型判定**信息不足、需要用户先补充**时产出本事件，随后**本轮立即结束**
        （不检索、不生成、不产出 done）。用户回答后由前端把「原问题 + 补充信息」
        作为**新一轮请求**发回，那时已有历史，不会再追问。

        故 `clarify` 与 `done` 是**互斥**的：前端收到 clarify 后不应期待 done。
        这一点必须写清，否则前端会一直等 done 而把界面卡在"生成中"。

        `auto_web_fallback=True` 时先判断本地知识库够不够用：
        不够则自动开启联网搜索并把判断理由写进系统提示词，
        让模型知道"本地没有答案、必须去查"。

        `on_status` 是可选的进度回调——自动判断要在首帧之前跑（含一次 LLM 判读），
        没有回调时用户会看到一段空白等待，不知道系统在做什么。
        """
        # ★ 清空「给用户看的质量说明」旁路（局限 6 的配套）
        #
        # `react_loop._USER_HINTS` 是模块级累积区（因为 `_validate_result`
        # 是 staticmethod、没有 self）。
        #
        # ⚠️ 为什么必须在这里清，而不只靠 `take_user_hints()` 的副作用
        # ------------------------------------------------------
        # `take_user_hints()` 在**组装 done 帧时**清空。若某次问答的
        # done 帧没被产出或被消费（异常路径、调用方没读完 generator），
        # 累积的提示会留到**下一轮** —— 用户会看到与本问题无关的
        # "依据可能不完整"。故在每次问答的**入口**无条件清一次。
        try:
            from src.agent.react_loop import reset_user_hints

            reset_user_hints()
        except Exception:  # noqa: BLE001 - 观测清理不能影响问答
            pass

        t_start = time.monotonic()
        meta = {
            "sources": [],
            "web_sources": [],
            "tool_called": False,
            "tool_name": "",       # 单字符串：最后一个执行完成的工具名（前端徽标 + eval 依赖）
            "phase_times": [],
            # 阶段偏移的基准（与 `t_start` 同一个 monotonic 读数）。
            # 单独放进 meta 是因为记录发生在 mixin 方法里，拿不到局部变量 t_start。
            "t_start": t_start,
            "last_stream_messages": None,
            "streamed": False,     # 是否已有正文流出：决定失败时走 RAG 降级还是直接报错
            "web_search_enabled": web_search_enabled,  # 决定决策轮暴露哪些工具（见 react_loop）
            # 用户原问题：工具执行时要传给执行器（联网工具据此重排并补算相关度）
            "question": question,
            # 带对话历史的检索提示：多轮时把上一轮用户问题拼进来，供知识库工具
            # 做指代消解（实测：追问的检索相似度 0.627 → 0.868）。
            # 与 `question` 分开两个键，是因为"当前问题"与"检索式"是两件事：
            # 生成阶段仍须用原始问题，否则会把上一轮的问题也当成待答内容。
            "retrieval_hint": self._retrieval_hint(question, history),
            # ★ 是否允许本轮追问（首轮澄清）
            #
            # 只在**没有任何对话历史**时允许 → 即"首轮"。
            #
            # 为什么必须限定首轮（而不是每轮都能问）：
            #   · 多轮里用户已经在追问细节，此时再要求"补充信息"会打断正常对话
            #   · 更要紧的是**死循环风险**：用户回答了澄清问题后，若仍允许追问，
            #     模型可以对同一处继续追问第二、第三轮，用户永远拿不到答案。
            #     首轮限定从结构上排除这种情况，不依赖模型自觉。
            #
            # 同时要求用户问题本身有一定长度：极短输入（"你好"、"?"）更可能是
            # 误触或寒暄，此时追问只会显得机械。
            "clarify_allowed": (not history) and len(question.strip()) >= CLARIFY_MIN_QUESTION_CHARS,
            # 自动兜底的判定结果（前端据此说明"为什么联网了"）
            "web_fallback": None,
            # 是否跑规划轮（Plan-and-Solve）。由 `_planning_enabled(question)` 决定：
            #   默认关闭（单源问题多付一次 LLM 调用是净亏，见该方法说明）
            #   `AGENT_PLANNING=auto` → **按问题复杂度分流**：复合问题才规划
            "planning_enabled": self._planning_enabled(question),
            # 规划结果（没跑或没产出时仍写入，便于排查）
            "plan": None,
            # 会话 ID：历史压缩的滚动水位线按它分桶（同一会话跨轮复用摘要，
            # 避免每轮重复压缩导致摘要层层衰减）
            "session_id": session_id,
            # 是否启用会话记忆（默认开启，见 `_memory_enabled`）
            "memory_enabled": self._memory_enabled(),
            # 历史压缩结果（默认开启，见 `compaction_enabled`）
            "compaction": None,
            # 本轮召回的历史示例（供 done 帧展示"为什么这么答"）
            "memory_examples": [],
        }
        yield "status", "正在初始化..."

        # ---- 自动兜底判断 ----
        fallback_note = ""
        if auto_web_fallback and not web_search_enabled:
            if on_status:
                on_status("正在检查本地知识库…")
            yield "status", "正在检查本地知识库…"
            cov = self._assess_local_coverage(question)
            meta["web_fallback"] = cov.to_dict()
            if not cov.enough:
                web_search_enabled = True
                meta["web_search_enabled"] = True
                fallback_note = (
                    "⚠️ 本地知识库**没有**能回答该问题的资料，"
                    f"原因：{cov.reason}。\n"
                    "请**必须**调用联网搜索工具（search_web / search_web_cn）获取最新资料后回答；"
                    "不要凭已有知识猜测，也不要把本地资料当作答案。"
                    "时效性内容（最新公告、法规更新）尤其要联网确认。"
                )
                yield "status", "本地知识库无匹配，正在联网搜索…"

        error_event = None
        # 首轮追问的载荷（非 None 表示本轮以"向用户提问"结束）。见循环内的 clarify 分支。
        clarified = None
        try:
            client = self._get_llm_client(provider)
            # ★ 每请求挂 LLM 调用追踪（usage / 单次耗时 / 工具名）
            #
            # 记录器**每请求新建**并挂到客户端上，由外层 `finally` 摘掉。
            # 不拷贝客户端：调用必须真的落在调用方持有的那个实例上，
            # 否则测试里 `fake.raw_calls` 恒为 0（实测踩到，5 个用例挂掉），
            # 而"在原对象上观察调用"是合理的测试写法。
            #
            # 用**工厂**而不是直接 `LLMTraceRecorder()`：工厂按配置决定是否
            # 返回会上报 Langfuse 的子类。这样"关掉 Langfuse"就是"不配 key"，
            # 调用方无需任何分支（见 `langfuse_tracing.build_recorder`）。
            recorder = build_recorder()
            attach_recorder(client, recorder)
            meta["llm_trace"] = recorder
            # 开启 Langfuse 的一轮 trace（未配置时是 no-op）
            _tagger = getattr(recorder, "tagger", None)
            if _tagger is not None:
                _tagger.begin(
                    "chat",
                    metadata={
                        "web_search_enabled": web_search_enabled,
                        "deep_thinking_enabled": deep_thinking_enabled,
                        "provider": provider or "",
                        "session_id": session_id or "",
                    },
                )
            # ★ 记录本轮用的是哪条 provider 链 / 哪一家（done 帧回传，不静默切换）
            meta["llm_provider_chain"] = list(getattr(client, "chain", []) or [])
            meta["llm_provider"] = getattr(client, "last_used_provider", "") or ""
            if deep_thinking_enabled:
                # ★ 无论单客户端还是链，都在**这里**按需切换（链构造时不再切）
                #
                # 原实现写的是 `if not hasattr(client, "chain")`——只对"注入的
                # 单一客户端"切换，理由是"链在构造时已经切过了"。而构造时那次
                # 切换是无条件的（不看开关），故那个理由本身就是缺陷的一部分：
                # 配置了思考模型的用户，链**一直**是推理模型，与开关无关。
                #
                # 现在链不带思考模型，切换统一在此处发生。`_thinking_model`
                # 返回的是**浅拷贝**，所以并发请求之间不会互相影响。
                client = self._thinking_model(client, provider)
                # 若真的切到了推理模型，后面走 `chat_stream_thinking` 能让
                # 思维链单独成流；没切到则退回提示词注入（generation 层判断）
                meta["deep_thinking_model"] = getattr(client, "model_name", "") or ""
                yield "status", "正在深度思考..."
            messages = self._build_context(
                question, history, web_search_enabled, fallback_note,
                memory_enabled=bool(meta.get("memory_enabled")), meta=meta,
                # 会话 ID 用于压缩状态的滚动水位线（同一会话跨轮复用摘要，
                # 避免每轮重复压缩导致摘要层层衰减）
                session_id=str(meta.get("session_id") or ""),
                # client 供压缩调用 LLM；此处已就绪（上面刚取到）
                client=client,
            )
            # 累积答案正文供生成后核对（`_verify_final_answer` 需要完整回答）。
            # 为什么在 core 层拦而不是让 react_loop 自己攒：
            # reset 事件（防泄漏清空重来）也经过这里，在此累积能保证
            # 攒到的就是**用户最终看到的那份**，与前端语义一致。
            for event_type, payload in self._chat_stream_tools(
                messages, client, meta, deep_thinking_enabled
            ):
                if event_type == "token":
                    meta["answer_text"] = meta.get("answer_text", "") + str(payload)
                elif event_type == "reset":
                    # 与前端语义一致：已流正文被清空，累积也要跟着清
                    meta["answer_text"] = ""
                elif event_type == "clarify":
                    # ★ 追问是**终止事件**：本轮不检索、不生成
                    #
                    # 记下来供下面的收尾分支使用。必须在这里捕获，因为
                    # `_chat_events` 后面还有"答案核对 / 同步 provider / 产 done"
                    # 一整套收尾逻辑，对追问而言**全都不该执行**：
                    #   · `_verify_final_answer` 核对的是"回答有没有引用依据"，
                    #     而本轮没有回答 → 会得出"无依据"的错误结论写进 done
                    #   · done 帧里的 sources/tool_called 都是空，
                    #     前端若按常规路径渲染会显示一张空白来源卡
                    clarified = payload
                yield event_type, payload
        except Exception:
            logger.exception("Agent 编排失败，尝试降级")
            if meta["streamed"]:
                # 正文已部分流出：不再降级重答（避免重复内容），直接报错由前端兜底
                error_event = {"content": "回答生成中断，请稍后重试。"}
            else:
                for event_type, payload in self._rag_fallback_events(question, meta):
                    if event_type == "error":
                        error_event = payload
                    else:
                        yield event_type, payload
        finally:
            # ★ 摘掉本轮的 LLM 追踪记录器
            #
            # 放在 `finally` 而不是函数末尾：`_chat_events` 是生成器，
            # 调用方可能中途放弃（前端断开、stop 按钮），此时函数体
            # **不会**执行到末尾。不摘的话记录器会留在被复用的客户端上，
            # 下一个请求的调用被记进**上一个**请求的记录器——
            # 表现为"token 数忽多忽少、调用条数对不上"。
            #
            # 用 `locals().get("client")`：`client` 在 try 块内才赋值，
            # 若在取客户端之前就失败（如 provider 未配置），变量不存在。
            detach_recorder(locals().get("client"))
            # 结束 Langfuse 的一轮 trace（未配置时 no-op）。
            #
            # **不主动 flush**：SDK 有后台定时导出，主动 flush 会阻塞请求收尾，
            # 而观测不该让用户多等。崩掉最多丢几条，比拖慢每个请求划算。
            _rec = meta.get("llm_trace")
            _tg = getattr(_rec, "tagger", None)
            if _tg is not None:
                try:
                    # ★ P1：把**已有的质量信号**一并上报
                    #
                    # 这些值在 `done` 帧里本来就有（`verify_*` / `rag_timing` /
                    # provider 降级），只是此前没往 Langfuse 传 —— 于是
                    # "哪批 trace 核对失败"、"哪批命中了检索缓存"都筛不出来。
                    #
                    # 只放**取到的**值：缺的键不出现，而不是补 None/0。
                    # 与 `_usage_details` 同一原则（补 0 会被读成"确实是 0"）。
                    _verify = meta.get("verification") or {}
                    _rag = meta.get("rag_timings") or []
                    _rag_sum = rag_timing_summary(_rag) if _rag else None
                    _quality: dict = {}
                    if _verify:
                        _quality["verify_ok"] = bool(_verify.get("ok"))
                        _quality["verify_critical_ok"] = bool(
                            _verify.get("critical_ok")
                        )
                        for _k, _fk in (
                            ("unsupported_laws", "unsupported_laws"),
                            ("unsupported_articles", "unsupported_articles"),
                            ("unsupported_numbers", "unsupported_numbers"),
                        ):
                            _v = _verify.get(_fk)
                            if _v is not None:
                                _quality[_k] = len(_v or [])
                    if _rag_sum:
                        _quality["rag_ms"] = _rag_sum.get("total_ms")
                        _quality["rag_cache_hit"] = bool(_rag_sum.get("all_cached"))
                        _quality["rag_rerank_failed"] = _rag_sum.get("rerank_failed")
                    # 提供商降级：>0 说明主模型不可用、答案来自备用模型
                    _sw = meta.get("llm_provider_switches")
                    if _sw:
                        _quality["provider_switches"] = int(_sw)

                    _tg.finish(
                        output_summary={
                            "answer_chars": len(meta.get("answer_text") or ""),
                            "tool_calls": len(meta.get("tool_timings") or []),
                            "sources": len(meta.get("sources") or []),
                        },
                        metadata={
                            "elapsed_ms": int((time.monotonic() - t_start) * 1000),
                            # ★ P2：`failing_components` 此前恒为 None（死字段）。
                            # 现在**只在真有降级时**才出现，且给出是哪一家失败的。
                            # 不做每请求深探测（`health.probe_llm_deep` 有成本）——
                            # `provider_errors` 是**已经发生的事实**，零成本。
                            "failing_components": (
                                _fail_names
                                if (_fail_names := [
                                    str(e.get("provider"))
                                    for e in (meta.get("llm_provider_errors") or [])
                                    if isinstance(e, dict) and e.get("provider")
                                ])
                                else None
                            ),
                            **_quality,
                        },
                    )
                except Exception:  # noqa: BLE001 - 观测不能搞挂业务
                    logger.debug("Langfuse finish 失败", exc_info=True)
                # ★ P0 前置：记下 session → trace_id
                #
                # 用户点赞是**之后**的独立请求（只带 session_id + message_id），
                # 而 trace_id 只活在内存里。故必须在这里落到能按 session_id
                # 找回的地方，反馈才能挂成 Langfuse Score。
                #
                # 放在 `finish()` **之后**：trace_id 在 `begin()` 时就已生成，
                # 但这时才确定这轮真的产出了内容（提前记会把"崩在半路"的
                # trace 也留成可反馈对象）。
                try:
                    from src.langfuse_feedback import remember_session_trace

                    _sid = str(meta.get("session_id") or "")
                    _tid = getattr(_tg, "_trace_id", None)
                    if _sid and _tid:
                        remember_session_trace(_sid, str(_tid))
                except Exception:  # noqa: BLE001 - 观测不能搞挂业务
                    logger.debug("记录 session→trace 映射失败", exc_info=True)

        if error_event is not None:
            yield "error", error_event
        elif clarified is not None:
            # ★ 追问路径：仍然产 done，但**跳过答案核对**
            #
            # 为什么必须产 done（而不是只产 clarify 就结束）：
            # 前端的流式状态由 done/error 收尾。只产 clarify 的话本轮
            # 没有终止帧，界面会一直停在"生成中"，直到超时——
            # 而用户看到的是几个问题挂在半空中，无处可答。
            #
            # 为什么跳过 `_verify_final_answer`：它核对的是"回答有没有引用依据"，
            # 本轮**没有回答**。硬跑会得出"无依据"这类结论写进 done 帧，
            # 属于凭空生成的质量判定。
            self._sync_provider_meta(meta, locals().get("client"))
            done = self._build_done(meta, t_start)
            # 显式标记，便于前端区分"追问结束"与"正常答完"
            done["clarified"] = True
            yield "done", done
        else:
            # ★ 答案核对放在**产 done 之前**，两条出口都能拿到结果。
            #
            # 最初把核对放在 `chat()` 里（非流式出口），结果流式路径的 done 帧
            # 早就发出去了，verification 永远带不上——实测确认过：
            # done 帧字段里根本没有 verification。
            # 放在这里则两条路径共用，且都在 done 之前完成。
            verification = self._verify_final_answer(meta.get("answer_text") or "", meta)
            # ★ 核对发现问题 → **定向修复一轮**（把"只上报"变成"能纠正"）
            #
            # 修复本身会再产 token（先 reset 清空再重发），故必须在这里、
            # 也就是 done 之前完成；且要用生成器 `yield from` 把事件透出去。
            if verification and not verification.get("ok"):
                yield from self._repair_answer_events(
                    meta, verification, locals().get("client")
                )
            # provider 信息在**生成结束后**再同步一次：`last_used_provider`
            # 要在真正产出内容之后才被写入，进入 try 时读到的还是空值。
            self._sync_provider_meta(meta, locals().get("client"))
            done = self._build_done(meta, t_start)
            # ★ 在线评测采样（真实请求）
            #
            # 放在 `_verify_final_answer` **之后**：它采集的 `verify_*` 指标
            # 就来自核对结果，早了拿不到。
            #
            # 为什么放在 `yield done` **之前**而不是之后：生成器在 yield 之后
            # 要等调用方再次 next() 才继续执行；若调用方在收到 done 后就不再
            # 迭代（前端断开、测试用 list() 后立即退出），yield 之后的代码
            # **可能不执行**，采样会静默丢失。
            #
            # `maybe_record` 本身是纯提取 + 一次 `put_nowait`（写盘在后台线程），
            # 故这里不会增加可感知的延迟。
            self._record_online_eval(meta, done, question, bool(history))
            yield "done", done

    @staticmethod
    def _record_online_eval(meta: dict, done: dict, question: str,
                            has_history: bool) -> None:
        """把本轮请求交给在线评测器。**永不抛异常、永不阻塞**。

        抽成静态方法是为了让这个"绝不能出错"的调用点显式、可测。
        """
        try:
            from src.eval_online import evaluator

            evaluator.maybe_record(
                meta, done, question=question, has_history=has_history
            )
        except Exception:  # noqa: BLE001 - 评测是旁路，绝不能影响请求
            logger.debug("在线评测接入失败（不影响请求）", exc_info=True)

    def _assess_local_coverage(self, question: str):
        """判断本地知识库能否回答该问题。**永不抛异常**。

        失败时返回"不够"——因为**证明不了够**时，联网的风险小于
        "凭不足的依据作答"（后者会产生看起来很确定的错误答案）。
        """
        try:
            from src.services.web_fallback import local_coverage

            return local_coverage(question)
        except Exception as exc:
            logger.warning("本地覆盖判断失败，按「不够」处理并联网", exc_info=True)
            from src.services.web_fallback import LocalCoverage

            return LocalCoverage(
                enough=False,
                best_relevance=None,
                threshold=0.0,
                reason=f"覆盖判断失败（{type(exc).__name__}），已改用联网搜索",
            )

    @staticmethod
    def tool_names() -> list:
        """当前注册的工具名列表（健康检查/诊断用）。"""
        return list(TOOL_EXECUTORS)

    @staticmethod
    def _sync_provider_meta(meta: dict, client) -> None:
        """把 provider 链的运行时状态同步进 meta，供 done 帧回传。

        ★ 为什么要在生成结束后同步，而不是进入 try 时读一次就完
        ------------------------------------------------------
        `last_used_provider` 是**产出内容之后**才被写入的（见 `fallback.py`），
        在开始生成前读只会拿到空字符串。故生成结束再同步一次。

        `client` 可能为 None（走 RAG 降级路径时没取过客户端）或注入的替身
        （没有这些属性）——两种情况都只是**少一条可观测信息**，
        不该影响回答本身，故全部用 getattr 兜底。
        """
        if client is None:
            return
        chain = getattr(client, "chain", None)
        if chain:
            meta["llm_provider_chain"] = list(chain)
        used = getattr(client, "last_used_provider", "")
        if used:
            meta["llm_provider"] = used
        switches = getattr(client, "switch_count", 0)
        if switches:
            meta["llm_provider_switches"] = switches
        errors = getattr(client, "last_error", None)
        if errors:
            # 只留最近 3 条，每条截断：够定位是哪家、什么错即可，
            # 不要把整段堆栈塞进 SSE 帧
            meta["llm_provider_errors"] = [
                {"provider": name, "error": str(text)[:200]}
                for name, text in list(errors)[-3:]
            ]

    @staticmethod
    def _mark_phase(meta: dict, name: str, t0: float) -> None:
        """记录一个阶段的耗时（含**相对流程开始的偏移**）。

        ★ 为什么统一走这里而不是各调用点自己 append
        --------------------------------------
        共 6 处调用点（ReAct 决策/检索各若干轮、最终生成、RAG 降级的两段）。
        若每处各写一遍 `append((name, ms))`，加偏移字段时就要改 6 处——
        漏一处那条阶段就没有偏移，前端时间轴会出现"缺一格"却看不出原因。
        统一入口后，格式只在一个地方定义。

        ★ 偏移从哪里算
        -----------
        以 `meta["t_start"]`（`_chat_events` 开头记的 monotonic）为基准。
        用 monotonic 而非 wall clock：后者会被系统对时/夏令时影响，
        算出负偏移或跳变。
        """
        t1 = time.monotonic()
        base = meta.get("t_start") or t0
        meta["phase_times"].append(
            (name, int((t1 - t0) * 1000), int((t0 - base) * 1000))
        )

    @staticmethod
    def _build_done(meta: dict, t_start: float) -> dict:
        """组装 done 帧（契约见开发文档 §6.4，字段结构与 tool_name 单字符串语义勿改）。

        `_meta` 是**内部传递用**的私有键：`chat()` 取出它做答案核对后立即 pop 掉，
        不会出现在对外返回里。挂在这里是因为这是 meta 最后一次可用之处，
        而核对需要 meta 里的工具结果文本与成功工具集合。
        """
        # ★ 把阶段耗时喂给监控聚合（局限 6）
        #
        # 为什么在这里而不是 api/server 的中间件：中间件只看得到 HTTP 状态码
        # 与总耗时，**拿不到** phase_times（它在 done 帧里）。
        # _build_done 是每次问答的唯一收口处，且已拿到最终 phase_times。
        #
        # 延迟 import + 吞异常：观测不能影响问答，也不能让 agent 包
        # 对 monitoring 产生硬依赖。
        try:
            from src.monitoring import record_phases

            record_phases(meta.get("phase_times") or [])
        except Exception:  # noqa: BLE001 - 观测不能影响问答
            pass

        return {
            "sources": meta["sources"],
            "web_sources": meta["web_sources"],
            "tool_called": meta["tool_called"],
            "tool_name": meta["tool_name"],
            # 每一次工具调用的完整序列（含重复）。评测据此算"打转"指标——
            # `tool_name` 是单字符串、`succeeded_tools` 是去重集合，都算不出重复次数。
            "tool_names": meta.get("tool_call_names") or [],
            # 本轮因"同工具同参数"被去重、没有真正执行的调用次数。
            #
            # ★ 这是一个**告警位，正常恒为 0**
            # ------------------------------
            # 实测：`tool_defense` 的 JSON 与 DSML 两条解析路径**都**已经做了
            # 同名同参去重，故正常情况下这一层兜底不会命中。
            # 它非 0 只有两种可能：① 新增了绕过解析器的调用路径；
            # ② 解析层去重被改坏。两种都值得立刻看见，故显式上报而不是藏着。
            "tool_deduped": int(meta.get("tool_deduped") or 0),
            "elapsed_ms": int((time.monotonic() - t_start) * 1000),
            # 阶段耗时：`[[名称, 该阶段毫秒, 该阶段开始的相对毫秒], ...]`
            #
            # ★ 第三个元素（偏移）为什么要有
            # -------------------------
            # 只有耗时的话，前端能列出"首轮分析 1.8s、检索 2.0s"，
            # 但看不出先后与间隔。有偏移才能画时间轴，也才能发现
            # "两次 LLM 调用之间有 3s 空档"这类问题。
            "phase_times": meta["phase_times"],
            # 分阶段合计。与 `elapsed_ms` 的差额 = 没被任何阶段覆盖的时间
            # （初始化、会话记忆检索、答案核对、SSE 收尾等）。
            #
            # ★ 为什么要显式给出而不是让前端自己加
            # -------------------------------
            # 前端自己 reduce 一次也能算，但把这个数放进契约有两点好处：
            #   · 后端与前端对"合计"的口径不会各算各的（例如是否含降级阶段）
            #   · 评测与监控能直接读它，不必先解析 phase_times
            "phase_total_ms": sum(
                int(item[1]) for item in meta["phase_times"] if len(item) > 1
            ),
            # 自动兜底的判定结果：前端据此说明"为什么这次联网了"，null 表示未启用自动模式
            "web_fallback": meta.get("web_fallback"),
            # ★ 给**用户**看的质量说明（局限 6 的修复）
            #
            # 原先 `tool_quality.auto_switch_hint` 有实现但**零调用点** ——
            # 用户从未真的收到过那句话。现在每次工具结果质量判定后记到旁路，
            # 这里随 done 帧带出。
            #
            # 为什么只放**一句话**而不是逐工具的 details：
            #   用户不需要知道是哪个工具弱（`format_hint` 那份给模型看）。
            # 为什么是 list 而不是 str：
            #   可能同时有"依据不完整"与"数据源不可用"两类，前端可自行拼接；
            #   空 list 表示没有问题 —— 比 null 更好用。
            "quality_hints": _take_user_hints_for_done(),
            # 答案核对结果（零延迟规则验证）。null 表示没跑（如无工具调用、或降级路径）。
            # 前端可据此提示"部分引用未在检索结果中找到依据"。
            "verification": meta.get("verification"),
            # 核对失败后的定向修复记录（默认关闭，见 `_repair_enabled`）。
            # null 表示没触发；applied=true 时含 before/after 两次核对结果 ——
            # 有了它，A/B 才能回答"修复到底把问题改掉了没有"，
            # 而不是只知道"跑了一次修复"。
            "verification_repair": meta.get("verification_repair"),
            # 规划轮结果（Plan-and-Solve，默认关闭）。null 表示没跑。
            # 暴露它是为了 A/B 可观测：能直接看出规划产出了哪些信息需求、是否被过滤。
            "plan": meta.get("plan"),
            # 本轮召回到的历史示例（会话记忆）。空列表表示没有足够相似的历史。
            "memory_examples": meta.get("memory_examples") or [],
            # 历史压缩结果。null 表示没跑；compacted=false 表示未达触发条件
            "compaction": meta.get("compaction"),
            # 本轮实际使用的 LLM provider（多 provider 自动切换的可观测出口）。
            #
            # ★ 为什么必须暴露而不是只记日志
            # ----------------------------
            # 自动切换若"静默"发生，用户以为在用 DeepSeek、实际用了智谱——
            # 在合规与成本核算上都是问题。故把"用了哪家、切换了几次、
            # 切换原因"明确回传：
            #     provider        实际产出内容的 provider（空 = 未走到生成）
            #     provider_chain  候选链（含兜底顺序）
            #     provider_switch_count  切换次数（频繁切换说明主 provider 不稳）
            #     provider_errors 每次失败的原因（截断，够定位即可）
            # ★ LLM 调用明细与 token 用量（本轮问答的所有 LLM 调用）
            #
            # 放在 done 帧里而不是只进日志，是因为它回答三个日志答不出的问题：
            #     · 这轮问答花了几次 LLM 调用、共多少 token（成本）
            #     · prompt 缓存命中多少（DeepSeek 对命中部分打折，是省钱信号）
            #     · 慢在决策还是慢在生成（by_kind 分组耗时）
            "llm_trace": (
                meta["llm_trace"].summary()
                if meta.get("llm_trace") is not None else None
            ),
            # ★ 每个工具的执行耗时（与 phase_times 的"整阶段合计"互补）
            "tool_timings": meta.get("tool_timings") or [],
            # ★ RAG 检索耗时（含各步骤分解）
            #
            # 与 phase_times 里的"检索"互补：那是**工具层的笼统阶段**，
            # 这是**检索内部的真实构成**（查询向量化/混合检索/精排/过滤）。
            # 实测一次检索 32.6s 里查询向量化占 28.6s —— 只给总数看不出该优化哪一步。
            "rag_timing": rag_timing_summary(meta.get("rag_timings") or []),
            # ★ 图表数据：从工具结果里提取的**结构化数值**，供前端画图
            #
            # 放在 done 帧而不是让前端解析回答正文，是因为：
            #   · 回答是模型写的自然语言，数字可能被改写或省略
            #   · 只有后端见过工具的**原始输出**，那才是唯一事实来源
            "charts": meta.get("charts") or [],
            "provider": meta.get("llm_provider") or "",
            "provider_chain": meta.get("llm_provider_chain") or [],
            "provider_switch_count": meta.get("llm_provider_switches") or 0,
            "provider_errors": meta.get("llm_provider_errors") or [],
            "_meta": meta,
        }

    def _verify_final_answer(self, answer: str, meta: dict) -> dict | None:
        """对最终答案做零延迟核对，结果写回 meta 并返回。

        只对有工具调用的回答做——没有工具结果可比对时，
        任何法规引用都会被判"未支持"，那是无意义的（模型只能凭参数记忆作答）。
        """
        if not meta.get("tool_called"):
            return None
        if not (answer or "").strip():
            return None
        try:
            from src.agent.verification import verify_answer

            result = verify_answer(
                answer,
                tool_text="\n".join(meta.get("tool_output_text") or []),
                succeeded_tools=meta.get("succeeded_tools") or [],
            )
        except Exception:
            # 核对是**增益项**：它失败不该影响回答本身
            logger.exception("答案核对失败（不影响回答）")
            return None
        meta["verification"] = result.to_dict()
        if not result.ok:
            logger.warning("答案核对发现问题：%s", result.summary())
        return meta["verification"]

    # --- 核对失败后的定向修复（把 Reflection 从"只上报"变成"能纠正"）---

    #: 修复轮的提示词。刻意**收窄任务**：只改引用，不改结论。
    #:
    #: ★ 为什么把"必须处理"写成命令式、并禁止保留原文表述（实测调优）
    #: --------------------------------------------------------
    #: 第一版写的是"请只做必要的修正"，实测失败的两例**逐字返回了原文**
    #: （517→517、1283→1283）—— 模型自己判断"引用没问题"，于是什么都没改，
    #: 而我们还白等了一次 LLM 调用 + 重发一遍全文。
    #: 故改成：逐项点名 + 明确"不允许保留这些字符串" + 给出删不掉时的退路。
    _REPAIR_PROMPT = """自动核对已判定：下面这份回答里的**下列引用在检索结果中找不到依据**。

{issues}

核对结论是确定的（不是建议复查）。请对上面列出的**每一项**都做出处理：
1. 若检索结果里有对应的、**确切出现**的表述 → 替换成那个表述（保留原句其他部分）。
2. 若检索结果里没有 → **删除该引用**（条款号/数字/法规名），并把句子改通顺；
   必要时改为概括说法（如"相关规定要求…"）。
3. **不允许**在输出里保留上面列出的那些原字符串。

其他约束：
- 不要改变结论、不要新增内容、不要调整结构；未被点名的部分逐字保留。
- 保持原回答的长度量级与中文表达风格，不要写"已修正"之类的说明。
- 直接输出修正后的完整回答，不要前言后语、不要 markdown 代码块。

【检索结果】
{tool_text}

【原回答】
{answer}
"""

    @staticmethod
    def _repair_enabled() -> bool:
        """是否开启"核对失败 → 定向修复"。`AGENT_VERIFY_REPAIR` 控制，默认关闭。

        ★ 为什么默认关闭（与项目"改行为先有数据"一致）
        ------------------------------------------
        修复轮要**再花一次 LLM 调用**（实测数秒），而它只在核对失败时触发
        （实测失败率约 1/4 ~ 1/5，取决于问题是否涉及具体条款号）。
        收益是把"引用编造"这类**高危害**问题当场改掉，代价是那部分请求变慢。
        这个取舍应由数据决定，故先做成开关、默认关，
        并把修复前后的核对结果都写进 done 帧（`verification_repair`）供 A/B。

        ★ 为什么复用 `reset` 事件而不是追加一段"更正说明"
        ---------------------------------------------
        `reset` 的语义是"已流出的正文作废、重新来"，本来就用在同一类场景
        （DSML 泄漏时清空重发）。前端已完整支持它（会清空气泡再收新 token），
        故修复后用户看到的仍是**一份完整、自洽的回答**，
        而不是"原文 + 一段补充说明"（后者会自相矛盾，用户还得自己判断哪句作数）。
        """
        return (os.getenv("AGENT_VERIFY_REPAIR") or "").strip().lower() in (
            "1", "true", "yes", "on",
        )

    def _repair_answer_events(self, meta: dict, verification: dict, client):
        """核对失败后的定向修复：产出 status → reset → token 事件。

        `client` 由调用方（`_chat_events`）传入 —— 与本轮生成用的是**同一个**
        已解析客户端（含 provider 降级结果），不在这里重新解析：
        重解析可能选到另一个 provider，甚至因为拿不到客户端而静默跳过。

        失败（开关关、无客户端、LLM 报错、修复后更差）时**什么都不产出**，
        调用方的原答案与既有 verification 保持不变 —— 修复是增益项，
        不能反过来把一次本来可用的回答搞坏。
        """
        if not self._repair_enabled():
            return
        answer = str(meta.get("answer_text") or "")
        tool_text = "\n".join(meta.get("tool_output_text") or [])
        if not answer.strip() or not tool_text.strip() or client is None:
            return

        issues = self._format_repair_issues(verification)
        if not issues:
            return

        prompt = self._REPAIR_PROMPT.format(
            issues=issues, tool_text=tool_text[:6000], answer=answer
        )
        try:
            repaired = client.chat_raw([
                {"role": "system", "content": "你是严谨的招投标法规问答校对员。"},
                {"role": "user", "content": prompt},
            ])
        except Exception:
            logger.exception("修复轮调用失败（保留原答案）")
            return

        repaired = str(repaired or "").strip()
        # 保守校验：太短就当修复失败（避免把一份完整回答换成残句）
        if len(repaired) < max(40, len(answer) * 0.5):
            logger.info("修复轮输出过短（%d 字 vs 原 %d 字），保留原答案",
                        len(repaired), len(answer))
            # 如实记录**为什么没修**：否则 A/B 只看到"未修复"，
            # 分不清是模型不肯改、还是被我们的保守闸拦下了
            meta["verification_repair"] = {
                "applied": False,
                "reason": "too_short",
                "before": verification,
                "answer_chars_after": len(repaired),
            }
            return

        # ★ 一字未改也要当"没修"处理（实测踩到）
        #
        # 第一版无论改没改都会 reset + 重发全文。实测有两例模型**逐字返回原文**
        # （模型自认引用没问题）—— 于是用户白等了一次 LLM 调用 + 一次全文重发，
        # 界面上还会闪一下（清空再逐字重画同一段文字）。
        # 故先比对：规范化后相同就什么都不做，并如实记 `no_change`。
        def _norm(text: str) -> str:
            return "".join(text.split())

        if _norm(repaired) == _norm(answer):
            logger.info("修复轮输出与原文一致（模型认为无需改动），按未修复处理")
            meta["verification_repair"] = {
                "applied": False,
                "reason": "no_change",
                "before": verification,
            }
            return

        logger.info("修复轮完成：%d 字 → %d 字", len(answer), len(repaired))
        yield "status", "正在按核对结果修正引用..."
        # reset 的语义与前端一致：作废已流出的正文，重新发
        yield "reset", None
        meta["answer_text"] = repaired
        for chunk in _pace_stream_chunks(repaired):
            yield "token", chunk

        # 修复后再核对一次：把"改好了没有"变成可观测的事实（而不是假设）
        try:
            from src.agent.verification import verify_answer

            after = verify_answer(
                repaired,
                tool_text=tool_text,
                succeeded_tools=meta.get("succeeded_tools") or [],
            )
            meta["verification_repair"] = {
                "applied": True,
                "before": verification,
                "after": after.to_dict(),
                "answer_chars_before": len(answer),
                "answer_chars_after": len(repaired),
            }
            meta["verification"] = after.to_dict()
            logger.info(
                "修复后核对：%s → %s",
                "通过" if verification.get("ok") else "未通过",
                "通过" if after.ok else "未通过",
            )
        except Exception:
            logger.exception("修复后核对失败（不影响已发出的答案）")
            meta["verification_repair"] = {"applied": True, "before": verification}

    @staticmethod
    def _format_repair_issues(verification: dict) -> str:
        """把核对结果整理成给模型看的"问题清单"（只列真正需要改的）。"""
        labels = (
            ("unsupported_articles", "条款号"),
            ("unsupported_numbers", "数字/百分比/金额"),
            ("unsupported_laws", "法规名称"),
            ("false_unavailable_claims", "误称工具不可用"),
        )
        lines = []
        for key, label in labels:
            values = verification.get(key) or []
            if values:
                shown = "、".join(str(v) for v in values[:8])
                lines.append(f"- {label}：{shown}")
        return "\n".join(lines)

    # --- 对外两条路径 ---

    def _chat(self, question: str, history: list = None, web_search_enabled: bool = False,
              provider: str = "", deep_thinking_enabled: bool = False,
              auto_web_fallback: bool = False, session_id: str = "") -> dict:
        """拉平事件流收集为 dict，并把 `meta` 挂在 `_meta` 上供核对使用。

        为什么返回 `_meta`：答案核对需要工具结果文本与成功工具集合，
        那些只在 meta 里。用私有键传递、由 `chat()` 取出并删除，
        避免污染对外契约（eval 与前端都读 `chat()` 的返回）。
        """
        result = {
            "answer": "",
            "sources": [],
            "web_sources": [],
            "tool_called": False,
            "tool_name": "",
            "elapsed_ms": 0,
            "phase_times": [],
            "web_fallback": None,
        }
        for event_type, payload in self._chat_events(
                question, history, web_search_enabled, provider,
                deep_thinking_enabled, auto_web_fallback, session_id=session_id):
            if event_type == "token":
                result["answer"] += payload
            elif event_type == "reset":
                # 与前端语义一致：清空已流正文，等后续 token 重来
                result["answer"] = ""
            elif event_type == "done":
                result.update(payload)
            elif event_type == "error":
                result["answer"] += f"\n\n【错误】{payload['content']}"
        return result

    def chat(self, question: str, history: list = None, web_search_enabled: bool = False,
             provider: str = "", deep_thinking_enabled: bool = False,
             auto_web_fallback: bool = False, session_id: str = "") -> dict:
        """非流式入口：拉平事件流收集为 dict。eval/agent_eval.py 依赖 tool_name 口径。

        核对已在 `_chat_events` 产出 done 之前完成（两条出口共用），
        故这里只负责把 `_meta` 这个内部键剔掉。
        """
        result = self._chat(question, history, web_search_enabled, provider,
                            deep_thinking_enabled, auto_web_fallback, session_id=session_id)
        result.pop("_meta", None)
        return result

    def chat_stream(self, question: str, history: list = None,
                    web_search_enabled: bool = False, provider: str = "",
                    deep_thinking_enabled: bool = False,
                    auto_web_fallback: bool = False, session_id: str = ""):
        """SSE 流式入口：事件元组 → SSE 帧。首帧 2KB padding 强制代理 flush。"""
        yield SSE_PADDING
        for event_type, payload in self._chat_events(
                question, history, web_search_enabled, provider,
                deep_thinking_enabled, auto_web_fallback, session_id=session_id):
            if event_type == "reset":
                yield _sse({"type": "reset"})
            elif event_type == "done":
                frame = {"type": "done"}
                # `_meta` 是内部传递用的私有键（见 `_build_done`），
                # **不能进 SSE 帧**：它含工具结果全文与内部结构，
                # 塞进网络响应既浪费带宽又泄露实现细节。
                frame.update({k: v for k, v in payload.items() if k != "_meta"})
                yield _sse(frame)
            elif event_type == "clarify":
                # ★ 追问是**结构化载荷**，不能走下面的兜底分支
                #
                # 兜底分支取 `payload.get("content", "")` —— 而 clarify 的载荷是
                # `{"questions": [...], "reason": "..."}`，**没有 content 键**。
                # 走兜底的话前端只会收到 `{"type":"clarify","content":""}`，
                # 追问的问题与选项**全部丢失**，界面上什么都没出现，
                # 而本轮又已经结束（不产出正文）——用户看到的是"转圈后什么都没发生"。
                #
                # 故这里显式展开载荷。`type` 放最后写入，避免载荷里若意外含
                # `type` 键覆盖事件类型。
                frame = dict(payload) if isinstance(payload, dict) else {}
                frame["type"] = "clarify"
                yield _sse(frame)
            else:
                content = payload if isinstance(payload, str) else payload.get("content", "")
                yield _sse({"type": event_type, "content": content})
        yield SSE_DONE_MARK

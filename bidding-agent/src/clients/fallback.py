"""多 provider 自动切换：按候选链依次尝试，失败则换下一个。

★ 背景：provider 参数早就通了，"自动切换"才是缺口
----------------------------------------------
`api/server.py` 的请求体已有 `provider` 字段，`BiddingAgent._get_llm_client(provider)`
也已按 provider 缓存客户端。但**没有任何兜底**：

    DeepSeek 密钥失效 / 限速 / 网络不通 → 整个问答失败
    智谱明明配好了、可用 → 却不会被尝试

本模块补上这一环。

★ 核心设计：切换只能在"还没吐正文"之前
------------------------------------
`chat_stream` 是流式的。若已经吐了 300 字再切换 provider，用户会看到
**前半段 DeepSeek、后半段智谱**的拼接答案——两段语气与口径都不一致，
比直接报错更糟（用户不知道哪句可信）。

故每个方法内部用 `emitted` 标记"是否已产出内容"：

    未产出 → 可安全切换（换 provider 重试）
    已产出 → **不再切换**，异常向上抛
             （由调用方走既有降级：RAG 原文 / error 事件，口径与现在一致）

★ 为什么 `available()` 也要参与
----------------------------
`pipeline` 用 `llm.available()` 决定"要不要降级为直接返回知识库原文"。
若链里第一个 provider 没配密钥、第二个配了，而 `available()` 只看第一个，
就会**误判为不可用**——明明有可用的 provider 却降级了。
故 `available()` 是"链里有没有任一可用"。
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Iterator

from src.clients.base_client import BaseLLMClient
from src.clients.failover_policy import (
    UNKNOWN_COOLDOWN_SECONDS,
    cooldown_seconds_for,
)


def _vendor_key(name: str) -> str:
    """provider 名 → **厂商**键（用于冷却字典）。

    ★ 为什么单独包一层
    --------------
    `llm_factory.vendor_of` 是权威实现，但它会 import 整个工厂（连带
    settings 与各客户端）——而本模块在**异常路径**上，不能因为
    "查一个厂商名"而引入新的导入失败点。故这里做**防御式调用**：
    取不到就退回 provider 名本身，绝不抛异常。
    """
    try:
        from src.clients.llm_factory import vendor_of

        return vendor_of(name)
    except Exception:  # noqa: BLE001 - 厂商名查不到不该让降级链路崩掉
        return (name or "").strip().lower()

logger = logging.getLogger(__name__)

#: 失败 provider 的冷却时长（秒）。
#:
#: 取值与 `src/mcp/web_search_exa.FAILURE_COOLDOWN_SECONDS` 一致（300 秒）：
#: 足够让一次问答内的后续轮次跳过坏 provider，又不至于长时间不恢复。
#: 失败原因（401 / 密钥无效 / 域名不通）不会在一次问答内自愈，故重试是纯浪费。
FAILURE_COOLDOWN_SECONDS = 300.0

#: 候选链的环境变量名。值为逗号分隔的 provider 名，按顺序尝试。
FALLBACK_ENV = "LLM_FALLBACK_PROVIDERS"

#: 默认候选链。
#:
#: 为什么默认是"deepseek → zhipu"：项目的默认 provider 是 deepseek
#: （`agent.DEFAULT_PROVIDER`），而智谱是第二个配置齐全的国内厂商
#: （`ZHIPU_API_KEY` 已在本项目 `.env` 里用于 Web Search）。
#: 两者都不可用时链会退化到 `UnavailableLLMClient`，行为与现在一致。
DEFAULT_FALLBACK: tuple[str, ...] = ("deepseek", "zhipu")


def parse_fallback_chain(primary: str = "", raw: str | None = None) -> list[str]:
    """构造候选链：**显式指定的 provider 排第一**，其后是配置的兜底顺序。

    ★ 为什么显式指定的要排第一
    ------------------------
    调用方传 `provider="zhipu"` 是**明确的意图**——可能因为额度、
    合规、或对比测试。若兜底顺序把 deepseek 排在前面，
    这个参数就等于被忽略了（用户选了 zhipu 却用了 deepseek，还不报错）。

    ★ `raw` 的三态语义（别把它和"读环境变量"混在一起）
    ------------------------------------------------
        raw=None      → 读环境变量 `LLM_FALLBACK_PROVIDERS`
        raw=""        → **显式关闭自动切换**（只用 primary）
        raw="a,b"     → 用给定的顺序

    第一版写成了 `text = raw if raw is not None else os.getenv(...)`
    之后又去判断 `FALLBACK_ENV in os.environ`——于是**显式传了 raw 也会被
    环境变量覆盖**，表现为"自定义顺序"完全不生效（实测踩过）。
    这里改成：只看 `raw` 是否为 None 来决定要不要读环境变量。

    去重且保序：`"deepseek,zhipu,deepseek"` → `["deepseek", "zhipu"]`。
    """
    primary = (primary or "").strip().lower()

    if raw is not None:
        # 调用方显式给了值：空串代表关闭，非空串代表顺序
        configured = [p.strip().lower() for p in raw.split(",") if p.strip()]
    else:
        env_text = os.getenv(FALLBACK_ENV)
        if env_text is None:
            # 变量不存在 → 用默认链
            configured = [p.lower() for p in DEFAULT_FALLBACK]
        else:
            # 变量存在：空串 = 显式关闭自动切换
            configured = [p.strip().lower() for p in env_text.split(",") if p.strip()]

    chain: list[str] = []
    # ★ 按**厂商**去重，而不是按 provider 名（局限 6，实测缺陷）
    #
    # `zhipu` 与 `zai` 是同一家的两个名字。原先只比 `name not in chain`
    # → `zhipu,zai` 两个都保留 → 一次问答里会**两次撞同一个坏凭据**，
    # 且冷却是按 name 记的，拉黑一个不会拉黑另一个（现已改为按厂商记）。
    #
    # 折叠策略：**保留先出现的那个名字**（保序）。`primary` 最靠前，
    # 故"显式指定 zai + 兜底里有 zhipu"时保留 zai —— 尊重调用方意图。
    chain: list[str] = []
    seen_vendors: set[str] = set()
    for name in ([primary] if primary else []) + configured:
        if not name:
            continue
        vendor = _vendor_key(name)
        if vendor in seen_vendors:
            continue
        seen_vendors.add(vendor)
        chain.append(name)
    return chain


class FallbackLLMClient(BaseLLMClient):
    """把多个 provider 客户端包成一条候选链。

    对外是**透明的 `BaseLLMClient`**——调用方（pipeline / react_loop）
    不需要知道背后有几个 provider。

    `last_used_provider` / `last_error` 是可观测性出口：
    回答到底由哪家产出、切换时发生了什么，前端与日志都要能看到。
    "静默切换"是不可接受的——用户以为在用 DeepSeek，实际用了智谱，
    这在合规与成本核算上都是问题。
    """

    def __init__(self, chain: list[str], resolver) -> None:
        """:param resolver: 可调用对象 `(provider_name) -> BaseLLMClient`。
        注入而不是直接调工厂，便于测试与避免循环导入。
        """
        self._chain = list(chain)
        self._resolver = resolver
        self._lock = threading.Lock()
        #: 最近一次**成功产出内容**的 provider（供前端/日志展示）
        self.last_used_provider: str = ""
        #: 最近一次的失败记录：`[(provider, 异常文本), ...]`
        self.last_error: list[tuple[str, str]] = []
        #: 失败过的 provider 数（可观测性：>0 说明主 provider 不稳）
        self.switch_count: int = 0
        #: provider -> 冷却截止时间戳。刚失败的短期拉黑，避免一次问答内反复重试
        self._cooldown: dict[str, float] = {}
        #: 因冷却被跳过的次数（可观测性：能看出"主 provider 一直坏着"）
        self._cooldown_hits: int = 0

    # --- 展示 ---

    @property
    def provider(self) -> str:  # type: ignore[override]
        """链的展示名，如 `deepseek+fallback(zhipu)`。

        ★ 为什么不是简单的 "fallback"
        ---------------------------
        前端与日志需要知道**实际用的是哪家**。用链名 + 已用 provider，
        既不丢失"背后有兜底"这个事实，也能看出真正生效的是谁。
        """
        used = self.last_used_provider
        base = "+".join(self._chain) if self._chain else "none"
        return f"{used}(via {base})" if used else base

    def __repr__(self) -> str:  # pragma: no cover - 仅调试用
        return f"<FallbackLLMClient chain={self._chain} used={self.last_used_provider!r}>"

    @property
    def chain(self) -> list[str]:
        return list(self._chain)

    # --- 透传"当前会用的那个客户端"的能力 ---
    #
    # ★ 为什么链必须透传这两项
    # --------------------
    # 深度思考的判定逻辑（`generation._generate_final`）读的是
    # `client.model_name` 与 `client.supports_reasoning()`。而生产环境里
    # `agent._get_llm_client()` 返回的是**本类**（自动切换链），不是单个客户端。
    #
    # 实测后果（修复前）：
    #     探针 deep_thinking=True model='' client_model=None capable=False
    # 链没有这两个成员 → 继承基类默认（model_name 不存在 / 能力 False）
    # → 判定恒为 False → **即使用户配了推理模型、请求也确实发给了
    #   deepseek-reasoner，注入分支照样执行**，走不成思考流。
    #
    # 这类"包一层就丢了能力声明"的问题很隐蔽：单个客户端测起来全对，
    # 一旦走链（也就是**生产路径**）就失效。

    @property
    def model_name(self) -> str:
        """当前实际会用到的客户端模型名；取不到返回空串。"""
        for name, client in self._available_pair():
            model = getattr(client, "model_name", "")
            if model:
                return str(model)
        return ""

    def supports_reasoning(self) -> bool:
        """链上**任一**可用客户端支持思维链即返回 True。

        取"任一"而不是"主 provider"：主 provider 失败后会自动切到备用，
        备用若支持思维链，思考流仍然可用。若都不可用则返回 False，
        上层自然回退到提示词注入。
        """
        pair = self._available_pair()
        if not pair:
            return False
        for _name, client in pair:
            checker = getattr(client, "supports_reasoning", None)
            if callable(checker) and checker():
                return True
        return False

    # --- 内部 ---

    @property
    def trace_recorder(self):
        """LLM 调用追踪记录器（透传给链上每个客户端）。

        ★ 为什么链必须实现它
        -----------------
        **链本身不调用 LLM**——真正发请求的是链上被选中的那个客户端。
        若链不把这个属性透传下去，多 provider 部署下追踪会**完全失效**：
        `agent` 把记录器挂在链上，而链上的客户端拿不到，
        于是 `done` 帧里 `llm_trace.call_count` 恒为 0，
        看起来像"这轮没调用 LLM"。

        而单 provider（不配 `LLM_FALLBACK_PROVIDERS`）时追踪正常——
        这种"取决于是否配了兜底 provider"的差异极难排查。
        """
        return getattr(self, "_trace_recorder", None)

    @trace_recorder.setter
    def trace_recorder(self, recorder) -> None:
        """挂载/摘除记录器，并**转发给链上每个客户端**。

        ★ 为什么 setter 必须转发（实测缺口）
        --------------------------------
        第一版 setter 只存 `self._trace_recorder`，靠 `_client_for` 在
        "解析客户端时"挂上。**挂载**时这没问题（客户端惰性解析），
        但**摘除时就漏了**：已解析过的底层客户端仍持有旧记录器。

        后果有两个，都不报错：
            ① **内存泄漏**：记录器随请求生灭，却被客户端长期持有
            ② 若某条路径绕过链直接调底层客户端，调用会被记进
               **已经结束的那个请求**的记录器里

        实测由 `test_recorder_cleared_on_detach` 抓到。

        故这里显式遍历链上所有 provider 转发。`_client_for` 对未解析的
        会按需创建——多创建一个客户端实例的代价可接受（它本来也会被
        `agent._provider_clients` 缓存），换来的是"摘除后不留旧引用"。
        """
        self._trace_recorder = recorder
        for name in list(self._chain):
            client = self._client_for(name)
            if client is None:
                continue
            try:
                client.trace_recorder = recorder
            except Exception:  # noqa: BLE001 - 观测是旁路，不能搞挂请求
                logger.debug("转发 trace_recorder 到 %s 失败", name, exc_info=True)

    def _client_for(self, name: str):
        """取客户端；解析失败返回 None（记日志，不中断整条链）。"""
        try:
            client = self._resolver(name)
        except Exception as exc:
            logger.warning("provider %s 客户端创建失败：%s", name, exc)
            return None
        # 记录器由 `trace_recorder` 的 setter 统一转发，这里**不再重复挂载**。
        #
        # ★ 为什么不能在这里也挂一次（实测踩到）
        # ----------------------------------
        # 第一版这里有 `if client.trace_recorder is None: client.trace_recorder = recorder`。
        # 那个 `is None` 守卫**恰好阻止了"摘除"**：detach 时 recorder 是 None，
        # 条件不成立，于是永远不覆盖 → 底层客户端留着旧记录器。
        # 表现为记录器泄漏 + 可能把调用记进已结束的请求。
        # 两处各挂一次还容易只改一处，故收归 setter 单点负责。
        return client

    def switch_thinking_model(self, model_name: str) -> "FallbackLLMClient":
        """把链上**每一个** provider 都切到思考模型，返回 self（便于链式调用）。

        ★ 为什么链也需要这个方法
        --------------------
        `agent.core._thinking_model()` 会先浅拷贝客户端再切换，以便"开了深度思考的
        请求"不影响"并发的普通请求"。若链对象没有本方法，那条路径会被
        `getattr(..., None)` 判为不支持，于是**一旦启用多 provider 自动切换，
        深度思考就永远退化成提示词注入**——而单 provider 时却能正常切换。
        这种"取决于是否配了兜底 provider"的差异极难排查。

        ★ 每个 provider 各切各的
        ----------------------
        DeepSeek 与智谱的推理模型是不同的名字（`DEEPSEEK_THINKING_MODEL` /
        `ZHIPU_THINKING_MODEL`）。故不能用一个全局模型名覆盖所有 provider，
        而是把 `model_name` 当作**该链自己的**思考模型名交给各 provider——
        实际生效的名字由调用方按 provider 解析后传入。

        实现方式：换掉 `_resolver`，让"取客户端"这一步顺带应用切换。
        `_resolver` 本身通常带缓存（见 `agent.core._build_llm_chain` 的
        `resolve`），故这里**不缓存**结果，避免把切换后的副本写进共享缓存。
        """
        if not model_name:
            return self

        base_resolver = self._resolver

        def thinking_resolver(name: str):
            client = base_resolver(name)
            switcher = getattr(client, "switch_thinking_model", None)
            if callable(switcher):
                client.switch_thinking_model(model_name)
            return client

        # 原地替换 resolver：调用方已经拷贝过本对象，故不影响共享链
        self._resolver = thinking_resolver
        logger.debug("provider 链已切换到思考模型 %s", model_name)
        return self

    def set_thinking(self, enabled: bool) -> "FallbackLLMClient":
        """在链上**每一个** provider 开启/关闭思考模式，返回 self。

        ★ 与 `switch_thinking_model` 的区别（两者都要有）
        -------------------------------------------
            `switch_thinking_model`  换**模型名**（确有独立推理模型的部署）
            `set_thinking`           开**请求参数** `{"thinking":{"type":"enabled"}}`
                                     （DeepSeek V4 之后的官方方式）

        链必须在两者上都透传，否则一旦启用多 provider 自动切换，
        深度思考就会静默退化——而单 provider 时正常。这类"取决于是否配了
        兜底 provider"的差异极难排查（同 `switch_thinking_model` 的说明）。

        `thinking_enabled()` 取"任一 provider 已开启"：链上真正生效的是
        当前可用的那家，任一家能开就说明本轮拿得到思维链。
        """
        base_resolver = self._resolver

        def thinking_resolver(name: str):
            client = base_resolver(name)
            setter = getattr(client, "set_thinking", None)
            if callable(setter):
                setter(bool(enabled))
            return client

        self._resolver = thinking_resolver
        # 同时记录在链自身，供 `thinking_enabled()` 在**尚未解析客户端时**也能回答
        self._thinking_enabled = bool(enabled)
        logger.debug("provider 链思考模式 = %s", enabled)
        return self

    def thinking_enabled(self) -> bool:
        """链上是否已开启思考模式。

        先看链自身记录（`set_thinking` 会写），再问各 provider 客户端——
        后者覆盖"没在链上调用过 `set_thinking`、但底层客户端本就开着"的情况。
        """
        if getattr(self, "_thinking_enabled", False):
            return True
        for _name, client in self._available_pair():
            checker = getattr(client, "thinking_enabled", None)
            if callable(checker) and checker():
                return True
        return False

    def _in_cooldown(self, name: str) -> bool:
        """该 provider **所属厂商**是否仍在冷却期。

        ★ 为什么用厂商而不是 provider 名（局限 6）
        ------------------------------------
        `zhipu` 与 `zai` 是同一家的两个名字，注册表里是两条独立记录。
        按 name 记冷却时它们各存一份 → 拉黑一个不会拉黑另一个 →
        候选链写 `zhipu,zai` 会**两次撞同一个坏凭据**。
        按厂商记后，拉黑 `zhipu` 等于同时拉黑 `zai`。
        """
        key = _vendor_key(name)
        with self._lock:
            until = self._cooldown.get(key, 0.0)
        return time.monotonic() < until

    def _enter_cooldown(self, name: str, exc: BaseException | None = None) -> None:
        """把刚失败的 provider 拉黑。**冷却时长按失败类型分档**。

        ★ 为什么必须有拉黑（实测数据）
        ------------------------
        用无效密钥实测 failover 时，**一次问答里 deepseek 被重试了 3 次**
        （ReAct 决策轮 + 最终生成各调一次），每次都撞 401、每次都要等一次
        网络往返。当时 `provider_switch_count` 报 3——但那不是"切换了 3 次"，
        而是"同一个坏 provider 被试了 3 次"。

        ★ 为什么按类型分档（局限 3，实测缺陷）
        ------------------------------
        原先一律 300 秒，而**不同失败的可自愈速度差几个数量级**：

            401 密钥错 / 无权限    → 几分钟不会自愈   → 长冷却合理（600s）
            429 限速               → 几秒~几十秒后可用 → 300s **过长**（30s）
            网络超时 / 5xx         → 多为瞬时抖动      → 300s **过长**（30s）
            无法归类               → 取中间值（300s）

        把 429／抖动屏蔽 300 秒的后果：**本可 10 秒后成功的一次问答，
        走了低质量的备用 provider**。故交给 `failover_policy.cooldown_seconds_for`。

        `exc=None` 时按 `UNKNOWN` 档（保持向后兼容）。
        """
        key = _vendor_key(name)
        seconds = (
            UNKNOWN_COOLDOWN_SECONDS if exc is None else cooldown_seconds_for(exc)
        )
        with self._lock:
            self._cooldown[key] = time.monotonic() + seconds
            self._cooldown_hits += 1

    def _available_pair(self) -> list[tuple[str, Any]]:
        """返回 [(provider, client)] 中**可用**的那些（保序）。

        跳过的两类：
            ① `available()` 为 False（凭据不全 / 模型名为空）
            ② **处于失败冷却期**（刚失败过，一次问答内不会自愈）

        让它们参与重试只会浪费一次调用、一段等待和一条无意义的错误日志。
        """
        pairs = []
        for name in self._chain:
            if self._in_cooldown(name):
                logger.debug("provider %s 处于失败冷却期，跳过", name)
                continue
            client = self._client_for(name)
            if client is None:
                continue
            try:
                if client.available():
                    pairs.append((name, client))
            except Exception as exc:
                logger.warning("provider %s 的 available() 抛错：%s", name, exc)
        return pairs

    def _note_success(self, name: str, failed_before: int = 0) -> None:
        """记录成功产出的 provider。

        ★ `switch_count` 的口径：**失败过几个 provider**，不是"重试了几次"
        ------------------------------------------------------------------
        第一版写成"若 `last_used_provider` 已存在且不等于本次的 name，则 +1"。
        实测**从不计数**：首次成功时 `last_used_provider` 是空的，
        于是条件不成立；而恰恰"第一次就切换"才是最常见的形态
        （主 provider 挂了、兜底接管）。

        第二版用 `failed_before=index`（链里第几个候选成功）。这个数**能**反映
        切换，但会把"同一个坏 provider 被重试 N 次"也算成 N 次切换——
        实测一次问答里报出 `switch_count=3`，而真实情况只是
        "deepseek 坏了、zhipu 接管"这**一件事**。
        现在有了冷却期，同一个 provider 在一次问答内只会失败一次，
        故 `failed_before` 与"失败过的 provider 数"一致，语义才对得上。

        含义：0 = 主 provider 直接成功；≥1 = 有 provider 失败后由兜底接管。
        """
        with self._lock:
            if failed_before > 0:
                self.switch_count += failed_before
            self.last_used_provider = name

    def _note_failure(self, name: str, exc: BaseException) -> None:
        with self._lock:
            self.last_error.append((name, f"{type(exc).__name__}: {exc}"))
        # 拉黑刚失败的 provider：一次问答内不再重试它（见 `_enter_cooldown`）
        #
        # ★ 必须把 exc 传进去：冷却时长按**失败类型**分档
        #   （401 → 600s、429/抖动 → 30s）。不传就退化成"一律 300s"，
        #   那正是局限 3 记的缺陷。
        self._enter_cooldown(name, exc)

    # --- BaseLLMClient 契约 ---

    def available(self) -> bool:
        """链里**有没有任一**可用。

        若只看第一个 provider，会出现"第一个没配密钥、第二个配了"
        却被误判为不可用 → 明明能答却降级成"返回知识库原文"。
        """
        return bool(self._available_pair())

    def chat_stream(self, messages: list[dict[str, Any]], **kwargs: Any) -> Iterator[str]:
        """流式生成。**仅在尚未产出内容时**允许切换 provider。"""
        pairs = self._available_pair()
        if not pairs:
            raise RuntimeError(
                f"没有可用的 LLM provider（候选链：{', '.join(self._chain) or '空'}）"
            )

        last_exc: BaseException | None = None
        for index, (name, client) in enumerate(pairs):
            emitted = False
            try:
                for chunk in client.chat_stream(messages, **kwargs):
                    emitted = True
                    yield chunk
                self._note_success(name, failed_before=index)
                return
            except Exception as exc:
                self._note_failure(name, exc)
                last_exc = exc
                if emitted:
                    # ★ 已经吐过正文：切换会造成两段拼接口径不一致的回答，
                    #   比直接失败更糟。交给调用方走既有降级路径。
                    logger.error(
                        "provider %s 在流式输出中途失败（已吐内容，不再切换）：%s",
                        name, exc,
                    )
                    raise
                remaining = [n for n, _ in pairs[index + 1:]]
                logger.warning(
                    "provider %s 失败，%s：%s",
                    name,
                    f"切换到 {remaining[0]}" if remaining else "无更多候选",
                    exc,
                )
        if last_exc is not None:
            raise last_exc

    def chat_raw(self, messages: list[dict[str, Any]], tools: list | None = None,
                 **kwargs: Any) -> str:
        """非流式调用（ReAct 决策轮）。无"部分输出"问题，可自由切换。"""
        pairs = self._available_pair()
        if not pairs:
            raise RuntimeError(
                f"没有可用的 LLM provider（候选链：{', '.join(self._chain) or '空'}）"
            )
        last_exc: BaseException | None = None
        for index, (name, client) in enumerate(pairs):
            try:
                result = client.chat_raw(messages, tools=tools, **kwargs)
                self._note_success(name, failed_before=index)
                return result
            except Exception as exc:
                self._note_failure(name, exc)
                last_exc = exc
                remaining = [n for n, _ in pairs[index + 1:]]
                logger.warning(
                    "provider %s 的 chat_raw 失败，%s：%s",
                    name,
                    f"切换到 {remaining[0]}" if remaining else "无更多候选",
                    exc,
                )
        if last_exc is not None:
            raise last_exc

    def chat_stream_thinking(self, messages: list[dict[str, Any]],
                             **kwargs: Any) -> Iterator[dict[str, str]]:
        """深度思考流。与 `chat_stream` 同样的"已产出则不切换"约束。"""
        pairs = self._available_pair()
        if not pairs:
            raise RuntimeError(
                f"没有可用的 LLM provider（候选链：{', '.join(self._chain) or '空'}）"
            )
        last_exc: BaseException | None = None
        for index, (name, client) in enumerate(pairs):
            emitted = False
            try:
                for event in client.chat_stream_thinking(messages, **kwargs):
                    emitted = True
                    yield event
                self._note_success(name, failed_before=index)
                return
            except Exception as exc:
                self._note_failure(name, exc)
                last_exc = exc
                if emitted:
                    logger.error(
                        "provider %s 在思考流输出中途失败（已吐内容，不再切换）：%s",
                        name, exc,
                    )
                    raise
                remaining = [n for n, _ in pairs[index + 1:]]
                logger.warning(
                    "provider %s 的思考流失败，%s：%s",
                    name,
                    f"切换到 {remaining[0]}" if remaining else "无更多候选",
                    exc,
                )
        if last_exc is not None:
            raise last_exc


def build_fallback_client(primary: str, resolver) -> FallbackLLMClient:
    """便捷构造：按 `primary` 与环境变量拼出候选链。"""
    return FallbackLLMClient(parse_fallback_chain(primary), resolver)

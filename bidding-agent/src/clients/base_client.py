"""LLM 客户端基类与共享约定。

本模块定义三件事：
    1. `BaseLLMClient`         —— 所有 provider 客户端必须实现的接口契约
    2. `UnavailableLLMClient`  —— 配置缺失时的空实现，保证链路能平滑降级
    3. 共享工具               —— SSE 解析、消息校验、增量提取

★ 为什么接口是**同步**的（重要，改动前先读）
------------------------------------------
项目里曾并存过一套 async 客户端骨架（`base.py` + `deepseek.py` + `zhipu.py` +
`openai_compatible.py`），但它们**从未被生产链路使用**，最终作为死代码删除。
原因很直接——生产链路的调用形状是同步的：

    pipeline.py    `for chunk in self.llm.chat_stream(messages)`
    react_loop.py  `client.chat_raw(conversation, tools=tool_schemas)`
    generation.py  `client.chat_stream_thinking(messages)`

这些调用点分布在 agent / rag / services / eval 共 6 处，且都在同步函数里。
把客户端改成 async 会连带要求整条链路 async 化（FastAPI 的 SSE 生成器、
工具层的线程池、eval 脚本的循环），改动面远超"重构客户端"本身，
而收益仅是"形式统一"。故此处明确选择同步接口。

★ 为什么不直接用厂商 SDK
----------------------
原先 DeepSeek 走 `openai` SDK、智谱走 `zai` SDK。改为原生 HTTP 的理由：

    1. **依赖更少**：少两个第三方包，且它们的版本漂移会直接影响调用（本项目
       已因 `openai` SDK 的行为差异踩过坑——`n` 参数、`trust_env` 继承代理）
    2. **出站策略统一**：复用 `src/http_client.py` 的全局 httpx 客户端，
       它带 `trust_env=False`（见该模块的长注释：继承系统代理会导致
       WinError 10061 且**只在部分机器复现**）。SDK 各自维护连接池，
       这个策略管不到它们
    3. **行为可控**：SSE 分帧、超时、重试都在自己手里，出问题能一眼看到原始报文

代价是要自己处理 SSE 分帧与增量提取——那正是本模块 `iter_sse_json` 与
`extract_delta` 的职责，两个 provider 共用，不需要各写一遍。
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from src.logging_config import get_logger

logger = get_logger(__name__)


class BaseLLMClient:
    """LLM 客户端的接口契约。

    四个方法构成完整能力集，**缺一个就会在某条路径上炸**（历史上都踩过）：

        available()               pipeline 据此决定要不要降级为"直接返回知识库原文"
        chat_stream()             用户可见的流式回答
        chat_raw()                ReAct 决策轮（带 tools，需要完整响应而非流式）
        chat_stream_thinking()    深度思考模式（要单独产出思维链）

    实现类必须全部实现——`UnavailableLLMClient` 里那几段"Bug 修复"注释记录了
    漏实现其中某一个的后果。
    """

    #: provider 名（如 "deepseek" / "zhipu"），日志与前端展示用
    provider: str = "unknown"

    #: 最近一次 `chat_raw` 产出的思维链（普通模式为空串）。
    #:
    #: 用途：思考模式 + `tools` 的请求**必须回传历史 `reasoning_content`**，
    #: 否则 API 返回 400（DeepSeek 官方明确要求，见「思考模式」文档的工具调用节）。
    #: 返回值只能是字符串，故用属性把思维链带出来。**每轮覆盖**。
    last_reasoning: str = ""

    #: 当前是否开启思考模式（由调用方按请求设置，见 `set_thinking`）。
    _thinking_enabled: bool = False

    #: LLM 调用追踪记录器。**由请求方按请求挂载**（见 `src/monitoring_trace.py`）。
    #:
    #: ★ 为什么不在客户端内部自己存调用历史
    #: --------------------------------
    #: 客户端实例在 `agent._provider_clients` 里被**跨请求缓存复用**。
    #: 若把调用历史存在客户端上：请求 A 的记录会被请求 B 追加，
    #: 且无限增长（内存泄漏）。记录器随请求生灭，天然没有这两个问题。
    #:
    #: 未挂载（None）时所有追踪代码静默跳过——零开销、不改既有行为。
    trace_recorder = None

    def _trace(self, call) -> None:
        """把一条调用记录交给记录器。

        **永不抛异常**：观测代码不能搞挂业务请求——与 `monitoring.safe_record`
        同一条原则。真出问题只记一条 debug 日志。
        """
        recorder = getattr(self, "trace_recorder", None)
        if recorder is None:
            return
        try:
            recorder.record(call)
        except Exception:  # noqa: BLE001
            logger.debug("LLM 调用记录失败（不影响请求）", exc_info=True)

    def set_thinking(self, enabled: bool) -> "BaseLLMClient":
        """开关思考模式，返回 self（便于链式调用）。

        ★ 为什么是"开关"而不是"换模型名"
        ---------------------------
        原本的设计是切到一个独立的推理模型名（如 `deepseek-reasoner`）。
        但 DeepSeek 已于 2026-04-24 发布 V4 并**取消了独立推理模型名**：
        现在只有 `deepseek-flash` / `deepseek-v4-pro`，两个模型都由
        **请求参数** `{"thinking": {"type": "enabled"}}` 控制思考模式
        （官方文档「思考模式」；旧名 `deepseek-reasoner` 将于
        2026-07-24 15:59 UTC 彻底退役）。

        故判定"本轮有没有真思维链"的依据从**模型名**变成**这个开关**。
        实测教训：迁移公告特别指出最常见的 bug 就是"忘记加 `thinking` 参数"，
        结果是**推理质量下降但没有任何报错**。

        默认实现只记录状态；真正把参数发出去由 `OpenAICompatClient._payload` 负责。
        """
        self._thinking_enabled = bool(enabled)
        return self

    def thinking_enabled(self) -> bool:
        """当前是否处于思考模式。"""
        return bool(getattr(self, "_thinking_enabled", False))

    def available(self) -> bool:
        """当前客户端是否可用（凭据齐全、模型名非空）。

        `False` 时上层会降级为"直接返回知识库原文"而不是报错——
        没配密钥不该让整个问答服务不可用。
        """
        raise NotImplementedError

    def chat_stream(self, messages: list[dict[str, Any]], **kwargs: Any) -> Iterator[str]:
        """流式生成回答，逐块吐出文本。

        实现约定：**异常向上抛**，由调用方（pipeline / generation）决定降级策略。
        在生成器里吞掉异常会让"流到一半断了"表现为"回答就这么短"。
        """
        raise NotImplementedError

    def chat_raw(
        self,
        messages: list[dict[str, Any]],
        tools: list | None = None,
        **kwargs: Any,
    ) -> str:
        """非流式调用，返回完整文本（ReAct 决策轮用）。

        传 `tools` 时要把模型的工具调用序列化成项目约定的文本格式
        （`{"tool_calls": [{"name": ..., "arguments": {...}}]}`），
        交给 `src/agent/tool_defense.py` 的 `_parse_tool_text` 解析——
        这样解析链路只有一处，各 provider 不必各写一套。
        """
        raise NotImplementedError

    #: 本客户端产出的思维链是否为**真实推理**（而非继承来的空壳实现）。
    #:
    #: ★ 为什么需要这个显式标志，而不是靠 `hasattr(client, "chat_stream_thinking")`
    #: ------------------------------------------------------------------
    #: 历史事故：`generation.py` 原来写的是
    #:
    #:     use_thinking = deep_thinking and hasattr(client, "chat_stream_thinking")
    #:     if deep_thinking and not use_thinking:
    #:         ...注入 DEEP_THINKING_INJECT...
    #:
    #: 而基类**实现了** `chat_stream_thinking`（只为让子类有统一接口），
    #: 于是 `hasattr` 对任何客户端都返回 True：
    #:
    #:     use_thinking 恒为 True  →  回退分支**永远不执行**  →  提示词注入从不发生
    #:
    #: 后果：**没有配推理模型时，深度思考开关打开后什么也没做**——
    #: 既不产出思维链（普通模型没有 reasoning_content），也不注入分析指令，
    #: 只是白传了一个更大的 max_tokens。用户看到"开关没效果"，且没有任何报错。
    #:
    #: `hasattr` 只能回答"有没有这个方法"，回答不了"这个方法能不能真的干活"。
    #: 故需要显式声明能力，见 `supports_reasoning()`。
    # ★ 注意：这里是**方法**而不是类属性
    # --------------------------------
    # 我第一版写成 `supports_reasoning: bool = False`（类属性）+ 同名方法，
    # 于是子类里 `supports_reasoning = True`（`OpenAICompatClient` 那样写）
    # 会把方法**整个遮蔽**掉，调用处 `client.supports_reasoning()` 就变成
    # "bool 不可调用" 的 TypeError。
    #
    # 属性与方法同名时，子类赋值会静默覆盖父类方法 —— 这类错误在定义处
    # 完全看不出来，只在调用时才炸。故只用方法，子类覆盖方法。
    def supports_reasoning(self) -> bool:
        """本客户端是否真的能产出思维链。

        默认 False（基类的 `chat_stream_thinking` 只是接口占位，产出空思维链）。
        `OpenAICompatClient` 覆盖为 True——它的实现会从
        `delta.reasoning_content` 取真实推理内容（DeepSeek / 智谱 / vLLM /
        Ollama 上的 R1 类模型都走这个字段）。

        **调用方应据此决定走思考流还是提示词注入**，不要用 `hasattr` 判断。**
        """
        return False

    def chat_stream_thinking(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> Iterator[dict[str, str]]:
        """深度思考流：产出 `{"type": "thinking"|"content", "content": 文本}`。

        非推理模型没有思维链字段，此时等价于 `chat_stream`（只产出 content），
        不影响正常回答。
        """
        raise NotImplementedError

    def switch_thinking_model(self, model_name: str) -> "BaseLLMClient":
        """切到"深度思考"用的推理模型，**返回客户端本身**以便链式调用。

        ★ 为什么基类给默认实现（而不是只定义 `raise NotImplementedError`）
        -------------------------------------------------------------
        历史事故：`src/agent/core.py` 的 `_thinking_model()` 写的是

            if model_name and hasattr(client, "switch_thinking_model"):
                client.switch_thinking_model(model_name)

        而**没有任何真实客户端实现过这个方法**——唯一实现它的是
        `tests/test_deep_thinking.py` 里的假客户端。于是：

            hasattr(...) 永远为 False
            → 配置了 `DEEPSEEK_THINKING_MODEL` 也**从不切换**
            → 深度思考永远只走"提示词注入"那条回退路径

        测试之所以通过，是因为它用自己造的假对象去测，把真实缺陷掩盖了。
        故这里在基类给出默认实现（OpenAI 兼容协议下切换模型就是改一个字段名），
        所有子类自动获得该能力，不再依赖"某个子类记得实现它"。

        ★ 默认行为：只改模型名
        --------------------
        OpenAI 兼容协议里推理模型与普通模型**只有模型名不同**，协议一致
        （思维链走 `delta.reasoning_content`，见 `extract_delta`）。故默认实现
        只替换 `_model`；确有额外差异的子类可覆盖本方法。

        ★ 调用方必须注意：本方法**原地修改**，不是返回新对象
        ------------------------------------------------
        客户端实例在 `agent.core._provider_clients` 里被缓存并**跨请求共享**
        （按链缓存）。原地改 `_model` 会让"开了深度思考的请求"把
        "没开的并发请求"也一起改成推理模型。
        故 `agent.core._thinking_model()` 会**先浅拷贝再切**，见那里的说明。
        本方法保持原地修改是为了让拷贝后的对象也能链式使用。
        """
        if model_name:
            self._model = model_name
        return self

    def __repr__(self) -> str:  # pragma: no cover - 仅调试用
        return f"<{type(self).__name__} provider={self.provider}>"


class UnavailableLLMClient(BaseLLMClient):
    """配置缺失时的空客户端，保证链路平滑降级到"直接返回知识库原文"。

    ★ 四个方法都必须实现，哪怕只是日志 + 空返回
    -----------------------------------------
    历史事故：本类最初只有 `available()` 与 `chat_stream()`，而
    `src/agent/react_loop.py` 的决策轮会直接调 `chat_raw()`，于是抛

        AttributeError: 'UnavailableLLMClient' object has no attribute 'chat_raw'

    结果是**设计好的优雅降级路径反而崩成 500**——"没配密钥"本该只是功能降级，
    却变成了服务不可用。故此处把四个方法都补齐。
    """

    def __init__(self, reason: str) -> None:
        self.reason = reason
        logger.warning("LLM 客户端不可用，将降级为知识库原文返回。原因: %s", reason)

    def available(self) -> bool:
        return False

    def chat_stream(self, messages: list[dict[str, Any]], **kwargs: Any) -> Iterator[str]:
        # 理论上 available() 为 False 时不会被调用；保留空实现以防调用方漏判
        logger.warning("LLM 不可用（%s），流式回答返回空", self.reason)
        yield ""

    def chat_raw(
        self,
        messages: list[dict[str, Any]],
        tools: list | None = None,
        **kwargs: Any,
    ) -> str:
        # 返回空串是**有意的**：ReAct 决策轮拿到空串即视为"不调用工具"，
        # 从而走进生成分支用已有资料作答，而不是让整轮问答失败
        logger.warning("LLM 不可用（%s），决策轮返回空串以触发降级", self.reason)
        return ""

    def chat_stream_thinking(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> Iterator[dict[str, str]]:
        logger.warning("LLM 不可用（%s），思考流返回空", self.reason)
        return
        yield  # pragma: no cover - 使其成为生成器函数


# ===========================================================================
# 共享工具（两个 provider 客户端都用）
# ===========================================================================


def iter_sse_json(response) -> Iterator[dict]:
    """把 SSE 响应逐行解析成 JSON 对象。

    为什么手写而不用现成库：需要控制的地方不多，但每一条都踩过坑——

        1. **`data: [DONE]` 是结束标记**，不是一条消息，直接 `json.loads` 会抛
        2. **空行与注释行**（`:` 开头的心跳）要跳过，某些网关会周期性发它们
        3. **一行解析失败不能让整流崩掉**：服务端偶发发不完整帧，
           跳过它比中断整轮回答好——用户看到的是回答略短，而不是报错
        4. 有些网关不按 SSE 规范加 `data: ` 前缀，或用了 `data:`（无空格），
           故前缀判断要宽松

    参数 `response` 是 httpx 的流式响应（`client.stream(...)` 的上下文产物）。
    """
    for raw_line in response.iter_lines():
        if not raw_line:
            continue
        line = raw_line.strip()
        if not line or line.startswith(":"):
            continue  # 空行 / 心跳注释
        if line.startswith("data:"):
            line = line[5:].lstrip()
        if not line or line == "[DONE]":
            continue
        try:
            payload = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            # 单行坏了不代表流坏了：记录并跳过
            logger.debug("跳过无法解析的 SSE 行：%s", line[:160])
            continue
        if isinstance(payload, dict):
            yield payload


def extract_delta(chunk: dict) -> tuple[str, str]:
    """从一条流式 chunk 里取出 `(思维链, 正文)`。

    两类字段都要认，因为不同模型/网关的命名不同：

        正文      `choices[0].delta.content`
        思维链    `choices[0].delta.reasoning_content`（DeepSeek reasoner 等）
                  `choices[0].delta.reasoning`         （部分智谱模型用这个）

    返回两个字符串（可能都是空串）。**不在这里判断"该不该输出思维链"**——
    那是调用方的策略（`chat_stream` 丢弃思维链，`chat_stream_thinking` 保留），
    提取器只负责把原始字段摊平。
    """
    try:
        choices = chunk.get("choices") or []
        if not choices:
            return "", ""
        delta = choices[0].get("delta") or {}
    except (AttributeError, IndexError, TypeError):
        return "", ""

    content = delta.get("content") or ""
    reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
    return str(reasoning), str(content)


def extract_message(chunk: dict) -> tuple[dict, str]:
    """从**非流式**响应里取出 `(message, finish_reason)`。

    非流式的形状是 `choices[0].message`（与流式的 `delta` 不同），
    这里把防御性取值集中一处，避免每个客户端各写一遍 try/except。
    """
    try:
        choices = chunk.get("choices") or []
        if not choices:
            return {}, ""
        choice = choices[0] or {}
        message = choice.get("message") or {}
        finish = choice.get("finish_reason") or ""
        return (message if isinstance(message, dict) else {}), str(finish)
    except (AttributeError, IndexError, TypeError):
        return {}, ""


def _tool_names_of(text: str) -> list[str]:
    """从项目约定的工具调用文本里取工具名（取不到返回空列表）。

    只做**轻量提取**，不复用 `tool_defense` 的完整解析器：
    后者要处理畸形 JSON / XML / DSML 等一堆形态，而这里只是给观测记个名字，
    不值得为它引入依赖与开销（观测代码不该成为业务链路的负担）。
    """
    if not text or "tool_calls" not in text:
        return []
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return []
    items = obj.get("tool_calls") if isinstance(obj, dict) else None
    if not isinstance(items, list):
        return []
    return [str(i["name"]) for i in items if isinstance(i, dict) and i.get("name")]


def _capture_limit() -> int:
    """正文采集上限（字符）。**0 表示不采集**——即默认状态。

    这是 `LANGFUSE_CAPTURE_CONTENT` 在客户端侧的唯一体现：
    默认关时返回 0，三次调用里对应字段都是空串，**零额外开销**
    （不做字符串拼接、不留引用）。

    放在函数里惰性 import：`langfuse_tracing` 是可选能力模块，
    客户端不该在 import 期就把它拖进来。取不到一律当"不采集"——
    观测配置读失败绝不能影响 LLM 调用本身。
    """
    try:
        from src.langfuse_tracing import capture_enabled, capture_limit

        return capture_limit() if capture_enabled() else 0
    except Exception:  # noqa: BLE001
        return 0


def format_tool_calls(tool_calls: list) -> str:
    """把 OpenAI 格式的 tool_calls 序列化成项目约定的文本。

    输出形状（**必须与 `tool_defense._parse_tool_text` 的期望一致**）：

        {"tool_calls": [{"name": "search_knowledge_base", "arguments": {...}}]}

    为什么要转成文本而不是直接把对象往上抛：ReAct 循环的解析链路
    （`_parse_tool_text`）已经处理了"模型把工具调用写进正文"的各种畸形情况，
    让它统一吃文本，各 provider 就不必各写一套工具调用解析。
    """
    calls = []
    for call in tool_calls or []:
        if not isinstance(call, dict):
            continue
        function = call.get("function") or {}
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if not name:
            continue
        calls.append({"name": name, "arguments": _loads_arguments(function.get("arguments"))})
    if not calls:
        return ""
    logger.info("决策轮返回工具调用: %s", [c["name"] for c in calls])
    return json.dumps({"tool_calls": calls}, ensure_ascii=False)


def _loads_arguments(raw: Any) -> dict:
    """把工具调用的 `arguments` 归一为 dict。

    不同厂商/版本给的东西不一样：dict、JSON 字符串、甚至畸形串。
    统一容错，避免一处格式差异让整轮工具调用崩掉。
    """
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, ValueError):
            return {}
    return {}


# ===========================================================================
# OpenAI 兼容协议的共用实现
# ===========================================================================


class OpenAICompatClient(BaseLLMClient):
    """OpenAI 兼容协议的共用实现，供各 provider 继承。

    DeepSeek 与智谱的对话接口都是 OpenAI 协议的子集
    （`POST {base_url}/chat/completions`），差异只在 base_url、鉴权头与少量参数。
    把这部分收在一处，provider 子类只需给出 `provider` / `base_url` / `headers`，
    以及个别差异点（如智谱不接受 `parallel_tool_calls`）。

    ★ 复用 `src/http_client.py` 的全局 httpx 客户端
    ----------------------------------------------
    不自己 new 一个 `httpx.Client`：共享客户端带 `trust_env=False`
    （见该模块的长注释——继承系统代理会导致 WinError 10061 且**只在部分机器复现**）。
    各客户端自建连接池的话，这个出站策略就管不到 LLM 请求了。

    超时按请求覆盖：共享客户端读超时是 60s（面向检索接口），
    而 LLM 生成一篇长回答可能更久，故本类默认 300s，通过 `timeout=` 传给单次请求。
    """

    #: 子类覆盖：provider 名
    provider = "openai-compat"
    #: 子类覆盖：API 根地址（**不含** /chat/completions）
    base_url = ""
    #: 子类覆盖：是否发送 parallel_tool_calls（智谱不接受该字段）
    supports_parallel_tool_calls = True

    def supports_reasoning(self) -> bool:
        """本实现会从 `delta.reasoning_content` 取真实思维链，故返回 True。

        注意这声明的是"**有能力**解析思维链"，而不是"当前模型一定产出思维链"——
        普通模型（如 deepseek-chat）该字段为空，此时思考流只产出正文，
        行为与普通流一致。是否**真的**走思考流由调用方结合"有没有配思考模型"
        一起判断（见 `generation._generate_final`）。
        """
        return True

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        base_url: str | None = None,
        temperature: float = 0.1,
        timeout: float = 300.0,
        provider: str | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = (base_url or self.base_url or "").rstrip("/")
        self._temperature = temperature
        self._timeout = timeout
        # 实例级 provider 名覆盖类属性：通用客户端要同时服务 ollama / vllm / qwen 等，
        # 靠子类逐个覆盖不现实（会变成一堆只有一个类属性的空子类）
        if provider:
            self.provider = provider

    # ---- 供子类覆盖的差异点 ----

    def _headers(self) -> dict[str, str]:
        """鉴权与内容类型。子类可覆盖（如智谱用 `Bearer` 但地址不同）。"""
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    @property
    def model_name(self) -> str:
        return self._model

    # ---- 契约实现 ----

    def available(self) -> bool:
        return bool(self._api_key) and bool(self._model) and bool(self._base_url)

    def _payload(
        self,
        messages: list[dict[str, Any]],
        *,
        stream: bool,
        tools: list | None = None,
        **kwargs: Any,
    ) -> dict:
        """组装请求体。子类可覆盖以增删字段。

        ★ 思考模式在这里落地（`temperature` 与 `thinking` 二选一）
        ---------------------------------------------------
        开启思考模式时发送 `{"thinking": {"type": "enabled"}}`，并**去掉
        `temperature`**。

        为什么必须去掉而不是留着：DeepSeek 官方文档明确
        「思考模式**不支持** `temperature`、`presence_penalty`、
        `frequency_penalty`；设置这些参数不会报错，但也不会生效」。
        留着它不会报错，只会让代码读起来以为它在起作用——本项目
        `LLM_TEMPERATURE=0.1` 是刻意调低的（招投标回答要求稳定），
        若以为思考模式下它仍生效，就会对输出稳定性做出错误判断。

        ★★ 关闭时必须**显式**发 `{"type": "disabled"}`
        -----------------------------------------
        真调实测（`deepseek-flash`，2026-10）：

            不发 thinking      → reasoning = 92 字符   ← 默认就是**开**的！
            thinking=disabled  → reasoning =  0 字符   ← 显式关才真关
            thinking=enabled   → reasoning = 89 字符

        官方文档也写明「思考模式**默认打开**，且 effort 默认为 high」。
        所以"不发参数"不等于"关闭"——**不显式关闭，每次普通问答都在为
        思维链付延迟与 token 成本**，而本项目此前的普通请求正是这么发的。

        更糟的是语义失真：用户在界面上看到「深度思考」是关的，实际上
        服务端一直在思考。这类"开关是假的"问题比没有开关更误导。

        注：`thinking` 走 **body 顶层**而不是包在 `extra_body` 里——
        `extra_body` 是 OpenAI **Python SDK** 的参数名，SDK 会把它摊平进
        请求体。本项目用原生 HTTP（`_payload` 产出的就是最终 body），
        故直接放顶层即可；照抄 SDK 写法反而会变成
        `{"extra_body": {...}}`，服务端不认。
        """
        body: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": stream,
        }
        if self.thinking_enabled():
            body["thinking"] = {"type": "enabled"}
        else:
            body["thinking"] = {"type": "disabled"}
            body["temperature"] = self._temperature
        if tools:
            body["tools"] = tools
            if self.supports_parallel_tool_calls:
                # 决策轮一次只调一个工具，降低工具风暴风险
                body["parallel_tool_calls"] = False
        body.update(kwargs)
        return body

    def _endpoint(self) -> str:
        return f"{self._base_url}/chat/completions"

    def _raise_for_status(self, response) -> None:
        """非 2xx 时抛出带响应体的异常。

        ★ 必须带上响应体：厂商的错误信息（如"model not found"、
        "Invalid n value"）只在 body 里，只看状态码会得到一句无用的 400，
        排查时得再复现一次才能看到原因。本项目已因忽略错误体踩过坑。
        """
        if response.status_code < 400:
            return
        try:
            detail = response.read().decode("utf-8", "replace")[:500]
        except Exception:  # pragma: no cover - 读取失败不影响抛错
            detail = "<响应体不可读>"
        raise RuntimeError(
            f"{self.provider} 接口返回 {response.status_code}：{detail}"
        )

    def chat_stream(self, messages: list[dict[str, Any]], **kwargs: Any) -> Iterator[str]:
        """流式生成回答，只产出正文（丢弃思维链）。"""
        if not self.available():
            raise RuntimeError(f"{self.provider} 客户端不可用（缺少 api_key / model）")

        from src.http_client import get_http_client

        client = get_http_client()
        body = self._payload(messages, stream=True, **kwargs)
        # ★ 必须显式要求回传 usage
        #
        # 实测：不加 `stream_options.include_usage` 时流式 chunk 里
        # **完全没有 usage**（非流式则顶层一直有）。加了之后最后一个 chunk
        # 会带 usage，且此时 `choices` 通常是空数组——故解析时不能
        # "没看到 choices 就跳过"。
        body.setdefault("stream_options", {"include_usage": True})

        import time as _time

        from src.monitoring_trace import LLMCall, parse_usage, render_messages

        # 正文采集（默认关）：关着时 cap=0，下面两个变量一直是空的，零开销
        cap = _capture_limit()
        captured: list[str] = []
        captured_len = 0

        t0 = _time.monotonic()
        content_chars = 0
        usage_payload: dict | None = None
        error_text = ""
        ok = True
        try:
            with client.stream(
                "POST", self._endpoint(), json=body,
                headers=self._headers(), timeout=self._timeout,
            ) as response:
                self._raise_for_status(response)
                for chunk in iter_sse_json(response):
                    if isinstance(chunk, dict) and chunk.get("usage"):
                        usage_payload = chunk
                    _, content = extract_delta(chunk)
                    if content:
                        content_chars += len(content)
                        if cap and captured_len < cap:
                            captured.append(content)
                            captured_len += len(content)
                        yield content
        except Exception as exc:  # noqa: BLE001
            ok = False
            error_text = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            # finally 保证**异常路径也记上**：失败次数是 provider 稳定性的
            # 直接信号，只在成功分支记会漏掉最重要的那种情况。
            self._trace(LLMCall(
                kind="stream",
                provider=self.provider,
                model=getattr(self, "model_name", "") or "",
                elapsed_ms=int((_time.monotonic() - t0) * 1000),
                ok=ok,
                error=error_text[:300],
                content_chars=content_chars,
                **parse_usage(usage_payload),
                # 采集时只留**前 cap 个字符**（流式下拿不到总长，只能顺序累积）；
                # `content_chars` 始终是真实总长，两者对照即可看出被截断
                prompt_text=render_messages(messages, cap) if cap else "",
                response_text="".join(captured) if cap else "",
            ))

    def chat_stream_thinking(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> Iterator[dict[str, str]]:
        """深度思考流：思维链与正文分别产出。

        非推理模型没有思维链字段，此时退化为只产出 content，不影响正常回答。
        """
        if not self.available():
            raise RuntimeError(f"{self.provider} 客户端不可用（缺少 api_key / model）")

        from src.http_client import get_http_client

        client = get_http_client()
        body = self._payload(messages, stream=True, **kwargs)
        body.setdefault("stream_options", {"include_usage": True})

        import time as _time

        from src.monitoring_trace import LLMCall, parse_usage, render_messages

        cap = _capture_limit()
        captured: list[str] = []
        captured_len = 0

        t0 = _time.monotonic()
        reasoning_chars = 0
        content_chars = 0
        usage_payload: dict | None = None
        error_text = ""
        ok = True
        try:
            with client.stream(
                "POST", self._endpoint(), json=body,
                headers=self._headers(), timeout=self._timeout,
            ) as response:
                self._raise_for_status(response)
                for chunk in iter_sse_json(response):
                    if isinstance(chunk, dict) and chunk.get("usage"):
                        usage_payload = chunk
                    reasoning, content = extract_delta(chunk)
                    if reasoning:
                        reasoning_chars += len(reasoning)
                        yield {"type": "thinking", "content": reasoning}
                    if content:
                        content_chars += len(content)
                        if cap and captured_len < cap:
                            captured.append(content)
                            captured_len += len(content)
                        yield {"type": "content", "content": content}
        except Exception as exc:  # noqa: BLE001
            ok = False
            error_text = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            # 思考流单独记 reasoning_chars：用于区分"慢是因为在想"
            # 还是"慢是因为输出长"——两者的优化方向完全不同。
            self._trace(LLMCall(
                kind="thinking",
                provider=self.provider,
                model=getattr(self, "model_name", "") or "",
                elapsed_ms=int((_time.monotonic() - t0) * 1000),
                ok=ok,
                error=error_text[:300],
                reasoning_chars=reasoning_chars,
                content_chars=content_chars,
                **parse_usage(usage_payload),
                # 只采正文，不采思维链：思维链长度已单独记录，正文才是"答了什么"
                prompt_text=render_messages(messages, cap) if cap else "",
                response_text="".join(captured) if cap else "",
            ))

    def chat_raw(
        self,
        messages: list[dict[str, Any]],
        tools: list | None = None,
        **kwargs: Any,
    ) -> str:
        """非流式调用，返回完整文本；有工具调用时返回项目约定的文本格式。

        ★ 副作用：把本轮思维链存到 `self.last_reasoning`
        --------------------------------------------
        返回值只能是字符串（调用方 `react_loop` 直接拿它解析工具调用），
        但**思考模式下思维链必须回传给 API**，否则带 `tools` 的请求返回 400。
        故用实例属性把它带出来，由 `react_loop` 在追加 assistant 消息时使用。

        为什么不做成返回元组：会改变所有调用方与测试替身的签名，
        代价远大于一个明确的副作用。属性是**每轮覆盖**的，故必须在
        `chat_raw` 返回后立刻读取（`react_loop` 正是这么做的）。
        """
        if not self.available():
            raise RuntimeError(f"{self.provider} 客户端不可用（缺少 api_key / model）")

        from src.http_client import get_http_client

        client = get_http_client()
        body = self._payload(messages, stream=False, tools=tools, **kwargs)

        # 计时从**发请求之前**开始：决策轮的耗时主要就是这一次网络往返 +
        # 模型思考，从解析后才开始计会漏掉整个往返。
        import time as _time

        from src.monitoring_trace import LLMCall, parse_usage, render_messages

        cap = _capture_limit()

        t0 = _time.monotonic()
        content = ""
        error_text = ""
        ok = True
        payload: dict = {}
        try:
            response = client.post(
                self._endpoint(), json=body,
                headers=self._headers(), timeout=self._timeout,
            )
            self._raise_for_status(response)
            try:
                payload = response.json()
            except Exception as exc:
                raise RuntimeError(f"{self.provider} 返回体不是合法 JSON：{exc}") from exc

            message, _finish = extract_message(payload)
            # 记录思维链（普通模式下为空串，行为不变）
            self.last_reasoning = str(message.get("reasoning_content") or "")
            tool_calls = message.get("tool_calls")
            formatted = format_tool_calls(tool_calls) if tool_calls else ""
            content = formatted or str(message.get("content") or "")
            return content
        except Exception as exc:  # noqa: BLE001
            ok = False
            error_text = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            # 工具名从序列化后的文本里取（`format_tool_calls` 产出的就是
            # 项目约定的 `{"tool_calls":[{"name":...}]}`）。记下来是为了
            # 排查"哪一轮在调工具、调了几次"，不必再翻日志。
            tools_called = _tool_names_of(content)
            self._trace(LLMCall(
                kind="decision",
                provider=self.provider,
                model=getattr(self, "model_name", "") or "",
                elapsed_ms=int((_time.monotonic() - t0) * 1000),
                ok=ok,
                error=error_text[:300],
                content_chars=len(content),
                reasoning_chars=len(getattr(self, "last_reasoning", "") or ""),
                tools=tools_called,
                **parse_usage(payload),
                prompt_text=render_messages(messages, cap) if cap else "",
                response_text=content if cap else "",
            ))


__all__ = [
    "BaseLLMClient",
    "OpenAICompatClient",
    "UnavailableLLMClient",
    "extract_delta",
    "extract_message",
    "format_tool_calls",
    "iter_sse_json",
]

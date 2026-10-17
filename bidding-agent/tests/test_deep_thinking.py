"""深度思考测试：思考模型切换 / max_tokens 预算 / 事件流 / 提示词注入回退。

★ 本文件曾经"全绿但功能是坏的"
============================
历史缺陷：`agent.core._thinking_model` 里写的是

    if model_name and hasattr(client, "switch_thinking_model"):

而**没有任何真实客户端实现过该方法**——唯一实现它的是本文件的假客户端。
于是 `hasattr` 永远为 False，配了 `DEEPSEEK_THINKING_MODEL` 也从不切换，
深度思考永远只走提示词注入。**测试却全绿**，因为假对象自己实现了该方法。

故本文件现在分两层：
    ① 行为层：用假客户端测事件流与预算（快，验证编排逻辑）
    ② 契约层：用**真实客户端**测 `switch_thinking_model` 确实存在且生效
              （TestRealClientContract）——这层才是能发现上述缺陷的那层

教训：**测试替身必须与真实实现有共同的接口契约**。用假对象测
"对象是否具备某能力"，等于在测自己刚写下的那行代码。
"""

import pytest

from src.agent import BiddingAgent
from src.agent.constants import MAX_TOKENS_DEEP, THINKING_MODEL_ENV
from src.agent.prompts import DEEP_THINKING_INJECT


class ScriptedLLM:
    def __init__(self, raw_responses=(), stream_chunks=()):
        self.raw_responses = list(raw_responses)
        self.stream_chunks = list(stream_chunks)
        self.stream_messages = []
        self.stream_kwargs = []
        self.raw_calls = 0

    def chat_raw(self, messages, tools=None):
        self.raw_calls += 1
        item = self.raw_responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def chat_stream(self, messages, **kwargs):
        """★ 必须接受 `**kwargs`（真实客户端签名如此）。

        第一版写成 `chat_stream(self, messages)`，于是深度思考回退路径传
        `max_tokens=` 时直接 `TypeError`。这恰好**反向证明了那次修复有效**——
        原实现根本不传该参数，所以假客户端不需要收它。
        替身签名与真实契约不一致，会把"实现改进"报成"测试失败"。
        """
        self.stream_messages.append(messages)
        self.stream_kwargs.append(kwargs)
        yield from self.stream_chunks


class ThinkingLLM(ScriptedLLM):
    """支持深度思考的客户端：chat_stream_thinking 产出 (kind, 文本)，kind ∈ thinking/content。

    ★ 必须声明 `model_name` 与 `supports_reasoning()`
    --------------------------------------------
    深度思考的判定要求"配置的思考模型名 == 实际模型名 **且** 客户端支持思维链"。
    替身若不声明这两项，判定就不通过，会走提示词注入而不是思考流——
    第一版正是如此，于是 `test_deep_thinking_emits_thinking_then_tokens`
    失败（`thinking` 事件为空）。

    这也说明判定条件确实收紧了：**光有 `chat_stream_thinking` 方法不够**，
    还要能证明"本轮真的会用推理模型"。替身必须与真实契约一致。
    """

    def __init__(self, model: str = "deepseek-flash"):
        super().__init__(raw_responses=["直接回答"])
        self.thinking_kwargs = []
        self._model = model
        self._thinking = False

    @property
    def model_name(self) -> str:
        return self._model

    def supports_reasoning(self) -> bool:
        return True

    def set_thinking(self, enabled: bool):
        """真实客户端由基类提供此方法，替身照做以保持一致契约。"""
        self._thinking = bool(enabled)
        return self

    def thinking_enabled(self) -> bool:
        return self._thinking

    def chat_stream_thinking(self, messages, max_tokens=None):
        self.thinking_kwargs.append({"max_tokens": max_tokens})
        self.stream_messages.append(messages)
        yield "thinking", "推理过程片段"
        yield "content", "最终答案"


def _events_by_type(events, event_type):
    return [e for e in events if e[0] == event_type]


def test_deep_thinking_emits_thinking_then_tokens(monkeypatch):
    # ★ 2026-04 之后是**参数开关**而非模型名（见 docs/52）
    monkeypatch.setenv("DEEPSEEK_THINKING_ENABLED", "true")
    client = ThinkingLLM()
    agent = BiddingAgent(llm_client=client)

    events = list(agent._chat_events("复杂问题", deep_thinking_enabled=True))

    thinking = _events_by_type(events, "thinking")
    tokens = _events_by_type(events, "token")
    assert [e[1] for e in thinking] == ["推理过程片段"]
    assert "".join(e[1] for e in tokens) == "最终答案"
    # 思考增量先于正文增量
    assert events.index(thinking[0]) < events.index(tokens[0])
    assert any(e[0] == "done" for e in events)

    # 思考输出预算：reasoning 计入 max_tokens
    assert client.thinking_kwargs == [{"max_tokens": MAX_TOKENS_DEEP}]


def test_thinking_stream_not_used_when_model_not_switched(monkeypatch):
    """★ 客户端有能力、但**没开**思考模式时不该走思考流。

    这是"光看 `hasattr(chat_stream_thinking)`"会漏掉的情况：方法存在、
    能力也声明为 True，但本轮没开思考模式（不发 `thinking` 参数，
    服务端就不产出 `reasoning_content`）。此时走思考流等于什么都不做——
    既没有思维链，也不会注入分析指令。故必须落到注入分支。

    ★ 必须显式清掉环境变量
    ------------------
    本项目的 `.env` 里可能写着 `DEEPSEEK_THINKING_ENABLED=true`（那是
    "**允许**使用真思考模式"的能力开关）。若不清掉，测试会读到它而启用，
    于是断言失败——实测踩过：全量回归时这个用例失败，单独跑却通过，
    因为 `.env` 被加载了。**测试不应依赖开发者的 `.env` 内容。**
    """
    from src.agent import constants as C

    monkeypatch.delenv(C.THINKING_ENABLED_ENV["deepseek"], raising=False)
    monkeypatch.delenv(C.THINKING_MODEL_ENV["deepseek"], raising=False)
    client = ThinkingLLM(model="deepseek-chat")
    agent = BiddingAgent(llm_client=client)

    events = list(agent._chat_events("复杂问题", deep_thinking_enabled=True))

    assert _events_by_type(events, "thinking") == [], "不该产出思维链事件"
    assert client.thinking_kwargs == [], "不该走思考流"
    system = client.stream_messages[0][0]["content"]
    assert DEEP_THINKING_INJECT in system, "应回退到提示词注入"


def test_deep_thinking_status_sequence(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_THINKING_MODEL", "deepseek-reasoner")
    client = ThinkingLLM()
    agent = BiddingAgent(llm_client=client)

    events = list(agent._chat_events("复杂问题", deep_thinking_enabled=True))
    statuses = [e[1] for e in _events_by_type(events, "status")]

    # §6.3 序列：正在初始化... → 正在深度思考... → ...
    assert statuses[0] == "正在初始化..."
    assert statuses[1] == "正在深度思考..."


def test_deep_thinking_falls_back_to_prompt_injection():
    # 客户端不支持 chat_stream_thinking：回退为提示词注入，且不产出 thinking 事件
    client = ScriptedLLM(raw_responses=["直接回答"], stream_chunks=["回", "答"])
    agent = BiddingAgent(llm_client=client)

    events = list(agent._chat_events("复杂问题", deep_thinking_enabled=True))

    assert _events_by_type(events, "thinking") == []
    assert any(e[0] == "done" for e in events)
    system = client.stream_messages[0][0]["content"]
    assert DEEP_THINKING_INJECT in system


def test_fallback_path_also_raises_max_tokens():
    """★ 回退路径也必须放宽输出预算。

    实测缺陷：原实现只给真推理模型那条路径传了 `max_tokens=MAX_TOKENS_DEEP`，
    提示词注入这条**完全没传**（`chat_stream(messages)`），于是用厂商默认值——
    而这条路径恰恰最需要放宽：模型没有原生思维链，"拆解 → 多角度分析 → 结论"
    全部内容都算在正文预算里，默认值下容易写到一半被截断。
    """
    client = ScriptedLLM(raw_responses=["直接回答"], stream_chunks=["答"])
    agent = BiddingAgent(llm_client=client)

    list(agent._chat_events("复杂问题", deep_thinking_enabled=True))

    assert client.stream_kwargs, "回退路径应调用 chat_stream"
    assert client.stream_kwargs[0].get("max_tokens") == MAX_TOKENS_DEEP


def test_normal_path_does_not_raise_max_tokens(monkeypatch):
    """不开深度思考时**不该**传 max_tokens —— 保持厂商默认，避免无谓地改变行为。"""
    from src.agent import constants as C

    monkeypatch.delenv(C.THINKING_ENABLED_ENV["deepseek"], raising=False)
    monkeypatch.delenv(C.THINKING_MODEL_ENV["deepseek"], raising=False)
    client = ScriptedLLM(raw_responses=["直接回答"], stream_chunks=["答"])
    agent = BiddingAgent(llm_client=client)

    list(agent._chat_events("普通问题", deep_thinking_enabled=False))

    assert client.stream_kwargs, "普通路径应调用 chat_stream"
    assert "max_tokens" not in client.stream_kwargs[0]


class TestThinkingParameterPayload:
    """★ 请求体里的 `thinking` 参数（真调实测得出的必要行为）。

    真调 `deepseek-flash` 实测（2026-10）：

        不发 thinking      → reasoning = 92 字符   ← **默认就是开的**！
        thinking=disabled  → reasoning =  0 字符   ← 显式关才真关
        thinking=enabled   → reasoning = 89 字符

    官方文档也写明「思考模式**默认打开**，且 effort 默认为 high」。
    故"不发参数"不等于"关闭"——不显式关闭，每次普通问答都在为思维链
    付延迟与 token，而且界面上「深度思考」显示为关、服务端却在思考，
    **开关是假的**。
    """

    def _client(self, thinking: bool):
        from src.clients.base_client import OpenAICompatClient

        c = OpenAICompatClient(
            api_key="k", model="deepseek-flash", base_url="http://x/v1"
        )
        c.set_thinking(thinking)
        return c

    def test_disabled_is_sent_explicitly(self):
        body = self._client(False)._payload([{"role": "user", "content": "x"}], stream=False)
        assert body.get("thinking") == {"type": "disabled"}, (
            "必须显式发 disabled —— 不发等于默认开启，普通问答会白付思维链成本"
        )

    def test_enabled_is_sent(self):
        body = self._client(True)._payload([{"role": "user", "content": "x"}], stream=False)
        assert body.get("thinking") == {"type": "enabled"}

    def test_temperature_only_in_non_thinking_mode(self):
        """思考模式下不发 temperature（官方：设置了也不生效）。"""
        plain = self._client(False)._payload([{"role": "user", "content": "x"}], stream=False)
        deep = self._client(True)._payload([{"role": "user", "content": "x"}], stream=False)
        assert "temperature" in plain
        assert "temperature" not in deep, (
            "思考模式不支持 temperature，发了不报错也不生效——留着会让人误以为它在起作用"
        )

    def test_thinking_goes_to_body_top_level(self):
        """`thinking` 放 body 顶层，**不是** `extra_body`。

        `extra_body` 是 OpenAI Python SDK 的参数名（SDK 会摊平进 body）。
        本项目用原生 HTTP，照抄 SDK 写法会发出 `{"extra_body": {...}}`，服务端不认。
        """
        body = self._client(True)._payload([{"role": "user", "content": "x"}], stream=False)
        assert "extra_body" not in body

    def test_chain_delegates_thinking_flag(self):
        """★ 链必须透传思考开关，否则多 provider 部署下深度思考静默退化。"""
        from src.clients.base_client import OpenAICompatClient
        from src.clients.fallback import build_fallback_client

        def resolver(name: str):
            return OpenAICompatClient(
                api_key="k", model="deepseek-flash", base_url="http://x/v1", provider=name
            )

        chain = build_fallback_client("deepseek", resolver)
        assert chain.thinking_enabled() is False
        chain.set_thinking(True)
        assert chain.thinking_enabled() is True

        # 链上客户端也要真的被开启（`_payload` 才会发 thinking 参数）
        _name, client = chain._available_pair()[0]
        assert client.thinking_enabled() is True


def test_prompt_injection_has_three_step_structure():
    """注入内容要有可执行的三步结构，而不是一句抽象的"请逐步推理"。

    "逐步推理"太抽象，模型往往直接给结论，或写一段与回答无实质差别的
    "我分析了以下方面"。给出步骤与每步产出要求才有稳定效果。
    """
    for kw in ("拆解", "多角度", "结论"):
        assert kw in DEEP_THINKING_INJECT, f"注入缺少「{kw}」环节"
    assert "第三步" in DEEP_THINKING_INJECT, "缺少明确的步骤划分"


def test_thinking_model_enables_on_copy(monkeypatch):
    """★ 深度思考在**浅拷贝**上开启，不污染共享客户端。

    客户端实例在 `_provider_clients` 里按链缓存、跨请求共享。若原地开思考模式：
        t0  A 请求 deep_thinking=True  → 共享 client 变成思考模式
        t1  B 请求 deep_thinking=False → 拿到同一对象，也跟着走思考模式
    即"一个请求的开关污染了并发请求"（后者莫名其妙变慢变贵）。
    """
    monkeypatch.setenv("DEEPSEEK_THINKING_ENABLED", "true")
    original = ThinkingLLM()
    agent = BiddingAgent(llm_client=original)

    client = agent._thinking_model(original, "deepseek")

    assert client is not original, "必须返回浅拷贝，避免污染共享客户端"
    assert client.thinking_enabled() is True, "副本应处于思考模式"
    assert original.thinking_enabled() is False, "原客户端不该被开启"


def test_thinking_model_not_configured_returns_client_unchanged(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_THINKING_ENABLED", raising=False)
    monkeypatch.delenv("DEEPSEEK_THINKING_MODEL", raising=False)
    client = ThinkingLLM()
    agent = BiddingAgent(llm_client=client)

    assert agent._thinking_model(client, "deepseek") is client


def test_switch_thinking_model_still_works(monkeypatch):
    """显式指定推理模型名的方式仍然保留。

    DeepSeek 已无独立推理模型名，但**确有独立推理模型的部署**（如本地 vLLM
    上单独部署的 R1）仍需这条路，故不能因为 API 变更就删掉。
    """
    switched = []

    class SwitchableLLM(ThinkingLLM):
        def switch_thinking_model(self, model_name):
            switched.append(model_name)
            return self

    monkeypatch.delenv("DEEPSEEK_THINKING_ENABLED", raising=False)
    monkeypatch.setenv("DEEPSEEK_THINKING_MODEL", "local-r1")
    original = SwitchableLLM()
    agent = BiddingAgent(llm_client=original)

    client = agent._thinking_model(original, "deepseek")

    assert switched == ["local-r1"], "应切到配置的推理模型名"
    assert client is not original


class TestRealClientContract:
    """★ 契约层：用**真实客户端**验证能力存在。

    这一层才是能发现"`hasattr` 永远为 False"那类缺陷的测试。
    上面用假客户端测切换行为，无论实现是否存在都会通过。
    """

    @pytest.fixture(autouse=True)
    def _env(self):
        import os

        from dotenv import load_dotenv

        from src.agent import constants as C

        load_dotenv()
        # 清掉可能来自 `.env` 的开关，避免测试结果取决于开发者本机配置
        for key in list(C.THINKING_ENABLED_ENV.values()) + list(C.THINKING_MODEL_ENV.values()):
            os.environ.pop(key, None)
        os.environ["DEEPSEEK_THINKING_MODEL"] = "thinking-model-for-test"
        os.environ["ZHIPU_THINKING_MODEL"] = "thinking-model-for-test"
        yield
        os.environ.pop("DEEPSEEK_THINKING_MODEL", None)
        os.environ.pop("ZHIPU_THINKING_MODEL", None)

    @pytest.mark.parametrize("provider", ["deepseek", "zhipu", "ollama", "vllm", "openai"])
    def test_real_clients_support_switch_thinking_model(self, provider):
        """每个登记的 provider 客户端都必须**真的**能切思考模型。

        实测（修复前）：
            deepseek  DeepSeekClient  switch_thinking_model=False
            zhipu     ZaiClient       switch_thinking_model=False
        唯一有的是测试里的假客户端。
        """
        from src.clients.llm_factory import get_llm_client

        client = get_llm_client(provider)
        assert hasattr(client, "switch_thinking_model"), (
            f"{provider} 的客户端不支持思考模型切换——"
            f"深度思考会永远退化为提示词注入"
        )

    def test_switch_actually_changes_model_name(self):
        from src.clients.llm_factory import get_llm_client

        client = get_llm_client("deepseek")
        before = client.model_name
        client.switch_thinking_model("X")
        assert client.model_name == "X"
        assert before != "X"

    def test_unavailable_client_also_supports_it(self):
        """凭据缺失时返回的 `UnavailableLLMClient` 也不能因缺方法而崩。

        历史事故：该类曾漏实现 `chat_raw`，让"设计好的优雅降级"崩成 500。
        """
        from src.clients.base_client import UnavailableLLMClient

        c = UnavailableLLMClient("测试")
        assert hasattr(c, "switch_thinking_model")
        c.switch_thinking_model("X")  # 不应抛

    def test_every_registered_provider_has_thinking_env_or_falls_back(self):
        """`THINKING_MODEL_ENV` 覆盖的 provider 必须都是已登记的。

        写错 provider 名（如 "zai" 而非 "zhipu"）会让配置**静默失效**：
        用户以为配了思考模型，实际一直走提示词注入，没有任何报错。
        """
        from src.clients.llm_factory import list_providers

        registered = set(list_providers())
        for name in THINKING_MODEL_ENV:
            assert name in registered, (
                f"THINKING_MODEL_ENV 里的 {name!r} 不是已登记的 provider："
                f"该配置永远不会生效"
            )


class TestReasoningContentEcho:
    """★ 思考模式 + `tools` 必须回传历史 `reasoning_content`。

    DeepSeek「思考模式」文档明确：**请求携带 `tools` 时，历史轮次的
    `reasoning_content` 都应回传给 API；若未正确回传，API 会返回 400。**

    本项目的 ReAct 决策轮**每轮都带 `tools`**（BASE_TOOL_NAMES 常驻暴露），
    故这条是硬约束——不回传不是"效果差一点"，而是整个问答直接失败。
    """

    class _ToolCallingLLM(ScriptedLLM):
        """第一轮调用工具（并产出思维链），第二轮给结论。"""

        def __init__(self):
            super().__init__(raw_responses=[
                '{"tool_calls": [{"name": "search_knowledge_base",'
                ' "arguments": {"query": "投标保证金"}}]}',
                "结论：保证金比例不超过 2%。",
            ])
            self.seen_conversations: list[list] = []
            self._reasonings = ["这是第一轮的推理过程", ""]

        def chat_raw(self, messages, tools=None, **kwargs):
            # 记录**模型实际收到的**会话（用于断言回传）
            self.seen_conversations.append([dict(m) for m in messages])
            self.last_reasoning = self._reasonings[
                min(len(self.seen_conversations) - 1, 1)
            ]
            return super().chat_raw(messages, tools=tools)

        def chat_stream(self, messages, **kwargs):
            self.stream_messages.append(messages)
            yield "最终回答"

    def test_reasoning_echoed_in_followup_request(self):
        client = self._ToolCallingLLM()
        agent = BiddingAgent(llm_client=client)

        list(agent._chat_events("投标保证金比例是多少", deep_thinking_enabled=True))

        assert len(client.seen_conversations) >= 2, "应至少两轮决策（一轮调工具）"
        second_round = client.seen_conversations[1]
        assistants = [m for m in second_round if m.get("role") == "assistant"]
        assert assistants, "第二轮应能看到第一轮的 assistant 消息"
        assert any(m.get("reasoning_content") == "这是第一轮的推理过程"
                   for m in assistants), (
            "带 tools 的后续请求必须回传 reasoning_content，否则 API 返回 400"
        )

    def test_no_reasoning_key_when_empty(self):
        """没产出思维链时不该添加该键（普通模式请求体保持不变）。"""
        client = ScriptedLLM(raw_responses=["直接回答"], stream_chunks=["答"])
        agent = BiddingAgent(llm_client=client)

        list(agent._chat_events("简单问题", deep_thinking_enabled=False))

        for conv in [client.stream_messages]:
            for msgs in conv:
                for m in msgs:
                    assert "reasoning_content" not in m

    def test_take_reasoning_consumes_value(self):
        """★ 取值后必须清空，避免旧思维链错配到下一轮。"""
        from src.agent.react_loop import ReActMixin

        class C:
            last_reasoning = "第一轮推理"

        c = C()
        assert ReActMixin._take_reasoning(None, c) == "第一轮推理"
        assert c.last_reasoning == "", "取走后应清空"
        assert ReActMixin._take_reasoning(None, c) == ""

    def test_take_reasoning_tolerates_missing_attr(self):
        """客户端没有该属性（第三方替身）时返回空串，不该抛。"""
        from src.agent.react_loop import ReActMixin

        class C:
            pass

        assert ReActMixin._take_reasoning(None, C()) == ""


class TestThinkingStreamUnpacking:
    """★ 思维链产出形态的归一化（曾有一个静默吞掉思维链的 bug）。

    原实现写的是 `for kind, chunk in stream:`，而链产出的每一项是**字典**
    `{"type": "thinking", "content": "We"}`。对字典迭代得到的是它的**键**：

        for kind, chunk in {"type": "thinking", "content": "We"}:
            第1轮 kind='type'     chunk='thinking'
            第2轮 kind='content'  chunk='We'

    于是 `kind == "thinking"` **永远不成立**：思维链的**内容字符串**
    （'thinking' / 'We' / …）被当成正文下发，而 `thinking` 事件恒为 0
    ——前端折叠块永远空白，同时正文里混进思维链。

    这类 bug 极隐蔽：不报错、有输出、只是"思维链功能看起来没用"。
    实测证据：在循环里打印 kind 的取值集合只得到 `{'type': 713}`。
    """

    def test_dict_stream_yields_thinking_events(self, monkeypatch):
        from src.agent import constants as C

        monkeypatch.setenv(C.THINKING_ENABLED_ENV["deepseek"], "true")

        class DictStreamLLM(ScriptedLLM):
            """★ 产出**字典**（与 `FallbackLLMClient` 一致的真实形态）。"""

            def __init__(self):
                super().__init__(raw_responses=["结论"])
                self._thinking = False

            def supports_reasoning(self):
                return True

            def set_thinking(self, enabled):
                self._thinking = bool(enabled)
                return self

            def thinking_enabled(self):
                return self._thinking

            def chat_stream_thinking(self, messages, max_tokens=None):
                self.stream_messages.append(messages)
                # 逐项字典：这正是链 `_stream_with_failover` 展开后的形态
                for piece in ("拆解", "问题", "分析", "结论"):
                    yield {"type": "thinking", "content": piece}
                yield {"type": "content", "content": "最终答案"}

        client = DictStreamLLM()
        agent = BiddingAgent(llm_client=client)

        events = list(agent._chat_events("复杂问题", deep_thinking_enabled=True))

        thinking = _events_by_type(events, "thinking")
        tokens = "".join(e[1] for e in _events_by_type(events, "token"))
        assert [e[1] for e in thinking] == ["拆解", "问题", "分析", "结论"], (
            "字典形态的流必须被正确解包成 thinking 事件"
        )
        assert tokens == "最终答案", "思维链不该混进正文"
        assert "拆解" not in tokens, "思维链内容泄漏到正文了"

    def test_tuple_stream_still_supported(self, monkeypatch):
        """元组形态（`(kind, chunk)`）也要支持——测试替身与第三方实现可能用它。"""
        from src.agent import constants as C

        monkeypatch.setenv(C.THINKING_ENABLED_ENV["deepseek"], "true")

        class TupleStreamLLM(ScriptedLLM):
            def __init__(self):
                super().__init__(raw_responses=["结论"])
                self._thinking = False

            def supports_reasoning(self):
                return True

            def set_thinking(self, enabled):
                self._thinking = bool(enabled)
                return self

            def thinking_enabled(self):
                return self._thinking

            def chat_stream_thinking(self, messages, max_tokens=None):
                self.stream_messages.append(messages)
                yield ("thinking", "推理")
                yield ("content", "答案")

        agent = BiddingAgent(llm_client=TupleStreamLLM())
        events = list(agent._chat_events("问题", deep_thinking_enabled=True))

        assert [e[1] for e in _events_by_type(events, "thinking")] == ["推理"]
        assert "".join(e[1] for e in _events_by_type(events, "token")) == "答案"


class TestFallbackChainDelegation:
    """★ 自动切换链必须透传"当前会用的客户端"的能力。

    实测缺陷（修复前）：
        探针 deep_thinking=True meta_model='' chain_model=None capable=False

    生产环境 `agent._get_llm_client()` 返回的是 `FallbackLLMClient`（链），
    不是单个客户端。链没有 `model_name`、也没有 `supports_reasoning()`，
    于是判定恒为 False —— **即使用户配了推理模型、请求确实发给了
    deepseek-reasoner，注入分支照样执行**，走不成思考流。

    最隐蔽的一点：**单个客户端测起来全对**（有 model_name、能力为 True），
    一旦包成链（也就是生产路径）就失效。故这一层必须单独测。
    """

    def _chain(self, models: dict[str, str]):
        from src.clients.base_client import OpenAICompatClient
        from src.clients.fallback import build_fallback_client

        def resolver(name: str):
            c = OpenAICompatClient(
                api_key="k", model=models[name], base_url="http://x/v1", provider=name
            )
            return c

        return build_fallback_client(list(models)[0], resolver)

    def test_chain_exposes_model_name(self):
        chain = self._chain({"deepseek": "deepseek-chat"})
        assert getattr(chain, "model_name", "") == "deepseek-chat"

    def test_chain_reports_reasoning_capability(self):
        chain = self._chain({"deepseek": "deepseek-chat"})
        assert chain.supports_reasoning() is True, (
            "链应在底层客户端支持思维链时返回 True，否则深度思考永远走注入"
        )

    def test_chain_model_name_reflects_switch(self):
        """链上切换思考模型后，`model_name` 要跟着变。

        判定逻辑靠"配置的思考模型名 == 实际模型名"，故这一项必须准确。
        """
        chain = self._chain({"deepseek": "deepseek-chat"})
        switched = chain.switch_thinking_model("deepseek-reasoner")
        assert switched.model_name == "deepseek-reasoner"

    def test_chain_without_capable_client_is_false(self):
        """链上客户端都不支持思维链时返回 False，让上层正常回退注入。"""
        from src.clients.base_client import BaseLLMClient
        from src.clients.fallback import build_fallback_client

        class NoReasoning(BaseLLMClient):
            provider = "plain"

            def __init__(self):
                self._model = "plain-model"

            @property
            def model_name(self):
                return self._model

            def available(self):
                return True

            def chat_stream(self, messages, **kw):
                yield ""

            def chat_raw(self, messages, tools=None, **kw):
                return ""

            def chat_stream_thinking(self, messages, **kw):
                return
                yield

        chain = build_fallback_client("plain", lambda name: NoReasoning())
        assert chain.supports_reasoning() is False


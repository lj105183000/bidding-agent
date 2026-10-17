"""把端到端验证固化进正式测试（DSML 泄漏的完整链路）。

★ 与单元测试的分工
==============
`test_dsml_tool_calls.py` 验证"检测与解析函数正确"。
这里验证**整条链路**：模型流里吐出标记 → generation 缓冲检出 →
reset 清空 → 兜底执行工具 → 重新生成干净回答，
且用户最终看到的 SSE 里**不含任何标记**。

★ 为什么要走完整链路
--------------
只测 `_parse_tool_text` 会漏掉"解析正确但没接到 reset 逻辑上"的情况。
实测第一版端到端脚本就因为**没装假工具执行器**而以"生成失败"收尾——
那会掩盖真正要验证的行为，故这里显式装上。
"""

from __future__ import annotations

import json

import pytest

from src.agent import BiddingAgent

from dsml_samples import FW

MARK = f"{FW}{FW}DSML{FW}{FW}"

#: 模型把工具调用写进流式正文（用户实际会看到的泄漏路径）
LEAK = (
    f"<{MARK}tool_calls>\n"
    f'<{MARK}invoke name="search_knowledge_graph">\n'
    f'<{MARK}parameter name="keyword" string="true">管理局</parameter>\n'
    f'<{MARK}parameter name="query_type" string="true">search_entity</parameter>\n'
    f"</invoke>\n"
    f"</{MARK}tool_calls>"
)

CLEAN_ANSWER = "管理局是负责行政管理的政府机构。"


class _LeakingLLM:
    """脚本化替身：① 决策轮给泄漏文本 ② 兜底时给干净回答。

    ★ 替身必须像真实模型的**序列**，而不是永远返回同一种输出。
    第一版让 `chat_raw` 永远返回泄漏文本，结果兜底重生成后又泄漏一次，
    最终用户看到"生成失败"——那不是被测逻辑的问题，是替身不像真实模型。
    """

    def __init__(self):
        self.stream_calls = 0
        self.raw_calls = 0

    def chat_raw(self, messages, tools=None, **kwargs):
        self.raw_calls += 1
        self.last_reasoning = ""
        return LEAK if self.raw_calls == 1 else CLEAN_ANSWER

    def chat_stream(self, messages, **kwargs):
        self.stream_calls += 1
        if self.stream_calls == 1:
            yield "根据查询："
            yield LEAK
        else:
            yield CLEAN_ANSWER


class _NativeToolCallLLM(_LeakingLLM):
    """对照：走原生 tool_calls（客户端把它序列化成项目约定的 JSON 文本）。"""

    def chat_raw(self, messages, tools=None, **kwargs):
        self.raw_calls += 1
        self.last_reasoning = ""
        if self.raw_calls == 1:
            return json.dumps({
                "tool_calls": [{
                    "name": "search_knowledge_graph",
                    "arguments": {"keyword": "管理局", "query_type": "search_entity"},
                }]
            }, ensure_ascii=False)
        return CLEAN_ANSWER


@pytest.fixture
def fake_executors(monkeypatch):
    """装上假的图谱/知识库执行器。

    ★ 必须装：不装会走真实实现去连 Neo4j，离线环境下失败，
    兜底重生成拿不到工具结果而报"生成失败"——实测踩过，
    那会让端到端测试以失败收尾从而掩盖真正要验证的行为。
    """
    import src.agent.react_loop as loop_mod

    monkeypatch.setitem(
        loop_mod.TOOL_EXECUTORS, "search_knowledge_graph",
        lambda arguments, question="": ("图谱结果：管理局是政府机构", []),
    )
    monkeypatch.setitem(
        loop_mod.TOOL_EXECUTORS, "search_knowledge_base",
        lambda arguments, question="": ("知识库结果", []),
    )


def _visible_text(events: list[tuple]) -> str:
    """用户最终会看到的文本（token 拼接；reset 清空之前的内容）。"""
    out = ""
    for etype, payload in events:
        if etype == "reset":
            out = ""
        elif etype == "token":
            out += str(payload)
    return out


class TestDsmlLeakEndToEnd:
    def test_dsml_leak_never_reaches_user(self, fake_executors):
        """★ 核心断言：DSML 标记绝不能出现在用户可见文本里。

        这是用户上报问题的直接验收：原实现检测不到该标记，
        于是它被当普通文本下发，用户看到的"回答"就是一串协议标记。
        """
        client = _LeakingLLM()
        agent = BiddingAgent(llm_client=client)
        events = list(agent._chat_events("管理局是什么机构？"))

        text = _visible_text(events)
        assert "DSML" not in text, f"标记泄漏到用户可见文本: {text[:120]!r}"
        assert "invoke" not in text, f"标签泄漏到用户可见文本: {text[:120]!r}"

    def test_leak_triggers_reset(self, fake_executors):
        """检出泄漏后必须 reset，否则已流出的半截标记会留在界面上。"""
        client = _LeakingLLM()
        agent = BiddingAgent(llm_client=client)
        events = list(agent._chat_events("管理局是什么机构？"))

        resets = [e for e in events if e[0] == "reset"]
        assert resets, "DSML 泄漏未触发 reset"

    def test_regenerated_answer_is_shown(self, fake_executors):
        """reset 之后要有干净回答顶上——只清空不补，用户就只看到空白。"""
        client = _LeakingLLM()
        agent = BiddingAgent(llm_client=client)
        events = list(agent._chat_events("管理局是什么机构？"))

        text = _visible_text(events)
        assert CLEAN_ANSWER in text, f"兜底重生成的回答未出现: {text[:120]!r}"

    def test_native_tool_calls_path_unaffected(self, fake_executors):
        """对照组：原生 tool_calls 路径行为不变（本次改动不能影响它）。"""
        client = _NativeToolCallLLM()
        agent = BiddingAgent(llm_client=client)
        events = list(agent._chat_events("管理局是什么机构？"))

        text = _visible_text(events)
        assert CLEAN_ANSWER in text
        assert "DSML" not in text

    def test_arguments_reach_executor_correctly(self, fake_executors, monkeypatch):
        """★ 泄漏文本里的参数要能正确传给工具——不能只是"拦住不下发"。

        若只挡住展示、却没把调用真的执行，用户拿到的是一句没有依据的回答，
        比看到标记更糟（标记至少还提示了"系统想查东西"）。
        """
        import src.agent.react_loop as loop_mod

        captured: list[dict] = []
        monkeypatch.setitem(
            loop_mod.TOOL_EXECUTORS, "search_knowledge_graph",
            lambda arguments, question="": (captured.append(arguments) or "图谱结果", []),
        )
        client = _LeakingLLM()
        agent = BiddingAgent(llm_client=client)
        list(agent._chat_events("管理局是什么机构？"))

        assert captured, "工具未被执行"
        assert captured[0].get("keyword") == "管理局", (
            f"参数未正确解析并传递: {captured[0]}"
        )

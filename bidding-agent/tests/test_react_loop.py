"""ReAct 多轮循环测试：多轮回填 / 轮次上限 / 交错执行顺序 / 失败降级。

mock 约定（开发文档 §9.3）：文件内局部 fake 类；TOOL_EXECUTORS 在
src.agent.core 与 src.agent.react_loop 有两处模块级绑定，必须同时替换。
"""

import src.agent.core as core_mod
import src.agent.react_loop as loop_mod
from src.agent import BiddingAgent


class ScriptedLLM:
    """按脚本队列返回 chat_raw 响应（可为异常），chat_stream 返回固定分片。"""

    def __init__(self, raw_responses=(), stream_chunks=()):
        self.raw_responses = list(raw_responses)
        self.stream_chunks = list(stream_chunks)
        self.raw_calls = 0
        self.stream_calls = 0

    def chat_raw(self, messages, tools=None):
        self.raw_calls += 1
        item = self.raw_responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def chat_stream(self, messages):
        self.stream_calls += 1
        yield from self.stream_chunks


class FakeRAGPipeline:
    """RAG 降级路径用：search 返回固定来源。"""

    def __init__(self):
        self.calls = []

    def search(self, question, top_k=5):
        self.calls.append((question, top_k))
        return [{"question": "知识库问题", "answer": "知识库答案", "score": 0.8}]


def _install_executors(monkeypatch, mapping, log=None):
    """同时替换两处 TOOL_EXECUTORS 绑定；log 用于记录执行顺序。"""
    def make(name, fn):
        def wrapper(arguments):
            if log is not None:
                log.append(name)
            return fn(arguments)
        return wrapper

    executors = {name: make(name, fn) for name, fn in mapping.items()}
    monkeypatch.setattr(core_mod, "TOOL_EXECUTORS", executors)
    monkeypatch.setattr(loop_mod, "TOOL_EXECUTORS", executors)


def _events_by_type(events, event_type):
    return [e for e in events if e[0] == event_type]


def _done(events):
    done = _events_by_type(events, "done")
    assert len(done) == 1
    return done[0][1]


def test_multi_round_tool_call_and_source_merge(monkeypatch):
    call = '{"name": "search_knowledge_base", "arguments": {"query": "投标流程"}}'
    fake = ScriptedLLM(
        raw_responses=[call, "根据检索结果，回答如下"],
        stream_chunks=["最终", "答案"],
    )
    _install_executors(monkeypatch, {
        "search_knowledge_base": lambda args: (
            "知识库结果文本",
            [{"question": "q1", "answer": "a1", "score": 0.9}],
        ),
    })
    agent = BiddingAgent(llm_client=fake)

    events = list(agent._chat_events("投标流程是什么"))
    statuses = [e[1] for e in _events_by_type(events, "status")]

    assert "正在检索与搜索..." in statuses
    assert "正在分析检索结果..." in statuses
    assert "正在生成回答..." in statuses
    assert not any("正在补充检索" in s for s in statuses)
    assert fake.raw_calls == 2 and fake.stream_calls == 1

    done = _done(events)
    assert done["tool_called"] is True
    assert done["tool_name"] == "search_knowledge_base"
    assert done["sources"] == [{"question": "q1", "answer": "a1", "score": 0.9}]
    assert [p[0] for p in done["phase_times"]] == [
        "首轮分析", "检索与搜索", "分析检索结果", "生成回答",
    ]


def test_tool_round_limit(monkeypatch):
    call = '{"name": "query_database", "arguments": {"query_type": "amount"}}'
    # 每轮都要求工具：4 轮上限后不再决策，直接最终生成
    fake = ScriptedLLM(raw_responses=[call] * 4, stream_chunks=["收尾回答"])
    exec_log = []
    _install_executors(monkeypatch, {
        "query_database": lambda args: ("数据库结果", []),
    }, log=exec_log)
    agent = BiddingAgent(llm_client=fake)

    events = list(agent._chat_events("统计中标金额"))
    statuses = [e[1] for e in _events_by_type(events, "status")]

    assert exec_log == ["query_database"] * 4
    assert fake.raw_calls == 4
    assert "正在补充检索（第1轮）..." in statuses
    assert "正在补充检索（第2轮）..." in statuses
    assert "正在补充检索（第3轮）..." in statuses
    assert not any("第4轮" in s for s in statuses)

    done = _done(events)
    assert done["tool_called"] is True
    assert done["tool_name"] == "query_database"
    assert [p[0] for p in done["phase_times"]][-1] == "生成回答"


def test_interleaved_execution_order(monkeypatch):
    # 一次决策输出两个调用：按出现顺序执行，tool_name 记录最后完成者
    decision = (
        '{"name": "tool_a", "arguments": {"q": "1"}}\n'
        '{"name": "tool_b", "arguments": {"q": "2"}}'
    )
    fake = ScriptedLLM(raw_responses=[decision, "好的"], stream_chunks=["答"])
    exec_log = []
    _install_executors(monkeypatch, {
        "tool_a": lambda args: ("A 结果", []),
        "tool_b": lambda args: ("B 结果", []),
    }, log=exec_log)
    agent = BiddingAgent(llm_client=fake)

    events = list(agent._chat_events("查一下"))

    assert exec_log == ["tool_a", "tool_b"]
    assert _done(events)["tool_name"] == "tool_b"


def test_tool_failure_degrades_without_crash(monkeypatch):
    fake = ScriptedLLM(
        raw_responses=['{"name": "bad_tool", "arguments": {}}', "直接回答"],
        stream_chunks=["答"],
    )

    def boom(arguments):
        raise RuntimeError("后端宕机")

    _install_executors(monkeypatch, {"bad_tool": boom})
    agent = BiddingAgent(llm_client=fake)

    events = list(agent._chat_events("随便问问"))
    done = _done(events)

    # 工具失败被隔离为结构化错误文本：流程继续，最终仍有 done
    assert done["tool_called"] is False
    assert done["tool_name"] == ""
    assert fake.stream_calls == 1
    assert any(e[0] == "token" and e[1] == "答" for e in events)


def test_llm_failure_falls_back_to_rag(monkeypatch):
    fake = ScriptedLLM(raw_responses=[RuntimeError("LLM 不可用")])
    rag = FakeRAGPipeline()
    agent = BiddingAgent(llm_client=fake, rag_pipeline=rag)

    events = list(agent._chat_events("政策怎么规定"))
    statuses = [e[1] for e in _events_by_type(events, "status")]

    assert "正在检索知识库..." in statuses
    assert "正在生成回答..." in statuses
    assert rag.calls == [("政策怎么规定", 5)]

    done = _done(events)
    assert done["tool_called"] is False
    assert done["sources"] == [{"question": "知识库问题", "answer": "知识库答案", "score": 0.8}]
    assert [p[0] for p in done["phase_times"]] == ["检索", "生成"]


def test_no_tool_direct_answer_phase_times(monkeypatch):
    # 无工具直答：phase_times 仅「首轮分析 + 生成回答」两项（契约 §6.4）
    fake = ScriptedLLM(raw_responses=["你好，我是助手"], stream_chunks=["你", "好"])
    _install_executors(monkeypatch, {})
    agent = BiddingAgent(llm_client=fake)

    events = list(agent._chat_events("你好"))
    done = _done(events)

    assert done["tool_called"] is False
    assert [p[0] for p in done["phase_times"]] == ["首轮分析", "生成回答"]
"""agent_eval 的判分口径测试（全程不构造真实 Agent）。

钉死文档 §6.4 的口径：tool_name 是单字符串、与 expected_tool 精确相等才算对；
error 题（ok=False）没有 match 键，按"答错"计入分母。
"""

import eval.agent_eval as mod


class _FakeAgent:
    def __init__(self, script):
        self._script = script  # question -> chat() 返回
        self.calls = []
        # 记录 kwargs 而非只记 question：评测必须走"无联网/无深度思考"的默认参数口径，
        # 只断言问题字符串钉不住这一点，eval_one 改传参时测试会静默失真
        self.kwargs = []

    def chat(self, question, history=None, web_search_enabled=False,
             provider="", deep_thinking_enabled=False):
        self.calls.append(question)
        self.kwargs.append({
            "history": history,
            "web_search_enabled": web_search_enabled,
            "provider": provider,
            "deep_thinking_enabled": deep_thinking_enabled,
        })
        return self._script[question]


SCRIPT = {
    "某公司中过哪些标？": {"answer": "……", "tool_name": "search_knowledge_graph",
                          "elapsed_ms": 1500, "sources": []},
    "最近的招标政策？": {"answer": "……", "tool_name": "query_database",
                        "elapsed_ms": 900, "sources": []},
}


def test_eval_one_exact_string_match():
    mod.agent = _FakeAgent(SCRIPT)
    result = mod.eval_one({"question": "某公司中过哪些标？", "expected_tool": "search_knowledge_graph"})
    assert result["ok"] is True and result["match"] is True
    result = mod.eval_one({"question": "最近的招标政策？", "expected_tool": "search_web"})
    assert result["match"] is False
    assert result["expected"] == "search_web" and result["actual"] == "query_database"


def test_eval_one_uses_default_params():
    """评测走非流式 chat() 且默认参数（无联网/无深度思考、无历史），文档 §10 口径。"""
    fake = _FakeAgent(SCRIPT)
    mod.agent = fake
    mod.eval_one({"question": "某公司中过哪些标？", "expected_tool": "search_knowledge_graph"})
    assert fake.calls == ["某公司中过哪些标？"]
    kwargs = fake.kwargs[0]
    assert kwargs["web_search_enabled"] is False
    assert kwargs["deep_thinking_enabled"] is False
    assert kwargs["history"] is None


def test_accuracy_counts_errors_as_wrong():
    class _BoomAgent:
        def chat(self, question, **kwargs):
            if "炸" in question:
                raise RuntimeError("LLM 超时")
            return {"answer": "", "tool_name": "search_knowledge_base", "elapsed_ms": 100}

    mod.agent = _BoomAgent()
    cases = [
        {"question": "正常题", "expected_tool": "search_knowledge_base"},
        {"question": "会炸的题", "expected_tool": "search_knowledge_base"},
    ]
    results = mod.run_cases(cases, mod.eval_one)
    matched = sum(1 for r in results if r.get("match"))
    # 2 题分母：error 题不算对
    assert len(results) == 2 and matched == 1


def test_eval_one_flags_degraded_zero_latency_row():
    """chat() 错误路径不抛异常而是把错误折进 answer 正常返回（elapsed_ms 停在 0）。

    这种降级行必须判失败：0ms 计入延迟均值会把均值往低了拽（系统最糟时指标最好看），
    明细若按计分行打印还会显示"实际为空"掩盖根因。
    """

    class _DegradedAgent:
        def chat(self, question, **kwargs):
            # 模拟 core.py chat() 的错误事件路径：无 done 帧，elapsed_ms/tool_name 保持初始值
            return {"answer": "\n\n【错误】知识库未就绪", "tool_name": "", "elapsed_ms": 0}

    mod.agent = _DegradedAgent()
    result = mod.eval_one({"question": "某题", "expected_tool": "search_knowledge_base"})
    assert result["ok"] is False
    assert "降级" in result["error"]
    assert "match" not in result  # 与 error 题同口径：没有 match 键，按答错计入分母

"""agent_eval 的判分口径测试（全程不构造真实 Agent）。

钉死文档 §6.4 的口径：tool_name 是单字符串、与 expected_tool 精确相等才算对；
error 题（ok=False）没有 match 键，按"答错"计入分母。
"""

import eval.agent_eval as mod


class _FakeAgent:
    def __init__(self, script):
        self._script = script  # question -> chat() 返回
        self.calls = []

    def chat(self, question, history=None, web_search_enabled=False,
             provider="", deep_thinking_enabled=False):
        self.calls.append(question)
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
    """评测走非流式 chat() 且默认参数（无联网/无深度思考），文档 §10 口径。"""
    fake = _FakeAgent(SCRIPT)
    mod.agent = fake
    mod.eval_one({"question": "某公司中过哪些标？", "expected_tool": "search_knowledge_graph"})
    assert fake.calls == ["某公司中过哪些标？"]


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

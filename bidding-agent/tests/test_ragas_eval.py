"""ragas_eval 的数据组装与容错契约测试（全程不调 ragas / LLM / Qdrant）。

脚本内 ragas 的 evaluate() 在模块顶层导入后被打成桩：要验证的不是 ragas 本身，
而是"我们喂给它的列名对不对、sources 为空跳不跳、NaN 处理不处理"——这些才是
我们自己的代码。
"""

from types import SimpleNamespace

import eval.ragas_eval as mod


class _FakePipeline:
    """脚本化 rag_pipeline.chat 的返回。"""

    def __init__(self, sources=None, answer="一般不超过项目估算价的2%"):
        self._sources = sources if sources is not None else [{"content": "保证金不超过2%"}]
        self._answer = answer

    def chat(self, question, history=None, top_k=None):
        return {"answer": self._answer, "sources": self._sources,
                "tool_called": False, "tool_name": ""}


def _patch_env(monkeypatch, sources=None, evaluate_result=None):
    monkeypatch.setattr(mod, "rag_pipeline", _FakePipeline(sources=sources))

    calls = []

    def fake_evaluate(dataset, metrics, llm, embeddings):
        calls.append(dataset)
        return evaluate_result or SimpleNamespace(scores=[{
            "faithfulness": 0.8, "answer_relevancy": 0.7, "llm_context_recall": 0.9,
        }])

    monkeypatch.setattr(mod, "evaluate", fake_evaluate)
    return calls


def test_eval_one_returns_three_scores(monkeypatch):
    calls = _patch_env(monkeypatch)
    result = mod.eval_one({"question": "保证金上限？", "reference": "不超过2%"}, llm=object(), embeddings=object())
    assert result["ok"] is True
    assert result["faithfulness"] == 0.8
    assert result["relevancy"] == 0.7
    assert result["recall"] == 0.9
    # 喂给 ragas 的数据列名是 ragas 标准四列
    row = calls[0].to_list()[0]
    assert row["user_input"] == "保证金上限？"
    assert row["response"] == "一般不超过项目估算价的2%"
    assert row["retrieved_contexts"] == ["保证金不超过2%"]
    assert row["reference"] == "不超过2%"


def test_eval_one_skips_empty_sources(monkeypatch):
    _patch_env(monkeypatch, sources=[])
    result = mod.eval_one({"question": "q", "reference": "r"}, llm=object(), embeddings=object())
    assert result["ok"] is False
    assert "sources 为空" in result["error"]


def test_eval_one_maps_nan_to_zero(monkeypatch):
    _patch_env(monkeypatch, evaluate_result=SimpleNamespace(scores=[{
        "faithfulness": float("nan"), "answer_relevancy": 0.7, "llm_context_recall": 0.9,
    }]))
    result = mod.eval_one({"question": "q", "reference": "r"}, llm=object(), embeddings=object())
    assert result["faithfulness"] == 0.0
    assert result["relevancy"] == 0.7


def test_extract_scores_prefers_scores_attr(monkeypatch):
    scores = mod._extract_scores(SimpleNamespace(scores=[{"faithfulness": 0.5}]))
    assert scores == {"faithfulness": 0.5}

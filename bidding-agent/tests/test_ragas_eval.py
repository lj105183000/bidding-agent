"""ragas_eval 的数据组装与容错契约测试（全程不调 LLM / Qdrant，无网络）。

脚本内 ragas 的 evaluate() 在模块顶层导入后被打成桩：要验证的不是 ragas 本身，
而是"我们喂给它的列名对不对、sources 为空/管线报错跳不跳、NaN 处理不处理"——
这些才是我们自己的代码。唯一例外是末尾的 test_metric_name_contract：仅实例化
ragas 指标读 .name（纯内存操作），把分数键名钉死在真实指标上。
"""

from types import SimpleNamespace
import sys

import eval.ragas_eval as mod


class _FakePipeline:
    """脚本化 rag_pipeline.chat 的返回。

    error 模拟 RAGPipeline 的部分失败路径：KB 命中（sources 非空）但生成报错，
    此时 chat 仍带 error 键返回——见 src/rag/pipeline.py 的事件收敛逻辑。
    ready/ready_error 供 main() 的 _check_ready 读取（真实 pipeline 的就绪探针）。
    """

    ready = True
    ready_error = ""

    def __init__(self, sources=None, answer="一般不超过项目估算价的2%", error=None):
        self._sources = sources if sources is not None else [{"content": "保证金不超过2%"}]
        self._answer = answer
        self._error = error

    def chat(self, question, history=None, top_k=None):
        result = {"answer": self._answer, "sources": self._sources,
                  "tool_called": False, "tool_name": ""}
        if self._error is not None:
            result["error"] = self._error
        return result


def _patch_env(monkeypatch, sources=None, evaluate_result=None, error=None):
    monkeypatch.setattr(mod, "rag_pipeline", _FakePipeline(sources=sources, error=error))

    calls = []

    def fake_evaluate(dataset, metrics, llm, embeddings):
        calls.append(dataset)
        # 键名镜像 ragas 0.4.3 的真实返回：LLMContextRecall().name == "context_recall"，
        # 而非类名暗示的 "llm_context_recall"——fake 越贴近现实，键名回归越藏不住
        return evaluate_result or SimpleNamespace(scores=[{
            "faithfulness": 0.8, "answer_relevancy": 0.7, "context_recall": 0.9,
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


def test_eval_one_skips_pipeline_error(monkeypatch):
    """KB 命中但生成失败：pipeline 带非空 sources + error 返回，降级回答不得计分。"""
    calls = _patch_env(monkeypatch, error="LLM 生成失败")
    result = mod.eval_one({"question": "q", "reference": "r"}, llm=object(), embeddings=object())
    assert result["ok"] is False
    assert "pipeline 报错" in result["error"]
    # 守卫必须短路在 ragas 判分之前：降级样本根本不该进 evaluate
    assert calls == []


def test_eval_one_maps_nan_to_zero(monkeypatch):
    _patch_env(monkeypatch, evaluate_result=SimpleNamespace(scores=[{
        "faithfulness": float("nan"), "answer_relevancy": 0.7, "context_recall": 0.9,
    }]))
    result = mod.eval_one({"question": "q", "reference": "r"}, llm=object(), embeddings=object())
    assert result["faithfulness"] == 0.0
    assert result["relevancy"] == 0.7


def test_extract_scores_prefers_scores_attr():
    scores = mod._extract_scores(SimpleNamespace(scores=[{"faithfulness": 0.5}]))
    assert scores == {"faithfulness": 0.5}


def test_metric_name_contract():
    """契约测试：分数读取键必须与真实 ragas 指标的 .name 完全一致。

    ragas evaluate() 按指标实例的 .name 索引 per-sample 分数。曾因硬编码
    "llm_context_recall" 而 0.4.x 实际返回 "context_recall"，召回列恒为默认 0.0
    且全套测试照样通过。这里双向钉死：字面量防手写键回归，现场实例化防
    ragas 升级改名后静默漂移——任何一端变了这条测试都会响。
    """
    from ragas.metrics import AnswerRelevancy, Faithfulness, LLMContextRecall

    # 字面量钉死：ragas 0.4.x 三个指标的实际 .name 与输出列的对应关系
    assert mod._NAME_TO_OUTPUT == {
        "faithfulness": "faithfulness",
        "answer_relevancy": "relevancy",
        "context_recall": "recall",
    }
    # 模块内 _METRICS 的 .name 与读取键一一对应（eval_one 的提取来源）
    assert {m.name for m in mod._METRICS} == set(mod._NAME_TO_OUTPUT)
    # 且与现场从 ragas 实例化的真实指标一致（防版本升级改名）
    real_names = {m.name for m in (Faithfulness(), AnswerRelevancy(), LLMContextRecall())}
    assert set(mod._NAME_TO_OUTPUT) == real_names
    # to_pandas 兜底路径读取同一组键，不能单独漂移
    assert set(mod._METRIC_KEYS) == real_names


def test_main_negative_limit_runs_all_cases(monkeypatch):
    """--limit 负数不得从尾部切片：钳到 0 走全量（与 agent_eval 的负数防护同口径）。

    不抽 _apply_limit 助手是为了与 agent_eval 的 main() 结构逐字对齐，故在 main() 层
    验证：--limit -3 配 4 条用例，负数若未钳制会切成 cases[:-3] 只剩 1 条。
    """
    calls = _patch_env(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["ragas_eval.py", "--limit", "-3"])
    monkeypatch.setattr(
        mod, "load_cases",
        lambda *a, **k: [{"question": f"第{i}题", "reference": "r"} for i in range(1, 5)],
    )
    monkeypatch.setattr(mod, "build_evaluator_llm", lambda: SimpleNamespace(model_name="fake-judge"))

    assert mod.main() == 0
    # 4 条全部进评测：负数 limit 被钳成全量，而非静默评测尾部切出的错误子集
    assert len(calls) == 4

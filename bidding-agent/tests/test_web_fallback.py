"""联网兜底单测。

只测**不依赖真实联网、不依赖 Qdrant** 的部分：覆盖判定的分支、结果归一、
评分降级、错误区分。真实联网由端到端脚本覆盖。

重点守住三处设计约束：
    - "搜不到"（空列表）与"搜不了"（异常）必须区分——混为一谈会让用户以为系统坏了
    - **部分数据源失败不能整体失败**（两路互补，一路挂了另一路仍有用）
    - 覆盖判定**永不抛异常**（证明不了"够"时联网，比凭不足依据作答安全）
"""

from __future__ import annotations

import pytest

from src.services import web_fallback as WF


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


def test_search_hit_to_dict_shape():
    h = WF.SearchHit(title="t", url="u", content="c", score=0.7, source="tavily", score_kind="engine")
    d = h.to_dict()
    for k in ("title", "url", "content", "score", "published", "source", "score_kind"):
        assert k in d
    assert d["score"] == 0.7 and d["score_kind"] == "engine"


def test_local_coverage_to_dict_shape():
    c = WF.LocalCoverage(enough=True, best_relevance=0.8, threshold=0.5, hits_count=3)
    d = c.to_dict()
    for k in ("enough", "best_relevance", "threshold", "hits_count", "reason", "judged"):
        assert k in d


def test_web_search_unavailable_is_runtime_error():
    assert issubclass(WF.WebSearchUnavailable, RuntimeError)


def test_defaults_are_sane():
    # 阈值必须与 vector_store.MIN_RELEVANCE 一致（否则"本地够不够"的判据会漂移）
    assert WF.DEFAULT_MIN_RELEVANCE == 0.5
    assert 1 <= WF.DEFAULT_WEB_TOP_K <= 10


# ---------------------------------------------------------------------------
# 覆盖判定：永不抛异常
# ---------------------------------------------------------------------------


class _FakePipeline:
    def __init__(self, sources=None, raises=None):
        self._sources = sources or []
        self._raises = raises

    def search(self, question, top_k=3):
        if self._raises:
            raise self._raises
        return self._sources


def test_coverage_enough_when_relevance_above_threshold():
    p = _FakePipeline([{"question": "q", "answer": "a", "relevance": 0.72}])
    c = WF.local_coverage("问题", pipeline=p, judge=False)
    assert c.enough is True
    assert c.best_relevance == 0.72
    assert c.judged is False  # judge=False 时如实标记未判读


def test_coverage_not_enough_when_relevance_below_threshold():
    p = _FakePipeline([{"question": "q", "answer": "a", "relevance": 0.31}])
    c = WF.local_coverage("问题", pipeline=p, judge=False)
    assert c.enough is False
    assert "低于阈值" in c.reason


def test_coverage_empty_sources_is_not_enough():
    c = WF.local_coverage("问题", pipeline=_FakePipeline([]), judge=False)
    assert c.enough is False
    assert "没有检索到任何内容" in c.reason


def test_coverage_without_relevance_field_is_not_enough():
    """缺少相关性字段时判"不够"——**不能证明够**时联网比凭不足依据作答安全。"""
    p = _FakePipeline([{"question": "q", "answer": "a"}])
    c = WF.local_coverage("问题", pipeline=p, judge=False)
    assert c.enough is False
    assert "没有相关性分数" in c.reason


def test_coverage_never_raises_on_pipeline_error():
    p = _FakePipeline(raises=RuntimeError("qdrant down"))
    c = WF.local_coverage("问题", pipeline=p, judge=False)
    assert c.enough is False
    assert "RuntimeError" in c.reason


def test_coverage_never_raises_when_pipeline_import_fails(monkeypatch):
    """RAG 未就绪时也应返回"不够"而不是抛异常——知识库没东西更该联网。"""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "src.rag.pipeline":
            raise ImportError("not ready")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    c = WF.local_coverage("问题", judge=False)
    assert c.enough is False


def test_coverage_takes_max_relevance_across_sources():
    p = _FakePipeline([{"relevance": 0.2}, {"relevance": 0.66}, {"relevance": 0.4}])
    c = WF.local_coverage("问题", pipeline=p, judge=False)
    assert c.best_relevance == 0.66


def test_coverage_ignores_non_numeric_relevance():
    p = _FakePipeline([{"relevance": "high"}, {"relevance": 0.7}])
    c = WF.local_coverage("问题", pipeline=p, judge=False)
    assert c.best_relevance == 0.7


# ---------------------------------------------------------------------------
# LLM 判读（第二层）
# ---------------------------------------------------------------------------


class _FakeLLM:
    def __init__(self, reply):
        self._reply = reply

    def available(self):
        return True

    def chat_raw(self, messages):
        return self._reply


def test_judge_parses_json_true():
    llm = _FakeLLM('{"can_answer": true, "reason": "资料直接包含答案"}')
    can, reason = WF._judge_by_llm("问题", [{"question": "q", "answer": "a"}], llm=llm)
    assert can is True and "直接包含" in reason


def test_judge_parses_json_false():
    llm = _FakeLLM('{"can_answer": false, "reason": "资料只涉及大陆法规"}')
    can, reason = WF._judge_by_llm("问题", [{"question": "q", "answer": "a"}], llm=llm)
    assert can is False and "大陆" in reason


def test_judge_handles_code_fence():
    llm = _FakeLLM('```json\n{"can_answer": true, "reason": "ok"}\n```')
    can, _ = WF._judge_by_llm("问题", [{"question": "q", "answer": "a"}], llm=llm)
    assert can is True


def test_judge_handles_json_with_prose():
    llm = _FakeLLM('判断如下：{"can_answer": false, "reason": "缺少对象"}')
    can, _ = WF._judge_by_llm("问题", [{"question": "q", "answer": "a"}], llm=llm)
    assert can is False


def test_judge_keyword_fallback_when_not_json():
    """模型没按 JSON 输出时做关键词兜底——判读价值太高，不该因格式丢掉。"""
    llm = _FakeLLM("这些资料不能回答该问题")
    can, reason = WF._judge_by_llm("问题", [{"question": "q", "answer": "a"}], llm=llm)
    assert can is False
    assert "非结构化" in reason


def test_judge_returns_none_on_garbage():
    llm = _FakeLLM("???")
    can, reason = WF._judge_by_llm("问题", [{"question": "q", "answer": "a"}], llm=llm)
    assert can is None
    assert reason


def test_judge_returns_none_when_llm_unavailable():
    class _Dead:
        def available(self):
            return False

    can, reason = WF._judge_by_llm("问题", [{"question": "q", "answer": "a"}], llm=_Dead())
    assert can is None
    assert "不可用" in reason


def test_judge_returns_none_on_empty_sources():
    can, reason = WF._judge_by_llm("问题", [], llm=_FakeLLM("{}"))
    assert can is None
    assert "没有可判读的资料" in reason


def test_judge_returns_none_on_llm_exception():
    class _Boom:
        def available(self):
            return True

        def chat_raw(self, messages):
            raise RuntimeError("api down")

    can, reason = WF._judge_by_llm("问题", [{"question": "q", "answer": "a"}], llm=_Boom())
    assert can is None
    assert "判读调用失败" in reason


def test_judge_truncates_long_sources():
    """超长资料要截断，否则会超出上下文。"""
    long = "啊" * 5000
    llm = _FakeLLM('{"can_answer": true, "reason": "ok"}')
    can, _ = WF._judge_by_llm("问题", [{"question": "q", "answer": long}], llm=llm, max_chars=500)
    assert can is True


def test_coverage_uses_judge_result_when_available():
    """判读说"不能回答"时，即使相似度过了门槛也必须改判"不够"。"""
    p = _FakePipeline([{"question": "q", "answer": "大陆法规", "relevance": 0.55}])
    llm = _FakeLLM('{"can_answer": false, "reason": "资料仅涉及大陆法规"}')
    c = WF.local_coverage("中国与香港法规的区别", pipeline=p, llm=llm)
    assert c.enough is False
    assert c.judged is True
    assert "不足以回答" in c.reason


def test_coverage_stays_enough_when_judge_confirms():
    p = _FakePipeline([{"question": "q", "answer": "退还时限 5 日", "relevance": 0.63}])
    llm = _FakeLLM('{"can_answer": true, "reason": "资料直接说明时限"}')
    c = WF.local_coverage("保证金退还时限", pipeline=p, llm=llm)
    assert c.enough is True
    assert c.judged is True


def test_coverage_falls_back_to_similarity_when_judge_unavailable():
    """判读不可用时退回相似度结论，但必须如实标记 judged=False（可信度较低）。"""
    p = _FakePipeline([{"question": "q", "answer": "a", "relevance": 0.7}])
    c = WF.local_coverage("问题", pipeline=p, llm=_FakeLLM("???"))
    assert c.enough is True
    assert c.judged is False
    assert "未能做覆盖判读" in c.reason


# ---------------------------------------------------------------------------
# 搜索：参数与错误处理
# ---------------------------------------------------------------------------


def test_search_empty_query_returns_empty_not_error():
    """空查询属于"无结果"而非"故障"，不该抛异常。"""
    r = WF.search_web("   ")
    assert r["results"] == []
    assert r["providers_used"] == []


def test_search_raises_when_all_providers_fail(monkeypatch):
    """全部数据源失败才抛 WebSearchUnavailable（前端据此提示"配密钥"）。"""

    def boom(*a, **kw):
        raise WF.WebSearchUnavailable("未配置密钥")

    monkeypatch.setattr(WF, "_from_tavily", boom)
    monkeypatch.setattr(WF, "_from_zhipu", boom)
    with pytest.raises(WF.WebSearchUnavailable):
        WF.search_web("问题")


def test_search_partial_failure_still_returns_results(monkeypatch):
    """★ 一路挂了另一路的结果仍必须可用（两路互补是有意设计）。"""
    monkeypatch.setattr(WF, "_from_tavily", lambda q, n: (_ for _ in ()).throw(
        WF.WebSearchUnavailable("tavily 挂了")))
    monkeypatch.setattr(
        WF, "_from_zhipu",
        lambda q, n: [WF.SearchHit(title="智谱结果", url="u", content="c", score=0.7, source="zhipu")],
    )
    r = WF.search_web("问题")
    assert len(r["results"]) == 1
    assert r["providers_used"] == ["zhipu"]
    assert r["errors"] and r["errors"][0]["provider"] == "tavily"


def test_search_sorts_by_score_desc(monkeypatch):
    monkeypatch.setattr(
        WF, "_from_tavily",
        lambda q, n: (
            [
                WF.SearchHit(title="低", url="u1", content="c", score=0.3, source="tavily"),
                WF.SearchHit(title="高", url="u2", content="c", score=0.9, source="tavily"),
            ],
            "",
        ),
    )
    monkeypatch.setattr(WF, "_from_zhipu", lambda q, n: [])
    r = WF.search_web("问题", providers=("tavily",))
    assert [h["score"] for h in r["results"]] == [0.9, 0.3]


def test_search_dedupes_same_url(monkeypatch):
    """两路可能搜到同一页面，只留分数最高的一条。"""
    monkeypatch.setattr(
        WF, "_from_tavily",
        lambda q, n: (
            [WF.SearchHit(title="A", url="same", content="c", score=0.5, source="tavily")],
            "",
        ),
    )
    monkeypatch.setattr(
        WF, "_from_zhipu",
        lambda q, n: [WF.SearchHit(title="A", url="same", content="c", score=0.8, source="zhipu")],
    )
    r = WF.search_web("问题")
    assert len(r["results"]) == 1
    assert r["results"][0]["score"] == 0.8


def test_search_min_score_filters_and_reports_count(monkeypatch):
    monkeypatch.setattr(
        WF, "_from_tavily",
        lambda q, n: (
            [
                WF.SearchHit(title="高", url="u1", content="c", score=0.9, source="tavily"),
                WF.SearchHit(title="低", url="u2", content="c", score=0.1, source="tavily"),
            ],
            "",
        ),
    )
    monkeypatch.setattr(WF, "_from_zhipu", lambda q, n: [])
    r = WF.search_web("问题", providers=("tavily",), min_score=0.5)
    assert len(r["results"]) == 1
    assert r["filtered"] == 1


def test_search_keeps_scoreless_results_but_sorts_last(monkeypatch):
    """无分数的结果不丢弃（可能仍有用），但排在最后。

    这里 mock 掉 `_score_by_cosine` 让它返回 None：
    否则补算会给"无分"那条也打上分数，就测不到"无分排最后"这个行为了
    （真实场景里补算失败或内容为空时才会留下无分结果）。
    """
    monkeypatch.setattr(
        WF, "_from_tavily",
        lambda q, n: (
            [
                WF.SearchHit(title="无分", url="u1", content="c", score=None, source="tavily"),
                WF.SearchHit(title="有分", url="u2", content="c", score=0.4, source="tavily"),
            ],
            "",
        ),
    )
    monkeypatch.setattr(WF, "_from_zhipu", lambda q, n: [])
    # 补算返回 None，保持"无分"状态
    monkeypatch.setattr(WF, "_score_by_cosine", lambda q, texts: [None] * len(texts))
    r = WF.search_web("问题", providers=("tavily",))
    assert r["results"][0]["title"] == "有分"
    assert r["results"][1]["title"] == "无分"


def test_search_scores_zhipu_results_with_cosine(monkeypatch):
    """智谱不给分数，必须用余弦补算——否则前端无法显示评分。"""
    monkeypatch.setattr(WF, "_from_tavily", lambda q, n: ([], ""))
    monkeypatch.setattr(
        WF, "_from_zhipu",
        lambda q, n: [WF.SearchHit(title="A", url="u", content="一些内容", source="zhipu")],
    )
    monkeypatch.setattr(WF, "_score_by_cosine", lambda q, texts: [0.61])
    r = WF.search_web("问题", providers=("zhipu",))
    assert r["results"][0]["score"] == 0.61
    assert r["results"][0]["score_kind"] == "cosine"


def test_search_keeps_engine_score_kind(monkeypatch):
    monkeypatch.setattr(
        WF, "_from_tavily",
        lambda q, n: (
            [WF.SearchHit(title="A", url="u", content="c", score=0.7, source="tavily",
                          score_kind="engine")],
            "",
        ),
    )
    monkeypatch.setattr(WF, "_from_zhipu", lambda q, n: [])
    r = WF.search_web("问题", providers=("tavily",))
    assert r["results"][0]["score_kind"] == "engine"


def test_cosine_scoring_failure_does_not_break_search(monkeypatch):
    """算不出分数不能让整次搜索失败。"""
    monkeypatch.setattr(WF, "_from_tavily", lambda q, n: ([], ""))
    monkeypatch.setattr(
        WF, "_from_zhipu",
        lambda q, n: [WF.SearchHit(title="A", url="u", content="c", source="zhipu")],
    )

    def boom(query, texts):
        return [None] * len(texts)

    monkeypatch.setattr(WF, "_score_by_cosine", boom)
    r = WF.search_web("问题", providers=("zhipu",))
    assert len(r["results"]) == 1
    assert r["results"][0]["score"] is None


def test_unknown_provider_is_skipped(monkeypatch):
    monkeypatch.setattr(WF, "_from_tavily", lambda q, n: ([], ""))
    monkeypatch.setattr(WF, "_from_zhipu", lambda q, n: [])
    r = WF.search_web("问题", providers=("tavily", "nonexistent"))
    assert r["providers_used"] == ["tavily"]


# ---------------------------------------------------------------------------
# 可用性
# ---------------------------------------------------------------------------


def test_is_available_shape():
    s = WF.is_available()
    assert set(s) >= {"tavily", "zhipu", "any"}
    assert s["any"] == (s["tavily"] or s["zhipu"])


def test_public_api_exports():
    for n in ("local_coverage", "search_web", "WebSearchUnavailable", "LocalCoverage", "SearchHit"):
        assert n in WF.__all__


# ---------------------------------------------------------------------------
# 与 Agent 的契约
# ---------------------------------------------------------------------------


def test_agent_done_frame_carries_web_fallback():
    """Agent 的 done 帧必须带 web_fallback 字段——前端据此解释"为什么联网了"。"""
    from src.agent.core import BiddingAgent

    meta = {
        "sources": [], "web_sources": [], "tool_called": False, "tool_name": "",
        "phase_times": [], "web_fallback": {"enough": False, "reason": "本地无匹配"},
    }
    done = BiddingAgent._build_done(meta, 0.0)
    assert "web_fallback" in done
    assert done["web_fallback"]["enough"] is False


def test_agent_done_frame_defaults_web_fallback_to_none():
    from src.agent.core import BiddingAgent

    meta = {
        "sources": [], "web_sources": [], "tool_called": False, "tool_name": "",
        "phase_times": [],
    }
    assert BiddingAgent._build_done(meta, 0.0)["web_fallback"] is None

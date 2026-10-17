"""工具结果质量校验测试。

★ 这一层解决什么问题
----------------
`_execute_tool` 的 `ok` 只表示"**没抛异常、没超时**"：
    返回 0 条          → ok=True，模型拿到"未找到相关内容"就懵了
    5 条但余弦 0.31    → ok=True，实际全是噪音（无关问题实测 0.284~0.421）
    被白名单拒绝        → ok=False，但模型不知道该换哪个

前两种会让模型基于空洞或噪音继续作答，退化成"看起来很确定的错误答案"。

★ 测试重点
--------
    ① 五种状态各自判对（空/弱/不可用/缺配置/被拒）
    ② `score` 与 `relevance` 不混用（RRF 分不能判相关性——本项目已验证的结论）
    ③ 替代建议**只提本轮允许的工具**（否则模型调了被拒，白耗一轮）
    ④ 「测不了」不能说成「测出来不好」（诚实性）
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.constants import BASE_TOOL_NAMES, WEB_TOOL_NAMES  # noqa: E402
from src.agent.tool_quality import (  # noqa: E402
    ALTERNATIVES,
    DEFAULT_MIN_RELEVANCE,
    QualityVerdict,
    ResultStatus,
    assess_tool_result,
    auto_switch_hint,
    format_hint,
)
from src.tools.rag_tools import (  # noqa: E402
    _TEXT_EXA_UNAVAILABLE,
    _TEXT_GRAPH_EMPTY,
    _TEXT_GRAPH_DOWN,
    _TEXT_PG_EMPTY,
    _TEXT_RAG_EMPTY,
    _TEXT_RAG_FAILED,
    _TEXT_WEB_EMPTY,
    _TEXT_WEB_NO_KEY,
)

BASE = set(BASE_TOOL_NAMES)
FULL = BASE | set(WEB_TOOL_NAMES)


def _src(relevance=None, score=None, kind=None) -> list[dict]:
    item: dict = {"question": "q", "answer": "a"}
    if relevance is not None:
        item["relevance"] = relevance
    if score is not None:
        item["score"] = score
    if kind is not None:
        item["score_kind"] = kind
    return [item]


class TestStatusDetection:
    def test_ok_with_good_relevance(self):
        v = assess_tool_result("search_knowledge_base", "结果", _src(relevance=0.72), True, BASE)
        assert v.status is ResultStatus.OK
        assert v.usable is True
        assert v.suggested == ()

    def test_empty_result(self):
        v = assess_tool_result("search_knowledge_base", _TEXT_RAG_EMPTY, [], True, BASE)
        assert v.status is ResultStatus.EMPTY
        assert v.usable is False

    def test_unavailable(self):
        v = assess_tool_result("search_knowledge_base", _TEXT_RAG_FAILED, [], False, BASE)
        assert v.status is ResultStatus.UNAVAILABLE

    def test_misconfigured_on_failure(self):
        v = assess_tool_result("search_web", _TEXT_WEB_NO_KEY, [], True, BASE)
        assert v.status is ResultStatus.MISCONFIGURED

    def test_rejected_not_confused_with_failure(self):
        """白名单拒绝与真正失败要分开——应对方式不同。"""
        v = assess_tool_result(
            "search_web", "工具 search_web 本轮不可用。请改用其他工具", [], False, BASE
        )
        assert v.status is ResultStatus.REJECTED

    def test_graph_empty(self):
        v = assess_tool_result("search_knowledge_graph", _TEXT_GRAPH_EMPTY, [], True, BASE)
        assert v.status is ResultStatus.EMPTY

    def test_graph_down_is_unavailable(self):
        v = assess_tool_result("search_knowledge_graph", _TEXT_GRAPH_DOWN, [], False, BASE)
        assert v.status is ResultStatus.UNAVAILABLE

    def test_exa_not_ready_is_misconfigured(self):
        v = assess_tool_result("search_exa", _TEXT_EXA_UNAVAILABLE, [], True, FULL)
        assert v.status is ResultStatus.MISCONFIGURED

    def test_pg_empty(self):
        v = assess_tool_result("query_database", _TEXT_PG_EMPTY, [], True, BASE)
        assert v.status is ResultStatus.EMPTY

    def test_substantive_text_without_sources_is_ok(self):
        """合规检查返回风险清单时没有 sources，但那是有效结果。

        不能因为"没来源"就判空——那会把可用结果丢掉。
        """
        text = "检出 2 项高风险：\n1. 以本地化条件限制投标人\n2. 注册资本门槛过高"
        v = assess_tool_result("check_tender_compliance", text, [], True, BASE)
        assert v.status is ResultStatus.OK, f"实际 {v.status}"

    def test_too_short_text_without_sources_is_empty(self):
        v = assess_tool_result("check_tender_compliance", "无", [], True, BASE)
        assert v.status is ResultStatus.EMPTY


class TestRelevanceSignals:
    """★ `score` 与 `relevance` 不能混用（本项目已验证的结论）。"""

    def test_low_relevance_is_weak(self):
        v = assess_tool_result(
            "search_knowledge_base", "结果", _src(relevance=0.31), True, BASE
        )
        assert v.status is ResultStatus.WEAK
        assert v.best_relevance == 0.31

    def test_weak_is_still_usable(self):
        """有序结果比丢光好——只是要告诉模型它弱。"""
        v = assess_tool_result(
            "search_knowledge_base", "结果", _src(relevance=0.31), True, BASE
        )
        assert v.usable is True

    def test_rrf_score_alone_is_not_relevance(self):
        """★ RRF 分**不能**判相关性：任何查询都有第一名，无关问题甚至分更高。

        故只有 `score`（无 `relevance`、非 cosine 口径）时**判断不了**切题程度，
        不能因为"分数看起来不低"就判 OK——那是拿排名分当相关性用。
        """
        v = assess_tool_result(
            "search_knowledge_base", "结果", _src(score=0.09), True, BASE
        )
        assert v.status is ResultStatus.WEAK
        assert v.best_relevance is None

    def test_cosine_kind_score_is_judged(self):
        """`score_kind == "cosine"` 是本地补算，与知识库同量纲，可判阈值。"""
        good = assess_tool_result(
            "search_web", "结果", _src(score=0.7, kind="cosine"), True, FULL
        )
        assert good.status is ResultStatus.OK
        bad = assess_tool_result(
            "search_web", "结果", _src(score=0.2, kind="cosine"), True, FULL
        )
        assert bad.status is ResultStatus.WEAK

    def test_engine_kind_score_not_judged(self):
        """引擎自带口径与余弦不同，**不可横比**，故不判阈值。"""
        v = assess_tool_result(
            "search_web", "结果", _src(score=0.30, kind="engine"), True, FULL
        )
        # 不是"已知弱"，而是"测不了"
        assert v.status is ResultStatus.WEAK
        assert v.on_topic_unknown is True

    def test_picks_best_relevance_across_sources(self):
        sources = [
            {"relevance": 0.20}, {"relevance": 0.80}, {"relevance": 0.30},
        ]
        v = assess_tool_result("search_knowledge_base", "结果", sources, True, BASE)
        assert v.best_relevance == 0.80
        assert v.status is ResultStatus.OK

    def test_threshold_boundary(self):
        at = assess_tool_result(
            "search_knowledge_base", "结果", _src(relevance=DEFAULT_MIN_RELEVANCE), True, BASE
        )
        assert at.status is ResultStatus.OK, "等于阈值应算达标（>=）"
        below = assess_tool_result(
            "search_knowledge_base", "结果",
            _src(relevance=DEFAULT_MIN_RELEVANCE - 0.01), True, BASE,
        )
        assert below.status is ResultStatus.WEAK

    def test_malformed_sources_tolerated(self):
        sources = ["不是 dict", None, {"relevance": "高"}, {"relevance": 0.7}]
        v = assess_tool_result("search_knowledge_base", "结果", sources, True, BASE)
        assert v.best_relevance == 0.7
        assert v.source_count == 4


class TestKnownWeakVsUnknown:
    """★ 「测不了」不能说成「测出来不好」——那是把不知道表述成结论。"""

    def test_unknown_flagged_for_engine_score(self):
        v = assess_tool_result("search_web", "结果", _src(score=0.3, kind="engine"), True, FULL)
        assert v.on_topic_unknown is True

    def test_known_weak_not_flagged_as_unknown(self):
        v = assess_tool_result("search_knowledge_base", "结果", _src(relevance=0.2), True, BASE)
        assert v.on_topic_unknown is False

    def test_ok_never_unknown(self):
        v = assess_tool_result("search_knowledge_base", "结果", _src(relevance=0.9), True, BASE)
        assert v.on_topic_unknown is False

    def test_hints_differ_between_known_and_unknown(self):
        known = format_hint(
            assess_tool_result("search_knowledge_base", "r", _src(relevance=0.2), True, BASE),
            "search_knowledge_base",
        )
        unknown = format_hint(
            assess_tool_result("search_web", "r", _src(score=0.3, kind="engine"), True, FULL),
            "search_web",
        )
        assert "低于阈值" in known
        assert "没有可用于判断" in unknown
        assert "不可横比" in unknown


class TestAlternativeSuggestions:
    def test_suggests_only_allowed_tools(self):
        """★ 联网关闭时不能建议联网工具——模型会去调、然后被拒，白耗一轮。"""
        v = assess_tool_result("search_knowledge_base", _TEXT_RAG_EMPTY, [], True, BASE)
        assert "search_web" not in v.suggested
        assert "search_web_cn" not in v.suggested
        assert "search_knowledge_graph" in v.suggested

    def test_includes_web_when_allowed(self):
        v = assess_tool_result("search_knowledge_base", _TEXT_RAG_EMPTY, [], True, FULL)
        assert "search_web" in v.suggested

    def test_never_suggests_itself(self):
        for name in ALTERNATIVES:
            v = assess_tool_result(name, _TEXT_RAG_EMPTY, [], True, FULL)
            assert name not in v.suggested, f"{name} 不该建议自己"

    def test_ok_result_has_no_suggestions(self):
        v = assess_tool_result("search_knowledge_base", "结果", _src(relevance=0.9), True, FULL)
        assert v.suggested == ()

    def test_weak_result_has_no_suggestions(self):
        """弱结果仍可用，不该建议换工具（换了未必更好，还多一轮延迟）。"""
        v = assess_tool_result("search_knowledge_base", "r", _src(relevance=0.2), True, FULL)
        assert v.suggested == ()

    def test_all_retrieval_tools_have_alternatives(self):
        """每个检索类工具都必须有替代，否则失败时只能干等。"""
        for name in ("search_knowledge_base", "search_knowledge_graph",
                     "query_database", "search_web", "search_web_cn", "search_exa"):
            assert ALTERNATIVES.get(name), f"{name} 缺少替代方案"


class TestFormatHint:
    def test_ok_result_gets_no_hint(self):
        v = assess_tool_result("search_knowledge_base", "结果", _src(relevance=0.9), True, BASE)
        assert format_hint(v, "search_knowledge_base") == ""

    def test_unusable_with_alternative_names_them(self):
        v = assess_tool_result("search_knowledge_base", _TEXT_RAG_EMPTY, [], True, BASE)
        hint = format_hint(v, "search_knowledge_base")
        assert "search_knowledge_graph" in hint
        assert "不要重复调用" in hint

    def test_unusable_without_alternative_tells_model_to_be_honest(self):
        """没有替代数据源时，要求模型**如实说明依据不足**而不是编造。"""
        v = assess_tool_result("search_knowledge_graph", _TEXT_GRAPH_EMPTY, [], True,
                               {"search_knowledge_graph"})
        hint = format_hint(v, "search_knowledge_graph")
        assert "没有其他可替代的数据源" in hint
        assert "不要凭推测编造" in hint

    def test_weak_result_warns_but_allows_use(self):
        v = assess_tool_result("search_knowledge_base", "r", _src(relevance=0.2), True, BASE)
        hint = format_hint(v, "search_knowledge_base")
        assert "不能作为唯一依据" in hint

    def test_hint_is_appended_not_replacing(self):
        """提示是**追加**在原文案之后，不是替换它。

        `format_hint` 只产出提示本身，拼接由 `_validate_result` 负责
        （`text + format_hint(...)`）。模型需要同时看到原始说明与质量提示：
        原始说明告诉它"没找到"，提示告诉它"该换哪个工具"。
        """
        v = assess_tool_result("search_knowledge_base", _TEXT_RAG_EMPTY, [], True, BASE)
        hint = format_hint(v, "search_knowledge_base")
        combined = _TEXT_RAG_EMPTY + hint
        assert combined.startswith(_TEXT_RAG_EMPTY), "原文案必须在最前"
        assert "数据质量提示" in combined
        assert combined.index(_TEXT_RAG_EMPTY) < combined.index("数据质量提示")


class TestUserFacingHint:
    """用户不需要知道内部换了数据源——只在无替代时才提示。"""

    def test_no_user_hint_when_alternative_exists(self):
        v = assess_tool_result("search_knowledge_base", _TEXT_RAG_EMPTY, [], True, BASE)
        assert auto_switch_hint(v, "search_knowledge_base") == ""

    def test_no_user_hint_when_usable(self):
        v = assess_tool_result("search_knowledge_base", "r", _src(relevance=0.9), True, BASE)
        assert auto_switch_hint(v, "search_knowledge_base") == ""

    def test_user_hint_when_no_alternative(self):
        v = assess_tool_result("search_knowledge_graph", _TEXT_GRAPH_EMPTY, [], True,
                               {"search_knowledge_graph"})
        hint = auto_switch_hint(v, "search_knowledge_graph")
        assert hint
        assert "search_" not in hint, "给用户的提示不该暴露内部工具名"


class TestSpinDetection:
    """★ 打转检测：单靠结果质量判定抓不到的缺口。

    实测一个问题触发了 **6 次 `search_knowledge_base`**，而每次返回的来源
    **主题相关、相关性也不低**（都是电子招投标/数字证书），故质量判定是 `ok`、
    不产生任何提示——模型一直在同一个坑里打转。

    问题不在"结果质量差"，而在"**结果切题但不回答问题**"：
    库里讲的是该领域的一般规定，用户问的是一个具体颗粒度的问题。
    这种落差测不出来（分数确实不低），能测出来的是**行为模式**。
    """

    def test_no_hint_before_threshold(self):
        from src.agent.react_loop import SPIN_REPEAT_THRESHOLD, ReActMixin

        meta: dict = {}
        for i in range(1, SPIN_REPEAT_THRESHOLD):
            assert ReActMixin._spin_hint("search_knowledge_base", meta, BASE) == "", (
                f"第 {i} 次不该提示"
            )

    def test_hint_at_threshold(self):
        from src.agent.react_loop import SPIN_REPEAT_THRESHOLD, ReActMixin

        meta: dict = {}
        hint = ""
        for _ in range(SPIN_REPEAT_THRESHOLD):
            hint = ReActMixin._spin_hint("search_knowledge_base", meta, BASE)
        assert hint, "到达阈值必须提示"
        assert "检索停滞" in hint
        assert "不要再用相同或相近的问法" in hint

    def test_spin_hint_respects_allowed(self):
        """联网关闭时不该建议联网工具（与质量提示同一约束）。"""
        from src.agent.react_loop import SPIN_REPEAT_THRESHOLD, ReActMixin

        meta: dict = {}
        hint = ""
        for _ in range(SPIN_REPEAT_THRESHOLD):
            hint = ReActMixin._spin_hint("search_knowledge_base", meta, BASE)
        assert "search_web" not in hint
        assert "search_knowledge_graph" in hint

    def test_spin_hint_includes_web_when_allowed(self):
        from src.agent.react_loop import SPIN_REPEAT_THRESHOLD, ReActMixin

        meta: dict = {}
        hint = ""
        for _ in range(SPIN_REPEAT_THRESHOLD):
            hint = ReActMixin._spin_hint("search_knowledge_base", meta, FULL)
        assert "search_web" in hint

    def test_counts_are_per_tool(self):
        """不同工具各自计数——换了工具就不该算在同一个坑里。"""
        from src.agent.react_loop import ReActMixin

        meta: dict = {}
        for name in ("search_knowledge_base", "search_knowledge_graph", "query_database"):
            assert ReActMixin._spin_hint(name, meta, BASE) == ""
        assert meta["_tool_call_counts"] == {
            "search_knowledge_base": 1,
            "search_knowledge_graph": 1,
            "query_database": 1,
        }

    def test_asks_for_honesty_not_padding(self):
        """提示要明确要求"如实说明没有"，而不是用同主题内容充数。"""
        from src.agent.react_loop import SPIN_REPEAT_THRESHOLD, ReActMixin

        meta: dict = {}
        hint = ""
        for _ in range(SPIN_REPEAT_THRESHOLD):
            hint = ReActMixin._spin_hint("search_knowledge_base", meta, BASE)
        assert "如实说明" in hint
        assert "充数" in hint

    def test_no_alternative_still_asks_to_stop(self):
        """没有可替代工具时，仍要让它停止重复调用。"""
        from src.agent.react_loop import SPIN_REPEAT_THRESHOLD, ReActMixin

        meta: dict = {}
        hint = ""
        for _ in range(SPIN_REPEAT_THRESHOLD):
            hint = ReActMixin._spin_hint("search_knowledge_graph", meta,
                                         {"search_knowledge_graph"})
        assert hint
        assert "不要再用相同或相近的问法" in hint


class TestVerdictDataclass:
    def test_defaults(self):
        v = QualityVerdict(status=ResultStatus.OK)
        assert v.source_count == 0
        assert v.best_relevance is None
        assert v.reasons == []
        assert v.suggested == ()

    def test_reasons_not_shared_between_instances(self):
        """字段默认值用 `field(default_factory=...)`——共享 list 会串味。"""
        a = QualityVerdict(status=ResultStatus.OK)
        b = QualityVerdict(status=ResultStatus.OK)
        a.reasons.append("x")
        assert b.reasons == [], "默认 list 被共享了"


class TestNeverRaises:
    """质量校验是**增强**，不能因为自身异常让检索结果丢失。"""

    @pytest.mark.parametrize(
        "text,sources,ok",
        [
            (None, None, True),
            ("", [], False),
            ("x", "不是列表", True),
            ("x", [{"relevance": None}], True),
            ("x", [{"relevance": float("nan")}], True),
            ("x", [{"relevance": float("inf")}], True),
        ],
    )
    def test_malformed_inputs_do_not_raise(self, text, sources, ok):
        v = assess_tool_result("search_knowledge_base", text, sources, ok, BASE)
        assert isinstance(v.status, ResultStatus)

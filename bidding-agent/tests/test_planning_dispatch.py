"""Plan-and-Solve **按复杂度分流**的测试（`is_compound_question` + `_planning_enabled`）。

★ 为什么要分流
============
规划轮是一次额外的非流式 LLM 调用（实测 2~5 秒）：
    · 单源问题只需 1~2 次工具调用、本来就不打转 → 多付这几秒是净亏
    · 复合问题（5~6 次调用、2/5 撞轮次上限）才有收益
故"一律开"与"一律关"都不对。判据必须只看**结构特征**（不能再调一次 LLM，
否则与省时间的初衷矛盾），且必须**保守**：宁可漏判（退回现状），不可误判
（给简单问题白等几秒、还把回答拖长）。
"""

import pytest

from src.agent.core import BiddingAgent
from src.agent.planning import is_compound_question


class TestIsCompoundQuestion:
    @pytest.mark.parametrize("question", [
        # 两个问号
        "投标保证金上限是多少？什么时候退还？",
        # 显式并列
        "对比一下中建三局和中铁十一局的中标情况",
        "请分别说明资格预审和资格后审的区别",
        "查一下联合体的规定，以及数据库里有没有案例",
        # 一句话两件事：不同类疑问词各命中一个
        "投标保证金最多能收多少，退还流程怎么走",
        "中标候选人公示期是几天，公示期间能不能提异议",
    ])
    def test_compound_questions_are_detected(self, question):
        assert is_compound_question(question) is True, f"应判为复合：{question}"

    @pytest.mark.parametrize("question", [
        "投标保证金上限是多少？",
        "评标委员会人数要求",
        "中标通知书发出后多久签订合同？",
        "履约保证金最高能收多少",
        "废标之后可以重新招标吗？",
        "资格审查由谁负责？",
    ])
    def test_single_source_questions_are_not_flagged(self, question):
        assert is_compound_question(question) is False, f"不该判为复合：{question}"

    def test_empty_and_none_are_safe(self):
        assert is_compound_question("") is False
        assert is_compound_question("   ") is False
        assert is_compound_question(None) is False  # type: ignore[arg-type]

    def test_english_question_mark_counts(self):
        assert is_compound_question("上限多少? 怎么退?") is True


class TestPlanningEnabled:
    """`AGENT_PLANNING` 的三态语义：默认关 / auto 分流 / 强制开。"""

    @pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "随便写的"])
    def test_default_and_invalid_are_off(self, monkeypatch, value):
        monkeypatch.setenv("AGENT_PLANNING", value)
        assert BiddingAgent._planning_enabled("复合问题？还有一个？") is False

    @pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
    def test_explicit_on_always_plans(self, monkeypatch, value):
        monkeypatch.setenv("AGENT_PLANNING", value)
        assert BiddingAgent._planning_enabled("简单问题") is True

    def test_auto_plans_only_for_compound(self, monkeypatch):
        monkeypatch.setenv("AGENT_PLANNING", "auto")
        assert BiddingAgent._planning_enabled("投标保证金上限是多少？什么时候退？") is True
        assert BiddingAgent._planning_enabled("投标保证金上限是多少？") is False

    def test_unset_keeps_current_default_off(self, monkeypatch):
        """★ 未设环境变量时**保持关闭**：auto 尚未被 A/B 证实，不擅自改默认行为。"""
        monkeypatch.delenv("AGENT_PLANNING", raising=False)
        assert BiddingAgent._planning_enabled("复合问题？还有一个？") is False

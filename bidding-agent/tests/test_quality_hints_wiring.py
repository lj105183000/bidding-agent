"""局限 6 的回归测试：`auto_switch_hint` 真正接到了 done 帧。

★ 锁住的行为
==========
1. `_note_user_hint` 把提示累积到旁路
2. `take_user_hints` 取出并**清空**
3. 去重（同一提示只出现一次）
4. done 帧带 `quality_hints` 字段
5. **跨请求不混入**：`_chat_events` 入口会 reset
6. 观测失败**不能影响问答**
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent import react_loop as RL  # noqa: E402
from src.agent.core import BiddingAgent, _take_user_hints_for_done  # noqa: E402
from src.agent.tool_quality import assess_tool_result  # noqa: E402

#: ★ 只放一个工具：没有**替代工具**时 uto_switch_hint 才会给用户提示。
#:
#: 实测语义（	ool_quality.auto_switch_hint）：
#:     有 suggested（有别的工具可换） → 返回 ""（换就完了，没必要打扰用户）
#:     没有 suggested 且不可用       → 返回提示
#: 故构造测试必须让 allowed 里**没有**可替代项。
ONLY_SELF = {"search_knowledge_base"}


def _bad_verdict():
    """空结果 + 无替代工具 → 会产生「给用户看的说明」。"""
    return assess_tool_result(
        "search_knowledge_base", "", [], ok=False, allowed=ONLY_SELF
    )


@pytest.fixture(autouse=True)
def _clean():
    RL.reset_user_hints()
    yield
    RL.reset_user_hints()


class TestAccumulateAndClear:
    def test_note_accumulates(self):
        RL._note_user_hint(_bad_verdict(), "search_knowledge_base")
        got = RL.take_user_hints()
        assert got, "应累积到至少一条提示"
        assert "依据" in got[0] or "不可用" in got[0]

    def test_take_clears(self):
        RL._note_user_hint(_bad_verdict(), "search_knowledge_base")
        first = RL.take_user_hints()
        assert first
        assert RL.take_user_hints() == [], "take 之后必须清空（否则会无限累积）"

    def test_duplicates_are_collapsed(self):
        """同一提示重复记只留一条 —— 否则用户看到重复的噪音。"""
        for _ in range(3):
            RL._note_user_hint(_bad_verdict(), "search_knowledge_base")
        got = RL.take_user_hints()
        assert len(got) == 1, f"应去重为 1 条，实为 {len(got)}: {got}"

    def test_reset_clears_without_returning(self):
        RL._note_user_hint(_bad_verdict(), "search_knowledge_base")
        RL.reset_user_hints()
        assert RL.take_user_hints() == []


class TestObservationNeverBreaks:
    def test_note_swallows_errors(self, monkeypatch):
        """`_note_user_hint` 内部抛异常不能传出去（观测不能影响问答）。"""

        def boom(*a, **k):
            raise RuntimeError("hint 炸了")

        monkeypatch.setattr("src.agent.tool_quality.auto_switch_hint", boom)
        RL._note_user_hint(_bad_verdict(), "search_knowledge_base")  # 不应抛
        assert RL.take_user_hints() == []

    def test_done_helper_swallows_errors(self, monkeypatch):
        """`_take_user_hints_for_done` 也不能抛 —— done 帧必须产得出来。"""

        def boom():
            raise RuntimeError("take 炸了")

        monkeypatch.setattr("src.agent.react_loop.take_user_hints", boom)
        assert _take_user_hints_for_done() == []

    def test_done_helper_returns_list(self):
        assert isinstance(_take_user_hints_for_done(), list)


class TestDoneFrameCarriesField:
    def test_build_done_has_quality_hints(self):
        """done 帧契约必须有 `quality_hints`（空 list 表示没问题）。"""
        meta = {
            "sources": [], "web_sources": [], "tool_called": False,
            "tool_name": "", "phase_times": [], "web_fallback": None,
        }
        done = BiddingAgent._build_done(meta, 0.0)
        assert "quality_hints" in done
        assert isinstance(done["quality_hints"], list)

    def test_build_done_surfaces_accumulated_hints(self):
        RL._note_user_hint(_bad_verdict(), "search_knowledge_base")
        meta = {
            "sources": [], "web_sources": [], "tool_called": False,
            "tool_name": "", "phase_times": [], "web_fallback": None,
        }
        done = BiddingAgent._build_done(meta, 0.0)
        assert done["quality_hints"], "累积的提示应出现在 done 帧里"
        # 且取走后必须清空（下一轮不该再看到）
        assert RL.take_user_hints() == []


class TestNoCrossRequestBleed:
    def test_chat_clears_previous_hints(self):
        """★ 核心：上一轮遗留的提示不能被下一轮带出。

        `take_user_hints()` 只在组装 done 帧时清空；若上一轮的 done 帧
        没被产出/消费（异常路径），提示会留到下一轮。故 `_chat_events`
        入口必须无条件 reset。
        """
        # 人为污染（模拟"上一轮 done 帧没被消费"）
        RL._USER_HINTS.append("上一轮的旧提示")
        assert RL.take_user_hints() == ["上一轮的旧提示"]

        # 再污染一次，然后跑一轮真实问答
        RL._USER_HINTS.append("上一轮的旧提示")
        agent = BiddingAgent()
        result = agent.chat("投标保证金比例上限是多少？")
        assert "上一轮的旧提示" not in (result.get("quality_hints") or []), (
            "旧的提示串到了新一轮 —— `_chat_events` 入口没有 reset"
        )

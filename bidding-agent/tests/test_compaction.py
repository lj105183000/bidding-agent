"""历史压缩（滚动摘要）单测。

要解决的缺口：`_truncate_history` 只保留最近 5 轮，**更早的信息永久丢失**。
真实场景：
    第 1 轮：在查"华中电力建设公司"的中标情况
    第 9 轮：刚才那家公司再查一下省份分布   ← 第 1 轮已被丢掉
             模型不知道"刚才那家"是谁，且**不会说自己忘了**，只会猜或反问。

测试重点：
    1. **触发时机**：短对话零成本；长对话压缩频率随长度下降（滚动而非每轮）
    2. **判断与行为一致**：`needs_compaction` 说"该压"时必须真的压
    3. **抗失败**：压缩是增益项，坏了要退化成硬截断而不是让问答失败
    4. **会话隔离**：不同 session_id 的摘要不能互相串
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent import compaction as C  # noqa: E402


class FakeClient:
    """记录调用次数的假 LLM。"""

    def __init__(self, reply: str = "早期对话摘要内容"):
        self.reply = reply
        self.calls = 0

    def chat_raw(self, messages, tools=None, **kwargs):
        self.calls += 1
        return self.reply


def mkhist(rounds: int, prefix: str = "") -> list[dict]:
    """造 `rounds` 轮对话（每轮 user + assistant）。"""
    out: list[dict] = []
    for i in range(rounds):
        out.append({"role": "user", "content": f"{prefix}第{i + 1}轮问题"})
        out.append({"role": "assistant", "content": f"{prefix}第{i + 1}轮回答"})
    return out


@pytest.fixture(autouse=True)
def _clean_state():
    """每个用例前后都清空进程内状态，避免用例间互相污染。"""
    C.clear_all()
    yield
    C.clear_all()


class TestTriggerTiming:
    def test_short_history_no_compaction(self):
        """★ 短对话零成本——不压缩、不调 LLM。

        这是默认开启的前提：若每次问答都压，就变成"每轮多一次 LLM 调用"，
        那与 `AGENT_PLANNING`（默认关）的成本结构一样，不该默认开。
        """
        client = FakeClient()
        for rounds in (1, 5, 9):
            r = C.compact_history(mkhist(rounds), client, session_id="s")
            assert r.compacted is False
        assert client.calls == 0

    def test_triggers_after_threshold(self):
        client = FakeClient()
        r = C.compact_history(mkhist(C.COMPACT_TRIGGER_ROUNDS + 1), client, session_id="s")
        assert r.compacted is True
        assert client.calls == 1
        assert r.summary

    def test_keeps_recent_rounds_uncompacted(self):
        """最近的若干轮**不进摘要**——它们是当前话题的上下文，摘要会丢细节。"""
        client = FakeClient()
        rounds = C.COMPACT_TRIGGER_ROUNDS + 1
        r = C.compact_history(mkhist(rounds), client, session_id="s")
        # 覆盖到"总消息数 - 保留窗口"
        expected = rounds * 2 - C.COMPACT_KEEP_RECENT_ROUNDS * 2
        assert r.covered == expected


class TestRollingBehavior:
    def test_does_not_recompress_every_turn(self):
        """★ 滚动策略的核心：不满足条件时复用已有摘要，不重复压缩。

        朴素做法（每轮重压）会让摘要**反复被摘要**，信息层层衰减——
        第 10 轮时第 1 轮的内容已被压了 10 次。
        """
        client = FakeClient()
        rounds = C.COMPACT_TRIGGER_ROUNDS + 1
        C.compact_history(mkhist(rounds), client, session_id="s")
        first_calls = client.calls
        # 多轮加一轮，仍不该重压
        for extra in range(1, 4):
            C.compact_history(mkhist(rounds + extra), client, session_id="s")
        assert client.calls == first_calls, "不该每轮都重压"

    def test_recompacts_when_gap_grows(self):
        """水位线之后的未覆盖内容攒够时，才再一次压缩。"""
        client = FakeClient()
        rounds = C.COMPACT_TRIGGER_ROUNDS + 1
        C.compact_history(mkhist(rounds), client, session_id="s")
        assert client.calls == 1
        # 加到远超水位线
        C.compact_history(mkhist(rounds * 2 + 2), client, session_id="s")
        assert client.calls == 2

    def test_watermark_never_goes_backwards(self):
        """水位线只能前进。回退会把已并入摘要的内容再压一次，重复计入。"""
        client = FakeClient()
        big = mkhist(C.COMPACT_TRIGGER_ROUNDS * 3)
        r1 = C.compact_history(big, client, session_id="s")
        # 用更短的历史再调（模拟前端传了截断过的历史）
        r2 = C.compact_history(mkhist(2), client, session_id="s")
        assert r2.covered >= r1.covered or r2.compacted is False
        # 关键：不该因历史变短而把水位线推回去
        assert C._get_state("s").covered >= r1.covered


class TestConsistency:
    """★ `needs_compaction` 与 `compact_history` 必须一致。

    实测踩过两次：一次是它俩各自实现判据，另一次是调用方传 covered=0
    而实际用会话水位——结果"说要压缩却没压"。
    """

    @pytest.mark.parametrize("rounds", [1, 5, 9, 10, 11, 12, 14, 16, 18, 25, 40])
    def test_prediction_matches_action(self, rounds):
        client = FakeClient()
        hist = mkhist(rounds)
        predicted = C.needs_compaction(hist, session_id="s")
        actual = C.compact_history(hist, client, session_id="s").compacted
        assert predicted == actual, (
            f"{rounds} 轮：预测 {predicted} 但实际 {actual}"
        )

    def test_call_frequency_grows_sublinearly(self):
        """★ 压缩次数必须**远小于**轮数，否则"定期压缩"就退化成"每轮压缩"。"""
        client = FakeClient()
        for rounds in range(1, 41):
            C.compact_history(mkhist(rounds), client, session_id="s")
        assert client.calls <= 6, f"40 轮压了 {client.calls} 次，过于频繁"


class TestResilience:
    """压缩是**增益项**：失败最多退化成现有的硬截断，不该让问答失败。"""

    def test_client_none_returns_not_compacted(self):
        r = C.compact_history(mkhist(20), None, session_id="s")
        assert r.compacted is False
        assert r.reason

    def test_client_exception_degrades(self):
        class Boom:
            def chat_raw(self, *a, **k):
                raise RuntimeError("LLM 挂了")

        r = C.compact_history(mkhist(20), Boom(), session_id="s")
        assert r.compacted is False
        assert "失败" in r.reason

    def test_empty_summary_degrades(self):
        client = FakeClient(reply="   ")
        r = C.compact_history(mkhist(20), client, session_id="s")
        assert r.compacted is False
        assert "空" in r.reason

    def test_summary_is_truncated(self):
        client = FakeClient(reply="长" * 5000)
        r = C.compact_history(mkhist(20), client, session_id="s")
        assert len(r.summary) <= C.MAX_SUMMARY_CHARS

    def test_failed_compaction_keeps_previous_summary(self):
        """压缩失败时不能把已有摘要弄丢——那等于回退到"更早内容全丢"。"""
        good = FakeClient(reply="第一次的摘要")
        C.compact_history(mkhist(20), good, session_id="s")

        class Boom:
            def chat_raw(self, *a, **k):
                raise RuntimeError("挂了")

        r = C.compact_history(mkhist(40), Boom(), session_id="s")
        assert r.summary == "第一次的摘要"

    def test_malformed_history_entries_ignored(self):
        client = FakeClient()
        hist = mkhist(20) + ["裸字符串", None, {"role": "system", "content": "x"}]
        r = C.compact_history(hist, client, session_id="s")
        assert r.compacted is True


class TestSessionIsolation:
    def test_different_sessions_have_separate_state(self):
        """★ 不同会话的摘要不能串。

        串了的后果**不会报错**：新会话凭空多出一段不属于它的早期摘要，
        而模型会把它当作真实历史，于是提到用户从没说过的话题。
        """
        client = FakeClient(reply="会话A的摘要")
        C.compact_history(mkhist(20), client, session_id="A")
        assert C._get_state("A").summary == "会话A的摘要"
        assert C._get_state("B").summary == ""

    def test_reset_session_clears_only_that_session(self):
        client = FakeClient()
        C.compact_history(mkhist(20), client, session_id="A")
        C.compact_history(mkhist(20), client, session_id="B")
        C.reset_session("A")
        assert C._get_state("A").summary == ""
        assert C._get_state("B").summary != ""

    def test_reset_nonexistent_session_is_safe(self):
        C.reset_session("不存在")  # 不该抛
        C.reset_session("")        # 空 ID 也不该抛

    def test_session_cap_evicts_oldest(self, monkeypatch):
        """长跑进程不能把内存吃光——超限时淘汰最旧的会话状态。"""
        monkeypatch.setattr(C, "MAX_TRACKED_SESSIONS", 3)
        client = FakeClient()
        for i in range(5):
            C.compact_history(mkhist(20), client, session_id=f"s{i}")
        assert len(C._store.sessions) <= 3
        # 最新的在
        assert C._get_state("s4").summary != ""


class TestBuildNote:
    def test_marks_as_summary_and_warns_about_loss(self):
        """★ 注入文本要标明"这是摘要"并提示可能失真。

        不提示的话，模型会拿摘要里的模糊记忆当作精确依据作答——
        而摘要必然会丢细节（数值、条款号）。
        """
        note = C.build_compaction_note(C.Compaction(summary="早期聊了投标保证金", compacted=True))
        assert "早期对话摘要" in note
        assert "可能损失细节" in note
        assert "重新检索确认" in note

    def test_empty_when_no_summary(self):
        assert C.build_compaction_note(C.Compaction()) == ""
        assert C.build_compaction_note(C.Compaction(compacted=False, summary="")) == ""

    def test_to_dict_shape(self):
        d = C.Compaction(summary="x", covered=4, compacted=True, reason="r").to_dict()
        assert set(d) == {"compacted", "covered", "summary_len", "reason"}
        assert d["summary_len"] == 1


class TestEnabledSwitch:
    def test_default_enabled(self, monkeypatch):
        """默认开启——它只在长会话触发，短对话零成本。

        与 `AGENT_PLANNING`（默认关）相反的取舍依据：没有固定成本的能力默认开。
        """
        monkeypatch.delenv("AGENT_COMPACT", raising=False)
        assert C.compaction_enabled() is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", "OFF", "False"])
    def test_explicit_disable(self, monkeypatch, value):
        monkeypatch.setenv("AGENT_COMPACT", value)
        assert C.compaction_enabled() is False

    @pytest.mark.parametrize("value", ["1", "true", "yes", "on", "bogus"])
    def test_other_values_enable(self, monkeypatch, value):
        """拼错环境变量不该把功能关掉——与其它开关一致，只认显式关闭值。"""
        monkeypatch.setenv("AGENT_COMPACT", value)
        assert C.compaction_enabled() is True


class TestSwitchActuallyGatesTheFeature:
    """★ 开关必须真的能关掉功能——不只是"有个函数返回 False"。

    实测踩坑（真实 bug）：`core.py` 里曾写成

        if compaction_enabled:      # ← 函数对象，恒为真

    后果是 `AGENT_COMPACT=0` **完全失效**、永远压缩。
    更麻烦的是它**不报错、也不改变行为**（恰好总是压缩），
    所以普通单测全绿——只有专门断言"关掉后 `_compact_history` 不被调用"
    才能发现。

    这类"假开关"是最值得写成测试的一类缺陷：没人会去怀疑一个看起来存在的开关。
    """

    @staticmethod
    def _hist():
        return mkhist(C.COMPACT_TRIGGER_ROUNDS + 4)

    def test_disabled_does_not_call_compact_history(self, monkeypatch):
        from src.agent.core import BiddingAgent

        called = {"n": 0}
        original = BiddingAgent._compact_history

        def spy(*args, **kwargs):
            called["n"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(BiddingAgent, "_compact_history", staticmethod(spy))
        monkeypatch.setenv("AGENT_COMPACT", "0")

        agent = BiddingAgent()
        agent._build_context("问题", self._hist(), False, "", False, {}, "s", FakeClient())

        assert called["n"] == 0, "开关关了却仍调用了压缩——开关是假的"

    def test_enabled_does_call_compact_history(self, monkeypatch):
        """对照：开着时必须调用。否则上面的测试可能因别的原因通过。"""
        from src.agent.core import BiddingAgent

        called = {"n": 0}
        original = BiddingAgent._compact_history

        def spy(*args, **kwargs):
            called["n"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(BiddingAgent, "_compact_history", staticmethod(spy))
        monkeypatch.delenv("AGENT_COMPACT", raising=False)

        agent = BiddingAgent()
        agent._build_context("问题", self._hist(), False, "", False, {}, "s", FakeClient())

        assert called["n"] == 1, "开关开着却没调用压缩"

    def test_disabled_injects_no_summary_note(self, monkeypatch):
        """关掉后 system 里不该出现摘要文本。"""
        from src.agent.core import BiddingAgent

        monkeypatch.setenv("AGENT_COMPACT", "0")
        messages = BiddingAgent()._build_context(
            "问题", self._hist(), False, "", False, {}, "s", FakeClient()
        )
        assert "早期对话摘要" not in messages[0]["content"]

    def test_enabled_injects_summary_note(self, monkeypatch):
        from src.agent.core import BiddingAgent

        monkeypatch.delenv("AGENT_COMPACT", raising=False)
        messages = BiddingAgent()._build_context(
            "问题", self._hist(), False, "", False, {}, "s", FakeClient()
        )
        assert "早期对话摘要" in messages[0]["content"]

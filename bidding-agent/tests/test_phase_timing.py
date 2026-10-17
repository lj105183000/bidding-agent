"""阶段耗时字段的测试。

★ 为什么只测**不变量**而不测具体毫秒
==============================
毫秒值取决于机器、网络、模型响应，测具体数字必然 flaky。
真正要锁住的是这些不变量——它们出错会直接导致前端展示错乱：

    ① 每条阶段是 3 元组 `(名称, 该阶段耗时, 该阶段起始偏移)`
    ② **偏移单调不减**（前端要按它排序/画时间轴；乱序会画出错乱的时间轴）
    ③ `phase_total_ms` == 各阶段耗时之和（前端会与总耗时做差看"其他"）
    ④ 偏移 ≤ 总耗时（否则说明基准取错，前端会显示"@999s"这种荒谬值）
    ⑤ **偏移的基准是 `meta["t_start"]`**（曾因基准取错导致偏移变成 34 秒的
       monotonic 绝对值——见下方 `test_offset_uses_meta_t_start_as_base`）

★ 第 ⑤ 条是有来历的真实 bug
========================
第一版 `_mark_phase` 写成 `base = meta.get("t_start") or t0`，
当 `t_start` 缺失时回落到 `t0`，偏移算出来**恒为 0**——看着正常但没信息量。
而更早一版我用 `@staticmethod` 装饰器插错位置，还让 `_build_done` 丢了装饰器、
26 个测试一起挂。故这两条都锁住。
"""

from __future__ import annotations

import json
import time

import pytest

from src.agent import BiddingAgent


class _ScriptedLLM:
    """最小替身：一轮决策不调工具 + 一段回答。"""

    def __init__(self, delay: float = 0.0):
        self.delay = delay
        self.last_reasoning = ""

    def chat_raw(self, messages, tools=None, **kwargs):
        if self.delay:
            time.sleep(self.delay)
        return "信息足够，直接回答。"

    def chat_stream(self, messages, **kwargs):
        if self.delay:
            time.sleep(self.delay)
        yield "回答正文"


def _done_of(client) -> dict:
    events = list(BiddingAgent(llm_client=client)._chat_events("测试问题"))
    dones = [e[1] for e in events if e[0] == "done"]
    assert dones, f"未产出 done 帧，事件类型: {[e[0] for e in events]}"
    return dones[0]


class TestPhaseTimesShape:
    def test_each_entry_is_triple(self):
        """① 每条阶段必须是 3 元组：名称、耗时、起始偏移。"""
        done = _done_of(_ScriptedLLM())
        phases = done["phase_times"]
        assert phases, "done 帧应含至少一个阶段"
        for item in phases:
            assert len(item) == 3, f"阶段项应为 3 元组，实得 {item!r}"
            name, ms, offset = item
            assert isinstance(name, str) and name, f"阶段名无效: {name!r}"
            assert isinstance(ms, int) and ms >= 0, f"耗时无效: {ms!r}"
            assert isinstance(offset, int) and offset >= 0, f"偏移无效: {offset!r}"

    def test_offsets_are_monotonic(self):
        """② 偏移单调不减——前端按它画时间轴，乱序会画出错乱的图。"""
        done = _done_of(_ScriptedLLM(delay=0.02))
        offsets = [p[2] for p in done["phase_times"]]
        assert offsets == sorted(offsets), f"偏移非单调: {offsets}"

    def test_phase_total_matches_sum(self):
        """③ `phase_total_ms` 必须等于各阶段耗时之和。"""
        done = _done_of(_ScriptedLLM(delay=0.01))
        phases = done["phase_times"]
        assert done["phase_total_ms"] == sum(p[1] for p in phases)

    def test_offset_within_elapsed(self):
        """④ 偏移不能超过总耗时（否则前端显示"@999s"这种荒谬值）。"""
        done = _done_of(_ScriptedLLM(delay=0.02))
        elapsed = done["elapsed_ms"]
        for name, _ms, offset in done["phase_times"]:
            assert offset <= elapsed + 50, (
                f"阶段 {name!r} 偏移 {offset}ms 超过总耗时 {elapsed}ms"
            )

    def test_offset_starts_near_zero(self):
        """首个阶段的偏移应接近 0（它是第一阶段，几乎紧接着流程开始）。"""
        done = _done_of(_ScriptedLLM())
        first_offset = done["phase_times"][0][2]
        assert first_offset < 2000, (
            f"首个阶段偏移 {first_offset}ms 过大——"
            f"说明基准取错（可能用了 monotonic 绝对值而非相对基准）"
        )

    def test_json_serializable(self):
        """原样进 SSE 帧，必须可 JSON 序列化。"""
        json.dumps(_done_of(_ScriptedLLM())["phase_times"])


class TestOffsetUsesMetaTStartAsBase:
    """⑤ 偏移基准必须是 `meta["t_start"]`。

    ★ 这是有来历的回归测试
    ------------------
    另一个校准失效的写法是 `base = meta.get("t_start") or t0`：
    `t_start` 缺失时回落到 `t0`，于是偏移 **恒为 0**——不报错、看着正常，
    但前端时间轴会全部叠在起点，丧失全部信息量。
    故这里直接断言"偏移确实相对流程起点，而不是相对阶段自身"。
    """

    def test_offset_not_always_zero(self):
        """若基准错写成 `t0`，所有偏移都会是 0。"""
        done = _done_of(_ScriptedLLM(delay=0.05))
        offsets = [p[2] for p in done["phase_times"]]
        assert any(o > 0 for o in offsets), (
            f"所有偏移都是 0（{offsets}）——基准可能被错写成阶段自身的 t0"
        )

    def test_mark_phase_uses_meta_t_start(self):
        """直接单元测 `_mark_phase`：偏移必须相对 `meta["t_start"]`。

        ★ 参数为什么用真实时间而不是构造的整数
        --------------------------------
        `_mark_phase` 内部读的是**真实** `time.monotonic()`（记录结束时刻），
        只有 `t0`/`t_start` 是外部传入的。第一版测试传 `t_start=1000.0`、
        `t0=1005.0` 这种构造值，实得耗时是当时的 monotonic 绝对值（8 百万毫秒）。
        故这里以真实 `now` 为锚，构造"1 秒前开始"的场景。
        """
        now = time.monotonic()
        meta = {"phase_times": [], "t_start": now - 10.0}
        # 该阶段在 1 秒前开始 → 耗时 ≈1000ms，偏移 ≈9000ms
        BiddingAgent._mark_phase(meta, "测试阶段", now - 1.0)
        name, ms, offset = meta["phase_times"][0]
        assert name == "测试阶段"
        assert ms == pytest.approx(1000, abs=120), f"耗时≈1000ms，实得 {ms}"
        assert offset == pytest.approx(9000, abs=200), f"偏移≈9000ms，实得 {offset}"

    def test_mark_phase_falls_back_to_t0_when_base_missing(self):
        """`t_start` 缺失时不崩，偏移按 0 处理（降级而非异常）。"""
        meta = {"phase_times": []}
        BiddingAgent._mark_phase(meta, "无基准", time.monotonic())
        assert meta["phase_times"][0][2] == 0


class TestMarkPhaseIsStatic:
    """`_mark_phase` 与 `_build_done` 都必须是 staticmethod。

    ★ 为什么专门测这个
    --------------
    实测事故：用"在 `def _build_done` 前插入新方法"的方式加 `_mark_phase` 时，
    **`_build_done` 上面的 `@staticmethod` 装饰器被留在了新方法上方**，
    于是出现两个装饰器叠在 `_mark_phase` 上、而 `_build_done` 变成普通方法，
    调用处 `self._build_done(meta, t_start)` 立刻
    `TypeError: takes 2 positional arguments but 3 were given`，
    **26 个测试一起失败**。

    这属于"编辑位置失误"而非逻辑错误，光看代码很难发现，故用测试锁住。
    """

    def test_mark_phase_is_static(self):
        assert isinstance(
            BiddingAgent.__dict__["_mark_phase"], staticmethod
        ), "_mark_phase 必须是 staticmethod（调用处用 self._mark_phase(meta, ...)）"

    def test_build_done_is_static(self):
        assert isinstance(
            BiddingAgent.__dict__["_build_done"], staticmethod
        ), "_build_done 必须是 staticmethod（调用处用 self._build_done(meta, t_start)）"

    def test_instances_can_call_both(self):
        """实例调用两个方法都不应抛（签名与装饰器匹配）。"""
        agent = BiddingAgent(llm_client=_ScriptedLLM())
        meta = {"phase_times": [], "t_start": time.monotonic()}
        agent._mark_phase(meta, "x", time.monotonic())
        assert meta["phase_times"]

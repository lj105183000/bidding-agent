"""多 provider 自动切换测试。

★ 核心要测什么
------------
不是"能构造出链"，而是**切换真的发生**，且在这三种边界上行为正确：

    ① 第一个 provider 失败 → 真的用了第二个（且报告出来）
    ② **已经吐出正文后失败 → 不切换**（避免两段拼接口径不一致的回答）
    ③ 链里全不可用 → `available()` 为 False（让 pipeline 正常降级，而不是崩）

★ 为什么全用替身
-------------
真实 provider 调用要网络、要额度、且失败形态不可控。
切换逻辑是纯控制流，用替身能把三条边界都钉死。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clients.base_client import BaseLLMClient  # noqa: E402
from src.clients.fallback import (  # noqa: E402
    DEFAULT_FALLBACK,
    FALLBACK_ENV,
    FallbackLLMClient,
    build_fallback_client,
    parse_fallback_chain,
)


class _FakeClient(BaseLLMClient):
    """可控替身：可设不可用、可在第 N 个 chunk 处抛错。"""

    def __init__(self, name: str, *, usable: bool = True, chunks=None,
                 fail_stream_after: int | None = None, fail_raw: bool = False,
                 fail_before_stream: bool = False) -> None:
        self.provider = name
        self._usable = usable
        self.chunks = ["来自", name, "的回答"] if chunks is None else chunks
        self.fail_stream_after = fail_stream_after
        self.fail_raw = fail_raw
        self.fail_before_stream = fail_before_stream
        self.stream_calls = 0
        self.raw_calls = 0

    def available(self) -> bool:
        return self._usable

    def chat_stream(self, messages, **kwargs):
        self.stream_calls += 1
        if self.fail_before_stream:
            raise RuntimeError(f"{self.provider} 连不上")
        for i, chunk in enumerate(self.chunks):
            if self.fail_stream_after is not None and i == self.fail_stream_after:
                raise RuntimeError(f"{self.provider} 流到一半断了")
            yield chunk

    def chat_raw(self, messages, tools=None, **kwargs):
        self.raw_calls += 1
        if self.fail_raw:
            raise RuntimeError(f"{self.provider} chat_raw 失败")
        return f"raw-{self.provider}"

    def chat_stream_thinking(self, messages, **kwargs):
        self.stream_calls += 1
        if self.fail_before_stream:
            raise RuntimeError(f"{self.provider} 连不上")
        for i, chunk in enumerate(self.chunks):
            if self.fail_stream_after is not None and i == self.fail_stream_after:
                raise RuntimeError(f"{self.provider} 思考流断了")
            yield {"type": "content", "content": chunk}


def _chain(clients: dict[str, _FakeClient], chain: list[str]) -> FallbackLLMClient:
    return FallbackLLMClient(chain, lambda name: clients.get(name))


# --- 候选链解析 ---

class TestParseChain:
    def test_default_chain(self, monkeypatch):
        monkeypatch.delenv(FALLBACK_ENV, raising=False)
        assert parse_fallback_chain("") == list(DEFAULT_FALLBACK)

    def test_explicit_provider_goes_first(self, monkeypatch):
        """★ 调用方显式指定 provider 是明确意图，不能被兜底顺序挤到后面。"""
        monkeypatch.delenv(FALLBACK_ENV, raising=False)
        assert parse_fallback_chain("zhipu")[0] == "zhipu"

    def test_normalizes_case_and_space(self, monkeypatch):
        monkeypatch.delenv(FALLBACK_ENV, raising=False)
        assert parse_fallback_chain(" DeepSeek ")[0] == "deepseek"

    def test_raw_argument_wins_over_env(self, monkeypatch):
        """★ 回归哨兵：第一版在这里出过 bug。

        显式传了 `raw` 却仍被环境变量覆盖，表现为"自定义顺序完全不生效"。
        """
        monkeypatch.setenv(FALLBACK_ENV, "ollama")
        assert parse_fallback_chain("", "zhipu,deepseek") == ["zhipu", "deepseek"]

    def test_empty_raw_disables_fallback(self, monkeypatch):
        """显式关闭自动切换：只留指定的那一个（对比评测需要 provider 固定）。"""
        monkeypatch.delenv(FALLBACK_ENV, raising=False)
        assert parse_fallback_chain("deepseek", "") == ["deepseek"]

    def test_empty_env_disables_fallback(self, monkeypatch):
        monkeypatch.setenv(FALLBACK_ENV, "")
        assert parse_fallback_chain("deepseek") == ["deepseek"]

    def test_env_none_uses_default(self, monkeypatch):
        monkeypatch.delenv(FALLBACK_ENV, raising=False)
        assert parse_fallback_chain("deepseek") == ["deepseek", "zhipu"]

    def test_dedupes_preserving_order(self, monkeypatch):
        monkeypatch.delenv(FALLBACK_ENV, raising=False)
        assert parse_fallback_chain("", "deepseek,zhipu,deepseek") == ["deepseek", "zhipu"]

    def test_build_fallback_client_helper(self, monkeypatch):
        monkeypatch.delenv(FALLBACK_ENV, raising=False)
        client = build_fallback_client("zhipu", lambda n: _FakeClient(n))
        assert client.chain == ["zhipu", "deepseek"]


# --- 可用性 ---

class TestAvailability:
    def test_available_if_any_usable(self):
        """★ 链里第一个不可用、第二个可用时，`available()` 必须是 True。

        否则 pipeline 会误判为"LLM 不可用"而**降级成直接返回知识库原文**——
        明明有可用的 provider 却不用。
        """
        client = _chain(
            {"deepseek": _FakeClient("deepseek", usable=False),
             "zhipu": _FakeClient("zhipu")},
            ["deepseek", "zhipu"],
        )
        assert client.available() is True

    def test_unavailable_when_none_usable(self):
        client = _chain(
            {"deepseek": _FakeClient("deepseek", usable=False),
             "zhipu": _FakeClient("zhipu", usable=False)},
            ["deepseek", "zhipu"],
        )
        assert client.available() is False

    def test_resolver_error_does_not_break_chain(self):
        """某个 provider 客户端建不出来时，不该让整条链失效。"""
        def resolver(name):
            if name == "deepseek":
                raise RuntimeError("缺少 DEEPSEEK_API_KEY")
            return _FakeClient(name)

        client = FallbackLLMClient(["deepseek", "zhipu"], resolver)
        assert client.available() is True


# --- 切换行为 ---

class TestSwitchOnFailure:
    def test_switches_when_first_fails_before_output(self):
        a = _FakeClient("deepseek", fail_before_stream=True)
        b = _FakeClient("zhipu")
        client = _chain({"deepseek": a, "zhipu": b}, ["deepseek", "zhipu"])

        out = "".join(client.chat_stream([{"role": "user", "content": "q"}]))
        assert "zhipu" in out, f"未切换到 zhipu，实际输出 {out!r}"
        assert a.stream_calls == 1 and b.stream_calls == 1
        assert client.last_used_provider == "zhipu"
        assert client.switch_count == 1

    def test_skips_unavailable_without_calling_it(self):
        """不可用的 provider 不该被调用——那只会浪费一次失败并刷日志。"""
        a = _FakeClient("deepseek", usable=False)
        b = _FakeClient("zhipu")
        client = _chain({"deepseek": a, "zhipu": b}, ["deepseek", "zhipu"])

        "".join(client.chat_stream([{"role": "user", "content": "q"}]))
        assert a.stream_calls == 0, "不可用的 provider 不应被调用"
        assert b.stream_calls == 1

    def test_no_switch_after_output_started(self):
        """★★ 最关键的一条：**已经吐出正文后不再切换**。

        否则用户会看到"前半段 DeepSeek、后半段智谱"的拼接答案，
        两段语气与口径都不一致——比直接报错更糟。
        """
        a = _FakeClient("deepseek", chunks=["A", "B", "C"], fail_stream_after=1)
        b = _FakeClient("zhipu")
        client = _chain({"deepseek": a, "zhipu": b}, ["deepseek", "zhipu"])

        got: list[str] = []
        with pytest.raises(RuntimeError, match="流到一半断了"):
            for chunk in client.chat_stream([{"role": "user", "content": "q"}]):
                got.append(chunk)

        assert got == ["A"], "应吐出第一块后失败，且不再继续"
        assert b.stream_calls == 0, "★ 已产出内容后不该调用下一个 provider"
        assert client.switch_count == 0

    def test_raises_when_all_fail(self):
        a = _FakeClient("deepseek", fail_before_stream=True)
        b = _FakeClient("zhipu", fail_before_stream=True)
        client = _chain({"deepseek": a, "zhipu": b}, ["deepseek", "zhipu"])

        with pytest.raises(RuntimeError):
            "".join(client.chat_stream([{"role": "user", "content": "q"}]))
        assert len(client.last_error) == 2, "两次失败都应被记录"

    def test_raises_when_chain_empty(self):
        client = FallbackLLMClient([], lambda n: None)
        with pytest.raises(RuntimeError, match="没有可用的 LLM provider"):
            "".join(client.chat_stream([{"role": "user", "content": "q"}]))

    def test_third_provider_used_when_two_fail(self):
        clients = {
            "deepseek": _FakeClient("deepseek", fail_before_stream=True),
            "zhipu": _FakeClient("zhipu", fail_before_stream=True),
            "ollama": _FakeClient("ollama"),
        }
        client = _chain(clients, ["deepseek", "zhipu", "ollama"])
        out = "".join(client.chat_stream([{"role": "user", "content": "q"}]))
        assert "ollama" in out
        assert client.last_used_provider == "ollama"


class TestChatRaw:
    def test_switches_on_raw_failure(self):
        """`chat_raw` 无"部分输出"问题，可自由切换。"""
        a = _FakeClient("deepseek", fail_raw=True)
        b = _FakeClient("zhipu")
        client = _chain({"deepseek": a, "zhipu": b}, ["deepseek", "zhipu"])

        assert client.chat_raw([{"role": "user", "content": "q"}]) == "raw-zhipu"
        assert client.last_used_provider == "zhipu"

    def test_raw_raises_when_all_fail(self):
        a = _FakeClient("deepseek", fail_raw=True)
        b = _FakeClient("zhipu", fail_raw=True)
        client = _chain({"deepseek": a, "zhipu": b}, ["deepseek", "zhipu"])
        with pytest.raises(RuntimeError):
            client.chat_raw([{"role": "user", "content": "q"}])


class TestThinkingStream:
    def test_switches_before_output(self):
        a = _FakeClient("deepseek", fail_before_stream=True)
        b = _FakeClient("zhipu")
        client = _chain({"deepseek": a, "zhipu": b}, ["deepseek", "zhipu"])
        events = list(client.chat_stream_thinking([{"role": "user", "content": "q"}]))
        assert events and events[0]["type"] == "content"
        assert client.last_used_provider == "zhipu"

    def test_no_switch_after_output(self):
        a = _FakeClient("deepseek", chunks=["A", "B"], fail_stream_after=1)
        b = _FakeClient("zhipu")
        client = _chain({"deepseek": a, "zhipu": b}, ["deepseek", "zhipu"])
        got = []
        with pytest.raises(RuntimeError):
            for ev in client.chat_stream_thinking([{"role": "user", "content": "q"}]):
                got.append(ev)
        assert b.stream_calls == 0, "★ 已产出后不该切换（与 chat_stream 同约束）"


class TestObservability:
    """静默切换是不可接受的——用户以为在用 A、实际用了 B，合规与成本都是问题。"""

    def test_provider_reports_actual_user(self):
        a = _FakeClient("deepseek", fail_before_stream=True)
        b = _FakeClient("zhipu")
        client = _chain({"deepseek": a, "zhipu": b}, ["deepseek", "zhipu"])
        "".join(client.chat_stream([{"role": "user", "content": "q"}]))
        assert "zhipu" in client.provider, f"展示名应含实际 provider，实际 {client.provider!r}"
        assert "deepseek" in client.provider, "也应体现候选链（让用户知道有兜底）"

    def test_provider_name_before_use_shows_chain(self):
        client = _chain({"deepseek": _FakeClient("deepseek")}, ["deepseek"])
        assert client.provider == "deepseek"

    def test_errors_recorded_with_provider(self):
        a = _FakeClient("deepseek", fail_before_stream=True)
        b = _FakeClient("zhipu")
        client = _chain({"deepseek": a, "zhipu": b}, ["deepseek", "zhipu"])
        "".join(client.chat_stream([{"role": "user", "content": "q"}]))
        assert client.last_error and client.last_error[0][0] == "deepseek"
        assert "连不上" in client.last_error[0][1]

    def test_no_switch_count_when_first_succeeds(self):
        client = _chain({"deepseek": _FakeClient("deepseek")}, ["deepseek"])
        "".join(client.chat_stream([{"role": "user", "content": "q"}]))
        assert client.switch_count == 0
        assert client.last_used_provider == "deepseek"


class TestCooldown:
    """失败 provider 会被短期拉黑，避免一次问答内反复重试。

    ★ 这条是实测逼出来的
    ------------------
    用无效密钥跑真实 failover 时，**一次问答里 deepseek 被重试了 3 次**
    （ReAct 决策轮 + 最终生成各调一次），每次都撞 401、每次都要等一次网络往返。
    而 401 这类原因不会在一次问答内自愈，重试纯属浪费延迟。
    """

    def test_failed_provider_not_retried(self):
        a = _FakeClient("deepseek", fail_before_stream=True)
        b = _FakeClient("zhipu")
        client = _chain({"deepseek": a, "zhipu": b}, ["deepseek", "zhipu"])

        # 第一次：deepseek 失败 → zhipu 接管
        "".join(client.chat_stream([{"role": "user", "content": "q1"}]))
        assert a.stream_calls == 1
        assert client.switch_count == 1

        # 第二次：deepseek 仍在冷却期，**不该再被调用**
        "".join(client.chat_stream([{"role": "user", "content": "q2"}]))
        assert a.stream_calls == 1, "冷却期内不该重试刚失败的 provider"
        assert b.stream_calls == 2
        assert client.switch_count == 1, (
            "同一个坏 provider 不该被算成多次切换（实测曾报 3）"
        )

    def test_cooldown_also_applies_to_chat_raw(self):
        """决策轮的失败同样要拉黑，否则生成轮还会再撞一次。"""
        a = _FakeClient("deepseek", fail_raw=True)
        b = _FakeClient("zhipu")
        client = _chain({"deepseek": a, "zhipu": b}, ["deepseek", "zhipu"])

        assert client.chat_raw([{"role": "user", "content": "q"}]) == "raw-zhipu"
        assert a.raw_calls == 1
        # 再问一次：该走 zhipu，不再碰 deepseek
        assert client.chat_raw([{"role": "user", "content": "q"}]) == "raw-zhipu"
        assert a.raw_calls == 1, "chat_raw 失败后也应冷却"

    def test_cooldown_counted(self):
        a = _FakeClient("deepseek", fail_before_stream=True)
        b = _FakeClient("zhipu")
        client = _chain({"deepseek": a, "zhipu": b}, ["deepseek", "zhipu"])
        "".join(client.chat_stream([{"role": "user", "content": "q1"}]))
        "".join(client.chat_stream([{"role": "user", "content": "q2"}]))
        assert client._cooldown_hits >= 1

    def test_healthy_provider_never_cooled(self):
        client = _chain({"deepseek": _FakeClient("deepseek")}, ["deepseek"])
        for _ in range(3):
            "".join(client.chat_stream([{"role": "user", "content": "q"}]))
        assert client._cooldown == {}
        assert client.switch_count == 0

    def test_all_cooled_raises_clear_error(self):
        """链里全部被拉黑后，报错要说清是"没有可用 provider"而不是别的。"""
        a = _FakeClient("deepseek", fail_before_stream=True)
        b = _FakeClient("zhipu", fail_before_stream=True)
        client = _chain({"deepseek": a, "zhipu": b}, ["deepseek", "zhipu"])
        with pytest.raises(RuntimeError):
            "".join(client.chat_stream([{"role": "user", "content": "q1"}]))
        # 两个都被拉黑 → 第二次直接说没有可用 provider
        with pytest.raises(RuntimeError, match="没有可用的 LLM provider"):
            "".join(client.chat_stream([{"role": "user", "content": "q2"}]))


class TestTransparency:
    """链对调用方透明——pipeline/react_loop 不需要知道背后有几个 provider。"""

    def test_all_four_contract_methods_present(self):
        client = _chain({"deepseek": _FakeClient("deepseek")}, ["deepseek"])
        for name in ("available", "chat_stream", "chat_raw", "chat_stream_thinking"):
            assert callable(getattr(client, name)), f"缺 {name}（契约要求四个都实现）"

    def test_is_a_base_llm_client(self):
        client = _chain({"deepseek": _FakeClient("deepseek")}, ["deepseek"])
        assert isinstance(client, BaseLLMClient)

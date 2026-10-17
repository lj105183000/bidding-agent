"""局限 3 + 6 合并修的回归测试。

★ 两个缺陷的根因是同一个
==================
冷却只按 **provider 名** 记、且**不看失败类型**。

    局限 6：`zhipu` 与 `zai` 是同一家 → 各记一份冷却 → 拉黑一个不拉黑另一个
            → 候选链写 `zhipu,zai` 会两次撞同一个坏凭据
    局限 3：401 / 429 / 超时 → 一律 300s
            → 429（几秒后可恢复）与抖动被屏蔽 300s，走了低质量备用 provider

故合并修：
    · 冷却键改为**厂商**（`vendor_of`）
    · 时长按失败类型分档（`failover_policy.cooldown_seconds_for`）
    · 候选链按**厂商**折叠
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.clients.failover_policy import (  # noqa: E402
    AUTH_COOLDOWN_SECONDS,
    RATE_LIMIT_COOLDOWN_SECONDS,
    TRANSIENT_COOLDOWN_SECONDS,
    UNKNOWN_COOLDOWN_SECONDS,
    FailureKind,
    classify_failure,
    cooldown_seconds_for,
)
from src.clients.fallback import FallbackLLMClient, parse_fallback_chain  # noqa: E402
from src.clients.llm_factory import vendor_of  # noqa: E402


class _Err(Exception):
    def __init__(self, msg: str, code: int | None = None) -> None:
        super().__init__(msg)
        self.status_code = code


class _TimeoutErr(Exception):
    pass


# ===========================================================================
# 局限 3：按错误类型分档
# ===========================================================================


class TestFailureClassification:
    @pytest.mark.parametrize("code", [401, 403])
    def test_auth_by_status_code(self, code):
        assert classify_failure(_Err("x", code)) is FailureKind.AUTH

    def test_rate_limit_by_status_code(self):
        assert classify_failure(_Err("x", 429)) is FailureKind.RATE_LIMIT

    @pytest.mark.parametrize("code", [500, 502, 503, 504])
    def test_server_errors_are_transient(self, code):
        assert classify_failure(_Err("x", code)) is FailureKind.TRANSIENT

    def test_rate_limit_by_message_when_no_code(self):
        """中转站可能不带 status_code，只能看文案。"""
        assert classify_failure(_Err("Rate limit exceeded")) is FailureKind.RATE_LIMIT
        assert classify_failure(_Err("请求过于频繁")) is FailureKind.RATE_LIMIT

    def test_auth_by_message(self):
        assert classify_failure(_Err("Invalid API key")) is FailureKind.AUTH
        assert classify_failure(_Err("密钥无效")) is FailureKind.AUTH

    def test_transient_by_type_name(self):
        assert classify_failure(_Err("boom", 503)) is FailureKind.TRANSIENT
        assert classify_failure(_TimeoutErr("timed out")) is FailureKind.TRANSIENT

    def test_unknown_is_not_guessed(self):
        """归不了类就 UNKNOWN —— **不猜**。"""
        assert classify_failure(_Err("something odd")) is FailureKind.UNKNOWN

    def test_classify_never_raises(self):
        """分类本身绝不能成为新的故障源。"""

        class Weird(Exception):
            @property
            def status_code(self):  # 故意抛
                raise RuntimeError("boom")

        assert classify_failure(Weird("x")) is FailureKind.UNKNOWN


class TestCooldownTiers:
    def test_auth_longest(self):
        assert cooldown_seconds_for(_Err("x", 401)) == AUTH_COOLDOWN_SECONDS

    def test_rate_limit_short(self):
        """★ 核心：429 是**短期**可恢复，不能按 300s 屏蔽。"""
        assert cooldown_seconds_for(_Err("x", 429)) == RATE_LIMIT_COOLDOWN_SECONDS
        assert RATE_LIMIT_COOLDOWN_SECONDS < 300, "429 的冷却必须明显短于旧值 300s"

    def test_transient_short(self):
        assert cooldown_seconds_for(_Err("x", 503)) == TRANSIENT_COOLDOWN_SECONDS
        assert TRANSIENT_COOLDOWN_SECONDS < 300

    def test_unknown_is_between(self):
        """未知档要"避免白撞"但也不长期拉黑 —— 取中间值。"""
        assert cooldown_seconds_for(_Err("odd")) == UNKNOWN_COOLDOWN_SECONDS
        assert (
            RATE_LIMIT_COOLDOWN_SECONDS
            < UNKNOWN_COOLDOWN_SECONDS
            < AUTH_COOLDOWN_SECONDS
        )


class TestCooldownApplied:
    def _client(self, chain=("deepseek",)):
        return FallbackLLMClient(list(chain), lambda n: None)

    def test_429_cooldown_is_short_in_practice(self):
        c = self._client()
        c._enter_cooldown("deepseek", _Err("rate limit", 429))
        remaining = c._cooldown["deepseek"] - time.monotonic()
        assert remaining <= RATE_LIMIT_COOLDOWN_SECONDS + 1
        assert remaining < 60, f"429 的冷却应短（实测 {remaining:.0f}s）"

    def test_401_cooldown_is_long_in_practice(self):
        c = self._client()
        c._enter_cooldown("deepseek", _Err("bad key", 401))
        remaining = c._cooldown["deepseek"] - time.monotonic()
        assert remaining >= AUTH_COOLDOWN_SECONDS - 1

    def test_no_exc_falls_back_to_unknown_tier(self):
        """`exc=None` 保持向后兼容（按 UNKNOWN 档）。"""
        c = self._client()
        c._enter_cooldown("deepseek")
        remaining = c._cooldown["deepseek"] - time.monotonic()
        assert remaining <= UNKNOWN_COOLDOWN_SECONDS + 1

    def test_note_failure_passes_exc_through(self):
        """★ 回归哨兵：`_note_failure` 必须把 exc 传给冷却。

        若有人把 `self._enter_cooldown(name, exc)` 改回 `(name)`，
        所有失败都会退回 UNKNOWN 档（300s）—— 局限 3 复发。
        """
        c = self._client()
        c._note_failure("deepseek", _Err("rate limit", 429))
        remaining = c._cooldown["deepseek"] - time.monotonic()
        assert remaining < UNKNOWN_COOLDOWN_SECONDS, (
            "429 被按 300s 处理了 —— _note_failure 没有把 exc 传给冷却"
        )


# ===========================================================================
# 局限 6：同一厂商的别名共享冷却、候选链折叠
# ===========================================================================


class TestVendorAliasing:
    def test_zai_and_zhipu_share_vendor(self):
        assert vendor_of("zai") == vendor_of("zhipu") == "zhipu"

    def test_unaliased_providers_keep_own_vendor(self):
        for n in ("deepseek", "vllm", "ollama", "openai"):
            assert vendor_of(n) == n

    def test_unknown_name_is_returned_as_is(self):
        assert vendor_of("nonexistent") == "nonexistent"

    def test_cooldown_on_one_alias_blocks_the_other(self):
        """★ 核心：拉黑 zhipu 必须同时拉黑 zai。"""
        c = FallbackLLMClient(["zhipu", "zai"], lambda n: None)
        assert not c._in_cooldown("zai")
        c._enter_cooldown("zhipu", _Err("bad key", 401))
        assert c._in_cooldown("zhipu") is True
        assert c._in_cooldown("zai") is True, (
            "同一厂商的另一个别名没有被拉黑 —— 会在候选链里被试第二遍"
        )

    def test_cooldown_dict_has_one_key_per_vendor(self):
        c = FallbackLLMClient(["zhipu", "zai"], lambda n: None)
        c._enter_cooldown("zhipu", _Err("x", 401))
        c._enter_cooldown("zai", _Err("x", 401))
        assert len(c._cooldown) == 1, f"冷却应为厂商级，实有键 {list(c._cooldown)}"

    def test_different_vendors_are_independent(self):
        c = FallbackLLMClient(["deepseek", "zhipu"], lambda n: None)
        c._enter_cooldown("deepseek", _Err("x", 401))
        assert c._in_cooldown("deepseek") is True
        assert c._in_cooldown("zhipu") is False, "不同厂商不该被连带拉黑"


class TestChainFolding:
    def test_alias_pair_folds_to_one(self):
        assert parse_fallback_chain("", "zhipu,zai") == ["zhipu"]

    def test_order_preserved_keeps_first_alias(self):
        """保留**先出现**的名字 —— 保序，不擅自改名。"""
        assert parse_fallback_chain("", "zai,zhipu") == ["zai"]

    def test_primary_wins_over_fallback_alias(self):
        """显式指定的 primary 优先保留（尊重调用方意图）。"""
        assert parse_fallback_chain("zai", "deepseek,zhipu") == ["zai", "deepseek"]

    def test_default_chain_unaffected(self):
        chain = parse_fallback_chain("", "deepseek,zhipu")
        assert chain == ["deepseek", "zhipu"]

    def test_empty_raw_still_disables_fallback(self):
        """空串 = 显式关闭自动切换（三态语义不能因折叠而改变）。"""
        assert parse_fallback_chain("", "") == []

    def test_same_name_still_deduped(self):
        assert parse_fallback_chain("", "zhipu,zhipu") == ["zhipu"]

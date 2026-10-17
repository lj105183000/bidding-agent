import threading
import time

import pytest

from src.rate_limiter import (
    DEFAULT_TIERS,
    MAX_TRACKED_CLIENTS,
    RateLimiter,
    build_limiters,
    check_request,
    client_ip,
    is_private_ip,
    parse_trusted_proxies,
    rate_limit_headers,
    tier_of,
)


def test_allows_requests_within_quota():
    limiter = RateLimiter(max_requests=3, window_seconds=60)

    assert limiter.allow("client-1")
    assert limiter.allow("client-1")
    assert limiter.allow("client-1")


def test_rejects_requests_over_quota():
    limiter = RateLimiter(max_requests=2, window_seconds=60)

    assert limiter.allow("client-1")
    assert limiter.allow("client-1")
    assert not limiter.allow("client-1")


def test_different_clients_have_independent_quotas():
    limiter = RateLimiter(max_requests=1, window_seconds=60)

    assert limiter.allow("client-1")
    assert not limiter.allow("client-1")
    assert limiter.allow("client-2")


def test_window_expires():
    """窗口过期后应重新放行。

    ★ 余量必须留够（实测 flaky）
    --------------------------
    原实现用 `window_seconds=0.05` + `sleep(0.06)`，只留 10ms 余量。
    单独跑必过，但**全量回归时机器负载高，sleep 会被拉长/调度延迟**，
    于是偶发失败——实测在 1827 个用例的全量跑里挂过一次，单独重跑 3 次全过。

    这只是测试余量问题，不是限流器缺陷。改成窗口 0.05 + 睡 0.2（4 倍余量），
    把"时钟精度"与"被测逻辑"解耦。
    """
    limiter = RateLimiter(max_requests=1, window_seconds=0.05)

    assert limiter.allow("client-1")
    assert not limiter.allow("client-1")

    time.sleep(0.2)

    assert limiter.allow("client-1")


def test_thread_safety_exact_quota():
    quota = 10
    limiter = RateLimiter(max_requests=quota, window_seconds=60)

    results = []
    lock = threading.Lock()

    def worker():
        allowed = limiter.allow("client-1")
        with lock:
            results.append(allowed)

    threads = [threading.Thread(target=worker) for _ in range(30)]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    assert sum(results) == quota


# ===========================================================================
# 以下为细化后新增：分层、真实 IP 识别、内存有界、响应头
#
# ★ 为什么重点在这四块而不是"滑动窗口算得对不对"
# ------------------------------------------
# 滑动窗口原本就是对的。而下面两点会让限流**在生产上完全失效**：
#     ① 反代下所有用户共用一个桶 → 全站用户一起被拒（比不限流更糟）
#     ② 63 个端点只有 2 个前缀被保护 → 重度接口可以被打爆
# 以及两个安全约束：
#     ③ X-Forwarded-For 不能无条件采信（客户端可伪造，绕过限流）
#     ④ 跟踪表必须有界（否则限流器自己成为内存耗尽入口）
# ===========================================================================


class TestMemoryBound:
    def test_tracked_clients_bounded(self):
        lim = RateLimiter(max_requests=10, window_seconds=60, max_tracked=50)
        for i in range(500):
            lim.allow(f"10.0.{i // 250}.{i % 250}")
        assert lim.snapshot()["tracked_clients"] <= 50

    def test_default_cap(self):
        assert MAX_TRACKED_CLIENTS == 10_000

    def test_prune_does_not_spam_warning(self, caplog):
        """★ 超上限时警告必须节流（实测踩过刷屏）。

        `_last_prune` 曾在提前返回前就被赋值，导致超上限后**每次请求**
        都触发一次丢弃 + 一条警告，实测刷了 17 条同样的警告。
        """
        import logging

        lim = RateLimiter(max_requests=10, window_seconds=60, max_tracked=20)
        with caplog.at_level(logging.WARNING):
            for i in range(300):
                lim.allow(f"10.1.{i // 250}.{i % 250}")
        warnings = [r for r in caplog.records if "超过上限" in r.message]
        assert len(warnings) <= 1, f"警告刷屏了：{len(warnings)} 条"


class TestRejectedNotRecorded:
    def test_rejected_does_not_extend_window(self):
        """★ 被拒的请求不写入时间戳。

        否则持续超限的客户端会把窗口一直往后推，形成"被限流后永不释放"
        的惩罚性封禁效果——与滑动窗口语义不符。
        """
        lim = RateLimiter(max_requests=1, window_seconds=60)
        assert lim.allow("a") is True
        for _ in range(5):
            assert lim.allow("a") is False
        snap = lim.snapshot()
        assert snap["allowed"] == 1, "被拒的请求不该被计入已放行"
        assert snap["rejected"] == 5


class TestTrustedProxyParsing:
    def test_empty_means_trust_nothing(self):
        """★ 安全默认值：未配置就不采信任何转发头。"""
        assert parse_trusted_proxies("") == ()
        assert parse_trusted_proxies(None) == ()

    def test_single_and_list(self):
        assert parse_trusted_proxies("172.18.0.1") == ("172.18.0.1",)
        assert parse_trusted_proxies("172.18.0.1, 10.0.0.1") == ("172.18.0.1", "10.0.0.1")

    def test_wildcard_supported(self):
        assert parse_trusted_proxies("*") == ("*",)

    def test_strips_and_drops_empties(self):
        assert parse_trusted_proxies(" a ,, b , ") == ("a", "b")


class TestClientIp:
    """★ 最严重缺陷的回归测试：反代下所有人共享一个桶。"""

    def test_untrusted_direct_ignores_forwarded(self):
        ip, used = client_ip("203.0.113.9", "1.2.3.4", "", ())
        assert ip == "203.0.113.9"
        assert used is False

    def test_trusted_proxy_uses_forwarded(self):
        ip, used = client_ip("172.18.0.1", "203.0.113.9", "", ("172.18.0.1",))
        assert ip == "203.0.113.9"
        assert used is True

    def test_takes_rightmost_untrusted(self):
        """★ 取最右侧非可信，而不是最左侧。

        XFF 是逐跳追加的，最左侧由客户端自己写（可任意伪造）。
        """
        ip, _ = client_ip("172.18.0.1", "9.9.9.9, 203.0.113.9", "", ("172.18.0.1",))
        assert ip == "203.0.113.9", "采信了客户端伪造的最左 IP"

    def test_skips_multiple_trusted_hops(self):
        ip, _ = client_ip(
            "172.18.0.1", "203.0.113.9, 172.18.0.2", "",
            ("172.18.0.1", "172.18.0.2"),
        )
        assert ip == "203.0.113.9"

    def test_falls_back_to_real_ip(self):
        ip, used = client_ip("172.18.0.1", "", "203.0.113.9", ("172.18.0.1",))
        assert ip == "203.0.113.9"
        assert used is True

    def test_falls_back_to_direct_when_all_trusted(self):
        ip, used = client_ip("172.18.0.1", "172.18.0.1", "", ("172.18.0.1",))
        assert ip == "172.18.0.1"
        assert used is False, "来源不是转发头时应为 False"

    def test_wildcard_trusts_everything(self):
        ip, used = client_ip("203.0.113.9", "1.2.3.4", "", ("*",))
        assert ip == "1.2.3.4"
        assert used is True

    def test_empty_direct_gives_unknown(self):
        assert client_ip("", "", "", ())[0] == "unknown"

    def test_prefix_match_for_container_networks(self):
        """容器网络里代理 IP 会变，写死精确值会被一次重建打破。"""
        ip, _ = client_ip("172.18.0.7", "203.0.113.9", "", ("172.18.",))
        assert ip == "203.0.113.9"


class TestTierAssignment:
    """★ 只有 2 个前缀被保护是原实现的第二个严重问题。"""

    @pytest.mark.parametrize("path,expected", [
        ("/api/health", "none"),
        ("/api/health/live", "none"),
        ("/api/health/ready", "none"),
        ("/api/logs", "none"),
        ("/api/extract", "heavy"),
        ("/api/vision", "heavy"),
        ("/api/tender/parse", "heavy"),
        ("/api/tender/extract", "heavy"),
        ("/api/report/tender", "heavy"),
        ("/api/compliance/check/upload", "heavy"),
        ("/api/qualification/check/upload", "heavy"),
        ("/api/chat", "llm"),
        ("/api/chat/stream", "llm"),
        ("/api/ask", "llm"),
        ("/api/web/search", "llm"),
        ("/api/price-compare", "llm"),
        ("/api/graph/query", "context"),
        ("/api/analysis/overview", "context"),
        ("/api/tender/documents", "context"),
        ("/api/unknown-thing", "context"),
    ])
    def test_tier(self, path, expected):
        assert tier_of(path) == expected

    def test_longest_prefix_wins(self):
        """★ `/api/tender/parse`（heavy）与 `/api/tender/documents`（context）
        必须分到不同层——靠最长匹配，不受列表顺序影响。"""
        assert tier_of("/api/tender/parse") == "heavy"
        assert tier_of("/api/tender/documents") == "context"

    def test_health_never_limited_is_critical(self):
        """限流健康检查会让编排器误判服务不可用并重启——必须永远放行。"""
        assert tier_of("/api/health") == "none"
        assert tier_of("/api/health/ready") == "none"


class TestBuildLimiters:
    def test_none_tier_not_constructed(self):
        """`none` 不构造限流器——"不限流"在代码里就是"没有这个对象"，
        少一处可能写错的分支。"""
        lims = build_limiters()
        assert "none" not in lims
        assert set(lims) == {"heavy", "llm", "context"}

    def test_defaults_applied(self):
        lims = build_limiters()
        for tier, (max_req, window) in DEFAULT_TIERS.items():
            if tier == "none":
                continue
            assert lims[tier].max_requests == max_req
            assert lims[tier].window_seconds == window

    def test_overrides(self):
        lims = build_limiters({"llm": (7, 30)})
        assert lims["llm"].max_requests == 7
        assert lims["llm"].window_seconds == 30
        assert lims["heavy"].max_requests == DEFAULT_TIERS["heavy"][0]

    def test_zero_quota_tier_skipped(self):
        assert "llm" not in build_limiters({"llm": (0, 60)})


class TestCheckRequest:
    """★ 纯函数版本：不必启动应用即可验证中间件判定逻辑。

    （实测：走真实 `/api/ask` 会触发模型下载并卡住十几分钟，
    故中间件逻辑刻意抽成纯函数以便测试。）
    """

    def test_none_tier_always_allows(self):
        lims = build_limiters({"llm": (0, 60)})
        for _ in range(100):
            v = check_request("/api/health", "1.2.3.4", limiters=lims)
            assert v["allowed"] is True
            assert v["limited"] is False

    def test_allows_then_rejects(self):
        lims = build_limiters({"llm": (2, 60)})
        assert check_request("/api/chat", "a", limiters=lims)["allowed"] is True
        assert check_request("/api/chat", "a", limiters=lims)["allowed"] is True
        v = check_request("/api/chat", "a", limiters=lims)
        assert v["allowed"] is False
        assert v["retry_after"] >= 1
        assert v["remaining"] == 0

    def test_headers_on_both_paths(self):
        lims = build_limiters({"llm": (1, 60)})
        ok = check_request("/api/chat", "a", limiters=lims)
        assert ok["headers"]["X-RateLimit-Limit"] == "1"
        assert ok["headers"]["X-RateLimit-Remaining"] == "0"
        bad = check_request("/api/chat", "a", limiters=lims)
        assert "Retry-After" in bad["headers"]

    def test_tiers_are_independent(self):
        """★ 一层被限不该影响另一层。"""
        lims = build_limiters({"llm": (1, 60), "context": (5, 60)})
        assert check_request("/api/chat", "a", limiters=lims)["allowed"] is True
        assert check_request("/api/chat", "a", limiters=lims)["allowed"] is False
        assert check_request("/api/graph/query", "a", limiters=lims)["allowed"] is True

    def test_clients_are_independent(self):
        lims = build_limiters({"llm": (1, 60)})
        assert check_request("/api/chat", "a", limiters=lims)["allowed"] is True
        assert check_request("/api/chat", "b", limiters=lims)["allowed"] is True

    def test_reverse_proxy_gives_per_user_buckets(self):
        """★ 核心回归：反代下两个用户必须各有各的桶。

        原实现在这里把两个用户当成同一个（都用代理 IP），
        于是用户 B 被用户 A 的请求挤掉。
        """
        lims = build_limiters({"llm": (1, 60)})
        trusted = ("172.18.0.1",)
        a = check_request("/api/chat", "172.18.0.1", "203.0.113.9", "", lims, trusted)
        b = check_request("/api/chat", "172.18.0.1", "203.0.114.7", "", lims, trusted)
        assert a["allowed"] is True
        assert b["allowed"] is True, "用户 B 被当成用户 A 了（共用桶）"
        assert a["key"] != b["key"]

    def test_untrusted_proxy_ignores_spoofed_header(self):
        """不可信直连方伪造 XFF 时仍按直连 IP 计数（不能绕过）。"""
        lims = build_limiters({"llm": (1, 60)})
        first = check_request("/api/chat", "203.0.113.9", "1.1.1.1", "", lims, ())
        second = check_request("/api/chat", "203.0.113.9", "2.2.2.2", "", lims, ())
        assert first["allowed"] is True
        assert second["allowed"] is False, "换了伪造 XFF 就绕过了限流"

    def test_returns_tier_and_flags(self):
        v = check_request("/api/vision", "a", limiters=build_limiters())
        assert v["tier"] == "heavy"
        assert v["limited"] is True
        assert "used_forwarded" in v


class TestHeaders:
    def test_basic_headers(self):
        lim = RateLimiter(max_requests=10, window_seconds=60)
        h = rate_limit_headers(lim, 7)
        assert h["X-RateLimit-Limit"] == "10"
        assert h["X-RateLimit-Remaining"] == "7"
        assert h["X-RateLimit-Window"] == "60"
        assert "Retry-After" not in h

    def test_retry_after_only_when_rejected(self):
        lim = RateLimiter(max_requests=10, window_seconds=60)
        assert rate_limit_headers(lim, 0, 30)["Retry-After"] == "30"

    def test_remaining_never_negative(self):
        lim = RateLimiter(max_requests=10, window_seconds=60)
        assert rate_limit_headers(lim, -5)["X-RateLimit-Remaining"] == "0"


class TestRemainingAndRetry:
    def test_remaining_counts_down(self):
        lim = RateLimiter(max_requests=2, window_seconds=60)
        lim.allow("a")
        assert lim.remaining("a") == 1
        lim.allow("a")
        assert lim.remaining("a") == 0

    def test_retry_after_zero_when_quota_available(self):
        assert RateLimiter(max_requests=5, window_seconds=60).retry_after("a") == 0

    def test_retry_after_positive_when_exhausted(self):
        lim = RateLimiter(max_requests=1, window_seconds=60)
        lim.allow("a")
        assert lim.retry_after("a") >= 1


class TestPrivateIp:
    @pytest.mark.parametrize("ip,expected", [
        ("127.0.0.1", True), ("10.1.2.3", True), ("192.168.1.1", True),
        ("172.18.0.1", True), ("8.8.8.8", False), ("203.0.113.9", False),
        ("", True),
    ])
    def test_detection(self, ip, expected):
        assert is_private_ip(ip) is expected


class TestSnapshot:
    def test_reports_counts_and_rate(self):
        lim = RateLimiter(max_requests=1, window_seconds=60)
        lim.allow("a")
        lim.allow("a")
        s = lim.snapshot()
        assert s["allowed"] == 1
        assert s["rejected"] == 1
        assert s["reject_rate"] == 0.5
        assert s["tracked_clients"] == 1

    def test_reject_rate_zero_when_no_traffic(self):
        assert RateLimiter().snapshot()["reject_rate"] == 0.0
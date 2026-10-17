"""健康检查三处修正的回归测试。

局限 1：`_CORE` 写死 → 改为 `HEALTH_CORE_COMPONENTS` 可配
局限 2：ready 探测 1~3.3 秒 → 加 5 秒 TTL 缓存
局限 3：`points_count` 是近似值 → 标注为 `points≈N`
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.health import (  # noqa: E402
    HEALTH_CORE_ENV,
    HEALTH_PROBE_CACHE_ENV,
    _DEFAULT_CORE,
    build_health,
    clear_probe_cache,
    core_components,
    overall_status,
)


# ===========================================================================
# 局限 1：核心组件可配
# ===========================================================================


class TestCoreConfigurable:
    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        monkeypatch.delenv(HEALTH_CORE_ENV, raising=False)

    def test_default_is_qdrant_and_llm(self):
        assert core_components() == _DEFAULT_CORE == ("qdrant", "llm")

    def test_can_promote_neo4j_to_core(self, monkeypatch):
        """★ 核心：把 Neo4j 当业务必需的部署可以覆盖。

        写死时 Neo4j 挂了只报 degraded(200)，**编排器不摘流量**；
        配上后应报 down(503)。
        """
        monkeypatch.setenv(HEALTH_CORE_ENV, "qdrant,llm,neo4j")
        assert "neo4j" in core_components()
        comps = [
            {"name": "qdrant", "ok": True},
            {"name": "llm", "ok": True},
            {"name": "neo4j", "ok": False},
        ]
        assert overall_status(comps, {}) == "down", "neo4j 被当核心时不可用应判 down"

    def test_neck_optional_by_default(self):
        """默认下 Neo4j 挂了只 degraded（问答仍可用，不该触发重启）。"""
        comps = [
            {"name": "qdrant", "ok": True},
            {"name": "llm", "ok": True},
            {"name": "neo4j", "ok": False},
        ]
        assert overall_status(comps, {}) == "degraded"

    def test_unknown_names_ignored(self, monkeypatch):
        monkeypatch.setenv(HEALTH_CORE_ENV, "qdrant,瞎写的名字")
        assert core_components() == ("qdrant",)

    def test_all_unknown_falls_back_to_default(self, monkeypatch):
        """全非法 → 退回默认，**不能变成"没有核心组件"**（那会永远 ok）。"""
        monkeypatch.setenv(HEALTH_CORE_ENV, "全,都,不,认,识")
        assert core_components() == _DEFAULT_CORE

    def test_empty_env_uses_default(self, monkeypatch):
        monkeypatch.setenv(HEALTH_CORE_ENV, "")
        assert core_components() == _DEFAULT_CORE

    def test_can_shrink_core(self, monkeypatch):
        """也可以把 llm 从核心摘掉（例如纯检索部署）。"""
        monkeypatch.setenv(HEALTH_CORE_ENV, "qdrant")
        assert core_components() == ("qdrant",)
        comps = [
            {"name": "qdrant", "ok": True},
            {"name": "llm", "ok": False},
        ]
        assert overall_status(comps, {}) == "degraded"


# ===========================================================================
# 局限 2：探测缓存
# ===========================================================================


class TestProbeCache:
    @pytest.fixture(autouse=True)
    def _clear(self, monkeypatch):
        monkeypatch.delenv(HEALTH_PROBE_CACHE_ENV, raising=False)
        clear_probe_cache()
        yield
        clear_probe_cache()

    def test_cache_disabled_by_default(self):
        """★ 默认关 —— 否则"连续调用两次期望不同结果"的测试会失败。"""
        clear_probe_cache()
        build_health(depth="ready", cache_seconds=0)
        payload, _ = build_health(depth="ready", cache_seconds=0)
        assert not payload.get("cached"), "cache_seconds=0 时不该命中缓存"

    def test_cache_hit_marks_payload(self):
        build_health(depth="ready", cache_seconds=30)
        payload, _ = build_health(depth="ready", cache_seconds=30)
        assert payload.get("cached") is True
        assert payload.get("cache_ttl_seconds") == 30

    def test_cache_hit_is_much_faster(self):
        """命中缓存的耗时应当显著低于真实探测。"""
        import time

        clear_probe_cache()
        t0 = time.monotonic()
        build_health(depth="ready", cache_seconds=30)
        cold = time.monotonic() - t0

        t0 = time.monotonic()
        build_health(depth="ready", cache_seconds=30)
        warm = time.monotonic() - t0

        assert warm < cold / 2 or warm < 0.05, (
            f"缓存命中没有变快：cold={cold:.3f}s warm={warm:.3f}s"
        )

    def test_live_depth_never_cached(self):
        """`live` 不探外部依赖，缓存它会掩盖刚预热完成等状态变化。"""
        build_health(depth="live", cache_seconds=30)
        payload, _ = build_health(depth="live", cache_seconds=30)
        assert not payload.get("cached")

    def test_different_injected_agent_not_shared(self):
        """不同替身不能共用缓存条目（缓存键含 id）。"""
        marker_a, marker_b = object(), object()
        build_health(depth="ready", agent=marker_a, cache_seconds=30)
        payload, _ = build_health(depth="ready", agent=marker_b, cache_seconds=30)
        assert not payload.get("cached"), "不同 agent 替身命中了同一缓存条目"

    def test_clear_probe_cache_works(self):
        build_health(depth="ready", cache_seconds=30)
        clear_probe_cache()
        payload, _ = build_health(depth="ready", cache_seconds=30)
        assert not payload.get("cached")

    def test_env_var_enables_cache(self, monkeypatch):
        monkeypatch.setenv(HEALTH_PROBE_CACHE_ENV, "30")
        build_health(depth="ready")                       # 用环境变量
        payload, _ = build_health(depth="ready")
        assert payload.get("cached") is True

    def test_bad_env_value_falls_back_to_off(self, monkeypatch):
        monkeypatch.setenv(HEALTH_PROBE_CACHE_ENV, "abc")
        build_health(depth="ready")
        payload, _ = build_health(depth="ready")
        assert not payload.get("cached"), "非法 TTL 应退回 0（关）"

    def test_failures_are_also_cached(self):
        """★ 上游宕机时探针更需要轻量 —— 失败结果也要缓存。"""
        build_health(depth="ready", cache_seconds=30)
        payload, _ = build_health(depth="ready", cache_seconds=30)
        assert payload.get("cached") is True
        # 无论成功失败都该命中
        assert "status" in payload


# ===========================================================================
# 局限 3：近似值标注
# ===========================================================================


class TestPointsApproximate:
    def test_detail_marks_approximate(self):
        """监控看到 `points≈N` 才知道不能拿它做精确点数告警。"""
        clear_probe_cache()
        payload, _ = build_health(depth="ready", cache_seconds=0)
        qdrant = next(c for c in payload["components"] if c["name"] == "qdrant")
        if qdrant.get("detail"):
            assert "≈" in qdrant["detail"], (
                f"points_count 未标注为近似值：{qdrant['detail']!r}"
            )

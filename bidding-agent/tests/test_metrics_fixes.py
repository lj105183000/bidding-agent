"""监控指标三处修正的回归测试。

局限 7：`/api/metrics` 无鉴权 → 已挂 `require_logs_access`
局限 5：`slow_ratio` 阈值固定 1000ms → 改为按端点
局限 6：阶段耗时未聚合 → 新增 `record_phases` / `phase_metrics`
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.monitoring import (  # noqa: E402
    SLOW_MS,
    SLOW_MS_BY_ENDPOINT,
    build_metrics,
    phase_metrics,
    record_phases,
    reset_phases,
    slow_threshold_ms,
)


# ===========================================================================
# 局限 5：按端点的慢阈值
# ===========================================================================


class TestSlowThresholdPerEndpoint:
    def test_chat_has_long_threshold(self):
        """★ 核心：问答端点正常十几秒，1000ms 会让 slow_ratio 恒为 1。"""
        assert slow_threshold_ms("/api/chat") >= 10000

    def test_chat_ratio_not_meaningless(self):
        """5 秒的问答请求在 /api/chat 上**不该**算慢。"""
        assert slow_threshold_ms("/api/chat") > 5000

    def test_health_has_short_threshold(self):
        assert slow_threshold_ms("/api/health") <= 3000

    def test_unknown_endpoint_uses_default(self):
        assert slow_threshold_ms("/api/whatever") == SLOW_MS

    def test_subpath_inherits_parent(self):
        """`/api/chat/stream` 应继承 `/api/chat` 的阈值。"""
        assert slow_threshold_ms("/api/chat/stream") == slow_threshold_ms("/api/chat")

    def test_similar_prefix_not_matched(self):
        """★ `/api/chatty` 不能误命中 `/api/chat` —— 否则是"前缀"而非"路径段"。"""
        assert slow_threshold_ms("/api/chatty") == SLOW_MS

    def test_query_string_ignored(self):
        assert slow_threshold_ms("/api/chat?x=1") == slow_threshold_ms("/api/chat")

    def test_trailing_slash_ignored(self):
        assert slow_threshold_ms("/api/chat/") == slow_threshold_ms("/api/chat")

    def test_snapshot_reports_threshold(self):
        """★ 输出必须带 `slow_threshold_ms`，否则读指标的人不知道"慢"怎么切。"""
        import src.monitoring as M

        M.metrics.reset()
        M.metrics.record("/api/chat", 200, 5000.0)     # 5 秒：chat 不算慢
        M.metrics.record("/api/chat", 200, 20000.0)    # 20 秒：算慢
        snap = M.metrics.snapshot()
        ep = next(e for e in snap["endpoints"] if e["endpoint"] == "/api/chat")
        assert ep["slow_threshold_ms"] == slow_threshold_ms("/api/chat")
        assert ep["slow_ratio"] == pytest.approx(0.5, abs=0.01), (
            f"2 个样本里 1 个超阈值，slow_ratio 应为 0.5，实为 {ep['slow_ratio']}"
        )

    def test_thresholds_are_sane(self):
        for ep, ms in SLOW_MS_BY_ENDPOINT.items():
            assert ms > 0, f"{ep} 阈值非法"
            assert ms <= 60000, f"{ep} 阈值过大（{ms}ms）"


# ===========================================================================
# 局限 6：阶段耗时聚合
# ===========================================================================


class TestPhaseAggregation:
    @pytest.fixture(autouse=True)
    def _clean(self):
        reset_phases()
        yield
        reset_phases()

    def test_empty_returns_empty_dict(self):
        """没记录过 → `{}`（前端据显示"暂无数据"，而不是画一条 0 线）。"""
        assert phase_metrics() == {}

    def test_aggregates_avg_and_share(self):
        for _ in range(3):
            record_phases([["检索", 2000, 0], ["生成", 3000, 2000]])
        m = phase_metrics()
        assert m["turns"] == 3
        by = {r["phase"]: r for r in m["phases"]}
        assert by["检索"]["avg_ms"] == pytest.approx(2000)
        assert by["生成"]["avg_ms"] == pytest.approx(3000)
        # 检索占 2000/(2000+3000) = 40%
        assert by["检索"]["share"] == pytest.approx(0.4, abs=0.01)

    def test_sorted_by_avg_desc(self):
        record_phases([["小", 100, 0], ["大", 5000, 100]])
        rows = phase_metrics()["phases"]
        assert rows[0]["phase"] == "大", "应按均值降序（最耗时的在前）"

    def test_p95_and_max(self):
        for ms in (100, 200, 300):
            record_phases([["检索", ms, 0]])
        row = phase_metrics()["phases"][0]
        assert row["max_ms"] == 300
        assert row["p95_ms"] >= 290

    def test_malformed_input_ignored(self):
        """非法结构静默忽略 —— 观测不能影响问答。"""
        for bad in (None, "x", 123, [{}], [["只有名字"]], [["名", "不是数字"]]):
            record_phases(bad)          # 不应抛
        assert phase_metrics() == {}

    def test_never_raises(self):
        class Boom:
            def __iter__(self):
                raise RuntimeError("boom")

        record_phases(Boom())           # 不应抛

    def test_capacity_bounded(self):
        import src.monitoring as M

        for i in range(M.MAX_PHASE_SAMPLES + 100):
            record_phases([["检索", float(i), 0]])
        row = phase_metrics()["phases"][0]
        assert row["samples"] <= M.MAX_PHASE_SAMPLES, "阶段样本必须有上限"

    def test_reset_works(self):
        record_phases([["检索", 100, 0]])
        reset_phases()
        assert phase_metrics() == {}

    def test_build_metrics_includes_phases(self):
        record_phases([["检索", 100, 0]])
        payload = build_metrics()
        assert "phases" in payload
        assert payload["phases"]["turns"] >= 1


# ===========================================================================
# 局限 7：/api/metrics 鉴权
# ===========================================================================


class TestMetricsAuth:
    def test_metrics_endpoint_has_dependency(self):
        """`/api/metrics` 必须挂 `Depends(require_logs_access)`。"""
        import inspect

        from api import server

        src = inspect.getsource(server.get_metrics)
        # 依赖挂在装饰器上，不在函数体里 —— 查 app.routes
        routes = [r for r in server.app.routes if getattr(r, "path", "") == "/api/metrics"]
        assert routes, "未找到 /api/metrics 路由"
        dep_names = str(getattr(routes[0], "dependencies", []))
        assert "require_logs_access" in dep_names, (
            f"/api/metrics 未挂访问控制，dependencies={dep_names}"
        )

    def test_logs_endpoint_has_dependency(self):
        """`/api/logs` 也必须有（第 1 条，上一轮已修）。"""
        from api import server

        routes = [r for r in server.app.routes if getattr(r, "path", "") == "/api/logs"]
        assert routes
        dep_names = str(getattr(routes[0], "dependencies", []))
        assert "require_logs_access" in dep_names

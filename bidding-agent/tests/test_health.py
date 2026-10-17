"""健康检查测试。

★ 这一层最容易出的错是"语义错"，不是"功能错"
----------------------------------------
探测代码跑通很容易，难的是**状态语义对不对**：

    · 把"正在预热"报成"不可用" → 编排器反复重启一个健康的容器
    · 把"可选组件坏了"报成"整体不可用" → Neo4j 挂了却去重启本服务
    · 永远返回 200 → 编排器认为"永远健康"，故障时不会摘流量
    · 把"集合为空"和"探测失败"混为一谈 → 监控误判向量库没数据

故本文件重点是 `overall_status` / `status_code` 的语义，
以及"不泄露密钥"这类安全约束。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import health as H  # noqa: E402


def _comp(name: str, ok: bool, required: bool = False) -> dict:
    return {"name": name, "kind": "data", "required": required, "ok": ok, "ms": 1.0}


class TestStatusSemantics:
    """★ 状态语义是这一层最该测的东西。"""

    def test_all_ok(self):
        comps = [_comp("qdrant", True, True), _comp("llm", True, True),
                 _comp("neo4j", True), _comp("postgres", True)]
        assert H.overall_status(comps, {}) == "ok"

    def test_core_down_is_down(self):
        comps = [_comp("qdrant", False, True), _comp("llm", True, True)]
        assert H.overall_status(comps, {}) == "down"

    def test_llm_down_is_down(self):
        """LLM 是核心：`/api/chat` 依赖它，不可用则问答必然失败。"""
        comps = [_comp("qdrant", True, True), _comp("llm", False, True)]
        assert H.overall_status(comps, {}) == "down"

    def test_optional_down_is_degraded_not_down(self):
        """★ 可选组件（图谱/业务库）坏了不该报 down。

        报 down 会让运维去重启本服务——而重启解决不了上游 Neo4j 的故障，
        只会让服务反复重启、日志被截断、更难排查。
        """
        comps = [_comp("qdrant", True, True), _comp("llm", True, True),
                 _comp("neo4j", False), _comp("postgres", True)]
        assert H.overall_status(comps, {}) == "degraded"

    def test_warming_is_not_down(self):
        """★ 预热中不是故障。

        预热要 100 秒（嵌入 20s + 精排 80s），期间三个数据源都正常。
        报 down 会让编排器杀掉一个健康的、正在预热的容器。
        """
        comps = [_comp("qdrant", True, True), _comp("llm", True, True)]
        assert H.overall_status(comps, {"in_progress": True}) == "warming"

    def test_core_down_takes_priority_over_warming(self):
        """★ 核心坏了就是坏了，不能因为"还在预热"把 down 说成 warming。"""
        comps = [_comp("qdrant", False, True), _comp("llm", True, True)]
        assert H.overall_status(comps, {"in_progress": True}) == "down"

    def test_warmup_error_is_degraded(self):
        comps = [_comp("qdrant", True, True), _comp("llm", True, True)]
        assert H.overall_status(comps, {"error": "OOM"}) == "degraded"

    def test_empty_components_is_ok(self):
        """`live` 深度不探组件，不该因此报异常状态。"""
        assert H.overall_status([], {}) == "ok"


class TestStatusCode:
    def test_only_down_is_503(self):
        assert H.status_code("down") == 503

    @pytest.mark.parametrize("status", ["ok", "degraded", "warming"])
    def test_others_are_200(self, status):
        """★ degraded / warming 必须是 200。

        编排器只看状态码：给 503 会触发重启，而这两种情况重启都没用
        （一个是上游故障、一个是正在预热）。
        """
        assert H.status_code(status) == 200


class TestProbePrimitive:
    def test_success(self):
        r = H.probe(lambda: 42)
        assert r["ok"] is True
        assert r["value"] == 42
        assert r["ms"] >= 0

    def test_exception_captured_with_type(self):
        def boom():
            raise ValueError("坏了")

        r = H.probe(boom)
        assert r["ok"] is False
        assert "ValueError" in r["error"]
        assert "坏了" in r["error"]

    def test_timeout_distinguished_from_error(self):
        """★ 超时与异常必须区分：前者是"慢"，后者是"错"，排查方向不同。"""
        import time as _t

        r = H.probe(lambda: _t.sleep(5), timeout=0.2)
        assert r["ok"] is False
        assert r["error"].startswith("Timeout")
        assert r["ms"] < 1000, "超时应立即返回，不等满 5 秒"

    def test_preload_runs_before_timing(self):
        """★ preload 的目的就是把 import 移出计时窗口。"""
        calls: list[str] = []

        def preload():
            import time as _t
            _t.sleep(0.25)
            calls.append("preload")

        r = H.probe(lambda: calls.append("fn") or True, preload=preload)
        assert r["ok"] is True
        assert calls == ["preload", "fn"]
        # 250ms 的 preload 不该计入 ms
        assert r["ms"] < 200, f"preload 被计入耗时了: {r['ms']}ms"

    def test_preload_failure_does_not_break_probe(self):
        def bad_preload():
            raise RuntimeError("预加载炸了")

        r = H.probe(lambda: True, preload=bad_preload)
        assert r["ok"] is True, "preload 失败不该让探测失败"

    def test_error_message_truncated(self):
        def boom():
            raise RuntimeError("x" * 2000)

        r = H.probe(boom)
        assert len(r["error"]) <= 320


class TestComponentShape:
    def test_required_and_kind_recorded(self):
        c = H._component("qdrant", "data", True, {"ok": True, "ms": 12.0},
                         detail="points=565")
        assert c["name"] == "qdrant"
        assert c["kind"] == "data"
        assert c["required"] is True
        assert c["detail"] == "points=565"

    def test_error_only_when_failed(self):
        ok = H._component("x", "data", False, {"ok": True, "ms": 1.0})
        assert "error" not in ok
        bad = H._component("x", "data", False,
                           {"ok": False, "ms": 1.0, "error": "boom"})
        assert bad["error"] == "boom"

    def test_core_set_is_exactly_qdrant_and_llm(self):
        """核心集合变更需要同步改文档与告警规则，故钉住。"""
        assert set(H._CORE) == {"qdrant", "llm"}


class TestBuildHealth:
    def _build(self, **kw):
        # 注入替身，不碰真实依赖
        return H.build_health(
            agent=kw.pop("agent", None),
            graph_data=kw.pop("graph_data", None),
            postgres_getter=kw.pop("postgres_getter", None),
            **kw,
        )

    def test_live_depth_probes_nothing(self):
        p, code = self._build(depth="live")
        assert p["components"] == []
        assert p["depth"] == "live"
        assert code == 200
        assert p["failing"] == []

    def test_unknown_depth_falls_back_to_ready(self):
        p, _ = self._build(depth="banana")
        assert p["depth"] == "ready"

    def test_live_depth_is_fast(self):
        """★ 存活探针不做任何 IO，必须微秒级返回。"""
        import time as _t

        t = _t.perf_counter()
        self._build(depth="live")
        assert (_t.perf_counter() - t) < 0.05

    def test_deep_depth_includes_llm_probe(self):
        p, _ = self._build(depth="deep")
        assert "llm_deep" in p, "deep 深度必须给出真实 LLM 探测结果"

    def test_ready_depth_has_no_llm_deep(self):
        """★ deep 会对 LLM 产生真实调用，默认深度绝不能带它。

        Docker HEALTHCHECK 默认每 30 秒一次 → 一天 2880 次计费调用。
        """
        p, _ = self._build(depth="ready")
        assert "llm_deep" not in p

    def test_payload_has_required_keys(self):
        p, _ = self._build(depth="ready")
        for key in ("status", "depth", "uptime_s", "check_ms", "warmup",
                    "components", "failing"):
            assert key in p, f"缺字段 {key}"

    def test_failing_lists_failed_names(self):
        p, _ = self._build(depth="ready")
        assert set(p["failing"]) == {c["name"] for c in p["components"] if not c["ok"]}

    def test_uptime_non_negative(self):
        p, _ = self._build(depth="live")
        assert p["uptime_s"] >= 0

    def test_config_included_only_when_asked(self):
        p1, _ = self._build(depth="live")
        assert "config" not in p1
        p2, _ = self._build(depth="live", include_config=True)
        assert "config" in p2


class TestConfigSnapshot:
    """★ 安全约束：绝不泄露密钥值。"""

    def test_no_secret_values_leaked(self):
        import os

        # 塞入可识别的假密钥，确认它们不出现在输出里
        os.environ["DEEPSEEK_API_KEY"] = "sk-SECRET-CANARY-12345"
        os.environ["TAVILY_API_KEY"] = "tvly-SECRET-CANARY-67890"
        try:
            snap = H.config_snapshot()
            blob = repr(snap)
            assert "SECRET-CANARY" not in blob, "配置快照泄露了密钥值！"
        finally:
            os.environ.pop("DEEPSEEK_API_KEY", None)
            os.environ.pop("TAVILY_API_KEY", None)

    def test_reports_boolean_key_presence(self):
        snap = H.config_snapshot()
        assert isinstance(snap["llm"]["deepseek_key_set"], bool)
        assert isinstance(snap["search"]["tavily_key_set"], bool)

    def test_has_key_sections(self):
        snap = H.config_snapshot()
        for section in ("qdrant", "postgres", "neo4j", "llm", "search",
                        "features", "logging"):
            assert section in snap, f"缺 {section} 段"

    def test_qdrant_collection_visible(self):
        """生产最常见故障是"连到了错的 collection"，必须能一眼核对。"""
        snap = H.config_snapshot()
        assert "collection" in snap["qdrant"]
        assert snap["qdrant"]["collection"]


class TestWarmupState:
    """预热状态。

    ★ 这些用例依赖**模块级全局状态**，必须显式复位
    -------------------------------------------
    实测：单独跑全过，全量跑时 `test_initial_state` 失败（`elapsed_ms`
    得到一个真实耗时而不是 -1）。原因是**别的测试先启动过应用**
    （`TestClient(app)` 的 lifespan 会拉起后台预热线程），
    于是 `mark_warming_started` / `mark_warmup_done` 真的被调用过，
    模块里的 `_warmup` 留下了两个真实时间戳。

    "从未预热"在这个进程里已经不成立了 —— 断言本身没错，
    错在**依赖了别人留下的全局状态**。故每个用例前显式复位。
    """

    @pytest.fixture(autouse=True)
    def _reset_warmup(self):
        with H._warmup_lock:
            H._warmup.update({
                "in_progress": False, "started_at": 0.0,
                "finished_at": 0.0, "error": "",
            })
        yield

    def test_initial_state(self):
        s = H.warmup_state()
        assert s["in_progress"] is False
        assert s["elapsed_ms"] == -1

    def test_mark_started(self):
        H.mark_warming_started()
        s = H.warmup_state()
        assert s["in_progress"] is True
        assert s["elapsed_ms"] >= 0

    def test_mark_done_clears_progress(self):
        H.mark_warming_started()
        H.mark_warmup_done()
        s = H.warmup_state()
        assert s["in_progress"] is False
        assert s["error"] == ""

    def test_mark_done_with_error(self):
        H.mark_warmup_done(error="CUDA out of memory")
        s = H.warmup_state()
        assert s["error"] == "CUDA out of memory"
        assert s["in_progress"] is False

    def test_snapshot_does_not_leak_internals(self):
        """内部时间戳（epoch 秒）对外无意义，只暴露 elapsed_ms。"""
        H.mark_warming_started()
        s = H.warmup_state()
        assert "started_at" not in s
        assert "finished_at" not in s


class TestTruthyHelper:
    @pytest.mark.parametrize("value,expected", [
        ("1", True), ("true", True), ("YES", True), ("on", True),
        ("0", False), ("false", False), ("no", False), ("off", False),
        ("", False), ("  ", False),
    ])
    def test_common_forms(self, value, expected):
        assert H._truthy(value) is expected

    def test_unknown_value_is_true(self):
        """与项目既有约定一致：非显式关闭即视为开启。"""
        assert H._truthy("maybe") is True


class TestProbeTimeouts:
    def test_documented_values(self):
        """超时值有实测依据（Qdrant Cloud 跨洲延迟），变更需重新标定。"""
        assert H.PROBE_TIMEOUT_SECONDS == 5.0
        assert H.DEEP_PROBE_TIMEOUT_SECONDS == 15.0

    def test_deep_timeout_larger_than_normal(self):
        """LLM 要建连 + 推理，必须比普通探测宽松。"""
        assert H.DEEP_PROBE_TIMEOUT_SECONDS > H.PROBE_TIMEOUT_SECONDS

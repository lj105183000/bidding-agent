"""系统监控测试。

★ 这一层最该测什么
--------------
监控代码的失败模式与业务代码不同，风险集中在三处：

    ① **监控搞挂业务**：指标记录处在每个请求的必经路径上，
       它一抛异常就把成功的请求变成 500。
    ② **数字骗人**：均值被稀释、首次 CPU 读数恒为 0、
       把"取不到"编造成 0 —— 这些比没有监控更糟（让人以为一切正常）。
    ③ **内存泄漏**：无界的样本容器在长跑服务里就是泄漏。

故本文件重点是这三块，而不是"分位数公式对不对"。
"""

from __future__ import annotations

import sys
import time
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import monitoring as M  # noqa: E402


class TestPercentile:
    def test_empty_is_zero(self):
        assert M.percentile([], 0.95) == 0.0

    def test_single_sample(self):
        assert M.percentile([42.0], 0.95) == 42.0

    def test_p50_median_of_odd(self):
        assert M.percentile([1.0, 2.0, 3.0], 0.5) == 2.0

    def test_p95_of_small_sample_is_max(self):
        """★ 小样本下 P95 必须是最大值，否则会系统性低估尾部延迟。"""
        assert M.percentile([1.0, 2.0, 3.0], 0.95) == 3.0

    def test_returns_real_observation_not_interpolation(self):
        """★ 最近秩法：返回值必须是**真实观测到**的延迟。

        插值会给出"两个样本之间的虚构值"，排障时无法据此定位那次请求。
        """
        values = [100.0, 200.0, 300.0, 400.0]
        got = M.percentile(values, 0.5)
        assert got in values
        assert got == 200.0  # 而不是插值出的 200.0 之外的数

    def test_p100_is_max(self):
        assert M.percentile([5.0, 9.0, 7.0], 1.0) == 9.0

    def test_accepts_unsorted_input(self):
        """★ 内部自行排序。

        第一版要求调用方预排序（形参名就叫 `sorted_values`），
        实测测试传了未排序的列表就得到无意义结果。那种 bug 极难发现，
        故改为自己排序——多一次 O(n log n) 但消除了整类错误。
        """
        assert M.percentile([9.0, 1.0, 5.0], 1.0) == 9.0
        assert M.percentile([9.0, 1.0, 5.0], 0.5) == 5.0

    def test_monotonic(self):
        values = sorted(float(i) for i in range(1, 101))
        assert M.percentile(values, 0.5) <= M.percentile(values, 0.95)
        assert M.percentile(values, 0.95) <= M.percentile(values, 0.99)
        assert M.percentile(values, 0.99) <= M.percentile(values, 0.999)


class TestRecordAndAggregate:
    def _reg(self, **kw):
        return M.MetricsRegistry(**kw)

    def test_counts_and_errors(self):
        r = self._reg()
        for _ in range(10):
            r.record("/api/chat", 200, 100.0)
        for _ in range(2):
            r.record("/api/chat", 500, 9000.0)
        e = r.snapshot()["endpoints"][0]
        assert e["endpoint"] == "/api/chat"
        assert e["requests"] == 12
        assert e["errors"] == 2
        assert e["error_rate"] == pytest.approx(2 / 12, abs=1e-4)

    def test_4xx_and_5xx_both_count_as_errors(self):
        r = self._reg()
        r.record("/a", 404, 1.0)
        r.record("/a", 429, 1.0)
        r.record("/a", 503, 1.0)
        r.record("/a", 200, 1.0)
        assert r.snapshot()["endpoints"][0]["errors"] == 3

    def test_codes_recorded_individually(self):
        r = self._reg()
        r.record("/a", 200, 1.0)
        r.record("/a", 200, 1.0)
        r.record("/a", 500, 1.0)
        assert r.snapshot()["endpoints"][0]["codes"] == {"200": 2, "500": 1}

    def test_percentile_catches_outliers_that_average_hides(self):
        """★★ 这是本模块存在的核心理由：均值会掩盖长尾。

        200 次 200ms + 1 次 60000ms：
            均值 ≈ 497ms  ← "看起来还行"
            P99  = 200ms  ← **抓不到**那个离群值
            max  = 60000ms ← 只有它和极高分位能看到
        """
        r = self._reg()
        for _ in range(200):
            r.record("/a", 200, 200.0)
        r.record("/a", 200, 60000.0)
        e = r.snapshot()["endpoints"][0]

        # 均值确实被稀释了（这正是问题）
        assert e["avg_ms"] < 500
        # ★ 分位数是"比例"概念：1/201 = 0.5%，天然落在 P99 之上，
        #   故 P99 看不到它 —— 这是定义决定的，不是 bug
        assert e["p99_ms"] == 200.0
        # 故必须同时给 max 与 p99.9，否则极端请求在监控上完全隐形
        assert e["max_ms"] == 60000.0
        assert e["p99_9_ms"] == 60000.0, "p99.9 应能抓到那 0.5% 的极端值"

    def test_slow_ratio(self):
        r = self._reg()
        for _ in range(8):
            r.record("/a", 200, 100.0)
        for _ in range(2):
            r.record("/a", 200, 2000.0)
        assert r.snapshot()["endpoints"][0]["slow_ratio"] == pytest.approx(0.2)

    def test_endpoints_sorted_by_p95_desc(self):
        """★ "哪里慢"是最常问的，故默认按 P95 降序。"""
        r = self._reg()
        r.record("/fast", 200, 10.0)
        r.record("/slow", 200, 5000.0)
        r.record("/mid", 200, 500.0)
        order = [e["endpoint"] for e in r.snapshot()["endpoints"]]
        assert order == ["/slow", "/mid", "/fast"]

    def test_global_overall(self):
        r = self._reg()
        r.record("/a", 200, 100.0)
        r.record("/b", 500, 300.0)
        o = r.snapshot()["overall"]
        assert o["requests"] == 2
        assert o["errors"] == 1
        assert o["error_rate"] == 0.5
        assert o["endpoints"] == 2

    def test_top_errors_sorted_by_count_then_name(self):
        """出错端点按**错误数降序**；数量相同时按端点名（确定性顺序）。"""
        r = self._reg()
        r.record("/ok", 200, 1.0)
        for _ in range(2):
            r.record("/bad", 500, 1.0)
        for _ in range(5):
            r.record("/worse", 500, 1.0)
        for _ in range(2):
            r.record("/also-bad", 500, 1.0)
        names = [t["endpoint"] for t in r.snapshot()["top_errors"]]
        assert "/ok" not in names, "没出错的端点不该出现在 top_errors"
        assert names == ["/worse", "/also-bad", "/bad"], (
            "应先按错误数降序，数量相同按端点名"
        )

    def test_avg_all_vs_window_avg_both_present(self):
        r = self._reg()
        r.record("/a", 200, 100.0)
        e = r.snapshot()["endpoints"][0]
        assert "avg_ms_all" in e and "avg_ms" in e

    def test_empty_registry_is_safe(self):
        s = self._reg().snapshot()
        assert s["overall"]["requests"] == 0
        assert s["overall"]["error_rate"] == 0.0
        assert s["endpoints"] == []


class TestMemoryBounds:
    """★ 无界容器在长跑服务里就是内存泄漏。"""

    def test_samples_capped_per_endpoint(self):
        r = M.MetricsRegistry(max_samples=50)
        for i in range(500):
            r.record("/a", 200, float(i))
        assert r.snapshot()["endpoints"][0]["samples"] <= 50

    def test_endpoints_capped(self):
        r = M.MetricsRegistry(max_samples=100)
        for i in range(300):
            r.record(f"/e{i}", 200, 1.0)
        assert r.snapshot()["overall"]["endpoints"] <= M.MAX_ENDPOINTS

    def test_window_drops_old_samples(self):
        r = M.MetricsRegistry(window_seconds=0.2)
        r.record("/a", 200, 1.0)
        assert r.snapshot()["endpoints"][0]["samples"] == 1
        import time

        time.sleep(0.3)
        r.record("/a", 200, 2.0)
        assert r.snapshot()["endpoints"][0]["samples"] == 1, "窗口外的样本应被丢弃"
        # 但累计计数保留
        assert r.snapshot()["endpoints"][0]["requests"] == 2

    def test_reset(self):
        r = M.MetricsRegistry()
        r.record("/a", 200, 1.0)
        r.reset()
        assert r.snapshot()["overall"]["requests"] == 0


class TestThreadSafety:
    def test_concurrent_record_no_loss(self):
        r = M.MetricsRegistry(max_samples=100000)
        n_threads, per_thread = 8, 200

        def worker():
            for _ in range(per_thread):
                r.record("/a", 200, 1.0)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert r.snapshot()["overall"]["requests"] == n_threads * per_thread

    def test_concurrent_record_and_snapshot(self):
        """边写边读不能崩（监控端点随时可能被调）。"""
        r = M.MetricsRegistry()
        stop = threading.Event()
        errors: list[Exception] = []

        def writer():
            while not stop.is_set():
                r.record("/a", 200, 1.0)

        def reader():
            try:
                while not stop.is_set():
                    r.snapshot()
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        ts = [threading.Thread(target=writer), threading.Thread(target=reader)]
        for t in ts:
            t.start()
        import time

        time.sleep(0.3)
        stop.set()
        for t in ts:
            t.join()
        assert not errors, f"边写边读出错: {errors[:2]}"


class TestSafeRecord:
    """★ 监控不能搞挂业务。"""

    def test_never_raises_on_broken_registry(self, monkeypatch):
        class Boom:
            def record(self, *a, **k):
                raise RuntimeError("指标炸了")

        monkeypatch.setattr(M, "metrics", Boom())
        M.safe_record("/a", 200, 1.0)  # 不应抛

    def test_never_raises_on_bad_input(self, monkeypatch):
        monkeypatch.setattr(M, "metrics", M.MetricsRegistry())
        M.safe_record("/a", None, None)  # type: ignore[arg-type]  # 不应抛


class TestSystemMetrics:
    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        """每个用例前清系统指标缓存。

        ★ 不清会让用例互相干扰，且 `test_cached_within_interval` 在
        缓存命中时**不调 psutil**（快），未命中时调（慢）——
        全量回归耗时实测因此多出 60 秒。清一下让行为确定。
        """
        M._sys_cache["at"] = 0.0
        M._sys_cache["data"] = {}
        yield

    def test_returns_expected_sections(self):
        d = M.system_metrics()
        assert isinstance(d, dict)
        # psutil 已安装时应拿到这些段
        for key in ("process", "memory", "cpu", "disk"):
            assert key in d, f"缺 {key} 段"

    def test_process_has_rss_and_threads(self):
        d = M.system_metrics()
        p = d.get("process", {})
        assert p.get("rss_mb", 0) > 0
        assert p.get("threads", 0) >= 1
        assert p.get("pid")

    def test_cpu_first_call_flagged(self):
        """★ 首次 `cpu_percent` 必定返回 0.0 —— 必须标记，不能让人当成真实读数。

        「首次调用返回 0.0」是 psutil 的已知行为（需要两次采样求差）。
        不标记的话，第一次查监控会看到"CPU 占用 0%"，容易误判成服务空闲。
        """
        # 清掉缓存与历史，模拟首次
        M._sys_cache["at"] = 0.0
        M._last_cpu.clear()
        d = M.system_metrics()
        assert d["process"]["cpu_first_call"] is True
        # 第二次应不再是首次
        M._sys_cache["at"] = 0.0
        d2 = M.system_metrics()
        assert d2["process"]["cpu_first_call"] is False

    def test_cached_within_interval(self):
        """缓存期内应返回**同一份**数据（断言 `is`，不是 `==`）。

        ★ 原写法 `a is b or a == b` 是个**脆弱测试**（实测在全量跑时偶发失败）

        两个问题：

        ① **没有控制缓存状态**。`a = system_metrics()` 这一行本身可能是
           "命中缓存"也可能是"重新采集"，取决于别的用例何时写过缓存。
           若两行之间超过 `SYSTEM_CACHE_SECONDS`（2.0s），缓存过期 →
           两次都重新采集。而覆盖率插桩/机器负载下，**2 秒很容易超**。

        ② **`a == b` 这个退路逻辑上就是错的**。采集到的内存读数
           （`available_mb` 等）每次都会变，故"重新采集的两次"**必然不等**。
           实测失败信息：
               `{'available_mb': 6492.2}` != `{'available_mb': 6482.9}`

        于是表现为最耗时的那种脆弱：单独跑（快）通过、全量跑（慢）失败。

        改成确定性：先清缓存逼出一次真实采集，紧接着的第二次必然命中缓存，
        此时断言 `is` 才成立，也才真正验证"缓存返回同一份数据"这个意图。
        """
        M._sys_cache["at"] = 0.0
        M._sys_cache["data"] = {}
        a = M.system_metrics()                 # 真采集一次
        # ★ 不管上面那次采集花了多久，都把时间戳设成"刚刚"。
        #
        # 为什么必须这样（我改的第二版漏了这点，仍然脆弱）：
        # system_metrics() 会调 psutil.net_connections()（已废弃、开销大），
        # 在负载下**可能耗时超过 SYSTEM_CACHE_SECONDS（2.0s）** ——
        # 于是一采集完缓存就过期，第二次又变成真采集，is 失败。
        # 显式设时间戳让"缓存期内命中"这个语义**独立于采集速度**。
        M._sys_cache["at"] = time.time()
        b = M.system_metrics()
        assert a is b, "缓存期内应返回**同一个对象**"

    def test_cache_expires_after_interval(self):
        """缓存过期后应**重新采集**（而不是永远返回旧值）。

        与上一条互补：那条验"期内命中"，这条验"期外失效"。
        只验前者会漏掉"缓存永不失效"这种缺陷 —— 那种情况下监控会
        一直显示几小时前的数据，而界面上看不出任何异常。
        """
        M._sys_cache["at"] = 0.0
        M._sys_cache["data"] = {}
        a = M.system_metrics()
        # 把时间戳推回窗口之外，模拟"已过期"
        M._sys_cache["at"] = time.time() - M.SYSTEM_CACHE_SECONDS - 1
        b = M.system_metrics()
        assert a is not b, "缓存过期后必须重新采集，否则监控一直显示旧数据"

    def test_missing_psutil_returns_error_not_crash(self, monkeypatch):
        """取不到依赖时返回结构化错误，**不编造 0**。"""
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "psutil":
                raise ImportError("模拟未安装")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        M._sys_cache["at"] = 0.0
        d = M.system_metrics()
        assert "error" in d
        assert "psutil" in d["error"]

    def test_load_avg_none_on_windows(self):
        """Windows 上 `os.getloadavg` 不存在 → 返回 None，**不编造 0**。

        编造 0 会让人以为"负载为零"，而实际是"这个平台没有该指标"。
        """
        v = M._load_avg()
        assert v is None or isinstance(v, float)


class TestComponentMetrics:
    def test_no_agent_still_returns_sections(self):
        d = M.component_metrics(None, None)
        assert "retrieval_cache" in d
        assert "logs" in d

    def test_limiters_included(self):
        from src.rate_limiter import RateLimiter

        d = M.component_metrics(None, {"llm": RateLimiter(max_requests=5)})
        assert "llm" in d["rate_limit"]
        assert d["rate_limit"]["llm"]["max_requests"] == 5

    def test_broken_limiter_does_not_break_section(self):
        class Boom:
            def snapshot(self):
                raise RuntimeError("炸")

        d = M.component_metrics(None, {"llm": Boom()})
        assert "error" in d["rate_limit"]["llm"]

    def test_log_level_counts(self):
        d = M.component_metrics(None, None)
        assert "counts" in d["logs"]
        assert isinstance(d["logs"]["counts"], dict)


class TestBuildMetrics:
    def test_shape(self):
        d = M.build_metrics(None, None)
        assert set(d) >= {"collected_at", "requests", "system", "components"}
        assert "overall" in d["requests"]

    def test_collected_at_is_iso_with_offset(self):
        """时间要带时区偏移（与结构化日志同一约定，便于人工读）。"""
        ts = M.build_metrics(None, None)["collected_at"]
        assert "T" in ts
        assert ("+" in ts or "-" in ts[10:])


class TestPrometheusFormat:
    def _text(self):
        return M.render_prometheus(None, None)

    def test_has_help_and_type_lines(self):
        t = self._text()
        assert "# HELP " in t
        assert "# TYPE " in t

    def test_metric_lines_parseable(self):
        """每行非注释内容必须是 `<name>{labels} <value>`，且 value 能转 float。"""
        import re

        for line in self._text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            assert re.match(r"^[a-zA-Z_:][a-zA-Z0-9_:]*(\{[^}]*\})?\s+\S+$", line), (
                f"不合格式: {line!r}"
            )
            value = line.rsplit(" ", 1)[1]
            float(value)  # 不能抛

    def test_contains_core_metrics(self):
        t = self._text()
        assert "bidding_requests_total" in t
        assert "bidding_errors_total" in t
        assert "bidding_process_rss_mb" in t

    def test_type_declared_once_per_metric(self):
        t = self._text()
        types = [ln.split()[2] for ln in t.splitlines() if ln.startswith("# TYPE ")]
        assert len(types) == len(set(types)), "同一指标重复声明 TYPE"

    def test_ends_with_newline(self):
        assert self._text().endswith("\n")

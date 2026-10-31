"""RAG 在线评测的测试。

★ 在线评测最容易做错的地方
====================
**拖慢或拖垮用户请求**。每个请求都跑一次 LLM 裁判，成本与延迟都不可接受；
把写盘放在请求路径上，一次慢 IO 就会让所有用户等。

故本文件的重心是**边界与不变量**，而不是"指标算得对不对"：

    ① 提取是**纯函数**：不抛异常、不补 0（缺数据必须是 None）
    ② 请求路径**永不阻塞**：队列满时丢弃而不是等待
    ③ **不抛异常**：任何坏输入都静默跳过，绝不影响回答
    ④ 采样口径：追问/极短问题被排除（否则指标被系统性拉低）
    ⑤ 落盘异步、坏行容错、聚合口径正确

★ 在线评测**测不了**什么
====================
没有参考答案，故本文件不验证"答案正确性"——那是离线 ragas 评测的事。
"""

from __future__ import annotations

import json
import pathlib
import queue
import threading
import time

import pytest

from src.eval_online.aggregate import summarize_rows
from src.eval_online.recorder import OnlineEvaluator
from src.eval_online.signals import evaluate, is_evaluable


def _done(**over) -> dict:
    """构造一个"正常答完"的 done 帧。"""
    base = {
        "sources": [{"relevance": 0.9, "question": "q"}],
        "web_sources": [],
        "verification": {"ok": True, "critical_ok": True},
        "rag_timing": {"all_cached": False, "total_ms": 1000,
                       "steps": [{"step": "encode_query", "sum": 20}],
                       "rerank_failed": 0},
        "llm_trace": {"call_count": 2, "failed_count": 0, "total_ms": 3000,
                      "total_tokens": 1000, "cache_hit_tokens": 800,
                      "cache_miss_tokens": 200},
        "elapsed_ms": 5000,
        "tool_called": True,
        "clarified": False,
    }
    base.update(over)
    return base


# ===========================================================================
# 一、纯函数提取
# ===========================================================================


class TestEvaluate:
    def test_basic_fields(self):
        row = evaluate({}, _done(), question="投标保证金比例上限是多少？")
        assert row["source_count"] == 1
        assert row["relevance_top1"] == 0.9
        assert row["verify_ok"] is True
        assert row["no_source"] is False
        assert row["duration_ms"] == 5000.0
        assert row["tokens"] == 1000.0

    def test_top1_vs_mean(self):
        """★ top1 与 mean 都要给。

        "检索到了相关内容吗"是 top-1 能回答的问题；均值会被 tail 拉平：
        5 条里 1 条相关（0.9）与全都中等相关（0.5）均值可能接近，
        但前者答案质量好得多。
        """
        row = evaluate({}, _done(sources=[
            {"relevance": 0.9}, {"relevance": 0.1}, {"relevance": 0.1},
        ]))
        assert row["relevance_top1"] == 0.9
        assert abs(row["relevance_mean"] - 0.3667) < 0.01

    def test_missing_relevance_is_none_not_zero(self):
        """★ 缺评分必须是 None，**不是 0**。

        `0` 会被读成"相关度极低"，而事实是"这个来源没给评分"。
        与项目其他模块同一原则（`parse_usage` 缺 usage 时不补 0）。
        """
        row = evaluate({}, _done(sources=[{"question": "q"}]))
        assert row["relevance_top1"] is None
        assert row["relevance_mean"] is None

    def test_no_source_detected(self):
        row = evaluate({}, _done(sources=[], web_sources=[]))
        assert row["no_source"] is True
        assert row["source_count"] == 0

    def test_web_sources_count_as_source(self):
        """★ 联网来源也算"有依据"。

        只看本地来源会把"自动联网后答得很好"的请求记成"无来源"，
        从而把 no_source_rate 抬高——那是个假警报。
        """
        row = evaluate({}, _done(sources=[], web_sources=[{"title": "t"}]))
        assert row["no_source"] is False

    def test_unsupported_counts(self):
        row = evaluate({}, _done(verification={
            "ok": False, "critical_ok": False,
            "unsupported_laws": ["x"], "unsupported_articles": ["a", "b"],
            "unsupported_numbers": ["1"],
        }))
        assert row["unsupported_laws"] == 1
        assert row["unsupported_articles"] == 2
        assert row["unsupported_numbers"] == 1
        assert row["verify_ok"] is False

    def test_web_fallback_used(self):
        row = evaluate({}, _done(web_fallback={"enough": False}))
        assert row["web_fallback_used"] is True
        row2 = evaluate({}, _done(web_fallback={"enough": True}))
        assert row2["web_fallback_used"] is False

    def test_rag_steps_extracted(self):
        row = evaluate({}, _done(rag_timing={
            "steps": [{"step": "encode_query", "sum": 20},
                      {"step": "hybrid_retrieve", "sum": 900}],
            "total_ms": 920,
        }))
        assert row["rag_steps"] == {"encode_query": 20.0, "hybrid_retrieve": 900.0}
        assert row["rag_ms"] == 920.0

    def test_never_raises_on_garbage(self):
        """★ 纯函数、永不抛异常。任何畸形输入都只是得到一个不完整的记录。"""
        garbage = [
            ({}, {}),
            ({}, {"sources": "not-a-list"}),
            ({}, {"sources": [None, "x", 3]}),
            ({}, {"verification": "x", "rag_timing": [], "llm_trace": 5}),
            ({}, {"rag_timing": {"steps": "x"}}),
            ({}, {"rag_timing": {"steps": [None, {"step": None}]}}),
            ({}, {"sources": [{"relevance": "0.9"}]}),   # 字符串不解析成数字
        ]
        for meta, done in garbage:
            evaluate(meta, done)  # 不应抛

    def test_bool_not_treated_as_number(self):
        """bool 是 int 子类，但不该被当成数值指标。"""
        row = evaluate({}, _done(llm_trace={"total_tokens": True}))
        assert row["tokens"] is None

    def test_clarified_flag_carried(self):
        row = evaluate({}, _done(clarified=True))
        assert row["clarified"] is True

    def test_question_chars_and_history(self):
        row = evaluate({}, _done(), question="一二三四五", has_history=True)
        assert row["question_chars"] == 5
        assert row["has_history"] is True


class TestIsEvaluable:
    """★ 排除口径。

    若不排除，指标会被**系统性拉低**：首轮追问没有答案（无来源、无核对），
    算进 no_source_rate 会得出"20% 的请求检索失败"——
    实际是"20% 的请求在按设计追问"。
    """

    def test_normal_row_is_evaluable(self):
        assert is_evaluable(evaluate({}, _done(), question="投标保证金上限？"))

    def test_clarified_excluded(self):
        row = evaluate({}, _done(clarified=True), question="我们这个项目可以单一来源吗")
        assert is_evaluable(row) is False

    def test_short_question_excluded(self):
        """寒暄/误触不代表真实检索场景。"""
        assert is_evaluable(evaluate({}, _done(), question="你好")) is False
        assert is_evaluable(evaluate({}, _done(), question="")) is False

    def test_empty_row_excluded(self):
        assert is_evaluable({}) is False


# ===========================================================================
# 二、请求路径不变量（本文件最重要的部分）
# ===========================================================================


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch, tmp_path):
    """每个用例用**独立的样本文件**，并强制开启采样。

    不隔离的话测试会写进真实的 `eval/online_samples.jsonl`
    （污染线上数据），且采样率 20% 会让断言随机失败。
    """
    monkeypatch.setenv("RAG_ONLINE_EVAL", "1")
    monkeypatch.setenv("RAG_ONLINE_EVAL_RATE", "1")
    monkeypatch.setenv("RAG_ONLINE_EVAL_PATH", str(tmp_path / "s.jsonl"))
    # ★ 必须显式关掉裁判：本地 .env 里可能配了 RAG_ONLINE_EVAL_JUDGE_RATE=1.0，
    #   而 src.config 在导入时会把 .env 灌进 os.environ（override=False）。
    #   不关的话每条基础记录会额外写一条 judge 行，`written == accepted` 之类的
    #   旧不变量会被环境悄悄破坏。
    monkeypatch.setenv("RAG_ONLINE_EVAL_JUDGE_RATE", "0")
    yield


class TestNeverBlocks:
    def test_full_queue_drops_instead_of_blocking(self, monkeypatch, tmp_path):
        """★ 队列满时必须**丢弃**，不能等待。

        在线采样丢几条无妨（趋势观测不需要每一条），
        但阻塞请求或无限堆积内存是不可接受的。
        """
        ev = OnlineEvaluator(path=tmp_path / "x.jsonl")
        # 填满队列且不让 worker 消费
        monkeypatch.setattr(ev, "_ensure_worker", lambda: None)
        for _ in range(ev._q.maxsize):
            ev._q.put_nowait({"x": 1})

        t0 = time.monotonic()
        ev.maybe_record({}, _done(), question="投标保证金上限是多少？")
        elapsed = time.monotonic() - t0

        assert elapsed < 0.5, f"不得阻塞（实测 {elapsed:.3f}s）"
        assert ev.dropped >= 1, "应计入 dropped（自己也要可观测）"

    def test_never_raises(self, monkeypatch, tmp_path):
        """任何异常都不得冒泡到请求路径。"""
        ev = OnlineEvaluator(path=tmp_path / "y.jsonl")

        class Boom:
            def get(self, *a, **k):
                raise RuntimeError("炸")

        # meta/done 都是"会炸"的对象
        ev.maybe_record(Boom(), Boom(), question="x")  # 不应抛

    def test_disabled_does_nothing(self, monkeypatch, tmp_path):
        monkeypatch.setenv("RAG_ONLINE_EVAL", "0")
        ev = OnlineEvaluator(path=tmp_path / "z.jsonl")
        ev.maybe_record({}, _done(), question="投标保证金上限是多少？")
        assert ev.accepted == 0


# ===========================================================================
# 三、异步落盘与读取
# ===========================================================================


class TestRecorder:
    def _drain(self, ev: OnlineEvaluator) -> None:
        """等后台写完（避免用 sleep 猜时间）。"""
        assert ev.flush(timeout=5.0) or True
        for _ in range(100):
            if ev.written >= ev.accepted and ev._q.empty():
                return
            time.sleep(0.02)

    def test_writes_rows(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RAG_ONLINE_EVAL_PATH", str(tmp_path / "w.jsonl"))
        ev = OnlineEvaluator(path=tmp_path / "w.jsonl")
        for i in range(5):
            ev.maybe_record({}, _done(), question=f"投标保证金上限是多少{i}？")
        self._drain(ev)
        rows = ev.read_rows()
        assert len(rows) == 5
        # 裁判实现后，每条基础记录还会额外写一条 judge 行；`written` 统计所有行，
        # 故不能再与采样条数精确相等。基础行数由上面的 `read_rows` 断言。
        assert ev.written >= 5

    def test_read_tolerates_bad_lines(self, tmp_path):
        """坏行跳过而不是整体失败：jsonl 可能被人工编辑或写了一半。"""
        p = tmp_path / "bad.jsonl"
        p.write_text(
            json.dumps({"source_count": 1}, ensure_ascii=False) + "\n"
            + "{ 这不是合法 JSON\n"
            + "\n"
            + json.dumps({"source_count": 2}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        ev = OnlineEvaluator(path=p)
        rows = ev.read_rows()
        assert len(rows) == 2
        assert [r["source_count"] for r in rows] == [1, 2]

    def test_read_missing_file(self, tmp_path):
        assert OnlineEvaluator(path=tmp_path / "nope.jsonl").read_rows() == []

    def test_reads_rotated_backup(self, tmp_path):
        """轮转后的备份也要读 —— 否则趋势对比会丢掉上一个窗口。"""
        p = tmp_path / "r.jsonl"
        p.write_text(json.dumps({"source_count": 1}) + "\n", encoding="utf-8")
        p.with_suffix(".jsonl.1").write_text(
            json.dumps({"source_count": 9}) + "\n", encoding="utf-8")
        rows = OnlineEvaluator(path=p).read_rows()
        assert [r["source_count"] for r in rows] == [9, 1], "备份应排在前面（更早）"

    def test_snapshot_self_observable(self, tmp_path):
        """采样器自身也要可观测：丢了多少条是"数据是否可信"的前提。"""
        ev = OnlineEvaluator(path=tmp_path / "s.jsonl")
        snap = ev.snapshot()
        for k in ("enabled", "rate", "judge_rate", "queue_size", "queue_maxsize",
                  "accepted", "dropped", "written", "write_errors"):
            assert k in snap, f"快照缺 {k}"

    def test_concurrent_records_do_not_lose_rows(self, tmp_path):
        """并发写入不丢行（工具层与多请求都可能并发）。"""
        ev = OnlineEvaluator(path=tmp_path / "c.jsonl")
        n_threads, per = 6, 20

        def worker(tag: int) -> None:
            for i in range(per):
                ev.maybe_record({}, _done(), question=f"投标保证金上限{tag}-{i}？")

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self._drain(ev)
        assert ev.accepted == n_threads * per
        # `_drain` 的 `written >= accepted` 条件在裁判行也计数后会提前满足，
        # 故这里等到基础记录真正全部落盘（最多 5 秒）再断言。
        deadline = time.monotonic() + 5.0
        rows: list[dict] = []
        while time.monotonic() < deadline:
            rows = ev.read_rows(limit=n_threads * per * 2)
            if len(rows) >= ev.accepted:
                break
            time.sleep(0.02)
        assert len(rows) == ev.accepted, "基础记录不能静默丢"
        assert ev.written + ev.write_errors >= ev.accepted


# ===========================================================================
# 四、聚合口径
# ===========================================================================


class TestAggregate:
    def test_empty(self):
        s = summarize_rows([])
        assert s["sample_count"] == 0
        assert s["evaluable_count"] == 0
        assert s["relevance_top1"] is None, "无样本时统计值应为 None（不是 0）"

    def test_excludes_clarified_and_short(self):
        rows = [
            evaluate({}, _done(), question="投标保证金上限是多少？"),
            evaluate({}, _done(clarified=True), question="我们这个项目可以单一来源吗"),
            evaluate({}, _done(), question="你好"),
        ]
        s = summarize_rows(rows)
        assert s["sample_count"] == 3
        assert s["evaluable_count"] == 1
        assert s["excluded_count"] == 2

    def test_rates_computed(self):
        rows = [
            evaluate({}, _done(sources=[]), question="投标保证金上限是多少？"),
            evaluate({}, _done(), question="履约保证金上限是多少？"),
        ]
        s = summarize_rows(rows)
        assert s["no_source_rate"] == 0.5
        assert s["verify_ok_rate"] == 1.0

    def test_rate_is_none_without_samples(self):
        """★ 没有样本时比例是 None 而不是 0。

        `0` 会被读成"0% 的请求无来源"（很好的成绩），
        而事实是"还没有数据"。这是最容易误导的一类默认值。
        """
        rows = [evaluate({}, _done(clarified=True), question="我们这个项目可以单一来源吗")]
        s = summarize_rows(rows)
        assert s["evaluable_count"] == 0
        assert s["no_source_rate"] is None

    def test_rag_steps_mean(self):
        rows = [
            evaluate({}, _done(rag_timing={"steps": [{"step": "a", "sum": 10}]}),
                     question="投标保证金上限是多少？"),
            evaluate({}, _done(rag_timing={"steps": [{"step": "a", "sum": 30}]}),
                     question="履约保证金上限是多少？"),
        ]
        s = summarize_rows(rows)
        assert s["rag_steps_mean"] == {"a": 20.0}

    def test_cache_ratio(self):
        rows = [evaluate({}, _done(), question="投标保证金上限是多少？")]
        s = summarize_rows(rows)
        assert s["cache_hit_ratio"]["hit_ratio"] == 0.8

    def test_feedback_counted(self):
        rows = [
            {**evaluate({}, _done(), question="投标保证金上限是多少？"), "feedback": 1},
            {**evaluate({}, _done(), question="履约保证金上限是多少？"), "feedback": -1},
            {**evaluate({}, _done(), question="中标公示期几天？"), "feedback": None},
        ]
        s = summarize_rows(rows)
        assert s["feedback_up"] == 1
        assert s["feedback_down"] == 1
        assert s["feedback_count"] == 2, "未评价的不计入分母"
        assert s["feedback_up_rate"] == 0.5

    def test_group_by_turns(self):
        rows = [
            evaluate({}, _done(sources=[]), question="投标保证金上限是多少？",
                     has_history=False),
            evaluate({}, _done(), question="那履约保证金呢？", has_history=True),
        ]
        s = summarize_rows(rows)
        assert s["by_turns"]["first_turn"]["n"] == 1
        assert s["by_turns"]["multi_turn"]["n"] == 1
        assert s["by_turns"]["first_turn"]["no_source_rate"] == 1.0
        assert s["by_turns"]["multi_turn"]["no_source_rate"] == 0.0

    def test_caveats_present(self):
        """★ 口径提醒必须随数据一起下发。

        不写清楚的话，`verify_ok_rate: 0.95` 会被读成"95% 的答案是对的"，
        而它实际只说"95% 的答案里引用的法条能在检索结果里找到"。
        """
        s = summarize_rows([evaluate({}, _done(), question="投标保证金上限是多少？")])
        caveats = s.get("caveats") or []
        assert len(caveats) >= 3
        joined = " ".join(caveats)
        assert "参考答案" in joined
        assert "verify_ok_rate" in joined


# ===========================================================================
# 五、★ 自查发现并修复的四个缺陷（回归锁）
#
# 这些都不是"跑不起来"，而是**算错了却不报错** —— 最危险的一类缺陷。
# 故每个都配回归测试，防止再次退化。
# ===========================================================================


class TestPercentileCorrectness:
    """★ `p95` 曾经算错（自查时手算比对发现）。

    原实现 `int(round(pct/100*n + 0.5)) - 1` 的净效果是**向上取整**，
    取到比应有位置更高的值：

        n=20,  p95 → 实际返回**最大值**（应为第 19 个）
        n=100, p95 → 实际返回第 96 个（应为第 95 个）

    即 API 返回的 `p95` 是错的，且偏差随 n 的小数部分变化、没有规律。

    正确式子：最近秩法 `k = ceil(pct/100 * n)`（1-based）→ 索引 k-1。
    """

    @pytest.mark.parametrize("n", [1, 2, 3, 5, 10, 19, 20, 21, 99, 100, 101, 1000])
    def test_p95_matches_nearest_rank(self, n):
        import math

        from src.eval_online.aggregate import _percentile

        vals = [float(i) for i in range(1, n + 1)]  # 已升序
        expect = vals[max(1, math.ceil(0.95 * n)) - 1]
        assert _percentile(vals, 95) == expect, (
            f"n={n} 时 p95 应为第 {math.ceil(0.95 * n)} 个值 {expect}"
        )

    def test_p50_is_median_like(self):
        from src.eval_online.aggregate import _percentile

        vals = [float(i) for i in range(1, 101)]
        assert _percentile(vals, 50) == 50.0

    def test_small_sample_falls_back_to_max(self):
        """小样本时 p95 自然退化为最大值（最近秩法的预期行为）。"""
        from src.eval_online.aggregate import _percentile

        assert _percentile([1.0, 2.0], 95) == 2.0
        assert _percentile([1.0], 95) == 1.0

    def test_empty(self):
        from src.eval_online.aggregate import _percentile

        assert _percentile([], 95) == 0.0


class TestSamplingUnbiased:
    """★ 采样曾经**有偏**（自查统计发现，本轮最严重的缺陷）。

    原实现用时间戳低位做判定：
        `(time.time_ns() % 1000) / 1000.0 >= rate`

    实测（各 2 万次）：
        rate=0.2 → 命中 **51.6%**
        rate=0.5 → 命中 **39.0%**

    形状像 `1 - rate/2`，说明纳秒计时的低位**不是均匀分布**。

    后果比"样本少"更糟：样本量是配置的好几倍、且偏离方向不可预测，
    所有聚合数字都建立在一个未声明也未验证的分布上。

    改用 `random.random()` 后只能验证**统计性质**（无法精确控制单次抽样）。
    """

    @pytest.mark.parametrize("rate", [0.1, 0.2, 0.5, 0.8])
    def test_hit_rate_matches_config(self, monkeypatch, tmp_path, rate):
        monkeypatch.setenv("RAG_ONLINE_EVAL_RATE", str(rate))
        ev = OnlineEvaluator(path=tmp_path / "s.jsonl")
        trials = 20000
        hits = sum(1 for _ in range(trials) if ev._sampled())
        got = hits / trials
        # 2 万次抽样下标准差约 sqrt(p(1-p)/n) ≈ 0.004，0.02 容差足够宽
        assert abs(got - rate) < 0.02, (
            f"rate={rate} 实测命中率 {got:.4f}，偏离过大（采样有偏）"
        )

    def test_rate_one_always_true(self, monkeypatch, tmp_path):
        monkeypatch.setenv("RAG_ONLINE_EVAL_RATE", "1")
        ev = OnlineEvaluator(path=tmp_path / "s.jsonl")
        assert all(ev._sampled() for _ in range(50))

    def test_rate_zero_always_false(self, monkeypatch, tmp_path):
        monkeypatch.setenv("RAG_ONLINE_EVAL_RATE", "0")
        ev = OnlineEvaluator(path=tmp_path / "s.jsonl")
        assert not any(ev._sampled() for _ in range(50))

    def test_sampling_happens_before_evaluate(self, monkeypatch, tmp_path):
        """★ 采样判定必须在 `evaluate()` **之前**。

        `evaluate()` 单次约 38µs；rate=0.2 时不提前判定就有 80% 的调用是白做的。
        """
        monkeypatch.setenv("RAG_ONLINE_EVAL_RATE", "0")  # 永不采样
        ev = OnlineEvaluator(path=tmp_path / "s.jsonl")
        called: list[int] = []

        import src.eval_online.recorder as rec

        original = rec.evaluate
        monkeypatch.setattr(
            rec, "evaluate",
            lambda *a, **k: (called.append(1), original(*a, **k))[1],
        )
        ev.maybe_record({}, _done(), question="投标保证金上限是多少？")
        assert called == [], "未采样时不应调用 evaluate（白做功）"


class TestJudgeSnapshot:
    """裁判已实现（逐轮 RAGAS 风格裁判）—— 快照必须如实声明。

    历史上这里断言的是"未实现"；裁判落地后源码改为 True，
    测试未同步会一直红。保留这组断言是为了防止快照把功能状态标错。
    """

    def test_snapshot_declares_judge_implemented(self, tmp_path):
        snap = OnlineEvaluator(path=tmp_path / "s.jsonl").snapshot()
        assert snap.get("judge_implemented") is True, (
            "裁判已实现，快照不能继续标成未实现"
        )
        assert snap.get("judge_sampled") == 0
        assert snap.get("judge_failed") == 0


class TestJudgePath:
    """裁判**开启**路径：此前全模块强制 `JUDGE_RATE=0`，这条链路没有覆盖。

    用 fake `judge_turn` 验证三件事：
        ① 采样率 1.0 时确实入队并落一条 `kind=judge` 记录；
        ② `read_rows()` 按 sample_id 把分数合并回基础行；
        ③ 裁判行只存分数，不落 question/answer/sources（隐私边界）。
    """

    _Q = "投标保证金比例上限是多少？"
    _A = "投标保证金不得超过招标项目估算价的百分之二。"

    @staticmethod
    def _fake_judge(monkeypatch, result: dict):
        monkeypatch.setattr(
            "src.eval_online.judge.judge_turn",
            lambda **kwargs: result,
        )

    def test_judge_on_merges_scores_into_base_row(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RAG_ONLINE_EVAL_JUDGE_RATE", "1")
        self._fake_judge(monkeypatch, {
            "ok": True, "faithfulness": 0.9, "relevancy": 0.8,
            "judge_model": "fake-judge", "error": "",
        })
        path = tmp_path / "j.jsonl"
        ev = OnlineEvaluator(path=path)
        ev.maybe_record(
            {}, _done(), question=self._Q, answer=self._A,
            sources=[{"question": "保证金", "answer": "不超过 2%"}],
        )
        # 两个队列分别 join：基础行与裁判行都真正落盘后再读，避免 sleep 猜时间
        ev._q.join()
        ev._jq.join()

        assert ev.judge_sampled == 1
        assert ev.judge_ok == 1
        assert ev.judge_failed == 0

        rows = ev.read_rows()
        assert len(rows) == 1
        assert rows[0]["faithfulness"] == pytest.approx(0.9)
        assert rows[0]["relevancy"] == pytest.approx(0.8)
        assert rows[0]["judge_model"] == "fake-judge"

        raw = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        judge_rows = [r for r in raw if r.get("kind") == "judge"]
        assert len(judge_rows) == 1
        for field in ("question", "answer", "sources"):
            assert field not in judge_rows[0], f"裁判行不得落 {field}"

    def test_judge_error_marks_base_row_without_scores(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RAG_ONLINE_EVAL_JUDGE_RATE", "1")
        self._fake_judge(monkeypatch, {
            "ok": False, "faithfulness": None, "relevancy": None,
            "judge_model": "fake-judge", "error": "boom",
        })
        ev = OnlineEvaluator(path=tmp_path / "j.jsonl")
        ev.maybe_record({}, _done(), question=self._Q, answer=self._A)
        ev._q.join()
        ev._jq.join()

        assert ev.judge_sampled == 1
        assert ev.judge_failed == 1
        rows = ev.read_rows()
        assert len(rows) == 1
        assert rows[0].get("judge_failed") is True
        assert "faithfulness" not in rows[0]


class TestTailReading:
    """★ `read_rows` 曾经全量读取再切片（自查发现）。

    实测 6.8MB / 17 万行耗时 0.34s，按 `MAX_FILE_BYTES=50MB` 外推约 **2.5s** ——
    而 `limit` 只影响返回值，不影响读取量。

    改为从尾部**自适应**反向读块：

        第一版按 `want * 2048` 定块大小（假设单行 2KB），实测 `want=2000`
        时块 = 4MB，6.8MB 的文件**一次就读掉 57%** —— 退化成了"大部分全量读取"。
        根因是**行宽未知**（真实 641 字节，我的测试数据 42 字节）。

    自适应后：18MB（行宽 641 字节）耗时 **0.031s**。
    """

    @staticmethod
    def _write(path, count: int, *, pad: int = 0):
        with path.open("w", encoding="utf-8") as f:
            for i in range(count):
                row: dict = {"i": i}
                if pad:
                    row["pad"] = "x" * pad
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def test_returns_last_n_in_order(self, tmp_path):
        p = tmp_path / "a.jsonl"
        self._write(p, 5000, pad=200)
        rows = OnlineEvaluator(path=p).read_rows(limit=100)
        assert len(rows) == 100
        assert [r["i"] for r in rows] == list(range(4900, 5000)), (
            "应是最后 100 条且顺序正确"
        )

    @pytest.mark.parametrize("pad", [0, 100, 900, 3000])
    def test_various_row_widths(self, tmp_path, pad):
        """★ 行宽未知是块大小必须自适应的原因。"""
        p = tmp_path / f"w{pad}.jsonl"
        self._write(p, 3000, pad=pad)
        rows = OnlineEvaluator(path=p).read_rows(limit=200)
        assert len(rows) == 200
        assert [r["i"] for r in rows] == list(range(2800, 3000))

    def test_limit_larger_than_file(self, tmp_path):
        p = tmp_path / "small.jsonl"
        self._write(p, 7)
        assert len(OnlineEvaluator(path=p).read_rows(limit=999)) == 7

    def test_reads_rotated_backup_in_order(self, tmp_path):
        """备份时间更早，应排在前面。"""
        p = tmp_path / "r.jsonl"
        p.write_text(
            json.dumps({"i": 1}) + "\n" + json.dumps({"i": 2}) + "\n",
            encoding="utf-8",
        )
        p.with_suffix(".jsonl.1").write_text(
            json.dumps({"i": 90}) + "\n" + json.dumps({"i": 91}) + "\n",
            encoding="utf-8",
        )
        rows = OnlineEvaluator(path=p).read_rows()
        assert [r["i"] for r in rows] == [90, 91, 1, 2]

    def test_tolerates_bad_lines_at_tail(self, tmp_path):
        """坏行（含被截断的最后一行）跳过，不影响整体。"""
        p = tmp_path / "bad.jsonl"
        body = "\n".join(json.dumps({"i": i}) for i in range(20))
        p.write_text(
            body + "\n" + "这不是 JSON\n" + '{"i": 999, "trunc',
            encoding="utf-8",
        )
        rows = OnlineEvaluator(path=p).read_rows(limit=5)
        # 有效数据是 0..19 共 20 条；尾部两条坏行被跳过后，最近 5 条是 15..19
        assert [r["i"] for r in rows] == [15, 16, 17, 18, 19], "坏行被跳过且顺序正确"

    def test_bounded_read_on_large_file(self, tmp_path, monkeypatch):
        """★ 大文件不得全量读取。

        直接统计**读入字节数**，断言远小于文件大小 ——
        否则"尾部读取"只是名义上的，实际仍在读全文件（第一版就是这样）。
        """
        p = tmp_path / "big.jsonl"
        self._write(p, 30000, pad=600)   # 约 20MB
        size = p.stat().st_size

        real_open = pathlib.Path.open
        read_bytes = {"n": 0}

        class _Counter:
            def __init__(self, fh):
                self._fh = fh

            def __enter__(self):
                self._fh.__enter__()
                return self

            def __exit__(self, *a):
                return self._fh.__exit__(*a)

            def read(self, n=-1):
                data = self._fh.read(n)
                read_bytes["n"] += len(data)
                return data

            def seek(self, *a):
                return self._fh.seek(*a)

        monkeypatch.setattr(
            pathlib.Path, "open",
            lambda self, *a, **k: _Counter(real_open(self, *a, **k)),
        )
        rows = OnlineEvaluator(path=p).read_rows(limit=500)

        assert len(rows) == 500
        assert read_bytes["n"] < size * 0.5, (
            f"读入 {read_bytes['n']} / 文件 {size} 字节 —— 超过一半，"
            f"说明退化成全量读取了"
        )


# ===========================================================================
# 六、Agent 编排行为信号
#
# 在线没有参考答案，故测不了"答得对不对"。但**编排行为**本身就是质量信号：
# 轮数多说明在打转、工具失败说明依据变弱、提供商切换说明主模型降级。
# 这些都能从 done 帧直接推导，**零额外开销**。
# ===========================================================================


def _done_agent(**over) -> dict:
    """构造一个带完整 Agent 行为的 done 帧。"""
    base = {
        "tool_calls": None,  # 占位，避免误用
        "tool_names": ["search_knowledge_base"],
        "tool_timings": [
            {"round": 1, "name": "search_knowledge_base", "ms": 100,
             "ok": True, "sources": 5, "chars": 900},
        ],
        "provider_chain": ["deepseek"],
        "provider_switch_count": 0,
        "provider_errors": [],
        "compaction": None,
        "memory_examples": [],
        "plan": None,
    }
    base.pop("tool_calls")
    base.update(over)
    return base


class TestAgentSignals:
    """从 done 帧推导 Agent 编排行为。"""

    def test_basic_counts(self):
        row = evaluate({}, {**_done(), **_done_agent()},
                       question="投标保证金比例上限是多少？")
        assert row["agent_tool_calls"] == 1
        assert row["agent_rounds"] == 1
        assert row["agent_distinct_tools"] == 1
        assert row["agent_tool_failures"] == 0
        assert row["agent_repeated_calls"] == 0

    def test_rounds_is_max_not_count(self):
        """★ 轮次取 `round` 的**最大值**，不是调用次数。

        同一轮可以并行调多个工具（`round` 相同），
        用调用次数当轮数会把"一轮内并行 3 个工具"误判成"打了 3 轮"。
        """
        row = evaluate({}, {**_done(), **_done_agent(
            tool_names=["a", "b", "c"],
            tool_timings=[
                {"round": 1, "name": "a", "ok": True, "sources": 1},
                {"round": 1, "name": "b", "ok": True, "sources": 1},
                {"round": 1, "name": "c", "ok": True, "sources": 1},
            ],
        )}, question="采购数量最多的标的物是哪些？")
        assert row["agent_tool_calls"] == 3
        assert row["agent_rounds"] == 1, "同一轮的 3 次调用轮数应为 1"

    def test_repeated_calls_counted(self):
        """同一工具调 3 次 → 重复 2 次。"""
        row = evaluate({}, {**_done(), **_done_agent(
            tool_names=["search_knowledge_base"] * 3,
            tool_timings=[
                {"round": i, "name": "search_knowledge_base", "ok": True, "sources": 3}
                for i in (1, 2, 3)
            ],
        )}, question="投标保证金比例上限是多少？")
        assert row["agent_repeated_calls"] == 2
        assert row["agent_distinct_tools"] == 1

    def test_failures_and_empty_results(self):
        row = evaluate({}, {**_done(), **_done_agent(
            tool_names=["a", "b", "c"],
            tool_timings=[
                {"round": 1, "name": "a", "ok": True, "sources": 3},
                {"round": 2, "name": "b", "ok": False, "sources": 0},
                {"round": 3, "name": "c", "ok": True, "sources": 0},
            ],
        )}, question="采购数量最多的标的物是哪些？")
        assert row["agent_tool_failures"] == 1
        # 只有"成功但 0 条"才算空结果；失败那次不计（它已被 failures 统计）
        assert row["agent_empty_result_calls"] == 1

    def test_provider_health(self):
        row = evaluate({}, {**_done(), **_done_agent(
            provider_chain=["deepseek", "zhipu"],
            provider_switch_count=1,
            provider_errors=[{"provider": "deepseek", "error": "503"}],
        )}, question="投标保证金比例上限是多少？")
        assert row["provider_switches"] == 1
        assert row["provider_errors"] == 1
        assert row["provider_count"] == 2

    def test_compaction_and_plan_flags(self):
        row = evaluate({}, {**_done(), **_done_agent(
            compaction={"dropped": 3}, plan={"steps": ["a"]},
            memory_examples=[{"q": "x"}],
        )}, question="投标保证金比例上限是多少？")
        assert row["compacted"] is True
        assert row["has_plan"] is True
        assert row["memory_examples"] == 1

    def test_no_tools_called(self):
        row = evaluate({}, {**_done(), **_done_agent(
            tool_names=[], tool_timings=[],
        )}, question="投标保证金比例上限是多少？")
        assert row["agent_tool_calls"] == 0
        assert row["agent_rounds"] == 0
        assert row["agent_distinct_tools"] == 0

    def test_malformed_timings_ignored(self):
        """畸形输入不抛异常（评测是旁路）。"""
        row = evaluate({}, {**_done(), **_done_agent(
            tool_timings=[None, "x", {"round": "abc"}, {}],
            tool_names=[None, "a", ""],
        )}, question="投标保证金比例上限是多少？")
        assert row["agent_tool_calls"] == 2, "只计 dict 项"
        assert row["agent_rounds"] == 0, "round 解析不出就是 0"

    def test_duplicate_clarified_key(self):
        """`agent_clarified` 与 `clarified` 同源 —— 不应出现两个不同值。"""
        d = {**_done(), **_done_agent(clarified=True)}
        row = evaluate({}, d, question="我们这个项目可以单一来源采购吗")
        assert row["clarified"] == row["agent_clarified"] is True


class TestAgentAggregation:
    """聚合口径 —— 尤其是"正常值不是 0"的那几个。"""

    def _row(self, **over):
        d = {**_done(), **_done_agent(**over)}
        return evaluate({}, d, question="投标保证金比例上限是多少？")

    def test_multi_round_rate(self):
        """★ 轮数的正常值是 1，不是 0。

        若用"大于 0 的比例"，所有正常请求都会被算成"发生过"，
        得到恒为 1.0 的无意义指标。故 `_multi_round_rate` 判的是 `> 1`。
        """
        rows = [
            self._row(),                                   # 1 轮
            self._row(tool_names=["a", "b"],
                      tool_timings=[{"round": 1, "name": "a", "ok": True, "sources": 1},
                                    {"round": 2, "name": "b", "ok": True, "sources": 1}]),
        ]
        s = summarize_rows(rows)
        assert s["agent_multi_round_rate"] == 0.5, (
            "1 轮不应被算作多轮（正常值不是 0）"
        )

    def test_no_tool_rate(self):
        """★ 工具调用数的正常值是 ≥1，故判的是 `== 0`。"""
        rows = [
            self._row(),
            self._row(tool_names=[], tool_timings=[]),
        ]
        assert summarize_rows(rows)["agent_no_tool_rate"] == 0.5

    def test_repeat_rate(self):
        rows = [
            self._row(),
            self._row(tool_names=["a", "a"],
                      tool_timings=[{"round": 1, "name": "a", "ok": True, "sources": 1},
                                    {"round": 2, "name": "a", "ok": True, "sources": 1}]),
        ]
        s = summarize_rows(rows)
        assert s["agent_repeat_rate"] == 0.5
        assert s["agent_repeated_calls_total"] == 1

    def test_provider_switch_rate(self):
        rows = [
            self._row(),
            self._row(provider_chain=["deepseek", "zhipu"], provider_switch_count=1,
                      provider_errors=[{"provider": "deepseek", "error": "503"}]),
        ]
        s = summarize_rows(rows)
        assert s["provider_switch_rate"] == 0.5
        assert s["provider_switches_total"] == 1
        assert s["provider_error_rate"] == 0.5

    def test_rates_none_without_samples(self):
        """无样本时为 None（不是 0）—— 与既有口径一致。"""
        rows = [evaluate({}, {**_done(clarified=True), **_done_agent()},
                         question="我们这个项目可以单一来源采购吗")]
        s = summarize_rows(rows)
        assert s["evaluable_count"] == 0
        for k in ("agent_multi_round_rate", "agent_no_tool_rate",
                  "agent_repeat_rate", "provider_switch_rate"):
            assert s[k] is None, f"{k} 无样本时应为 None"

    def test_caveats_mention_proxy_nature(self):
        """口径提醒仍在下发（新增指标也属代理信号）。"""
        s = summarize_rows([self._row()])
        joined = " ".join(s.get("caveats") or [])
        assert "参考答案" in joined


class TestFreeChecksFirst:
    """★ 免费判据必须先于 `evaluate()`（自查发现的浪费）。

    `maybe_record` 的判定顺序曾经是：
        采样判定 → `evaluate()` → `is_evaluable()`

    但 `clarified` 与问题长度**在 done 帧里本来就有**，是免费的，
    而 `evaluate()` 要做 27 项提取。追问请求（实测占样本 1/4）
    本可以在提取前就退出。

    实现修复后，这里用"永不采样"来隔离：若免费判据真的前置，
    `evaluate()` 一次都不该被调用。
    """

    def test_clarified_skips_evaluate(self, monkeypatch, tmp_path):
        monkeypatch.setenv("RAG_ONLINE_EVAL_RATE", "1")
        ev = OnlineEvaluator(path=tmp_path / "c.jsonl")
        called: list[int] = []

        import src.eval_online.recorder as rec

        original = rec.evaluate
        monkeypatch.setattr(
            rec, "evaluate",
            lambda *a, **k: (called.append(1), original(*a, **k))[1],
        )

        # 追问请求：done 帧里 clarified=True
        ev.maybe_record({}, {**_done(clarified=True)},
                        question="我们这个项目可以单一来源采购吗")
        assert called == [], (
            "追问请求应在 evaluate() 之前就被排除（clarified 是免费判据）"
        )
        assert ev.accepted == 0

    def test_short_question_skips_evaluate(self, monkeypatch, tmp_path):
        monkeypatch.setenv("RAG_ONLINE_EVAL_RATE", "1")
        ev = OnlineEvaluator(path=tmp_path / "s.jsonl")
        called: list[int] = []

        import src.eval_online.recorder as rec

        original = rec.evaluate
        monkeypatch.setattr(
            rec, "evaluate",
            lambda *a, **k: (called.append(1), original(*a, **k))[1],
        )
        ev.maybe_record({}, _done(), question="你好")
        assert called == [], "极短问题应在 evaluate() 之前被排除"
        assert ev.accepted == 0

    def test_normal_still_evaluated(self, monkeypatch, tmp_path):
        """对照组：正常请求仍要走进 evaluate（确认短路没有过宽）。"""
        monkeypatch.setenv("RAG_ONLINE_EVAL_RATE", "1")
        ev = OnlineEvaluator(path=tmp_path / "n.jsonl")
        called: list[int] = []

        import src.eval_online.recorder as rec

        original = rec.evaluate
        monkeypatch.setattr(
            rec, "evaluate",
            lambda *a, **k: (called.append(1), original(*a, **k))[1],
        )
        ev.maybe_record({}, _done(), question="投标保证金比例上限是多少？")
        assert len(called) == 1
        assert ev.accepted == 1

    def test_constant_matches_signals(self):
        """短路的阈值必须与 `signals.is_evaluable` 用的一致。

        两处若不一致不会产生错误数据（`is_evaluable` 仍会兜底），
        但会出现"某长度的问题被短路了、另一处却认为可评"的困惑。
        """
        from src.eval_online.recorder import MIN_EVALUABLE_QUESTION_CHARS
        from src.eval_online.signals import _num, is_evaluable

        n = MIN_EVALUABLE_QUESTION_CHARS
        assert is_evaluable({"question_chars": n}) is True
        assert is_evaluable({"question_chars": n - 1}) is False

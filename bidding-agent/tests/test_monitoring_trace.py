"""LLM 调用追踪的测试（usage / 单次耗时 / 工具耗时）。

★ 为什么需要它
==========
审计发现四个真实缺口，都属"数据本来就有但被丢掉"：

    ① **token 用量完全没读**。实测 DeepSeek 响应里一直带着 `usage`：
           {prompt_tokens, completion_tokens, total_tokens,
            prompt_cache_hit_tokens, prompt_cache_miss_tokens}
       而 `extract_message` / `iter_sse_json` 里 `usage` 出现 **0 次**。
    ② **prompt 缓存命中数丢失**——DeepSeek 对命中部分打折计费，
       它是直接的省钱信号。
    ③ 单次 LLM 调用明细缺失（只有阶段合计）。
    ④ 单个工具执行耗时缺失（只有整阶段合计）。

★ 本文件测什么
==========
    ① `parse_usage` 的字段兼容与"不编造 0"
    ② `LLMTraceRecorder` 的聚合口径与内存上界
    ③ `attach/detach` 的行为（含"不拷贝客户端"这条关键契约）
    ④ 客户端真的把 usage 记下来（用假 HTTP，不打真实 API）
"""

from __future__ import annotations

import json

import pytest

from src.monitoring_trace import (
    MAX_RECORDS,
    LLMCall,
    LLMTraceRecorder,
    Stopwatch,
    attach_recorder,
    detach_recorder,
    parse_usage,
    render_messages,
)


class TestParseUsage:
    """字段名在不同厂商/版本间不一致，必须兼容；取不到返回 None。"""

    def test_standard_openai_fields(self):
        got = parse_usage({"usage": {
            "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
        }})
        assert got["prompt_tokens"] == 10
        assert got["completion_tokens"] == 5
        assert got["total_tokens"] == 15

    def test_deepseek_cache_fields(self):
        """DeepSeek 额外给缓存命中/未命中——**省钱信号**，必须取到。"""
        got = parse_usage({"usage": {
            "prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
            "prompt_cache_hit_tokens": 76, "prompt_cache_miss_tokens": 24,
        }})
        assert got["cache_hit_tokens"] == 76
        assert got["cache_miss_tokens"] == 24

    def test_cache_via_prompt_tokens_details(self):
        """部分实现把缓存数放在 `prompt_tokens_details.cached_tokens`。"""
        got = parse_usage({"usage": {
            "prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11,
            "prompt_tokens_details": {"cached_tokens": 7},
        }})
        assert got["cache_hit_tokens"] == 7

    def test_explicit_field_wins_over_details(self):
        """两个字段都在时，显式字段优先（它是 DeepSeek 的正式口径）。"""
        got = parse_usage({"usage": {
            "prompt_cache_hit_tokens": 9,
            "prompt_tokens_details": {"cached_tokens": 3},
        }})
        assert got["cache_hit_tokens"] == 9

    def test_alternative_field_names(self):
        """`input_tokens` / `output_tokens`（部分兼容层用）。"""
        got = parse_usage({"usage": {"input_tokens": 4, "output_tokens": 2}})
        assert got["prompt_tokens"] == 4
        assert got["completion_tokens"] == 2

    def test_missing_usage_returns_none_not_zero(self):
        """★ 取不到必须是 None，**绝不能编造 0**。

        `total_tokens=0` 会被读成"这次没花 token"，而事实是"没拿到这个数据"——
        与本项目"不编造数据"的一贯原则一致。
        """
        for payload in (None, {}, {"usage": None}, {"usage": "bad"}):
            got = parse_usage(payload)
            for k, v in got.items():
                assert v is None, f"{payload!r} 的 {k} 应为 None，实得 {v}"

    def test_partial_usage_keeps_others_none(self):
        got = parse_usage({"usage": {"prompt_tokens": 3}})
        assert got["prompt_tokens"] == 3
        assert got["completion_tokens"] is None
        assert got["total_tokens"] is None


class TestRecorder:
    def test_empty_summary(self):
        s = LLMTraceRecorder().summary()
        assert s["call_count"] == 0
        assert s["total_ms"] == 0
        # 无调用时 token 全为 None（不是 0）
        assert s["total_tokens"] is None

    def test_aggregates_tokens_and_ms(self):
        rec = LLMTraceRecorder()
        rec.record(LLMCall(kind="decision", elapsed_ms=100, total_tokens=10,
                           prompt_tokens=8, completion_tokens=2))
        rec.record(LLMCall(kind="stream", elapsed_ms=200, total_tokens=30,
                           prompt_tokens=20, completion_tokens=10))
        s = rec.summary()
        assert s["total_ms"] == 300
        assert s["total_tokens"] == 40
        assert s["prompt_tokens"] == 28
        assert s["by_kind"]["decision"] == {"count": 1, "ms": 100}
        assert s["by_kind"]["stream"] == {"count": 1, "ms": 200}

    def test_partial_token_data_not_zeroed(self):
        """★ 只要有一条记录没带 token，求和就不该把它当 0 拉低总数。

        这里两条都没带 → 返回 None；一条带一条不带 → 只累加带的那条。
        """
        rec = LLMTraceRecorder()
        rec.record(LLMCall(kind="decision", elapsed_ms=1))
        rec.record(LLMCall(kind="stream", elapsed_ms=1, total_tokens=50))
        assert rec.summary()["total_tokens"] == 50, (
            "缺失的 token 不该被当作 0 参与求和（会得出'花了 50'这个正确值，"
            "但若实现写成 sum(c.total_tokens or 0) 则两条都不带时会给 0 而非 None）"
        )
        rec2 = LLMTraceRecorder()
        rec2.record(LLMCall(kind="decision", elapsed_ms=1))
        assert rec2.summary()["total_tokens"] is None

    def test_failed_calls_counted_separately(self):
        """失败次数是 provider 稳定性的直接信号，必须单独统计。"""
        rec = LLMTraceRecorder()
        rec.record(LLMCall(kind="decision", elapsed_ms=5, ok=False, error="401"))
        rec.record(LLMCall(kind="decision", elapsed_ms=7, ok=True))
        s = rec.summary()
        assert s["call_count"] == 2
        assert s["failed_count"] == 1
        # 失败的耗时也要计入总耗时（它也真的花了时间）
        assert s["total_ms"] == 12

    def test_reasoning_chars_summed(self):
        """思维链长度用于区分'慢是因为在想'还是'慢是因为输出长'。"""
        rec = LLMTraceRecorder()
        rec.record(LLMCall(kind="thinking", elapsed_ms=1, reasoning_chars=100))
        rec.record(LLMCall(kind="thinking", elapsed_ms=1, reasoning_chars=50))
        assert rec.summary()["reasoning_chars"] == 150

    def test_memory_bounded(self):
        """★ 无界增长在长跑服务里就是内存泄漏。"""
        rec = LLMTraceRecorder()
        for i in range(MAX_RECORDS + 50):
            rec.record(LLMCall(kind="stream", elapsed_ms=i))
        s = rec.summary()
        assert s["call_count"] == MAX_RECORDS
        # 溢出**显式计数**，不静默丢弃——"记录被截断"本身是要知道的事实
        assert s["dropped_count"] == 50

    def test_calls_returns_copy(self):
        """`calls()` 返回副本，外部修改不影响记录器内部状态。"""
        rec = LLMTraceRecorder()
        rec.record(LLMCall(kind="stream", elapsed_ms=1))
        got = rec.calls()
        got.clear()
        assert rec.summary()["call_count"] == 1

    def test_summary_is_json_serializable(self):
        """原样进 SSE 帧，必须可 JSON 序列化（`asdict` 已把 dataclass 摊平）。"""
        rec = LLMTraceRecorder()
        rec.record(LLMCall(kind="decision", elapsed_ms=1, tools=["search_knowledge_base"]))
        json.dumps(rec.summary(), ensure_ascii=False)  # 不应抛


class TestAttachDetachContract:
    """★ 关键契约：**不拷贝客户端**。

    第一版 `_attach_tracer` 用 `copy.copy(client)` 再挂记录器，
    导致调用落到副本上——测试里 `assert fake.raw_calls == 4` 恒失败
    （实测一次挂掉 5 个用例）。而"在原对象上观察调用"是合理的测试写法，
    不该为观测功能逼测试改写法。

    故这里锁住：`attach_recorder` 必须**原地**挂载，同一个对象。
    """

    def test_attach_is_in_place(self):
        class Fake:
            trace_recorder = None

        f = Fake()
        rec = LLMTraceRecorder()
        attach_recorder(f, rec)
        assert f.trace_recorder is rec, "必须原地挂载，不能返回副本"

    def test_detach_clears(self):
        class Fake:
            pass

        f = Fake()
        attach_recorder(f, LLMTraceRecorder())
        assert f.trace_recorder is not None
        detach_recorder(f)
        assert f.trace_recorder is None

    def test_attach_tolerates_readonly_attribute(self):
        """只读属性 / `__slots__` 客户端：挂不上也不该抛（观测是旁路）。"""

        class ReadOnly:
            @property
            def trace_recorder(self):
                return None

        attach_recorder(ReadOnly(), LLMTraceRecorder())  # 不应抛

    def test_detach_tolerates_broken_object(self):
        class Boom:
            @property
            def trace_recorder(self):
                return None

            @trace_recorder.setter
            def trace_recorder(self, v):
                raise RuntimeError("炸")

        detach_recorder(Boom())  # 不应抛

    def test_none_inputs_are_safe(self):
        attach_recorder(None, LLMTraceRecorder())  # 不应抛
        attach_recorder(object(), None)  # 不应抛
        detach_recorder(None)  # 不应抛


class TestStopwatch:
    def test_measures_elapsed(self):
        import time

        with Stopwatch() as w:
            time.sleep(0.05)
        assert w.ms >= 40, f"应至少 40ms，实得 {w.ms}"

    def test_records_on_exception(self):
        """★ 异常路径也要计时——失败调用的耗时是排查超时的关键数据。"""
        with pytest.raises(ValueError):
            with Stopwatch() as w:
                raise ValueError("boom")
        # 上下文管理器退出后 ms 已被写入（`__exit__` 在异常传播前执行）
        assert w.ms >= 0


class TestClientRecordsUsage:
    """客户端层：真的把 usage 与耗时记下来（假 HTTP，不打真实 API）。"""

    @staticmethod
    def _patch_http(monkeypatch, body: dict, *, stream_lines=None):
        import src.http_client as hc

        class Resp:
            status_code = 200

            def __init__(self):
                self.text = json.dumps(body)

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def json(self):
                return body

            def read(self):
                return json.dumps(body).encode()

            def iter_lines(self):
                yield from (stream_lines or [])

        class Http:
            def post(self, url, **kw):
                return Resp()

            def stream(self, method, url, **kw):
                return Resp()

        monkeypatch.setattr(hc, "get_http_client", lambda: Http())

    def test_non_streaming_records_usage(self, monkeypatch):
        from src.clients.base_client import OpenAICompatClient

        self._patch_http(monkeypatch, {
            "choices": [{"message": {"content": "答"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10,
                      "prompt_cache_hit_tokens": 5, "prompt_cache_miss_tokens": 2},
        })
        rec = LLMTraceRecorder()
        c = OpenAICompatClient(api_key="k", model="m", base_url="http://x/v1",
                               provider="deepseek")
        attach_recorder(c, rec)
        out = c.chat_raw([{"role": "user", "content": "q"}])

        assert out == "答"
        calls = rec.calls()
        assert len(calls) == 1
        assert calls[0].total_tokens == 10
        assert calls[0].cache_hit_tokens == 5
        assert calls[0].kind == "decision"
        assert calls[0].model == "m"

    def test_failed_call_still_recorded(self, monkeypatch):
        """★ 失败也要记：失败次数是 provider 稳定性信号。"""
        from src.clients.base_client import OpenAICompatClient

        import src.http_client as hc

        class Boom:
            def post(self, url, **kw):
                raise RuntimeError("网络炸了")

            def stream(self, method, url, **kw):
                raise RuntimeError("网络炸了")

        monkeypatch.setattr(hc, "get_http_client", lambda: Boom())
        rec = LLMTraceRecorder()
        c = OpenAICompatClient(api_key="k", model="m", base_url="http://x/v1")
        attach_recorder(c, rec)

        with pytest.raises(RuntimeError):
            c.chat_raw([{"role": "user", "content": "q"}])

        calls = rec.calls()
        assert len(calls) == 1, "失败调用必须被记录"
        assert calls[0].ok is False
        assert "网络炸了" in calls[0].error

    def test_streaming_requests_usage(self, monkeypatch):
        """★ 流式必须显式要求回传 usage。

        实测：不加 `stream_options.include_usage` 时流式 chunk 里
        **完全没有 usage**（非流式则顶层一直有）。故断言请求体里带了它。
        """
        from src.clients.base_client import OpenAICompatClient

        import src.http_client as hc

        seen: list[dict] = []

        class Resp:
            status_code = 200
            text = ""

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def iter_lines(self):
                yield 'data: {"choices":[{"delta":{"content":"答"}}]}'
                yield 'data: {"choices":[],"usage":{"prompt_tokens":1,' \
                      '"completion_tokens":1,"total_tokens":2}}'
                yield "data: [DONE]"

        class Http:
            def stream(self, method, url, **kw):
                seen.append(kw.get("json") or {})
                return Resp()

        monkeypatch.setattr(hc, "get_http_client", lambda: Http())
        rec = LLMTraceRecorder()
        c = OpenAICompatClient(api_key="k", model="m", base_url="http://x/v1")
        attach_recorder(c, rec)

        out = "".join(c.chat_stream([{"role": "user", "content": "q"}]))

        assert out == "答"
        assert seen[0].get("stream_options") == {"include_usage": True}, (
            "流式请求必须带 stream_options.include_usage，否则拿不到 token 用量"
        )
        calls = rec.calls()
        assert len(calls) == 1
        assert calls[0].total_tokens == 2, (
            "usage 在最后一个 chunk 且该 chunk 的 choices 为空——"
            "解析时不能'没看到 choices 就跳过'"
        )
        assert calls[0].kind == "stream"

    def test_no_recorder_is_noop(self, monkeypatch):
        """未挂记录器时追踪代码静默跳过（零开销、不改既有行为）。"""
        from src.clients.base_client import OpenAICompatClient

        self._patch_http(monkeypatch, {
            "choices": [{"message": {"content": "答"}, "finish_reason": "stop"}],
        })
        c = OpenAICompatClient(api_key="k", model="m", base_url="http://x/v1")
        assert c.chat_raw([{"role": "user", "content": "q"}]) == "答"


class TestFallbackChainPropagation:
    """★ 链必须把记录器透传给链上每个客户端。

    链本身**不调用 LLM**——真正发请求的是被选中的那个客户端。
    不透传的话，多 provider 部署下追踪**完全失效**：
    `done` 帧里 `llm_trace.call_count` 恒为 0，看起来像"这轮没调用 LLM"。
    而单 provider 时正常——这种"取决于是否配了兜底 provider"的差异极难排查。
    """

    def test_recorder_reaches_resolved_client(self):
        from src.clients.base_client import OpenAICompatClient
        from src.clients.fallback import build_fallback_client

        # ★ resolver 必须**缓存实例**，与真实用法一致
        #
        # 真实 `core._build_llm_chain` 的 resolver 带 `lru_cache`，同一 provider
        # 永远返回同一个客户端。第一版测试替身每次调用都新建，于是
        # `_client_for` 返回的实例上没有记录器（记录器挂在前一个实例上），
        # 断言失败——**是替身不真实，不是实现有问题**。
        cache: dict[str, OpenAICompatClient] = {}

        def resolver(name: str):
            if name not in cache:
                cache[name] = OpenAICompatClient(
                    api_key="k", model=f"{name}-model",
                    base_url="http://x/v1", provider=name,
                )
            return cache[name]

        chain = build_fallback_client("deepseek", resolver)
        rec = LLMTraceRecorder()
        attach_recorder(chain, rec)

        client = chain._client_for("deepseek")
        assert client is not None
        assert client.trace_recorder is rec, (
            "链必须把记录器透传给链上客户端，否则多 provider 部署下追踪失效"
        )
        assert client is cache["deepseek"], "应拿到缓存的那个实例"

    def test_recorder_cleared_on_detach(self):
        from src.clients.base_client import OpenAICompatClient
        from src.clients.fallback import build_fallback_client

        cache: dict[str, OpenAICompatClient] = {}

        def resolver(name: str):
            if name not in cache:
                cache[name] = OpenAICompatClient(
                    api_key="k", model="m", base_url="http://x/v1", provider=name,
                )
            return cache[name]

        chain = build_fallback_client("deepseek", resolver)
        rec = LLMTraceRecorder()
        attach_recorder(chain, rec)
        first = chain._client_for("deepseek")
        assert first.trace_recorder is rec

        detach_recorder(chain)
        assert chain.trace_recorder is None
        assert first.trace_recorder is None, (
            "摘除必须转发到已解析的底层客户端——否则记录器被长期持有"
            "（内存泄漏），且绕过链的调用会记进已结束的请求"
        )


class TestContentCapture:
    """正文采集（`LANGFUSE_CAPTURE_CONTENT`）：**默认关**，开了才多两个字段。

    为什么这两个字段值得单独测：它们会离开本机（进 Langfuse），
    所以"默认是不是真的不发"必须被测试钉住，而不能靠"记得别开"。
    """

    @staticmethod
    def _patch_http(monkeypatch, body: dict, *, stream_lines=None):
        TestClientRecordsUsage._patch_http(monkeypatch, body, stream_lines=stream_lines)

    def test_off_by_default_no_text_collected(self, monkeypatch):
        """★ 默认关闭时，`prompt_text` / `response_text` 必须是空串。"""
        from src.clients.base_client import OpenAICompatClient

        self._patch_http(monkeypatch, {
            "choices": [{"message": {"content": "答"}, "finish_reason": "stop"}],
        })
        rec = LLMTraceRecorder()
        c = OpenAICompatClient(api_key="k", model="m", base_url="http://x/v1")
        attach_recorder(c, rec)
        c.chat_raw([{"role": "user", "content": "绝密问题ABC"}])

        call = rec.calls()[0]
        assert call.prompt_text == ""
        assert call.response_text == ""

    def test_on_collects_prompt_and_response(self, monkeypatch):
        from src.clients.base_client import OpenAICompatClient

        monkeypatch.setenv("LANGFUSE_CAPTURE_CONTENT", "true")
        monkeypatch.setenv("LANGFUSE_MAX_CAPTURE_CHARS", "1000")
        self._patch_http(monkeypatch, {
            "choices": [{"message": {"content": "答"}, "finish_reason": "stop"}],
        })
        rec = LLMTraceRecorder()
        c = OpenAICompatClient(api_key="k", model="m", base_url="http://x/v1")
        attach_recorder(c, rec)
        c.chat_raw([{"role": "user", "content": "绝密问题ABC"}])

        call = rec.calls()[0]
        assert "绝密问题ABC" in call.prompt_text
        assert call.response_text == "答"

    def test_streaming_accumulates_response(self, monkeypatch):
        from src.clients.base_client import OpenAICompatClient

        monkeypatch.setenv("LANGFUSE_CAPTURE_CONTENT", "true")
        monkeypatch.setenv("LANGFUSE_MAX_CAPTURE_CHARS", "1000")
        lines = [
            'data: {"choices": [{"delta": {"content": "你"}}]}',
            'data: {"choices": [{"delta": {"content": "好"}}]}',
            "data: [DONE]",
        ]
        self._patch_http(monkeypatch, {}, stream_lines=lines)
        rec = LLMTraceRecorder()
        c = OpenAICompatClient(api_key="k", model="m", base_url="http://x/v1")
        attach_recorder(c, rec)
        assert "".join(c.chat_stream([{"role": "user", "content": "q"}])) == "你好"

        call = rec.calls()[0]
        assert call.response_text == "你好"
        assert call.content_chars == 2

    def test_streaming_respects_capture_limit(self, monkeypatch):
        """流式下只能顺序累积，到了上限就停止缓存（内存不能随回答长度涨）。"""
        from src.clients.base_client import OpenAICompatClient

        monkeypatch.setenv("LANGFUSE_CAPTURE_CONTENT", "true")
        monkeypatch.setenv("LANGFUSE_MAX_CAPTURE_CHARS", "4")
        lines = ['data: {"choices": [{"delta": {"content": "abcdefgh"}}]}', "data: [DONE]"]
        self._patch_http(monkeypatch, {}, stream_lines=lines)
        rec = LLMTraceRecorder()
        c = OpenAICompatClient(api_key="k", model="m", base_url="http://x/v1")
        attach_recorder(c, rec)
        c.chat_stream([{"role": "user", "content": "q"}]).__next__()

        call = rec.calls()[0]
        assert len(call.response_text) <= 8
        assert call.content_chars == 8, "真实长度照旧统计（不会被采集上限影响）"

    def test_summary_excludes_text_fields(self, monkeypatch):
        """★ 正文不进 `summary()`：它会原样进 SSE 的 done 帧，把响应体撑大几十倍。"""
        rec = LLMTraceRecorder()
        rec.record(LLMCall(kind="stream", elapsed_ms=1, prompt_text="很长的prompt",
                           response_text="很长的回答"))
        row = rec.summary()["calls"][0]
        assert "prompt_text" not in row and "response_text" not in row
        assert row["kind"] == "stream", "其余字段照旧保留"

    def test_render_messages_keeps_head_and_tail(self):
        messages = [{"role": "system", "content": "S" * 100},
                    {"role": "user", "content": "尾部问题"}]
        out = render_messages(messages, 40)
        assert out.startswith("system:")
        assert out.endswith("尾部问题"), "★ 最后一条用户消息必须留着"
        assert "省略" in out

    def test_render_messages_disabled_by_zero_limit(self):
        assert render_messages([{"role": "user", "content": "q"}], 0) == ""
        assert render_messages(None, 100) == ""

    def test_render_messages_handles_multimodal_and_tool_calls(self):
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "看图"}]},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
        ]
        out = render_messages(messages, 500)
        assert "看图" in out
        assert "tool_calls=1" in out

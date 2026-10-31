"""压测脚本的纯函数与协议测试（不访问真实服务）。

统计/分类逻辑必须能单测：它们出错时不会崩，只会给出看起来合理的错数字。
"""

from __future__ import annotations

import httpx
import pytest

from eval.load_eval import (
    LoadSample,
    _request_once,
    build_payload,
    classify_status,
    percentile,
    run_level,
    summarize,
)


class TestPercentile:
    def test_known_values(self):
        values = [1.0, 2.0, 3.0, 4.0]
        assert percentile(values, 50) == pytest.approx(2.5)
        assert percentile(values, 95) == pytest.approx(3.85)
        assert percentile(values, 99) == pytest.approx(3.97)
        assert percentile(values, 100) == 4.0

    def test_empty_and_single(self):
        assert percentile([], 95) == 0.0
        assert percentile([7.0], 95) == 7.0


class TestSummarize:
    def test_counts_errors_and_percentiles(self):
        samples = [
            LoadSample(ok=True, latency_ms=100.0, status=200),
            LoadSample(ok=True, latency_ms=200.0, status=200),
            LoadSample(ok=False, latency_ms=300.0, status=429,
                       error="rate_limited: HTTP 429 tier=llm"),
            LoadSample(ok=True, latency_ms=400.0, status=200),
        ]
        s = summarize(samples, wall_s=2.0)
        assert s["requests"] == 4
        assert s["ok"] == 3
        assert s["errors"] == 1
        assert s["error_rate"] == 0.25
        assert s["rps"] == 2.0
        assert s["latency_ms"]["p50"] == pytest.approx(250.0)
        assert s["status_codes"] == {"200": 3, "429": 1}
        assert s["error_kinds"] == {"rate_limited": 1}

    def test_empty_samples_does_not_divide_by_zero(self):
        s = summarize([], wall_s=0.0)
        assert s["requests"] == 0
        assert s["rps"] == 0.0
        assert s["error_rate"] is None
        assert s["latency_ms"]["p95"] is None


class TestPayloadAndClassification:
    def test_payload_shapes(self):
        assert build_payload("ask", "问题", top_k=3) == {"question": "问题", "top_k": 3}
        chat = build_payload("chat", "问题")
        assert chat["web_search_mode"] == "off"
        assert chat["deep_thinking_enabled"] is False

    def test_unknown_mode_rejected(self):
        with pytest.raises(ValueError):
            build_payload("nope", "问题")

    def test_status_classification(self):
        assert classify_status(200) == "ok"
        assert classify_status(429) == "rate_limited"
        assert classify_status(500) == "server_error"
        assert classify_status(404) == "client_error"


class TestRequestOnce:
    def test_successful_json_request(self):
        def handler(request):
            return httpx.Response(200, json={"answer": "ok", "elapsed_ms": 12.5})

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            sample = _request_once(client, "http://test", "ask", "问题", 5)
        assert sample.ok is True
        assert sample.status == 200
        assert sample.server_ms == 12.5
        assert sample.latency_ms >= 0

    def test_rate_limited_reports_tier_and_retry(self):
        def handler(request):
            return httpx.Response(
                429, json={"detail": "频繁", "retry_after": 37, "tier": "llm"}
            )

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            sample = _request_once(client, "http://test", "chat", "问题", 5)
        assert sample.ok is False
        assert "rate_limited" in sample.error
        assert "tier=llm" in sample.error
        assert "retry_after=37" in sample.error

    def test_stream_records_ttft_and_done_elapsed(self):
        body = (
            ":" + " " * 20 + "\n\n"
            'data: {"type": "status", "content": "检索中"}\n\n'
            'data: {"type": "token", "content": "你"}\n\n'
            'data: {"type": "token", "content": "好"}\n\n'
            'data: {"type": "done", "elapsed_ms": 42}\n\n'
            "data: [DONE]\n\n"
        )

        def handler(request):
            return httpx.Response(200, content=body.encode("utf-8"))

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            sample = _request_once(client, "http://test", "stream", "问题", 5)
        assert sample.ok is True
        assert sample.ttft_ms is not None
        assert sample.server_ms == 42.0

    def test_stream_error_event_is_failure(self):
        body = 'data: {"type": "error", "content": "模型不可用"}\n\ndata: [DONE]\n\n'

        def handler(request):
            return httpx.Response(200, content=body.encode("utf-8"))

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            sample = _request_once(client, "http://test", "stream", "问题", 5)
        assert sample.ok is False
        assert "sse_error" in sample.error


class TestRunLevel:
    def test_runs_requested_number_with_concurrency(self):
        seen: list[str] = []

        def handler(request):
            seen.append(request.url.path)
            return httpx.Response(200, json={"answer": "ok"})

        samples, wall = run_level(
            "http://test", "ask", ["q1", "q2"],
            concurrency=2, requests=4, timeout_s=5, top_k=5,
            transport=httpx.MockTransport(handler),
        )
        assert len(samples) == 4
        assert all(s.ok for s in samples)
        assert seen == ["/api/ask"] * 4
        assert wall >= 0

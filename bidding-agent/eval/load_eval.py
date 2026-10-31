"""HTTP 并发压测：吞吐、P50/P95/P99、错误率、流式首 token 时间。

★ 为什么单独一个脚本，而不进 `run_all.py` 的评测线
-----------------------------------------------
`run_all` 的契约是「每条线可独立离线运行、产出可比 `SuiteReport`」。
压测需要**已启动的 API 服务**，结果强依赖机器与网络，而门禁的 2% 相对容差
会把环境抖动当成退化。故本脚本独立运行，`--json` 归档，发布前人工对比。

★ 两种口径
----------
    ask     `POST /api/ask`    只测检索，不调 LLM，适合 CI/日常冒烟
    chat    `POST /api/chat`   端到端（LLM + ReAct + 工具），发布前正式压测
    stream  `POST /api/chat/stream`  额外测首 token 时间（TTFT）

★ 压测前必须知道的服务端限制
--------------------------
    · 默认限流 30 次/60s/IP；本机所有请求共享 127.0.0.1，第 31 次起 429。
      没有关闭开关（设 0 无效），必须**启动服务前**设大 RATE_LIMIT_LLM。
    · 全局出站连接池 20：并发 LLM 流超过约 20 会排队，pool timeout 30s。
    · 同步端点线程池约 40：超过后在 ASGI 层排队，会计入客户端延迟。
    · 必须单进程启动（--workers 1），多 worker 会让限流/缓存各持一份。
    · 检索缓存 256 条：重复问题会命中缓存，测到的是缓存路径；
      用 `--clear-cache` 在每档前清空，或准备足够多的问题。

用法::

    python eval/load_eval.py --mode ask --concurrency 1,5,20 --requests 50
    python eval/load_eval.py --mode chat --concurrency 4 --requests 20 --json perf.json
    python eval/load_eval.py --mode stream --concurrency 4 --requests 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from eval.common import setup_stdio  # noqa: E402

DEFAULT_QUESTIONS = Path(__file__).resolve().parent / "test_questions.json"

ENDPOINTS = {
    "ask": "/api/ask",
    "chat": "/api/chat",
    "stream": "/api/chat/stream",
}

#: 并发上限：不是技术限制，是防止手滑把一台开发机打满。
MAX_CONCURRENCY = 200


@dataclass
class LoadSample:
    ok: bool
    latency_ms: float
    status: int
    error: str = ""
    server_ms: float | None = None
    ttft_ms: float | None = None


def percentile(values: list[float], q: float) -> float:
    """线性插值分位数。空列表返回 0.0（调用方负责显示"无样本"）。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * (q / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return float(ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo))


def summarize(samples: list[LoadSample], wall_s: float) -> dict:
    """把一批样本汇总成可比较的一行。**纯函数**，便于单测。"""
    latencies = [s.latency_ms for s in samples]
    ok_samples = [s for s in samples if s.ok]
    failures = [s for s in samples if not s.ok]
    error_kinds = Counter(s.error.split(":", 1)[0] for s in failures)
    ttfts = [s.ttft_ms for s in samples if s.ttft_ms is not None]
    server = [s.server_ms for s in samples if s.server_ms is not None]
    status_codes = Counter(str(s.status) for s in samples)
    return {
        "requests": len(samples),
        "ok": len(ok_samples),
        "errors": len(failures),
        "error_rate": round(len(failures) / len(samples), 4) if samples else None,
        "rps": round(len(samples) / wall_s, 2) if wall_s > 0 else 0.0,
        "latency_ms": {
            "mean": round(sum(latencies) / len(latencies), 1) if latencies else None,
            "p50": round(percentile(latencies, 50), 1) if latencies else None,
            "p95": round(percentile(latencies, 95), 1) if latencies else None,
            "p99": round(percentile(latencies, 99), 1) if latencies else None,
            "max": round(max(latencies), 1) if latencies else None,
        },
        "ttft_ms": {
            "p50": round(percentile(ttfts, 50), 1) if ttfts else None,
            "p95": round(percentile(ttfts, 95), 1) if ttfts else None,
        } if ttfts else None,
        "server_ms": {
            "p50": round(percentile(server, 50), 1) if server else None,
            "p95": round(percentile(server, 95), 1) if server else None,
        } if server else None,
        "status_codes": dict(status_codes),
        "error_kinds": dict(error_kinds),
    }


def classify_status(status: int) -> str:
    """HTTP 状态码 → 错误类别。429 必须单独点名（它是压测配置问题）。"""
    if status == 429:
        return "rate_limited"
    if status >= 500:
        return "server_error"
    if status >= 400:
        return "client_error"
    return "ok"


def build_payload(mode: str, question: str, top_k: int = 5) -> dict:
    if mode == "ask":
        return {"question": question, "top_k": top_k}
    if mode in ("chat", "stream"):
        # 关掉联网与思考：它们引入外部不确定性且会额外调用 LLM，
        # 会让"服务端并发能力"的测量变成"外部搜索/思考时长"的测量。
        return {
            "question": question,
            "web_search_mode": "off",
            "deep_thinking_enabled": False,
        }
    raise ValueError(f"未知模式：{mode}")


def load_questions(path: Path | None = None, limit: int = 0) -> list[str]:
    """读问题列表。支持 `[{"question": ...}]` 与 `["..."]` 两种形状。"""
    target = Path(path) if path is not None else DEFAULT_QUESTIONS
    data = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(f"问题文件必须是 JSON 数组：{target}")
    questions: list[str] = []
    for item in data:
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, dict):
            text = str(item.get("question") or "").strip()
        else:
            text = ""
        if text:
            questions.append(text)
    if not questions:
        raise SystemExit(f"问题文件里没有可用问题：{target}")
    if limit:
        questions = questions[: max(0, limit)]
    return questions


def _server_elapsed(payload: dict) -> float | None:
    value = payload.get("elapsed_ms")
    return float(value) if isinstance(value, (int, float)) else None


def _parse_sse(response, sample: LoadSample, started: float) -> None:
    """解析 SSE 流：记录首 token 时间、done 帧耗时、流内 error 事件。"""
    first_token_at: float | None = None
    for line in response.iter_lines():
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data: "):
            continue
        raw = line[6:].strip()
        if raw == "[DONE]":
            break
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            sample.ok = False
            sample.error = "sse_parse: 非 JSON 数据帧"
            continue
        etype = event.get("type")
        if etype == "token" and first_token_at is None:
            first_token_at = time.perf_counter()
            sample.ttft_ms = round((first_token_at - started) * 1000, 1)
        elif etype == "done":
            sample.server_ms = _server_elapsed(event)
        elif etype == "error":
            sample.ok = False
            sample.error = f"sse_error: {str(event.get('content') or '')[:120]}"


def _request_once(client: httpx.Client, base_url: str, mode: str,
                  question: str, top_k: int) -> LoadSample:
    url = base_url.rstrip("/") + ENDPOINTS[mode]
    payload = build_payload(mode, question, top_k=top_k)
    started = time.perf_counter()
    sample = LoadSample(ok=True, latency_ms=0.0, status=0)
    try:
        if mode == "stream":
            with client.stream("POST", url, json=payload) as response:
                sample.status = response.status_code
                kind = classify_status(response.status_code)
                if kind != "ok":
                    sample.ok = False
                    sample.error = f"{kind}: HTTP {response.status_code}"
                    response.read()
                else:
                    _parse_sse(response, sample, started)
        else:
            response = client.post(url, json=payload)
            sample.status = response.status_code
            kind = classify_status(response.status_code)
            if kind != "ok":
                sample.ok = False
                sample.error = f"{kind}: HTTP {response.status_code}"
                if kind == "rate_limited":
                    try:
                        body = response.json()
                        sample.error += (
                            f" tier={body.get('tier', '?')}"
                            f" retry_after={body.get('retry_after', '?')}"
                        )
                    except Exception:  # noqa: BLE001
                        pass
            else:
                try:
                    body = response.json()
                except Exception:  # noqa: BLE001
                    body = {}
                if isinstance(body, dict):
                    sample.server_ms = _server_elapsed(body)
                    # 非流式错误事件（如 agent 降级）也应算失败
                    if body.get("error"):
                        sample.ok = False
                        sample.error = f"api_error: {str(body['error'])[:120]}"
    except Exception as exc:  # noqa: BLE001 - 单请求异常计入错误率，不中断整轮
        sample.ok = False
        sample.error = f"exception: {type(exc).__name__}: {exc}"
    sample.latency_ms = round((time.perf_counter() - started) * 1000, 1)
    return sample


def run_level(base_url: str, mode: str, questions: list[str], *,
              concurrency: int, requests: int, timeout_s: float, top_k: int,
              transport: httpx.BaseTransport | None = None
              ) -> tuple[list[LoadSample], float]:
    """跑一档并发，返回 (样本, 墙钟秒数)。

    `transport` 只给测试注入 `httpx.MockTransport`，CLI 不暴露。
    """
    requests = max(1, requests)
    workers = max(1, min(concurrency, requests))
    limits = httpx.Limits(max_connections=workers, max_keepalive_connections=workers)
    # trust_env=False：与项目出站客户端一致，避免系统代理把本机请求路由到失效端口
    with httpx.Client(timeout=httpx.Timeout(timeout_s, connect=10.0),
                      limits=limits, trust_env=False, transport=transport) as client:
        def one(index: int) -> LoadSample:
            question = questions[index % len(questions)]
            return _request_once(client, base_url, mode, question, top_k)

        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(one, i) for i in range(requests)]
            samples = [f.result() for f in as_completed(futures)]
        return samples, time.perf_counter() - started


def _health_check(client: httpx.Client, base_url: str) -> str:
    """返回健康状态文本；不可达时抛 SystemExit 并给出启动命令。"""
    try:
        live = client.get(base_url.rstrip("/") + "/api/health/live")
        if live.status_code != 200:
            raise SystemExit(f"服务不可用：/api/health/live 返回 {live.status_code}")
        ready = client.get(base_url.rstrip("/") + "/api/health")
        try:
            body = ready.json()
        except Exception:  # noqa: BLE001
            body = {}
        status = str(body.get("status") or "unknown")
        if status == "warming":
            print("⚠️  服务仍处于 warming：嵌入/精排模型可能还在加载，首批请求会很慢。")
        elif status == "down":
            print("⚠️  服务健康检查为 down（依赖不可用）；压测结果可能全是错误，先修依赖。")
        return status
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"无法连接 {base_url}（{type(exc).__name__}: {exc}）。\n"
            "先启动服务：uv run python main.py api\n"
            "压测前记得把限流调大：RATE_LIMIT_LLM=1000000 RATE_LIMIT_LLM_WINDOW=60"
        ) from exc


def _print_level(concurrency: int, summary: dict) -> None:
    lat = summary["latency_ms"]
    print(
        f"并发 {concurrency:<4} 请求 {summary['requests']:<5} "
        f"成功 {summary['ok']:<5} RPS {summary['rps']:<7} "
        f"P50 {lat['p50']}ms  P95 {lat['p95']}ms  P99 {lat['p99']}ms  "
        f"错误率 {summary['error_rate']}"
    )
    if summary["ttft_ms"]:
        print(f"        首 token P50 {summary['ttft_ms']['p50']}ms  "
              f"P95 {summary['ttft_ms']['p95']}ms")


def main() -> int:
    setup_stdio()
    parser = argparse.ArgumentParser(description="HTTP 并发压测（不新增依赖）")
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--mode", default="ask", choices=tuple(ENDPOINTS))
    parser.add_argument(
        "--concurrency", default="1",
        help="并发档位，逗号分隔（如 1,5,20）；每档各跑 --requests 个请求",
    )
    parser.add_argument("--requests", type=int, default=20, help="每档总请求数")
    parser.add_argument("--warmup", type=int, default=3, help="正式统计前的预热请求数")
    parser.add_argument("--timeout", type=float, default=180.0, help="单请求超时（秒）")
    parser.add_argument("--top-k", type=int, default=5, help="ask 模式的 top_k")
    parser.add_argument("--questions", default="", help="问题 JSON 文件（默认 test_questions.json）")
    parser.add_argument("--question-limit", type=int, default=0, help="只用前 N 个问题")
    parser.add_argument("--json", default="", help="把结果写到指定 JSON 文件")
    parser.add_argument("--skip-health-check", action="store_true",
                        help="跳过 /api/health 预检（测试用）")
    parser.add_argument("--clear-cache", action="store_true",
                        help="每档前清空检索缓存（DELETE /api/cache?scope=retrieval）")
    parser.add_argument("--allow-errors", action="store_true",
                        help="有错误也返回 0；默认任何错误都返回非零")
    args = parser.parse_args()

    try:
        levels = [int(x) for x in args.concurrency.split(",") if x.strip()]
    except ValueError:
        raise SystemExit("--concurrency 必须是正整数，逗号分隔（如 1,5,20）")
    if not levels or any(n <= 0 for n in levels):
        raise SystemExit("--concurrency 必须是正整数，逗号分隔")
    if max(levels) > MAX_CONCURRENCY:
        raise SystemExit(f"并发 {max(levels)} 超过安全上限 {MAX_CONCURRENCY}")
    if args.requests <= 0:
        raise SystemExit("--requests 必须为正整数")

    questions = load_questions(
        Path(args.questions) if args.questions else None,
        limit=args.question_limit,
    )
    mode = args.mode
    print(
        f"目标 {args.base_url}{ENDPOINTS[mode]} | 模式 {mode} | "
        f"问题 {len(questions)} 条 | 档位 {levels} | 每档 {args.requests} 请求"
    )

    result = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "base_url": args.base_url,
            "mode": mode,
            "concurrency": levels,
            "requests": args.requests,
            "warmup": args.warmup,
            "timeout_s": args.timeout,
            "top_k": args.top_k,
        },
        "levels": [],
    }

    with httpx.Client(timeout=httpx.Timeout(args.timeout, connect=10.0),
                      trust_env=False) as client:
        if not args.skip_health_check:
            status = _health_check(client, args.base_url)
            result["health"] = status

        if args.clear_cache:
            try:
                client.delete(args.base_url.rstrip("/") + "/api/cache",
                              params={"scope": "retrieval"})
                print("已清空检索缓存")
            except Exception as exc:  # noqa: BLE001
                print(f"⚠️  清空缓存失败（继续）：{type(exc).__name__}: {exc}")

        if args.warmup > 0:
            print(f"预热 {args.warmup} 次（不计入统计）…")
            for _ in range(args.warmup):
                _request_once(client, args.base_url, mode, questions[0], args.top_k)

    any_errors = False
    for concurrency in levels:
        samples, wall = run_level(
            args.base_url, mode, questions,
            concurrency=concurrency, requests=args.requests,
            timeout_s=args.timeout, top_k=args.top_k,
        )
        summary = summarize(samples, wall)
        result["levels"].append({
            "concurrency": concurrency,
            "summary": summary,
        })
        _print_level(concurrency, summary)
        any_errors = any_errors or summary["errors"] > 0

    if any_errors:
        print()
        print("⚠️  出现错误。若错误类别是 rate_limited：")
        print("    本机所有请求共享 127.0.0.1 一个限流桶，默认 30 次/60s。")
        print("    没有关闭开关（设 0 无效），必须**重启服务前**设大：")
        print("      RATE_LIMIT_LLM=1000000 RATE_LIMIT_LLM_WINDOW=60")
        print("    再重跑本脚本。")

    if args.json:
        Path(args.json).write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"结果已写入 {args.json}")

    if any_errors and not args.allow_errors:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

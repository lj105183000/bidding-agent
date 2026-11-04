"""多 Agent 灰度的进程内指标：按模式/决策/降级维度聚合 done 帧。

★ 为什么单独成模块
----------------
`src/monitoring.py` 已经有请求/阶段/组件指标，但没有任何"多 Agent"维度。
灰度最需要的三个问题——auto 到底激活了多少、被拒的原因是什么、
降级/部分失败有没有变多——都只能从每条 done 帧里读，无法聚合。

本模块只做**进程内**聚合（项目单进程部署），与 `src/monitoring.py`
的定位一致：不引外部依赖，`/api/metrics` 直接读取。

★ 为什么区分 mode 与 decision
---------------------------
`AGENT_MULTI_AGENT=auto` 时，"配置了 auto"、"实际生效 auto"、
"路由后决定不走图"是三件事。只看 `orchestration` 时 off 与 auto 被拒
都是 null；这里把 `mode/decision/fallback` 分开，灰度才能看出
"配置生效了但全被拒"这类问题。

样本按组保留最近 1000 条；计数是进程启动至今累计。重启归零，
在线样本（eval/online_samples.jsonl）负责跨重启持久化。
"""

from __future__ import annotations

import statistics
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from src.monitoring import percentile

SCHEMA_VERSION = 1
MAX_SAMPLES_PER_GROUP = 1000

MODES = ("off", "auto", "on", "unknown")
DECISIONS = (
    "off",
    "activated",
    "declined_short",
    "declined_single_expert",
    "declined_no_tasks",
    "declined_error",
    "unknown",
)
FALLBACKS = ("none", "legacy", "fragment", "unknown")
REASONS = ("active", "off", "invalid_env", "langgraph_unavailable", "unknown")


def _num(value: Any) -> float | None:
    """只接受真实数值；bool 不算数字，None/字符串一律视为缺失。"""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _pick(value: Any, allowed: tuple[str, ...], default: str = "unknown") -> str:
    text = str(value or "").strip()
    return text if text in allowed else default


def _stat(values: list[float]) -> dict[str, float | int] | None:
    """均值/中位/P95/max；无样本返回 None（不补 0）。"""
    if not values:
        return None
    return {
        "mean": round(statistics.mean(values), 2),
        "median": round(percentile(values, 0.5), 2),
        "p95": round(percentile(values, 0.95), 2),
        "max": round(max(values), 2),
        "total": round(sum(values), 2),
        "samples": len(values),
    }


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


@dataclass
class _Bucket:
    requests: int = 0
    clarified: int = 0
    activated: int = 0
    declined: int = 0
    partial: int = 0
    latency_ms: list[float] = field(default_factory=list)
    llm_calls: list[float] = field(default_factory=list)
    tokens: list[float] = field(default_factory=list)
    verify_checked: int = 0
    verify_ok: int = 0
    verify_critical_checked: int = 0
    verify_critical_ok: int = 0
    answered: int = 0
    no_source: int = 0

    def add_sample(self, bucket: list[float], value: float | None) -> None:
        if value is None:
            return
        bucket.append(value)
        if len(bucket) > MAX_SAMPLES_PER_GROUP:
            del bucket[0]


class MultiAgentMetrics:
    """线程安全的 done 帧聚合器；所有方法都不应影响问答。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._groups: dict[tuple[str, str, str, bool], _Bucket] = {}
        self._configured_counts: dict[str, int] = {}
        self._reason_counts: dict[str, int] = {}
        self._configured_reason_counts: dict[tuple[str, str], int] = {}
        self._started_at = time.time()

    def reset(self) -> None:
        with self._lock:
            self._groups.clear()
            self._configured_counts.clear()
            self._reason_counts.clear()
            self._configured_reason_counts.clear()
            self._started_at = time.time()

    def record(self, done: dict, *, clarified: bool = False) -> None:
        """记录一条 done；字段缺失/类型异常一律降级，不抛异常。"""
        try:
            ma = done.get("multi_agent")
            ma = ma if isinstance(ma, dict) else {}
            mode = _pick(ma.get("mode"), MODES)
            decision = _pick(ma.get("decision"), DECISIONS)
            activated = bool(ma.get("activated"))
            fallback = _pick(ma.get("fallback"), FALLBACKS) if activated else "none"
            partial = bool(ma.get("partial")) if activated else False
            configured = _pick(ma.get("configured_mode"), MODES)
            reason = _pick(ma.get("mode_reason"), REASONS)

            with self._lock:
                self._configured_counts[configured] = (
                    self._configured_counts.get(configured, 0) + 1
                )
                self._reason_counts[reason] = self._reason_counts.get(reason, 0) + 1
                pair = (configured, reason)
                self._configured_reason_counts[pair] = (
                    self._configured_reason_counts.get(pair, 0) + 1
                )
                bucket = self._groups.setdefault(
                    (mode, decision, fallback, partial), _Bucket()
                )
                bucket.requests += 1
                bucket.clarified += int(clarified)
                bucket.activated += int(activated)
                if activated:
                    bucket.partial += int(partial)
                elif mode in ("auto", "on"):
                    # off/unknown 不是"被拒"，只有 auto/on 未激活才算 declined。
                    bucket.declined += 1

                bucket.add_sample(bucket.latency_ms, _num(done.get("elapsed_ms")))
                llm = done.get("llm_trace")
                if isinstance(llm, dict):
                    bucket.add_sample(bucket.llm_calls, _num(llm.get("call_count")))
                    bucket.add_sample(bucket.tokens, _num(llm.get("total_tokens")))

                verification = done.get("verification")
                if isinstance(verification, dict) and verification:
                    bucket.verify_checked += 1
                    bucket.verify_ok += int(bool(verification.get("ok")))
                    critical = verification.get("critical_ok")
                    if critical is not None:
                        bucket.verify_critical_checked += 1
                        bucket.verify_critical_ok += int(bool(critical))

                if not clarified:
                    bucket.answered += 1
                    sources = done.get("sources") or []
                    web_sources = done.get("web_sources") or []
                    if not sources and not web_sources:
                        bucket.no_source += 1
        except Exception:  # noqa: BLE001 - 观测不能影响问答
            return

    @staticmethod
    def _bucket_snapshot(
        mode: str, decision: str, fallback: str, partial: bool, b: _Bucket
    ) -> dict:
        return {
            "mode": mode,
            "decision": decision,
            "fallback": fallback,
            "partial": partial,
            "requests": b.requests,
            "clarified": b.clarified,
            "activated": b.activated,
            "declined": b.declined,
            "partial_count": b.partial,
            "latency_ms": _stat(b.latency_ms),
            "llm_calls": _stat(b.llm_calls),
            "tokens": _stat(b.tokens),
            "verification": {
                "checked": b.verify_checked,
                "ok": b.verify_ok,
                "ok_rate": _rate(b.verify_ok, b.verify_checked),
                "critical_checked": b.verify_critical_checked,
                "critical_ok": b.verify_critical_ok,
                "critical_ok_rate": _rate(
                    b.verify_critical_ok, b.verify_critical_checked
                ),
            },
            "sources": {
                "answered": b.answered,
                "no_source": b.no_source,
                "no_source_rate": _rate(b.no_source, b.answered),
            },
        }

    def snapshot(self) -> dict:
        with self._lock:
            groups = [
                self._bucket_snapshot(mode, decision, fallback, partial, b)
                for (mode, decision, fallback, partial), b in sorted(
                    self._groups.items(), key=lambda item: str(item[0])
                )
            ]
            configured = dict(sorted(self._configured_counts.items()))
            reasons = dict(sorted(self._reason_counts.items()))
            configured_reasons = [
                {"mode": mode, "reason": reason, "requests": count}
                for (mode, reason), count in sorted(
                    self._configured_reason_counts.items()
                )
            ]
            started = self._started_at

        summary = {
            "off": 0,
            "auto_activated": 0,
            "auto_declined": 0,
            "on_activated": 0,
            "on_declined": 0,
            "unknown": 0,
            "fallback_none": 0,
            "fallback_legacy": 0,
            "fallback_fragment": 0,
            "partial": 0,
        }
        by_mode: dict[str, dict] = {}
        for g in groups:
            mode = g["mode"]
            activated = bool(g["activated"])
            if mode == "off":
                summary["off"] += g["requests"]
            elif mode == "auto":
                summary["auto_activated" if activated else "auto_declined"] += g[
                    "requests"
                ]
            elif mode == "on":
                summary["on_activated" if activated else "on_declined"] += g["requests"]
            else:
                summary["unknown"] += g["requests"]
            if activated and g["fallback"] in ("none", "legacy", "fragment"):
                summary[f"fallback_{g['fallback']}"] += g["requests"]
            summary["partial"] += g["partial_count"]

            m = by_mode.setdefault(
                mode,
                {
                    "requests": 0,
                    "activated": 0,
                    "declined": 0,
                    "clarified": 0,
                    "decision_counts": {},
                    "fallback_counts": {},
                    "partial_count": 0,
                    "latency_samples": [],
                    "llm_samples": [],
                    "token_samples": [],
                    "verify_checked": 0,
                    "verify_ok": 0,
                    "verify_critical_checked": 0,
                    "verify_critical_ok": 0,
                    "answered": 0,
                    "no_source": 0,
                },
            )
            m["requests"] += g["requests"]
            m["activated"] += g["activated"]
            m["declined"] += g["declined"]
            m["clarified"] += g["clarified"]
            m["partial_count"] += g["partial_count"]
            m["decision_counts"][g["decision"]] = (
                m["decision_counts"].get(g["decision"], 0) + g["requests"]
            )
            if activated:
                m["fallback_counts"][g["fallback"]] = (
                    m["fallback_counts"].get(g["fallback"], 0) + g["requests"]
                )
            for key, target in (
                ("latency_ms", "latency_samples"),
                ("llm_calls", "llm_samples"),
                ("tokens", "token_samples"),
            ):
                stats = g.get(key)
                if stats:
                    # 只保留统计量，不把原始样本跨组复制；均值/分位在下方按组重算
                    m[target].append(stats)
            ver = g["verification"]
            m["verify_checked"] += ver["checked"]
            m["verify_ok"] += ver["ok"]
            m["verify_critical_checked"] += ver["critical_checked"]
            m["verify_critical_ok"] += ver["critical_ok"]
            m["answered"] += g["sources"]["answered"]
            m["no_source"] += g["sources"]["no_source"]

        for mode, m in by_mode.items():
            # 用各组统计量近似合并：按 samples 加权均值；分位取组间最大（保守）。
            def merge(stats_list: list[dict]) -> dict | None:
                if not stats_list:
                    return None
                weights = [max(1, s["samples"]) for s in stats_list]
                total = sum(weights)
                mean = sum(s["mean"] * w for s, w in zip(stats_list, weights)) / total
                return {
                    "mean": round(mean, 2),
                    "median": round(
                        sum(s["median"] * w for s, w in zip(stats_list, weights))
                        / total,
                        2,
                    ),
                    "p95": round(max(s["p95"] for s in stats_list), 2),
                    "max": round(max(s["max"] for s in stats_list), 2),
                    "total": round(sum(s.get("total", 0) for s in stats_list), 2),
                    "samples": total,
                }

            m["latency_ms"] = merge(m.pop("latency_samples"))
            m["llm_calls"] = merge(m.pop("llm_samples"))
            m["tokens"] = merge(m.pop("token_samples"))
            m["n"] = m["requests"]  # 与在线聚合的 by_mode 字段名对齐
            denom = m["activated"] + m["declined"]
            m["activated_rate"] = _rate(m["activated"], denom) if denom else 0.0
            m["partial_rate"] = _rate(m["partial_count"], m["activated"])
            m["verification"] = {
                "checked": m.pop("verify_checked"),
                "ok": m.pop("verify_ok"),
                "ok_rate": None,
                "critical_checked": m.pop("verify_critical_checked"),
                "critical_ok": m.pop("verify_critical_ok"),
                "critical_ok_rate": None,
            }
            m["verification"]["ok_rate"] = _rate(
                m["verification"]["ok"], m["verification"]["checked"]
            )
            m["verification"]["critical_ok_rate"] = _rate(
                m["verification"]["critical_ok"],
                m["verification"]["critical_checked"],
            )
            m["sources"] = {
                "answered": m.pop("answered"),
                "no_source": m.pop("no_source"),
                "no_source_rate": None,
            }
            m["sources"]["no_source_rate"] = _rate(
                m["sources"]["no_source"], m["sources"]["answered"]
            )

        # 稳定输出：始终带 off/auto/on，unknown 只在有样本时出现。
        ordered = {mode: by_mode[mode] for mode in ("off", "auto", "on") if mode in by_mode}
        if "unknown" in by_mode:
            ordered["unknown"] = by_mode["unknown"]

        total = sum(g["requests"] for g in groups)
        return {
            "schema_version": SCHEMA_VERSION,
            "since": time.strftime(
                "%Y-%m-%dT%H:%M:%S%z", time.localtime(started)
            ),
            "uptime_s": int(time.time() - started),
            "total_requests": total,
            "summary": summary,
            "configured_mode_counts": configured,
            "mode_reason_counts": reasons,
            "configured_reason_counts": configured_reasons,
            "by_mode": ordered,
            "groups": groups,
        }


_REGISTRY = MultiAgentMetrics()


def record_done(done: dict, *, clarified: bool = False) -> None:
    _REGISTRY.record(done, clarified=clarified)


def multi_agent_snapshot() -> dict:
    try:
        return _REGISTRY.snapshot()
    except Exception as exc:  # noqa: BLE001 - 观测不能影响 /api/metrics
        return {"schema_version": SCHEMA_VERSION, "error": f"{type(exc).__name__}: {exc}"}


def reset_multi_agent_metrics() -> None:
    _REGISTRY.reset()


__all__ = [
    "DECISIONS",
    "FALLBACKS",
    "MAX_SAMPLES_PER_GROUP",
    "MODES",
    "REASONS",
    "MultiAgentMetrics",
    "multi_agent_snapshot",
    "record_done",
    "reset_multi_agent_metrics",
]

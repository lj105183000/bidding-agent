"""手动触发评测任务（轻量评测中台）。

★ 为什么用 subprocess + 后台线程，而不是在 API 进程里直接跑
--------------------------------------------------
`retrieval_offline` 也要加载 BGE 模型、重建内存索引；在事件循环/线程池里
直接跑会占住请求处理，且评测代码崩溃会波及 API 进程。子进程隔离后，
API 只负责记录状态与回收超龄任务。

★ 为什么只允许 `retrieval_offline`
-------------------------------
它确定性、不调 LLM/Qdrant/密钥，分钟级可完成；agent/retrieval/rag 动辄
数分钟到数小时，且会消耗真实额度。昂贵线仍走 GitHub Actions 的
`eval-gate.yml`，本接口对其它线返回 400 并直接给出该指引。

★ 已知限制
--------
`python main.py dev` 的 reload 会杀掉后台线程。子进程可能继续跑完并写入
`eval_runs`，但 `eval_jobs` 状态会停在 running/lost。手动触发只保证在
生产单进程 `uvicorn`（无 reload）下可靠；读取时会按时间把超龄任务标 lost，
不伪造成功。
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT
from src.eval_runs import store
from src.eval_runs.db import EvalStoreError

logger = logging.getLogger(__name__)

#: 可从 API 触发的评测线。要扩展时先确认运行时长与外部依赖可控。
TRIGGERABLE_SUITES = ("retrieval_offline",)

DEFAULT_MIN_INTERVAL_SECONDS = 60
DEFAULT_TIMEOUT_SECONDS = 1200
#: 超龄回收的宽限：任务超时后给子进程/收尾留两分钟。
STALE_GRACE_SECONDS = 120

_lock = threading.Lock()
_active_job_id: str | None = None
_last_requested_at = 0.0


class EvalTriggerError(RuntimeError):
    """触发失败；`status_code` 由 API 层映射成 HTTP 状态码。"""

    status_code = 400


class EvalTriggerDisabled(EvalTriggerError):
    status_code = 403


class EvalTriggerConflict(EvalTriggerError):
    status_code = 409


class EvalTriggerUnavailable(EvalTriggerError):
    status_code = 503


def _env_int(key: str, default: int, *, minimum: int = 0) -> int:
    raw = (os.getenv(key) or "").strip()
    if not raw:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        return default


def trigger_enabled() -> bool:
    raw = (os.getenv("EVAL_TRIGGER_ENABLED") or "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def min_interval_seconds() -> int:
    return _env_int(
        "EVAL_TRIGGER_MIN_INTERVAL_SECONDS", DEFAULT_MIN_INTERVAL_SECONDS, minimum=0
    )


def timeout_seconds() -> int:
    return _env_int(
        "EVAL_TRIGGER_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS, minimum=60
    )


def stale_seconds() -> int:
    return timeout_seconds() + STALE_GRACE_SECONDS


def trigger_status() -> dict[str, Any]:
    enabled = trigger_enabled()
    return {
        "enabled": enabled,
        "suites": list(TRIGGERABLE_SUITES),
        "reason": "" if enabled else "EVAL_TRIGGER_ENABLED=0（默认关闭）",
    }


def _new_job_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def _resolve_log_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _build_command(job: dict[str, Any]) -> tuple[list[str], Path]:
    json_path = _resolve_log_path(job["log_path"]).with_suffix(".json")
    command = [
        sys.executable,
        "-u",
        "eval/run_all.py",
        "--suites",
        ",".join(job["suites"]),
        "--save-db",
        "--json",
        str(json_path),
    ]
    if int(job.get("limit") or 0) > 0:
        command.extend(["--limit", str(int(job["limit"]))])
    return command, json_path


def trigger_job(
    suites: list[str], *, limit: int = 0, note: str = ""
) -> dict[str, Any]:
    """创建并启动一个评测任务；返回任务字典（状态 queued）。

    失败时抛 `EvalTriggerError` 子类，由 API 映射成 403/400/409/503。
    """
    if not trigger_enabled():
        raise EvalTriggerDisabled(
            "手动触发未开启（EVAL_TRIGGER_ENABLED=0）；"
            "昂贵评测请使用 GitHub Actions 的 eval-gate.yml"
        )

    selected = [str(item).strip() for item in suites if str(item).strip()]
    if not selected:
        raise EvalTriggerError("至少选择一条评测线")
    unknown = [item for item in selected if item not in TRIGGERABLE_SUITES]
    if unknown:
        raise EvalTriggerError(
            f"API 只允许触发 {list(TRIGGERABLE_SUITES)}，收到 {unknown}；"
            "其它评测线请使用 GitHub Actions 的 eval-gate.yml"
        )
    if limit < 0 or limit > 200:
        raise EvalTriggerError("limit 必须在 0~200 之间（0 表示全量）")
    note = (note or "").strip()[:200]

    try:
        store.reconcile_stale_jobs(stale_seconds())
        active = store.active_job()
    except EvalStoreError as exc:
        raise EvalTriggerUnavailable(
            "评测任务存储不可用，未启动评测；请检查 PostgreSQL"
        ) from exc
    if active is not None:
        raise EvalTriggerConflict(f"已有评测任务进行中：{active['job_id']}")

    global _active_job_id, _last_requested_at
    with _lock:
        now = time.monotonic()
        if _active_job_id:
            raise EvalTriggerConflict(f"已有评测任务进行中：{_active_job_id}")
        if (
            _last_requested_at
            and now - _last_requested_at < min_interval_seconds()
        ):
            raise EvalTriggerConflict(
                f"触发过于频繁，请等待 {min_interval_seconds()} 秒后再试"
            )

        job_id = _new_job_id()
        log_path = f"logs/eval_jobs/{job_id}.log"
        job = {
            "job_id": job_id,
            "status": "queued",
            "suites": selected,
            "limit": limit,
            "note": note,
            "requested_at": datetime.now(timezone.utc),
            "started_at": None,
            "finished_at": None,
            "run_id": job_id,
            "exit_code": None,
            "error": None,
            "log_path": log_path,
        }
        try:
            store.create_job(job)
        except EvalStoreError as exc:
            raise EvalTriggerUnavailable(
                "评测任务写入失败，未启动评测；请检查 PostgreSQL"
            ) from exc

        _active_job_id = job_id
        _last_requested_at = now
        _resolve_log_path(log_path).parent.mkdir(parents=True, exist_ok=True)
        thread = threading.Thread(
            target=_run_job,
            args=(job,),
            daemon=True,
            name=f"eval-job-{job_id}",
        )
        try:
            thread.start()
        except Exception as exc:  # noqa: BLE001 - 线程都起不来只能如实失败
            _active_job_id = None
            try:
                store.update_job(
                    job_id,
                    status="failed",
                    finished_at=datetime.now(timezone.utc),
                    error="无法启动评测线程，请查看日志",
                )
            except Exception:  # noqa: BLE001
                logger.exception("评测任务启动失败后回写状态失败：%s", job_id)
            raise EvalTriggerUnavailable("无法启动评测任务") from exc

    return job


def _run_job(job: dict[str, Any]) -> None:
    """后台线程：启动子进程、等待、落任务状态。

    任何异常都只把任务标 failed，并写固定短文案；完整堆栈进日志。
    """
    job_id = job["job_id"]
    exit_code: int | None = None
    error: str | None = None
    try:
        store.update_job(
            job_id, status="running", started_at=datetime.now(timezone.utc)
        )
        command, _ = _build_command(job)
        log_path = _resolve_log_path(job["log_path"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update(
            {
                "EVAL_RUN_ID": job_id,
                "EVAL_RUN_SOURCE": "api",
                "EVAL_JOB_ID": job_id,
            }
        )
        with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
            process = subprocess.Popen(  # noqa: S603 - 参数为列表，无 shell
                command,
                cwd=str(PROJECT_ROOT),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
                shell=False,
            )
            try:
                exit_code = process.wait(timeout=timeout_seconds())
            except subprocess.TimeoutExpired:
                process.kill()
                # kill 后的 wait 只用于回收僵尸进程；即便它再次抛异常，
                # 也不该把“超时”这个更重要的结论覆盖成“执行失败”。
                try:
                    process.wait()
                except Exception:  # noqa: BLE001 - 回收失败无补救价值
                    pass
                exit_code = None
                error = "评测超时，已终止"

        if error is None:
            if exit_code not in (0, 1, 2):
                error = f"评测进程异常退出（exit_code={exit_code}）"
            elif not store.has_run(job_id):
                error = "评测进程结束但结果未写入数据库，请查看日志"
    except Exception:  # noqa: BLE001 - 后台任务不能把异常抛给线程
        logger.exception("评测任务执行失败：%s", job_id)
        error = "评测任务执行失败，请查看日志"

    status = "succeeded" if error is None else "failed"
    try:
        store.update_job(
            job_id,
            status=status,
            finished_at=datetime.now(timezone.utc),
            exit_code=exit_code,
            error=error,
        )
    except Exception:  # noqa: BLE001
        logger.exception("评测任务状态回写失败：%s", job_id)

    global _active_job_id
    with _lock:
        if _active_job_id == job_id:
            _active_job_id = None


def reset_state() -> None:
    """清除进程内单任务锁（测试用）。"""
    global _active_job_id, _last_requested_at
    with _lock:
        _active_job_id = None
        _last_requested_at = 0.0


__all__ = [
    "DEFAULT_MIN_INTERVAL_SECONDS",
    "DEFAULT_TIMEOUT_SECONDS",
    "EvalTriggerConflict",
    "EvalTriggerDisabled",
    "EvalTriggerError",
    "EvalTriggerUnavailable",
    "TRIGGERABLE_SUITES",
    "min_interval_seconds",
    "reset_state",
    "stale_seconds",
    "timeout_seconds",
    "trigger_enabled",
    "trigger_job",
    "trigger_status",
]

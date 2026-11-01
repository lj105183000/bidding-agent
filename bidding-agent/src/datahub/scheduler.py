"""数据中台轻量调度器（**独立进程**）。

为什么独立进程
--------------
API 进程有“必须单进程”的约束（限流器/缓存/MCP 子进程都是进程内单例）。
把调度器塞进 API lifespan 会让“起两个 worker”从“限流器失效”升级成
“同一任务跑两份”。独立进程不导入 `api.server`、不监听 8001，互不影响。

为什么只调度这三个任务
--------------------
调度器的原则是“**只做幂等、可重复、失败可重跑**的事”：

    - 法规增量入库：UUID5 稳定 id，重复跑只覆盖同一条；
    - 数据质量扫描：只读业务数据 + 追加报告；
    - 目录快照：只读探测 + 追加快照。

以下任务**永远不放进调度器**，必须人工执行：
    - `main.py ingest`：删集合重建 `bid_qa`；
    - `main.py pg --force`：DROP 重建 `bidding_records`；
    - `main.py graph --clear`：清空图谱；
    - `batch/` 五步管道：真实 LLM 计费 + 人工归置文件。

运行方式：
    python main.py scheduler                 # 常驻（需 DATA_SCHEDULER_ENABLED=1）
    python main.py scheduler --list          # 只列任务，不连任何后端
    python main.py scheduler --run-once quality
    python main.py scheduler --run-once regulations
    python main.py scheduler --run-once catalog
    python main.py scheduler --run-once all
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass
from typing import Callable
from zoneinfo import ZoneInfo

from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from src.config import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class JobDef:
    job_id: str
    cron_setting: str
    description: str
    run: Callable[[], dict]


def job_quality_scan() -> dict:
    from src.datahub.quality import run_quality_scan

    report = run_quality_scan(source="scheduler", persist=True)
    return {
        "overall": report.get("overall"),
        "summary": report.get("summary"),
        "report_id": report.get("report_id"),
    }


def job_catalog_snapshot() -> dict:
    from src.datahub.catalog_store import save_snapshot

    result = save_snapshot(source="scheduler", persist=True)
    return {
        "asset_count": result.get("asset_count"),
        "drift_summary": result.get("drift_summary"),
        "snapshot_id": result.get("snapshot_id"),
    }


def job_regulations_incremental() -> dict:
    # 延迟导入：会加载嵌入模型，`--list` 与目录端点不该受它影响。
    from src.rag.regulation_store import upsert_regulations, verify_regulations

    # ★ 绝不传 recreate=True：那是“删集合重建”，必须人工执行。
    written = upsert_regulations(recreate=False)
    stats = verify_regulations()
    return {
        "written": written,
        "points": stats.get("points"),
        "in_sync": stats.get("in_sync"),
    }


JOB_DEFS: tuple[JobDef, ...] = (
    JobDef(
        "datahub.regulations_incremental",
        "data_scheduler_regulations_cron",
        "法规增量入库（UUID5 幂等）",
        job_regulations_incremental,
    ),
    JobDef(
        "datahub.quality_scan",
        "data_scheduler_quality_cron",
        "数据质量扫描",
        job_quality_scan,
    ),
    JobDef(
        "datahub.catalog_snapshot",
        "data_scheduler_catalog_cron",
        "资产目录快照与漂移检测",
        job_catalog_snapshot,
    ),
)

#: `--run-once` 的短名 → job_id。只暴露白名单里的任务。
_JOB_ALIASES = {
    "quality": "datahub.quality_scan",
    "regulations": "datahub.regulations_incremental",
    "catalog": "datahub.catalog_snapshot",
}


def _timezone() -> ZoneInfo:
    name = (settings.data_scheduler_timezone or "Asia/Shanghai").strip()
    try:
        return ZoneInfo(name)
    except Exception:  # noqa: BLE001 - 非法时区回退，不让整个调度器起不来
        logger.warning("非法时区 %r，回退 Asia/Shanghai", name)
        return ZoneInfo("Asia/Shanghai")


def _job_by_id(job_id: str) -> JobDef | None:
    return next((job for job in JOB_DEFS if job.job_id == job_id), None)


def _run_job(job_id: str) -> dict:
    """执行一个任务；异常只记日志并返回 `ok=False`，不向上抛。"""
    job = _job_by_id(job_id)
    if job is None:
        logger.error("未知调度任务：%s", job_id)
        return {"ok": False, "job_id": job_id, "error": "unknown job"}

    started = time.perf_counter()
    try:
        result = dict(job.run() or {})
        result.update(
            {
                "ok": True,
                "job_id": job_id,
                "elapsed_ms": max(0, int((time.perf_counter() - started) * 1000)),
            }
        )
        logger.info("调度任务完成：%s（%dms）", job_id, result["elapsed_ms"])
        return result
    except Exception as exc:  # noqa: BLE001 - 单任务失败不拖垮调度器
        logger.exception("调度任务失败：%s", job_id)
        return {
            "ok": False,
            "job_id": job_id,
            "error": f"调度任务失败：{type(exc).__name__}",
            "elapsed_ms": max(0, int((time.perf_counter() - started) * 1000)),
        }


def build_scheduler(*, start: bool = True) -> BlockingScheduler:
    """注册白名单任务并（默认）启动。

    `start=False` 供测试与 `--run-once` 路径使用：只构造、不阻塞。
    """
    timezone = _timezone()
    scheduler = BlockingScheduler(
        timezone=timezone,
        executors={"default": ThreadPoolExecutor(max_workers=2)},
    )

    for job in JOB_DEFS:
        cron = getattr(settings, job.cron_setting, "")
        try:
            trigger = CronTrigger.from_crontab(cron, timezone=timezone)
        except Exception:
            # 单个 cron 写错只跳过该任务；其余任务照常注册。
            logger.warning(
                "任务 %s 的 cron 非法（%r），已跳过", job.job_id, cron
            )
            continue
        scheduler.add_job(
            _run_job,
            trigger=trigger,
            args=[job.job_id],
            id=job.job_id,
            name=job.description,
            max_instances=max(1, int(settings.data_scheduler_max_instances)),
            coalesce=bool(settings.data_scheduler_coalesce),
            misfire_grace_time=max(
                1, int(settings.data_scheduler_misfire_grace_seconds)
            ),
            replace_existing=True,
        )

    if start:
        scheduler.start()
    return scheduler


def _select_jobs(name: str) -> list[JobDef]:
    if name == "all":
        return list(JOB_DEFS)
    job_id = _JOB_ALIASES.get(name)
    if not job_id:
        return []
    job = _job_by_id(job_id)
    return [job] if job is not None else []


def run_scheduler(argv: list[str] | None = None) -> int:
    """CLI 入口；返回进程退出码。"""
    args = list(argv or [])

    if "--list" in args:
        for job in JOB_DEFS:
            cron = getattr(settings, job.cron_setting, "")
            print(f"{job.job_id}\t{cron}\t{job.description}")
        return 0

    if "--run-once" in args:
        index = args.index("--run-once")
        name = args[index + 1] if index + 1 < len(args) else ""
        jobs = _select_jobs(name)
        if not jobs:
            print(f"未知任务：{name}（可选：quality / regulations / catalog / all）")
            return 1
        results = [_run_job(job.job_id) for job in jobs]
        for result in results:
            print(json.dumps(result, ensure_ascii=False, default=str))
        return 0 if all(result.get("ok") for result in results) else 1

    if not settings.data_scheduler_enabled:
        print(
            "调度器未启用：设置 DATA_SCHEDULER_ENABLED=1 后运行；"
            "或用 --run-once 手工执行一次。"
        )
        return 1

    logger.info(
        "数据中台调度器启动（时区 %s，任务数 %d）",
        settings.data_scheduler_timezone,
        len(JOB_DEFS),
    )
    try:
        build_scheduler(start=True)
    except KeyboardInterrupt:
        logger.info("调度器已停止")
    return 0


if __name__ == "__main__":  # pragma: no cover - 手工运行入口
    from src.logging_config import setup_logging

    setup_logging()
    raise SystemExit(run_scheduler(sys.argv[1:]))

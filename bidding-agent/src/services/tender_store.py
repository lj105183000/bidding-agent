"""招标文件抽取结果的持久化与截止日期监控。

为什么需要持久化
--------------
抽取结果此前只存在内存里（前端刷新即丢）。持久化带来两件此前做不到的事：

    1. **截止日期监控**（P3）——`bid_deadline` 是从招标文件里抽出来的，
       但它必须在库里才能做"哪些项目快截止了"这类查询。
       这是 P3 从"阻塞"变成"可行"的关键：`bidding_records` 是历史中标记录
       （22 列里确实没有投标截止时间），而**新上传的招标文件有**。
    2. **重复上传检测**——同一项目编号再传一次时能提示"已解析过"，
       避免用户重复花 1~3 分钟做同样的抽取。

为什么单独一个模块而不是塞进 postgresql_client
------------------------------------------
`postgresql_client` 负责**会话/消息/反馈**（应用自身的对话存储），
本模块负责**招标文件业务数据**。两者生命周期与查询模式都不同：
前者按 session_id 查、后者按项目编号与截止时间查。

★ 重要：`bid_deadline` 必须是**独立的时间列**而不是埋在 JSONB 里
------------------------------------------------------------
"哪些项目 7 天内截止"需要 `WHERE bid_deadline BETWEEN now() AND now() + 7 days`。
埋进 JSONB 后只能全表扫描 + 逐行解析 JSON，且无法建索引。
故标量字段中时间与金额类都单独成列，其余明细留在 `parsed` JSONB 里。
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

import psycopg
from dotenv import load_dotenv

from src.config import PROJECT_ROOT
from src.logging_config import get_logger

logger = get_logger(__name__)

TABLE = "tender_documents"

# 截止日期紧急度分档（天）。取值依据是投标实务：
#   ≤3 天  必须立刻决定是否投（编标书来不及精雕）
#   ≤7 天  进入编标阶段
#   ≤30 天 正常准备期
URGENCY_BANDS: tuple[tuple[str, int], ...] = (
    ("urgent", 3),
    ("soon", 7),
    ("normal", 30),
)


class TenderStoreError(RuntimeError):
    """持久化失败（数据库不可用 / 数据不合法）。"""


def _dsn() -> dict[str, Any]:
    load_dotenv(PROJECT_ROOT / ".env")
    host = (os.getenv("POSTGRES_HOST") or "").strip()
    if not host:
        raise TenderStoreError("未配置 PostgreSQL（POSTGRES_HOST 为空）")
    return {
        "host": host,
        "port": (os.getenv("POSTGRES_PORT") or "5432").strip(),
        "user": (os.getenv("POSTGRES_USER") or "postgres").strip(),
        # Windows 下编辑 .env 常带入 \r，会让认证失败
        "password": (os.getenv("POSTGRES_PASSWORD") or "").rstrip("\r\n"),
        "dbname": (os.getenv("POSTGRES_DB") or "chatbot").strip(),
    }


class _Conn:
    """连库上下文管理器：用后必关。"""

    def __enter__(self):
        try:
            self._conn = psycopg.connect(**_dsn())
        except Exception as exc:
            raise TenderStoreError(f"数据库连接失败：{type(exc).__name__}") from exc
        self._cur = self._conn.cursor()
        return self._conn, self._cur

    def __exit__(self, *exc_info):
        try:
            self._conn.close()
        except Exception:  # pragma: no cover
            pass
        return False


def _parse_dt(value: Any) -> datetime | None:
    """把抽取出的日期字符串转成 datetime。

    抽取结果是 `YYYY-MM-DD HH:MM:SS` 字符串；解析失败返回 None
    （而不是抛错）——一份时间格式异常的文件不该让整条落库失败，
    其余字段仍然有价值。
    """
    if not value or not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone(timedelta(hours=8)))
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def ensure_schema() -> bool:
    """建表（幂等）。成功返回 True，失败抛 `TenderStoreError`。

    由本模块自己建表，不让使用者手工执行 DDL——否则拿到代码跑不起来。

    `project_id` 上建**非唯一**索引而非唯一约束：同一个项目可能因为
    澄清公告而重新上传（内容不同、需要各存一份做变更追踪）。
    """
    ddl = (
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE} (
            id                BIGSERIAL PRIMARY KEY,
            filename          TEXT NOT NULL,
            kind              TEXT,
            project_name      TEXT,
            project_id        TEXT,
            purchaser         TEXT,
            agency            TEXT,
            budget            NUMERIC,
            max_price         NUMERIC,
            deposit           NUMERIC,
            bid_deadline      TIMESTAMPTZ,
            open_time         TIMESTAMPTZ,
            evaluation_method TEXT,
            price_weight      TEXT,
            pages             INTEGER,
            total_chars       INTEGER,
            completeness      REAL,
            provider          TEXT,
            model             TEXT,
            parsed            JSONB NOT NULL,
            warnings          JSONB,
            created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """,
        # 按截止时间查"快截止的项目"是核心查询，必须有索引
        f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_deadline ON {TABLE} (bid_deadline)",
        # 按项目编号查历史版本（变更追踪要取同一项目的多份文件）
        f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_project_id ON {TABLE} (project_id)",
    )
    try:
        with _Conn() as (conn, cur):
            for stmt in ddl:
                cur.execute(stmt)
            conn.commit()
        return True
    except TenderStoreError:
        raise
    except Exception as exc:
        logger.warning("建表失败：%s", exc)
        raise TenderStoreError(f"建表失败：{type(exc).__name__}") from exc


def save_document(result: dict[str, Any]) -> dict[str, Any]:
    """保存一份抽取结果。返回 `{saved, id, duplicate_of?, reason?}`。

    `result` 即 `/api/tender/extract` 的返回体（含 `parsed` 与 `meta`）。

    **不做唯一性拦截**：同一项目编号可以有多份（原文件 + 澄清公告），
    变更追踪正需要它们。但会回报 `duplicate_of`（同编号最近一份的 id）
    供前端提示"该项目已解析过，是否要做变更对比"。
    """
    if not isinstance(result, dict) or "parsed" not in result:
        raise TenderStoreError("无效的抽取结果：缺少 parsed 字段")

    parsed = result.get("parsed") or {}
    fields = parsed.get("fields") or {}
    meta = result.get("meta") or {}

    ensure_schema()

    project_id = (fields.get("project_id") or "") or None

    try:
        with _Conn() as (conn, cur):
            duplicate_of = None
            if project_id:
                cur.execute(
                    f"SELECT id FROM {TABLE} WHERE project_id = %s "
                    f"ORDER BY created_at DESC LIMIT 1",
                    (project_id,),
                )
                row = cur.fetchone()
                if row:
                    duplicate_of = row[0]

            cur.execute(
                f"""
                INSERT INTO {TABLE} (
                    filename, kind, project_name, project_id, purchaser, agency,
                    budget, max_price, deposit, bid_deadline, open_time,
                    evaluation_method, price_weight, pages, total_chars,
                    completeness, provider, model, parsed, warnings
                ) VALUES (
                    %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s, %s
                ) RETURNING id
                """,
                (
                    meta.get("filename") or "unnamed",
                    meta.get("kind"),
                    fields.get("project_name") or None,
                    project_id,
                    fields.get("purchaser") or None,
                    fields.get("agency") or None,
                    fields.get("budget"),
                    fields.get("max_price"),
                    fields.get("deposit"),
                    _parse_dt(fields.get("bid_deadline")),
                    _parse_dt(fields.get("open_time")),
                    fields.get("evaluation_method") or None,
                    fields.get("price_weight") or None,
                    meta.get("pages"),
                    meta.get("total_chars"),
                    meta.get("completeness"),
                    meta.get("provider"),
                    meta.get("model"),
                    json.dumps(parsed, ensure_ascii=False),
                    json.dumps(meta.get("warnings") or [], ensure_ascii=False),
                ),
            )
            new_id = cur.fetchone()[0]
            conn.commit()
            return {"saved": True, "id": new_id, "duplicate_of": duplicate_of}
    except TenderStoreError:
        raise
    except Exception as exc:
        logger.warning("保存招标文件失败：%s", exc)
        return {"saved": False, "id": None, "reason": f"写入失败：{type(exc).__name__}"}


def list_documents(limit: int = 50, with_deadline_only: bool = False) -> list[dict[str, Any]]:
    """列出已保存的招标文件（按上传时间倒序）。"""
    n = max(1, min(200, int(limit or 50)))
    where = "WHERE bid_deadline IS NOT NULL" if with_deadline_only else ""
    try:
        with _Conn() as (_conn, cur):
            cur.execute(
                f"""
                SELECT id, filename, kind, project_name, project_id, purchaser,
                       budget, max_price, deposit, bid_deadline, open_time,
                       evaluation_method, pages, completeness, created_at
                FROM {TABLE} {where}
                ORDER BY created_at DESC LIMIT %s
                """,
                (n,),
            )
            rows = cur.fetchall()
    except TenderStoreError:
        raise
    except Exception as exc:
        # 表不存在（首次使用、还没保存过任何文件）时返回空列表而不是报错——
        # 前端"暂无记录"比"服务异常"更准确
        logger.info("查询招标文件列表失败（可能尚未建表）：%s", exc)
        return []

    now = datetime.now(timezone.utc)
    out = []
    for r in rows:
        deadline = r[9]
        days_left = None
        if deadline:
            days_left = (deadline - now).total_seconds() / 86400
        out.append(
            {
                "id": r[0],
                "filename": r[1],
                "kind": r[2],
                "project_name": r[3],
                "project_id": r[4],
                "purchaser": r[5],
                "budget": float(r[6]) if r[6] is not None else None,
                "max_price": float(r[7]) if r[7] is not None else None,
                "deposit": float(r[8]) if r[8] is not None else None,
                "bid_deadline": deadline.isoformat() if deadline else None,
                "open_time": r[10].isoformat() if r[10] else None,
                "evaluation_method": r[11],
                "pages": r[12],
                "completeness": r[13],
                "created_at": r[14].isoformat() if r[14] else None,
                "days_left": round(days_left, 1) if days_left is not None else None,
                "urgency": classify_urgency(days_left),
            }
        )
    return out


def classify_urgency(days_left: float | None) -> str:
    """按剩余天数分档：expired / urgent / soon / normal / unknown。

    已过期的单独一档（`expired`），不并入 `urgent`——
    "还有 1 天" 和 "已经截止" 需要完全不同的行动，混在一起会误事。
    """
    if days_left is None:
        return "unknown"
    if days_left < 0:
        return "expired"
    for label, threshold in URGENCY_BANDS:
        if days_left <= threshold:
            return label
    return "normal"


def get_document(doc_id: int) -> dict[str, Any] | None:
    """取单份文件的完整抽取结果（含 parsed 明细）。"""
    try:
        with _Conn() as (_conn, cur):
            cur.execute(
                f"""
                SELECT id, filename, kind, project_name, project_id, bid_deadline,
                       completeness, parsed, warnings, created_at
                FROM {TABLE} WHERE id = %s
                """,
                (int(doc_id),),
            )
            r = cur.fetchone()
    except TenderStoreError:
        raise
    except Exception as exc:
        logger.warning("查询招标文件失败：%s", exc)
        return None

    if not r:
        return None
    return {
        "id": r[0],
        "filename": r[1],
        "kind": r[2],
        "project_name": r[3],
        "project_id": r[4],
        "bid_deadline": r[5].isoformat() if r[5] else None,
        "completeness": r[6],
        "parsed": r[7],
        "warnings": r[8],
        "created_at": r[9].isoformat() if r[9] else None,
    }


def delete_document(doc_id: int) -> bool:
    """删除一份文件记录。"""
    try:
        with _Conn() as (conn, cur):
            cur.execute(f"DELETE FROM {TABLE} WHERE id = %s", (int(doc_id),))
            deleted = cur.rowcount > 0
            conn.commit()
            return deleted
    except TenderStoreError:
        raise
    except Exception as exc:
        logger.warning("删除招标文件失败：%s", exc)
        return False


# ---------------------------------------------------------------------------
# 截止日期监控（P3）
# ---------------------------------------------------------------------------

Window = Literal["7d", "30d", "all"]


def upcoming_deadlines(days: int = 30, include_expired: bool = False) -> dict[str, Any]:
    """未来 N 天内截止的项目（按截止时间升序）。

    ★ 这个功能以前做不了：`bidding_records` 是历史中标记录，没有投标截止时间字段。
    现在它建立在**用户上传的招标文件**之上——`bid_deadline` 是抽取出来的。

    覆盖范围必须在响应里说明（`coverage`）：只有被上传解析过的文件才有截止时间，
    所以这里的"监控"**不等于"全网项目监控"**，而是"我关心的这些项目"。
    不说明这点，用户会以为系统漏报了项目。
    """
    n = max(1, min(365, int(days or 30)))
    try:
        with _Conn() as (_conn, cur):
            cur.execute(f"SELECT count(*) FROM {TABLE}")
            total = cur.fetchone()[0]
            cur.execute(f"SELECT count(*) FROM {TABLE} WHERE bid_deadline IS NOT NULL")
            with_deadline = cur.fetchone()[0]

            cur.execute(
                f"""
                SELECT id, filename, project_name, project_id, purchaser,
                       budget, max_price, bid_deadline, open_time, completeness
                FROM {TABLE}
                WHERE bid_deadline IS NOT NULL
                  AND bid_deadline >= now()
                  AND bid_deadline <= now() + (%s || ' days')::interval
                ORDER BY bid_deadline ASC
                """,
                (n,),
            )
            rows = cur.fetchall()

            expired_rows: list = []
            if include_expired:
                cur.execute(
                    f"""
                    SELECT id, filename, project_name, project_id, purchaser,
                           budget, max_price, bid_deadline, open_time, completeness
                    FROM {TABLE}
                    WHERE bid_deadline IS NOT NULL AND bid_deadline < now()
                    ORDER BY bid_deadline DESC LIMIT 50
                    """
                )
                expired_rows = cur.fetchall()
    except TenderStoreError:
        raise
    except Exception as exc:
        logger.info("截止日期查询失败（可能尚未建表）：%s", exc)
        return {
            "items": [],
            "expired": [],
            "summary": {"total_documents": 0, "with_deadline": 0, "window_days": n},
            "coverage": _coverage_note(0, 0),
        }

    now = datetime.now(timezone.utc)

    def to_item(r) -> dict[str, Any]:
        deadline = r[7]
        days_left = (deadline - now).total_seconds() / 86400 if deadline else None
        return {
            "id": r[0],
            "filename": r[1],
            "project_name": r[2],
            "project_id": r[3],
            "purchaser": r[4],
            "budget": float(r[5]) if r[5] is not None else None,
            "max_price": float(r[6]) if r[6] is not None else None,
            "bid_deadline": deadline.isoformat() if deadline else None,
            "open_time": r[8].isoformat() if r[8] else None,
            "completeness": r[9],
            "days_left": round(days_left, 1) if days_left is not None else None,
            "urgency": classify_urgency(days_left),
        }

    items = [to_item(r) for r in rows]
    expired = [to_item(r) for r in expired_rows]

    buckets = {label: 0 for label, _ in URGENCY_BANDS}
    buckets["expired"] = len(expired)
    for it in items:
        buckets[it["urgency"]] = buckets.get(it["urgency"], 0) + 1

    return {
        "items": items,
        "expired": expired,
        "summary": {
            "total_documents": total,
            "with_deadline": with_deadline,
            "window_days": n,
            "buckets": buckets,
        },
        "coverage": _coverage_note(total, with_deadline),
    }


def _coverage_note(total: int, with_deadline: int) -> str:
    missing = max(0, total - with_deadline)
    parts = [
        "本监控只覆盖**已上传并成功抽取到截止时间**的招标文件，"
        "不是全网项目监控——未上传的项目不会出现在这里",
    ]
    if total == 0:
        parts.append("当前库里还没有任何招标文件，请先到「招标文件解析」页上传")
    elif missing:
        parts.append(
            f"另有 {missing} 份文件未抽到截止时间（文件里没有该信息，或抽取失败），"
            "它们不参与提醒"
        )
    return "。".join(parts)


__all__ = [
    "TABLE",
    "URGENCY_BANDS",
    "TenderStoreError",
    "classify_urgency",
    "delete_document",
    "ensure_schema",
    "get_document",
    "list_documents",
    "save_document",
    "upcoming_deadlines",
]

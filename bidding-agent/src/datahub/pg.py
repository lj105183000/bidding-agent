"""PostgreSQL 统一适配器。

为什么要有这一层
----------------
P0 之前，`analysis / competitor / material` 各自复制了一份 `_dsn()` 与 `_Cursor`。
三份实现的连接参数、超时、异常文案已经开始漂移（例如只有 `postgresql_client`
设了 `statement_timeout`）。本模块把连接参数与超时收口到一处：

    - `connect_timeout=10`：地址填错/数据库没起时快速失败，不拖满工具 30s 预算；
    - `statement_timeout=15000`：慢查询由服务端主动掐断，异常照常降级；
    - 异常统一映射到调用方给的 `error_cls`（`AnalysisError` / `MaterialError` /
      `CompetitorError`），服务层不必 import psycopg。

低层 `PGCursor` 只负责“连上/连不上”和“关闭”；语句本身的异常仍原样抛给
调用方（`material._fetchall` 等会把它收成业务异常）。`query_dicts` 是给质量
扫描用的高层入口，会把语句异常也收成 `DataHubQueryError`。
"""

from __future__ import annotations

import logging
import os
from typing import Any

import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row

from src.config import PROJECT_ROOT
from src.datahub.contract import (
    DataHubConfigError,
    DataHubError,
    DataHubQueryError,
)
from src.database.postgresql_client import _jsonable

logger = logging.getLogger(__name__)

# 与 postgresql_client 保持同一组超时：工具总预算 30s，连接/查询各留一半。
CONNECT_TIMEOUT_SECONDS = 10
STATEMENT_TIMEOUT_MS = 15000


def dsn() -> dict[str, Any]:
    """从 .env / 环境变量组装连接参数。

    密码只去行尾换行、不 strip 首位空格（首位空格可能是密码本身）。
    缺 `POSTGRES_HOST` 时抛 `DataHubConfigError`，由调用方映射成自己的业务异常。
    """
    load_dotenv(PROJECT_ROOT / ".env")
    host = (os.getenv("POSTGRES_HOST") or "").strip()
    if not host:
        raise DataHubConfigError("未配置 PostgreSQL（POSTGRES_HOST 为空）")
    return {
        "host": host,
        "port": (os.getenv("POSTGRES_PORT") or "5432").strip(),
        "user": (os.getenv("POSTGRES_USER") or "postgres").strip(),
        "password": (os.getenv("POSTGRES_PASSWORD") or "").rstrip("\r\n"),
        "dbname": (os.getenv("POSTGRES_DB") or "chatbot").strip(),
    }


def _connect(
    config: dict[str, Any] | None = None,
    *,
    dict_rows: bool = False,
) -> psycopg.Connection:
    cfg = dict(config or dsn())
    kwargs: dict[str, Any] = {
        **cfg,
        "connect_timeout": CONNECT_TIMEOUT_SECONDS,
        "options": f"-c statement_timeout={STATEMENT_TIMEOUT_MS}",
    }
    if dict_rows:
        kwargs["row_factory"] = dict_row
    return psycopg.connect(**kwargs)


class PGCursor:
    """只读游标上下文：连库 → 用 → 必关。

    与 `psycopg.connect()` 的 `with` 不同：那个管的是**事务**，退出时关闭的是
    事务而不是连接。本服务是长跑 Web 后端，查询又全是只读，必须显式 close，
    否则连接数会被慢慢耗尽。
    """

    def __init__(
        self,
        error_cls: type[Exception] = DataHubQueryError,
        *,
        config: dict[str, Any] | None = None,
        dict_rows: bool = False,
    ) -> None:
        self._error_cls = error_cls
        self._config = config
        self._dict_rows = dict_rows
        self._conn: psycopg.Connection | None = None

    def __enter__(self):
        try:
            self._conn = _connect(self._config, dict_rows=self._dict_rows)
        except DataHubConfigError as exc:
            raise self._error_cls(str(exc)) from exc
        except Exception as exc:
            raise self._error_cls(f"数据库连接失败：{type(exc).__name__}") from exc
        return self._conn.cursor()

    def __exit__(self, *exc_info):
        self._close()
        return False

    def _close(self) -> None:
        if self._conn is None:
            return
        try:
            self._conn.close()
        except Exception:  # pragma: no cover - 关闭失败没有补救价值
            pass
        finally:
            self._conn = None


class PGTransaction:
    """写事务上下文：成功 commit / 异常 rollback / 必关连接。

    给质量报告落库这类“必须原子”的写操作使用。只读查询请用 `PGCursor`，
    避免误提交。
    """

    def __init__(
        self,
        error_cls: type[Exception] = DataHubQueryError,
        *,
        config: dict[str, Any] | None = None,
        dict_rows: bool = False,
    ) -> None:
        self._error_cls = error_cls
        self._config = config
        self._dict_rows = dict_rows
        self._conn: psycopg.Connection | None = None

    def __enter__(self):
        try:
            self._conn = _connect(self._config, dict_rows=self._dict_rows)
        except DataHubConfigError as exc:
            raise self._error_cls(str(exc)) from exc
        except Exception as exc:
            raise self._error_cls(f"数据库连接失败：{type(exc).__name__}") from exc
        return self._conn.cursor()

    def __exit__(self, exc_type, exc, tb):
        commit_error: Exception | None = None
        try:
            if self._conn is not None:
                if exc_type is None:
                    self._conn.commit()
                else:
                    self._conn.rollback()
        except Exception as err:  # pragma: no cover - 提交失败属基础设施异常
            commit_error = err
        finally:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None
        if commit_error is not None and exc_type is None:
            raise self._error_cls(
                f"事务提交失败：{type(commit_error).__name__}"
            ) from commit_error
        return False


def query_dicts(
    sql: str,
    params: tuple = (),
    *,
    error_cls: type[Exception] = DataHubQueryError,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """执行查询并返回 JSON 原生类型的 dict 行。

    失败统一抛 `error_cls`，不返回空列表：质量扫描需要区分
    “查到了 0 条”和“表不存在/连不上”，否则会把故障误报成 pass。
    """
    try:
        with PGCursor(error_cls, config=config, dict_rows=True) as cur:
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()
    except error_cls:
        raise
    except Exception as exc:
        raise error_cls(f"查询失败：{type(exc).__name__}") from exc
    return [{key: _jsonable(value) for key, value in row.items()} for row in rows]


def health() -> dict[str, Any]:
    """连通性检查；失败不抛，返回 `{ok, error}`。"""
    try:
        with PGCursor(DataHubQueryError) as cur:
            cur.execute("SELECT 1")
            cur.fetchall()
        return {"ok": True, "error": None}
    except DataHubError as exc:
        return {"ok": False, "error": str(exc)}

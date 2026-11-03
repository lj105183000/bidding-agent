"""PostgreSQL 适配器测试。

不连真库：用局部 `_FakeConn/_FakeCursor` 顶替 `psycopg.connect`，
验证 DSN、超时参数、异常映射、事务提交/回滚与 JSON 归一化。
"""

from datetime import datetime
from decimal import Decimal

import pytest

from src.datahub import pg as P
from src.datahub.contract import DataHubConfigError, DataHubQueryError

_CONFIG = {
    "host": "localhost",
    "port": "5432",
    "user": "test",
    "password": "test",
    "dbname": "test",
}


class _FakeCursor:
    def __init__(self, rows=None):
        self.rows = list(rows or [])
        self.executed: list[tuple] = []

    def execute(self, sql, params=()):
        self.executed.append((sql, params))

    def fetchall(self):
        return self.rows

    def fetchone(self):
        return self.rows[0] if self.rows else None


class _FakeConn:
    def __init__(self, rows=None):
        self._cursor = _FakeCursor(rows)
        self.closed = False
        self.committed = False
        self.rolled_back = False

    def cursor(self):
        return self._cursor

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True

    def close(self):
        self.closed = True


def test_dsn_requires_postgres_host(monkeypatch):
    monkeypatch.setenv("POSTGRES_HOST", "")
    with pytest.raises(DataHubConfigError) as exc:
        P.dsn()
    assert "PostgreSQL" in str(exc.value)


def test_dsn_strips_trailing_cr(monkeypatch):
    monkeypatch.setenv("POSTGRES_HOST", "127.0.0.1")
    monkeypatch.setenv("POSTGRES_PASSWORD", "secret\r\n")
    assert P.dsn()["password"] == "secret"


def test_cursor_passes_timeouts_and_closes(monkeypatch):
    captured = {}
    fake_conn = _FakeConn()

    def fake_connect(**kwargs):
        captured.update(kwargs)
        return fake_conn

    monkeypatch.setattr(P.psycopg, "connect", fake_connect)
    with P.PGCursor(DataHubQueryError, config=_CONFIG) as cur:
        cur.execute("SELECT 1")
    assert captured["connect_timeout"] == P.CONNECT_TIMEOUT_SECONDS
    assert f"statement_timeout={P.STATEMENT_TIMEOUT_MS}" in captured["options"]
    assert fake_conn.closed is True


def test_cursor_maps_connect_error(monkeypatch):
    monkeypatch.setattr(
        P.psycopg,
        "connect",
        lambda **_: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    with pytest.raises(DataHubQueryError) as exc:
        with P.PGCursor(DataHubQueryError, config=_CONFIG):
            pass
    assert "数据库连接失败" in str(exc.value)


class _TaskError(RuntimeError):
    pass


def test_transaction_commits_and_closes(monkeypatch):
    fake_conn = _FakeConn()
    monkeypatch.setattr(P, "_connect", lambda *a, **k: fake_conn)
    with P.PGTransaction(_TaskError, config=_CONFIG) as cur:
        cur.execute("INSERT INTO t VALUES (1)")
    assert fake_conn.committed is True
    assert fake_conn.rolled_back is False
    assert fake_conn.closed is True


def test_transaction_rolls_back_on_exception(monkeypatch):
    fake_conn = _FakeConn()
    monkeypatch.setattr(P, "_connect", lambda *a, **k: fake_conn)
    with pytest.raises(ValueError):
        with P.PGTransaction(_TaskError, config=_CONFIG):
            raise ValueError("boom")
    assert fake_conn.rolled_back is True
    assert fake_conn.committed is False
    assert fake_conn.closed is True


class _FakeCursorContext:
    def __init__(self, rows):
        self._cursor = _FakeCursor(rows)

    def __enter__(self):
        return self._cursor

    def __exit__(self, *exc_info):
        return False


def test_query_dicts_normalizes_json_types(monkeypatch):
    rows = [{"amount": Decimal("1.50"), "ts": datetime(2026, 1, 2, 3, 4, 5)}]
    monkeypatch.setattr(P, "PGCursor", lambda *a, **k: _FakeCursorContext(rows))
    result = P.query_dicts("SELECT 1", config=_CONFIG)
    assert result == [{"amount": 1.5, "ts": "2026-01-02T03:04:05"}]


def test_query_dicts_maps_query_error(monkeypatch):
    class _BrokenCursor(_FakeCursor):
        def execute(self, sql, params=()):
            raise RuntimeError("bad sql")

    # 直接构造一个会抛的上下文
    class _BrokenContext:
        def __enter__(self):
            return _BrokenCursor()

        def __exit__(self, *exc_info):
            return False

    monkeypatch.setattr(P, "PGCursor", lambda *a, **k: _BrokenContext())
    with pytest.raises(DataHubQueryError) as exc:
        P.query_dicts("SELECT 1", config=_CONFIG)
    assert "查询失败" in str(exc.value)

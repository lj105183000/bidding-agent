"""feedback_summary / conversation_stats 只读聚合的契约测试（全程不连真实数据库）。

dashboard 是面向运营的只读入口：这里钉死三件事——SQL 参数化（days 走 %s 不拼接）、
查询失败降级为空 dict（Agent/脚本不因库挂而崩）、返回形状稳定（dashboard 直接取键打印）。

fake 与 test_pg_validation.py 同款：`_FakeCursor` 记录发出去的 SQL 与参数，靠它断言
"days 到底有没有进占位符"。注意 `_query` 内部是 `with conn.cursor() as cur`，所以假游标
必须带 `__enter__` / `__exit__`——缺了它们 `_query` 会兜住 AttributeError 静默返回 []，
测试就变成永远失败的空转，而不是真正走到断言。
"""

import src.database.postgresql_client as pgmod

POSTGRES_ENV = {
    "POSTGRES_HOST": "127.0.0.1",
    "POSTGRES_PORT": "5432",
    "POSTGRES_DB": "chatbot",
    "POSTGRES_USER": "postgres",
    "POSTGRES_PASSWORD": "test-password",
}


class _FakeCursor:
    """按调用次序弹出一组结果集：feedback_summary 会连发两条 SQL（汇总 + 每日）。"""

    def __init__(self, result_sets):
        self.result_sets = [list(rs) for rs in result_sets]
        self.executed = []

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchall(self):
        return self.result_sets.pop(0) if self.result_sets else []

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class _FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def _make_client(monkeypatch):
    for name, value in POSTGRES_ENV.items():
        monkeypatch.setenv(name, value)
    return pgmod.PostgresClient()


def _install(monkeypatch, result_sets):
    client = _make_client(monkeypatch)
    cursor = _FakeCursor(result_sets)
    monkeypatch.setattr(pgmod.psycopg, "connect", lambda **kwargs: _FakeConnection(cursor))
    return client, cursor


def test_feedback_summary_shape_and_parametrization(monkeypatch):
    summary_row = {"total": 5, "up": 3, "down": 2}
    daily_rows = [{"day": "2026-09-29", "cnt": 4}, {"day": "2026-09-30", "cnt": 1}]
    client, cursor = _install(monkeypatch, [[summary_row], daily_rows])

    result = client.feedback_summary(days=7)

    assert result["total"] == 5 and result["up"] == 3 and result["down"] == 2
    assert result["daily"] == daily_rows
    # 两条 SQL；days 走 %s 参数（形如 "7 days"），不拼接进语句
    assert len(cursor.executed) == 2
    second_sql, second_params = cursor.executed[1]
    assert "%s::interval" in second_sql
    assert second_params == ("7 days",)


def test_feedback_summary_clamps_days(monkeypatch):
    client, cursor = _install(monkeypatch, [[{"total": 0, "up": 0, "down": 0}], []])
    client.feedback_summary(days=999)
    assert cursor.executed[1][1] == ("90 days",)


def test_feedback_summary_degrades_to_empty(monkeypatch):
    client = _make_client(monkeypatch)

    def _fail(**kwargs):
        raise ConnectionError("down")

    monkeypatch.setattr(pgmod.psycopg, "connect", _fail)
    assert client.feedback_summary(days=7) == {}


def test_conversation_stats(monkeypatch):
    client, cursor = _install(monkeypatch, [[{"conversations": 12, "messages": 340}]])
    result = client.conversation_stats()
    assert result == {"conversations": 12, "messages": 340}
    sql = cursor.executed[0][0]
    assert "FROM conversations" in sql and "FROM messages" in sql


def test_conversation_stats_degrades_to_empty(monkeypatch):
    client = _make_client(monkeypatch)

    def _fail(**kwargs):
        raise ConnectionError("down")

    monkeypatch.setattr(pgmod.psycopg, "connect", _fail)
    assert client.conversation_stats() == {}

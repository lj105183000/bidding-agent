"""find_id_by_session：按 session_id 反查会话 id（server 端会话/反馈路由依赖）。"""

from src.database.postgresql_client import PostgresClient


def test_find_id_by_session_returns_id(monkeypatch):
    monkeypatch.setattr(
        PostgresClient, "_query",
        lambda self, sql, params: [{"id": 7}],
    )
    assert PostgresClient().find_id_by_session("s-1") == 7


def test_find_id_by_session_returns_negative_one_when_missing(monkeypatch):
    monkeypatch.setattr(
        PostgresClient, "_query",
        lambda self, sql, params: [],
    )
    assert PostgresClient().find_id_by_session("s-none") == -1

"""Neo4j 适配器测试。

不连真库：局部 `_FakeDriver/_FakeSession` 顶替 `GraphDatabase`，
验证共享 driver、严格/非严格模式、配置缺失与 close 幂等。
"""

import pytest

from src.datahub import neo4j as N
from src.datahub.contract import DataHubConfigError, DataHubUnavailableError


class _FakeSession:
    def __init__(self, rows=None, error=None):
        self.rows = list(rows or [])
        self.error = error
        self.database = None
        self.last_call = None

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def run(self, cypher, **params):
        self.last_call = (cypher, params)
        if self.error is not None:
            raise self.error
        return list(self.rows)


class _FakeDriver:
    def __init__(self, rows=None, error=None):
        self.rows = rows
        self.error = error
        self.sessions: list[_FakeSession] = []
        self.closed = False

    def session(self, database=None):
        session = _FakeSession(self.rows, self.error)
        session.database = database
        self.sessions.append(session)
        return session

    def close(self):
        self.closed = True


class _FakeGraphDatabase:
    def __init__(self, driver):
        self._driver = driver
        self.calls: list[tuple] = []

    def driver(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self._driver


def _install(monkeypatch, driver):
    fake_gdb = _FakeGraphDatabase(driver)
    monkeypatch.setattr(N, "GraphDatabase", fake_gdb)
    monkeypatch.setattr(N, "_driver", None)
    monkeypatch.setattr(N, "_config", lambda: ("neo4j://test", ("u", "p"), None))
    return fake_gdb


def test_run_returns_rows(monkeypatch):
    driver = _FakeDriver(rows=[{"label": "SubjectMatter", "count": 3}])
    _install(monkeypatch, driver)
    assert N.run("MATCH (n) RETURN n") == [{"label": "SubjectMatter", "count": 3}]


def test_run_passes_params(monkeypatch):
    driver = _FakeDriver(rows=[])
    _install(monkeypatch, driver)
    N.run("MATCH (n {name: $name}) RETURN n", name="硒鼓")
    assert driver.sessions[0].last_call == (
        "MATCH (n {name: $name}) RETURN n",
        {"name": "硒鼓"},
    )


def test_run_strict_mode_raises(monkeypatch):
    driver = _FakeDriver(error=RuntimeError("down"))
    _install(monkeypatch, driver)
    with pytest.raises(DataHubUnavailableError) as exc:
        N.run("MATCH (n)", raise_on_error=True, error_cls=DataHubUnavailableError)
    assert "知识图谱查询失败" in str(exc.value)


def test_run_non_strict_returns_empty(monkeypatch):
    driver = _FakeDriver(error=RuntimeError("down"))
    _install(monkeypatch, driver)
    assert N.run("MATCH (n)") == []


def test_driver_is_singleton(monkeypatch):
    driver = _FakeDriver()
    fake_gdb = _install(monkeypatch, driver)
    first = N.get_driver()
    second = N.get_driver()
    assert first is second is driver
    assert len(fake_gdb.calls) == 1


def test_missing_config_raises(monkeypatch):
    monkeypatch.setattr(N, "_driver", None)
    monkeypatch.setenv("NEO4J_URI", "")
    monkeypatch.setenv("NEO4J_USERNAME", "")
    monkeypatch.setenv("NEO4J_PASSWORD", "")
    with pytest.raises(DataHubConfigError) as exc:
        N.get_driver()
    assert "NEO4J_URI" in str(exc.value)


def test_close_is_idempotent(monkeypatch):
    driver = _FakeDriver()
    monkeypatch.setattr(N, "_driver", driver)
    N.close()
    N.close()
    assert driver.closed is True
    assert N._driver is None

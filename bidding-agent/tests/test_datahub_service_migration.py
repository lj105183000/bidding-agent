"""服务层迁移到 DataHub 的回归测试。

守住三件事：
    - `analysis / competitor / material` 不再直接 `psycopg.connect` / `GraphDatabase.driver`；
    - `_Cursor()` 返回 DataHub 适配器（而不是各自复制的实现）；
    - `analysis.data_quality()` 的响应形状不变（前端与既有测试依赖）。
"""

import inspect

from src.datahub.pg import PGCursor
from src.services import analysis as A
from src.services import competitor as C
from src.services import material as M


def test_services_do_not_directly_connect():
    for module, name in ((A, "analysis"), (C, "competitor"), (M, "material")):
        source = inspect.getsource(module)
        assert "psycopg.connect(" not in source, f"{name} 仍有直连 PG"
        assert "GraphDatabase.driver(" not in source, f"{name} 仍有直连 Neo4j"


def test_cursor_helpers_return_datahub_cursor():
    assert isinstance(A._Cursor(), PGCursor)
    assert isinstance(M._Cursor(), PGCursor)
    assert isinstance(C._Cursor(), PGCursor)


def test_dsn_helpers_delegate_to_datahub(monkeypatch):
    monkeypatch.setenv("POSTGRES_HOST", "")
    for module, error_cls in (
        (A, A.AnalysisError),
        (C, C.CompetitorError),
        (M, M.MaterialError),
    ):
        try:
            module._dsn()
        except error_cls as exc:
            assert "PostgreSQL" in str(exc)
        else:  # pragma: no cover
            raise AssertionError(f"{module.__name__}._dsn() 未抛业务异常")


class _ScriptedCursor:
    def __init__(self, results):
        self._results = list(results)
        self._current = None

    def execute(self, sql, params=()):
        self._current = self._results.pop(0)

    def fetchone(self):
        return self._current

    def fetchall(self):
        return self._current


class _CursorContext:
    def __init__(self, cursor):
        self.cursor = cursor

    def __enter__(self):
        return self.cursor

    def __exit__(self, *exc_info):
        return False


def test_data_quality_shape_unchanged(monkeypatch):
    cursor = _ScriptedCursor(
        [
            (10,),                              # count(*)
            [("amount",), ("project_id",)],     # information_schema.columns
            (5,),                               # amount 空值数
            (0,),                               # project_id 空值数
            [(1, "托养服务", "采购人", "供应商", 660000000.0, None, "服务")],
            (3,),                               # suspicious 真值计数
            (50.0,),                            # P90
            [("服务", 5)],                      # category 分布
        ]
    )
    monkeypatch.setattr(A, "_Cursor", lambda: _CursorContext(cursor))

    result = A.data_quality()

    assert result["total_rows"] == 10
    assert result["empty_rates"] == [
        {"column": "amount", "empty_count": 5, "empty_percent": 50.0}
    ]
    assert result["suspicious_count"] == 3
    assert result["suspicious_returned"] == 1
    assert "巨额" in result["suspicious_records"][0]["reason"]
    assert result["category_values"] == [{"name": "服务", "count": 5}]
    assert result["suspicious_rules"]["huge_amount"] == A.HUGE_AMOUNT
    assert "notes" in result

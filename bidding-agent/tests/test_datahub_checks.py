"""共享检查 SQL 测试（可疑金额判据 / 空值率 / 可疑明细）。

判据本身是从 `analysis.py` 迁来的，测试重点：
    - `%` 必须转义成 `%%`（psycopg 占位符）；
    - 空值率口径是 NULL + ''；
    - 明细与真值计数分开返回；
    - 表名必须是简单标识符（防未来有人把外部输入传进来）。
"""

import pytest

from src.datahub import checks


class _ScriptedCursor:
    def __init__(self, results):
        self._results = list(results)
        self._current = None
        self.executed: list[tuple] = []

    def execute(self, sql, params=()):
        self.executed.append((sql, params))
        self._current = self._results.pop(0)

    def fetchone(self):
        return self._current

    def fetchall(self):
        return self._current


def test_suspicious_sql_contains_both_rules():
    sql = checks.suspicious_sql()
    assert "10000000000" in sql
    assert str(checks.HUGE_AMOUNT) in sql
    assert "ILIKE" in sql


def test_suspicious_sql_has_ok_keyword_exemption():
    sql = checks.suspicious_sql()
    for keyword in ("工程", "建设", "施工"):
        assert keyword in sql


def test_suspicious_sql_escapes_percent_for_psycopg():
    """psycopg 把单个 % 当占位符，LIKE 的 % 必须写成 %%。"""
    sql = checks.suspicious_sql()
    assert "%%" in sql
    assert "ILIKE '%" not in sql.replace("%%", "")


def test_empty_rates_counts_null_and_empty_string():
    cursor = _ScriptedCursor(
        [
            (10,),                       # count(*)
            [("amount",), ("project_id",)],  # information_schema.columns
            (5,),                        # amount 空值数
            (0,),                        # project_id 空值数
        ]
    )
    total, empties = checks.empty_rates(cursor, "bidding_records")
    assert total == 10
    assert empties == [
        {"column": "amount", "empty_count": 5, "empty_percent": 50.0}
    ]
    # 表名走参数化；列名来自 information_schema，不进 params
    assert cursor.executed[1][1] == ("bidding_records",)


def test_empty_rates_rejects_injection_in_table_name():
    with pytest.raises(ValueError):
        checks.empty_rates(_ScriptedCursor([]), "bidding_records; DROP TABLE users")


def test_suspicious_rows_returns_detail_and_true_count():
    rows = [(1, "托养服务", "采购人", "供应商", 660000000.0, None, "服务")]
    cursor = _ScriptedCursor([rows, (3,)])
    detail, total = checks.suspicious_rows(cursor, "bidding_records", limit=30)
    assert detail == rows
    assert total == 3


def test_suspicious_count_returns_int():
    cursor = _ScriptedCursor([(5,)])
    assert checks.suspicious_count(cursor, "bidding_records") == 5

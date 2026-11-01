"""`to_postgresql.py` 的破坏性操作守卫测试。

重点：
    - 表已存在且没给 `--force` 时，**必须在 DROP 之前**拒绝；
    - `--dry-run` 不读 .env、不连库、不写库；
    - 给了 `--force` 才进入真正的导入。

全部用局部 fake + monkeypatch，不连真实 PostgreSQL。
"""

import sys

import pandas as pd
import pytest

from src.database import to_postgresql as T


def _df() -> pd.DataFrame:
    return pd.DataFrame({"项目编号": ["P-001"], "中标金额": [100.0]})


def test_require_force_refuses_existing_table():
    with pytest.raises(SystemExit) as exc:
        T.require_force(True, False)
    assert exc.value.code == 2


def test_require_force_allows_missing_table():
    T.require_force(False, False)  # 不抛即通过


def test_require_force_allows_explicit_force():
    T.require_force(True, True)


def test_dry_run_does_not_connect(monkeypatch):
    monkeypatch.setattr(T, "read_xlsx", lambda path, sheet: _df())
    monkeypatch.setattr(
        T,
        "read_config",
        lambda *_args, **_kwargs: pytest.fail("--dry-run 不应读取数据库配置"),
    )
    monkeypatch.setattr(
        T,
        "table_exists",
        lambda *_args, **_kwargs: pytest.fail("--dry-run 不应连接数据库"),
    )
    monkeypatch.setattr(sys, "argv", ["to_postgresql.py", "dummy.xlsx", "--dry-run"])
    assert T.main() == 0


def test_main_without_force_exits_nonzero(monkeypatch):
    monkeypatch.setattr(T, "read_xlsx", lambda path, sheet: _df())
    monkeypatch.setattr(
        T,
        "read_config",
        lambda *_args, **_kwargs: {
            "host": "localhost",
            "port": 5432,
            "dbname": "test",
            "user": "test",
            "password": "test",
        },
    )
    monkeypatch.setattr(T, "table_exists", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(sys, "argv", ["to_postgresql.py", "dummy.xlsx"])
    with pytest.raises(SystemExit) as exc:
        T.main()
    assert exc.value.code == 2


def test_force_reaches_import(monkeypatch):
    calls: dict[str, int] = {}

    def fake_import(config, plan, rows):
        calls["rows"] = len(rows)
        calls["columns"] = len(plan)
        return len(rows)

    monkeypatch.setattr(T, "read_xlsx", lambda path, sheet: _df())
    monkeypatch.setattr(
        T,
        "read_config",
        lambda *_args, **_kwargs: {
            "host": "localhost",
            "port": 5432,
            "dbname": "test",
            "user": "test",
            "password": "test",
        },
    )
    monkeypatch.setattr(T, "table_exists", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(T, "import_to_postgres", fake_import)
    monkeypatch.setattr(sys, "argv", ["to_postgresql.py", "dummy.xlsx", "--force"])
    assert T.main() == 0
    assert calls == {"rows": 1, "columns": 2}


def test_dry_run_wins_over_force(monkeypatch):
    """同时给 --force 和 --dry-run 时，dry-run 仍然不连库。"""
    monkeypatch.setattr(T, "read_xlsx", lambda path, sheet: _df())
    monkeypatch.setattr(
        T,
        "read_config",
        lambda *_args, **_kwargs: pytest.fail("--dry-run 不应读取数据库配置"),
    )
    monkeypatch.setattr(sys, "argv", ["to_postgresql.py", "dummy.xlsx", "--force", "--dry-run"])
    assert T.main() == 0

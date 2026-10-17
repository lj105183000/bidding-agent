"""`messages.image_analyses`（图片解析文本落库）的回归测试。

★ 这个文件存在的原因：一个**只在真库上才暴露**的静默失效
====================================================
第一版把"该列是否可用"缓存在 `_has_image_analyses`，默认 False、只在
`ensure_schema()` 里置位。测试环境里我先手动跑过 ensure_schema，于是
存储层测起来一切正常；但 API 层用的是**另一个 PostgresClient 实例**
（没人给它跑过 ensure_schema）→ `save_message` 静默跳过该列 →
读回来永远是 None，**而且没有任何报错**。

真正的发现方式是"打一遍真端点再读回来"（见 logs/verify_image_persistence.py）。
这里把它固化成用例：**新建的、没跑过 ensure_schema 的实例也必须能存上**。
"""

import pytest

from src.database import image_analyses_migration


class _FakePG:
    """记录所有 SQL 的假客户端（不连库）。

    `image_insert_ok=False` 模拟"列其实不存在"：**只对带该列的 INSERT 报错**，
    不带该列的照常成功 —— 这样才测得出"退化为普通写入"这条路径。
    """

    def __init__(self, *, ddl_ok=True, image_insert_ok=True):
        self.ddl_ok = ddl_ok
        self.image_insert_ok = image_insert_ok
        self.sqls: list[str] = []
        self.params: list[tuple] = []

    def _query(self, sql, params=None, fetch=True):
        self.sqls.append(" ".join(sql.split()))
        self.params.append(tuple(params or ()))
        if "ALTER TABLE" in sql:
            if not self.ddl_ok:
                raise RuntimeError("没有 DDL 权限")
            return []
        if "INSERT INTO messages" in sql:
            if "image_analyses" in sql and not self.image_insert_ok:
                raise RuntimeError("列不存在")
            return [{"id": 42}]
        return []


class TestMigration:
    def test_ddl_is_idempotent_and_fetch_false(self):
        pg = _FakePG()
        assert image_analyses_migration.ensure_column(pg) is True
        sql = pg.sqls[0]
        assert "ADD COLUMN IF NOT EXISTS image_analyses" in sql
        assert "jsonb" in sql, "结构化解析结果应存 JSONB，便于按字段查询"

    def test_ddl_failure_degrades_gracefully(self):
        """加不上列（无 DDL 权限）时返回 False，不抛异常。"""
        pg = _FakePG(ddl_ok=False)
        assert image_analyses_migration.ensure_column(pg) is False


class _ClientUnderTest:
    """借 PostgresClient 的真实方法，但把 `_query` 换成假的。

    用真实类而不是重写一份逻辑：要测的正是"方法里怎么用那个标志"。
    """

    def __init__(self, fake: _FakePG):
        from src.database.postgresql_client import PostgresClient

        self.client = PostgresClient.__new__(PostgresClient)  # 跳过 __init__（不读 env）
        self.client._has_image_analyses = None
        self.client._query = fake._query  # type: ignore[method-assign]
        self.fake = fake


class TestLazyProbe:
    """★ 惰性探测：没跑过 ensure_schema 的实例也要能存上。"""

    def test_fresh_client_still_persists(self):
        fake = _FakePG()
        under = _ClientUnderTest(fake)
        mid = under.client.save_message(
            conversation_id=1,
            role="user",
            content="这两张图说了什么？",
            image_analyses=[{"name": "a.png", "analysis": "编号 A-1"}],
        )
        assert mid == 42
        assert any("ALTER TABLE" in sql for sql in fake.sqls), (
            "首次使用时必须先补列 —— 否则字段会被静默跳过（真实踩过的坑）"
        )
        insert = next((sql for sql in fake.sqls if "INSERT INTO messages" in sql), "")
        assert "image_analyses" in insert, "补列之后插入必须带上该列"

    def test_probe_runs_once_per_instance(self):
        """探测一次就够：第二次保存不该再跑 DDL。"""
        fake = _FakePG()
        under = _ClientUnderTest(fake)
        for _ in range(3):
            under.client.save_message(1, "user", "x",
                                      image_analyses=[{"name": "a", "analysis": "b"}])
        ddl = [sql for sql in fake.sqls if "ALTER TABLE" in sql]
        assert len(ddl) == 1, f"DDL 应只跑一次（实际 {len(ddl)} 次）"

    def test_no_ddl_when_no_analyses(self):
        """没有图片的消息不该触发 DDL（绝大多数消息都没有图）。"""
        fake = _FakePG()
        under = _ClientUnderTest(fake)
        mid = under.client.save_message(1, "user", "纯文字")
        assert mid == 42
        assert not any("ALTER TABLE" in sql for sql in fake.sqls)

    def test_insert_falls_back_when_column_missing(self):
        """列在补上之后仍插入失败 → 退化为不带该列的插入，消息本身不能丢。"""
        fake = _FakePG(image_insert_ok=False)
        under = _ClientUnderTest(fake)
        mid = under.client.save_message(1, "user", "x",
                                        image_analyses=[{"name": "a", "analysis": "b"}])
        assert mid == 42, "图片解析存不上不该导致整条消息写不进去"
        last = [sql for sql in fake.sqls if "INSERT INTO messages" in sql][-1]
        assert "image_analyses" not in last

    def test_ddl_failure_keeps_messages_working(self):
        """没有 DDL 权限的环境：消息照常保存，只是不带图片解析。"""
        fake = _FakePG(ddl_ok=False)
        under = _ClientUnderTest(fake)
        mid = under.client.save_message(1, "user", "x",
                                        image_analyses=[{"name": "a", "analysis": "b"}])
        assert mid == 42
        assert under.client._has_image_analyses is False


class TestLoadSelectsColumnSafely:
    """★ 读端绝不能赌该列存在：`_query` 会吞异常，赌错=整个历史变空。"""

    def _select_sql(self, available: bool) -> str:
        fake = _FakePG()
        under = _ClientUnderTest(fake)
        under.client._has_image_analyses = available
        under.client.load_messages(1)
        return next(sql for sql in fake.sqls if "SELECT" in sql)

    def test_column_included_when_available(self):
        assert "image_analyses" in self._select_sql(True)

    def test_column_excluded_when_unavailable(self):
        sql = self._select_sql(False)
        assert "image_analyses" not in sql, (
            "老库没有该列时必须退回老 SELECT —— 否则查询失败被吞成空列表，"
            "用户看到整个会话历史凭空消失"
        )
        # 基本列一个都不能少
        for column in ("id", "role", "content", "sources", "tool_name", "created_at"):
            assert column in sql

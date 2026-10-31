"""Langfuse 反馈评分链路的测试。

★ 本轮实施的四处补足
==============
    P0  用户反馈 → Langfuse Score（session→trace 映射 + create_score）
    P1  质量信号随 turn_summary 上报（verify_* / rag_* / provider 降级）
    P2  `failing_components` 死字段改为"只在真有降级时给出是哪一家"
    P3  （未做，见文档）

★ 测什么
=====
重点是**不变量**（这四处全是观测旁路，绝不能影响业务）：
    ① 任何一环失败都不抛异常、不阻塞
    ② 未配 Langfuse 时全部 no-op
    ③ 未知 rating 不猜语义
    ④ DDL 必须 `fetch=False`（本轮踩到的真坑）
"""

from __future__ import annotations

import pathlib
import sys

import pytest

from src import langfuse_feedback as lf


def _fake_langfuse(monkeypatch, create_score):
    """把一个**假的** langfuse 模块注入 sys.modules，并返回它。

    ★ 为什么不用 import langfuse（原写法，会挂）

    langfuse 是**可选依赖**（pyproject.toml 的
    [project.optional-dependencies].langfuse），主依赖里**没有**它，
    而 CI 的 uv sync --frozen 只装默认组 ——
    所以裸导入在 CI 上必然 ModuleNotFoundError，**把 CI 变红**。

    更要紧的是：这两条测试断言的是**我们自己**的参数选择
    （rating→数值映射、大小写归一），与 SDK 的真实行为无关。
    用假模块不仅去掉可选依赖，还让测试**更贴合意图**。

    另外 src.langfuse_feedback.score_feedback 是在**函数内部**
    rom langfuse import Langfuse 的，故 patch sys.modules 即可生效
    （不需要 patch 调用方属性）。
    """
    import sys
    import types

    mod = types.ModuleType("langfuse")

    class _C:
        def create_score(self, **kw):
            create_score(kw)

    mod.Langfuse = lambda **kw: _C()          # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langfuse", mod)
    return mod


class TestScoreValueMapping:
    """rating → Score 数值的映射。"""

    def test_known_values(self):
        assert lf._SCORE_VALUES == {"up": 1.0, "down": -1.0}

    def test_unknown_rating_does_not_guess(self, monkeypatch):
        """★ 未知 rating 必须拒绝，不能猜语义。

        猜错的后果是**静默写错数据**：把"不确定"的反馈写成"满意"，
        之后所有基于 Score 的统计都偏了，而且看不出来。
        """
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-x")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-x")

        class _PG:
            def find_trace_id_by_message(self, mid):
                return "t-1"

        assert lf.score_feedback("s", 1, "maybe", pg=_PG()) is False
        assert lf.score_feedback("s", 1, "", pg=_PG()) is False
        assert lf.score_feedback("s", 1, None, pg=_PG()) is False

    def test_case_insensitive(self, monkeypatch):
        """`rating` 大小写不该决定成败（'UP' 与 'up' 同义）。"""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-x")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-x")
        calls: list = []

        class _PG:
            def find_trace_id_by_message(self, mid):
                return "t-1"

        _fake_langfuse(monkeypatch, calls.append)
        assert lf.score_feedback("s", 1, "UP", pg=_PG()) is True
        assert calls and calls[0]["value"] == 1.0


class TestScoreFeedbackGuards:
    """三条前置检查：任一不满足就静默返回 False。"""

    def test_not_configured_returns_false(self, monkeypatch):
        """未配 Langfuse → 没有 trace 可挂，**不是错误**。"""
        for k in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
            monkeypatch.delenv(k, raising=False)
        assert lf.score_feedback("s", 1, "up") is False

    def test_missing_trace_id_returns_false(self, monkeypatch):
        """消息上没有 trace_id（老消息 / 加列之前写入的）→ 跳过，不报错。"""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-x")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-x")

        class _PG:
            def find_trace_id_by_message(self, mid):
                return None

        assert lf.score_feedback("s", 1, "up", pg=_PG()) is False

    def test_never_raises(self, monkeypatch):
        """★ 观测旁路绝不能抛异常。

        反馈端点已经成功保存了反馈；这一步失败只该少一个评分，
        不该让用户看到 503。
        """
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-x")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-x")

        class _Boom:
            def find_trace_id_by_message(self, mid):
                raise RuntimeError("炸")

        lf.score_feedback("s", 1, "up", pg=_Boom())  # 不应抛

    def test_sdk_exception_swallowed(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-x")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-x")

        class _PG:
            def find_trace_id_by_message(self, mid):
                return "t-1"

        def boom(kw):
            raise RuntimeError("网络炸了")

        _fake_langfuse(monkeypatch, boom)
        assert lf.score_feedback("s", 1, "up", pg=_PG()) is False


class TestSessionTraceMapping:
    """session → trace 的跨请求映射。

    ★ 为什么用数据库而不是内存字典
    ========================
        ① 多进程部署（uvicorn 多 worker）下内存字典失效：
           写它的进程与读它的进程可能不是同一个
        ② 重启即丢：用户还没点赞就重启，那条反馈永远接不上
    """

    @pytest.mark.integration
    def test_remember_and_lookup(self):
        from src.langfuse_feedback import lookup_trace_id, remember_session_trace

        sid = "pytest-session-trace-map"
        assert remember_session_trace(sid, "pytest-trace-abc") is True
        assert lookup_trace_id(sid) == "pytest-trace-abc"
        # 同一会话再写一次应**覆盖**（只保留最近一次）
        assert remember_session_trace(sid, "pytest-trace-def") is True
        assert lookup_trace_id(sid) == "pytest-trace-def"
        # 清理
        from src.database.postgresql_client import PostgresClient

        PostgresClient()._query(
            "DELETE FROM session_traces WHERE session_id = %s", (sid,), fetch=False
        )

    def test_empty_inputs(self):
        from src.langfuse_feedback import lookup_trace_id, remember_session_trace

        assert remember_session_trace("", "t") is False
        assert remember_session_trace("s", "") is False
        assert lookup_trace_id("") is None

    def test_unknown_session_returns_none(self):
        from src.langfuse_feedback import lookup_trace_id

        assert lookup_trace_id("definitely-not-a-real-session") is None

    def test_never_raises_on_broken_db(self, monkeypatch):
        """库挂了也不能抛 —— 它挂在请求收尾路径上。"""
        import src.database.postgresql_client as pgmod

        class _Boom:
            def __init__(self, *a, **k):
                raise RuntimeError("库连不上")

        monkeypatch.setattr(pgmod, "PostgresClient", _Boom)
        from src.langfuse_feedback import lookup_trace_id, remember_session_trace

        assert remember_session_trace("s", "t") is False
        assert lookup_trace_id("s") is None


class TestDdlUsesFetchFalse:
    """★ DDL 必须 `fetch=False`（本轮踩到的真坑）。

    `PostgresClient._query` 默认 `fetch=True`，会执行 `cur.fetchall()`，
    而 **DDL 不产生结果集** → psycopg 抛
    `ProgrammingError: the last operation didn't produce records`。

    后果很隐蔽：**语句其实执行成功了**，但异常被 `_query` 吞掉并返回 `[]`，
    调用方的 `except` 又把它当失败 → 全程没有任何报错，
    只有"反馈接不上 Langfuse"这一个结果。

    这里用源码断言锁住，因为"语句成功但被判为失败"无法用返回值测出来。
    """

    def test_ensure_table_source_uses_fetch_false(self):
        src = pathlib.Path(lf.__file__).read_text(encoding="utf-8")
        i = src.find("def ensure_table")
        seg = src[i:i + 1400]
        assert "fetch=False" in seg, (
            "ensure_table 的 CREATE TABLE 必须传 fetch=False，"
            "否则 DDL 会被误判为失败（且表其实建成了）"
        )

    def test_ensure_column_source_uses_fetch_false(self):
        from src.database import trace_migration

        src = pathlib.Path(trace_migration.__file__).read_text(encoding="utf-8")
        i = src.find("def ensure_column")
        seg = src[i:i + 900]
        assert "fetch=False" in seg, "ALTER TABLE 同样必须 fetch=False"

    def test_ddl_actually_works(self):
        """端到端：两个幂等 DDL 都应返回 True，且可重复调用。"""
        from src.database.postgresql_client import PostgresClient
        from src.database.trace_migration import ensure_column
        from src.langfuse_feedback import ensure_table

        pg = PostgresClient()
        assert ensure_table(pg) is True
        assert ensure_table(pg) is True, "重复调用也必须成功（IF NOT EXISTS）"
        assert ensure_column(pg) is True
        assert ensure_column(pg) is True


class TestTraceIdOnMessage:
    """`messages.trace_id` 的写入与查回。"""

    @pytest.mark.integration
    def test_save_and_find(self):
        from src.database.postgresql_client import PostgresClient

        pg = PostgresClient()
        cid = pg.find_id_by_session("pytest-trace-id-session")
        if cid < 0:
            cid = pg.create_conversation(title="pytest-trace-id",
                                         session_id="pytest-trace-id-session")
        mid = pg.save_message(conversation_id=cid, role="assistant",
                              content="pytest trace_id 测试", trace_id="t-xyz")
        assert mid >= 0
        assert pg.find_trace_id_by_message(mid) == "t-xyz"

    @pytest.mark.integration
    def test_save_without_trace_id(self):
        """不带 trace_id 时照旧写入（trace_id 为 NULL）。"""
        from src.database.postgresql_client import PostgresClient

        pg = PostgresClient()
        cid = pg.find_id_by_session("pytest-trace-id-session")
        mid = pg.save_message(conversation_id=cid, role="assistant",
                              content="pytest 无 trace_id")
        assert mid >= 0
        assert pg.find_trace_id_by_message(mid) is None

    def test_find_unknown_message(self):
        from src.database.postgresql_client import PostgresClient

        assert PostgresClient().find_trace_id_by_message(999999999) is None

    def test_find_invalid_message_id(self):
        """非法 id 不抛（`_as_id` 会抛 ValueError，这里要吞掉）。"""
        from src.database.postgresql_client import PostgresClient

        assert PostgresClient().find_trace_id_by_message("not-an-int") is None


class TestQualityMetadataWiring:
    """P1：质量信号要真的被塞进 `_tagger.finish(metadata=...)`。

    用源码断言，因为这条路径要跑真实问答才能触发，
    而单测里不值得为它起一次完整问答。
    """

    @staticmethod
    def _core_src() -> str:
        root = pathlib.Path(__file__).resolve().parents[1]
        return (root / "src/agent/core.py").read_text(encoding="utf-8")

    def test_quality_keys_present(self):
        src = self._core_src()
        for key in ("verify_ok", "verify_critical_ok", "unsupported_articles",
                    "rag_ms", "rag_cache_hit", "provider_switches"):
            assert f'"{key}"' in src, f"质量信号 {key} 未接入 finish()"

    def test_failing_components_no_longer_hardcoded_none(self):
        """★ P2：`failing_components` 此前恒为 `None`（死字段）。

        现在只在**真有降级**时给出是哪一家失败的，且来自
        `llm_provider_errors`（**已发生的事实**，零成本），
        不做每请求深探测。
        """
        src = self._core_src()
        assert '"failing_components": None,' not in src, (
            "不该再硬编码为 None —— 那是个从不被读取的死字段"
        )
        assert "failing_components" in src

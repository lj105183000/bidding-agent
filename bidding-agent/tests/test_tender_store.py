"""招标文件持久化与截止监控单测。

只测**不依赖数据库**的纯逻辑：紧急度分档、日期解析、覆盖率文案。
真实的建表/读写/查询由端到端脚本覆盖（需要连库）。

重点守住：
    - `expired` 必须独立于 `urgent`——"还有 1 天"与"已截止"要采取完全不同的行动
    - 日期解析失败要返回 None 而不是抛错（一份时间格式异常的文件不该让整条落库失败）
    - 覆盖率文案必须说明"不是全网监控"（否则用户以为系统漏报）
"""

from datetime import datetime, timedelta, timezone

import pytest

from src.services import tender_store as TS


# ---------------------------------------------------------------------------
# 紧急度分档
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "days,expected",
    [
        (-10.0, "expired"),
        (-0.1, "expired"),
        (0.0, "urgent"),  # 刚好到期算紧急，不算已过期
        (0.5, "urgent"),
        (3.0, "urgent"),  # 边界含在 urgent
        (3.1, "soon"),
        (7.0, "soon"),  # 边界含在 soon
        (7.1, "normal"),
        (30.0, "normal"),  # 边界含在 normal
        (30.1, "normal"),  # 超出所有分档仍归 normal（不是 unknown）
        (365.0, "normal"),
    ],
)
def test_classify_urgency_bands(days, expected):
    assert TS.classify_urgency(days) == expected


def test_expired_is_distinct_from_urgent():
    """★ 核心：已截止不能并进紧急。

    "还有 1 天"（赶紧编标书）与"已经截止"（归档或放弃）需要完全不同的行动，
    混在一档会误事。
    """
    assert TS.classify_urgency(-1) == "expired"
    assert TS.classify_urgency(1) == "urgent"
    assert TS.classify_urgency(-1) != TS.classify_urgency(1)


def test_classify_urgency_none_is_unknown():
    """没有截止时间不是"已过期"，而是"未知"——两者含义完全不同。"""
    assert TS.classify_urgency(None) == "unknown"


def test_urgency_bands_sorted_ascending():
    """分档阈值必须升序，否则后面的档永远命中不到。"""
    thresholds = [t for _, t in TS.URGENCY_BANDS]
    assert thresholds == sorted(thresholds)


# ---------------------------------------------------------------------------
# 日期解析
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    ["2026-11-20 09:30:00", "2026-11-20 09:30", "2026-11-20", "2026-11-20T09:30:00"],
)
def test_parse_dt_accepts_common_formats(value):
    got = TS._parse_dt(value)
    assert isinstance(got, datetime)
    assert got.year == 2026 and got.month == 11 and got.day == 20


def test_parse_dt_preserves_time():
    """时分不能丢——投标截止时间差几小时意味着废标。"""
    got = TS._parse_dt("2026-11-20 09:30:00")
    assert got.hour == 9 and got.minute == 30


@pytest.mark.parametrize("value", ["", None, "另行通知", "   ", "2026年11月20日"])
def test_parse_dt_returns_none_on_unparseable(value):
    """解析失败必须返回 None 而不是抛错。

    一份时间格式异常的文件不该让整条记录落库失败——其余字段仍有价值。
    """
    assert TS._parse_dt(value) is None


def test_parse_dt_naive_string_gets_timezone():
    """抽取出的时间是北京时间，落库应带时区，否则与 now() 比较会差 8 小时。"""
    got = TS._parse_dt("2026-11-20 09:30:00")
    assert got.tzinfo is not None


# ---------------------------------------------------------------------------
# 覆盖率文案
# ---------------------------------------------------------------------------


def test_coverage_note_states_not_global():
    """★ 必须说明"不是全网监控"——否则用户以为系统漏报了项目。"""
    note = TS._coverage_note(total=10, with_deadline=8)
    assert "不是全网项目监控" in note


def test_coverage_note_mentions_missing_count():
    note = TS._coverage_note(total=10, with_deadline=7)
    assert "3 份" in note


def test_coverage_note_when_empty_library():
    note = TS._coverage_note(total=0, with_deadline=0)
    assert "还没有任何招标文件" in note


def test_coverage_note_no_missing_mention_when_all_have_deadline():
    note = TS._coverage_note(total=5, with_deadline=5)
    assert "另有" not in note


# ---------------------------------------------------------------------------
# 契约
# ---------------------------------------------------------------------------


def test_store_error_is_runtime_error():
    assert issubclass(TS.TenderStoreError, RuntimeError)


def test_table_name_is_stable():
    """表名被写进 DDL 与查询，改动会破坏已有数据的可读性。"""
    assert TS.TABLE == "tender_documents"


def test_urgency_labels_cover_all_classify_outputs():
    """`classify_urgency` 的所有返回值都必须在分档表里有对应档位。"""
    labels = {label for label, _ in TS.URGENCY_BANDS}
    labels.add("expired")
    labels.add("unknown")
    for days in (-5, 1, 5, 20, 400, None):
        assert TS.classify_urgency(days) in labels


def test_dsn_requires_pg_host(monkeypatch):
    monkeypatch.setenv("POSTGRES_HOST", "")
    with pytest.raises(TS.TenderStoreError) as ei:
        TS._dsn()
    assert "PostgreSQL" in str(ei.value)


def test_dsn_strips_trailing_cr(monkeypatch):
    """Windows 下编辑 .env 常带入 \\r，会让认证失败。"""
    monkeypatch.setenv("POSTGRES_HOST", "127.0.0.1")
    monkeypatch.setenv("POSTGRES_PASSWORD", "secret\r\n")
    assert TS._dsn()["password"] == "secret"


def test_save_document_rejects_invalid_payload():
    """缺少 parsed 字段应在**连库之前**就拒绝。"""
    with pytest.raises(TS.TenderStoreError) as ei:
        TS.save_document({"meta": {}})
    assert "parsed" in str(ei.value)


def test_save_document_rejects_non_dict():
    with pytest.raises(TS.TenderStoreError):
        TS.save_document("not a dict")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 与 tender_schema 的衔接（防止两处字段名漂移）
# ---------------------------------------------------------------------------


def test_deadline_field_name_matches_schema():
    """本模块把 `bid_deadline` 单独成列，字段名必须与 schema 一致。

    若 schema 改了名而这里没跟，抽取结果里的截止时间会静默丢失，
    截止监控直接失效且不报错——这是最难查的一类问题。
    """
    from src.constants import tender_schema as TSCH

    assert "bid_deadline" in TSCH.field_keys()
    assert "budget" in TSCH.field_keys()
    assert "max_price" in TSCH.field_keys()


def test_columns_cover_deadline_and_money_fields():
    """DDL 里必须真的有这几列——它们是被单独提出来做查询的。"""
    import inspect

    src = inspect.getsource(TS.ensure_schema)
    for col in ("bid_deadline", "open_time", "budget", "max_price", "deposit", "project_id"):
        assert col in src, f"DDL 缺少列 {col}"


def test_deadline_has_index():
    """截止时间必须有索引，否则"未来 N 天截止"的查询会全表扫描。

    断言的是**结构**（建了 deadline 索引、且索引名由表名派生）而不是字面量
    索引名——DDL 用的是 f-string，源码里并不存在拼好的名字。
    """
    import inspect

    src = inspect.getsource(TS.ensure_schema)
    assert "CREATE INDEX IF NOT EXISTS idx_{TABLE}_deadline" in src
    assert f"idx_{TS.TABLE}_deadline" == "idx_tender_documents_deadline"


def test_parsed_stored_as_jsonb():
    """明细必须存 JSONB 而非 TEXT。

    存成 JSON 字符串后想按章节/条目过滤只能全表 LIKE；
    JSONB 至少能直接用 ->> 取字段。
    """
    import inspect

    src = inspect.getsource(TS.ensure_schema)
    assert "parsed            JSONB" in src or "parsed JSONB" in src


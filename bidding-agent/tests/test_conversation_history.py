"""对话历史持久化单测：搜索 / 增强列表 / 重命名 / 片段提取。

只测**纯逻辑**（`_snippet`）与**需要真库的集成**（搜索、列表、重命名）。
集成测试在库不可用时自动 skip，不阻塞 CI。

重点守住三处**实测踩过的坑**：
    - `PostgreSQL 全文检索对中文不可用` —— 故必须用 ILIKE（本文件不断言实现细节，
      而是断言"中文短词搜得到"这个**行为**）
    - `_like_param` 已自带两侧 `%`，再手工包一层会让搜索变成字面量匹配
    - 空标题必须被拒绝（会话没标题后用户认不出是哪次对话）
"""

from __future__ import annotations

import uuid

import pytest

from src.database.postgresql_client import _like_param, _snippet


# ===========================================================================
# _snippet：命中片段提取（纯逻辑）
# ===========================================================================


def test_snippet_centers_on_keyword():
    """★ 片段必须包含关键词，否则用户不知道"为什么这条被搜出来"。"""
    content = "前言" * 50 + "投标保证金不得超过预算金额的2%" + "后记" * 50
    out = _snippet(content, "保证金", width=40)
    assert "保证金" in out
    assert out.startswith("…") and out.endswith("…")


def test_snippet_at_start_has_no_leading_ellipsis():
    out = _snippet("保证金退还时限是5日", "保证金", width=40)
    assert "保证金" in out
    assert not out.startswith("…")


def test_snippet_at_end_has_no_trailing_ellipsis():
    out = _snippet("投标保证金", "保证金", width=40)
    assert "保证金" in out
    assert not out.endswith("…")


def test_snippet_falls_back_to_head_when_keyword_missing():
    """关键词找不到时退回开头截断，**不能返回空串**。

    返回空会让用户以为数据有问题；而理论上不该发生（SQL 已用 ILIKE 过滤过），
    兜底也要给出内容。
    """
    out = _snippet("这是一段完全不含关键词的文本", "不存在的词", width=10)
    assert out
    assert len(out) <= 11  # width + 省略号


def test_snippet_handles_none_and_empty():
    assert _snippet(None, "x") == ""
    assert _snippet("", "x") == ""


def test_snippet_collapses_whitespace():
    """消息里的换行/多余空白要压掉，否则列表里排版会乱。"""
    out = _snippet("第一行\n\n第二行   第三行", "第二行")
    assert "\n" not in out
    assert "  " not in out


def test_snippet_handles_non_string_content():
    """content 可能是 JSONB 反序列化出的非字符串（防御性）。"""
    assert isinstance(_snippet(12345, "1"), str)
    assert isinstance(_snippet({"a": 1}, "a"), str)


# ===========================================================================
# _like_param：通配符转义（纯逻辑）
# ===========================================================================


def test_like_param_adds_wildcards():
    """★ 它**自带**两侧 %，调用方不要再包一层（实测踩过这个坑）。"""
    assert _like_param("香港") == "%香港%"


def test_like_param_escapes_percent():
    """用户搜 `100%` 时原样的 % 会变成"匹配全表"。"""
    out = _like_param("100%")
    assert "\\%" in out
    assert out == "%100\\%%"


def test_like_param_escapes_underscore():
    assert "\\_" in _like_param("A_1")


def test_like_param_escapes_backslash_first():
    """反斜杠必须最先转义——它是转义符本身，放后面会把刚加的 \\% 再转一遍。"""
    out = _like_param("a\\b")
    assert out == "%a\\\\b%"


def test_like_param_wrapping_twice_breaks_search():
    """记录这个坑：再包一层会把通配符转义成字面量。

    这不是"期望行为"，而是**回归哨兵**——若有人把 `_like_param` 改成
    "不补两侧 %"，这个测试会失败，提醒他检查调用方是否也跟着改了。
    """
    doubled = _like_param(_like_param("香港"))
    assert doubled != "%香港%"
    assert "\\%" in doubled  # 通配符被转义了 → 搜索会变成字面量匹配


# ===========================================================================
# 集成测试（需要真库）
# ===========================================================================


@pytest.fixture(scope="module")
def pg():
    """真库客户端；连不上则 skip 整个模块的集成测试。"""
    try:
        from src.database.postgresql_client import PostgresClient

        client = PostgresClient()
        health = client.health()
        if not health.get("ok"):
            pytest.skip(f"PostgreSQL 不可用：{health.get('error')}")
        client.ensure_schema()
        return client
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"PostgreSQL 不可用：{type(exc).__name__}: {exc}")


@pytest.fixture
def seeded(pg):
    """造一个带独特关键词的会话，用完删掉。

    关键词带 uuid 前缀，避免与真实数据碰撞，也避免测试之间互相干扰。
    """
    marker = f"zqtest{uuid.uuid4().hex[:8]}"
    sid = f"sess-{marker}"
    cid = pg.create_conversation(title=f"测试会话{marker}", session_id=sid)
    if cid < 0:
        pytest.skip("建会话失败")
    pg.save_message(cid, "user", f"这是一条包含{marker}关键词的用户消息", None, None)
    pg.save_message(cid, "assistant", f"回答里也提到{marker}与投标保证金", None, "search_web")
    yield {"marker": marker, "sid": sid, "cid": cid}
    pg.delete_conversation(cid)


def test_search_finds_chinese_keyword(pg, seeded):
    """★ 中文关键词必须搜得到——这是选 ILIKE 而非全文检索的原因。

    PostgreSQL 的 `to_tsvector('simple', '中国大陆和香港的招投标法规')`
    会把整句当成 1 个 token（按空格分词，中文没有空格），
    搜「香港」的 `@@` 匹配是 False。本测试守住"中文短词能搜到"这个行为。
    """
    hits = pg.search_conversations(seeded["marker"], limit=10)
    assert hits, "中文关键词搜不到结果"
    assert any(h["id"] == seeded["cid"] for h in hits)


def test_search_reports_matched_in(pg, seeded):
    """必须告知命中位置——只给标题的话用户看不出为什么这条被搜出来。"""
    hits = pg.search_conversations(seeded["marker"], limit=10)
    target = next(h for h in hits if h["id"] == seeded["cid"])
    # marker 同时出现在标题与消息里
    assert target["matched_in"] in ("title", "message", "both")


def test_search_snippet_contains_keyword(pg, seeded):
    hits = pg.search_conversations(seeded["marker"], limit=10)
    target = next(h for h in hits if h["id"] == seeded["cid"])
    assert seeded["marker"] in target["snippet"], "片段里没有关键词，用户看不出命中原因"


def test_search_returns_message_count(pg, seeded):
    hits = pg.search_conversations(seeded["marker"], limit=10)
    target = next(h for h in hits if h["id"] == seeded["cid"])
    assert target["message_count"] == 2


def test_search_message_only_keyword(pg, seeded):
    """只出现在消息里（不在标题）的关键词也要能搜到，且标为 message 命中。"""
    hits = pg.search_conversations("投标保证金", limit=30)
    target = next((h for h in hits if h["id"] == seeded["cid"]), None)
    assert target is not None, "消息里的关键词没搜到"
    assert target["matched_in"] in ("message", "both")


def test_search_empty_keyword_returns_empty(pg):
    assert pg.search_conversations("") == []
    assert pg.search_conversations("   ") == []


def test_search_no_match_returns_empty(pg):
    assert pg.search_conversations("zzz绝不存在zzz", limit=5) == []


def test_search_wildcard_is_literal(pg, seeded):
    """★ `%` 必须按**字面量**搜索，不能匹配全表。

    参数化管注入，管不了通配符语义：原样的 `%` 交给 ILIKE 会变成
    "匹配一切"，返回一堆不相干记录却不报错——比注入更难发现。

    判据不是"命中 0 条"（真实内容里本来就可能有 `%`，如"占投资额10%以上"），
    而是**命中数严格少于会话总数**——那才说明它没退化成通配符。
    """
    total = len(pg.list_conversations_detailed(limit=500))
    hits_pct = pg.search_conversations("%", limit=500)
    hits_underscore = pg.search_conversations("_", limit=500)

    if total == 0:
        pytest.skip("库里没有会话，无法对照")

    assert len(hits_pct) < total, (
        f"「%」匹配了 {len(hits_pct)}/{total} 条会话——通配符未被转义，"
        "搜索会返回一堆不相干结果"
    )
    assert len(hits_underscore) < total, (
        f"「_」匹配了 {len(hits_underscore)}/{total} 条会话——通配符未被转义"
    )


def test_search_wildcard_matches_only_literal_occurrences(pg):
    """搜 `%` 时，命中的每条片段里都必须**真的含** `%`。

    这比"数量少于总数"更精确：它排除了"碰巧少匹配几条但语义仍是通配"的情况。
    """
    hits = pg.search_conversations("%", limit=500)
    for h in hits:
        # 片段是围绕关键词截的，故含关键词时片段里必然有 %
        blob = f"{h['title']} {h['snippet']}"
        assert "%" in blob, f"id={h['id']} 被搜出来但标题与片段里都没有 %，说明是通配匹配"


def test_search_limit_respected(pg, seeded):
    hits = pg.search_conversations(seeded["marker"], limit=1)
    assert len(hits) <= 1


# ---------------------------------------------------------------------------
# 增列列表
# ---------------------------------------------------------------------------


def test_detailed_list_has_new_fields(pg, seeded):
    """恢复会话需要消息数与预览——只有 id+title 的话认不出是哪次对话。"""
    items = pg.list_conversations_detailed(limit=50)
    target = next((x for x in items if x["id"] == seeded["cid"]), None)
    assert target is not None
    assert target["message_count"] == 2
    assert target["preview"], "预览为空"
    assert target["last_message_at"], "缺最后消息时间"


def test_detailed_list_preview_uses_user_message(pg, seeded):
    """预览取**首条 user 消息**而不是首条消息。

    首条 assistant 通常是欢迎语，各会话都一样，用它做预览没有区分度。
    """
    items = pg.list_conversations_detailed(limit=50)
    target = next(x for x in items if x["id"] == seeded["cid"])
    assert seeded["marker"] in target["preview"]


def test_detailed_list_empty_conversation(pg):
    """空会话（点了新建但没提问）应能被识别出来。"""
    sid = f"sess-empty-{uuid.uuid4().hex[:8]}"
    cid = pg.create_conversation(title="空会话测试", session_id=sid)
    try:
        items = pg.list_conversations_detailed(limit=100)
        target = next((x for x in items if x["id"] == cid), None)
        assert target is not None
        assert target["message_count"] == 0
        assert target["preview"] == ""
    finally:
        pg.delete_conversation(cid)


# ---------------------------------------------------------------------------
# 重命名
# ---------------------------------------------------------------------------


def test_rename_conversation(pg, seeded):
    new_title = f"重命名后{seeded['marker']}"
    assert pg.rename_conversation(seeded["cid"], new_title) is True
    items = pg.list_conversations_detailed(limit=100)
    target = next(x for x in items if x["id"] == seeded["cid"])
    assert target["title"] == new_title


@pytest.mark.parametrize("bad", ["", "   ", "\n\t"])
def test_rename_rejects_blank_title(pg, seeded, bad):
    """★ 空标题必须拒绝：会话没标题后列表里只剩 id，用户再也认不出是哪次对话。"""
    assert pg.rename_conversation(seeded["cid"], bad) is False


def test_rename_nonexistent_returns_false(pg):
    assert pg.rename_conversation(999_999_999, "x") is False


def test_rename_truncates_long_title(pg, seeded):
    ok = pg.rename_conversation(seeded["cid"], "长" * 300)
    assert ok is True
    items = pg.list_conversations_detailed(limit=100)
    target = next(x for x in items if x["id"] == seeded["cid"])
    assert len(target["title"]) <= 120


# ---------------------------------------------------------------------------
# 契约
# ---------------------------------------------------------------------------


def test_search_result_shape(pg, seeded):
    hits = pg.search_conversations(seeded["marker"], limit=5)
    assert hits
    for key in (
        "id", "session_id", "title", "created_at",
        "matched_in", "snippet", "message_count", "relevance",
    ):
        assert key in hits[0], f"搜索结果缺字段 {key}"


def test_detailed_list_shape(pg):
    items = pg.list_conversations_detailed(limit=5)
    if not items:
        pytest.skip("库里没有会话")
    for key in ("id", "session_id", "title", "created_at", "message_count", "preview", "last_message_at"):
        assert key in items[0], f"列表缺字段 {key}"

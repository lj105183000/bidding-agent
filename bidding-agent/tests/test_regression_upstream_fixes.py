"""上游 bug 修复的回归测试。

这些 bug 由并行的子代理在集成测试中发现并报告，本文件把修复钉死，
防止以后再犯。每一条都对应一个**真实后果**，不是理论风险。

文件按"危害程度"排序：
    P0  document_parser 静默丢正文（整章丢失 → 抽取结果为空 → 下游误报变更）
    P1  金额归一化拼出荒谬数字（55 亿 → 污染所有金额统计）
    P2  日期/分值归一化两模块不一致（跨模块比对误报"变更"）
"""

import pytest

from src.constants import bid_schema as BS
from src.constants import tender_schema as TS
from src.services import document_parser as DP


# ===========================================================================
# P0：`split_into_chunks` 曾静默丢掉"（一）（二）…"编号的整段内容
# ===========================================================================
#
# 原实现把 `（一）投标文件未按要求密封的。` 这类**正文条目**当成章节标题，
# 而连续多行都是这种条目时，每次 flush() 的 body 都是空的——
# 这些行既没进正文、也没被保留成标题，**整段消失**。
#
# 实锤后果：招标文件若用"（一）"编号废标条款，抽取出的条目数为 0
# （用"1."编号时是 8 条），LLM 根本没看到这些条款；下游变更追踪还会把
# 已有的 8 条报成"删除"、把真实新增漏掉。
#
# 招标文件里"（一）"编号在废标条款/资格要求/评分项中极常见，属"整章丢失"级别。


@pytest.mark.parametrize(
    "markers",
    [
        ("（一）", "（二）", "（三）"),
        ("一、", "二、", "三、"),
        ("1.", "2.", "3."),
        ("(1)", "(2)", "(3)"),
    ],
)
def test_numbered_items_are_not_swallowed(markers):
    """每种常见编号样式的条目内容都必须保留在块正文里。"""
    a, b, c = markers
    text = (
        "第六章 废标条款\n"
        "投标文件出现下列情形之一的，作废标处理：\n"
        f"{a}投标文件未按要求密封的。\n"
        f"{b}未在规定时间内送达的。\n"
        f"{c}未按要求提交投标保证金的。"
    )
    chunks = DP.split_into_chunks(text)
    body = "\n".join(ch.text for ch in chunks)
    for kw in ("密封", "送达", "保证金"):
        assert kw in body, f"编号样式 {markers} 下「{kw}」丢失；body={body!r}"


def test_sentence_ending_markers_are_not_headings():
    """以句末标点结尾的编号行是**正文**，不是章节标题。"""
    is_head, _ = DP._is_heading("（一）投标文件未按要求密封的。")
    assert is_head is False
    is_head2, _ = DP._is_heading("一、本项目采用综合评分法。")
    assert is_head2 is False


def test_real_weak_headings_still_detected():
    """不能因为修 bug 就把真正的弱标题也判成正文。"""
    for line in ("（一）总则", "一、项目概况", "1.1 项目背景", "(2) 投标人资格"):
        is_head, _ = DP._is_heading(line)
        assert is_head is True, f"{line!r} 应被识别为标题"


def test_strong_headings_always_detected():
    """强标题（第X章/附件）不受弱标题约束影响。"""
    for line in ("第六章 废标条款。", "附件一：投标文件格式", "第 3 节 评标办法"):
        is_head, kind = DP._is_heading(line)
        assert is_head is True, f"{line!r} 应被识别为标题"


def test_heading_text_preserved_in_body():
    """兜底保护：标题行也要出现在块正文里。

    这是"即使标题误判也不丢内容"的第二道防线。漏判标题只是章节归属不准，
    而丢掉正文是**内容损失**，后者严重得多。
    """
    text = "第六章 废标条款\n未密封的作废标处理。"
    chunks = DP.split_into_chunks(text)
    assert chunks
    assert "第六章 废标条款" in chunks[0].text
    assert "未密封" in chunks[0].text


def test_no_content_loss_on_heading_dense_text():
    """连续标题行（无正文）不应导致内容消失。"""
    text = "一、总则\n二、资格要求\n三、评标办法"
    chunks = DP.split_into_chunks(text)
    body = "\n".join(ch.text for ch in chunks)
    for kw in ("总则", "资格要求", "评标办法"):
        assert kw in body, f"{kw} 丢失"


# ===========================================================================
# P1：金额归一化曾把多个数字拼成一个荒谬金额
# ===========================================================================
#
# 原实现"删掉所有非数字字符再拼接"，于是：
#     '25000元（含税13%）'           → 2500013
#     '5 万元（保证金 50,000.00 元）' → 5,500,000,000（55 亿）
#     '预算 120 万，限价 108 万'      → 1,201,080,000
# 公告里"金额 + 税率 + 限价 + 保证金"同句出现极常见，这些数字会直接进库。


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("25000元（含税13%）", 25000),
        ("5 万元（保证金 50,000.00 元）", 50000),
        ("预算 120 万，限价 108 万", 1200000),
        ("保证金人民币贰万元整 20000 元", 20000),
        ("36.65万元（含税）", 366500),
        ("人民币 25,000 元整", 25000),
    ],
)
def test_money_takes_only_first_number(raw, expected):
    """★ 只取第一个数值，不把后续数字拼进来。"""
    assert BS.normalize_record({"中标金额": raw})["amount"] == expected
    assert TS.normalize_tender({"预算金额": raw})["fields"]["budget"] == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("125万元", 1250000),
        ("1,250,000.00 元", 1250000),
        ("1.2亿", 120000000),
        ("1000000", 1000000),
        ("2500万元", 25000000),
    ],
)
def test_money_normal_cases_unchanged(raw, expected):
    """修复不能破坏原本正确的换算。"""
    assert BS.normalize_record({"中标金额": raw})["amount"] == expected
    assert TS.normalize_tender({"预算金额": raw})["fields"]["budget"] == expected


@pytest.mark.parametrize("raw", ["", None, "面议", "详见招标文件"])
def test_money_empty_is_none_not_zero(raw):
    """无法解析必须是 None——0 会被下游当成"金额为零"这种荒谬结论。"""
    assert BS.normalize_record({"中标金额": raw})["amount"] is None
    assert TS.normalize_tender({"预算金额": raw})["fields"]["budget"] is None


def test_money_thousand_separator_not_split():
    """千分位必须整体匹配，否则 '1,250,000' 会变成 1。"""
    assert BS.normalize_record({"中标金额": "1,250,000"})["amount"] == 1250000


def test_money_unit_applies_to_nearest_number():
    """单位要作用到**就近**的数值上，不能把句尾另一个金额的单位套上来。"""
    # 5 万元是主金额；后面 50,000 元是保证金，不带"万"
    got = TS.normalize_tender({"预算金额": "5 万元（保证金 50,000.00 元）"})["fields"]["budget"]
    assert got == 50000, f"期望 50000，得到 {got}"


# ===========================================================================
# P2：日期归一化两模块不一致（会让跨模块比对误报"变更"）
# ===========================================================================


@pytest.mark.parametrize(
    "raw",
    [
        "2026-1-5",
        "2026-01-05",
        "2026年1月5日",
        "2026-1-5 09:30",
        "2026-11-20 09:30",
        "2026年11月20日 09:30",
        "2026年11月20日 9：30",
        "2026/1/5 14:00",
        "2026年11月",
        "2026-11-20T09:30:00",
        "另行通知",
        "",
    ],
)
def test_datetime_consistent_across_modules(raw):
    """★ 两个 schema 模块对同一输入必须给出**完全相同**的结果。

    它们处理同一批招投标文本；结果不一致会让变更追踪把"格式差异"
    误报成"内容变更"，用户会白跑一趟核对。
    """
    a = BS.normalize_record({"发布时间": raw})["publish_time"]
    b = TS.normalize_tender({"投标截止时间": raw})["fields"]["bid_deadline"]
    assert a == b, f"{raw!r}: bid={a!r} vs tender={b!r}"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("2026-1-5", "2026-01-05 00:00:00"),
        ("2026-1-5 09:30", "2026-01-05 09:30:00"),
        ("2026/1/5 14:00", "2026-01-05 14:00:00"),
    ],
)
def test_unpadded_iso_dates_normalized(raw, expected):
    """未补零日期要归一——否则同一日期存在两种写法，比对会误报。"""
    assert TS.normalize_tender({"投标截止时间": raw})["fields"]["bid_deadline"] == expected
    assert BS.normalize_record({"发布时间": raw})["publish_time"] == expected


def test_chinese_date_keeps_time_of_day():
    """★ 中文日期的时间部分不能丢——投标截止差几小时意味着废标。"""
    got = TS.normalize_tender({"投标截止时间": "2026年11月20日 09:30"})["fields"]["bid_deadline"]
    assert "09:30" in got


# ===========================================================================
# P2：非纯数字分值被静默丢弃
# ===========================================================================


@pytest.mark.parametrize(
    "raw,expected",
    [("10", 10.0), ("10分", 10.0), ("10 分", 10.0), ("满分10分", 10.0), ("25.5", 25.5)],
)
def test_score_falls_back_to_leading_number(raw, expected):
    """分值常被写成 "10分"，不能因 float() 失败就静默丢成空。

    分值是评分标准里最关键的数值，静默丢失会让用户以为文件没写分值。
    """
    items = TS.normalize_tender(
        {"评分标准": [{"text": "技术方案", "score": raw, "category": "技术"}]}
    )["sections"]["scoring_items"]
    assert items[0]["score"] == expected


@pytest.mark.parametrize("raw", ["", None, "无", "另定"])
def test_score_without_number_is_none(raw):
    """确实没有数字才置 None。"""
    items = TS.normalize_tender(
        {"评分标准": [{"text": "技术方案", "score": raw, "category": "技术"}]}
    )["sections"]["scoring_items"]
    assert items[0]["score"] is None


# ===========================================================================
# analysis.py：空库必崩 + 可疑计数口径不一致
# ===========================================================================


def test_overview_caveats_do_not_crash_on_empty_table():
    """★ 空表 / bid_time 全为 NULL 时不能崩。

    原实现 `f"{t_min:%Y-%m}"` 在 `t_min=None` 且 `months=0`（仍满足 `<= 1`）时
    抛 `TypeError: unsupported format string passed to NoneType.__format__`，
    整个概览接口 500 —— 首页与集采分析页会同时打不开。

    当前库有 8746 行且 bid_time 全非空，所以一直没暴露；
    一旦清库或重新导入到空表就会炸。故这条例行必须有。
    """
    from src.services.analysis import build_overview_caveats

    caveats = build_overview_caveats(t_min=None, months=0, suspicious=0, avg=None, p50=None)
    assert isinstance(caveats, list)
    assert any("暂无时间数据" in c for c in caveats)
    # 不能出现"覆盖 0 个月"这种基于 None 的格式化结果
    assert not any("覆盖 0 个月" in c for c in caveats)


def test_overview_caveats_single_month_mentions_no_trend():
    """只有一个月的覆盖度必须明确提示"无法做趋势分析"。"""
    from datetime import datetime

    from src.services.analysis import build_overview_caveats

    caveats = build_overview_caveats(
        t_min=datetime(2023, 10, 13), months=1, suspicious=0, avg=None, p50=None
    )
    assert any("无法做时间趋势分析" in c for c in caveats)
    assert any("2023-10" in c for c in caveats)


def test_overview_caveats_flags_skewed_distribution():
    """均值远大于中位数时必须提示"以中位数判断典型水平"。"""
    from datetime import datetime

    from src.services.analysis import build_overview_caveats

    # 实例数据：均值 174 万 vs 中位 4836 元（361 倍）
    caveats = build_overview_caveats(
        t_min=datetime(2023, 10, 13), months=1, suspicious=1,
        avg=1_744_620.0, p50=4_836.0,
    )
    assert any("中位数" in c and "均值不具代表性" in c for c in caveats)


def test_suspicious_count_matches_overview_contract():
    """★ `data_quality().suspicious_count` 必须是真值，不受 LIMIT 30 截断。

    原实现用 `len(suspicious)`，而明细 SQL 带 `LIMIT 30`。可疑记录超过 30 条时，
    数据质量面板与总览提示条会显示两个不同的数字——它们在同一页上，对不上就是 bug。
    """
    import inspect

    from src.services import analysis as A

    src = inspect.getsource(A.data_quality)
    assert "suspicious_total" in src, "未单独查真值计数"
    assert '"suspicious_count": suspicious_total' in src, "suspicious_count 未用真值"
    # 明细条数另用独立字段表达，语义不混淆
    assert "suspicious_returned" in src


def test_overview_suspicious_count_uses_count_star():
    """`overview()` 的口径是真值 count(*)——两者必须同为真值才一致。"""
    import inspect

    from src.services import analysis as A

    src = inspect.getsource(A.overview)
    assert "count(*) FROM" in src and "suspicious_sql()" in src


def test_suspicious_rules_expose_relative_factor():
    """相对规则已被弃用但字段保留，需能看出它不再是判据。

    历史原因：曾用"P90 的 500 倍"做判据，但数据跨 6 个数量级，
    统计离群 ≠ 数据错误，故改为「金额 + 品类」关键词规则。
    字段保留是为了前端兼容与可追溯。
    """
    from src.services import analysis as A

    assert A.RELATIVE_OUTLIER_FACTOR == 500
    assert A.HUGE_AMOUNT == 100_000_000
    assert A.NON_HUGE_CATEGORY_KEYWORDS  # 非空
    assert A.HUGE_OK_CATEGORY_KEYWORDS  # 非空


"""集采分析服务单测。

只测**不依赖数据库**的部分：SQL 片段生成、参数校验、常量契约。
真实聚合结果由 `logs/test_analysis.py` 端到端脚本验证（需要连库）。

重点守住的是**踩过的坑**：
    - 可疑判定的阈值必须与真实数据量级匹配（第一版阈值 10 亿 > 数据最大值 6.6 亿，
      规则从不生效）
    - 不能把"统计离群"当成"数据错误"（曾误伤 23.68 亿的真实工程）
    - 维度白名单必须存在（列名直接拼 SQL，是注入面）
"""

import pytest

from src.services import analysis as A


# ---------------------------------------------------------------------------
# 可疑判定规则（本项目最容易出错的地方）
# ---------------------------------------------------------------------------


def test_huge_amount_threshold_is_reachable():
    """阈值必须低于数据集最大值，否则规则永不生效。

    ★ 这是真实踩过的坑：第一版用 `amount >= 10_000_000_000`（100 亿），
    而数据集最大值只有 6.6 亿，结果"可疑记录 0 条"，规则等于没写。
    """
    assert A.HUGE_AMOUNT < 6_600_000_000, "阈值必须小于已知最大脏数据，否则规则不生效"
    assert A.HUGE_AMOUNT == 100_000_000


def test_suspicious_sql_contains_both_rules():
    sql = A.suspicious_sql()
    # 兜底规则：绝对巨额
    assert "10000000000" in sql
    # 品类规则：巨额 + 品类关键词
    assert str(A.HUGE_AMOUNT) in sql
    assert "ILIKE" in sql


def test_suspicious_sql_has_ok_keyword_exemption():
    """工程类关键词必须出现在豁免条件里，否则真实工程会被误伤。"""
    sql = A.suspicious_sql()
    for kw in ("工程", "建设", "施工"):
        assert kw in sql, f"豁免关键词 {kw} 缺失"


def test_suspicious_sql_has_bad_keywords():
    sql = A.suspicious_sql()
    for kw in ("托养", "办公", "保洁"):
        assert kw in sql, f"可疑品类关键词 {kw} 缺失"


def test_suspicious_sql_does_not_use_relative_outlier():
    """相对离群规则已被移除——它会误伤真实大额工程（见模块文档）。

    回归防线：如果哪天有人把 `percentile_cont` 加回可疑判定，这条测试会失败。
    """
    sql = A.suspicious_sql()
    assert "percentile_cont" not in sql, (
        "可疑判定不应使用相对分位数阈值：本数据集"
        "'统计离群' != '数据错误'，会误伤真实工程"
    )


def test_relative_factor_kept_for_documentation():
    """常量保留是为了记录踩坑历史，不应再被 suspicious_sql 引用。"""
    assert A.RELATIVE_OUTLIER_FACTOR == 500


def test_suspicious_sql_supports_alias():
    sql = A.suspicious_sql(col="t.amount", subject_col="t.subject_matter")
    assert "t.amount" in sql
    assert "t.subject_matter" in sql


def test_ok_and_bad_keywords_disjoint():
    """一个关键词不能同时出现在"支持巨额"和"不支持巨额"两个列表里。"""
    overlap = set(A.HUGE_OK_CATEGORY_KEYWORDS) & set(A.NON_HUGE_CATEGORY_KEYWORDS)
    assert not overlap, f"关键词同时出现在两个列表：{overlap}"


# ---------------------------------------------------------------------------
# 维度白名单（注入防线）
# ---------------------------------------------------------------------------


def test_groupable_is_whitelist():
    """分组维度必须走白名单——列名是直接拼进 SQL 的标识符，无法参数化。"""
    assert "subject_matter" in A.GROUPABLE
    assert "purchaser" in A.GROUPABLE
    assert "supplier" in A.GROUPABLE
    # 危险输入不在白名单里
    assert "1=1" not in A.GROUPABLE
    assert "amount; DROP TABLE" not in A.GROUPABLE


@pytest.mark.parametrize("bad", ["", "amount", "id", "1=1", "title"])
def test_group_ranking_rejects_unknown_dimension(bad):
    """非法维度要在**查库之前**就拒绝（否则就是注入面）。"""
    with pytest.raises(A.AnalysisError) as ei:
        A.group_ranking(bad)
    assert "不支持的维度" in str(ei.value)


def test_error_message_lists_available_dimensions():
    with pytest.raises(A.AnalysisError) as ei:
        A.group_ranking("nope")
    msg = str(ei.value)
    assert "subject_matter" in msg


# ---------------------------------------------------------------------------
# 参数校验
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [(0, 1), (-5, 1), (10, 10), (500, A.MAX_LIMIT), ("20", 20), (None, A.DEFAULT_LIMIT)],
)
def test_clamp_limit(raw, expected):
    assert A._clamp_limit(raw) == expected


@pytest.mark.parametrize("bad", ["", "   ", None])
def test_price_profile_requires_subject(bad):
    with pytest.raises(A.AnalysisError):
        A.price_profile(bad or "")


# ---------------------------------------------------------------------------
# 数量级分桶
# ---------------------------------------------------------------------------


def test_magnitude_buckets_are_contiguous():
    """分桶必须首尾相接，不能有缝隙（否则有记录落不进任何桶）。"""
    prev_hi = 0.0
    for label, lo, _hi in A.MAGNITUDE_BUCKETS:
        assert lo == prev_hi, f"桶 {label} 的下界 {lo} 与上一个桶的上界 {prev_hi} 不连续"
        prev_hi = _hi
    assert A.MAGNITUDE_BUCKETS[-1][2] == float("inf"), "最后一个桶必须是开区间"


def test_magnitude_buckets_cover_all_range():
    """数据实测 12.80 ~ 6.6e9，分桶必须覆盖。"""
    labels = [b[0] for b in A.MAGNITUDE_BUCKETS]
    assert labels[0].startswith("<")
    assert "≥1亿" in labels[-1] or "亿" in labels[-1]


def test_magnitude_bucket_labels_unique():
    labels = [b[0] for b in A.MAGNITUDE_BUCKETS]
    assert len(labels) == len(set(labels))


# ---------------------------------------------------------------------------
# 数据库不可用时的行为
# ---------------------------------------------------------------------------


def test_analysis_error_on_missing_pg_config(monkeypatch):
    """未配置 POSTGRES_HOST 时给出明确错误，而不是抛 psycopg 原生异常。"""
    monkeypatch.setenv("POSTGRES_HOST", "")
    with pytest.raises(A.AnalysisError) as ei:
        A._dsn()
    assert "PostgreSQL" in str(ei.value)


def test_dsn_strips_trailing_cr(monkeypatch):
    """密码末尾的 \\r 必须清掉——Windows 下编辑 .env 常见，会导致认证失败。"""
    monkeypatch.setenv("POSTGRES_HOST", "127.0.0.1")
    monkeypatch.setenv("POSTGRES_PASSWORD", "secret\r\n")
    assert A._dsn()["password"] == "secret"


def test_analysis_error_is_runtime_error():
    assert issubclass(A.AnalysisError, RuntimeError)

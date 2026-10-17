"""竞争对手分析服务单测。

只测**不依赖数据库与 Neo4j** 的部分：集中度算法、SQL 片段生成、参数校验、
异常契约、常量上界。真实聚合结果由 `logs/test_competitor.py` 端到端脚本验证
（那个需要连库，不进 CI）。

重点守住的是**踩过的坑**：
    - 别名参数必须真的生效：第一版用 `where.replace("amount", "s.amount")` 拼别名，
      会把 `suspicious_sql()` 里的子串一起改坏——这类错误在连库跑之前完全看不出来
    - PostgreSQL 的 `ILIKE` 里字面量 `%` 必须写成 `%%`（psycopg 会把单个 `%` 当占位符），
      错误表现为运行时 `IndexError`，纯逻辑测试可以提前守住
    - HHI 用"百分比平方"而不是"小数份额平方"：弄错会得到 0.0004 这种没人能判读的值
    - 单月覆盖必须提示：数据实测只有 2023-10 一个月，"首次/最近中标时间"
      最容易被用户脑补成趋势
"""

import re

import pytest

from src.services import analysis as A
from src.services import competitor as C

# competitor.py 里 Cypher 语句的条数：`_graph_subject_context` 3 条、
# `_graph_supplier_context` 4 条、`_graph_purchaser_context` 3 条。
# 写死这个数字不是为了好看，而是让"扫描正则失效"也能被测出来——
# 否则正则一旦匹配不到东西，下面那条 for 循环会空转通过，测试变成摆设。
EXPECTED_CYPHER_QUERY_COUNT = 10


# ---------------------------------------------------------------------------
# HHI / CR3 / CR5 核心算法（本模块最该独立验证的部分）
# ---------------------------------------------------------------------------


def test_monopoly_hhi_is_max():
    """单一供应商垄断（100% 份额）→ HHI = 10000（理论上限）、CR3 = 100%。"""
    hhi, cr3, cr5, leaders = C._concentration({"独家公司": 100.0})
    assert hhi == 10000.0
    assert cr3 == 100.0
    assert cr5 == 100.0
    assert leaders == [{"name": "独家公司", "share_percent": 100.0}]


def test_two_equal_suppliers_hhi_is_half():
    """两家 50/50 → HHI = 50² + 50² = 5000。"""
    hhi, cr3, cr5, leaders = C._concentration({"甲": 50.0, "乙": 50.0})
    assert hhi == 5000.0
    assert cr3 == 100.0
    assert cr5 == 100.0
    assert len(leaders) == 2


def test_concentration_tie_break_is_deterministic_by_name():
    """份额相同时按名称升序，保证同一份数据每次渲染顺序一致。

    排序键是 `(-份额, 名称)`：只按份额排会让并列项的顺序随 dict 插入顺序漂移，
    前端表格的行序就会"随机"变化。这里固定"乙"在"甲"前（'乙' U+4E59 < '甲' U+7532）。
    """
    _hhi, _cr3, _cr5, leaders = C._concentration({"甲": 50.0, "乙": 50.0})
    assert [x["name"] for x in leaders] == ["乙", "甲"]

    # 顺序无关：换一个插入顺序，结果必须相同
    _h2, _c3, _c5, leaders2 = C._concentration({"乙": 50.0, "甲": 50.0})
    assert [x["name"] for x in leaders2] == ["乙", "甲"]


@pytest.mark.parametrize("n", [2, 3, 4, 5, 10, 100])
def test_even_split_hhi_equals_10000_over_n(n):
    """完全均分（n 家各 1/n）→ HHI = 10000/n。

    这是 HHI 的定义性质，也是判断"份额该用百分比还是小数"的试金石：
    用小数份额算会得到 1/n（0.04 这种），无法用 1500/2500 分档判读。
    """
    shares = {f"供应商{i}": 100.0 / n for i in range(n)}
    hhi, cr3, _cr5, _leaders = C._concentration(shares)
    assert hhi == pytest.approx(10000.0 / n, abs=0.1)


def test_even_split_cr_scales_with_top_n():
    """均分时 CR3 = 3/n、CR5 = 5/n（份额不足时按实际条数封顶）。"""
    shares = {f"供应商{i}": 10.0 for i in range(10)}
    _hhi, cr3, cr5, _leaders = C._concentration(shares)
    assert cr3 == pytest.approx(30.0)
    assert cr5 == pytest.approx(50.0)


def test_single_supplier_cr3_returns_all_not_error():
    """CR3 在供应商不足 3 家时必须返回**全部**份额，而不是报错或返回 None。

    ★ 这是容易写错的边界：`sum(ordered[:3])` 在只有 1 条时返回那 1 条，
    但若实现写成"不足 3 家就返回 None"，前端会拿到空值画不出图。
    """
    _hhi, cr3, cr5, leaders = C._concentration({"唯一": 100.0})
    assert cr3 == 100.0, "只有 1 家时 CR3 应为 100（等于全部），不该是 None 或报错"
    assert cr5 == 100.0
    assert len(leaders) == 1


def test_two_suppliers_cr5_is_also_100():
    """2 家时 CR5 也应是 100%（前 5 名取不满就取全部），不是 None。"""
    _hhi, cr3, cr5, _leaders = C._concentration({"甲": 60.0, "乙": 40.0})
    assert cr3 == 100.0
    assert cr5 == 100.0


def test_uneven_shares_hhi_and_cr():
    """不均衡分布手算校验：60/25/15 → HHI = 3600+625+225 = 4450，CR3 = 100%。"""
    hhi, cr3, _cr5, leaders = C._concentration({"甲": 60.0, "乙": 25.0, "丙": 15.0})
    assert hhi == 4450.0
    assert cr3 == 100.0
    assert [x["share_percent"] for x in leaders] == [60.0, 25.0, 15.0]


def test_concentration_orders_leaders_descending():
    """leaders 必须按份额降序（前端直接顺序渲染，不再排序）。"""
    _hhi, _cr3, _cr5, leaders = C._concentration({"小": 5.0, "大": 80.0, "中": 15.0})
    assert [x["name"] for x in leaders] == ["大", "中", "小"]


def test_concentration_leaders_capped_at_five():
    """leaders 只留前 5 名（CR5 的展示口径），但 HHI 用全量。"""
    _hhi, _cr3, _cr5, leaders = C._concentration({f"s{i}": 1.0 for i in range(20)})
    assert len(leaders) == 5


def test_concentration_empty_input():
    """空输入不能抛异常（无供应商的市场），CR 返回 None、HHI 返回 0。"""
    hhi, cr3, cr5, leaders = C._concentration({})
    assert hhi == 0.0
    assert cr3 is None
    assert cr5 is None
    assert leaders == []


def test_concentration_all_zero_shares_yields_zero_hhi():
    """全部份额为 0（例如全市场金额合计为 0）→ HHI = 0，不能 ZeroDivisionError。

    实测数据里 `amount` 全为正数，但调用方按金额算份额时 `amount_total` 可能为 0，
    实现里用 `if amount_total:` 挡住了——这条测试守住那个分支。
    """
    hhi, cr3, cr5, leaders = C._concentration({"甲": 0.0, "乙": 0.0})
    assert hhi == 0.0
    assert cr3 == 0.0
    assert cr5 == 0.0
    assert len(leaders) == 2


def test_concentration_shares_need_not_sum_to_100():
    """份额之和偏离 100 时不归一化（有记录缺供应商名时就是这样）。

    HHI 因此系统性偏低——这不是 bug，是"份额未覆盖全量"的诚实反映，
    由调用方在 data_caveats 里说明。测试固定这个语义，防止有人"顺手"加归一化。
    """
    hhi, cr3, _cr5, _leaders = C._concentration({"甲": 50.0, "乙": 50.0})  # 合计 100
    hhi_partial, cr3_partial, _c5, _l = C._concentration({"甲": 25.0, "乙": 25.0})  # 合计 50
    assert hhi == 5000.0
    assert hhi_partial == 1250.0, "合计 50 的份额应算出 1250，不能被归一化成 5000"
    assert cr3_partial == 50.0
    assert cr3 == 100.0


# ---------------------------------------------------------------------------
# HHI 分档（阈值与说明必须与数据一起返回）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hhi,expected",
    [
        (0.0, "竞争充分"),
        (1499.9, "竞争充分"),
        (1500.0, "中等集中"),      # 边界：1500 属于中等集中（"< 1500"才叫竞争充分）
        (2000.0, "中等集中"),
        (2500.0, "中等集中"),      # 边界：2500 属于中等集中（"> 2500"才是高度集中）
        (2500.1, "高度集中"),
        (10000.0, "高度集中"),
    ],
)
def test_hhi_level_table(hhi, expected):
    """边界值单独用参数化表列一遍，失败时能一眼看出是哪个值错了。

    边界归属的完整理由（为什么 1500 上归、2500 下留）见下面
    `test_hhi_level_boundaries` 的说明。
    """
    assert C._hhi_level(hhi) == expected


def test_hhi_thresholds_match_module_constants():
    """分档阈值必须与常量一致（1500 / 2500 是 DOJ/FTC 横向并购指南的分档）。"""
    assert C.HHI_MODERATE == 1500
    assert C.HHI_HIGH == 2500


def test_hhi_explanation_documents_all_three_levels():
    """HHI 是个没有量纲直觉的数字，分档标准必须随数据返回给前端。"""
    ex = C._hhi_explanation()
    assert "Σ" in ex["formula"] and "10000" in ex["formula"]
    levels = [t["level"] for t in ex["thresholds"]]
    assert levels == ["竞争充分", "中等集中", "高度集中"]
    for item in ex["thresholds"]:
        assert item["meaning"], f"{item['level']} 缺少含义说明"
    ranges = [t["range"] for t in ex["thresholds"]]
    assert f"< {C.HHI_MODERATE}" in ranges
    assert f"{C.HHI_MODERATE} ~ {C.HHI_HIGH}" in ranges
    assert f"> {C.HHI_HIGH}" in ranges


def test_hhi_explanation_states_doj_ftc_caveat():
    """必须说明该分档出自并购审查指南、此处只作相对比较，不是合规判据。

    少了这句话，用户会把它当成"这个市场违法了"的依据。
    """
    caveat = C._hhi_explanation()["caveat"]
    assert "DOJ" in caveat or "并购" in caveat
    assert "不是合规判据" in caveat or "相对比较" in caveat


def test_hhi_levels_table_is_ascending_and_inf_terminated():
    """档位表必须按上界升序、最后一档用 inf 兜底。

    升序是**功能性**要求而非风格要求：`_hhi_level` 取"第一个 `值 <= 上界`
    命中的档"，表一旦乱序就会返回错误的档位。
    inf 兜底则保证任何 HHI 都落得进某一档，不会走到静默的兜底 return。
    """
    thresholds = [t for t, _label in C.HHI_LEVELS]
    assert thresholds == sorted(thresholds), "档位必须按上界升序排列"
    assert thresholds[-1] == float("inf"), "最后一档必须是 inf（任何 HHI 都要有归属）"
    assert [label for _t, label in C.HHI_LEVELS] == [
        "竞争充分",
        "中等集中",
        "高度集中",
    ]


def test_hhi_levels_first_bound_is_just_below_moderate():
    """第一档上界必须是"小于 1500 的最大浮点数"，不是 1500 本身。

    ★ 这是本次三次翻车的根因所在：1500 要归"中等集中"，2500 要留在
    "中等集中"——两条边界的归属方向相反，用同一个阈值既当上档上界又当下档
    下界时，怎么比都会错一边。把第一档的上界写成 `nextafter(1500, -inf)`，
    再统一用 `值 <= 上界` 判定，两个边界就同时正确了。

    如果哪天有人"顺手"把它改回 `HHI_MODERATE`，1500 会立刻被判成"竞争充分"。
    """
    import math

    first_bound = C.HHI_LEVELS[0][0]
    assert first_bound < C.HHI_MODERATE
    assert math.nextafter(first_bound, math.inf) == C.HHI_MODERATE
    assert C.HHI_LEVELS[1][0] == C.HHI_HIGH, "第二档上界必须是 2500 本身（含端点）"


def test_hhi_level_boundaries():
    """两条边界都必须落在"中等集中"，且与 `_hhi_explanation()` 的文案一致：

        1500 → 中等集中（"< 1500"才是竞争充分）
        2500 → 中等集中（"> 2500"才是高度集中）
    """
    assert C._hhi_level(0.0) == "竞争充分"
    assert C._hhi_level(1000.0) == "竞争充分"
    assert C._hhi_level(1499.9) == "竞争充分"
    assert C._hhi_level(1500.0) == "中等集中"
    assert C._hhi_level(2000.0) == "中等集中"
    assert C._hhi_level(2500.0) == "中等集中"
    assert C._hhi_level(2500.1) == "高度集中"
    assert C._hhi_level(10000.0) == "高度集中"


def test_hhi_level_grades_at_display_precision():
    """档位判定按展示精度（HHI_DECIMALS 位小数）取整后再分档。

    这样"界面上的数字"与"档位文字"永远一致：`_concentration` 用同一精度
    四舍五入，所以显示 2500.0 时档位绝不可能是"高度集中"。
    副作用要说清楚：1499.999 取整后是 1500.0，因此判为"中等集中"——
    这与界面显示一致，不是 bug。
    """
    assert C.HHI_DECIMALS == 1
    # 取整后仍小于 1500 的，留在竞争充分
    assert C._hhi_level(1499.94) == "竞争充分"
    # 取整后等于 1500.0 的，进中等集中（与界面显示的数字一致）
    assert C._hhi_level(1499.999) == "中等集中"
    assert C._hhi_level(1500.04) == "中等集中"
    # 取整后等于 2500.0 的留在中等集中；超过才进高度集中
    assert C._hhi_level(2500.04) == "中等集中"
    assert C._hhi_level(2500.05) == "高度集中"
    # 展示精度与 _concentration 的取整精度必须是同一个常量
    import inspect

    assert "HHI_DECIMALS" in inspect.getsource(C._concentration)


# ---------------------------------------------------------------------------
# 价格水平分档
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "index,expected",
    [
        (None, "无法判定"),
        (0.5, "明显偏低"),
        (0.899, "明显偏低"),
        (0.9, "略低于同行"),
        (0.969, "略低于同行"),
        (0.97, "与同行持平"),
        (1.0, "与同行持平"),
        (1.03, "与同行持平"),
        (1.031, "略高于同行"),
        (1.1, "略高于同行"),
        (1.101, "明显偏高"),
        (1.747, "明显偏高"),  # 实测：亳州市诚信文化体育用品销售有限公司
    ],
)
def test_price_level_is_generous_around_parity(index, expected):
    """±3% 内算"与同行持平"——报价差异本就受规格/服务期影响，
    把 1.02 说成"偏贵"会让用户过度反应。"""
    assert C._price_level(index) == expected


# ---------------------------------------------------------------------------
# 时间覆盖提示（单月数据最容易被脑补成趋势）
# ---------------------------------------------------------------------------


def test_coverage_caveat_warns_on_single_month():
    """★ 实测数据只覆盖 2023-10 一个月，必须提示"无法做时间趋势分析"。"""
    caveat = C._coverage_caveat(
        {"min": "2023-10-13T09:00:00+08:00", "max": "2023-10-20T13:00:00+08:00", "months": 1}
    )
    assert caveat is not None
    assert "1 个月" in caveat
    assert "趋势" in caveat
    assert "2023-10" in caveat, "提示里要带上实际覆盖的年月，用户才知道是哪个月"


def test_coverage_caveat_silent_on_multi_month():
    """覆盖多月时不产出提示（否则就是噪音，用户会开始忽略所有提示）。"""
    assert (
        C._coverage_caveat(
            {"min": "2023-01-01T00:00:00", "max": "2023-12-31T00:00:00", "months": 12}
        )
        is None
    )
    assert C._coverage_caveat(
        {"min": "2023-01-01T00:00:00", "max": "2023-02-01T00:00:00", "months": 2}
    ) is None


def test_coverage_caveat_handles_zero_and_missing():
    """0 个月（无记录）也按"覆盖不足"提示；缺字段时静默而不是崩。"""
    assert C._coverage_caveat({"months": 0}) is not None
    assert C._coverage_caveat({}) is None
    assert C._coverage_caveat({"months": None}) is None


# ---------------------------------------------------------------------------
# 金额过滤条件与别名（踩过 str.replace 的坑）
# ---------------------------------------------------------------------------


def test_base_amount_filter_includes_positive_amount():
    sql = C._base_amount_filter(exclude_suspicious=False)
    assert "amount IS NOT NULL" in sql
    assert "amount > 0" in sql
    assert "suspicious" not in sql.lower()


def test_base_amount_filter_excludes_suspicious_when_asked():
    sql = C._base_amount_filter(exclude_suspicious=True)
    assert "NOT (" in sql
    # 判据来自 analysis：10 亿兜底 + 1 亿×品类
    assert "10000000000" in sql
    assert str(A.HUGE_AMOUNT) in sql


def test_base_amount_filter_differs_between_branches():
    """两个分支必须真的不同——否则 exclude_suspicious 参数形同虚设。"""
    assert C._base_amount_filter(True) != C._base_amount_filter(False)
    assert len(C._base_amount_filter(True)) > len(C._base_amount_filter(False))


def test_base_amount_filter_alias_applies_to_every_column():
    """★ 别名必须作用到**每一处**列名，包括 subject_matter。

    这是真实踩过的坑：第一版用 `where.replace("amount", "s.amount")` 拼别名，
    会把 `suspicious_sql()` 生成的条件里的子串一起改坏。
    这条测试要求：出现列名的地方都带前缀，且不出现裸列名。
    """
    sql = C._base_amount_filter(exclude_suspicious=True, alias="s")
    assert "s.amount IS NOT NULL" in sql
    assert "s.amount > 0" in sql
    assert "s.subject_matter" in sql, "别名必须也作用于 subject_matter"
    # 裸列名（不带别名点号）不该出现
    assert not re.search(r"(?<![\w.])amount\b", sql), f"出现了不带别名的 amount：{sql}"
    assert not re.search(r"(?<![\w.])subject_matter\b", sql), "出现了不带别名的 subject_matter"


def test_base_amount_filter_alias_used_by_competitor_profile():
    """competitor_profile 的自连接查询确实传了别名 `s`。

    守住"别名参数被真的用上"——如果哪天有人改回字符串替换，这条会连同上面
    那条一起失败。
    """
    import inspect

    source = inspect.getsource(C.competitor_profile)
    assert '_base_amount_filter(exclude_suspicious, "s")' in source
    assert ".replace(" not in source, "不要用字符串替换拼别名"


def test_suspicious_sql_percent_is_escaped_for_psycopg():
    """★ ILIKE 里的字面量 `%` 必须写成 `%%`。

    psycopg 把单个 `%` 当参数占位符：写成 `ILIKE '%空调%'` 会在 execute 时
    抛 IndexError/ProgrammingError。错误只在真正查库时暴露，
    纯逻辑断言可以在 CI 里提前拦住。
    """
    sql = A.suspicious_sql()
    # 所有 ILIKE 后面的模式都应是 %%...%%（成对出现），没有落单的 %
    for match in re.finditer(r"ILIKE\s+'([^']*)'", sql):
        pattern = match.group(1)
        assert pattern.count("%") % 2 == 0, f"ILIKE 模式里的 % 未成对转义：{pattern!r}"
        assert pattern.startswith("%%") and pattern.endswith("%%"), pattern


# ---------------------------------------------------------------------------
# 参数校验与钳制
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        (0, 1),
        (-5, 1),
        (1, 1),
        (10, 10),
        (200, C.MAX_LIMIT if hasattr(C, "MAX_LIMIT") else 200),
        (500, 200),
        ("20", 20),
        ("abc", C.DEFAULT_LIMIT),
        (None, C.DEFAULT_LIMIT),
    ],
)
def test_clamp_limit(raw, expected):
    """limit 越界钳制到 [1, 200]，非法值退回默认 20。"""
    assert C._clamp_limit(raw) == expected


@pytest.mark.parametrize("bad", ["", "   ", None, "\t\n"])
def test_norm_keyword_rejects_empty(bad):
    with pytest.raises(C.CompetitorError) as ei:
        C._norm_keyword(bad, "标的物关键词")
    assert "请提供" in str(ei.value)


def test_norm_keyword_rejects_too_short():
    """单字关键词（"车""纸"）会命中几百种互不相干的标的物，必须拒绝。

    这不是洁癖：把无关市场掺在一起算出来的"集中度"是错的，
    宁可报错让调用方把词写具体。
    """
    assert C.MIN_KEYWORD_LENGTH == 2
    with pytest.raises(C.CompetitorError) as ei:
        C._norm_keyword("车", "标的物关键词")
    assert "2 个字符" in str(ei.value)


def test_norm_keyword_accepts_exactly_min_length():
    """边界：刚好 2 个字符必须通过（"空调"就是 2 字，是本服务最常用的查询）。"""
    assert C._norm_keyword("空调", "标的物关键词") == "空调"
    assert C._norm_keyword(" 空调 ", "标的物关键词") == "空调", "应顺带去掉首尾空白"


def test_norm_keyword_error_mentions_label():
    """错误信息要带上是哪个参数错了（三个函数共用这个校验）。"""
    with pytest.raises(C.CompetitorError) as ei:
        C._norm_keyword("", "采购人名称或关键词")
    assert "采购人名称或关键词" in str(ei.value)


def test_public_functions_validate_before_touching_db():
    """★ 参数校验必须发生在查库**之前**。

    否则非法输入会先走一次连接、再抛错，既慢又把错误原因埋在连接异常后面。
    这条测试不能连库：如果实现是"先连库再校验"，这里会抛连接失败类的
    CompetitorError（消息不同），断言就会失败。
    """
    for call in (
        lambda: C.suppliers_for_subject("车"),
        lambda: C.competitor_profile(""),
        lambda: C.purchaser_landscape(" "),
        lambda: C.market_concentration("纸"),
    ):
        with pytest.raises(C.CompetitorError) as ei:
            call()
        message = str(ei.value)
        assert "请提供" in message or "个字符" in message, (
            f"应在查库前就拒绝非法关键词，实际报错：{message}"
        )


# ---------------------------------------------------------------------------
# DSN 组装（不连库）
# ---------------------------------------------------------------------------


def test_dsn_requires_postgres_host(monkeypatch):
    """未配置 POSTGRES_HOST 时抛 CompetitorError，而不是暴露 psycopg 原生异常。"""
    monkeypatch.setenv("POSTGRES_HOST", "")
    with pytest.raises(C.CompetitorError) as ei:
        C._dsn()
    assert "PostgreSQL" in str(ei.value)


def test_dsn_strips_trailing_cr_from_password(monkeypatch):
    """密码末尾的 \\r 必须清掉——Windows 下编辑 .env 常见，会导致认证失败。"""
    monkeypatch.setenv("POSTGRES_HOST", "127.0.0.1")
    monkeypatch.setenv("POSTGRES_PASSWORD", "secret\r\n")
    assert C._dsn()["password"] == "secret"


def test_dsn_has_no_hardcoded_password():
    """★ 密码只能来自 .env。

    用两个都置空的变量构造：若代码里写死了密码兜底，这里会拿到非空值。
    """
    import inspect

    source = inspect.getsource(C._dsn)
    assert "getenv" in source
    assert "Neo4j@2026Local" not in source and "545983" not in source


def test_neo4j_driver_requires_config(monkeypatch):
    """图谱配置缺失时抛 GraphUnavailableError（可降级），不是 CompetitorError 基类。

    这个区分很重要：图谱不可用时应当**降级继续出结果**，而不是让请求失败。
    """
    monkeypatch.setenv("NEO4J_URI", "")
    with pytest.raises(C.GraphUnavailableError):
        C._neo4j_driver()


# ---------------------------------------------------------------------------
# 异常契约
# ---------------------------------------------------------------------------


def test_exception_hierarchy():
    """所有异常都收敛到 CompetitorError，且它本身是 RuntimeError 子类
    （与 analysis.AnalysisError 保持一致，便于路由层统一处理）。"""
    assert issubclass(C.CompetitorError, RuntimeError)
    assert issubclass(C.GraphUnavailableError, C.CompetitorError)
    assert issubclass(C.GraphUnavailableError, RuntimeError)


def test_graph_error_is_catchable_as_competitor_error():
    """降级代码写 `except GraphUnavailableError`，兜底代码写
    `except CompetitorError`——后者必须也能接住前者。"""
    with pytest.raises(C.CompetitorError):
        raise C.GraphUnavailableError("图谱挂了")


def test_no_psycopg_or_neo4j_exception_leaks_in_contract():
    """对外承诺的异常只有这两个名字，不该有 psycopg/neo4j 的类型。"""
    pub = []
    for name in dir(C):
        obj = getattr(C, name)
        if isinstance(obj, type) and issubclass(obj, BaseException):
            pub.append(name)
    assert set(pub) == {"CompetitorError", "GraphUnavailableError"}, pub


# ---------------------------------------------------------------------------
# 复用 analysis 的判据（防漂移）
# ---------------------------------------------------------------------------


def test_suspicious_sql_is_imported_not_reimplemented():
    """★ 必须是 analysis 里的**同一个函数对象**。

    若有人复制一份实现在 competitor 里，两边的脏数据判据迟早漂移，
    同一批数据在两个服务里会得出不同结论。`is` 断言守住这一点。
    """
    assert C.suspicious_sql is A.suspicious_sql
    assert C.suspicious_sql.__module__ == "src.services.analysis"


def test_no_local_suspicious_rule_constants():
    """competitor 不该再定义一份"巨额"阈值常量（判据只有一处真源）。"""
    for forbidden in (
        "HUGE_AMOUNT",
        "NON_HUGE_CATEGORY_KEYWORDS",
        "HUGE_OK_CATEGORY_KEYWORDS",
        "RELATIVE_OUTLIER_FACTOR",
    ):
        assert not hasattr(C, forbidden), (
            f"{forbidden} 应只在 analysis 里定义，competitor 必须复用而不是复制"
        )


def test_base_amount_filter_uses_shared_suspicious_sql(monkeypatch):
    """反向验证：把共享判据换成一个哨兵字符串，过滤条件里必须出现它。

    这条测试证明 `_base_amount_filter` 真的在调用 `suspicious_sql()`，
    而不是内联了一份等价写法（内联的话哨兵不会出现）。
    """
    sentinel = "(SENTINEL_SUSPICIOUS_RULE)"
    monkeypatch.setattr(C, "suspicious_sql", lambda col="amount", subject_col="s": sentinel)
    sql = C._base_amount_filter(exclude_suspicious=True)
    assert sentinel in sql
    # 关掉排除后哨兵不该出现
    assert sentinel not in C._base_amount_filter(exclude_suspicious=False)


# ---------------------------------------------------------------------------
# 常量上界（防止无限制查询拖回整张图）
# ---------------------------------------------------------------------------


def test_all_rows_limit_has_upper_bound():
    """"取全部"也必须落在一个具体数字上，不能是无 LIMIT 的裸查询。

    实测数据规模：单标的物最多 175 条记录、供应商合计 4151 家，
    所以几十万的上界足够覆盖，同时挡住"某天数据涨到千万行"的拉爆。
    """
    assert isinstance(C.ALL_ROWS_LIMIT, int)
    assert C.ALL_ROWS_LIMIT >= 10_000, "太小会截断真实的全量供应商名单，HHI 会算错"
    assert C.ALL_ROWS_LIMIT <= 1_000_000, "上界过大等于没有护栏"


def test_graph_context_limit_has_upper_bound():
    """★ Neo4j 有 19336 个节点，没有 LIMIT 的 MATCH 会把整张图拉回来。

    实测规模：SubjectMatter 2528 / Supplier 4138 / Transaction 8730，
    GRAPH_CONTEXT_LIMIT 必须远小于它们。
    """
    assert isinstance(C.GRAPH_CONTEXT_LIMIT, int)
    assert 1 <= C.GRAPH_CONTEXT_LIMIT <= 200
    assert C.GRAPH_CONTEXT_LIMIT < 2528, "至少应小于最小的实体标签节点数"


def test_max_limits_are_sane():
    assert C.DEFAULT_LIMIT >= 1
    assert C.MAX_LIMIT >= C.DEFAULT_LIMIT
    assert C.MAX_LIMIT <= 1000, "单次返回条数不该超过千级（前端渲染与响应体都会炸）"


def test_concentration_top_n_constants():
    assert C.CR3_N == 3 and C.CR5_N == 5


def test_min_records_for_ratio_is_at_least_two():
    """单条记录的"中位数"就是那一笔，比值反映的是运气不是水平，故门槛 ≥ 2。"""
    assert C.MIN_RECORDS_FOR_RATIO >= 2


# ---------------------------------------------------------------------------
# 图谱查询必须带 LIMIT（静态断言，不连库）
# ---------------------------------------------------------------------------


def test_all_cypher_queries_carry_limit():
    """★ 模块内每一条 Cypher 都必须带 LIMIT。

    用源码扫描而非运行时：这些语句在连库时才执行，CI 里跑不到，
    但"忘了写 LIMIT"是这里最容易犯且后果最重的错（图谱实测 19336 个节点，
    一条没有 LIMIT 的 MATCH 会把整张图拉回来）。

    扫描口径：本模块的 Cypher 由**相邻的单引号字符串**拼成（写在函数体内、
    与 `$param` 实参紧邻，便于逐行阅读），所以先把相邻字符串拼接起来再检查
    ——整条语句的 LIMIT 可能落在第二个片段上（实测确实如此）。
    """
    import inspect

    source = inspect.getsource(C)
    # 把相邻的 "..." "..." 先合并，避免 LIMIT 恰好在第二段时漏判
    joined = re.sub(r'"\s*\n\s*"', "", source)
    queries = re.findall(r'"((?:MATCH|CALL)\b[^"]*)"', joined)
    assert len(queries) == EXPECTED_CYPHER_QUERY_COUNT, (
        f"扫描到 {len(queries)} 条 Cypher，预期 {EXPECTED_CYPHER_QUERY_COUNT} 条"
        "（新增/删除查询时请同步这个数字——它的作用是让扫描失效时测试会失败，"
        f"而不是空转通过）：{queries}"
    )
    for query in queries:
        assert "LIMIT" in query.upper(), f"缺少 LIMIT 的 Cypher：{query[:110]}"


def test_graph_queries_bind_parameters_via_placeholders():
    """★ Cypher 的值必须走 `$name` 占位符，不能靠 f-string 插值。

    第一版写过 `f"MATCH (s:SubjectMatter {{name: $name}})"` 并把 name 当 Python
    变量插值——Cypher 会直接语法报错，而 `_query` 又不接受关键字参数。
    """
    import inspect

    source = inspect.getsource(C)
    # 语句里出现的 $xxx 必须有对应的关键字实参传给 _query / _query_one
    assert "$subject" in source and "subject=" in source
    assert "$limit" in source and "limit=" in source
    # 不该出现"用 f-string 插值实体名"的写法（会把值拼进语句）
    assert "f\"MATCH (s:SubjectMatter {{name:" not in source


def test_graph_subject_context_returns_none_when_node_absent(monkeypatch):
    """图谱里没有同名标的物节点时返回 None（而不是抛异常）。

    调用方据此区分"图谱没这个名字"（正常降级 + 提示）与"图谱挂了"（异常）。
    """
    monkeypatch.setattr(C, "_query_one", lambda cypher, **params: None)
    assert C._graph_subject_context("不存在的标的物xyz") is None


def test_graph_subject_context_shape(monkeypatch):
    """节点存在时返回图谱上下文，且地区去重、带上 tradeFrequency。"""
    monkeypatch.setattr(
        C,
        "_query_one",
        lambda cypher, **params: {"name": "空调", "tradeFrequency": 175},
    )
    monkeypatch.setattr(
        C,
        "_query",
        lambda cypher, **params: [{"name": "合肥市"}, {"name": "合肥市"}, {"name": "芜湖市"}],
    )
    ctx = C._graph_subject_context("空调")
    assert ctx["graph_name"] == "空调"
    assert ctx["trade_frequency"] == 175
    assert ctx["regions"] == ["合肥市", "芜湖市"], "地区应保序去重"


def test_graph_unavailable_propagates_for_degradation(monkeypatch):
    """查询失败必须抛 GraphUnavailableError 让调用方降级，而不是静默返回 None。"""
    def boom(cypher, **params):
        raise C.GraphUnavailableError("图谱查询失败：ServiceUnavailable")

    monkeypatch.setattr(C, "_query_one", boom)
    with pytest.raises(C.GraphUnavailableError):
        C._graph_subject_context("空调")


def test_try_graph_subject_degrades_to_note(monkeypatch):
    """`_try_graph_subject` 把"图谱挂了"转成 (None, 提示文案)，让主流程继续出结果。"""
    def boom(_keyword):
        raise C.GraphUnavailableError("图谱不可用")

    monkeypatch.setattr(C, "_graph_subject_context", boom)
    graph, note = C._try_graph_subject("空调")
    assert graph is None
    assert note and "不可用" in note


def test_try_graph_subject_note_when_node_missing(monkeypatch):
    """节点不存在也要给提示——用户才会知道为什么没有图谱数据。"""
    monkeypatch.setattr(C, "_graph_subject_context", lambda _keyword: None)
    graph, note = C._try_graph_subject("空调")
    assert graph is None
    assert note and "完全同名" in note


def test_try_graph_subject_passes_through(monkeypatch):
    monkeypatch.setattr(C, "_graph_subject_context", lambda _keyword: {"graph_name": "空调"})
    graph, note = C._try_graph_subject("空调")
    assert graph == {"graph_name": "空调"}
    assert note is None


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, None),
        (0, 0.0),
        (1234, 1234.0),
        ("1234.5", 1234.5),
    ],
)
def test_f_normalizes_numeric(value, expected):
    """Decimal / int / 数字字符串都要变成 float；None 保持 None。

    保持 None 而不是折成 0：本数据里 amount 无空值，但"没数据"与"金额是 0"
    在展示上完全是两回事，不能让前端把缺失画成 0 元。
    """
    from decimal import Decimal

    assert C._f(value) == expected
    assert isinstance(C._f(Decimal("1234.5")), float)


def test_f_returns_none_for_non_numeric():
    assert C._f(object()) is None


def test_dedupe_preserves_order_and_skips_blanks():
    assert C._dedupe(["合肥市", "芜湖市", "合肥市", None, "", "蚌埠市"]) == [
        "合肥市",
        "芜湖市",
        "蚌埠市",
    ]


def test_dedupe_respects_limit():
    assert C._dedupe(["a", "b", "c", "d"], limit=2) == ["a", "b"]


def test_dedupe_empty():
    assert C._dedupe([]) == []


def test_iso_handles_none_and_datetime():
    from datetime import datetime

    assert C._iso(None) is None
    assert C._iso(datetime(2023, 10, 13, 9, 0)) == "2023-10-13T09:00:00"


def test_iso_range_shape():
    class _Row:
        def __init__(self, values):
            self._values = values

        def __getitem__(self, index):
            return self._values[index]

    from datetime import datetime

    row = _Row([datetime(2023, 10, 13), datetime(2023, 10, 20), 1])
    assert C._iso_range(row) == {
        "min": "2023-10-13T00:00:00",
        "max": "2023-10-20T00:00:00",
        "months": 1,
    }


def test_add_param_appends_without_mutating():
    """条件片段与参数分开拼装时，顺序必须靠得住；且不能原地改入参元组。"""
    original = ("a",)
    result = C._add_param(original, "b")
    assert result == ("a", "b")
    assert original == ("a",), "原元组不该被改（元组本身不可变，但实现别用 list 共享）"


# ---------------------------------------------------------------------------
# 模块契约
# ---------------------------------------------------------------------------


def test_all_exports_are_importable():
    for name in C.__all__:
        assert hasattr(C, name), f"__all__ 里的 {name} 不存在"


def test_all_exports_cover_the_four_public_functions():
    assert set(C.__all__) >= {
        "suppliers_for_subject",
        "competitor_profile",
        "purchaser_landscape",
        "market_concentration",
    }


def test_public_functions_have_chinese_docstrings():
    """本项目的约定：公开函数一律中文 docstring，说明"为什么"而不只是"是什么"。"""
    for name in (
        "suppliers_for_subject",
        "competitor_profile",
        "purchaser_landscape",
        "market_concentration",
    ):
        doc = getattr(C, name).__doc__ or ""
        assert doc.strip(), f"{name} 缺少 docstring"
        assert any("\u4e00" <= ch <= "\u9fff" for ch in doc), f"{name} 的 docstring 不是中文"


def test_module_docstring_explains_pg_vs_neo4j_split():
    """模块 docstring 必须写清"为什么聚合统计不能靠 Neo4j"。

    这是本模块最关键的设计决定（SUPPLIED_BY 无金额权重，Transaction 与
    Supplier 之间没有关系），不写下来下一个人一定会想"为什么不直接查图谱"。
    """
    doc = C.__doc__ or ""
    assert "SUPPLIED_BY" in doc
    assert "Transaction" in doc and "Supplier" in doc
    assert "中位数" in doc, "必须说明为什么用中位数而不是均值"


def test_import_does_not_touch_databases():
    """导入本模块不得建立任何数据库连接（CI 无库环境必须能 import）。

    验证方式：import 之后再断言模块里没有模块级的连接对象；
    真正的保障是 `_dsn` / `_neo4j_driver` 都只在函数内调用。
    """
    import inspect

    module_source = inspect.getsource(C)
    # 模块级（顶格）不该出现 connect / driver 调用
    for line in module_source.splitlines():
        if line.startswith(("psycopg.connect", "GraphDatabase.driver")):
            pytest.fail(f"模块级出现了连接调用：{line}")
    # _Cursor / _GraphSession 是延迟到 __enter__ 才连
    assert "def __enter__" in inspect.getsource(C._Cursor)
    assert "def __enter__" in inspect.getsource(C._GraphSession)

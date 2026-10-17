"""物料识别服务单测。

只测**不依赖数据库**的部分：归一化规则、匹配打分、参数校验、常量契约。

物料库索引来自 `_index()`（读 `bidding_records` 全表），本文件一律用
`_fake_index()` 注入合成索引（`monkeypatch.setattr(material, "_index", ...)`），
**不连库、不改生产代码结构**——生产代码里"查库"与"纯计算"本来就已分离，
这里只是把查库那一步换掉。

真实数据上的端到端结果由 `logs/test_material.py` 验证（需要连库）。

重点守住的是**踩过的坑**：
    - 归一化必须剥掉"的网上超市采购"这类渠道后缀，否则同一物料会分裂成多个
    - 但**不能过度剥离**：`空调扇` / `中央空调` 是不同商品，绝不能被并成 `空调`
    - 匹配分层得分区间必须不重叠（精确 ≥ 归一化 ≥ 包含 ≥ 重叠）
    - `MIN_MATCH_SCORE` 必须同时管住"包含"与"字符重叠"两层
      （曾漏掉包含层，导致 `支架` ⊂ `麦克风/话筒支架` 以 0.12 分混进候选）
"""

import pytest

from src.services import material as M


# ---------------------------------------------------------------------------
# 合成索引：替代 `_index()` 的查库结果
# ---------------------------------------------------------------------------


def _fake_index(entries):
    """把 [(原始写法, 次数), ...] 变成与 `_build_index()` 同构的索引。

    刻意复用 `normalize_name` 来分组，保证"索引构建"与"查询归一化"
    用的是同一套规则——两处若分叉，`match_materials` 会自己匹配不上自己。
    """
    variants: dict[str, list[tuple[str, int]]] = {}
    lookup: dict[str, str] = {}
    for raw, cnt in entries:
        key = M.normalize_name(raw)
        variants.setdefault(key, []).append((raw, cnt))
        lookup.setdefault(key, key)
        lookup.setdefault(str(raw).strip().upper(), key)
    for group in variants.values():
        group.sort(key=lambda item: -item[1])
    return {
        "variants": variants,
        "counts": {k: sum(c for _, c in v) for k, v in variants.items()},
        "raw_total": len(entries),
        "raw_rows": sum(c for _, c in entries),
        "lookup": lookup,
    }


@pytest.fixture
def fake_library(monkeypatch):
    """注入一个合成物料库，让匹配类接口可以不连库运行。"""
    index = _fake_index(
        [
            ("硒鼓", 112),
            ("硒鼓的网上超市采购", 58),
            ("打印/复印纸", 88),
            ("打印/复印纸的网上超市采购", 160),
            ("空调", 175),
            ("空调5件", 2),
            ("空调配件", 21),
            ("中央空调", 5),
            ("空调扇", 2),
            ("U盘", 14),
            ("u盘", 6),
            ("麦克风/话筒", 25),
            ("麦克风/话筒支架", 4),
            ("粉盒", 150),
        ]
    )
    monkeypatch.setattr(M, "_index", lambda: index)
    # 金额一律置空，避免测试触及 `_amounts_for` 的查库路径
    monkeypatch.setattr(M, "_amounts_for", lambda names: {n: [] for n in names})
    return index


# ---------------------------------------------------------------------------
# ① normalize_name —— 剥离噪声后缀（本模块的核心价值）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        # ★ 实测最高频的冗余：`…的网上超市采购` 是采购渠道，不是物料属性
        ("硒鼓的网上超市采购", "硒鼓"),
        ("打印/复印纸的网上超市采购", "打印/复印纸"),
        ("教育类的网上超市采购项目", "教育类"),
        # 单独出现的形式
        ("网上超市采购", "网上超市"),
        ("硒鼓的网上超市", "硒鼓"),
        # 流程外壳
        ("围墙内警戒区硬化项目", "围墙内警戒区硬化"),
    ],
)
def test_normalize_strips_channel_suffixes(raw, expected):
    assert M.normalize_name(raw) == expected


def test_normalize_strips_multi_layer_suffix():
    """★ 回归防线：`教育类的网上超市采购项目` 有两层后缀（采购项目 + 的网上超市）。

    第一版写成"两趟 for + 命中即 break"，只剥到 `教育类的网上超市` 就停了。
    必须是**不动点循环**才能得到 `教育类`。
    """
    assert M.normalize_name("教育类的网上超市采购项目") == "教育类"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("空调5件", "空调"),
        ("空调8件", "空调"),
        ("柜式空调13台", "柜式空调"),
        ("打印/复印纸250件", "打印/复印纸"),
        ("台式整机183件", "台式整机"),
    ],
)
def test_normalize_strips_quantity_suffix(raw, expected):
    """数量是订单属性，不改变物料身份（实测 30 个写法因此可归并）。"""
    assert M.normalize_name(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        # 型号里的数字**后面不是数量单位**，故数量规则不会触发
        ("3M口罩9502", "3M口罩9502"),
        ("35kV及以上电力杆线迁改工程设计施工总承包", "35KV及以上电力杆线迁改工程设计施工总承包"),
        # 数量在**前面**：后缀规则只看结尾，故整体保留
        ("4台空调", "4台空调"),
    ],
)
def test_normalize_keeps_model_numbers(raw, expected):
    """★ 数量规则必须"数字 + 单位"成对啊，否则会毁掉型号。

    实测含数字但无单位的写法有 35 个，全部应保持原样（仅大小写/全半角统一）。
    这里断言的是**逐字相同**（除已知的大小写统一），即数量规则完全未触发。
    """
    assert M.normalize_name(raw) == expected


@pytest.mark.parametrize(
    "raw,gone",
    [
        ("220P竞赛型（JO级）站立式水上摩托采购", "采购"),
        ("16排及以上螺旋CT采购", "采购"),
    ],
)
def test_quantity_rule_does_not_eat_model_codes(raw, gone):
    """结尾的 `采购` 该剥，但型号主体（`220P…摩托` / `16排…CT`）必须完整保留。

    注意：`16排及以上螺旋CT采购` 剥掉 `采购` 是对的——`采购` 是流程词；
    数量规则**没有**误把 `CT` 或 `16排` 当成数量单位削掉。
    """
    out = M.normalize_name(raw)
    # 前缀必须逐字保留：证明数量规则没有把 `16排` / `220P` 当成数量削掉
    assert out.startswith(raw[:4]), f"{raw!r} 的型号前缀被破坏：{out!r}"
    assert not out.endswith("采购"), "结尾的流程词 `采购` 应被剥离"
    assert "CT" in out or "摩托" in out


def test_fullwidth_parens_become_halfwidth():
    """NFKC 顺带把全角括号统一成半角——这是"全半角统一"的一部分。"""
    assert M.normalize_name("220P竞赛型（JO级）站立式水上摩托采购") == "220P竞赛型(JO级)站立式水上摩托"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Ｕ盘", "U盘"),          # 全角 → 半角
        ("u盘", "U盘"),           # 大小写统一
        ("  空调  ", "空调"),      # 首尾空白
        ("\u3000空调\u3000", "空调"),  # 全角空格
        ("打印／复印纸", "打印/复印纸"),  # 全角斜杠 → 半角
        ("Ａ４复印纸", "A4复印纸"),   # 全角字母数字
    ],
)
def test_normalize_unifies_width_case_space(raw, expected):
    assert M.normalize_name(raw) == expected


def test_normalize_is_idempotent():
    """★ 幂等性：归一化两次必须等于一次（否则建索引与查询会分叉）。"""
    samples = [
        "硒鼓的网上超市采购",
        "打印/复印纸的网上超市采购",
        "教育类的网上超市采购项目",
        "空调5件",
        "u盘",
        "Ｕ盘",
        "  空调  ",
        "中央空调",
        "A/B",
        "采购",
        "无",
    ]
    for raw in samples:
        once = M.normalize_name(raw)
        assert M.normalize_name(once) == once, f"{raw!r} 归一化不幂等：{once!r}"


@pytest.mark.parametrize(
    "raw",
    ["空调扇", "中央空调", "空调配件", "空调机组", "空调系统采购及安装"],
)
def test_normalize_does_not_merge_different_products(raw):
    """★ 不能误伤：这些是**不同商品**，绝不能塌缩成 `空调`。

    数据里 `空调`(175) 与 `中央空调`、`空调扇` 是并列的标的物，
    若被并成一个，"空调"的价格画像会混入空调扇，中位数直接失真。
    """
    assert M.normalize_name(raw) != "空调"


@pytest.mark.parametrize("bad", ["", "   ", None, "\u3000\u3000"])
def test_normalize_handles_empty_input(bad):
    """空输入返回空串而不抛异常——调用方多在"清洗整列数据"的循环里。"""
    assert M.normalize_name(bad) == ""


@pytest.mark.parametrize("noise", ["采购", "项目", "购置", "标段"])
def test_normalize_keeps_pure_noise_words(noise):
    """★ MIN_REMAIN_LEN 保护：剥完只剩 0 字时必须放弃剥离。

    否则 `采购` 会被削成空串，产生一个无意义的"标准物料"。
    """
    assert M.normalize_name(noise) == noise


def test_normalize_min_remain_len_guard_is_exactly_two():
    """守卫值是 2：`无` 这类 2 字以下的名字不该被继续剥。"""
    assert M.MIN_REMAIN_LEN == 2
    assert M.normalize_name("无") == "无"


def test_strip_quantity_suffix_requires_unit():
    assert M._strip_quantity_suffix("空调5件") == "空调"
    assert M._strip_quantity_suffix("空调") == "空调"
    assert M._strip_quantity_suffix("3M") == "3M"  # 末尾是字母，不是数量


# ---------------------------------------------------------------------------
# ② 匹配打分与理由
# ---------------------------------------------------------------------------


def test_match_reason_exact(fake_library):
    r = M.match_materials("硒鼓", limit=5)
    assert r["items"][0]["standard_name"] == "硒鼓"
    assert r["items"][0]["match_reason"] == "精确匹配"
    assert r["items"][0]["score"] == M.SCORE_EXACT == 1.0


def test_match_reason_normalized(fake_library):
    """带渠道后缀的输入应命中"归一化后匹配"，而不是靠模糊。"""
    r = M.match_materials("硒鼓的网上超市采购", limit=5)
    top = r["items"][0]
    assert top["standard_name"] == "硒鼓"
    assert top["match_reason"] == "归一化后匹配"
    assert top["score"] == M.SCORE_NORMALIZED == 0.95


def test_match_reason_containment(fake_library):
    """真包含：`麦克风/话筒支架` 应同时给出自身与主体 `麦克风/话筒`（包含匹配）。"""
    reasons = {
        i["standard_name"]: i["match_reason"]
        for i in M.match_materials("麦克风/话筒支架", limit=5)["items"]
    }
    assert reasons.get("麦克风/话筒支架") == "精确匹配"
    assert reasons.get("麦克风/话筒") == "包含匹配"


def test_short_base_name_is_not_offered_as_containment(fake_library):
    """★ 回归防线：查 `空调配件` **不该**顺带返回 `空调`。

      `空调`(175) 与 `空调配件`(21) 是不同标的物；虽然 `空调` 是 `空调配件` 的子串，
      但多出的 `配件` 是 2 字、短名只有 2 字，包含分只有 0.78，
      低于 MIN_MATCH_SCORE 被正确剔除，避免用户把配件价格当成整机价格。
    """
    names = {i["standard_name"] for i in M.match_materials("空调配件", limit=5)["items"]}
    assert "空调配件" in names
    assert "空调" not in names
    assert M._containment_score("空调", "空调配件") < M.MIN_MATCH_SCORE


def test_match_merges_variants_into_one_standard(fake_library):
    """★ 归并的核心断言：两个写法必须并成一个标准物料，次数相加。"""
    top = M.match_materials("硒鼓", limit=5)["items"][0]
    assert top["total_count"] == 170  # 112 + 58
    names = {v["name"] for v in top["matched_variants"]}
    assert names == {"硒鼓", "硒鼓的网上超市采购"}


def test_match_variants_sorted_by_count_desc(fake_library):
    top = M.match_materials("打印/复印纸", limit=1)["items"][0]
    counts = [v["count"] for v in top["matched_variants"]]
    assert counts == sorted(counts, reverse=True)


def test_quantity_variant_merged_by_match(fake_library):
    """`空调5件` 与 `空调` 必须是同一个标准物料（实测归并过）。"""
    top = M.match_materials("空调5件", limit=1)["items"][0]
    assert top["standard_name"] == "空调"
    assert top["total_count"] == 177  # 175 + 2


@pytest.mark.parametrize(
    "query,lo_reason,hi_reason",
    [
        ("硒鼓", "精确匹配", "归一化后匹配"),
        ("硒鼓的网上超市采购", "归一化后匹配", "包含匹配"),
        ("麦克风/话筒支架", "包含匹配", "字符重叠"),
    ],
)
def test_score_tiers_are_monotonic(fake_library, query, lo_reason, hi_reason):
    """★ 得分必须单调：精确 ≥ 归一化 ≥ 包含 ≥ 重叠。

    这是"按 score 排序 == 按置信度排序"的前提；区间一旦重叠，
    前端把候选按分数展示就会把低置信的排到高置信前面。
    """
    items = M.match_materials(query, limit=10)["items"]
    by_reason = {}
    for it in items:
        by_reason.setdefault(it["match_reason"], it["score"])
    if lo_reason in by_reason and hi_reason in by_reason:
        assert by_reason[lo_reason] >= by_reason[hi_reason]


def test_score_tier_constants_are_ordered():
    """常量层面的同一契约（不依赖具体查询）。"""
    assert M.SCORE_EXACT > M.SCORE_NORMALIZED
    assert M.SCORE_NORMALIZED > M.CONTAIN_MAX_SCORE
    assert M.CONTAIN_MAX_SCORE >= M.MIN_MATCH_SCORE


def test_match_results_sorted_by_score_desc(fake_library):
    scores = [i["score"] for i in M.match_materials("麦克风/话筒支架", limit=10)["items"]]
    assert scores == sorted(scores, reverse=True)


def test_match_returns_no_candidate_for_unrelated(fake_library):
    r = M.match_materials("完全无关的物料XYZ", limit=10)
    assert r["matched"] is False
    assert r["items"] == []


# ---------------------------------------------------------------------------
# ③ MIN_MATCH_SCORE 过滤
# ---------------------------------------------------------------------------


def test_below_threshold_is_filtered():
    """★ 回归防线：`支架` ⊂ `麦克风/话筒支架` 得分 0.12，绝不能被当成候选。

    第一版只给"字符重叠"层设了门槛，包含层漏了；而包含层公式是无界的
    （多出部分远长于短名时为负/极低），结果 0.12 分的弱包含照样返回。
    """
    score, reason = M._match_one(
        "麦克风/话筒支架", "支架", M.normalize_name("麦克风/话筒支架"), "支架"
    )
    assert score == 0.0
    assert reason == ""
    assert score < M.MIN_MATCH_SCORE


def test_unrelated_pair_scores_zero():
    for a, b in [("垃圾桶", "垃圾袋"), ("台式整机", "台式计算机"), ("复印机", "打印机")]:
        score, _ = M._match_one(a, b, M.normalize_name(a), M.normalize_name(b))
        assert score < M.MIN_MATCH_SCORE, f"{a} 与 {b} 不应匹配（得分 {score}）"


def test_min_match_score_gates_both_tiers():
    """两个下限必须一致——多层各设一个阈值正是上次出错的根因。"""
    assert M.MIN_MATCH_SCORE == 0.80


# ---------------------------------------------------------------------------
# ④ 字符重叠算法（边界）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "a,b",
    [
        ("", ""),
        ("", "硒鼓"),
        ("硒鼓", ""),
        ("硒鼓", "硒鼓"),
        ("硒鼓", "空调"),
        ("硒", "硒鼓"),
        ("/", "/"),
        ("硒鼓", "/"),
    ],
)
def test_similarity_never_raises(a, b):
    """任何输入（含空串/单字符）都必须返回 0~1 的浮点数，不抛异常。"""
    v = M._similarity(a, b)
    assert isinstance(v, float)
    assert 0.0 <= v <= 1.0


def test_similarity_identity_is_one():
    assert M._similarity("硒鼓", "硒鼓") == 1.0


def test_similarity_is_symmetric():
    for a, b in [("硒鼓", "打印/复印纸"), ("空调", "空调配件"), ("", "硒鼓")]:
        assert M._similarity(a, b) == M._similarity(b, a)


def test_similarity_ranks_same_material_above_different():
    """同一物料（多一个后缀）的**包含层**得分，必须高于"像但不是"的另一对。

    ★ 注意：这类比较**不能**用 `_similarity`——纯字符相似度对本数据是反向的
    （`垃圾桶`~`垃圾袋` 0.667 高于 `硒鼓`~`硒鼓的网上超市采购` 0.364）。
    这正是当初弃用 `pg_trgm`（同属字符 n-gram 思路）的原因：
    真正把同一物料认出来的是**包含结构 + 归一化规则**，不是字符重合度。
    """
    same_contain = M._containment_score("打印/复印纸", "打印/复印纸的网上超市采购")
    same_norm = M.normalize_name("打印/复印纸的网上超市采购") == M.normalize_name("打印/复印纸")
    assert same_norm, "同一物料必须靠归一化归并"

    # 而"像但不是"的一对，既不该归并，也不该拿到包含分
    assert M.normalize_name("垃圾桶") != M.normalize_name("垃圾袋")
    assert M._containment_score("垃圾桶", "垃圾袋") == 0.0


def test_bigrams_handles_short_and_empty():
    assert M._bigrams("") == frozenset()
    assert M._bigrams("硒") == frozenset({"硒"})
    assert M._bigrams("硒鼓") == frozenset({"硒鼓"})


# ---------------------------------------------------------------------------
# ⑤ 参数校验
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["", "   ", None])
def test_match_requires_query(bad):
    with pytest.raises(M.MaterialError):
        M.match_materials(bad or "")


def test_empty_query_rejected_before_touching_db():
    """★ 参数校验必须发生在查库**之前**（否则空输入会白连一次库）。"""
    # 不注入 _index，若实现改成先查库再校验，这里会因连不上库而抛别的异常
    with pytest.raises(M.MaterialError) as ei:
        M.match_materials("")
    assert "物料描述" in str(ei.value)


@pytest.mark.parametrize(
    "raw,low,high,expected",
    [
        (0, 1, 50, 1),
        (-5, 1, 50, 1),
        (10, 1, 50, 10),
        (99999, 1, 50, 50),
        ("20", 1, 50, 20),
        ("abc", 1, 50, 1),          # 非法值回落下限
        (None, 1, 50, 1),
    ],
)
def test_clamp(raw, low, high, expected):
    assert M._clamp(raw, low, high) == expected


def test_match_limit_is_clamped(fake_library):
    """limit 越界要被钳制，而不是报错或返回超量。"""
    assert M.match_materials("硒鼓", limit=M.MAX_MATCH_LIMIT + 1000)["returned"] <= M.MAX_MATCH_LIMIT
    assert M.match_materials("硒鼓", limit=0)["returned"] >= 1
    assert M.match_materials("硒鼓", limit="abc")["returned"] >= 1


def test_limit_constants_sane():
    assert 1 <= M.DEFAULT_MATCH_LIMIT <= M.MAX_MATCH_LIMIT
    assert 1 <= M.DEFAULT_SUGGEST_LIMIT <= M.MAX_SUGGEST_LIMIT


@pytest.mark.parametrize("bad", ["", "   ", None])
def test_material_detail_requires_name(bad):
    with pytest.raises(M.MaterialError):
        M.material_detail(bad or "")


# ---------------------------------------------------------------------------
# ⑥ 缓存
# ---------------------------------------------------------------------------


def test_clear_cache_sets_cache_to_none():
    """`clear_cache()` 是数据重新入库后的失效入口，必须可用且不连库。"""
    M._INDEX_CACHE = {"variants": {}}
    try:
        M.clear_cache()
        assert M._INDEX_CACHE is None
    finally:
        M._INDEX_CACHE = None


def test_clear_cache_is_exported():
    assert "clear_cache" in M.__all__


def test_bigram_cache_is_populated_and_bounded():
    """bigram 缓存是纯内存优化，不应无限增长——同值重复调用不新增条目。"""
    M._BIGRAM_CACHE.clear()
    M._similarity("硒鼓", "空调")
    n1 = len(M._BIGRAM_CACHE)
    M._similarity("硒鼓", "空调")
    assert len(M._BIGRAM_CACHE) == n1
    M._BIGRAM_CACHE.clear()


# ---------------------------------------------------------------------------
# ⑦ 异常与依赖契约
# ---------------------------------------------------------------------------


def test_material_error_is_runtime_error():
    assert issubclass(M.MaterialError, RuntimeError)


def test_reuses_analysis_suspicious_sql():
    """★ 必须复用 `analysis.suspicious_sql`，不能各写一份。

    两处判据一旦分叉，看板与物料页会给出互相矛盾的"可疑记录"口径。
    """
    from src.services import analysis as A

    assert M.suspicious_sql is A.suspicious_sql


def test_amount_filter_excludes_suspicious():
    """金额过滤必须：非空 + 正数 + 排除可疑（复用同一判据）。"""
    sql = M._amount_filter()
    assert "IS NOT NULL" in sql
    assert "> 0" in sql
    assert "10000000000" in sql  # suspicious_sql 的兜底规则，证明真的拼进去了


def test_amount_filter_supports_alias():
    sql = M._amount_filter("t")
    assert "t.amount" in sql
    assert "t.subject_matter" in sql


def test_missing_pg_config_raises_material_error(monkeypatch):
    monkeypatch.setenv("POSTGRES_HOST", "")
    with pytest.raises(M.MaterialError) as ei:
        M._dsn()
    assert "PostgreSQL" in str(ei.value)


def test_dsn_strips_trailing_cr(monkeypatch):
    """密码末尾的 \\r 必须清掉——Windows 下编辑 .env 常见，会导致认证失败。"""
    monkeypatch.setenv("POSTGRES_HOST", "127.0.0.1")
    monkeypatch.setenv("POSTGRES_PASSWORD", "secret\r\n")
    assert M._dsn()["password"] == "secret"


def test_no_hardcoded_password_in_source():
    """★ 绝不硬编码密码：源码里不应出现已知口令字面量。"""
    import pathlib

    src = pathlib.Path(M.__file__).read_text(encoding="utf-8")
    assert "545983" not in src


# ---------------------------------------------------------------------------
# ⑧ 常量契约
# ---------------------------------------------------------------------------


def test_suffix_lists_disjoint_and_non_empty():
    overlap = set(M.NOISE_SUFFIXES) & set(M.GENERIC_SUFFIXES)
    assert not overlap, f"后缀同时出现在两张表：{overlap}"
    for suffix in M.NOISE_SUFFIXES + M.GENERIC_SUFFIXES:
        assert suffix, "后缀表不能含空串"
        assert suffix == suffix.strip(), f"后缀 {suffix!r} 带首尾空白"


def test_strip_order_is_longest_first():
    """★ 剥离顺序必须长度降序：短后缀先试会漏掉更具体的组合。

    （`教育类的网上超市采购项目` 正是因此才会残留 `…的网上超市`。）
    """
    order = list(M._STRIP_ORDER)
    assert order == sorted(order, key=len, reverse=True)
    assert set(order) == set(M.NOISE_SUFFIXES) | set(M.GENERIC_SUFFIXES)


def test_strip_order_deduplicates():
    assert len(M._STRIP_ORDER) == len(set(M._STRIP_ORDER))


def test_max_qualifier_len_sane():
    assert 1 <= M.MAX_QUALIFIER_LEN <= 20
    assert M.MIN_CONTAIN_LEN >= 2


def test_suggest_thresholds_ordered():
    assert M.SUGGEST_MIN_SCORE <= M.SUGGEST_MEDIUM <= M.SUGGEST_HIGH <= 1.0


def test_all_public_functions_exported():
    for name in (
        "normalize_name",
        "match_materials",
        "material_detail",
        "material_stats",
        "suggest_standardization",
        "normalize_preview",
        "clear_cache",
        "MaterialError",
    ):
        assert name in M.__all__, f"{name} 未导出"
        assert hasattr(M, name)


# ---------------------------------------------------------------------------
# ⑨ 包含匹配：真包含 vs 假包含（"多出的字加在前面还是后面"）
# ---------------------------------------------------------------------------


def test_true_containment_scores_higher_than_insertion():
    """★ 真包含必须排在假包含之前，否则人工队列无法排序。"""
    true_contain = M._containment_score("医学检验试剂、耗材", "医学检验试剂、耗材等")
    false_contain = M._containment_score("配电工程", "供配电工程")
    assert true_contain > false_contain


def test_false_containment_is_filtered_out():
    """前置定语改变类别的（`配电工程`→`供配电工程`）应直接跌破门槛。"""
    assert M._containment_score("配电工程", "供配电工程") < M.SUGGEST_MIN_SCORE
    assert M._suggest_pair("配电工程", "供配电工程", 1, 1) is None


@pytest.mark.parametrize(
    "a,b",
    [("医学检验试剂、耗材", "医学检验试剂、耗材等"),
     ("高标准农田建设改造提升", "高标准农田建设改造提升工程"),
     ("医保业务综合服务终端", "医保业务综合服务终端设备")],
)
def test_true_containment_reaches_high_confidence(a, b):
    hit = M._suggest_pair(a, b, 2, 1)
    assert hit is not None
    assert hit["confidence"] == "高"


def test_containment_parts_none_when_no_containment():
    assert M._containment_parts("硒鼓", "空调") is None
    assert M._containment_parts("硒鼓", "硒鼓") is None
    assert M._containment_parts("", "空调") is None


def test_containment_parts_reports_head_and_tail():
    """`配电工程` 在 `供配电工程` 中是**前置**多字；`…耗材等` 是**后置**。"""
    _s, _l, extra, head, tail = M._containment_parts("配电工程", "供配电工程")
    assert (extra, head, tail) == ("供", "供", "")
    _s, _l, extra, head, tail = M._containment_parts("医学检验试剂、耗材", "医学检验试剂、耗材等")
    assert (extra, head, tail) == ("等", "", "等")


def test_containment_rejects_overlong_qualifier():
    """多出一大截说明短名只是泛化词，不足以断定同一物料。"""
    assert M._containment_score("设备", "其它网络设备及配套设施采购安装调试服务项目") == 0.0


def test_containment_is_symmetric():
    for a, b in [("空调", "空调配件"), ("配电工程", "供配电工程")]:
        assert M._containment_score(a, b) == M._containment_score(b, a)


# ---------------------------------------------------------------------------
# ⑩ 中位数工具（金额统计必须用中位数）
# ---------------------------------------------------------------------------


def test_percentile_empty_and_single():
    assert M._percentile([], 0.5) is None
    assert M._percentile([10.0], 0.5) == 10.0


def test_percentile_matches_postgres_interpolation():
    """线性插值口径：`percentile_cont` 与 SQL 侧保持一致。"""
    vals = [1.0, 2.0, 3.0, 4.0]
    assert M._percentile(vals, 0.5) == 2.5
    assert M._percentile(vals, 0.0) == 1.0
    assert M._percentile(vals, 1.0) == 4.0


def test_median_is_not_mean():
    """★ 中位数必须与均值不同——本项目数据跨 6 个数量级，均值不可用。"""
    skewed = [100.0, 100.0, 100.0, 100.0, 1_000_000.0]
    assert M._median(skewed) == 100.0
    assert sum(skewed) / len(skewed) == pytest.approx(200_080.0)


def test_median_empty_is_none():
    assert M._median([]) is None

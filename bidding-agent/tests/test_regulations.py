"""法规语料 + 法规库的单测。

分三层，**只有第一层是必跑的**：

1. **纯数据测试**（绝大多数）：只读 `src.constants.regulations`，不连网、不连
   Qdrant、不加载嵌入模型。CI 与无网络环境可裸跑。
2. **内存 Qdrant 测试**：注入 `QdrantClient(":memory:")` + 假嵌入器，
   验证入库幂等、payload 契约、检索过滤。不碰真实服务、不加载模型。
3. **真实服务测试**（`@_requires_qdrant`）：默认 **skip**，只有显式设了
   `REGULATION_TEST_LIVE=1` 且本地 Qdrant 可达时才跑。这是刻意的——
   入库是部署动作，不该让 `pytest` 在别人机器上因为 Qdrant 没开而变红。

★ 本文件**绝不写入 `bid_qa`**：所有写操作都被指向 `bid_regulations`
（或内存集合），并有专门的守卫测试固化这一约束。
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from src.constants.regulations import (
    ABRIDGED_KEYS,
    ABRIDGED_MARKER,
    LEVEL_ORDER,
    REGULATIONS,
    TOPIC_ABRIDGED,
    RegulationArticle,
    abridged_marker_keys,
    all_articles,
    articles_by_law,
    corpus_stats,
    duplicate_keys,
)
from src.rag.vector_store import VectorStore
from src.services import regulation_store as rs

# 条号格式：第四十三条 / 第43条 / 第四十三条之一 / 第一百零八条
ARTICLE_PATTERN = re.compile(r"^第[0-9一二三四五六七八九十百零两]+条(之[一二三四五六七八九十]+)?$")

# 语料规模底线（低于此值说明语料被误删，或新增法规时漏了合并）
MIN_ARTICLES = 120
MIN_LAWS = 6

# 合规检查最依赖的主题，每个主题至少要有多条支撑，否则检索会"只有唯一答案"
REQUIRED_TOPICS = (
    "保证金",
    "资格条件",
    "差别待遇",
    "评审因素",
    "异议投诉",
    "不合理条件",
    "时限",
    "地域歧视",
)
MIN_ARTICLES_PER_REQUIRED_TOPIC = 3

# 占位符/未完成标记。**不要**把 `……` 加进来——节选条文的裁剪位置用它标记，
# 见 `ABRIDGED_KEYS`（那三条单独校验，不是占位符）。
PLACEHOLDER_MARKERS = ("TODO", "FIXME", "XXX", "待补充", "待定", "（略）", "(略)", "???", "N/A")

DIM = 16


# ---------------------------------------------------------------------
# 夹具：内存 Qdrant + 假嵌入器（不加载任何模型）
# ---------------------------------------------------------------------


class FakeEmbedder:
    """确定性假嵌入器。

    为什么需要它：真嵌入器要加载 bge 模型（数十秒、数百 MB），而这些测试要验证的
    是**存储与检索逻辑**，不是模型效果。用一个由文本哈希决定的单位向量，
    就能让"相同文本 → 相同向量、不同文本 → 不同向量"这一契约成立，
    从而让余弦相似度检索的结果可复现。

    ★ 必须带 `model_name` 属性：向量维度靠它对齐（与 `ensure_collection` 一致）。
    """

    model_name = "fake-embedder"
    dim = DIM

    def _vec(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = [float(b) - 127.5 for b in digest[: self.dim]]
        norm = math.sqrt(sum(v * v for v in raw)) or 1.0
        return [v / norm for v in raw]

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def encode_query(self, text: str) -> list[float]:
        return self._vec(text)


@pytest.fixture()
def memory_store() -> VectorStore:
    return VectorStore(
        url=":memory:",
        collection=rs.REGULATION_COLLECTION,
        dim=DIM,
        client=QdrantClient(":memory:"),
    )


@pytest.fixture()
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture()
def seeded(memory_store: VectorStore, fake_embedder: FakeEmbedder) -> VectorStore:
    """已写入全量内置语料的内存法规库。"""
    rs.upsert_regulations(store=memory_store, embedder=fake_embedder, progress=False)
    return memory_store


def _sample_articles() -> list[RegulationArticle]:
    """跨法规取几条，避免只测到某一部的结构。"""
    grouped = articles_by_law()
    picked: list[RegulationArticle] = []
    for items in grouped.values():
        picked.append(items[0])
    return picked


# =====================================================================
# 一、纯数据测试
# =====================================================================


def test_minimum_scale():
    """规模底线：>=120 条、>=6 部法规。"""
    assert len(REGULATIONS) >= MIN_ARTICLES, f"条文只有 {len(REGULATIONS)} 条"
    assert len({a.law for a in REGULATIONS}) >= MIN_LAWS


def test_every_article_has_required_fields():
    for item in REGULATIONS:
        assert item.law.strip(), item.key
        assert item.law_short.strip(), item.key
        assert item.article.strip(), item.key
        assert len(item.text.strip()) >= 8, f"{item.key} 正文过短，疑似残缺"
        assert item.topics, f"{item.key} 没有主题标签"
        assert item.level in LEVEL_ORDER, f"{item.key} 的层级非法: {item.level}"


def test_topics_are_non_empty_unique_and_trimmed():
    for item in REGULATIONS:
        assert all(t.strip() for t in item.topics), f"{item.key} 有空白主题"
        assert len(set(item.topics)) == len(item.topics), f"{item.key} 主题重复"
        for topic in item.topics:
            assert topic == topic.strip(), f"{item.key} 主题未去空白: {topic!r}"


def test_no_duplicate_law_article():
    """同 law+article 必须唯一——否则入库时会互相覆盖成一条。"""
    assert duplicate_keys() == []


def test_article_number_format_is_legal():
    for item in REGULATIONS:
        assert ARTICLE_PATTERN.match(item.article), f"{item.key} 条号格式异常: {item.article}"


def test_no_placeholder_text():
    """不能出现 TODO / XXX / 待补充 之类的未完成标记。"""
    for item in REGULATIONS:
        for marker in PLACEHOLDER_MARKERS:
            assert marker not in item.text, f"{item.key} 含占位符 {marker!r}"
            assert marker not in item.article, f"{item.key} 条号含占位符 {marker!r}"


def test_original_text_is_not_obviously_truncated():
    """正文不应以逗号/顿号/冒号结尾——那是被截断的典型痕迹。"""
    for item in REGULATIONS:
        tail = item.text.strip()[-1]
        assert tail not in ("，", "、", "：", ",", ";"), f"{item.key} 结尾异常: ...{item.text[-20:]!r}"


def test_abridged_marker_consistency():
    """节选标记与 `ABRIDGED_KEYS` 必须双向一致，防止悄悄截断原文。

    判定"节选"看的是**整行只有 `……`** 的裁剪标记行，不是"正文里有省略号"：
    财政部令第87号第五十五条的 `F1×A1＋……＋Fn×An` 是官方原文的公式，
    不是裁剪（开发时把它误当成节选，测试立刻报了不一致）。
    """
    assert abridged_marker_keys() == set(ABRIDGED_KEYS), (
        f"实际节选 {abridged_marker_keys()} != 声明 {set(ABRIDGED_KEYS)}"
    )

    for item in REGULATIONS:
        has_topic = TOPIC_ABRIDGED in item.topics
        assert has_topic == (item.key in ABRIDGED_KEYS), f"{item.key} 的节选主题标记不一致"


def test_ellipsis_in_the_middle_of_a_line_is_not_treated_as_abridgement():
    """行内省略号（官方原文的公式/列举）不得被误判为节选。"""
    inline_only = {
        item.key
        for item in REGULATIONS
        if ABRIDGED_MARKER in item.text and item.key not in ABRIDGED_KEYS
    }
    # 至少要有一条这样的条文，否则上面那条断言就失去意义
    assert inline_only, "语料里应存在行内省略号（如评标总得分公式），否则本测试形同虚设"
    for item in REGULATIONS:
        if item.key in inline_only:
            assert TOPIC_ABRIDGED not in item.topics, f"{item.key} 被误标为节选"


def test_required_topics_are_covered():
    """关键主题必须存在，且每个主题至少有 3 条支撑。"""
    stats = corpus_stats()
    topics = stats["topics"]
    for topic in REQUIRED_TOPICS:
        assert topic in topics, f"缺少关键主题: {topic}"
        assert topics[topic] >= MIN_ARTICLES_PER_REQUIRED_TOPIC, (
            f"主题 {topic} 只有 {topics[topic]} 条，少于 {MIN_ARTICLES_PER_REQUIRED_TOPIC} 条"
        )


def test_key_compliance_rules_are_present():
    """合规检查最依赖的具体规则必须在语料里（按法规+条号定位，防止改错条号）。"""
    keys = {a.key for a in REGULATIONS}
    expected = [
        # 以不合理条件限制/排斥潜在投标人
        "中华人民共和国招标投标法实施条例|第三十二条",
        "中华人民共和国政府采购法实施条例|第二十条",
        "中华人民共和国招标投标法|第十八条",
        # 保证金比例与退还
        "中华人民共和国招标投标法实施条例|第二十六条",
        "中华人民共和国招标投标法实施条例|第五十七条",
        "中华人民共和国政府采购法实施条例|第三十三条",
        # 时限：发售 5 日 / 澄清 15 日 / 中标公示 3 日
        "中华人民共和国招标投标法实施条例|第十六条",
        "中华人民共和国招标投标法实施条例|第二十一条",
        "中华人民共和国招标投标法实施条例|第五十四条",
        # 评审因素量化与价格分
        "政府采购货物和服务招标投标管理办法（财政部令第87号）|第五十五条",
        "中华人民共和国政府采购法实施条例|第三十四条",
        # 必须招标的规模标准
        "必须招标的工程项目规定（国家发展改革委令第16号）|第五条",
        # 地方保护 / 所有制歧视
        "招标投标领域公平竞争审查规则|第六条",
        "招标投标领域公平竞争审查规则|第七条",
        "优化营商环境条例|第十三条",
        # 废标 / 投标无效
        "中华人民共和国政府采购法|第三十六条",
        "中华人民共和国招标投标法实施条例|第五十一条",
        "政府采购货物和服务招标投标管理办法（财政部令第87号）|第六十三条",
        # 异议投诉
        "中华人民共和国招标投标法实施条例|第六十条",
        "中华人民共和国政府采购法|第五十二条",
    ]
    missing = [key for key in expected if key not in keys]
    assert missing == [], f"缺少关键条款: {missing}"


def test_critical_numeric_thresholds_are_verbatim():
    """数字型硬指标必须逐字在正文里（写错数字比写错条号更危险）。"""
    by_key = {a.key: a.text for a in REGULATIONS}
    checks = {
        "中华人民共和国招标投标法实施条例|第二十六条": "不得超过招标项目估算价的2%",
        "中华人民共和国招标投标法实施条例|第五十八条": "不得超过中标合同金额的10%",
        "中华人民共和国政府采购法实施条例|第三十三条": "不得超过采购项目预算金额的2%",
        "中华人民共和国政府采购法实施条例|第四十八条": "不得超过政府采购合同金额的10%",
        "必须招标的工程项目规定（国家发展改革委令第16号）|第五条": "400万元人民币以上",
        "中华人民共和国招标投标法实施条例|第十六条": "不得少于5日",
        "中华人民共和国招标投标法实施条例|第二十一条": "投标截止时间至少15日前",
        "中华人民共和国招标投标法实施条例|第五十四条": "公示期不得少于3日",
        "政府采购货物和服务招标投标管理办法（财政部令第87号）|第五十五条": "不得低于30%",
    }
    for key, fragment in checks.items():
        assert key in by_key, f"缺少条款 {key}"
        assert fragment in by_key[key], f"{key} 正文里找不到 {fragment!r}"


def test_effective_dates_are_iso_or_empty():
    pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    for item in REGULATIONS:
        assert item.effective == "" or pattern.match(item.effective), (
            f"{item.key} 施行日期格式异常: {item.effective!r}"
        )


def test_effective_date_is_consistent_within_a_law():
    """同一部法规的所有条文必须报同一个施行日期（防止逐条手写时写岔）。"""
    for law, items in articles_by_law().items():
        dates = {item.effective for item in items}
        assert len(dates) == 1, f"{law} 的施行日期不一致: {dates}"


def test_level_index_is_not_mislabeled():
    """索引式断言：法规名与效力层级的对应关系不能被改错。"""
    by_law = {a.law: a.level for a in REGULATIONS}
    assert by_law["中华人民共和国招标投标法"] == "法律"
    assert by_law["中华人民共和国政府采购法"] == "法律"
    assert by_law["中华人民共和国招标投标法实施条例"] == "行政法规"
    assert by_law["中华人民共和国政府采购法实施条例"] == "行政法规"
    assert by_law["优化营商环境条例"] == "行政法规"
    assert by_law["政府采购货物和服务招标投标管理办法（财政部令第87号）"] == "部门规章"
    assert by_law["必须招标的工程项目规定（国家发展改革委令第16号）"] == "部门规章"
    assert by_law["招标投标领域公平竞争审查规则"] == "部门规章"


def test_key_and_to_payload_contract():
    """`key` 与 payload 字段是入库契约，改动会破坏 point id 的稳定性。"""
    item = REGULATIONS[0]
    assert item.key == f"{item.law}|{item.article}"

    payload = item.to_payload()
    assert set(payload) == {"law", "law_short", "article", "text", "topics", "level", "effective"}
    assert isinstance(payload["topics"], list), "Qdrant 不接受 tuple，必须转 list"
    assert payload["topics"] == list(item.topics)


def test_dataclass_is_frozen():
    item = REGULATIONS[0]
    with pytest.raises(Exception):
        item.text = "改一下"  # type: ignore[misc]


# ---- corpus_stats ----

def test_corpus_stats_structure():
    stats = corpus_stats()
    assert set(stats) == {
        "articles",
        "laws",
        "law_count",
        "law_names",
        "by_level",
        "level_count",
        "by_law",
        "by_law_short",
        "topics",
        "topic_total",
        "laws_with_dates",
    }
    assert stats["articles"] == len(REGULATIONS)
    assert stats["laws"] == len({a.law for a in REGULATIONS})
    assert stats["law_count"] == stats["laws"]
    assert stats["level_count"] == len(stats["by_level"])
    assert stats["laws_with_dates"] <= stats["laws"]


def test_corpus_stats_counts_add_up():
    stats = corpus_stats()
    assert sum(stats["by_level"].values()) == stats["articles"]
    assert sum(stats["by_law"].values()) == stats["articles"]
    assert sum(stats["by_law_short"].values()) == stats["articles"]
    assert stats["topic_total"] == sum(len(a.topics) for a in REGULATIONS)
    assert stats["topic_total"] > stats["articles"], "平均每条应带多个主题"


def test_corpus_stats_level_order_is_stable():
    """层级分布按效力从高到低排序，且只出现语料里真实存在的层级。"""
    by_level = corpus_stats()["by_level"]
    present = [lv for lv in LEVEL_ORDER if lv in by_level]
    assert list(by_level) == present


def test_corpus_stats_topics_sorted_by_count_desc():
    topics = corpus_stats()["topics"]
    counts = list(topics.values())
    assert counts == sorted(counts, reverse=True)


def test_corpus_stats_is_pure_and_repeatable():
    assert corpus_stats() == corpus_stats()


def test_all_articles_matches_module_constant():
    assert all_articles() == REGULATIONS
    assert isinstance(REGULATIONS, tuple), "语料应是不可变元组，避免被调用方就地改动"


def test_articles_by_law_groups_everything():
    grouped = articles_by_law()
    assert sum(len(v) for v in grouped.values()) == len(REGULATIONS)
    for law, items in grouped.items():
        assert {item.law for item in items} == {law}


# =====================================================================
# 二、内存 Qdrant 测试（不连真实服务、不加载模型）
# =====================================================================


def test_point_id_is_stable_and_content_addressed():
    item = REGULATIONS[0]
    assert rs.point_id(item) == rs.point_id(item.law, item.article)
    assert rs.point_id(item) == rs.point_id(item)  # 跨调用稳定

    other = REGULATIONS[1]
    assert rs.point_id(item) != rs.point_id(other)
    # 同条号不同法规不能撞 id
    assert rs.point_id("中华人民共和国招标投标法", "第一条") != rs.point_id(
        "中华人民共和国政府采购法", "第一条"
    )


def test_point_id_rejects_bad_arguments():
    with pytest.raises(rs.RegulationError):
        rs.point_id("中华人民共和国招标投标法")


def test_document_text_includes_law_and_article():
    item = REGULATIONS[0]
    doc = rs.document_text(item)
    assert item.law_short in doc
    assert item.article in doc
    assert item.text in doc


def test_collection_name_is_independent_from_qa_collection():
    """法规库必须是独立集合——复用 bid_qa 会污染经验问答语料。"""
    from src.config import settings

    assert rs.REGULATION_COLLECTION == "bid_regulations"
    assert rs.REGULATION_COLLECTION != settings.qdrant_collection


def test_ensure_collection_is_idempotent(memory_store):
    rs.ensure_regulation_collection(memory_store)
    rs.ensure_regulation_collection(memory_store)
    info = memory_store.client.get_collection(memory_store.collection)
    assert rs.DENSE_VECTOR_NAME in info.config.params.vectors


def test_upsert_writes_every_article(seeded):
    assert seeded.count() == len(REGULATIONS)


def test_upsert_is_idempotent(memory_store, fake_embedder):
    """重复入库不产生重复点——这是稳定 point id 的核心收益。"""
    first = rs.upsert_regulations(store=memory_store, embedder=fake_embedder, progress=False)
    second = rs.upsert_regulations(store=memory_store, embedder=fake_embedder, progress=False)
    assert first == second == len(REGULATIONS)
    assert memory_store.count() == len(REGULATIONS)


def test_upsert_partial_subset_does_not_touch_others(seeded, fake_embedder):
    """再写入 2 条不应影响库里其余条文（增量增补语料的真实场景）。"""
    before = seeded.count()
    rs.upsert_regulations(
        REGULATIONS[:2], store=seeded, embedder=fake_embedder, progress=False
    )
    assert seeded.count() == before


def test_upsert_reports_progress_in_batches(memory_store, fake_embedder, capsys):
    rs.upsert_regulations(
        REGULATIONS[:5],
        store=memory_store,
        embedder=fake_embedder,
        batch_size=2,
        progress=True,
    )
    out = capsys.readouterr().out
    assert "入库进度" in out
    assert memory_store.count() == 5


def test_upsert_recreate_clears_collection(seeded, fake_embedder):
    rs.upsert_regulations(
        REGULATIONS[:3],
        recreate=True,
        store=seeded,
        embedder=fake_embedder,
        progress=False,
    )
    assert seeded.count() == 3


def test_upsert_rejects_duplicate_articles(memory_store, fake_embedder):
    """语料自身重复必须报错，不能静默去重掩盖数据问题。"""
    with pytest.raises(rs.RegulationError):
        rs.upsert_regulations(
            [REGULATIONS[0], REGULATIONS[0]],
            store=memory_store,
            embedder=fake_embedder,
            progress=False,
        )


def test_upsert_rejects_empty_input(memory_store, fake_embedder):
    with pytest.raises(rs.RegulationError):
        rs.upsert_regulations([], store=memory_store, embedder=fake_embedder, progress=False)


def test_upsert_rejects_non_article_objects(memory_store, fake_embedder):
    with pytest.raises(rs.RegulationError):
        rs.upsert_regulations(
            [{"law": "x"}],  # type: ignore[list-item]
            store=memory_store,
            embedder=fake_embedder,
            progress=False,
        )


def test_payload_contains_locator_fields(seeded):
    """payload 必须能定位到"哪部法规第几条"——这是合规依据的最低要求。"""
    records, _ = seeded.client.scroll(
        collection_name=seeded.collection, limit=len(REGULATIONS), with_payload=True
    )
    for record in records:
        payload = record.payload or {}
        assert payload.get("law")
        assert payload.get("article")
        assert payload.get("text")
        assert payload.get("topics")
        assert payload.get("level")
        assert "law_short" in payload and "effective" in payload


def test_search_returns_required_contract_fields(seeded, fake_embedder):
    results = rs.search_regulations(
        "投标保证金", top_k=3, store=seeded, embedder=fake_embedder
    )
    assert results
    for item in results:
        for field in ("law", "article", "text", "score", "topics", "id", "level", "effective"):
            assert field in item, f"检索结果缺少字段 {field}"
        assert isinstance(item["topics"], list)
        assert isinstance(item["score"], float)
        assert -1.0001 <= item["score"] <= 1.0001, "余弦相似度必须在 [-1, 1]"
        assert item["score_source"] == "cosine"


def test_search_respects_top_k(seeded, fake_embedder):
    results = rs.search_regulations(
        "资格条件", top_k=4, store=seeded, embedder=fake_embedder
    )
    assert len(results) == 4


def test_search_is_deterministic(seeded, fake_embedder):
    a = rs.search_regulations("招标文件澄清", top_k=5, store=seeded, embedder=fake_embedder)
    b = rs.search_regulations("招标文件澄清", top_k=5, store=seeded, embedder=fake_embedder)
    assert [r["id"] for r in a] == [r["id"] for r in b]


def test_search_scores_are_valid_cosine_values(seeded, fake_embedder):
    """`score` 是余弦相似度（不是 RRF 融合分），必须在合法区间内。

    ★ 刻意**不**断言 `scores` 降序：结果的**顺序**由双路 RRF 融合决定，
    而 `score` 是融合之后补算的稠密余弦，两者不必同序。
    把顺序断言写在这里是错的（开发时写过一版，被这个测试自己纠正）。
    """
    results = rs.search_regulations("保证金", top_k=8, store=seeded, embedder=fake_embedder)
    assert results
    scores = [r["score"] for r in results]
    assert all(-1.0001 <= s <= 1.0001 for s in scores), scores
    assert all(r["score_source"] == "cosine" for r in results)


def test_hybrid_ranking_differs_from_dense_only(seeded, fake_embedder):
    """混合检索的排序必须与纯稠密不同——否则关键词路是死代码。

    用假嵌入器时两路都可能命中，只要给出**任一个**排序差异即说明
    sparse 路确实参与了融合。
    """
    query = "保证金 第二十六条 比例"
    hybrid = rs.search_regulations(query, top_k=6, store=seeded, embedder=fake_embedder)
    dense = rs.search_regulations(
        query, top_k=6, store=seeded, embedder=fake_embedder, hybrid=False
    )
    hybrid_top = [r["id"] for r in hybrid]
    dense_top = [r["id"] for r in dense]
    assert hybrid_top and dense_top
    assert hybrid_top != dense_top, "两条路的排序完全一致，sparse 路可能没生效"


def test_dense_only_mode_still_scores_by_cosine(seeded, fake_embedder):
    results = rs.search_regulations(
        "保证金", top_k=5, store=seeded, embedder=fake_embedder, hybrid=False
    )
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True), "纯稠密模式下按余弦排序"
    assert all(r["score_source"] == "cosine" for r in results)


def test_search_exact_document_scores_highest(memory_store, fake_embedder):
    """把某条原文当查询，它自己必须排第一 —— 验证 id/payload/向量没有错位。"""
    target = next(
        a for a in REGULATIONS if a.law_short == "招标投标法实施条例" and a.article == "第二十六条"
    )
    rs.upsert_regulations(store=memory_store, embedder=fake_embedder, progress=False)
    results = rs.search_regulations(
        rs.document_text(target), top_k=1, store=memory_store, embedder=fake_embedder
    )
    assert results[0]["article"] == target.article
    assert results[0]["law"] == target.law
    assert results[0]["score"] > 0.99


def test_search_topic_filter_narrows_results(seeded, fake_embedder):
    results = rs.search_regulations(
        "比例上限", top_k=20, topics=["保证金"], store=seeded, embedder=fake_embedder
    )
    assert results, "按保证金过滤后不应为空"
    assert len(results) < len(REGULATIONS)
    for item in results:
        assert "保证金" in item["topics"], item


def test_search_topic_filter_accepts_multiple_topics(seeded, fake_embedder):
    results = rs.search_regulations(
        "资格", top_k=20, topics=["差别待遇", "歧视待遇"], store=seeded, embedder=fake_embedder
    )
    assert results
    for item in results:
        assert set(item["topics"]) & {"差别待遇", "歧视待遇"}


def test_search_topic_filter_with_unknown_topic_returns_empty(seeded, fake_embedder):
    results = rs.search_regulations(
        "任意问题", top_k=5, topics=["并不存在的主题"], store=seeded, embedder=fake_embedder
    )
    assert results == []


def test_search_blank_query_raises(seeded, fake_embedder):
    with pytest.raises(rs.RegulationError):
        rs.search_regulations("   ", store=seeded, embedder=fake_embedder)


def test_search_invalid_top_k_raises(seeded, fake_embedder):
    with pytest.raises(rs.RegulationError):
        rs.search_regulations("保证金", top_k=0, store=seeded, embedder=fake_embedder)


def test_search_on_empty_collection_returns_empty(memory_store, fake_embedder):
    rs.ensure_regulation_collection(memory_store)
    assert rs.search_regulations("保证金", store=memory_store, embedder=fake_embedder) == []


def test_regulation_stats_reports_stored_counts(seeded):
    stats = rs.regulation_stats(seeded)
    assert stats["collection"] == rs.REGULATION_COLLECTION
    assert stats["points"] == len(REGULATIONS)
    assert stats["laws"] == len({a.law for a in REGULATIONS})
    assert stats["corpus_articles"] == len(REGULATIONS)
    assert stats["in_sync"] is True
    assert sum(stats["by_law"].values()) == len(REGULATIONS)


def test_regulation_stats_detects_partial_ingest(memory_store, fake_embedder):
    rs.upsert_regulations(
        REGULATIONS[:5], store=memory_store, embedder=fake_embedder, progress=False
    )
    stats = rs.regulation_stats(memory_store)
    assert stats["points"] == 5
    assert stats["in_sync"] is False


def test_regulation_stats_swallows_connection_errors(memory_store, monkeypatch):
    def _boom() -> int:
        raise RuntimeError("connection refused")

    monkeypatch.setattr(memory_store, "count", _boom)
    stats = rs.regulation_stats(memory_store)
    assert stats["points"] == -1
    assert stats["in_sync"] is False


def test_regulation_stats_can_raise_on_error(memory_store, monkeypatch):
    def _boom() -> int:
        raise RuntimeError("connection refused")

    monkeypatch.setattr(memory_store, "count", _boom)
    with pytest.raises(rs.RegulationError):
        rs.regulation_stats(memory_store, raise_on_error=True)


def test_verify_regulations_returns_sample(seeded):
    result = rs.verify_regulations(seeded)
    assert result["points"] == len(REGULATIONS)
    assert len(result["sample"]) == 3
    for item in result["sample"]:
        assert item["law"]
        assert item["article"]
        assert item["text"]
        assert item["topics"]


# ---- 稀疏向量（无词表 BM25）----


def test_sparse_index_is_stable_within_process():
    assert rs._sparse_index("保证金") == rs._sparse_index("保证金")
    assert rs._sparse_index("保证金") != rs._sparse_index("资格条件")


def test_sparse_index_is_non_negative_and_in_range():
    for term in ("保证金", "2%", "第四十三条", "brand", "。"):
        idx = rs._sparse_index(term)
        assert 0 <= idx < 2**31, term


def test_sparse_index_is_process_independent():
    """下标不能依赖 `hash()`——它对 str 有进程级随机盐。

    这是**最容易踩且最难排查**的坑：换进程后下标全变，入库向量与查询向量
    静默错位（检索结果莫名其妙变差，且本地复现不了）。这里起一个全新进程
    算同一个词的下标，必须与当前进程一致。
    """
    import subprocess
    import sys

    term = "招标投标保证金"
    expected = rs._sparse_index(term)
    code = (
        "import sys; sys.path.insert(0, r'{}'); "
        "from src.services.regulation_store import _sparse_index; "
        "print(_sparse_index({!r}))"
    ).format(str(Path(__file__).resolve().parents[1]), term)

    for seed in ("0", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True
        )
        assert int(out.stdout.strip()) == expected, f"PYTHONHASHSEED={seed} 时下标变了"


def test_tf_weights_shape_and_weights():
    vec = rs._tf_weights("投标保证金 投标保证金 资格条件")
    assert len(vec.indices) == len(vec.values)
    assert len(vec.indices) == len(set(vec.indices)), "下标必须去重"
    assert all(0.0 < v < 1.0 for v in vec.values)
    assert vec.indices == sorted(vec.indices), "下标应有序，便于比对与调试"


def test_tf_weights_saturates_repeated_terms():
    """同一词出现越多权重越大，但始终 < 1（词频饱和）。"""
    once = rs._tf_weights("保证金")
    thrice = rs._tf_weights("保证金 保证金 保证金")
    assert once.values[0] < thrice.values[0] < 1.0
    assert once.indices == thrice.indices


def test_tf_weights_is_deterministic():
    text = "评审因素应当量化"
    assert rs._tf_weights(text).indices == rs._tf_weights(text).indices
    assert rs._tf_weights(text).values == rs._tf_weights(text).values


def test_tf_weights_handles_empty_and_symbol_only_text():
    """空/纯符号文本返回空稀疏向量，而不是抛异常（Qdrant 接受空向量）。"""
    for text in ("", "   ", "。。。", "!!!"):
        vec = rs._tf_weights(text)
        assert vec.indices == []
        assert vec.values == []


def test_document_text_includes_topics_for_recall():
    """topics 参与向量化文本——这是"口语化问法也能召回"的关键。"""
    item = next(a for a in REGULATIONS if "保证金" in a.topics)
    doc = rs.document_text(item)
    assert item.topics[0] in doc
    assert item.law_short in doc
    assert item.article in doc


# =====================================================================
# 三、真实 Qdrant 集成测试（默认 skip）
# =====================================================================


def _live_enabled() -> bool:
    import os

    return os.getenv("REGULATION_TEST_LIVE", "").strip() in ("1", "true", "yes")


def _live_store() -> VectorStore | None:
    """真实服务不可达时返回 None（调用方据 skip）。"""
    import os

    from src.config import settings

    try:
        store = VectorStore(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            collection=rs.REGULATION_COLLECTION,
            dim=settings.qdrant_vector_size,
            distance=settings.qdrant_distance,
            timeout=settings.qdrant_timeout,
        )
        if not store.client.collection_exists(rs.REGULATION_COLLECTION):
            return None
        return store
    except Exception:
        return None


_requires_qdrant = pytest.mark.skipif(
    not _live_enabled() or _live_store() is None,
    reason="需要真实 Qdrant 且设 REGULATION_TEST_LIVE=1（集成测试，默认跳过）",
)


@_requires_qdrant
def test_live_collection_has_articles():
    store = _live_store()
    assert store is not None
    assert store.count() >= MIN_ARTICLES


@_requires_qdrant
def test_live_payloads_are_complete():
    store = _live_store()
    assert store is not None
    records, _ = store.client.scroll(
        collection_name=rs.REGULATION_COLLECTION, limit=200, with_payload=True
    )
    for record in records:
        payload = record.payload or {}
        assert payload.get("law") and payload.get("article") and payload.get("text")


@_requires_qdrant
def test_live_search_returns_relevant_articles():
    """真实模型 + 真实库的检索冒烟：合法性问题必须命中带"资格条件"主题的条文。"""
    results = rs.search_regulations("以特定品牌作为资格条件", top_k=3)
    assert results
    joined = " ".join(r["text"] for r in results)
    assert "品牌" in joined

"""问题分类 + RRF k 映射：按问句特征分桶，映射到不同的检索参数。

**为什么需要分类**：不同问法对稠密向量与 BM25 关键词的倚重不同——

1. **关键词查询**（"单一来源采购"几个词）：BM25 精确匹配信号强，RRF 应偏向 BM25 路——
   k 取小值，放大头部排名优势，让关键词命中排在前面；
2. **自然语言问句**（"什么是政府采购的单一来源采购方式？"）：语义检索更关键，RRF 应
   偏向 dense 路——k 取大值，让各排名贡献更平均，语义近邻不被关键词硬盖；
3. **法条引用**（"政府采购法第二十六条"）：关键词与语义都重要，需更大召回池——
   k 取中值，两路均衡。

RRF（Reciprocal Rank Fusion）公式：`score = Σ 1/(k + rank)`，rank 从 1 起。
k 控制排名靠后项的贡献衰减速度：k 小 → 头部优势放大；k 大 → 更平均。

**为什么用规则而非 LLM 分类**：分类是检索的前置步骤，每次问答都要跑，LLM 调用
慢且贵。三类规则（法条正则 / 长度 + 标点 / 其余）已覆盖真实问句分布，且结果确定
可复现——测试与 CI 不会因模型随机性抖动。
"""

from __future__ import annotations

import re
from enum import Enum


class QuestionCategory(Enum):
    """问题类别。新增类别须同步更新 RRF_PARAMS。"""

    KEYWORD = "keyword"        # 关键词查询：短、无标点
    SENTENCE = "sentence"     # 自然语言问句：带标点、较长
    CITATION = "citation"     # 法条/法规引用：含"第X条""XX法"等


# 各类别 → 检索参数映射（RRF k + 双路权重 + 召回深度 + top_k）
#
# ★ k 与权重是两个独立维度，别混为一谈
# -----------------------------------
#     dense_weight / sparse_weight   调"**信哪一路**"（整体影响力）
#     rrf_k                          调"**多看重头部**"（排名衰减速度）
#
# 例如关键词查询既要偏 BM25（权重），又要放大其头部优势（小 k）。
# 只调 k 无法表达"这一路整体不重要"，只调权重也无法表达"头部要更突出"。
#
# ★ 权重取值依据
# ------------
# 权重与 k 的初始值由 `eval/retrieval_eval.py` 实测标定（见 docs/39）。
# 归一化后只表达相对倚重，绝对值无意义（`[7,3]` 与 `[0.7,0.3]` 等价）。
RRF_PARAMS: dict[QuestionCategory, dict] = {
    QuestionCategory.KEYWORD: {
        "rrf_k": 20, "prefetch_limit": 30, "top_k": 5,
        # 关键词查询：BM25 精确匹配是主信号，故偏稀疏路
        "dense_weight": 0.35, "sparse_weight": 0.65,
    },
    QuestionCategory.SENTENCE: {
        "rrf_k": 60, "prefetch_limit": 40, "top_k": 5,
        # 自然语言问句：语义是主信号，故偏稠密路
        "dense_weight": 0.7, "sparse_weight": 0.3,
    },
    QuestionCategory.CITATION: {
        "rrf_k": 40, "prefetch_limit": 50, "top_k": 5,
        # 法条引用：条文编号（关键词）与条文语义同等重要，两路均衡
        "dense_weight": 0.5, "sparse_weight": 0.5,
    },
}

# 法条/法规引用模式：匹配"第N条""XX法""XX条例"等
# 注意："规定""办法"在自然语言中也是常用动词/名词（"是怎么规定的"），
#       所以 `的` 结尾的模式只保留"法""条例"这类不会作动词的法规专名。
_CITATION_PATTERNS = [
    re.compile(r"第\s*\d+\s*条"),          # 第二十六条、第3条
    re.compile(r"第\s*[一二三四五六七八九十百千]+\s*条"),  # 第二十条（中文数字）
    re.compile(r"[\u4e00-\u9fa5]{2,8}(法|条例|办法|规定|目录|清单)$"),  # 政府采购法（字符串结尾）
    re.compile(r"[\u4e00-\u9fa5]{2,6}(法|条例)的"),  # 政府采购法的（只匹配法规专名+的）
]

# 判断"短查询"的阈值：短于这个字符数且无句末标点 → 关键词
_KEYWORD_MAX_LEN = 15
# 句末标点：有这些说明是完整问句而非关键词堆
_SENTENCE_PUNCT = set("。？?！!")


def classify(question: str) -> QuestionCategory:
    """对问题分类，返回类别。纯函数，无副作用，结果确定可复现。

    分类优先级（从前到后，命中即返回）：
    1. 法条引用——含"第N条"或"XX法/条例/办法"等法规名称模式；
    2. 关键词——长度 ≤15 且不含句末标点（短、无标点 = 关键词堆叠）；
    3. 自然语言——其余全部归入（长句、带标点）。
    """
    text = (question or "").strip()
    if not text:
        return QuestionCategory.SENTENCE  # 空查询走默认参数，不阻断流程

    # ① 法条引用优先：法条编号/法规名称是强信号，即使短也归 citation
    for pattern in _CITATION_PATTERNS:
        if pattern.search(text):
            return QuestionCategory.CITATION

    # ② 关键词查询：短且无句末标点
    if len(text) <= _KEYWORD_MAX_LEN and not (set(text) & _SENTENCE_PUNCT):
        return QuestionCategory.KEYWORD

    # ③ 其余归自然语言
    return QuestionCategory.SENTENCE


def get_params(question: str) -> dict:
    """便捷方法：分类 → 返回对应的检索参数字典。

    返回的 key：`rrf_k`（RRF 常数）、`dense_weight` / `sparse_weight`（双路权重）、
    `prefetch_limit`（每路召回数）、`top_k`（最终返回条数）。

    ★ 权重与 k 都会传给两条融合路径
    ----------------------------
    `k == 60` 走 Qdrant 服务端 RRF（用 `Rrf(weights=..., k=...)`），
    其余走应用层 `fuse_rrf`。**两条路径都必须收到同一组参数**——
    否则同一个问题在两种 k 下排序会不一致（实测踩过：只改一条等于写死代码）。
    """
    return dict(RRF_PARAMS[classify(question)])


def fuse_rrf(
    dense_results: list[dict],
    sparse_results: list[dict],
    k: int = 60,
    limit: int = 5,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
) -> list[dict]:
    """应用层 RRF 融合：把两路检索结果按加权 `1/(k+rank)` 合并排序。

    Qdrant 的内置 `FusionQuery(RRF)` 不暴露 k 参数（固定 k=60），当分类器要求
    非 60 的 k 值时，改为在应用层做两路独立查询后由此函数融合。

    两路结果按 `question` 字段去重（同一问答对可能在两路都召回），合并后取 limit 条。
    分数 = `Σ weight_路 × 1/(k+rank)`，rank 从 1 起。

    ★ 权重的作用与 k 不同（为什么要两个都留）
    --------------------------------------
        权重  改变某一路的**整体影响力**（0.7 比 0.3 强一倍多）
        k     改变**排名衰减速度**（k 小 → 头部优势放大；k 大 → 更平均）

    两者是独立的调节维度：权重调"信哪一路"，k 调"多看重头部"。
    例如"关键词查询"既要偏 BM25（权重），又要放大其头部优势（小 k）。

    ★ 默认值 1.0/1.0 等价于无权重的经典 RRF
    -------------------------------------
    故本参数是**纯增量**：既有调用方与测试不传权重时行为完全不变
    （`1.0×1/(k+rank)` 与原来的 `1/(k+rank)` 数值相同）。

    ★ 归一化：权重按比例生效，绝对值不影响排序
    ----------------------------------------
    `[0.7, 0.3]` 与 `[7, 3]` 结果相同。故这里先归一化到和为 1，
    让传入值只表达"相对倚重"，避免调用方误以为必须凑成 1。
    两路权重都为 0（或负）时退回均权，避免把所有分数算成 0 而丢失排序。
    """
    dw = max(0.0, float(dense_weight))
    sw = max(0.0, float(sparse_weight))
    total = dw + sw
    if total <= 0:
        dw = sw = 1.0
    else:
        dw, sw = dw / total, sw / total

    scores: dict[str, float] = {}
    # 用 question 作 key 聚合（payload 中 question 唯一）
    payload: dict[str, dict] = {}

    for rank, item in enumerate(dense_results, start=1):
        key = item.get("question", "")
        scores[key] = scores.get(key, 0.0) + dw / (k + rank)
        payload[key] = item

    for rank, item in enumerate(sparse_results, start=1):
        key = item.get("question", "")
        scores[key] = scores.get(key, 0.0) + sw / (k + rank)
        # sparse 路若命中了 dense 没有的，补进 payload；已有的保留 dense 的（内容一致）
        payload.setdefault(key, item)

    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:limit]
    result = []
    for key, score in ordered:
        merged = dict(payload[key])
        merged["score"] = score
        result.append(merged)
    return result

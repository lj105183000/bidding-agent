"""编码器：BGE dense 向量 + jieba BM25 稀疏向量 + 可选 CrossEncoder 精排。

三个模型都遵循同一套加载策略——**懒加载 + 双重检查锁**（见 docs/开发文档.md
§8.1）：首次实际使用才加载，且并发首用只加载一份。这样做是因为模型加载耗时
数十秒、占数百 MB，放在 import 期会让启动、测试、CI 全部变慢。
"""

from __future__ import annotations

import json
import math
import threading
from collections import Counter
from pathlib import Path

import jieba
from qdrant_client import models

from src.logging_config import get_logger
from src.rag.constants import (
    BM25_B,
    BM25_K1,
    EMBED_MODEL_NAME,
    QUERY_PREFIX,
    RERANK_MODEL_NAME,
    VOCAB_FILE,
)

logger = get_logger(__name__)


def _tokenize(text: str) -> list[str]:
    """中文分词。

    精确模式 + 过滤单字：招投标领域的大量单字（"的"、"和"）只会稀释 BM25 的
    区分度，而领域关键词（"单一来源"、"异议"、"标的物"）几乎都是多字词。
    """
    tokens = []
    for token in jieba.lcut(text or "", cut_all=False):
        token = token.strip()
        if len(token) < 2:
            continue  # 单字与标点区分度低，只会稀释 BM25
        if token.isdigit():
            continue  # 纯数字（年份、金额）对语义检索无贡献
        if all(not ch.isalnum() for ch in token):
            continue  # 纯符号
        tokens.append(token)
    return tokens


class Embedder:
    """BGE dense 编码器。"""

    def __init__(self, model_name: str = EMBED_MODEL_NAME) -> None:
        self.model_name = model_name
        self._model = None
        self._lock = threading.Lock()
        self._device = "cpu"

    @property
    def model(self):
        """双重检查锁：快路径无锁返回，慢路径加锁且只加载一次。"""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    try:
                        import torch

                        self._device = "cuda" if torch.cuda.is_available() else "cpu"
                    except Exception:  # torch 不可用时退回 CPU，不阻断流程
                        self._device = "cpu"

                    logger.info("加载嵌入模型 %s (device=%s)...", self.model_name, self._device)
                    self._model = SentenceTransformer(self.model_name, device=self._device)
                    logger.info("嵌入模型就绪")
        return self._model

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _encode(self, texts: list[str], **kwargs) -> list[list[float]]:
        try:
            vectors = self.model.encode(texts, normalize_embeddings=True, **kwargs)
        except RuntimeError as exc:
            # 显存不足时把模型挪到 CPU 重试一次：宁可慢，也不要让整个检索链路失败
            if "out of memory" in str(exc).lower() and self._device == "cuda":
                logger.warning("GPU 显存不足，降级到 CPU 重试")
                self.model.to("cpu")
                self._device = "cpu"
                vectors = self.model.encode(texts, normalize_embeddings=True, **kwargs)
            else:
                raise
        return [list(map(float, v)) for v in vectors]

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        """文档侧编码：**不加**指令前缀。"""
        return self._encode(texts, show_progress_bar=len(texts) > 64)

    def encode_query(self, text: str) -> list[float]:
        """查询侧编码：加 bge 检索指令前缀（见 constants.QUERY_PREFIX 说明）。"""
        return self._encode([QUERY_PREFIX + (text or "")])[0]


class Reranker:
    """CrossEncoder 精排（默认关闭，见 `settings.rerank_enabled`）。

    BM25/dense 都是「查询与文档各自编码后算距离」，精排则是把两者拼起来过一遍
    模型，能捕捉更细的相关性——代价是每个候选一次前向推理（CPU 上约 0.9s/查询）。

    **当前语料下的实测结论：开了不会变好，因此默认关闭。** 用 LLM 改写的口语化
    查询跑 `eval/retrieval_eval.py`（75 条），混合检索基线的 Top-1 已是 98.7%、
    Top-5 100%，精排（配对问题字段）同样是 98.7%/100%，MRR 都是 0.993——**没有
    提升空间**。语料只有 75 条且问答对彼此差异明显，RRF 融合已经够用。

    什么时候值得打开：语料扩大到数千条、出现大量相互混淆的近义问答时，精排才有
    发挥空间。**打开前先跑一次评测**，别凭感觉。
    """

    def __init__(self, model_name: str = RERANK_MODEL_NAME) -> None:
        self.model_name = model_name
        self._model = None
        self._lock = threading.Lock()

    @property
    def model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import CrossEncoder

                    logger.info("加载精排模型 %s...", self.model_name)
                    self._model = CrossEncoder(self.model_name)
                    logger.info("精排模型就绪")
        return self._model

    def rerank(
        self, query: str, candidates: list[dict], top_k: int, text_key: str = "question"
    ) -> list[dict]:
        """对候选按相关性重排，并把分数归一化到 0-1。

        **配对字段为什么默认是"问题"而不是"答案"**：用户查询是短问句，知识库答案
        是大段说明文，二者文体差距过大，CrossEncoder 会偏向"文体相似"的错误候选。
        实测（`eval/retrieval_eval.py`，keyword 查询）用答案配对把 top-1 命中率从
        98.7% 打到 78.7%，用问题配对则与混合检索持平。改动此默认值前请先跑评测。

        **为什么显式请求未过激活的输出**：`CrossEncoder.predict()` 对单标签模型
        默认已经套了一次 Sigmoid（见 sentence-transformers 的 activation_fn 默认值）。
        若直接拿它的输出再做归一化，就成了**双重 sigmoid**：一个 logit 为 -5.9 的
        不相关候选会被抬到 0.50，分数彻底失去区分度。这里统一要原始 logit，
        由本类做一次 sigmoid，行为不随依赖库的默认值变化。

        字段语义：`rerank_score` 是原始 logit（排查用，可为负），`score` 是它的
        sigmoid（0-1，前端与来源卡片展示用）。logit 常达 ±8 以上，sigmoid 会饱和到
        0/1，因此 `score` 适合作展示、不适合作精细阈值。
        """
        if not candidates:
            return []

        from torch import nn

        pairs = [(query, c.get(text_key) or c.get("question") or "") for c in candidates]
        scores = self.model.predict(pairs, activation_fn=nn.Identity())

        ordered = sorted(zip(candidates, scores), key=lambda kv: float(kv[1]), reverse=True)[:top_k]
        out = []
        for item, score in ordered:
            merged = dict(item)
            merged["rerank_score"] = float(score)
            merged["score"] = _sigmoid(float(score))
            out.append(merged)
        return out


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class BM25Encoder:
    """把 BM25 打分转成 Qdrant 稀疏向量，使关键词检索与向量检索能在**同一个
    集合、同一次查询**里完成（Qdrant 原生 Prefetch + RRF 融合）。

    做法：`fit` 阶段统计词表、idf 与平均文档长度并落盘 `data/vocab.json`；
    文档侧与查询侧分别按 BM25 公式算出每个词的权重，写入 sparse 向量的
    (indices, values)；检索时点积即 BM25 分数。

    索引一旦重建，**旧向量全部失效**——这就是「换数据必须重新 ingest」的
    根本原因（见 docs/开发文档.md §4.1）。
    """

    VERSION = 1

    def __init__(self, k1: float = BM25_K1, b: float = BM25_B) -> None:
        self.k1 = k1
        self.b = b
        self.vocab: dict[str, int] = {}
        self.idf: list[float] = []
        self.avgdl: float = 0.0
        self.n_docs: int = 0

    # ---- 拟合与持久化 ----

    def fit(self, documents: list[str]) -> "BM25Encoder":
        df: Counter[str] = Counter()
        total_len = 0
        for doc in documents:
            tokens = _tokenize(doc)
            total_len += len(tokens)
            df.update(set(tokens))

        # 词表按字典序编号：同一份语料无论何时 fit，索引都一致，
        # 便于比对不同机器上生成的 vocab.json
        self.vocab = {term: idx for idx, term in enumerate(sorted(df))}
        self.n_docs = len(documents)
        self.avgdl = (total_len / self.n_docs) if self.n_docs else 0.0

        # BM25 的 idf 用 ln(1 + ...) 形式，恒为正数：避免高频词出现负权重，
        # 否则稀疏向量点积会出现"词越常见越扣分"的反常行为
        n = self.n_docs
        self.idf = [0.0] * len(self.vocab)
        for term, idx in self.vocab.items():
            self.idf[idx] = math.log(1.0 + (n - df[term] + 0.5) / (df[term] + 0.5))
        return self

    def save(self, path: Path | None = None) -> Path:
        target = Path(path or VOCAB_FILE)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.VERSION,
            "k1": self.k1,
            "b": self.b,
            "avgdl": self.avgdl,
            "n_docs": self.n_docs,
            "vocab": self.vocab,
            "idf": self.idf,
        }
        target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: Path | None = None) -> "BM25Encoder":
        source = Path(path or VOCAB_FILE)
        if not source.exists():
            raise FileNotFoundError(
                f"词表文件不存在: {source}（请先执行 `python main.py ingest` 导入知识库）"
            )
        payload = json.loads(source.read_text(encoding="utf-8"))
        encoder = cls(k1=payload.get("k1", BM25_K1), b=payload.get("b", BM25_B))
        encoder.vocab = payload["vocab"]
        encoder.idf = payload["idf"]
        encoder.avgdl = payload.get("avgdl", 0.0)
        encoder.n_docs = payload.get("n_docs", 0)
        return encoder

    # ---- 编码 ----

    def _weights(self, tokens: list[str], doc_length: float | None) -> models.SparseVector:
        """按 BM25 公式把词序列转成稀疏向量。

        `doc_length=None` 表示查询侧——查询不做长度归一化（BM25 标准做法：
        长查询本身就带更多信息量，不该因此被惩罚）。
        """
        counts = Counter(tokens)
        indices: list[int] = []
        values: list[float] = []
        for term, tf in counts.items():
            idx = self.vocab.get(term)
            if idx is None:
                continue  # 词表外的词无从取 idf，直接丢弃
            idf = self.idf[idx]
            if doc_length is None:
                weight = idf * (tf * (self.k1 + 1)) / (tf + self.k1)
            else:
                norm = 1 - self.b + self.b * (doc_length / self.avgdl if self.avgdl else 1.0)
                weight = idf * (tf * (self.k1 + 1)) / (tf + self.k1 * norm)
            if weight > 0:
                indices.append(idx)
                values.append(weight)
        return models.SparseVector(indices=indices, values=values)

    def encode_documents(self, documents: list[str]) -> list[models.SparseVector]:
        vectors = []
        for doc in documents:
            tokens = _tokenize(doc)
            vectors.append(self._weights(tokens, float(len(tokens))))
        return vectors

    def encode_query(self, query: str) -> models.SparseVector:
        return self._weights(_tokenize(query), None)


# ---- 模块级单例（与 settings 同风格，便于测试整体替换）----
embedder = Embedder()
reranker = Reranker()

"""RAG 链路常量。

集中在此并附语义说明，避免 magic number 散落各处（见 docs/开发文档.md §13.1）。

说明：docs/开发文档.md 约定常量集中到 `src/agent/constants.py`，但当前落地的
是 RAG 链路本身、`src/agent/` 属成员 A 的分工范围，故 RAG 侧常量就近放在本
文件；待 Agent 层落地后可再上提合并。
"""

from __future__ import annotations

from pathlib import Path

from src.config import PROJECT_ROOT

# ---- 路径 ----
DATA_DIR = PROJECT_ROOT / "data"
VOCAB_FILE = DATA_DIR / "vocab.json"

# ---- 嵌入模型（BAAI/bge-small-zh-v1.5 原生输出 512 维）----
EMBED_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
EMBED_DIM = 512
# bge 系列中文模型的检索指令前缀：查询侧必须加，文档侧不加。
# 漏加会让查询与文档落在同一空间的不同区域，召回率明显下降。
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

# ---- 精排模型（可选，默认关闭）----
RERANK_MODEL_NAME = "BAAI/bge-reranker-base"

# ---- 检索参数 ----
DEFAULT_TOP_K = 5
# 混合检索每条路径的召回数：先各召回 30 条再融合，给 RRF 留出足够的重排空间
PREFETCH_LIMIT = 30
# 精排候选池大小：精排只能重排它看到的这些，池子太小会成为效果上限
RERANK_CANDIDATE_K = 20

# ---- BM25 参数（Qdrant sparse 向量承载）----
BM25_K1 = 1.5  # 词频饱和系数
BM25_B = 0.75  # 文档长度归一化强度

# ---- 流式输出 ----
# 小帧切分：LLM 的字符流粒度不均，按固定长度切能避免"一次蹦一行"的观感；
# 帧间延时让前端渲染节奏平滑（40 字符 / 20ms ≈ 2000 字符/秒）
STREAM_CHUNK_CHARS = 40
STREAM_CHUNK_DELAY = 0.02
# 首帧填充：部分反向代理与浏览器缓冲会攒够一定字节才下发，先塞一段注释强制 flush
SSE_PADDING_BYTES = 2048

# ---- 对话历史 ----
MAX_HISTORY_ROUNDS = 5

# ---- 工具语义 ----
# done 帧的 tool_name 取此值（单字符串，前端徽标与 eval 依赖，勿改成数组）
TOOL_NAME_KB = "search_knowledge_base"

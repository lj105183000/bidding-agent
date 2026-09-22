"""工具定义 + 执行器注册表 + 结果格式化（五路工具的落地文件）。

职责
----
把四个数据源翻译成**给 LLM 看的文本**，并把它们的来源列表交给上层：

    search_knowledge_base   → src.rag.pipeline.rag_pipeline（成员 B，未就绪时降级）
    search_knowledge_graph  → src.database.neo4j_client.Neo4jClient
    query_database          → src.database.postgresql_client.PostgresClient
    search_web              → src.web_search.web_search_client（Tavily）
    search_exa              → src.mcp.web_search_exa（成员 C 的另一部分，未落地时降级）

执行器契约（冻结，见 分配说明.md §5 / docs/组件工作机制.md §13）
--------------------------------------------------------------
`executor(arguments: dict, question: str = "") -> (格式化文本, sources 列表)`

- **永不抛异常**：失败返回 `("错误文案", [])`，让 Agent 带着"这一路没有数据"继续往下答。
  后端宕机不该让整轮问答崩掉——这是本项目对 Agent 的硬承诺。
- 契约是二元组，所以**成败只能体现在文本里**（`src/tools/base.py` 的 `_as_tool_result`
  把元组路径恒定为 `success=True`）。不要改成抛异常。
- `question` 可缺省：`src/agent/react_loop.py` 只传 arguments，`ToolRunner` 才会注入
  question（靠 `_accepts_question` 签名探测）。联网类需要它做结果重排，检索类用不上。

为什么不在顶层 import 数据源
----------------------------
1. `src/agent/constants.py`：**顶层绝不能 import**。导入链是
   `src.agent.__init__` → `core` →（try）`rag_tools` → `src.tools.base` → 回头 import
   `src.agent.constants` 时 `src.agent` 只初始化了一半，抛出的 ImportError 会被
   `core.py` 的 `except ImportError` 静默吞掉，**整个工具层悄悄变成空字典**。
   需要常量时在函数内 import。
2. `src.rag` / `src.mcp`：顶层 import 会让整个 `TOOL_EXECUTORS` 构造失败，五个工具一起
   消失；放进函数里则只有那一路降级。`src.mcp` 还没落地；`src.rag` 虽已随 PR #3 合入 main，
   但它模块级依赖成员 D 的 `src.config` / `src.logging_config` / `src.clients`，
   那三个文件还只在 `feat/api-llm` 上——**在这条分支上 import 仍然必然失败**，
   所以降级路径不是历史包袱，而是当前的真实运行状态。

参数名为什么要容错
------------------
工具调用走的是**文本解析**（`src/agent/tool_defense.py` 正则抽 `{"name","arguments"}`），
不是原生 Function Calling；系统提示里只给了工具名和一句话描述、**没有参数名**，
模型只能猜。所以每个执行器都用 `_pick` 接受多个同义键，猜错时回落到默认分支，
而不是直接回一句"缺少必填参数"。
"""

import logging
import math
import threading

from src.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

# --- 数值上限 ---

# 与两个数据库客户端的 MAX_LIMIT 对齐（neo4j_client.py:65 / postgresql_client.py:97）。
# 客户端自己也会再收敛一次，这里先挡一道是为了不让明显越界的值走完整个调用链。
MAX_LIMIT = 200

RAG_DEFAULT_TOP_K = 5      # 与 src/agent/generation.py 的 RAG_TOP_K 一致
RAG_MAX_TOP_K = 20
GRAPH_DEFAULT_LIMIT = 20
PG_DEFAULT_LIMIT = 20

WEB_DEFAULT_MAX_RESULTS = 5
WEB_MAX_RESULTS = 10
WEB_DISPLAY_LIMIT = 5      # 组件工作机制.md §13：_fmt_web 只展示前 5 条
PG_DISPLAY_ROWS = 20       # 组件工作机制.md §13：_fmt_pg 只展示前 20 行

# --- 文本截断 ---

# 工具结果会原样进 LLM 上下文，而 MAX_TOOL_ROUNDS=4 意味着最多累积四轮结果。
# 单条记录截得短、整体再兜一道，防止一次检索把上下文预算吃光。
FIELD_TRUNCATE_CHARS = 120
ANSWER_TRUNCATE_CHARS = 300
WEB_SNIPPET_CHARS = 200
PG_FIELD_TRUNCATE_CHARS = 40
MAX_OBSERVATION_CHARS = 4000

# --- 用户可见文案 ---
#
# 三条 RAG 文案与 src/agent/generation.py:145-156 的降级路径保持同一措辞：
# 同一类故障在"工具路径"与"RAG 降级路径"下必须给出一致的说法，否则排障时
# 会被当成两个不同的问题。文案里不含地址、SQL、异常类型（那些只进日志）。

_TEXT_RAG_UNAVAILABLE = "知识库暂时不可用，请稍后重试。"
_TEXT_RAG_FAILED = "知识库检索失败，请稍后重试。"
_TEXT_RAG_EMPTY = "知识库中暂未找到相关内容，请换个问法试试。"

_TEXT_GRAPH_EMPTY = "知识图谱中未找到符合条件的记录。"
_TEXT_GRAPH_DOWN = "知识图谱服务暂时不可用，已跳过该数据源。"
_TEXT_PG_EMPTY = "业务数据库中未找到符合条件的记录。"
_TEXT_PG_DOWN = "业务数据库暂时不可用，已跳过该数据源。"

_TEXT_WEB_NO_KEY = "未配置联网搜索密钥（TAVILY_API_KEY），联网搜索不可用。"
_TEXT_WEB_FAILED = "联网搜索暂时不可用，请稍后重试。"
_TEXT_WEB_EMPTY = "联网没有搜到相关结果。"
_TEXT_EXA_UNAVAILABLE = "语义联网搜索（Exa）暂未就绪，已跳过该数据源；如需时效性信息请改用关键词联网搜索。"
_TEXT_EXA_FAILED = "语义联网搜索暂时不可用，请稍后重试。"
_TEXT_EXA_EMPTY = "语义联网搜索没有返回结果。"

_TEXT_MISSING_QUERY = "缺少检索内容，请提供关键词或问题后重试。"


# --------------------------------------------------------------------------
# 懒加载单例
# --------------------------------------------------------------------------
#
# 为什么由本文件持有而不是放在数据库模块里：两个客户端的 __all__ 只导出常量与类，
# 没有模块级实例；而每次调用都新建 Neo4jClient 会重建 driver 与连接池，
# 一轮问答里同一个工具被调多次就会反复建连（Aura 上尤其慢）。
# 与 core._get_rag_pipeline 一样用"探测一次 + 双检锁"，构造失败也缓存：
# 构造只读配置、不发网络包（见 Neo4jClient.__init__ 的说明），没有任何自愈空间。

_client_lock = threading.Lock()
_neo4j_client = None
_neo4j_probed = False
_pg_client = None
_pg_probed = False
_rag_pipeline = None
_rag_probed = False


def _get_neo4j():
    """取图谱客户端单例；配置缺失或驱动初始化失败时返回 None（不抛）。"""
    global _neo4j_client, _neo4j_probed
    if _neo4j_probed:
        return _neo4j_client
    with _client_lock:
        if _neo4j_probed:
            return _neo4j_client
        _neo4j_probed = True
        try:
            from src.database.neo4j_client import Neo4jClient

            _neo4j_client = Neo4jClient()
        except Exception:
            logger.exception("知识图谱客户端不可用，图谱工具将走降级文案")
            _neo4j_client = None
    return _neo4j_client


def _get_postgres():
    """取业务库客户端单例；配置缺失或驱动初始化失败时返回 None（不抛）。"""
    global _pg_client, _pg_probed
    if _pg_probed:
        return _pg_client
    with _client_lock:
        if _pg_probed:
            return _pg_client
        _pg_probed = True
        try:
            from src.database.postgresql_client import PostgresClient

            _pg_client = PostgresClient()
        except Exception:
            logger.exception("业务数据库客户端不可用，数据库工具将走降级文案")
            _pg_client = None
    return _pg_client


def _get_rag_pipeline():
    """取 RAG 流水线单例；成员 B 尚未合入时返回 None（不抛）。

    取的是 `src.rag.pipeline.rag_pipeline` 这个**模块级对象**而不是工厂函数——
    `src/agent/core.py:83` 就是这么取的，两边必须一致，否则 B 按一处实现、这边按另一处调用。
    """
    global _rag_pipeline, _rag_probed
    if _rag_probed:
        return _rag_pipeline
    with _client_lock:
        if _rag_probed:
            return _rag_pipeline
        _rag_probed = True
        try:
            from src.rag.pipeline import rag_pipeline

            _rag_pipeline = rag_pipeline
        except Exception:
            logger.info("RAG 流水线尚未合入，知识库工具将走降级文案")
            _rag_pipeline = None
    return _rag_pipeline


# --------------------------------------------------------------------------
# 通用小工具
# --------------------------------------------------------------------------


def _pick(arguments, *names, default=""):
    """按候选名依次取第一个非空值（理由见模块 docstring「参数名为什么要容错」）。"""
    if not isinstance(arguments, dict):
        return default
    for name in names:
        value = arguments.get(name)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return default


def _as_text(value) -> str:
    """把任意值转成可放进文本的字符串；None 与空串统一成空串。"""
    if value is None:
        return ""
    return str(value).strip()


def _safe_int(value, default: int, maximum: int, minimum: int = 1) -> int:
    """把 LLM 给的条数收敛成合法整数：非法退回默认值，越界截断。

    与 `postgresql_client._safe_limit` 同一套路——外部数值不可信，字符串或浮点数
    直接塞进 LIMIT / Cypher 会被驱动拒绝，整条查询连"有没有结果"都问不出来。
    """
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(number, maximum))


def _truncate(text, limit: int) -> str:
    """按字符截断并补省略号；中文按字符计，不做字节换算。"""
    text = _as_text(text)
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


def _finalize(text: str) -> str:
    """所有格式化结果的统一出口：去空白 + 总量兜底截断。"""
    return _truncate(text, MAX_OBSERVATION_CHARS)


def _norm_score(value):
    """把分数收敛成 0-1 的 float；缺失 / None / NaN / 非数值一律返回 None。

    为什么不用 `float(value or 0)`：那样缺失会变成 0.0，与"算出来就是 0 分"混为一谈，
    前端与质量标签都无法区分。None 表示"这个来源没有分数"。

    为什么负数也夹到 0：分数会进 done 帧的 sources，前端按 0-1 渲染进度条，
    而精排模型吐出的原始分可能是负数（logits）。夹住范围比让前端拿到越界值安全。
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return max(0.0, min(number, 1.0))


def _fmt_amount(value) -> str:
    """金额显示成"1234567.00"，非数值原样返回（后端可能给 Decimal / None / 字符串）。"""
    if value is None:
        return "—"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return _as_text(value) or "—"


def _fmt_rows(rows) -> str:
    """兜底渲染：按行输出"键：值"，不假设任何固定列。

    存在的意义是**新增查询类型时不会渲染成空白**——格式化器是最后一道防线，
    宁可给出一份朴素但可读的列表，也不要因为列名没登记就丢掉整份结果。
    """
    lines = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        parts = [
            f"{_GRAPH_FIELD_LABELS.get(key, key)}：{_truncate(value, FIELD_TRUNCATE_CHARS)}"
            for key, value in row.items()
            if _as_text(value)
        ]
        if parts:
            lines.append("；".join(parts))
    if not lines:
        return ""
    return "\n".join(f"- {line}" for line in lines)


# --------------------------------------------------------------------------
# 格式化器
# --------------------------------------------------------------------------

_GRAPH_FIELD_LABELS = {
    "label": "类型",
    "name": "名称",
    "tradeFrequency": "交易频次",
    "subject": "标的物",
    "relation": "关系",
    "entity": "实体",
    "supplier": "供应商",
    "amount": "金额",
    "count": "数量",
    "id": "编号",
    "period": "时间",
    "cnt": "数量",
}

# 图谱节点标签与关系类型的展示名：英文枚举进不了给用户的正文
_GRAPH_LABEL_LABELS = {
    "SubjectMatter": "标的物",
    "Purchaser": "采购单位",
    "Supplier": "供应商",
    "Agency": "代理机构",
    "Location": "地区",
    "Transaction": "交易记录",
}
_GRAPH_RELATION_LABELS = {
    "PURCHASED_BY": "采购单位",
    "SUPPLIED_BY": "供应商",
    "AGENT_BY": "代理机构",
    "LOCATED_AT": "所在地",
    "HAS_TRANSACTION": "交易记录",
    "SELF": "标的物本身",
}

# entity_detail 的子列表字段 → 中文小标题（顺序即展示顺序）
_ENTITY_DETAIL_SECTIONS = (
    ("suppliers", "供应商"),
    ("purchasers", "采购单位"),
    ("agencies", "代理机构"),
    ("locations", "所在地"),
    ("transactions", "交易记录"),
)

# 业务库字段 → 中文表头。未登记的字段原样显示（见 _fmt_rows 的说明）
_PG_FIELD_LABELS = {
    "project_id": "项目编号",
    "project_name": "项目名称",
    "subject_matter": "标的物",
    "purchaser": "采购单位",
    "agency": "代理机构",
    "supplier": "供应商",
    "amount": "中标金额",
    "location_city": "所在地",
    "created_at": "时间",
    "bid_time": "中标时间",
    "publish_time": "发布时间",
    "period": "时间",
    "cnt": "数量",
    "avg_amount": "平均金额",
    "match_count": "命中记录数",
    "total_records": "记录总数",
    "total_amount": "中标金额合计",
    "min_bid_time": "最早中标时间",
    "max_bid_time": "最近中标时间",
    "distinct_subject": "标的物种类数",
    "distinct_purchaser": "采购单位数",
}

# 统计类字典的展示顺序（不依赖 dict 的插入顺序，保证同一份数据每次渲染一致）
_PG_STAT_ORDER = (
    "total_records",
    "total_amount",
    "distinct_subject",
    "distinct_purchaser",
    "min_bid_time",
    "max_bid_time",
)


def _fmt_rag(items, limit: int = RAG_DEFAULT_TOP_K) -> str:
    """知识库检索结果 → 编号资料列表（问 / 答 / 相关度）。

    同时把整体匹配度写进文本：`src/agent/react_loop.py` 的 `_validate_result` 只有
    图谱 / 业务库 / 联网三个质量标签分支，没有 RAG 的；在不改动他人文件的前提下，
    由格式化器自己给出提示，用户才知道这批资料是否值得引用。
    """
    blocks = []
    scores = []
    # 再切一刀 limit：检索侧已按 top_k 截过，但工具参数与检索参数未必同源，
    # 这里兜住"调用方给了个很大的 top_k"的情况
    for item in (items[:limit] if isinstance(items, list) else []):
        if not isinstance(item, dict):
            continue
        question = _as_text(item.get("question"))
        answer = _as_text(item.get("answer"))
        if not question and not answer:
            continue
        score = _norm_score(item.get("score"))
        if score is not None:
            scores.append(score)
        lines = [f"[{len(blocks) + 1}] 问：{_truncate(question, FIELD_TRUNCATE_CHARS)}"]
        lines.append(f"    答：{_truncate(answer, ANSWER_TRUNCATE_CHARS)}")
        if score is not None:
            lines.append(f"    相关度：{score:.2f}")
        blocks.append("\n".join(lines))

    if not blocks:
        return _TEXT_RAG_EMPTY

    header = f"知识库检索到 {len(blocks)} 条相关资料："
    if scores:
        average = sum(scores) / len(scores)
        header += "（匹配度良好，可直接引用）" if average >= 0.5 else "（匹配度一般，请结合其他信息判断）"
    return _finalize(header + "\n" + "\n".join(blocks))


def _fmt_graph_stats(data: dict) -> str:
    """图谱规模统计：`{标签: 数量}` → 中文计数列表。"""
    lines = []
    for label, count in data.items():
        name = _GRAPH_LABEL_LABELS.get(str(label), str(label))
        lines.append(f"- {name}：{count}")
    if not lines:
        return _TEXT_GRAPH_EMPTY
    return _finalize("知识图谱规模：\n" + "\n".join(lines))


def _fmt_graph_detail(data: dict) -> str:
    """标的物完整关系详情：按供应商 / 采购单位 / 代理机构 / 所在地 / 交易记录分段。"""
    subject = _as_text(data.get("subject"))
    blocks = [f"标的物「{subject}」的关联信息："] if subject else ["标的物关联信息："]
    for key, title in _ENTITY_DETAIL_SECTIONS:
        values = data.get(key)
        if not isinstance(values, list) or not values:
            continue
        rendered = []
        for value in values[:MAX_LIMIT]:
            if isinstance(value, dict):
                # 子项也是 dict（如交易记录 {id, amount}），复用 _fmt_rows 的"键：值"渲染
                rendered.append(_truncate("；".join(
                    f"{_GRAPH_FIELD_LABELS.get(key, key)}：{value[key]}"
                    for key in value
                    if _as_text(value[key])
                ), FIELD_TRUNCATE_CHARS))
            else:
                rendered.append(_as_text(value))
        rendered = [item for item in rendered if item]
        if rendered:
            blocks.append(f"【{title}】共 {len(values)} 项：\n" + "\n".join(f"  - {r}" for r in rendered))
    if len(blocks) == 1 and not subject:
        return _TEXT_GRAPH_EMPTY
    return _finalize("\n".join(blocks))


def _fmt_graph(query_type: str, data) -> str:
    """图谱查询结果 → 按查询类型分派的文本。空 / 异常形状一律给可读文案。"""
    if query_type == "graph_stats" and isinstance(data, dict):
        return _fmt_graph_stats(data)
    if query_type == "entity_detail" and isinstance(data, dict):
        return _fmt_graph_detail(data)

    rows = [row for row in data if isinstance(row, dict)] if isinstance(data, list) else []
    if not rows:
        return _TEXT_GRAPH_EMPTY

    if query_type == "top_traded":
        lines = [
            f"{index}. {_as_text(row.get('name'))}（交易频次 {_as_text(row.get('tradeFrequency')) or '—'}）"
            for index, row in enumerate(rows[:MAX_LIMIT], 1)
        ]
        return _finalize(f"采购频次最高的标的物（前 {len(lines)} 名）：\n" + "\n".join(lines))

    if query_type == "search_entity":
        lines = []
        for row in rows[:MAX_LIMIT]:
            label = _GRAPH_LABEL_LABELS.get(_as_text(row.get("label")), _as_text(row.get("label")) or "实体")
            frequency = _as_text(row.get("tradeFrequency"))
            suffix = f"，交易频次 {frequency}" if frequency else ""
            lines.append(f"- {_as_text(row.get('name'))}（{label}{suffix}）")
        return _finalize("匹配到的实体：\n" + "\n".join(lines))

    if query_type == "items_by_entity":
        lines = []
        for row in rows[:MAX_LIMIT]:
            relation = _GRAPH_RELATION_LABELS.get(_as_text(row.get("relation")), _as_text(row.get("relation")))
            lines.append(f"- {_as_text(row.get('subject'))}（关联方式：{relation}）")
        return _finalize("关联到的标的物：\n" + "\n".join(lines))

    if query_type == "supplier_by_subject":
        lines = [
            f"- {_as_text(row.get('supplier'))}（最高中标金额 {_fmt_amount(row.get('amount'))}）"
            for row in rows[:MAX_LIMIT]
        ]
        return _finalize("该标的物的供应商：\n" + "\n".join(lines))

    body = _fmt_rows(rows[:MAX_LIMIT])
    return _finalize(body) if body else _TEXT_GRAPH_EMPTY


def _fmt_pg(query_type: str, rows) -> str:
    """业务库查询结果 → 中文表头表格（前 20 行，字段截断）。

    两种形状都要认：绝大多数查询返回 `list[dict]`，而 `stats_overview` 返回单个 dict。
    """
    if isinstance(rows, dict):
        if not rows:
            return _TEXT_PG_EMPTY
        ordered = [key for key in _PG_STAT_ORDER if key in rows]
        ordered += [key for key in rows if key not in _PG_STAT_ORDER]
        lines = []
        for key in ordered:
            value = rows[key]
            if key == "total_amount":
                value = _fmt_amount(value)
            lines.append(f"- {_PG_FIELD_LABELS.get(key, key)}：{value}")
        return _finalize("业务数据库整体统计：\n" + "\n".join(lines))

    records = [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []
    if not records:
        return _TEXT_PG_EMPTY

    # 表头取所有行的键的保序并集：不同查询返回的列不一样，写死一份列清单会在
    # 新增查询时静默丢列
    columns: list[str] = []
    for row in records:
        for key in row:
            if key not in columns:
                columns.append(key)

    shown = records[:PG_DISPLAY_ROWS]
    header = " | ".join(_PG_FIELD_LABELS.get(key, key) for key in columns)
    lines = [header, "-" * max(8, len(header))]
    for row in shown:
        cells = []
        for key in columns:
            value = row.get(key)
            if key == "amount":
                value = _fmt_amount(value)
            cells.append(_truncate(value, PG_FIELD_TRUNCATE_CHARS))
        lines.append(" | ".join(cells))

    title = f"业务数据库查询结果（共 {len(records)} 条"
    title += f"，显示前 {len(shown)} 条）：" if len(records) > len(shown) else "）："
    return _finalize(title + "\n" + "\n".join(lines))


def _fmt_web(answer: str, items, limit: int = WEB_DISPLAY_LIMIT) -> str:
    """联网结果 → 摘要段 + 前 N 条（标题 / 链接 / 摘要）。

    摘要单独标注"AI 生成"：它是搜索服务对网页的二次概括，可能有过时或臆测成分，
    而用户往往直接引用。链接必须原文给出，供核对。
    """
    blocks = []
    answer = _as_text(answer)
    if answer:
        blocks.append("【联网摘要】" + _truncate(answer, ANSWER_TRUNCATE_CHARS))
        blocks.append("（摘要由搜索服务自动生成，引用请核对下方原始链接）")

    records = [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []
    lines = []
    for index, item in enumerate(records[:limit], 1):
        title = _as_text(item.get("title")) or _as_text(item.get("url")) or "（无标题）"
        url = _as_text(item.get("url"))
        content = _truncate(item.get("content"), WEB_SNIPPET_CHARS)
        block = f"[{index}] {title}"
        if url:
            block += f"\n    {url}"
        if content:
            block += f"\n    {content}"
        lines.append(block)

    if lines:
        blocks.append(f"联网搜索到 {len(records)} 条结果" + (f"，显示前 {len(lines)} 条：" if len(records) > len(lines) else "："))
        blocks.append("\n".join(lines))

    if not blocks:
        return _TEXT_WEB_EMPTY
    return _finalize("\n".join(blocks))


# --------------------------------------------------------------------------
# 联网结果重排
# --------------------------------------------------------------------------


def _rerank_web_results(question: str, items, top_k=None):
    """用精排模型按用户问题给联网结果打分并降序排列。

    对接 `src/rag/embedder.py` 的 `Reranker.rerank(query, candidates, top_k)`：
    候选必须是 `list[dict]`（它按 `text_key` 从字典里取值配对），返回的是**候选的副本**
    重排后的列表，每项带 `rerank_score`（原始 logit）与 `score`（sigmoid 后的 0-1）。
    它已经做完降序与归一化，这里**不再二次归一化**——min-max 会把"精排认为这批都不太相关"
    强行拉出一个满分 1.0，让用户以为首条很准。

    任何一步失败都**保持原顺序返回**：联网结果本身是可用的（引擎分照旧），
    因为精排器没就绪就把整批结果丢掉，是把可选优化当成了硬依赖。

    返回新列表，不原地改入参——上游的缓存会按引用共享同一份结果，
    就地改分数会污染后续请求（web_search 里已用深拷贝防了一层，这里再明确一次）。
    """
    records = [dict(item) for item in items if isinstance(item, dict)] if isinstance(items, list) else []
    if len(records) < 2:
        # 只有一条时排序没有意义，直接原样返回；此时沿用引擎分（Tavily 的相关度本就是 0-1），
        # 不要自作主张改成 1.0——那会把"引擎认为它不太相关"伪装成满分
        return records

    question = _as_text(question)
    if not question:
        # 没有可对照的问题就无从重排，保持引擎顺序与引擎分
        return records

    try:
        # 函数内导入：src.rag 依赖成员 D 的 src.config 等模块，未就绪时顶层导入会让
        # 五个工具一起消失。取的是**包级属性**（src/rag/__init__.py 导出）而非子模块——
        # 精排器住在 embedder.py 里，没有 src/rag/reranker.py 这个文件
        from src.rag import reranker
    except Exception:
        logger.info("精排模块尚未就绪，联网结果保持引擎顺序")
        return records

    # `_idx` 是回填锚点：对方用 `dict(item)` 造副本，对象身份会丢，只能靠自己塞的键认回来。
    # 配对文本给"标题 + 正文"而不是只给标题——招投标网页的标题常常只有项目名，
    # 判断相关性靠的其实是正文里的金额与采购单位
    candidates = [
        {"question": f"{_as_text(item.get('question'))} {_as_text(item.get('answer'))}".strip(), "_idx": index}
        for index, item in enumerate(records)
    ]
    try:
        # top_k 传满：要不要截断由调用方决定，少给几条不该由精排器替我们做主
        reranked = reranker.rerank(question, candidates, top_k=len(candidates))
    except Exception:
        logger.warning("联网结果重排失败，保留引擎顺序", exc_info=True)
        return records

    scored = []
    for entry in reranked if isinstance(reranked, list) else []:
        if not isinstance(entry, dict):
            continue
        index = entry.get("_idx")
        if not isinstance(index, int) or not 0 <= index < len(records):
            continue
        scored.append((index, _norm_score(entry.get("score"))))
    if not scored:
        logger.warning("精排返回的形状无法识别，保留引擎顺序")
        return records

    # 自己再排一次，不吃对方"已降序"的承诺。没有分数的排到最后——它不等于 0 分，
    # 排到"确实不相关"前面会误导；同分按下标稳定排序，保证同一批结果每次顺序一致
    scored.sort(key=lambda pair: (-(pair[1] if pair[1] is not None else -1.0), pair[0]))

    limit = _safe_int(top_k, default=len(scored), maximum=len(scored))
    picked = []
    for index, score in scored[:limit]:
        record = records[index]
        if score is not None:
            record["score"] = round(score, 4)
        picked.append(record)
    return picked


# --------------------------------------------------------------------------
# 来源列表
# --------------------------------------------------------------------------


def _rag_sources(items) -> list[dict]:
    """知识库来源：`{question, answer, score}`（见 docs/开发文档.md §6.2）。"""
    sources = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        question = _as_text(item.get("question"))
        answer = _as_text(item.get("answer"))
        if not question and not answer:
            continue
        sources.append({"question": question, "answer": answer, "score": _norm_score(item.get("score"))})
    return sources


def _web_sources(items) -> list[dict]:
    """联网来源：`{question, answer, score, url}`。

    `question` 承载标题（跨轮去重键是 `(question, answer, url)`，见 `react_loop._merge_sources`），
    **严格只放这四个键**——前端与 eval 若按元素精确比较，多一个键就会失配。
    """
    sources = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        url = _as_text(item.get("url"))
        title = _as_text(item.get("title"))
        if not url and not title:
            continue
        sources.append(
            {
                "question": title or url,
                "answer": _as_text(item.get("content")),
                "score": _norm_score(item.get("score")),
                "url": url,
            }
        )
    return sources


# --------------------------------------------------------------------------
# 执行器
# --------------------------------------------------------------------------


def search_knowledge_base(arguments: dict, question: str = "") -> tuple[str, list[dict]]:
    """检索招投标问答知识库（Qdrant 混合检索 + 精排）。"""
    arguments = arguments if isinstance(arguments, dict) else {}
    # 模型没给 query 时回落到用户原始问题：整句话是合法且通常更好的检索式
    query = _as_text(_pick(arguments, "query", "question", "keyword", "q", "text", "search_query")) or _as_text(question)
    if not query:
        return _TEXT_MISSING_QUERY, []
    top_k = _safe_int(_pick(arguments, "top_k", "topk", "topK", "limit", "k", "num"), RAG_DEFAULT_TOP_K, RAG_MAX_TOP_K)

    pipeline = _get_rag_pipeline()
    if pipeline is None:
        return _TEXT_RAG_UNAVAILABLE, []
    try:
        # 关键字传 top_k：成员 B 的签名是 search(question, top_k=5)
        items = pipeline.search(query, top_k=top_k)
    except Exception:
        logger.exception("知识库检索失败：%s", query)
        return _TEXT_RAG_FAILED, []

    records = [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []
    if not records:
        return _TEXT_RAG_EMPTY, []
    return _fmt_rag(records, top_k), _rag_sources(records)


def search_knowledge_graph(arguments: dict, question: str = "") -> tuple[str, list[dict]]:
    """查询知识图谱（标的物 / 采购单位 / 供应商 / 地区之间的关系与频次）。"""
    arguments = arguments if isinstance(arguments, dict) else {}
    query_type = _as_text(_pick(arguments, "query_type", "queryType", "type", "template"))
    if query_type not in _GRAPH_QUERY_TYPES:
        # 猜错或缺失时回落到模糊搜索：这是"用户问了个名字"最常见的意图，
        # 比直接回一句"不支持的查询类型"有用得多
        if query_type:
            logger.info("图谱查询类型无法识别（%s），回落到 search_entity", query_type)
        query_type = "search_entity"

    limit = _safe_int(_pick(arguments, "limit", "top_k", "topk", "count", "num"), GRAPH_DEFAULT_LIMIT, MAX_LIMIT)
    entity = _as_text(_pick(arguments, "entity", "keyword", "name", "subject", "query", "q", "text")) or _as_text(question)

    client = _get_neo4j()
    if client is None:
        return _TEXT_GRAPH_DOWN, []

    try:
        if query_type == "graph_stats":
            return _fmt_graph(query_type, client.graph_stats()), []
        if query_type == "top_traded":
            return _fmt_graph(query_type, client.top_traded(limit=limit)), []

        if not entity:
            return "请提供要查询的实体名称（标的物 / 采购单位 / 供应商 / 地区）。", []

        if query_type == "search_entity":
            data = client.search_entity(entity, limit=limit)
        elif query_type == "items_by_entity":
            data = client.items_by_entity(entity, limit=limit)
        elif query_type == "entity_detail":
            data = client.entity_detail(entity, limit=limit)
        else:  # supplier_by_subject
            data = client.supplier_by_subject(entity, limit=limit)
    except Exception:
        # 客户端的查询方法自己会吞异常返回空，走到这里说明是更外层的意外
        logger.exception("知识图谱查询失败：%s / %s", query_type, entity)
        return _TEXT_GRAPH_DOWN, []

    # 客户端对"没查到"与"没连上"都返回空，靠 health() 把两种情况分开报——
    # 只在空结果这条路上多花一次探活，正常路径不受影响
    if not data:
        status = client.health()
        return (_TEXT_GRAPH_DOWN if not status.get("ok") else _TEXT_GRAPH_EMPTY), []
    return _fmt_graph(query_type, data), []


def query_database(arguments: dict, question: str = "") -> tuple[str, list[dict]]:
    """查询业务数据库（中标金额、采购频次、地区分布、时间趋势等统计）。"""
    arguments = arguments if isinstance(arguments, dict) else {}
    query_type = _as_text(_pick(arguments, "query_type", "queryType", "type", "template"))
    if query_type not in _PG_QUERY_TYPES:
        if query_type:
            logger.info("数据库查询类型无法识别（%s），回落到 search_by_keyword", query_type)
        query_type = "search_by_keyword"

    limit = _safe_int(_pick(arguments, "limit", "top_k", "topk", "count", "num"), PG_DEFAULT_LIMIT, MAX_LIMIT)
    keyword = _as_text(_pick(arguments, "keyword", "query", "q", "text", "subject")) or _as_text(question)

    client = _get_postgres()
    if client is None:
        return _TEXT_PG_DOWN, []

    try:
        data, missing = _dispatch_pg(client, query_type, arguments, keyword, limit)
    except ValueError as exc:
        # 客户端对非法参数（金额写错、group_by 不支持）抛 ValueError，属**调用方**的错误：
        # 把原因回给 LLM，让它改参数重试，而不是报"服务不可用"
        logger.info("业务数据库参数不合法：%s", exc)
        return f"查询参数有误：{exc}", []
    except Exception:
        logger.exception("业务数据库查询失败：%s", query_type)
        return _TEXT_PG_DOWN, []

    if missing:
        return f"缺少参数：{missing}。请补充后重试。", []
    if not data:
        status = client.health()
        return (_TEXT_PG_DOWN if not status.get("ok") else _TEXT_PG_EMPTY), []
    return _fmt_pg(query_type, data), []


def _dispatch_pg(client, query_type: str, arguments: dict, keyword: str, limit: int):
    """按查询类型调用对应方法，返回 `(数据, 缺失的参数说明)`。

    返回二元组而不是直接返回数据：有些查询必须带业务参数（"谁的供应商"），
    缺参数时应该明确告诉 LLM 缺什么，而不是拿空字符串去查、再回一句"没查到"。
    """
    if query_type == "search_by_keyword":
        return (client.search_by_keyword(keyword, limit=limit), "") if keyword else ([], "keyword（关键词）")
    if query_type == "by_purchaser":
        value = _as_text(_pick(arguments, "purchaser", "buyer", "entity", "name", "keyword"))
        return (client.by_purchaser(value, limit=limit), "") if value else ([], "purchaser（采购单位名称）")
    if query_type == "by_supplier":
        value = _as_text(_pick(arguments, "supplier", "vendor", "entity", "name", "keyword"))
        return (client.by_supplier(value, limit=limit), "") if value else ([], "supplier（供应商名称）")
    if query_type == "avg_amount_by_subject":
        value = _as_text(_pick(arguments, "subject", "subject_matter", "entity", "name", "keyword"))
        return (client.avg_amount_by_subject(value, limit=limit), "") if value else ([], "subject（标的物名称）")
    if query_type == "amount_range":
        # 两个边界都缺时用全表默认区间没有意义，明确要求至少给一个
        low = _pick(arguments, "min_amount", "minAmount", "min", default=None)
        high = _pick(arguments, "max_amount", "maxAmount", "max", default=None)
        if low is None and high is None:
            return [], "min_amount 或 max_amount（金额区间，至少给一个）"
        return client.amount_range(min_amount=low, max_amount=high, limit=limit), ""
    if query_type == "trend":
        # 数据里只有 month / year 两种粒度（postgresql_client._TREND_EXPRESSIONS），
        # 别的值会被客户端拒绝，这里先收敛掉
        group_by = _as_text(_pick(arguments, "group_by", "groupBy", "granularity", "period")) or "month"
        if group_by not in ("month", "year"):
            logger.info("时间粒度 %s 不支持，回落到 month", group_by)
            group_by = "month"
        return client.trend(group_by=group_by, limit=limit), ""
    if query_type == "location_dist":
        return client.location_dist(limit=limit), ""
    if query_type == "purchaser_ranking":
        return client.purchaser_ranking(limit=limit), ""
    if query_type == "supplier_ranking":
        return client.supplier_ranking(limit=limit), ""
    if query_type == "top_subject":
        return client.top_subject(limit=limit), ""
    if query_type == "stats_overview":
        return client.stats_overview(), ""
    return [], ""  # 类型已在调用方校验过，兜底返回空


def search_web(arguments: dict, question: str = "") -> tuple[str, list[dict]]:
    """联网关键词搜索（Tavily），结果经精排后按用户问题排序。"""
    arguments = arguments if isinstance(arguments, dict) else {}
    query = _as_text(_pick(arguments, "query", "keyword", "q", "text", "search_query", "question")) or _as_text(question)
    if not query:
        return _TEXT_MISSING_QUERY, []

    # 延迟导入：让"没配密钥"这类纯配置问题不依赖于模块能否导入，
    # 也避免 web_search 的 import 失败连带影响其他四个工具
    try:
        from src import web_search
    except Exception:
        logger.exception("联网搜索模块导入失败")
        return _TEXT_WEB_FAILED, []

    if not web_search.is_configured():
        return _TEXT_WEB_NO_KEY, []

    max_results = _safe_int(
        _pick(arguments, "max_results", "maxResults", "limit", "count", "num"), WEB_DEFAULT_MAX_RESULTS, WEB_MAX_RESULTS
    )
    depth = _as_text(_pick(arguments, "depth", "search_depth")) or "basic"

    try:
        payload = web_search.web_search_client.search(query, max_results=max_results, depth=depth)
    except Exception:
        logger.exception("联网搜索失败：%s", query)
        return _TEXT_WEB_FAILED, []

    items = payload.get("results") if isinstance(payload, dict) else None
    records = _rerank_web_results(_as_text(question) or query, items)
    if not records:
        return _TEXT_WEB_EMPTY, []
    return _fmt_web(payload.get("answer", ""), records), _web_sources(records)


def search_exa(arguments: dict, question: str = "") -> tuple[str, list[dict]]:
    """语义联网搜索（Exa MCP）。

    `src/mcp/web_search_exa.py` 尚未落地（`src/mcp/` 目前只有一个空 `__init__.py`）。
    约定该模块导出模块级单例 `exa_search_client`，提供
    `search(query, num_results=N) -> {"results": [{title, url, content}], "answer": str}`。
    未就绪时本工具返回可读的降级文案，落地后无需改动本文件即可自动挂上。
    """
    arguments = arguments if isinstance(arguments, dict) else {}
    query = _as_text(_pick(arguments, "query", "keyword", "q", "text", "search_query", "question")) or _as_text(question)
    if not query:
        return _TEXT_MISSING_QUERY, []

    try:
        from src.mcp.web_search_exa import exa_search_client
    except Exception:
        logger.info("Exa MCP 客户端尚未就绪，语义联网搜索降级")
        return _TEXT_EXA_UNAVAILABLE, []

    num_results = _safe_int(
        _pick(arguments, "num_results", "numResults", "max_results", "limit", "count"), WEB_DEFAULT_MAX_RESULTS, WEB_MAX_RESULTS
    )
    try:
        payload = exa_search_client.search(query, num_results=num_results)
    except Exception:
        logger.exception("语义联网搜索失败：%s", query)
        return _TEXT_EXA_FAILED, []

    # 兼容两种返回形状：带 results 的 dict，或直接给列表
    items = payload.get("results") if isinstance(payload, dict) else payload
    records = _rerank_web_results(_as_text(question) or query, items)
    if not records:
        return _TEXT_EXA_EMPTY, []
    answer = payload.get("answer", "") if isinstance(payload, dict) else ""
    return _fmt_web(answer, records), _web_sources(records)


# 支持的路由图谱查询类型：与 Neo4jClient 的 6 个查询方法一一对应。
# 文档 §14 里的 list_entities 在客户端中没有对应方法，不收；supplier_by_subject 则是
# 客户端有、文档漏写的，按"以已合并代码为准"补上。
_GRAPH_QUERY_TYPES = (
    "search_entity",
    "items_by_entity",
    "top_traded",
    "entity_detail",
    "supplier_by_subject",
    "graph_stats",
)

# 业务库查询类型：与 PostgresClient 的 11 个查询方法一一对应。
# 文档 §6 的 6 个旧枚举（filter_by_field / aggregate_stats / top_by_amount ...）与实际方法
# 对不上，映射会有损，故直接以方法名为枚举值，1:1 路由。
_PG_QUERY_TYPES = (
    "search_by_keyword",
    "by_purchaser",
    "by_supplier",
    "amount_range",
    "location_dist",
    "trend",
    "purchaser_ranking",
    "supplier_ranking",
    "avg_amount_by_subject",
    "top_subject",
    "stats_overview",
)


# --------------------------------------------------------------------------
# 工具元信息（给 ToolRunner 的参数校验 / API 层的工具清单用）
# --------------------------------------------------------------------------


# 每个工具的执行器与元信息声明在一处，避免"加了工具忘了注册"或两处参数名漂移
_TOOL_SPECS = (
    {
        "name": "search_knowledge_base",
        "description": "招投标问答知识库（政策法规、办理流程、常见问题），优先使用",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "检索用的自然语言问题，保留用户原话中的关键实体与限定词",
                },
                "top_k": {"type": "integer", "description": "返回条数，默认 5，上限 20"},
            },
            "required": ["query"],
        },
        "executor": search_knowledge_base,
    },
    {
        "name": "search_knowledge_graph",
        "description": "企业与标的物、采购单位、供应商之间的关系查询（知识图谱）",
        "parameters": {
            "type": "object",
            "properties": {
                "query_type": {
                    "type": "string",
                    "enum": list(_GRAPH_QUERY_TYPES),
                    "description": (
                        "search_entity=模糊查找实体；items_by_entity=某单位/供应商关联的标的物；"
                        "top_traded=采购频次排行；entity_detail=某标的物的完整关系；"
                        "supplier_by_subject=某标的物的供应商与金额；graph_stats=图谱规模统计"
                    ),
                },
                "entity": {
                    "type": "string",
                    "description": "实体名称（标的物 / 采购单位 / 供应商 / 地区）",
                },
                "limit": {"type": "integer", "description": "返回条数，默认 20，上限 200"},
            },
            "required": ["query_type"],
        },
        "executor": search_knowledge_graph,
    },
    {
        "name": "query_database",
        "description": "结构化业务数据查询（中标金额、交易频次等统计）",
        "parameters": {
            "type": "object",
            "properties": {
                "query_type": {
                    "type": "string",
                    "enum": list(_PG_QUERY_TYPES),
                    "description": (
                        "search_by_keyword=关键词模糊搜索；by_purchaser/by_supplier=按采购单位/供应商查；"
                        "amount_range=按中标金额区间；location_dist=地区分布；trend=时间趋势（group_by 取 month/year）；"
                        "purchaser_ranking/supplier_ranking=排行；avg_amount_by_subject=某标的物平均金额；"
                        "top_subject=标的物频次排行；stats_overview=总览统计"
                    ),
                },
                "keyword": {"type": "string", "description": "关键词，search_by_keyword 用"},
                "purchaser": {"type": "string", "description": "采购单位名称，by_purchaser 用"},
                "supplier": {"type": "string", "description": "供应商名称，by_supplier 用"},
                "subject": {"type": "string", "description": "标的物名称，avg_amount_by_subject 用"},
                "min_amount": {"type": "number", "description": "最小中标金额（元），amount_range 用"},
                "max_amount": {"type": "number", "description": "最大中标金额（元），amount_range 用"},
                "group_by": {
                    "type": "string",
                    "enum": ["month", "year"],
                    "description": "trend 的聚合粒度，默认 month",
                },
                "limit": {"type": "integer", "description": "返回条数，默认 20，上限 200"},
            },
            "required": ["query_type"],
        },
        "executor": query_database,
    },
    {
        "name": "search_web",
        "description": "联网关键词搜索（最新公告、时效性强的信息）",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "联网搜索关键词，尽量具体并带上时间限定词（如「2025年 招标公告」）",
                },
                "max_results": {"type": "integer", "description": "抓取条数，默认 5，上限 10"},
                "depth": {
                    "type": "string",
                    "enum": ["basic", "advanced"],
                    "description": "basic 快而省（默认）；仅在 basic 结果不足时用 advanced",
                },
            },
            "required": ["query"],
        },
        "executor": search_web,
    },
    {
        "name": "search_exa",
        "description": "联网语义搜索（开放性、研究类问题）",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "语义搜索描述，写完整句子而非关键词，适合开放性、研究类问题",
                },
                "num_results": {"type": "integer", "description": "返回条数，默认 5，上限 10"},
            },
            "required": ["query"],
        },
        "executor": search_exa,
    },
)


def _build_tool(spec: dict) -> BaseTool:
    """按元信息动态产出一个 BaseTool 实例。

    为什么不手写五个子类：五个工具的差别**只在元信息**，执行逻辑都在上面的函数里。
    动态构造让"加工具"变成在 `_TOOL_SPECS` 里加一条，"注册进 TOOL_EXECUTORS"与
    "出现在工具清单里"由同一份数据派生，不可能出现两者不一致。
    """
    executor = spec["executor"]

    class _Tool(BaseTool):
        """由 `_TOOL_SPECS` 派生出的工具定义（参数校验 + 异常兜底由基类提供）。"""

        def run(self, **kwargs) -> ToolResult:
            """调用对应执行器；question 由 ToolRunner 注入，可能不存在。"""
            question = kwargs.pop("question", "")
            text, sources = executor(kwargs, question)
            # 与二元组契约保持一致：失败也走 success=True + 错误文本，
            # 两条调用路径（直接执行 / 经 ToolRunner）行为必须一样
            return ToolResult(success=True, data={"text": text, "sources": sources})

    _Tool.__name__ = f"Tool_{spec['name']}"
    return _Tool(name=spec["name"], description=spec["description"], parameters=spec["parameters"])


# 工具定义清单（供 API 层列出工具、ToolRunner 做参数校验）
TOOLS: list[BaseTool] = [_build_tool(spec) for spec in _TOOL_SPECS]

# 冻结契约：工具名 → 执行器。键名必须与 src/agent/constants.py 的
# BASE_TOOL_NAMES + WEB_TOOL_NAMES 完全一致（src/agent/core.py 与
# src/agent/react_loop.py 各持一份模块级绑定，只读 .get(name) / list(...)）。
# 五个键**无条件注册**：模块缺失时由执行器给出可读的降级文案，
# 不注册的话上层只会说"工具暂不可用（未注册或后端未就绪）"，分不清是哪一路没通。
TOOL_EXECUTORS = {spec["name"]: spec["executor"] for spec in _TOOL_SPECS}

__all__ = [
    "TOOLS",
    "TOOL_EXECUTORS",
    "search_knowledge_base",
    "search_knowledge_graph",
    "query_database",
    "search_web",
    "search_exa",
]

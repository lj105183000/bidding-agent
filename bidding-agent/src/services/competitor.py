"""竞争对手分析服务：某标的物 / 某采购人维度下的供应商竞争格局与价格水平。

要回答的四个问题
----------------
    ① 这个标的物上，谁经常中标？他们什么价位？      → suppliers_for_subject
    ② 这家供应商是什么来头？报价贵还是便宜？        → competitor_profile
    ③ 这家采购人把活给了谁？是不是就那么几家？      → purchaser_landscape
    ④ 这个市场是充分竞争还是高度集中？              → market_concentration

数据来源的分工（★ 这是本模块最关键的实现决定）
----------------------------------------------
**聚合统计一律以 PostgreSQL 的 `bidding_records` 为准，Neo4j 只做图谱侧补充。**

为什么不能让 Neo4j 承担统计职责——实测结论，不是猜测：

    MATCH (s:SubjectMatter)-[r:SUPPLIED_BY]->(u:Supplier) RETURN r LIMIT 3
      → 关系属性全部为空 {}；`CALL db.propertyKeys()` 只有
        amount / id / name / tradeFrequency

即图谱里的 `SUPPLIED_BY` 是**没有权重的存在性边**（实测 7968 条），
金额只挂在 `(:SubjectMatter)-[:HAS_TRANSACTION]->(:Transaction {amount})` 上，
而 `Transaction` 与 `Supplier` **之间没有任何关系**——无法把某笔金额归到某个供应商头上。
（`neo4j_client.supplier_by_subject` 取 `max(交易金额)` 作为供应商的代表金额，
 那是"该标的物下最大的那笔"，不是这家供应商自己的出价，两者不是一回事。）

所以：**谁中标几次、中位价多少、份额多少，只能从 `bidding_records` 逐条记录算**；
图谱负责回答"这个标的物关联了哪些地区与代理机构""这家供应商还投过哪些标的物"
这类**关系型**问题，且**失败即降级**（`GraphUnavailableError` 单独成类，
因为它与"业务库挂了"的处理方式完全不同：前者只是少一块补充信息，后者必须失败）。

金额统计为什么必须用中位数
--------------------------
沿用 `src/services/analysis.py` 的结论：实测 8746 条记录的金额横跨 **6 个数量级**
（办公用品几千元 ↔ 基建工程数亿元），均值是中位数的 **517 倍**，报出"均价"
会严重误导。故本模块**所有"典型价格"一律用中位数**，均值只作为对照字段返回，
并且必须与中位数并排展示，让人一眼看出偏态。

可疑记录
--------
直接 import 复用 `analysis.suspicious_sql()`，不重复实现——判据是"金额 + 品类"
的组合规则（服务类项目出现数亿元即判可疑），实测全库命中 1 条
（`智力、精神和重度残肢体疾人托养服务采购` 记成 66 亿，应为 660 万）。
默认排除出统计，但计数并回报，绝不静默丢弃。
"""

from __future__ import annotations

import math
import os
from typing import Any, Iterable

import psycopg
from dotenv import load_dotenv

from src.config import PROJECT_ROOT
from src.logging_config import get_logger
from src.services.analysis import MAX_LIMIT, suspicious_sql

logger = get_logger(__name__)

TABLE = "bidding_records"

# ---------------------------------------------------------------------------
# 模块级常量
# ---------------------------------------------------------------------------

# 返回条数默认值；上限复用 analysis.MAX_LIMIT（200），
# 免得同一份数据在两个服务里给出不同的截断行为。
DEFAULT_LIMIT = 20

# "取全部"用的护栏。仍需落在一个具体数字上——绝不写无 LIMIT 的裸查询。
# 实测本数据集：供应商 4151 家、标的物 2529 种、单标的物最多 175 条记录，
# 6 万足够覆盖任何单标的物的供应商全量，又不会把内存拉爆。
ALL_ROWS_LIMIT = 60_000

# 关键词最短长度：1 个汉字（"车""纸"）在 ILIKE '%%x%%' 下会命中几百种互不相干的
# 标的物，聚合出来的"市场集中度"是把若干无关市场掺在一起算的。
# 与其给一个看着像结论的错误数字，不如明确报错让调用方把词写具体。
MIN_KEYWORD_LENGTH = 2

# 一家供应商在某标的物上至少要有这么多条记录，它的"价格水平比值"才算数：
# 只有 1 条记录时，比值就是"这一笔 ÷ 市场中位"，那是运气而不是水平。
MIN_RECORDS_FOR_RATIO = 2

# 供应商的服务地区（location_city）最多列几个：实测有的供应商横跨十几个市，
# 全列会把响应体撑大，而用户真正关心的只是"主要在哪儿"。
MAX_REGIONS = 5

# 图谱侧一次取回的关系条数上限。图谱实测共 19336 个节点
# （8730 Transaction / 4138 Supplier / 3577 Purchaser / 2528 SubjectMatter
#   / 325 Agency / 38 Location），没有 LIMIT 的 MATCH 就是把整张图拖回来。
GRAPH_CONTEXT_LIMIT = 50

# 图谱关联标的物最多列几个（同上：只给概况，不做完整枚举）
MAX_GRAPH_SUBJECTS = 15

# HHI（赫芬达尔-赫希曼指数）= Σ(各主体市场份额百分比的平方)，取值 0~10000。
# 用"百分比的平方"而非"小数份额的平方"，是为了对齐反垄断实务中的通行读法
# （否则读出来是 0~1 的小数，没人能一眼判读）。
#
# 判定标准的来历：美国 DOJ/FTC 2010 年《横向并购指南》的分档——
#     < 1500        竞争充分（unconcentrated）
#     1500 ~ 2500   中等集中（moderately concentrated）
#     > 2500        高度集中（highly concentrated）
# 该阈值出自**并购审查**场景；招投标不是并购市场，故本模块只把它当
# "相对比较的标尺"（同一框架下比不同标的物谁更集中），不是合规判据。
HHI_MODERATE = 1500
HHI_HIGH = 2500

# 集中度档位 → 上界与标签（判定见 `_hhi_level`：取精度后 `值 <= 上界` 即命中该档）。
#
# ★ 规格（与 `_hhi_explanation()` 给前端展示的文字逐字一致）
# ----------------------------------------------------------
#     HHI <  1500                竞争充分
#     HHI 1500 ~ 2500（含两端）   中等集中
#     HHI >  2500                高度集中
#
# ★ 为什么第一档的上界是 `_JUST_BELOW_MODERATE` 而不是 HHI_MODERATE
# ------------------------------------------------------------------
# 因为 1500 必须归"中等集中"，而 2500 必须留在"中等集中"——两条边界的
# 归属方向相反，用"一个阈值既当上档上界又当下档下界"的直觉写法必然顾此失彼。
# 本次真实踩中了四次（前三次都在这一个函数上反复翻车）：
#   ① `(1500,"竞争充分")` + `hhi <= 1500` → 1500 被判成"竞争充分"
#   ② `(1500,"竞争充分")` + `hhi <  1500` → 2500 被判成"高度集中"，与前端文案矛盾
#   ③ 按直觉写区间 `[1500, 2500)`        → 2500 又跑到"高度集中"（坑换了张皮）
#   ④ 用等宽区间 `[lo, hi]` 配 `lo <= hhi <= hi` → 相邻档在 1500/2500 处重叠
#
# 现在把两条边界**分别**写清楚，代码里不再有任何需要推理的边界逻辑：
#     第一档 = 严格小于 1500（故上界取"小于 1500 的最大浮点数"）
#     第二档 = 小于等于 2500（故上界就是 2500，含端点）
# 配合 `_hhi_level` 里统一的一句 `value <= 上界`，两个边界同时正确。
_JUST_BELOW_MODERATE = math.nextafter(HHI_MODERATE, -math.inf)

HHI_LEVELS: tuple[tuple[float, str], ...] = (
    (_JUST_BELOW_MODERATE, "竞争充分"),  # HHI <  1500
    (HHI_HIGH, "中等集中"),              # 1500 <= HHI <= 2500
    (float("inf"), "高度集中"),          # HHI >  2500
)

# HHI 的展示精度（小数位）。`_concentration` 按此精度四舍五入，
# `_hhi_level` 判定前也先取到同一精度——两者必须一致，否则会出现
# "界面显示 2500.0 却判成高度集中"这种自相矛盾的输出。
HHI_DECIMALS = 1

# 集中度榜单看前几名
CR3_N = 3
CR5_N = 5


class CompetitorError(RuntimeError):
    """竞争对手分析失败（数据库不可用 / 参数不合法 / 无匹配数据）。

    对外**只抛这一种异常**：psycopg 与 neo4j 的原生异常一律在这里收敛。
    上层（API 路由）因此不必知道底层用的是哪个库，也不会把驱动报文里的
    主机名、端口、SQL 泄露给用户。
    """


class GraphUnavailableError(CompetitorError):
    """知识图谱不可用（未配置 NEO4J_* / 连不上 / 查询报错）。

    单独分一个子类，是因为它与"业务库挂了"应当被区别对待：
    **图谱只是补充信息，不可用时降级继续出结果**，而不是让整个请求失败。
    调用方用 `except GraphUnavailableError` 把 graph_enriched 标成 False 即可。
    """


# ---------------------------------------------------------------------------
# 连接与基础设施
# ---------------------------------------------------------------------------


def _dsn() -> dict[str, Any]:
    """从 .env 组装 PostgreSQL 连接参数（**密码绝不硬编码**）。

    与 analysis.py 同一实现，含一个来自实测的细节：Windows 上用手写/编辑器
    保存的 .env，密码行尾常带 `\\r`，不 rstrip 掉就会报"密码验证失败"。
    """
    load_dotenv(PROJECT_ROOT / ".env")
    host = (os.getenv("POSTGRES_HOST") or "").strip()
    if not host:
        raise CompetitorError("未配置 PostgreSQL（POSTGRES_HOST 为空）")
    return {
        "host": host,
        "port": (os.getenv("POSTGRES_PORT") or "5432").strip(),
        "user": (os.getenv("POSTGRES_USER") or "postgres").strip(),
        # 密码末尾可能带 \r（Windows 编辑器保存 .env 时常见），必须清掉
        "password": (os.getenv("POSTGRES_PASSWORD") or "").rstrip("\r\n"),
        "dbname": (os.getenv("POSTGRES_DB") or "chatbot").strip(),
    }


class _Cursor:
    """轻量上下文管理器：连库 → 用 → 必关（与 analysis.py 同款）。

    为什么不用 psycopg 的 `with psycopg.connect() as conn`：那个写法管的是
    **事务**（正常退出提交、异常回滚），连接本身在退出时并不关闭。本服务是
    长跑的 Web 后端，查询又全是只读，连接必须显式 close 才不会把连接数耗光。
    """

    def __enter__(self):
        try:
            self._conn = psycopg.connect(**_dsn())
        except Exception as exc:
            raise CompetitorError(f"数据库连接失败：{type(exc).__name__}") from exc
        self._cur = self._conn.cursor()
        return self._cur

    def __exit__(self, *exc_info):
        try:
            self._conn.close()
        except Exception:  # pragma: no cover
            pass
        return False


def _neo4j_driver():
    """构造 Neo4j driver；配置缺失或初始化失败一律抛 GraphUnavailableError。

    driver 本身完全不发网络包（只准备配置与连接池），所以这里失败只可能是
    "配置缺失"或"URI 协议写错"，不会是"服务没起来"——后者在查询时才暴露。
    """
    load_dotenv(PROJECT_ROOT / ".env")
    uri = (os.getenv("NEO4J_URI") or "").strip()
    user = (os.getenv("NEO4J_USERNAME") or "").strip()
    # 同 _dsn：密码只去行尾换行，首位空格可能是密码本身
    password = (os.getenv("NEO4J_PASSWORD") or "").rstrip("\r\n")
    missing = [
        name
        for name, value in (
            ("NEO4J_URI", uri),
            ("NEO4J_USERNAME", user),
            ("NEO4J_PASSWORD", password),
        )
        if not value
    ]
    if missing:
        raise GraphUnavailableError(f"未配置知识图谱（缺少 {'、'.join(missing)}）")
    try:
        from neo4j import GraphDatabase

        return GraphDatabase.driver(uri, auth=(user, password))
    except Exception as exc:
        raise GraphUnavailableError(f"知识图谱驱动初始化失败：{type(exc).__name__}") from exc


class _GraphSession:
    """Neo4j 只读会话的上下文管理器：开 driver → 开会话 → **两层都关**。

    与 `_Cursor` 同样是显式类而非 @contextmanager 生成器：退出路径看得见，
    且每个 close 各自 try，关闭时的异常不会盖掉真正的业务异常。
    """

    def __enter__(self):
        self._driver = _neo4j_driver()
        # dotenv 把 `NEO4J_DATABASE=` 读成空串而不是 None，空串会被服务端当成
        # "名为空的数据库"直接报错，故统一收敛为 None（= 用 driver 默认库）
        database = (os.getenv("NEO4J_DATABASE") or "").strip() or None
        try:
            self._session = self._driver.session(database=database)
        except Exception as exc:
            self._close_driver()
            raise GraphUnavailableError(f"知识图谱会话创建失败：{type(exc).__name__}") from exc
        return self._session

    def _close_driver(self):
        try:
            self._driver.close()
        except Exception:  # pragma: no cover
            pass

    def __exit__(self, *exc_info):
        try:
            self._session.close()
        except Exception:  # pragma: no cover
            pass
        self._close_driver()
        return False


def _query(cypher: str, **params: Any) -> list[dict]:
    """跑一条 Cypher 并返回 `list[dict]`；任何失败都收敛成 GraphUnavailableError。

    值一律走 `$name` 占位符经 `**params` 交给驱动——Cypher 的**标签、关系类型、
    属性名都不能参数化**，本模块把它们全部写死在语句字面量里（不接受外部传入），
    需要拼进标识符的地方因此没有注入面。

    **每条语句都必须自带 LIMIT**（本模块的调用方无一例外）：图谱有 19336 个节点，
    一条没有 LIMIT 的 MATCH 会把整张图拉回来。
    """
    try:
        with _GraphSession() as session:
            return [dict(record) for record in session.run(cypher, **params)]
    except GraphUnavailableError:
        raise
    except Exception as exc:
        # 驱动异常里带主机名与端口，只进日志；对外只给异常类型
        logger.warning("知识图谱查询失败：%s", exc)
        raise GraphUnavailableError(f"知识图谱查询失败：{type(exc).__name__}") from exc


def _query_one(cypher: str, **params: Any) -> dict | None:
    """跑一条 Cypher 取首行（找不到返回 None）。LIMIT 由调用方写在语句里。"""
    rows = _query(cypher, **params)
    return rows[0] if rows else None


def _execute(cur, sql: str, params: tuple = ()) -> None:
    """执行 SQL 并把驱动异常收敛成 CompetitorError。

    与 `_Cursor.__enter__` 的分工：那里管"连不上"，这里管"连上了但语句失败"
    （表不存在、列名写错、超时）。两者都不能把 psycopg 的原生异常抛给上层。
    """
    try:
        cur.execute(sql, params)
    except CompetitorError:
        raise
    except Exception as exc:
        logger.warning("SQL 执行失败：%s", exc)
        raise CompetitorError(f"查询失败：{type(exc).__name__}") from exc


# ---------------------------------------------------------------------------
# 参数与数值工具
# ---------------------------------------------------------------------------


def _clamp_limit(limit: int) -> int:
    """把 limit 收敛到 [1, MAX_LIMIT]；非法值退回 DEFAULT_LIMIT。"""
    try:
        n = int(limit)
    except (TypeError, ValueError):
        n = DEFAULT_LIMIT
    return max(1, min(MAX_LIMIT, n))


def _norm_keyword(value: str, label: str) -> str:
    """校验并归一化查询关键词（去首尾空白 + 最短长度校验）。"""
    keyword = (value or "").strip()
    if not keyword:
        raise CompetitorError(f"请提供{label}")
    if len(keyword) < MIN_KEYWORD_LENGTH:
        raise CompetitorError(
            f"{label}至少需要 {MIN_KEYWORD_LENGTH} 个字符（当前 {keyword!r} 过于宽泛，"
            "会把互不相干的市场混在一起统计）"
        )
    return keyword


def _f(value: Any) -> float | None:
    """数值归一化：None 保持 None，Decimal / int 统一成 float。

    为什么不把 None 折成 0：本数据里 amount 无空值，但"没有数据"与"金额是 0"
    在展示上完全是两回事，不能让前端把缺失画成 0 元。
    """
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):  # pragma: no cover - numeric 列不会走到
        return None


def _dedupe(values: Iterable[Any], limit: int | None = None) -> list[Any]:
    """保序去重（跳过 None 与空串）。地区/代理机构这类多行同值的场景要用。"""
    seen: set[Any] = set()
    out: list[Any] = []
    for value in values:
        if value is None or value == "":
            continue
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
        if limit is not None and len(out) >= limit:
            break
    return out


def _iso(value: Any) -> str | None:
    """时间戳 → ISO 字符串；None 保持 None（JSON 友好）。"""
    return value.isoformat() if value is not None else None


def _iso_range(row, offset: int = 0) -> dict[str, Any]:
    """把 `(min_time, max_time, months)` 三元组转成 JSON 友好的时间范围。"""
    return {
        "min": _iso(row[offset]),
        "max": _iso(row[offset + 1]),
        "months": int(row[offset + 2] or 0),
    }


def _base_amount_filter(exclude_suspicious: bool, alias: str = "") -> str:
    """金额有效性条件：`amount > 0`，可选再排除可疑记录。

    与 `analysis._amount_filter` 判据一致，但**不 import 它**：那是带下划线的
    模块私有函数，跨模块依赖私有名会在对方重构时无声断掉；而 `suspicious_sql()`
    是 analysis.py 在 `__all__` 里公开承诺的契约，真正的风险（脏数据判据漂移）
    由复用它来消除，这就够了。

    `alias` 供需要自连接的查询使用（本模块在 competitor_profile 里用 `s` 前缀），
    因此返回的是一整段可直接拼进 WHERE 的条件，由调用方负责补上自己的限定条件。
    """
    col = f"{alias}.amount" if alias else "amount"
    subj = f"{alias}.subject_matter" if alias else "subject_matter"
    condition = f"{col} IS NOT NULL AND {col} > 0"
    if exclude_suspicious:
        condition += f" AND NOT {suspicious_sql(col, subj)}"
    return condition


def _add_param(params: tuple, value: Any) -> tuple:
    """往参数元组末尾追加一个值并返回新元组。

    存在的意义是让"条件片段"与"参数"能分开拼装：本模块多处需要根据
    （精确匹配 / 模糊匹配）切换 WHERE 片段，但参数必须严格按出现顺序排列，
    用这个小函数拼装比在每个分支里手工对齐问号安全。
    """
    return params + (value,)


# ---------------------------------------------------------------------------
# 图谱侧补充（失败即降级，绝不让整个分析失败）
# ---------------------------------------------------------------------------


def _graph_subject_context(subject: str) -> dict[str, Any] | None:
    """标的物在图谱里的上下文：交易笔数、关联地区、关联代理机构。

    只在图谱里**精确匹配** `toLower(name) = toLower($subject)`——图谱里的标的物
    是归一化过的独立实体（实测 2528 个，如"空调""粉盒"），PG 里的 subject_matter
    则是逐条记录的原文，两者粒度不同。用 CONTAINS 去凑关键词会把"空调"与
    "空调维修"混为一谈，**宁可匹配不上也不给错信息**（调用方会拿到 None 并提示）。

    返回 None 表示"图谱里没有这个名字的节点"（与"图谱不可用"区分开：
    后者抛 GraphUnavailableError）。
    """
    node = _query_one(
        "MATCH (s:SubjectMatter) WHERE toLower(s.name) = toLower($subject) "
        "RETURN s.name AS name, s.tradeFrequency AS tradeFrequency LIMIT 1",
        subject=subject,
    )
    if not node:
        return None
    # 地区：一个标的物可关联多个 Location，必须 LIMIT
    locations = _query(
        "MATCH (s:SubjectMatter {name: $subject})-[:LOCATED_AT]->(l:Location) "
        "RETURN l.name AS name LIMIT $limit",
        subject=node.get("name"),
        limit=GRAPH_CONTEXT_LIMIT,
    )
    agencies = _query(
        "MATCH (s:SubjectMatter {name: $subject})-[:AGENT_BY]->(a:Agency) "
        "RETURN a.name AS name LIMIT $limit",
        subject=node.get("name"),
        limit=GRAPH_CONTEXT_LIMIT,
    )
    return {
        "graph_name": node.get("name"),
        # tradeFrequency 是图谱侧独立维护的采购频次，与 PG 的逐条计数可能不一致，
        # 故作为补充字段返回而不是覆盖 PG 的统计值
        "trade_frequency": node.get("tradeFrequency"),
        "regions": _dedupe([row.get("name") for row in locations], MAX_REGIONS),
        "agencies": _dedupe([row.get("name") for row in agencies], MAX_REGIONS),
    }


def _graph_supplier_context(supplier_name: str) -> dict[str, Any] | None:
    """供应商在图谱里的上下文：关联标的物、服务地区、代理机构。

    `SUPPLIED_BY` 的边没有金额属性（见模块 docstring），所以这里给出的**只是
    "还投过哪些标的物"**，不参与任何金额计算——避免把图谱的"存在性"误当成"价格"。
    """
    node = _query_one(
        "MATCH (u:Supplier) WHERE toLower(u.name) = toLower($supplier) "
        "RETURN u.name AS name LIMIT 1",
        supplier=supplier_name,
    )
    if not node:
        return None
    name = node.get("name")
    subjects = _query(
        "MATCH (s:SubjectMatter)-[:SUPPLIED_BY]->(u:Supplier {name: $supplier}) "
        "RETURN s.name AS name LIMIT $limit",
        supplier=name,
        limit=GRAPH_CONTEXT_LIMIT,
    )
    agencies = _query(
        "MATCH (u:Supplier {name: $supplier})<-[:SUPPLIED_BY]-(s:SubjectMatter)"
        "-[:AGENT_BY]->(a:Agency) "
        "RETURN DISTINCT a.name AS name LIMIT $limit",
        supplier=name,
        limit=GRAPH_CONTEXT_LIMIT,
    )
    regions = _query(
        "MATCH (u:Supplier {name: $supplier})<-[:SUPPLIED_BY]-(s:SubjectMatter)"
        "-[:LOCATED_AT]->(l:Location) "
        "RETURN DISTINCT l.name AS name LIMIT $limit",
        supplier=name,
        limit=GRAPH_CONTEXT_LIMIT,
    )
    return {
        "graph_name": name,
        "subjects": _dedupe(
            [row.get("name") for row in subjects], MAX_GRAPH_SUBJECTS
        ),
        "agencies": _dedupe([row.get("name") for row in agencies], MAX_REGIONS),
        "regions": _dedupe([row.get("name") for row in regions], MAX_REGIONS),
    }


def _graph_purchaser_context(purchaser_name: str) -> dict[str, Any] | None:
    """采购人在图谱里的上下文：关联标的物、代理机构。"""
    node = _query_one(
        "MATCH (p:Purchaser) WHERE toLower(p.name) = toLower($purchaser) "
        "RETURN p.name AS name LIMIT 1",
        purchaser=purchaser_name,
    )
    if not node:
        return None
    name = node.get("name")
    subjects = _query(
        "MATCH (s:SubjectMatter)-[:PURCHASED_BY]->(p:Purchaser {name: $purchaser}) "
        "RETURN DISTINCT s.name AS name LIMIT $limit",
        purchaser=name,
        limit=GRAPH_CONTEXT_LIMIT,
    )
    agencies = _query(
        "MATCH (p:Purchaser {name: $purchaser})<-[:PURCHASED_BY]-(s:SubjectMatter)"
        "-[:AGENT_BY]->(a:Agency) "
        "RETURN DISTINCT a.name AS name LIMIT $limit",
        purchaser=name,
        limit=GRAPH_CONTEXT_LIMIT,
    )
    return {
        "graph_name": name,
        "subjects": _dedupe([row.get("name") for row in subjects], MAX_GRAPH_SUBJECTS),
        "agencies": _dedupe([row.get("name") for row in agencies], MAX_REGIONS),
    }


def _count_suspicious(cur, **likes: str) -> int:
    """数本次分析范围内有多少条可疑记录（用于如实回报，而非静默丢弃）。

    `**likes` 的键必须是 `subject_matter` / `purchaser` / `supplier` 之一，
    值按 ILIKE 模糊匹配。列名来自本函数自己的白名单，不接受外部传入。
    """
    allowed = ("subject_matter", "purchaser", "supplier")
    where = [suspicious_sql()]
    params: tuple = ()
    for column in allowed:
        value = likes.get(column)
        if value is not None:
            where.append(f"{column} ILIKE %s")
            params = _add_param(params, value)
    _execute(
        cur,
        f"SELECT count(*) FROM {TABLE} WHERE {' AND '.join(where)}",
        params,
    )
    return int(cur.fetchone()[0])


def _coverage_caveat(time_range: dict[str, Any]) -> str | None:
    """时间覆盖面的提示；覆盖充足时返回 None。

    ★ 为什么每个公开函数都必须带上它：实测本数据集**只覆盖 2023-10 一个月**
    （见 analysis.overview 的 data_caveats，同一份数据同一个结论）。
    在单月切片上，"某供应商最近在发力""这个市场正在集中"这类**趋势性**判断
    全部不成立——而那些正是用户看到"首次/最近中标时间"后最容易自行脑补的结论。
    提示必须跟着数据走，而不是指望用户记得去看 analysis 那一页。
    """
    if not time_range:
        return None
    months = time_range.get("months")
    if months is None:
        return None
    if months <= 1:
        return (
            f"数据仅覆盖 {months} 个月（{str(time_range.get('min'))[:7]} 前后），"
            "**无法做时间趋势分析**：本页的『首次/最近中标时间』只反映该月内的先后，"
            "不代表供应商的进入或退出趋势"
        )
    return None


def _time_range(cur, extra_where: str, params: tuple) -> dict[str, Any]:
    """返回 `{"min","max","months"}`：最早/最近中标时间与覆盖月数。

    月数一并返回是有原因的：本数据集实测**只覆盖 2023-10 一个月**，
    任何"趋势"结论都无从谈起，必须把这一点带进响应，否则用户会读出不存在的趋势。
    """
    _execute(
        cur,
        f"SELECT min(bid_time), max(bid_time), "
        f"count(DISTINCT to_char(bid_time, 'YYYY-MM')) "
        f"FROM {TABLE} WHERE {extra_where}",
        params,
    )
    return _iso_range(cur.fetchone())


# ---------------------------------------------------------------------------
# 集中度计算
# ---------------------------------------------------------------------------


def _concentration(shares: dict[str, float]) -> tuple[float, float | None, float | None, list[dict]]:
    """按份额字典算 HHI / CR3 / CR5，返回 `(hhi, cr3, cr5, 前 5 名明细)`。

    `shares` 是 `{供应商: 份额百分比}`，**不要求合计恰为 100**：数据里存在
    "有记录没有供应商名"的情况，强行归一化会把这个事实掩盖掉。
    HHI 按定义用 Σ(百分比²) 计算，所以份额之和偏离 100 时 HHI 会系统性偏低——
    这不是 bug，是"份额未覆盖全量"的诚实反映，由调用方在 data_caveats 里说明。
    """
    ordered = sorted(shares.items(), key=lambda kv: (-kv[1], kv[0]))
    # 精度用 HHI_DECIMALS：`_hhi_level` 判定前也取到同一精度，
    # 保证"界面显示 2500.0 + 档位是高度集中"这种自相矛盾的输出不会出现
    hhi = round(sum(share * share for share in shares.values()), HHI_DECIMALS)
    total3 = sum(share for _, share in ordered[:CR3_N])
    total5 = sum(share for _, share in ordered[:CR5_N])
    leaders = [
        {"name": name, "share_percent": round(share, 2)} for name, share in ordered[:CR5_N]
    ]
    return (
        hhi,
        round(total3, 1) if ordered else None,
        round(total5, 1) if ordered else None,
        leaders,
    )


def _hhi_level(hhi: float) -> str:
    """HHI → 中文档位（规格与踩过的坑见 HHI_LEVELS 上方的注释）。

    先按 `HHI_DECIMALS` 取整再比阈值，保证**档位与用户看到的数字一致**：
    `_concentration` 也用同一精度，所以界面上写 2500.0 时档位绝不会是
    "高度集中"。取整后规格就是最朴素的 `<= 上界`，两条边界都落在
    "中等集中"，与 `_hhi_explanation()` 展示的 "1500 ~ 2500" 逐字对得上：

        _hhi_level(1499.9) → 竞争充分     _hhi_level(1500.0) → 中等集中
        _hhi_level(2500.0) → 中等集中     _hhi_level(2500.1) → 高度集中

    边界由 `tests/test_competitor.py::test_hhi_level_boundaries` 守住。
    """
    value = round(hhi, HHI_DECIMALS)
    for threshold, label in HHI_LEVELS:
        if value <= threshold:
            return label
    return HHI_LEVELS[-1][1]  # pragma: no cover - 最后一档上界是 inf，恒真


def _hhi_explanation() -> dict[str, Any]:
    """HHI 判定标准的可读说明（前端直接展示，用户不必去查资料）。

    为什么要把"标准"随"数据"一起返回：HHI 是个没有量纲直觉的数字——
    单看 3200 没人知道算高还是低。把分档与逐档含义一并给出，用户才能自己判读，
    而不是只能相信我们给的"高度集中"四个字。
    """
    return {
        "formula": "HHI = Σ(各供应商市场份额百分比的平方)，取值 0~10000",
        "thresholds": [
            {
                "range": f"< {HHI_MODERATE}",
                "level": "竞争充分",
                "meaning": "供应商众多、无人主导，价格通常接近成本",
            },
            {
                "range": f"{HHI_MODERATE} ~ {HHI_HIGH}",
                "level": "中等集中",
                "meaning": "头部几家有影响力，但仍有新进入者空间",
            },
            {
                "range": f"> {HHI_HIGH}",
                "level": "高度集中",
                "meaning": "少数供应商主导、议价能力强，需关注是否存在围标或长期绑定",
            },
        ],
        "caveat": (
            "该分档出自美国 DOJ/FTC 横向并购指南，本意是并购审查；"
            "招投标不是并购市场，此处仅作为不同标的物之间**相对比较**的标尺，"
            "不是合规判据"
        ),
    }


def _price_level(index: float | None) -> str:
    """价格指数 → 中文档位。

    分档只为给出一个可读的定性标签，**阈值故意宽松**（±3% 内算持平）：
    招投标的报价差异本就受规格、配置、服务期影响，把 1.02 说成"偏贵"
    会让用户做出过度反应——定性标签不该显得比数据本身更精确。
    """
    if index is None:
        return "无法判定"
    if index < 0.9:
        return "明显偏低"
    if index < 0.97:
        return "略低于同行"
    if index <= 1.03:
        return "与同行持平"
    if index <= 1.1:
        return "略高于同行"
    return "明显偏高"


# ---------------------------------------------------------------------------
# ① 某标的物的供应商竞争格局
# ---------------------------------------------------------------------------


def suppliers_for_subject(
    subject: str,
    limit: int = DEFAULT_LIMIT,
    exclude_suspicious: bool = True,
) -> dict[str, Any]:
    """某标的物的供应商竞争格局：谁经常中标、什么价位、覆盖哪些地区。

    这是"我要投这个标，对手是谁、该报多少"的核心数据源。

    每个供应商都带**金额中位数**而不是均值——同一标的物下的报价同样可能跨数量级
    （"空调"实测从 ¥1,800 到 ¥108,625），均值会被少数大单带偏。

    参数
    ----
    subject            标的物关键词，按 `ILIKE %keyword%` 模糊匹配（至少 2 个字符）
    limit              返回的供应商条数，收敛到 1~200，默认 20
    exclude_suspicious 是否排除金额可疑的记录（默认排除，判据见 analysis.suspicious_sql）

    `win_count` 与 `count` 是同一个值（中标次数）：`count` 对齐
    `analysis.group_ranking` 的字段名便于前端复用表格组件，`win_count` 在竞争
    语境下语义更明确，两个都给，前端用哪个都行。

    返回的 `market` 是整个标的物的行情（条数 / 中位数 / P25 / P75 / 均值 / 合计），
    供应商名单要有"在多大的盘子里"才有意义，故两者同一次调用一起给。
    """
    keyword = _norm_keyword(subject, "标的物关键词")
    n = _clamp_limit(limit)
    like = f"%{keyword}%"
    where = _base_amount_filter(exclude_suspicious)

    with _Cursor() as cur:
        # 市场规模先行
        _execute(
            cur,
            f"""
            SELECT
                count(*),
                percentile_cont(0.5) WITHIN GROUP (ORDER BY amount),
                percentile_cont(0.25) WITHIN GROUP (ORDER BY amount),
                percentile_cont(0.75) WITHIN GROUP (ORDER BY amount),
                avg(amount),
                sum(amount)
            FROM {TABLE}
            WHERE {where} AND subject_matter ILIKE %s
            """,
            (like,),
        )
        row = cur.fetchone()
        total_count = int(row[0])
        suspicious = _count_suspicious(cur, subject_matter=like)

        if total_count == 0:
            # 这一支有一个必须说清楚的情形：命中记录**全被判为可疑**而排除干净。
            # 此时 matched=False 是事实，但不能让用户以为"数据里根本没有这个标的物"——
            # 得告诉他被排除了几条、怎么把它们调出来看。
            hint = (
                f"。该关键词命中的 {suspicious} 条记录金额全部被判定为可疑（服务类项目"
                "出现数亿元），故排除后无可用记录；可置 exclude_suspicious=False 复看原始数据"
                if suspicious and exclude_suspicious
                else ""
            )
            return {
                "subject": keyword,
                "matched": False,
                "excluded_suspicious": exclude_suspicious,
                "suspicious_count": suspicious,
                "message": f"未找到与「{keyword}」匹配的中标记录{hint}",
            }

        market = {
            "count": total_count,
            "median_amount": _f(row[1]),
            "p25": _f(row[2]),
            "p75": _f(row[3]),
            "mean_amount": _f(row[4]),
            "total_amount": _f(row[5]),
        }
        market["mean_over_median"] = (
            round(market["mean_amount"] / market["median_amount"], 1)
            if market["mean_amount"] and market["median_amount"]
            else None
        )

        # 每个供应商：次数 / 中位价 / 合计 / 首末中标时间。
        # 地区单独查而不是 JOIN 进来——JOIN 会把行数放大（供应商 × 记录 × 地区），
        # 一旦在同一层做 percentile_cont，中位数就跟着算错了。
        _execute(
            cur,
            f"""
            SELECT
                supplier,
                count(*) AS win_count,
                percentile_cont(0.5) WITHIN GROUP (ORDER BY amount) AS median_amount,
                sum(amount) AS total_amount,
                min(bid_time) AS first_win,
                max(bid_time) AS last_win,
                avg(amount) AS mean_amount
            FROM {TABLE}
            WHERE {where} AND subject_matter ILIKE %s
              AND supplier IS NOT NULL AND supplier <> ''
            GROUP BY supplier
            ORDER BY win_count DESC, total_amount DESC NULLS LAST
            LIMIT %s
            """,
            (like, n),
        )
        rows = cur.fetchall()

        # 服务地区：只查上榜的这几家（= ANY(数组)），避免为全表做一次大聚合
        names = [r[0] for r in rows]
        regions: dict[str, list[str]] = {}
        if names:
            _execute(
                cur,
                f"""
                SELECT supplier, location_city, count(*) AS c
                FROM {TABLE}
                WHERE {where} AND subject_matter ILIKE %s
                  AND supplier = ANY(%s)
                  AND location_city IS NOT NULL AND location_city <> ''
                GROUP BY supplier, location_city
                ORDER BY supplier, c DESC
                """,
                (like, names),
            )
            for supplier_name, city, _c in cur.fetchall():
                bucket = regions.setdefault(supplier_name, [])
                if len(bucket) < MAX_REGIONS:
                    bucket.append(city)

        _execute(
            cur,
            f"SELECT count(DISTINCT supplier) FROM {TABLE} "
            f"WHERE {where} AND subject_matter ILIKE %s "
            f"AND supplier IS NOT NULL AND supplier <> ''",
            (like,),
        )
        distinct_suppliers = int(cur.fetchone()[0])

        time_range = _time_range(
            cur, f"{where} AND subject_matter ILIKE %s", (like,)
        )

    suppliers = [
        {
            "name": r[0],
            "count": int(r[1]),
            "win_count": int(r[1]),
            "median_amount": _f(r[2]),
            "total_amount": _f(r[3]),
            "first_win": _iso(r[4]),
            "last_win": _iso(r[5]),
            "mean_amount": _f(r[6]),
            "regions": regions.get(r[0], []),
        }
        for r in rows
    ]

    # 图谱补充：标的物在图谱里的采购频次与地区/代理机构（不可用时降级）
    graph, graph_note = _try_graph_subject(keyword)

    caveats = _subject_caveats(
        market, distinct_suppliers, suspicious, exclude_suspicious, time_range
    )

    return {
        "subject": keyword,
        "matched": True,
        "excluded_suspicious": exclude_suspicious,
        "suspicious_count": suspicious,
        "market": market,
        "distinct_suppliers": distinct_suppliers,
        "returned": len(suppliers),
        "suppliers": suppliers,
        "time_range": time_range,
        "graph": graph,
        "graph_note": graph_note,
        "data_caveats": caveats,
    }


def _try_graph_subject(keyword: str) -> tuple[dict | None, str | None]:
    """取标的物的图谱上下文；不可用或节点不存在时返回 `(None, 说明文案)`。

    把 try/except 收在一处：四个公开函数都要做这件事，各自的 except 分支
    又会写得不一致（有的漏了 None 分支，有的把异常文案当数据返回）。
    """
    try:
        graph = _graph_subject_context(keyword)
    except GraphUnavailableError as exc:
        return None, f"知识图谱不可用，已跳过图谱侧补充：{exc}"
    if graph is None:
        return None, (
            f"知识图谱中没有与「{keyword}」完全同名的标的物节点，已跳过图谱侧补充"
            "（图谱实体是归一化后的名字，与逐条记录的原文粒度不同）"
        )
    return graph, None


def _subject_caveats(
    market: dict[str, Any],
    distinct_suppliers: int,
    suspicious: int,
    exclude_suspicious: bool,
    time_range: dict[str, Any] | None = None,
) -> list[str]:
    """标的物维度的数据局限说明（前端必须展示）。

    集中在一处而不是散落各分支：这些提示是"结论能不能信"的前提，
    与数据一起返回才能保证前端不会漏掉其中某一条。
    """
    caveats: list[str] = []
    coverage = _coverage_caveat(time_range or {})
    if coverage:
        caveats.append(coverage)
    if suspicious and exclude_suspicious:
        caveats.append(
            f"范围内有 {suspicious} 条记录的金额被判定为可疑（服务类项目出现数亿元），"
            "已排除出统计。判据是「金额 + 品类」组合规则，详见 analysis.data_quality()"
        )
    elif suspicious:
        caveats.append(
            f"本次统计**含 {suspicious} 条金额可疑记录**（如服务类项目出现数亿元），"
            "中位数与合计均可能偏高；建议改用 exclude_suspicious=True 对比"
        )
    median = market.get("median_amount")
    ratio = market.get("mean_over_median")
    if median and ratio and ratio > 10:
        caveats.append(
            f"金额分布极度偏态：均值 ¥{market['mean_amount']:,.0f} 是中位数 "
            f"¥{median:,.0f} 的 {ratio:.0f} 倍，**请以中位数判断价格水平**，"
            "均值不具代表性"
        )
    if distinct_suppliers and distinct_suppliers <= 3:
        caveats.append(
            f"该标的物仅 {distinct_suppliers} 家供应商中标，样本过小，"
            "集中度类指标在此规模下不具统计意义"
        )
    if market["count"] < 5:
        caveats.append(f"该标的物仅 {market['count']} 条历史记录，结论仅供参考")
    return caveats


# ---------------------------------------------------------------------------
# ② 单个供应商画像
# ---------------------------------------------------------------------------


def competitor_profile(
    supplier_name: str,
    exclude_suspicious: bool = True,
    top_n: int = 8,
) -> dict[str, Any]:
    """单个供应商画像：中标规模 + 常投标的物 + 常中标的采购人 + 价格水平。

    价格水平怎么算（**这是本函数最需要解释的部分**）
    ----------------------------------------------
    直接拿"这家供应商的中位价"和"全库中位价"比是**没有意义**的：中标额首先
    取决于卖什么——卖基建的天然比卖办公用品的贵几个数量级。实测全库中位价
    ¥4,837，若某供应商专做工程（中位价上百万），会被误判成"报价偏高 200 倍"。

    所以比值**逐标的物算，再按金额加权**：

        对每个标的物 k：  ratio_k = 该供应商在 k 上的中位价 ÷ k 的市场中位价
        价格水平指数    = Σ(ratio_k × 该供应商在 k 的合计金额)
                          ÷ Σ(该供应商在 k 的合计金额)

    读法：**1.0 = 与同行持平；1.15 = 系统性地比同行贵 15%；0.8 = 便宜 20%**。
    用金额加权而不是简单平均，是因为大额项目才是这家公司的业务主体，
    几个几千元的小单不该和上亿的工程同等权重。

    只纳入 `记录数 >= MIN_RECORDS_FOR_RATIO` 且市场侧也有多条记录的标的物：
    单条记录的"中位数"就是那一笔，比值反映的是运气不是水平。
    指数算不出来时返回 None 并在 data_caveats 里说明原因，绝不退回一个假的 1.0。
    """
    keyword = _norm_keyword(supplier_name, "供应商名称")
    like = f"%{keyword}%"
    n = _clamp_limit(top_n)
    where = _base_amount_filter(exclude_suspicious)

    with _Cursor() as cur:
        _execute(
            cur,
            f"""
            SELECT
                count(*),
                count(DISTINCT subject_matter),
                count(DISTINCT purchaser),
                percentile_cont(0.5) WITHIN GROUP (ORDER BY amount),
                percentile_cont(0.25) WITHIN GROUP (ORDER BY amount),
                percentile_cont(0.75) WITHIN GROUP (ORDER BY amount),
                avg(amount),
                sum(amount),
                min(amount),
                max(amount),
                min(bid_time),
                max(bid_time)
            FROM {TABLE}
            WHERE {where} AND supplier ILIKE %s
            """,
            (like,),
        )
        row = cur.fetchone()
        win_count = int(row[0])
        if win_count == 0:
            return {
                "supplier": keyword,
                "matched": False,
                "excluded_suspicious": exclude_suspicious,
                "message": (
                    f"未找到与「{keyword}」匹配的中标记录"
                    "（该供应商可能未在本数据集覆盖的时间范围内中标，或名称写法不同）"
                ),
            }

        stats = {
            "win_count": win_count,
            "distinct_subjects": int(row[1] or 0),
            "distinct_purchasers": int(row[2] or 0),
            "median_amount": _f(row[3]),
            "p25": _f(row[4]),
            "p75": _f(row[5]),
            "mean_amount": _f(row[6]),
            "total_amount": _f(row[7]),
            "min_amount": _f(row[8]),
            "max_amount": _f(row[9]),
            "first_win": _iso(row[10]),
            "last_win": _iso(row[11]),
        }
        stats["mean_over_median"] = (
            round(stats["mean_amount"] / stats["median_amount"], 1)
            if stats["mean_amount"] and stats["median_amount"]
            else None
        )

        suspicious = _count_suspicious(cur, supplier=like)

        # 常投标的物 TOP：同时给出该标的物的**市场中位价**，让"贵还是便宜"可比。
        # 用别名 s 是为了让 _base_amount_filter 的条件整体指向本表（自连接场景）。
        alias_where = _base_amount_filter(exclude_suspicious, "s")
        _execute(
            cur,
            f"""
            SELECT
                s.subject_matter,
                count(*) AS cnt,
                percentile_cont(0.5) WITHIN GROUP (ORDER BY s.amount) AS my_median,
                sum(s.amount) AS my_total
            FROM {TABLE} s
            WHERE {alias_where}
              AND s.supplier ILIKE %s
              AND s.subject_matter IS NOT NULL AND s.subject_matter <> ''
            GROUP BY s.subject_matter
            ORDER BY cnt DESC, my_total DESC NULLS LAST
            LIMIT %s
            """,
            (like, n),
        )
        subject_rows = cur.fetchall()

        # 这些标的物的市场侧中位价与记录数：一次查完，避免 N+1 往返
        subject_names = [r[0] for r in subject_rows]
        market_medians: dict[str, float | None] = {}
        market_counts: dict[str, int] = {}
        if subject_names:
            _execute(
                cur,
                f"""
                SELECT subject_matter, count(*),
                       percentile_cont(0.5) WITHIN GROUP (ORDER BY amount)
                FROM {TABLE}
                WHERE {where} AND subject_matter = ANY(%s)
                GROUP BY subject_matter
                """,
                (subject_names,),
            )
            for subject_name, market_count, market_median in cur.fetchall():
                market_medians[subject_name] = _f(market_median)
                market_counts[subject_name] = int(market_count)

        _execute(
            cur,
            f"""
            SELECT purchaser, count(*) AS cnt, sum(amount) AS total,
                   percentile_cont(0.5) WITHIN GROUP (ORDER BY amount) AS median_amount
            FROM {TABLE}
            WHERE {where} AND supplier ILIKE %s
              AND purchaser IS NOT NULL AND purchaser <> ''
            GROUP BY purchaser
            ORDER BY cnt DESC, total DESC NULLS LAST
            LIMIT %s
            """,
            (like, n),
        )
        purchaser_rows = cur.fetchall()

        # 代表项目：取金额最大的几笔（注意是"最大"不是"最新"——
        # 本数据集实测只覆盖 2023-10 一个月，按时间排序等于按 id 排序，没有信息量）
        _execute(
            cur,
            f"""
            SELECT project_name, subject_matter, purchaser, amount, bid_time, location_city
            FROM {TABLE}
            WHERE {where} AND supplier ILIKE %s
            ORDER BY amount DESC NULLS LAST
            LIMIT %s
            """,
            (like, min(n, 5)),
        )
        top_records_rows = cur.fetchall()

        _execute(
            cur,
            f"""
            SELECT location_city, count(*) AS c
            FROM {TABLE}
            WHERE {where} AND supplier ILIKE %s
              AND location_city IS NOT NULL AND location_city <> ''
            GROUP BY location_city ORDER BY c DESC LIMIT %s
            """,
            (like, MAX_REGIONS),
        )
        region_rows = cur.fetchall()

        supplier_time_range = _time_range(cur, f"{where} AND supplier ILIKE %s", (like,))

    # 价格水平：逐标的物算比值，再按该供应商在各标的物的合计金额加权
    subjects: list[dict[str, Any]] = []
    weighted_ratio = 0.0
    weighted_amount = 0.0
    price_points: list[dict[str, Any]] = []
    for subject_name, cnt, my_median, my_total in subject_rows:
        market_median = market_medians.get(subject_name)
        my_med = _f(my_median)
        count_i = int(cnt)
        ratio = round(my_med / market_median, 3) if (my_med and market_median) else None
        subjects.append(
            {
                "subject_matter": subject_name,
                "count": count_i,
                "median_amount": my_med,
                "market_median_amount": market_median,
                "market_count": market_counts.get(subject_name),
                "total_amount": _f(my_total),
                "price_ratio": ratio,
                "ratio_basis": (
                    f"仅 {count_i} 条记录，比值仅作参考（需 ≥{MIN_RECORDS_FOR_RATIO} 条）"
                    if count_i < MIN_RECORDS_FOR_RATIO
                    else "该标的物内中位数对比"
                ),
            }
        )
        weight = _f(my_total) or 0.0
        # 三个条件缺一不可：有比值、样本够、市场侧也不止一条（否则"市场中位"
        # 就是对方那一条，比值恒为 1.0，纯属自欺）
        if (
            ratio is not None
            and count_i >= MIN_RECORDS_FOR_RATIO
            and weight > 0
            and market_counts.get(subject_name, 0) > 1
        ):
            weighted_ratio += ratio * weight
            weighted_amount += weight
            price_points.append(
                {
                    "subject_matter": subject_name,
                    "price_ratio": ratio,
                    "weight_amount": round(weight, 2),
                }
            )

    price_index = round(weighted_ratio / weighted_amount, 3) if weighted_amount > 0 else None

    # 图谱补充：这家供应商在图谱里还投过哪些标的物（失败降级）
    graph: dict[str, Any] | None = None
    graph_note: str | None = None
    try:
        graph = _graph_supplier_context(keyword)
    except GraphUnavailableError as exc:
        graph_note = f"知识图谱不可用，已跳过图谱侧补充：{exc}"
    else:
        if graph is None:
            graph_note = (
                f"知识图谱中没有与「{keyword}」匹配的供应商节点，已跳过图谱侧补充"
                "（图谱按归一化后的公司全名建节点，简称或关键词往往匹配不上）"
            )

    caveats: list[str] = []
    # 覆盖月数必须真的去查，不能拿 first_win / last_win 各截前 7 位再数集合大小：
    # 跨月的两天（10-31 与 11-01）会被算成 2 个月，实际只差一天。
    coverage = _coverage_caveat(supplier_time_range)
    if coverage:
        caveats.append(coverage)
    if suspicious and exclude_suspicious:
        caveats.append(f"该供应商有 {suspicious} 条记录的金额被判定为可疑，已排除出统计")
    elif suspicious:
        caveats.append(f"本次统计含 {suspicious} 条金额可疑记录，合计金额可能被显著放大")
    if stats["win_count"] < 5:
        caveats.append(
            f"该供应商仅 {stats['win_count']} 条中标记录，样本偏少，画像仅供参考"
        )
    if stats["mean_over_median"] and stats["mean_over_median"] > 10:
        caveats.append(
            f"其金额分布极度偏态：均值是中位数的 {stats['mean_over_median']:.0f} 倍，"
            "说明个别超大额项目主导了合计金额，**不要用均值代表其报价水平**"
        )
    if price_index is None:
        caveats.append(
            "无法计算价格水平指数：需要同一标的物下同时满足「该供应商 ≥"
            f"{MIN_RECORDS_FOR_RATIO} 条记录」且「市场 ≥2 条记录」才可比"
        )
    elif len(price_points) < 3:
        caveats.append(
            f"价格水平指数仅由 {len(price_points)} 个标的物加权得出，覆盖面窄，"
            "不宜当作该供应商的全局定价风格"
        )

    return {
        "supplier": keyword,
        "matched": True,
        "excluded_suspicious": exclude_suspicious,
        "suspicious_count": suspicious,
        "stats": stats,
        "price_level": {
            "index": price_index,
            "level": _price_level(price_index),
            "readme": (
                "1.0 = 与同行持平；>1 系统性偏贵；<1 偏便宜。"
                "按各标的物的合计金额加权，口径是「与该标的物内所有供应商的中位数对比」"
            ),
            "basis": price_points,
        },
        "top_subjects": subjects,
        "top_purchasers": [
            {
                "name": r[0],
                "count": int(r[1]),
                "total_amount": _f(r[2]),
                "median_amount": _f(r[3]),
            }
            for r in purchaser_rows
        ],
        "representative_records": [
            {
                "project_name": r[0],
                "subject_matter": r[1],
                "purchaser": r[2],
                "amount": _f(r[3]),
                "bid_time": _iso(r[4]),
                "location_city": r[5],
            }
            for r in top_records_rows
        ],
        "regions": [{"name": r[0], "count": int(r[1])} for r in region_rows],
        "graph": graph,
        "graph_note": graph_note,
        "data_caveats": caveats,
    }


# ---------------------------------------------------------------------------
# ③ 某采购人的合作供应商格局
# ---------------------------------------------------------------------------


def purchaser_landscape(
    purchaser_name_or_keyword: str,
    limit: int = DEFAULT_LIMIT,
    exclude_suspicious: bool = True,
) -> dict[str, Any]:
    """某采购人的合作供应商格局：供应商数量、集中度、各家中标次数与占比。

    匹配策略（**先精确、后模糊**）
    ----------------------------
    ① 先按 `purchaser = 关键词` **精确匹配**。这是最可信的路径——同一采购主体的
       记录全数落在一起，"它把活给了谁"才有确定答案。
    ② 精确匹配为空时才退化为 `ILIKE %关键词%` **模糊匹配**。此时会命中多个不同
       采购人（实测关键词"中学"命中 24 次阜南县第五初级中学、16 次合肥市第四十六
       中学南校区……），**聚合口径变成"这批单位合起来"**。因此响应里必须列出
       `matched_purchasers` 让人看清混进了哪些单位，否则"前 3 名占比"会被误读成
       某一家采购人的数据。

    `concentration.top3_share_percent` 是**前 3 名供应商的中标次数占比**；
    同时给出按金额的占比。两者常常背离，而背离本身就是结论：
    次数占比高说明"活给得频繁"，金额占比高说明"给的是大项目"。

    `limit` 只影响返回的供应商条数，**不影响任何占比**——占比一律按全量供应商
    排序计算，否则百分比会随 limit 变化，那就不是指标而是参数了。
    """
    keyword = _norm_keyword(purchaser_name_or_keyword, "采购人名称或关键词")
    n = _clamp_limit(limit)
    where = _base_amount_filter(exclude_suspicious)

    with _Cursor() as cur:
        # ① 精确匹配探路：只数条数，轻量
        _execute(
            cur,
            f"SELECT count(*) FROM {TABLE} WHERE {where} AND purchaser = %s",
            (keyword,),
        )
        exact_used = int(cur.fetchone()[0]) > 0
        op = "=" if exact_used else "ILIKE"
        pattern = keyword if exact_used else f"%{keyword}%"

        _execute(
            cur,
            f"""
            SELECT count(*), count(DISTINCT supplier),
                   percentile_cont(0.5) WITHIN GROUP (ORDER BY amount),
                   avg(amount), sum(amount)
            FROM {TABLE}
            WHERE {where} AND purchaser {op} %s
            """,
            (pattern,),
        )
        row = cur.fetchone()
        total_count = int(row[0])
        if total_count == 0:
            return {
                "purchaser": keyword,
                "matched": False,
                "exact_match_used": False,
                "excluded_suspicious": exclude_suspicious,
                "message": f"未找到与「{keyword}」匹配的采购记录",
            }

        # 用 op 而不是一律 ILIKE：精确匹配下若用 ILIKE，会把
        # 「XX中学附属小学」的可疑记录也算到「XX中学」的账上
        suspicious = _count_suspicious_purchaser(cur, op, pattern)

        overview = {
            "total_count": total_count,
            "distinct_suppliers": int(row[1] or 0),
            "median_amount": _f(row[2]),
            "mean_amount": _f(row[3]),
            "total_amount": _f(row[4]),
        }
        overview["mean_over_median"] = (
            round(overview["mean_amount"] / overview["median_amount"], 1)
            if overview["mean_amount"] and overview["median_amount"]
            else None
        )

        # 供应商明细（受 limit 截断，仅用于展示）
        _execute(
            cur,
            f"""
            SELECT supplier, count(*) AS cnt, sum(amount) AS total,
                   percentile_cont(0.5) WITHIN GROUP (ORDER BY amount) AS median_amount,
                   min(bid_time) AS first_win, max(bid_time) AS last_win
            FROM {TABLE}
            WHERE {where} AND purchaser {op} %s
              AND supplier IS NOT NULL AND supplier <> ''
            GROUP BY supplier
            ORDER BY cnt DESC, total DESC NULLS LAST
            LIMIT %s
            """,
            (pattern, n),
        )
        supplier_rows = cur.fetchall()

        # 全量供应商（不截断）：占比与排序必须基于全量，见 docstring
        _execute(
            cur,
            f"""
            SELECT supplier, count(*) AS cnt, sum(amount) AS total
            FROM {TABLE}
            WHERE {where} AND purchaser {op} %s
              AND supplier IS NOT NULL AND supplier <> ''
            GROUP BY supplier ORDER BY cnt DESC
            LIMIT %s
            """,
            (pattern, ALL_ROWS_LIMIT),
        )
        all_suppliers = cur.fetchall()

        # 命中的采购人清单（关键词模式下必须展示，见 docstring 的匹配策略）
        _execute(
            cur,
            f"""
            SELECT purchaser, count(*) AS c, count(DISTINCT supplier) AS sup
            FROM {TABLE}
            WHERE {where} AND purchaser {op} %s
              AND purchaser IS NOT NULL AND purchaser <> ''
            GROUP BY purchaser ORDER BY c DESC LIMIT %s
            """,
            (pattern, min(n, 10)),
        )
        matched_purchasers = [
            {"name": r[0], "count": int(r[1]), "distinct_suppliers": int(r[2])}
            for r in cur.fetchall()
        ]

        _execute(
            cur,
            f"""
            SELECT subject_matter, count(*) AS c
            FROM {TABLE}
            WHERE {where} AND purchaser {op} %s
              AND subject_matter IS NOT NULL AND subject_matter <> ''
            GROUP BY subject_matter ORDER BY c DESC LIMIT %s
            """,
            (pattern, n),
        )
        subject_rows = cur.fetchall()

        _execute(
            cur,
            f"SELECT count(DISTINCT purchaser) FROM {TABLE} "
            f"WHERE {where} AND purchaser {op} %s "
            f"AND purchaser IS NOT NULL AND purchaser <> ''",
            (pattern,),
        )
        distinct_purchasers = int(cur.fetchone()[0])

        time_range = _time_range(
            cur, f"{where} AND purchaser {op} %s", (pattern,)
        )

    def _shares(rows, use_amount: bool) -> dict[str, float]:
        """按次数或金额算份额百分比（分母=该口径下的合计）。"""
        raw: dict[str, float] = {}
        for supplier_name, cnt, total in rows:
            value = (_f(total) or 0.0) if use_amount else float(cnt)
            if value > 0:
                raw[supplier_name] = value
        denominator = sum(raw.values())
        if not denominator:
            return {}
        return {name: value / denominator * 100 for name, value in raw.items()}

    count_shares = _shares(all_suppliers, use_amount=False)
    amount_shares = _shares(all_suppliers, use_amount=True)

    def _top_share(shares: dict[str, float], k: int) -> float | None:
        """前 k 名份额之和。份额字典无序，故先降序再取前 k。"""
        if not shares:
            return None
        return round(sum(sorted(shares.values(), reverse=True)[:k]), 1)

    concentration = {
        "top3_share_percent": _top_share(count_shares, CR3_N),
        "top5_share_percent": _top_share(count_shares, CR5_N),
        "top3_share_by_amount_percent": _top_share(amount_shares, CR3_N),
        "top5_share_by_amount_percent": _top_share(amount_shares, CR5_N),
        "note": (
            "按**中标次数**占比计算；金额占比另列。两者背离说明头部供应商"
            "拿的是大项目（金额占比高）或只拿零散小单（次数占比高）"
        ),
    }

    caveats: list[str] = []
    coverage = _coverage_caveat(time_range)
    if coverage:
        caveats.append(coverage)
    if not exact_used:
        caveats.append(
            f"未找到名称恰为「{keyword}」的采购人，已退化为**关键词模糊匹配**："
            "本次聚合覆盖多个不同采购单位，「前 3 名占比」是这批单位合起来的口径，"
            "不能当成单一采购人的数据；请对照 matched_purchasers 核对范围"
        )
    if suspicious and exclude_suspicious:
        caveats.append(
            f"该范围内有 {suspicious} 条记录的金额被判定为可疑，已排除出统计"
        )
    if overview["distinct_suppliers"] and overview["distinct_suppliers"] <= 3:
        caveats.append(
            f"该采购人仅有 {overview['distinct_suppliers']} 家供应商中标，"
            "集中度指标在如此小的样本下不具统计意义"
        )
    if overview["mean_over_median"] and overview["mean_over_median"] > 10:
        caveats.append(
            f"金额分布极度偏态：均值 ¥{overview['mean_amount']:,.0f} 是中位数 "
            f"¥{overview['median_amount']:,.0f} 的 {overview['mean_over_median']:.0f} 倍，"
            "请以中位数判断典型项目规模"
        )
    if distinct_purchasers > 1 and exact_used:
        caveats.append(
            f"精确匹配命中 {distinct_purchasers} 个同名采购主体记录，"
            "如为不同单位建议核对名称"
        )

    # 图谱补充：采购人的关联标的物与代理机构（仅精确名称可查，失败降级）
    graph: dict[str, Any] | None = None
    graph_note: str | None = None
    if exact_used:
        try:
            graph = _graph_purchaser_context(keyword)
        except GraphUnavailableError as exc:
            graph_note = f"知识图谱不可用，已跳过图谱侧补充：{exc}"
        else:
            if graph is None:
                graph_note = f"知识图谱中没有与「{keyword}」匹配的采购人节点，已跳过图谱侧补充"
    else:
        graph_note = "关键词模式下命中的是多个采购单位，图谱侧补充不适用（图谱按完整名称建节点）"

    return {
        "purchaser": keyword,
        "matched": True,
        "exact_match_used": exact_used,
        "excluded_suspicious": exclude_suspicious,
        "suspicious_count": suspicious,
        "overview": overview,
        "suppliers": [
            {
                "name": r[0],
                "count": int(r[1]),
                "win_count": int(r[1]),
                "total_amount": _f(r[2]),
                "median_amount": _f(r[3]),
                "share_percent": round(count_shares.get(r[0], 0.0), 2),
                "share_by_amount_percent": round(amount_shares.get(r[0], 0.0), 2),
                "first_win": _iso(r[4]),
                "last_win": _iso(r[5]),
            }
            for r in supplier_rows
        ],
        "concentration": concentration,
        "matched_purchasers": matched_purchasers,
        "distinct_purchasers": distinct_purchasers,
        "top_subjects": [{"name": r[0], "count": int(r[1])} for r in subject_rows],
        "time_range": time_range,
        "graph": graph,
        "graph_note": graph_note,
        "data_caveats": caveats,
    }


def _count_suspicious_purchaser(cur, op: str, pattern: str) -> int:
    """数某采购人范围内金额可疑的记录数。

    为什么单独一个函数而不是复用 `_count_suspicious`：那个函数一律用 ILIKE，
    精确匹配场景下会把"XX中学附属小学"的可疑记录也算进"XX中学"的账上。
    """
    _execute(
        cur,
        f"SELECT count(*) FROM {TABLE} WHERE {suspicious_sql()} AND purchaser {op} %s",
        (pattern,),
    )
    return int(cur.fetchone()[0])


# ---------------------------------------------------------------------------
# ④ 市场集中度
# ---------------------------------------------------------------------------


def market_concentration(
    subject: str,
    exclude_suspicious: bool = True,
    top_n: int = 10,
) -> dict[str, Any]:
    """某标的物的市场集中度：供应商数量、CR3 / CR5、HHI 指数。

    HHI 怎么判读（**这条必须跟着数据一起返回**）
    ------------------------------------------
        HHI = Σ(各供应商市场份额百分比的平方)，取值 0 ~ 10000

            < 1500        竞争充分——供应商众多、无人主导，价格通常接近成本
            1500 ~ 2500   中等集中——头部几家有影响力，但仍有新进入者空间
            > 2500        高度集中——少数供应商主导、议价能力强，
                          需关注是否存在围标或长期绑定

    阈值出自美国 DOJ/FTC 横向并购指南。招投标不是并购市场，故这里只把它当
    **相对比较的标尺**（同一框架下比不同标的物谁更集中），不是合规判据——
    响应里的 `hhi_explanation` 会把这段话原样带给前端。

    **份额按金额算还是按次数算？两个都给。**
    ------------------------------------
    本数据集的金额跨 6 个数量级（几千元 ↔ 数亿元），按金额算份额时，一个上亿的
    工程能把所有小单供应商压成 0.0%，HHI 直接顶到满分；按次数算则完全忽略项目
    大小，"中了 100 个小单"与"中了 1 个亿级工程"被看作同等重要。
    **两种口径各有偏差，单给一个都会误导**，故两个都算（`by_amount` / `by_count`），
    并把偏差来源写进 data_caveats，让用户自己判读。

    `top_n` 只影响 `top_suppliers` 的展示条数；HHI 与 CR 一律用**全量供应商**计算
    ——用截断后的名单算集中度会系统性偏低，且数值随 top_n 变化，那不是指标。
    """
    keyword = _norm_keyword(subject, "标的物关键词")
    like = f"%{keyword}%"
    n = _clamp_limit(top_n)
    where = _base_amount_filter(exclude_suspicious)

    with _Cursor() as cur:
        _execute(
            cur,
            f"""
            SELECT count(*), count(DISTINCT supplier),
                   percentile_cont(0.5) WITHIN GROUP (ORDER BY amount),
                   avg(amount), sum(amount)
            FROM {TABLE}
            WHERE {where} AND subject_matter ILIKE %s
            """,
            (like,),
        )
        row = cur.fetchone()
        total_count = int(row[0])
        if total_count == 0:
            return {
                "subject": keyword,
                "matched": False,
                "excluded_suspicious": exclude_suspicious,
                "message": f"未找到与「{keyword}」匹配的中标记录，无法计算市场集中度",
            }

        suspicious = _count_suspicious(cur, subject_matter=like)

        # 全量供应商名单（不截断）：CR / HHI 必须是全市场的。
        # ALL_ROWS_LIMIT 只作护栏——实测单个标的物最多 175 条记录、≤169 家供应商。
        _execute(
            cur,
            f"""
            SELECT supplier, count(*) AS cnt, sum(amount) AS total
            FROM {TABLE}
            WHERE {where} AND subject_matter ILIKE %s
              AND supplier IS NOT NULL AND supplier <> ''
            GROUP BY supplier
            ORDER BY cnt DESC, total DESC NULLS LAST
            LIMIT %s
            """,
            (like, ALL_ROWS_LIMIT),
        )
        rows = cur.fetchall()

        # 没有供应商名的记录：它们不参与份额计算，会让 HHI 偏低，必须如实回报
        _execute(
            cur,
            f"SELECT count(*) FROM {TABLE} WHERE {where} AND subject_matter ILIKE %s "
            f"AND (supplier IS NULL OR supplier = '')",
            (like,),
        )
        unnamed = int(cur.fetchone()[0])

        time_range = _time_range(
            cur, f"{where} AND subject_matter ILIKE %s", (like,)
        )

    count_pct: dict[str, float] = {}
    amount_pct: dict[str, float] = {}
    count_total = sum(int(r[1]) for r in rows)
    amount_total = sum(_f(r[2]) or 0.0 for r in rows)
    for supplier_name, cnt, total in rows:
        if count_total:
            count_pct[supplier_name] = int(cnt) / count_total * 100
        if amount_total:
            amount_pct[supplier_name] = (_f(total) or 0.0) / amount_total * 100

    hhi_amount, cr3_amount, cr5_amount, leaders_amount = _concentration(amount_pct)
    hhi_count, cr3_count, cr5_count, leaders_count = _concentration(count_pct)

    distinct_suppliers = int(row[1] or 0)
    top_amount_share = leaders_amount[0]["share_percent"] if leaders_amount else None
    dominant_supplier = leaders_amount[0]["name"] if leaders_amount else None

    caveats: list[str] = []
    coverage = _coverage_caveat(time_range)
    if coverage:
        caveats.append(coverage)
    if suspicious and exclude_suspicious:
        caveats.append(
            f"范围内有 {suspicious} 条记录的金额被判定为可疑，已排除出统计"
        )
    elif suspicious:
        caveats.append(
            f"本次统计**含 {suspicious} 条金额可疑记录**，金额口径的 HHI 可能严重失真"
        )
    if distinct_suppliers <= 3:
        caveats.append(
            f"该标的物仅 {distinct_suppliers} 家供应商中标，样本过小，"
            "HHI / CR 在此规模下不具统计意义（HHI 的理论下限就是 10000 ÷ 供应商数）"
        )
    if top_amount_share is not None and top_amount_share >= 50:
        caveats.append(
            f"金额口径下「{dominant_supplier}」一家就占 {top_amount_share:.1f}%，"
            "**金额口径的『高度集中』基本由这一个项目决定**；"
            "请对照次数口径的 HHI，判断它到底是长期主导还是偶然中了一个大标"
        )
    if amount_pct and hhi_amount > hhi_count * 2 and hhi_amount > HHI_HIGH:
        caveats.append(
            f"两种口径结论背离（金额 HHI {hhi_amount:,.0f} vs 次数 HHI {hhi_count:,.0f}）："
            "说明少数几家拿走大项目、其余分食零散小单。此时按金额算会显得比实际更垄断"
        )
    if total_count < 10:
        caveats.append(f"该标的物仅 {total_count} 条历史记录，集中度指标波动大，仅供参考")
    if unnamed:
        caveats.append(
            f"有 {unnamed} 条记录没有供应商名，未参与份额计算，HHI 因此略偏低"
        )

    graph, graph_note = _try_graph_subject(keyword)

    return {
        "subject": keyword,
        "matched": True,
        "excluded_suspicious": exclude_suspicious,
        "suspicious_count": suspicious,
        "distinct_suppliers": distinct_suppliers,
        "total_count": total_count,
        "median_amount": _f(row[2]),
        "mean_amount": _f(row[3]),
        "total_amount": _f(row[4]),
        "time_range": time_range,
        # 主口径 = 金额（"市场份额"的通常含义）；次数口径并列给出供对照
        "by_amount": {
            "basis": "按中标金额计算份额（市场份额的通常含义）",
            "hhi": hhi_amount,
            "hhi_level": _hhi_level(hhi_amount),
            "cr3_percent": cr3_amount,
            "cr5_percent": cr5_amount,
            "leaders": leaders_amount,
        },
        "by_count": {
            "basis": "按中标次数计算份额（衡量「活给得频不频繁」）",
            "hhi": hhi_count,
            "hhi_level": _hhi_level(hhi_count),
            "cr3_percent": cr3_count,
            "cr5_percent": cr5_count,
            "leaders": leaders_count,
        },
        "top_suppliers": [
            {
                "name": r[0],
                "count": int(r[1]),
                "total_amount": _f(r[2]),
                "share_by_count_percent": round(count_pct.get(r[0], 0.0), 2),
                "share_by_amount_percent": round(amount_pct.get(r[0], 0.0), 2),
            }
            for r in rows[:n]
        ],
        "hhi_explanation": _hhi_explanation(),
        "graph": graph,
        "graph_note": graph_note,
        "data_caveats": caveats,
    }


__all__ = [
    "ALL_ROWS_LIMIT",
    "CompetitorError",
    "DEFAULT_LIMIT",
    "GraphUnavailableError",
    "HHI_HIGH",
    "HHI_LEVELS",
    "HHI_MODERATE",
    "MIN_KEYWORD_LENGTH",
    "MIN_RECORDS_FOR_RATIO",
    "competitor_profile",
    "market_concentration",
    "purchaser_landscape",
    "suppliers_for_subject",
]

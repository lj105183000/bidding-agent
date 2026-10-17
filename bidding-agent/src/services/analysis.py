"""集采分析服务：bidding_records 的多维聚合统计（供前端看板与报价策略使用）。

为什么必须用中位数而不是均值
----------------------------
实测数据（8746 条真实中标记录）的金额分布：

    最小      12.80
    P25    1,700.00
    中位   4,837.40
    P75   19,800.00
    均值 2,499,051.64     ← 是中位数的 516 倍
    最大 6,600,000,000    ← 66 亿

数据横跨 **6 个数量级**（办公用品几千元 ↔ 基建工程数亿元），
均值被少数超大工程完全带偏，**报出"中标均价 249 万"会严重误导**
（实际一半的项目在 5 千元以下）。故本模块：

    - 所有"典型值"一律用**中位数 / 分位数**
    - 均值只作为参考字段返回，且必须与中位数并排展示，让人一眼看出偏态

数据清洗策略
------------
实测发现脏数据：`智力、精神和重度残肢体疾人托养服务` 金额为 **66 亿元**
（服务类项目不可能到这个量级，应为 660 万）。但**不能一刀切删掉大额记录**——
同一批数据里 `安徽建工三建` 的 23.68 亿是真的城乡一体化工程。

故采用**标记而非删除**：
    - `SUSPICIOUS_AMOUNT`（10 亿）以上标记为可疑，默认排除出统计，但计数并回报
    - 同时用**分位数离群**（超过 P75 的 50 倍）作辅助判定

调用方通过 `exclude_suspicious` 决定是否排除；响应里始终带 `data_caveats`
说明数据局限（时间维度只有单月、category 语义混杂等），供前端如实展示。
"""

from __future__ import annotations

import os
from typing import Any

import psycopg
from dotenv import load_dotenv

from src.config import PROJECT_ROOT
from src.logging_config import get_logger

logger = get_logger(__name__)

TABLE = "bidding_records"

# 巨额下限（元）。用于"按品类判断是否合理"的规则：
# 单笔 ≥ 此值的记录，只有工程/基建这类品类才可能是真的。
HUGE_AMOUNT = 100_000_000  # 1 亿

# 天然不支持巨额单笔的品类关键词：命中即判可疑。
#
# ★ 这条规则的由来（踩过的坑）
# ---------------------------
# 第一版只用绝对阈值 `amount >= 10亿` 判可疑，结果是**一条都没匹配上**——
# 因为数据集的最大值才 6.6 亿，阈值比最大值还大，规则等于从不生效。
# 实测那条脏数据是 `智力、精神和重度残肢体疾人托养服务` = 6.6 亿，
# 而托养服务不可能到这个量级（应为 660 万，"万元"当"元"录入的典型错误）。
#
# 但**不能简单地"服务类不能超 1 亿"**：同一批数据里
# `水体综合治理及管网提升工程` 4.86 亿（安徽省城建设计研究总院）是真的。
# 所以判据必须是"品类 + 金额"的组合，而不是单看金额。
NON_HUGE_CATEGORY_KEYWORDS = (
    "托养", "养护服务", "残疾", "养老", "食堂", "餐饮", "保洁", "物业",
    "印刷", "纸张", "办公", "耗材", "书本", "教材", "绿化养护", "安保",
    "咨询", "培训", "会议", "差旅", "保险", "体检",
)

# 天然支持巨额单笔的品类关键词：命中则**即使金额很大也判为真实**。
# 这些是基建/工程类的常见标的，金额上亿属正常。
HUGE_OK_CATEGORY_KEYWORDS = (
    "工程", "建设", "施工", "总承包", "EPC", "市政", "道路", "桥梁", "隧道",
    "管网", "水利", "电力", "铁路", "轨道", "房建", "安置", "产业园",
    "土地", "复垦", "整治", "改造", "扩建", "景观", "绿化工程",
)

# 相对离群倍数：**已废弃，保留仅为记录踩过的坑**。
#
# 第二版曾用「金额 > P90 × 此倍数」作辅助判据，误伤了 23.68 亿的真实城乡
# 一体化工程、11.01 亿的产业园项目等一批真实基建记录——因为本数据集的 P90
# 只有 30.75 万（半数是办公用品），其 500 倍约 1.5 亿，真实工程必然超线。
#
# 结论：本数据集"统计离群"≠"数据错误"，相对阈值不可用，改用品类关键词。
# 详见 suspicious_sql 的文档字符串。
RELATIVE_OUTLIER_FACTOR = 500


class AnalysisError(RuntimeError):
    """分析失败（数据库不可用 / 参数不合法）。"""


def _dsn() -> dict[str, Any]:
    load_dotenv(PROJECT_ROOT / ".env")
    host = (os.getenv("POSTGRES_HOST") or "").strip()
    if not host:
        raise AnalysisError("未配置 PostgreSQL（POSTGRES_HOST 为空）")
    return {
        "host": host,
        "port": (os.getenv("POSTGRES_PORT") or "5432").strip(),
        "user": (os.getenv("POSTGRES_USER") or "postgres").strip(),
        # 密码末尾可能带 \r（Windows 编辑器保存 .env 时常见），必须清掉
        "password": (os.getenv("POSTGRES_PASSWORD") or "").rstrip("\r\n"),
        "dbname": (os.getenv("POSTGRES_DB") or "chatbot").strip(),
    }


class _Cursor:
    """轻量上下文管理器：连库 → 用 → 必关。"""

    def __enter__(self):
        try:
            self._conn = psycopg.connect(**_dsn())
        except Exception as exc:
            raise AnalysisError(f"数据库连接失败：{type(exc).__name__}") from exc
        self._cur = self._conn.cursor()
        return self._cur

    def __exit__(self, *exc_info):
        try:
            self._conn.close()
        except Exception:  # pragma: no cover
            pass
        return False


def _clamp_limit(limit: int) -> int:
    try:
        n = int(limit)
    except (TypeError, ValueError):
        n = DEFAULT_LIMIT
    return max(1, min(MAX_LIMIT, n))


# 返回条数上限，防止前端渲染被拖垮
DEFAULT_LIMIT = 20
MAX_LIMIT = 200


def suspicious_sql(col: str = "amount", subject_col: str = "subject_matter") -> str:
    """生成"该记录金额可疑"的 SQL 条件（供统计排除与明细查询共用）。

    两条规则取**并集**（命中任一即可疑）：

        ① 品类规则：金额 ≥ 1 亿，且标的物命中"天然不支持巨额"的品类关键词
                     （托养/办公/保洁…）。命中"支持巨额"关键词（工程/建设…）时豁免。
        ② 兜底规则：金额 ≥ 100 亿（无论如何都不可能是单笔采购）

    ★ 为什么删掉了"相对统计离群"规则
    ---------------------------------
    第二版用过「金额 > P90 × 500」作辅助规则，结果**误伤大量真实工程**：

        id=245   23.68亿  城乡一体化发展项目   （安徽建工三建，真实）
        id=942   11.01亿  工业产业园项目       （中建四局，真实）
        id=8045   4.86亿  水体综合治理及管网工程（真实）

    原因是本数据集的 P90 只有 30.75 万——半数记录是办公用品（16.1% 甚至
    低于 1000 元），其 500 倍仅约 1.5 亿，**数亿的真实工程必然超线**。

    根本教训：**本数据集的"统计离群"与"数据错误"不是一回事**。
    数据类型天然混合（几千元的耗材 ↔ 数亿元的基建），任何相对阈值都会
    把"大额但真实"和"大额且错误"一起圈进去。而品类关键词是**语义判断**，
    能区分"托养服务 6.6 亿（错）"与"治水工程 4.86 亿（对）"，且规则可审计、
    可解释、用户能自行判断是否认同。

    **宁可漏判也不误伤**：漏掉一条脏数据只是让均值偏高（而均值本就不被推荐
    使用），误伤真实工程却会把重要项目从分析里删掉。
    """
    ok_kw = " OR ".join(
        f"COALESCE({subject_col}, '') ILIKE '%%{k}%%'" for k in HUGE_OK_CATEGORY_KEYWORDS
    )
    bad_kw = " OR ".join(
        f"COALESCE({subject_col}, '') ILIKE '%%{k}%%'" for k in NON_HUGE_CATEGORY_KEYWORDS
    )
    return (
        f"("
        # ② 兜底：绝对巨额
        f"  {col} >= 10000000000"
        # ① 品类规则：巨额 + 不该巨额的品类 - 该巨额的品类
        f"  OR ({col} >= {HUGE_AMOUNT} AND ({bad_kw}) AND NOT ({ok_kw}))"
        f")"
    )


def _amount_filter(exclude_suspicious: bool, alias: str = "") -> str:
    """金额有效性条件。`exclude_suspicious=True` 时再排除可疑记录。"""
    col = f"{alias}.amount" if alias else "amount"
    subj = f"{alias}.subject_matter" if alias else "subject_matter"
    base = f"{col} IS NOT NULL AND {col} > 0"
    if exclude_suspicious:
        base += f" AND NOT {suspicious_sql(col, subj)}"
    return base


# ---------------------------------------------------------------------------
# ① 总览
# ---------------------------------------------------------------------------


def build_overview_caveats(
    t_min, months, suspicious: int, avg, p50
) -> list[str]:
    """生成概览页的数据局限提示（**纯函数，便于单测**）。

    抽出来的理由：这段逻辑此前内联在 `overview()` 里，而它有一个**只在空库时
    才触发**的崩溃（`f"{None:%Y-%m}"` → TypeError）。空库没法在测试里轻易构造，
    于是那个 bug 一直没被发现，直到有人用同形状的 SQL 复现。

    纯函数后可以直接喂 `t_min=None, months=0` 来验证，不必连库。
    """
    caveats: list[str] = []

    # 空表 / bid_time 全为 NULL 时 t_min 是 None，而 months 会是 0 —— 仍满足 `<= 1`，
    # 于是 `f"{t_min:%Y-%m}"` 直接抛 TypeError，整个概览接口 500。
    #
    # 当前库有 8746 行且 bid_time 全非空，所以一直没暴露；一旦清库或重新导入到空表，
    # 首页与集采分析页会同时打不开。故把"有没有时间数据"与"覆盖几个月"分开判断。
    if t_min is None:
        caveats.append("暂无时间数据（或 bid_time 全为空），无法做时间趋势分析")
    elif months is not None and months <= 1:
        caveats.append(
            f"数据仅覆盖 {months} 个月（{t_min:%Y-%m} 前后），**无法做时间趋势分析**，"
            "故本页不提供趋势图"
        )

    if suspicious:
        caveats.append(
            f"有 {suspicious} 条记录的金额被判定为可疑（如服务类项目出现数亿元），"
            "已从统计中排除。判据是「金额 + 品类」组合规则，详见数据质量面板；"
            "可在高级选项里选择纳入对比"
        )

    if avg and p50 and float(avg) > float(p50) * 10:
        caveats.append(
            f"金额分布极度偏态：均值 ¥{float(avg):,.0f} 是中位数 ¥{float(p50):,.0f} 的 "
            f"{float(avg)/float(p50):.0f} 倍，**请以中位数判断典型水平**，均值不具代表性"
        )

    caveats.append(
        "category 字段语义混杂（同时含「学校/政府/医院」等采购主体与「货物/工程/服务」等品类），"
        "按主体筛选时请注意"
    )
    return caveats


def overview(exclude_suspicious: bool = True) -> dict[str, Any]:
    """总体概览：条数、金额分位数、维度基数、数据质量提示。"""
    with _Cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {TABLE}")
        total_rows = cur.fetchone()[0]

        where = _amount_filter(exclude_suspicious)
        cur.execute(
            f"""
            SELECT
                count(*),
                min(amount), max(amount),
                percentile_cont(0.25) WITHIN GROUP (ORDER BY amount),
                percentile_cont(0.5)  WITHIN GROUP (ORDER BY amount),
                percentile_cont(0.75) WITHIN GROUP (ORDER BY amount),
                percentile_cont(0.9)  WITHIN GROUP (ORDER BY amount),
                avg(amount), sum(amount)
            FROM {TABLE} WHERE {where}
            """
        )
        n, mn, mx, p25, p50, p75, p90, avg, total = cur.fetchone()

        # 可疑记录计数（单独查，不受排除影响）
        cur.execute(f"SELECT count(*) FROM {TABLE} WHERE {suspicious_sql()}")
        suspicious = cur.fetchone()[0]

        # 维度基数
        dims: dict[str, int] = {}
        for col in ("subject_matter", "purchaser", "supplier", "province", "category"):
            cur.execute(
                f"SELECT count(DISTINCT {col}) FROM {TABLE} "
                f"WHERE {col} IS NOT NULL AND {col} <> ''"
            )
            dims[col] = cur.fetchone()[0]

        # 时间跨度（实测只有一个月的可能，必须在提示里说明）
        cur.execute(
            f"SELECT min(bid_time), max(bid_time), "
            f"count(DISTINCT to_char(bid_time, 'YYYY-MM')) FROM {TABLE}"
        )
        t_min, t_max, months = cur.fetchone()

    def f(v):
        return float(v) if v is not None else None

    caveats = build_overview_caveats(t_min, months, suspicious, avg, p50)

    return {
        "total_rows": total_rows,
        "amount_stats": {
            "count": n,
            "min": f(mn),
            "p25": f(p25),
            "median": f(p50),
            "p75": f(p75),
            "p90": f(p90),
            "max": f(mx),
            "mean": f(avg),
            "sum": f(total),
        },
        "suspicious_count": suspicious,
        "excluded_suspicious": exclude_suspicious,
        "dimension_cardinality": dims,
        "time_range": {
            "min": t_min.isoformat() if t_min else None,
            "max": t_max.isoformat() if t_max else None,
            "months": months,
        },
        "data_caveats": caveats,
    }


# ---------------------------------------------------------------------------
# ② 维度分组排行
# ---------------------------------------------------------------------------

# 允许分组的维度白名单。**必须白名单**：列名直接拼进 SQL，
# 用参数化无法覆盖标识符，只能靠白名单防注入。
GROUPABLE = {
    "subject_matter": "标的物",
    "purchaser": "采购人",
    "supplier": "供应商",
    "province": "省份",
    "location_city": "市区",
    "category": "类别",
    "source": "来源",
}


def group_ranking(
    dimension: str,
    limit: int = DEFAULT_LIMIT,
    exclude_suspicious: bool = True,
    order_by: str = "count",
) -> dict[str, Any]:
    """按维度分组统计：条数 / 金额中位数 / 合计 / 均值。

    `order_by` 取 `count`（按次数）或 `amount`（按合计金额）。
    每组都给出**中位数**，因为组内同样可能有超大额记录。
    """
    if dimension not in GROUPABLE:
        raise AnalysisError(
            f"不支持的维度 {dimension!r}，可用：{'、'.join(GROUPABLE)}"
        )
    n = _clamp_limit(limit)
    label = GROUPABLE[dimension]
    where = _amount_filter(exclude_suspicious)
    order_col = "total_amount" if order_by == "amount" else "cnt"

    with _Cursor() as cur:
        cur.execute(
            f"""
            SELECT
                {dimension} AS name,
                count(*) AS cnt,
                percentile_cont(0.5) WITHIN GROUP (ORDER BY amount) AS median_amount,
                avg(amount) AS mean_amount,
                sum(amount) AS total_amount,
                min(amount), max(amount)
            FROM {TABLE}
            WHERE {where} AND {dimension} IS NOT NULL AND {dimension} <> ''
            GROUP BY {dimension}
            ORDER BY {order_col} DESC NULLS LAST
            LIMIT %s
            """,
            (n,),
        )
        rows = cur.fetchall()

        cur.execute(
            f"SELECT count(DISTINCT {dimension}) FROM {TABLE} "
            f"WHERE {where} AND {dimension} IS NOT NULL AND {dimension} <> ''"
        )
        distinct_total = cur.fetchone()[0]

    items = [
        {
            "name": r[0],
            "count": r[1],
            "median_amount": float(r[2]) if r[2] is not None else None,
            "mean_amount": float(r[3]) if r[3] is not None else None,
            "total_amount": float(r[4]) if r[4] is not None else None,
            "min_amount": float(r[5]) if r[5] is not None else None,
            "max_amount": float(r[6]) if r[6] is not None else None,
        }
        for r in rows
    ]

    return {
        "dimension": dimension,
        "dimension_label": label,
        "order_by": order_by,
        "distinct_total": distinct_total,
        "returned": len(items),
        "items": items,
    }


# ---------------------------------------------------------------------------
# ③ 金额分布（直方图 + 数量级分桶）
# ---------------------------------------------------------------------------

# 数量级分桶：比等宽直方图更适合本数据集（跨 6 个数量级，等宽会把 99% 挤在第一格）
MAGNITUDE_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("<1千", 0, 1_000),
    ("1千~1万", 1_000, 10_000),
    ("1万~10万", 10_000, 100_000),
    ("10万~100万", 100_000, 1_000_000),
    ("100万~1000万", 1_000_000, 10_000_000),
    ("1000万~1亿", 10_000_000, 100_000_000),
    ("≥1亿", 100_000_000, float("inf")),
)


def amount_distribution(exclude_suspicious: bool = True) -> dict[str, Any]:
    """金额数量级分布（前端柱状图数据源）。

    为什么用数量级分桶而非等宽：数据跨 6 个数量级，等宽直方图会把
    99% 的记录挤进第一格，看不出任何结构。
    """
    where = _amount_filter(exclude_suspicious)
    buckets: list[dict[str, Any]] = []

    with _Cursor() as cur:
        for label, lo, hi in MAGNITUDE_BUCKETS:
            if hi == float("inf"):
                cond = f"amount >= {lo}"
            else:
                cond = f"amount >= {lo} AND amount < {hi}"
            cur.execute(
                f"SELECT count(*), avg(amount) FROM {TABLE} WHERE {where} AND {cond}"
            )
            cnt, avg = cur.fetchone()
            buckets.append(
                {
                    "label": label,
                    "count": cnt,
                    "mean_amount": float(avg) if avg is not None else None,
                }
            )
        cur.execute(f"SELECT count(*) FROM {TABLE} WHERE {where}")
        total = cur.fetchone()[0]

    for b in buckets:
        b["percent"] = round(b["count"] / total * 100, 2) if total else 0.0

    return {
        "total": total,
        "buckets": buckets,
        "note": "按金额数量级分桶（等宽直方图在此数据上会把 99% 挤进第一格）",
    }


# ---------------------------------------------------------------------------
# ④ 报价策略（单标的物的价格画像）
# ---------------------------------------------------------------------------


def price_profile(
    subject: str,
    exclude_suspicious: bool = True,
    top_n: int = 10,
) -> dict[str, Any]:
    """某标的物的价格画像：分位数 + 建议报价区间 + 历史案例。

    这是"我该报多少"的核心数据源。建议区间取 **P25~P75**：
    低于 P25 可能亏本或引发低价质疑，高于 P75 中标概率显著下降。
    """
    keyword = (subject or "").strip()
    if not keyword:
        raise AnalysisError("请提供标的物关键词")

    where = _amount_filter(exclude_suspicious)
    like = f"%{keyword}%"

    with _Cursor() as cur:
        cur.execute(
            f"""
            SELECT
                count(*),
                percentile_cont(0.1) WITHIN GROUP (ORDER BY amount),
                percentile_cont(0.25) WITHIN GROUP (ORDER BY amount),
                percentile_cont(0.5)  WITHIN GROUP (ORDER BY amount),
                percentile_cont(0.75) WITHIN GROUP (ORDER BY amount),
                percentile_cont(0.9)  WITHIN GROUP (ORDER BY amount),
                min(amount), max(amount), avg(amount)
            FROM {TABLE}
            WHERE {where} AND subject_matter ILIKE %s
            """,
            (like,),
        )
        row = cur.fetchone()
        cnt = row[0]

        if cnt == 0:
            return {
                "subject": keyword,
                "matched": False,
                "count": 0,
                "message": f"未找到与「{keyword}」匹配的历史中标记录，无法给出报价建议",
            }

        cur.execute(
            f"""
            SELECT project_name, purchaser, supplier, amount, bid_time, location_city
            FROM {TABLE}
            WHERE {where} AND subject_matter ILIKE %s
            ORDER BY amount DESC LIMIT %s
            """,
            (like, _clamp_limit(top_n)),
        )
        top_rows = cur.fetchall()

        # 常见供应商（竞争对手线索）
        cur.execute(
            f"""
            SELECT supplier, count(*) c, percentile_cont(0.5) WITHIN GROUP (ORDER BY amount)
            FROM {TABLE}
            WHERE {where} AND subject_matter ILIKE %s AND supplier IS NOT NULL AND supplier <> ''
            GROUP BY supplier ORDER BY c DESC LIMIT 5
            """,
            (like,),
        )
        supplier_rows = cur.fetchall()

    f = lambda v: float(v) if v is not None else None  # noqa: E731
    p10, p25, p50, p75, p90 = (f(row[i]) for i in range(1, 6))

    advice: list[str] = []
    if p25 is not None and p75 is not None:
        advice.append(
            f"建议报价区间 **¥{p25:,.0f} ~ ¥{p75:,.0f}**（历史 P25~P75，覆盖中间 50% 的成交）"
        )
    if p50 is not None:
        advice.append(f"历史中位价 ¥{p50:,.0f}，报价接近此值中标概率最高")
    if p10 is not None:
        advice.append(
            f"历史 P10 为 ¥{p10:,.0f}；显著低于此价需评估成本，也可能触发低价质疑"
        )
    if cnt < 5:
        advice.append(f"该标的物仅 {cnt} 条历史记录，样本偏少，结论仅供参考")
    if f(row[8]) and p50 and f(row[8]) > p50 * 3:
        advice.append(
            f"均价（¥{f(row[8]):,.0f}）远高于中位数，说明存在个别超大额项目，**不要按均价报价**"
        )

    return {
        "subject": keyword,
        "matched": True,
        "count": cnt,
        "percentiles": {
            "p10": p10, "p25": p25, "median": p50, "p75": p75, "p90": p90,
        },
        "min": f(row[6]),
        "max": f(row[7]),
        "mean": f(row[8]),
        "suggested_range": {"low": p25, "high": p75},
        "top_records": [
            {
                "project_name": r[0],
                "purchaser": r[1],
                "supplier": r[2],
                "amount": float(r[3]) if r[3] is not None else None,
                "bid_time": r[4].isoformat() if r[4] else None,
                "location_city": r[5],
            }
            for r in top_rows
        ],
        "top_suppliers": [
            {
                "name": r[0],
                "count": r[1],
                "median_amount": float(r[2]) if r[2] is not None else None,
            }
            for r in supplier_rows
        ],
        "advice": advice,
    }


# ---------------------------------------------------------------------------
# ⑤ 数据质量（本数据集的重要一环）
# ---------------------------------------------------------------------------


def data_quality() -> dict[str, Any]:
    """字段空值率 + 可疑金额明细（含判定原因）。前端"数据质量"面板的数据源。

    为什么要单独做这个：实测发现 `budget` 空值 99.5%、`agency` 空值 88%，
    还有 `托养服务` 记成 6.6 亿这种单位错误。这些不说明白，
    用户会拿分析结论当真。
    """
    with _Cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {TABLE}")
        total = cur.fetchone()[0]

        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = %s ORDER BY ordinal_position",
            (TABLE,),
        )
        cols = [r[0] for r in cur.fetchall()]

        empties: list[dict[str, Any]] = []
        for col in cols:
            cur.execute(
                f"SELECT count(*) FROM {TABLE} WHERE {col} IS NULL OR {col}::text = ''"
            )
            n = cur.fetchone()[0]
            if n:
                empties.append(
                    {
                        "column": col,
                        "empty_count": n,
                        "empty_percent": round(n / total * 100, 2) if total else 0.0,
                    }
                )
        empties.sort(key=lambda x: x["empty_percent"], reverse=True)

        cur.execute(
            f"""
            SELECT id, subject_matter, purchaser, supplier, amount, bid_time, category
            FROM {TABLE} WHERE {suspicious_sql()}
            ORDER BY amount DESC LIMIT 30
            """
        )
        rows = cur.fetchall()

        # 可疑记录的**真值**计数（不受上面 LIMIT 30 影响）。
        # 单列一次 count 是为了与 overview() 的口径一致——两个接口的数字
        # 会出现在同一个页面上，对不上就是 bug。
        cur.execute(f"SELECT count(*) FROM {TABLE} WHERE {suspicious_sql()}")
        suspicious_total = cur.fetchone()[0]

        # P90 用于说明相对规则
        cur.execute(
            f"SELECT percentile_cont(0.9) WITHIN GROUP (ORDER BY amount) "
            f"FROM {TABLE} WHERE amount IS NOT NULL AND amount > 0"
        )
        p90 = cur.fetchone()[0]

        # category 语义混杂的证据
        cur.execute(
            f"SELECT category, count(*) c FROM {TABLE} "
            f"WHERE category IS NOT NULL AND category <> '' "
            f"GROUP BY category ORDER BY c DESC LIMIT 12"
        )
        categories = [{"name": r[0], "count": r[1]} for r in cur.fetchall()]

    def why(subject: str, amount: float) -> str:
        """给出这条记录被判可疑的具体原因（让用户能自行判断是否认同）。"""
        s = subject or ""
        if amount >= 10_000_000_000:
            return "金额 ≥ 100 亿，远超单笔采购的合理上限"
        hits = [k for k in NON_HUGE_CATEGORY_KEYWORDS if k in s]
        if amount >= HUGE_AMOUNT and hits:
            return (
                f"标的物「{s}」属于不支持巨额单笔的品类（命中关键词：{hits[0]}），"
                f"但金额达 {amount/100_000_000:.2f} 亿，疑似把「万元」当「元」录入"
            )
        return "命中可疑规则"

    suspicious = [
        {
            "id": r[0],
            "subject_matter": r[1],
            "purchaser": r[2],
            "supplier": r[3],
            "amount": float(r[4]) if r[4] is not None else None,
            "bid_time": r[5].isoformat() if r[5] else None,
            "category": r[6],
            "reason": why(r[1], float(r[4]) if r[4] is not None else 0.0),
        }
        for r in rows
    ]

    return {
        "total_rows": total,
        "empty_rates": empties,
        "suspicious_records": suspicious,
        # ★ `suspicious_count` 必须是**真值**（与 overview() 的 count(*) 一致），
        # 而不是 `len(suspicious)` —— 后者受上面 SQL 的 `LIMIT 30` 截断，
        # 可疑记录超过 30 条时两个接口会给出互相矛盾的数字，
        # 而它们展示在同一个页面上（总览的提示条与数据质量面板），用户会看到两个数对不上。
        "suspicious_count": suspicious_total,
        # 明细另用 `returned` 表达"这次返回了几条"，语义清晰不混淆
        "suspicious_returned": len(suspicious),
        "suspicious_rules": {
            "huge_amount": HUGE_AMOUNT,
            "relative_factor": RELATIVE_OUTLIER_FACTOR,
            "non_huge_keywords": list(NON_HUGE_CATEGORY_KEYWORDS),
            "huge_ok_keywords": list(HUGE_OK_CATEGORY_KEYWORDS),
            "p90": float(p90) if p90 is not None else None,
        },
        "category_values": categories,
        "notes": [
            "budget / duration 字段空值率极高（>99%），做预算分析时不可依赖",
            "agency 字段空值率 88%，代理机构分析仅覆盖少数记录",
            "category 字段同时存放采购主体（学校/政府/医院）与品类（货物/工程/服务），使用前需自行归类",
            "可疑金额采用「金额 + 品类」组合规则：服务类标的出现数亿元即判可疑，"
            "而工程类标的数亿元属正常。被排除的记录并未删除，可在数据质量面板逐条查看",
        ],
    }


def recent_records(limit: int = 10, exclude_suspicious: bool = True) -> dict[str, Any]:
    """最近的中标记录（按中标时间倒序）。

    供工作台展示"最近中标动态"。

    ⚠️ 本数据集只覆盖 2023-10 单月，所以"最近"指的是**数据里**最近，
    不代表当前正在发生的采购——响应里必须带 `coverage` 说明，
    否则用户会以为是实时动态。
    """
    n = _clamp_limit(limit)
    where = _amount_filter(exclude_suspicious)

    with _Cursor() as cur:
        cur.execute(
            f"""
            SELECT project_name, purchaser, supplier, amount, bid_time,
                   subject_matter, province, location_city, source
            FROM {TABLE}
            WHERE {where} AND project_name IS NOT NULL AND project_name <> ''
            ORDER BY bid_time DESC NULLS LAST
            LIMIT %s
            """,
            (n,),
        )
        rows = cur.fetchall()

        cur.execute(
            f"SELECT min(bid_time), max(bid_time), count(DISTINCT to_char(bid_time,'YYYY-MM')) "
            f"FROM {TABLE}"
        )
        t_min, t_max, months = cur.fetchone()

    def f(v):
        return float(v) if v is not None else None

    return {
        "items": [
            {
                "project_name": r[0],
                "purchaser": r[1],
                "supplier": r[2],
                "amount": f(r[3]),
                "bid_time": r[4].isoformat() if r[4] else None,
                "subject_matter": r[5],
                "province": r[6],
                "location_city": r[7],
                "source": r[8],
            }
            for r in rows
        ],
        "returned": len(rows),
        "coverage": {
            "min": t_min.isoformat() if t_min else None,
            "max": t_max.isoformat() if t_max else None,
            "months": months,
            "caveat": (
                f"数据仅覆盖 {months} 个月"
                f"（{t_min:%Y-%m-%d} ~ {t_max:%Y-%m-%d}）"
                if t_min and months is not None
                else "时间范围未知"
            ),
        },
    }


__all__ = [
    "AnalysisError",
    "GROUPABLE",
    "HUGE_AMOUNT",
    "MAGNITUDE_BUCKETS",
    "amount_distribution",
    "data_quality",
    "group_ranking",
    "overview",
    "price_profile",
    "recent_records",
    "suspicious_sql",
]

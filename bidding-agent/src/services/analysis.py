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

from typing import Any

from src.datahub.checks import (
    HUGE_AMOUNT,
    HUGE_OK_CATEGORY_KEYWORDS,
    NON_HUGE_CATEGORY_KEYWORDS,
    RELATIVE_OUTLIER_FACTOR,
    empty_rates,
    suspicious_rows,
    suspicious_sql as _checks_suspicious_sql,
)
from src.datahub.contract import DataHubConfigError as _DataHubConfigError
from src.datahub.pg import PGCursor, dsn as pg_dsn
from src.logging_config import get_logger

logger = get_logger(__name__)

TABLE = "bidding_records"


class AnalysisError(RuntimeError):
    """分析失败（数据库不可用 / 参数不合法）。"""


def _dsn() -> dict[str, Any]:
    """兼容包装：DSN 组装已收口到 `src.datahub.pg`。

    保留这个函数名是因为既有测试直接调用 `A._dsn()`；实现只有一行，
    真正的 `.env` 读取与密码清洗在 DataHub 适配器里。
    """
    try:
        return pg_dsn()
    except _DataHubConfigError as exc:
        raise AnalysisError(str(exc)) from exc


def _Cursor():
    """兼容包装：连接、超时与异常映射统一由 `src.datahub.pg.PGCursor` 负责。

    仍然返回上下文管理器（`with _Cursor() as cur`），故调用方无需改动。
    """
    return PGCursor(AnalysisError)


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
    """兼容包装：判据的唯一事实来源在 `src.datahub.checks.suspicious_sql`。

    保留这个公开函数是因为 `competitor` / `material` 与既有测试都从这里导入；
    判据的踩坑历史与详细说明已随实现迁到 `checks.py`。
    """
    return _checks_suspicious_sql(col, subject_col)


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
        # 空值率与可疑明细的口径在 `datahub.checks` 里，质量扫描复用同一份实现。
        total, empties = empty_rates(cur, TABLE)
        # `suspicious_rows` 同时返回明细与**真值**计数（不受 LIMIT 30 影响）：
        # 两个数字会出现在同一页面（总览提示条 + 质量面板），口径必须一致。
        rows, suspicious_total = suspicious_rows(cur, TABLE, limit=30)

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

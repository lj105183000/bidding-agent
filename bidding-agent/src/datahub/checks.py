"""共享的数据检查 SQL（金额可疑判定 / 空值率 / 可疑明细）。

为什么要从 `src/services/analysis.py` 抽出来
------------------------------------------
同一套“可疑金额”判据被三处使用：分析看板、物料服务、竞对服务。
如果质量扫描再抄一份，四处判据迟早分叉——用户会看到“分析页说 1 条可疑、
质量报告说 3 条”这种互相矛盾的数字。故这里作为唯一事实来源，
`analysis` 只做 re-export，保持既有导入路径不变。

判据本身的历史与踩坑记录在 `suspicious_sql` 的 docstring 里，迁移时一并保留。
"""

from __future__ import annotations

import re
from typing import Any


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

# 只允许简单标识符拼进 SQL：表名/列名来自调用方常量，但这里再加一道防线，
# 避免未来有人把外部输入传进来。值仍然一律走 %s 参数化。
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _safe_identifier(name: str) -> str:
    if not _IDENTIFIER.fullmatch(name or ""):
        raise ValueError(f"非法 SQL 标识符：{name!r}")
    return name


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


def empty_rates(cur: Any, table: str = "bidding_records") -> tuple[int, list[dict[str, Any]]]:
    """统计全表空值率，返回 `(total, [{column, empty_count, empty_percent}])`。

    与 `analysis.data_quality()` 的口径完全一致：`NULL` 与 `''` 都算空。
    列名来自 `information_schema`（可信），但仍过标识符校验后才拼进 SQL。
    """
    table = _safe_identifier(table)
    cur.execute(f"SELECT count(*) FROM {table}")
    total = int(cur.fetchone()[0])

    cur.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = %s ORDER BY ordinal_position",
        (table,),
    )
    columns = [row[0] for row in cur.fetchall()]

    empties: list[dict[str, Any]] = []
    for column in columns:
        if not _IDENTIFIER.fullmatch(column):
            continue
        cur.execute(
            f"SELECT count(*) FROM {table} WHERE {column} IS NULL OR {column}::text = ''"
        )
        count = int(cur.fetchone()[0])
        if count:
            empties.append(
                {
                    "column": column,
                    "empty_count": count,
                    "empty_percent": round(count / total * 100, 2) if total else 0.0,
                }
            )
    empties.sort(key=lambda item: item["empty_percent"], reverse=True)
    return total, empties


def suspicious_rows(
    cur: Any,
    table: str = "bidding_records",
    limit: int = 30,
) -> tuple[list[tuple], int]:
    """返回 `(明细行, 真值计数)`。

    `limit` 只限制明细条数；总数单独 `count(*)`，因为这两个数字会同时出现在
    “总览提示条”和“数据质量面板”，口径必须一致。
    """
    table = _safe_identifier(table)
    cur.execute(
        f"""
        SELECT id, subject_matter, purchaser, supplier, amount, bid_time, category
        FROM {table} WHERE {suspicious_sql()}
        ORDER BY amount DESC LIMIT %s
        """,
        (limit,),
    )
    rows = cur.fetchall()
    cur.execute(f"SELECT count(*) FROM {table} WHERE {suspicious_sql()}")
    total = int(cur.fetchone()[0])
    return rows, total


def suspicious_count(cur: Any, table: str = "bidding_records") -> int:
    table = _safe_identifier(table)
    cur.execute(f"SELECT count(*) FROM {table} WHERE {suspicious_sql()}")
    return int(cur.fetchone()[0])

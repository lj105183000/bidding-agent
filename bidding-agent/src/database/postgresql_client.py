"""PostgreSQL 业务库客户端（成员 C）。

用途
----
把招投标结构化数据（`bidding_records` 表）的统计查询（6 类固定模板 + 排行 / 模糊搜索 /
整体概览），以及会话 / 消息 / 反馈的增删查接口封装成一组短事务方法。业务查询供
src/tools/rag_tools.py 的 `query_database` 工具调用；会话历史与反馈的读写
（`save_message` / `load_messages` / `save_feedback`）供 API 层的落库链路调用。
本模块只负责"查得到什么 / 写得进什么"，不负责措辞与包装——成功 / 失败的结构化包装
由调用方统一完成。

依赖
----
psycopg 3（`psycopg[binary]`）。配置来自环境变量：`POSTGRES_HOST` / `POSTGRES_DB` /
`POSTGRES_USER` / `POSTGRES_PASSWORD`（必填）、`POSTGRES_PORT`（可选，默认 5432）。

表结构（简述）
--------------
    bidding_records（由 src/database/to_postgresql.py 从 xlsx 导入）
        project_id / project_name / subject_matter / purchaser / agency /
        supplier / location_city   TEXT
        amount                     NUMERIC     交易额 / 中标金额
        created_at                 TIMESTAMPTZ（可选，默认 now()）

    conversations(id, session_id, title, created_at)          会话
    messages(id, conversation_id, role, content, sources JSONB, tool_name, created_at)
        conversation_id 外键 → conversations.id，ON DELETE CASCADE：删会话即删消息
    feedback(id, conversation_id, message_id, rating, comment, created_at)
    后三张表由 `ensure_schema()` 负责创建，只在会话 / 消息 / 反馈链路使用。

设计要点（改本文件前先读）
--------------------------
- **白名单是字段名与聚合函数名的唯一防线**：SQL 的"值"可以交给 psycopg 的参数化占位符
  （`%s`），驱动会负责转义；但**字段名、表名、聚合函数名、GROUP BY 的表达式不是值**，
  占位符顶替不了它们——把 `purchaser; DROP TABLE` 塞进标识符的位置，参数化救不了。
  故凡是要拼进语句的标识符，必须先过 `validate_field` / `validate_agg`；本模块自己
  拼的片段也只从自带常量表（如 `_TREND_EXPRESSIONS`）里取，不接受外部字符串。
- **失败返回空而不抛异常**：工具层统一把失败包成结构化错误，Agent 不能因为数据库宕机
  而崩，故查询方法异常时返回 `[]` / `False` / `{"ok": False, ...}`，完整异常（可能含
  主机名、SQL、参数值）只进日志，用户可见文案只用固定措辞。
- **参数非法则相反——直接抛 `ValueError`**：那是调用方（工具层 / LLM 给的参数）的缺陷，
  不是运营故障；静默返回空会让"字段名写错了"看起来像"库里没数据"。故 `validate_*`
  与金额边界的校验一律放在 `try` 之外先做。
- **出口统一转成 JSON 原生类型**：NUMERIC 出来是 `Decimal`、TIMESTAMPTZ 出来是
  `datetime`，`json.dumps` 都不认识——这些行最终要进 SSE 帧，故在 `_query` 出口统一
  归一化（见 `_jsonable`）。
- **精确匹配与模糊匹配并存，模糊匹配必须先转义通配符**：`by_purchaser` / `by_supplier` 走
  `= %s`；`avg_amount_by_subject` / `search_by_keyword` 走 `ILIKE %s`，其模式串一律经
  `_like_param` 处理。参数化管的是**注入**，管不了**通配符语义**——用户搜 `100%` 时若把
  原样的 `%` 交给 LIKE，会变成"匹配全表"，返回一大堆不相干记录却不报错，比注入更难发现。
  新增任何 LIKE / ILIKE 查询时，模式串都必须过 `_like_param`。
- **JSONB 写入要显式包 `Jsonb`**：JSONB 列收到裸 dict 时 psycopg 会把它当普通 Python 对象
  去适配而报错，`Jsonb` 才是"这是一个 JSON 值"的声明。读出方向相反——驱动的默认加载器
  直接给出 dict / list，不需要手工 `json.loads`（`_jsonable` 也不碰它们）。

"""

import logging
import os
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Sequence

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from pathlib import Path
from dotenv import load_dotenv

# 自动定位项目根目录的 .env（假设本文件在 src/database/ 下）
env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=env_path)

logger = logging.getLogger(__name__)

# 业务表名：与 src/database/to_postgresql.py 的导入目标绑定，改导入脚本要同步这里
BUSINESS_TABLE = "bidding_records"

# 分组 / 过滤字段白名单。字段名不能参数化（`%s` 只能出现在"值"的位置），
# 所以只要某个字段名会被拼进 SQL，它就必须先登记在这里——这是防注入的唯一防线。
ALLOWED_GROUP_FIELDS = {
    "project_name",
    "subject_matter",
    "purchaser",
    "agency",
    "supplier",
    "location_city",
}

# 聚合函数白名单：同样是标识符，同样不能参数化。
# 本模块的固定查询直接写 count(*)，这份白名单是给工具层组装聚合查询（docs/组件工作机制.md
# §15 的 aggregate_stats）用的，故 validate_agg 与它一起导出。
ALLOWED_AGG = {"count", "sum", "avg", "min", "max"}

# 单次查询行数上限：limit 多由 LLM 给出，业务表不是分页导出通道。
# 这个上限同时是本地 PG 的保护线——一条 SELECT * 拉爆内存对演示毫无帮助。
MAX_LIMIT = 200

# 建连超时必须快失败：地址填错 / 数据库没启动时，绝不能让工具调用干等，
# 工具层的整体预算是 30s（CLAUDE.md 的超时表）。
CONNECT_TIMEOUT_SECONDS = 10

# 语句超时：bidding_records 目前没有索引（索引由导入脚本按需建），
# 一个全表扫描 + 排序可能吃掉整轮问答的时间。超时后 PG 主动断开，异常照常降级成 []。
STATEMENT_TIMEOUT_MS = 15000

# 业务表返回列：显式列出而不是 SELECT *。导入脚本的列顺序 / 增减列不该改变工具层的
# 输出形状（格式化器按列名取值），列集合要改时这里与 to_postgresql.py 一起改。
_RECORD_COLUMNS = (
    "project_id, project_name, subject_matter, purchaser, "
    "agency, supplier, amount, location_city, created_at"
)

# 时间趋势的列表达式：key 是 group_by 的合法取值，value 是**本模块自带的常量片段**。
# 用查表而不是拼字符串，是为了让"外部输入"永远只当 key 用，不参与 SQL 组装。
_TREND_EXPRESSIONS = {
    "month": "to_char(date_trunc('month', created_at), 'YYYY-MM')",
    "year": "to_char(date_trunc('year', created_at), 'YYYY')",
}

# 连接类异常 → 中文原因。只报方向不报细节：驱动原始报文里带主机名、端口、库名，
# 而 health() 的结果会进 /api/health 与用户可见区。
#
# 为什么这里只有一条（实测结论，别按 neo4j_client 的三条照搬）：psycopg 把**连接阶段**的
# 失败一律包成 OperationalError，且 sqlstate 为 None——密码错、库不存在、端口不通都是
# 同一个类、同一个属性。"到底哪个变量写错了"只存在于 PG 自己的本地化报文里，靠异常类型
# 区分不了，所以文案一次点出整个 POSTGRES_* 组，让人对着 .env 逐项核。
# （业务表没导入属于**查询阶段**的 UndefinedTable，由 _query 记日志并返回 []，
#   不会走到这里；排查时去日志里找「关系 "bidding_records" 不存在」。）
_REASON_BY_EXCEPTION = (
    (psycopg.OperationalError, "无法连接到数据库服务（检查 POSTGRES_* 配置、数据库是否已启动、库是否已创建）"),
)


def _explain(exc: Exception) -> str:
    """把驱动异常翻译成中文原因，不含地址、端口与原始报文（那些只进日志）。"""
    for exc_type, reason in _REASON_BY_EXCEPTION:
        if isinstance(exc, exc_type):
            return reason
    return "未知错误，详见后端日志"


def _summarize(sql: str) -> str:
    """把多行 SQL 压成单行并截断，免得一条长语句把日志刷屏。"""
    one_line = " ".join(sql.split())
    return one_line if len(one_line) <= 160 else one_line[:157] + "..."


def _jsonable(value: Any) -> Any:
    """把 PG 返回的 Python 对象收敛成 JSON 原生类型。

    `Decimal` / `datetime` 都是 `json.dumps` 认不出的类型，而这些行会被工具层塞进
    SSE 的 done 帧——不在出口归一化，就会在序列化那一步炸掉，且报错位置离数据库很远，
    排查成本极高。`datetime` 必须先于 `date` 判断（前者是后者的子类）。
    """
    if isinstance(value, bool):  # bool 是 int 子类，别把 True 变成 1.0
        return value
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return value


def validate_field(field: str) -> str:
    """字段名白名单校验，合法返回规范化后的字段名，非法抛 `ValueError`。

    规范化 = 去首尾空白 + 转小写：LLM 抽出来的字段名常见 `Purchaser`、`supplier ` 这类
    写法，与库里的列名只差大小写；这里统一收敛，省得调用方各自 lower()。
    校验不通过必须抛：字段名会拼进 SQL，放行任何一个未登记的值都等于开了一个注入口子。
    """
    normalized = (field or "").strip().lower()
    if normalized not in ALLOWED_GROUP_FIELDS:
        raise ValueError(f"不支持的查询字段: {field}")
    return normalized


def validate_agg(agg: str) -> str:
    """聚合函数名白名单校验，合法返回规范化后的函数名，非法抛 `ValueError`。"""
    normalized = (agg or "").strip().lower()
    if normalized not in ALLOWED_AGG:
        raise ValueError(f"不支持的聚合函数: {agg}")
    return normalized


def _safe_limit(limit: Any, default: int = 20) -> int:
    """把调用方（多半是 LLM）给的 limit 收敛成合法整数：非法退回默认值，超限截断。

    不信任外部传来的数值——`LIMIT %s` 收到字符串或浮点会被驱动拒绝，
    整条查询连"有没有结果"都问不出来。
    """
    try:
        value = int(limit)
    except (TypeError, ValueError):
        return default
    return max(1, min(value, MAX_LIMIT))


def _like_param(keyword: str) -> str:
    """把关键词转成 ILIKE 用的参数值：转义通配符后两侧补 `%`。

    值仍然走 `%s` 占位符（不拼进 SQL），这里处理的不是注入，而是**通配符语义**：
    `%` 与 `_` 在 LIKE 里是"任意串 / 任意单字符"，用户搜 `100%` 或 `A_1` 时若原样传入，
    会变成"匹配一切"或"匹配任意中间字符"，结果是查出一堆不相干的记录还看不出错。
    反斜杠必须最先转义——它是转义符本身，放在后面会把刚加上的 `\\%` 再转一遍。
    PG 的 LIKE 默认转义符就是 `\\`，故无需额外写 `ESCAPE` 子句。
    """
    escaped = keyword.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def _as_amount(value: Any) -> float | None:
    """金额边界归一化为 float；`None` 表示"这一侧不限制"。

    非数值（含 "1,234.50" 这类带千分位的字符串）直接抛：金额是调用方给的过滤条件，
    静默忽略一半边界会得到"看起来查到了、其实条件没生效"的错误答案。
    """
    if value is None:
        return None
    if isinstance(value, bool):  # bool 是 int 子类，True 当金额没有意义
        raise ValueError(f"金额边界不是有效数字: {value}")
    if isinstance(value, (int, float, Decimal)):
        return float(value)
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"金额边界不是有效数字: {value}") from exc


def _as_id(value: Any, name: str) -> int:
    """自增主键归一化为 int，非法抛 `ValueError`。"""
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 不是有效的整数 id: {value}") from exc


class PostgresClient:
    """招投标业务库 + 会话 / 反馈存储的客户端。

    实例本身不持有连接，也不缓存任何查询结果：工具层的 4 线程并行会同时调这些方法，
    方法全部无副作用（只读或自带事务），故无需加锁。
    """

    def __init__(self) -> None:
        """读取 POSTGRES_* 配置并保存连接参数（**不**在这里连库）。

        懒连接的理由与 Neo4jClient 一致：构造或 import 阶段就要求连得上远程，
        没装数据库的机器上跑测试与 CI 会直接挂；连通性交给 health() 按需检查。
        """
        host = (os.getenv("POSTGRES_HOST") or "").strip()
        dbname = (os.getenv("POSTGRES_DB") or "").strip()
        user = (os.getenv("POSTGRES_USER") or "").strip()
        # 密码只去行尾换行（Windows 上手写的 .env 常带 \r），不做 strip——首位空格可能是密码本身
        password = (os.getenv("POSTGRES_PASSWORD") or "").rstrip("\r\n")
        missing = [
            name
            for name, value in (
                ("POSTGRES_HOST", host),
                ("POSTGRES_DB", dbname),
                ("POSTGRES_USER", user),
                ("POSTGRES_PASSWORD", password),
            )
            if not value
        ]
        if missing:
            raise ValueError(f"缺少必需配置：{'、'.join(missing)}（请检查 .env，可参照 .env.example）")

        self.host = host
        self.dbname = dbname
        self.user = user
        self.password = password
        self.port = self._read_port()

    @staticmethod
    def _read_port() -> int:
        """读 POSTGRES_PORT：未配置按 Postgres 默认 5432；写成非数字时退默认值并告警。

        这里不抛异常：端口缺失有众所周知的默认值，把它当成"配置错误"会让整台机器
        起不来，代价远大于收益；真连错了，health() 会给出方向性的中文提示。
        """
        raw = (os.getenv("POSTGRES_PORT") or "").strip()
        if not raw:
            return 5432
        try:
            return int(raw)
        except ValueError:
            logger.warning("POSTGRES_PORT 不是数字（%s），回退到默认端口 5432", raw)
            return 5432

    def _conn(self) -> psycopg.Connection:
        """返回一个新的 psycopg 连接（**不**复用连接池），供 `with` 使用。

        连接的上下文语义正是我们要的：块内正常结束则提交、抛异常则回滚、退出即关闭。
        每次新建而不是共享一条连接，是因为工具层 4 线程并行调用：共享连接既不线程安全，
        一个查询失败还会污染同连接上其他线程的事务状态（PG 里事务失败后整条连接只能回滚）。
        本地 / 内网的握手成本远小于一次远程查询本身，这个交换是划算的。
        """
        return psycopg.connect(
            host=self.host,
            port=self.port,
            dbname=self.dbname,
            user=self.user,
            password=self.password,
            connect_timeout=CONNECT_TIMEOUT_SECONDS,
            # 行工厂设在连接级：所有 cursor 都拿 dict，调用方不必记住逐列取数的顺序
            row_factory=dict_row,
            # statement_timeout 走 options 传入（psycopg 没有独立的 connect 参数），
            # 让"跑太久的查询"由服务端主动掐断，而不是拖垮整轮问答
            options=f"-c statement_timeout={STATEMENT_TIMEOUT_MS}",
        )

    def health(self) -> dict:
        """连接健康检查，返回 {"ok": bool, "error": str | None}。失败不抛。"""
        try:
            with self._conn() as conn, conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchall()
        except Exception as exc:
            logger.exception("PostgreSQL 连接检查失败")
            return {"ok": False, "error": f"数据库连接失败：{_explain(exc)}"}
        return {"ok": True, "error": None}

    def _query(
        self,
        sql: str,
        params: Sequence[Any] | None = None,
        fetch: bool = True,
    ) -> list[dict]:
        """执行 SQL 返回 `list[dict]`；任何失败只记日志并返回 `[]`。

        值一律通过 `params` 交给驱动（对应语句里的 `%s`），本方法**不接受**要拼接的
        标识符——字段名 / 聚合函数名只能由调用方先用 `validate_*` 校验后拼进 sql 参数。
        `fetch=False` 用于 DDL 等不取结果的语句（提交由连接的上下文完成）。
        """
        try:
            with self._conn() as conn, conn.cursor() as cur:
                cur.execute(sql, tuple(params or ()))
                if not fetch:
                    return []
                rows = cur.fetchall()
        except Exception:
            # 只记语句骨架（含白名单标识符与 %s 占位符），不记参数值：日志里不需要用户关键词
            logger.exception("PostgreSQL 查询失败：%s", _summarize(sql))
            return []
        logger.debug("PostgreSQL 查询返回 %d 行：%s", len(rows), _summarize(sql))
        # 出口统一归一化：格式化层与 SSE 序列化都不该再关心 Decimal / datetime
        return [{key: _jsonable(value) for key, value in row.items()} for row in rows]

    # --- bidding_records 的查询模板（6 类固定模板 + 排行 / 模糊搜索 / 概览）---
    #
    # 下面每条 ORDER BY 都带一个次级排序键（名称 / project_id）：数据由导入脚本一次写入，
    # created_at 与各种 count 平局是常态而非例外，没有次级键时 PG 可以任意决定平局行序——
    # 同一个问题问两次拿到不同顺序，前端的来源排序与 eval 的比对都会跟着抖。

    def top_subject(self, limit: int = 10) -> list[dict]:
        """采购次数最多的标的物排行，返回 `[{"subject_matter": ..., "cnt": N}]`。"""
        safe_limit = _safe_limit(limit, default=10)
        # 空串与 NULL 排除在分组之外：否则排行第一可能是一个没有名字的分组，
        # 对"哪些标的物最热门"这个问题毫无意义
        sql = f"""
        SELECT subject_matter, count(*) AS cnt
        FROM {BUSINESS_TABLE}
        WHERE subject_matter IS NOT NULL AND subject_matter <> ''
        GROUP BY subject_matter
        ORDER BY cnt DESC, subject_matter
        LIMIT %s
        """
        return self._query(sql, (safe_limit,))

    def by_purchaser(self, purchaser: str, limit: int = 20) -> list[dict]:
        """某个采购人的采购记录，返回记录列表（列见 `_RECORD_COLUMNS`）。"""
        name = (purchaser or "").strip()
        if not name:  # 空关键词查不出任何东西，省掉一次远程往返
            return []
        safe_limit = _safe_limit(limit)
        sql = f"""
        SELECT {_RECORD_COLUMNS}
        FROM {BUSINESS_TABLE}
        WHERE purchaser = %s
        ORDER BY created_at DESC NULLS LAST, project_id
        LIMIT %s
        """
        return self._query(sql, (name, safe_limit))

    def by_supplier(self, supplier: str, limit: int = 20) -> list[dict]:
        """某个供应商（中标人）的成交记录，返回记录列表。"""
        name = (supplier or "").strip()
        if not name:
            return []
        safe_limit = _safe_limit(limit)
        sql = f"""
        SELECT {_RECORD_COLUMNS}
        FROM {BUSINESS_TABLE}
        WHERE supplier = %s
        ORDER BY created_at DESC NULLS LAST, project_id
        LIMIT %s
        """
        return self._query(sql, (name, safe_limit))

    def amount_range(
        self,
        min_amount: float | None = None,
        max_amount: float | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """交易额落在 [min_amount, max_amount] 内的记录，按金额从高到低返回。

        边界归一化放在 try 之外（非数字直接抛 `ValueError`）：静默忽略一个写错的边界，
        会让"金额大于 100 万"悄悄变成"全表前 50 条"，答案错得没有任何提示。
        两个边界都给 `None` 时不报错也不返回空，退化成"金额最高的若干条"——
        LLM 有时只问"金额情况",返回空会让它以为库里没有数据。
        """
        low = _as_amount(min_amount)
        high = _as_amount(max_amount)
        safe_limit = _safe_limit(limit, default=50)

        clauses: list[str] = []
        params: list[Any] = []
        if low is not None:
            clauses.append("amount >= %s")
            params.append(low)
        if high is not None:
            clauses.append("amount <= %s")
            params.append(high)
        # f-string 里只有本模块常量与固定片段（列名、表名、AND），边界值与 limit 一律走占位符
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"""
        SELECT {_RECORD_COLUMNS}
        FROM {BUSINESS_TABLE}
        {where}
        ORDER BY amount DESC NULLS LAST, project_id
        LIMIT %s
        """
        params.append(safe_limit)
        return self._query(sql, params)

    def location_dist(self, limit: int = 20) -> list[dict]:
        """按所在地（市区）分组统计，返回 `[{"location_city": ..., "cnt": N}]`。"""
        safe_limit = _safe_limit(limit)
        sql = f"""
        SELECT location_city, count(*) AS cnt
        FROM {BUSINESS_TABLE}
        WHERE location_city IS NOT NULL AND location_city <> ''
        GROUP BY location_city
        ORDER BY cnt DESC, location_city
        LIMIT %s
        """
        return self._query(sql, (safe_limit,))

    def trend(self, group_by: str = "month", limit: int = 20) -> list[dict]:
        """按时间趋势统计，返回 `[{"period": "2024-03", "cnt": N}]`（year 时为 "2024"）。

        `group_by` 只接受 "month" / "year"，其它值抛 `ValueError`——它决定 SQL 里的表达式，
        属于标识符位置，不能参数化。
        """
        key = (group_by or "").strip().lower()
        if key not in _TREND_EXPRESSIONS:
            raise ValueError(f"不支持的时间粒度: {group_by}（只支持 month / year）")
        # 表达式取自上行校验过的常量表：外部输入只当 key 用，不参与拼装
        expression = _TREND_EXPRESSIONS[key]
        safe_limit = _safe_limit(limit)
        sql = f"""
        SELECT {expression} AS period, count(*) AS cnt
        FROM {BUSINESS_TABLE}
        WHERE created_at IS NOT NULL
        GROUP BY {expression}
        ORDER BY period
        LIMIT %s
        """
        return self._query(sql, (safe_limit,))

    def purchaser_ranking(self, limit: int = 10) -> list[dict]:
        """采购次数排行：出现次数最多的采购人。

        用途：回答"哪些单位采购最活跃 / 谁是我们的重点客户"。
        参数：limit —— 返回条数，超上限自动截断，非法值退回默认。
        返回：`[{"purchaser": "某某局", "cnt": 12}, ...]`，按 cnt 降序；查询失败返回 `[]`。

        排序里带 `purchaser` 次级键：采购次数相同的单位很常见，没有次级键时 PG 可以任意
        决定平局行序，同一问题两次回答的名单顺序会不一样（与 top_subject 同一处理）。
        """
        safe_limit = _safe_limit(limit, default=10)
        # 空串与 NULL 排除在分组之外：否则排行第一可能是"没有名字的采购人"
        sql = f"""
        SELECT purchaser, count(*) AS cnt
        FROM {BUSINESS_TABLE}
        WHERE purchaser IS NOT NULL AND purchaser <> ''
        GROUP BY purchaser
        ORDER BY cnt DESC, purchaser
        LIMIT %s
        """
        return self._query(sql, (safe_limit,))

    def supplier_ranking(self, limit: int = 10) -> list[dict]:
        """中标次数排行：中标最多的供应商（中标人）。

        用途：回答"哪些企业中标最多"，与 purchaser_ranking 一起支撑"谁在主导这类项目"。
        参数：limit —— 返回条数，超上限自动截断，非法值退回默认。
        返回：`[{"supplier": "某某公司", "cnt": 8}, ...]`，按 cnt 降序；查询失败返回 `[]`。
        """
        safe_limit = _safe_limit(limit, default=10)
        sql = f"""
        SELECT supplier, count(*) AS cnt
        FROM {BUSINESS_TABLE}
        WHERE supplier IS NOT NULL AND supplier <> ''
        GROUP BY supplier
        ORDER BY cnt DESC, supplier
        LIMIT %s
        """
        return self._query(sql, (safe_limit,))

    def avg_amount_by_subject(self, subject: str, limit: int = 20) -> list[dict]:
        """按标的物关键词查中标记录（模糊匹配），并附上该标的物的平均中标金额。

        用途：回答"某类标的物的项目都有谁做、金额什么水平"，比只给一个 AVG 数字更能支撑
        结论——LLM 需要看到具体项目才能引用。
        参数：subject —— 标的物关键词，按 `ILIKE %关键词%` 模糊匹配 subject_matter，
              空关键词直接返回 `[]`（不查库：空串会让 ILIKE '%%' 命中全表）。
              limit —— 返回条数，超上限自动截断，非法值退回默认。
        返回：`[{"project_name", "purchaser", "supplier", "amount", "bid_time",
              "avg_amount", "match_count"}, ...]`
              按 amount 降序；查询失败返回 `[]`。

        avg_amount 用窗口函数 `AVG(amount) OVER ()` 而不是再查一次：一条 SQL 拿到"每行 +
        该组的平均值"，省一次往返；窗口函数在 LIMIT 之前算完，故它统计的是**全部命中行**，
        不是被截断后的前 limit 行——这正是"这类标的物的平均金额"该有的口径。

        但"全量均值"和"top-N 行"是两种口径，摆进同一条记录极易被读错：返回的行按金额降序，
        随行的 amount 是这批里最贵的几个，而 avg_amount 是全量均值，两者常差一两个数量级。
        故同时给 match_count（同样走窗口函数，统计全部命中行）把分母摆出来——"命中 219 条，
        这条 8803 万是最贵的，全量均价 2.9 万"就成了一句可自证的话。窗口函数会忽略 NULL 的
        amount，故 avg_amount 的分母实际是"命中行里有金额的那些"，可能小于 match_count。
        """
        keyword = (subject or "").strip()
        if not keyword:
            return []
        safe_limit = _safe_limit(limit)
        # 关键词与模式串分开：模式串是**值**（走 %s），字段名是本模块常量，两者都不拼进 SQL
        sql = f"""
        SELECT project_name, purchaser, supplier, amount, bid_time,
               avg(amount) OVER () AS avg_amount,   -- 全部命中行（LIMIT 之前算完）
               count(*)    OVER () AS match_count   -- 上面这个均值基于多少行
        FROM {BUSINESS_TABLE}
        WHERE subject_matter ILIKE %s
        ORDER BY amount DESC NULLS LAST, project_id
        LIMIT %s
        """
        return self._query(sql, (_like_param(keyword), safe_limit))

    def search_by_keyword(self, keyword: str, limit: int = 20) -> list[dict]:
        """跨字段模糊搜索：关键词出现在标的物 / 项目名称 / 采购人 / 供应商任一即可。

        用途：用户问题里给的是零散线索（一个公司名、一个项目词），不告诉我们要按哪个字段查，
        所以四个字段一起 OR——分四次查要么多三个来回，要么得先让 LLM 猜字段（猜错就查空）。
        参数：keyword —— 关键词，`ILIKE %关键词%`；空白关键词直接返回 `[]`。
              limit —— 返回条数，超上限自动截断，非法值退回默认。
        返回：`[{"project_name", "subject_matter", "purchaser", "supplier", "amount"}, ...]`
              按 bid_time 降序（最新的在前）；查询失败返回 `[]`。
        """
        kw = (keyword or "").strip()
        if not kw:
            return []
        safe_limit = _safe_limit(limit)
        # 同一个模式串参数在四个字段上复用（psycopg 按下标绑定，同一个 %s 用四次也没问题）
        sql = f"""
        SELECT project_name, subject_matter, purchaser, supplier, amount
        FROM {BUSINESS_TABLE}
        WHERE subject_matter  ILIKE %s
           OR project_name    ILIKE %s
           OR purchaser       ILIKE %s
           OR supplier        ILIKE %s
        ORDER BY bid_time DESC NULLS LAST, project_id
        LIMIT %s
        """
        pattern = _like_param(kw)
        return self._query(sql, (pattern, pattern, pattern, pattern, safe_limit))

    def stats_overview(self) -> dict:
        """业务库整体概览：记录数、总金额、时间跨度、标的物 / 采购人种类数。

        用途：回答"这个库大概有什么"这类开场问题，也是排查"数据到底导进去没有"的第一入口。
        参数：无。
        返回：`{"total_records", "total_amount", "min_bid_time", "max_bid_time",
              "distinct_subject", "distinct_purchaser"}`；查询失败返回 `{}`（空 dict 表示
              "没查到"，与"表是空的"区分开）。
              表为空时 total_records 为 0、total_amount 为 None（SQL 的 SUM 对空集返回 NULL，
              不伪造成 0.0——0 会被读成"有过交易但金额为零"）。

        一条 SQL 查完而不是六条：六个数字属于同一张表同一时刻，分开查不仅多五个来回，
        还可能因为并发导入拿到互相矛盾的一组数（总数 100，金额却是 101 条的）。
        """
        sql = f"""
        SELECT count(*)                        AS total_records,
               sum(amount)                     AS total_amount,
               min(bid_time)                   AS min_bid_time,
               max(bid_time)                   AS max_bid_time,
               count(DISTINCT subject_matter)  AS distinct_subject,
               count(DISTINCT purchaser)       AS distinct_purchaser
        FROM {BUSINESS_TABLE}
        """
        rows = self._query(sql)
        return rows[0] if rows else {}

    # --- 会话 / 消息 / 反馈 CRUD ---

    def ensure_schema(self) -> bool:
        """建会话 / 消息 / 反馈三张表（幂等，已存在则跳过），成功返回 True。

        与业务表不同，这三张表由本模块负责：它们是应用自己写的存储，不该让使用者
        先手工建表。多条 DDL 放在同一个事务里（Postgres 支持事务化 DDL），
        要么三张都建好，要么一条都不留，不会出现"消息表建了、会话表没有"的半成品。

        升级注意（老库必读）：`CREATE TABLE IF NOT EXISTS` 只保证"表在"，**不会**给已存在的表
        补列、补外键。本模块早期版本建的 messages 没有 sources / tool_name、也没有外键，
        若你的库是那时建的，本方法会静默跳过，随后 `save_message` 报"字段不存在"并返回 -1。
        当时还没有任何写入路径，那张表一定是空的，直接 `DROP TABLE messages;` 再跑一次本方法
        即可（业务表 bidding_records 不受影响）。
        """
        statements = (
            # created_at 一律 NOT NULL DEFAULT now()：写入方（工具层 / API 层）不该关心时间字段，
            # 也不该能写入一个没有时间的会话
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id         SERIAL PRIMARY KEY,
                session_id TEXT,
                title      TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """,
            # messages 比另外两张表"严"：role / content 都 NOT NULL（一条没有说话人的消息没有
            # 意义），conversation_id 有外键且 ON DELETE CASCADE —— 删会话时消息跟着走，不必在
            # delete_conversation 里手工级联，将来新增写入路径也不会漏删。
            # sources 存 JSONB 而不是 TEXT：参考来源本就是结构化的（工具名、片段、分数），
            # 存成 JSON 字符串后想按来源过滤只能全表 LIKE，JSONB 至少能直接用 ->> 取字段。
            """
            CREATE TABLE IF NOT EXISTS messages (
                id              BIGSERIAL PRIMARY KEY,
                conversation_id BIGINT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                role            TEXT NOT NULL,
                content         TEXT NOT NULL,
                sources         JSONB,
                tool_name       TEXT,
                created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id              SERIAL PRIMARY KEY,
                conversation_id INTEGER,
                message_id      INTEGER,
                rating          TEXT,
                comment         TEXT,
                created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """,
            # 会话下的消息 / 反馈只能按 conversation_id 查，没有索引就是全表扫描。
            # messages 的索引带上 created_at：load_messages 的排序与"取最近 N 条"都走它，
            # 复合索引一次覆盖"先定位会话、再按时间排"两步。
            # feedback 没建外键（它的删除仍由 delete_conversation 显式完成）——给已存在的表
            # 补外键要连带确认历史数据没有孤儿行，不在这轮改动范围内。
            "CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages (conversation_id, created_at)",
            "CREATE INDEX IF NOT EXISTS idx_feedback_conversation ON feedback (conversation_id)",
        )
        try:
            with self._conn() as conn, conn.cursor() as cur:
                for statement in statements:
                    cur.execute(statement)
        except Exception:
            logger.exception("建表失败（ensure_schema）")
            return False
        logger.info("会话 / 反馈表已就绪")
        return True

    def create_conversation(self, title: str, session_id: str | None = None) -> int:
        """新建会话，返回 `conversation_id`；失败返回 `-1`。

        `-1` 而不是 0 作失败哨兵：SERIAL 从 1 开始自增，负数不可能是合法 id，
        调用方用 `if cid < 0` 判断即可（与 docs/开发文档.md §5.3 里 -1 表示异常的约定一致）。
        """
        sql = "INSERT INTO conversations (session_id, title) VALUES (%s, %s) RETURNING id"
        rows = self._query(sql, (session_id, title))
        if not rows:
            return -1
        return int(rows[0]["id"])

    def list_conversations(self, limit: int = 50) -> list[dict]:
        """会话列表（新建的在前），返回 `[{"id", "session_id", "title", "created_at"}]`。"""
        safe_limit = _safe_limit(limit, default=50)
        sql = """
        SELECT id, session_id, title, created_at
        FROM conversations
        ORDER BY created_at DESC NULLS LAST, id DESC
        LIMIT %s
        """
        return self._query(sql, (safe_limit,))

    def delete_conversation(self, conversation_id: int) -> bool:
        """删除会话及其消息 / 反馈，返回是否真的删掉了一条会话。

        messages 不再手工删：外键的 ON DELETE CASCADE 会随会话一起带走它们（见
        `ensure_schema`），级联和下面两条 DELETE 处在同一个事务里，同样不会删一半。
        仍需手工删的是 feedback——它没有外键。`id DESC` 之外的顺序无关紧要，这里不排序。
        """
        cid = _as_id(conversation_id, "conversation_id")
        try:
            with self._conn() as conn, conn.cursor() as cur:
                cur.execute("DELETE FROM feedback WHERE conversation_id = %s", (cid,))
                cur.execute("DELETE FROM conversations WHERE id = %s RETURNING id", (cid,))
                deleted = cur.fetchone() is not None
        except Exception:
            logger.exception("删除会话失败：conversation_id=%s", cid)
            return False
        return deleted

    def save_feedback(
        self,
        conversation_id: int,
        message_id: int,
        rating: str,
        comment: str | None = None,
    ) -> bool:
        """保存一条反馈（点赞 / 点踩 + 可选评论），返回是否写入成功。"""
        cid = _as_id(conversation_id, "conversation_id")
        mid = _as_id(message_id, "message_id")
        sql = """
        INSERT INTO feedback (conversation_id, message_id, rating, comment)
        VALUES (%s, %s, %s, %s)
        RETURNING id
        """
        return bool(self._query(sql, (cid, mid, rating, comment)))

    def save_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
        sources: Any = None,
        tool_name: str | None = None,
    ) -> int:
        """写入一条会话消息，返回新消息 `id`；失败返回 `-1`。

        用途：把一轮问答的 user / assistant 消息落库，供刷新页面后恢复历史。字段与前端落库
        契约 `PersistedMessage` 对齐（role / content / sources / toolName）；契约里的
        `thinking` 与耗时**不落库**，故本方法没有这两个参数。
        参数：conversation_id —— 所属会话 id，必须是 conversations 里已有的行（有外键约束）。
              role            —— `'user'` / `'assistant'`。
              content         —— 消息正文。
              sources         —— 参考来源，dict / list，或 None（没有来源时）。
              tool_name       —— 本轮调用的工具名，没调工具时传 None。
        返回：新消息 id；写入失败返回 `-1`（与 create_conversation 同一套哨兵约定）。
              conversation_id 不是整数时抛 `ValueError`——那是调用方的缺陷而非运营故障，
              与 save_feedback / delete_conversation 的处理一致。
              会话 id 不存在、或列不匹配（老库的 messages 表）同样落成 -1，原因只进日志。

        `sources` 必须用 `Jsonb` 包一层：JSONB 列收到裸 dict 时 psycopg 会把它当普通 Python
        对象去适配而直接报错。`None` **不**包装——它该落成 SQL NULL（"这条没有来源"），
        而不是 JSON 的 null（"来源字段存在，值为 null"），读端对这两者的判断不同。

        role 不做枚举校验：写入方是自家 API 层而不是 LLM 的自由文本，值域由前端契约保证；
        真正容易犯的"忘了传 role / content"由 NOT NULL 在库这一层挡下，落成 -1 + 日志。
        """
        cid = _as_id(conversation_id, "conversation_id")
        payload = Jsonb(sources) if sources is not None else None
        sql = """
        INSERT INTO messages (conversation_id, role, content, sources, tool_name)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id
        """
        rows = self._query(sql, (cid, role, content, payload, tool_name))
        if not rows:
            return -1
        return int(rows[0]["id"])

    def load_messages(self, conversation_id: int, limit: int = 100) -> list[dict]:
        """读回某会话的消息，返回 `[{"id", "role", "content", "sources", "tool_name",
        "created_at"}, ...]`；失败返回 `[]`。

        用途：刷新页面 / 重建上下文时按会话还原对话。
        参数：conversation_id —— 会话 id。
              limit —— 最多取几条，默认 100，超 MAX_LIMIT 自动截断。
        返回：按 created_at **升序**（前端顺序渲染即可）；查询失败返回 `[]`。

        取的是**最近** limit 条、而不是最早的 limit 条：先按 DESC 截断再翻回升序，长会话
        （超过 limit 条）丢掉的才是开头那几句寒暄，而不是用户刚说的话——按 ASC 直接 LIMIT
        会让最新消息永远进不了上下文，前端看起来像"对话卡住了"。
        子查询里截断、外层只排序不加 LIMIT，故返回顺序天然就是展示顺序。

        排序带 `id` 次级键：`created_at` 默认 now()，而 now() 是**事务开始时刻**，同一个
        事务里连写两条会拿到完全相同的时间戳，只按时间排时两者的先后由 PG 任意决定。
        sources 由驱动的默认加载器反序列化成 dict / list，这里不需要手工 json.loads。
        """
        cid = _as_id(conversation_id, "conversation_id")
        safe_limit = _safe_limit(limit, default=100)
        sql = """
        SELECT id, role, content, sources, tool_name, created_at
        FROM (
            SELECT id, role, content, sources, tool_name, created_at
            FROM messages
            WHERE conversation_id = %s
            ORDER BY created_at DESC, id DESC
            LIMIT %s
        ) AS recent
        ORDER BY created_at, id
        """
        return self._query(sql, (cid, safe_limit))


__all__ = [
    "ALLOWED_AGG",
    "ALLOWED_GROUP_FIELDS",
    "BUSINESS_TABLE",
    "MAX_LIMIT",
    "PostgresClient",
    "validate_agg",
    "validate_field",
]


if __name__ == "__main__":
    # 简易自测：`python -m src.database.postgresql_client`（在 bidding-agent/ 下执行）。
    # 只读——不建表、不写数据，用来在真库上快速看一眼各查询实际返回什么；
    # 正式的断言归 tests/，这里不重复做判定，只打印，方便肉眼核对。
    import json
    import logging
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # 库本身不读 .env（那是应用启动时的事，见 api/server.py 的 lifespan）；自测块要能脱离
    # 应用独立跑起来，所以在这里补一次。已存在的环境变量优先，不覆盖。
    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    except ImportError:  # 没装 python-dotenv 时仍可跑，只是要求自己先导出环境变量
        pass

    client = PostgresClient()
    status = client.health()
    print("health:", status)
    if not status["ok"]:
        raise SystemExit(f"数据库不可用：{status['error']}（自测跳过）")

    # 通配符转义的效果：这几个字符不转义会被当成"匹配任意串 / 任意单字符"
    print("\n_like_param('100%') =", _like_param("100%"), "（% 已被转义，不会变成全表匹配）")

    for name, run in (
        ("stats_overview()", lambda: client.stats_overview()),
        ("purchaser_ranking(3)", lambda: client.purchaser_ranking(3)),
        ("supplier_ranking(3)", lambda: client.supplier_ranking(3)),
        ("search_by_keyword('空调', 2)", lambda: client.search_by_keyword("空调", 2)),
        ("avg_amount_by_subject('空调', 2)", lambda: client.avg_amount_by_subject("空调", 2)),
        ("top_subject(3) / by_purchaser 等既有方法", lambda: client.top_subject(3)),
    ):
        print(f"\n--- {name} ---")
        print(json.dumps(run(), ensure_ascii=False, indent=2))

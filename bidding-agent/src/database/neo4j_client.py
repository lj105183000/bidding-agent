"""Neo4j 知识图谱客户端（成员 C）。

用途
----
把招投标数据的星型图谱查询封装成 6 个只读方法，供 src/tools/rag_tools.py 的
search_knowledge_graph 工具调用。本模块只负责"查得到什么"，不负责措辞与包装——
成功/失败的结构化包装由工具层统一完成。

依赖
----
neo4j 官方驱动 6.x（`GraphDatabase.driver`）。配置来自环境变量：
`NEO4J_URI` / `NEO4J_USERNAME` / `NEO4J_PASSWORD`（必填）、`NEO4J_DATABASE`
（可选，留空表示使用 driver 的默认库）。这些变量由本模块在 import 时
自动从仓库根的 `.env` 加载，故 `python -c "from src.database.neo4j_client import ..."`
这类脱离应用的用法不必先手动 export。

数据模型（星型 schema，中心节点是标的物）
----------------------------------------
    (:SubjectMatter {name, tradeFrequency})                 标的物 + 采购频次
        -[:HAS_TRANSACTION]->(:Transaction {id, amount})    交易记录
        -[:PURCHASED_BY]->(:Purchaser {name})               采购人
        -[:AGENT_BY]->(:Agency {name})                      代理机构
        -[:LOCATED_AT]->(:Location {name})                  项目地址
        -[:SUPPLIED_BY]->(:Supplier {name})                 中标人 / 供应商

设计要点（改本文件前先读）
--------------------------
- **懒连接**：`__init__` 只创建 driver（不发任何网络包），连通性由 `health()` 按需检查。
  若在构造或 import 阶段就要求连上远程，没有图谱的机器上跑测试与 CI 会直接挂。
- **白名单是唯一防线**：Cypher 的标签、关系类型、属性名**都不能参数化**
  （`$param` 只能出现在"值"的位置），只能拼字符串，所以凡是要拼进语句的标识符，
  必须先过 `_safe_label` / `_safe_rel` / `_safe_prop`；值则一律走 `$` 占位符。
- **失败返回空而不抛异常**：工具层统一把失败包成结构化错误，Agent 不能因为图谱宕机
  而崩，故查询方法异常时返回 `[]` / `{}`，完整异常只进日志。代价是"查不到"与
  "查询失败"在返回值上不可区分——需要区分时用 `health()`（工具层判断"未连接"就走这条）。
"""

import logging
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import AuthError, ConfigurationError, ServiceUnavailable

# 自动定位项目根目录的 .env（假设本文件在 src/database/ 下），与 postgresql_client.py 同一处理。
# 必须由模块自己加载而不是指望调用方：本客户端要能被脚本、`python -c`、测试单独构造，
# 而 src/config.py（集中加载 .env 的那层）在本仓库尚未落地，少了这行就只有"先手动 export
# 一遍环境变量"才用得起来。load_dotenv 默认不覆盖已有环境变量，故不影响显式传参的用法。
load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

logger = logging.getLogger(__name__)

# 标签 / 属性 / 关系白名单：Cypher 不支持参数化标识符，拼进语句前必须用它们校验。
# 节点类型增加时这几行与模块 docstring 的数据模型要一起改。
ALLOWED_LABELS = {"SubjectMatter", "Purchaser", "Agency", "Location", "Supplier", "Transaction"}
ALLOWED_PROPS = {"name", "tradeFrequency", "id", "amount"}
# 关系类型同样不能参数化，与标签属于同一条防线
ALLOWED_RELS = {"HAS_TRANSACTION", "PURCHASED_BY", "AGENT_BY", "LOCATED_AT", "SUPPLIED_BY"}

# 单次查询行数上限：limit 多由 LLM 给出，图谱不是分页导出通道；
# 这个上限同时是 Aura 实例的保护线（一条查询拉爆连接与配额对演示毫无帮助）
MAX_LIMIT = 200

# 驱动异常 → 中文原因。只报方向不报细节：驱动的原始报文里带主机名与端口，
# 而 health() 的结果会进 /health 与用户可见区。按"具体在前"的顺序匹配。
_REASON_BY_EXCEPTION = (
    (AuthError, "账号或密码不正确（检查 NEO4J_USERNAME / NEO4J_PASSWORD）"),
    (ServiceUnavailable, "无法连接到图谱服务（检查地址与网络，Aura 实例可能已暂停）"),
    (ConfigurationError, "连接配置无效（检查 NEO4J_URI 的协议前缀与端口）"),
)


def _explain(exc: Exception) -> str:
    """把驱动异常翻译成中文原因，不含地址、端口与原始报文（那些只进日志）。"""
    for exc_type, reason in _REASON_BY_EXCEPTION:
        if isinstance(exc, exc_type):
            return reason
    return "未知错误，详见后端日志"


def _safe_label(label: str) -> str:
    """标签白名单校验：不通过就抛，绝不放行任何未登记的值去拼 Cypher。"""
    if label not in ALLOWED_LABELS:
        raise ValueError(f"非法节点标签：{label}")
    return label


def _safe_prop(prop: str) -> str:
    """属性名白名单校验（属性名同样不能参数化）。"""
    if prop not in ALLOWED_PROPS:
        raise ValueError(f"非法属性名：{prop}")
    return prop


def _safe_rel(rel: str) -> str:
    """关系类型白名单校验。"""
    if rel not in ALLOWED_RELS:
        raise ValueError(f"非法关系类型：{rel}")
    return rel


def _safe_limit(limit: Any, default: int = 20) -> int:
    """把调用方（多半是 LLM）给的 limit 收敛成合法整数：非法退回默认值，超限截断。

    不信任外部传来的数值——`LIMIT $limit` 收到字符串或浮点会被驱动拒绝，
    整条查询连"有没有结果"都问不出来。
    """
    try:
        value = int(limit)
    except (TypeError, ValueError):
        return default
    return max(1, min(value, MAX_LIMIT))


def _as_amount(value: Any) -> float | None:
    """交易金额归一化为 float；非数值（含 CSV 导入常见的 "1,234.50" 字符串）尽力解析。

    格式化层要拿金额做比较与求和，字符串和整数混着来迟早出错，故在出口处统一类型。
    """
    if value is None or isinstance(value, bool):  # bool 是 int 子类，别把 True 变成 1.0
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _summarize(cypher: str) -> str:
    """把多行 Cypher 压成单行并截断，免得一条长语句把日志刷屏。"""
    one_line = " ".join(cypher.split())
    return one_line if len(one_line) <= 160 else one_line[:157] + "..."


class Neo4jClient:
    """招投标知识图谱的只读客户端（6 类查询模板）。

    实例不是线程安全的共享状态持有者，但 driver 的连接池本身可并发使用；
    工具层的 4 线程并行会同时调这些方法，故各方法不修改实例状态。
    """

    def __init__(self) -> None:
        """读取 NEO4J_* 配置并创建 driver（**不**在这里连服务端）。"""
        uri = (os.getenv("NEO4J_URI") or "").strip()
        username = (os.getenv("NEO4J_USERNAME") or "").strip()
        # 密码只去行尾换行（Windows 上手写的 .env 常带 \r），不做 strip——首位空格可能是密码本身
        password = (os.getenv("NEO4J_PASSWORD") or "").rstrip("\r\n")
        missing = [
            name
            for name, value in (
                ("NEO4J_URI", uri),
                ("NEO4J_USERNAME", username),
                ("NEO4J_PASSWORD", password),
            )
            if not value
        ]
        if missing:
            raise ValueError(f"缺少必需配置：{'、'.join(missing)}（请检查 .env，可参照 .env.example）")

        # 库名留空表示用 driver 的默认库：dotenv 把 `NEO4J_DATABASE=` 读成空串而不是 None，
        # 空串会被服务端当成"名为空的数据库"直接报错，故统一收敛为 None
        self.database = (os.getenv("NEO4J_DATABASE") or "").strip() or None

        try:
            # 懒连接：GraphDatabase.driver 只准备配置与连接池，一个网络包都不发
            self._driver = GraphDatabase.driver(uri, auth=(username, password))
        except Exception as exc:
            # 不回显 uri/用户名/密码，只给排查方向；原始异常进日志
            logger.exception("Neo4j driver 初始化失败")
            raise RuntimeError(f"NEO4J_URI 无法初始化驱动：{_explain(exc)}") from exc

    def close(self) -> None:
        """关闭 driver 与其连接池（FastAPI lifespan 退出时调用）。可重复调用。"""
        try:
            self._driver.close()
        except Exception:
            logger.exception("关闭 Neo4j driver 失败")

    def health(self) -> dict:
        """连接健康检查，返回 {"ok": bool, "error": str | None}。失败不抛。"""
        try:
            self._driver.verify_connectivity()
        except Exception as exc:
            logger.exception("Neo4j 连接检查失败")
            return {"ok": False, "error": f"知识图谱连接失败：{_explain(exc)}"}
        return {"ok": True, "error": None}

    def _run(self, cypher: str, **params: Any) -> list[dict]:
        """执行只读 Cypher，返回 `list[dict]`；任何失败只记日志并返回 `[]`。

        值一律通过 `**params` 交给驱动（对应语句里的 `$name`）；本方法**不接受**
        标签 / 关系 / 属性名——它们只能由查询方法先用 `_safe_*` 校验后拼进语句。
        """
        try:
            # execute_read 是托管读事务：驱动会在瞬时故障时自动重试，写语句不会走到这里
            with self._driver.session(database=self.database) as session:
                rows = session.execute_read(lambda tx: tx.run(cypher, **params).data())
        except Exception:
            # 只记语句骨架（含白名单标识符与 $占位符），不记参数值：日志里不需要用户关键词
            logger.exception("Neo4j 查询失败：%s", _summarize(cypher))
            return []
        logger.debug("Neo4j 查询返回 %d 行：%s", len(rows), _summarize(cypher))
        return rows

    # --- 6 类查询模板 ---

    def search_entity(self, keyword: str, limit: int = 20) -> list[dict]:
        """按名称模糊搜索标的物 / 采购人 / 供应商（大小写不敏感）。

        返回 `[{"label": ..., "name": ..., "tradeFrequency": N 或 None}]`。
        三类实体一条语句查完（UNION）：调用方通常只拿到一个名字，并不知道它属于哪一类，
        分三次往返要么多花两个来回，要么得先让 LLM 猜类型（猜错就查空）。
        """
        kw = (keyword or "").strip()
        if not kw:
            return []
        safe_limit = _safe_limit(limit)
        subject = _safe_label("SubjectMatter")
        purchaser = _safe_label("Purchaser")
        supplier = _safe_label("Supplier")
        freq = _safe_prop("tradeFrequency")
        cypher = f"""
        MATCH (s:{subject}) WHERE toLower(s.name) CONTAINS toLower($kw)
        RETURN '{subject}' AS label, s.name AS name, s.{freq} AS tradeFrequency
        LIMIT $limit
        UNION
        MATCH (p:{purchaser}) WHERE toLower(p.name) CONTAINS toLower($kw)
        RETURN '{purchaser}' AS label, p.name AS name, null AS tradeFrequency
        LIMIT $limit
        UNION
        MATCH (u:{supplier}) WHERE toLower(u.name) CONTAINS toLower($kw)
        RETURN '{supplier}' AS label, u.name AS name, null AS tradeFrequency
        LIMIT $limit
        """
        try:
            rows = self._run(cypher, kw=kw, limit=safe_limit)
        except Exception:
            logger.exception("实体搜索失败：%s", kw)
            return []

        # 排序放在 Python 侧：跨 UNION 分支写 ORDER BY 时"作用在哪一层、null 排前还是排后"
        # 都容易踩坑。有频次的（标的物）优先，同频次按名称排，保证输出顺序稳定
        def _sort_key(row: dict) -> tuple:
            value = _as_amount(row.get("tradeFrequency"))
            return (value is None, -(value or 0.0), str(row.get("name") or ""))

        rows.sort(key=_sort_key)
        return rows[:safe_limit]

    def items_by_entity(self, entity: str, limit: int = 20) -> list[dict]:
        """输入实体名，返回它关联的标的物。

        返回 `[{"subject": ..., "relation": "PURCHASED_BY"/"SUPPLIED_BY"/"SELF", "entity": ...}]`。
        三类身份都试一遍：同一串名字在数据里可能既是采购人又是标的物，只猜一种会漏结果。
        """
        name = (entity or "").strip()
        if not name:
            return []
        safe_limit = _safe_limit(limit)
        subject = _safe_label("SubjectMatter")
        rows: list[dict] = []
        try:
            # 采购人 / 供应商：从实体反向找到标的物（箭头方向见模块 docstring 的数据模型）
            for label, rel in (("Purchaser", "PURCHASED_BY"), ("Supplier", "SUPPLIED_BY")):
                node = _safe_label(label)
                relation = _safe_rel(rel)
                cypher = f"""
                MATCH (x:{node} {{name: $name}})<-[:{relation}]-(s:{subject})
                RETURN DISTINCT s.name AS subject
                LIMIT $limit
                """
                for row in self._run(cypher, name=name, limit=safe_limit):
                    rows.append({"subject": row.get("subject"), "relation": relation, "entity": name})

            # 输入本身就是标的物：直接返回它自己。SELF 只是给调用方的标记，不拼进 Cypher
            cypher = f"MATCH (s:{subject} {{name: $name}}) RETURN s.name AS subject LIMIT $limit"
            for row in self._run(cypher, name=name, limit=safe_limit):
                rows.append({"subject": row.get("subject"), "relation": "SELF", "entity": name})
        except Exception:
            logger.exception("实体关联标的物查询失败：%s", name)
            return []
        return rows

    def top_traded(self, limit: int = 10) -> list[dict]:
        """采购频次最高的标的物排行，返回 `[{"name": ..., "tradeFrequency": N}]`。"""
        safe_limit = _safe_limit(limit, default=10)
        subject = _safe_label("SubjectMatter")
        freq = _safe_prop("tradeFrequency")
        # WHERE 先滤掉没有频次的节点：否则 null 会参与排序（Neo4j 里 null 视为最大值）
        cypher = f"""
        MATCH (s:{subject})
        WHERE s.{freq} IS NOT NULL
        RETURN s.name AS name, s.{freq} AS tradeFrequency
        ORDER BY tradeFrequency DESC
        LIMIT $limit
        """
        try:
            return self._run(cypher, limit=safe_limit)
        except Exception:
            logger.exception("采购频次排行查询失败")
            return []

    def entity_detail(self, entity: str, limit: int = 20) -> dict:
        """标的物的完整关系详情，返回 `{"subject", "suppliers", "purchasers", "agencies",
        "locations", "transactions"}`；标的物不存在时各项为空列表（键恒定存在，
        格式化层不必判断键有没有）。

        失败时返回 `{}`——与"标的物不存在"区分开，供工具层判断是"没查到"还是"查询故障"。
        """
        name = (entity or "").strip()
        detail: dict = {
            "subject": name,
            "suppliers": [],
            "purchasers": [],
            "agencies": [],
            "locations": [],
            "transactions": [],
        }
        if not name:
            return detail
        safe_limit = _safe_limit(limit)
        subject = _safe_label("SubjectMatter")
        try:
            # 先确认标的物存在：不存在就直接返回空结构，省掉后面 5 次远程查询
            exists = self._run(
                f"MATCH (s:{subject} {{name: $name}}) RETURN s.name AS name LIMIT 1", name=name
            )
            if not exists:
                return detail

            # 一类关系一条语句：串成一条多路 OPTIONAL MATCH 会先做笛卡尔积再折叠
            # （供应商 × 采购人 × 代理机构 …），行数与耗时都随关系数放大
            for key, label, rel in (
                ("suppliers", "Supplier", "SUPPLIED_BY"),
                ("purchasers", "Purchaser", "PURCHASED_BY"),
                ("agencies", "Agency", "AGENT_BY"),
                ("locations", "Location", "LOCATED_AT"),
            ):
                node = _safe_label(label)
                relation = _safe_rel(rel)
                cypher = f"""
                MATCH (s:{subject} {{name: $name}})-[:{relation}]->(x:{node})
                RETURN DISTINCT x.name AS name
                LIMIT $limit
                """
                detail[key] = [
                    row.get("name")
                    for row in self._run(cypher, name=name, limit=safe_limit)
                    if row.get("name")
                ]

            transaction = _safe_label("Transaction")
            has_transaction = _safe_rel("HAS_TRANSACTION")
            id_prop = _safe_prop("id")
            amount_prop = _safe_prop("amount")
            cypher = f"""
            MATCH (s:{subject} {{name: $name}})-[:{has_transaction}]->(t:{transaction})
            RETURN t.{id_prop} AS id, t.{amount_prop} AS amount
            LIMIT $limit
            """
            detail["transactions"] = [
                {"id": row.get("id"), "amount": _as_amount(row.get("amount"))}
                for row in self._run(cypher, name=name, limit=safe_limit)
            ]
        except Exception:
            # _run 已把查询失败降级成 []；这里兜住的是白名单校验与拼装阶段的意外（属于代码缺陷，
            # 必须留痕，故返回 {} 让调用方看得出是故障而不是"这个标的物没有关系"）
            logger.exception("标的物详情查询失败：%s", name)
            return {}
        return detail

    def graph_stats(self) -> dict:
        """各类节点数量统计，返回 `{"SubjectMatter": N, ...}`；某类不存在则为 0。"""
        # 按名称排序而非直接迭代集合：字符串集合的迭代顺序受哈希随机化影响，跨进程会变
        stats = {label: 0 for label in sorted(ALLOWED_LABELS)}
        # 一条语句查完（UNWIND 展开标签），比按标签循环发 6 次远程请求省一个数量级的往返
        cypher = "MATCH (n) UNWIND labels(n) AS label RETURN label AS label, count(*) AS count"
        try:
            for row in self._run(cypher):
                label = row.get("label")
                if label in stats:  # 白名单外的标签不进结果，输出形状对调用方恒定
                    stats[label] = int(row.get("count") or 0)
        except Exception:
            logger.exception("图谱统计查询失败")
            return {}
        return stats

    def supplier_by_subject(self, subject: str, limit: int = 20) -> list[dict]:
        """标的物的供应商列表，返回 `[{"supplier": ..., "amount": float 或 None}]`。

        amount 取自该标的物的交易记录；没有交易记录的标的物，供应商照样返回
        （OPTIONAL MATCH），此时 amount 为 None。
        """
        name = (subject or "").strip()
        if not name:
            return []
        safe_limit = _safe_limit(limit)
        subject_label = _safe_label("SubjectMatter")
        supplier_label = _safe_label("Supplier")
        transaction_label = _safe_label("Transaction")
        supplied_by = _safe_rel("SUPPLIED_BY")
        has_transaction = _safe_rel("HAS_TRANSACTION")
        amount_prop = _safe_prop("amount")
        # 用 max 聚合而不是逐条返回：同一标的物若有多条交易记录，逐条会把"一个供应商"放大成
        # 多行，amount 归属也说不清；取最大值给一个确定的代表值
        cypher = f"""
        MATCH (s:{subject_label} {{name: $name}})-[:{supplied_by}]->(u:{supplier_label})
        OPTIONAL MATCH (s)-[:{has_transaction}]->(t:{transaction_label})
        RETURN u.name AS supplier, max(t.{amount_prop}) AS amount
        LIMIT $limit
        """
        try:
            rows = self._run(cypher, name=name, limit=safe_limit)
        except Exception:
            logger.exception("供应商查询失败：%s", name)
            return []
        return [
            {"supplier": row.get("supplier"), "amount": _as_amount(row.get("amount"))} for row in rows
        ]


__all__ = ["ALLOWED_LABELS", "ALLOWED_PROPS", "ALLOWED_RELS", "MAX_LIMIT", "Neo4jClient"]

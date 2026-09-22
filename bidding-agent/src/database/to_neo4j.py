"""CSV → Neo4j 知识图谱导入脚本（成员 C）。

用途
----
把 batch/准备图谱CSV.py 派生的两份 CSV 灌进 Neo4j，供 src/database/neo4j_client.py 的
6 类查询与 src/tools/rag_tools.py 的 `search_knowledge_graph` 工具使用：

    data/processed/标的物_交易频次.csv                        -> (:SubjectMatter {name, tradeFrequency})
    data/processed/招标采购标的物信息提取训练数据_2_知识图谱.csv   -> 交易记录 + 4 类实体节点与关系

图 schema（与 neo4j_client.py 的查询**强绑定**，改这里必须同步改那边）
--------------------------------------------------------------------
    (:SubjectMatter {name: string, tradeFrequency: int})   标的物（图谱中心）
        -[:HAS_TRANSACTION]->(:Transaction {id: int, amount: float})   交易记录
        -[:PURCHASED_BY]->(:Purchaser {name: string})       采购人
        -[:AGENT_BY]->(:Agency {name: string})              代理机构
        -[:LOCATED_AT]->(:Location {name: string})          项目地址
        -[:SUPPLIED_BY]->(:Supplier {name: string})         中标人 / 供应商

用法
----
    python src/database/to_neo4j.py                  # 增量导入（MERGE，重复跑不产生重复数据）
    python src/database/to_neo4j.py --clear          # 先清空图谱再导入（推荐：改过 CSV 后用）
    python src/database/to_neo4j.py --password       # 交互式输入密码（不回显，不进 shell 历史）
    python src/database/to_neo4j.py --password mypass

依赖
----
neo4j 官方驱动 6.x、python-dotenv（读 .env）。配置项与 neo4j_client.py 同一组：
`NEO4J_URI` / `NEO4J_USERNAME` / `NEO4J_PASSWORD`（必填）、`NEO4J_DATABASE`（可选）。

设计要点（改本文件前先读）
--------------------------
- **为什么不用 LOAD CSV**：`LOAD CSV FROM 'file:///...'` 里的路径是**服务端**的文件系统。
  本项目连的是 Aura（云端托管实例），CSV 在本机磁盘上，服务端根本看不见这个路径，
  会直接报 "Couldn't load the external resource"；改用 `https://` 形式则要求两个 CSV
  挂在公网可访问的地址上，等于把项目数据公开出去。所以走 driver 把数据读进内存、
  用 `UNWIND $rows` 批量发给服务端——这套在 Aura 上也是官方推荐的导入姿势。
- **为什么空值必须跳过**：`MERGE (:Agency {name: ""})` 不会报错，它会安静地造出一个
  名为空的节点，然后把**所有**没有代理机构的项目挂到它上面（本表 8730 行里有 7692 行
  是空的）。结果是一个 7692 度的假中心节点，`entity_detail` 还会返回一个叫 "" 的代理机构。
  故在 Python 侧先把空串滤掉，而不是指望 Cypher 侧兜住。
- **为什么不复用 Neo4jClient**：与 to_postgresql.py 同一理由——那个客户端的契约是
  "失败返回空、绝不抛异常"，那是给 Agent 用的；导入脚本恰恰相反，出问题就要大声失败退出。
- **幂等靠 MERGE**：全篇没有一条 CREATE。Transaction 的 id 用 **CSV 行号**（1 起）而不是
  uuid：行号在"同一份 CSV 重跑"时稳定，MERGE 才会命中已有节点；uuid 每次跑都是新的，
  跑两遍图谱里就有两份交易记录。代价是**改过 CSV 的行序或行数后要配 `--clear` 重跑**，
  否则旧行号对应的 Transaction 会残留成孤点。
- **标识符只能拼字符串**：Cypher 的标签与关系类型不能参数化（`$param` 只能放在"值"的位置），
  所以 `LABELS` / `RELS` 是写死在本文件里的常量、绝不来自 CSV；只有值走 `$rows`。

"""

import argparse
import csv
import getpass
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import AuthError, ConfigurationError, ServiceUnavailable

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 仓库根（本文件在 src/database/ 下，上溯三层）：.env 与数据路径都以它为基准，
# 这样在任意工作目录下执行脚本行为都一致——CWD 相关的路径是"我这儿能跑"的经典来源
REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
FREQ_CSV = PROCESSED_DIR / "标的物_交易频次.csv"
GRAPH_CSV = PROCESSED_DIR / "招标采购标的物信息提取训练数据_2_知识图谱.csv"

# 每批发给服务端的行数。太小则往返次数多（8700 行 × 逐条 = 8700 个来回），
# 太大则单事务参数体积膨胀，Aura 免费实例上容易撞事务内存上限且失败后毫无进度可看。
BATCH_SIZE = 500

# 清空图谱时单批删除的节点数：同样是为了把"一次事务删全库"拆成可观察的小事务
CLEAR_BATCH = 5000

# 标签 / 关系类型与属性名（与 neo4j_client.py 的 ALLOWED_* 白名单一一对应）。
# 它们会被 f-string 拼进 Cypher——Cypher 不支持参数化标识符——所以只能来自本文件的常量，
# 任何情况下都不得从 CSV 内容拼进来。
LABEL_SUBJECT = "SubjectMatter"
LABEL_TRANSACTION = "Transaction"
PROP_NAME = "name"
PROP_FREQUENCY = "tradeFrequency"
PROP_ID = "id"
PROP_AMOUNT = "amount"

# 知识图谱 CSV 的列名 -> (节点标签, 关系类型)。顺序按 CSV 列序，日志读起来与文件一致。
# 单列一项的实体节点，同一份数据里既有多个标的物共用同一个采购商，也有同一标的物
# 对应多个采购商，故这 4 类都建成独立节点 + 关系，而不是把名字塞进标的物的属性里。
RELATION_COLUMNS = (
    ("采购商", "Purchaser", "PURCHASED_BY"),
    ("代理机构", "Agency", "AGENT_BY"),
    ("供应商", "Supplier", "SUPPLIED_BY"),
    ("所在地", "Location", "LOCATED_AT"),
)

FREQ_COLUMNS = ("标的物", "交易频次")
GRAPH_REQUIRED_COLUMNS = ("标的物", "交易额") + tuple(column for column, _, _ in RELATION_COLUMNS)

# 这些写法的意思是"没有值"，与 batch/准备图谱CSV.py 的 NULL_LIKE 同一套判断。
# 正常路径上那个脚本已经清过一遍，这里是第二道防线：CSV 可能是别人手改过的，
# 而漏掉一个 "-" 就会在图上多一个名为 "-" 的节点。
NULL_LIKE = {"", "-", "—", "--", "无", "None", "none", "nan", "NaN", "NULL", "null"}


def _setup_console() -> None:
    """让控制台编码不认识字符时不炸掉整个导入。

    Windows 控制台默认 GBK，数据里的生僻字（公司名、地名）可能打不出来；
    一次 print 抛 UnicodeEncodeError 会让"数据其实已经写进去了"的脚本看起来像失败。
    errors="replace" 只把打不出的字符显示成 ?，不影响库里存的内容。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):  # 被重定向到不支持 reconfigure 的对象时忽略
            pass


def _clean(value: str | None) -> str:
    """单元格归一：去首尾空白，并把 NULL_LIKE 占位符统一成空串。"""
    text = (value or "").strip()
    return "" if text in NULL_LIKE else text


def read_rows(path: Path, required: tuple[str, ...]) -> list[dict[str, str]]:
    """读 CSV 成 `list[dict]`，每格都过 `_clean`；缺列时抛 SystemExit。

    表头先剥 BOM：Excel 另存为 UTF-8 会给首列名加上 U+FEFF，「标的物」于是变成
    「\\ufeff标的物」，缺列报错会把锅甩给一个看不见的字符。这里剥掉后，
    校验失败时给出的就是"真的缺了哪一列"，直接可修。
    """
    if not path.exists():
        raise SystemExit(f"找不到输入文件：{path}")

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = [(name or "").lstrip("\ufeff").strip() for name in (reader.fieldnames or [])]
        missing = [column for column in required if column not in headers]
        if missing:
            raise SystemExit(
                f"{path.name} 缺少列：{'、'.join(missing)}（实际列：{'、'.join(headers)}）"
            )
        # 按剥过 BOM 的表头重新取键，否则 row["标的物"] 会取不到带 BOM 的那一列
        return [
            {column: _clean(raw.get(column)) for column in headers}
            for raw in reader
            if any((value or "").strip() for value in raw.values())  # 跳过尾部空行
        ]


def read_config(password_override: str | None) -> dict:
    """读 .env 与进程环境，返回连接参数。缺必填项时报中文错误退出。

    load_dotenv 默认**不覆盖**已存在的环境变量：临时在 shell 里
    `NEO4J_PASSWORD=xxx python ...` 就能覆盖，不必改文件。
    """
    load_dotenv(REPO_ROOT / ".env")

    def required(key: str) -> str:
        value = (os.getenv(key) or "").strip()
        if not value:
            raise SystemExit(f"缺少必需配置 {key}（检查 {REPO_ROOT / '.env'}，可参照 .env.example）")
        return value

    password = password_override
    if password is None:
        # 密码只去行尾换行（Windows 手写的 .env 常带 \r），不 strip——首位空格可能是密码本身
        password = (os.getenv("NEO4J_PASSWORD") or "").rstrip("\r\n")
        if not password:
            raise SystemExit(f"缺少必需配置 NEO4J_PASSWORD（检查 {REPO_ROOT / '.env'}）")

    return {
        "uri": required("NEO4J_URI"),
        "auth": (required("NEO4J_USERNAME"), password),
        # 库名留空表示用 driver 的默认库：dotenv 把 `NEO4J_DATABASE=` 读成空串而不是 None，
        # 空串会被服务端当成"名为空的数据库"报错，故统一收敛为 None
        "database": (os.getenv("NEO4J_DATABASE") or "").strip() or None,
    }


def _explain_connect_error(exc: Exception) -> str:
    """把驱动异常翻译成能直接照做的下一步。导入脚本面向开发者，故给得比客户端详细。"""
    if isinstance(exc, AuthError):
        return "账号或密码不正确：核对 .env 里的 NEO4J_USERNAME / NEO4J_PASSWORD，或用 --password 覆盖"
    if isinstance(exc, ServiceUnavailable):
        return (
            "连不上图谱服务：核对 .env 里的 NEO4J_URI（Aura 是 neo4j+s://<id>.databases.neo4j.io）；"
            "若实例已暂停，去 console.neo4j.io 点 Resume 唤醒后再跑"
        )
    if isinstance(exc, ConfigurationError):
        return "连接配置无效：检查 NEO4J_URI 的协议前缀与端口"
    return "未知错误，详见上面的日志"


def _run_batched(session, cypher: str, rows: list[dict], label: str) -> dict:
    """分批执行写语句，返回累计的计数器（新建/删除的节点与关系数）。

    每个批次是一个独立的事务：中途失败时前面的批次已经落库，日志能准确指出停在第几批，
    重跑一遍即可（MERGE 幂等）。整份数据一个事务则相反——失败全部回滚，
    但事务本身的体积和耗时也让"到底卡在哪"无从判断。
    """
    total = len(rows)
    batches = max(1, (total + BATCH_SIZE - 1) // BATCH_SIZE)
    counters = {"nodes_created": 0, "relationships_created": 0, "properties_set": 0}
    for index, start in enumerate(range(0, total, BATCH_SIZE), start=1):
        chunk = rows[start : start + BATCH_SIZE]
        # 用默认参数把 chunk 绑进闭包：直接在 lambda 里引用循环变量是经典的晚绑定陷阱，
        # 这里虽然每轮立刻调用不会踩到，但写成绑定形式省得日后被改坏
        summary = session.execute_write(
            lambda tx, batch=chunk: tx.run(cypher, rows=batch).consume()
        )
        for key in counters:
            counters[key] += getattr(summary.counters, key)
        logger.info(
            "  %s：第 %d/%d 批，累计 %d/%d 行（本批新建节点 %d、关系 %d）",
            label, index, batches, min(start + len(chunk), total), total,
            summary.counters.nodes_created, summary.counters.relationships_created,
        )
    return counters


def clear_graph(session) -> int:
    """清空整个图谱，返回删除的节点数。

    分批删而不是一条 `MATCH (n) DETACH DELETE n`：后者要把全库节点读进一个事务，
    节点一多就撞事务内存上限，且中断后只删掉一部分、状态难以判断。
    带 LIMIT 的循环每批独立提交，日志里有稳定进度，中途失败重跑即可。
    """
    # DETACH DELETE：删节点前先删掉它的关系，否则会报 "still has relationships"
    cypher = f"MATCH (n) WITH n LIMIT {CLEAR_BATCH} DETACH DELETE n"
    deleted = 0
    while True:
        summary = session.execute_write(lambda tx: tx.run(cypher).consume())
        removed = summary.counters.nodes_deleted
        deleted += removed
        if removed == 0:
            break
        logger.info("  已删除 %d 个节点", deleted)
    logger.info("图谱已清空，共删除 %d 个节点", deleted)
    return deleted


def import_frequency(session, rows: list[dict]) -> dict:
    """写入标的物节点与其采购频次。

    频次单独成表的原因：它是聚合结果，直接从交易记录现算要扫全表；
    而且标的物集合与交易记录是一对多，把频次挂在标的物节点上最省事。
    """
    prepared = []
    for row in rows:
        name = _clean(row.get("标的物"))
        if not name:
            continue
        try:
            frequency = int(float(_clean(row.get("交易频次")) or 0))
        except ValueError:
            logger.warning("频次不是数字，按 0 处理：%r -> %r", name, row.get("交易频次"))
            frequency = 0
        prepared.append({PROP_NAME: name, PROP_FREQUENCY: frequency})

    skipped = len(rows) - len(prepared)
    if skipped:
        logger.warning("跳过 %d 行没有标的物的频次记录", skipped)

    # MERGE 只按 name 匹配，再把频次 SET 上去：把频次写进 MERGE 的模式里当匹配条件，
    # 会在频次变化时匹配不到旧节点、凭空建出一个同名新节点
    cypher = f"""
    UNWIND $rows AS row
    MERGE (sm:{LABEL_SUBJECT} {{{PROP_NAME}: row.{PROP_NAME}}})
    SET sm.{PROP_FREQUENCY} = row.{PROP_FREQUENCY}
    """
    return _run_batched(session, cypher, prepared, "标的物频次")


def import_graph(session, rows: list[dict]) -> dict:
    """写入交易记录与 4 类实体关系。

    Transaction 的 id 取 CSV 行号（1 起）：同一份文件重跑得到同一批 id，MERGE 才能命中
    已有节点；换成 uuid 则每次跑都是全新的 8730 个节点，"幂等"当场失效。
    """
    transactions = []
    for line_number, row in enumerate(rows, start=1):
        subject = _clean(row.get("标的物"))
        if not subject:
            # 正常路径上 batch/准备图谱CSV.py 已经把这些行滤掉了，这里只是第二道防线
            logger.warning("跳过第 %d 行：标的物为空", line_number)
            continue
        amount_text = _clean(row.get("交易额"))
        try:
            amount = float(amount_text) if amount_text else None
        except ValueError:
            logger.warning("第 %d 行交易额不是数字，amount 留空：%r", line_number, amount_text)
            amount = None
        transactions.append(
            {PROP_ID: line_number, PROP_NAME: subject, PROP_AMOUNT: amount}
        )

    # 一笔交易一个 Transaction 节点 + 一条 HAS_TRANSACTION：
    # 标的物与交易是一对多，把金额直接挂成标的物的属性会只剩最后一个值。
    # amount 为 None 时 SET 会把属性置为 null（即"没有这个属性"），重复跑不会残留旧值
    cypher = f"""
    UNWIND $rows AS row
    MERGE (sm:{LABEL_SUBJECT} {{{PROP_NAME}: row.{PROP_NAME}}})
    MERGE (t:{LABEL_TRANSACTION} {{{PROP_ID}: row.{PROP_ID}}})
    SET t.{PROP_AMOUNT} = row.{PROP_AMOUNT}
    MERGE (sm)-[:HAS_TRANSACTION]->(t)
    """
    counters = _run_batched(session, cypher, transactions, "交易记录")

    # 4 类实体各跑一遍：一类一条语句，比串成一条巨型语句好读，进度也能分类打印。
    # 空值在这里滤掉——MERGE 一个 name 为 "" 的节点不会报错，只会造出垃圾节点（见模块 docstring）
    for column, label, relation in RELATION_COLUMNS:
        prepared = []
        for row in rows:
            subject = _clean(row.get("标的物"))
            name = _clean(row.get(column))
            if subject and name:
                prepared.append({PROP_NAME: subject, "entity": name})
        if not prepared:
            logger.info("  %s：无有效值，跳过", column)
            continue
        # 先 MERGE 标的物再连关系：理论上交易记录那批已经建好了全部标的物，
        # 但这里仍用 MERGE——若图谱 CSV 里出现了频次表没有的标的物，MATCH 会静默丢关系
        cypher = f"""
        UNWIND $rows AS row
        MERGE (sm:{LABEL_SUBJECT} {{{PROP_NAME}: row.{PROP_NAME}}})
        MERGE (x:{label} {{{PROP_NAME}: row.entity}})
        MERGE (sm)-[:{relation}]->(x)
        """
        batch_counters = _run_batched(session, cypher, prepared, f"{column}→{label}")
        for key in counters:
            counters[key] += batch_counters[key]

    return counters


def fetch_stats(session) -> tuple[list[dict], list[dict]]:
    """读出节点与关系统计（按标签 / 类型分组），供导入后核对。"""
    nodes = session.execute_read(
        lambda tx: tx.run(
            "MATCH (n) UNWIND labels(n) AS label RETURN label AS label, count(*) AS count ORDER BY label"
        ).data()
    )
    relationships = session.execute_read(
        lambda tx: tx.run(
            "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS count ORDER BY type"
        ).data()
    )
    return nodes, relationships


def main() -> int:
    parser = argparse.ArgumentParser(
        description="把两份 CSV 导入 Neo4j 知识图谱（MERGE，可重复执行）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--clear", action="store_true", help="导入前先清空图谱（MATCH (n) DETACH DELETE n）")
    # nargs="?" + const=""：`--password` 单独出现时提示输入，避免明文密码进 shell 历史与进程列表
    parser.add_argument("--password", nargs="?", const="", help="覆盖 .env 的密码；不带值则交互式输入（不回显）")
    args = parser.parse_args()

    _setup_console()

    password_override = args.password
    if password_override == "":  # 裸 --password
        password_override = getpass.getpass("Neo4j 密码（不回显）：")
        if not password_override:
            logger.error("密码为空，已取消")
            return 1
    config = read_config(password_override)

    # 读文件放在连库之前：文件不在是本地问题，先报它，别让人以为是连不上 Aura
    freq_rows = read_rows(FREQ_CSV, FREQ_COLUMNS)
    graph_rows = read_rows(GRAPH_CSV, GRAPH_REQUIRED_COLUMNS)
    logger.info("读入 %s：%d 行", FREQ_CSV.name, len(freq_rows))
    logger.info("读入 %s：%d 行", GRAPH_CSV.name, len(graph_rows))

    started = time.perf_counter()
    # 只打印"连到哪台、用哪个账号"，绝不打密码
    logger.info("目标图谱：%s（用户 %s，库 %s）", config["uri"], config["auth"][0], config["database"] or "默认")

    driver = GraphDatabase.driver(config["uri"], auth=config["auth"])
    try:
        try:
            # 先探活：Aura 免费实例闲置会暂停，这一步失败就不该继续往下写
            driver.verify_connectivity()
        except Exception as exc:
            logger.exception("Neo4j 连接失败：%s", type(exc).__name__)
            logger.error("连接失败：%s", _explain_connect_error(exc))
            return 1
        logger.info("连接正常")

        with driver.session(database=config["database"]) as session:
            if args.clear:
                logger.info("--clear：清空图谱")
                clear_graph(session)
            else:
                logger.info("未指定 --clear：增量导入（同名的节点与关系会被复用）")

            logger.info("导入标的物频次……")
            freq_counters = import_frequency(session, freq_rows)

            logger.info("导入交易记录与实体关系……")
            graph_counters = import_graph(session, graph_rows)

            nodes, relationships = fetch_stats(session)
    finally:
        driver.close()

    elapsed = time.perf_counter() - started
    logger.info("导入完成，耗时 %.1f 秒", elapsed)
    logger.info(
        "本次写入：节点 %d 个、关系 %d 条（频次表 %d / 知识图谱 %d）",
        freq_counters["nodes_created"] + graph_counters["nodes_created"],
        freq_counters["relationships_created"] + graph_counters["relationships_created"],
        len(freq_rows), len(graph_rows),
    )

    # 最终统计从库里现查，而不是用计数器累加：重跑时"新建 0 个"才说明真的幂等，
    # 而库里到底有多少东西只有查一遍才知道
    logger.info("图谱现有节点：")
    for row in nodes:
        logger.info("  %-16s %d", row["label"], row["count"])
    logger.info("图谱现有关系：")
    for row in relationships:
        logger.info("  %-16s %d", row["type"], row["count"])
    if not nodes:
        logger.warning("图谱里没有任何节点，检查 CSV 是否有数据")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""XLSX → PostgreSQL 导入脚本（成员 C）。

用途
----
把招投标 xlsx（招标采购提取结果 / 标的物训练数据）整表灌进 PostgreSQL 的
`bidding_records` 表，供 src/database/postgresql_client.py 的 6 类查询与
src/tools/rag_tools.py 的 `query_database` 工具使用。

用法
----
    # 1) 默认路径（下面的 DEFAULT_XLSX）；文件不存在则弹窗选择
    python src/database/to_postgresql.py

    # 2) 直接指定文件
    python src/database/to_postgresql.py "lj_rag_agent/data/processed/招标采购_提取结果_去重.xlsx"

    # 3) 覆盖 .env 里的密码（写完就跑，不进 shell 历史）
    python src/database/to_postgresql.py --password
    python src/database/to_postgresql.py --password mypass

    # 4) 多 sheet 的 xlsx（默认自动挑行数最多的那张）
    python src/database/to_postgresql.py --sheet Sheet2

依赖
----
pandas + openpyxl（读 xlsx）、psycopg 3（写库）、python-dotenv（读 .env）。
配置项与 postgresql_client.py 同一组：`POSTGRES_HOST` / `POSTGRES_PORT` /
`POSTGRES_DB` / `POSTGRES_USER` / `POSTGRES_PASSWORD`。

设计要点（改本文件前先读）
--------------------------
- **为什么 DROP 再 CREATE，而不是 CREATE IF NOT EXISTS + TRUNCATE**：xlsx 换一版，
  列集合就可能变（本仓库的训练数据就有 17 列与 21 列两种），只 TRUNCATE 会把旧表的
  列结构留下，新列全部丢失、旧列全为 NULL——表面"导入成功"，实际数据残缺。
  DROP + CREATE 让表结构永远等于"当前这个文件"，脚本也可以反复跑。
  ⚠️ 这是**破坏性**操作：它会删掉 bidding_records 里已有的数据（该表是纯导入产物，
  没有人工写入，故可安全重建；往表里存过手工数据的机器不要跑本脚本）。
- **为什么不复用 PostgresClient**：它的契约是"失败返回空、绝不抛异常"，那是给 Agent
  用的；导入脚本恰恰相反——出错就要立刻大声失败退出，不能吞掉。而且它只暴露固定查询，
  不开放任意 DDL。故本脚本自己连库、自己报错。
- **列名为什么必须过映射表 + 消毒**：SQL 的标识符（表名、列名）不能参数化，只能拼字符串。
  列名来自 xlsx 表头，属于**外部输入**，所以走"中文→英文白名单映射 + 只留 [a-z0-9_]"
  两道处理；映射表里没有的列不会丢，会退化成 col_<序号> 并打印告警，提醒补映射。
- **为什么整表一个事务**：中途失败则整体回滚，不会留下"导了一半"的表——半张表比空表
  更危险，检索会给出"数据只有一半"的错误结论却看不出异常。
- **金额为什么先转 Decimal**：xlsx 里的金额经 pandas 是 float64，二进制浮点存进
  NUMERIC 会带出 0.30000000000000004 这类尾巴；先 str 再 Decimal 才得到精确值。

"""

import argparse
import getpass
import logging
import os
import re
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg
from dotenv import load_dotenv
from psycopg import sql
from psycopg.rows import dict_row

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 仓库根（本文件在 src/database/ 下，上溯三层）：.env 与默认数据路径都以它为基准，
# 这样在任意工作目录下执行脚本行为都一致——CWD 相关的路径是"我这儿能跑"的经典来源
REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_XLSX = REPO_ROOT / "data" / "processed" / "招标采购标的物信息提取训练数据_2_标的物.xlsx"

TABLE_NAME = "bidding_records"
PK_COLUMN = "id"

# created_at：postgresql_client.trend() 按它分组的列。它不是 xlsx 里的列，而是**派生列**——
# 取记录自身的时间（发布时间 → 中标时间 → 导入时刻）。若让它等于导入时刻，8700 条会全挤在
# 同一个月，趋势查询永远只返回一根柱子，等于这个字段白建。
CREATED_AT_COLUMN = "created_at"
CREATED_AT_SOURCES = ("发布时间", "中标时间")

# xlsx 中文表头 → PostgreSQL 列名。**与 postgresql_client.py 的查询字段强绑定**：
# 那边写死的 subject_matter / purchaser / supplier / amount / location_city / project_id /
# project_name 必须在这里有对应项，否则查询会稳定返回空（表建出来了，但列名对不上）。
# 同义列（标的物 / 核心标的物）指向同一目标名，靠下面的重名处理兜底。
COLUMN_NAME_MAP = {
    "标题": "title",
    "类别": "category",
    "来源": "source",
    "发布时间": "publish_time",
    "省份": "province",
    "市区": "location_city",  # postgresql_client.location_dist / 所在地
    "县城": "county",
    "项目编号": "project_id",  # postgresql_client.by_* 返回列
    "项目名称": "project_name",
    "采购人": "purchaser",
    "代理机构": "agency",
    "预算": "budget",
    "项目地址": "project_address",
    "周期": "duration",
    "中标人": "supplier",  # 供应商 / 中标人
    "供应商": "supplier",
    "中标金额": "amount",  # 交易额 / 中标金额
    "交易额": "amount",
    "中标时间": "bid_time",
    "核心标的物": "subject_matter",  # 标的物
    "标的物": "subject_matter",
    "实体_主体": "entity_subject",
    "实体_地名": "entity_location",
    "实体_品类": "entity_category",
}

# 金额列强制 NUMERIC（精确十进制），不吃 pandas 推出来的 DOUBLE PRECISION：
# NUMERIC 是金额的默认选择，SUM/比较都不会有浮点误差。
COLUMN_TYPE_OVERRIDES = {
    "amount": "NUMERIC",
    "budget": "NUMERIC",
}

# pandas dtype → PostgreSQL 类型。未命中的一律 TEXT：xlsx 的列类型本来就靠猜，
# 猜错代价最小的是 TEXT（存得进、读得出），而 NUMBER 猜错会直接插入失败。
DTYPE_TO_PG = {
    "int64": "BIGINT",
    "int32": "INTEGER",
    "float64": "DOUBLE PRECISION",
    "float32": "REAL",
    "bool": "BOOLEAN",
    "datetime64[us]": "TIMESTAMPTZ",
    "datetime64[ns]": "TIMESTAMPTZ",
    "datetime64[s]": "TIMESTAMPTZ",
}


def _setup_console() -> None:
    """让控制台编码不认识字符时不炸掉整个导入。

    Windows 控制台默认 GBK，中文没问题，但数据里可能混进生僻字或 emoji；
    一次 print 抛 UnicodeEncodeError 会让"数据其实已经写进去了"的脚本看起来像失败。
    errors="replace" 只把打不出的字符显示成 ?，不影响库里存的内容。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):  # 被重定向到不支持 reconfigure 的对象时忽略
            pass


def sanitize_column(name: str, fallback_index: int) -> str:
    """把任意表头转成安全的 SQL 列名：只留小写字母/数字/下划线，且不以数字开头。

    白名单式消毒而不是加双引号引用：列名会进 DDL，也会被 postgresql_client 的 SQL
    以裸标识符引用，两边必须长得一样；双引号能保住怪字符，但会让列名与那侧的查询对不上。
    """
    cleaned = "".join(ch if ch.isascii() and (ch.isalnum() or ch == "_") else "_" for ch in name.strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_") or f"col_{fallback_index}"
    if cleaned[0].isdigit():
        cleaned = f"col_{cleaned}"
    return cleaned


def build_column_plan(df: pd.DataFrame) -> list[tuple[str, str, str]]:
    """产出 `[(原始列名, 目标列名, PG 类型)]`；这是整个导入的"列契约"。

    重名处理：两列映射到同一个目标名时（同义列并存）给后者加 `_2` 后缀，而不是覆盖——
    覆盖会静默丢一列数据，是"导入成功但少了一半字段"的源头。
    """
    plan: list[tuple[str, str, str]] = []
    used: set[str] = set()
    used.add(PK_COLUMN)
    for index, raw_name in enumerate(df.columns, start=1):
        original = str(raw_name).strip()
        target = COLUMN_NAME_MAP.get(original) or sanitize_column(original, index)
        if target not in COLUMN_NAME_MAP.values():
            # 没登记映射的列：不丢，但要让使用者看见，否则列名会悄悄变成 col_7
            logger.warning("列 %r 不在映射表中，列名按 %r 处理（建议补进 COLUMN_NAME_MAP）", original, target)
        base = target
        suffix = 2
        while target in used:
            target = f"{base}_{suffix}"
            suffix += 1
        if target != base:
            logger.warning("列 %r 与前面的列重名（%s），已改名为 %s", original, base, target)
        used.add(target)

        dtype = str(df[raw_name].dtype)
        pg_type = COLUMN_TYPE_OVERRIDES.get(target) or DTYPE_TO_PG.get(dtype, "TEXT")
        plan.append((original, target, pg_type))
    return plan


def clean_value(value: Any, pg_type: str) -> Any:
    """把 pandas 读出来的单元格值转成 psycopg 能直接发送的 Python 原生值。

    - NaN / NaT / None → None（SQL NULL）。不转换的话，数值列的缺失会变成字符串 "nan"
      或 float('nan') 写进 NUMERIC，从此每个聚合都被污染。
    - numpy 标量（np.int64 等）psycopg 没有适配器，必须 `.item()` 退回 Python 原生类型。
    - 空字符串 / 纯空白 → None：xlsx 里"没填"和"填了个空格"是同一件事，留成 '' 会让
      `WHERE purchaser = ''` 这种查询把一堆空记录当成真记录返回。
    """
    if value is None:
        return None
    try:
        if pd.isna(value):  # 标量判断；数组类值会返回数组，故下面用 try 兜住
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if hasattr(value, "item"):  # numpy 标量（int64 / float64 / bool_ / str_）
        value = value.item()
        if isinstance(value, str):
            return value.strip() or None
    if pg_type == "NUMERIC" and isinstance(value, float):
        # 先转字符串再进 Decimal：直接 Decimal(0.1) 会展开成 0.1000000000000000055511151231257827
        # 这种二进制长尾；str() 只给最短往返表示（0.1），把误差限制在 float64 自身的精度内
        return Decimal(str(value))
    return value


def read_xlsx(path: Path, sheet: str | None) -> pd.DataFrame:
    """读 xlsx 为 DataFrame；未指定 sheet 时自动选行数最多的那张。

    为什么要自动挑：本仓库的训练数据 xlsx 里 Sheet1 是空表、数据全在 Sheet2，
    "读第一张 sheet"会安静地导入 0 行——这种失败最难查，因为脚本退出码是 0。
    """
    if not path.exists():
        raise SystemExit(f"找不到文件：{path}")
    if path.suffix.lower() not in (".xlsx", ".xls", ".xlsm"):
        raise SystemExit(f"不是 Excel 文件：{path}")

    book = pd.ExcelFile(path)
    if sheet is not None:
        if sheet not in book.sheet_names:
            raise SystemExit(f"文件里没有 sheet {sheet!r}，可选：{book.sheet_names}")
        chosen = sheet
    else:
        # 用 openpyxl 的只读模式数行数，避免为了挑 sheet 把整本都读进内存
        import openpyxl

        wb = openpyxl.load_workbook(path, read_only=True)
        try:
            sizes = {ws.title: ws.max_row for ws in wb.worksheets}
        finally:
            # openpyxl 的 Workbook 不支持 with（它没有 __enter__），只读模式下必须显式 close，
            # 否则文件句柄一直开着，Windows 上表现为"文件正被另一个程序占用"
            wb.close()
        if not sizes:
            raise SystemExit(f"文件里没有任何 sheet：{path}")
        chosen = max(sizes, key=lambda name: sizes[name])
        logger.info("文件含 %d 张 sheet %s，选用行数最多的 %r", len(sizes), sizes, chosen)

    df = book.parse(chosen)
    df = df.dropna(how="all")  # 末尾常见的空行：全空的行没有导入价值
    if df.empty:
        raise SystemExit(f"sheet {chosen!r} 里没有数据行")
    return df


def read_config(password_override: str | None) -> dict:
    """读 .env 与进程环境，返回 psycopg.connect 的参数。缺必填项时报中文错误退出。

    load_dotenv 默认**不覆盖**已存在的环境变量：临时在 shell 里 `POSTGRES_DB=test python ...`
    就能覆盖，不必改文件；密钥仍然只在 .env 里，不进代码、不进日志。
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
        password = (os.getenv("POSTGRES_PASSWORD") or "").rstrip("\r\n")
        if not password:
            raise SystemExit(f"缺少必需配置 POSTGRES_PASSWORD（检查 {REPO_ROOT / '.env'}）")

    port_raw = (os.getenv("POSTGRES_PORT") or "").strip() or "5432"
    try:
        port = int(port_raw)
    except ValueError:
        raise SystemExit(f"POSTGRES_PORT 不是数字：{port_raw!r}") from None

    return {
        "host": required("POSTGRES_HOST"),
        "port": port,
        "dbname": required("POSTGRES_DB"),
        "user": required("POSTGRES_USER"),
        "password": password,
    }


def pick_xlsx(initial_dir: Path) -> Path | None:
    """弹窗选一个 xlsx；用户取消或环境不支持 GUI 时返回 None。

    tkinter 在无桌面/服务环境下会抛 TclError（它连不上显示服务），这不是"程序坏了"，
    故整体兜住并提示改用命令行参数——脚本要能在服务器上无人值守地跑。
    """
    # 非交互式调用（定时任务、CI、`python x.py < /dev/null`）绝不能弹窗：
    # 对话框会在无人应答的情况下一直等，表现就是"进程挂住"，比直接报错难查得多
    if not sys.stdin.isatty():
        logger.error("当前不是交互式终端，无法弹窗选择文件，请把 xlsx 路径作为参数传入")
        return None
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        logger.error("当前 Python 没有 tkinter，无法弹窗，请把文件路径作为参数传入")
        return None

    try:
        root = tk.Tk()
        root.withdraw()  # 只要文件对话框，不要那个空白主窗口
        root.attributes("-topmost", True)  # 否则对话框可能被终端压在后面，看起来像卡死
        selected = filedialog.askopenfilename(
            title="选择要导入 PostgreSQL 的 XLSX 文件",
            initialdir=str(initial_dir if initial_dir.exists() else REPO_ROOT),
            filetypes=[("Excel 文件", "*.xlsx *.xls *.xlsm"), ("所有文件", "*.*")],
        )
        root.destroy()
    except Exception:
        logger.exception("打开文件选择框失败")
        return None
    return Path(selected) if selected else None


def build_ddl(plan: list[tuple[str, str, str]]) -> sql.Composed:
    """拼出 DROP + CREATE 两条 DDL。

    用 psycopg.sql.Identifier 拼标识符：它做的就是加双引号并转义，等价于"标识符参数化"。
    列名虽然已经过映射表 / 消毒，但把转义交给驱动比自己拼引号更不容易漏。
    （表名是我们自己的常量，同样走 Identifier 保持一致。）
    """
    table = sql.Identifier(TABLE_NAME)
    columns = [sql.SQL("{} BIGSERIAL PRIMARY KEY").format(sql.Identifier(PK_COLUMN))]
    columns += [sql.SQL("{} {}").format(sql.Identifier(name), sql.SQL(pg_type)) for _, name, pg_type in plan]
    # created_at 放最后：它是派生列而非 xlsx 原列，摆在末尾能让"哪些列来自文件"一眼可辨
    columns.append(
        sql.SQL("{} TIMESTAMPTZ NOT NULL DEFAULT now()").format(sql.Identifier(CREATED_AT_COLUMN))
    )
    # DROP ... IF EXISTS 让脚本可重复执行；不加 CASCADE：真要误删别的表时宁可报错停下
    return sql.SQL("DROP TABLE IF EXISTS {}; CREATE TABLE {} ({});").format(
        table, table, sql.SQL(", ").join(columns)
    )


def build_rows(df: pd.DataFrame, plan: list[tuple[str, str, str]]) -> list[tuple]:
    """按列计划把 DataFrame 摊成待插入的行（每行一个 tuple）。

    逐行构造而不是 df.values：DataFrame 的 values 会把 int 列和 NaN 列揉成 float64，
    金额会变成 1684000.0 这种带浮点尾巴的形式，还要再拆回来；按原列取值则保持各列自己的类型。
    """
    # 候选时间列的下标先算好：逐行再遍历全部列去找，是把几十万次比较花在一件常量上
    time_indexes = [
        index
        for source in CREATED_AT_SOURCES
        for index, (original, _, _) in enumerate(plan)
        if original == source
    ]
    rows: list[tuple] = []
    for record in df.itertuples(index=False, name=None):
        row = [clean_value(value, pg_type) for value, (_, _, pg_type) in zip(record, plan)]
        # created_at 按候选顺序取第一个有值的（发布时间 → 中标时间）；都没有则留 NULL，
        # 由表默认值 now() 兜底——DEFAULT 只在插入值为 NULL 时生效，正是这里要的语义
        created_at = next((row[index] for index in time_indexes if row[index] is not None), None)
        rows.append((*row, created_at))
    return rows


def import_to_postgres(config: dict, plan: list[tuple[str, str, str]], rows: list[tuple]) -> int:
    """建表并批量写入，返回实际提交的行数；任一步失败都抛异常（不吞）。"""
    insert_columns = [name for _, name, _ in plan] + [CREATED_AT_COLUMN]
    # 占位符按列数生成，值一律走 %s：本表没有用户输入，但仍不给 SQL 拼接留任何口子
    statement = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
        sql.Identifier(TABLE_NAME),
        sql.SQL(", ").join(sql.Identifier(name) for name in insert_columns),
        sql.SQL(", ").join(sql.Placeholder() * len(insert_columns)),
    )

    with psycopg.connect(**config) as conn:  # 连接上下文：正常退出提交，异常回滚
        with conn.cursor() as cur:
            logger.info("执行 DDL（DROP + CREATE %s，含 %d 列）", TABLE_NAME, len(plan) + 2)
            cur.execute(build_ddl(plan))
            # executemany 在 psycopg3 里走 pipeline，一次往返发一批：8700 行是秒级，
            # 换成逐条 execute 则是 8700 次往返，慢到让人以为卡死
            cur.executemany(statement, rows)
            inserted = cur.rowcount
            cur.execute(
                sql.SQL("SELECT count(*) AS n FROM {}").format(sql.Identifier(TABLE_NAME))
            )
            total = cur.fetchone()[0]
    logger.info("已提交 %d 行；SELECT count(*) 复核 = %d", inserted, total)
    return total


def main() -> int:
    parser = argparse.ArgumentParser(
        description="把 XLSX 导入 PostgreSQL 的 bidding_records 表（会先 DROP 再重建）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("xlsx", nargs="?", help=f"xlsx 路径；省略则用 {DEFAULT_XLSX.name}，文件不存在时弹窗选择")
    # nargs="?" + const=""：`--password` 单独出现时提示输入，避免明文密码进 shell 历史与进程列表
    parser.add_argument("--password", nargs="?", const="", help="覆盖 .env 的密码；不带值则交互式输入（不回显）")
    parser.add_argument("--sheet", help="指定 sheet 名；默认自动选行数最多的那张")
    args = parser.parse_args()

    _setup_console()

    path = Path(args.xlsx).expanduser() if args.xlsx else DEFAULT_XLSX
    if not path.exists() and not args.xlsx:
        logger.warning("默认文件不存在：%s", path)
        picked = pick_xlsx(DEFAULT_XLSX.parent)
        if picked is None:
            logger.error("未选择文件，已取消。可显式传路径：python src/database/to_postgresql.py <xlsx>")
            return 1
        path = picked
    logger.info("数据源：%s", path)

    password_override = args.password
    if password_override == "":  # 裸 --password
        password_override = getpass.getpass("PostgreSQL 密码（不回显）：")
        if not password_override:
            logger.error("密码为空，已取消")
            return 1
    config = read_config(password_override)

    df = read_xlsx(path, args.sheet)
    plan = build_column_plan(df)
    logger.info("读到 %d 行 × %d 列，列映射如下：", len(df), len(plan))
    for original, target, pg_type in plan:
        logger.info("  %-8s -> %-18s %s", original, target, pg_type)

    rows = build_rows(df, plan)
    # 只打印"连到哪个库的哪台机"，绝不打密码
    logger.info(
        "目标库：postgresql://%s@%s:%s/%s（表 %s 将被 DROP 重建）",
        config["user"], config["host"], config["port"], config["dbname"], TABLE_NAME,
    )
    try:
        total = import_to_postgres(config, plan, rows)
    except psycopg.Error as exc:
        # 驱动的原始报文可能含主机名与 SQL 片段，只进日志；给使用者的提示指明下一步
        logger.exception("导入失败：%s", type(exc).__name__)
        logger.error("导入失败（已回滚），请检查 POSTGRES_* 配置、数据库是否启动；详见上面的日志")
        return 1
    logger.info("导入完成：%s 共 %d 行（%d 列 + id + created_at）", TABLE_NAME, total, len(plan))
    return 0


if __name__ == "__main__":
    sys.exit(main())

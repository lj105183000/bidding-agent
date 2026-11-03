"""评测运行与任务的持久化层（轻量评测中台）。

★ 为什么单独成包
-------------
评测报告原本只落 `eval/trend.jsonl`（供门禁逐行比较）。看板要看历史、
要选两次运行做对比，就需要可查询的存储；但门禁又必须保持“离线可跑、
可提交”的既有行为。故这里只新增 PostgreSQL 一侧，`trend.jsonl` 仍是
门禁的唯一事实源，两边通过 `run_all.py --save-db` 显式双写。

★ 为什么不复用 `src/datahub/pg.py` 之外的连接
----------------------------------------
项目要求数据访问收口。本包的所有 SQL 都经 `src/datahub/pg.py` 的
`PGCursor` / `PGTransaction`，超时与异常映射与其他数据访问一致。
"""

import sys
from pathlib import Path

# `eval/` 是无 `__init__.py` 的命名空间包。API 进程用 `uvicorn api.server:app`
# 启动时，`src` 可导入不代表项目根在 `sys.path` 上；先补一次，后续
# `records.py` / `compare.py` 才能稳定导入 `eval.report`。
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ★ 这里**不导入 `db` / `store`**：`eval/backfill_db.py --dry-run` 只读趋势文件，
#   不应因为包初始化就 import psycopg。需要数据库的模块各自显式导入。

"""项目 CLI 入口。

命令：
    python main.py api      # 生产模式启动后端
    python main.py dev      # 开发模式（热重载）
    python main.py ingest   # 构建向量知识库（Qdrant：Excel 问答 → 向量）
    python main.py graph    # 构建知识图谱（Neo4j：两份 CSV → 图数据库）
    python main.py pg       # 构建业务数据库（PostgreSQL：xlsx → bidding_records）
    python main.py all      # 依次构建全部三套数据（ingest → graph → pg）
    python main.py scheduler # 数据中台维护任务（--list / --run-once，独立进程）

设计说明：`graph` / `pg` 以**子进程**方式调用 src/database/ 下的两个导入脚本，
而不是 import 它们的函数。那两个脚本刻意不复用 Neo4jClient / PostgresClient
（客户端契约是"失败返回空、绝不抛异常"，导入脚本恰恰相反：出错必须立刻大声失败退出），
各自带 argparse 参数解析与 SystemExit 退出码。走子进程能原样保留它们全部行为与
命令行参数（如 `python main.py graph --clear`），导入失败也会以非零码自然冒泡到本 CLI，
不必在本文件里重复一遍错误处理。
"""

import subprocess
import sys
from pathlib import Path

import uvicorn

from src.config import settings
from src.logging_config import setup_logging

PROJECT_ROOT = Path(__file__).resolve().parent


def _run_builder(script: str, extra_args: list, title: str) -> int:
    """以子进程运行 src/database 下的导入脚本，透传附加参数并继承输出。

    用 `-u` 关闭 Python 输出缓冲：导入脚本日志量大、耗时以分钟计，
    不加的话输出会攒在缓冲区，用户看到的是"卡住不动"而不是实时进度。
    """
    target = PROJECT_ROOT / "src" / "database" / script
    if not target.exists():
        print(f"[main] 找不到构建脚本: {target}")
        return 1
    print(f"[main] === {title} ===")
    result = subprocess.run(
        [sys.executable, "-u", str(target), *extra_args],
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode == 0:
        print(f"[main] {title}：完成")
    else:
        print(f"[main] {title}：失败（退出码 {result.returncode}）")
    return result.returncode


def _ingest(extra_args: list) -> int:
    """构建 Qdrant 向量知识库。支持 `python main.py ingest [数据文件]`。"""
    try:
        from src.rag.ingest import ingest_data  # type: ignore

        # 默认导入问答知识库 data/qa.xlsx，避免 data/processed 下的
        # 招投标等非问答表抢走默认定位。
        ingest_data(force=True, path=extra_args[0] if extra_args else "data/qa.xlsx")
        return 0
    except ImportError:
        print(
            "[main] ingest 模块尚未就绪（src/rag/ingest.py 由 B 组实现），"
            "请等待 RAG 数据管道合入后再执行。"
        )
        return 1


def main() -> None:
    setup_logging()

    command = sys.argv[1] if len(sys.argv) > 1 else "dev"
    extra_args = sys.argv[2:]

    if command == "api":
        uvicorn.run(
            "api.server:app",
            host=settings.api_host,
            port=settings.api_port,
        )
    elif command == "dev":
        uvicorn.run(
            "api.server:app",
            host=settings.api_host,
            port=settings.api_port,
            reload=True,
        )
    elif command == "ingest":
        raise SystemExit(_ingest(extra_args))
    elif command == "graph":
        # 透传参数，例如：python main.py graph --clear
        raise SystemExit(_run_builder("to_neo4j.py", extra_args, "构建知识图谱（Neo4j）"))
    elif command == "pg":
        # 透传参数，例如：python main.py pg --sheet Sheet2
        raise SystemExit(
            _run_builder("to_postgresql.py", extra_args, "构建业务数据库（PostgreSQL）")
        )
    elif command == "all":
        # 一键构建三套数据。任一步失败即停：后续步骤大概率依赖同一批数据，
        # 继续跑只会刷出更多看不懂的错误。
        #
        # ★ 为什么 all 必须显式 --force：
        #   pg 导入会 DROP 重建 bidding_records，而运行时 /api/extract?save=true
        #   也写同一张表。一键脚本最容易在“只想更新知识库”时被顺手执行，
        #   所以把破坏性确认放在 all 入口，而不是等子进程里再问。
        if "--force" not in extra_args:
            print(
                "[main] 拒绝执行 all：pg 导入会 DROP 重建 bidding_records，"
                "可能删除运行时抽取写入的数据。确认后加 --force。"
            )
            raise SystemExit(1)
        # 过滤掉 --force，避免被 _ingest 当成数据文件路径
        ingest_args = [arg for arg in extra_args if not arg.startswith("--")]
        code = _ingest(ingest_args)
        if code == 0:
            code = _run_builder("to_neo4j.py", [], "构建知识图谱（Neo4j）")
        if code == 0:
            code = _run_builder("to_postgresql.py", ["--force"], "构建业务数据库（PostgreSQL）")
        raise SystemExit(code)
    elif command == "scheduler":
        # 独立进程：不导入 api.server、不监听 8001。
        # 只调度幂等任务；破坏性导入仍需人工执行 `pg --force` / `ingest`。
        from src.datahub.scheduler import run_scheduler

        raise SystemExit(run_scheduler(extra_args))
    else:
        print(f"未知命令: {command}")
        print(
            "可用命令: ingest（向量库）, graph（知识图谱）, pg（业务库，需 --force）, "
            "all（全部，需 --force）, scheduler（数据维护）, api, dev"
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()

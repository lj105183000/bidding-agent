"""项目 CLI 入口。

命令：
    python main.py api      # 生产模式启动后端
    python main.py dev      # 开发模式（热重载）
    python main.py ingest   # 全量重建知识库（B 组实现）
"""

import sys

import uvicorn

from src.config import settings
from src.logging_config import setup_logging


def main() -> None:
    setup_logging()

    command = sys.argv[1] if len(sys.argv) > 1 else "dev"

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
        # 知识库导入由 B 组（RAG）实现，此处懒加载其入口
        try:
            from src.rag.ingest import ingest_data  # type: ignore

            ingest_data(force=True)
        except ImportError:
            print(
                "[main] ingest 模块尚未就绪（src/rag/ingest.py 由 B 组实现），"
                "请等待 RAG 数据管道合入后再执行。"
            )
            raise SystemExit(1)
    else:
        print(f"未知命令: {command}")
        print("可用命令: ingest, api, dev")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

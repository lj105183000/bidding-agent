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
        print("ingest command is not implemented yet")
    else:
        print(f"未知命令: {command}")
        print("可用命令: ingest, api, dev")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
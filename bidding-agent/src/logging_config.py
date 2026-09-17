import logging
import sys


LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def setup_logging() -> None:
    """配置项目统一日志输出。"""
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        stream=sys.stdout,
        force=True,
    )

    logging.getLogger("src").setLevel(logging.INFO)

    for logger_name in (
        "httpx",
        "openai",
        "urllib3",
        "sentence_transformers",
        "jieba",
    ):
        logging.getLogger(logger_name).setLevel(logging.WARNING)
import os
import sys
from dataclasses import dataclass

from dotenv import load_dotenv


# .env 只填充未设置的变量，容器/CI 注入的真实环境变量优先
load_dotenv(override=False)


def _get_int(key: str, default: int) -> int:
    """安全读取整数环境变量，非法值回退默认。"""
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass
class Settings:
    """项目运行配置。缺失必需项时 __post_init__ 会打印清单并退出。"""

    # ---- LLM 选择 ----
    llm_provider: str = os.getenv("LLM_PROVIDER", "deepseek")

    # ---- DeepSeek ----
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    deepseek_base_url: str = os.getenv(
        "DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"
    )
    deepseek_thinking_model: str = os.getenv("DEEPSEEK_THINKING_MODEL", "")

    # ---- 智谱 ----
    zhipu_api_key: str = os.getenv("ZHIPU_API_KEY", "")
    zhipu_model: str = os.getenv("ZHIPU_MODEL", "glm-4.7-flashx")
    zhipu_vision_model: str = os.getenv("ZHIPU_VISION_MODEL", "glm-4.6v-flashx")
    zhipu_thinking_model: str = os.getenv("ZHIPU_THINKING_MODEL", "")

    # ---- vLLM（OpenAI 兼容） ----
    vllm_base_url: str = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    vllm_api_key: str = os.getenv("VLLM_API_KEY", "EMPTY")
    vllm_model: str = os.getenv("VLLM_MODEL", "deepseek-r1-0528-qwen3-8b")

    # ---- Ollama（OpenAI 兼容） ----
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    ollama_api_key: str = os.getenv("OLLAMA_API_KEY", "ollama")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "deepseek-r1:8b")

    # ---- Qdrant 向量库 ----
    qdrant_url: str = os.getenv("QDRANT_URL", "")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "bid_qa")
    qdrant_vector_size: int = _get_int("QDRANT_VECTOR_SIZE", 512)
    qdrant_timeout: int = _get_int("QDRANT_TIMEOUT", 30)

    # ---- Neo4j 知识图谱 ----
    neo4j_uri: str = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
    neo4j_username: str = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "")
    neo4j_database: str = os.getenv("NEO4J_DATABASE", "neo4j")

    # ---- PostgreSQL 结构化数据库 ----
    postgres_host: str = os.getenv("POSTGRES_HOST", "localhost")
    postgres_port: int = _get_int("POSTGRES_PORT", 5432)
    postgres_user: str = os.getenv("POSTGRES_USER", "postgres")
    postgres_password: str = os.getenv("POSTGRES_PASSWORD", "")
    postgres_db: str = os.getenv("POSTGRES_DB", "chatbot")

    # ---- 联网搜索 ----
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")
    exa_api_key: str = os.getenv("EXA_API_KEY", "")

    # ---- API 服务 ----
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = _get_int("API_PORT", 8001)
    # 逗号分隔白名单；* 允许所有；空 = 本地开发端口
    cors_origins: str = os.getenv("CORS_ORIGINS", "")

    # ---- 限流 ----
    rate_limit_max: int = _get_int("RATE_LIMIT_MAX", 30)
    rate_limit_window: int = _get_int("RATE_LIMIT_WINDOW", 60)

    def __post_init__(self) -> None:
        """校验必需配置，缺失则打印清单并退出进程。"""
        missing: list[str] = []

        # Qdrant 始终必需（RAG 底座）
        if not self.qdrant_url:
            missing.append("QDRANT_URL")
        if not self.qdrant_api_key:
            missing.append("QDRANT_API_KEY")

        # DeepSeek key 仅在选择 deepseek 提供商时必需
        if self.llm_provider == "deepseek" and not self.deepseek_api_key:
            missing.append("DEEPSEEK_API_KEY")

        if missing:
            print(
                "[config] 缺少必需配置项："
                + ", ".join(missing)
                + "。请对照 docs/开发文档.md §2.3 在 .env 中补全。",
                file=sys.stderr,
            )
            raise SystemExit(1)

    @property
    def cors_origin_list(self) -> list[str]:
        """解析 CORS_ORIGINS 为列表。"""
        raw = self.cors_origins.strip()
        if not raw:
            return [
                "http://localhost:3000",
                "http://127.0.0.1:3000",
            ]
        if raw == "*":
            return ["*"]
        return [o.strip() for o in raw.split(",") if o.strip()]


settings = Settings()

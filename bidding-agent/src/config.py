import os
from pathlib import Path

from dotenv import load_dotenv


# 项目根目录（bidding-agent/），src/rag 用于定位 data 目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# .env 只填充未设置的变量，容器/CI 注入的真实环境变量优先
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _get_int(key: str, default: int) -> int:
    """安全读取整数环境变量，非法值回退默认。"""
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# 全局配置定义
class Settings:
    # ---- Neo4j Aura 配置 ----
    neo4j_uri: str = os.getenv("NEO4J_URI", "")
    neo4j_username: str = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "")
    neo4j_database: str = os.getenv("NEO4J_DATABASE", "neo4j")

    # ---- Qdrant 向量数据库配置 (对应 pipeline.py 的 store 懒加载) ----
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "bidding_qa")
    qdrant_vector_size: int = int(os.getenv("QDRANT_VECTOR_SIZE", "1024"))
    qdrant_distance: str = os.getenv("QDRANT_DISTANCE", "Cosine")
    qdrant_timeout: float = float(os.getenv("QDRANT_TIMEOUT", "10.0"))

    # ---- 检索与重排配置 ----
    # 对应 pipeline.py 里的 settings.rerank_enabled
    rerank_enabled: bool = os.getenv("RERANK_ENABLED", "false").lower() in ("true", "1", "yes")

    # ---- 大模型(LLM)配置 (对应 llm_factory.py) ----
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai")  # 默认使用 openai
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_base_url: str | None = os.getenv("LLM_BASE_URL", None)  # 如果是默认OpenAI可留空
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))

    # ---- DeepSeek（feat/api-llm 引入，逐请求 provider 选项） ----
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
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

    @property
    def cors_origin_list(self) -> list[str]:
        """解析 CORS_ORIGINS 为列表；空 = 本地开发端口默认白名单。"""
        raw = self.cors_origins.strip()
        if not raw:
            return [
                "http://localhost:3000",
                "http://127.0.0.1:3000",
                "http://localhost:5173",
                "http://127.0.0.1:5173",
            ]
        if raw == "*":
            return ["*"]
        return [o.strip() for o in raw.split(",") if o.strip()]


# 实例化全局配置对象
settings = Settings()

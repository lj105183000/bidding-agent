import os
from pathlib import Path
from dotenv import load_dotenv

# 1. 加载项目根目录下的 .env 文件
# 注意：为了确保能读取到 .env，我们要先定位项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

# 2. 全局配置定义
class Settings:
    # ---- Neo4j Aura 配置 ----
    neo4j_uri: str = os.getenv("NEO4J_URI", "")
    neo4j_username: str = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "")

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
    llm_base_url: str | None = os.getenv("LLM_BASE_URL", None) # 如果是默认OpenAI可留空
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))

# 3. 实例化全局配置对象
settings = Settings()
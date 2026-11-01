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


def _get_bool(key: str, default: bool = False) -> bool:
    """安全读取布尔环境变量：1/true/yes/on 为真，其余为假，空值回退默认。"""
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


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
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
    deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    deepseek_thinking_model: str = os.getenv("DEEPSEEK_THINKING_MODEL", "")

    # ---- 智谱 ----
    zhipu_api_key: str = os.getenv("ZHIPU_API_KEY", "")
    zhipu_model: str = os.getenv("ZHIPU_MODEL", "glm-4.7-flashx")
    # 视觉模型：前端「附件」上传图片后由它解析。
    # glm-4.6v-flashx 经实测可用：项目编号 / 预算金额 / 时间等关键字段识别正确，
    # 与 glm-5.3-flashx 在同一测试图上能力相当。
    # 实测其他可用值：glm-5.3-flashx / glm-5.3-flash / glm-4v-flash。
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

    # ---- 数据中台 ----
    # 资产目录 / 血缘端点的进程内缓存秒数。0 = 关闭缓存。
    # 目录是静态注册表，正常不会变；留 TTL 是为了将来支持“注册表热更新后生效”，
    # 而不是每次请求都重新构造几十个资产与字段的响应。
    data_catalog_cache_seconds: int = _get_int("DATA_CATALOG_CACHE_SECONDS", 60)
    # 质量报告保留天数与 latest 端点缓存秒数。
    # 报告由 scheduler 写入、API 读取；跨进程共享，故 30 秒缓存足够。
    data_quality_retention_days: int = _get_int("DATA_QUALITY_RETENTION_DAYS", 30)
    data_quality_cache_seconds: int = _get_int("DATA_QUALITY_CACHE_SECONDS", 30)

    # ---- 数据中台：轻量调度（独立进程，见 src/datahub/scheduler.py） ----
    # 常驻调度器默认关闭：它只做数据维护，不该因为起了 API 就被动开跑。
    data_scheduler_enabled: bool = _get_bool("DATA_SCHEDULER_ENABLED", False)
    data_scheduler_timezone: str = os.getenv("DATA_SCHEDULER_TIMEZONE", "Asia/Shanghai")
    data_scheduler_regulations_cron: str = os.getenv(
        "DATA_SCHEDULER_REGULATIONS_CRON", "30 3 * * *"
    )
    data_scheduler_quality_cron: str = os.getenv(
        "DATA_SCHEDULER_QUALITY_CRON", "15 */6 * * *"
    )
    data_scheduler_catalog_cron: str = os.getenv(
        "DATA_SCHEDULER_CATALOG_CRON", "0 4 * * *"
    )
    # 同一任务最大并发；默认 1，避免质量扫描/法规入库重叠。
    data_scheduler_max_instances: int = _get_int("DATA_SCHEDULER_MAX_INSTANCES", 1)
    # 错过多次只补跑一次；misfire 超过 300s 的任务直接跳过。
    data_scheduler_coalesce: bool = _get_bool("DATA_SCHEDULER_COALESCE", True)
    data_scheduler_misfire_grace_seconds: int = _get_int(
        "DATA_SCHEDULER_MISFIRE_GRACE_SECONDS", 300
    )

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

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv(override=False)


@dataclass
class Settings:
    """项目运行配置。"""

    
    llm_provider: str = os.getenv("LLM_PROVIDER", "deepseek")

    # DeepSeek
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    deepseek_model: str = os.getenv(
        "DEEPSEEK_MODEL",
        "deepseek-v4-flash",
    )
    deepseek_base_url: str = os.getenv(
        "DEEPSEEK_BASE_URL",
        "https://api.deepseek.com/v1",
    )

    # 智谱
    zhipu_api_key: str = os.getenv("ZHIPU_API_KEY", "")
    zhipu_model: str = os.getenv(
        "ZHIPU_MODEL",
        "glm-4.7-flashx",
    )
    zhipu_vision_model: str = os.getenv(
        "ZHIPU_VISION_MODEL",
        "glm-4.6v-flashx",
    )

    # vLLM
    vllm_base_url: str = os.getenv(
        "VLLM_BASE_URL",
        "http://localhost:8000/v1",
    )
    vllm_model: str = os.getenv(
        "VLLM_MODEL",
        "deepseek-r1-0528-qwen3-8b",
    )

    # Ollama
    ollama_base_url: str = os.getenv(
        "OLLAMA_BASE_URL",
        "http://localhost:11434/v1",
    )
    ollama_model: str = os.getenv(
        "OLLAMA_MODEL",
        "deepseek-r1:8b",
    )

    # Qdrant
    qdrant_url: str = os.getenv("QDRANT_URL", "")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")

    # API
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8001"))

    # OpenRouter
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    openrouter_model: str = os.getenv(
        "OPENROUTER_MODEL",
        "openrouter/free",
    )


settings = Settings()
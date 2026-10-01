"""pytest 根配置：预置必需 env，无 .env 也可运行（CI 同样受益）。

配置语义见 docs/开发文档.md §2.4——这里只填充占位值，
真实密钥永远只在本地 .env 中，不入库。
"""

import os

os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("QDRANT_API_KEY", "test-key")
os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")
os.environ.setdefault("LLM_PROVIDER", "deepseek")
# 测试离线运行：嵌入/精排模型已本地缓存（HF_HOME），不依赖外网。
# 未设此项时 huggingface_hub 会尝试联网刷新 token，无外网环境会长时间阻塞导致测试挂起。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
# PostgresClient 构造只校验 env 是否填满（不连库），占位即可让依赖其构造的测试离线通过；
# 真正连库的集成路径由测试自身 monkeypatch 或跳过。
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_DB", "test")
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
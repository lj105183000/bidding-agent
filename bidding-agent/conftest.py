"""pytest 根配置：预置必需 env，无 .env 也可运行（CI 同样受益）。

配置语义见 docs/开发文档.md §2.4——这里只填充占位值，
真实密钥永远只在本地 .env 中，不入库。
"""

import os

os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("QDRANT_API_KEY", "test-key")
os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")
os.environ.setdefault("LLM_PROVIDER", "deepseek")
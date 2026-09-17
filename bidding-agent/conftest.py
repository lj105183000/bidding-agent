"""pytest 根配置：预置必需环境变量，使测试无需 .env 即可运行。

config.Settings.__post_init__ 会校验 QDRANT_URL / QDRANT_API_KEY，
且 LLM_PROVIDER=deepseek 时校验 DEEPSEEK_API_KEY。
此处在导入任何业务模块前注入占位值。
"""

import os


# 在 config 被导入前设置必需项（占位值，测试中会 mock 真实客户端）
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("QDRANT_API_KEY", "test-qdrant-key")
os.environ.setdefault("DEEPSEEK_API_KEY", "test-deepseek-key")
os.environ.setdefault("LLM_PROVIDER", "deepseek")

"""pytest 根配置：**先加载真实 .env**，仅在确实缺失时才填占位值。

为什么必须先加载 .env（实测踩坑）
--------------------------------
原先这里只有 `os.environ.setdefault("POSTGRES_USER", "test")` 这类占位。

问题出在 `setdefault` 与 `load_dotenv` 的**交互**：

    1. 本文件在 collect 阶段先执行 → 环境里被填上 POSTGRES_USER="test"
    2. 之后 `postgresql_client` 模块内的 `load_dotenv()` 才执行
       —— 而 **python-dotenv 默认不覆盖已存在的环境变量**
    3. 于是真实 .env 里的 postgres 配置被占位值挡住，
       集成测试全部连不上库（报"用户 test 密码验证失败"），
       表现为 20 个用例静默 skip。

更隐蔽的是：**单独跑那个测试文件时它可能是通过的**——
因为占位值恰好"已存在"与否取决于环境继承顺序。这种不确定性比直接失败更糟。

修法：先 `load_dotenv(override=False)` 把真实 .env 装进来，**再** setdefault 兜底。
这样 CI（无 .env）仍有占位值可跑，本地（有 .env）则用真实配置。
"""

import os
from pathlib import Path

# ① 先加载真实 .env（不覆盖外部显式设置的环境变量，尊重调用方意图）
try:
    from dotenv import load_dotenv

    _env = Path(__file__).resolve().parent / ".env"
    if _env.exists():
        load_dotenv(_env, override=False)
except Exception:  # pragma: no cover - dotenv 不可用时退回纯占位
    pass

# ② 缺失的才填占位值（CI / 无 .env 环境靠这批值离线通过）
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("QDRANT_API_KEY", "test-key")
os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")
os.environ.setdefault("LLM_PROVIDER", "deepseek")
# 测试离线运行：嵌入/精排模型已本地缓存（HF_HOME），不依赖外网。
# 未设此项时 huggingface_hub 会尝试联网刷新 token，无外网环境会长时间阻塞导致测试挂起。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
# PostgresClient 构造只校验 env 是否填满（不连库），占位即可让依赖其构造的测试离线通过。
# 注意：**真正连库的集成测试必须拿到真实配置**，故上面第 ① 步是关键，
# 不能删——只留这批占位值会让集成测试全部 skip 且原因难查。
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_DB", "test")
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")

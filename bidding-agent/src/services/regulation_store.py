"""兼容别名：本模块已迁移到 `src.rag.regulation_store`。

★ 为什么保留这个文件
================
`src/rag/__init__.py` 明确写着"公共导入路径为对外契约（测试与 eval 依赖），
勿破坏"。迁移是**整理目录结构**，不该顺带破坏外部引用。

★ 新代码请直接 import 新路径
=======================
    from src.rag.regulation_store import search_regulations   # ← 推荐
    from src.services.regulation_store import search_regulations  # ← 仍可用（本文件）

★ 为什么不能只写 `from src.rag.regulation_store import *`
==================================================
被迁移的模块**没有 `__all__`**，而 `import *` 只复制**公开名**
（不以下划线开头的）。但测试用到了两个私有名：

    _sparse_index    稀疏向量编码（测试直接验算权重）
    _tf_weights      词频权重

它们在 `import *` 下会丢失，报错点在**测试里**而不是这里——
排查时容易误判成"迁移把功能改坏了"。故显式再导出。
"""

from src.rag.regulation_store import *  # noqa: F401,F403 - 公开 API 全量转发
from src.rag.regulation_store import (  # 私有名：测试依赖，import * 覆盖不到
    _sparse_index,
    _tf_weights,
)

__all__ = [name for name in dir() if not name.startswith("__")]

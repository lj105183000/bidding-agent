"""轻量数据中台包。

只在这里导入 `DataHub`，**不要**在模块级导入 `quality` / `pg` / `neo4j` / `qdrant`：
- `quality.py` 会用 `analysis` 的检查逻辑，而 `analysis` 在 P1 会反过来用 `datahub.pg`；
  在 `__init__` 里提前导入会形成循环。
- 适配器需要读配置、建客户端，导入即副作用；目录端点不应该因为“看一眼目录”
  就去连数据库。

因此：目录/血缘用 `from src.datahub import datahub`；
适配器与质量模块由调用方在方法内延迟导入。
"""

from src.datahub.service import DataHub

#: 进程级单例。与项目其它单例一致：目录缓存、注册表都是进程内共享状态；
#: 结构不可变，缓存自带锁，故无需额外双检锁。
datahub = DataHub()

__all__ = ["DataHub", "datahub"]

"""数据中台的统一返回契约与异常层次。

为什么单独成模块
----------------
API 层、服务层、测试都要判断“成功 / 失败”的形状。把 `Envelope` 与
`DataHubError` 放在一起，避免目录门面和底层适配器各自定义一套，
最后出现“同一个端点两种错误结构”。

约定（与项目既有工具契约保持一致）：
    - 成功：`success=True`，列表用 `results`，对象用 `data`，二者互斥；
    - 失败：`success=False` + `error`，`error` 是**用户可见中文**，
      不含服务地址、SQL、异常原文（完整异常只进日志）；
    - `elapsed_ms` 始终非负，便于前端显示“目录查询耗时”。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


class DataHubError(RuntimeError):
    """数据中台异常的基类。

    低层适配器（pg / neo4j / qdrant）可以抛它；高层门面
    （catalog / asset / lineage / quality_latest）必须捕获并转成 Envelope，
    不能让“看目录”这种只读操作把 API 打挂。
    """


class DataHubConfigError(DataHubError):
    """配置缺失或非法（例如 POSTGRES_HOST 为空）。"""


class DataHubUnavailableError(DataHubError):
    """后端暂时不可用（连接失败、超时、集合不存在）。"""


class DataHubQueryError(DataHubError):
    """查询本身失败（语句错误、字段不存在）。"""


@dataclass(frozen=True)
class Envelope:
    """数据中台统一响应体。

    `count` 单独放在顶层而不是塞进 `data`：
    目录列表的 `count` 与 `data.count` 在语义上不是一回事，
    混在一起会让前端分不清“返回了几条”和“业务字段叫什么”。
    """

    success: bool
    asset: str
    data: Any = None
    results: list[Any] | None = None
    error: str | None = None
    elapsed_ms: int = 0
    cached: bool = False
    cache_ttl_seconds: int = 0
    generated_at: str | None = None
    count: int = 0

    def __post_init__(self) -> None:
        if self.data is not None and self.results is not None:
            raise ValueError("Envelope 的 data 与 results 不能同时出现")
        if self.success and self.error is not None:
            raise ValueError("成功响应不能携带 error")
        if not self.success and not self.error:
            raise ValueError("失败响应必须携带 error")

    def to_dict(self) -> dict[str, Any]:
        """转成可直接 JSON 序列化的 dict。

        保留 `None` 字段而不是过滤掉：前端按固定形状读取，
        “字段不存在”和“字段为 null”是两种不同的契约，前者会让可选链失效。
        """
        return asdict(self)

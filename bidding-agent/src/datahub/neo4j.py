"""Neo4j 统一适配器。

与 `src/database/neo4j_client.py` 的分工
--------------------------------------
- `Neo4jClient` 是**给 Agent 工具用的**：查询模板固定、失败返回空/降级，
  标识符白名单在客户端里维护。
- 本适配器是**给服务层与质量扫描用的**低层入口：只负责“共享 driver + 只读
  session + 超时 + 异常映射”，不提供业务查询模板。

共享 driver 的理由：`competitor` 原来每次查询都 `GraphDatabase.driver(...)`
再 close，连接池反复建/拆；质量扫描也会查图谱，两处各建一份更浪费。这里用
双检锁懒加载一个进程级 driver，`close()` 可重复调用（测试用）。

Cypher 的标签/关系/属性名不能参数化，故本适配器**不接受**外部传入标识符；
调用方必须把标识符写死在语句字面量里，值走 `$name`。
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

from dotenv import load_dotenv

from src.config import PROJECT_ROOT
from src.datahub.contract import (
    DataHubConfigError,
    DataHubError,
    DataHubQueryError,
)

try:  # neo4j 是主依赖；这里兜住“缺包”以便目录/血缘端点仍可用
    from neo4j import GraphDatabase
except Exception:  # pragma: no cover - 依赖缺失属部署异常
    GraphDatabase = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

CONNECT_TIMEOUT_SECONDS = 10
ACQUISITION_TIMEOUT_SECONDS = 15

_driver: Any = None
_driver_lock = threading.Lock()


def _config() -> tuple[str, tuple[str, str], str | None]:
    """返回 `(uri, auth, database)`；缺配置抛 DataHubConfigError。"""
    load_dotenv(PROJECT_ROOT / ".env")
    uri = (os.getenv("NEO4J_URI") or "").strip()
    user = (os.getenv("NEO4J_USERNAME") or "").strip()
    # 同 postgres：密码只去行尾换行，首位空格可能是密码本身
    password = (os.getenv("NEO4J_PASSWORD") or "").rstrip("\r\n")
    missing = [
        name
        for name, value in (
            ("NEO4J_URI", uri),
            ("NEO4J_USERNAME", user),
            ("NEO4J_PASSWORD", password),
        )
        if not value
    ]
    if missing:
        raise DataHubConfigError(f"未配置知识图谱（缺少 {'、'.join(missing)}）")
    # dotenv 把 `NEO4J_DATABASE=` 读成空串而不是 None；空串会被服务端当成
    # “名为空的数据库”直接报错，故统一收敛为 None（= 用 driver 默认库）
    database = (os.getenv("NEO4J_DATABASE") or "").strip() or None
    return uri, (user, password), database


def get_driver() -> Any:
    """懒加载共享 driver（双检锁）。配置缺失/驱动初始化失败抛 DataHubConfigError。"""
    global _driver
    if _driver is not None:
        return _driver
    with _driver_lock:
        if _driver is None:
            uri, auth, _database = _config()
            if GraphDatabase is None:
                raise DataHubConfigError("neo4j 驱动未安装")
            try:
                _driver = GraphDatabase.driver(
                    uri,
                    auth=auth,
                    connection_timeout=CONNECT_TIMEOUT_SECONDS,
                    connection_acquisition_timeout=ACQUISITION_TIMEOUT_SECONDS,
                )
            except Exception as exc:
                raise DataHubConfigError(
                    f"知识图谱驱动初始化失败：{type(exc).__name__}"
                ) from exc
    return _driver


def run(
    cypher: str,
    *,
    raise_on_error: bool = False,
    error_cls: type[Exception] = DataHubQueryError,
    **params: Any,
) -> list[dict[str, Any]]:
    """执行只读 Cypher，返回 `list[dict]`。

    - `raise_on_error=False`（默认）：失败记日志并返回 `[]`，供“降级继续”的
      调用方使用；
    - `raise_on_error=True`：把失败映射成 `error_cls`，供质量扫描区分
      “查到 0 行”与“图谱不可用”。
    """
    try:
        _uri, _auth, database = _config()
        driver = get_driver()
        with driver.session(database=database) as session:
            return [dict(record) for record in session.run(cypher, **params)]
    except DataHubError as exc:
        if raise_on_error:
            raise error_cls(str(exc)) from exc
        logger.warning("知识图谱查询失败：%s", exc)
        return []
    except Exception as exc:
        # 驱动异常里带主机名与端口，只进日志；对外只给异常类型
        logger.warning("知识图谱查询失败：%s", exc)
        if raise_on_error:
            raise error_cls(f"知识图谱查询失败：{type(exc).__name__}") from exc
        return []


def close() -> None:
    """关闭共享 driver（可重复调用）。测试与进程退出时用。"""
    global _driver
    with _driver_lock:
        driver, _driver = _driver, None
    if driver is not None:
        try:
            driver.close()
        except Exception:  # pragma: no cover
            pass

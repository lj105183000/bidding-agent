"""带锁的进程内 TTL 缓存。

为什么不用 `functools.lru_cache`
------------------------------
1. 目录数据会变（注册表改了、部署换了），需要**按时间过期**；lru_cache 没有时间维度。
2. 响应要带 `cached` / `elapsed_ms`，调用方需要知道“这次是命中还是现算”；
   lru_cache 不暴露这个信息。

实现参照 `src/health.py:551-570` 的探测缓存，但独立成类：
数据中台的目录、血缘、质量报告缓存互不影响，不能共用一个全局字典。
"""

from __future__ import annotations

import copy
import threading
import time
from typing import Any, Hashable


class TTLCache:
    """线程安全的简单 TTL 缓存。

    `ttl_seconds <= 0` 表示关闭缓存：`get()` 永远返回 None，`put()` 不写入。
    这与配置项“0 = 关闭”的语义一致，测试也靠它绕开缓存。
    """

    def __init__(self, ttl_seconds: int, max_entries: int = 256) -> None:
        self.ttl_seconds = max(0, int(ttl_seconds))
        self.max_entries = max(1, int(max_entries))
        self._lock = threading.Lock()
        self._store: dict[Hashable, tuple[float, Any]] = {}

    def get(self, key: Hashable) -> tuple[Any, int] | None:
        """返回 `(value, age_seconds)`；未命中 / 已过期返回 None。

        返回深拷贝：调用方（API 序列化、测试）若修改了结果，
        不能把缓存里的共享对象一起改掉——那会让下一个请求读到脏数据。
        """
        if self.ttl_seconds <= 0:
            return None
        with self._lock:
            hit = self._store.get(key)
        if hit is None:
            return None
        timestamp, value = hit
        age = time.monotonic() - timestamp
        if age > self.ttl_seconds:
            with self._lock:
                self._store.pop(key, None)
            return None
        return copy.deepcopy(value), int(age)

    def put(self, key: Hashable, value: Any) -> None:
        if self.ttl_seconds <= 0:
            return
        with self._lock:
            # 条目数很少（目录 × 过滤组合 + 血缘 × 深度），但仍设上限：
            # 防止未来有人把 asset 名直接当 key 拼出无限增长。
            if len(self._store) >= self.max_entries:
                self._store.clear()
            self._store[key] = (time.monotonic(), copy.deepcopy(value))

    def clear(self) -> None:
        """清空缓存（测试与“注册表热更新后立即生效”时用）。"""
        with self._lock:
            self._store.clear()

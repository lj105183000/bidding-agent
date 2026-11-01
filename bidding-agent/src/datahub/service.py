"""数据中台门面：资产目录、血缘与后续质量报告的只读入口。

设计边界
--------
- 这一层**不连数据库**。目录和血缘是静态注册表，三库全挂也能返回；
  真正的数据查询由 P1 的 `pg.py` / `neo4j.py` / `qdrant.py` 适配器负责。
- 高层方法（catalog / asset / lineage）**永不抛异常**：参数非法抛 `ValueError`
  让 API 映射 400，其余异常一律转成 `success=False` 的 Envelope。
  这与项目“工具永不抛异常”的硬性约定一致。
- `quality_latest` 在 P1 接入；本阶段不导入 `quality.py`，
  避免 `analysis -> datahub -> quality -> analysis` 的循环导入。
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

from src.config import settings
from src.datahub.cache import TTLCache
from src.datahub.contract import Envelope
from src.datahub.lineage import lineage_subgraph
from src.datahub.registry import (
    ASSET_KINDS,
    ASSET_LAYERS,
    asset_to_dict,
    find_asset,
    list_assets,
)

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class DataHub:
    """资产目录与血缘的只读门面。"""

    def __init__(self, cache_ttl_seconds: int | None = None) -> None:
        ttl = (
            settings.data_catalog_cache_seconds
            if cache_ttl_seconds is None
            else cache_ttl_seconds
        )
        self._cache_ttl = max(0, int(ttl))
        self._catalog_cache = TTLCache(self._cache_ttl)
        self._lineage_cache = TTLCache(self._cache_ttl)
        self._quality_cache = TTLCache(settings.data_quality_cache_seconds)

    # ------------------------------------------------------------------
    # 对外方法
    # ------------------------------------------------------------------

    def catalog(self, kind: str | None = None, layer: str | None = None) -> dict[str, Any]:
        """列出资产。`kind` / `layer` 可选过滤；非法取值抛 ValueError。"""
        kind = (kind or "").strip() or None
        layer = (layer or "").strip() or None
        if kind is not None and kind not in ASSET_KINDS:
            raise ValueError(f"未知资产类型：{kind}")
        if layer is not None and layer not in ASSET_LAYERS:
            raise ValueError(f"未知资产分层：{layer}")

        started = time.perf_counter()
        cache_key = ("catalog", kind, layer)
        hit = self._catalog_cache.get(cache_key)
        if hit is not None:
            payload, _age = hit
            return self._success(
                "catalog",
                results=payload["results"],
                count=payload["count"],
                started=started,
                cached=True,
            )

        try:
            assets = list_assets(kind=kind, layer=layer)
            results = [asset_to_dict(asset, include_fields=False) for asset in assets]
            payload = {"results": results, "count": len(results)}
            self._catalog_cache.put(cache_key, payload)
            return self._success(
                "catalog",
                results=results,
                count=len(results),
                started=started,
            )
        except Exception:
            return self._failure("catalog", started)

    def asset(self, name: str) -> dict[str, Any] | None:
        """单个资产详情；未知资产返回 None（由 API 映射 404）。"""
        target = find_asset(name)
        if target is None:
            return None

        started = time.perf_counter()
        cache_key = ("asset", target.name)
        hit = self._catalog_cache.get(cache_key)
        if hit is not None:
            payload, _age = hit
            return self._success(
                target.name,
                data=payload,
                started=started,
                cached=True,
            )

        try:
            payload = asset_to_dict(target, include_fields=True)
            # P1 的质量规则注册表接上后在这里填充；P0 先给空列表，
            # 让前端按固定形状读取，不必等 P1 再改契约。
            payload["quality_rules"] = []
            self._catalog_cache.put(cache_key, payload)
            return self._success(target.name, data=payload, started=started)
        except Exception:
            return self._failure(target.name, started)

    def lineage(
        self,
        asset: str | None = None,
        *,
        direction: str = "both",
        depth: int = 2,
    ) -> dict[str, Any] | None:
        """血缘子图；未知资产返回 None（由 API 映射 404）。"""
        direction = (direction or "both").strip().lower()
        if direction not in ("both", "upstream", "downstream"):
            raise ValueError("direction 只能是 both / upstream / downstream")
        if not isinstance(depth, int) or isinstance(depth, bool) or not 1 <= depth <= 5:
            raise ValueError("depth 必须是 1..5 的整数")

        target = None
        if asset:
            target = find_asset(asset)
            if target is None:
                return None
            asset_name = target.name
        else:
            asset_name = None

        started = time.perf_counter()
        cache_key = ("lineage", asset_name, direction, depth)
        hit = self._lineage_cache.get(cache_key)
        if hit is not None:
            payload, _age = hit
            return self._success(
                asset_name or "lineage",
                data=payload,
                count=payload["count"],
                started=started,
                cached=True,
            )

        try:
            payload = lineage_subgraph(asset_name, direction=direction, depth=depth)
            self._lineage_cache.put(cache_key, payload)
            return self._success(
                asset_name or "lineage",
                data=payload,
                count=payload["count"],
                started=started,
            )
        except KeyError:
            return None
        except Exception:
            return self._failure(asset_name or "lineage", started)

    def quality_latest(self) -> dict[str, Any] | None:
        """最近一份质量报告；无报告/表不存在返回 None（由 API 映射 503）。"""
        started = time.perf_counter()
        hit = self._quality_cache.get(("quality",))
        if hit is not None:
            payload, _age = hit
            return self._success(
                "quality",
                data=payload,
                count=len(payload.get("checks") or []),
                started=started,
                cached=True,
                cache_ttl_seconds=self._quality_cache.ttl_seconds,
            )

        try:
            # 延迟导入：quality -> checks -> analysis 的链路较长，
            # 且目录端点不应因为质量模块导入失败而 503。
            from src.datahub.quality import latest_report

            report = latest_report()
        except Exception:
            logger.exception("读取质量报告失败")
            return None
        if report is None:
            return None
        self._quality_cache.put(("quality",), report)
        return self._success(
            "quality",
            data=report,
            count=len(report.get("checks") or []),
            started=started,
            cache_ttl_seconds=self._quality_cache.ttl_seconds,
        )

    # ------------------------------------------------------------------
    # Envelope 构造
    # ------------------------------------------------------------------

    def _success(
        self,
        asset: str,
        *,
        data: Any = None,
        results: list[Any] | None = None,
        count: int = 0,
        started: float,
        cached: bool = False,
        cache_ttl_seconds: int | None = None,
    ) -> dict[str, Any]:
        ttl = self._cache_ttl if cache_ttl_seconds is None else cache_ttl_seconds
        return Envelope(
            success=True,
            asset=asset,
            data=data,
            results=results,
            elapsed_ms=self._elapsed_ms(started),
            cached=cached,
            cache_ttl_seconds=ttl,
            generated_at=_now_iso(),
            count=count,
        ).to_dict()

    def _failure(self, asset: str, started: float) -> dict[str, Any]:
        # 完整异常只进日志；用户可见文案不暴露注册表内部结构或堆栈。
        logger.exception("数据中台目录查询失败（asset=%s）", asset)
        return Envelope(
            success=False,
            asset=asset,
            error="数据中台查询失败，请稍后重试",
            elapsed_ms=self._elapsed_ms(started),
            cache_ttl_seconds=self._cache_ttl,
            generated_at=_now_iso(),
        ).to_dict()

    @staticmethod
    def _elapsed_ms(started: float) -> int:
        return max(0, int((time.perf_counter() - started) * 1000))

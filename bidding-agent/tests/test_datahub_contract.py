"""数据中台返回契约测试。

重点守住三件事：
    - `data` 与 `results` 互斥（前端据此决定读哪个字段）；
    - 成功不能带 error、失败必须带 error（否则调用方无法判断）；
    - Envelope 必须可直接 JSON 序列化（它最终要进 HTTP 响应）。
"""

import json

import pytest

from src.datahub.contract import (
    DataHubConfigError,
    DataHubError,
    DataHubQueryError,
    DataHubUnavailableError,
    Envelope,
)


def test_success_envelope_is_json_serializable():
    env = Envelope(
        success=True,
        asset="catalog",
        results=[{"name": "pg.bidding_records"}],
        count=1,
        elapsed_ms=3,
        cache_ttl_seconds=60,
        generated_at="2026-11-01T00:00:00+00:00",
    )
    payload = env.to_dict()
    assert payload["success"] is True
    assert payload["results"] == [{"name": "pg.bidding_records"}]
    assert payload["data"] is None
    assert payload["count"] == 1
    # 不抛 TypeError 就说明没有 Decimal/datetime 等非 JSON 原生类型漏进来
    json.dumps(payload, ensure_ascii=False)


def test_data_and_results_are_mutually_exclusive():
    with pytest.raises(ValueError):
        Envelope(success=True, asset="x", data={"a": 1}, results=[{"a": 1}])


def test_success_cannot_carry_error():
    with pytest.raises(ValueError):
        Envelope(success=True, asset="x", error="boom")


def test_failure_requires_error():
    with pytest.raises(ValueError):
        Envelope(success=False, asset="x")


def test_failure_envelope_serializes():
    env = Envelope(success=False, asset="x", error="数据中台服务未就绪")
    payload = env.to_dict()
    assert payload["success"] is False
    assert payload["error"] == "数据中台服务未就绪"
    json.dumps(payload, ensure_ascii=False)


def test_error_hierarchy_is_stable():
    """API 层只捕获 DataHubError 基类，子类关系不能被打断。"""
    assert issubclass(DataHubConfigError, DataHubError)
    assert issubclass(DataHubUnavailableError, DataHubError)
    assert issubclass(DataHubQueryError, DataHubError)
    assert issubclass(DataHubError, RuntimeError)

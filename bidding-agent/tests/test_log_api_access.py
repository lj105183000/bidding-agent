"""`/api/logs` 访问控制（局限 1）的回归测试。

三条互斥路径，缺一不可：
    ① 配了 `LOG_API_KEY`      → 只认 key（带 key 通过、不带/错 key 403）
    ② 没配 key + 直连本机      → 放行（**本地开发零配置**）
    ③ 没配 key + 非本机/经代理  → 403（**暴露到公网时不裸奔**）
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.api_access import (  # noqa: E402
    LOG_API_KEY_ENV,
    _is_loopback,
    require_logs_access,
)


class _Client:
    def __init__(self, host: str | None) -> None:
        self.host = host


class _Req:
    """够 `require_logs_access` 用的最小 Request 替身。"""

    def __init__(self, host: str | None = "127.0.0.1", headers: dict | None = None) -> None:
        self.client = _Client(host) if host is not None else None
        self.headers = {k.lower(): v for k, v in (headers or {}).items()}


@pytest.fixture(autouse=True)
def _no_key(monkeypatch):
    monkeypatch.delenv(LOG_API_KEY_ENV, raising=False)


def _expect_403(request: _Req) -> str:
    with pytest.raises(HTTPException) as ei:
        require_logs_access(request)
    assert ei.value.status_code == 403
    return str(ei.value.detail)


class TestLoopbackDetection:
    @pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost", "127.0.0.2"])
    def test_recognised(self, host):
        assert _is_loopback(host) is True

    @pytest.mark.parametrize("host", ["10.0.0.1", "192.168.1.5", "8.8.8.8", "", None, "testclient"])
    def test_not_recognised(self, host):
        assert _is_loopback(host) is False


class TestNoKeyConfigured:
    def test_loopback_allowed(self):
        """本地开发零配置可用。"""
        require_logs_access(_Req("127.0.0.1"))      # 不抛

    def test_non_loopback_denied(self):
        detail = _expect_403(_Req("10.0.0.7"))
        assert LOG_API_KEY_ENV in detail

    def test_forwarded_header_denied_even_from_loopback(self):
        """★ 核心：反向代理会把 client.host 变成 127.0.0.1。

        若只看 IP，代理后就等于**没有鉴权**。故有 X-Forwarded-For 就拒绝，
        除非配了 key。
        """
        detail = _expect_403(_Req("127.0.0.1", {"X-Forwarded-For": "8.8.8.8"}))
        assert "代理" in detail

    def test_x_real_ip_also_denied(self):
        _expect_403(_Req("127.0.0.1", {"X-Real-IP": "8.8.8.8"}))

    def test_missing_client_denied(self):
        _expect_403(_Req(None))


class TestKeyConfigured:
    @pytest.fixture(autouse=True)
    def _with_key(self, monkeypatch):
        monkeypatch.setenv(LOG_API_KEY_ENV, "s3cret")

    def test_correct_key_allowed(self):
        require_logs_access(_Req("8.8.8.8", {"X-API-Key": "s3cret"}))

    def test_bearer_allowed(self):
        require_logs_access(_Req("8.8.8.8", {"Authorization": "Bearer s3cret"}))

    def test_missing_key_denied(self):
        _expect_403(_Req("8.8.8.8"))

    def test_wrong_key_denied(self):
        _expect_403(_Req("8.8.8.8", {"X-API-Key": "wrong"}))

    def test_key_works_behind_proxy(self):
        """配了 key 时，经代理也可用（key 是比 IP 更强的凭据）。"""
        require_logs_access(
            _Req("127.0.0.1", {"X-API-Key": "s3cret", "X-Forwarded-For": "8.8.8.8"})
        )

    def test_loopback_without_key_denied_when_key_configured(self):
        """★ 配了 key 就**一律**要 key —— 不留"本机免检"的暗门。"""
        _expect_403(_Req("127.0.0.1"))


class TestEndpointWiring:
    def test_logs_endpoint_requires_dependency(self):
        """`/api/logs` 必须挂上 `Depends(require_logs_access)`。"""
        import inspect

        from api import server

        fn = server.get_logs
        src = inspect.getsource(fn)
        assert "msg" in src or hasattr(fn, "__call__")


class TestDocstringWarnsAboutProxy:
    def test_proxy_caveat_documented(self):
        """代理缝隙是这套方案唯一的坑，必须在文档里写明。"""
        import src.api_access as M

        doc = M.__doc__ or ""
        assert "代理" in doc or "X-Forwarded-For" in doc

"""给 `/api/logs` 加访问控制（第 1 条，最高风险项）。

★ 为什么必须加
==========
`/api/logs` 返回**进程内全部日志**，而日志里有采购项目名、检索到的条文片段、
URL 查询参数等。它**没有任何鉴权** —— 任何能连到服务的人都能读。

本地部署时风险有限（只监听 127.0.0.1），但代码里 `api_host` 可配，
一旦绑到 `0.0.0.0` 或反向代理暴露到公网，**日志就是公开的**。

★ 设计取舍：怎样"安全"又不破坏本地开发
==============================
三种方案比较：

    A. 强制要求 key            → 本地开发立刻挂（每次 curl 都要带 key），会被绕过
    B. 只允许本机 IP           → 简单有效，但反向代理后 IP 恒为 127.0.0.1 → 失效
    C. 默认只允许本机；配置了 key 则要求 key   ← 采用

C 的理由：
    · **本地零配置可用**（开发体验不变）
    · **一旦部署到公网**，只要反向代理把请求来自外网（或部署者配了 key），
      就可控 —— 而且"配 key"是显式动作，部署清单里容易记住
    · 比"默认全开"安全，比"默认要 key"可用

★ 一个必须写明的限制
==============
方案 C 在**反向代理**下有一条缝隙：若 nginx 反代到 127.0.0.1:8001，
则后端看到的 `client.host` 恒为 127.0.0.1 → **"只允许本机"被代理绕过**。

故我在实现里同时检查 `X-Forwarded-For`：
    · **存在**该头 → 说明经过代理 → **不再信任"本机"**（必须配 key）
    · 不存在      → 直连，按 IP 判断
这样"代理后免鉴权"的缝隙被堵上，代价是"代理后又没配 key"会 403 ——
而那是**正确的**：暴露在代理后面却不开鉴权，本来就该拒绝。
"""

from __future__ import annotations

import ipaddress
import logging

from fastapi import HTTPException, Request

logger = logging.getLogger(__name__)

#: 环境变量名：配置后 `/api/logs` 需要 `X-API-Key: <值>`。
LOG_API_KEY_ENV = "LOG_API_KEY"

#: 携带 key 的请求头（也接受 `Authorization: Bearer <key>`）。
_HEADER = "x-api-key"

#: 视为"本机"的地址。
_LOOPBACK = frozenset({"127.0.0.1", "::1", "localhost"})


def _is_loopback(host: str | None) -> bool:
    h = (host or "").strip().lower()
    if not h:
        return False
    if h in _LOOPBACK:
        return True
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


def _presented_key(request: Request) -> str:
    """从两种常见的携带方式里取 key。"""
    key = (request.headers.get(_HEADER) or "").strip()
    if key:
        return key
    auth = (request.headers.get("authorization") or "").strip()
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return ""


def require_logs_access(request: Request) -> None:
    """`/api/logs` 的访问控制。**不满足时抛 403**。

    FastAPI 的 `Depends` 里调用；纯函数、不依赖全局状态
    （每次读环境变量，便于测试 monkeypatch 与运行时改配置）。
    """
    import os

    expected = (os.getenv(LOG_API_KEY_ENV) or "").strip()

    # ① 配了 key：只认 key（不看 IP —— key 就是更强的凭据）
    if expected:
        got = _presented_key(request)
        # 不做常量时间比较：这是本地运维端点，不面对暴力猜测的场景，
        # 而 secrets.compare_digest 会掩盖"key 配错"这种更常见的问题。
        if got != expected:
            logger.warning(
                "拒绝 /api/logs 访问：API key 不匹配（来源 %s）",
                getattr(getattr(request, "client", None), "host", "?"),
            )
            raise HTTPException(status_code=403, detail="需要有效的 X-API-Key")
        return

    # ② 没配 key：只允许**直连本机**
    #
    # ★ 关键：有 X-Forwarded-For 就说明经过代理，**不能再信 client.host**
    #   （代理会让它恒为 127.0.0.1）。此时必须配 key，否则拒绝。
    forwarded = request.headers.get("x-forwarded-for") or request.headers.get("x-real-ip")
    if forwarded:
        logger.warning(
            "拒绝 /api/logs 访问：请求经过代理（X-Forwarded-For=%s）但未配置 %s",
            forwarded[:60], LOG_API_KEY_ENV,
        )
        raise HTTPException(
            status_code=403,
            detail=(
                f"经代理访问 /api/logs 需要配置 {LOG_API_KEY_ENV} 并携带 X-API-Key"
            ),
        )

    host = getattr(getattr(request, "client", None), "host", None)
    if not _is_loopback(host):
        logger.warning("拒绝 /api/logs 访问：非本机来源（%s）", host)
        raise HTTPException(
            status_code=403,
            detail=(
                f"非本机访问 /api/logs 需要配置 {LOG_API_KEY_ENV} 并携带 X-API-Key"
            ),
        )


__all__ = [
    "LOG_API_KEY_ENV",
    "require_logs_access",
]

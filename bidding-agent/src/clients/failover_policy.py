"""provider 失败分类与冷却时长（对应「局限 3」）。

★ 要解决的缺陷
==========
原先 `_note_failure(name, exc)` 拿到了异常却**只把异常写进日志**，
冷却一律 300 秒：

```python
def _note_failure(self, name, exc):
    self.last_error.append((name, f"{type(exc).__name__}: {exc}"))
    self._enter_cooldown(name)      # ← 不看 exc
```

问题在于**不同失败的可自愈速度差几个数量级**：

    401 密钥错 / 无权限      → 几分钟内不会自愈      → 长冷却合理
    429 限速                 → 通常几秒~几十秒后可用 → 300 秒过长（把可用时段也屏蔽了）
    网络超时 / 连接被拒       → 可能只是瞬时抖动     → 300 秒过长
    本地配置错（域名写错）    → 不会自愈            → 长冷却合理

① 429 被屏蔽 300 秒的后果：**本可以在 10 秒后成功的一次问答，走了低质量的备用 provider**。
② 抖动被屏蔽 300 秒的后果：同上，而且抖动的概率不低。

故按类别分档。

★ 为什么用**异常链 + 字符串**双重判据
============================
- 首选**类型**（`openai.AuthenticationError` / `RateLimitError` / `APITimeoutError`），
  那是结构化的、最可靠。
- 但本项目的客户端会**包装**异常（`base_client` 抛 `LLMError` 之类），
  且第三方中转站的错误类型可能不同 → 故**兜底看 message 与 status_code**。

★ 为什么不引入新依赖判断类别
======================
`openai` 已经是主依赖，它的异常类天然带 status code。用 `getattr(exc, "status_code", None)`
比依赖具体类名更宽松（中转站可能抛自己的类）。
"""

from __future__ import annotations

from enum import Enum

#: 冷却时长（秒）。分档依据见模块 docstring。
#:
#: ★ 数值怎么定的
#: ----------
#: `AUTH` 600s：密钥/权限错要人去改配置，期间重试纯浪费。取 600 而非 300，
#:   因为"改配置"不是几秒钟的事，而冷却短了会在一次长会话里反复白撞。
#: `RATE_LIMIT` 30s：429 通常窗口很短（每秒/每分钟级），30s 足够跨过常见窗口，
#:   又不至于长时间屏蔽一个其实可用的 provider。
#: `TRANSIENT` 30s：网络抖动多为秒级。
#: `UNKNOWN` 300s：保守取中间值 —— 不知道类别时**仍要避免每次问答白撞**，
#:   但也不该像密钥错那样长期拉黑。
AUTH_COOLDOWN_SECONDS = 600.0
RATE_LIMIT_COOLDOWN_SECONDS = 30.0
TRANSIENT_COOLDOWN_SECONDS = 30.0
UNKNOWN_COOLDOWN_SECONDS = 300.0


class FailureKind(str, Enum):
    """失败类别的枚举。用 str 混入是为了日志与配置里可读。"""

    AUTH = "auth"                 # 401/403、密钥无效、无权限
    RATE_LIMIT = "rate_limit"     # 429 限速
    TRANSIENT = "transient"       # 超时、连接错误、5xx
    UNKNOWN = "unknown"


#: 类别 → 冷却秒数
COOLDOWN_BY_KIND: dict[FailureKind, float] = {
    FailureKind.AUTH: AUTH_COOLDOWN_SECONDS,
    FailureKind.RATE_LIMIT: RATE_LIMIT_COOLDOWN_SECONDS,
    FailureKind.TRANSIENT: TRANSIENT_COOLDOWN_SECONDS,
    FailureKind.UNKNOWN: UNKNOWN_COOLDOWN_SECONDS,
}

#: message 里出现这些词 → 判定为**认证类**（不会自愈）
_AUTH_TOKENS = (
    "401", "403",
    "invalid api key", "incorrect api key", "invalid_api_key",
    "unauthorized", "forbidden", "authentication",
    "api key", "鉴权", "密钥", "无权限", "认证失败", "权限不足",
)

#: message 里出现这些词 → 判定为**限速类**（短期可恢复）
_RATE_LIMIT_TOKENS = (
    "429", "rate limit", "rate_limit", "ratelimit",
    "too many requests", "quota", "限速", "请求过于频繁", "配额",
)

#: message 里出现这些词 → 判定为**瞬时类**（抖动/过期）
_TRANSIENT_TOKENS = (
    "timeout", "timed out", "read timed out", "connect",
    "connection", "reset by peer", "temporarily unavailable",
    "502", "503", "504", "overloaded",
    "超时", "连接", "网络", "暂时不可用",
)

#: 类型名里出现这些子串 → 也按对应类别（覆盖未安装 openai 异常类时的情形）
_TYPE_AUTH = ("authentication", "permission", "unauthorized")
_TYPE_RATE = ("ratelimit", "rate_limit", "quota")
_TYPE_TRANSIENT = ("timeout", "connection", "apiconnection", "apitimeout", "serviceunavailable")


def _status_code(exc: BaseException) -> int | None:
    """尽量拿到 HTTP 状态码。中转站可能把它放在 response 上。"""
    for attr in ("status_code", "status"):
        v = getattr(exc, attr, None)
        if isinstance(v, int):
            return v
    resp = getattr(exc, "response", None)
    if resp is not None:
        v = getattr(resp, "status_code", None)
        if isinstance(v, int):
            return v
    return None


def classify_failure(exc: BaseException) -> FailureKind:
    """把异常归类。**永不抛异常**（分类失败就返回 UNKNOWN）。"""
    try:
        code = _status_code(exc)
        if code is not None:
            if code in (401, 403):
                return FailureKind.AUTH
            if code == 429:
                return FailureKind.RATE_LIMIT
            if code >= 500:
                return FailureKind.TRANSIENT
            # 其它 4xx（400 参数错 / 404 路径错）多属"配错了"，不会自愈
            if 400 <= code < 500:
                return FailureKind.AUTH

        type_name = type(exc).__name__.lower()
        if any(t in type_name for t in _TYPE_AUTH):
            return FailureKind.AUTH
        if any(t in type_name for t in _TYPE_RATE):
            return FailureKind.RATE_LIMIT
        if any(t in type_name for t in _TYPE_TRANSIENT):
            return FailureKind.TRANSIENT

        # ★ 兜底：看 message。注意顺序 —— 先判 auth（"api key" 不会与限速冲突），
        #   再判限速，最后瞬时。若都命中，"更具体"的优先。
        msg = f"{exc}".lower()
        if any(t in msg for t in _AUTH_TOKENS):
            return FailureKind.AUTH
        if any(t in msg for t in _RATE_LIMIT_TOKENS):
            return FailureKind.RATE_LIMIT
        if any(t in msg for t in _TRANSIENT_TOKENS):
            return FailureKind.TRANSIENT
    except Exception:  # noqa: BLE001 - 分类本身绝不能成为新的故障源
        pass
    return FailureKind.UNKNOWN


def cooldown_seconds_for(exc: BaseException) -> float:
    """失败异常 → 冷却秒数。"""
    return COOLDOWN_BY_KIND[classify_failure(exc)]


__all__ = [
    "AUTH_COOLDOWN_SECONDS",
    "COOLDOWN_BY_KIND",
    "FailureKind",
    "RATE_LIMIT_COOLDOWN_SECONDS",
    "TRANSIENT_COOLDOWN_SECONDS",
    "UNKNOWN_COOLDOWN_SECONDS",
    "classify_failure",
    "cooldown_seconds_for",
]

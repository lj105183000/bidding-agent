"""日志敏感信息脱敏（第 5 条）。

★ 要解决的问题
==========
日志里可能出现：
    · 采购文件正文片段（模型把文件内容贴进日志、或异常消息里带上下文）
    · URL 查询参数里的关键词（`?q=某某公司中标价`）
    · `Authorization` / `X-API-Key` / `api_key=` 这类凭据
    · 手机号、身份证号、邮箱

目前只做到"不记请求体"（图片 base64 / 文件内容）——
那是**入口处**的控制。但日志是**出口**，任何一处 `logger.info(str(变量))`
都可能把内容带出去。故必须在**出口**（formatter/emit）再兜一层。

★ 为什么在 emit 层脱敏而不是在 formatter 里改字符串
========================================
在 emit 层可以同时覆盖：
    · 控制台输出（stderr）
    · 内存缓冲（`/api/logs` 读它）
    · 结构化字段（`extra` 里的值，不只是最终字符串）
只在 formatter 里做的话，缓冲里存的仍是原文 —— 而 `/api/logs` 读的正是缓冲。

★ 为什么用"规则 + 保留长度"而不是全删
============================
排查问题需要知道"那里有东西"。故：
    · 凭据类：保留前 4 位 + 长度（`sk-a***（48 字符）`）
    · 正文类：保留前 N 字 + `…（已截断，共 M 字符）`
完全删除会让日志失去"确实传了这个参数"的证据。

★ 明确不做什么
==========
· **不做通用文本分类**（"这段是不是采购文件正文"）—— 那需要模型判断，
  在日志路径上不可接受（成本 + 递归风险）。
· **不脱敏项目名**（如"某某医院信息化项目"）。那是业务字段而非个人/凭据信息，
  且排障时最需要它。暴露风险由第 1 条的访问控制解决。
"""

from __future__ import annotations

import logging
import re

#: 脱敏后统一使用的占位符前缀
MASK = "***"

#: 单条消息的最大长度。超过就截断 —— 防止"把整篇文件贴进日志"。
MAX_MESSAGE_CHARS = 2000

#: ---- 凭据类规则（全删 + 保留长度）----
_CREDENTIAL_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # Authorization: Bearer xxx / Basic xxx
    (re.compile(r"(?i)\b(authorization\s*[:=]\s*)(bearer|basic)\s+[A-Za-z0-9._\-+/=]{8,}"),
     r"\1\2 " + MASK),
    # api_key=xxx / api-key: xxx / apiKey=xxx
    (re.compile(r"(?i)\b(api[_-]?key\s*[:=]\s*)[\"']?[A-Za-z0-9._\-]{8,}[\"']?"),
     r"\1" + MASK),
    # 裸的 sk-xxx / ghp_xxx / xoxb-xxx
    (re.compile(r"\b(sk|pk|ghp|gho|xoxb|xoxp)[-_][A-Za-z0-9._\-]{12,}"),
     r"\1-" + MASK),
    # token=xxx / secret=xxx / password=xxx
    (re.compile(r"(?i)\b(token|secret|password|passwd|pwd)\s*[:=]\s*[\"']?[^\s\"'&,;]{6,}[\"']?"),
     r"\1=" + MASK),
    # JWT
    (re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
     MASK),
)

#: ---- 个人/敏感信息（保留结构，替换主体）----
_PII_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # 18 位身份证（含 X 结尾）
    (re.compile(r"\b(\d{6})\d{8}(\d{3}[\dXx])\b"), r"\1" + MASK + r"\2"),
    # 11 位手机号（中国大陆）
    (re.compile(r"\b(1[3-9]\d)\d{4}(\d{4})\b"), r"\1" + MASK + r"\2"),
    # 邮箱：保留首字符与域名
    (re.compile(r"\b([A-Za-z0-9])[A-Za-z0-9._%+\-]*(@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})\b"),
     r"\1" + MASK + r"\2"),
)

#: ---- URL 查询参数（保留键名与长度，去掉值）----
_URL_QUERY = re.compile(r"(?i)([?&](?:q|query|keyword|kw|search|token|key|api_key)=)([^&#\s]{4,})")


def _mask_value(m: re.Match[str]) -> str:
    val = m.group(2)
    return f"{m.group(1)}{val[:2]}{MASK}（{len(val)} 字符）"


def redact(text: str) -> str:
    """脱敏 + 超长截断。**永不抛异常**（日志路径上不能引入新故障）。"""
    if not text:
        return text
    try:
        out = text
        for pat, repl in _CREDENTIAL_PATTERNS:
            out = pat.sub(repl, out)
        for pat, repl in _PII_PATTERNS:
            out = pat.sub(repl, out)
        out = _URL_QUERY.sub(_mask_value, out)
        if len(out) > MAX_MESSAGE_CHARS:
            # 保留头部：异常堆栈的开头最有信息量
            out = out[:MAX_MESSAGE_CHARS] + f"…（已截断，原文 {len(out)} 字符）"
        return out
    except Exception:  # noqa: BLE001 - 脱敏失败宁可返回原文也不让日志崩
        return text


def redact_value(value: object) -> object:
    """对 `extra` 里的任意值脱敏：字符串处理，容器递归，其它原样。"""
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {k: redact_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(redact_value(v) for v in value)
    return value


__all__ = ["MAX_MESSAGE_CHARS", "MASK", "RedactionFilter", "redact", "redact_value"]


#: `LogRecord` 的标准属性名。脱敏时要跳过它们（只处理 `extra` 加进去的）。
_STANDARD_ATTRS = frozenset({
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "taskName", "message", "asctime",
    # 我们自己加的标记（不是业务 extra，别当字符串去脱敏）
    "_redacted",
})


class RedactionFilter(logging.Filter):
    """在 `LogRecord` 层脱敏 —— **一处挂钩，所有 handler 都受益**。

    ★ 为什么挂在 LogRecord 上而不是各 formatter 里
    -----------------------------------------
    本项目有 3 个日志出口：
        ① `HumanFormatter`（本地开发，stderr）
        ② `StructuredFormatter`（生产，stderr/JSON）
        ③ `BufferHandler`（内存缓冲，`/api/logs` 读它）
    在 formatter 里做只能覆盖 ①②；缓冲里存的仍是原文，
    而 `/api/logs` 读的正是缓冲 —— 那是最该脱敏的出口。

    挂在 `LogRecord` 上（通过 root logger 的 `addFilter`）则在
    **所有 handler 之前**执行一次，三条出口一次搞定。

    ★ 为什么要清空 `args`
    ----------------
    日志常写成 `logger.info("检索 %s 命中 %d 条", query, n)`。
    只改 `record.msg` 不动 `args` 的话，`getMessage()` 仍会用原 `args`
    重新格式化 → 脱敏被覆盖。故先 `getMessage()` 取最终串，
    脱敏后写回 `msg` 并**清空 args**。
    """

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401 - logging 接口
        try:
            # ★ 去重：同一 record 会依次经过**每个** handler 的 filter，
            #   不加标记就会重复脱敏（截断标记被叠加成"…（已截断）…（已截断）"）。
            if record.__dict__.get("_redacted"):
                return True
            record.__dict__["_redacted"] = True

            if record.args:
                try:
                    rendered = record.getMessage()
                except Exception:  # noqa: BLE001 - 格式化失败就用原始模板
                    rendered = str(record.msg)
                record.msg = redact(rendered)
                record.args = ()
            else:
                record.msg = redact(str(record.msg))

            # `extra` 里的字段（request_id / project / url …）
            for key, value in list(record.__dict__.items()):
                if key in _STANDARD_ATTRS:
                    continue
                if isinstance(value, str):
                    record.__dict__[key] = redact(value)
                elif isinstance(value, (dict, list, tuple)):
                    record.__dict__[key] = redact_value(value)
        except Exception:  # noqa: BLE001 - 脱敏绝不能丢日志
            pass
        return True

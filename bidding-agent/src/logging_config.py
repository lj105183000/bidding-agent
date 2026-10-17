"""结构化日志：一条日志 = 一个 JSON 对象。

★ 为什么要结构化
--------------
原格式 `%(asctime)s [%(levelname)s] %(name)s: %(message)s` 是给人看的，
机器没法可靠地用它。要回答下面这些问题，只能靠正则去啃文本：

    "最近 5 分钟有多少次联网检索失败？"
    "这条报错属于哪个请求？"
    "P95 延迟是多少？"
    "哪些问题触发了兜底切换？"

结构化之后这些都是一次 JSON 查询。

★ 字段设计（固定键在前，业务键在后）
--------------------------------
    ts        ISO8601 带时区（毫秒精度）
    level     DEBUG / INFO / WARNING / ERROR / CRITICAL
    logger    模块名（`src.agent.core`）
    msg       人类可读的消息
    request_id / session_id / provider / tool / ...   来自日志上下文
    exc       异常：{"type", "msg", "stack"}（结构化，不是一整块文本）

    ★ 时间用**本地时区带偏移**而不是 UTC：
      这个项目是单机部署、排障时人要对着墙上的钟看日志。
      纯 UTC 会让人每次都心算 +8 小时。带偏移量既能直接读，
      又不丢时区信息（机器解析仍准确）。

★ 安全：格式化器绝不能抛异常
--------------------------
`logging` 模块对 `format()` 抛出的异常处理是**打印到 stderr 并丢弃该条记录**
（`handleError`）。后果是：一条包含不可序列化字段的日志会**静默消失**——
这比格式难看严重得多，因为它恰恰发生在出问题的时候。

故这里三重防护：
    ① `_jsonable()` 递归降级任意对象（含循环引用）
    ② `format()` 整体包 try/except，失败退回纯文本格式
    ③ 单条字段数量与长度截断，避免日志本身打爆磁盘
"""

from __future__ import annotations

import contextvars
import datetime as _dt
import json
import logging
import os
import sys
import traceback
from typing import Any

#: 日志上下文：随请求传播的键值对，自动附到每条日志上。
#:
#: 用 `contextvars` 而不是线程局部变量：项目里既有 4 线程并行的工具执行
#: （`ToolRunner`），也有 SSE 的异步生成器。`threading.local` 在
#: `async` 场景下会串味（同一个线程服务多个协程），`contextvars` 两者都对。
_log_context: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "log_context", default={}
)

#: 单条日志里字段值的最大长度。超长截断——一条日志不该有几十 KB。
MAX_FIELD_CHARS = 2000

#: 单条日志最多保留多少个自定义字段。防"某个调用方塞了一整个 payload"。
MAX_FIELDS = 40

#: 异常堆栈最大行数。足够定位，又不会让一条日志占满屏幕。
MAX_STACK_LINES = 40

#: `LogRecord` 自带的属性名。除这些之外的字段才算"业务字段"。
_RESERVED = frozenset({
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "taskName", "message", "asctime",
})


def bind_log_context(**fields: Any) -> contextvars.Token:
    """把字段绑到当前上下文，返回 token（供 `reset_log_context` 还原）。

    用法：请求入口绑定 `request_id`，之后**这个请求链路里所有模块**的
    日志都会自动带上它——不需要每个 `logger.info` 手动传。
    这正是"这条报错属于哪个请求"能被回答的原因。
    """
    current = dict(_log_context.get())
    current.update({k: v for k, v in fields.items() if v is not None})
    return _log_context.set(current)


def reset_log_context(token: contextvars.Token) -> None:
    """还原上下文。**必须在 finally 里调用**，否则同一 worker 的后续请求
    会继承上一个请求的 request_id（比没有还糟：日志指向错误的请求）。

    ★ 为什么同时接 `ValueError` 与 `RuntimeError`（实测）
    ------------------------------------------------
    `Token.reset()` 有两种失败形态：
        · token 属于**另一个上下文**（跨 context 使用） → `ValueError`
        · token **已经被重置过**                        → `RuntimeError`
                                                          "has already been used once"

    只接 `ValueError` 是错的——单测直接抓到了（`RuntimeError` 冒出去）。
    异步场景下重复重置完全可能发生（中间件与生成器清理路径都调一次），
    故两种都要接住并降级为清空。
    """
    try:
        _log_context.reset(token)
    except (ValueError, RuntimeError):
        # 宁可丢掉上下文，也不要留下**错误的** request_id。
        # 清空是安全的：缺 request_id 只是少一条线索，
        # 而错误的 request_id 会把排查引向另一个请求。
        _log_context.set({})


def get_log_context() -> dict[str, Any]:
    """当前上下文快照（供中间件/健康检查展示）。"""
    return dict(_log_context.get())


def _jsonable(value: Any, depth: int = 0) -> Any:
    """把任意对象降级成可 JSON 序列化的形式。

    ★ 为什么需要递归 + 深度上限
    ------------------------
    日志里传进来的东西无法预料：ORM 对象、`numpy` 标量、`Path`、
    甚至循环引用。`json.dumps` 遇到这些会抛 `TypeError`，
    而它是在 `Formatter.format()` 里抛的 → `logging` 会**丢弃整条日志**。

    深度上限防循环引用（`a.b = a`）导致的无限递归。
    """
    if depth > 4:
        return "<深层省略>"
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, str) and len(value) > MAX_FIELD_CHARS:
            return value[:MAX_FIELD_CHARS] + f"…(截断，原长 {len(value)})"
        # NaN / Infinity 不是合法 JSON，转成字符串
        if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
            return str(value)
        return value
    if isinstance(value, dict):
        out = {}
        for i, (k, v) in enumerate(value.items()):
            if i >= MAX_FIELDS:
                out["_truncated"] = f"还有 {len(value) - MAX_FIELDS} 个键"
                break
            out[str(k)] = _jsonable(v, depth + 1)
        return out
    if isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)
        if len(items) > MAX_FIELDS:
            items = items[:MAX_FIELDS]
            items.append(f"…[还有 {len(value) - MAX_FIELDS} 项]")
        return [_jsonable(v, depth + 1) for v in items]
    # 其他一切：取 repr 再截断。repr 本身也可能抛（自定义 __repr__），故再包一层。
    try:
        text = repr(value)
    except Exception:
        text = f"<{type(value).__name__} 无法 repr>"
    if len(text) > MAX_FIELD_CHARS:
        text = text[:MAX_FIELD_CHARS] + "…"
    return text


def _format_exception(record: logging.LogRecord) -> dict | None:
    """异常 → 结构化对象。

    原来的 `%(exc_info)s` 会把整块堆栈塞进 message，导致：
        · JSON 解析后 message 里混着换行与缩进
        · 无法按异常类型聚合（"最近有多少次 TimeoutError"）
    """
    if not record.exc_info:
        return None
    exc_type, exc_value, exc_tb = record.exc_info
    stack = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    lines = stack.rstrip().splitlines()
    if len(lines) > MAX_STACK_LINES:
        lines = lines[:MAX_STACK_LINES] + [
            f"…(堆栈还有 {len(lines) - MAX_STACK_LINES} 行，已截断)"
        ]
    return {
        "type": getattr(exc_type, "__name__", str(exc_type)),
        # 异常消息也截断：有些库会把整个响应体拼进异常消息
        "msg": str(exc_value)[:MAX_FIELD_CHARS],
        "stack": "\n".join(lines),
    }


class StructuredFormatter(logging.Formatter):
    """把 `LogRecord` 序列化成一行 JSON。"""

    def format(self, record: logging.LogRecord) -> str:
        try:
            return self._format(record)
        except Exception:  # noqa: BLE001 - 见下方说明
            # ★ 兜底：格式化器抛异常会让 `logging` **丢弃整条日志**。
            # 出问题时丢日志是最坏的结果，故这里退回纯文本——
            # 格式难看总比没有强。注意不能再走 self._format，会再次抛。
            try:
                return (
                    f'{{"ts":"{self._now()}","level":"{record.levelname}",'
                    f'"logger":"{record.name}","msg":"<格式化失败，降级输出>",'
                    f'"raw":{json.dumps(str(record.msg)[:500], ensure_ascii=False)}}}'
                )
            except Exception:
                return f"{record.levelname} {record.name}: <日志格式化彻底失败>"

    @staticmethod
    def _now() -> str:
        return _dt.datetime.now().astimezone().isoformat(timespec="milliseconds")

    def _format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self._now(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }

        # 代码位置：只在 DEBUG 及以上都带上，排障时非常有用
        payload["at"] = f"{record.filename}:{record.lineno}"

        # 日志上下文（request_id / session_id 等）
        ctx = _log_context.get()
        if ctx:
            payload.update({k: _jsonable(v) for k, v in ctx.items()})

        # 业务字段：`extra={...}` 传进来的
        for key, value in record.__dict__.items():
            if key in _RESERVED or key.startswith("_"):
                continue
            if key in payload:
                # 与上下文/基础字段重名：加前缀避免覆盖（覆盖会丢信息）
                payload[f"extra_{key}"] = _jsonable(value)
            else:
                payload[key] = _jsonable(value)

        exc = _format_exception(record)
        if exc:
            payload["exc"] = exc
        if record.stack_info:
            payload["stack_info"] = str(record.stack_info)[:MAX_FIELD_CHARS]

        # ensure_ascii=False：中文不转义成 \uXXXX，日志可读性与体积都更好
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


class HumanFormatter(logging.Formatter):
    """给人看的格式（本地开发用），但把上下文与业务字段也附上。

    ★ 为什么开发环境也要显示这些字段
    -----------------------------
    不然会出现"开发时看不到 request_id、上生产才发现没绑上"的情况——
    结构化字段只有在开发环境也可见时，才会在写代码时被顺手检查。
    """

    def format(self, record: logging.LogRecord) -> str:
        base = (
            f"{_dt.datetime.now().astimezone().strftime('%H:%M:%S')} "
            f"[{record.levelname:<7}] {record.name}: {record.getMessage()}"
        )
        ctx = _log_context.get()
        if ctx:
            base += "  " + " ".join(f"{k}={v}" for k, v in ctx.items())
        extras = {
            k: v for k, v in record.__dict__.items()
            if k not in _RESERVED and not k.startswith("_")
        }
        if extras:
            base += "  " + " ".join(f"{k}={v}" for k, v in extras.items())
        if record.exc_info:
            base += "\n" + "".join(traceback.format_exception(*record.exc_info))
        return base


#: 输出格式的环境变量开关。
LOG_FORMAT_ENV = "LOG_FORMAT"

#: 日志级别开关。
LOG_LEVEL_ENV = "LOG_LEVEL"

#: 被降噪的第三方 logger（它们的 DEBUG/INFO 对本项目无价值，且量极大）。
_NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "openai",
    "urllib3",
    "sentence_transformers",
    "jieba",
    "neo4j",
    "qdrant_client",
    "asyncio",
    "multipart",
    "python_multipart",
)


def setup_logging(force_json: bool | None = None) -> None:
    """配置全项目日志。**幂等**，重复调用不叠加 handler。

    `force_json`：
        None  按环境自动判断（见下）
        True  强制 JSON
        False 强制人类可读

    ★ 默认格式怎么选
    -------------
        LOG_FORMAT 环境变量显式指定   → 用它（json / text）
        否则：stderr 是终端 → text（本地开发，人能读）
              否则           → json（容器/重定向到文件，机器要解析）

    这个判断比"一律 JSON"实用：本地 `python main.py api` 看到的是彩色可读日志，
    而 `docker logs` / 重定向到文件时自动变结构化。
    """
    if force_json is None:
        env = os.getenv(LOG_FORMAT_ENV, "").strip().lower()
        if env in ("json", "text"):
            use_json = env == "json"
        else:
            # stderr 是终端 → 人在看 → 可读格式；否则视为被重定向/容器收集 → JSON
            use_json = not sys.stderr.isatty()
    else:
        use_json = bool(force_json)

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(StructuredFormatter() if use_json else HumanFormatter())

    level_name = os.getenv(LOG_LEVEL_ENV, "INFO").strip().upper()
    level = getattr(logging, level_name, logging.INFO)
    if not isinstance(level, int):
        level = logging.INFO

    root = logging.getLogger()
    # ★ 先清空已有 handler 再装：`main.py` 与各 eval 脚本都会调 setup_logging，
    #   不清会**同一条日志打印多次**（每次调用叠一个 handler）。
    #   这是原来就存在的隐患（`basicConfig(force=True)` 只处理 root 的默认
    #   handler，管不到别人已经加上的）。
    for old in list(root.handlers):
        root.removeHandler(old)
    root.addHandler(handler)
    # 内存缓冲：供 /api/logs 查询（不输出，只留存）
    root.addHandler(log_buffer)
    root.setLevel(level)

    # ★ 出口脱敏（局限 5，见 `src/log_redaction.py`）
    #
    # 挂在 **root logger 的 filter** 上（而不是各 formatter）：
    # 这样它在**所有 handler 之前**执行一次，三个出口
    # （HumanFormatter / StructuredFormatter / BufferHandler）都受益 ——
    # 尤其 BufferHandler：`/api/logs` 读的正是它的缓冲，
    # 只在 formatter 里脱敏会让缓冲里留原文。
    #
    # 幂等：先移除同类 filter 再装，避免重复调用 setup_logging 时叠加。
    from src.log_redaction import RedactionFilter

    # ⚠️ 必须挂到 **handler**，不能挂到 logger
    # ------------------------------------
    # `Logger.filter()` 只运行自己的 filters；`callHandlers()` 爬父链时
    # 只调用父 logger 的 **handlers**，**不调用它们的 filters**。
    # 故 `root.addFilter(...)` 对 `logging.getLogger("src.xxx").info(...)`
    # **完全不生效**（实测：缓冲里全是原文）。
    # 而 `Handler.handle()` 会先 `self.filter(record)` → 挂 handler 才对。
    redactor = RedactionFilter()
    for h in (handler, log_buffer):
        for f in list(h.filters):
            if isinstance(f, RedactionFilter):
                h.removeFilter(f)
        h.addFilter(redactor)

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """取 logger。

    ★ 为什么现在只做 `getLogger`
    -------------------------
    原实现给每个 logger **单独加一个 StreamHandler**。问题是：

        logger = logging.getLogger("src.rag.pipeline")
        logger.addHandler(h)      # 自己的 handler
        logger.propagate 仍为 True → 记录继续上传到 root
        root 上还有 setup_logging 装的 handler
        ⇒ **同一条日志输出两次**

    实测这个隐患真实存在：25 个文件用 `get_logger`、20 个用标准写法，
    两种入口混用，重复与否取决于调用顺序（`setup_logging` 在 import 之后跑，
    于是这些 logger 的 handler 与 root 的 handler 同时生效）。

    故这里去掉自建 handler：**只保留 root 一个出口**，
    格式/级别/上下文全部由 `setup_logging` 统一决定。
    保留本函数是为了兼容既有 25 个文件的 `get_logger(__name__)` 调用，
    **不需要逐个改文件**。
    """
    return logging.getLogger(name)


#: 内存中保留的最近日志条数。
#:
#: ★ 为什么需要内存缓冲（这是"结构化"最实用的一步）
# ---------------------------------------------
#: 结构化之后日志能被程序读了，但**读哪里**？默认只写到 stderr：
#:     · Docker 里要 `docker logs`，本地要翻重定向文件
#:     · 用户报"刚才问答失败了"，你得先找到那个时刻的日志
#:     · 前端没有入口，只能口头描述现象
#:
#: 环形缓冲把最近 N 条留在进程内，配一个 /api/logs 就能直接按
#: level / logger / request_id 过滤查询——排障从"翻文件"变成"发一个请求"。
#: 上限 500 条：够覆盖"刚才出了什么问题"，又不会吃内存。
LOG_BUFFER_SIZE = 500


class BufferHandler(logging.Handler):
    """把日志记录留在内存环形缓冲里（线程安全）。"""

    def __init__(self, capacity: int = LOG_BUFFER_SIZE) -> None:
        super().__init__()
        self._capacity = capacity
        self._items: list[dict[str, Any]] = []
        self._lock = __import__("threading").Lock()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            item = {
                "ts": StructuredFormatter._now(),
                "level": record.levelname,
                "logger": record.name,
                "msg": record.getMessage(),
                "at": f"{record.filename}:{record.lineno}",
            }
            item.update({k: _jsonable(v) for k, v in _log_context.get().items()})
            for key, value in record.__dict__.items():
                if key in _RESERVED or key.startswith("_"):
                    continue
                item.setdefault(key, _jsonable(value))
            exc = _format_exception(record)
            if exc:
                # 缓冲里只留异常类型与消息：堆栈占地方且看的时候再取
                item["exc"] = {"type": exc["type"], "msg": exc["msg"]}
            with self._lock:
                self._items.append(item)
                if len(self._items) > self._capacity:
                    # 丢最旧的：排障关心的是"刚刚"
                    del self._items[: len(self._items) - self._capacity]
        except Exception:  # noqa: BLE001
            # 缓冲是**附加能力**，坏了不能影响正常日志输出
            self.handleError(record)

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._items)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


#: 进程内日志缓冲单例（`setup_logging` 时装到 root 上）。
log_buffer = BufferHandler()

"""结构化日志测试。

★ 这一层的风险点与普通功能不同
--------------------------
日志代码**出问题的方式是"静默丢日志"**，而不是抛异常。原因是
`logging` 对 `Formatter.format()` 抛出的异常处理是
`handleError()` → 打印到 stderr 并**丢弃该条记录**。

后果：恰恰在出问题（字段不可序列化、上下文异常）时，日志消失。
故本文件重点是**异常路径**，不是"格式好不好看"。
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.logging_config import (  # noqa: E402
    LOG_BUFFER_SIZE,
    MAX_FIELD_CHARS,
    BufferHandler,
    HumanFormatter,
    StructuredFormatter,
    bind_log_context,
    get_log_context,
    get_logger,
    log_buffer,
    reset_log_context,
    setup_logging,
)


@pytest.fixture
def json_caplog(caplog):
    """把 caplog 配成结构化格式，返回可解析的记录列表。"""
    caplog.handler.setFormatter(StructuredFormatter())
    return caplog


def _record(msg: str = "测试", level: int = logging.INFO, **extra) -> logging.LogRecord:
    rec = logging.LogRecord(
        name="src.test", level=level, pathname="t.py", lineno=1,
        msg=msg, args=(), exc_info=None,
    )
    for k, v in extra.items():
        setattr(rec, k, v)
    return rec


def _parse(fmt: StructuredFormatter, rec: logging.LogRecord) -> dict:
    return json.loads(fmt.format(rec))


class TestStructuredFormat:
    def test_required_fields(self):
        d = _parse(StructuredFormatter(), _record())
        assert d["level"] == "INFO"
        assert d["logger"] == "src.test"
        assert d["msg"] == "测试"
        assert "ts" in d and "at" in d

    def test_timestamp_is_local_with_offset(self):
        """★ 时间用本地时区带偏移，而不是 UTC。

        这个项目是单机部署、排障时人对着墙上的钟看日志；
        纯 UTC 会让人每次心算 +8 小时。带偏移量既能直接读，
        机器解析也不丢时区。
        """
        d = _parse(StructuredFormatter(), _record())
        ts = d["ts"]
        assert ("+" in ts or "-" in ts[10:]), f"时间戳缺时区偏移: {ts}"

    def test_output_is_single_line(self):
        """一条日志一行——多行会破坏 JSONL 的可解析性。"""
        out = StructuredFormatter().format(_record("含\n换行的消息"))
        assert "\n" not in out

    def test_chinese_not_escaped(self):
        """`ensure_ascii=False`：中文不转成 \\uXXXX，可读性与体积都更好。"""
        out = StructuredFormatter().format(_record("投标保证金"))
        assert "投标保证金" in out
        assert "\\u" not in out

    def test_extra_fields_included(self):
        d = _parse(StructuredFormatter(), _record(tool="search_web", elapsed_ms=431))
        assert d["tool"] == "search_web"
        assert d["elapsed_ms"] == 431

    def test_reserved_names_not_leaked_as_fields(self):
        """LogRecord 自带的属性不该被当成业务字段输出。"""
        d = _parse(StructuredFormatter(), _record())
        for leaked in ("pathname", "lineno", "args", "levelno", "process"):
            assert leaked not in d, f"{leaked} 不该出现在输出里"


class TestExceptionFormatting:
    def test_exception_is_structured(self):
        """★ 异常要结构化成对象，而不是混进 message。

        原来的 `%(exc_info)s` 会把整块堆栈塞进 message：
            · 解析出来 message 里混着换行与缩进
            · 无法按异常类型聚合（"最近有多少次 TimeoutError"）
        """
        try:
            raise ValueError("演示异常")
        except ValueError:
            rec = logging.LogRecord(
                name="src.test", level=logging.ERROR, pathname="t.py", lineno=1,
                msg="失败", args=(), exc_info=sys.exc_info(),
            )
        d = _parse(StructuredFormatter(), rec)
        assert d["exc"]["type"] == "ValueError"
        assert d["exc"]["msg"] == "演示异常"
        assert "ValueError" in d["exc"]["stack"]
        # message 本身要保持干净
        assert "\n" not in d["msg"]

    def test_no_exc_key_without_exception(self):
        d = _parse(StructuredFormatter(), _record())
        assert "exc" not in d


class TestNeverLosesLogs:
    """★ 核心风险：日志代码不能"静默丢日志"。"""

    def test_unserializable_object_degrades(self):
        d = _parse(StructuredFormatter(), _record(obj=object()))
        assert isinstance(d["obj"], str)

    def test_circular_reference_does_not_hang(self):
        a: dict = {}
        a["self"] = a
        d = _parse(StructuredFormatter(), _record(data=a))
        assert isinstance(d["data"], (dict, str))

    def test_nan_and_infinity_become_strings(self):
        """NaN / Infinity 不是合法 JSON，`json.dumps` 默认会写出裸的 NaN。

        那会让下游（严格 JSON 解析器）解析失败——一条日志毒化整个采集管道。
        """
        d = _parse(StructuredFormatter(), _record(a=float("nan"), b=float("inf")))
        assert d["a"] == "nan"
        assert d["b"] == "inf"
        # 且整体仍是合法 JSON
        assert isinstance(json.loads(json.dumps(d)), dict)

    def test_long_string_truncated(self):
        d = _parse(StructuredFormatter(), _record(big="x" * (MAX_FIELD_CHARS * 2)))
        assert len(d["big"]) < MAX_FIELD_CHARS * 2
        assert "截断" in d["big"]

    def test_many_keys_truncated(self):
        big = {f"k{i}": i for i in range(200)}
        d = _parse(StructuredFormatter(), _record(data=big))
        assert len(d["data"]) <= 41  # 40 + _truncated

    def test_broken_repr_does_not_raise(self):
        class Bad:
            def __repr__(self):
                raise RuntimeError("repr 坏了")

        d = _parse(StructuredFormatter(), _record(obj=Bad()))
        assert "Bad" in d["obj"]

    def test_formatter_fallback_on_internal_error(self):
        """格式化器内部出错时必须**退回可读文本**，而不是让 logging 丢弃该条。"""

        class Exploding(StructuredFormatter):
            def _format(self, record):
                raise RuntimeError("故意炸")

        out = Exploding().format(_record("保命"))
        assert out  # 非空
        assert "降级" in out or "保命" in out


class TestLogContext:
    def test_bind_and_reset(self):
        assert get_log_context() == {}
        tok = bind_log_context(request_id="r1")
        assert get_log_context()["request_id"] == "r1"
        reset_log_context(tok)
        assert get_log_context() == {}

    def test_context_appears_in_output(self):
        tok = bind_log_context(request_id="r2", session_id="s2")
        try:
            d = _parse(StructuredFormatter(), _record())
        finally:
            reset_log_context(tok)
        assert d["request_id"] == "r2"
        assert d["session_id"] == "s2"

    def test_nested_binding_restores_outer(self):
        """嵌套绑定（如子任务再加字段）后要能还原到外层，而不是清空。"""
        outer = bind_log_context(request_id="outer")
        inner = bind_log_context(tool="search_web")
        assert get_log_context() == {"request_id": "outer", "tool": "search_web"}
        reset_log_context(inner)
        assert get_log_context() == {"request_id": "outer"}
        reset_log_context(outer)
        assert get_log_context() == {}

    def test_none_values_ignored(self):
        tok = bind_log_context(request_id="r3", session_id=None)
        try:
            assert "session_id" not in get_log_context()
        finally:
            reset_log_context(tok)

    def test_reset_out_of_order_does_not_raise(self):
        """token 乱序重置（异步场景可能发生）时不能抛。

        宁可丢掉上下文，也不要留下**错误的** request_id。
        """
        tok = bind_log_context(request_id="r4")
        reset_log_context(tok)
        reset_log_context(tok)  # 再重置一次
        assert get_log_context() == {}

    def test_context_is_isolated_per_thread(self):
        """★ 线程隔离：工具层是 4 线程并行，串味会导致日志归属错误。"""
        import threading

        results: dict[str, object] = {}
        tok = bind_log_context(request_id="main-thread")

        def worker():
            results["child_sees"] = get_log_context().get("request_id")
            bind_log_context(request_id="child")
            results["child_after"] = get_log_context().get("request_id")

        t = threading.Thread(target=worker)
        t.start()
        t.join()
        results["main_after"] = get_log_context().get("request_id")
        reset_log_context(tok)

        # 子线程默认**看不到**父线程后绑的上下文（contextvars 不跨线程复制）
        assert results["child_sees"] is None
        assert results["child_after"] == "child"
        # 且子线程的绑定不影响主线程
        assert results["main_after"] == "main-thread"


class TestBufferHandler:
    def test_captures_records(self):
        buf = BufferHandler(capacity=10)
        buf.emit(_record("第一条"))
        buf.emit(_record("第二条"))
        items = buf.snapshot()
        assert [i["msg"] for i in items] == ["第一条", "第二条"]

    def test_ring_buffer_drops_oldest(self):
        """★ 丢最旧的：排障关心"刚刚发生了什么"。"""
        buf = BufferHandler(capacity=3)
        for i in range(5):
            buf.emit(_record(f"msg{i}"))
        msgs = [i["msg"] for i in buf.snapshot()]
        assert msgs == ["msg2", "msg3", "msg4"]

    def test_captures_context(self):
        buf = BufferHandler(capacity=5)
        tok = bind_log_context(request_id="buf-1")
        try:
            buf.emit(_record("带上下文"))
        finally:
            reset_log_context(tok)
        assert buf.snapshot()[0]["request_id"] == "buf-1"

    def test_clear(self):
        buf = BufferHandler(capacity=5)
        buf.emit(_record())
        buf.clear()
        assert buf.snapshot() == []

    def test_emit_never_raises(self):
        """缓冲是附加能力，坏了不能影响正常日志输出。"""
        class Bad:
            def __repr__(self):
                raise RuntimeError("炸")

        buf = BufferHandler(capacity=5)
        buf.emit(_record(obj=Bad()))  # 不应抛

    def test_exception_stored_compactly(self):
        """缓冲里只留异常类型与消息，不留整块堆栈（省内存）。"""
        buf = BufferHandler(capacity=5)
        try:
            raise ValueError("x")
        except ValueError:
            rec = logging.LogRecord(
                name="t", level=logging.ERROR, pathname="t.py", lineno=1,
                msg="err", args=(), exc_info=sys.exc_info(),
            )
        buf.emit(rec)
        exc = buf.snapshot()[0]["exc"]
        assert exc == {"type": "ValueError", "msg": "x"}

    def test_snapshot_is_a_copy(self):
        buf = BufferHandler(capacity=5)
        buf.emit(_record())
        buf.snapshot().clear()
        assert len(buf.snapshot()) == 1


class TestSetupLogging:
    def test_idempotent_no_duplicate_handlers(self):
        """★ 重复调用不能叠加 handler —— 否则同一条日志打印多次。

        `main.py` 与各 eval 脚本都会调 `setup_logging`，这个隐患真实存在。
        """
        setup_logging(force_json=True)
        n1 = len(logging.getLogger().handlers)
        setup_logging(force_json=True)
        n2 = len(logging.getLogger().handlers)
        assert n1 == n2, f"handler 数从 {n1} 变成 {n2}（重复叠加）"

    def test_no_duplicate_output(self, capsys):
        """同一条日志只应输出一次。"""
        setup_logging(force_json=False)
        logging.getLogger("src.dup-test").info("只应出现一次")
        err = capsys.readouterr().err
        assert err.count("只应出现一次") == 1, f"实际出现 {err.count('只应出现一次')} 次"

    def test_json_mode_emits_parseable_json(self, capsys):
        setup_logging(force_json=True)
        logging.getLogger("src.json-test").info("可解析", extra={"n": 1})
        line = [ln for ln in capsys.readouterr().err.splitlines() if "可解析" in ln][-1]
        d = json.loads(line)
        assert d["n"] == 1

    def test_human_mode_is_readable(self, capsys):
        setup_logging(force_json=False)
        logging.getLogger("src.human-test").info("给人看的")
        err = capsys.readouterr().err
        assert "给人看的" in err
        assert "src.human-test" in err

    def test_returns_something_installable_twice(self):
        setup_logging(force_json=True)
        setup_logging(force_json=True)
        assert log_buffer in logging.getLogger().handlers


class TestGetLogger:
    def test_get_logger_does_not_add_own_handler(self):
        """★ 关键契约：`get_logger` 不能再自己加 handler。

        原实现给每个 logger 单独加 StreamHandler，而 `propagate` 仍为 True，
        于是同一条日志被自己的 handler 与 root 的 handler **各输出一次**。
        25 个文件用 `get_logger`、20 个用标准写法，两种入口混用。
        """
        logger = get_logger("src.no-own-handler")
        assert logger.handlers == [], f"get_logger 不该加 handler，实际 {logger.handlers}"

    def test_get_logger_returns_standard_logger(self):
        assert get_logger("src.x") is logging.getLogger("src.x")

    def test_noisy_loggers_are_quieted(self):
        setup_logging(force_json=True)
        assert logging.getLogger("httpx").level == logging.WARNING
        assert logging.getLogger("jieba").level == logging.WARNING


class TestBufferSize:
    def test_default_capacity_documented(self):
        assert LOG_BUFFER_SIZE == 500

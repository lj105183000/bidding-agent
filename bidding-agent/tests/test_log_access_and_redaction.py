"""`/api/logs` 访问控制 + 日志脱敏的回归测试。

对应两条局限：
    1. `/api/logs` 无鉴权 → **已加** `require_logs_access`
    5. 未做敏感信息脱敏 → **已加** `RedactionFilter`
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.log_redaction import (  # noqa: E402
    MAX_MESSAGE_CHARS,
    MASK,
    RedactionFilter,
    redact,
    redact_value,
)


# ===========================================================================
# 脱敏：凭据类
# ===========================================================================


class TestCredentialRedaction:
    @pytest.mark.parametrize(
        "text",
        [
            "Authorization: Bearer sk-abcdefghijklmnopqrstuvwxyz",
            "api_key=abc123def456ghi789",
            "apiKey=abc123def456ghi789",
            "token=xyzsecret123",
            "password=hunter2hunter2",
            "sk-proj-abcdefghijklmnopqrstuvwxyz123456",
            "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abcdefghijklmnop",
        ],
    )
    def test_secret_values_removed(self, text):
        out = redact(text)
        # 原文里的密钥主体不应再出现
        for secret in ("abcdefghijklmnopqrstuvwxyz", "abc123def456ghi789",
                       "xyzsecret123", "hunter2hunter2"):
            assert secret not in out, f"密钥泄漏：{out}"
        assert MASK in out


class TestPiiRedaction:
    def test_phone_keeps_head_and_tail(self):
        out = redact("手机 13812345678 请联系")
        assert "13812345678" not in out
        assert "138" in out and "5678" in out, "应保留头尾便于排查"

    def test_id_card(self):
        out = redact("身份证 110101199001011234")
        assert "110101199001011234" not in out
        assert "110101" in out

    def test_email_keeps_domain(self):
        out = redact("邮箱 zhangsan@example.com")
        assert "zhangsan" not in out
        assert "@example.com" in out, "域名应保留（判断是哪家）"


class TestUrlQueryRedaction:
    def test_query_value_masked(self):
        out = redact("GET /api/search?q=某某公司中标价&page=2")
        assert "某某公司中标价" not in out
        assert "page=2" in out, "非敏感参数不该被改"
        assert "q=" in out


class TestNoFalsePositives:
    @pytest.mark.parametrize(
        "text",
        [
            "正常日志：检索完成，命中 5 条",
            "阶段耗时 首轮分析=1800ms",
            "工具 search_knowledge_base 返回 0 条",
            "REPEAT=3 次取中位",
        ],
    )
    def test_normal_lines_unchanged(self, text):
        assert redact(text) == text, f"正常日志被误改：{redact(text)!r}"


class TestTruncation:
    def test_long_message_truncated(self):
        out = redact("x" * (MAX_MESSAGE_CHARS + 500))
        assert len(out) < MAX_MESSAGE_CHARS + 100
        assert "已截断" in out

    def test_short_message_not_truncated(self):
        assert "已截断" not in redact("短消息")


class TestRobustness:
    def test_empty_and_none_safe(self):
        assert redact("") == ""

    def test_redact_value_containers(self):
        got = redact_value({"a": "token=abc123def456", "n": 5, "l": ["api_key=zzzzzzzzzzzz"]})
        assert "abc123def456" not in str(got)
        assert got["n"] == 5


# ===========================================================================
# LogRecord 层：三个出口都覆盖
# ===========================================================================


class TestRedactionFilter:
    def test_args_are_redacted_and_cleared(self):
        """★ 关键：`logger.info("%s", secret)` 这种写法也必须脱敏。

        只改 `msg` 不动 `args` 的话，`getMessage()` 会用原 args 重新拼 → 泄漏。
        """
        rec = logging.LogRecord(
            "t", logging.INFO, "f.py", 1, "调用 %s 检索", ("api_key=abc123def456ghi789",), None
        )
        RedactionFilter().filter(rec)
        assert "abc123def456ghi789" not in rec.getMessage()
        assert rec.args in ((), None)

    def test_extra_fields_are_redacted(self):
        rec = logging.LogRecord("t", logging.INFO, "f.py", 1, "x", (), None)
        rec.__dict__["url"] = "/api?q=某某公司中标价"
        RedactionFilter().filter(rec)
        assert "某某公司中标价" not in rec.__dict__["url"]

    def test_idempotent(self):
        """同一 record 经过多个 handler 的 filter 时不能重复脱敏。"""
        rec = logging.LogRecord("t", logging.INFO, "f.py", 1, "手机 13812345678", (), None)
        f = RedactionFilter()
        f.filter(rec)
        once = rec.getMessage()
        f.filter(rec)
        assert rec.getMessage() == once, "重复脱敏导致内容变化"

    def test_never_raises(self):
        class Bad(logging.LogRecord):
            def getMessage(self):  # 故意抛
                raise RuntimeError("boom")

        rec = Bad("t", logging.INFO, "f.py", 1, "x", ("y",), None)
        assert RedactionFilter().filter(rec) is True   # 不抛，且返回 True 保留日志


class TestFilterWiredToHandlers:
    def test_buffer_contains_redacted_text(self):
        """★ 端到端：`/api/logs` 读的缓冲里必须是脱敏后的内容。

        这是最初挂错位置（root logger）时**唯一能发现**的测试 ——
        挂 root 时 formatter 可能看起来对，但缓冲里是原文。
        """
        from src.logging_config import log_buffer, setup_logging

        setup_logging(force_json=False)
        log_buffer.clear()
        logging.getLogger("probe_redact").warning("api_key=abc123def456ghi789 请求失败")

        items = log_buffer.snapshot()
        assert items, "缓冲应有记录（否则测试无效）"
        blob = " ".join(i["msg"] for i in items)
        assert "abc123def456ghi789" not in blob, "缓冲里残留了明文密钥"
        assert MASK in blob

    def test_setup_logging_idempotent_filter(self):
        """重复 setup 不应叠加 filter（否则每条日志被多次处理）。"""
        from src.logging_config import log_buffer, setup_logging
        from src.log_redaction import RedactionFilter as RF

        setup_logging(force_json=False)
        setup_logging(force_json=False)
        n = sum(1 for f in log_buffer.filters if isinstance(f, RF))
        assert n == 1, f"filter 叠加了 {n} 个"

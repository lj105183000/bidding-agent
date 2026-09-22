"""工具层基座测试：结果契约 / 参数校验 / 并行执行 / 超时与异常隔离 / 导入顺序守卫。

mock 约定（开发文档 §9.3）：fake 定义在文件内，不建全局 fixture；
全部同步测试，不依赖真实数据源。执行器按冻结契约写成 `executor(arguments, question="")`。
"""

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from pydantic import Field

from src.tools.base import BaseTool, ToolResult, ToolRunner

ROOT = Path(__file__).resolve().parents[1]


# --- fake 工具（BaseTool 路径：用于元信息/参数校验/兜底） ---

class _EchoTool(BaseTool):
    """回显工具：把收到的参数原样放进 data，用于断言参数传递。"""

    name: str = "echo"
    description: str = "回显参数"
    parameters: dict = {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "top_k": {"type": "integer"},
            "mode": {"type": "string", "enum": ["fast", "deep"]},
        },
        "required": ["text"],
    }

    def run(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, data=kwargs)


class _BoomTool(BaseTool):
    """总是抛异常，且异常文本里带内部地址与密钥（用于验证脱敏）。"""

    name: str = "boom"
    description: str = "抛异常"

    def run(self, **kwargs) -> ToolResult:
        raise RuntimeError("连接 http://10.0.0.5:8001 失败，password=secret123")


# --- fake 执行器（ToolRunner 路径：冻结契约的可调用对象） ---

def _echo_executor(arguments, question=""):
    """最简执行器：把参数回显成格式化文本，sources 为空。"""
    return f"回显 {arguments}", []


def _web_executor(arguments, question=""):
    """联网类执行器：返回摘要 + 来源列表。"""
    return "联网摘要", [{"title": "钢材行情", "url": "http://example.com/a"}]


def _sleep_executor(arguments, question=""):
    """睡指定秒数，供并行/超时两条路径共用。"""
    time.sleep(float(arguments.get("seconds", 0)))
    return f"睡了 {arguments.get('seconds')}", []


def _boom_executor(arguments, question=""):
    """抛异常，且异常文本里带内部地址与密钥（用于验证脱敏）。"""
    raise RuntimeError("连接 http://10.0.0.5:8001 失败，password=secret123")


# --- ToolResult 契约 ---

def test_to_dict_uses_data_key_by_default():
    result = ToolResult(success=True, data=[{"q": 1}])
    assert result.to_dict() == {"success": True, "data": [{"q": 1}], "error": None}


def test_to_dict_failure_keeps_all_three_keys():
    result = ToolResult.failure("工具执行失败，已跳过该数据源。")
    assert result.success is False
    assert result.data is None
    assert result.to_dict()["success"] is False
    assert result.to_dict()["error"] == "工具执行失败，已跳过该数据源。"


def test_web_tool_result_lands_under_results_key():
    """联网类工具（WEB_TOOL_NAMES）的结果键必须是 results，前端与格式化器按此读取。"""
    results = ToolRunner().execute_parallel(
        [{"name": "search_web", "arguments": {"query": "钢材"}}],
        "钢材价格",
        {"search_web": _web_executor},
    )
    payload = results[0].to_dict()
    assert results[0].payload_key == "results"
    assert "results" in payload and "data" not in payload
    assert payload["success"] is True
    assert payload["results"]["sources"][0]["url"] == "http://example.com/a"


def test_non_web_tool_result_lands_under_data_key():
    results = ToolRunner().execute_parallel(
        [{"name": "query_database", "arguments": {"text": "x"}}],
        "问",
        {"query_database": _echo_executor},
    )
    assert results[0].payload_key == "data"
    assert results[0].to_dict()["data"]["text"] == "回显 {'text': 'x'}"


# --- 参数校验（BaseTool 路径） ---

def test_missing_required_argument_is_rejected():
    tool = _EchoTool()
    assert tool.validate_arguments({}) == "缺少必填参数：text。"
    assert tool.safe_run(top_k=3).success is False


def test_enum_argument_outside_whitelist_is_rejected():
    tool = _EchoTool()
    message = tool.validate_arguments({"text": "x", "mode": "turbo"})
    assert message is not None
    assert "参数 mode 取值须为 fast / deep" in message


def test_wrong_argument_type_is_rejected():
    tool = _EchoTool()
    message = tool.validate_arguments({"text": "x", "top_k": {"a": 1}})
    assert message == "参数 top_k 类型不正确。"


def test_arguments_must_be_object():
    tool = _EchoTool()
    assert tool.validate_arguments("text=x") == "参数格式错误：应为 JSON 对象。"


def test_undeclared_extra_argument_is_tolerated():
    """Runner 会注入 question；未声明的多余参数不应把正常调用判成失败。"""
    tool = _EchoTool()
    assert tool.validate_arguments({"text": "x", "question": "用户问题"}) is None


def test_malformed_property_spec_is_tolerated():
    """schema 里某个字段描述写错时宁可放行，也不让一个笔误把工具变成永远失败。"""
    tool = _EchoTool(parameters={"type": "object", "properties": {"text": "不是字典"}})
    assert tool.validate_arguments({"text": "x"}) is None


# --- 异常脱敏 ---

def test_safe_run_catches_exception_and_hides_internals():
    result = _BoomTool().safe_run()
    assert result.success is False
    dumped = json.dumps(result.to_dict(), ensure_ascii=False)
    # 内部地址、密钥、原始异常文本一个都不能出现在用户可见结果里
    assert "10.0.0.5" not in dumped
    assert "secret123" not in dumped
    assert "RuntimeError" not in dumped
    assert result.error == "工具执行失败，已跳过该数据源，请结合其他信息回答。"


def test_safe_run_wraps_non_toolresult_return():
    class _BareTool(BaseTool):
        name: str = "bare"
        description: str = "返回裸值"

        def run(self, **kwargs):
            return "裸文本"

    result = _BareTool().safe_run()
    assert result.success is True and result.data == "裸文本"


def test_executor_exception_is_hidden_from_result():
    """执行器抛异常时的脱敏：地址与密钥只进日志。"""
    results = ToolRunner().execute_parallel(
        [{"name": "search_knowledge_graph", "arguments": {}}],
        "问",
        {"search_knowledge_graph": _boom_executor},
    )
    dumped = json.dumps(results[0].to_dict(), ensure_ascii=False)
    assert results[0].success is False
    assert "10.0.0.5" not in dumped and "secret123" not in dumped
    assert results[0].error == "工具执行失败，已跳过该数据源，请结合其他信息回答。"


# --- 执行器返回值归一化 ---

def test_tuple_executor_result_carries_text_and_sources():
    results = ToolRunner().execute_parallel(
        [{"name": "search_knowledge_base", "arguments": {"q": "钢材"}}],
        "问",
        {"search_knowledge_base": lambda arguments, question="": ("资料 1", [{"question": "q"}])},
    )
    assert results[0].success is True
    assert results[0].data == {"text": "资料 1", "sources": [{"question": "q"}]}


def test_sources_not_a_list_degrades_to_empty():
    results = ToolRunner().execute_parallel(
        [{"name": "query_database", "arguments": {}}],
        "问",
        {"query_database": lambda arguments, question="": ("文本", "不是列表")},
    )
    assert results[0].data["sources"] == []


def test_toolresult_returning_executor_passes_through():
    """用 BaseTool 定义的工具包出来的执行器，直接返回 ToolResult 也能注册。"""
    results = ToolRunner().execute_parallel(
        [{"name": "query_database", "arguments": {}}],
        "问",
        {"query_database": lambda arguments, question="": ToolResult(success=True, data="结构化结果")},
    )
    assert results[0].data == "结构化结果"


def test_bare_value_degrades_to_text():
    """非契约形状按 react_loop 的先例降级为纯文本，不丢弃内容。"""
    results = ToolRunner().execute_parallel(
        [{"name": "query_database", "arguments": {}}],
        "问",
        {"query_database": lambda arguments, question="": "裸文本"},
    )
    assert results[0].success is True
    assert results[0].data == {"text": "裸文本", "sources": []}


# --- 并行执行 ---

def test_unknown_tool_returns_structured_failure():
    results = ToolRunner().execute_parallel([{"name": "nope", "arguments": {}}], "问", {})
    assert results[0].success is False
    assert "暂不可用" in results[0].error


def test_non_callable_executor_is_rejected_explicitly():
    """executors 契约是工具名 → 可调用对象；传 BaseTool 实例要给出明确失败而不是崩掉。"""
    results = ToolRunner().execute_parallel(
        [{"name": "echo", "arguments": {}}], "问", {"echo": _EchoTool()}
    )
    assert results[0].success is False
    assert "未按契约注册" in results[0].error


def test_run_one_directly_returns_failure_for_unknown_tool():
    """_run_one 是并行执行的单次实现，单独调用也必须不抛异常。"""
    result = ToolRunner()._run_one({"name": "nope", "arguments": {}}, "问", {})
    assert result.success is False


def test_tool_outside_manifest_logs_warning(caplog):
    """constants 的工具清单是 SYSTEM_PROMPT 与来源分流共用的 manifest，漏维护要有线索。"""
    with caplog.at_level(logging.WARNING):
        results = ToolRunner().execute_parallel(
            [{"name": "custom_tool", "arguments": {}}], "问", {"custom_tool": _echo_executor}
        )
    assert results[0].success is True  # 只是提醒，不拦执行
    assert "不在 constants 的工具清单中" in caplog.text


def test_json_string_arguments_are_parsed():
    """LLM 常把 arguments 作为 JSON 字符串给出（文档 §13 的解析步骤）。"""
    results = ToolRunner().execute_parallel(
        [{"name": "query_database", "arguments": '{"text": "钢材"}'}], "问", {"query_database": _echo_executor}
    )
    assert results[0].success is True
    assert "钢材" in results[0].data["text"]


def test_malformed_json_arguments_returns_failure():
    results = ToolRunner().execute_parallel(
        [{"name": "query_database", "arguments": "{不是 JSON"}], "问", {"query_database": _echo_executor}
    )
    assert results[0].success is False
    assert "参数解析失败" in results[0].error


def test_missing_tool_name_returns_failure():
    results = ToolRunner().execute_parallel([{"arguments": {}}], "问", {})
    assert results[0].success is False
    assert "缺少工具名" in results[0].error


def test_empty_calls_returns_empty_list():
    assert ToolRunner().execute_parallel([], "问", {}) == []


def test_results_follow_input_order_and_run_in_parallel():
    """结果顺序必须与 tool_calls 一致（回填消息按序拼接），且确实是并行执行。"""
    calls = [
        {"name": "search_knowledge_base", "arguments": {"seconds": 0.5}},
        {"name": "query_database", "arguments": {"seconds": 0.5}},
    ]
    executors = {"search_knowledge_base": _sleep_executor, "query_database": _sleep_executor}
    started = time.monotonic()
    results = ToolRunner(timeout=2).execute_parallel(calls, "问", executors)
    elapsed = time.monotonic() - started

    assert [r.success for r in results] == [True, True]
    assert [r.data["text"] for r in results] == ["睡了 0.5", "睡了 0.5"]
    # 两个睡 0.5s 的工具串行需 ~1.0s，并行只需 ~0.5s；留余量避免 CI 抖动误报
    assert elapsed < 0.85, f"未并行执行，耗时 {elapsed:.2f}s"


def test_timeout_is_isolated_per_tool():
    """一个工具超时不影响同轮其他工具，且超时文案为固定措辞。"""
    results = ToolRunner(timeout=0.2).execute_parallel(
        [
            {"name": "search_knowledge_base", "arguments": {"seconds": 1.0}},
            {"name": "query_database", "arguments": {"text": "我很快"}},
        ],
        "问",
        {"search_knowledge_base": _sleep_executor, "query_database": _echo_executor},
    )
    assert results[0].success is False
    assert results[0].error == "工具执行超时，请稍后重试或缩小查询范围。"
    assert results[1].success is True


def test_exception_in_one_tool_does_not_break_others():
    results = ToolRunner().execute_parallel(
        [
            {"name": "search_knowledge_graph", "arguments": {}},
            {"name": "query_database", "arguments": {"text": "正常"}},
        ],
        "问",
        {"search_knowledge_graph": _boom_executor, "query_database": _echo_executor},
    )
    assert results[0].success is False
    assert results[1].success is True


def test_question_injected_only_when_declared():
    seen = []

    def with_question(arguments, question=""):
        seen.append(question)
        return "带 question", []

    def without_question(arguments):
        return "不带 question", []

    results = ToolRunner().execute_parallel(
        [
            {"name": "search_web", "arguments": {}},
            {"name": "query_database", "arguments": {}},
        ],
        "钢材供应关系",
        {"search_web": with_question, "query_database": without_question},
    )
    assert seen == ["钢材供应关系"]
    # 只收 arguments 的执行器（react_loop 的调用方式）不能被注入搞崩
    assert results[1].success is True


def test_run_parallel_is_alias_of_execute_parallel():
    """文档 §13 写作 run_parallel，别名指向同一实现，两种叫法都可用。"""
    assert ToolRunner.run_parallel is ToolRunner.execute_parallel
    results = ToolRunner().run_parallel(
        [{"name": "query_database", "arguments": {"text": "别名"}}], "问", {"query_database": _echo_executor}
    )
    assert results[0].success is True


# --- 导入顺序守卫 ---

def test_importing_base_does_not_pull_agent_package():
    """守住模块 docstring 里的循环导入约束。

    base.py 顶层一旦 import src.agent.constants，就会经 src.agent.__init__ → core
    →（try）rag_tools → 回到部分初始化的 src.tools.base，使 rag_tools 的
    `from src.tools.base import BaseTool` 抛 ImportError 并被 core 静默吞掉——
    工具层会悄悄变空且只在特定导入顺序下复现。故在干净子进程里断言。
    """
    code = (
        "import sys\n"
        "import src.tools.base\n"
        "leaked = [m for m in ('src.agent', 'src.agent.core', 'src.tools.rag_tools') if m in sys.modules]\n"
        "assert not leaked, 'base 顶层引入了 ' + repr(leaked)\n"
    )
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr

"""工具层基座：ToolResult / BaseTool / ToolRunner（成员 C）。

契约与设计理由
--------------
- **结果契约**（分配说明.md §5）：所有工具统一返回 `{"success", "data"/"results", "error"}`，
  失败不抛异常——Agent 不因后端宕机崩溃。`ToolResult.to_dict()` 是这条契约的唯一出口：
  联网类工具的结果放 `"results"` 键下，其余放 `"data"`。
- **执行器契约**（docs/组件工作机制.md §13）：`executor(arguments, question="")` →
  `(格式化文本, sources 列表)`。`question` 必须可缺省——已合并的 `src/agent/react_loop.py`
  只传一个 arguments；ToolRunner 按签名探测决定是否注入，两种写法都能跑。
- **为什么用 Pydantic 模型驱动**：结果对象要在工具、格式化器、SSE 帧、评测四处流转，
  模型给出唯一定义 + 类型校验 + 一键 to_dict；参数校验复用同一套机制
  （`BaseTool.parameters` 是 JSON Schema 子集，编译成 Pydantic 模型后校验）。
- **脱敏**（分配说明.md §5 安全红线）：完整异常（服务地址、SQL、密钥）只进日志；
  用户可见文案只用固定措辞，绝不回声原始异常文本。
- **与 react_loop 的关系**：已合并的 `src/agent/react_loop.py` 自己实现「解析 → 执行 → 回填」，
  直接调用 `TOOL_EXECUTORS` 里的执行器；本模块的 `ToolRunner` 是**同一批执行器**的并行执行
  入口（4 线程），供 API 层与二期联调复用。两条路径互不影响，本模块不改动 react_loop。

⚠️ 循环导入（改本文件前必读）
----------------------------
本模块**不能在顶层** `from src.agent.constants import ...`：那会触发
`src/agent/__init__.py` → `core` →（`try:`）`src.tools.rag_tools` → `src.tools.base`，
而此刻 base 只执行到 import 行（部分初始化状态），rag_tools 里的
`from src.tools.base import BaseTool` 会抛 ImportError，又被 core 的
`except ImportError` 静默吞掉——**结果是整个工具层悄悄变空**，且是否翻车取决于
"谁先被导入"，极难排查。因此 `TOOL_TIMEOUT_SECONDS` / `BASE_TOOL_NAMES` /
`WEB_TOOL_NAMES` 一律在**函数内延迟导入**（执行到时导入状态已稳定）。
tests/test_tools_base.py 用子进程断言守着这条约束。
"""

import inspect
import json
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, create_model

logger = logging.getLogger(__name__)

# 并行度：与 docs/组件工作机制.md §13 的 ThreadPoolExecutor(4) 对齐。
# 只服务本模块，故就地定义而非进 src/agent/constants.py（那边是跨模块共享常量的家）。
PARALLEL_MAX_WORKERS = 4

# 用户可见兜底文案：与 react_loop._execute_tool 保持同一措辞——
# 前端徽标与评测脚本若按文案断言，不会因为走了不同执行路径而漂移。
_FAILURE_TEXT = "工具执行失败，已跳过该数据源，请结合其他信息回答。"
_TIMEOUT_TEXT = "工具执行超时，请稍后重试或缩小查询范围。"

# JSON Schema 基础类型 → Python 注解；未声明的类型一律按 Any 放行
_JSON_TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


class ToolResult(BaseModel):
    """工具执行结果（契约对象，勿随意增删字段）。

    `data` 承载工具自己的负载；约定把来源列表也放进去（`{"text": ..., "sources": [...]}`），
    因为 SSE done 帧的 sources 需要它，而契约只定义了三个字段。
    """

    success: bool
    data: Any = None
    error: str | None = None
    # 输出键名：联网类工具用 "results"，其余用 "data"（由 ToolRunner._run_one 依工具名判定）；
    # 失败结果不带负载，保持默认 "data" —— 消费者按 success 分支，键名无关紧要
    payload_key: Literal["data", "results"] = "data"

    @classmethod
    def failure(cls, error: str) -> "ToolResult":
        """构造失败结果：成功标志与 error 文案成对出现，避免调用方写反。"""
        return cls(success=False, error=error)

    def to_dict(self) -> dict:
        """输出契约字典 `{"success", "data"/"results", "error"}`。

        三个键恒定存在（失败时对应负载为 None），格式化器只需按 success 分支，
        不必再判断键是否存在。
        """
        result = {"success": self.success}
        result[self.payload_key] = self.data
        result["error"] = self.error
        return result


class BaseTool(BaseModel, ABC):
    """工具基类：Pydantic 声明元信息 + 抽象 run + 兜底 safe_run。

    执行器**不强制**用它——ToolRunner 接受任意可调用对象。用 BaseTool 的好处是
    元信息（给 LLM 的 name/description/parameters）、参数白名单校验、异常兜底都在一处，
    由它包出来的执行器自然满足契约。
    """

    name: str
    description: str
    # JSON Schema 子集：type / required / enum 会被编译成校验模型（供 LLM 选择参数与本地拦截）
    parameters: dict = Field(default_factory=dict)

    @abstractmethod
    def run(self, **kwargs) -> ToolResult:
        """实际执行。实现应返回 ToolResult；抛出的异常由 safe_run 兜底。"""

    # --- 参数校验 ---

    def validate_arguments(self, arguments: Any) -> str | None:
        """校验参数，返回中文错误说明；通过则返回 None。

        只回字段名不回参值——参值可能含密钥，而这段文案会进 LLM 上下文与用户可见区。
        """
        if not isinstance(arguments, Mapping):
            return "参数格式错误：应为 JSON 对象。"
        model = self._params_model()
        if model is None:
            return None
        try:
            model(**arguments)
        except ValidationError as exc:
            return self._describe(exc)
        except TypeError:
            # 键不是字符串（JSON 反序列化正常不会产生，兜底不当成崩溃）
            return "参数格式错误：字段名必须为字符串。"
        return None

    def _describe(self, exc: ValidationError) -> str:
        """把 Pydantic 报错翻译成给 LLM 看的中文提示（LLM 据此自我修正后重试）。

        枚举取值从本地 schema 取，而不是解析 Pydantic 的 ctx["expected"]——
        后者是 `"'a' or 'b'"` 这类自然语言串，切分不可靠。
        """
        parts = []
        for item in exc.errors():
            field = str(item.get("loc", ("?",))[-1])
            kind = item.get("type", "")
            if kind == "missing":
                parts.append(f"缺少必填参数：{field}")
            elif kind in ("literal_error", "enum"):
                allowed = self._enum_values(field)
                parts.append(
                    f"参数 {field} 取值须为 {' / '.join(allowed)}" if allowed
                    else f"参数 {field} 取值不在允许范围内"
                )
            else:
                parts.append(f"参数 {field} 类型不正确")
        return "；".join(parts) + "。"

    def _enum_values(self, field: str) -> list[str]:
        """取 schema 中该字段的枚举白名单。取自本地 schema，回显无泄露风险。"""
        properties = self.parameters.get("properties") if isinstance(self.parameters, Mapping) else None
        spec = properties.get(field) if isinstance(properties, Mapping) else None
        enum = spec.get("enum") if isinstance(spec, Mapping) else None
        return [str(v) for v in enum] if isinstance(enum, (list, tuple)) else []

    def _params_model(self):
        """把 parameters 编译成 Pydantic 模型；schema 不可用则返回 None（降级为不校验）。

        只支持 type（含 required）/ enum 两类约束——它们覆盖本项目全部工具参数。
        编译失败说明 schema 写错了：记日志并放行，宁可漏拦一次调用，
        也不让一个笔误把整个工具变成永远失败。
        """
        spec = self.parameters
        properties = spec.get("properties") if isinstance(spec, Mapping) else None
        if not isinstance(properties, Mapping) or not properties:
            return None

        required = set(spec.get("required") or ())
        fields: dict[str, tuple] = {}
        for field_name, field_spec in properties.items():
            field_spec = field_spec if isinstance(field_spec, Mapping) else {}
            annotation = _JSON_TYPE_MAP.get(field_spec.get("type"), Any)
            enum = field_spec.get("enum")
            if isinstance(enum, (list, tuple)) and enum:
                # 枚举白名单：与文档「参数枚举白名单」要求一致，从 schema 动态构造
                annotation = Literal[tuple(enum)]
            fields[field_name] = (annotation, ...) if field_name in required else (annotation | None, None)

        try:
            return create_model(f"ToolParams_{self.name}", **fields)
        except Exception:
            logger.exception("工具 %s 的 parameters 无法编译为校验模型，本次跳过参数校验", self.name)
            return None

    # --- 执行入口 ---

    def safe_run(self, **kwargs) -> ToolResult:
        """参数校验 + 执行 + 异常兜底。本方法**保证不抛异常**（工具层对 Agent 的硬承诺）。

        校验只用于拦截明显错误的调用，通过后仍把原始参数交给 run——
        避免把未在 schema 里声明的参数（例如 Runner 注入的 question）悄悄丢掉。
        """
        params_error = self.validate_arguments(kwargs)
        if params_error is not None:
            logger.info("工具 %s 参数校验未通过：%s", self.name, params_error)
            return ToolResult.failure(params_error)
        try:
            result = self.run(**kwargs)
        except Exception:
            # 完整异常（含地址、SQL、密钥）只进日志；用户可见文案固定
            logger.exception("工具 %s 执行异常", self.name)
            return ToolResult.failure(_FAILURE_TEXT)
        if not isinstance(result, ToolResult):
            # 实现漏了契约：包一层而不是把裸值透传给下游格式化器
            logger.warning("工具 %s 未按契约返回 ToolResult，已自动包装", self.name)
            return ToolResult(success=True, data=result)
        return result


def _accepts_question(executor: Callable) -> bool:
    """探测执行器是否接收 question，决定 Runner 要不要注入。

    联网类工具需要用户问题做结果重排，检索类不需要。用签名探测而不是强制所有执行器
    都声明 question，是为了让 `def f(arguments)` 这种只收参数的写法也能直接注册
    （已合并的 react_loop 就是这么调的）。
    """
    try:
        params = inspect.signature(executor).parameters
    except (TypeError, ValueError):  # pragma: no cover - 内置/动态实现
        return False
    if "question" in params:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def _call_name(call: Any) -> str:
    """取工具名供日志使用；调用畸形时给占位符，避免日志代码自己抛异常。"""
    name = call.get("name") if isinstance(call, Mapping) else None
    return name if isinstance(name, str) and name else "<未知>"


def _as_tool_result(value: Any) -> ToolResult:
    """把执行器返回值归一化为 ToolResult。

    执行器会返回三种形态，都要认——工具层是 Agent 的最后一道防线，兼容优先于严格：
      1. `(格式化文本, sources)`：TOOL_EXECUTORS 的冻结契约，也是 react_loop 认识的形状
      2. `ToolResult`：BaseTool.safe_run 的返回，便于用 BaseTool 定义工具后直接注册
      3. 其他裸值：按 react_loop 的先例「降级为纯文本，不丢弃内容」处理

    元组路径恒为 `success=True`：冻结契约要求执行器用**错误文本**表达失败而不是抛异常，
    成败体现在文本里（前端与 SSE 只关心 tool_name 与 sources）。
    """
    if isinstance(value, ToolResult):
        return value
    if isinstance(value, tuple) and len(value) == 2:
        text, sources = value
        return ToolResult(
            success=True,
            data={"text": str(text), "sources": sources if isinstance(sources, list) else []},
        )
    logger.debug("执行器返回了非契约形状（%s），按纯文本降级处理", type(value).__name__)
    return ToolResult(success=True, data={"text": str(value), "sources": []})


class ToolRunner:
    """工具并行执行器：4 线程 / 单工具 30s 超时 / 异常隔离。

    为什么并行：一次决策可能同时点名多个数据源，串行时延是各自耗时之和，
    并行后只取决于最慢的一个。为什么按「每个工具」而非「总体」计时：
    一个卡住的数据源不应该吃掉其他工具已经拿到的结果。
    """

    def __init__(self, max_workers: int = PARALLEL_MAX_WORKERS, timeout: float | None = None):
        """timeout=None → 取 src.agent.constants.TOOL_TIMEOUT_SECONDS（见 _resolve_timeout）。"""
        self.max_workers = max_workers
        self.timeout = timeout

    def _resolve_timeout(self) -> float:
        """延迟导入超时常量（见模块 docstring 的循环导入说明）。"""
        if self.timeout is not None:
            return self.timeout
        from src.agent.constants import TOOL_TIMEOUT_SECONDS

        return TOOL_TIMEOUT_SECONDS

    def execute_parallel(
        self,
        tool_calls: Sequence[Mapping[str, Any]],
        question: str,
        executors: Mapping[str, Callable],
    ) -> list[ToolResult]:
        """并行执行一轮工具调用，返回与 tool_calls **同序**的结果列表。

        参数：
            tool_calls：`[{"name": ..., "arguments": {...}}, ...]`；arguments 允许是
                        JSON 字符串（LLM 常这么给，文档 §13 的解析步骤）
            question：当前用户问题，注入给声明了该参数的执行器
            executors：工具名 → 可调用对象，`executor(arguments, question="") -> (文本, sources)`

        任何一类失败（工具不存在、参数错误、超时、异常）都变成 ToolResult.failure，
        调用方拿到的永远是等长列表，不需要处理 None 或异常。
        """
        calls = list(tool_calls or ())
        if not calls:
            return []

        results: list[ToolResult | None] = [None] * len(calls)
        timeout = self._resolve_timeout()
        pool = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="tool")
        try:
            # _run_one 内部兜住所有异常，线程里不会有未捕获异常逃逸
            pending: dict[Any, int] = {
                pool.submit(self._run_one, call, question, executors): index
                for index, call in enumerate(calls)
            }
            # 所有工具共享同一个截止时刻：并发等待不会让总耗时随工具数叠加
            deadline = time.monotonic() + timeout
            for future, index in pending.items():
                name = _call_name(calls[index])
                try:
                    results[index] = future.result(timeout=max(0.0, deadline - time.monotonic()))
                except FutureTimeoutError:
                    logger.warning("工具 %s 执行超时（%ss）", name, timeout)
                    results[index] = ToolResult.failure(_TIMEOUT_TEXT)
                except Exception:
                    # _run_one 已兜底，此处是最后一道保险（例如执行器抛出了 BaseException 之外的东西）
                    logger.exception("工具 %s 执行异常（未走 _run_one 兜底）", name)
                    results[index] = ToolResult.failure(_FAILURE_TEXT)
        finally:
            # 不等超时任务：那条线程随后自行结束，不阻塞回答流程（与 react_loop 同一理由）
            pool.shutdown(wait=False)

        # 正常路径下每个下标都已赋值（失败或 FUTURE 结果二选一），
        # 这里的兜底只防将来改坏控制流后出现空洞，保证返回等长且无 None
        return [r if r is not None else ToolResult.failure(_FAILURE_TEXT) for r in results]

    def _run_one(self, call: Any, question: str, executors: Mapping[str, Callable]) -> ToolResult:
        """执行单个工具调用。本方法**不抛异常**：所有失败都变成 ToolResult.failure。"""
        if not isinstance(call, Mapping):
            return ToolResult.failure("工具调用格式错误：应为包含 name/arguments 的对象。")

        tool_name = call.get("name")
        if not isinstance(tool_name, str) or not tool_name:
            return ToolResult.failure("工具调用格式错误：缺少工具名。")

        # 延迟导入，见模块 docstring 的循环导入说明
        from src.agent.constants import BASE_TOOL_NAMES, WEB_TOOL_NAMES

        # 联网类工具的结果落在 results 键下，其余落 data（分配说明.md §5 的 data/results 二选一）
        payload_key: Literal["data", "results"] = "results" if tool_name in WEB_TOOL_NAMES else "data"
        if tool_name not in BASE_TOOL_NAMES and tool_name not in WEB_TOOL_NAMES:
            # 这两个清单是 SYSTEM_PROMPT 与来源分流共用的 manifest（constants.py 顶部注释）：
            # 已注册却不在清单里，通常是新增工具时漏改了 constants.py，留一条运行期线索
            logger.warning("工具 %s 不在 constants 的工具清单中，确认是否需要同步维护", tool_name)

        executor = executors.get(tool_name) if isinstance(executors, Mapping) else None
        if executor is None:
            logger.warning("工具 %s 未注册，已跳过（当前可用：%s）", tool_name, list(executors or ()))
            return ToolResult.failure(f"工具 {tool_name} 暂不可用（未注册或后端未就绪）。")
        if not callable(executor):
            logger.error("executors[%s] 不是可调用对象，而是 %r", tool_name, type(executor))
            return ToolResult.failure(f"工具 {tool_name} 未按契约注册，已跳过（executors 应为可调用对象）。")

        arguments = call.get("arguments", call.get("parameters", {}))
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments or "{}")
            except (ValueError, TypeError):
                logger.warning("工具 %s 的 arguments 不是合法 JSON", tool_name)
                return ToolResult.failure("参数解析失败：arguments 不是合法 JSON。")
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, Mapping):
            return ToolResult.failure("参数格式错误：应为 JSON 对象。")

        try:
            # 执行器契约：executor(arguments, question)；question 可缺省（react_loop 只传 arguments）
            if _accepts_question(executor):
                value = executor(dict(arguments), question)
            else:
                value = executor(dict(arguments))
        except Exception:
            # 完整异常（含服务地址、SQL、密钥）只进日志；用户可见文案固定
            logger.exception("工具 %s 执行异常", tool_name)
            return ToolResult.failure(_FAILURE_TEXT)

        result = _as_tool_result(value)
        result.payload_key = payload_key
        return result

    # 兼容 docs/组件工作机制.md §13 的命名，联调后再统一。
    run_parallel = execute_parallel


__all__ = ["BaseTool", "ToolResult", "ToolRunner", "PARALLEL_MAX_WORKERS"]

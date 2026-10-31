"""pytest 全局配置。

★ 这个文件解决一个**真实的数据外泄风险**（不是理论问题）
==============================================
`tests/` 里大量用例会走 `agent._chat_events(...)`。而 `_chat_events` 现在
按配置创建 LLM 调用记录器：

    recorder = build_recorder()   # 配了 key 就返回会上报 Langfuse 的子类

测试进程会 `load_dotenv()` 读到开发者 `.env`，而那里**可能真的配了
Langfuse key**（实测本项目就配了，指向 `https://jp.cloud.langfuse.com`）。
后果：

    跑一次 `pytest` → 每个走到问答的用例都建一个真实 Langfuse 客户端
    → SDK 后台线程把**测试数据**（含 prompt / 回答）发到真实云端账号

实测证据：加上 Langfuse 集成后，测试输出里出现

    Failed to export spans batch code: 401, reason: Unauthorized

那是**真实网络请求**（用假 key 打到真实端点）。危害有两个：

    ① 用测试数据污染用户的真实 Langfuse 账号
    ② 测试依赖网络，CI 上会慢甚至超时

★ 修法：在测试会话里**禁用上报**
==========================
把 `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` 从环境里删掉，
于是 `is_configured()` 为假、`build_recorder()` 返回纯本地记录器——
**本地 `llm_trace` 与耗时数据照旧完整**，只是不往外交数据。

这与"未配置的用户"走同一条路径，故顺带验证了默认关闭的降级逻辑。

★ 为什么放在 conftest 而不是各测试文件
================================
要覆盖**所有**测试文件（任何走 `_chat_events` 的用例都可能触发）。
放 conftest 是 pytest 保证生效的唯一位置。
"""

from __future__ import annotations

import os
import tempfile

import pytest

#: 这些环境变量一出现就意味着"会往 Langfuse 发数据"
_LANGFUSE_ENV_VARS = (
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_HOST",
    "LANGFUSE_BASE_URL",
    "LANGFUSE_TRACING_ENABLED",
)


@pytest.fixture(autouse=True, scope="session")
def _disable_online_eval_writes():
    """整个测试会话禁用**在线评测写盘**。

    ★ 为什么必须在 conftest 层挡（实测踩到的数据污染）
    --------------------------------------------
    在线评测默认是开启的（`RAG_ONLINE_EVAL` 缺省视为 1），采样率 0.2。
    而**任何**跑 `agent._chat_events(...)` 的测试都会走到
    `core._record_online_eval` → 于是约 1/5 的测试被写进
    `eval/online_samples.jsonl`（**真实数据文件**）。

    实测确认：该文件被写进 17 行测试数据（`duration_ms` 只有 16~141ms，
    而真实请求是 6~15 秒），**没有任何报错**。

    这是与 Langfuse 同一类的风险：一个"默认开启的旁路功能 + 测试会走到的
    代码路径" = 测试静默污染生产数据。故在**测试层**强制关掉。

    注意：`tests/test_eval_online.py` 自己会覆盖这些变量来测试写盘逻辑，
    且它用的是**临时目录**，故不受影响。

    ★ 同时关掉裁判采样：本地 `.env` 可能配了
    `RAG_ONLINE_EVAL_JUDGE_RATE=1.0`，而 `src.config` 在导入时会把
    `.env` 灌进 `os.environ`（override=False）。不关的话，任何显式打开
    在线评测的测试都会触发**真实裁判 LLM 调用**。
    """
    saved = {
        k: os.environ.get(k)
        for k in (
            "RAG_ONLINE_EVAL",
            "RAG_ONLINE_EVAL_RATE",
            "RAG_ONLINE_EVAL_PATH",
            "RAG_ONLINE_EVAL_JUDGE_RATE",
        )
    }
    os.environ["RAG_ONLINE_EVAL"] = "0"
    os.environ["RAG_ONLINE_EVAL_RATE"] = "0"
    os.environ["RAG_ONLINE_EVAL_JUDGE_RATE"] = "0"
    # 即便某些测试显式打开，也把数据引到临时目录，**绝不碰真实文件**
    tmp_dir = tempfile.mkdtemp(prefix="dsh-online-eval-")
    os.environ["RAG_ONLINE_EVAL_PATH"] = os.path.join(tmp_dir, "samples.jsonl")
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@pytest.fixture(autouse=True, scope="session")
def _disable_langfuse_upload():
    """整个测试会话禁用 Langfuse 上报，并在结束时关掉残留的 SDK 单例。

    **session 级 autouse**：任何测试文件、任何用例都拿不到有效的 Langfuse 凭据。

    用 `session` 而不是 function 作用域：`build_recorder()` 在每次
    `_chat_events` 里被调用，逐用例设置会有窗口期；且 session 级一次性更省。
    """
    saved = {k: os.environ.pop(k, None) for k in _LANGFUSE_ENV_VARS}
    yield
    _shutdown_stray_langfuse_clients()
    # 还原（避免影响同一进程里后续的非测试代码，如 pytest 插件）
    for k, v in saved.items():
        if v is not None:
            os.environ[k] = v


def _shutdown_stray_langfuse_clients() -> None:
    """关掉测试期间可能被建起来的 Langfuse SDK 单例。

    ★ 为什么光"删环境变量"不够
    ----------------------
    SDK 按 public key 维护**进程级单例**（`LangfuseResourceManager._instances`），
    并带一个后台 OTLP 导出线程。若在删环境变量**之前**已经有单例存在
    （例如某个用例显式建了客户端，或 `.env` 在更早被加载），
    那个线程会继续把队列里的 span 发出去——实测表现为测试结束后仍打印：

        Failed to export spans batch code: 401, reason: Unauthorized

    那是**真实网络请求**打到真实端点。故这里显式 shutdown 所有单例，
    让后台线程停下（`shutdown()` 内部会 flush + join）。

    整段包 try/except：清理失败不该让测试报错（SDK 内部结构可能变）。
    """
    try:
        from langfuse._client.resource_manager import LangfuseResourceManager

        for instance in list(LangfuseResourceManager._instances.values()):
            try:
                instance.shutdown()
            except Exception:  # noqa: BLE001
                pass
        LangfuseResourceManager._instances.clear()
    except Exception:  # noqa: BLE001 - 未装 SDK 或内部结构变化
        pass


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """确保用例改动的环境变量不外泄。

    pytest 的 `monkeypatch` 本身就会还原；这条 fixture 的作用是**显式声明**
    环境隔离的意图，并给后续可能新增的全局状态留一个统一入口。
    """
    yield

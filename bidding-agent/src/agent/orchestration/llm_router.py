"""Phase 1b：可选的 LLM specialist 路由。

规则路由（`router.route_question`）零额外调用、可解释，是默认路径。
但关键词会漏掉"语义上是多领域、字面上却没有命中关键词"的问题。
本模块提供一次**有界、可失败回落**的 LLM 路由，仅在
`AGENT_MULTI_ROUTER=auto|llm` 时由 runtime 调用：

- `auto`：只对"规则只给出一个专家、但问题明显复合"的题调用 LLM；
- `llm`：每道非 off 的多 Agent 题都调用 LLM；
- 任何超时/非法 JSON/空结果都回落规则路由，绝不阻断问答。

LLM 路由的额外成本必须可控：一次 `chat_raw`、`MULTI_AGENT_ROUTER_TIMEOUT_S=5s`
硬上限、最多 `MAX_SPECIALISTS` 个专家。模型输出只作为**建议**，仍要过
白名单与 web 开关校验。
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass

from src.agent.constants import MAX_SPECIALISTS, MULTI_AGENT_ROUTER_TIMEOUT_S
from src.agent.orchestration.router import SPECIALIST_TOOLS, AgentTask

_ROUTER_PROMPT = """你是招投标智能问答系统的专家路由器。

从下列专家中选择**真正需要**的，最多 {max_agents} 个：
- retrieval：制度/流程/法规问答，企业与标的物、采购人、供应商关系
  （工具：search_knowledge_base、search_knowledge_graph）
- data：中标金额、数量、趋势、排行、地区分布等结构化统计
  （工具：query_database）
- compliance：招标文件条款合规检查
  （工具：check_tender_compliance）
- web：最新公告、政策、新闻（工具：联网搜索）{web_note}

规则：
1. 只输出一个 JSON 对象，不要解释文字、不要 markdown 代码块：
   {{"agents": ["data", "retrieval"], "reason": "一句话理由"}}
2. agents 只能是 retrieval/data/compliance/web 中的名字，不得重复，最多 {max_agents} 个。
3. 一个问题只涉及一个领域时只给一个专家；**不要为了并行而拆分**。
4. 只有确实需要多个领域的信息时才给多个专家。
5. 不确定时宁可只给一个最相关的专家。

用户问题：{question}
"""


@dataclass(frozen=True)
class RouterResult:
    """一次 LLM 路由的结果；失败时 tasks 为空，调用方回落规则。"""

    tasks: tuple[AgentTask, ...] = ()
    reason: str = ""
    error: str = ""
    elapsed_ms: int = 0
    attempted: bool = False


def _parse_agents(raw: str) -> list[str] | None:
    """从模型输出里尽力提取 agents 列表；无法解析返回 None。"""
    text = str(raw or "").strip()
    if not text:
        return None
    # 去掉 markdown fence；模型经常把 JSON 包在 ```json ... ``` 里
    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    start, end = text.find("{"), text.rfind("}")
    payload = text[start : end + 1] if start >= 0 and end > start else text
    try:
        data = json.loads(payload)
    except Exception:
        # 兜底：只抓 "agents": ["a","b"] 这一段
        match = re.search(r'"agents"\s*:\s*\[(.*?)\]', text, flags=re.DOTALL)
        if not match:
            return None
        names = re.findall(r'"([^"]+)"', match.group(1))
        return names or None
    if not isinstance(data, dict):
        return None
    agents = data.get("agents")
    if not isinstance(agents, list):
        return None
    out: list[str] = []
    for item in agents:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict):
            name = item.get("agent") or item.get("name")
            if isinstance(name, str):
                out.append(name)
    return out


def _normalize_agents(
    names: list[str], *, web_enabled: bool
) -> list[str]:
    """白名单 + web 开关 + 去重 + 上限；顺序保持模型给出的顺序。"""
    out: list[str] = []
    for raw in names:
        name = str(raw or "").strip().lower()
        if name not in SPECIALIST_TOOLS:
            continue
        if name == "web" and not web_enabled:
            continue
        if name in out:
            continue
        out.append(name)
        if len(out) >= MAX_SPECIALISTS:
            break
    return out


def route_with_llm(
    question: str,
    *,
    web_enabled: bool,
    client,
    trace_recorder=None,
    timeout_s: float = MULTI_AGENT_ROUTER_TIMEOUT_S,
    deadline: float | None = None,
) -> RouterResult:
    """尝试一次 LLM 路由；**永不抛异常**（意外错误也转成 error 字段）。"""
    started = time.monotonic()
    if client is None:
        return RouterResult(error="client_unavailable", attempted=False)

    remaining = timeout_s
    if deadline is not None:
        remaining = min(remaining, deadline - time.monotonic())
        if remaining <= 0.2:
            return RouterResult(error="deadline", attempted=False)

    try:
        clone = getattr(client, "clone", None)
        router_client = clone() if callable(clone) else client
        if trace_recorder is not None:
            try:
                router_client.trace_recorder = trace_recorder
            except Exception:  # noqa: BLE001 - 观测挂载失败不影响路由
                pass
        prompt = _ROUTER_PROMPT.format(
            max_agents=MAX_SPECIALISTS,
            web_note="" if web_enabled else "（本轮联网关闭，禁止选择 web）",
            question=str(question or "").strip(),
        )
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm-router")
        future = pool.submit(
            router_client.chat_raw,
            [{"role": "user", "content": prompt}],
        )
        try:
            raw = future.result(timeout=max(0.2, remaining))
        except FutureTimeoutError:
            return RouterResult(
                error="timeout",
                elapsed_ms=int((time.monotonic() - started) * 1000),
                attempted=True,
            )
        finally:
            # 不等待超时线程：它只是 provider 调用 + 局部解析，不写 meta
            pool.shutdown(wait=False)
    except Exception as exc:  # noqa: BLE001 - 路由是增强项，失败回落规则
        return RouterResult(
            error=f"exception:{type(exc).__name__}",
            elapsed_ms=int((time.monotonic() - started) * 1000),
            attempted=True,
        )

    names = _parse_agents(raw)
    if names is None:
        return RouterResult(
            error="invalid_json",
            elapsed_ms=int((time.monotonic() - started) * 1000),
            attempted=True,
        )
    selected = _normalize_agents(names, web_enabled=web_enabled)
    if not selected:
        return RouterResult(
            error="no_allowed_agents",
            elapsed_ms=int((time.monotonic() - started) * 1000),
            attempted=True,
        )
    tasks = tuple(
        AgentTask(
            agent=name,
            index=index,
            task=str(question or ""),
            tools=SPECIALIST_TOOLS[name],
        )
        for index, name in enumerate(selected)
    )
    reason = ""
    try:
        data = json.loads(raw if isinstance(raw, str) else "")
        reason = str(data.get("reason") or "")[:200]
    except Exception:
        reason = ""
    return RouterResult(
        tasks=tasks,
        reason=reason,
        elapsed_ms=int((time.monotonic() - started) * 1000),
        attempted=True,
    )


__all__ = ["RouterResult", "route_with_llm"]

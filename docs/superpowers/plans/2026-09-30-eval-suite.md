# 评测三线落地（ragas_eval / agent_eval / dashboard）实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 落地阶段二验收剩余的 eval 三条评测线——`ragas_eval.py`（ragas 库 LLM-as-Judge）、`agent_eval.py`（工具选择准确率+延迟）、`dashboard.py`（feedback/会话/缓存统计），共享 `eval/common.py`，配套测试全绿。

**Architecture:** 三个独立可跑的单文件评测脚本 + 约 100 行共享基建 `common.py`（用例加载/逐题容错/聚合）。ragas 集成走 DeepSeek OpenAI 兼容端点（`trust_env=False` 红线）+ 项目本地 BGE 嵌入适配。dashboard 复用 `PostgresClient` 新增的两个只读聚合方法。设计文档：`docs/superpowers/specs/2026-09-30-eval-suite-design.md`。

**Tech Stack:** Python 3.12 + uv、ragas 0.4.x（dev 依赖）、langchain-openai、pytest（文件内局部 fake 类）、FastAPI 项目现有单例（`rag_pipeline`、`BiddingAgent`、`PostgresClient`）。

**工作目录约定**：除特别说明，所有命令在 `D:\bidding-agent\bidding-agent\` 下执行（后端项目根）。

---

### Task 1: 添加 ragas dev 依赖并验证导入

**Files:**
- Modify: `bidding-agent/pyproject.toml`（通过 uv 命令自动改）
- Modify: `bidding-agent/uv.lock`（自动改）

- [ ] **Step 1: 用 uv 添加 dev 依赖（锁定 ragas 0.4.x 防 API 漂移）**

```bash
uv add --dev "ragas>=0.4,<0.5" langchain-openai
```

- [ ] **Step 2: 验证依赖树无冲突且关键导入可用**

```bash
uv sync
uv run python -c "
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import Faithfulness, AnswerRelevancy, LLMContextRecall
from langchain_openai import ChatOpenAI
print('ragas imports OK')
"
```

Expected: 输出 `ragas imports OK`。若某个导入名在该版本不存在（v0.4 期间 API 有漂移），用 `uv run python -c "import ragas.metrics as m; print([n for n in dir(m) if 'Recall' in n or 'Relevancy' in n])"` 实测成员名，并**把实际可用的导入名写回本计划后续任务的代码里**再继续。

- [ ] **Step 3: 全量测试确认没有破坏现有环境**

```bash
uv run pytest -q -x
```

Expected: 全绿（基线 561 例）。

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore(eval): 添加 ragas 与 langchain-openai dev 依赖（评测线一 LLM-as-Judge 用，锁定 0.4.x 防 API 漂移）"
```

---

### Task 2: eval/common.py 共享基建（TDD）

**Files:**
- Create: `bidding-agent/eval/common.py`
- Test: `bidding-agent/tests/test_eval_common.py`

说明：pytest 以 rootdir（`bidding-agent/`）的 conftest 所在目录为 sys.path 前置目录，测试里 `from eval.common import ...` 可直接导入（`eval/` 无需 `__init__.py`，namespace package 即可）。

- [ ] **Step 1: 写失败测试**

```python
"""eval/common.py 的加载与容错契约测试。

评测脚本的"不中断"语义是三条评测线的公共口径：单题失败计入分母、
错误信息可读、其余题继续。这里把口径钉死，防止后续某个脚本单独漂移。
"""

import json

import pytest

from eval.common import load_cases, mean, run_cases


def _write(tmp_path, payload, name="cases.json"):
    path = tmp_path / name
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_load_cases_rejects_missing_field(tmp_path):
    path = _write(tmp_path, [{"question": "q1"}])
    with pytest.raises(SystemExit, match="reference"):
        load_cases(path, required=("question", "reference"))


def test_load_cases_rejects_missing_file(tmp_path):
    with pytest.raises(SystemExit, match="不存在"):
        load_cases(tmp_path / "nope.json", required=("question",))


def test_load_cases_rejects_empty_list(tmp_path):
    path = _write(tmp_path, [])
    with pytest.raises(SystemExit, match="为空"):
        load_cases(path, required=("question",))


def test_load_cases_ok(tmp_path):
    path = _write(tmp_path, [{"question": "q1", "reference": "r1"}])
    assert load_cases(path, required=("question", "reference")) == [
        {"question": "q1", "reference": "r1"}
    ]


def test_run_cases_isolates_failures():
    """第 2 题抛异常：1、3 题正常返回，2 题记 ok=False 且带可读错误。"""
    cases = [{"n": 1}, {"n": 2}, {"n": 3}]

    def fn(case):
        if case["n"] == 2:
            raise RuntimeError("boom")
        return {"value": case["n"]}

    results = run_cases(cases, fn)
    assert results[0]["ok"] is True and results[0]["value"] == 1
    assert results[2]["ok"] is True and results[2]["value"] == 3
    assert results[1]["ok"] is False
    assert "boom" in results[1]["error"]
    # 结果数 == 用例数：失败题也占一个分母位置
    assert len(results) == 3


def test_mean_empty_is_zero():
    assert mean([]) == 0.0
    assert mean([0.5, 1.0]) == pytest.approx(0.75)
```

- [ ] **Step 2: 运行确认失败**

```bash
uv run pytest tests/test_eval_common.py -v
```

Expected: 收集错误（`ModuleNotFoundError: No module named 'eval'` 或 ImportError）。

- [ ] **Step 3: 实现 eval/common.py**

```python
"""eval 三条评测线共享的基建：用例加载、逐题容错、聚合。

独立成模块而不是各脚本抄一遍：'单题失败计入分母、不中断整轮'的容错口径
必须三处一致，口径散落迟早漂移。retrieval_eval.py 先于本模块存在，保持原样不迁移。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable


def setup_stdio() -> None:
    """Windows 控制台默认 GBK，打印中文表格会直接抛 UnicodeEncodeError。

    文档 §10 约定由脚本自行 reconfigure；getattr 判空是为了兼容 CI 里的非 tty 流。
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def load_cases(path: Path, required: tuple[str, ...]) -> list[dict]:
    """加载用例 JSON 并校验必需字段；任何问题直接 SystemExit（评测前置条件不满足就别开跑）。"""
    if not path.exists():
        raise SystemExit(f"用例文件不存在：{path}")
    try:
        cases = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"用例文件不是合法 JSON：{path}（{exc}）") from exc
    if not isinstance(cases, list) or not cases:
        raise SystemExit(f"用例文件为空或不是列表：{path}")
    for i, case in enumerate(cases, 1):
        missing = [f for f in required if f not in case]
        if missing:
            raise SystemExit(f"用例 #{i} 缺少字段：{', '.join(missing)}")
    return cases


def run_cases(cases: list[dict], fn: Callable[[dict], dict]) -> list[dict]:
    """逐题容错执行：fn 抛任何异常只让该题失败（error 计入分母），整轮不中断。

    返回与 cases 等长的结果列表——失败题也占一个分母位置，这是三条线共用的口径。
    """
    results: list[dict] = []
    total = len(cases)
    for i, case in enumerate(cases, 1):
        try:
            result = fn(case)
            result.setdefault("ok", True)
        except Exception as exc:  # noqa: BLE001 - 容错是本函数存在的唯一目的
            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        results.append(result)
        print(f"  进度 {i}/{total}", end="\r", flush=True)
    print()
    return results


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
```

- [ ] **Step 4: 运行确认通过**

```bash
uv run pytest tests/test_eval_common.py -v
```

Expected: 6 passed。

- [ ] **Step 5: Commit**

```bash
git add eval/common.py tests/test_eval_common.py
git commit -m "feat(eval): 新增 common.py 共享基建（用例加载/逐题容错/聚合），三条评测线口径统一"
```

---

### Task 3: PostgresClient 新增 feedback_summary / conversation_stats（TDD）

**Files:**
- Modify: `bidding-agent/src/database/postgresql_client.py`（在 `stats_overview` 方法之后、`# --- 会话 / 消息 / 反馈 CRUD ---` 注释之前插入）
- Test: `bidding-agent/tests/test_pg_read_summary.py`

- [ ] **Step 1: 写失败测试**

沿用 `tests/test_pg_validation.py` 的文件内局部 fake 风格（不连真实库；假游标记录 SQL 与参数，断言参数化生效）：

```python
"""feedback_summary / conversation_stats 只读聚合的契约测试（全程不连真实数据库）。

dashboard 是面向运营的只读入口：这里钉死三件事——SQL 参数化（days 走 %s 不拼接）、
查询失败降级为空 dict（Agent/脚本不因库挂而崩）、返回形状稳定（dashboard 直接取键打印）。
"""

import src.database.postgresql_client as pgmod

POSTGRES_ENV = {
    "POSTGRES_HOST": "127.0.0.1",
    "POSTGRES_PORT": "5432",
    "POSTGRES_DB": "chatbot",
    "POSTGRES_USER": "postgres",
    "POSTGRES_PASSWORD": "test-password",
}


class _FakeCursor:
    """按调用次序弹出一组结果集：feedback_summary 会连发两条 SQL（汇总 + 每日）。"""

    def __init__(self, result_sets):
        self.result_sets = [list(rs) for rs in result_sets]
        self.executed = []

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchall(self):
        return self.result_sets.pop(0) if self.result_sets else []


class _FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def _make_client(monkeypatch):
    for name, value in POSTGRES_ENV.items():
        monkeypatch.setenv(name, value)
    return pgmod.PostgresClient()


def _install(monkeypatch, result_sets):
    client = _make_client(monkeypatch)
    cursor = _FakeCursor(result_sets)
    monkeypatch.setattr(pgmod.psycopg, "connect", lambda **kwargs: _FakeConnection(cursor))
    return client, cursor


def test_feedback_summary_shape_and_parametrization(monkeypatch):
    summary_row = {"total": 5, "up": 3, "down": 2}
    daily_rows = [{"day": "2026-09-29", "cnt": 4}, {"day": "2026-09-30", "cnt": 1}]
    client, cursor = _install(monkeypatch, [ [summary_row], daily_rows ])

    result = client.feedback_summary(days=7)

    assert result["total"] == 5 and result["up"] == 3 and result["down"] == 2
    assert result["daily"] == daily_rows
    # 两条 SQL；days 走 %s 参数（形如 "7 days"），不拼接进语句
    assert len(cursor.executed) == 2
    second_sql, second_params = cursor.executed[1]
    assert "%s::interval" in second_sql
    assert second_params == ("7 days",)


def test_feedback_summary_clamps_days(monkeypatch):
    client, cursor = _install(monkeypatch, [ [{"total": 0, "up": 0, "down": 0}], [] ])
    client.feedback_summary(days=999)
    assert cursor.executed[1][1] == ("90 days",)


def test_feedback_summary_degrades_to_empty(monkeypatch):
    client = _make_client(monkeypatch)

    def _fail(**kwargs):
        raise ConnectionError("down")

    monkeypatch.setattr(pgmod.psycopg, "connect", _fail)
    assert client.feedback_summary(days=7) == {}


def test_conversation_stats(monkeypatch):
    client, cursor = _install(monkeypatch, [ [{"conversations": 12, "messages": 340}] ])
    result = client.conversation_stats()
    assert result == {"conversations": 12, "messages": 340}
    sql = cursor.executed[0][0]
    assert "FROM conversations" in sql and "FROM messages" in sql
```

- [ ] **Step 2: 运行确认失败**

```bash
uv run pytest tests/test_pg_read_summary.py -v
```

Expected: FAIL / ERROR（`AttributeError: ... no attribute 'feedback_summary'`）。

- [ ] **Step 3: 实现两个方法**

在 `src/database/postgresql_client.py` 的 `stats_overview` 方法结束后、`# --- 会话 / 消息 / 反馈 CRUD ---` 注释之前插入：

```python
    # --- 运行看板只读聚合（eval/dashboard.py 用；与 stats_overview 同款降级口径）---

    def feedback_summary(self, days: int = 7) -> dict:
        """反馈汇总：总数、赞 / 踩、最近 N 天每日反馈数。

        rating 取值已被 API 层校验器钉死为 "up"/"down"（server.py FeedbackRequest），
        这里按枚举统计即可。days 上限 90：这是运营看板不是审计系统，窗口太大会把
        "最近趋势"稀释成"历史总量"。
        """
        days = max(1, min(int(days), 90))
        summary_sql = """
        SELECT count(*)                                AS total,
               count(*) FILTER (WHERE rating = 'up')   AS up,
               count(*) FILTER (WHERE rating = 'down') AS down
        FROM feedback
        """
        daily_sql = """
        SELECT date_trunc('day', created_at)::date AS day, count(*) AS cnt
        FROM feedback
        WHERE created_at >= now() - %s::interval
        GROUP BY 1 ORDER BY 1
        """
        rows = self._query(summary_sql)
        if not rows:
            return {}
        return {**rows[0], "daily": self._query(daily_sql, (f"{days} days",))}

    def conversation_stats(self) -> dict:
        """会话与消息总量（看板"系统真实使用量"一栏）。失败返回空 dict。"""
        rows = self._query(
            """
            SELECT (SELECT count(*) FROM conversations) AS conversations,
                   (SELECT count(*) FROM messages)      AS messages
            """
        )
        return rows[0] if rows else {}
```

- [ ] **Step 4: 运行确认通过**

```bash
uv run pytest tests/test_pg_read_summary.py tests/test_pg_validation.py -v
```

Expected: 全部 passed（新测试 4 例 + 既有 PG 测试不回归）。

- [ ] **Step 5: Commit**

```bash
git add src/database/postgresql_client.py tests/test_pg_read_summary.py
git commit -m "feat(database): 新增 feedback_summary/conversation_stats 只读聚合（dashboard 看板数据源）"
```

---

### Task 4: 生成评测用例 JSON（从知识库抽样 + 人工核对）

**Files:**
- Create: `bidding-agent/eval/test_questions.json`（25 条）
- Create: `bidding-agent/eval/test_cases.json`（15 条）

用例来自知识库真实问题（`load_qa()`），reference 用其答案字段——保证"标准答案"与知识库同源，Recall 指标才有意义。`expected_tool` 的取值必须是 `src/tools/rag_tools.py:1109` 的 `TOOL_EXECUTORS` 键：`search_knowledge_base / search_knowledge_graph / query_database / search_web / search_exa`，闲聊拒答类为空字符串。

- [ ] **Step 1: 运行一次性生成脚本（直接终端执行，不留文件）**

```bash
uv run python -c "
import json, sys
from pathlib import Path
sys.path.insert(0, '.')
from src.rag.ingest import load_qa

entries = load_qa()
print('知识库共', len(entries), '条')

questions = [
    {'question': e['question'], 'reference': e['answer']}
    for e in entries[:25]
]
Path('eval/test_questions.json').write_text(
    json.dumps(questions, ensure_ascii=False, indent=2), encoding='utf-8')

def guess_tool(q):
    if any(k in q for k in ('公司', '供应商', '供应商', '中标', '供应')):
        return 'search_knowledge_graph'
    if any(k in q for k in ('多少', '统计', '金额', '趋势', '排名', '平均', '几家')):
        return 'query_database'
    return 'search_knowledge_base'

cases = [
    {'question': e['question'], 'expected_tool': guess_tool(e['question']), 'reference': e['answer']}
    for e in entries[:12]
]
cases += [
    {'question': '你好，你能帮我做什么？', 'expected_tool': '', 'reference': ''},
    {'question': '今天天气怎么样？', 'expected_tool': '', 'reference': ''},
    {'question': '给我讲个笑话吧', 'expected_tool': '', 'reference': ''},
]
Path('eval/test_cases.json').write_text(
    json.dumps(cases, ensure_ascii=False, indent=2), encoding='utf-8')
print('test_questions.json:', len(questions), '条; test_cases.json:', len(cases), '条')
"
```

Expected: 打印 `test_questions.json: 25 条; test_cases.json: 15 条`。

- [ ] **Step 2: 人工核对（不可跳过）**

打开两个 JSON 逐条过一遍：
1. `guess_tool` 是关键词启发式，**必然有错标**——把明显该走图谱/数据库的问题手工改对；
2. 三条闲聊题确认 `expected_tool: ""` 合理（Agent 对闲聊应直接回答不调工具）；
3. 若知识库不足 25 条，取实际条数并在总结里说明。

- [ ] **Step 3: Commit**

```bash
git add eval/test_questions.json eval/test_cases.json
git commit -m "test(eval): 新增评测用例 JSON（知识库抽样 25 题含参考答案 + 15 题 agent 用例含期望工具）"
```

---

### Task 5: eval/ragas_eval.py（TDD）

**Files:**
- Create: `bidding-agent/eval/ragas_eval.py`
- Test: `bidding-agent/tests/test_ragas_eval.py`

- [ ] **Step 1: 写失败测试**

```python
"""ragas_eval 的数据组装与容错契约测试（全程不调 ragas / LLM / Qdrant）。

脚本内 ragas 的 evaluate() 在模块顶层导入后被打成桩：要验证的不是 ragas 本身，
而是"我们喂给它的列名对不对、sources 为空跳不跳、NaN 处理不处理"——这些才是
我们自己的代码。
"""

from types import SimpleNamespace

import eval.ragas_eval as mod


class _FakePipeline:
    """脚本化 rag_pipeline.chat 的返回。"""

    def __init__(self, sources=None, answer="一般不超过项目估算价的2%"):
        self._sources = sources if sources is not None else [{"content": "保证金不超过2%"}]
        self._answer = answer

    def chat(self, question, history=None, top_k=None):
        return {"answer": self._answer, "sources": self._sources,
                "tool_called": False, "tool_name": ""}


def _patch_env(monkeypatch, sources=None, evaluate_result=None):
    monkeypatch.setattr(mod, "rag_pipeline", _FakePipeline(sources=sources))

    calls = []

    def fake_evaluate(dataset, metrics, llm, embeddings):
        calls.append(dataset)
        return evaluate_result or SimpleNamespace(scores=[{
            "faithfulness": 0.8, "answer_relevancy": 0.7, "llm_context_recall": 0.9,
        }])

    monkeypatch.setattr(mod, "evaluate", fake_evaluate)
    return calls


def test_eval_one_returns_three_scores(monkeypatch):
    calls = _patch_env(monkeypatch)
    result = mod.eval_one({"question": "保证金上限？", "reference": "不超过2%"}, llm=object(), embeddings=object())
    assert result["ok"] is True
    assert result["faithfulness"] == 0.8
    assert result["relevancy"] == 0.7
    assert result["recall"] == 0.9
    # 喂给 ragas 的数据列名是 ragas 标准四列
    row = calls[0].to_list()[0]
    assert row["user_input"] == "保证金上限？"
    assert row["response"] == "一般不超过项目估算价的2%"
    assert row["retrieved_contexts"] == ["保证金不超过2%"]
    assert row["reference"] == "不超过2%"


def test_eval_one_skips_empty_sources(monkeypatch):
    _patch_env(monkeypatch, sources=[])
    result = mod.eval_one({"question": "q", "reference": "r"}, llm=object(), embeddings=object())
    assert result["ok"] is False
    assert "sources 为空" in result["error"]


def test_eval_one_maps_nan_to_zero(monkeypatch):
    _patch_env(monkeypatch, evaluate_result=SimpleNamespace(scores=[{
        "faithfulness": float("nan"), "answer_relevancy": 0.7, "llm_context_recall": 0.9,
    }]))
    result = mod.eval_one({"question": "q", "reference": "r"}, llm=object(), embeddings=object())
    assert result["faithfulness"] == 0.0
    assert result["relevancy"] == 0.7


def test_extract_scores_prefers_scores_attr(monkeypatch):
    scores = mod._extract_scores(SimpleNamespace(scores=[{"faithfulness": 0.5}]))
    assert scores == {"faithfulness": 0.5}
```

- [ ] **Step 2: 运行确认失败**

```bash
uv run pytest tests/test_ragas_eval.py -v
```

Expected: 收集错误（模块不存在）。

- [ ] **Step 3: 实现 eval/ragas_eval.py**

```python
"""RAG 回答质量评测（ragas 库，LLM-as-Judge）：Faithfulness / AnswerRelevancy / LLMContextRecall。

为什么用 ragas 而不是自写裁判提示词：拆论断、要点召回这类判分逻辑在社区库里
经过反复打磨，自写中文提示词的稳定性没有验证手段；代价是 dev 依赖树变大，
故 ragas 锁 0.4.x 防 API 漂移（见 docs/superpowers/specs/2026-09-30-eval-suite-design.md）。

出站红线：openai SDK 底层 httpx 默认 trust_env=True 会继承系统代理
（WinError 10061 坑源，见 CLAUDE.md），这里显式注入 trust_env=False 的 http_client。

用法::

    python eval/ragas_eval.py                # 全量
    python eval/ragas_eval.py --limit 5      # 快速验证
    python eval/ragas_eval.py --verbose      # 附带每题明细
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerRelevancy, Faithfulness, LLMContextRecall

from eval.common import load_cases, mean, run_cases, setup_stdio
from src.logging_config import setup_logging
from src.rag import rag_pipeline

CASES_PATH = Path(__file__).resolve().parent / "test_questions.json"

# 汇总时 NaN 视为 0：ragas 对无法判分的样本返回 NaN，直接平均会把整列拖成 NaN
_METRIC_KEYS = ("faithfulness", "answer_relevancy", "llm_context_recall")


class _BGEEmbeddings:
    """把项目本地 BGE 模型适配成 langchain Embeddings 接口。

    AnswerRelevancy 需要 embeddings；DeepSeek 没有 embeddings API、OpenAI 要额外
    key——而项目本地就有 BGE，复用零成本。返回 list[float]：langchain 契约要
    纯 Python 列表，直接回 np.ndarray 会在 ragas 内部算相似度时踩 dtype 坑。
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [np.asarray(v).tolist() for v in rag_pipeline.embedder.encode_documents(list(texts))]

    def embed_query(self, text: str) -> list[float]:
        return np.asarray(rag_pipeline.embedder.encode_query(text)).tolist()


def _check_ready() -> None:
    if not rag_pipeline.ready:
        raise SystemExit(f"知识库未就绪：{rag_pipeline.ready_error}；先运行 python main.py ingest")


def build_evaluator_llm():
    """裁判 LLM：DeepSeek 的 OpenAI 兼容端点。http_client 见文件头红线说明。"""
    import httpx
    from langchain_openai import ChatOpenAI

    from src.config import settings

    return ChatOpenAI(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=0.0,
        http_client=httpx.Client(trust_env=False, timeout=60.0),
    )


def _extract_scores(result) -> dict[str, float]:
    """ragas 各版本的 evaluate() 返回类型有差异，按属性逐级兼容提取单行分数。"""
    if hasattr(result, "scores"):  # EvaluationResult：单行数据集取第一行
        row = result.scores[0]
        return {k: float(v) for k, v in dict(row).items() if isinstance(v, (int, float))}
    if hasattr(result, "to_pandas"):
        row = result.to_pandas().iloc[0]
        return {k: float(row[k]) for k in _METRIC_KEYS if k in row}
    return {k: float(v) for k, v in dict(result).items() if isinstance(v, (int, float))}


def eval_one(case: dict, llm, embeddings) -> dict:
    result = rag_pipeline.chat(case["question"])
    sources = [
        s.get("content", "") if isinstance(s, dict) else str(s)
        for s in result.get("sources", [])
    ]
    if not sources:
        return {"ok": False, "error": "sources 为空（知识库未命中），该题不计入指标分子"}

    scores = _extract_scores(evaluate(
        dataset=EvaluationDataset.from_list([{
            "user_input": case["question"],
            "response": result.get("answer", ""),
            "retrieved_contexts": sources,
            "reference": case.get("reference", ""),
        }]),
        metrics=[Faithfulness(), AnswerRelevancy(), LLMContextRecall()],
        llm=LangchainLLMWrapper(llm),
        embeddings=LangchainEmbeddingsWrapper(embeddings),
    ))
    return {
        "ok": True,
        "faithfulness": 0.0 if math.isnan(scores.get("faithfulness", 0.0)) else scores.get("faithfulness", 0.0),
        "relevancy": 0.0 if math.isnan(scores.get("answer_relevancy", 0.0)) else scores.get("answer_relevancy", 0.0),
        "recall": 0.0 if math.isnan(scores.get("llm_context_recall", 0.0)) else scores.get("llm_context_recall", 0.0),
        "answer": result.get("answer", ""),
    }


def main() -> int:
    setup_stdio()
    parser = argparse.ArgumentParser(description="RAG 回答质量评测（ragas LLM-as-Judge）")
    parser.add_argument("--limit", type=int, default=0, help="只用前 N 条做快速验证")
    parser.add_argument("--verbose", action="store_true", help="打印每题分数与答案")
    args = parser.parse_args()

    setup_logging()
    _check_ready()
    cases = load_cases(CASES_PATH, required=("question", "reference"))
    if args.limit:
        cases = cases[: args.limit]

    llm = build_evaluator_llm()
    embeddings = _BGEEmbeddings()

    print(f"用例 {len(cases)} 条 | 裁判模型：{llm.model_name}")
    started = time.perf_counter()
    results = run_cases(cases, lambda c: eval_one(c, llm, embeddings))
    ok_rows = [r for r in results if r.get("ok")]
    errors = len(results) - len(ok_rows)

    print()
    print(f"{'指标':<20}{'平均分':>10}")
    print("-" * 30)
    for label, key in (("Faithfulness 忠实度", "faithfulness"),
                       ("AnswerRelevancy 相关性", "relevancy"),
                       ("LLMContextRecall 召回率", "recall")):
        print(f"{label:<20}{mean([r[key] for r in ok_rows]):>10.3f}")
    print(f"\n有效 {len(ok_rows)}/{len(results)}（error {errors}，计入分母） | 总耗时 {time.perf_counter() - started:.1f}s")

    if args.verbose:
        print("\n===== 每题明细 =====")
        for case, r in zip(cases, results):
            if not r.get("ok"):
                print(f"[失败] {case['question']} -> {r.get('error')}")
                continue
            print(f"[{r['faithfulness']:.2f}/{r['relevancy']:.2f}/{r['recall']:.2f}] {case['question']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

注意：若 Task 1 Step 2 实测的导入名不同（如 `AnswerRelevancy` 叫 `ResponseRelevancy`），同步修改本文件与测试桩里的键名——**指标键名以实际安装版本返回的 dict 键为准**（在 `_extract_scores` 返回后 `print` 一次实测确认）。

- [ ] **Step 4: 运行确认通过**

```bash
uv run pytest tests/test_ragas_eval.py -v
```

Expected: 4 passed。若 `EvaluationDataset.from_list(...)` 的桩断言取不到 `to_list()`，改用 `calls[0].to_list` 失败信息里的真实方法名（如 `.rows` / `.data`）并对齐测试与实现——**以实测为准，两处必须一致**。

- [ ] **Step 5: 真实冒烟（小样本）**

```bash
uv run python eval/ragas_eval.py --limit 3
```

Expected: 能跑出 3 题的分数表格（首次会触发 BGE 模型加载，慢属正常）。若 ragas API 报错，按报错信息修正导入/参数后重跑。

- [ ] **Step 6: Commit**

```bash
git add eval/ragas_eval.py tests/test_ragas_eval.py
git commit -m "feat(eval): ragas_eval.py 基于 ragas 库的 RAG 质量评测（DeepSeek 裁判 + 本地 BGE 嵌入，trust_env=False 红线适配）"
```

---

### Task 6: eval/agent_eval.py（TDD）

**Files:**
- Create: `bidding-agent/eval/agent_eval.py`
- Test: `bidding-agent/tests/test_agent_eval.py`

- [ ] **Step 1: 写失败测试**

```python
"""agent_eval 的判分口径测试（全程不构造真实 Agent）。

钉死文档 §6.4 的口径：tool_name 是单字符串、与 expected_tool 精确相等才算对；
error 题（ok=False）没有 match 键，按"答错"计入分母。
"""

import eval.agent_eval as mod


class _FakeAgent:
    def __init__(self, script):
        self._script = script  # question -> chat() 返回
        self.calls = []

    def chat(self, question, history=None, web_search_enabled=False,
             provider="", deep_thinking_enabled=False):
        self.calls.append(question)
        return self._script[question]


SCRIPT = {
    "某公司中过哪些标？": {"answer": "……", "tool_name": "search_knowledge_graph",
                          "elapsed_ms": 1500, "sources": []},
    "最近的招标政策？": {"answer": "……", "tool_name": "query_database",
                        "elapsed_ms": 900, "sources": []},
}


def test_eval_one_exact_string_match():
    mod.agent = _FakeAgent(SCRIPT)
    result = mod.eval_one({"question": "某公司中过哪些标？", "expected_tool": "search_knowledge_graph"})
    assert result["ok"] is True and result["match"] is True
    result = mod.eval_one({"question": "最近的招标政策？", "expected_tool": "search_web"})
    assert result["match"] is False
    assert result["expected"] == "search_web" and result["actual"] == "query_database"


def test_eval_one_uses_default_params():
    """评测走非流式 chat() 且默认参数（无联网/无深度思考），文档 §10 口径。"""
    fake = _FakeAgent(SCRIPT)
    mod.agent = fake
    mod.eval_one({"question": "某公司中过哪些标？", "expected_tool": "search_knowledge_graph"})
    assert fake.calls == ["某公司中过哪些标？"]


def test_accuracy_counts_errors_as_wrong():
    class _BoomAgent:
        def chat(self, question, **kwargs):
            if "炸" in question:
                raise RuntimeError("LLM 超时")
            return {"answer": "", "tool_name": "search_knowledge_base", "elapsed_ms": 100}

    mod.agent = _BoomAgent()
    cases = [
        {"question": "正常题", "expected_tool": "search_knowledge_base"},
        {"question": "会炸的题", "expected_tool": "search_knowledge_base"},
    ]
    results = mod.run_cases(cases, mod.eval_one)
    matched = sum(1 for r in results if r.get("match"))
    # 2 题分母：error 题不算对
    assert len(results) == 2 and matched == 1
```

- [ ] **Step 2: 运行确认失败**

```bash
uv run pytest tests/test_agent_eval.py -v
```

Expected: 收集错误（模块不存在）。

- [ ] **Step 3: 实现 eval/agent_eval.py**

```python
"""Agent 端到端评测：工具选择准确率 + 延迟。

口径（docs/开发文档.md §6.4/§10）：走非流式 chat()、默认参数（无联网/深度思考）；
tool_name 是"最后一个执行完的工具"单字符串，与 expected_tool 精确相等才算对。
多轮 ReAct 后工具名可能与第一直觉不同——用例标注时就按这个口径来。

用法::

    python eval/agent_eval.py                # 全量
    python eval/agent_eval.py --limit 5      # 快速验证
    python eval/agent_eval.py --verbose      # 附带每题答案全文
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.common import load_cases, run_cases, setup_stdio
from src.agent import BiddingAgent
from src.logging_config import setup_logging

CASES_PATH = Path(__file__).resolve().parent / "test_cases.json"

# main() 里构造；测试直接替换本模块属性（文件内局部 fake 的常规打法）
agent: BiddingAgent | None = None


def eval_one(case: dict) -> dict:
    result = agent.chat(case["question"])
    expected = case["expected_tool"]
    actual = result.get("tool_name", "")
    return {
        "ok": True,
        "match": actual == expected,
        "expected": expected,
        "actual": actual,
        "elapsed_ms": result.get("elapsed_ms", 0),
        "answer": result.get("answer", ""),
    }


def _p95(values: list[int]) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]


def main() -> int:
    global agent
    setup_stdio()
    parser = argparse.ArgumentParser(description="Agent 工具选择准确率 + 延迟评测")
    parser.add_argument("--limit", type=int, default=0, help="只用前 N 条做快速验证")
    parser.add_argument("--verbose", action="store_true", help="打印每题答案全文")
    args = parser.parse_args()

    setup_logging()
    agent = BiddingAgent()
    cases = load_cases(CASES_PATH, required=("question", "expected_tool"))
    if args.limit:
        cases = cases[: args.limit]

    print(f"用例 {len(cases)} 条 | 非流式 chat()，默认参数")
    started = time.perf_counter()
    results = run_cases(cases, eval_one)
    matched = sum(1 for r in results if r.get("match"))
    latencies = [r["elapsed_ms"] for r in results if r.get("ok")]

    print()
    print(f"工具选择准确率：{matched}/{len(results)} = {matched / len(results):.1%}")
    print(f"延迟：平均 {sum(latencies) / len(latencies):.0f}ms | P95 {_p95(latencies)}ms | 总耗时 {time.perf_counter() - started:.1f}s")

    print("\n===== 每题明细 =====")
    for case, r in zip(cases, results):
        if not r.get("ok"):
            print(f"[失败] {case['question']} -> {r.get('error')}")
        elif r["match"]:
            print(f"[√] {case['question']} -> {r['actual']} ({r['elapsed_ms']}ms)")
        else:
            print(f"[×] {case['question']} -> 期望 {r['expected']}，实际 {r['actual']}")
        if args.verbose and r.get("ok"):
            print(f"    答案：{r['answer'][:120]}{'…' if len(r['answer']) > 120 else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 运行确认通过**

```bash
uv run pytest tests/test_agent_eval.py -v
```

Expected: 3 passed。

- [ ] **Step 5: 真实冒烟（小样本）**

```bash
uv run python eval/agent_eval.py --limit 3
```

Expected: 打印准确率/延迟/明细（首次会触发嵌入模型加载，慢属正常）。

- [ ] **Step 6: Commit**

```bash
git add eval/agent_eval.py tests/test_agent_eval.py
git commit -m "feat(eval): agent_eval.py 工具选择准确率与延迟评测（非流式 chat 口径，error 计入分母）"
```

---

### Task 7: eval/dashboard.py（薄胶水层，无单测）

**Files:**
- Create: `bidding-agent/eval/dashboard.py`

- [ ] **Step 1: 实现**

```python
"""运行看板：feedback 汇总 + 会话统计 + 检索缓存统计（受限口径）。

缓存统计的口径局限必须如实标注：rag_pipeline 的缓存与命中账本只存在于本进程
内存，本脚本是独立进程，读不到长驻 API 服务的账本（打印出来接近全 0）。
它只作"管线通没通"的冒烟验证，真实命中率要在 API 服务进程内观察。

用法::

    python eval/dashboard.py                 # 默认最近 7 天
    python eval/dashboard.py --days 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.common import setup_stdio
from src.database.postgresql_client import PostgresClient
from src.logging_config import setup_logging


def main() -> int:
    setup_stdio()
    parser = argparse.ArgumentParser(description="运行看板（feedback / 会话 / 缓存统计）")
    parser.add_argument("--days", type=int, default=7, help="反馈趋势统计窗口（天）")
    args = parser.parse_args()

    setup_logging()
    pg = PostgresClient()
    health = pg.health()
    if not health["ok"]:
        # health() 的错误文案已脱敏（不含主机名/端口/驱动报文），可直接展示
        print(f"PostgreSQL 不可用：{health['error']}")
        return 1

    fb = pg.feedback_summary(days=args.days)
    print("===== 用户反馈汇总 =====")
    if not fb:
        print("（无数据或查询失败）")
    else:
        total = fb.get("total", 0)
        up, down = fb.get("up", 0), fb.get("down", 0)
        print(f"总反馈 {total} | 赞 {up}（{up / total:.0%}）| 踩 {down}（{down / total:.0%}）" if total else "总反馈 0")
        for row in fb.get("daily", []):
            print(f"  {row['day']}  {'█' * min(row['cnt'], 40)} {row['cnt']}")

    cs = pg.conversation_stats()
    print("\n===== 会话统计 =====")
    if not cs:
        print("（无数据或查询失败）")
    else:
        print(f"会话 {cs.get('conversations', 0)} 个 | 消息 {cs.get('messages', 0)} 条")

    from src.rag import cache_info

    ci = cache_info()
    print("\n===== 检索缓存统计（受限口径）=====")
    print(f"缓存条目 {ci.get('size', 0)} | 命中 {ci.get('hits', 0)}")
    print("说明：以上为脚本本进程数据，仅作冒烟验证；真实命中率请在长驻 API 服务进程内观察。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: 语法检查 + 真实冒烟**

```bash
uv run python -c "import ast; ast.parse(open('eval/dashboard.py', encoding='utf-8').read()); print('syntax OK')"
uv run python eval/dashboard.py
```

Expected: 第一条输出 `syntax OK`；第二条打印三段统计（数据库可用时是真实数据；不可用时走降级提示且 exit 1）。

- [ ] **Step 3: Commit**

```bash
git add eval/dashboard.py
git commit -m "feat(eval): dashboard.py 运行看板（feedback/会话统计 + 缓存统计如实标注进程内口径局限）"
```

---

### Task 8: 全量验证 + 真实跑通三线 + 文档同步

**Files:**
- Modify: `bidding-agent/docs/开发文档.md`（§1 目录树、§10 评测指南）
- Modify: `bidding-agent/docs/技术栈.md`（§9 评测线说明 + ragas 依赖注记）
- Modify: `bidding-agent/docs/组件工作机制.md`（§20 评测图更新为"已落地"）
- Modify: `CLAUDE.md`（测试数量、CI 状态描述）

- [ ] **Step 1: 全量测试**

```bash
uv run pytest -q
```

Expected: 全绿（561 + 新增约 17 例）。记录实际数字。

- [ ] **Step 2: 三条线真实各跑一次并记录结果**

```bash
uv run python eval/ragas_eval.py > eval_report_ragas.txt 2>&1 || true
uv run python eval/agent_eval.py > eval_report_agent.txt 2>&1 || true
uv run python eval/dashboard.py > eval_report_dashboard.txt 2>&1 || true
```

检查三个报告文件有实际数字（不是 traceback）。这三份 txt 是临时产物，**验证后删除，不提交**。

- [ ] **Step 3: 文档同步（对照 CLAUDE.md §文档同步义务）**

1. `docs/开发文档.md` §1 目录树 `eval/` 行改为：`├── eval/                      # common + ragas_eval / agent_eval / dashboard / retrieval_eval + 用例 JSON`
2. `docs/开发文档.md` §10 表格：三行脚本的"说明"确认与实现一致（ragas 0.4、工具口径、dashboard 局限），删除"待落地"类措辞；补充 `common.py` 与 `feedback_summary/conversation_stats` 一句
3. `docs/技术栈.md` §9：补一行 ragas 0.4.x（dev 依赖，锁定 `<0.5`）+ `trust_env=False` 适配注记
4. `docs/组件工作机制.md` §20：评测图与文字改为已落地描述
5. `CLAUDE.md`：`uv run pytest -q`（561 例）改为实际例数；"Dashboard/extract/…仍为 mock"等描述不动

- [ ] **Step 4: Commit**

```bash
git add docs/开发文档.md docs/技术栈.md docs/组件工作机制.md ../CLAUDE.md
git commit -m "docs: 同步评测三线落地（ragas/agent/dashboard 已可跑，测试数与文档更新）"
```

- [ ] **Step 5: 清理临时报告文件**

```bash
rm -f eval_report_ragas.txt eval_report_agent.txt eval_report_dashboard.txt
```

---

## 自查记录

- **Spec 覆盖**：spec §1 交付物表逐项有对应 Task（依赖→T1、common→T2、PG 方法→T3、用例 JSON→T4、三脚本→T5/T6/T7、测试→各任务、文档同步→T8）；spec §8"明确不做"未引入。
- **类型一致性**：`run_cases(cases, fn)` 签名在 T2 定义、T5/T6 使用一致；`eval_one` 返回含 `ok` 键与 `run_cases` 的 `setdefault("ok", True)` 契约一致；`feedback_summary` 返回 `daily` 键与 dashboard 取键一致；测试补丁目标 `mod.agent` / `mod.evaluate` / `mod.rag_pipeline` 均为模块顶层属性。
- **已知风险**：ragas v0.4 API 漂移——T1 Step 2 与 T5 Step 4 都有"以实测为准"的对齐指引，且 T5 Step 5 强制真实冒烟。

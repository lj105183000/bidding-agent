"""RAG 回答质量评测（ragas 库，LLM-as-Judge）：Faithfulness / AnswerRelevancy / LLMContextRecall。

为什么用 ragas 而不是自写裁判提示词：拆论断、要点召回这类判分逻辑在社区库里
经过反复打磨，自写中文提示词的稳定性没有验证手段；代价是 dev 依赖树变大，
故 ragas 锁 0.4.x 防 API 漂移（见 docs/superpowers/specs/2026-09-30-eval-suite-design.md）。

★ 本脚本的定位：**为数据驱动优化服务**，而不只是打一张分数表
------------------------------------------------------------------
一张"忠实度 0.750"的表格不能支撑任何决策——你不知道 0.750 是好是坏、
改动之后是升是降、是哪几道题拖低的。故本脚本提供四件配套能力：

    --configs        多配置 A/B 对比（top_k / 精排开关），用数据判断改动值不值
    --baseline       与基线比对，退化超过容差则以非 0 退出（可进 CI）
    --diagnose       逐题诊断：列出拖低指标的题目 + 低于阈值的个案
    --kb-overlap     用例-知识库重合度自检，防止把"背答案"当成"质量好"

★ 为什么需要 --kb-overlap（实测踩坑）
------------------------------------
实测本项目的 `test_questions.json` 与 75 条知识库**大量重合**：
第 10~19 题的最相近库条目**就是同名问题**（相似度 0.81~0.94）。

后果是：这套指标实际测的是"**检索管道是否正常**"（回归检测），
**不是**"面对知识库外的新问题能答多好"（真实质量）。
两者都会给出高分，但只有前者能在改动检索参数时反映退化。

若不加这条自检，很容易把 0.94 的 top-1 读成"RAG 质量优秀"——
而它其实只说明"照着答案问，系统能找到答案"。

★ 指标怎么读（每条都有对应的失败模式）
------------------------------------
    Faithfulness 低   → 回答里有检索文档**支持不了**的内容（幻觉）。
                        先查检索是否漏召回，再怀疑生成。
    AnswerRelevancy 低 → 答非所问（跑了题或过度保守说"不知道"）。
    LLMContextRecall 低→ 检索**没找到**回答参考答案所需的内容。
                        这是检索侧指标，与生成无关——优化方向完全不同。

出站红线：openai SDK 底层 httpx 默认 trust_env=True 会继承系统代理
（WinError 10061 坑源，见 CLAUDE.md）。ragas 只驱动异步链路（LangchainLLMWrapper.
agenerate_text → agenerate_prompt），走的是 ChatOpenAI 独立的 http_async_client
字段——同步/异步两个客户端都必须显式注入 trust_env=False，只配其一等于没配。

用法::

    python eval/ragas_eval.py                          # 全量
    python eval/ragas_eval.py --limit 5 --verbose      # 快速验证
    python eval/ragas_eval.py --save-baseline          # 跑一遍并存为新基线
    python eval/ragas_eval.py --baseline               # 与基线比对（退化则退出码 1）
    python eval/ragas_eval.py --configs top3,top5,rerank   # A/B 对比
    python eval/ragas_eval.py --diagnose --json out.json   # 诊断 + 机器可读输出
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 必须先于 ragas 导入：ragas → huggingface_hub 在 import 期读取 HF_ENDPOINT 并冻结为
# 模块常量，晚了就回落 huggingface.co——境内被 DNS 污染后 TCP SYN 直接挂死（无超时）。
# main.py 天然先 import src.config 所以没踩过这个坑；本脚本是入口，必须显式保证顺序。
import src.config  # noqa: F401  - 仅副作用：加载 .env（HF_ENDPOINT 等）

import numpy as np
from ragas import EvaluationDataset, evaluate
from ragas import RunConfig as RagasRunConfig
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerRelevancy, Faithfulness, LLMContextRecall

from eval.common import load_cases, mean, run_cases, setup_stdio
from src.logging_config import setup_logging
from src.rag import rag_pipeline

HERE = Path(__file__).resolve().parent
CASES_PATH = HERE / "test_questions.json"
BASELINE_PATH = HERE / "baseline.json"

# ragas 的 evaluate() 内部按「指标实例的 .name」索引 per-sample 分数（evaluation.py）。
# 0.4.x 里 LLMContextRecall 是旧 ContextRecall 的重导出，.name 仍是 "context_recall"
# 而非类名暗示的 "llm_context_recall"——曾因手写键名与实际返回不一致，召回列恒为
# 默认 0.0 且无任何报错。因此读取键一律从实例化指标推导，禁止手写；契约测试
# （test_ragas_eval.py::test_metric_name_contract）把键名双向钉死防漂移。
_OUTPUT_KEYS = ("faithfulness", "relevancy", "recall")  # 对外输出列名，main() 表格与测试依赖

# ★ AnswerRelevancy 必须显式指定 strictness=1（实测踩坑，会伪造 0 分）
# ------------------------------------------------------------------
# ragas 默认 `strictness=3`，意味着它用 `n=3` 请求裁判生成 3 个问题。
# 而 DeepSeek 只支持 `n=1`，直接返回：
#
#     400 BadRequestError: Invalid n value (currently only n = 1 is supported)
#
# 该样本的分数于是变成 **NaN**，而评测脚本原本把 NaN 当 0 处理
# → 界面上显示"相关性 0.000"，看起来像模型答得极差。
#
# 实测影响：25 题里有 2 题被这样伪造成 0 分（占 8%）。其中一例是
# "回答完全正常（703 字、叙述严谨）却得 0.000"，把一次**完全正常的回答**
# 误判成重灾，并让我基于它做出了错误的优化方向判断。
#
# 修正后（strictness=1）同一输入三次判分为 0.788 / 0.795 / 0.788（极差 0.007），
# 既不再报错、也不再伪造 0 分。
#
# 注：Faithfulness 与 LLMContextRecall 不走 n>1 路径，实测不受影响
# （重复判分给出 0.444 / 0.750 等一致的实数）。
_METRICS = [Faithfulness(), AnswerRelevancy(strictness=1), LLMContextRecall()]
_NAME_TO_OUTPUT = {m.name: key for m, key in zip(_METRICS, _OUTPUT_KEYS)}
# to_pandas 兜底路径按同一组键从 DataFrame 列取值
_METRIC_KEYS = tuple(_NAME_TO_OUTPUT)

# 指标中文名（表格与诊断共用，避免两处漂移）
_LABELS = {
    "faithfulness": "忠实度",
    "relevancy": "相关性",
    "recall": "召回率",
}

# 浮点比较用的极小量。二进制浮点减法会有 ~1e-16 的误差，
# 使"恰好等于容差"被误判（实测 0.75-0.80 = -0.050000000000000044）。
# 取 1e-9：远大于浮点误差，又远小于任何有意义的指标差异。
_EPS = 1e-9

# 生产默认配置名。基线必须锚定它，而不是"本次第一个跑的配置"——
# 否则 `--configs top3,top5` 会把 top3 的分数存成基线，之后所有比对都虚高。
_PRODUCTION_CONFIG = "top5"


# ===========================================================================
# 配置：可对比的管线参数
# ===========================================================================


@dataclass
class RunConfig:
    """一次评测使用的管线配置。

    为什么要有这个抽象：优化的前提是"能对比"。`top_k`、精排开关、提示词变体
    都能在运行时改，故不需要重启进程就能跑 A/B：

        top_k          `chat(top_k=...)`，逐次传入，无全局状态
        rerank         `settings.rerank_enabled`，在 `pipeline.search` 里**调用时**读取
        prompt_variant `pipeline.ACTIVE_SYSTEM_PROMPT`，在 `build_messages` 里**调用时**读取

    后两者是全局的，`_run_config` 会在 `finally` 里还原——
    否则一次 A/B 会把状态留给后续配置，测出来不是各自的真实表现。
    """

    name: str
    top_k: int = 5
    rerank: bool = False
    # 提示词变体名；空串表示"用当前生效的"（便于单跑时不干扰）
    prompt_variant: str = ""


# 预置配置：名字 → RunConfig。命令行 --configs 用这些名字
_PRESETS: dict[str, RunConfig] = {
    "top3": RunConfig("top3", top_k=3, rerank=False),
    "top5": RunConfig("top5", top_k=5, rerank=False),  # 生产默认
    "top8": RunConfig("top8", top_k=8, rerank=False),
    "top10": RunConfig("top10", top_k=10, rerank=False),
    "rerank": RunConfig("rerank", top_k=5, rerank=True),
    "rerank8": RunConfig("rerank8", top_k=8, rerank=True),
    # 提示词 A/B：同 top_k、同精排，只差生成约束
    "prompt-default": RunConfig("prompt-default", top_k=5, prompt_variant="default"),
    "prompt-balanced": RunConfig("prompt-balanced", top_k=5, prompt_variant="balanced"),
    "prompt-strict": RunConfig("prompt-strict", top_k=5, prompt_variant="strict"),
    "prompt-minimal": RunConfig("prompt-minimal", top_k=5, prompt_variant="minimal"),
}


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


#: 裁判模型默认值：智谱 GLM。
#:
#: ★ 为什么用 GLM 而不是 DeepSeek（原本就是 DeepSeek，这是本轮改动的核心理由）
#: --------------------------------------------------------------
#: **被评测的生成方是 DeepSeek**（`LLM_PROVIDER=deepseek`）。裁判若也用
#: DeepSeek，就是"自己评自己"——同族模型倾向给同族输出更高分
#: （self-preference bias），指标系统性偏高，拿它做 A/B 或回归门禁都可能误判。
#:
#: 换 GLM 后裁判与生成**不同家**，偏差显著减小。实测 `glm-5.3-flash` 可用。
DEFAULT_JUDGE_MODEL = "glm-5.3-flash"

#: 从环境变量覆盖裁判模型（模型会下线，硬编码早晚会坏——DeepSeek 刚发生过
#: `deepseek-chat` 退役；且本地/离线部署可能想换成本地模型）。
ENV_JUDGE_MODEL = "RAGAS_JUDGE_MODEL"

#: 判分失败时的**最大尝试次数**（含首次）。
#:
#: ★ 为什么需要重试（实测数据决定，不是保险起见）
#: ----------------------------------------
#: 把裁判换成智谱后，`glm-5.3-flash` 单次成功率仅 **4/6 ≈ 0.67**，
#: 失败原因是 ragas 的"从回答抽取陈述"返回空列表（→ `np.nan`）。
#: 但**同一份输入重试就能成功**（实测 p=0.67 时，2 次 ≈ 0.89、3 次 ≈ 0.96）。
#:
#: 对照（同一条用例）：DeepSeek 3/3、glm-5.3-flash 4/6、
#: glm-4.7-flashx 0/4、glm-4.6 0/3、glm-4.7 0/3。
#: 即智谱家族在 ragas 的结构化输出上明显弱——但为了消除自评偏差
#: （裁判必须与生成方不同家）只能用它，故用重试把可靠性补回来。
#:
#: 用 3 次：p≥0.9。可用 `RAGAS_JUDGE_ATTEMPTS` 覆盖。
DEFAULT_JUDGE_ATTEMPTS = 3
ENV_JUDGE_ATTEMPTS = "RAGAS_JUDGE_ATTEMPTS"


def _judge_attempts() -> int:
    """取最大尝试次数：环境变量优先，非法值回落默认。永不小于 1。"""
    import os

    raw = (os.getenv(ENV_JUDGE_ATTEMPTS, "") or "").strip()
    if not raw:
        return DEFAULT_JUDGE_ATTEMPTS
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_JUDGE_ATTEMPTS


#: 模块级默认值（供测试与打印引用）。
#:
#: ⚠️ 真正生效的次数由 `_judge_attempts()` 在**运行期**解析——不要在评分逻辑里
#: 直接用这个常量，否则改 `RAGAS_JUDGE_ATTEMPTS` 会不生效（读的是导入时冻结的值）。
JUDGE_ATTEMPTS = DEFAULT_JUDGE_ATTEMPTS


def _judge_model_name() -> str:
    """取裁判模型名：环境变量优先，否则默认 GLM。

    延迟 import os 而不是放文件顶部：本模块的导入曾经全部延迟（见文件头
    对"重依赖 + 代理残留"的说明），新增模块级 import 会改变导入时序。
    """
    import os

    return (os.getenv(ENV_JUDGE_MODEL, "") or "").strip() or DEFAULT_JUDGE_MODEL


def judge_provider_matches_generator() -> tuple[bool, str, str]:
    """裁判与生成是否**同源**（同 provider）。返回 `(是否同源, 生成方, 裁判方)`。

    ★ 为什么要做这个自检
    -----------------
    换成 GLM 只在"生成方是 DeepSeek"时消除偏差。若用户把 `LLM_PROVIDER`
    改成 `zhipu`，裁判与生成**又变成同家**，自评偏差回来了——
    而**不会有任何提示**，指标悄悄变得不可信。

    故显式检查并警告。**不阻断**：同源评测仍有参考价值（纵向比较同一配置的
    变化趋势），只是不能当作"绝对质量"的独立证据。由使用者决定。
    """
    from src.config import settings

    generator = (settings.llm_provider or "").strip().lower()
    judge = "zhipu"  # 裁判固定走智谱端点
    return generator == judge, generator, judge


def build_evaluator_llm():
    """裁判 LLM：**智谱 GLM** 的 OpenAI 兼容端点。

    双客户端注入见文件头红线说明（ragas 走异步路径，两个客户端都要钉死）。

    ★ 与生成方分离是**刻意设计**，不是随手换家
    ------------------------------------
    生成方是 DeepSeek，裁判用 GLM —— 不同家族才能降低自评偏差。
    改动前两者都是 DeepSeek（自评）。模型名可用 `RAGAS_JUDGE_MODEL` 覆盖。
    """
    import httpx
    from langchain_openai import ChatOpenAI

    from src.clients.zai_client import ZHIPU_BASE_URL
    from src.config import settings

    return ChatOpenAI(
        model=_judge_model_name(),
        api_key=settings.zhipu_api_key,
        base_url=ZHIPU_BASE_URL,
        temperature=0.0,
        # ragas 只走异步路径，异步用的是 ChatOpenAI 独立的 http_async_client 字段：
        # 只配同步 http_client 的话真实链路会回落到 trust_env=True 的默认客户端，
        # 代理残留场景（WinError 10061）照样炸，两个都要钉死
        # ★ 180s 而不是 60s：ragas 的 faithfulness 会为一篇回答并发多次
        #   "抽陈述 + 逐条 NLI"，智谱在并发下经常 60s 内拿不到全部响应；
        #   实测 60s 会让每个用例前两次尝试超时、第三次才成功（3 倍耗时）。
        http_client=httpx.Client(trust_env=False, timeout=180.0),
        http_async_client=httpx.AsyncClient(trust_env=False, timeout=180.0),
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


def _source_text(source) -> str:
    """从一条 RAG source 里取正文，供 ragas 的 `retrieved_contexts` 使用。

    ★ 为什么必须取 `answer` 而不是 `content`（实测踩坑）
    --------------------------------------------------
    本项目有**两种** sources 契约，字段名不同：

        RAG 检索（`rag/vector_store.py` 的 `_payload_to_record`）
            `{question(标题), answer(正文), score, relevance}`   ← 正文在 answer
        知识图谱 / 联网 / API（`rag_tools._web_sources` 等）
            `{question(标题), answer(正文), score, url}`          ← 联网亦用 answer
        `/api/web/search` 端点
            `{title, content, ...}`                              ← 只有这里用 content

    第一版这里写的是 `s.get("content", "")` —— RAG source 里**没有** `content`，
    于是每条都取到空串，`retrieved_contexts` 变成 `["", "", "", "", ""]`。

    后果是**静默的错误结论**：Faithfulness 与 LLMContextRecall 恒为 0.000，
    而 AnswerRelevancy（只看 response 与 question，不看 context）仍显示 0.847。
    指标表看起来"跑通了"，却把一次有充分依据的回答判成"完全不忠实于上下文"——
    比直接报错更有害。

    故这里按 `answer` → `content` → `str(source)` 依次回落，并**过滤空串**：
    空 context 会被 ragas 当成"没有检索到任何依据"，同样污染指标。
    """
    if isinstance(source, dict):
        for key in ("answer", "content", "text", "page_content"):
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return value
        # 兜底：把整条记录序列化，总好过喂空串给裁判
        return json.dumps(source, ensure_ascii=False)
    return str(source)


def eval_one(case: dict, llm, embeddings, config: RunConfig) -> dict:
    result = rag_pipeline.chat(case["question"], top_k=config.top_k)
    sources = [
        text
        for text in (_source_text(s) for s in result.get("sources", []))
        if text.strip()
    ]
    if not sources:
        return {"ok": False, "error": "sources 为空（知识库未命中），该题不计入指标分子"}

    # KB 命中但生成失败时，pipeline 走部分失败路径：sources 非空却带 error 键返回
    # （pipeline.py 的 chat 收敛逻辑）。这种降级回答不能当有效样本计分，否则指标被污染。
    if result.get("error"):
        return {"ok": False, "error": f"pipeline 报错：{result['error']}"}

    # ★ 判分失败自动重试（换 GLM 裁判后必需）
    #
    # 背景（实测数据）：把裁判从 DeepSeek 换成智谱 GLM 后，
    # `glm-5.3-flash` 单次判分成功率仅 **4/6 ≈ 0.67**，失败信息是
    # `faithfulness: 裁判未给出分数（NaN/Inf）`。
    #
    # 根因**不是 JSON 解析**，而是 ragas 的第一步"从回答抽取陈述"返回了空列表
    # （`ragas/metrics/_faithfulness.py` L191/L210 → `np.nan`）。
    # 关键观察：**同一份输入换个时刻重试就能成功**（实测 2 次内成功率 ≈ 0.89，
    # 3 次 ≈ 0.96）。故这是**随机性**问题，重试是有效解。
    #
    # 对照数据（同一条用例、同一条 _run_config 路径）：
    #     DeepSeek（原裁判）  3/3        glm-5.3-flash      4/6
    #     glm-4.7-flashx     0/4        glm-4.6            0/3
    #     glm-4.7            0/3        glm-5.3（标准档）   1/2
    # 即：**智谱这个家族在 ragas 要求的结构化输出上明显弱于 DeepSeek**。
    # 但为了消除自评偏差（裁判必须与生成方不同家），只能接受它 + 重试兜底。
    #
    # 次数量级：失败只发生在"抽不出陈述"时，故绝大多数样本 1 次就过；
    # 只有失败样本才多花 1~2 次裁判调用。
    scores: dict[str, float] = {}
    _attempts = _judge_attempts()
    for attempt in range(1, _attempts + 1):
        scores = _extract_scores(evaluate(
            dataset=EvaluationDataset.from_list([{
                "user_input": case["question"],
                "response": result.get("answer", ""),
                "retrieved_contexts": sources,
                "reference": case.get("reference", ""),
            }]),
            metrics=_METRICS,
            llm=LangchainLLMWrapper(llm),
            embeddings=LangchainEmbeddingsWrapper(embeddings),
            # ★ ragas 默认 max_workers=16 / timeout=180：智谱裁判在 16 路并发下
            #   经常整组超时（实测每用例要重试 2 次才出分，单次约 3 分钟）。
            #   降到 8 路并放宽到 300s：既能并行跑完 faithfulness 的多条 NLI，
            #   又不至于把智谱打到限流/整组超时。
            #   max_retries=1：ragas 内层 tenacity 默认重试 10 次、每次最多等 60s，
            #   一次坏响应会把整个用例拖到 5 分钟以上；外层已有 3 次整体重试，
            #   内层快速失败反而更快（失败样本不会把整轮拖死）。
            run_config=RagasRunConfig(
                timeout=300, max_workers=8, max_retries=1, max_wait=10
            ),
        ))
        # 三项都拿到有限值 → 成功，不必再试
        #
        # 判据是"**所有**指标都有有限值"而不是"任一有值"：只要有一项是 NaN，
        # 该样本最终仍会被标为判分失败（下面的 `unjudged` 逻辑），
        # 故必须整体成功才算数。
        if all(
            isinstance(scores.get(n), (int, float))
            and not math.isnan(scores.get(n, float("nan")))
            and not math.isinf(scores.get(n, float("nan")))
            for n in _NAME_TO_OUTPUT
        ):
            break
        if attempt < _attempts:
            # 用 `print` 而不是 `logging`：本模块**没有** logging 配置
            # （全文用 print 输出），保持一致。
            # 实测踩到：第一版写了 `logger.warning`，而本文件从未定义 logger
            # → 一旦真的触发重试就 `NameError`，把"判分失败"升级成"整轮崩溃"。
            print(
                f"    ⚠️ 判分未出分，第 {attempt + 1}/{_attempts} 次重试"
                f"（裁判 {getattr(llm, 'model_name', '?')}）"
            )

    row = {"ok": True}
    unjudged: list[str] = []
    for name, out_key in _NAME_TO_OUTPUT.items():
        v = scores.get(name, float("nan"))
        # ★ 指标名**缺失**与**判分失败**都必须显式失败，不能当 0 分
        # ------------------------------------------------------
        # 原先把 NaN 静默当 0 处理。这带来两类假象：
        #   ① 指标名写错 → 整列恒为 0 且无任何报错（历史上真发生过）
        #   ② 裁判调用失败（如 AnswerRelevancy 的 n=3 触发 400）→ 该样本 0 分
        # 两者都表现为"模型答得很差"，而实际是度量本身坏了。
        # 一个静默的错误 0 比一个响亮的报错有害得多——前者会被当成质量结论，
        # 后者才会有人去查。
        #
        # 故这里改为：判分失败 → 该题标记为失败（进入 error 计数），
        # 并从指标均值中排除（`run_cases`/`summarize` 只统计 ok 行）。
        if name not in scores:
            unjudged.append(f"{out_key}: 指标名 {name!r} 不在 ragas 返回里（版本漂移？）")
            continue
        if math.isnan(v) or math.isinf(v):
            unjudged.append(f"{out_key}: 裁判未给出分数（NaN/Inf）")
            continue
        row[out_key] = v

    if unjudged:
        return {"ok": False, "error": "判分失败 —— " + "；".join(unjudged)}

    row["answer"] = result.get("answer", "")
    row["sources"] = sources
    return row


# ===========================================================================
# 汇总与统计
# ===========================================================================


def summarize(rows: list[dict]) -> dict:
    """对有效样本算每项指标的均值 / 中位 / 最低 / 零分数占比。

    ★ 为什么不能只报均值（实测教训）
    ------------------------------
    全量 25 题的 Faithfulness 是 0.750，而前 2 题是 0.977。
    只看均值会以为"整体还行"，实际是**少数题拿 0 分把均值拉下来了**——
    优化方向完全不同（前者要提升普遍质量，后者要定位那几道题）。

    故额外给出 `zeros`（拿 0 分的题数）与 `min`：
    若 zeros 较多，应该去看 `--diagnose` 的具体题目，而不是调参。
    """
    out: dict = {"n": len(rows)}
    for key in _OUTPUT_KEYS:
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        if not vals:
            out[key] = {"mean": 0.0, "median": 0.0, "min": 0.0, "zeros": 0}
            continue
        out[key] = {
            "mean": round(mean(vals), 4),
            "median": round(statistics.median(vals), 4),
            "min": round(min(vals), 4),
            "zeros": sum(1 for v in vals if v <= 1e-9),
        }
    return out


def _fmt_table(summary: dict, *, indent: str = "  ") -> str:
    lines = [
        f"{indent}{'指标':<12}{'均值':>8}{'中位':>8}{'最低':>8}{'0分题数':>9}",
        f"{indent}{'-' * 45}",
    ]
    for key in _OUTPUT_KEYS:
        s = summary[key]
        lines.append(
            f"{indent}{_LABELS[key]:<12}{s['mean']:>8.3f}{s['median']:>8.3f}"
            f"{s['min']:>8.3f}{s['zeros']:>9}"
        )
    return "\n".join(lines)


# ===========================================================================
# 逐题诊断
# ===========================================================================


def diagnose(cases: list[dict], results: list[dict], threshold: float = 0.6) -> list[str]:
    """列出拖低指标的题目，并归类原因。

    只报"哪些题低于阈值"是不够的——还需要指出**该往哪个方向查**：

        三项都低            → 检索没找到依据（查语料/召回），不是生成问题
        Faithfulness 低其余高 → 生成时超出文档自由发挥（查提示词/温度）
        只有 Recall 低        → 检索侧漏召回（查 top_k / 查询改写）
        只有 Relevancy 低     → 答非所问（查提示词约束）
    """
    lines: list[str] = []
    weak: list[tuple[int, dict, dict]] = []
    failed: list[tuple[int, dict, dict]] = []

    for i, (case, r) in enumerate(zip(cases, results), 1):
        if not r.get("ok"):
            failed.append((i, case, r))
            continue
        vals = [r.get(k, 0.0) for k in _OUTPUT_KEYS]
        if min(vals) < threshold:
            weak.append((i, case, r))

    if failed:
        lines.append(f"■ 执行失败 {len(failed)} 题（不计入指标分子，但计入分母）")
        for i, case, r in failed:
            lines.append(f"  [{i:>2}] {case['question'][:52]}")
            lines.append(f"       {r.get('error')}")
        lines.append("")

    if not weak:
        lines.append(f"■ 无低于 {threshold:.2f} 的题目 —— 三项指标均已达标")
        return lines

    lines.append(f"■ 低于阈值 {threshold:.2f} 的题目 {len(weak)} 道（按最弱项排序）")
    lines.append("")

    # 按"最弱项"升序，最需要关注的排最前
    weak.sort(key=lambda t: min(t[2].get(k, 0.0) for k in _OUTPUT_KEYS))

    for i, case, r in weak:
        f, rel, rec = (r.get(k, 0.0) for k in _OUTPUT_KEYS)
        lines.append(f"  [{i:>2}] 忠实 {f:.2f} | 相关 {rel:.2f} | 召回 {rec:.2f}")
        lines.append(f"       {case['question'][:60]}")

        # 归类：三项全低 → 检索问题；单独偏低 → 归因到对应环节
        if f < threshold and rel < threshold and rec < threshold:
            lines.append("       → 三项全低：**检索没找到依据**，查语料覆盖与召回，不是生成问题")
        elif rec < threshold and f >= threshold and rel >= threshold:
            lines.append("       → 只有召回低：检索漏召回（考虑调 top_k 或查询改写）")
        elif f < threshold and rec >= threshold:
            lines.append("       → 召回够但忠实度低：生成超出了文档内容，查提示词约束")
        elif rel < threshold and f >= threshold:
            lines.append("       → 忠实但答非所问：查提示词是否要求直接回答")
        lines.append("")

    return lines


# ===========================================================================
# 用例-知识库重合度自检
# ===========================================================================


def check_kb_overlap(cases: list[dict], sample_kb: int = 200) -> dict:
    """检查评测题与知识库的重合度，判断这套指标能说明什么。

    ★ 为什么必须有这一步（实测发现）
    ------------------------------
    实测 25 道评测题里有 10 道的最相近库条目**就是同名问题**（相似度 0.81~0.94）。
    这意味着指标衡量的是"照着答案问，系统能不能找到答案"——
    是**回归检测**能力，不是"面对新问题能答多好"的**真实质量**。

    两者在数值上都很好看，但只有前者能在调检索参数时反映变化。
    不把这件事说清楚，很容易把 0.94 的 top-1 读成"RAG 质量优秀"。

    返回 `{"mean_similarity", "verbatim", "high", "verdict", "detail"}`。
    """
    result: dict = {
        "mean_similarity": 0.0,
        "verbatim": 0,   # 相似度 >= 0.85：基本是原题
        "high": 0,       # 0.70 ~ 0.85：库里有对应内容
        "low": 0,        # < 0.60：库里可能没有
        "verdict": "",
        "detail": [],
    }
    try:
        kb = rag_pipeline.search("招投标 采购 法规 保证金 中标 招标", top_k=sample_kb)
    except Exception as exc:
        result["verdict"] = f"重合度检查跳过（检索失败：{type(exc).__name__}）"
        return result
    if not kb:
        result["verdict"] = "重合度检查跳过（知识库为空）"
        return result

    kb_vecs = rag_pipeline.embedder.encode_documents(
        [f"{p.get('question', '')} {p.get('answer', '')}".strip()[:400] for p in kb]
    )
    kb_questions = [p.get("question", "") for p in kb]

    sims: list[float] = []
    for i, case in enumerate(cases, 1):
        qv = rag_pipeline.embedder.encode_query(case["question"])
        na = sum(a * a for a in qv) ** 0.5
        best, best_idx = 0.0, 0
        for j, kv in enumerate(kb_vecs):
            nb = sum(b * b for b in kv) ** 0.5
            if not na or not nb:
                continue
            c = sum(a * b for a, b in zip(qv, kv)) / (na * nb)
            if c > best:
                best, best_idx = c, j
        sims.append(best)
        if best >= 0.85:
            result["verbatim"] += 1
        elif best >= 0.70:
            result["high"] += 1
        elif best < 0.60:
            result["low"] += 1
        result["detail"].append({
            "index": i,
            "question": case["question"],
            "best_similarity": round(best, 4),
            "nearest_kb_question": kb_questions[best_idx],
        })

    result["mean_similarity"] = round(mean(sims), 4)
    n = len(cases)
    if result["verbatim"] >= n * 0.3:
        result["verdict"] = (
            f"⚠️ 重合度过高：{result['verbatim']}/{n} 题与库内问题几乎相同（相似度≥0.85）。"
            "本指标主要反映**检索管道回归**，不能代表「面对新问题的真实质量」。"
        )
    elif result["mean_similarity"] >= 0.75:
        result["verdict"] = (
            f"重合度偏高（均值 {result['mean_similarity']:.3f}）：库内多有对应内容，"
            "指标偏乐观，宜作回归检测用。"
        )
    else:
        result["verdict"] = f"重合度适中（均值 {result['mean_similarity']:.3f}），指标有区分度。"
    return result


# ===========================================================================
# 基线
# ===========================================================================


def save_baseline(summary: dict, config: dict | RunConfig, n_cases: int) -> None:
    """存档基线。`config` 接受 dict（来自 run 结果）或 RunConfig。"""
    if isinstance(config, RunConfig):
        cfg = {
            "name": config.name,
            "top_k": config.top_k,
            "rerank": config.rerank,
            "prompt_variant": config.prompt_variant or "<active>",
        }
    else:
        cfg = dict(config)
    payload = {
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": cfg,
        "cases": n_cases,
        "metrics": {k: summary[k]["mean"] for k in _OUTPUT_KEYS},
        "detail": {k: summary[k] for k in _OUTPUT_KEYS},
    }
    BASELINE_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n基线已保存：{BASELINE_PATH}（锚定 {cfg.get('name')}）")


def compare_baseline(
    summary: dict,
    tolerance: float,
    config: RunConfig,
    n_cases: int | None = None,
) -> tuple[bool, list[str]]:
    """与基线比对。返回 (是否通过, 报告行)。

    `tolerance` 是**绝对差**容差（如 0.05 表示允许下降 5 个百分点）。
    为什么用绝对差而不是相对差：0.90→0.81 是相对降 10%，0.10→0.09 也是 10%，
    但后者毫无意义（低分区的波动本来就大）。绝对差配合"只看下降"更贴合
    "指标不能退化"这个诉求。

    ★ `n_cases` 是**本次数据集规模**，不是有效判分数
    --------------------------------------------------
    `summary["n"]` 是有效判分题数（裁判失败的不算），它会随裁判抖动而变；
    而 `baseline.json` 的 `cases` 存的是数据集规模（40）。若拿 35 去比 40，
    一次正常的全量运行会被误判成"用例数不一致"，独立基线门禁永远不判。
    传 `n_cases` 时用它做可比性判据；不传时保留旧行为（用 `summary["n"]`，
    兼容只关心样本量的调用方与测试）。
    """
    if not BASELINE_PATH.exists():
        return False, [f"未找到基线文件 {BASELINE_PATH}；先跑一次 --save-baseline"]

    base = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    lines = [
        f"基线：{base.get('saved_at')} | 配置 {base.get('config', {}).get('name')} "
        f"| {base.get('cases')} 题 | 容差 {tolerance:.3f}",
        "",
        f"  {'指标':<12}{'基线':>9}{'本次':>9}{'变化':>10}{'判定':>8}",
        f"  {'-' * 50}",
    ]
    passed = True
    for key in _OUTPUT_KEYS:
        b = float(base.get("metrics", {}).get(key, 0.0))
        c = float(summary[key]["mean"])
        delta = c - b
        # 浮点容差：二进制浮点减法会引入 ~1e-16 量级的误差，使"恰好等于容差"
        # 被误判为退化。实测 `0.75 - 0.80 = -0.050000000000000044`，
        # 与容差 0.05 比较时 `< -0.05` 成立 → 明明在容差内却报红。
        # 这属于**假阳性**：用户设了 0.05 容差，恰好降 0.05 却被拦，
        # 会让人不再信任这个门禁。故比较时放宽一个极小的 eps。
        if delta < -(tolerance + _EPS):
            passed = False
            verdict = "退化"
        elif delta > tolerance + _EPS:
            verdict = "提升"
        else:
            verdict = "持平"
        lines.append(
            f"  {_LABELS[key]:<12}{b:>9.3f}{c:>9.3f}{delta:>+10.3f}{verdict:>8}"
        )

    # ★ 不可比的情形必须显式列出并**取消判据**，而不是照常给"提升/退化"
    #
    # 两种会得出错误结论的情况：
    #   1. 配置不同（top_k / 精排开关）——指标本来就会不同，判退化是错的
    #   2. 用例数不同（如 --limit 2 对 25 题基线）——样本不同，比较无意义
    #
    # 实测踩过第 2 种：`--limit 2 --baseline` 报了"忠实度 +0.139 提升"，
    # 而真相是 2 道简单题的均分对不上 25 题的均分。这类假提升比假退化更危险——
    # 它让人以为优化见效了。
    incomparable: list[str] = []
    base_cfg = base.get("config", {})
    if (base_cfg.get("top_k"), base_cfg.get("rerank")) != (config.top_k, config.rerank):
        incomparable.append(
            f"配置不一致：基线 top_k={base_cfg.get('top_k')} rerank={base_cfg.get('rerank')}，"
            f"本次 top_k={config.top_k} rerank={config.rerank}"
        )
    # 提示词变体不同同样是不可比的：它直接改变回答风格，
    # 拿 strict 的分数比 default 的基线会把"提示词变更的影响"误读成"管线退化"
    base_prompt = base_cfg.get("prompt_variant")
    cur_prompt = getattr(config, "prompt_variant", "") or base_prompt
    if base_prompt and cur_prompt and base_prompt != cur_prompt:
        incomparable.append(
            f"提示词变体不一致：基线 {base_prompt}，本次 {cur_prompt}"
        )
    base_cases = base.get("cases")
    cur_cases = n_cases if isinstance(n_cases, int) else summary.get("n")
    if isinstance(base_cases, int) and cur_cases != base_cases:
        incomparable.append(
            f"用例数不一致：基线 {base_cases} 题，本次 {cur_cases} 题"
            "（--limit 的结果不能与全量基线比较）"
        )

    if incomparable:
        lines.append("")
        for reason in incomparable:
            lines.append(f"  ⚠️ {reason}")
        lines.append("  → 以上对比**不作为退化判据**（仅展示数值，不判通过/失败）")
        passed = True

    lines.append("")
    lines.append("  ✅ 指标未退化" if passed else "  ❌ 指标退化超过容差")
    return passed, lines


# ===========================================================================
# 主流程
# ===========================================================================


def _resolve_configs(spec: str) -> list[RunConfig]:
    names = [s.strip() for s in (spec or "").split(",") if s.strip()]
    out: list[RunConfig] = []
    for name in names:
        if name not in _PRESETS:
            raise SystemExit(
                f"未知配置 {name!r}；可用：{', '.join(_PRESETS)}"
            )
        out.append(_PRESETS[name])
    return out


def _run_config(
    config: RunConfig,
    cases: list[dict],
    llm,
    embeddings,
    *,
    diagnose_on: bool,
    threshold: float,
) -> dict:
    from src.config import settings
    from src.rag import pipeline as pipeline_mod

    prev_rerank = settings.rerank_enabled
    prev_prompt = pipeline_mod.ACTIVE_SYSTEM_PROMPT
    prev_variant = pipeline_mod.PROMPT_VARIANT
    settings.rerank_enabled = config.rerank
    if config.prompt_variant:
        variants = pipeline_mod._PROMPT_VARIANTS
        if config.prompt_variant not in variants:
            raise SystemExit(
                f"未知提示词变体 {config.prompt_variant!r}；"
                f"可用：{', '.join(variants)}"
            )
        pipeline_mod.ACTIVE_SYSTEM_PROMPT = variants[config.prompt_variant]
        pipeline_mod.PROMPT_VARIANT = config.prompt_variant
    try:
        started = time.perf_counter()
        results = run_cases(cases, lambda c: eval_one(c, llm, embeddings, config))
        elapsed = time.perf_counter() - started
    finally:
        # 必须还原：否则一次 A/B 会把状态留给后续配置，测出来不是各自的真实表现。
        # 提示词尤其重要——它直接决定回答风格，残留会让"下一组配置"实际仍在用
        # 上一组的提示词，而输出上完全看不出来。
        settings.rerank_enabled = prev_rerank
        pipeline_mod.ACTIVE_SYSTEM_PROMPT = prev_prompt
        pipeline_mod.PROMPT_VARIANT = prev_variant

    ok_rows = [r for r in results if r.get("ok")]
    errors = len(results) - len(ok_rows)
    summary = summarize(ok_rows)
    out = {
        "config": {
            "name": config.name,
            "top_k": config.top_k,
            "rerank": config.rerank,
            "prompt_variant": config.prompt_variant or prev_variant,
        },
        "summary": summary,
        "valid": len(ok_rows),
        "total": len(results),
        "errors": errors,
        "elapsed_s": round(elapsed, 1),
        "results": results,
    }
    if diagnose_on:
        out["diagnosis"] = diagnose(cases, results, threshold)
    return out


def main() -> int:
    setup_stdio()
    parser = argparse.ArgumentParser(description="RAG 回答质量评测（ragas LLM-as-Judge）")
    parser.add_argument("--limit", type=int, default=0, help="只用前 N 条做快速验证")
    parser.add_argument("--verbose", action="store_true", help="打印每题分数与答案")
    parser.add_argument("--configs", default="", help=f"逗号分隔的配置名：{', '.join(_PRESETS)}")
    parser.add_argument("--save-baseline", action="store_true", help="把本次结果存为基线")
    parser.add_argument("--baseline", action="store_true", help="与基线比对，退化则退出码 1")
    parser.add_argument(
        "--baseline-config",
        default=_PRODUCTION_CONFIG,
        help=f"基线锚定哪个配置（默认 {_PRODUCTION_CONFIG}）。"
             "跑提示词 A/B 时应指定为对照组的名字",
    )
    parser.add_argument("--tolerance", type=float, default=0.05, help="允许的绝对下降幅度（默认 0.05）")
    parser.add_argument("--diagnose", action="store_true", help="逐题诊断拖低指标的题目")
    parser.add_argument("--threshold", type=float, default=0.6, help="诊断阈值（默认 0.6）")
    parser.add_argument("--no-overlap-check", action="store_true", help="跳过用例-知识库重合度自检")
    parser.add_argument("--json", dest="json_path", default="", help="把结果写到该 JSON 文件")
    parser.add_argument(
        "--json-per-config",
        dest="json_per_config",
        default="",
        help="逐题明细按配置分文件写到该前缀下（形如 <前缀>.prompt-strict.json）",
    )
    args = parser.parse_args()

    if args.tolerance < 0:
        raise SystemExit("--tolerance 不能为负")

    setup_logging()
    _check_ready()
    cases = load_cases(CASES_PATH, required=("question", "reference"))
    # 负数 limit 会从尾部切片甚至切出空列表，静默评测错误子集（mean 对空列表回 0，汇总列
    # 全 0 也看不出异常，比直接崩更危险）；钳到 0 后复用 "--limit 0 = 全量" 的既有语义
    # （0 为假值，不进切片分支）
    limit = max(args.limit, 0)
    if limit:
        cases = cases[:limit]

    try:
        llm = build_evaluator_llm()
    except Exception as exc:
        # ChatOpenAI 在 main() 构造期就 eagerly 校验 api_key（openai SDK 行为），失败发生在
        # run_cases 的逐题容错之前——必须在此拦住，否则裸的英文 traceback 直接甩给用户，
        # 违反异常分级约定。OpenAIError 文案本身干净（不含密钥），str(exc) 可直接展示。
        print(
            f"裁判 LLM 构建失败：{exc}"
            f"（裁判走智谱端点，请检查 .env 的 ZHIPU_API_KEY；"
            f"模型名由 {ENV_JUDGE_MODEL} 控制，当前 {_judge_model_name()}）"
        )
        return 1
    embeddings = _BGEEmbeddings()

    configs = _resolve_configs(args.configs) or [_PRESETS["top5"]]

    # ---- 重合度自检：先说清"这套指标能说明什么" ----
    overlap = None
    if not args.no_overlap_check:
        print("正在检查用例与知识库的重合度…")
        overlap = check_kb_overlap(cases)
        print(f"  {overlap['verdict']}")
        print(f"  分布：几乎原题 {overlap['verbatim']} | 有对应内容 {overlap['high']} "
              f"| 库内缺失 {overlap['low']} | 均值 {overlap['mean_similarity']:.3f}")
        print()

    print(f"用例 {len(cases)} 条 | 裁判模型：{llm.model_name} | 配置 {len(configs)} 组")
    # ★ 同源自检：裁判与生成同家时，自评偏差回来了，必须让用户知道
    _same, _gen, _judge = judge_provider_matches_generator()
    if _same:
        print()
        print("⚠️  裁判与生成方**同源**（都是 %s）——存在自评偏差，指标会系统性偏高。" % _gen)
        print("    建议把 LLM_PROVIDER 保持 deepseek（裁判走智谱），"
              "或把裁判换成第三家：export %s=<其它厂商模型>" % ENV_JUDGE_MODEL)
    else:
        print(f"裁判与生成方不同源（生成 {_gen} / 裁判 {_judge}）→ 已规避自评偏差")

    runs: list[dict] = []
    for config in configs:
        print()
        print("=" * 70)
        print(f"配置 {config.name}（top_k={config.top_k}, rerank={config.rerank}）")
        print("=" * 70)
        run = _run_config(
            config, cases, llm, embeddings,
            diagnose_on=args.diagnose, threshold=args.threshold,
        )
        runs.append(run)
        print()
        print(_fmt_table(run["summary"]))
        print(f"\n  有效 {run['valid']}/{run['total']}（error {run['errors']}，计入分母）"
              f" | 耗时 {run['elapsed_s']:.1f}s")

        if args.diagnose and run.get("diagnosis"):
            print()
            for line in run["diagnosis"]:
                print(line)

    # ---- 多配置对比 ----
    if len(runs) > 1:
        print()
        print("=" * 70)
        print("配置对比")
        print("=" * 70)
        header = f"  {'配置':<16}{'top_k':>6}{'精排':>6}{'提示词':>10}"
        for key in _OUTPUT_KEYS:
            header += f"{_LABELS[key]:>10}"
        header += f"{'耗时':>9}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for run in runs:
            c = run["config"]
            line = (
                f"  {c['name']:<16}{c['top_k']:>6}"
                f"{'是' if c['rerank'] else '否':>6}"
                f"{c.get('prompt_variant', '-') or '-':>10}"
            )
            for key in _OUTPUT_KEYS:
                line += f"{run['summary'][key]['mean']:>10.3f}"
            line += f"{run['elapsed_s']:>8.0f}s"
            print(line)

        # 给出"相对生产默认（top5）"的变化，直接回答"改动值不值"
        base = next(
            (r for r in runs if r["config"]["name"] == _PRODUCTION_CONFIG),
            runs[0],
        )
        print()
        print(f"  以 {base['config']['name']} 为参照的变化：")
        for run in runs:
            if run is base:
                continue
            parts = []
            for key in _OUTPUT_KEYS:
                d = run["summary"][key]["mean"] - base["summary"][key]["mean"]
                parts.append(f"{_LABELS[key]} {d:+.3f}")
            print(f"    {run['config']['name']:<10} " + " | ".join(parts))

    # ★ 基线锚定哪个配置，由 `--baseline-config` 决定（默认生产配置 top5）
    #
    # 为什么做成可指定（实测踩坑）
    # --------------------------
    # 原先硬编码 `_PRODUCTION_CONFIG = "top5"`。跑提示词 A/B 时配置列表是
    # `prompt-default,prompt-strict,prompt-minimal`，里面**没有** top5，
    # 于是每次都打印"未包含生产默认配置"的警告并退回第一个配置。
    # 虽然结果是想要的，但**每次都报警告**会训练人忽略警告——
    # 等哪天真出了问题时，那条警告也不会被看见。
    #
    # 改为可指定后，跑对照实验时显式说明"以哪一组为基准"，不再有噪音警告。
    exit_code = 0
    target = args.baseline_config
    production = next((r for r in runs if r["config"]["name"] == target), None)
    if production is None:
        available = ", ".join(r["config"]["name"] for r in runs)
        print()
        print(
            f"  ⚠️ 配置列表里没有 {target}（本次：{available}）；"
            f"基线将锚定 {runs[0]['config']['name']}"
        )
    primary = production or runs[0]
    if args.save_baseline:
        save_baseline(primary["summary"], primary["config"], len(cases))
    if args.baseline:
        print()
        print("=" * 70)
        print(f"与基线比对（锚定 {primary['config']['name']}）")
        print("=" * 70)
        passed, lines = compare_baseline(
            primary["summary"], args.tolerance,
            RunConfig(
                primary["config"]["name"],
                primary["config"]["top_k"],
                primary["config"]["rerank"],
                primary["config"].get("prompt_variant", ""),
            ),
            # 用数据集规模而不是有效判分数：裁判失败会让 summary["n"] 抖动，
            # 拿它比 cases 会把正常全量运行误判成"用例数不一致"
            n_cases=len(cases),
        )
        for line in lines:
            print(line)
        if not passed:
            exit_code = 1

    # ---- JSON 输出 ----
    if args.json_path:
        payload = {
            "cases": len(cases),
            "overlap": overlap,
            "runs": [
                {k: v for k, v in run.items() if k != "results"} for run in runs
            ],
            "per_case": [
                {
                    "index": i,
                    "question": case["question"],
                    **{k: r.get(k) for k in _OUTPUT_KEYS},
                    "ok": bool(r.get("ok")),
                    "error": r.get("error"),
                }
                for i, (case, r) in enumerate(zip(cases, primary["results"]), 1)
            ],
            "config": primary["config"],
            # 逐题明细**按配置分组**保留。
            #
            # ★ 为什么必须留（实测踩坑）
            # ------------------------
            # 第一版只输出 `per_case`（首个配置的），于是 A/B 跑完后
            # **无法回答"相关性下降落在哪几道题上"**——而那正是判断
            # "这个权衡能不能接受"的关键。只能重跑一遍 27 分钟的实验。
            #
            # 只留前 3 项指标与答案长度，不留回答全文：
            # 全文会让 JSON 膨胀到几 MB，而指标本身已足够做逐题对比。
            "per_config": {
                run["config"]["name"]: [
                    {
                        "index": i,
                        "question": case["question"],
                        **{k: r.get(k) for k in _OUTPUT_KEYS},
                        "ok": bool(r.get("ok")),
                        "error": r.get("error"),
                        "answer_len": len(str(r.get("answer") or "")),
                    }
                    for i, (case, r) in enumerate(zip(cases, run["results"]), 1)
                ]
                for run in runs
            },
        }
        Path(args.json_path).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n结果已写入 {args.json_path}")

    # 逐题明细按配置分文件：单文件里所有配置挤在一起时，
    # 想单独看某一组的分布要写筛选代码；分文件后可直接加载比较。
    if args.json_per_config:
        base_path = Path(args.json_per_config)
        for run in runs:
            name = run["config"]["name"]
            detail = {
                "config": run["config"],
                "summary": run["summary"],
                "elapsed_s": run["elapsed_s"],
                "per_case": [
                    {
                        "index": i,
                        "question": case["question"],
                        "faithfulness": r.get("faithfulness"),
                        "relevancy": r.get("relevancy"),
                        "recall": r.get("recall"),
                        "ok": bool(r.get("ok")),
                        "error": r.get("error"),
                        "answer_len": len(str(r.get("answer") or "")),
                    }
                    for i, (case, r) in enumerate(zip(cases, run["results"]), 1)
                ],
            }
            out_path = base_path.with_name(f"{base_path.stem}.{name}.json")
            out_path.write_text(
                json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        print(f"逐题明细已写入 {base_path.stem}.*.json（{len(runs)} 个文件）")

    if args.verbose:
        print("\n===== 每题明细（首个配置）=====")
        for i, (case, r) in enumerate(zip(cases, primary["results"]), 1):
            if not r.get("ok"):
                print(f"[失败] {case['question']} -> {r.get('error')}")
                continue
            print(f"[{r['faithfulness']:.2f}/{r['relevancy']:.2f}/{r['recall']:.2f}] "
                  f"{case['question']}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

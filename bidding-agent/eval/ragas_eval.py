"""RAG 回答质量评测（ragas 库，LLM-as-Judge）：Faithfulness / AnswerRelevancy / LLMContextRecall。

为什么用 ragas 而不是自写裁判提示词：拆论断、要点召回这类判分逻辑在社区库里
经过反复打磨，自写中文提示词的稳定性没有验证手段；代价是 dev 依赖树变大，
故 ragas 锁 0.4.x 防 API 漂移（见 docs/superpowers/specs/2026-09-30-eval-suite-design.md）。

出站红线：openai SDK 底层 httpx 默认 trust_env=True 会继承系统代理
（WinError 10061 坑源，见 CLAUDE.md）。ragas 只驱动异步链路（LangchainLLMWrapper.
agenerate_text → agenerate_prompt），走的是 ChatOpenAI 独立的 http_async_client
字段——同步/异步两个客户端都必须显式注入 trust_env=False，只配其一等于没配。

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

# 必须先于 ragas 导入：ragas → huggingface_hub 在 import 期读取 HF_ENDPOINT 并冻结为
# 模块常量，晚了就回落 huggingface.co——境内被 DNS 污染后 TCP SYN 直接挂死（无超时）。
# main.py 天然先 import src.config 所以没踩过这个坑；本脚本是入口，必须显式保证顺序。
import src.config  # noqa: F401  - 仅副作用：加载 .env（HF_ENDPOINT 等）

import numpy as np
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerRelevancy, Faithfulness, LLMContextRecall

from eval.common import load_cases, mean, run_cases, setup_stdio
from src.logging_config import setup_logging
from src.rag import rag_pipeline

CASES_PATH = Path(__file__).resolve().parent / "test_questions.json"

# ragas 的 evaluate() 内部按「指标实例的 .name」索引 per-sample 分数（evaluation.py）。
# 0.4.x 里 LLMContextRecall 是旧 ContextRecall 的重导出，.name 仍是 "context_recall"
# 而非类名暗示的 "llm_context_recall"——曾因手写键名与实际返回不一致，召回列恒为
# 默认 0.0 且无任何报错。因此读取键一律从实例化指标推导，禁止手写；契约测试
# （test_ragas_eval.py::test_metric_name_contract）把键名双向钉死防漂移。
_OUTPUT_KEYS = ("faithfulness", "relevancy", "recall")  # 对外输出列名，main() 表格与测试依赖
_METRICS = [Faithfulness(), AnswerRelevancy(), LLMContextRecall()]
_NAME_TO_OUTPUT = {m.name: key for m, key in zip(_METRICS, _OUTPUT_KEYS)}
# to_pandas 兜底路径按同一组键从 DataFrame 列取值
_METRIC_KEYS = tuple(_NAME_TO_OUTPUT)


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
    """裁判 LLM：DeepSeek 的 OpenAI 兼容端点。双客户端注入见文件头红线说明。"""
    import httpx
    from langchain_openai import ChatOpenAI

    from src.config import settings

    return ChatOpenAI(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=0.0,
        # ragas 只走异步路径，异步用的是 ChatOpenAI 独立的 http_async_client 字段：
        # 只配同步 http_client 的话真实链路会回落到 trust_env=True 的默认客户端，
        # 代理残留场景（WinError 10061）照样炸，两个都要钉死
        http_client=httpx.Client(trust_env=False, timeout=60.0),
        http_async_client=httpx.AsyncClient(trust_env=False, timeout=60.0),
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

    # KB 命中但生成失败时，pipeline 走部分失败路径：sources 非空却带 error 键返回
    # （pipeline.py 的 chat 收敛逻辑）。这种降级回答不能当有效样本计分，否则指标被污染。
    if result.get("error"):
        return {"ok": False, "error": f"pipeline 报错：{result['error']}"}

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
    ))
    row = {"ok": True}
    for name, out_key in _NAME_TO_OUTPUT.items():
        v = scores.get(name, 0.0)
        # NaN 视为 0：ragas 对无法判分的样本返回 NaN，直接平均会把整列拖成 NaN
        row[out_key] = 0.0 if math.isnan(v) else v
    row["answer"] = result.get("answer", "")
    return row


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

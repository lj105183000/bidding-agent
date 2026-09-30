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

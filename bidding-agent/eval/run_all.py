"""统一评估运行器：一条命令跑全部评测线，产出一份可比对的报告。

★ 为什么需要它
-----------
现有三条评测线各自 `print` 到终端，于是：
    · 想回答"这次改动有没有让某个指标退化"→ 只能人眼比对两段输出
    · 想回答"上周到现在趋势如何"→ 没有任何历史
    · 想在提交前自动拦住退化 → 无从下手

本运行器把三条线的结果收敛成 `EvalReport`（见 `eval/report.py`），
再交给趋势存储与门禁处理。

★ 用法
----
    python eval/run_all.py                      # 全部（含 RAGAS，较慢）
    python eval/run_all.py --suites agent       # 只跑 Agent 线
    python eval/run_all.py --suites agent,retrieval --limit 5
    python eval/run_all.py --save               # 存进趋势文件
    python eval/run_all.py --gate               # 与最近一次比，退化则非零退出
    python eval/run_all.py --json out.json      # 落盘完整报告

★ 降级原则
--------
某条线跑不了（缺依赖、缺 key、外部服务不可用）时**记为 NOT_RUN 并说明原因**，
不中断整轮、也不算作失败。理由：把"没测"当"退化"会频繁误报，
久而久之没人看报告——那比不测更糟。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.common import load_cases, mean, setup_stdio  # noqa: E402
from eval.metrics import (  # noqa: E402
    TOOL_ROUNDS_CAP,
    aggregate,
    extract_process_metrics,
)
from eval.report import (  # noqa: E402
    SCHEMA_VERSION,
    EvalReport,
    SuiteReport,
    make_result,
    render_text,
)

EVAL_DIR = Path(__file__).resolve().parent
AGENT_CASES = EVAL_DIR / "test_cases.json"

#: 各条线的名字。用于 --suites 过滤与报告分组。
ALL_SUITES = ("agent", "retrieval", "rag")


def _git_version() -> str:
    """被测版本标识：git 短哈希 + 脏标记。

    没有它，趋势文件里两行分数无法对应到具体代码——而那正是趋势的意义。
    不在 git 仓库（或没装 git）时返回 "unknown"，不抛异常。
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
            cwd=str(EVAL_DIR.parent), check=False,
        )
        if sha.returncode != 0:
            return "unknown"
        version = sha.stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=10,
            cwd=str(EVAL_DIR.parent), check=False,
        )
        if dirty.returncode == 0 and dirty.stdout.strip():
            version += "-dirty"
        return version or "unknown"
    except Exception:
        return "unknown"


def _runtime_config() -> dict:
    """配置快照。**没有它，两次分数不可比**。

    只放会影响指标的键（模型、检索参数、各功能开关），不放密钥。

    ★ 为什么把裁判模型也算进"会影响指标"的键
    ------------------------------------
    RAG 线的分数由**裁判模型**打出（`RAGAS_JUDGE_MODEL`，默认 GLM）。
    换裁判等于换尺子 —— 实测同一批数据换裁判后分数会整体平移。
    不记进 config，趋势文件里两行 RAGAS 分数就无法判断"是系统变了还是尺子变了"。
    """
    keys = (
        "LLM_PROVIDER", "DEEPSEEK_MODEL", "ZHIPU_MODEL",
        "RAG_PROMPT_VARIANT", "RERANK_ENABLED",
        "AGENT_PLANNING", "AGENT_MEMORY", "AGENT_COMPACT",
        # 裁判（RAG 线专用；不设时留空，由 ragas_eval 的默认值兜底）
        "RAGAS_JUDGE_MODEL", "RAGAS_JUDGE_ATTEMPTS",
    )
    cfg = {k: os.getenv(k, "") for k in keys}
    try:
        from src.agent.constants import MAX_TOOL_ROUNDS
        from src.rag import constants as rag_const

        cfg["MAX_TOOL_ROUNDS"] = MAX_TOOL_ROUNDS
        cfg["DEFAULT_TOP_K"] = getattr(rag_const, "DEFAULT_TOP_K", "")
        cfg["MIN_RELEVANCE"] = getattr(rag_const, "MIN_RELEVANCE", "")
    except Exception:
        pass
    return {k: v for k, v in cfg.items() if v not in ("", None)}


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return float(ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))])


# ===========================================================================
# Agent 线
# ===========================================================================


def _tool_matched(case: dict, result: dict) -> bool:
    """判断工具选择是否算对。

    两种口径：
        ① 精确相等：`tool_name == expected_tool`
           `tool_name` 是"最后一个执行成功的工具"（开发文档 §6.4 的契约，
           eval/agent_eval.py 也依赖它，**不能改**）。
        ② `expected_tools_any`：只要调用了列表里任一工具即算对。

    ★ 为什么需要 ②（实测依据）
    -----------------------
    多工具题上，口径 ① 会误判。实测 #9：
        期望 query_database
        实际序列 ['search_knowledge_graph','query_database','search_knowledge_base']
        tool_name = 'search_knowledge_base'   ← 最后一个成功的
    题目考的是"有没有去业务库查项目清单"，Agent 确实查了（还查了两次），
    但因为它最后又补了一次知识库检索，就被判成错——**这是口径问题不是能力问题**。

    用"任一命中"而不是改成"最后一个"：后者会让 #9 永远无法通过。
    """
    actual = result.get("tool_name", "")
    any_of = case.get("expected_tools_any")
    if isinstance(any_of, list) and any_of:
        called = set(result.get("tool_names") or [])
        return actual in any_of or bool(called & set(any_of))
    return actual == case.get("expected_tool", "")


def run_agent_suite(limit: int = 0, web_mode: str = "off") -> SuiteReport:
    """Agent 线：工具选择准确率 + 延迟 + **过程指标**。

    `web_mode="off"` 是刻意的：联网会引入外部不确定性（搜索结果每次都不同），
    让指标不可比。评测要的是**可复现**地测自家逻辑，故默认关联网。
    """
    suite = SuiteReport(suite="agent")
    try:
        from src.agent import BiddingAgent

        agent = BiddingAgent()
    except Exception as exc:
        suite.note = f"Agent 初始化失败：{type(exc).__name__}: {exc}"
        return suite

    cases = load_cases(AGENT_CASES, required=("question", "expected_tool"))
    if limit:
        cases = cases[:limit]

    # ★ 剔除"当前口径下不可能答对"的用例，并如实计入 note。
    #
    # 实测踩过：`#15 2026 年最新法规` 期望 `search_web`，但 `--web-mode off`
    # **不暴露联网工具**——该题在 off 口径下必然失败，把 Agent 的准确率
    # 从 0.94 拉到 0.89。**这不是 Agent 的缺陷，是评测设置的缺陷**：
    # 用"工具不可用"的口径去测"该用那个工具"的题。
    #
    # 剔除而非记 0 分：0 分会污染指标并指向错误的优化方向。
    # 但必须**如实说明剔了几题**，否则就变成"挑好看的题报数"。
    skipped: list[str] = []
    if web_mode == "off":
        kept = []
        for c in cases:
            if c.get("requires_web"):
                skipped.append(c["question"][:40])
                continue
            kept.append(c)
        cases = kept

    suite.case_count = len(cases)
    if skipped:
        suite.note = f"web_mode={web_mode}；跳过 {len(skipped)} 题需联网用例（当前口径不暴露联网工具）"
    else:
        suite.note = f"web_mode={web_mode}"

    matched = 0
    latencies: list[float] = []
    per_case_metrics = []
    errors: list[str] = []

    for i, case in enumerate(cases, 1):
        try:
            result = agent.chat(
                case["question"],
                web_search_enabled=(web_mode == "on"),
                auto_web_fallback=(web_mode == "auto"),
            )
        except Exception as exc:
            errors.append(f"#{i} {type(exc).__name__}: {exc}")
            per_case_metrics.append(extract_process_metrics({}))
            continue

        # elapsed_ms 为 0 只可能是降级/错误路径（见 agent_eval.py 的说明）。
        # 必须判失败而不是计 0 分：否则延迟均值被往低了拽，系统最糟时指标最漂亮。
        if not result.get("elapsed_ms"):
            errors.append(f"#{i} 走了降级/错误路径（无计时）")
            per_case_metrics.append(extract_process_metrics(result))
            continue

        if _tool_matched(case, result):
            matched += 1
        latencies.append(float(result["elapsed_ms"]))
        per_case_metrics.append(extract_process_metrics(result))
        print(f"  进度 {i}/{len(cases)}", end="\r", flush=True)
    print()

    n = len(cases)
    agg = aggregate(per_case_metrics)
    suite.ran = True
    suite.note = f"web_mode={web_mode}"
    if errors:
        suite.note += f"；{len(errors)} 题异常：{errors[0][:80]}"

    suite.results.append(make_result("agent.tool_accuracy", matched / n if n else None))
    suite.results.append(make_result("agent.latency_p95", _p95(latencies)))
    suite.results.append(make_result("agent.tool_calls_mean", agg["tool_calls_mean"]))
    # ★ 轮次与调用次数**分开报**
    #
    # 调用次数高但轮次低是**正常**的：模型会在同一轮里发多个不同 query
    # 做多路召回（实测：一轮内两个 `search_knowledge_base`）。
    # 两者**同高**才是打转。此前只有一个"撞轮次上限比例"却按调用次数算，
    # 把"一轮内两路召回"误判成"打转"（见 `eval/metrics.py` 的注释）。
    suite.results.append(make_result(
        "agent.rounds_mean",
        agg["rounds_mean"],
        note=f"最大 {agg['rounds_max']} 轮（上限 {TOOL_ROUNDS_CAP}）",
    ))
    suite.results.append(make_result(
        "agent.hit_rounds_cap_rate",
        (agg["hit_rounds_cap"] / n) if n else None,
        note=(
            f"{agg['hit_rounds_cap']}/{n} 题撞上限"
            f"（cap={TOOL_ROUNDS_CAP} 轮；按 tool_timings[].round 最大值判定）"
        ),
    ))
    suite.results.append(make_result(
        "agent.redundant_calls_mean",
        (agg["redundant_calls_total"] / n) if n else None,
        note=(
            f"重复调用合计 {agg['redundant_calls_total']} 次"
            "；**只看工具名不看参数**，一轮内两路不同 query 的检索也会计入"
        ),
    ))
    suite.results.append(make_result(
        "agent.verification_critical_ok_rate",
        agg["verification_critical_ok_rate"],
        note=(
            "未测（本轮没有题目产生核对结果）"
            if agg["verification_critical_ok_rate"] is None
            else f"条款号编造 {agg['unsupported_article_cases']} 题、"
                 f"数字编造 {agg['unsupported_number_cases']} 题"
        ),
    ))
    suite.results.append(make_result(
        "agent.verification_ok_rate",
        agg["verification_ok_rate"],
        note=(
            "未测（本轮没有题目产生核对结果）"
            if agg["verification_ok_rate"] is None
            else f"含法规名未在结果中 {agg['unsupported_law_cases']} 题（低风险）"
        ),
    ))
    suite.results.append(make_result(
        "agent.false_unavailable_rate",
        (agg["false_unavailable_cases"] / n) if n else None,
        note=f"编造工具不可用 {agg['false_unavailable_cases']} 题",
    ))
    suite.results.append(make_result("agent.answer_chars_mean", agg["answer_chars_mean"]))
    return suite


# ===========================================================================
# Retrieval 线
# ===========================================================================


def ranking_metrics(
    per_topic: list[tuple[list[dict], tuple[str, ...]]],
) -> dict:
    """从"每个主题的召回结果"算排序质量指标。**纯函数**（不检索、不联网）。

    参数：`[(docs, keys), ...]` —— 每个主题的 top-k 文档与切题关键词。

    返回：
        `ranked_total`        有结果的**主题**数（不是文档数）
        `top1_ok`             首条即切题的主题数
        `first_ranks`         每个"前 k 条内有切题文档"的主题的首个切题位次
        `no_hit`              前 k 条**全不切题**的主题数

    ★ 为什么抽成纯函数：这两个指标的口径极易写错（分母、位次从 1 还是 0 起、
    "全不切题"算不算分母），而跑一次真检索要连 Qdrant。抽出来就能用构造数据
    把每种情形钉死（见 `tests/test_eval_gaps.py`）。**统计逻辑不该只能靠真跑验证。**
    """
    first_ranks: list[int] = []
    top1_ok = 0
    ranked_total = 0
    for docs, keys in per_topic:
        if not docs:
            continue
        ranked_total += 1
        pos = next(
            (
                i + 1  # 位次从 1 起（人对"第几条"的直觉），不是下标
                for i, d in enumerate(docs)
                if any(k in f"{d.get('question') or ''}{d.get('answer') or ''}" for k in keys)
            ),
            None,
        )
        if pos == 1:
            top1_ok += 1
        if pos is not None:
            first_ranks.append(pos)
    return {
        "ranked_total": ranked_total,
        "top1_ok": top1_ok,
        "first_ranks": first_ranks,
        "no_hit": ranked_total - len(first_ranks),
    }


def run_retrieval_suite(limit: int = 0) -> SuiteReport:
    """检索线：**切题率**（召回的结果有多少真正回答了该主题）。

    ★ 为什么需要它（`rag.context_recall` 测不出的盲区）
    ------------------------------------------------
    `rag.context_recall` 实测 1.000，看起来检索完美。但那个指标衡量的是
    "**期望答案里的信息是否出现在上下文中**"，而评测集与知识库高度重叠
    （`docs/23`：25 题里 8 题近乎原题），故它测不出：

        问「投标保证金」（法定 2% / 80 万上限）
        召回「履约保证金」（10% 上限）—— 都是保证金，答案却不同

    本项用**精确主题词 + 人工切题判据**，刻意不用语义相似度——
    相似度会把「履约保证金」也算成切题，而那正是要抓的问题。

    复用 `eval/retrieval_precision.py` 的 TOPICS 与判据，
    不另写一套：口径必须与那份独立脚本一致，否则两处结论会打架。
    """
    suite = SuiteReport(suite="retrieval")
    try:
        sys.path.insert(0, str(EVAL_DIR))
        from retrieval_precision import TOPICS  # type: ignore[import-not-found]

        from src.rag.pipeline import RAGPipeline
    except Exception as exc:
        suite.note = f"导入失败：{type(exc).__name__}: {exc}"
        return suite

    try:
        pipe = RAGPipeline()
    except Exception as exc:
        suite.note = f"检索流水线初始化失败：{type(exc).__name__}: {exc}"
        return suite

    topics = TOPICS[:limit] if limit else TOPICS
    suite.case_count = len(topics)
    total_recall = total_relevant = 0
    weak: list[str] = []
    errors = 0

    for query, keys in topics:
        try:
            docs = pipe.search(query, top_k=5) or []
        except Exception:
            errors += 1
            continue
        relevant = 0
        for d in docs:
            blob = f"{d.get('question') or ''}{d.get('answer') or ''}"
            if any(k in blob for k in keys):
                relevant += 1
        total_recall += len(docs)
        total_relevant += relevant
        rate = relevant / len(docs) if docs else 0.0
        if docs and rate < 0.6:
            weak.append(query)

    suite.ran = True
    suite.note = f"top-5；{len(weak)} 个主题切题率 <60%"
    if weak:
        suite.note += f"：{weak[:4]}"
    if errors:
        suite.note += f"；{errors} 个主题检索异常"

    rate = (total_relevant / total_recall) if total_recall else None
    suite.results.append(make_result(
        "retrieval.precision", rate,
        note=f"召回 {total_recall} 条，切题 {total_relevant} 条",
    ))
    # ★ 两个**排序质量**指标（此前缺）
    #
    # 为什么需要：`retrieval.precision` 只看"召回的一堆里有多少切题"，
    # **完全不看顺序** —— 把最相关的那条排到第 5 位，precision 一模一样。
    # 而 RAG 的上下文窗口有限，**第一条切不切题**直接决定答案质量。
    #
    # ★ 为什么**不**做 recall@k / MRR（如实说明）
    # --------------------------------------
    # 那需要**完整的相关集**（哪些文档算相关）。本评测用的是
    # `TOPICS = [(query, keys)]` —— 靠关键词判相关，只能判"某条是否相关"，
    # **判不出"一共有多少条相关"**，故 recall 的分母不可知。
    # 硬算出来的"recall"会是伪指标。故这里只做**能诚实算出**的两个：
    #
    #   首位切题率      top-1 那条是否切题
    #   首个切题位次均值  第一个切题的文档排在第几（越接近 1 越好）
    #
    # 统计逻辑在 `ranking_metrics()`（纯函数，可单测）。
    #
    # ★ 两套题都跑（评测集 + 难例集），并**分别出指标**
    # -------------------------------------------------
    # 评测集与语料高度重叠，首位切题率长期 0.9444、位次 1.06，**贴着天花板**；
    # 排序类改动在它上面几乎分辨不出来（只剩 1 题的余量）。
    # 难例集（`eval/hard_topics.HARD_TOPICS`，真实问法）实测 0.7857 / 1.43，
    # 才看得见收益与代价 —— 实测已用它否掉了"开精排"与"纯余弦排序"两个方案。
    # 两套**分别报**而不是合并：合并会把差距抹平，也说不清退化发生在哪种问法上。
    def _run_set(items) -> tuple[list[tuple[list[dict], tuple[str, ...]]], int]:
        docs: list[tuple[list[dict], tuple[str, ...]]] = []
        failed = 0
        for query, keys in items:
            try:
                docs.append((pipe.search(query, top_k=5) or [], keys))
            except Exception:
                failed += 1                # 上面已计过 errors，不重复计
        return docs, failed

    topic_docs, _failed = _run_set(topics)
    rank = ranking_metrics(topic_docs)
    ranked_total = rank["ranked_total"]
    top1_ok = rank["top1_ok"]
    first_ranks = rank["first_ranks"]

    if ranked_total:
        suite.results.append(make_result(
            "retrieval.top1_relevant_rate",
            top1_ok / ranked_total,
            note=f"{top1_ok}/{ranked_total} 个主题的首条结果切题",
        ))
    else:
        suite.results.append(make_result(
            "retrieval.top1_relevant_rate", None,
            note="无可用主题（全部检索异常或返回空）",
        ))

    if first_ranks:
        suite.results.append(make_result(
            "retrieval.first_relevant_rank_mean",
            sum(first_ranks) / len(first_ranks),
            note=(
                f"{len(first_ranks)}/{ranked_total} 个主题在前 5 条内找到切题文档；"
                f"另 {ranked_total - len(first_ranks)} 个**前 5 条全不切题**"
            ),
        ))
    else:
        suite.results.append(make_result(
            "retrieval.first_relevant_rank_mean", None,
            note="没有任何主题在前 5 条内出现切题文档",
        ))

    # ★ 分母陷阱的补丁：把"前 5 条全不切题"的比例单独报成一个**有门槛的指标**。
    #
    # 为什么必须补：`first_relevant_rank_mean` 的分母是"前 5 条内有切题文档的
    # 主题"，于是"一条都没切题"的主题**不会拉高那个均值**。极端情形：
    # 100 个主题里 90 个前 5 条全不切题，只要剩下 10 个首位命中，
    # 位次均值照样是 1.0 —— 看起来完美，实际检索几乎全废。
    # 原来这个情形只写在 note 里（"另 N 个前 5 条全不切题"），
    # 而 **note 不进门槛判定**，等于没人守。单列成比率后门禁会管它。
    if ranked_total:
        no_hit = ranked_total - len(first_ranks)
        suite.results.append(make_result(
            "retrieval.no_relevant_in_topk_rate",
            no_hit / ranked_total,
            note=(
                f"{no_hit}/{ranked_total} 个主题的前 5 条**完全没有**切题文档；"
                "该比例不进 `first_relevant_rank_mean` 的分母，故必须单独看"
            ),
        ))
    else:
        suite.results.append(make_result(
            "retrieval.no_relevant_in_topk_rate", None,
            note="无可用主题（全部检索异常或返回空）",
        ))

    # ★ 难例集：同样三个指标，前缀加 `hard_`（真实问法下的召回/排序）
    #
    # 为什么单列而不是并入上面那三条：两套题的绝对水平差 15 个百分点以上
    # （0.9444 vs 0.7857），合并后既看不出差距，也说不清"退化发生在哪种问法"。
    # 难例集是**排序类改动唯一有分辨力的地方**（评测集已贴天花板）。
    try:
        sys.path.insert(0, str(EVAL_DIR))
        from hard_topics import as_pairs as _hard_pairs  # type: ignore[import-not-found]
        from hard_topics import corpus_gap_note as _gap_note

        hard_items = _hard_pairs()[:limit] if limit else _hard_pairs()
        gap_note = _gap_note()
    except Exception:
        hard_items = []
        gap_note = ""

    if hard_items:
        hard_docs, hard_failed = _run_set(hard_items)
        hard = ranking_metrics(hard_docs)
        h_total = hard["ranked_total"]
        if h_total:
            suite.results.append(make_result(
                "retrieval.hard_top1_relevant_rate",
                hard["top1_ok"] / h_total,
                note=(
                    f"{hard['top1_ok']}/{h_total} 道**真实问法**难例的首条切题"
                    "（问法不含语料原话，故与上面那条不是同一水平）"
                ),
            ))
            if hard["first_ranks"]:
                suite.results.append(make_result(
                    "retrieval.hard_first_relevant_rank_mean",
                    sum(hard["first_ranks"]) / len(hard["first_ranks"]),
                    note=f"{len(hard['first_ranks'])}/{h_total} 道在前 5 条内找到切题文档",
                ))
            else:
                suite.results.append(make_result(
                    "retrieval.hard_first_relevant_rank_mean", None,
                    note="没有任何难例在前 5 条内出现切题文档",
                ))
            suite.results.append(make_result(
                "retrieval.hard_no_relevant_in_topk_rate",
                (h_total - len(hard["first_ranks"])) / h_total,
                note=(
                    f"{h_total - len(hard['first_ranks'])}/{h_total} 道难例的前 5 条"
                    "完全没有切题文档（召回彻底失手，比排序差严重得多）"
                ),
            ))
            suite.note = f"{suite.note}；难例集 {h_total} 题" + (
                f"（{hard_failed} 题检索异常）" if hard_failed else ""
            )
            if gap_note:
                suite.note = f"{suite.note}；{gap_note}"
    return suite


# ===========================================================================
# RAG 线
# ===========================================================================


def run_rag_suite(limit: int = 0) -> SuiteReport:
    """RAG 线：RAGAS 三项指标。

    RAGAS 依赖重、耗时长（25 题约 10 分钟），且需要 LLM。跑不了时记 NOT_RUN。
    """
    suite = SuiteReport(suite="rag")
    try:
        import ragas  # noqa: F401
    except Exception as exc:
        suite.note = f"未安装 RAGAS（{type(exc).__name__}），跳过。装法：uv pip install ragas"
        return suite

    # 复用 ragas_eval 的评测函数而不是另写一套：口径必须与既有基线一致，
    # 否则 baseline.json 里锚定的分数就不可比了。
    #
    # ⚠️ `_run_config` 的签名是 (config, cases, llm, embeddings, *, diagnose_on,
    # threshold)，**不接受 limit**。第一版按 `_run_config(cfg, limit=...)` 调用，
    # 直接 TypeError 被降级成 NOT_RUN——而 NOT_RUN 在报告里只显示"未执行"，
    # 不会让总判定失败，于是**RAG 线悄悄消失了**（实测踩过）。
    # 故这里按 ragas_eval.main() 的组装方式原样复刻。
    try:
        sys.path.insert(0, str(EVAL_DIR))
        from ragas_eval import (  # type: ignore[import-not-found]
            CASES_PATH,
            _BGEEmbeddings,
            _PRESETS,
            _PRODUCTION_CONFIG,
            _run_config,
            build_evaluator_llm,
            judge_provider_matches_generator,
            load_cases,
        )
    except Exception as exc:
        suite.note = f"导入 ragas_eval 失败：{type(exc).__name__}: {exc}"
        return suite

    cfg = _PRESETS.get(_PRODUCTION_CONFIG)
    if cfg is None:
        suite.note = f"生产配置 {_PRODUCTION_CONFIG} 不在预设里，跳过"
        return suite

    try:
        cases = load_cases(CASES_PATH, required=("question", "reference"))
        if limit:
            cases = cases[:limit]
        judge = build_evaluator_llm()
        run = _run_config(
            cfg, cases, judge, _BGEEmbeddings(),
            diagnose_on=False, threshold=0.0,
        )
    except Exception as exc:
        suite.note = f"RAGAS 运行失败：{type(exc).__name__}: {exc}"
        return suite

    suite.ran = True

    summary = run.get("summary") or {}
    # ★ `case_count` 用**数据集规模**（本线跑了多少题），不是"有效判分题数"
    # --------------------------------------------------------------
    # `summary["n"]` 是**有效**例数（判分失败的不算），而它随裁判抖动而变：
    # 同一套 25 题，实测一次 21 题有效（4 题判分失败）、下一次可能是 23 题。
    #
    # 而 `case_count` 是**可比性判据**（`compare_with_previous` 用它判断
    # "两次是不是同一批样本"）—— 用有效数会让裁判抖一次就触发
    # "⚠️ 样本量不一致 → 不据分数判退化"，等于**悄悄把 rag 线的门禁关掉**。
    # 有效/失败数写进 note（下面那行），信息一点没少，只是不再污染可比性口径。
    suite.case_count = len(cases)
    suite.note = (
        f"配置 {_PRODUCTION_CONFIG}；有效 {run.get('valid')}/{run.get('total')}"
        f"（error {run.get('errors')}，计入分母；case_count 记为数据集规模 {len(cases)}）"
    )
    # ★ 裁判模型与"是否同源"追加到 note（**必须放在上面那次赋值之后**）
    #
    # `_run_config` 不产出这两项，而它们是**解读分数可信度的前提**：
    # 裁判与生成同源时存在自评偏差（self-preference bias），分数系统性偏高——
    # 不写出来，看聚合报告的人无从判断这组 RAGAS 分数能否当独立证据。
    #
    # ⚠️ 位置有讲究：第一版放在 `suite.ran = True` 之后、这条赋值之前，
    # 结果**被整条覆盖**（实测 note 里完全没有裁判信息）。
    # 故必须在最后一次 `suite.note = ...` 之后追加。
    try:
        _same, _gen, _judge = judge_provider_matches_generator()
        _bias = (
            f"⚠️ 裁判与生成同源（{_gen}），存在自评偏差"
            if _same else f"裁判 {_judge} / 生成 {_gen}（不同源）"
        )
        suite.note = f"{suite.note}｜裁判模型 {judge.model_name}｜{_bias}"
    except Exception:  # noqa: BLE001 - 报告注记失败不该让评测失败
        pass

    # ⚠️ 键名要用 `ragas_eval._OUTPUT_KEYS` 里的**短名**，不是 RAGAS 的类名。
    # 实测踩过：按 `answer_relevancy` / `llm_context_recall` 去匹配，
    # 只有 `faithfulness` 能对上，另两项静默变成"未测"——
    # 而"未测"**不会**让总判定失败，于是 RAG 线看起来跑通了、实际只测了一项。
    mapping = (
        ("faithfulness", "rag.faithfulness"),
        ("relevancy", "rag.answer_relevancy"),
        ("recall", "rag.context_recall"),
    )
    for src_key, metric_name in mapping:
        raw = None
        for k, v in summary.items():
            if src_key in str(k).lower().replace("-", "_"):
                raw = v
                break
        # ★ `summarize()` 返回的是**嵌套结构** `{mean, median, min, zeros}`
        # （见 ragas_eval.summarize 的说明：只报均值会掩盖"少数题拿 0 分"）。
        # 第一版直接把整个 dict 传给 `make_result`，于是 `value >= threshold`
        # 拿 dict 和 float 比 → TypeError 崩在报告组装处。
        # 这里取 `mean`，并**显式判类型**：非数值一律记 None（= 未测），
        # 不让任何形状变化把整个评估打挂。
        value = None
        if isinstance(raw, dict):
            candidate = raw.get("mean")
            value = float(candidate) if isinstance(candidate, (int, float)) else None
            zeros = raw.get("zeros")
            note = f"mean；中位 {raw.get('median')}、最低 {raw.get('min')}、拿 0 分 {zeros} 题"
        elif isinstance(raw, (int, float)):
            value = float(raw)
            note = ""
        else:
            note = "未能从 summary 里取到该指标"
        suite.results.append(make_result(metric_name, value, note=note))
    return suite


# ===========================================================================
# 主流程
# ===========================================================================


def build_report(
    suites: list[str], limit: int, web_mode: str, repeat: int = 1
) -> EvalReport:
    """跑一遍（或 N 遍取中位）并组装报告。

    ★ `repeat` 为什么在这层实现（而不是各 run_*_suite 内部）
    --------------------------------------------------
    因为"取中位"是**跨整条线**的操作：要先拿到 N 份同结构的 `SuiteReport`
    才能按指标名对齐求中位。放在各 suite 内部就得重复实现 N 遍聚合逻辑。

    ★ `repeat > 1` 的行为
    ------------------
    每条线跑 N 次，然后按**指标名**取中位（见 `eval/repeat.py`），
    并把 `repeat` 记进 `config` —— 否则"3 次中位"与"1 次实测"
    在趋势文件里长得一样，门禁会把它们当可比。
    """
    report = EvalReport(
        schema_version=SCHEMA_VERSION,
        run_id=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        version=_git_version(),
        config=_runtime_config(),
    )
    if repeat > 1:
        # ★ 必须进 config：趋势文件按 config 判可比，
        #   否则"3 次中位"会被当成基线去比"1 次实测"。
        report.config["REPEAT"] = repeat
    # ★ 数据集登记与重叠核算（`eval/dataset_registry.py`）：结论必须带着
    #   "这是在什么数据上测的"一起出现 —— 三条线用的是三份不同的数据，
    #   跨集分数不可直接比。取不到时留空 dict（报告里就不打这一节）。
    try:
        from eval.dataset_registry import summary as _datasets_summary

        report.datasets = _datasets_summary()
    except Exception as exc:  # noqa: BLE001 - 数据登记失败不该让评测跑不起来
        print(f"（数据集登记读取失败：{type(exc).__name__}: {exc}）")

    # 各线的 runner（顺序即打印顺序）
    runners: list[tuple[str, str, object]] = []
    if "agent" in suites:
        runners.append((
            "agent", "[1/3] 跑 Agent 线…",
            lambda: run_agent_suite(limit=limit, web_mode=web_mode),
        ))
    if "retrieval" in suites:
        runners.append((
            "retrieval", "[2/3] 跑 检索线…",
            lambda: run_retrieval_suite(limit=limit),
        ))
    if "rag" in suites:
        runners.append((
            "rag", "[3/3] 跑 RAG 线…",
            lambda: run_rag_suite(limit=limit),
        ))

    for name, banner, run in runners:
        print(banner)
        if repeat <= 1:
            report.suites.append(run())  # type: ignore[operator]
            continue
        # ---- 重复采样 ----
        from eval.repeat import merge_suite_runs

        collected = []
        for i in range(repeat):
            print(f"    第 {i + 1}/{repeat} 次…")
            collected.append(run())  # type: ignore[operator]
        merged = merge_suite_runs(collected)
        print(f"    {name} 已按 {repeat} 次取中位")
        report.suites.append(merged)

    return report


def parse_suites(raw: str) -> list[str]:
    """解析 --suites，**空与未知都报错**。

    ★ 为什么抽成独立函数（它此前内联在 main 里）
    --------------------------------------
    内联时只能靠**启动子进程**来测，而那会真的跑评测
    （实测：--suites agent, --limit 0 里的 --limit 0 是不限，
    于是跑完整个 agent 套件直到超时）。抽出来后校验变成纯字符串逻辑，
    测试毫秒级完成。

    ★ 为什么空要**单独**报错（原先不报）
    ------------------------------
    "".split(",") → [""] → 被 if s.strip() 滤成 []，
    于是**逃过 unknown 校验**，然后：

        build_report([]) → 没有任何 suite
        报告打印「总体：✅ 通过（0 项未测）」、退出码 0

    这是**最危险的一类缺陷**：什么都没测的流水线**报成功**。
    CI 里若写 --suites "" 而变量为空，会得到
    全绿但零覆盖，且没有任何迹象。

    空与未知**成因不同**，提示词要分开，否则误导排查方向。
    """
    suites = [s.strip() for s in (raw or "").split(",") if s.strip()]
    if not suites:
        raise SystemExit(
            f"--suites 不能为空（收到 {raw!r}）。"
            f"可选：{list(ALL_SUITES)}；不传则默认全部。"
        )
    unknown = [s for s in suites if s not in ALL_SUITES]
    if unknown:
        raise SystemExit(f"未知的评测线：{unknown}（可选：{list(ALL_SUITES)}）")
    return suites


def parse_suite_list(raw: str, flag: str) -> list[str]:
    """解析 `--fail-on` / `--allow-fail` 的线名列表。**空 = 未指定（None 语义）**。

    与 `--suites` 不同：这里空串是合法的（表示"没给这个开关"），
    但给了就必须是已知的线名 —— 写错一个线名会让整个开关**静默失效**
    （比如 `--fail-on retrival` 拼错 → 谁都没被选中 → 门禁永远绿），
    那正是门禁最不该有的失败方式。
    """
    value = (raw or "").strip()
    if not value:
        return []
    items = [s.strip() for s in value.split(",") if s.strip()]
    if not items:
        return []
    unknown = [s for s in items if s not in ALL_SUITES]
    if unknown:
        raise SystemExit(
            f"{flag} 里有未知的评测线：{unknown}（可选：{list(ALL_SUITES)}）"
        )
    return items


def main() -> int:
    setup_stdio()
    parser = argparse.ArgumentParser(description="统一 Agent 评估体系")
    parser.add_argument("--suites", default="agent,retrieval,rag",
                        help="要跑的线，逗号分隔（默认全部）")
    parser.add_argument("--limit", type=int, default=0, help="每条线只用前 N 条用例")
    parser.add_argument(
        "--repeat", type=int, default=1, metavar="N",
        help=(
            "每条线跑 N 次、按指标取中位（默认 1）。"
            "实测 agent 线的指标在同代码同配置下可波动 2~3 倍，"
            "而门禁容差只有 2%% —— 单次运行会结构性地产出假阳性。"
            "N 会记进 config，故「3 次中位」与「1 次实测」在趋势里不会混为可比。"
        ),
    )
    parser.add_argument("--web-mode", default="off", choices=("off", "on", "auto"),
                        help="联网模式。默认 off：联网引入外部不确定性，让指标不可比")
    parser.add_argument("--json", default="", help="把完整报告写到指定文件")
    parser.add_argument("--save", action="store_true", help="存入趋势文件")
    parser.add_argument("--gate", action="store_true", help="与最近一次比，退化则非零退出")
    parser.add_argument("--tolerance", type=float, default=0.02,
                        help="门禁容差（默认 0.02，避免浮点与正常波动误报）")
    # ---- 分线退出码（拆 CI job 用）----
    parser.add_argument(
        "--fail-on", default="",
        help="只有这些线的未达标/退化会让退出码非零（逗号分隔）。"
             "配 `--suites X --gate --fail-on X` 可让每条线独立红",
    )
    parser.add_argument(
        "--allow-fail", default="",
        help="这些线不参与退出码判定（逗号分隔），其余线照常判定",
    )
    parser.add_argument(
        "--gate-json", default="",
        help="把门禁决策（退出码 + 分线结论）写成 JSON，供 CI 读取",
    )
    parser.add_argument(
        "--trend", default="",
        help="趋势文件路径（默认 eval/trend.jsonl）。CI 里按线/按分支隔离历史时用",
    )
    parser.add_argument(
        "--list-suites", action="store_true",
        help="只打印评测线清单（JSON）后退出，供 CI 生成 matrix",
    )
    parser.add_argument(
        "--calibrate", action="store_true",
        help="只打印门槛标定结论（读趋势历史）后退出",
    )
    args = parser.parse_args()

    # ---- 两个"只看不跑"的快路径：CI 的 plan 阶段与门槛巡检用 ----
    if args.list_suites:
        print(json.dumps({"suites": list(ALL_SUITES)}, ensure_ascii=False))
        return 0
    if args.calibrate:
        from eval.calibrate import calibration, render as render_calibration
        from eval.trend import load_history

        trend_path = Path(args.trend) if args.trend else None
        print(render_calibration(calibration(load_history(trend_path))))
        return 0

    fail_on = parse_suite_list(args.fail_on, "--fail-on")
    allow_fail = parse_suite_list(args.allow_fail, "--allow-fail")
    if fail_on and allow_fail:
        raise SystemExit(
            "--fail-on 与 --allow-fail 不能同时使用："
            "一个说「只看这些线」、另一个说「除了这些线都看」，口径会自相矛盾"
        )

    suites = parse_suites(args.suites)

    # ★ 空套件必须报错，不能"零条线却报通过"
    #
    # 实测：`--suites ""` 时 `"".split(",")` → `[""]` → 被 `if s.strip()` 滤成 `[]`，
    # 于是**逃过上面的 unknown 校验**，然后：
    #     build_report([], ...)  → 没有任何 suite
    #     报告打印「总体：✅ 通过（0 项未测）」
    #     退出码 0
    #
    # 这是**最危险的一类缺陷**：一个什么都没测的流水线**报成功**。
    # 若 CI 用它（如脚本里 `--suites "$SUITES"` 而变量为空），
    # 会得到"全绿但零覆盖"，而且没有任何迹象。
    #
    # 故显式拦住，并说明是"空"而不是"未知"（两者成因不同，提示要给对）。
    if not suites:
        raise SystemExit(
            f"--suites 不能为空（收到 {args.suites!r}）。"
            f"可选：{list(ALL_SUITES)}；不传则默认全部。"
        )

    started = time.perf_counter()
    report = build_report(suites, max(args.limit, 0), args.web_mode, repeat=args.repeat)
    print()
    print(render_text(report))
    print(f"耗时 {time.perf_counter() - started:.1f}s")

    if args.json:
        Path(args.json).write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"报告已写入 {args.json}")

    comparison = None
    trend_path = Path(args.trend) if args.trend else None

    if args.save or args.gate:
        from eval.trend import (
            append_report,
            compare_with_previous,
            load_history,
            render_exit_decision,
        )

        history = load_history(trend_path)
        if args.gate and history:
            comparison = compare_with_previous(report, history, tolerance=args.tolerance)
            # ★ 按线填独立结论：让门禁输出能指出**是哪条线**没过。
            #   只有全局一个布尔值时，CI 只说评测没过，
            #   而 RAG 与 Agent 的修法完全不同（检索/生成 vs 编排）。
            comparison.suite_verdicts = {
                sr.suite: (sr.verdict if sr.ran else "not_run")
                for sr in report.suites
            }
            print()
            print(comparison.render())
        elif args.gate:
            print("\n门禁：没有历史记录可比，本次只做存档。")
            print("      建基线：`python eval/run_all.py --suites "
                  f"{','.join(suites)} --save`（不带 --gate），之后才有得比。")

        append_report(report, trend_path)
        print(f"已存入趋势文件（累计 {len(load_history(trend_path))} 次运行）")

    # ★ 分线退出码：把"哪些线算数"从全局收窄到分线，这是拆 CI job 的前提。
    #   决策放在 `eval.trend.decide_exit_code`（纯函数），便于单测覆盖。
    from eval.trend import decide_exit_code, render_exit_decision

    exit_code, detail = decide_exit_code(
        report, comparison, fail_on=fail_on, allow_fail=allow_fail
    )
    print(render_exit_decision(detail))

    if args.gate_json:
        Path(args.gate_json).write_text(
            json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"门禁决策已写入 {args.gate_json}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

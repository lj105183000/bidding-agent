"""评测数据集登记表：哪条线用哪份数据、两套集重叠多少、分数能不能直接比。

★ 为什么需要它（三个实测暴露的坑）
============================
    ① **文件命名反直觉**：Agent 线用 `test_cases.json`，RAG 线用
       `test_questions.json` —— "cases" 与 "questions" 并不告诉你哪条线用哪个；
       而检索线**根本没有文件**（`retrieval_precision.TOPICS` 直接内联在 .py 里）。
       结果是"这次评测用的是哪套数据"要靠读代码才知道。
    ② **两套集不是同一套**：实测 Agent 18 条、RAG 25 条，**交集只有 12 条**。
       两套集里"独有题"的难度分布不同，所以**跨集的分数不可直接比**
       （"RAG 忠实度 0.79"与"Agent 准确率 1.00"是在不同题目上测出来的）。
       而报告里此前**没有任何提示**。
    ③ **数据集会被改**：加一道题、改一条 reference，历史分数立刻不可比，
       但趋势文件里只记了分数，没有记录"当时那份数据长什么样"。

故本模块把这三件事显式化，并由 `run_all` 写进报告 ——
**结论必须和"在哪套数据上测的"一起出现**，否则数字会被过度解读。

★ 为什么不做"统一成一套集"
=====================
两套集的**用途不同**，不该合并：
    · Agent 集测的是**工具路由**（每题标 `expected_tool`），
      题目要覆盖各类工具，很多题不需要标准答案
    · RAG 集测的是**答案质量**（每题必须有 `reference` 供 RAGAS 比对），
      题目要能写出明确参考答案
强行合并会得到一份"两边都不合适"的数据：要么缺 expected_tool，
要么缺 reference。故登记 + 标注重叠，而不是合并。

★ 为什么这个文件**不叫** `datasets.py`（实测踩到的坑，改名原因）
=====================================================
它原来就叫 `eval/datasets.py`。而 `run_all.run_retrieval_suite()` 会执行
`sys.path.insert(0, str(EVAL_DIR))`（为了 `import retrieval_precision`），
于是 **`eval/` 目录本身进了模块搜索路径**。后果：

    `sentence_transformers` → `from datasets import Dataset`
       → 解析到 **eval/datasets.py**（我们的登记表）
       → ImportError: cannot import name 'Dataset'

表现为**检索线 18 个主题全部"检索异常"**、四条指标全是"未测"，
而报告只轻描淡写地写"18 个主题检索异常"—— 原因（一次 import 撞名）
离现象（检索失败）很远，极难排查。实测就是这么踩的。

故改名 `dataset_registry.py`：**`eval/` 里不得出现与已安装第三方包同名的模块**。
`tests/test_eval_gaps.py::test_eval_modules_do_not_shadow_third_party_packages`
把这条约束钉成了测试（用 `importlib.metadata.packages_distributions()` 对照）。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class DatasetInfo:
    """一份评测数据的登记信息。"""

    suite: str
    label: str
    #: 数据来源：文件路径（`path`）或代码内联（`inline`）
    source_kind: str
    #: 文件型数据的路径；内联数据为 None
    path: Path | None
    #: 这道数据是干什么的（写清用途，避免被跨集比较）
    description: str
    #: 每题必带的字段
    key_fields: tuple[str, ...] = ()


#: 评测线 → 数据登记。**新增评测线必须在这里登记**，否则报告里会出现
#: "未知数据来源"，而那正是本模块要消灭的状态。
DATASETS: dict[str, DatasetInfo] = {
    "agent": DatasetInfo(
        suite="agent",
        label="Agent 工具路由集",
        source_kind="file",
        path=EVAL_DIR / "test_cases.json",
        description="每题标 expected_tool，测工具选择是否符合预期；含少量多工具题",
        key_fields=("question", "expected_tool"),
    ),
    "retrieval": DatasetInfo(
        suite="retrieval",
        label="检索切题主题集",
        source_kind="inline",
        path=EVAL_DIR / "retrieval_precision.py",
        description=(
            "内联在 retrieval_precision.TOPICS 里的 (query, 切题关键词) 对；"
            "**没有独立文件**，改题目要改代码"
        ),
        key_fields=("query", "keys"),
    ),
    "rag": DatasetInfo(
        suite="rag",
        label="RAG 答案质量集",
        source_kind="file",
        path=EVAL_DIR / "test_questions.json",
        description="每题带 reference，供 RAGAS 三项指标比对；不标期望工具",
        key_fields=("question", "reference"),
    ),
}


def _load_json_list(path: Path) -> list[dict]:
    """读 JSON 数组；失败返回空列表（数据集缺失不该让整个评估崩）。"""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return []
    if isinstance(data, list):
        return [c for c in data if isinstance(c, dict)]
    return []


def questions_of(suite: str) -> list[str]:
    """取某条线的题目文本（去空白）。取不到返回空列表。"""
    info = DATASETS.get(suite)
    if info is None:
        return []
    if suite == "retrieval":
        # 内联数据：`retrieval_precision.TOPICS = [(query, keys), ...]`
        try:
            import sys

            if str(EVAL_DIR) not in sys.path:
                sys.path.insert(0, str(EVAL_DIR))
            from retrieval_precision import TOPICS  # type: ignore[import-not-found]

            return [str(q).strip() for q, _keys in TOPICS]
        except Exception:  # noqa: BLE001
            import os
            import traceback

            if os.getenv("EVAL_DATASETS_DEBUG"):
                traceback.print_exc()
            return []
    if info.path is None:
        return []
    out: list[str] = []
    for case in _load_json_list(info.path):
        q = str(case.get("question") or "").strip()
        if q:
            out.append(q)
    return out


def references_of(suite: str) -> dict[str, str]:
    """题目 → reference（没有 reference 的题不入表）。"""
    info = DATASETS.get(suite)
    if info is None or info.path is None or suite == "retrieval":
        return {}
    out: dict[str, str] = {}
    for case in _load_json_list(info.path):
        q = str(case.get("question") or "").strip()
        ref = case.get("reference")
        if q and isinstance(ref, str):
            out[q] = ref.strip()
    return out


def fingerprint(path: Path | None) -> dict:
    """文件指纹：内容哈希前 12 位 + 字节数。

    ★ 为什么记哈希而不是记 mtime：mtime 在 checkout / 复制后会变，
    但内容没变——那种"看起来数据变了"的假信号会让人无谓地怀疑历史分数。
    内容哈希只在**真的改了题**时才变。

    ★ 为什么这里**不**返回条数（`cases`）
    -----------------------------------
    调用方（`summary`）已经用 `len(questions_of(...))` 数过条数了。
    第一版这里也返回 `cases`，于是 `{"cases": len(qs), **fingerprint(...)}`
    里指纹的 `0` **把真实条数覆盖掉** —— 报告上显示"检索线 0 条"，
    看起来像题目丢了。字典展开的同名键覆盖是静默的，故这里索性不产出该键。
    """
    if path is None or not path.exists():
        return {"sha256_12": "", "bytes": 0}
    try:
        raw = path.read_bytes()
    except Exception:  # noqa: BLE001
        return {"sha256_12": "", "bytes": 0}
    return {
        "sha256_12": hashlib.sha256(raw).hexdigest()[:12],
        "bytes": len(raw),
    }


def overlap(suite_a: str, suite_b: str) -> dict:
    """两份数据的重叠核算。

    返回 `{shared, only_a, only_b, jaccard, reference_mismatch, mismatch_examples}`。

    `reference_mismatch` 值得单独看：同一条问题在两套集里若给出**不同的参考答案**，
    那么"横向比两边的分数"就是拿两把尺子量同一根棍子 —— 必须先统一参考答案。
    """
    qa, qb = questions_of(suite_a), questions_of(suite_b)
    sa, sb = set(qa), set(qb)
    shared = sa & sb
    union = sa | sb
    refs_a, refs_b = references_of(suite_a), references_of(suite_b)
    mismatched = sorted(
        q for q in shared if q in refs_a and q in refs_b and refs_a[q] != refs_b[q]
    )
    return {
        "pair": [suite_a, suite_b],
        "count_a": len(sa),
        "count_b": len(sb),
        "shared": len(shared),
        "only_a": len(sa - sb),
        "only_b": len(sb - sa),
        "jaccard": round(len(shared) / len(union), 4) if union else 0.0,
        "reference_mismatch": len(mismatched),
        "mismatch_examples": mismatched[:3],
    }


def summary() -> dict:
    """给报告用的一段：登记信息 + 两两重叠 + 必须随结论一起读的提醒。"""
    entries: dict[str, dict] = {}
    for suite, info in DATASETS.items():
        questions = questions_of(suite)
        entries[suite] = {
            "label": info.label,
            "source_kind": info.source_kind,
            "source": str(info.path.relative_to(EVAL_DIR.parent))
            if info.path is not None
            else "",
            "description": info.description,
            "key_fields": list(info.key_fields),
            "cases": len(questions),
            **fingerprint(info.path if info.source_kind == "file" else None),
        }

    pairs = [
        overlap(a, b)
        for i, a in enumerate(DATASETS)
        for b in list(DATASETS)[i + 1:]
    ]

    caveats: list[str] = []
    for pair in pairs:
        if pair["count_a"] and pair["count_b"] and pair["shared"]:
            caveats.append(
                f"「{pair['pair'][0]}」与「{pair['pair'][1]}」两套数据只有 "
                f"{pair['shared']} 条重合（各 {pair['count_a']}/{pair['count_b']} 条），"
                "**两条线的分数不可直接横向比较**（题目难度分布不同）"
            )
        if pair["reference_mismatch"]:
            caveats.append(
                f"「{pair['pair'][0]}」与「{pair['pair'][1]}」有 "
                f"{pair['reference_mismatch']} 条同题但参考答案不同，"
                "横向比分数前必须先统一参考答案"
            )
    return {
        "datasets": entries,
        "overlaps": pairs,
        "caveats": caveats,
    }


def render(data: dict | None = None) -> str:
    """渲染成终端可读的一段。"""
    data = data or summary()
    lines = ["-" * 78, "评测数据集", "-" * 78]
    for suite, info in sorted(data.get("datasets", {}).items()):
        src = info.get("source") or "（内联在代码里）"
        fp = info.get("sha256_12") or "—"
        lines.append(
            f"  {suite:<10} {info.get('label', ''):<12} {info.get('cases', 0):>3} 条"
            f"  ← {src}  [{info.get('source_kind')}, sha256:{fp}]"
        )
    for pair in data.get("overlaps", []):
        a, b = pair["pair"]
        lines.append(
            f"  重叠 {a} ∩ {b}：{pair['shared']} 条"
            f"（仅 {a} {pair['only_a']} 条 / 仅 {b} {pair['only_b']} 条，"
            f"Jaccard {pair['jaccard']}）"
            + (f"｜⚠️ 同题不同参考答案 {pair['reference_mismatch']} 条"
               if pair.get("reference_mismatch") else "")
        )
    for caveat in data.get("caveats", []):
        lines.append(f"  ⚠️  {caveat}")
    lines.append("-" * 78)
    return "\n".join(lines)


__all__ = [
    "DATASETS",
    "DatasetInfo",
    "fingerprint",
    "overlap",
    "questions_of",
    "references_of",
    "render",
    "summary",
]


if __name__ == "__main__":  # 手动查看：python -m eval.dataset_registry
    import sys

    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    print(render())

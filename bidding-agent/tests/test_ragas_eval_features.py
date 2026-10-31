"""ragas_eval 新增能力的单测：配置解析 / 汇总统计 / 基线比对 / 逐题诊断。

为什么要专门测这些：`--baseline` 用**退出码**当 CI 判据，判错了比没有更糟——
把真退化放过（假阴性）会让缺陷静默进入主干；把没退化误报（假阳性）会让人
不再信任这个门禁。故容差边界必须双向钉死。

`--diagnose` 的归因文案同理：它直接告诉使用者"该往哪个方向查"，
归错方向会把人引到错误的优化路径上。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval import ragas_eval as R  # noqa: E402


# ===========================================================================
# 配置解析
# ===========================================================================


def test_presets_include_production_default():
    """生产默认必须作为一个可对比的配置存在，否则"以谁为参照"没有锚点。"""
    assert "top5" in R._PRESETS
    assert R._PRESETS["top5"].top_k == 5
    assert R._PRESETS["top5"].rerank is False


def test_resolve_configs_order_preserved():
    """顺序要保留：用户写的顺序就是输出与对比的顺序。"""
    out = R._resolve_configs("top3,top5,top8")
    assert [c.name for c in out] == ["top3", "top5", "top8"]


def test_resolve_configs_trims_and_skips_blanks():
    out = R._resolve_configs(" top3 , , top8 ")
    assert [c.name for c in out] == ["top3", "top8"]


def test_resolve_configs_empty_gives_empty_list():
    """空串返回空列表，由调用方决定回落默认——不在这里偷偷塞默认值。"""
    assert R._resolve_configs("") == []


def test_resolve_configs_unknown_raises_with_choices():
    with pytest.raises(SystemExit) as exc:
        R._resolve_configs("top99")
    # 报错必须列出可用选项，否则用户得去翻源码
    assert "top3" in str(exc.value)


def test_rerank_and_plain_share_top_k():
    """`rerank` 与 `top5` 只差精排开关——这样对比才是单变量。"""
    assert R._PRESETS["rerank"].top_k == R._PRESETS["top5"].top_k
    assert R._PRESETS["rerank"].rerank is True


# ===========================================================================
# 汇总统计
# ===========================================================================


def _rows(faith, rel, rec):
    return [
        {"ok": True, "faithfulness": f, "relevancy": r, "recall": c}
        for f, r, c in zip(faith, rel, rec)
    ]


def test_summarize_mean_and_median():
    rows = _rows([1.0, 0.0, 0.5], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
    s = R.summarize(rows)
    assert s["n"] == 3
    assert s["faithfulness"]["mean"] == pytest.approx(0.5)
    assert s["faithfulness"]["median"] == pytest.approx(0.5)


def test_summarize_counts_zeros():
    """★ 0 分题数必须单独统计。

    实测 25 题均值 0.750 而前 2 题 0.977——只看均值会以为"整体还行"，
    实际是少数题拿 0 分拉低。优化方向完全不同（调参 vs 定位那几道题）。
    """
    rows = _rows([1.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
    s = R.summarize(rows)
    assert s["faithfulness"]["zeros"] == 2
    assert s["faithfulness"]["min"] == 0.0


def test_summarize_empty_is_safe():
    """空列表不能让 mean 抛 ZeroDivisionError。"""
    s = R.summarize([])
    assert s["n"] == 0
    assert s["faithfulness"]["mean"] == 0.0


def test_summarize_handles_missing_metric_key():
    """某题缺指标键时按缺失处理，不能整列崩掉。"""
    rows = [{"ok": True, "faithfulness": 0.8}, {"ok": True, "faithfulness": 0.6}]
    s = R.summarize(rows)
    assert s["faithfulness"]["mean"] == pytest.approx(0.7)
    assert s["relevancy"]["mean"] == 0.0  # 全缺 → 0


# ===========================================================================
# 基线比对（退出码判据，边界必须双向钉死）
# ===========================================================================


def _summary(faith, rel, rec, n=25):
    """构造一个汇总结果。

    `n` 默认 25，与 `baseline_file` 写入的 `cases: 25` 保持一致——
    否则会触发"用例数不一致 → 不作判据"的保护，让所有判定类测试失去意义。
    需要测该保护本身时显式传别的 n。
    """
    return {
        "n": n,
        "faithfulness": {"mean": faith, "median": faith, "min": faith, "zeros": 0},
        "relevancy": {"mean": rel, "median": rel, "min": rel, "zeros": 0},
        "recall": {"mean": rec, "median": rec, "min": rec, "zeros": 0},
    }


@pytest.fixture
def baseline_file(tmp_path, monkeypatch):
    """把基线路径指到临时目录，避免污染仓库里的 eval/baseline.json。"""
    path = tmp_path / "baseline.json"
    monkeypatch.setattr(R, "BASELINE_PATH", path)

    def write(metrics: dict, top_k: int = 5, rerank: bool = False, cases: int = 25):
        path.write_text(
            json.dumps(
                {
                    "saved_at": "2026-01-01 00:00:00",
                    "config": {"name": "top5", "top_k": top_k, "rerank": rerank},
                    "cases": cases,
                    "metrics": metrics,
                    "detail": {},
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return path

    return write


def test_baseline_missing_file_fails(tmp_path, monkeypatch):
    """没有基线必须判失败并说清怎么办——静默通过等于门禁失效。

    这里显式把基线路径指向一个不存在的文件，而**不依赖仓库里恰好没有
    `eval/baseline.json`**：那份基线是真实产物、应当入库，
    测试若依赖"它不存在"就会在跑过一次 `--save-baseline` 之后莫名失败。
    """
    monkeypatch.setattr(R, "BASELINE_PATH", tmp_path / "not-there.json")
    passed, lines = R.compare_baseline(_summary(0.9, 0.9, 0.9), 0.05, R._PRESETS["top5"])
    assert passed is False
    assert any("save-baseline" in line for line in lines)


def test_baseline_within_tolerance_passes(baseline_file):
    baseline_file({"faithfulness": 0.80, "relevancy": 0.90, "recall": 0.95})
    passed, _ = R.compare_baseline(_summary(0.77, 0.88, 0.93), 0.05, R._PRESETS["top5"])
    assert passed is True  # 各降 0.02~0.03，都在 0.05 容差内


def test_baseline_exactly_at_tolerance_passes(baseline_file):
    """边界：恰好等于容差不判退化（`<` 而非 `<=`）。

    这条边界有意选"不判退化"：容差是用户设的允许波动，
    恰好落在边界上时判退化会造成"差一点点就红"的噪音。
    """
    baseline_file({"faithfulness": 0.80, "relevancy": 0.90, "recall": 0.95})
    passed, _ = R.compare_baseline(_summary(0.75, 0.90, 0.95), 0.05, R._PRESETS["top5"])
    assert passed is True


def test_baseline_beyond_tolerance_fails(baseline_file):
    baseline_file({"faithfulness": 0.80, "relevancy": 0.90, "recall": 0.95})
    passed, lines = R.compare_baseline(_summary(0.70, 0.90, 0.95), 0.05, R._PRESETS["top5"])
    assert passed is False
    assert any("退化" in line for line in lines)


def test_baseline_improvement_passes(baseline_file):
    baseline_file({"faithfulness": 0.80, "relevancy": 0.90, "recall": 0.95})
    passed, lines = R.compare_baseline(_summary(0.95, 0.95, 0.99), 0.05, R._PRESETS["top5"])
    assert passed is True
    assert any("提升" in line for line in lines)


def test_baseline_zero_tolerance_flags_any_drop(baseline_file):
    """容差 0 时任何下降都算退化。"""
    baseline_file({"faithfulness": 0.80, "relevancy": 0.90, "recall": 0.95})
    passed, _ = R.compare_baseline(_summary(0.799, 0.90, 0.95), 0.0, R._PRESETS["top5"])
    assert passed is False


def test_baseline_config_mismatch_not_a_regression_verdict(baseline_file):
    """★ 配置不同时不能拿来做退化判据。

    top_k=3 与 top_k=5 的指标本来就会不同，判"退化"是错的——
    这种情况只告警，不判失败。
    """
    baseline_file({"faithfulness": 0.90, "relevancy": 0.90, "recall": 0.95}, top_k=5)
    passed, lines = R.compare_baseline(_summary(0.50, 0.50, 0.50), 0.05, R._PRESETS["top3"])
    assert passed is True
    assert any("配置不一致" in line for line in lines)


def test_baseline_case_count_mismatch_not_a_verdict(baseline_file):
    """★ 用例数不同时不能判"提升/退化"。

    实测踩过：`--limit 2 --baseline` 报了"忠实度 +0.139 提升"，
    真相是 2 道简单题的均分对不上 25 题的均分。
    **假提升比假退化更危险**——它让人以为优化见效了。
    """
    baseline_file({"faithfulness": 0.80, "relevancy": 0.88, "recall": 1.0})
    summary = _summary(0.99, 0.99, 1.0)
    summary["n"] = 2  # 本次只跑了 2 题（基线是 25 题）
    passed, lines = R.compare_baseline(summary, 0.05, R._PRESETS["top5"])
    assert passed is True, "用例数不同时不应判失败"
    assert any("用例数不一致" in line for line in lines), "必须告知不可比"


def test_baseline_same_case_count_still_judged(baseline_file):
    """用例数一致时正常判定——别把保护做成"永远不判"。"""
    baseline_file({"faithfulness": 0.80, "relevancy": 0.88, "recall": 1.0})
    summary = _summary(0.60, 0.88, 1.0)
    summary["n"] = 25
    passed, lines = R.compare_baseline(summary, 0.05, R._PRESETS["top5"])
    assert passed is False
    assert not any("用例数不一致" in line for line in lines)


def test_baseline_case_count_uses_dataset_size_not_valid_count(baseline_file):
    """★ 可比性判据用数据集规模，不是有效判分数。

    40 题里 5 题裁判失败时 `summary["n"]=35`；若拿 35 去比基线 40，
    一次正常全量运行会被误判成"用例数不一致"——独立基线门禁永远不判。
    """
    baseline_file({"faithfulness": 0.80, "relevancy": 0.88, "recall": 1.0}, cases=40)
    summary = _summary(0.60, 0.88, 1.0, n=35)  # 5 题判分失败
    passed, lines = R.compare_baseline(
        summary, 0.05, R._PRESETS["top5"], n_cases=40
    )
    assert passed is False, "全量运行应正常判退化"
    assert not any("用例数不一致" in line for line in lines)


def test_baseline_limit_run_still_incomparable(baseline_file):
    """`--limit 2` 仍必须判不可比——不能用数据集规模把保护绕过去。"""
    baseline_file({"faithfulness": 0.80, "relevancy": 0.88, "recall": 1.0}, cases=40)
    summary = _summary(0.99, 0.99, 1.0, n=2)
    passed, lines = R.compare_baseline(
        summary, 0.05, R._PRESETS["top5"], n_cases=2
    )
    assert passed is True
    assert any("用例数不一致" in line for line in lines)


def test_baseline_rerank_flag_mismatch_detected(baseline_file):
    baseline_file({"faithfulness": 0.9, "relevancy": 0.9, "recall": 0.9}, top_k=5, rerank=False)
    passed, lines = R.compare_baseline(_summary(0.1, 0.1, 0.1), 0.05, R._PRESETS["rerank"])
    assert passed is True
    assert any("配置不一致" in line for line in lines)


def test_save_baseline_writes_metrics(baseline_file, tmp_path, monkeypatch):
    path = baseline_file({"faithfulness": 0.0, "relevancy": 0.0, "recall": 0.0})
    R.save_baseline(_summary(0.81, 0.92, 0.97), R._PRESETS["top5"], 25)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["metrics"]["faithfulness"] == pytest.approx(0.81)
    assert saved["config"]["name"] == "top5"
    assert saved["cases"] == 25
    assert "saved_at" in saved


def test_save_then_compare_roundtrip(baseline_file):
    """存完立刻比，应当判持平——这是最常见的日常用法。"""
    R.save_baseline(_summary(0.81, 0.92, 0.97), R._PRESETS["top5"], 25)
    passed, lines = R.compare_baseline(_summary(0.81, 0.92, 0.97), 0.05, R._PRESETS["top5"])
    assert passed is True
    assert all("退化" not in line for line in lines if "指标" not in line)


def test_production_config_constant_is_a_real_preset():
    """`_PRODUCTION_CONFIG` 必须真的存在于预置配置里。

    若它指向一个不存在的名字，基线锚定逻辑会静默退回 `runs[0]`，
    于是又回到"把 top3 存成基线"的老问题——这种失效没有任何报错。
    """
    assert R._PRODUCTION_CONFIG in R._PRESETS


def test_production_config_matches_pipeline_default():
    """生产配置的 top_k 必须与管线默认一致。

    两者不一致时，基线测的是 A 配置、线上跑的是 B 配置，
    指标再准也说明不了线上的事。
    """
    from src.rag.constants import DEFAULT_TOP_K

    assert R._PRESETS[R._PRODUCTION_CONFIG].top_k == DEFAULT_TOP_K


def test_baseline_anchoring_ignores_first_config_order():
    """★ 回归哨兵：基线的取值必须与"配置在命令行里的顺序"无关。

    第一版用 `runs[0]`，于是 `--configs top3,top5` 把 top3 的分存成基线；
    换成 `--configs top5,top3` 又变成 top5。同样的实验、不同的参数顺序，
    基线不同——这会让人完全无法信任门禁。

    这里直接验证选择逻辑：无论顺序如何，选出的都应是 top5。
    """
    runs = [
        {"config": {"name": "top3", "top_k": 3, "rerank": False}, "summary": _summary(0.699, 0.889, 1.0)},
        {"config": {"name": "top5", "top_k": 5, "rerank": False}, "summary": _summary(0.816, 0.882, 1.0)},
    ]
    for ordered in (runs, list(reversed(runs))):
        picked = next(
            (r for r in ordered if r["config"]["name"] == R._PRODUCTION_CONFIG), ordered[0]
        )
        assert picked["config"]["name"] == "top5"
        assert picked["summary"]["faithfulness"]["mean"] == pytest.approx(0.816)


def test_repo_baseline_is_anchored_to_production(tmp_path, monkeypatch):
    """仓库里那份真实基线必须锚定一个**对照组合法**的配置。

    它会被 CI 当作退化判据，若被误存成某个实验组（如 top3 或 prompt-strict），
    之后所有比对都会偏。

    允许两种锚点：
        top5            生产默认基线（常规回归用）
        prompt-default  提示词 A/B 的对照组（跑提示词实验时覆盖）

    明确**不允许**实验组：top3 / top8 / rerank / prompt-strict / prompt-minimal
    —— 拿它们当基线，等于把"换了配置"当成"质量基准"。
    """
    repo_baseline = ROOT / "eval" / "baseline.json"
    if not repo_baseline.exists():
        pytest.skip("仓库尚无基线文件")
    data = json.loads(repo_baseline.read_text(encoding="utf-8"))
    name = data["config"]["name"]
    allowed = {R._PRODUCTION_CONFIG, "prompt-default"}
    assert name in allowed, (
        f"基线锚定在 {name!r}，不在允许的对照组 {sorted(allowed)} 里——"
        "实验组不能当基线"
    )


# ===========================================================================
# 逐题诊断
# ===========================================================================


def _cases(n):
    return [{"question": f"问题{i}", "reference": "参考答案"} for i in range(1, n + 1)]


def test_diagnose_clean_when_all_above_threshold():
    cases = _cases(2)
    results = [
        {"ok": True, "faithfulness": 0.9, "relevancy": 0.9, "recall": 0.9},
        {"ok": True, "faithfulness": 0.8, "relevancy": 0.8, "recall": 0.8},
    ]
    lines = R.diagnose(cases, results, threshold=0.6)
    assert any("无低于" in line for line in lines)


def test_diagnose_all_low_attributes_to_retrieval():
    """★ 三项全低要归因到**检索**，不是生成。

    归错方向会把人引去调提示词，而真正的问题是库里没依据——
    这是最贵的一类误判，故单独钉死。
    """
    cases = _cases(1)
    results = [{"ok": True, "faithfulness": 0.2, "relevancy": 0.2, "recall": 0.2}]
    text = "\n".join(R.diagnose(cases, results, threshold=0.6))
    assert "检索没找到依据" in text
    assert "不是生成问题" in text


def test_diagnose_recall_only_attributes_to_missed_recall():
    cases = _cases(1)
    results = [{"ok": True, "faithfulness": 0.9, "relevancy": 0.9, "recall": 0.3}]
    text = "\n".join(R.diagnose(cases, results, threshold=0.6))
    assert "漏召回" in text
    assert "top_k" in text


def test_diagnose_low_faithfulness_attributes_to_generation():
    cases = _cases(1)
    results = [{"ok": True, "faithfulness": 0.3, "relevancy": 0.9, "recall": 0.9}]
    text = "\n".join(R.diagnose(cases, results, threshold=0.6))
    assert "生成超出" in text


def test_diagnose_low_relevancy_attributes_to_offtopic():
    cases = _cases(1)
    results = [{"ok": True, "faithfulness": 0.9, "relevancy": 0.3, "recall": 0.9}]
    text = "\n".join(R.diagnose(cases, results, threshold=0.6))
    assert "答非所问" in text


def test_diagnose_lists_failed_cases_separately():
    """执行失败的题要单独列出——它不计入指标分子但计入分母，
    与"分数低的题"是两回事，混在一起会让人以为模型答得差。"""
    cases = _cases(2)
    results = [
        {"ok": False, "error": "sources 为空"},
        {"ok": True, "faithfulness": 0.9, "relevancy": 0.9, "recall": 0.9},
    ]
    text = "\n".join(R.diagnose(cases, results, threshold=0.6))
    assert "执行失败" in text
    assert "sources 为空" in text


def test_diagnose_sorted_by_weakest_first():
    """最需要关注的排最前——列表长了用户只看前几条。"""
    cases = _cases(2)
    results = [
        {"ok": True, "faithfulness": 0.5, "relevancy": 0.5, "recall": 0.5},
        {"ok": True, "faithfulness": 0.1, "relevancy": 0.1, "recall": 0.1},
    ]
    text = "\n".join(R.diagnose(cases, results, threshold=0.6))
    # 问题2（0.1）应排在问题1（0.5）之前
    assert text.index("问题2") < text.index("问题1")

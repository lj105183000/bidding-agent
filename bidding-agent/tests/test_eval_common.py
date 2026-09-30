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

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
        # 先验类型再取字段：非 dict 元素（手工编辑写成字符串/数字）上做 `in` 会抛 TypeError，
        # 与"用例问题 → 中文 SystemExit"的契约不一致，必须在这里统一拦截成可读报错。
        if not isinstance(case, dict):
            raise SystemExit(f"用例 #{i} 不是对象（dict），实际类型：{type(case).__name__}")
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
        except Exception as exc:  # noqa: BLE001 - 容错只兜业务执行异常，不兜调用方契约违反
            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        else:
            # 返回值校验放 else（try 外）：fn 返回非 dict 是调用方 bug，
            # 必须响亮抛出而非被 except 吞掉伪装成"题目失败"。
            if not isinstance(result, dict):
                raise TypeError(f"fn 必须返回 dict，实际返回 {type(result).__name__}（题目：{case!r}）")
            result.setdefault("ok", True)
        results.append(result)
        print(f"  进度 {i}/{total}", end="\r", flush=True)
    print()
    return results


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0

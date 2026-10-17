"""RAG 在线评测。

从**真实请求**里采集质量信号（与 `eval/` 的离线评测互补）。

    `signals.py`     从 done 帧提取指标（纯函数、零额外开销）
    `recorder.py`    采样 + 后台异步落盘（不阻塞请求）
    `aggregate.py`   聚合视图（供 API 与诊断）

对外只需要 `recorder.evaluator` 这一个单例。
"""

from __future__ import annotations

from src.eval_online.aggregate import summarize_rows
from src.eval_online.recorder import OnlineEvaluator, evaluator
from src.eval_online.signals import evaluate, is_evaluable

__all__ = [
    "OnlineEvaluator",
    "evaluate",
    "evaluator",
    "is_evaluable",
    "summarize_rows",
]

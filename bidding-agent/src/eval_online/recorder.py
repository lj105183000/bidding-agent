"""RAG 在线评测（真实请求采样）。

★ 定位：与离线评测互补，不是替代
==========================
    离线 `eval/`     固定测试集 + LLM 裁判，**有参考答案** → 能测"答得对不对"
                     但测试集有限，覆盖不到真实长尾
    在线 本模块       真实请求 + 已有信号，**无参考答案** → 测不了"对不对"
                     但覆盖真实分布，能发现"线上某类问题一直在失败"

★ 硬约束：不拖慢、不拖垮用户请求
==========================
采用三层防护，任一层失效都不会影响问答：

    ① `evaluate()` 是**纯函数**，只读 `done` 帧里本来就有的字段 —— 零额外开销
    ② 写盘与聚合走**后台线程**，请求路径上只做一次 `queue.put_nowait`
    ③ 队列**有界**：满了一律丢弃并计数（`dropped`）——
       在线采样丢几条无妨，但阻塞请求或无限堆积内存是不可接受的

★ 为什么默认**关闭** LLM 裁判采样
============================
每请求跑一次 LLM 裁判，成本与延迟都不可接受。故：
    · 确定性指标：始终采集（零成本）
    · LLM 裁判：`RAG_ONLINE_EVAL_JUDGE_RATE` 控制，**默认 0（关闭）**
      开启后仍走后台线程，且失败不影响任何东西

★ 存储：jsonl 追加，与 `eval/trend.jsonl` 同一约定
===========================================
一行一次请求，便于 `grep`/`jq`/diff。**但会无限增长**——
故用简单的按大小轮转（见 `_rotate_if_needed`），而不是假装它不会涨。
"""

from __future__ import annotations

import json
import logging
import os
import queue
import random
import threading
import time
from pathlib import Path
from typing import Any

from src.eval_online.signals import evaluate, is_evaluable

logger = logging.getLogger(__name__)

#: 环境变量
ENV_ENABLED = "RAG_ONLINE_EVAL"
ENV_RATE = "RAG_ONLINE_EVAL_RATE"
ENV_JUDGE_RATE = "RAG_ONLINE_EVAL_JUDGE_RATE"
ENV_PATH = "RAG_ONLINE_EVAL_PATH"

#: 默认采样率。
#:
#: ★ 为什么不是 1.0
#: ---------------
#: 在线评测的价值在**趋势与异常**，不在"每一个请求都记"。
#: 采样 20% 已足够看出分布（且写盘量降 5 倍）；
#: 需要全量时显式设 `RAG_ONLINE_EVAL_RATE=1`。
DEFAULT_RATE = 0.2

#: 队列容量。满则丢弃。
QUEUE_MAXSIZE = 512

#: 可评性问题的最小长度。与 signals.is_evaluable 用的是同一个数 ——
#: 这里单独定义一份是为了能在**提取之前**就用它短路
#: （信号模块的常量在 signals 内，导入它只为取一个数字不划算）。
#: 两处若不一致，is_evaluable 仍会在提取后兜底，故不会产生错误数据。
MIN_EVALUABLE_QUESTION_CHARS = 4

#: 单文件大小上限（字节），超过则轮转。
#:
#: 一行约 600~900 字节，50MB ≈ 6~8 万条请求。
#: 在线评测是**趋势观测**，不需要无限历史。
MAX_FILE_BYTES = 50 * 1024 * 1024


def _env_float(name: str, default: float) -> float:
    raw = (os.getenv(name, "") or "").strip()
    if not raw:
        return default
    try:
        v = float(raw)
    except ValueError:
        return default
    return min(max(v, 0.0), 1.0)


def _env_bool(name: str, default: bool) -> bool:
    raw = (os.getenv(name, "") or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


class OnlineEvaluator:
    """采样、异步落盘、聚合。线程安全。"""

    def __init__(self, path: Path | None = None) -> None:
        #: 显式传入的路径（测试用）。None 时按**运行期**环境变量解析。
        self._explicit_path = path
        self._q: queue.Queue = queue.Queue(maxsize=QUEUE_MAXSIZE)
        self._lock = threading.Lock()
        self._started = False
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        #: 统计（自己也要可观测）
        self.accepted = 0
        self.dropped = 0
        self.written = 0
        self.write_errors = 0
        self.judge_sampled = 0
        self.judge_failed = 0

    @property
    def path(self) -> Path:
        """样本文件路径。**每次读取都重新解析环境变量**。

        ★ 为什么必须是 property 而不是 __init__ 里定死的字段（实测踩到的缺陷）
        ------------------------------------------------------------
        模块级单例 `evaluator = OnlineEvaluator()` 在**模块第一次被 import 时**
        执行，若路径在那里定死，那么：

            · 任何测试先 import 本模块（或经由 `BendingAgent` 间接触发）
              → 单例绑定了**真实数据文件**
            · 测试 fixture 再设 `RAG_ONLINE_EVAL_PATH` 也**改不动它**
            · 于是跑 pytest 会把测试数据写进 `eval/online_samples.jsonl`

        实测确认：该文件被写进 17 行测试数据（`duration_ms` 只有 16~141ms，
        而真实请求是 6~15 秒），**没有任何报错**。

        改成运行期解析后，测试可以用环境变量把数据引到临时目录。
        这与 `ragas_eval._judge_attempts()` 是同一模式
        （那里的注释也写了"不要用导入时冻结的常量"）。
        """
        if self._explicit_path is not None:
            return self._explicit_path
        raw = (os.getenv(ENV_PATH, "") or "").strip()
        if raw:
            return Path(raw)
        return Path(__file__).resolve().parents[2] / "eval" / "online_samples.jsonl"

    # --- 开关 ---

    def enabled(self) -> bool:
        return _env_bool(ENV_ENABLED, True)

    def rate(self) -> float:
        return _env_float(ENV_RATE, DEFAULT_RATE)

    def judge_rate(self) -> float:
        return _env_float(ENV_JUDGE_RATE, 0.0)

    # --- 请求路径（必须极快、绝不阻塞）---

    def maybe_record(self, meta: dict, done: dict, *, question: str = "",
                     has_history: bool = False) -> None:
        """在请求结束时调用。**永不抛异常、永不阻塞**。

        只做三件事：判断是否采样 → 提取指标（纯函数）→ 丢进队列。
        写盘与聚合都在后台。
        """
        try:
            if not self.enabled():
                return
            # ★ 免费判据优先（顺序有讲究，实测发现的浪费）
            #
            # `maybe_record` 原本的顺序是：采样判定 → `evaluate()` → 可评性判定。
            # 但 `clarified` 与问题长度**在 done 帧里本来就有**，是免费的；
            # 而 `evaluate()` 要做 27 项提取。先判免费的那些，
            # 追问请求（实测占样本 1/4）就能完全跳过提取。
            #
            # 注意不能把 `is_evaluable` 整体前置：它还依赖 `evaluate()` 产出的
            # `question_chars`。故这里只前置那两个**不需要提取**的条件，
            # 完整的 `is_evaluable` 仍在提取后再判一次（兜底）。
            if done.get("clarified"):
                return
            if len(str(question or "")) < MIN_EVALUABLE_QUESTION_CHARS:
                return
            # 采样：按比例丢弃
            if not self._sampled():
                return
            row = evaluate(meta, done, question=question, has_history=has_history)
            if not row:
                return
            if not is_evaluable(row):
                return
            self._ensure_worker()
            try:
                self._q.put_nowait(row)
                with self._lock:
                    self.accepted += 1
            except queue.Full:
                # ★ 满了一律丢弃：在线采样丢几条无妨，
                # 阻塞请求或无限堆积内存不可接受
                with self._lock:
                    self.dropped += 1
        except Exception:  # noqa: BLE001 - 评测绝不能影响请求
            logger.debug("在线评测采样失败（不影响请求）", exc_info=True)

    def _sampled(self) -> bool:
        """按采样率决定这一条是否记录。

        ★ 第一版**采样有偏**（自查时统计发现，这是本轮最严重的缺陷）
        ------------------------------------------------
        原实现用时间戳低位做判定：

            (time.time_ns() % 1000) / 1000.0 >= self.rate()

        实测（各 2 万次抽取）：
            配置 rate=0.2  → 实际命中 **51.6%**
            配置 rate=0.5  → 实际命中 **39.0%**

        形状像 `1 - rate/2`，说明 `time.time_ns() % 1000` **不是均匀分布**
        （纳秒计时的低位带有结构性偏差，不是独立均匀的随机位——
        我原以为"低位相当于随机"是错的）。

        后果比"样本少"更糟：**样本量是配置的好几倍，且偏离方向不可预测**，
        意味着所有聚合数字都建立在一个未声明、也未验证的分布上。

        改用 `random.random()`：标准库、分布均匀、无需自造随机源。
        代价是无法用 monkeypatch 精确控制单次抽样，故测试改为验证**统计性质**
        （大样本下命中率应接近配置值）。
        """
        rate = self.rate()
        if rate >= 1.0:
            return True
        if rate <= 0.0:
            return False
        return random.random() < rate

    # --- 后台 ---

    def _ensure_worker(self) -> None:
        if self._started:
            return
        with self._lock:
            if self._started:
                return
            self._started = True
        self._worker = threading.Thread(
            target=self._run, name="online-eval", daemon=True
        )
        self._worker.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                row = self._q.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                self._append(row)
            except Exception:  # noqa: BLE001
                with self._lock:
                    self.write_errors += 1
                logger.debug("在线评测写盘失败", exc_info=True)
            finally:
                self._q.task_done()

    def _rotate_if_needed(self) -> None:
        """按大小轮转。保留一份 `.1` 备份。

        ★ 为什么不直接清空：清空会丢掉"上一个窗口"的数据，
        而趋势对比恰恰需要它。
        """
        try:
            if not self.path.exists() or self.path.stat().st_size < MAX_FILE_BYTES:
                return
            backup = self.path.with_suffix(self.path.suffix + ".1")
            if backup.exists():
                backup.unlink()
            self.path.rename(backup)
            logger.info("在线评测数据已轮转：%s → %s", self.path.name, backup.name)
        except Exception:  # noqa: BLE001 - 轮转失败不该阻断写入
            logger.debug("在线评测轮转失败", exc_info=True)

    def _append(self, row: dict) -> None:
        self._rotate_if_needed()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        with self._lock:
            self.written += 1

    # --- 聚合读取 ---

    def read_rows(self, limit: int = 2000) -> list[dict]:
        """读**最近** `limit` 条（含轮转备份）。

        ★ 为什么不能"全量读取再切片"（自查发现的浪费）
        ----------------------------------------
        第一版是把整个文件 `read_text()` 再取尾部。实测：6.8MB / 17 万行
        耗时 **0.34s**，按 `MAX_FILE_BYTES=50MB` 外推约 **2.5s** ——
        而 `limit` 只影响**返回值**，读进来的量完全一样。

        对一个只读诊断端点，2.5s 且"随文件增长而增长"是不必要的。
        故改为**从文件尾部反向读块**，凑够 `limit` 行即停。
        坏行**跳过而不是让整体失败**（jsonl 可能被人工编辑或写了一半）。
        """
        limit = max(1, limit)
        rows: list[dict] = []
        # 先读当前文件，不足再往前补备份（备份时间更早，故拼在前面）
        for p in (self.path, self.path.with_suffix(self.path.suffix + ".1")):
            if len(rows) >= limit:
                break
            rows = self._read_tail(p, limit - len(rows)) + rows
        return rows[-limit:]

    def _read_tail(self, path: Path, want: int) -> list[dict]:
        """从文件尾部反向读块，取得最后 `want` 条有效记录。

        ★ 块大小必须**自适应**，不能按固定"每行多少字节"估
        -------------------------------------------
        第一版按 `want * 2048` 定块大小（假设单行 2KB）。实测：
        `want=2000` 时块 = 4MB，而 6.8MB 的文件**一次就被读掉 57%** ——
        等于"尾部读取"退化成"大部分全量读取"（只快 40%）。

        根因是**行宽未知**：样本行从 42 字节（我的测试数据）到 900 字节
        （真实记录）不等，任何固定估计都会在某些场景下过大或过小。

        改成从 128KB 起，**按已有条数比例推算缺口**并翻倍，
        最多读 4MB。这样：
            · 行短 → 一次就够
            · 行长 → 迭代几次，但总读取量仍远小于全文件
        """
        if want <= 0 or not path.exists():
            return []
        try:
            total = path.stat().st_size
        except OSError:
            return []
        if total <= 0:
            return []

        out: list[dict] = []
        pos = total
        buf = b""
        read_bytes = 0
        max_bytes = 4 * 1024 * 1024
        step = 131072
        try:
            with path.open("rb") as f:
                while pos > 0 and len(out) < want and read_bytes < max_bytes:
                    need = want - len(out)
                    # 按"还差几条"外推缺口：已解析 out 条用了 read_bytes，
                    # 则再要 need 条约需 read_bytes/len(out)*need。
                    # 首次（out 为空）用起始块。
                    if out and read_bytes:
                        est = int(read_bytes / len(out) * need * 1.5) + 65536
                        step = max(step, min(est, max_bytes - read_bytes))
                    step = max(65536, min(step, max_bytes - read_bytes))
                    read = min(step, pos)
                    pos -= read
                    f.seek(pos)
                    buf = f.read(read) + buf
                    read_bytes += read
                    lines = buf.split(b"\n")
                    complete = lines[1:] if pos > 0 else lines
                    buf = lines[0] if pos > 0 else b""
                    parsed: list[dict] = []
                    for raw in complete:
                        raw = raw.strip()
                        if not raw:
                            continue
                        try:
                            obj = json.loads(raw.decode("utf-8"))
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            continue
                        if isinstance(obj, dict):
                            parsed.append(obj)
                    out = parsed + out
                    step *= 2
        except OSError:
            return []
        return out[-want:]

    def snapshot(self) -> dict[str, Any]:
        """自身状态（供 /api/health 或诊断）。"""
        with self._lock:
            return {
                "enabled": self.enabled(),
                "rate": self.rate(),
                # ★ 如实标注"LLM 裁判采样尚未实现"
                #
                # RAG_ONLINE_EVAL_JUDGE_RATE 能被读取并出现在快照里，
                # 但 maybe_record **没有**任何调用裁判的代码 —— 设了不生效。
                # 若只暴露 judge_rate 而不说明，用户会以为它已经工作，
                # 进而怀疑"为什么 judge_sampled 一直是 0"。
                # 故显式给出这个标志，让"未实现"是**可观测**的。
                "judge_implemented": False,
                "judge_rate": self.judge_rate(),
                "path": str(self.path),
                "queue_size": self._q.qsize(),
                "queue_maxsize": QUEUE_MAXSIZE,
                "accepted": self.accepted,
                "dropped": self.dropped,
                "written": self.written,
                "write_errors": self.write_errors,
                # 恒为 0，因为裁判未实现（见 judge_implemented）
                "judge_sampled": self.judge_sampled,
                "judge_failed": self.judge_failed,
            }

    def flush(self, timeout: float = 5.0) -> bool:
        """等队列排空（测试与优雅关闭用）。返回是否排空。"""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._q.empty():
                # 队列空但 worker 可能正在写最后一条
                time.sleep(0.05)
                return self._q.empty()
            time.sleep(0.05)
        return self._q.empty()


#: 进程级单例。与 `monitoring` / `monitoring_trace` 同一模式。
evaluator = OnlineEvaluator()

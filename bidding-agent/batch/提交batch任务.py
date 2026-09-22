"""提交智谱 Batch 任务并取回结果（第 2 步与第 3 步之间的运维环节，见 docs/开发文档.md §4.2）。

**它不属于 §4.2 的 5 个数据处理步骤**，而是那 5 步之间原本靠人工在控制台点击的一环：
上传请求文件 → 创建批量任务 → 等模型跑完 → 下载结果。做成脚本是为了可复现——
浏览器里点出来的任务，谁也说不清用的是哪个文件、哪个模型。

**为什么要有状态文件**：批量任务按完成先后排队，跑几分钟到几小时都有可能，跨得过一次
会话。把 `batch_id` 落到 `data/batch/batch_job.json`，中断后用同一条命令接着等即可。

**绝不自动重建任务**：已有记录时默认**续用**，要重新提交必须显式 `--new`。重复提交不是
"再来一次"那么便宜——它是 8751 次真实计费的调用，而且两份结果混在一起会让人分不清哪份
才是当前数据的。

**下载的文件名按项目约定写死**：`batch_results_raw.jsonl` 正是第 3 步（按custom_id排序.py）
的输入名，`batch_results_errors.jsonl` 收平台单独返回的失败请求（官方文档说明失败请求不
混在结果文件里，走 `error_file_id`）。

用法::

    python batch/提交batch任务.py                  # 上传 + 创建 + 等完成 + 下载
    python batch/提交batch任务.py --no-wait        # 只上传建任务，记下 batch_id 就走
    python batch/提交batch任务.py --status         # 只看当前任务状态
    python batch/提交batch任务.py --batch-id <id>  # 指定任务（继续等 / 下载）
    python batch/提交batch任务.py --new            # 忽略已有记录，重新提交（会重复计费）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# 项目根：batch/提交batch任务.py → 上一级。所有默认路径都以此为基准，脚本可在任意 CWD 下运行
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=False)

DATA_DIR = PROJECT_ROOT / "data" / "batch"
DEFAULT_SRC = DATA_DIR / "batch_requests.jsonl"
# 第 3 步认这个名字
DEFAULT_OUTPUT = DATA_DIR / "batch_results_raw.jsonl"
DEFAULT_ERROR_OUTPUT = DATA_DIR / "batch_results_errors.jsonl"
DEFAULT_STATE = DATA_DIR / "batch_job.json"

# 智谱 batch 目前只支持这一个 endpoint（官方文档的 endpoint 参数说明）
DEFAULT_ENDPOINT = "/v4/chat/completions"

POLL_INTERVAL = 30.0
DEFAULT_TIMEOUT = 2 * 3600.0
# 下载整个结果文件可能几十 MB，别用默认的短超时
DOWNLOAD_TIMEOUT = 600.0

DONE_OK = {"completed"}
DONE_BAD = {"failed", "expired", "cancelled"}
PENDING = {"validating", "in_progress", "finalizing", "cancelling"}


def _force_utf8() -> None:
    """Windows 控制台默认 GBK，中文日志容易乱码或抛 UnicodeEncodeError。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except Exception:
            pass


def make_client():
    """按 `.env` 的 `ZHIPU_API_KEY` 建客户端。密钥不进日志、不进错误提示。"""
    from zai import ZhipuAiClient

    api_key = (os.getenv("ZHIPU_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("缺少 ZHIPU_API_KEY，无法提交批量任务（见 docs/开发文档.md §2.3）")
    return ZhipuAiClient(api_key=api_key)


# ---- 状态记录 ----


def load_state(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # 状态文件坏了不该让整件事卡死：当作没有记录，让调用方走"重新提交"的显式分支
        return {}


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


# ---- 各步骤 ----


def upload_input(client, src: Path):
    """上传请求文件，返回 FileObject。purpose 必须是 batch，否则建任务时会被拒。"""
    print(f"上传中 : {src}（{src.stat().st_size / 1024 / 1024:.1f}MB）")
    with src.open("rb") as handle:
        return client.files.create(file=handle, purpose="batch")


def create_batch(client, file_id: str, endpoint: str, model: str | None = None):
    """创建批量任务。model 只写进 metadata，方便日后在控制台里认出这是哪一批。"""
    metadata = {"source": DEFAULT_SRC.name}
    if model:
        metadata["model"] = model
    return client.batches.create(
        input_file_id=file_id,
        endpoint=endpoint,
        metadata=metadata,
    )


def describe(batch) -> str:
    counts = getattr(batch, "request_counts", None)
    if not counts:
        return batch.status
    return f"{batch.status}（{counts.completed}/{counts.total} 完成，{counts.failed} 失败）"


def wait_for(client, batch_id: str, timeout: float, interval: float):
    """轮询到终态。超时只是本脚本放弃等待，任务在平台上照跑，batch_id 已落盘。"""
    deadline = time.monotonic() + timeout
    last = ""
    while True:
        batch = client.batches.retrieve(batch_id)
        line = describe(batch)
        if line != last:
            print(f"[{time.strftime('%H:%M:%S')}] {line}")
            last = line
        if batch.status in DONE_OK or batch.status in DONE_BAD:
            return batch
        if time.monotonic() >= deadline:
            print(f"等待超时（{timeout / 60:.0f} 分钟），任务仍在平台运行。")
            print(f"稍后用同一条命令继续：python batch/提交batch任务.py --batch-id {batch_id}")
            return batch
        time.sleep(interval)


def download(client, file_id: str, dest: Path) -> int:
    """把平台文件落到本地，返回字节数。"""
    response = client.files.content(file_id, timeout=DOWNLOAD_TIMEOUT)
    payload = response.content
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(payload)
    return len(payload)


def read_model_from_requests(src: Path) -> str | None:
    """从请求文件首行取模型名——只用于 metadata 标注，取不到就算了。"""
    try:
        with src.open("r", encoding="utf-8-sig") as handle:
            first = handle.readline().strip()
        return (json.loads(first).get("body") or {}).get("model")
    except Exception:
        return None


def fetch_results(client, batch, output: Path, error_output: Path) -> int:
    """下载结果；有失败请求文件时一并下载。返回退出码。"""
    if batch.status in DONE_BAD:
        print(f"任务以 {batch.status} 结束，没有结果文件可下载。")
        return 1
    if not getattr(batch, "output_file_id", None):
        print("任务已完成但没有 output_file_id——请在平台控制台确认。")
        return 1

    size = download(client, batch.output_file_id, output)
    print(f"已下载 : {output}（{size / 1024 / 1024:.1f}MB）")

    error_file_id = getattr(batch, "error_file_id", None)
    if error_file_id:
        size = download(client, error_file_id, error_output)
        print(f"失败请求: {error_output}（{size / 1024:.1f}KB）——这些行的 custom_id 不会出现在结果里")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="提交智谱 Batch 任务并取回结果")
    parser.add_argument("--src", type=Path, default=None, help="请求文件（默认 data/batch/batch_requests.jsonl）")
    parser.add_argument("-o", "--output", type=Path, default=None, help="结果文件（默认 batch_results_raw.jsonl）")
    parser.add_argument("--state", type=Path, default=None, help="任务记录文件")
    parser.add_argument("--batch-id", default=None, help="直接指定已有任务")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="批量任务 endpoint")
    parser.add_argument("--new", action="store_true", help="忽略已有记录，重新提交（会重复计费）")
    parser.add_argument("--no-wait", action="store_true", help="创建后立即返回，不等待")
    parser.add_argument("--status", action="store_true", help="只查状态")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="最长等待秒数")
    parser.add_argument("--interval", type=float, default=POLL_INTERVAL, help="轮询间隔秒数")
    args = parser.parse_args(argv)

    _force_utf8()

    src = args.src or DEFAULT_SRC
    src = src if src.is_absolute() else (PROJECT_ROOT / src)
    output = args.output or DEFAULT_OUTPUT
    output = output if output.is_absolute() else (PROJECT_ROOT / output)
    error_output = DEFAULT_ERROR_OUTPUT
    state_path = args.state or DEFAULT_STATE
    state_path = state_path if state_path.is_absolute() else (PROJECT_ROOT / state_path)

    try:
        client = make_client()
    except RuntimeError as exc:
        print(exc)
        return 1

    state = load_state(state_path)
    batch_id = args.batch_id or state.get("batch_id")

    # 只有"要新建任务"的分支才需要源文件在
    needs_upload = not batch_id or args.new
    if needs_upload and not src.exists():
        print(f"请求文件不存在: {src}")
        print("先跑 python batch/提数据_编JSONL.py 生成。")
        return 1

    if args.new and batch_id:
        print(f"⚠ --new：忽略已有任务 {batch_id}，将重新上传并提交（重复计费）。")

    if needs_upload:
        try:
            file_object = upload_input(client, src)
        except Exception as exc:
            print(f"上传失败: {exc}")
            return 1
        if not file_object.id:
            print("上传返回里没有文件 id，无法继续。")
            return 1
        print(f"文件 id : {file_object.id}")

        try:
            batch = create_batch(
                client, file_object.id, args.endpoint, read_model_from_requests(src)
            )
        except Exception as exc:
            print(f"创建任务失败: {exc}")
            return 1
        batch_id = batch.id
        state = {
            "batch_id": batch_id,
            "input_file_id": file_object.id,
            "endpoint": args.endpoint,
            "source": str(src),
        }
        save_state(state_path, state)
        print(f"任务 id : {batch_id}")
        print(f"已记录 : {state_path}")
    else:
        print(f"续用任务: {batch_id}（要重新提交加 --new）")

    if args.status:
        batch = client.batches.retrieve(batch_id)
        print(f"状态   : {describe(batch)}")
        return 0

    if args.no_wait:
        print("已提交，未等待。稍后用同一条命令继续。")
        return 0

    try:
        batch = wait_for(client, batch_id, args.timeout, args.interval)
    except Exception as exc:
        print(f"查询任务失败: {exc}")
        return 1

    state.update({"status": batch.status, "output_file_id": getattr(batch, "output_file_id", None)})
    save_state(state_path, state)

    return fetch_results(client, batch, output, error_output)


if __name__ == "__main__":
    raise SystemExit(main())

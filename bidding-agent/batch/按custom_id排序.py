"""按 custom_id 重排 JSONL（数据管道第 3 步，见 docs/开发文档.md §4.2）。

**为什么需要这一步**：batch 接口不保证结果顺序——它按完成先后返回，同一个文件里
第 8000 条的请求可能第 3 个才返回。第 4 步要按行序把标的物回填进 Excel，所以必须
先用 custom_id 把结果排回原始行号顺序。

**为什么不能直接排字符串**：`request-10` 的字典序排在 `request-2` 前面，字符串排序
会把 10 号往后的结果整体插到 2 号前面——回填的标的物于是错位，而且全程不报错，
要等到看统计结果觉得"这标的物怎么这么怪"才发现。序号一律按**数值**比较。

契约里的输入输出是 `batch_results_raw.jsonl` → `batch_results_processed.jsonl`；
对其他 JSONL（比如第 2 步生成的请求文件）同样适用，输出落在源文件旁边加 `_sorted`。
对本来就是升序的文件跑一遍是幂等的——这正好可以用来验证第 2 步的产物。

**坏行一律报错并指出行号**，不猜测、不跳过：排序本身不会让错位暴露，等到第 4 步
标的物填错了行才排查，成本高得多。

约定（与其他 batch 脚本一致）：CWD 无关、支持 `--dry-run`、空数据提前返回、
目标文件被占用时给出可操作的提示。

用法::

    python batch/按custom_id排序.py                                    # 结果文件 raw → processed
    python batch/按custom_id排序.py --src data/batch/batch_requests.jsonl
    python batch/按custom_id排序.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 项目根：batch/按custom_id排序.py → 上一级。所有默认路径都以此为基准，脚本可在任意 CWD 下运行
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "batch"
# 契约里的输入：平台下载的原始结果文件（顺序不保证）
DEFAULT_SRC = DATA_DIR / "batch_results_raw.jsonl"
# 契约里的原始结果文件名 → 处理后的名字
RAW_NAME = "batch_results_raw.jsonl"
PROCESSED_NAME = "batch_results_processed.jsonl"

SUFFIX = "_sorted"


def _force_utf8() -> None:
    """Windows 控制台默认 GBK，中文日志容易乱码或抛 UnicodeEncodeError。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except Exception:
            pass


def resolve_output(src: Path) -> Path:
    """输出路径。

    契约里这一步的产物叫 `batch_results_processed.jsonl`；对其他文件（请求文件等）
    沿用 去重.py 的"挨着源文件加后缀"约定，免得两种命名规则各自为政。
    """
    if src.name == RAW_NAME:
        return src.with_name(PROCESSED_NAME)
    return src.with_name(f"{src.stem}{SUFFIX}{src.suffix}")


def parse_custom_id(value: object) -> int:
    """`request-12` → 12。只取最后一段数字，前缀长什么样不影响——契约见第 2 步。"""
    suffix = str(value).strip().rsplit("-", 1)[-1]
    try:
        return int(suffix)
    except ValueError:
        raise ValueError(
            f"无法解析为序号: {value!r}（应为 request-<数字> 形式）"
        ) from None


def load_jsonl(path: Path) -> tuple[list[tuple[int, dict]], dict]:
    """读取 JSONL，返回 `([(序号, 记录)], 统计)`。

    空行跳过（结果文件末尾常有），其余坏行直接抛错——见模块开头"坏行为什么要报错"。
    """
    stats = {"input": 0, "blank": 0}
    indexed: list[tuple[int, dict]] = []

    # utf-8-sig：平台/Windows 工具导出的文件可能带 BOM，带 BOM 时首行的 "{" 会解析失败
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                stats["blank"] += 1
                continue
            stats["input"] += 1
            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"第 {line_number} 行不是合法 JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"第 {line_number} 行不是 JSON 对象: {text[:60]}")
            if "custom_id" not in record:
                raise ValueError(f"第 {line_number} 行缺少 custom_id 字段")
            try:
                seq = parse_custom_id(record["custom_id"])
            except ValueError as exc:
                raise ValueError(f"第 {line_number} 行 {exc}") from exc
            indexed.append((seq, record))

    return indexed, stats


def sort_records(indexed: list[tuple[int, dict]]) -> tuple[list[dict], dict]:
    """按序号升序排列，返回 `(记录列表, 统计)`。

    `sorted` 是稳定排序：万一出现重复序号，保持文件里的先后——同一份输入两次跑出
    同一个结果，下游才可复现。
    """
    stats = {"first": None, "last": None, "gaps": 0, "duplicates": 0}
    if not indexed:
        return [], stats

    ordered = sorted(indexed, key=lambda item: item[0])
    seqs = [seq for seq, _ in ordered]
    stats["first"], stats["last"] = seqs[0], seqs[-1]
    stats["duplicates"] = len(seqs) - len(set(seqs))
    # 缺口 = 序号区间里"本该有却没有"的个数。缺口是正常的（第 2 步跳过空名称的行），
    # 但结果文件出现缺口也可能是平台漏返回了某些请求，值得报出来。
    stats["gaps"] = (seqs[-1] - seqs[0] + 1) - len(set(seqs))
    return [record for _, record in ordered], stats


def render_jsonl(records: list[dict]) -> str:
    """序列化为 JSONL 文本（每行一个对象，末尾留换行）。

    `ensure_ascii=False` 是必须的：转义成 `\\uXXXX` 的中文在人工抽查时完全不可读。
    """
    if not records:
        return ""
    return "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n"


def write_text(path: Path, payload: str) -> None:
    try:
        path.write_text(payload, encoding="utf-8")
    except PermissionError as exc:
        raise PermissionError(
            f"无法写入 {path}——文件可能正被其他程序打开。请关闭后重试。"
        ) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="按 custom_id 序号重排 JSONL")
    parser.add_argument("--src", type=Path, default=None, help="输入 JSONL（默认结果文件）")
    parser.add_argument("-o", "--output", type=Path, default=None, help="输出 JSONL")
    parser.add_argument("--dry-run", action="store_true", help="只统计，不写出文件")
    args = parser.parse_args(argv)

    _force_utf8()

    src = args.src or DEFAULT_SRC
    src = src if src.is_absolute() else (PROJECT_ROOT / src)
    if not src.exists():
        print(f"输入文件不存在: {src}")
        print("默认读平台下载的 batch_results_raw.jsonl；排序其他文件请用 --src 指定。")
        return 1

    try:
        indexed, stats = load_jsonl(src)
    except (ValueError, OSError) as exc:
        print(f"读取失败: {exc}")
        return 1

    if not indexed:
        print(f"{src} 里没有可排序的记录（空行 {stats['blank']} 行）。")
        return 0

    records, order = sort_records(indexed)
    print(f"源文件 : {src}")
    print(f"记录数 : {stats['input']}")
    if stats["blank"]:
        print(f"空行   : {stats['blank']} 行（已跳过）")
    print(f"序号   : {order['first']} .. {order['last']}")
    print(f"缺口   : {order['gaps']} 个（第 2 步跳过空「项目名称」的行所致）")
    if order["duplicates"]:
        # 不致命但必须说出来：同一行有两条结果时，第 4 步回填会取到后一条
        print(f"重复   : {order['duplicates']} 个序号重复，已保持文件内先后顺序")

    if args.dry_run:
        print("\n--dry-run：未写出文件。")
        return 0

    dest = args.output or resolve_output(src)
    dest = dest if dest.is_absolute() else (PROJECT_ROOT / dest)
    if dest == src:
        print("\n输出路径与输入相同，会覆盖源文件。请用 -o 指定其他路径。")
        return 1
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        write_text(dest, render_jsonl(records))
    except PermissionError as exc:
        print(f"\n{exc}")
        return 1
    print(f"\n已写出: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

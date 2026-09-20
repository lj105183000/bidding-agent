"""把模型抽出的标的物回填进 Excel（数据管道第 4 步，见 docs/开发文档.md §4.2）。

**读的是"结果"不是"请求"**：第 2 步产出的 `batch_requests.jsonl` 里只有问题（项目名称），
标的物是模型在**结果文件**里给回来的答案。所以这一步读第 3 步排好序的
`batch_results_processed.jsonl`。拿请求文件来跑会明确报错，不会静默填出一张空表。

**怎么对齐**：第 2 步定的契约是 `custom_id = request-<数据行号>`，于是序号 N 的结果
填进第 N 条数据（`.iloc[N-1]`）。

- **缺号留空**：那是第 2 步跳过空「项目名称」的行导致的，正常，只报数不报错；
- **越界是致命的**：序号超出表长说明结果文件和这张表根本不是一对（比如去重规则改过、
  表重跑过），硬填会把整表错位且全程不报错——所以直接拦下。

输出写成源文件旁的 `<原名>_标的物.xlsx`，正好是第 5 步的输入（沿用 去重.py 的加后缀约定）。
「标的物」列已存在就原地填，不存在就追加到**最后一列**——不插在中间，免得打乱下游按
列序读表的假设（本项目下游都是按列名取的）。

约定（与其他 batch 脚本一致）：CWD 无关、支持 `--dry-run`、空数据提前返回、
Excel 被占用时给出可操作的提示。

用法::

    python batch/提数据_插标的物.py                      # 结果文件 + 去重版 → …_标的物.xlsx
    python batch/提数据_插标的物.py --dry-run            # 只统计不落盘
    python batch/提数据_插标的物.py --src <结果.jsonl> --excel <表.xlsx> -o <输出.xlsx>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

# 项目根：batch/提数据_插标的物.py → 上一级。所有默认路径都以此为基准，脚本可在任意 CWD 下运行
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "batch"
# 第 3 步的产物（已按 custom_id 排回行号顺序）
DEFAULT_SRC = DATA_DIR / "batch_results_processed.jsonl"
# 与第 2 步读同一份：去重产出固定落 data/processed、加 `_2` 后缀（见 去重.py），
# 原始数据与加工数据分离。这里刻意不做多路径探测——探测出"读错了批次"的隐患，
# 比少写两行候选路径的便利更值钱。
DEFAULT_EXCEL = PROJECT_ROOT / "data" / "processed" / "招标采购标的物信息提取训练数据_2.xlsx"

SUBJECT_COLUMN = "标的物"
SUFFIX = "_标的物"

# content 里可能出现的包装：markdown 代码围栏（官方文档的示例 content 就是被 ``` 包着的）、
# 模型自作主张加上的标签、引号、句末标点。提示词里明确要求纯文本，但"要求"不等于"保证"
_FENCE_HEAD = re.compile(r"^```[a-zA-Z]*\s*")
_FENCE_TAIL = re.compile(r"\s*```$")
_LABEL_HEAD = re.compile(r"^标的物\s*[:：]\s*")
_TRIM = "\"'“”‘’ \t\r\n"
_TAIL_PUNCT = "。.，,、;；"


def _force_utf8() -> None:
    """Windows 控制台默认 GBK，中文日志容易乱码或抛 UnicodeEncodeError。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except Exception:
            pass


def resolve_default_excel() -> Path:
    """约定的去重版输入（data/processed 下的 `_2.xlsx`）；不存在时由调用方报错。"""
    return DEFAULT_EXCEL


def resolve_output(src: Path) -> Path:
    """输出到输入表旁边的 `<原名>_标的物.xlsx`（第 5 步就认这个名字）。"""
    return src.with_name(f"{src.stem}{SUFFIX}.xlsx")


def parse_custom_id(value: object) -> int:
    """`request-12` → 12。只取最后一段数字，前缀长什么样不影响——契约见第 2 步。"""
    suffix = str(value).strip().rsplit("-", 1)[-1]
    try:
        return int(suffix)
    except ValueError:
        raise ValueError(
            f"无法解析为序号: {value!r}（应为 request-<数字> 形式）"
        ) from None


def clean_subject_matter(raw: str) -> str:
    """清掉模型偶尔附带的包装，只留标的物本身。"""
    text = raw.strip()
    text = _FENCE_TAIL.sub("", _FENCE_HEAD.sub("", text)).strip()
    text = _LABEL_HEAD.sub("", text).strip()
    # 引号与句末标点会交替挡路（`"胶圈"。` 里末尾的句号挡住了引号），来回剥到稳定为止
    while True:
        trimmed = text.strip(_TRIM).rstrip(_TAIL_PUNCT).strip()
        if trimmed == text:
            return text
        text = trimmed


def _subject_from_json_object(text: str) -> str | None:
    """模型改用 JSON 回答时（官方示例的 content 就是 JSON 字符串）取其中的值。"""
    if not text.startswith("{"):
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict):
        for key in (SUBJECT_COLUMN, "subject", "subject_matter"):
            value = payload.get(key)
            if isinstance(value, str):
                return value
    return None


def extract_subject_matter(record: dict) -> tuple[str, str]:
    """从一条结果里取标的物，返回 `(值, 状态)`。

    状态取值：`ok` 取到了；`empty` 模型答了但清完是空的；`failed` 这条请求失败
    （平台错误或非 200）；`no_response` 压根没有 response 字段——通常是拿请求文件来跑了。
    """
    if "response" not in record:
        return "", "no_response"

    response = record.get("response") or {}
    if record.get("error") or response.get("status_code") != 200:
        return "", "failed"

    choices = (response.get("body") or {}).get("choices") or []
    if not choices:
        return "", "failed"
    content = (choices[0].get("message") or {}).get("content") or ""

    text = _subject_from_json_object(content.strip()) or content
    cleaned = clean_subject_matter(str(text))
    return (cleaned, "ok") if cleaned else ("", "empty")


def load_jsonl(path: Path) -> list[dict]:
    """逐行读 JSONL，坏行直接报错并指出行号——静默跳过会让整表错位且没人发现。"""
    records: list[dict] = []
    # utf-8-sig：平台导出的文件可能带 BOM，带 BOM 时首行的 "{" 会解析失败
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"第 {line_number} 行不是合法 JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"第 {line_number} 行不是 JSON 对象: {text[:60]}")
            if "custom_id" not in record:
                raise ValueError(f"第 {line_number} 行缺少 custom_id 字段")
            records.append(record)
    return records


def apply_subject_matter(df: pd.DataFrame, records: list[dict]) -> tuple[pd.DataFrame, dict]:
    """把结果填进「标的物」列，返回 `(填好的副本, 统计)`。不修改传入的 DataFrame。"""
    stats = {
        "rows": len(df),
        "records": len(records),
        "filled": 0,
        "empty": 0,
        "failed": 0,
        "no_response": 0,
    }
    result = df.copy()
    if result.empty or not records:
        return result, stats

    # 越界一次性查完再报，避免填了一半才发现对不上
    out_of_range: list[int] = []
    parsed: list[tuple[int, dict]] = []
    for record in records:
        seq = parse_custom_id(record["custom_id"])
        if not 1 <= seq <= len(result):
            out_of_range.append(seq)
        else:
            parsed.append((seq, record))
    if out_of_range:
        sample = ", ".join(str(s) for s in out_of_range[:5])
        raise ValueError(
            f"{len(out_of_range)} 条结果的序号越界（表只有 {len(result)} 行，如 {sample}）——"
            f"结果文件与这张表不是一对，请确认用的是同一次 batch 对应的去重版 Excel。"
        )

    if SUBJECT_COLUMN not in result.columns:
        result[SUBJECT_COLUMN] = None
    column = result.columns.get_loc(SUBJECT_COLUMN)

    for seq, record in parsed:
        value, status = extract_subject_matter(record)
        if status == "ok":
            result.iloc[seq - 1, column] = value
            stats["filled"] += 1
        elif status in ("empty", "failed", "no_response"):
            stats[status] += 1

    # 请求文件当结果文件用是最容易犯的错，给一句能立刻定位的提示
    if stats["no_response"] == len(records):
        raise ValueError(
            f"{len(records)} 条记录都没有 response 字段——这看着是**请求文件**，"
            f"不是平台返回的结果文件。请用第 3 步产出的 batch_results_processed.jsonl。"
        )
    return result, stats


def load_data(path: Path) -> tuple[pd.DataFrame, str]:
    """读取第一个非空工作表，返回 `(数据, 工作表名)`。

    源文件常有空的 Sheet1、数据在 Sheet2 的情况，写死 sheet 名会在换数据时踩空。
    """
    excel = pd.ExcelFile(path)
    for sheet in excel.sheet_names:
        frame = excel.parse(sheet)
        if not frame.empty:
            return frame, sheet
    raise ValueError(f"文件里没有任何非空工作表: {path}")


def write_excel(df: pd.DataFrame, path: Path, sheet_name: str) -> None:
    try:
        df.to_excel(path, index=False, sheet_name=sheet_name)
    except PermissionError as exc:
        raise PermissionError(
            f"无法写入 {path}——文件可能正被 Excel 打开。请关闭后重试。"
        ) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="把 batch 结果里的标的物回填进 Excel")
    parser.add_argument("--src", type=Path, default=None, help="结果 JSONL（默认第 3 步产物）")
    parser.add_argument("--excel", type=Path, default=None, help="目标 Excel（默认自动探测）")
    parser.add_argument("-o", "--output", type=Path, default=None, help="输出 Excel")
    parser.add_argument("--dry-run", action="store_true", help="只统计，不写出文件")
    args = parser.parse_args(argv)

    _force_utf8()

    src = args.src or DEFAULT_SRC
    src = src if src.is_absolute() else (PROJECT_ROOT / src)
    if not src.exists():
        print(f"结果文件不存在: {src}")
        print("先把平台下载的结果跑一遍 python batch/按custom_id排序.py，或用 --src 指定。")
        return 1

    excel_path = args.excel or resolve_default_excel()
    excel_path = excel_path if excel_path.is_absolute() else (PROJECT_ROOT / excel_path)
    if not excel_path.exists():
        print(f"Excel 不存在: {excel_path}")
        print("先跑 python batch/去重.py 生成去重版，或用 --excel 指定。")
        return 1

    try:
        records = load_jsonl(src)
    except (ValueError, OSError) as exc:
        print(f"读取结果失败: {exc}")
        return 1

    try:
        raw, sheet_name = load_data(excel_path)
    except Exception as exc:
        print(f"读取 Excel 失败: {exc}")
        return 1

    if raw.empty:
        print("Excel 为空，无需回填。")
        return 0
    if not records:
        print(f"{src} 里没有任何结果记录。")
        return 0

    try:
        filled, stats = apply_subject_matter(raw, records)
    except ValueError as exc:
        print(f"回填失败: {exc}")
        return 1

    print(f"结果文件 : {src}")
    print(f"目标表   : {excel_path}（工作表 {sheet_name}）")
    print(f"数据行   : {stats['rows']}")
    print(f"结果记录 : {stats['records']}")
    print(f"已回填   : {stats['filled']}")
    if stats["empty"]:
        print(f"空答案   : {stats['empty']} 条（模型返回空，该行留空）")
    if stats["failed"]:
        print(f"失败     : {stats['failed']} 条（平台错误或非 200）")
    if stats["no_response"]:
        print(f"无结果   : {stats['no_response']} 条（缺 response 字段）")
    print(f"未填行   : {stats['rows'] - stats['filled']}（含第 2 步跳过空「项目名称」的行）")

    if args.dry_run:
        print("\n--dry-run：未写出文件。")
        return 0

    dest = args.output or resolve_output(excel_path)
    dest = dest if dest.is_absolute() else (PROJECT_ROOT / dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        write_excel(filled, dest, sheet_name)
    except PermissionError as exc:
        print(f"\n{exc}")
        return 1
    print(f"\n已写出: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

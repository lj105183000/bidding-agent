"""按「项目编号 + 中标金额」联合去重（数据管道第 1 步，见 docs/开发文档.md §4.2）。

**为什么用两个键联合去重**：同一项目可能因重新发布、变更公告产生多条记录，只按
项目编号去重会把"同一项目不同标段/不同次中标"的记录压掉——那些是真实的不同交易；
只按金额去重更荒谬，不同项目撞金额很常见。两个键一起才定位到"同一项目的同一次中标"。

**同组内保留哪一条**（按优先级）：

1. 空值最少的一条——信息最完整，后续提取标的物时可用字段更多；
2. 空值数相同时，取发布时间最新的一条——变更/更正公告通常后发且更准确。

约定（与其他 batch 脚本一致）：CWD 无关、支持 `--dry-run`、空数据提前返回、
Excel 被占用时给出可操作的提示。

用法::

    python batch/去重.py                      # 正式执行，结果写入 data/processed/<原名>_2.xlsx
    python batch/去重.py --dry-run            # 只统计不落盘
    python batch/去重.py --src <输入.xlsx> -o <输出.xlsx>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# 项目根：batch/去重.py → 上一级。所有默认路径都以此为基准，脚本可在任意 CWD 下运行
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SRC = PROJECT_ROOT / "data" / "raw" / "招标采购标的物信息提取训练数据.xlsx"
# 去重后的产出统一落到 data/processed，与原始数据分离，便于管道下游消费
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# 联合去重键。缺任一键都无法判定"是否同一笔交易"，这类行不参与去重（见 dedupe）
KEY_COLUMNS = ("项目编号", "中标金额")
# 同组内空值数相同时的次序依据，按顺序取第一个存在的列
DATE_COLUMNS = ("发布时间", "中标时间")

SUFFIX = "_2"


def _force_utf8() -> None:
    """Windows 控制台默认 GBK，中文日志容易乱码或抛 UnicodeEncodeError。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except Exception:
            pass


def resolve_output(src: Path) -> Path:
    """默认输出到 data/processed 目录的 `<原名>_2.xlsx`。"""
    return DEFAULT_PROCESSED_DIR / f"{src.stem}{SUFFIX}{src.suffix}"


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


def _pick_date_column(df: pd.DataFrame) -> str | None:
    for column in DATE_COLUMNS:
        if column in df.columns:
            return column
    return None


def dedupe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """执行去重，返回 `(去重后数据, 统计信息)`。

    **缺键的行不参与去重**：项目编号缺失时，无法判断两条记录是否指向同一笔交易，
    按金额硬合并会把不同项目错误地压成一条——宁可留着重复，也不能丢真实数据。
    """
    stats = {"input": len(df), "output": 0, "removed": 0, "skipped_no_key": 0}
    if df.empty:
        return df.copy(), stats

    missing_keys = [column for column in KEY_COLUMNS if column not in df.columns]
    if missing_keys:
        raise ValueError(
            f"数据缺少去重所需的列: {', '.join(missing_keys)}；"
            f"实际列: {', '.join(map(str, df.columns))}"
        )

    work = df.copy()

    # 排序 + 取每组第一条，把"保留哪条"的规则集中在一处表达
    work["_null_count"] = work.isna().sum(axis=1)
    date_column = _pick_date_column(work)
    if date_column:
        # 解析失败的日期置为 NaT，排序时排在最后（na_position="last"）
        work["_published"] = pd.to_datetime(work[date_column], errors="coerce")
    else:
        work["_published"] = pd.NaT

    # kind="stable" 是必须的：两个排序键都相同的组（实测很常见，同一笔交易的
    # "见证书"与"中标通知书"两条空值数与发布时间往往一致）若用默认的快速排序，
    # 保留哪条就成了未定义行为——同一份输入两次跑出不同结果，下游管道无法复现
    work = work.sort_values(
        ["_null_count", "_published"],
        ascending=[True, False],
        na_position="last",
        kind="stable",
    )

    has_key = work["项目编号"].notna()
    keyed = work[has_key].drop_duplicates(subset=list(KEY_COLUMNS), keep="first")
    unkeyed = work[~has_key]

    result = (
        pd.concat([keyed, unkeyed])
        .sort_index()
        .drop(columns=["_null_count", "_published"])
    )

    stats.update(
        output=len(result),
        removed=len(work) - len(result),
        skipped_no_key=int((~has_key).sum()),
    )
    return result, stats


def write_excel(df: pd.DataFrame, path: Path, sheet_name: str) -> None:
    try:
        df.to_excel(path, index=False, sheet_name=sheet_name)
    except PermissionError as exc:
        raise PermissionError(
            f"无法写入 {path}——文件可能正被 Excel 打开。请关闭后重试。"
        ) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="按「项目编号 + 中标金额」联合去重")
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC, help="输入 Excel")
    parser.add_argument("-o", "--output", type=Path, default=None, help="输出 Excel")
    parser.add_argument("--dry-run", action="store_true", help="只统计，不写出文件")
    args = parser.parse_args(argv)

    _force_utf8()
    src = args.src if args.src.is_absolute() else (PROJECT_ROOT / args.src)
    if not src.exists():
        print(f"输入文件不存在: {src}")
        return 1

    try:
        raw, sheet_name = load_data(src)
    except Exception as exc:
        print(f"读取失败: {exc}")
        return 1

    if raw.empty:
        print("数据为空，无需去重。")
        return 0

    result, stats = dedupe(raw)
    print(f"数据源 : {src}")
    print(f"工作表 : {sheet_name}")
    print(f"去重键 : {' + '.join(KEY_COLUMNS)}")
    print(f"原始行 : {stats['input']}")
    print(f"删除行 : {stats['removed']}")
    print(f"结果行 : {stats['output']}")
    if stats["skipped_no_key"]:
        print(
            f"无编号 : {stats['skipped_no_key']} 行（缺「项目编号」，无法判重，全部保留）"
        )

    if args.dry_run:
        print("\n--dry-run：未写出文件。")
        return 0

    dest = args.output or resolve_output(src)
    dest = dest if dest.is_absolute() else (PROJECT_ROOT / dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        write_excel(result, dest, sheet_name)
    except PermissionError as exc:
        print(f"\n{exc}")
        return 1
    print(f"\n已写出: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

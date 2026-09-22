"""统计标的物交易频次（数据管道第 5 步，见 docs/开发文档.md §4.2）。

**频次是什么**：第 1 步已按「项目编号+中标金额」联合去重，所以**每一行 = 一笔中标交易**。
某个标的物出现多少次，就是它被交易了多少次。

**只按字面统计，不做同义归并**：`空调` 与 `空调设备` 算两个标的物。归并是语义问题，
应该在第 2 步的提示词里解决（那是唯一能看见上下文的地方）；在这一步猜测性地合并，
只会把"到底合并了什么"变成没人能复现的暗箱。

**空值行不计入**：标的物没抽出来的行（模型返回空 / 平台失败 / 第 2 步跳过的空名称行）
不是一笔可统计的交易，硬当成一个空标的物去计数会污染榜首。

输出固定写 `data/processed/标的物_交易频次.xlsx`（契约指定，与第 4 步"写在源文件旁"
的约定不同——这是张新的汇总表，不属于任何一份源数据），两列：标的物、交易频次。

约定（与其他 batch 脚本一致）：CWD 无关、支持 `--dry-run`、空数据提前返回、
Excel 被占用时给出可操作的提示。

用法::

    python batch/统计数据_插交易频次.py                # → data/processed/标的物_交易频次.xlsx
    python batch/统计数据_插交易频次.py --dry-run      # 只统计不落盘
    python batch/统计数据_插交易频次.py --src <带标的物的表.xlsx> -o <输出.xlsx>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# 项目根：batch/统计数据_插交易频次.py → 上一级。所有默认路径都以此为基准，脚本可在任意 CWD 下运行
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# 输入候选，按优先级。第 4 步的产物名是从输入表名派生的，而输入表在人工归置中被改过几轮
# 名字（_去重版 在 raw、_2_ 之类），所以认不下时退到"data/ 下最新的 *_标的物.xlsx"。
DEFAULT_SRC_CANDIDATES = (
    DATA_DIR / "processed" / "招标采购标的物信息提取训练数据_2_标的物.xlsx",
    DATA_DIR / "processed" / "招标采购标的物信息提取训练数据_去重版_标的物.xlsx",
    DATA_DIR / "raw" / "招标采购标的物信息提取训练数据_去重版_标的物.xlsx",
)
SRC_GLOB = "*_标的物.xlsx"

# 契约指定的汇总表位置（不是"源文件旁"——它是新表，不属于任何一份源数据）
DEFAULT_OUTPUT = DATA_DIR / "processed" / "标的物_交易频次.xlsx"

SUBJECT_COLUMN = "标的物"
COUNT_COLUMN = "交易频次"
# 控制台预览几行
PREVIEW_ROWS = 5


def _force_utf8() -> None:
    """Windows 控制台默认 GBK，中文日志容易乱码或抛 UnicodeEncodeError。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except Exception:
            pass


def resolve_default_src() -> Path | None:
    """探测带标的物的表。

    先认候选里的确切名字（确定性优先），认不下再取 `data/` 下最新的 `*_标的物.xlsx`
    ——文件在人工归置中改过名，硬编码单一路径会让脚本在换机器时直接趴窝。
    都没有则返回 None，由调用方给出提示。
    """
    for candidate in DEFAULT_SRC_CANDIDATES:
        if candidate.exists():
            return candidate

    found = [p for p in DATA_DIR.rglob(SRC_GLOB) if p.is_file()]
    if not found:
        return None
    return max(found, key=lambda p: p.stat().st_mtime)


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


def count_subject_matter(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """统计频次，返回 `(两列表, 统计)`。不修改传入的 DataFrame。"""
    stats = {"rows": len(df), "counted": 0, "skipped_empty": 0, "unique": 0}
    empty_table = pd.DataFrame({SUBJECT_COLUMN: [], COUNT_COLUMN: []})
    if df.empty:
        return empty_table, stats

    if SUBJECT_COLUMN not in df.columns:
        raise ValueError(
            f"数据缺少「{SUBJECT_COLUMN}」列；实际列: {', '.join(map(str, df.columns))}"
        )

    values = df[SUBJECT_COLUMN].map(lambda v: "" if pd.isna(v) else str(v).strip())
    counted = values[values != ""]
    stats["counted"] = int(len(counted))
    stats["skipped_empty"] = int(len(values) - len(counted))
    stats["unique"] = int(counted.nunique())
    if counted.empty:
        return empty_table, stats

    table = (
        counted.value_counts().rename_axis(SUBJECT_COLUMN).reset_index(name=COUNT_COLUMN)
    )
    # 频次降序；并列时按标的物的码点升序（不是拼音序——这里要的是"确定"，不是"语感"）。
    # value_counts 对并列项的顺序不保证，显式排序才能让同一份输入两次跑出同一个文件
    # ——汇总表是要拿去比对、入库的，顺序不能随机
    table = table.sort_values(
        [COUNT_COLUMN, SUBJECT_COLUMN], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)
    table[COUNT_COLUMN] = table[COUNT_COLUMN].astype(int)
    return table, stats


def write_excel(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_excel(path, index=False)
    except PermissionError as exc:
        raise PermissionError(
            f"无法写入 {path}——文件可能正被 Excel 打开。请关闭后重试。"
        ) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="统计标的物交易频次（两列：标的物、交易频次）")
    parser.add_argument("--src", type=Path, default=None, help="带「标的物」列的表（默认自动探测）")
    parser.add_argument("-o", "--output", type=Path, default=None, help="输出 Excel")
    parser.add_argument("--dry-run", action="store_true", help="只统计，不写出文件")
    args = parser.parse_args(argv)

    _force_utf8()

    src = args.src or resolve_default_src()
    if src is None:
        print(f"没有找到带标的物的表（找的是 data/ 下的 {SRC_GLOB}）。")
        print("先跑 python batch/提数据_插标的物.py 生成，或用 --src 指定。")
        return 1
    src = src if src.is_absolute() else (PROJECT_ROOT / src)
    if not src.exists():
        print(f"输入文件不存在: {src}")
        return 1

    try:
        raw, sheet_name = load_data(src)
    except Exception as exc:
        print(f"读取失败: {exc}")
        return 1

    if raw.empty:
        print("数据为空，无需统计。")
        return 0

    try:
        table, stats = count_subject_matter(raw)
    except ValueError as exc:
        print(f"统计失败: {exc}")
        return 1

    print(f"数据源 : {src}")
    print(f"工作表 : {sheet_name}")
    print(f"数据行 : {stats['rows']}（每行 = 一笔中标交易）")
    print(f"已统计 : {stats['counted']}")
    if stats["skipped_empty"]:
        print(f"空标的物: {stats['skipped_empty']} 行（未抽出标的物的行不计入频次）")
    print(f"唯一数 : {stats['unique']}")

    if table.empty:
        print("\n没有任何「标的物」可统计，未写出文件。")
        return 0

    print(f"\n交易频次 Top {min(PREVIEW_ROWS, len(table))}:")
    for _, row in table.head(PREVIEW_ROWS).iterrows():
        print(f"  {row[COUNT_COLUMN]:>6}  {row[SUBJECT_COLUMN]}")

    if args.dry_run:
        print("\n--dry-run：未写出文件。")
        return 0

    dest = args.output or DEFAULT_OUTPUT
    dest = dest if dest.is_absolute() else (PROJECT_ROOT / dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        write_excel(table, dest)
    except PermissionError as exc:
        print(f"\n{exc}")
        return 1
    print(f"\n已写出: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

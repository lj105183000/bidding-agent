"""提取「项目名称」生成 Batch 请求文件（数据管道第 2 步，见 docs/开发文档.md §4.2）。

**这一步在干什么**：把去重版 Excel 里每一行的「项目名称」交给大模型，抽出**标的物**
——这次采购真正买的东西。第 4 步按行号把结果回填成「标的物」列，第 5 步再统计标的物
频次。所以本步产出的 custom_id 必须能反查回原始行号，否则后面两步无从对齐。

**custom_id 约定（第 3 / 4 步依赖，勿改）**：`request-<N>`，N 是该行在去重版 Excel 中的
**数据行序号**（第 1 条数据 = request-1，即 Excel 第 2 行，表头占第 1 行）。空「项目名称」
的行不发请求，序号因此留缺口——这是刻意的：序号即行号，缺口让第 4 步知道"这一行没有
结果"，而不是把后面的结果整体错位一行填进去。

**排序陷阱**：`request-10` 的字典序排在 `request-2` 前面，所以第 3 步"按 custom_id 排序"
必须按序号**数值**排（`int(custom_id.split("-")[-1])`），不能直接排字符串。

请求体格式照 https://docs.bigmodel.cn/cn/guide/tools/batch ：每行一个 JSON 对象，含
`custom_id` / `method` / `url` / `body` 四个字段；单文件上限 50000 行且不超过 100MB。
超限的文件上传时必然失败，所以本地直接拦下——与其让人在网页上传时才撞墙，不如当场报错。

约定（与其他 batch 脚本一致）：CWD 无关、支持 `--dry-run`、空数据提前返回、
目标文件被占用时给出可操作的提示。

用法::

    python batch/提数据_编JSONL.py                 # 写出 data/batch/batch_requests.jsonl
    python batch/提数据_编JSONL.py --dry-run       # 只统计不落盘
    python batch/提数据_编JSONL.py --limit 10      # 先试跑 10 条验证提示词
    python batch/提数据_编JSONL.py --model glm-4-air-250414

⚠ 模型名必须落在**批量接口**的支持列表内：`.env` 里的 `ZHIPU_MODEL=glm-4.7-flashx` 是
Chat 侧模型名，批量接口不认，平台会在上传时以 1210「模型名称错误」整批拒掉。默认值见
下面的 `BATCH_MODEL`。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# 项目根：batch/提数据_编JSONL.py → 上一级。所有默认路径都以此为基准，脚本可在任意 CWD 下运行
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# 与 src/config.py 同款：只填充未设置的变量，容器/CI 注入的真实环境变量优先
load_dotenv(PROJECT_ROOT / ".env", override=False)

# 去重版由第 1 步（去重.py）产出，固定落在 data/processed，文件名沿用
# `<原名>_2.xlsx` 的约定（见 去重.py 的 SUFFIX）。不再去 raw 目录探测——
# 原始数据与加工数据分离，避免管道里"读错了批次"的隐患。
DEFAULT_SRC = PROJECT_ROOT / "data" / "processed" / "招标采购标的物信息提取训练数据_2.xlsx"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "batch" / "batch_requests.jsonl"

# 要抽取的列。列名写死是有意的：换数据源时它必须是显式改动，而不是静默抽了另一列
MODEL_COLUMN = "项目名称"
# 智谱 batch 目前只支持这一个 endpoint（见官方文档的 endpoint 参数说明）
BATCH_URL = "/v4/chat/completions"
CUSTOM_ID_PREFIX = "request-"

# 批量任务用的模型，与 .env 的 ZHIPU_MODEL **不是一回事**：后者是 Chat 侧模型名
# （glm-4.7-flashx），批量接口不认——上传时会被平台以 1210「模型名称错误」整批拒掉，
# 所以这里不读 .env，要换用 --model 显式指定。取值须在平台批量接口的支持列表内
# （实测可用：glm-4-flashx-250414 / glm-4-air-250414 / glm-4-flash / glm-4-plus / glm-5.1）。
BATCH_MODEL = "glm-4-flashx-250414"

# 抽取是短输出任务，256 足够放下任何标的物词组；给太大只会让模型有机会"解释两句"
MAX_TOKENS = 256
TEMPERATURE = 0.1

# 智谱 batch 单文件硬限制（https://docs.bigmodel.cn/cn/guide/tools/batch）
MAX_REQUESTS_PER_FILE = 50_000
MAX_FILE_MB = 100.0

# 提示词。三条要求各有原因：
#   1) 去掉地区/年份/采购人等公文套话——它们对"买的是什么"没有信息量，
#      留着会让第 5 步的频次统计把同一个标的物拆成几十个变体；
#   2) 保留规格型号等限定词——"空调"和"5匹柜式空调"是不同的采购对象；
#   3) 只输出标的物本身——第 4 步直接取 message.content 当值用，多一句解释就得多写一层解析。
# 示例取自本项目真实数据，用真实句式做 few-shot 比抽象规则更管用。
SYSTEM_PROMPT = """你是招投标领域的标的物抽取器。标的物指这次采购真正买的东西：具体的货物、服务或工程。

抽取规则：
1. 去掉地区、年份、采购人/招标人名称，以及"项目""采购""招标""工程"等公文套话；
2. 保留标的物本身的规格、型号、材质、品牌、数量等限定词；
3. 一律沿用原文用词，不改写、不翻译、不脑补原文没有的信息；
4. 只输出标的物本身，不要引号、句号，不要任何解释。

示例：
项目名称：庐江县2023年大气污染防控精准溯源服务项目
标的物：大气污染防控精准溯源服务
项目名称：2023年信息化基础运维服务项目
标的物：信息化基础运维服务
项目名称：庐江县水务集团供水工程2023-2025年度胶圈采购
标的物：胶圈"""


def _force_utf8() -> None:
    """Windows 控制台默认 GBK，中文日志容易乱码或抛 UnicodeEncodeError。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except Exception:
            pass


def resolve_default_src() -> Path:
    """返回约定的去重版输入路径（data/processed 下的 `_2.xlsx`）。"""
    return DEFAULT_SRC


def default_model() -> str:
    """批量接口支持的默认模型；用 `--model` 覆盖。见 BATCH_MODEL 处的说明。"""
    return BATCH_MODEL


def load_data(path: Path) -> tuple[pd.DataFrame, str]:
    """读取第一个非空工作表，返回 `(数据, 工作表名)`。

    源文件常有空的 Sheet1、数据在 Sheet2 的情况，写死 sheet 名会在换数据时踩空。
    与 batch/去重.py 同款实现——两个脚本各自独立，不互相 import（脚本名带中文，
    import 需要 importlib 绕路，不如让这十行重复着）。
    """
    excel = pd.ExcelFile(path)
    for sheet in excel.sheet_names:
        frame = excel.parse(sheet)
        if not frame.empty:
            return frame, sheet
    raise ValueError(f"文件里没有任何非空工作表: {path}")


def make_custom_id(row_number: int) -> str:
    """行号 → custom_id。row_number 从 1 起，与 excel 数据行序号一致。"""
    return f"{CUSTOM_ID_PREFIX}{row_number}"


def build_request(row_number: int, model: str, project_name: str) -> dict:
    """拼一条 batch 请求。四个字段顺序无关，这里按官方文档的写法排。"""
    return {
        "custom_id": make_custom_id(row_number),
        "method": "POST",
        "url": BATCH_URL,
        "body": {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                # 项目名称嵌在 user 消息里，与系统提示词中的示例保持同一句式
                # （`项目名称：<原文>`），few-shot 的格式一致性直接决定抽取稳定性
                {"role": "user", "content": f"项目名称：{project_name}"},
            ],
            # 抽取要的是可复现，不是文采：默认采样温度下同一行两次能跑出不同标的物，
            # 第 5 步的频次统计就失去了可比性
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
        },
    }


def build_requests(
    df: pd.DataFrame, model: str, limit: int | None = None
) -> tuple[list[dict], dict]:
    """整个表 → 请求列表，返回 `(请求列表, 统计信息)`。

    **空「项目名称」的行跳过**（没法抽出标的物），但行号照旧占位，见模块开头的
    custom_id 约定。
    """
    stats = {"rows": len(df), "emitted": 0, "skipped_empty": 0}
    if df.empty:
        return [], stats

    if MODEL_COLUMN not in df.columns:
        raise ValueError(
            f"数据缺少「{MODEL_COLUMN}」列；实际列: {', '.join(map(str, df.columns))}"
        )

    requests: list[dict] = []
    for position, value in enumerate(df[MODEL_COLUMN].tolist()):
        if limit is not None and position >= limit:
            break
        # NaN / None / 纯空白都归为"没有项目名称"，别让 "  " 去消耗一次请求
        name = "" if pd.isna(value) else str(value).strip()
        if not name:
            stats["skipped_empty"] += 1
            continue
        requests.append(build_request(position + 1, model, name))

    stats["emitted"] = len(requests)
    return requests, stats


def render_jsonl(requests: list[dict]) -> str:
    """序列化为 JSONL 文本（每行一个对象，末尾留换行）。

    `ensure_ascii=False` 是必须的：转义成 `\\uXXXX` 的中文在人工抽查时完全不可读，
    而 batch 文件动辄上千行，抽查是常态。
    """
    if not requests:
        return ""
    return "\n".join(json.dumps(r, ensure_ascii=False) for r in requests) + "\n"


def limit_violation(count: int, payload: str) -> str | None:
    """超出智谱单文件限制时返回可读原因，否则 None。"""
    if count > MAX_REQUESTS_PER_FILE:
        return (
            f"请求数 {count} 超过单文件上限 {MAX_REQUESTS_PER_FILE} 行。"
            f"本脚本只处理首个非空工作表，需要分批请先拆分输入表再逐个生成。"
        )
    size_mb = len(payload.encode("utf-8")) / 1024 / 1024
    if size_mb > MAX_FILE_MB:
        return (
            f"文件约 {size_mb:.1f}MB 超过单文件上限 {MAX_FILE_MB:g}MB，请分批生成。"
        )
    return None


def write_text(path: Path, payload: str) -> None:
    try:
        path.write_text(payload, encoding="utf-8")
    except PermissionError as exc:
        raise PermissionError(
            f"无法写入 {path}——文件可能正被其他程序打开。请关闭后重试。"
        ) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="提取「项目名称」生成 Batch 请求文件（智谱 /v4/chat/completions）"
    )
    parser.add_argument("--src", type=Path, default=None, help="输入 Excel（默认自动探测）")
    parser.add_argument("-o", "--output", type=Path, default=None, help="输出 JSONL")
    parser.add_argument("--model", default=None, help="请求体里的模型名（默认取 ZHIPU_MODEL）")
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 行（试跑用）")
    parser.add_argument("--dry-run", action="store_true", help="只统计，不写出文件")
    args = parser.parse_args(argv)

    _force_utf8()

    if args.limit is not None and args.limit <= 0:
        print("--limit 必须为正整数。")
        return 1

    model = args.model or default_model()
    src = args.src or resolve_default_src()
    src = src if src.is_absolute() else (PROJECT_ROOT / src)
    if not src.exists():
        print(f"输入文件不存在: {src}")
        print("先跑 python batch/去重.py 生成去重版，或用 --src 指定输入。")
        return 1

    try:
        raw, sheet_name = load_data(src)
    except Exception as exc:
        print(f"读取失败: {exc}")
        return 1

    if raw.empty:
        print("数据为空，无需生成请求。")
        return 0

    try:
        requests, stats = build_requests(raw, model, args.limit)
    except ValueError as exc:
        # 缺列是"数据换了一批"的常见症状，给一句能照做的提示，别甩一串堆栈
        print(f"生成失败: {exc}")
        return 1

    print(f"数据源 : {src}")
    print(f"工作表 : {sheet_name}")
    print(f"模型   : {model}")
    print(f"数据行 : {stats['rows']}")
    if args.limit is not None:
        # 否则"数据行 8751 / 生成数 10"看着像出错了
        print(f"限制   : 只处理前 {args.limit} 行（--limit）")
    print(f"生成数 : {stats['emitted']}")
    if stats["skipped_empty"]:
        print(f"跳过   : {stats['skipped_empty']} 行（「{MODEL_COLUMN}」为空）")

    if not requests:
        print("\n没有可抽取的「项目名称」，未生成任何请求。")
        return 0

    payload = render_jsonl(requests)
    violation = limit_violation(len(requests), payload)
    if violation:
        print(f"\n{violation}")
        return 1

    if args.dry_run:
        print("\n--dry-run：未写出文件。")
        return 0

    dest = args.output or DEFAULT_OUTPUT
    dest = dest if dest.is_absolute() else (PROJECT_ROOT / dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        write_text(dest, payload)
    except PermissionError as exc:
        print(f"\n{exc}")
        return 1
    print(f"\n已写出: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

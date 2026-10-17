"""从工具输出里提取**可图表化的结构化数据**，随 `done` 帧下发给前端。

★ 为什么需要这一层（而不是让前端解析纯文本）
====================================
工具执行器的契约是 `(文本, sources)` —— 返回的是**已经格式化好的中文文本**：

    业务数据库查询结果（共 6 条）：
    标的物 | 数量
    --------
    空调 | 175
    打印/复印纸的网上超市采购 | 160
    ...

    知识图谱规模：
    - 采购单位：3577
    - 供应商：4138

前端拿到的 `content` 是**模型写的回答**（自然语言 + 可能的 Markdown 表格），
而这份工具原文只存在于后端。若让前端去解析模型回答，会有两个问题：
    ① 模型可能改写、省略甚至编造数字 —— 图表必须画**工具给的真实数据**
    ② 同一份数据在回答里出现的形式不固定，解析规则会非常脆

故在**工具结果产生处**提取（那里是唯一事实来源），以结构化形式随 done 帧下发。

★ 为什么不改工具执行器直接返回结构化数据
==============================
那要改 `TOOL_EXECUTORS` 的冻结契约（`(文本, sources)`）与全部 8 个执行器，
波及面远大于收益。这里的提取是**纯函数、只读**，且失败时静默返回空
（图表是增益项，不能影响问答）。

★ 设计原则：宁可漏，不可错
====================
图表画错数字比不画图表有害得多。故这里对"什么算可图表数据"设了**严格**门槛
（见 `_parse_table` / `_parse_kv_list` 的注释），实测过的真实形态全部覆盖，
但形状稍有偏差就放弃——放弃的代价只是少一张图。
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

#: 单张图表最多几条数据。
#:
#: 超过这个数量横向条形图会挤成一团。截断而不是全画，并在 note 里说明
#: （`chart.note`），让用户知道"图里不是全部"。
MAX_BARS = 12

#: 一个回答最多下发几张图。
#:
#: ★ 值定义在 `constants` 里，与 `react_loop` 共用**同一个**上限。
#: 第一版这里也写死了一个 3，两处各有一个 —— 改一处忘另一处就会出现
#: "提取了 5 张但只能下发 3 张"这类对不上的行为。
from src.agent.constants import MAX_CHARTS  # noqa: E402

#: 数值解析：允许千分位、小数、负数与百分号结尾。
_NUMBER_RE = re.compile(r"^-?[\d,]+(?:\.\d+)?%?$")

#: Markdown 表格的分隔行，如 `--------` 或 `--- | ---`。
_SEP_RE = re.compile(r"^[\s|:-]+$")

#: 键值行：`- 采购单位：3577` / `- 代理机构: 325`
_KV_RE = re.compile(r"^\s*[-*•]?\s*([^：:]{1,40})[：:]\s*([\d,]+(?:\.\d+)?%?)\s*$")

#: 括号内数值的排名行：`1. 空调（交易频次 175）` / `2. 粉盒（数量 150）`
#:
#: ★ 这是**第三种**真实形态（知识图谱的 top_traded 用它，与数据库的
#: Markdown 表格不同）。第一版只认表格与 `键：值`，于是漏了它——
#: 实测发现后补上。三处都覆盖，是因为这些输出格式由**不同工具**生成，
#: 格式不统一是既成事实，提取器必须同时容忍。
_PAREN_RANK_RE = re.compile(
    r"^\s*\d+\s*[.、)]\s*(?P<label>[^（(]{1,60}?)\s*[（(][^）)]*?(?P<value>[\d,]+(?:\.\d+)?)\s*[）)]\s*$"
)


def _to_number(text: str) -> float | None:
    """把单元格文本转成数字；转不了返回 None。

    ★ 为什么必须**严格**
    ----------------
    若放宽到"能提取出数字就算"（如用正则搜第一串数字），那么
    `2023-10`（时间）、`23996.00 元`、`第二十七条第 2 款` 都会被当成数值，
    于是把一张**项目明细表**画成"按项目名的金额排行"——数字对但语义错，
    比不画更误导。故这里要求整格**恰好**是一个数字（允许千分位与百分号）。
    """
    t = (text or "").strip()
    if not t or not _NUMBER_RE.match(t):
        return None
    is_pct = t.endswith("%")
    t = t.rstrip("%").replace(",", "")
    try:
        v = float(t)
    except ValueError:
        return None
    return v


def _parse_table(lines: list[str], start: int) -> tuple[dict | None, int]:
    """尝试从 `lines[start]` 起解析一个 Markdown 表格。

    返回 `(chart 或 None, 消费的行数)`。

    ★ 接受条件（三条同时满足，否则放弃）
    ------------------------------
    ① **恰好两列**。真实数据里所有可图表的表都是"类别 | 数值"两列
       （标的物/地区/供应商/采购单位 × 数量）。五列的明细表
       （项目名称|标的物|采购单位|供应商|中标金额）**不是**排行的语义，
       画成条形图会变成"按项目名的金额排行"——那不是用户要的。
    ② 有分隔行（`--------`）。这是"这确实是表格"的强信号，
       避免把正文里偶然含 `|` 的句子当成表。
    ③ 数值列**每一行**都能解析成数字。只要有一行不是数字就整表放弃——
       部分解析会画出缺项的图，比不画更容易被误读。
    """
    header = lines[start]
    if "|" not in header:
        return None, 0
    # 分隔行必须在表头下一行
    if start + 1 >= len(lines) or not _SEP_RE.match(lines[start + 1].strip()):
        return None, 0

    cols = [c.strip() for c in header.split("|")]
    if len(cols) != 2:
        return None, 0
    label_col, value_col = cols[0], cols[1]
    if not label_col or not value_col:
        return None, 0

    rows: list[dict[str, Any]] = []
    i = start + 2
    while i < len(lines):
        raw = lines[i]
        if "|" not in raw:
            break
        cells = [c.strip() for c in raw.split("|")]
        if len(cells) != 2:
            break
        value = _to_number(cells[1])
        if value is None:
            # ③ 一行解析失败 → 整表放弃
            return None, 0
        label = cells[0]
        if not label:
            return None, 0
        rows.append({"label": label, "value": value})
        i += 1

    if len(rows) < 2:
        # 只有一行画不出"分布"，多半是误匹配
        return None, 0

    consumed = i - start
    return (
        {
            "kind": "bar",
            "title": _title_before(lines, start),
            "value_label": value_col,
            "label_label": label_col,
            "data": rows[:MAX_BARS],
            "note": (
                f"共 {len(rows)} 项，图中显示前 {MAX_BARS} 项"
                if len(rows) > MAX_BARS else ""
            ),
        },
        consumed,
    )


def _parse_kv_list(lines: list[str], start: int) -> tuple[dict | None, int]:
    """尝试从 `lines[start]` 起解析"键：数值"列表。

    真实形态（知识图谱规模 / 数据库统计概览）：
        知识图谱规模：
        - 代理机构：325
        - 地区：38
        - 采购单位：3577

    ★ 接受条件
    --------
    ① **至少 3 行**都能匹配 `键：数值`。少于 3 行更像正文里的偶然短语。
    ② 键**不重复**。重复键说明这不是"规模统计"，而是别的结构。
    ③ 行首必须是 `-`/`*`/`•` 或直接是键——不接受缩进的正文句子。

    这三个条件把"统计概览"与"正文里提到某个数字"区分开。
    """
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    i = start
    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()
        if not stripped:
            break
        m = _KV_RE.match(raw)
        if not m:
            break
        key = m.group(1).strip()
        value = _to_number(m.group(2))
        if value is None or not key or key in seen:
            break
        seen.add(key)
        rows.append({"label": key, "value": value})
        i += 1

    if len(rows) < 3:
        return None, 0
    return (
        {
            "kind": "bar",
            "title": _title_before(lines, start) or "统计概览",
            "value_label": "数值",
            "label_label": "项目",
            "data": rows[:MAX_BARS],
            "note": "",
        },
        i - start,
    )


def _parse_rank_list(lines: list[str], start: int) -> tuple[dict | None, int]:
    """尝试解析"序号 + 名称（说明 数值）"形态的排名列表。

    真实形态（知识图谱 `top_traded`）：
        采购频次最高的标的物（前 5 名）：
        1. 空调（交易频次 175）
        2. 打印/复印纸的网上超市采购（交易频次 160）

    ★ 接受条件
    --------
    ① **至少 3 行**都能匹配。少于 3 行更像正文里的编号列举。
    ② 行首必须是"数字 + `.`/`、`/`)`"，且**整行**匹配到结尾——
       正文里的"1. 只能从唯一供应商处采购的：1"不会匹配
       （它的数字在冒号后，不在括号里）。
    """
    rows: list[dict[str, Any]] = []
    i = start
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            break
        m = _PAREN_RANK_RE.match(lines[i])
        if not m:
            break
        label = m.group("label").strip()
        value = _to_number(m.group("value"))
        if value is None or not label:
            break
        rows.append({"label": label, "value": value})
        i += 1

    if len(rows) < 3:
        return None, 0
    return (
        {
            "kind": "bar",
            "title": _title_before(lines, start) or "排名",
            # 括号里的说明文字形如"交易频次"，取不到就用通用名
            "value_label": "数值",
            "label_label": "名称",
            "data": rows[:MAX_BARS],
            "note": (
                f"共 {len(rows)} 项，图中显示前 {MAX_BARS} 项"
                if len(rows) > MAX_BARS else ""
            ),
        },
        i - start,
    )


def _title_before(lines: list[str], index: int) -> str:
    """取表格/列表**上一行**作为标题（形如"业务数据库查询结果（共 6 条）："）。

    取不到就返回空串——不编造标题（图表上宁可不显示标题，
    也不要写一个与数据不符的名字）。
    """
    for k in range(index - 1, max(-1, index - 3), -1):
        prev = lines[k].strip()
        if not prev:
            continue
        if prev.endswith(("：", ":")):
            return prev.rstrip("：:")
        return prev[:40]
    return ""


def extract_charts(tool_text: str, tool_name: str = "") -> list[dict]:
    """从一段工具输出里提取可图表的系列。**永不抛异常**。

    `tool_name` 只用于日志（便于定位"哪个工具的格式变了导致提取不到"）。
    """
    text = str(tool_text or "")
    # 快速排除：两种可图表形态都必须含 `|`（表格）或全角冒号（键值）
    if "|" not in text and "：" not in text:
        return []
    try:
        lines = text.splitlines()
        charts: list[dict] = []
        i = 0
        while i < len(lines) and len(charts) < MAX_CHARTS:
            chart, consumed = _parse_table(lines, i)
            if chart is None:
                chart, consumed = _parse_rank_list(lines, i)
            if chart is None:
                chart, consumed = _parse_kv_list(lines, i)
            if chart is not None and consumed > 0:
                chart["source_tool"] = tool_name
                charts.append(chart)
                i += consumed
            else:
                i += 1
        return charts
    except Exception:  # noqa: BLE001 - 图表是增益项，任何异常都不影响问答
        logger.debug("图表数据提取失败（不影响回答）tool=%s", tool_name, exc_info=True)
        return []

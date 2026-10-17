"""工具文本防御：识别并处理模型把函数调用格式当正文输出的情况。

为什么需要：部分模型（尤其本地小模型）会用文本形式输出工具调用 JSON，
直接下发会污染用户可见的回答。策略是先检测 → 归一化 → 解析，
上层据此决定「约束重试」（决策轮）或「reset + 兜底执行」（流式轮）。

这里冻结的统一格式（LLM 客户端需把 API tool_calls 序列化为同一文本格式）：
  {"name": "工具名", "arguments": {...}}
多个调用可放在 {"tool_calls": [...]} 数组里，或用 <tool_call> 标签/代码围栏包裹。
"""

import json
import logging
import re

logger = logging.getLogger(__name__)

# 代码围栏：<tool_call> 标签：疑似调用起始（用于流式缓冲判定）
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)
_TOOL_TAG_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
_TOOL_JSON_RE = re.compile(r"\{\s*\"name\"\s*:")
_TOOL_CALLS_RE = re.compile(r"[\"']tool_calls[\"']")

# ---------------------------------------------------------------------------
# XML / DSML 风格的工具调用（2026-10 补）
#
# ★ 为什么要补这一族（实测缺口，不是猜测）
# ------------------------------------
# 原实现只查 3 个子串：`<tool_call>` / `tool_calls` 关键字 / `{"name":`。
# 而项目自己的文档（docs/组件工作机制.md §8）声称检测范围含
# "XML 标签: tool / invoke / function_call、DSML 标记、裸函数名"——
# **文档描述的是设想的系统，代码只实现了一个子集。**
#
# 实测场景：泄漏标记为「全角竖线 + DSML + 全角竖线」包裹的 XML：
#     <FW>DSML<FW>tool_calls>
#     <FW>DSML<FW>invoke name="search_knowledge_graph">
#     <FW>DSML<FW>parameter name="keyword" string="true">管理局</parameter>
#     </invoke>
#
# DeepSeek **官方 API** 走原生 tool_calls 字段时不会这样输出
# （实测 5 种组合——非流式/流式 × 思考开/关 × 单/多实体——全部返回
# 结构化 tool_calls，content 里无任何工具文本）。会见到它的场景是：
#     · 第三方 OpenAI 兼容网关 / 中转（把 tool_calls 降级成文本）
#     · 不具备原生 function calling 的本地小模型
#     · 上游把 provider 的 XML 风格工具协议原样透传
# 本项目对接的不止官方 API，故这层防线必须真的覆盖它。
# ---------------------------------------------------------------------------

#: 全角竖线（U+FF5C）。模型偶尔用全角，也可能用 ASCII 的 `||`，两者都要认。
_FW_BAR = "\uff5c"

#: DSML 标记：竖线 + DSML + 竖线。区分大小写地宽松匹配（`dsml` 也认）。
#:
#: 为什么单独建一个正则而不是并进 `_TOOL_TEXT_MARKERS`：
#: 解析阶段需要**去掉**它才能拿到干净的标签名与参数值。
_DSML_MARK_RE = re.compile(
    rf"(?:{_FW_BAR}{{1,2}}|\|{{1,2}})\s*DSML\s*(?:{_FW_BAR}{{1,2}}|\|{{1,2}})",
    re.IGNORECASE,
)

#: 工具文本的"标记家族"。命中任一即判定为疑似工具调用。
#:
#: ★ 为什么按家族匹配而不是逐个字符串
#: -----------------------------
#: 逐个 `in` 检查的问题不是慢，而是**每遇到新变体就要加一行**，
#: 且漏了不会有任何提示（原实现就这么漏掉了 DSML）。
#: 用正则族一次覆盖：XML 标签名（tool_call / invoke / function_call /
#: tool_calls / parameter）、DSML 标记、裸函数名调用形态。
#: 标签名前可能出现的"噪音"：
#:   · `\s*`  —— DSML 标记与标签之间的空白（实测：`||DSML|| invoke` 有空格）
#:   · `<!--\s*` —— 有些网关把协议标记包成 XML 注释
#:   · `(?:\w+:)?` —— 命名空间前缀（实测：`<ns:invoke ...>`）
_TAG_NOISE = r"(?:<!--\s*)?(?:\w+:)?"
_TOOL_TEXT_MARKERS = (
    re.compile(rf"<\s*/?\s*{_TAG_NOISE}tool_calls?\b", re.IGNORECASE),
    re.compile(rf"<\s*/?\s*{_TAG_NOISE}invoke\b", re.IGNORECASE),
    re.compile(rf"<\s*/?\s*{_TAG_NOISE}function_call\b", re.IGNORECASE),
    re.compile(rf"<\s*/?\s*{_TAG_NOISE}parameter\b", re.IGNORECASE),
    re.compile(rf"<\s*/?\s*{_TAG_NOISE}tool\b", re.IGNORECASE),
    _DSML_MARK_RE,
)

#: DSML 标记与其后标签之间的空白/注释前缀。
#:
#: ★ 为什么解析前也要清它
#: -------------------
#: 实测：`||DSML|| invoke name=...` 里标记后有个**空格**。`_strip_dsml`
#: 只去标记，剩下的 ` invoke` 前缀会让 `_INVOKE_BLOCK_RE` 的
#: `<\s*(?:\w+:)?invoke` 匹配失败（`<` 后面直接跟空格再跟字母，
#: 而 `\s*` 允许空格，但 `(?:\w+:)?` 之后要求紧接 `invoke`——
#: 实际是能匹配的，问题在于**标记本身没被去掉时** `<||DSML|| invoke`）。
#: 这里统一把"标记 + 其后的空白/注释"一起清掉，让剩余文本是干净 XML。
_DSML_LEAD_RE = re.compile(
    _DSML_MARK_RE.pattern + r"\s*(?:<!--\s*)?",
    re.IGNORECASE,
)

#: XML 风格 invoke 块：`<invoke name="x">` + 若干 `<parameter name="k">v</parameter>`。
#:
#: 宽松点（都因实测泄漏样本而必须）：
#:   · `invoke` **允许不闭合**——泄漏样本里只有开标签与 parameter，没有收尾
#:   · 标签内可有额外属性（样本里 parameter 带 `string="true"`）
#:   · 属性值单双引号都要认
_INVOKE_BLOCK_RE = re.compile(
    r"<\s*(?:\w+:)?invoke\b(?P<attrs>[^>]*)>(?P<body>.*?)(?=<\s*(?:\w+:)?invoke\b|\Z)",
    re.IGNORECASE | re.S,
)
_PARAM_BLOCK_RE = re.compile(
    r"<\s*(?:\w+:)?parameter\b(?P<attrs>[^>]*)>(?P<value>.*?)"
    r"(?=<\s*(?:\w+:)?parameter\b|<\s*/\s*(?:\w+:)?parameter\b|\Z)",
    re.IGNORECASE | re.S,
)
_ATTR_NAME_RE = re.compile(r"name\s*=\s*[\"\']([^\"\']+)[\"\']", re.IGNORECASE)

#: 裸函数名调用形态：`search_knowledge_base(` 或 `search_knowledge_base({...})`。
#:
#: 只在"后面紧跟左括号"时才算（避免把正文里提到工具名误判）：
#: 提到"可以用 search_knowledge_base 检索"是正常说明，不该触发兜底执行。
_BARE_CALL_RE = re.compile(r"\b([a-z_][a-z0-9_]{3,})\s*\(")


def _attr(text: str, key: str = "name") -> str:
    """从标签属性串里取 `key="value"` 的值，取不到返回空串。"""
    found = _ATTR_NAME_RE.findall(text or "")
    return found[0].strip() if found else ""


def _strip_dsml(text: str) -> str:
    """去掉 DSML 标记**及其后的空白/注释前缀**，保留标签名。

    泄漏样本里标记是**前缀**：`<FW>DSML<FW>tool_calls>`、
    `<FW>DSML<FW>invoke name="...">`。去掉标记后剩下的就是标准 XML，
    可直接复用同一解析器——这正是把标记做成独立正则的目的。

    ★ 连"标记后的空白"一起清（实测踩到）
    --------------------------------
    变体 `||DSML|| invoke name="..."`（ASCII 竖线 + **一个空格**）：
    只去标记会留下 ` invoke`，虽然 `<\\s*` 能容忍空白，但若标记紧贴
    `<` 写成 `<||DSML|| invoke`，清掉标记后就变成 `< invoke`——
    标签名与 `<` 之间隔了空格，某些更严格的正则就认不出。
    统一清掉"标记 + 空白"，剩余文本一定是紧凑的 XML。
    """
    return _DSML_LEAD_RE.sub("", text or "")


def _looks_like_tool_text(text: str) -> bool:
    """判断文本是否疑似工具调用格式。宁可疑高不可漏：漏检会把调用 JSON 直接展示给用户。

    ★ 检测口径按"标记家族"而非逐个字符串
    --------------------------------
    原实现是四个 `if ... in text`。这种写法的问题是：**每遇到一种新变体
    就要补一行，而漏了不会有任何提示**——DSML 标记就是这么漏掉的。

    现在改成：先扫 `_TOOL_TEXT_MARKERS` 家族正则（覆盖 XML 标签名、
    DSML 标记），再退回 JSON 形态。新增变体只需往家族里加一条正则，
    且 `tests/test_tool_defense.py` 有一组"形态清单"逐项断言，
    漏了会在测试里暴露。
    """
    if not text:
        return False
    for pat in _TOOL_TEXT_MARKERS:
        if pat.search(text):
            return True
    if _TOOL_CALLS_RE.search(text):
        return True
    if _TOOL_JSON_RE.search(text):
        return True
    # 裸函数名调用：仅当形如 `name(` 且名字像工具名时才算。
    # 放宽会误判正常说明文字（如"可以用 search_knowledge_base 检索"），
    # 故这里要求**紧跟左括号**。
    if _BARE_CALL_RE.search(text) and _TOOL_CALLS_RE.search(text) is None:
        # 仅当同时出现工具名痕迹时才算，避免把普通"函数()"写法误判
        return bool(re.search(r"\b(?:search_|query_|check_)\w*\s*\(", text))
    return False


def _normalize_tool_text(text: str) -> str:
    """去掉代码围栏、`<tool_call>` 标签与 **DSML 标记**，提取纯片段便于解析。

    ★ DSML 标记也必须在这里去掉（原实现漏了）
    ------------------------------------
    标记是**前缀**形态：`<FW>DSML<FW>invoke name="...">`。若不清掉，
    后续的 XML 正则虽然能靠 `<?\\w+:` 之类的宽松匹配勉强认出标签名，
    但参数值里也会夹带标记，使 `arguments` 变成脏数据。
    统一在归一化阶段清掉，解析器就只需面对标准 XML。
    """
    t = _strip_dsml((text or "").strip())
    if "<tool_call>" in t:
        t = "\n".join(_TOOL_TAG_RE.findall(t))
    fences = _CODE_FENCE_RE.findall(t)
    if fences:
        t = "\n".join(fences)
    return t.strip()


def _parse_xml_tool_calls(text: str) -> list:
    """解析 XML / DSML 风格的 invoke 块，返回 `[{"name", "arguments"}]`。

    ★ 支持的形态（都由实测泄漏样本决定）
    ------------------------------
        <invoke name="search_knowledge_graph">
          <parameter name="keyword" string="true">管理局</parameter>
          <parameter name="query_type">search_entity</parameter>
        </invoke>

    必须在以下几处**宽松**，否则真实泄漏样本解析不出来：

    ① **invoke 允许不闭合**：泄漏样本里往往只有开标签与 parameter，
       没有收尾 `</invoke>`（模型被截断或本就不写）。故用"遇到下一个
       invoke 或字符串结束"作为边界，而不是要求闭合标签。
    ② **标签内可有额外属性**：样本里 parameter 带 `string="true"`。
       只取 `name`，忽略其余。
    ③ **单双引号都要认**。
    ④ **DSML 前缀要先去掉**：标记是前缀（`<标记>invoke ...>`），
       去掉后就是标准 XML，可复用同一解析器。

    解析失败返回空列表（与 `_parse_tool_text` 契约一致，不抛异常）。
    """
    if not text:
        return []
    cleaned = _strip_dsml(text)
    calls: list[dict] = []
    for block in _INVOKE_BLOCK_RE.finditer(cleaned):
        name = _attr(block.group("attrs"))
        if not name:
            continue
        args: dict = {}
        for pm in _PARAM_BLOCK_RE.finditer(block.group("body") or ""):
            key = _attr(pm.group("attrs"))
            if not key:
                continue
            # 值可能被 DSML 标记与空白污染，统一清洗
            value = _strip_dsml(pm.group("value") or "").strip()
            # 去掉可能残留的收尾标签
            value = re.sub(r"<\s*/\s*(?:\w+:)?parameter\s*>", "", value).strip()
            if value:
                args[key] = value
        calls.append({"name": name, "arguments": args})

    # 同名同参去重（模型偶发重复输出同一调用，重复执行会放大后端压力）
    seen, unique = set(), []
    for call in calls:
        key = (call["name"], json.dumps(call["arguments"], sort_keys=True, ensure_ascii=False))
        if key not in seen:
            seen.add(key)
            unique.append(call)
    return unique


def _parse_tool_text(text: str) -> list:
    """从文本中解析工具调用，返回 [{"name": str, "arguments": dict}]。

    解析失败一律返回空列表而不抛异常——调用方据此区分「无调用」与「格式损坏」，
    两者都走无工具分支，Agent 不因模型输出畸形而崩溃。
    """
    # ★ XML / DSML 风格优先：它的形态与 JSON 完全不同，先试它，
    #   失败再走原来的 JSON 路径。两者都失败就返回空列表。
    xml_calls = _parse_xml_tool_calls(text)
    if xml_calls:
        logger.info("从 XML/DSML 文本解析出工具调用: %s", [c["name"] for c in xml_calls])
        return xml_calls

    t = _normalize_tool_text(text)
    if not t:
        return []

    candidates = [t]
    # 逐个 {...} 块尝试：模型可能一次输出多个调用或夹带说明文字
    candidates.extend(re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", t))

    calls = []
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(obj, dict):
            continue
        items = obj.get("tool_calls")
        if items is None and "name" in obj:
            items = [obj]
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict) or not item.get("name"):
                continue
            args = item.get("arguments", item.get("parameters", {}))
            if isinstance(args, str):
                # 部分模型把 arguments 序列化成字符串，尝试二次解析
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {}
            if not isinstance(args, dict):
                args = {}
            calls.append({"name": str(item["name"]), "arguments": args})

    # 同名同参去重：模型偶发重复输出同一调用，重复执行会放大后端压力
    seen, unique = set(), []
    for call in calls:
        key = (call["name"], json.dumps(call["arguments"], sort_keys=True, ensure_ascii=False))
        if key not in seen:
            seen.add(key)
            unique.append(call)
    return unique


class ToolDefenseMixin:
    """以方法形式暴露模块级函数，便于与 ReActMixin / GenerationMixin 组合及测试调用。"""

    @staticmethod
    def _detect_tool_text(text: str) -> bool:
        return _looks_like_tool_text(text)

    @staticmethod
    def _normalize_tool_text(text: str) -> str:
        return _normalize_tool_text(text)

    @staticmethod
    def _parse_tool_text(text: str) -> list:
        calls = _parse_tool_text(text)
        if not calls and text and _looks_like_tool_text(text):
            logger.debug("工具文本检测命中但解析为空，按无调用处理")
        return calls

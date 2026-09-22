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


def _looks_like_tool_text(text: str) -> bool:
    """判断文本是否疑似工具调用格式。宁可疑高不可漏：漏检会把调用 JSON 直接展示给用户。"""
    if not text:
        return False
    if "<tool_call>" in text:
        return True
    if _TOOL_CALLS_RE.search(text):
        return True
    if _TOOL_JSON_RE.search(text):
        return True
    return False


def _normalize_tool_text(text: str) -> str:
    """去掉代码围栏与 <tool_call> 标签，提取出纯 JSON 片段便于解析。"""
    t = (text or "").strip()
    if "<tool_call>" in t:
        t = "\n".join(_TOOL_TAG_RE.findall(t))
    fences = _CODE_FENCE_RE.findall(t)
    if fences:
        t = "\n".join(fences)
    return t.strip()


def _parse_tool_text(text: str) -> list:
    """从文本中解析工具调用，返回 [{"name": str, "arguments": dict}]。

    解析失败一律返回空列表而不抛异常——调用方据此区分「无调用」与「格式损坏」，
    两者都走无工具分支，Agent 不因模型输出畸形而崩溃。
    """
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

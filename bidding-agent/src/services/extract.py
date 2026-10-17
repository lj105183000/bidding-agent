"""标讯智能提取服务：公告文本 / 公告截图 → 17 个结构化字段。

两种输入，同一个输出
--------------------
    text  → 大模型按 schema 抽 JSON
    image → 视觉模型按同一个 schema 抽 JSON（复用 src/clients/vision.py）

两条路都经 `bolt_schema.normalize_record` 归一成**同一份字段结构**，
故前端与入库侧不必区分来源。

设计要点
--------
- **提示词与 schema 同源**：`extraction_prompt()` 由字段定义生成，
  新增字段只需改 `bid_schema` 一处，不会出现"prompt 漏字段"。
- **JSON 解析容错**：模型偶发加 Markdown 代码块或前后说明文字，
  解析侧三级兜底（直接解析 → 剥代码块 → 截取首尾大括号）。
- **不编造**：提示词明确要求缺失字段留空；解析后仍会剔除明显非法的值
  （如金额为 0 或负数、日期早于 2000 年）。
- **可选入库**：`save=True` 时写入 `bidding_records`（表结构与列映射已具备）。
"""

from __future__ import annotations

import json
import re
import threading
from typing import Any

from src.constants.bid_schema import (
    BID_FIELDS,
    completeness,
    extraction_prompt,
    normalize_record,
)
from src.logging_config import get_logger

logger = get_logger(__name__)

# 文本过长时截断：招标公告常见 2000~8000 字，但偶有整本招标文件贴进来。
# 12000 字足够覆盖公告类文本，再长既拖慢又超出多数模型的可靠抽取范围。
MAX_TEXT_CHARS = 12_000

# 输入下限：太短的文本不可能包含公告要素，提前拒绝比让模型瞎猜好
MIN_TEXT_CHARS = 10


class ExtractError(RuntimeError):
    """提取失败（输入不合法 / 模型不可用 / 输出无法解析）。"""


def _strip_code_fence(text: str) -> str:
    """剥掉 Markdown 代码块围栏（与 price_compare 同一处理，但独立实现以免互相耦合）。"""
    t = (text or "").strip()
    if "```" not in t:
        return t
    parts = t.split("```")
    if len(parts) >= 3:
        body = parts[1]
        if body.lstrip().lower().startswith("json"):
            body = body.lstrip()[4:]
        return body.strip()
    return t


def parse_json_object(text: str) -> dict[str, Any]:
    """从模型输出中解析 JSON 对象，多级兜底。

    模型有时会在 JSON 前后夹一句「好的，以下是提取结果：」，
    或把 JSON 包在 ```json 里，或用英文键名。多级兜底能覆盖这些实战情况，
    而不必把 prompt 调到完美。

    **关键顺序**：先判整体是否为合法 JSON，再做"截取大括号"兜底。
    反过来的话，`[{"标题": "x"}]` 这种顶层是数组的响应会被从中间截出对象、
    当成合法结果接受——那掩盖了模型没按契约返回对象的问题。
    """
    body = _strip_code_fence(text)

    # ① 先试整体解析：能解析就说明响应本身是完整 JSON
    try:
        whole = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        whole = None
    if whole is not None:
        if isinstance(whole, dict):
            return whole
        # 是合法 JSON 但不是对象（数组 / 数字 / null）→ 明确拒绝，
        # 不再尝试截取：这说明模型根本没按对象契约返回
        raise ExtractError(
            f"模型返回的 JSON 顶层是 {type(whole).__name__}，本接口要求对象"
        )

    # ② 整体不是合法 JSON（前后有说明文字）→ 截取首尾大括号
    start, end = body.find("{"), body.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(body[start : end + 1])
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            pass

    # ③ 逐行尝试（模型把多个 JSON 分行输出时）
    for line in body.splitlines():
        line = line.strip().rstrip(",")
        if line.startswith("{") and line.endswith("}"):
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    return obj
            except (json.JSONDecodeError, ValueError):
                continue

    raise ExtractError("模型输出无法解析为 JSON（可能未按格式返回）")


def _sanitize(record: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """剔除明显不合法的值，返回 `(清洗后记录, 警告列表)`。

    为什么要在提示词之外再做一层："不要编造"是软约束，模型仍可能填
    `中标金额: 0`、`发布时间: 1900-01-01` 这类占位值。与其让它们进库污染统计，
    不如清空并告诉用户"这几项模型给了可疑值，请人工确认"。
    """
    warnings: list[str] = []

    for key in ("amount", "budget"):
        value = record.get(key)
        if isinstance(value, (int, float)) and value <= 0:
            record[key] = None
            warnings.append(f"{key} 为非正数，已清空")

    for key in ("publish_time", "bid_time"):
        value = str(record.get(key) or "")
        m = re.match(r"^(\d{4})", value)
        if m and (int(m.group(1)) < 2000 or int(m.group(1)) > 2100):
            record[key] = ""
            warnings.append(f"{key} 年份异常（{m.group(1)}），已清空")

    # 标题与项目名称完全相同是正常的；但项目名称等于标题且含"公告"后缀，
    # 说明模型没执行"去后缀"规则，这里补一刀
    name = str(record.get("project_name") or "")
    if name and re.search(r"(中标结果|招标|采购|成交)公告$", name):
        record["project_name"] = re.sub(
            r"(中标结果|招标|采购|成交)公告$", "", name
        ).strip()
        warnings.append("项目名称已自动去除公告后缀")

    return record, warnings


# ---------------------------------------------------------------------------
# 文本抽取（走 LLM）
# ---------------------------------------------------------------------------

def _get_llm_client():
    """取当前配置的 LLM 客户端（复用工厂，跟随 .env 的 LLM_PROVIDER）。"""
    from src.clients.llm_factory import get_llm_client
    from src.config import settings

    provider = (settings.llm_provider or "deepseek").strip()
    client = get_llm_client(provider)
    if client is None or not getattr(client, "available", lambda: False)():
        raise ExtractError(f"LLM 客户端不可用（provider={provider}），请检查 .env 配置")
    return client, provider


def extract_from_text(text: str, *, timeout_hint: int = 60) -> dict[str, Any]:
    """从公告文本抽取 17 个字段。

    `timeout_hint` 仅用于日志提示（真正的超时由客户端决定）：
    本地 Ollama 生成 300 token 约需 90 秒，云端 DeepSeek 约 5 秒，
    这个差距对使用者是重要信息，故写进返回体的 `meta` 里。
    """
    content = (text or "").strip()
    if len(content) < MIN_TEXT_CHARS:
        raise ExtractError(f"文本过短（至少 {MIN_TEXT_CHARS} 个字符）")
    truncated = False
    if len(content) > MAX_TEXT_CHARS:
        content = content[:MAX_TEXT_CHARS]
        truncated = True

    client, provider = _get_llm_client()
    messages = [
        {"role": "system", "content": extraction_prompt()},
        {"role": "user", "content": f"公告内容：\n\n{content}"},
    ]

    try:
        raw = client.chat_raw(messages)
    except Exception as exc:
        logger.warning("标讯抽取调用失败：%s", exc)
        raise ExtractError(f"模型调用失败：{type(exc).__name__}") from exc

    if not (raw or "").strip():
        raise ExtractError("模型返回空内容（若使用本地小模型，可能是上下文或 token 限制）")

    parsed = parse_json_object(raw)
    record = normalize_record(parsed)
    record, warnings = _sanitize(record)

    return {
        "record": record,
        "meta": {
            "source": "text",
            "provider": provider,
            "model": getattr(client, "model", ""),
            "input_chars": len(content),
            "truncated": truncated,
            "completeness": completeness(record),
            "warnings": warnings,
        },
    }


# ---------------------------------------------------------------------------
# 图片抽取（走视觉模型）
# ---------------------------------------------------------------------------

def extract_from_image(image_base64: str, hint: str = "") -> dict[str, Any]:
    """从公告截图抽取 17 个字段（复用视觉模型，schema 与文本路径相同）。"""
    import asyncio

    from src.clients.vision import VisionClient

    client = VisionClient()
    if not client.available:
        raise ExtractError("视觉模型未配置（检查 ZHIPU_API_KEY / ZHIPU_VISION_MODEL）")

    prompt = extraction_prompt() + "\n\n请从这张招投标公告图片中提取上述字段。"
    if hint.strip():
        prompt += f"\n\n用户补充说明：{hint.strip()}"

    try:
        raw = asyncio.run(client.analyze(image_base64, prompt))
    except RuntimeError:
        # 已在事件循环中 → 交给独立线程
        raw = _run_in_thread(client.analyze(image_base64, prompt))
    except Exception as exc:
        logger.warning("图片抽取失败：%s", exc)
        raise ExtractError(f"图片解析失败：{type(exc).__name__}") from exc

    parsed = parse_json_object(raw)
    record = normalize_record(parsed)
    record, warnings = _sanitize(record)

    return {
        "record": record,
        "meta": {
            "source": "image",
            "provider": "zhipu",
            "model": client.model_name,
            "input_chars": 0,
            "truncated": False,
            "completeness": completeness(record),
            "warnings": warnings,
        },
    }


def _run_in_thread(coro_factory) -> str:
    box: dict[str, Any] = {}

    def runner() -> None:
        import asyncio

        try:
            box["value"] = asyncio.run(coro_factory())
        except Exception as exc:  # pragma: no cover
            box["error"] = exc

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    t.join()
    if "error" in box:
        raise box["error"]
    return box.get("value", "")


# ---------------------------------------------------------------------------
# 入库
# ---------------------------------------------------------------------------

def save_record(record: dict[str, Any]) -> dict[str, Any]:
    """把抽取结果写入 `bidding_records`。

    只写入表里**真实存在**的列——用 information_schema 查一次列名再过滤，
    这样即使 schema 与表结构有出入也不会因 "column does not exist" 整条失败。

    返回 `{"saved": bool, "id": int|None, "skipped_columns": [...]}`。
    """
    import os

    import psycopg
    from dotenv import load_dotenv

    from src.config import PROJECT_ROOT

    load_dotenv(PROJECT_ROOT / ".env")

    host = (os.getenv("POSTGRES_HOST") or "").strip()
    if not host:
        return {"saved": False, "id": None, "reason": "未配置 PostgreSQL（POSTGRES_HOST 为空）"}

    non_empty = {k: v for k, v in record.items() if v not in (None, "")}
    if not non_empty:
        return {"saved": False, "id": None, "reason": "没有可写入的字段"}

    try:
        conn = psycopg.connect(
            host=host,
            port=(os.getenv("POSTGRES_PORT") or "5432").strip(),
            user=(os.getenv("POSTGRES_USER") or "postgres").strip(),
            password=(os.getenv("POSTGRES_PASSWORD") or "").rstrip("\r\n"),
            dbname=(os.getenv("POSTGRES_DB") or "chatbot").strip(),
        )
    except Exception as exc:
        logger.warning("入库连接失败：%s", exc)
        return {"saved": False, "id": None, "reason": f"数据库连接失败：{type(exc).__name__}"}

    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = 'bidding_records'"
            )
            table_cols = {row[0] for row in cur.fetchall()}
            usable = {k: v for k, v in non_empty.items() if k in table_cols}
            skipped = sorted(set(non_empty) - set(usable))
            if not usable:
                return {
                    "saved": False,
                    "id": None,
                    "reason": "抽取字段与表列无交集",
                    "skipped_columns": skipped,
                }

            cols = ", ".join(usable.keys())
            placeholders = ", ".join(["%s"] * len(usable))
            cur.execute(
                f"INSERT INTO bidding_records ({cols}) VALUES ({placeholders}) RETURNING id",
                list(usable.values()),
            )
            row = cur.fetchone()
        conn.commit()
        return {
            "saved": True,
            "id": row[0] if row else None,
            "skipped_columns": skipped,
        }
    except Exception as exc:
        conn.rollback()
        logger.warning("入库失败：%s", exc)
        return {"saved": False, "id": None, "reason": f"写入失败：{type(exc).__name__}"}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 对外入口
# ---------------------------------------------------------------------------

def extract(source: str, text: str = "", image_base64: str = "", hint: str = "") -> dict[str, Any]:
    """统一入口：`source` 取 `text` 或 `image`。"""
    if source == "text":
        if not text.strip():
            raise ExtractError("text 模式下必须提供 text")
        return extract_from_text(text)
    if source == "image":
        if not image_base64.strip():
            raise ExtractError("image 模式下必须提供 image_base64")
        return extract_from_image(image_base64, hint)
    raise ExtractError(f"不支持的 source：{source!r}（应为 text 或 image）")


__all__ = [
    "ExtractError",
    "extract",
    "extract_from_image",
    "extract_from_text",
    "parse_json_object",
    "save_record",
]

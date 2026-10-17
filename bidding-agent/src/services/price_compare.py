"""图片商品比价服务：图片识别 → 商品信息提取 → 多平台价格检索 → 择优建议。

工作流
------
    用户上传商品图片（铭牌 / 包装 / 报价单）
        ↓ ① 视觉模型提取结构化商品信息（JSON）
        ↓ ② 按平台并行检索公开报价（Tavily include_domains 限定站点）
        ↓ ③ 从搜索结果摘要中抽取价格 + 判定可信度
        ↓ ④ 生成比价结论与风险提示

**必须如实告知的能力边界**（前端会原样展示这些提示）
----------------------------------------------------
这套实现基于**公开搜索引擎的摘要**，不是电商官方价格 API，因此：

1. **并非所有平台都能取到价**。实测京东、天猫的摘要通常是
   "登录查看价格"，抓不到；淘宝、苏宁、1688、西域等能取到。
2. **搜到的价格可能不是同一个规格**。同一型号名下有仿品与不同规格，
   实测出现过 ¥15 与 ¥999 并存的情况。故本模块用**中位数**做基准，
   把偏离过大的报价标为「疑似非同一商品或仿品」而不是当成"最低价"推荐。
3. **价格有时效性**。搜索结果有索引延迟，最终价以商品页为准。

因此返回结构里带 `caveats` 字段，前端必须展示——**宁可说清楚局限，
也不要给用户一个看起来很确定但可能错的"最优平台"**。
"""

from __future__ import annotations

import json
import re
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import httpx

from src.http_client import get_http_client
from src.logging_config import get_logger

logger = get_logger(__name__)

TAVILY_URL = "https://api.tavily.com/search"

# 检索超时：比价是同步接口，用户在前端等结果，故比工具层的 25s 再收紧一些，
# 配合并行检索（5 个平台同时发），整体控制在 ~30s 内。
SEARCH_TIMEOUT = 25.0
CONNECT_TIMEOUT = 8.0

# 并行度上限：平台数与线程数取小，避免为几个平台开一堆线程
MAX_WORKERS = 6

# 单平台拉取的条数：够抽价格即可，太多会拖慢且引入噪声
RESULTS_PER_PLATFORM = 5

# --- 报价可信度阈值（相对全平台中位数的倍数）---
# 两档设计，依据是"可比性"而非"贵贱"：
#   extreme    : 超出即判定为**不同商品**，从比价中排除
#   suspicious : 参与比价，但结论里提醒核实规格与真伪
# 实测依据：同一型号在电商与工业品渠道可能差 5 倍以上（¥48 vs ¥332），
# 若不排除，"较次低价低 85%"这种结论毫无意义。
EXTREME_RATIO_HIGH = 2.5
EXTREME_RATIO_LOW = 0.4
SUSPICIOUS_RATIO_HIGH = 1.5
SUSPICIOUS_RATIO_LOW = 0.6

# 样本少于这个数时**不做离群排除**：2 个价格的中位数就是两者平均，毫无代表性。
# 实测教训：真品 ¥48.61 与乱码聚合页 ¥332 两条，中位数 ¥190 把真品判成
# "不同商品"排除掉，结论完全反了。样本不足时应把判断权交回用户。
MIN_SAMPLES_FOR_OUTLIER = 3


@dataclass(frozen=True)
class Platform:
    """一个比价平台。`domains` 传给 Tavily 的 include_domains 限定站点。"""

    key: str
    name: str
    domains: tuple[str, ...]
    kind: str = "综合电商"  # 综合电商 / 工业品 / 比价参考


# 平台清单：按实测可抓取性排序（能取到价的放前面，前端展示顺序即此顺序）
PLATFORMS: tuple[Platform, ...] = (
    Platform("taobao", "淘宝", ("taobao.com",)),
    Platform("suning", "苏宁易购", ("suning.com",)),
    Platform("jd", "京东", ("jd.com",)),
    Platform("tmall", "天猫", ("tmall.com",)),
    Platform("alibaba", "阿里巴巴 1688", ("1688.com",), "工业品"),
    Platform("ehsy", "西域", ("ehsy.com",), "工业品"),
    Platform("zkh", "震坤行", ("zkh.com",), "工业品"),
)

# 价格正则：只认带币种符号的数字，避免把型号（iC65N）、电流（20A）、
# 分断能力（6kA）当成价格。要求 2~7 位整数，可选两位小数。
_PRICE_RE = re.compile(r"[¥￥]\s*(\d{2,7}(?:\.\d{1,2})?)")

# 需要登录 / 无公开报价的标记：命中则跳过该条，不把"¥1??"这类残缺值当成价格
_NO_PRICE_MARKERS = (
    "登录查看",
    "登录后查看",
    "请登录",
    "会员可见",
    "价格面议",
    "询价",
    "暂无报价",
)


class PriceCompareError(RuntimeError):
    """比价流程失败（视觉解析失败 / 搜索服务不可用 / 无任何可用结果）。"""


@dataclass
class Quote:
    """一条平台报价。"""

    platform: str
    platform_key: str
    kind: str
    price: float | None
    title: str
    url: str
    snippet: str
    has_price: bool
    trust: str = "normal"  # normal / suspicious_high / suspicious_low
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "platform": self.platform,
            "platform_key": self.platform_key,
            "kind": self.kind,
            "price": self.price,
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "has_price": self.has_price,
            "trust": self.trust,
            "note": self.note,
        }


@dataclass
class PlatformResult:
    """单个平台的检索结果（可能有多条报价，也可能一条都没有）。"""

    platform: str
    platform_key: str
    kind: str
    ok: bool
    quotes: list[Quote] = field(default_factory=list)
    error: str = ""
    # 因"型号/规格不符"被排除的条目。回传给前端是为了让用户看到
    # "为什么没有更便宜的选项"——否则会怀疑系统漏搜。
    rejected: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "platform": self.platform,
            "platform_key": self.platform_key,
            "kind": self.kind,
            "ok": self.ok,
            "error": self.error,
            "quotes": [q.to_dict() for q in self.quotes],
            "rejected": list(self.rejected),
            "rejected_count": len(self.rejected),
        }


# ---------------------------------------------------------------------------
# ① 商品信息提取（视觉模型）
# ---------------------------------------------------------------------------

PRODUCT_SCHEMA_PROMPT = """请提取图片中的商品信息，严格只输出一个 JSON 对象。
不要输出任何解释文字，不要用 Markdown 代码块包裹。

字段定义：
- product_name: 商品名称（简洁，如「小型断路器」）
- brand: 品牌（如「施耐德」）
- model: 型号（如「iC65N 2P C20A」）
- spec: 关键规格（如「20A 2P 400V 6kA」）
- category: 品类（如「低压电器/断路器」）
- sku: 物料编码/货号（图中没有则空字符串）
- raw_text: 图片中出现的全部文字，按原顺序用换行连接
- confidence: 提取准确度自评，0-1 之间的小数

识别不到的字段填空字符串，不要编造。"""

_PRODUCT_FIELDS = (
    "product_name",
    "brand",
    "model",
    "spec",
    "category",
    "sku",
    "raw_text",
)


def _strip_code_fence(text: str) -> str:
    """剥掉模型可能加上的 Markdown 代码块围栏。

    提示词已经明确要求"不要代码块"，但模型仍偶发添加。这是**已知不稳定点**，
    故解析侧必须容错，而不是假定模型永远听话。
    """
    t = (text or "").strip()
    if "```" not in t:
        return t
    parts = t.split("```")
    # 取第一个围栏内的内容（形如 ```json {...} ```）
    if len(parts) >= 3:
        body = parts[1]
        if body.lstrip().lower().startswith("json"):
            body = body.lstrip()[4:]
        return body.strip()
    return t


def _extract_json_object(text: str) -> dict[str, Any]:
    """从模型输出中解析出 JSON 对象；失败时尝试截取首尾大括号兜底。"""
    body = _strip_code_fence(text)
    try:
        obj = json.loads(body)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass

    # 兜底：模型可能在 JSON 前后夹了说明文字，截取最外层大括号再试一次
    start = body.find("{")
    end = body.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(body[start : end + 1])
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            pass
    raise PriceCompareError("无法从图片中解析出结构化商品信息")


def extract_product_from_image(image_base64: str, hint: str = "") -> dict[str, Any]:
    """用视觉模型从图片提取商品信息。

    `hint` 用于用户补充说明（如"这是包装盒背面"），会追加到提示词末尾。
    """
    import asyncio

    from src.clients.vision import VisionClient

    client = VisionClient()
    if not client.available:
        raise PriceCompareError("视觉模型未配置（检查 ZHIPU_API_KEY / ZHIPU_VISION_MODEL）")

    prompt = PRODUCT_SCHEMA_PROMPT
    if hint and hint.strip():
        prompt += f"\n\n用户补充说明：{hint.strip()}"

    try:
        raw = asyncio.run(client.analyze(image_base64, prompt))
    except RuntimeError:
        # 已在事件循环中（FastAPI 的 async 端点里调本函数）→ 交给新线程跑
        raw = _run_coro_in_thread(client.analyze(image_base64, prompt))
    except Exception as exc:
        logger.warning("视觉解析失败：%s", exc)
        raise PriceCompareError(f"图片解析失败：{type(exc).__name__}") from exc

    obj = _extract_json_object(raw)
    product: dict[str, Any] = {}
    for key in _PRODUCT_FIELDS:
        value = obj.get(key)
        product[key] = str(value).strip() if value is not None else ""
    try:
        product["confidence"] = max(0.0, min(1.0, float(obj.get("confidence") or 0.0)))
    except (TypeError, ValueError):
        product["confidence"] = 0.0
    return product


def _run_coro_in_thread(coro_factory) -> str:
    """在独立线程里跑协程（用于从同步上下文调用异步方法）。"""
    box: dict[str, Any] = {}

    def runner() -> None:
        import asyncio

        try:
            box["value"] = asyncio.run(coro_factory())
        except Exception as exc:  # pragma: no cover - 由调用方处理
            box["error"] = exc

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    t.join()
    if "error" in box:
        raise box["error"]
    return box.get("value", "")


# ---------------------------------------------------------------------------
# ② 平台检索
# ---------------------------------------------------------------------------


def build_price_query(product: dict[str, Any]) -> str:
    """把商品信息拼成检索词。

    为什么要拼而不是只搜商品名：电商搜索对**型号**最敏感，
    只给"小型断路器"会返回一堆不相关商品；带上品牌与型号才能落到具体 SKU。
    """
    parts: list[str] = []
    brand = (product.get("brand") or "").strip()
    model = (product.get("model") or "").strip()
    name = (product.get("product_name") or "").strip()
    if brand:
        parts.append(brand)
    if model:
        parts.append(model)
    if name and name not in " ".join(parts):
        parts.append(name)
    query = " ".join(parts).strip()
    return f"{query} 价格" if query else "商品价格"


def _parse_price(text: str) -> float | None:
    """从文本中抽取价格；不可靠时返回 None。

    只认带 ¥/￥ 的数字：型号里的字母数字混合（iC65N）、电流（20A）、
    分断能力（6kA）都不带币种符号，天然被排除。

    **为什么"多个不同价格"要判为不可靠**：实测 1688 的聚合页/乱码页里
    会同时出现多个价格，取第一个会得到随机值（曾抽出 ¥332 而真品在 ¥48）。
    这类页面的价格没有单品对应关系，不该参与比价。
    """
    if not text:
        return None
    for marker in _NO_PRICE_MARKERS:
        if marker in text:
            return None
    matches = _PRICE_RE.findall(text)
    if not matches:
        return None
    values = []
    for raw in matches:
        try:
            value = float(raw)
        except ValueError:
            continue
        if value >= 1.0:
            values.append(value)
    if not values:
        return None

    # 多个差异较大的价格 → 无法判断哪个是本品价格
    if len(values) > 1 and max(values) / min(values) >= 2.0:
        return None
    return min(values)


# ---------------------------------------------------------------------------
# 相关性过滤：只保留"确实是同一个商品"的条目
# ---------------------------------------------------------------------------
#
# 为什么必须有这一层（实测教训）
# ------------------------------
# 不加过滤时，搜「施耐德 iC65N 2P C20A 断路器」会返回：
#     ¥15.66   配件/仿品
#     ¥38.00   A9系列不同型号
#     ¥332.00  EA9R 漏电保护（不同产品线）
#     ¥9999.00 「施耐德报价批发」聚合页（不是单品）
# 直接取最低价会把 ¥15.66 当成"最优平台"，是**误导性结论**。
#
# 过滤策略（两级，宽松优先，避免误杀）：
#   ① 型号必须出现（去掉空格与大小写差异）；型号缺失时才退回品牌+名称双匹配
#   ② 规格数字不得冲突：查询里有 20A 而结果里只有 32A/16A → 判为不同规格
#
# 关键取舍：**宁可少给几条可靠报价，也不给一条错误的最优推荐**。
# 过滤后若某平台无结果，前端会显示"未找到同型号报价"，这是诚实且有信息量的。

_NUM_UNIT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([a-zA-Z]{1,3})")


def _normalize(text: str) -> str:
    """归一化用于比对：去空格、转小写、统一全角括号。"""
    return re.sub(r"\s+", "", (text or "").lower()).replace("（", "(").replace("）", ")")


def _numeric_tokens(text: str) -> set[tuple[str, str]]:
    """抽出「数字+单位」组合，如 20A / 6kA / 400V，用于规格冲突检测。"""
    return {(m.group(1), m.group(2).lower()) for m in _NUM_UNIT_RE.finditer(text or "")}


def _relevance_ok(product: dict[str, Any], quote_text: str) -> tuple[bool, str]:
    """判断一条搜索结果是否与目标商品是同一型号。

    返回 `(是否相关, 原因说明)`；原因用于前端展示"为什么这条被排除"，
    避免用户以为系统漏掉了更便宜的选项。
    """
    text = _normalize(quote_text)
    if not text:
        return False, "无有效内容"

    model = _normalize(product.get("model") or "")
    brand = _normalize(product.get("brand") or "")
    name = _normalize(product.get("product_name") or "")

    # ① 型号匹配：型号是最强的身份标识
    if model:
        # 型号里可能带空格与连字符（iC65N 2P C20A），归一化后再找
        if model not in text:
            # 宽松匹配：型号的字母数字部分（iC65NC20A 这种被电商压缩过的写法）
            compact = re.sub(r"[^0-9a-z]", "", model)
            if not compact or compact not in re.sub(r"[^0-9a-z]", "", text):
                return False, "型号不符"
    elif brand and name:
        if brand not in text or name not in text:
            return False, "品牌或名称不符"
    else:
        return False, "商品信息不足，无法判定相关性"

    # ② 规格冲突：查询里的"数字+单位"不应被同单位的不同数字取代
    want = _numeric_tokens(product.get("spec") or "") | _numeric_tokens(product.get("model") or "")
    if want:
        got = _numeric_tokens(quote_text)
        for number, unit in want:
            same_unit = {n for n, u in got if u == unit}
            # 结果里出现了同单位但不同的数字 → 规格不同（如查 20A 得到 32A）
            if same_unit and number not in same_unit:
                return False, f"规格不符（需 {number}{unit}，结果为 {'/'.join(sorted(same_unit))}{unit}）"

    return True, ""


def _search_platform(platform: Platform, query: str, product: dict[str, Any]) -> PlatformResult:
    """检索单个平台。任何失败都收敛为 `ok=False`，不向上抛——单站失败不该毁掉整次比价。"""
    import os

    api_key = (os.getenv("TAVILY_API_KEY") or "").strip()
    if not api_key:
        return PlatformResult(platform.name, platform.key, platform.kind, False, error="未配置 TAVILY_API_KEY")

    payload = {
        "query": query,
        "search_depth": "advanced",
        "max_results": RESULTS_PER_PLATFORM,
        "include_domains": list(platform.domains),
    }
    try:
        resp = get_http_client().post(
            TAVILY_URL,
            json=payload,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=httpx.Timeout(SEARCH_TIMEOUT, connect=CONNECT_TIMEOUT),
        )
    except Exception as exc:
        logger.warning("平台 %s 检索失败：%s", platform.name, type(exc).__name__)
        return PlatformResult(platform.name, platform.key, platform.kind, False, error=f"检索失败：{type(exc).__name__}")

    if resp.status_code != 200:
        return PlatformResult(platform.name, platform.key, platform.kind, False, error=f"搜索服务返回 {resp.status_code}")

    try:
        items = resp.json().get("results") or []
    except (json.JSONDecodeError, ValueError):
        return PlatformResult(platform.name, platform.key, platform.kind, False, error="搜索响应无法解析")

    quotes: list[Quote] = []
    rejected: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        url = str(item.get("url") or "").strip()
        content = str(item.get("content") or "").strip()

        # 相关性过滤：标题与摘要任一命中型号即算相关
        # （电商常把型号放在标题、规格放在摘要，只查一处会漏）
        relevant, why = _relevance_ok(product, f"{title} {content}")
        if not relevant:
            rejected.append({"title": (title or url)[:120], "url": url, "reason": why})
            continue

        price = _parse_price(f"{title} {content}")
        quotes.append(
            Quote(
                platform=platform.name,
                platform_key=platform.key,
                kind=platform.kind,
                price=price,
                title=title[:120] or url,
                url=url,
                snippet=content[:300],
                has_price=price is not None,
            )
        )

    result = PlatformResult(platform.name, platform.key, platform.kind, True, quotes=quotes)
    result.rejected = rejected
    return result


def search_all_platforms(
    query: str,
    product: dict[str, Any],
    platforms: tuple[Platform, ...] = PLATFORMS,
) -> list[PlatformResult]:
    """并行检索全部平台。并行是必须的：串行 7 个平台 × 数秒 = 用户等半分钟以上。"""
    workers = min(MAX_WORKERS, max(1, len(platforms)))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="price-search") as pool:
        return list(pool.map(lambda p: _search_platform(p, query, product), platforms))


# ---------------------------------------------------------------------------
# ③ 报价可信度判定
# ---------------------------------------------------------------------------


def _pick_best_quote(result: PlatformResult) -> Quote | None:
    """单平台内挑一条作为该平台报价：优先有价的，多条取中位数量级的那条。

    取中位数而非最低值：一个平台内常混有配件、仿品、不同规格，
    取最低会把"¥15 的仿品"当成该平台报价。
    """
    priced = [q for q in result.quotes if q.has_price and q.price is not None]
    if not priced:
        # 没有明确报价时，仍返回第一条作为"可点击查看"的入口
        return result.quotes[0] if result.quotes else None
    if len(priced) == 1:
        return priced[0]
    prices = [q.price for q in priced if q.price is not None]
    median = statistics.median(prices)
    return min(priced, key=lambda q: abs((q.price or 0) - median))


def annotate_trust(best_quotes: list[Quote]) -> None:
    """按全平台中位数给各报价打可信度标签（就地修改）。

    **样本不足时不做排除**（`MIN_SAMPLES_FOR_OUTLIER`）：
    只有 2 个价格时中位数就是两者的平均值，毫无代表性——实测出现过
    「真品 ¥48.61」与「乱码聚合页 ¥332」两条，中位数 ¥190 把真品判成了
    "不同商品"并排除，结论完全反了。故样本 < 3 时一律只标 suspicious，
    让用户带着完整信息自己判断，而不是由系统做一个不可靠的裁决。

    阈值设计（相对中位数倍数），依据是**"可比性"而非"贵贱"**：

        >= 2.5 或 <= 0.4   → extreme   ：判定为不同商品，从比价中排除
        >= 1.5 或 <= 0.6   → suspicious：参与比价，但提醒核实规格与真伪
    """
    prices = [q.price for q in best_quotes if q.price is not None]
    if len(prices) < 2:
        return
    median = statistics.median(prices)
    if median <= 0:
        return

    small_sample = len(prices) < MIN_SAMPLES_FOR_OUTLIER

    for q in best_quotes:
        if q.price is None:
            continue
        ratio = q.price / median

        if small_sample:
            # 样本太少：不做排除，只在偏离明显时提示
            if ratio >= SUSPICIOUS_RATIO_HIGH:
                q.trust = "suspicious_high"
                q.note = (
                    f"高于样本中位价（¥{median:,.2f}）{int((ratio - 1) * 100)}%，"
                    "因样本过少无法判定是否为同一商品，请人工核实"
                )
            elif ratio <= SUSPICIOUS_RATIO_LOW:
                q.trust = "suspicious_low"
                q.note = (
                    f"低于样本中位价（¥{median:,.2f}）{int((1 - ratio) * 100)}%，"
                    "因样本过少无法判定是否为同一商品，请人工核实"
                )
            # 其余保持 normal：比值接近 1，没有可提示的信息
            continue

        if ratio >= EXTREME_RATIO_HIGH:
            q.trust = "extreme"
            q.note = (
                f"是同类中位价（¥{median:,.2f}）的 {ratio:.1f} 倍，"
                "极可能不是同一商品（不同产品线 / 含服务 / 聚合页），已从比价中排除"
            )
        elif ratio <= EXTREME_RATIO_LOW:
            q.trust = "extreme"
            q.note = (
                f"仅为同类中位价（¥{median:,.2f}）的 {int(ratio * 100)}%，"
                "极可能是仿品、二手或配件，已从比价中排除"
            )
        elif ratio >= SUSPICIOUS_RATIO_HIGH:
            q.trust = "suspicious_high"
            q.note = f"高于同类中位价（¥{median:,.2f}）{int((ratio - 1) * 100)}%，可能含服务费或非同一规格"
        elif ratio <= SUSPICIOUS_RATIO_LOW:
            q.trust = "suspicious_low"
            q.note = f"仅为同类中位价（¥{median:,.2f}）的 {int(ratio * 100)}%，需核实是否仿品或不同规格"


# ---------------------------------------------------------------------------
# ④ 结论生成
# ---------------------------------------------------------------------------


def build_recommendation(product: dict[str, Any], results: list[PlatformResult]) -> dict[str, Any]:
    """按"可信且最低价"给出推荐，并汇总风险提示。

    推荐池的优先级：
        1. `extreme` 报价**永远排除**（不是同一商品，参与比较会得出荒谬结论）
        2. 优先在 `normal` 里选最低价
        3. `normal` 为空时退回 `suspicious`，但结论里必须说明"需核实"
    """
    best_quotes: list[Quote] = []
    for r in results:
        q = _pick_best_quote(r)
        if q is not None:
            best_quotes.append(q)
    annotate_trust(best_quotes)

    priced = [q for q in best_quotes if q.price is not None]
    extreme = [q for q in priced if q.trust == "extreme"]
    comparable = [q for q in priced if q.trust != "extreme"]  # 参与比价的池子
    trusted = [q for q in comparable if q.trust == "normal"]
    pool = trusted or comparable
    winner = min(pool, key=lambda q: q.price or float("inf")) if pool else None

    caveats: list[str] = []
    no_price = [r.platform for r in results if r.ok and not any(q.has_price for q in r.quotes)]
    failed = [r.platform for r in results if not r.ok]
    if no_price:
        caveats.append(
            "以下平台未取到同型号公开报价（多为需登录查看价格，或只有不同规格）："
            + "、".join(no_price)
        )
    if failed:
        caveats.append("以下平台检索失败：" + "、".join(failed))

    rejected_total = sum(len(r.rejected) for r in results)
    if rejected_total > 0:
        caveats.append(
            f"另有 {rejected_total} 条搜索结果因型号或规格不符已排除"
            "（例如用了不同电流规格或不同产品线），未参与比价"
        )

    if extreme:
        names = "、".join(f"{q.platform}（¥{q.price:,.2f}）" for q in extreme)
        caveats.append(
            f"以下报价与同类中位价差异过大，判断为不同商品，已从比价中排除：{names}"
        )

    suspicious = [q for q in comparable if q.trust != "normal"]
    if suspicious:
        names = "、".join(f"{q.platform}（¥{q.price:,.2f}）" for q in suspicious)
        caveats.append(f"以下报价需人工核实规格与真伪：{names}")

    if not trusted and pool:
        caveats.append("没有完全可信的报价，以下结论基于需核实的报价，请谨慎参考")
    if len(comparable) < 2:
        caveats.append("可比报价少于 2 条，结论参考价值有限，建议人工复核")
    caveats.append("价格来自公开搜索摘要，可能滞后于商品页实时价，下单前请以平台实际价格为准")

    # ── 结论 ──────────────────────────────────────────────────────
    if winner is None:
        summary = (
            "未能在各平台取到同型号的可比报价。建议用商品关键词手动到平台核实，"
            "或换一张更清晰的商品图（型号越清楚，匹配越准）。"
        )
    else:
        # "次低价"只在同一个可比池里找，避免拿不同商品当参照
        others = sorted(q.price for q in pool if q.price is not None and q is not winner)
        gap = ""
        if others and others[0] > 0:
            gap = f"，较次低价（¥{others[0]:,.2f}）低 {((others[0] - (winner.price or 0)) / others[0] * 100):.1f}%"
        summary = (
            f"在取到同型号报价的 {len(comparable)} 个平台中，{winner.platform} 报价 "
            f"¥{winner.price:,.2f} 最低{gap}，建议优先询价。"
        )
        if winner.trust != "normal":
            summary += "（该报价需先核实规格与真伪）"

        # 工业品渠道建议：只在池内（已排除极端离群）找，且必须有真实价格
        industrial = [q for q in pool if q.kind == "工业品" and q.trust == "normal"]
        if industrial:
            cheapest_ind = min(industrial, key=lambda q: q.price or float("inf"))
            summary += (
                f" 若为工程/企业采购，工业品渠道 {cheapest_ind.platform} "
                f"（¥{cheapest_ind.price:,.2f}）在开票与账期上通常更适配。"
            )

    return {
        "winner": winner.to_dict() if winner else None,
        "priced_count": len(comparable),
        "excluded_count": len(extreme),
        "summary": summary,
        "caveats": caveats,
    }


# ---------------------------------------------------------------------------
# 对外入口
# ---------------------------------------------------------------------------


def compare_by_image(image_base64: str, hint: str = "", include_raw_text: bool = True) -> dict[str, Any]:
    """完整比价流程。返回可直接 JSON 序列化的结果。

    返回结构：
        {
          "product": {...},                  # 从图片提取的商品信息
          "query": "...",                    # 实际用于检索的关键词
          "platforms": [PlatformResult...],  # 各平台结果（含多条原始报价）
          "recommendation": {...},           # 结论 + 风险提示
        }

    失败抛 PriceCompareError（路由层转 400/503）。
    """
    product = extract_product_from_image(image_base64, hint)
    if not any(product.get(k) for k in ("product_name", "model", "brand")):
        raise PriceCompareError(
            "未能从图片中识别出商品名称、品牌或型号，请换一张更清晰的商品图片"
        )

    query = build_price_query(product)
    results = search_all_platforms(query, product)
    recommendation = build_recommendation(product, results)

    if not include_raw_text:
        # raw_text 可能很长（整张图的 OCR 文本），无需回传前端时省掉
        product = {k: v for k, v in product.items() if k != "raw_text"}

    return {
        "product": product,
        "query": query,
        "platforms": [r.to_dict() for r in results],
        "recommendation": recommendation,
    }


__all__ = [
    "PLATFORMS",
    "Platform",
    "PriceCompareError",
    "build_price_query",
    "build_recommendation",
    "compare_by_image",
    "extract_product_from_image",
    "search_all_platforms",
]

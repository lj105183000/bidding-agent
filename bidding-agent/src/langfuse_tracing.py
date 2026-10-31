"""Langfuse 集成：把每轮问答的 LLM 调用与工具执行上报到 Langfuse。

★ 设计原则：**完全可选、默认关闭、绝不拖累业务**
=======================================
这是本项目**第一个向外发数据**的模块。此前的可观测性（health / logs /
metrics / 阶段耗时 / LLM 追踪）全部是自持的——不开容器、不连外部服务、
不发任何数据出机器。引入 Langfuse 意味着：

    · 多一个**网络依赖**（Langfuse Cloud 不可达时不能影响问答）
    · **prompt 与回答可能离开本机**（招投标场景含敏感信息）

故这里的每条设计都在约束这个风险：

    ① **未配置 `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` 时完全 no-op**
       —— 不 import SDK、不建客户端、不联网。默认状态就是"没这回事"。
    ② **SDK 未安装时也 no-op**（它是 optional-dependencies，不装不影响运行）
    ③ **所有上报路径包 try/except**：观测失败只记 debug 日志，
       绝不让"上报出错"变成"用户拿不到回答"
    ④ **失败后进入冷却**：与 `web_search_exa` / `FallbackLLMClient` 同一策略——
       上游挂了不该让每个请求都白等一次超时
    ⑤ 用**独立线程**（SDK 自带后台导出）而不是在请求路径上同步发送
    ⑥ **默认不发正文**：只发模型名、token、工具名、字符数等元数据。
       要读完整对话必须显式打开 `LANGFUSE_CAPTURE_CONTENT=true`，
       且打开后正文会**强制脱敏**（见下）

★ 六个开关（全部从环境变量读，改 .env 即可，无需改代码）
=================================================
    LANGFUSE_TRACING_ENABLED   总开关。**默认开**；`false`/`0`/`no`/`off` 关闭。
                               置 false 后本模块一条数据都不发（含反馈评分）。
    LANGFUSE_SAMPLE_RATE       采样率 0~1，默认 1.0（全量）。**按 trace 采样**：
                               一条被抽中的 trace 记全过程，没抽中的一条都不记
                               —— 只采一半 observation 会让 UI 里出现残缺的调用树。
    LANGFUSE_FLUSH_ON_FINISH   每轮问答结束是否主动 flush，默认 false。
                               true 时"必达"但结尾多一次等待（有超时上界）；
                               false 时依赖 SDK 后台导出 + 进程退出时兜底 flush，
                               崩溃（被 kill / 断电）最多丢几秒的 span。
    LANGFUSE_FLUSH_TIMEOUT_S   flush 等待上界（秒），默认 3。**必须要有上界**：
                               网络挂起时不能让收尾线程无限等。
    LANGFUSE_CAPTURE_CONTENT   是否上报 prompt 与回答正文，默认 false。
    LANGFUSE_MAX_CAPTURE_CHARS 单侧正文上限，默认 4000（超出保留首尾）。
    LANGFUSE_PRICE_TABLE       单价表覆盖（JSON），用于算成本；见 `estimate_cost`。

★ 脱敏是强制的，没有开关
=====================
一旦打开正文上报，`redact()` 会在**每条**正文与元数据上执行：手机号、
身份证、统一社会信用代码、银行账号、邮箱、金额、项目/招标编号、企业名称、
联系人姓名一律替换为 `[金额]` 这类占位符。同一份脱敏函数还作为 SDK 的
`mask=` 钩子传入（SDK 会对 input/output/metadata 都调它）——**双保险**，
将来新增的调用点忘了脱敏也挡得住。

不做"关闭脱敏"的开关：能关掉的保护等于没有保护，而 Langfuse 是外部服务。

★ 为什么挂在 `LLMTraceRecorder` 上，而不是改客户端
============================================
上一轮做可观测性时已经建好了扩展点：`LLMTraceRecorder.record()` 是**所有**
LLM 调用的唯一收口（三种调用形态都走它）。故这里只需：

    · 把 recorder 换成 Langfuse 版本（`LangfuseTraceRecorder`）
    · 它在 `record()` 里顺手记一条 Langfuse observation

不改客户端、不改 agent 主流程、不新增第二套采集逻辑。

（唯一的例外是 `LANGFUSE_CAPTURE_CONTENT`：正文只存在于客户端里，
故客户端在**该开关打开时**才多填两个字段，默认路径零开销。）

★ trace 结构
==========
    trace: chat（一次问答 = 一个 trace）
      ├─ observation(generation)  decision   决策轮
      ├─ observation(tool)        search_knowledge_base
      ├─ observation(generation)  decision   第二轮决策
      └─ observation(generation)  stream     最终生成

这样在 Langfuse UI 里看到的是一棵**有层级的调用树**，
而不是一串散记录——排查时能直接看出"哪次调用属于哪一步"。
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import random
import re
import threading
import time
from typing import Any, Callable

from src.monitoring_trace import LLMCall, LLMTraceRecorder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 环境变量
# ---------------------------------------------------------------------------

#: 凭据与地址
#:
#: ★ `LANGFUSE_BASE_URL` 而不是 `LANGFUSE_HOST`（实测踩到）
#: ------------------------------------------------
#: Langfuse 官方 SDK 的历史文档里两种都出现过，而**用户 `.env` 里写的是
#: `LANGFUSE_BASE_URL`**（形如 `https://jp.cloud.langfuse.com`）。
#: 我第一版只读 `LANGFUSE_HOST`，后果是：
#:
#:     host 取不到 → 传 None → SDK 回落到**默认云端**
#:     → 用户以为是自建/日本区，数据实际发到了默认区
#:
#: 这类"配置项名字对不上"的失败**完全静默**——不发错、不报错，
#: 只是数据去了别处。故两个名字都读，且 `BASE_URL` 优先（它更具体，
#: 且是用户已有的写法，也与 SDK 自己的优先级一致）。
ENV_PUBLIC_KEY = "LANGFUSE_PUBLIC_KEY"
ENV_SECRET_KEY = "LANGFUSE_SECRET_KEY"
ENV_HOST = "LANGFUSE_BASE_URL"
#: 兼容别名（官方 SDK 也接受这个写法）
ENV_HOST_ALIASES = ("LANGFUSE_BASE_URL", "LANGFUSE_HOST")

#: 总开关。★ SDK 只认 `"false"` 这一个字符串（`== "false"` 才算关），
#: 用户写 `0` / `no` / `off` 时 SDK 会**继续上报**——故本模块自己解析，
#: 并把结果**显式传给 SDK**（`tracing_enabled=`），两边判定永远一致。
ENV_TRACING_ENABLED = "LANGFUSE_TRACING_ENABLED"

#: 采样率（0~1）。本模块按 trace 采样，并把 `sample_rate=1.0` 显式传给 SDK，
#: 避免 SDK 再独立采一次（两次采样相乘会把实际采样率变成 rate²）。
ENV_SAMPLE_RATE = "LANGFUSE_SAMPLE_RATE"

#: 每轮 flush 开关与等待上界（秒）
ENV_FLUSH_ON_FINISH = "LANGFUSE_FLUSH_ON_FINISH"
ENV_FLUSH_TIMEOUT = "LANGFUSE_FLUSH_TIMEOUT_S"

#: 正文上报开关与单侧长度上限
ENV_CAPTURE_CONTENT = "LANGFUSE_CAPTURE_CONTENT"
ENV_MAX_CAPTURE_CHARS = "LANGFUSE_MAX_CAPTURE_CHARS"

#: 单价表覆盖（JSON 字符串）
ENV_PRICE_TABLE = "LANGFUSE_PRICE_TABLE"

#: 传给 SDK 的 HTTP 超时（秒）。SDK 自己也会读 `LANGFUSE_TIMEOUT`。
ENV_TIMEOUT = "LANGFUSE_TIMEOUT"

#: 视为"关"的取值。覆盖常见写法，避免用户写了 `off` 却被静默忽略。
_FALSEY = {"0", "false", "no", "off", "none", "null", "disabled", "disable", "否", "不", "关", "关闭"}
_TRUTHY = {"1", "true", "yes", "on", "enabled", "enable", "是", "开", "打开"}

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

#: 上报失败后的冷却时长（秒）。
#:
#: 与 `web_search_exa` / `FallbackLLMClient` 同一策略：上游挂了不该让
#: **每个请求**都白等一次连接超时。300 秒足够让一次网络抖动自愈，
#: 又不至于长时间不恢复上报。
FAILURE_COOLDOWN_SECONDS = 300.0

#: 一次问答最多向 Langfuse 推进多少条 observation。
#:
#: 与记录器的 `MAX_RECORDS` 同量级即可；这里存在的意义是**崩溃隔离**：
#: 若上游不断重试，单请求的推送量有上界，不会无限堆积。
MAX_PUSH_PER_TRACE = 120

#: 默认 flush 等待上界（秒）。
#:
#: 为什么必须有上界：flush 会等后台导出队列排空，网络挂起时可能长时间不返回。
#: 收尾阶段最多等这么久，超了就放弃（数据仍在 SDK 队列里，由后台线程继续尝试）。
DEFAULT_FLUSH_TIMEOUT_S = 3.0

#: 进程退出时的 flush 上界（秒）。比每轮 flush 宽松一点：
#: 此时用户已经拿到回答，多等一会儿换"数据不丢"是划算的。
EXIT_FLUSH_TIMEOUT_S = 5.0

#: 默认的单侧正文上限（字符）。超出后保留**首尾**（见 `_clip`）。
#: 4000 字符 ≈ 2000 汉字，足够看清"问了什么、答了什么"，又不至于把
#: 一次问答的 span 撑到几百 KB。
DEFAULT_MAX_CAPTURE_CHARS = 4000

#: 默认 HTTP 超时（秒）。与 SDK 默认一致，显式传给 SDK 便于在 status() 里显示。
DEFAULT_TIMEOUT_S = 5.0


# ---------------------------------------------------------------------------
# 环境变量解析
# ---------------------------------------------------------------------------

def _env_str(name: str) -> str:
    """读环境变量并去掉 .env 里常见的包裹引号与空白。"""
    return (os.getenv(name, "") or "").strip().strip('"').strip("'")


def _env_bool(name: str, default: bool) -> bool:
    """布尔环境变量：认 `true/1/yes/on/是` 与 `false/0/no/off/否`，其余取默认。

    认不出的值**不抛异常**——配置写错不该让服务起不来，但也必须留下痕迹，
    故调用方（`status()`）会把"读到了什么"暴露出来供排查。
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().strip('"').strip("'").lower()
    if value in _FALSEY:
        return False
    if value in _TRUTHY:
        return True
    return default


def _env_float(name: str, default: float) -> float:
    """浮点环境变量：非法值取默认（不抛）。

    与 Langfuse SDK 的行为**刻意不同**：SDK 在 `LANGFUSE_SAMPLE_RATE=abc` 时
    直接 `float()` 抛 ValueError，客户端建不起来 → 整条上报静默失效。
    这里取默认值并告警，至少服务是活的、`status()` 里能看出配置有问题。
    """
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip().strip('"').strip("'"))
    except ValueError:
        logger.warning("%s=%r 不是数字，按默认值 %s 处理", name, raw, default)
        return default


def resolve_host() -> str:
    """解析 Langfuse 地址，兼容两种变量名。

    返回空串表示"未配置"（此时由 SDK 用它的默认值）。**不编造默认值**：
    若我们硬编码 `https://cloud.langfuse.com`，用户配了 `LANGFUSE_BASE_URL`
    指向自建实例时会**静默发错地方**——那正是这一版修掉的 bug。
    """
    for name in ENV_HOST_ALIASES:
        value = _env_str(name)
        if value:
            return value
    return ""


def tracing_enabled() -> bool:
    """总开关：`LANGFUSE_TRACING_ENABLED`，未设置时**默认开**（与 SDK 一致）。"""
    return _env_bool(ENV_TRACING_ENABLED, True)


def sample_rate() -> float:
    """采样率，收敛到 [0, 1]；非法值退回 1.0。"""
    value = _env_float(ENV_SAMPLE_RATE, 1.0)
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def flush_on_finish() -> bool:
    """每轮问答结束是否主动 flush（默认否，见模块文档的取舍）。"""
    return _env_bool(ENV_FLUSH_ON_FINISH, False)


def flush_timeout_s() -> float:
    """flush 等待上界（秒），下限 0.1 防止写出 0 导致永远超时。"""
    return max(0.1, _env_float(ENV_FLUSH_TIMEOUT, DEFAULT_FLUSH_TIMEOUT_S))


def capture_enabled() -> bool:
    """是否上报 prompt / 回答正文。**默认否**（招投标场景含敏感信息）。"""
    return _env_bool(ENV_CAPTURE_CONTENT, False)


def capture_limit() -> int:
    """单侧正文长度上限（字符）。0 或负数视为不采集。"""
    value = int(_env_float(ENV_MAX_CAPTURE_CHARS, float(DEFAULT_MAX_CAPTURE_CHARS)))
    return max(0, value)


def timeout_s() -> float:
    """HTTP 超时（秒）。"""
    return max(1.0, _env_float(ENV_TIMEOUT, DEFAULT_TIMEOUT_S))


def is_configured() -> bool:
    """是否配置了 Langfuse 凭据（两个 key 都要有）。

    注意它**不看**总开关：`is_configured()` 回答的是"凭据齐不齐"，
    "到底会不会发数据"要问 `is_enabled()`。两者分开是为了让 `status()` 能
    如实区分"没配"与"配了但被关掉"。
    """
    return bool(_env_str(ENV_PUBLIC_KEY) and _env_str(ENV_SECRET_KEY))


def _sdk_available() -> bool:
    """SDK 是否可用（可选依赖，未装时静默降级）。"""
    try:
        import langfuse  # noqa: F401
    except Exception:  # noqa: BLE001 - ImportError 及任何导入期异常
        return False
    return True


def sdk_version() -> str:
    """SDK 版本号（诊断用；取不到返回空串）。"""
    try:
        import importlib.metadata as md

        return md.version("langfuse")
    except Exception:  # noqa: BLE001
        return ""


def is_enabled() -> bool:
    """当前是否**真的会**往 Langfuse 发数据。

    这是唯一的总判据：凭据齐 + 总开关开 + SDK 装好。三者缺一都是 no-op，
    且 `status()["reason"]` 会说明是哪一条不满足。
    """
    return is_configured() and tracing_enabled() and _sdk_available()


# ---------------------------------------------------------------------------
# 进程级计数（给 /api/metrics 与 status() 用）
# ---------------------------------------------------------------------------
#
# 为什么要有计数：Langfuse 是**外部**依赖，"到底发出去了没有"是运维最常问的
# 问题。只报 enabled=true 不足以回答它——可能开着但每条都失败。
# 故这里记"试了多少、成了多少、失败多少、最近一次错误是什么"。

_STATS_LOCK = threading.Lock()
_STATS: dict[str, Any] = {
    "traces_started": 0,        # 开启的 trace 数（已通过采样）
    "traces_sampled_out": 0,    # 因采样被跳过的 trace 数
    "spans_pushed": 0,          # 成功推入 SDK 队列的 observation 数
    "spans_failed": 0,          # 推送时抛异常的 observation 数
    "spans_dropped_by_cap": 0,  # 超出 MAX_PUSH_PER_TRACE 被丢弃的条数
    "cooldowns_entered": 0,     # 进入冷却的次数
    "flushes": 0,               # 主动 flush 次数
    "flush_timeouts": 0,        # flush 超时次数
    "captured_chars": 0,        # 实际发出去的正文总字符数（脱敏后）
    "redactions": 0,            # 脱敏替换次数
}


def stats() -> dict[str, Any]:
    """进程级上报计数快照（线程安全）。"""
    with _STATS_LOCK:
        return dict(_STATS)


def reset_stats() -> None:
    """清零计数（测试用；生产中不应调用）。"""
    with _STATS_LOCK:
        for key in _STATS:
            _STATS[key] = 0


def _bump(key: str, delta: int = 1) -> None:
    with _STATS_LOCK:
        _STATS[key] = _STATS.get(key, 0) + delta


# ---------------------------------------------------------------------------
# 脱敏（★ 强制，无开关）
# ---------------------------------------------------------------------------

#: 脱敏规则。**顺序有意义**：先长后短、先具体后笼统——
#: 身份证 18 位若不先处理，会被"银行账号 16~19 位"规则截走一半。
#:
#: 为什么这些字段：招投标文件里出现频率最高且**最不该外流**的就是这些
#: （项目编号能反查到具体项目，金额与投标人名称是商业敏感信息）。
#: 替换成 `[金额]` 这类占位符而不是删除：保留句子结构，人读 trace 时
#: 仍能看懂"这里原本有个金额"。
REDACTION_RULES: tuple[tuple[str, re.Pattern[str], str], ...] = (
    ("id_card", re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"), "[身份证号]"),
    ("phone", re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"), "[手机号]"),
    ("uscc", re.compile(r"(?<![0-9A-Z])[0-9A-HJ-NPQRTUWXY]{18}(?![0-9A-Z])"), "[统一社会信用代码]"),
    ("bank_account", re.compile(r"(?<!\d)\d{16,19}(?!\d)"), "[银行账号]"),
    ("email", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"), "[邮箱]"),
    ("money", re.compile(r"(?:人民币|RMB|¥|￥)?\s*\d[\d,]*(?:\.\d+)?\s*(?:亿元|万元|亿|万|元)"), "[金额]"),
    (
        "project_no",
        re.compile(r"((?:项目|招标|采购|标段|包)?(?:编号|编码|文号)\s*[:：]?\s*)([A-Za-z0-9\u4e00-\u9fa5\-—_/]{3,})"),
        r"\1[编号]",
    ),
    (
        "company",
        re.compile(
            r"[\u4e00-\u9fa5]{2,20}(?:股份有限公司|有限责任公司|有限公司|集团有限公司"
            r"|集团|设计院|研究院|工程公司|服务中心|采购中心)"
        ),
        "[企业名称]",
    ),
    (
        "person",
        re.compile(
            r"((?:联系人|法定代表人|法人代表|授权代表|项目负责人|项目经理|拟派人员|姓名)\s*[:：]?\s*)"
            r"([\u4e00-\u9fa5]{2,4})"
        ),
        r"\1[姓名]",
    ),
)

#: 脱敏后的占位符形态：`[xxx]`。第二次脱敏时用它做幂等判断，
#: 避免 `mask=` 钩子对已脱敏内容重复计数。
_REDACTED_MARK = re.compile(r"\[(?:金额|编号|企业名称|姓名|手机号|身份证号|统一社会信用代码|银行账号|邮箱)\]")


def redact(text: str) -> str:
    """对文本做脱敏，返回脱敏后的文本。

    幂等：已经替换过的占位符不会再被匹配（占位符里没有数字/公司后缀）。

    为什么不做"可关闭的开关"：Langfuse 在境外、是第三方服务，
    脱敏一旦能关，迟早会有人在排查问题时关掉它并忘记打开。
    """
    if not text:
        return text
    out = str(text)
    hits = 0
    for _name, pattern, replacement in REDACTION_RULES:
        out, n = pattern.subn(replacement, out)
        hits += n
    if hits:
        _bump("redactions", hits)
    return out


def redact_payload(data: Any) -> Any:
    """递归脱敏任意 JSON 结构（str 脱敏、dict/list 逐层下钻、其余原样）。

    这个函数同时被用作 SDK 的 `mask=` 钩子——SDK 会对每条 observation 的
    input / output / metadata 都调它，等于给"将来新增的调用点忘了脱敏"
    加了一道兜底。
    """
    if isinstance(data, str):
        return redact(data)
    if isinstance(data, dict):
        return {k: redact_payload(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [redact_payload(v) for v in data]
    return data


def _clip(text: str, limit: int) -> str:
    """按上限截断，**保留首尾**并注明省略了多少字符。

    为什么首尾都留：
        · 首部是结论（回答的第一句往往就是答案、prompt 开头是角色设定）
        · 尾部是最新上下文（用户最后问的那句话）
    只留首部会把"用户到底问了什么"丢掉，只留尾部则看不出回答的开头。
    """
    text = str(text or "")
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    head = limit // 3
    tail = limit - head
    return f"{text[:head]}…[省略 {len(text) - limit} 字符]…{text[-tail:]}"


# ---------------------------------------------------------------------------
# 成本估算（Langfuse 控制台里"金额为 0"的修法）
# ---------------------------------------------------------------------------
#
# ★ 为什么在本模块自己算，而不是依赖 Langfuse 侧的模型价目表
# ------------------------------------------------------
# Langfuse 能按 model 名匹配它内置的价目表算成本，但：
#     ① 它内置的表**未必收录**本地/国产模型（如 qwen2.5:7b、glm-*-flashx）
#     ② 用户在 Langfuse 侧没配 model definition 时，金额就是 0
#     ③ 缓存命中的折扣价（DeepSeek 的 `prompt_cache_hit_tokens`）在
#        对方表里不一定有对应配置
# 故这里按 token 数**显式发 `cost_details`**：Langfuse 收到即采用，不再猜。
#
# ⚠️ 单价会变，且不同渠道（官方/代理/包年）差异很大。下表是**参考值**，
#    单位是「美元 / 100 万 token」，请按官方价目自行核对；
#    可用 LANGFUSE_PRICE_TABLE 覆盖（JSON），无需改代码。

DEFAULT_PRICE_TABLE: dict[str, dict[str, float]] = {
    # DeepSeek 官方价（2025 年前后）。cached_input 是缓存命中价（约 1/4）。
    "deepseek-chat": {"input": 0.27, "output": 1.10, "cached_input": 0.07},
    "deepseek-reasoner": {"input": 0.55, "output": 2.19, "cached_input": 0.14},
    # ★ 项目默认模型。官方按时段浮动（2026-10 价目：
    #   https://api-docs.deepseek.com/quick_start/pricing/），
    #   峰值 input 0.30 / output 1.20 / 缓存命中 0.006，空闲时段为一半。
    #   单值表只能存一组数，故取**峰值**——宁可高估也不低估成本；
    #   只在空闲时段跑评测时，可用 LANGFUSE_PRICE_TABLE 覆盖成半价。
    "deepseek-flash": {"input": 0.30, "output": 1.20, "cached_input": 0.006},
    # 本地 Ollama：没有 API 费用（电费与显卡不算），显式记 0 比"缺数据"更准确
    "qwen2.5:7b": {"input": 0.0, "output": 0.0},
    # 智谱（glm-4.7-flashx / glm-5.3-flashx）：flashx 系列按低价档估
    "glm-4.7-flashx": {"input": 0.10, "output": 0.10},
    "glm-5.3-flashx": {"input": 0.10, "output": 0.10},
    # OpenAI 参考价（项目里有 OpenAI 兼容客户端，可能被切过去）
    "gpt-4o-mini": {"input": 0.15, "output": 0.60, "cached_input": 0.075},
}


def _price_table() -> tuple[dict[str, dict[str, float]], str]:
    """返回 `(单价表, 来源)`。来源取值 `builtin` / `env`，供 metadata 记录。

    `LANGFUSE_PRICE_TABLE` 是**合并**而不是替换：只写要覆盖的那几个模型即可，
    没写的仍用内置值（否则用户改一个模型就得把整张表抄一遍）。
    """
    table = dict(DEFAULT_PRICE_TABLE)
    raw = _env_str(ENV_PRICE_TABLE)
    if not raw:
        return table, "builtin"
    try:
        override = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        logger.warning("%s 不是合法 JSON，改用内置单价表", ENV_PRICE_TABLE)
        return table, "builtin"
    if not isinstance(override, dict):
        return table, "builtin"
    for model, price in override.items():
        if isinstance(price, dict):
            table[str(model).lower()] = {k: float(v) for k, v in price.items() if isinstance(v, (int, float))}
    return table, "env"


def _lookup_price(model: str) -> dict[str, float] | None:
    """按模型名找单价：先精确匹配，再前缀匹配（`deepseek-chat-0324` → `deepseek-chat`）。"""
    name = (model or "").strip().lower()
    if not name:
        return None
    table, _source = _price_table()
    if name in table:
        return table[name]
    for key, price in table.items():
        if key and name.startswith(key):
            return price
    return None


def estimate_cost(model: str, usage: dict[str, int] | None) -> dict[str, float] | None:
    """按 token 用量估算成本，返回 Langfuse 的 `cost_details`（美元）。

    返回 `None` 表示**算不出来**（模型不在价目表里 / 没有任何 token 数据）。
    此时宁可不发：发 0 会让控制台显示"这次没花钱"，而事实是"不知道多少钱"。

    缓存命中的处理：DeepSeek 的 `input` 是**含**缓存命中的总量
    （`hit + miss = prompt_tokens`），故命中部分按折扣价、其余按全价，
    不能把 input 整体再按全价算一遍（那样成本会偏高）。
    """
    if not usage:
        return None
    price = _lookup_price(model)
    if price is None:
        return None

    def _cost(tokens: float, unit_price: float | None) -> float:
        return (tokens / 1_000_000.0) * float(unit_price or 0.0)

    input_tokens = usage.get("input")
    output_tokens = usage.get("output")
    cached = usage.get("input_cached_tokens")
    if input_tokens is None and output_tokens is None:
        return None

    input_cost = 0.0
    if input_tokens is not None:
        cached_price = price.get("cached_input")
        if cached is not None and cached_price is not None:
            billed_cached = min(int(cached), int(input_tokens))
            input_cost = _cost(billed_cached, cached_price) + _cost(
                int(input_tokens) - billed_cached, price.get("input")
            )
        else:
            input_cost = _cost(input_tokens, price.get("input"))
    output_cost = _cost(output_tokens or 0, price.get("output"))

    total = input_cost + output_cost
    return {
        "input": round(input_cost, 8),
        "output": round(output_cost, 8),
        "total": round(total, 8),
    }


# ---------------------------------------------------------------------------
# 客户端
# ---------------------------------------------------------------------------

_CLIENT_LOCK = threading.Lock()
#: 已建过的客户端（`(key, base_url)` → client）。SDK 内部本就按 public key
#: 复用资源，这里缓存是为了让 `flush()` / 自检 / 反馈评分拿到同一个对象。
_CLIENTS: dict[tuple[str, str], Any] = {}
_ATEXIT_REGISTERED = False


def _register_atexit_flush() -> None:
    """首次真正建客户端时注册退出 flush（没建过就什么都不做）。

    为什么不在 import 阶段注册：模块被 import 不代表会用到 Langfuse，
    注册了会让**每次**进程退出都白跑一遍（包括测试）。
    """
    global _ATEXIT_REGISTERED
    with _CLIENT_LOCK:
        if _ATEXIT_REGISTERED:
            return
        _ATEXIT_REGISTERED = True
    try:
        atexit.register(_atexit_flush)
    except Exception:  # pragma: no cover - 解释器收尾阶段的极端情况
        pass


def _atexit_flush() -> None:
    """进程**正常退出**时排空队列。

    覆盖不到的情况要如实说明：`SIGKILL` / 断电 / 崩溃退出时不会执行，
    那部分 span 就丢了——这正是 `LANGFUSE_FLUSH_ON_FINISH=true` 存在的意义。
    """
    for client in list(_CLIENTS.values()):
        _safe_flush(client, EXIT_FLUSH_TIMEOUT_S, source="atexit")


def build_client() -> Any:
    """**新建**一个 Langfuse 客户端（不缓存）。不可用时返回 `None`。

    构造参数集中在这里，`get_client()`（带缓存）与 `langfuse_feedback`
    （每次现建，避免把评分挂到旧凭据的客户端上）共用同一份——参数只写一处，
    将来 SDK 改参数名时不会漏改其中一个调用点。
    """
    if not is_enabled():
        return None
    try:
        from langfuse import Langfuse
    except Exception:  # noqa: BLE001
        logger.warning("导入 Langfuse SDK 失败，本轮不上报", exc_info=True)
        return None

    kwargs: dict[str, Any] = {
        "public_key": _env_str(ENV_PUBLIC_KEY),
        "secret_key": _env_str(ENV_SECRET_KEY),
        # ★ v5 的规范参数名是 `base_url`（`host` 是旧名，且**环境变量优先于
        # 构造参数**：SDK 先读 LANGFUSE_BASE_URL 再读 host）。用 `base_url`
        # 与我们 resolve_host() 的优先级保持一致，避免"传了却没生效"。
        "base_url": resolve_host() or None,
        # 把总开关**显式**传下去：SDK 只把字面量 "false" 当关，
        # 用户写 "0" 时 SDK 仍会上报，而我们已判定为关——显式传参消除分歧。
        "tracing_enabled": True,
        # 采样由本模块按 trace 做（见 begin()），故这里恒为 1.0：
        # 否则 SDK 会**再采一次**，实际采样率变成 rate² 且我们无法统计。
        "sample_rate": 1.0,
        "timeout": int(timeout_s()),
        # 双保险：SDK 会对每条 observation 的 input/output/metadata 调 mask。
        "mask": redact_payload if capture_enabled() else None,
    }
    try:
        client = Langfuse(**kwargs)
    except TypeError:
        # 老版本 SDK 没有 mask / sample_rate 参数：去掉新参数重试一次，
        # 保证"换了 SDK 版本"不会让整个上报链路静默失效。
        for drop in ("mask", "sample_rate"):
            kwargs.pop(drop, None)
        try:
            client = Langfuse(**kwargs)
        except Exception:  # noqa: BLE001
            logger.warning("初始化 Langfuse 客户端失败，本轮不上报", exc_info=True)
            return None
    except Exception:  # noqa: BLE001
        logger.warning("初始化 Langfuse 客户端失败，本轮不上报", exc_info=True)
        return None
    return client


def get_client() -> Any:
    """返回（必要时创建并缓存）Langfuse 客户端；不可用时返回 `None`。

    同一个 `(public_key, base_url)` 只建一次：SDK 内部按 public key 复用
    资源管理器，重复 `Langfuse(...)` 不会多开线程，但我们的计数与 flush
    需要拿到**同一个**对象。
    """
    if not is_enabled():
        return None
    key = (_env_str(ENV_PUBLIC_KEY), resolve_host())
    with _CLIENT_LOCK:
        cached = _CLIENTS.get(key)
    if cached is not None:
        return cached
    client = build_client()
    if client is None:
        return None
    with _CLIENT_LOCK:
        _CLIENTS[key] = client
    _register_atexit_flush()
    return client


def _safe_flush(client: Any, timeout: float, source: str = "manual") -> bool:
    """带**超时上界**的 flush。返回是否在超时内完成。

    为什么用线程 + join(timeout)：SDK 的 `flush()` 没有超时参数，
    它要等后台导出队列排空。网络挂起时它会一直等——放在请求收尾路径上
    就是一个"偶发卡死 30 秒"的 bug。丢一个守护线程比拖住用户强。
    """
    if client is None:
        return False
    done = threading.Event()

    def _run() -> None:
        try:
            client.flush()
        except Exception:  # noqa: BLE001
            logger.debug("Langfuse flush 失败（%s）", source, exc_info=True)
        finally:
            done.set()

    _bump("flushes")
    thread = threading.Thread(target=_run, name=f"langfuse-flush-{source}", daemon=True)
    try:
        thread.start()
    except Exception:  # noqa: BLE001 - 解释器收尾阶段可能不让起线程
        logger.debug("Langfuse flush 线程启动失败（%s）", source, exc_info=True)
        return False
    if done.wait(timeout):
        return True
    _bump("flush_timeouts")
    logger.debug("Langfuse flush 超时（%s，>%.1fs），交由后台导出继续", source, timeout)
    return False


def flush(timeout_s_: float | None = None) -> dict[str, Any]:
    """主动排空所有已建客户端的上报队列。返回执行摘要。

    用途：`LANGFUSE_FLUSH_ON_FINISH` 关掉时也能"关键轮次必达"
    （比如用户点了导出、或服务要重启），以及自检脚本收尾。
    """
    limit = flush_timeout_s() if timeout_s_ is None else max(0.1, float(timeout_s_))
    clients = list(_CLIENTS.values())
    results = [_safe_flush(c, limit, source="manual") for c in clients]
    return {"clients": len(clients), "completed": sum(1 for r in results if r), "timeout_s": limit}


def flush_client(client: Any, timeout_s_: float | None = None) -> bool:
    """排空**指定**客户端的队列。

    与 `flush()` 的区别：`flush()` 只管本模块缓存过的客户端。用户反馈
    （`langfuse_feedback.score_feedback`）自己现建客户端，不在缓存里，
    却同样需要"这条数据立刻出去"（人工产生的数据无法重放）。
    """
    limit = flush_timeout_s() if timeout_s_ is None else max(0.1, float(timeout_s_))
    return _safe_flush(client, limit, source="explicit")


# ---------------------------------------------------------------------------
# Tagger：把一轮问答推给 Langfuse
# ---------------------------------------------------------------------------

class LangfuseTagger:
    """把一次问答的观测数据推送到 Langfuse。

    生命周期与"一轮问答"一致：请求开始时 `begin()`，结束时 `finish()`。
    线程安全（工具层是 4 线程并行，可能同时 `push_tool`）。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._trace_id = ""
        self._pushed = 0
        self._failed = False
        #: 失败冷却截止（monotonic）。0 表示不在冷却
        self._cooldown_until = 0.0
        #: 本轮是否被采样命中。未命中则整轮不上报（保证 trace 完整而非半棵树）
        self._sampled = True

    # --- 可用性 ---

    def _in_cooldown(self) -> bool:
        with self._lock:
            return time.monotonic() < self._cooldown_until

    def _enter_cooldown(self) -> None:
        with self._lock:
            self._cooldown_until = time.monotonic() + FAILURE_COOLDOWN_SECONDS
        _bump("cooldowns_entered")

    def available(self) -> bool:
        """当前能否上报。**不做网络请求**（只反映本地状态与开关）。"""
        if not is_enabled() or self._in_cooldown():
            return False
        with self._lock:
            return self._sampled

    def sampled(self) -> bool:
        """本轮是否被采样命中（`begin()` 之后才有意义）。"""
        with self._lock:
            return self._sampled

    def _get_client(self):
        """惰性建客户端。失败返回 None 并进入冷却。"""
        client = get_client()
        if client is None:
            if is_enabled():
                self._enter_cooldown()
            return None
        return client

    # --- 一轮问答 ---

    def begin(self, name: str, *, input_summary: dict | None = None,
              metadata: dict | None = None) -> None:
        """开启一轮 trace。失败时静默降级（不影响问答）。

        采样在**这里**决定，且是 trace 级：要么整棵树都记，要么一条都不记。
        若按 observation 采样，UI 里会出现"有工具调用但没有对应的 LLM 调用"
        这种残缺调用树，比没有数据更容易误导排查。

        trace id 由 SDK 生成并由我们保存——后续 observation 靠它挂到同一棵树上。
        """
        rate = sample_rate()
        hit = rate >= 1.0 or (rate > 0.0 and random.random() < rate)
        with self._lock:
            self._sampled = hit
            self._pushed = 0
        if not hit:
            _bump("traces_sampled_out")
            return
        if not self.available():
            return
        client = self._get_client()
        if client is None:
            return
        try:
            # `create_trace_id` 让我们显式持有 id，便于把并行产生的工具
            # observation 挂到同一 trace 上（工具是 4 线程并行的）
            trace_id = client.create_trace_id()
            with self._lock:
                self._trace_id = trace_id
            _bump("traces_started")
            client.create_event(
                name=name,
                trace_context={"trace_id": trace_id},
                input=input_summary,
                # 采样率与开关状态写进 trace 元数据：事后看一条 trace 时
                # 能立刻知道"这条为什么没有正文"，不用去翻当时的 .env
                metadata={
                    **(metadata or {}),
                    "sample_rate": rate,
                    "capture_content": capture_enabled(),
                },
            )
        except Exception:  # noqa: BLE001
            logger.debug("Langfuse begin 失败", exc_info=True)
            self._failed = True

    def push_llm_call(self, call: LLMCall) -> None:
        """上报一次 LLM 调用（generation observation）。"""
        usage = _usage_details(call)
        capture = capture_enabled()
        limit = capture_limit()
        payload_out: dict[str, Any] = {"chars": call.content_chars}
        payload_in: dict[str, Any] | None = None
        if capture and limit > 0:
            # ★ 先脱敏再截断？**先截断再脱敏**更省 CPU，但会让"截断点恰好
            # 切在手机号中间"的残片逃过脱敏。故顺序是**先脱敏再截断**。
            prompt_text = redact(getattr(call, "prompt_text", "") or "")
            response_text = redact(getattr(call, "response_text", "") or "")
            if prompt_text:
                payload_in = {"prompt": _clip(prompt_text, limit)}
            if response_text:
                payload_out["response"] = _clip(response_text, limit)
            _bump("captured_chars", len(prompt_text) + len(response_text))
        cost = estimate_cost(call.model, usage)
        metadata = {
            "provider": call.provider,
            "ok": call.ok,
            "error": call.error or None,
            "reasoning_chars": call.reasoning_chars,
            "tools": call.tools or None,
            "cache_hit_tokens": call.cache_hit_tokens,
            "cache_miss_tokens": call.cache_miss_tokens,
        }
        if cost is not None:
            metadata["cost_usd"] = cost["total"]
            metadata["cost_source"] = _price_table()[1]
        self._push(
            as_type="generation",
            name=call.kind or "llm",
            model=call.model or None,
            usage_details=usage,
            cost_details=cost,
            input=payload_in,
            output=payload_out,
            metadata=metadata,
            level="ERROR" if not call.ok else "DEFAULT",
            status_message=call.error or None,
        )

    def push_tool(self, timing: dict) -> None:
        """上报一次工具执行（tool observation）。

        工具入参/出参**不采集**：它们常含检索到的原文片段（可能整段招标文件），
        体积大且敏感度最高，而排查"慢在哪、有没有失败"并不需要看它们。
        """
        self._push(
            as_type="tool",
            name=str(timing.get("name") or "tool"),
            model=None,
            usage_details=None,
            cost_details=None,
            input=None,
            output={
                "ms": timing.get("ms"),
                "ok": timing.get("ok"),
                "sources": timing.get("sources"),
                "chars": timing.get("chars"),
                "round": timing.get("round"),
            },
            metadata=None,
            level="DEFAULT" if timing.get("ok") else "ERROR",
            status_message=None if timing.get("ok") else "工具执行失败",
        )

    def _push(self, *, as_type: str, name: str, model, usage_details, cost_details,
              input, output, metadata, level, status_message) -> None:
        """统一的推送入口。

        ★ 为什么所有路径都要包 try/except
        -----------------------------
        这是**唯一向外发数据**的代码。它一旦抛异常，可能把一次成功的问答
        变成 500 —— 与 `monitoring.safe_record` 同一条原则：
        观测代码绝不能搞挂业务。
        """
        if not self.available():
            return
        client = self._get_client()
        if client is None:
            return
        with self._lock:
            if self._pushed >= MAX_PUSH_PER_TRACE:
                _bump("spans_dropped_by_cap")
                return
            self._pushed += 1
            trace_id = self._trace_id
        try:
            kwargs = {
                "name": name,
                "as_type": as_type,
                "output": output,
                "level": level,
                "trace_context": {"trace_id": trace_id} if trace_id else None,
            }
            if model:
                kwargs["model"] = model
            if usage_details:
                kwargs["usage_details"] = usage_details
            if cost_details:
                kwargs["cost_details"] = cost_details
            if input is not None:
                kwargs["input"] = input
            if metadata:
                kwargs["metadata"] = metadata
            if status_message:
                kwargs["status_message"] = status_message

            obs = client.start_observation(**kwargs)
            # 我们记的是**已完成**的调用（耗时已知），故直接 end 而不是留 open。
            # 传 `end_time` 由 SDK 用当前时间；耗时通过 metadata 表达。
            obs.end()
            _bump("spans_pushed")
        except Exception:  # noqa: BLE001
            _bump("spans_failed")
            logger.debug("Langfuse 上报失败（不影响问答）", exc_info=True)
            self._enter_cooldown()

    def finish(self, *, output_summary: dict | None = None,
               metadata: dict | None = None) -> None:
        """结束一轮 trace；按 `LANGFUSE_FLUSH_ON_FINISH` 决定是否立刻 flush。

        flush 的取舍（两条路都有代价，故做成开关）：
            · 不 flush（默认）：零收尾延迟，但进程被 kill / 崩溃时最多丢几秒 span
            · flush：这一轮**必达**，代价是结尾多一次等待（有超时上界，
              不会无限等）

        无论开关如何，进程正常退出时都会兜底 flush 一次（见 `_atexit_flush`）。
        """
        if not self.available():
            return
        client = self._get_client()
        if client is None:
            return
        try:
            # 用 event 记录本轮汇总（trace 级输出由 SDK 从 observations 汇总）
            client.create_event(
                name="turn_summary",
                trace_context={"trace_id": self._trace_id} if self._trace_id else None,
                output=output_summary,
                metadata=metadata,
            )
        except Exception:  # noqa: BLE001
            logger.debug("Langfuse finish 失败", exc_info=True)
        if flush_on_finish():
            _safe_flush(client, flush_timeout_s(), source="finish")


def _usage_details(call: LLMCall) -> dict[str, int] | None:
    """构造 Langfuse 的 `usage_details`。

    ★ 只放**真实取到**的字段（不补 0）
    -----------------------------
    Langfuse 用这些值算成本。补 0 会让"某次调用没拿到 usage"看起来像
    "那次没花 token"，从而把成本曲线压低——比缺数据更糟。
    """
    details: dict[str, int] = {}
    if call.prompt_tokens is not None:
        details["input"] = call.prompt_tokens
    if call.completion_tokens is not None:
        details["output"] = call.completion_tokens
    if call.total_tokens is not None:
        details["total"] = call.total_tokens
    if call.cache_hit_tokens is not None:
        # Langfuse 约定：缓存命中的部分单独计入，用于算折扣价
        details["input_cached_tokens"] = call.cache_hit_tokens
    return details or None


class LangfuseTraceRecorder(LLMTraceRecorder):
    """在原有记录器基础上，顺手把每条记录推给 Langfuse。

    ★ 为什么用继承而不是包装
    --------------------
    `LLMTraceRecorder` 是 agent 与客户端都认识的类型；包装一层会改变类型，
    需要改动调用方的类型假设。继承则**对外完全一致**——agent 拿到的还是
    `LLMTraceRecorder`，`summary()` 行为不变，只是多了上报副作用。

    这也让"关闭 Langfuse"变成一行：用基类而不是本类。
    """

    def __init__(self, tagger: LangfuseTagger | None = None) -> None:
        super().__init__()
        self.tagger = tagger or LangfuseTagger()

    def record(self, call: LLMCall) -> None:
        super().record(call)  # 先落本地（保证即使上报失败也有本地数据）
        try:
            self.tagger.push_llm_call(call)
        except Exception:  # noqa: BLE001 - 双保险：push 内部已兜底
            logger.debug("Langfuse push_llm_call 异常", exc_info=True)

    def push_tool(self, timing: dict) -> None:
        """上报工具执行。由 `react_loop` 在记录本地 tool_timings 后调用。"""
        try:
            self.tagger.push_tool(timing)
        except Exception:  # noqa: BLE001
            logger.debug("Langfuse push_tool 异常", exc_info=True)


def build_recorder() -> LLMTraceRecorder:
    """按配置返回记录器。

    未启用（未配置 / 总开关关 / SDK 不可用）时返回**基类**（纯本地记录），
    这样调用方无需分支判断，且关闭时没有额外开销。

    ★ 为什么用工厂而不是让调用方判断
    --------------------------
    `is_enabled()` 的判断若散落在 agent 里，新增一个开关就要改多处；
    工厂把决策收在一处。
    """
    if not is_enabled():
        return LLMTraceRecorder()
    return LangfuseTraceRecorder()


# ---------------------------------------------------------------------------
# 状态 / 指标
# ---------------------------------------------------------------------------

def status() -> dict:
    """供 `/api/health`、`/api/metrics` 与诊断用：是否上报、为什么、当前开关。"""
    configured = is_configured()
    switch = tracing_enabled()
    sdk = _sdk_available()
    reason = ""
    if not configured:
        reason = f"未配置 {ENV_PUBLIC_KEY} / {ENV_SECRET_KEY}"
    elif not switch:
        reason = f"已被 {ENV_TRACING_ENABLED} 关闭"
    elif not sdk:
        reason = "langfuse 未安装（uv pip install -e .[langfuse]）"
    return {
        "enabled": configured and switch and sdk,
        "configured": configured,
        "tracing_enabled": switch,
        "sdk_installed": sdk,
        "sdk_version": sdk_version(),
        "host": resolve_host() or "(SDK 默认)",
        "host_env": next((n for n in ENV_HOST_ALIASES if _env_str(n)), ""),
        "sample_rate": sample_rate(),
        "capture_content": capture_enabled(),
        "max_capture_chars": capture_limit(),
        "flush_on_finish": flush_on_finish(),
        "flush_timeout_s": flush_timeout_s(),
        "reason": reason,
    }


def metrics() -> dict:
    """`/api/metrics` 用的一段：状态 + 进程级计数。**永不抛异常**。"""
    try:
        return {**status(), "counters": stats()}
    except Exception as exc:  # noqa: BLE001
        return {"enabled": False, "error": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------------------
# 自检（解决"到底有没有真的发出去"）
# ---------------------------------------------------------------------------

def self_test(send: bool = False, timeout_s_: float = 15.0, poll_s: float = 12.0) -> dict:
    """自检 Langfuse 链路。返回结构化结果，**永不抛异常**。

    为什么需要它：`status()` 只能说明"配置齐了、开关开了"，回答不了
    "凭据有效吗、地址对吗、数据真的进得去吗"。这三件事只有真发一次请求才知道。

    两档，默认只做**只读**那一档：

        `send=False`（默认）
            ① SDK 是否装好、配置是否齐、总开关状态
            ② 用凭据 `GET {host}/api/public/projects`——**只读**，
               不往账号里写任何数据。返回 200 即证明"地址对 + 凭据有效"，
               401/403 则是 key 错，连不上是网络/地址问题。

        `send=True`（**显式要求**才做）
            ③ 真发一条 trace（名字里带 `self-test`，**不含任何业务正文**），
               flush 后轮询公开 API 看它是否已经可查——这是唯一能证明
               "控制台里看得到"的方式。会在账号里留下一条测试 trace。

    返回字段：`ok`（只读检查是否全过）、`auth_ok`、`project`、`checked`、
    `sent`、`trace_id`、`trace_url`、`verified`、`reasons`。

    ⚠️ 网络抖动会被如实报成失败，值得多试几次
    --------------------------------------
    实测本机到 Langfuse Cloud 的连通性**不稳定**：同一分钟内三次探测里
    两次 `ConnectTimeout`、一次 200（0.3s）。这类失败与配置无关，
    重试即可。另外一次探测可能要几十秒——DNS 解析出多个地址时，
    httpx 会对每个地址各等一次超时，故"慢"不等于"错"。
    """
    result: dict[str, Any] = {
        "ok": False,
        "checked": [],
        "reasons": [],
        "sent": False,
        "verified": None,
        "trace_id": "",
        "trace_url": "",
        "project": "",
        "auth_ok": None,
        "status": status(),
    }

    def _fail(msg: str) -> dict:
        result["reasons"].append(msg)
        return result

    if not _sdk_available():
        return _fail("langfuse SDK 未安装：uv pip install -e .[langfuse]")
    result["checked"].append("sdk")
    if not is_configured():
        return _fail(f"未配置 {ENV_PUBLIC_KEY} / {ENV_SECRET_KEY}")
    result["checked"].append("credentials")
    if not tracing_enabled():
        return _fail(f"{ENV_TRACING_ENABLED} 已关闭（本模块不会发送任何数据）")
    result["checked"].append("switch")

    host = resolve_host()
    if not host:
        return _fail("未配置 LANGFUSE_BASE_URL / LANGFUSE_HOST")
    result["checked"].append("host")

    # ① 只读鉴权探测
    try:
        import httpx

        url = host.rstrip("/") + "/api/public/projects"
        resp = httpx.get(
            url,
            auth=(_env_str(ENV_PUBLIC_KEY), _env_str(ENV_SECRET_KEY)),
            timeout=timeout_s_,
        )
        result["auth_ok"] = resp.status_code == 200
        if resp.status_code == 200:
            data = resp.json()
            items = data.get("data") if isinstance(data, dict) else None
            if isinstance(items, list) and items:
                result["project"] = str(items[0].get("name") or "")
            result["checked"].append("auth")
        elif resp.status_code in (401, 403):
            return _fail(f"凭据被拒绝（HTTP {resp.status_code}）：检查 LANGFUSE_PUBLIC_KEY / SECRET_KEY")
        else:
            return _fail(f"公开 API 返回 HTTP {resp.status_code}（地址或网络有问题）")
    except Exception as exc:  # noqa: BLE001
        return _fail(f"无法访问 {host}：{type(exc).__name__}: {exc}")

    result["ok"] = True

    # ② 可选：真发一条自检 trace
    if not send:
        result["reasons"].append("只做了只读检查；要验证端到端上报请用 self_test(send=True)")
        return result

    try:
        import uuid

        tagger = LangfuseTagger()
        tagger.begin(
            "bidding-agent/self-test",
            input_summary={"note": "连通性自检，无业务数据"},
            metadata={"self_test": True, "trace": uuid.uuid4().hex[:8]},
        )
        if not tagger.sampled():
            result["reasons"].append("自检 trace 被采样率排除（把 LANGFUSE_SAMPLE_RATE 设为 1 再试）")
            return result
        tagger.push_llm_call(
            LLMCall(kind="self_test", provider="self-test", model="", elapsed_ms=0,
                    content_chars=0)
        )
        tagger.finish(output_summary={"self_test": True}, metadata={"self_test": True})
        trace_id = tagger._trace_id
        result["sent"] = True
        result["trace_id"] = trace_id
        result["trace_url"] = f"{host.rstrip('/')}/trace/{trace_id}" if trace_id else ""
        flush(flush_timeout_s())

        # ③ 轮询公开 API 确认"真的进去了"（ingestion 是异步的，要等）
        import httpx

        deadline = time.monotonic() + max(0.0, poll_s)
        while trace_id and time.monotonic() < deadline:
            try:
                resp = httpx.get(
                    host.rstrip("/") + "/api/public/traces",
                    params={"limit": 20},
                    auth=(_env_str(ENV_PUBLIC_KEY), _env_str(ENV_SECRET_KEY)),
                    timeout=timeout_s_,
                )
                if resp.status_code == 200:
                    rows = (resp.json() or {}).get("data") or []
                    if any(str(r.get("id")) == trace_id for r in rows):
                        result["verified"] = True
                        break
            except Exception:  # noqa: BLE001
                pass
            time.sleep(1.5)
        if result["verified"] is None:
            result["verified"] = False
            result["reasons"].append(
                "已发出但轮询窗口内没在公开 API 里查到——ingestion 有延迟，"
                f"可稍后打开 {result['trace_url']} 确认"
            )
    except Exception as exc:  # noqa: BLE001
        result["reasons"].append(f"发送自检 trace 失败：{type(exc).__name__}: {exc}")
    return result


__all__ = [
    "DEFAULT_MAX_CAPTURE_CHARS",
    "DEFAULT_PRICE_TABLE",
    "ENV_CAPTURE_CONTENT",
    "ENV_FLUSH_ON_FINISH",
    "ENV_FLUSH_TIMEOUT",
    "ENV_HOST",
    "ENV_HOST_ALIASES",
    "ENV_MAX_CAPTURE_CHARS",
    "ENV_PRICE_TABLE",
    "ENV_PUBLIC_KEY",
    "ENV_SECRET_KEY",
    "ENV_SAMPLE_RATE",
    "ENV_TRACING_ENABLED",
    "EXIT_FLUSH_TIMEOUT_S",
    "FAILURE_COOLDOWN_SECONDS",
    "LangfuseTagger",
    "LangfuseTraceRecorder",
    "MAX_PUSH_PER_TRACE",
    "REDACTION_RULES",
    "build_client",
    "build_recorder",
    "capture_enabled",
    "capture_limit",
    "estimate_cost",
    "flush",
    "flush_client",
    "flush_on_finish",
    "flush_timeout_s",
    "get_client",
    "is_configured",
    "is_enabled",
    "metrics",
    "redact",
    "redact_payload",
    "reset_stats",
    "resolve_host",
    "sample_rate",
    "sdk_version",
    "self_test",
    "stats",
    "status",
    "tracing_enabled",
]


if __name__ == "__main__":
    # 自检入口：`python -m src.langfuse_tracing`
    #
    # 默认**只读**：查 SDK / 配置 / 开关，并用凭据 GET 一次公开 API
    # （证明地址对、凭据有效），**不往账号里写任何数据**。
    # 加 `--send` 才会真发一条名为 `bidding-agent/self-test` 的 trace
    # 并回读确认——那是唯一能证明"控制台里看得到"的方式。
    import json as _json
    import sys as _sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:  # 独立跑时自己读 .env（应用启动时由 api/server.py 读）
        from dotenv import load_dotenv

        from src.config import PROJECT_ROOT

        load_dotenv(PROJECT_ROOT / ".env")
    except Exception:  # pragma: no cover - 缺 dotenv 时仍可用已导出的环境变量
        pass

    _report = self_test(send="--send" in _sys.argv)
    print(_json.dumps(_report, ensure_ascii=False, indent=2))

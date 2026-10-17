"""Agent 模块共享常量。

所有 magic number 集中在此并附语义说明；改动前先确认上下游不受影响
（前端 SSE 协议见 docs/开发文档.md §6，测试断言见 tests/）。
"""

import os
import threading

# --- 对话与工具循环 ---
MAX_HISTORY_ROUNDS = 5       # 对话历史保留轮数（1 轮 = user + assistant 两条）
MAX_TOOL_ROUNDS = 4          # ReAct 工具轮次上限（不含最终生成回合）
TOOL_TIMEOUT_SECONDS = 30    # 单个工具执行超时；超时按工具失败处理而非中断整个回答

# --- 流式输出 ---
#
# ★ STREAM_FLUSH_CHARS 是**安全参数**，其下界已实测标定，不要凭感觉调
# ---------------------------------------------------------------------
# 它同时承担两个方向相反的职责：
#   ① 防泄漏：缓冲要攒够字符，才能识别"模型把工具调用当正文输出"并拦下；
#   ② 观感：缓冲越小，用户越早看到字。
#
# 2026-10-11 用 5 种泄漏形态标定了安全下界（每种形态长度/格式不同）：
#
#     FLUSH   短64字  长102字  标签140字  tool_calls数组  说明文字+JSON   判定
#       80     ✅      ✅       ✅        ✅             ✅           安全
#       48     ✅      ✅       ✅        ✅             ✅           安全
#       40     ✅      ✅       ✅        ✅             ✅           安全 ← 最小安全值
#       32     ✅      ✅       ❌        ✅             ✅           不安全
#       24     ✅      ❌       ❌        ✅             ✅           不安全
#       16     ❌      ❌       ❌        ❌             ❌           不安全
#
# 取 40（而非保守的 80）：首字可见时间少等约 40 个字符（≈0.2~0.5s），
# 且 40 已在全部 5 种形态上验证通过。若未来新增更长的泄漏形态，需重新标定。
#
# ★ 一个曾经犯过的推理错误（记下来避免重犯）
# ----------------------------------------
# 我先看到"短泄漏在 FLUSH=24 下仍被拦下"，就推断"检测是逐字符触发的，
# 缓冲大小无关"，据此把 80 调到 24 —— 结果**长泄漏与标签泄漏直接下发给用户**。
# 错在两点：① 检测的触发粒度**就等于**缓冲大小，不是逐字符；
# ② 短用例能过只是因为它本身就短，**在单一配置下测出的结论不能外推**。
STREAM_FLUSH_CHARS = 40      # 防泄漏缓冲阈值：**安全参数**，下界已标定为 40

# 下面两个**只影响观感**，与泄漏防御无关（切片发生在检测之后，早已通过安全检查）。
#
# 实测一段 148 字的回答，帧数与每帧字数随 CHUNK 的变化：
#
#     CHUNK=40 →  4 帧，平均 37.0 字/帧   （打字机一段段跳）
#     CHUNK=12 → 13 帧，平均 11.4 字/帧   （较平滑，现值）
#     CHUNK=4  → 37 帧，平均  4.0 字/帧   （最平滑，但帧数偏多）
#
# 为什么原来的观感是"一次蹦 40 字"：LLM 客户端的真实粒度是 **1.8 字/chunk**
# （实测 DeepSeek SSE），但 `_pace_stream_chunks` 把它重新按 40 字切分，
# 等于把已经拿到的逐字信息又攒批丢掉了。改成 12 后视觉上接近逐字。
#
# 代价：帧数从 4 增到 13，额外延时 13 × 0.02s = 0.26s（远小于首 token 延迟 1~20s）。
# 若低端设备上长回答出现渲染卡顿，把 CHUNK 调回 24 或 40 即可（不影响安全）。
STREAM_CHUNK_CHARS = 12      # token 小帧切分长度，只影响观感
STREAM_CHUNK_DELAY = 0.02    # 帧间延时（秒），打字机效果并减轻前端渲染压力

# --- LLM ---
# 深度思考输出预算（reasoning 计入 max_tokens）。
#
# ★ 这个常量同时用在**两条**路径上
# -----------------------------
#   ① 真推理模型（`chat_stream_thinking`）：思维链 + 正文共享这个预算，
#      故必须比普通请求宽——思考模式的思维链动辄数千 token。
#   ② 提示词注入回退：内容全在正文里，同样需要更宽的预算，
#      否则"拆解 → 多角度分析 → 结论"会写到一半被截断。
#
# 原实现只给了 ①，② 走的是 `chat_stream(messages)` 且**不带 max_tokens**，
# 于是用厂商默认值——恰恰是最需要放宽的那条路径没有放宽。
#
# 可用 `MAX_TOKENS_DEEP` 环境变量覆盖：本地小模型（ollama/vllm）的上下文窗口
# 可能远小于 8192，传一个超过窗口的值会被服务端拒绝或静默截断。
MAX_TOKENS_DEEP = int(os.getenv("MAX_TOKENS_DEEP", "8192"))

# --- 工具名清单（SYSTEM_PROMPT 与来源分流共用，新增工具同步维护） ---

#: 一次回答最多随 done 帧下发几张图。
#:
#: 一轮可能调 4 个工具（含重复），不限制的话 done 帧会塞进十几张图，
#: 前端渲染成一长串图表墙，反而淹没回答。
MAX_CHARTS = 3

#: 追问工具名（多处引用，避免字符串字面量写歪）。
CLARIFY_TOOL_NAME = "ask_clarification"

#: 追问相关的展示上限。
#:
#: 这些是**面向用户**的约束，不是技术限制。追问出现在用户面前的门槛应当
#: 比检索高：一次糟糕的追问（空问题、200 字的"问题"、重复选项）比不追问
#: 更伤害体验。故在 Agent 层做长度与数量夹取，而不是信任模型输出。
MAX_CLARIFY_QUESTIONS = 3        # 最多问几条
MAX_CLARIFY_OPTIONS = 5          # 每条最多几个候选项
MIN_CLARIFY_QUESTION_CHARS = 4   # 短于此视为无信息量（"是吗"、"怎么办"）
MAX_CLARIFY_QUESTION_CHARS = 60  # 超过则截断（模型偶发把整段解释塞进来）
MAX_CLARIFY_OPTION_CHARS = 20    # 候选项过长不适合做成按钮
MAX_CLARIFY_REASON_CHARS = 80    # "为什么追问"的说明

#: 首轮追问的最小问题长度（字符）。
#:
#: 极短输入（"你好"、"?"、"在吗"）更可能是误触或寒暄，
#: 对这类输入追问"请补充项目信息"只会显得机械。
#: 8 个汉字大致是"能表达一个完整意图"的下限。
CLARIFY_MIN_QUESTION_CHARS = 8

BASE_TOOL_NAMES = (
    # 追问工具：问题指向用户自身项目却缺关键信息时，先问清楚再检索。
    # 与检索类并列常驻，但 Agent 层**只在首轮、且无历史时**暴露它
    # （见 react_loop._resolve_tool_schemas）——避免多轮对话里反复追问。
    CLARIFY_TOOL_NAME,
    "search_knowledge_base",
    "search_knowledge_graph",
    "query_database",
    # 规则判定类工具：逐条比对招标文件条款与已知违规特征，并附法规依据原文。
    # 与三个检索类工具并列常驻暴露——它不产生外部请求、不依赖联网开关，
    # 且用户问「这条要求合规吗」时**必须先有它**才能给出可核对的结论
    # （此前 Agent 只能用检索拼原则性判断，见 docs/34）。
    "check_tender_compliance",
)
# 联网类工具：只在用户打开「联网搜索」开关时暴露给模型（见 react_loop._resolve_tool_schemas），
# 其结果也会被归入 done 帧的 web_sources（见 react_loop._record_tool_meta）。
# search_web=Tavily（英文强 + AI 摘要）；search_web_cn=智谱（中文站点 + 发布时间），两者互补并存。
WEB_TOOL_NAMES = ("search_web", "search_web_cn", "search_exa")

# 深度思考（按 provider 配置）
#
# ★ 语义在 2026-04 之后变了：从"换模型名"变成"开请求参数"
# --------------------------------------------------
# 原设计：`DEEPSEEK_THINKING_MODEL=deepseek-reasoner` 切到独立推理模型。
# 但 DeepSeek 发布 V4 后取消了独立推理模型名——现在只有 `deepseek-flash`
# 与 `deepseek-v4-pro`，两者都由请求参数 `{"thinking": {"type": "enabled"}}`
# 控制思考模式；旧名 `deepseek-reasoner` 将于 2026-07-24 15:59 UTC 退役。
#
# 故这里保留**两种**配置，语义不同（`resolve_thinking_model` 仍可用于显式
# 指定推理模型，例如本地 vLLM 上真的有一个独立推理模型时）：
#
#     DEEPSEEK_THINKING_MODEL    切到该模型名（可选，用于确有独立推理模型的部署）
#     DEEPSEEK_THINKING_ENABLED  开思考模式参数（推荐，DeepSeek 官方方式）
#
# 两者取"或"：任一为真即启用深度思考的模型侧能力。
THINKING_MODEL_ENV = {
    "deepseek": "DEEPSEEK_THINKING_MODEL",
    "zhipu": "ZHIPU_THINKING_MODEL",
}

#: provider -> "是否开思考模式"的环境变量名
THINKING_ENABLED_ENV = {
    "deepseek": "DEEPSEEK_THINKING_ENABLED",
    "zhipu": "ZHIPU_THINKING_ENABLED",
}

#: 默认是否开启思考模式。
#:
#: 取 `true` 与官方默认一致（DeepSeek 思考模式默认打开、effort 默认 high）。
#: 但本项目**显式控制**：普通请求不传 `thinking`，避免每次问答都付出
#: 思维链的延迟与 token 成本；只有用户打开「深度思考」开关时才传。
#: 若希望默认就带思考（例如全部场景都要最强推理），把
#: `DEEPSEEK_THINKING_ENABLED=true` 写进 `.env` 即可。
DEFAULT_THINKING_ENABLED = False


def _truthy(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


def resolve_thinking_model(provider: str = "") -> str:
    """取该 provider 显式配置的推理模型名；未配置返回空串。

    **注意**：DeepSeek 当前没有独立推理模型名，故这个通常为空——
    真正控制思考模式的是 `thinking_enabled_for(provider)`。
    保留本函数是为了支持"确有独立推理模型"的部署（如本地 vLLM
    上单独部署的 R1）。
    """
    name = (provider or os.getenv("LLM_PROVIDER", "") or "").strip().lower()
    env_key = THINKING_MODEL_ENV.get(name)
    return (os.getenv(env_key, "") if env_key else "").strip()


def thinking_enabled_for(provider: str = "") -> bool:
    """该 provider 是否启用思考模式（请求参数方式）。

    provider 未登记时回落到 `DEFAULT_THINKING_ENABLED`，这样新增 provider
    不必改 `THINKING_ENABLED_ENV` 也能通过环境变量控制（只是没有专属别名）。
    """
    name = (provider or os.getenv("LLM_PROVIDER", "") or "").strip().lower()
    env_key = THINKING_ENABLED_ENV.get(name)
    if not env_key:
        return DEFAULT_THINKING_ENABLED
    raw = os.getenv(env_key, "")
    if not raw.strip():
        return DEFAULT_THINKING_ENABLED
    return _truthy(raw)



# LLM 客户端缓存锁：多请求并发首用时只创建一份客户端（双检锁的外层保护）
llm_client_lock = threading.Lock()



# 来源条数上限（done 帧回填时生效）。
#
# 实测：多轮工具调用会累加来源，一次提问拿到 24~36 条联网来源
# （3 轮 × 每轮 8~12 条）。前端把 30 多个卡片全渲染出来用户读不完，
# 真正相关的前几条反而被淹没。
#
# 限条数时**按分数降序保留**，而不是截断前 N 条——后者会把后面轮次
# 检索到的、可能更相关的结果直接丢掉。
MAX_WEB_SOURCES = 10
MAX_LOCAL_SOURCES = 8

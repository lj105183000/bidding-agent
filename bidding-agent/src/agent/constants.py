"""Agent 模块共享常量。

所有 magic number 集中在此并附语义说明；改动前先确认上下游不受影响
（前端 SSE 协议见 docs/开发文档.md §6，测试断言见 tests/）。
"""

import threading

# --- 对话与工具循环 ---
MAX_HISTORY_ROUNDS = 5       # 对话历史保留轮数（1 轮 = user + assistant 两条）
MAX_TOOL_ROUNDS = 4          # ReAct 工具轮次上限（不含最终生成回合）
TOOL_TIMEOUT_SECONDS = 30    # 单个工具执行超时；超时按工具失败处理而非中断整个回答

# --- 流式输出 ---
STREAM_FLUSH_CHARS = 80      # 防泄漏缓冲阈值：攒够这么多字符且未检出工具格式才下发
STREAM_CHUNK_CHARS = 40      # token 小帧切分长度，避免前端一次收到大块
STREAM_CHUNK_DELAY = 0.02    # 帧间延时（秒），打字机效果并减轻前端渲染压力

# --- LLM ---
MAX_TOKENS_DEEP = 8192       # 深度思考输出预算（reasoning 计入 max_tokens）

# --- 工具名清单（SYSTEM_PROMPT 与来源分流共用，新增工具同步维护） ---
BASE_TOOL_NAMES = ("search_knowledge_base", "search_knowledge_graph", "query_database")
WEB_TOOL_NAMES = ("search_web", "search_exa")

# 深度思考模型配置键：provider -> .env 变量名（留空则回退提示词注入）
THINKING_MODEL_ENV = {
    "deepseek": "DEEPSEEK_THINKING_MODEL",
    "zhipu": "ZHIPU_THINKING_MODEL",
}

# LLM 客户端缓存锁：多请求并发首用时只创建一份客户端（双检锁的外层保护）
llm_client_lock = threading.Lock()

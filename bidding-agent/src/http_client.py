"""全项目统一的出站 HTTP 客户端。

用途
----
Tavily 搜索、Exa MCP，以及任何需要访问公网服务的模块共用同一份出站策略，
避免每个数据源各写一套 httpx 参数、各自踩代理、超时与连接池的坑。

设计原则
--------
- **单例**：连接池按进程复用，出口 IP 与 TLS 会话稳定，也不必每次问答重建连接。
- **trust_env=False**：不继承系统代理，行为与机器环境解耦（理由见下方常量区）。
- **显式超时**：连接 / 读 / 写 / 池四类超时分别设定，不依赖 httpx 的默认值。
- **可关闭**：FastAPI lifespan 退出时释放连接池，释放后可重新创建。

本模块只做连接管理，不含任何业务逻辑——搜索、图谱、数据库的调用留在各自模块里。
"""

import logging
import threading

import httpx

logger = logging.getLogger(__name__)

# --- 出站参数（数值为什么这样取） ---

# 对方站点与网关按 UA 做限流、排查问题时也能认出流量来自本服务
USER_AGENT = "bidding-qa-chatbot/1.0"

# 四类超时分开设：外网检索的响应可能很慢，读超时给足 60s；但**建连**必须快失败（10s），
# 否则一个不可达的域名会拖住整轮问答。写超时 30s 覆盖 MCP 初始化报文这类较大的请求体。
HTTP_TIMEOUT = httpx.Timeout(connect=10.0, read=60.0, write=30.0, pool=30.0)

# 池上限与工具层的 4 线程并行匹配：20 个连接够四路数据源并发 + 重试余量，
# 保活 10 条让连续问答复用连接，省掉重复的 TLS 握手。
HTTP_LIMITS = httpx.Limits(max_connections=20, max_keepalive_connections=10)

# 为什么 trust_env=False —— 这条最容易被"顺手改回去"，改动前先读完：
# Windows 上大量用户装了系统级代理软件，并把地址写进了 HTTP_PROXY / HTTPS_PROXY /
# ALL_PROXY。本项目要直连 Qdrant Cloud / Neo4j Aura / Tavily / DeepSeek 等公网服务，
# 一旦继承这些环境变量，流量会被送到代理端口：代理软件恰好关闭时表现为
# Connection refused (WinError 10061) 或 HTTPS CONNECT 失败，代理开启时又可能撞上
# TLS 证书校验错误。更麻烦的是它**只在部分同学的机器上复现**，排查成本极高。
# 关掉之后出站行为与机器环境解耦；确实需要代理时，请在构造处显式传 proxy=，
# 而不是把这个开关打开（打开等于把"是否走代理"交给每个用户的系统设置）。
TRUST_ENV = False

# --- 单例状态 ---
_client: httpx.Client | None = None
_async_client: httpx.AsyncClient | None = None

# 双检锁：并发首用只创建一份客户端，与 embedder / neo4j_client 等单例同一套路。
# 同步与异步各一把锁，避免两者初始化时互相阻塞。
_client_lock = threading.Lock()
_async_client_lock = threading.Lock()


def _build_client_kwargs() -> dict:
    """同步 / 异步客户端共用的构造参数。

    集中在这里是为了防止两条路径参数漂移——出站策略一旦只有一边改对，
    排查时会出现"同样的调用，异步那条就是慢/就是超时"的怪现象。
    """
    return {
        "trust_env": TRUST_ENV,
        "timeout": HTTP_TIMEOUT,
        "limits": HTTP_LIMITS,
        # 搜索站点与文档站常见 301/302：跟随跳转省掉每个调用方各自处理重定向
        "follow_redirects": True,
        "headers": {"User-Agent": USER_AGENT},
    }


def get_http_client() -> httpx.Client:
    """返回全局复用的 httpx.Client（首次调用时创建）。"""
    global _client
    if _client is None:
        with _client_lock:
            # 二次检查：等锁期间可能已有别的线程完成了创建
            if _client is None:
                _client = httpx.Client(**_build_client_kwargs())
                logger.info("已创建全局 httpx.Client")
    return _client


def get_async_http_client() -> httpx.AsyncClient:
    """返回全局复用的 httpx.AsyncClient（结构与同步版对称）。

    目前没有调用方；异步链路落地时直接复用同一个工厂，不必再补一份出站策略。
    """
    global _async_client
    if _async_client is None:
        with _async_client_lock:
            if _async_client is None:
                _async_client = httpx.AsyncClient(**_build_client_kwargs())
                logger.info("已创建全局 httpx.AsyncClient")
    return _async_client


def close_http_client() -> None:
    """关闭并丢弃全局同步客户端，供 FastAPI lifespan 退出时调用。

    幂等：没有已创建的实例时直接返回，重复调用只打一次日志。
    """
    global _client
    with _client_lock:
        if _client is None:
            return
        # 先摘引用再关闭：即使 close() 抛异常，模块也不会卡在一个已失效的实例上，
        # 后续 get_http_client() 仍能重新建立连接池
        client, _client = _client, None
        try:
            client.close()
        except Exception:
            logger.exception("关闭全局 httpx.Client 时出错（引用已丢弃，可重新创建）")
        else:
            logger.info("已关闭全局 httpx.Client")


async def aclose_http_client() -> None:
    """关闭并丢弃全局异步客户端。

    为什么单独一个函数而不是合并进 close_http_client()：AsyncClient 必须 await
    aclose() 才真正释放连接池，同步函数里做不到——合并的话异步池会一直挂着不释放。
    """
    global _async_client
    with _async_client_lock:
        if _async_client is None:
            return
        client, _async_client = _async_client, None
    try:
        # 锁外 await：关闭可能等待连接收回，不该把其他线程挡在初始化路径上
        await client.aclose()
    except Exception:
        logger.exception("关闭全局 httpx.AsyncClient 时出错（引用已丢弃，可重新创建）")
    else:
        logger.info("已关闭全局 httpx.AsyncClient")

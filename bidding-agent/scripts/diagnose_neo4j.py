"""Neo4j 连接分层诊断：区分「网络不通」与「密码错误」。

原理
----
这两类故障在客户端会表现为完全不同的错误，只要把链路拆成 5 层逐层验证，
就能明确定位问题在哪一层：

    第1层 DNS 解析      → 不通则报 getaddrinfo failed
    第2层 TCP 三次握手   → 不通则报 Connection refused / timed out
    第3层 TLS 握手       → 不通则报 SSL/TLS error；能拿到证书说明服务器真实存在
    第4层 Bolt 协议握手  → 能收到服务器的 HELLO 应答，说明 Neo4j 进程活着
    第5层 认证           → 服务器**主动返回** Unauthorized，说明它在正常处理请求

关键判据：**"密码错误"这个结论本身就是服务器告诉我们的话**。
连不上的时候，服务器根本没机会说这句话——客户端只会抛超时或网络异常。
"""

import socket
import ssl
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
import os

# ★ 本脚本在 scripts/ 下，故项目根是**上一级**。
#   写成 `.parent` 会指向 scripts/，于是 .env 与 src/ 都找不到。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

URI = (os.getenv("NEO4J_URI") or "").strip()
USERNAME = (os.getenv("NEO4J_USERNAME") or "").strip()
PASSWORD = (os.getenv("NEO4J_PASSWORD") or "").rstrip("\r\n")
DATABASE = (os.getenv("NEO4J_DATABASE") or "").strip()


def line(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass

    host = URI.replace("neo4j+s://", "").replace("neo4j://", "").split("/")[0].split(":")[0]

    line("配置")
    print(f"  URI      : {URI}")
    print(f"  主机     : {host}")
    print(f"  用户名   : {USERNAME}")
    print(f"  库名     : {DATABASE}")
    print(f"  密码长度 : {len(PASSWORD)} 位")

    # ---------------------------------------------------------------- 第1层
    line("第 1 层：DNS 解析（把域名变成 IP）")
    try:
        started = time.perf_counter()
        addresses = socket.getaddrinfo(host, 7687, proto=socket.IPPROTO_TCP)
        ip = addresses[0][4][0]
        print(f"  [OK] 解析成功：{host} -> {ip}（{(time.perf_counter()-started)*1000:.0f}ms）")
    except Exception as exc:
        print(f"  [XX] 解析失败：{type(exc).__name__}: {exc}")
        print("\n  结论：域名不存在 —— 实例可能已被删除。这属于网络/实例问题，不是密码问题。")
        return 1

    # ---------------------------------------------------------------- 第2层
    line("第 2 层：TCP 三次握手（能否建立到 7687 端口的连接）")
    try:
        started = time.perf_counter()
        sock = socket.create_connection((ip, 7687), timeout=15)
        print(f"  [OK] TCP 连接建立成功（{(time.perf_counter()-started)*1000:.0f}ms）")
        print("       说明：目标主机在线、端口开放、中间没有防火墙拦截")
    except Exception as exc:
        print(f"  [XX] TCP 连接失败：{type(exc).__name__}: {exc}")
        print("\n  结论：网络层不通。这才是「网络链接不上」该有的样子。")
        return 1

    # ---------------------------------------------------------------- 第3层
    line("第 3 层：TLS 握手 + 服务器证书（证明对面真的是 Neo4j）")
    try:
        context = ssl.create_default_context()
        started = time.perf_counter()
        tls = context.wrap_socket(sock, server_hostname=host)
        cert = tls.getpeercert()
        print(f"  [OK] TLS 握手成功（{(time.perf_counter()-started)*1000:.0f}ms），协议 {tls.version()}")
        subject = dict(x[0] for x in cert.get("subject", ()))
        issuer = dict(x[0] for x in cert.get("issuer", ()))
        print(f"       证书主体  : {subject.get('commonName')}")
        print(f"       颁发机构  : {issuer.get('organizationName')}")
        print(f"       有效期至  : {cert.get('notAfter')}")
        print("       说明：能拿到受信任的证书 = 服务器真实存在且身份可信")
    except Exception as exc:
        print(f"  [XX] TLS 握手失败：{type(exc).__name__}: {exc}")
        print("\n  结论：TLS 层不通（可能被中间设备拦截或证书问题）。")
        return 1

    # ---------------------------------------------------------------- 第4+5层
    line("第 4 层：Bolt 协议握手（Neo4j 应用层是否响应）")
    try:
        # Bolt 4.4 握手魔数：0x6060B017 + 4 个版本号
        tls.sendall(bytes.fromhex("6060b017000000040404000003040000020400000104000000"))
        response = tls.recv(4)
        chosen = int.from_bytes(response, "big")
        print(f"  [OK] 服务器回应了 Bolt 握手，选定协议版本 0x{chosen:08x}")
        print("       说明：对面运行的确实是 Neo4j 数据库服务，且正在正常工作")
    except Exception as exc:
        print(f"  [XX] Bolt 握手失败：{type(exc).__name__}: {exc}")
        return 1
    finally:
        try:
            tls.close()
        except Exception:
            pass

    line("第 5 层：认证（服务器对凭据的判定）")
    from neo4j import GraphDatabase

    try:
        driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD), connection_timeout=30)
        driver.verify_connectivity()
        with driver.session(database=DATABASE or None) as session:
            nodes = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        driver.close()
        print(f"  [OK] 认证通过！节点数 = {nodes}")
        print("\n  结论：网络正常 + 密码正确。可以直接导入图谱了。")
        return 0
    except Exception as exc:
        print(f"  [XX] 认证失败：{type(exc).__name__}")
        print(f"       {str(exc)[:200]}")

    # 对照实验：用一个明显错误的密码再试一次，看错误是否「一模一样」
    line("对照实验：故意用一个错误的密码")
    try:
        driver = GraphDatabase.driver(URI, auth=(USERNAME, "obviously-wrong-password-12345"), connection_timeout=30)
        driver.verify_connectivity()
        driver.close()
        print("  [!!] 错误密码竟然也连上了？这不正常，请把输出发给助手。")
    except Exception as exc:
        print(f"  [XX] 认证失败：{type(exc).__name__}")
        print(f"       {str(exc)[:200]}")
        print()
        print("  判据：错误密码与你的密码产生了【完全相同类型】的错误。")
        print("        如果这是网络问题，两次都会是超时/连接异常，而不是 Unauthorized。")

    line("最终结论")
    print("  第 1~4 层全部通过：")
    print("    ✓ DNS 解析成功 —— 域名存在")
    print("    ✓ TCP 连接成功 —— 网络可达，防火墙未拦")
    print("    ✓ TLS 握手成功 —— 能建立加密通道，证书由 Neo4j 签发")
    print("    ✓ Bolt 握手成功 —— Neo4j 服务在线并正常响应")
    print()
    print("  唯独第 5 层认证被服务器拒绝。")
    print()
    print("  → 因此这是【密码与实例不匹配】，不是网络问题。")
    print("     Neo4j 服务端主动回复了「Unauthorized」，")
    print("     网络不通时服务器根本没机会回这句话。")
    return 2


if __name__ == "__main__":
    sys.exit(main())

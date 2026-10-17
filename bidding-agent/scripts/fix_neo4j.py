"""一步修复 Neo4j 连接并导入知识图谱。

背景：Neo4j Aura 的实例密码是「创建时单独生成、只显示一次」的，
换实例（或重建实例）后必须同步更新 NEO4J_PASSWORD，否则报
    AuthError {neo4j_code: Neo.ClientError.Security.Unauthorized}
而这个报错不会提示"密码属于另一个实例"，很容易误判成网络问题。

本脚本把「验证密码 → 写入 .env → 导入图谱 → 复核」四步合成一条命令。

用法
----
    # 方式一：密码作为参数（会明文出现在命令历史里，仅本地临时用）
    python fix_neo4j.py --password "你的Aura密码"

    # 方式二：交互输入（推荐，不回显、不进历史）
    python fix_neo4j.py

    # 只测试连接，不改 .env、不导入
    python fix_neo4j.py --password "xxx" --test-only

    # 密码已在 .env 里改好了，只跑导入
    python fix_neo4j.py --skip-password
"""

import argparse
import getpass
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

# ★ 本脚本在 scripts/ 下，故项目根是**上一级**。
#   写成 `.parent` 会指向 scripts/，于是 .env 与 src/ 都找不到。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
IMPORT_SCRIPT = PROJECT_ROOT / "src" / "database" / "to_neo4j.py"


def _setup_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass


def read_env() -> dict:
    """读取当前 .env 的 Neo4j 配置（不覆盖已存在的环境变量）。"""
    load_dotenv(ENV_PATH)
    return {
        "uri": (os.getenv("NEO4J_URI") or "").strip(),
        "username": (os.getenv("NEO4J_USERNAME") or "").strip(),
        "password": (os.getenv("NEO4J_PASSWORD") or "").rstrip("\r\n"),
        "database": (os.getenv("NEO4J_DATABASE") or "").strip(),
    }


def test_connection(cfg: dict, password: str | None = None, timeout: int = 30) -> tuple[bool, str]:
    """尝试连接，返回 (是否成功, 说明文本)。"""
    pwd = (password if password is not None else cfg["password"]).rstrip("\r\n")
    database = cfg["database"] or None
    started = time.perf_counter()
    try:
        driver = GraphDatabase.driver(
            cfg["uri"], auth=(cfg["username"], pwd), connection_timeout=timeout
        )
        driver.verify_connectivity()
        with driver.session(database=database) as session:
            nodes = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            try:
                rels = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
            except Exception:
                rels = "?"
        driver.close()
        elapsed = time.perf_counter() - started
        return True, f"连接成功（{elapsed:.1f}s）｜现有节点 {nodes} 个、关系 {rels} 条"
    except Exception as exc:
        name = type(exc).__name__
        msg = str(exc)
        if "Unauthorized" in msg or "authentication failure" in msg:
            hint = (
                "认证失败：密码与该实例不匹配。\n"
                "     → Aura 的密码是每个实例创建时单独生成的，换实例必须换密码。\n"
                "     → 去 https://console.neo4j.io 打开实例，Download credentials 或 Reset password。"
            )
        elif "Cannot resolve" in msg or "NameResolution" in msg:
            hint = "域名解析失败：该实例 ID 可能已不存在或被删除。"
        elif "timed out" in msg.lower() or "Timeout" in name:
            hint = "连接超时：实例可能处于暂停状态（Aura 免费版闲置 3 天自动暂停），去控制台点 Resume。"
        else:
            hint = f"{name}: {msg[:200]}"
        return False, hint


def write_password(new_password: str) -> None:
    """把新密码写回 .env 的 NEO4J_PASSWORD 行（其余内容原样保留）。

    只替换该行，不重写整个文件：.env 里还有别人写的注释与其它密钥，
    整体重写会丢掉它们。密码一律不加引号——dotenv 会把引号当成密码的一部分。
    """
    text = ENV_PATH.read_text(encoding="utf-8")
    pattern = re.compile(r"(?m)^NEO4J_PASSWORD=.*$")
    if pattern.search(text):
        text = pattern.sub(f"NEO4J_PASSWORD={new_password}", text, count=1)
    else:
        text = text.rstrip("\n") + f"\nNEO4J_PASSWORD={new_password}\n"
    ENV_PATH.write_text(text, encoding="utf-8")


def run_import(with_clear: bool) -> int:
    """调用 to_neo4j.py 导入图谱（先清空，避免旧行号残留孤点）。"""
    cmd = [sys.executable, "-u", str(IMPORT_SCRIPT)]
    if with_clear:
        cmd.append("--clear")
    print(f"\n[fix] 开始导入知识图谱：{' '.join(cmd[-2:]) if with_clear else cmd[-1]}")
    return subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode


def main() -> int:
    _setup_console()
    parser = argparse.ArgumentParser(
        description="修复 Neo4j 连接并导入知识图谱",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--password", nargs="?", const="", help="Aura 密码；不带值则交互输入（不回显）")
    parser.add_argument("--test-only", action="store_true", help="只测试连接，不写 .env、不导入")
    parser.add_argument("--skip-password", action="store_true", help="跳过密码处理，直接用 .env 里现有的密码导入")
    parser.add_argument("--no-clear", action="store_true", help="导入时不清空图谱（默认会 --clear）")
    args = parser.parse_args()

    cfg = read_env()
    print("=" * 68)
    print("当前配置")
    print("=" * 68)
    print(f"  NEO4J_URI      = {cfg['uri'] or '(空)'}")
    print(f"  NEO4J_USERNAME = {cfg['username'] or '(空)'}")
    print(f"  NEO4J_DATABASE = {cfg['database'] or '(空)'}")
    if cfg["password"]:
        p = cfg["password"]
        print(f"  NEO4J_PASSWORD = 长度{len(p)}，前4位 {p[:4]}...，尾4位 ...{p[-4:]}")
    else:
        print("  NEO4J_PASSWORD = (空)")

    if not cfg["uri"] or not cfg["username"]:
        print("\n[fix] 缺少 NEO4J_URI / NEO4J_USERNAME，请先在 .env 里填好。")
        return 1

    # ---- 1. 确定候选密码 ----
    candidate: str | None = None
    if not args.skip_password:
        if args.password == "":  # 裸 --password
            candidate = getpass.getpass("请输入 Aura 密码（不回显，直接回车则跳过）：").strip()
            if not candidate:
                candidate = None
        elif args.password:
            candidate = args.password.strip()

    # ---- 2. 测试 ----
    print()
    print("=" * 68)
    if candidate:
        print("测试新密码")
        print("=" * 68)
        ok, message = test_connection(cfg, password=candidate)
        print(f"  → {message}")
        if ok:
            if args.test_only:
                print("\n[fix] --test-only：连接正常，未修改 .env、未导入。")
                return 0
            write_password(candidate)
            print(f"\n[fix] 已把新密码写入 {ENV_PATH.name}")
            cfg["password"] = candidate
        else:
            print("\n[fix] 新密码也不通，未修改 .env。请核对控制台里的凭据。")
            return 1
    else:
        print("测试 .env 中的现有密码")
        print("=" * 68)
        ok, message = test_connection(cfg)
        print(f"  → {message}")
        if not ok:
            print("\n[fix] 现有密码不可用。请带 --password 传入新密码，或直接运行本脚本交互输入。")
            return 1
        if args.test_only:
            print("\n[fix] --test-only：连接正常。")
            return 0

    # ---- 3. 导入 ----
    code = run_import(with_clear=not args.no_clear)
    print()
    print("=" * 68)
    if code == 0:
        print("完成！知识图谱已导入。")
        print("验证：重启后端，然后提问涉及企业/标的物关系的问题，")
        print("      回答下方来源会标注「数据来源：知识图谱」。")
    else:
        print(f"导入失败，退出码 {code}。请把上面的日志发给助手排查。")
    print("=" * 68)
    return code


if __name__ == "__main__":
    sys.exit(main())

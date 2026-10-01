"""运行看板：feedback 汇总 + 会话统计 + 检索缓存统计（受限口径）。

缓存统计的口径局限必须如实标注：rag_pipeline 的缓存与命中账本只存在于本进程
内存，本脚本是独立进程，读不到长驻 API 服务的账本（打印出来接近全 0）。
它只作"管线通没通"的冒烟验证，真实命中率要在 API 服务进程内观察。

用法::

    python eval/dashboard.py                 # 默认最近 7 天
    python eval/dashboard.py --days 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.common import setup_stdio
from src.database.postgresql_client import PostgresClient
from src.logging_config import setup_logging


def main() -> int:
    setup_stdio()
    parser = argparse.ArgumentParser(description="运行看板（feedback / 会话 / 缓存统计）")
    parser.add_argument("--days", type=int, default=7, help="反馈趋势统计窗口（天）")
    args = parser.parse_args()

    setup_logging()
    try:
        pg = PostgresClient()
    except ValueError as exc:
        # 构造期就抛 ValueError 是懒连接设计的另一半：连接留给 health()，但 POSTGRES_*
        # 配置缺失不等到连库才暴露。这里只拦 ValueError——构造不碰网络，其余异常是
        # 真 bug，应当响亮崩掉而不是被吞成"数据库不可用"误导排障方向。
        # 异常文案本身是中文脱敏的（不含主机名/密钥），与 health() 的错误同口径可直接展示
        print(f"PostgreSQL 不可用：{exc}")
        return 1
    health = pg.health()
    if not health["ok"]:
        # health() 的错误文案已脱敏（不含主机名/端口/驱动报文），可直接展示
        print(f"PostgreSQL 不可用：{health['error']}")
        return 1

    fb = pg.feedback_summary(days=args.days)
    print("===== 用户反馈汇总 =====")
    if not fb:
        print("（无数据或查询失败）")
    else:
        total = fb.get("total", 0)
        up, down = fb.get("up", 0), fb.get("down", 0)
        print(f"总反馈 {total} | 赞 {up}（{up / total:.0%}）| 踩 {down}（{down / total:.0%}）" if total else "总反馈 0")
        for row in fb.get("daily", []):
            print(f"  {row['day']}  {'█' * min(row['cnt'], 40)} {row['cnt']}")

    cs = pg.conversation_stats()
    print("\n===== 会话统计 =====")
    if not cs:
        print("（无数据或查询失败）")
    else:
        print(f"会话 {cs.get('conversations', 0)} 个 | 消息 {cs.get('messages', 0)} 条")

    # 延迟导入：src.rag 会连带加载 torch 等重依赖，只有走到这一段才付这个代价
    from src.rag import cache_info

    ci = cache_info()
    print("\n===== 检索缓存统计（受限口径）=====")
    print(f"缓存条目 {ci.get('size', 0)} | 命中 {ci.get('hits', 0)}")
    print("说明：以上为脚本本进程数据，仅作冒烟验证；真实命中率请在长驻 API 服务进程内观察。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

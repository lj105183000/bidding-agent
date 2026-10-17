"""P0 最终方案：从法规库**程序化生成**全覆盖问答语料。

★ 为什么不继续手工逐条补
--------------------
手工补了两轮（37 条），切题率 38% → 77%，但：
    · 覆盖靠人想，必然遗漏（法规库有 180 条）
    · 每条都要人工核对条号，效率低且易错
    · 新法规入库后又要补一轮

而 `RegulationArticle` 自带 `topics` 标签（如 `('保证金','投标保证金','比例上限')`），
**可以按标签程序化生成**：每条 (条文, 标签) 生成一条问答。

★ 为什么这样**零编造风险**
------------------------
答案 100% 取自 `article.text` 原文，问题由标签与法规名拼成——
全程没有人工写入的法律内容。且引用自检能保证条号真实存在。

★ 与手工补语料的关系
------------------
手工那 37 条**质量更高**（问题更口语化、更像用户真实问法、带对比说明），
故保留；本脚本生成的是**广度覆盖**，两者互补。

为避免重复，生成时会跳过"与已有语料问题高度相似"的条目。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(r"C:\Users\10518\bidding-agent\bidding-agent")
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from src.constants import regulations as R  # noqa: E402

#: 标签太泛，单独成题没有信息量
SKIP_TOPICS = frozenset({
    "时限", "信息公开", "基本原则", "法律责任", "违法行为",
    "招标方式", "异议投诉",
})

#: 问题模板：把 (法规名, 标签) 组织成自然问法
TEMPLATES = (
    "{law}对「{topic}」是怎么规定的？",
    "关于{topic}，{law}有什么要求？",
    "{topic}方面，{law}的规定是什么？",
)


def main() -> int:
    cur = pd.read_excel(ROOT / "data" / "qa.xlsx")
    existing_qs = [str(x).strip() for x in cur.iloc[:, 0].tolist()]

    rows: list[tuple[str, str]] = []
    seen_q: set[str] = set()
    for art in R.REGULATIONS:
        for topic in art.topics:
            if topic in SKIP_TOPICS or len(topic) < 3:
                continue
            # 轮流用模板，避免 180 条全是同一个句式
            tpl = TEMPLATES[len(rows) % len(TEMPLATES)]
            q = tpl.format(law=f"《{art.law_short}》", topic=topic)
            if q in seen_q:
                continue
            seen_q.add(q)
            ans = (
                f"依据《{art.law_short}》{art.article}（{art.level}，"
                f"施行日期 {art.effective}）：\n\n{str(art.text).strip()}"
            )
            rows.append((q, ans))

    print("=" * 96)
    print("程序化生成全覆盖语料")
    print("=" * 96)
    print(f"  法规库条文: {len(R.REGULATIONS)} 条")
    print(f"  生成问答:   {len(rows)} 条（按 (条文, 标签) 展开，已去重）")
    print(f"  现有语料:   {len(cur)} 条")

    # 自检：答案里的「《法规》第X条」必须在库里
    #
    # ⚠️ 匹配时**全称与简称都要认**。第一版只用 `law_short`（简称，如「招标投标法」）
    # 建索引，而答案用的是 `law`（全称「中华人民共和国招标投标法」），
    # 于是 3 条真实存在的引用被误报成"无效"——**自检本身成了误报源**。
    import re

    def law_keys(a) -> set[str]:
        return {str(a.law).strip(), str(a.law_short).strip()}

    ART: set[tuple[str, str]] = set()
    for a in R.REGULATIONS:
        for key in law_keys(a):
            ART.add((key, a.article))

    CITE = re.compile(r"《([^》]+)》(第[一二三四五六七八九十百零\d]+条)")
    bad = [(q, l, n) for q, a in rows for l, n in CITE.findall(a) if (l, n) not in ART]
    print(f"  引用自检: {len(bad)} 处无效引用")
    if bad:
        for b in bad[:5]:
            print(f"    ❌ {b}")
        raise SystemExit("存在无效引用，已中止")

    # 去重：与现有语料问题完全相同或高度相似的跳过
    # （用前 8 字做粗判——中文问句开头 8 字足以区分主题）
    def head(s: str) -> str:
        return re.sub(r"\s+", "", s)[:8]

    existing_heads = {head(q) for q in existing_qs}
    kept = [(q, a) for q, a in rows if head(q) not in existing_heads]
    print(f"  与现有语料冲突跳过: {len(rows) - len(kept)} 条")
    print(f"  实际新增: {len(kept)} 条")

    merged = pd.concat(
        [cur, pd.DataFrame(kept, columns=["问", "答"])], ignore_index=True
    ).drop_duplicates(subset=["问"], keep="first").reset_index(drop=True)
    merged.to_excel(ROOT / "data" / "qa.xlsx", index=False)
    print(f"\n  语料已写回 data/qa.xlsx：{len(cur)} → {len(merged)} 条")
    print()
    print("  样例:")
    for q, a in kept[:3]:
        print(f"    Q: {q}")
        print(f"    A: {a[:90]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

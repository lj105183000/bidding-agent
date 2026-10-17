"""技能加载与匹配：SKILL.md 是上下文注入（无额外工具往返）。

★ 触发方式：显式声明，不再从 description 机械截取
----------------------------------------------
原实现用「description 中破折号/冒号前半段」当触发短语。实测有两个问题：

    '深度分析招投标问题'    → 匹配不到 bid-analysis（别名是「投标文件分析」）
    '帮我看看这个招标文件'   → 匹配不到（同上）

即用户稍微换个说法就命中不了，而**技能没生效是静默的**——
模型不会说"我没拿到技能"，它只是按普通方式回答，没人会发现。

故改为在 frontmatter 里**显式声明 `triggers`**：

    ---
    name: bid-analysis
    description: 投标文件分析——按资质、业绩、报价、偏离项四个维度逐项核对
    triggers: 投标文件分析, 投标分析, 分析投标, 招标文件分析
    ---

`description` 保留展示用途；`triggers` 专管匹配。两者分开的好处是
**改触发词不用动描述文案**，反之亦然。

★ 兼容性
-------
未声明 `triggers` 时回落到原行为（从 description 截取别名），
故新增技能不写 triggers 也能工作，不会因为忘写而失效。

★ 为什么不做 LLM 选择技能
-----------------------
LLM 选择更灵活，但要多一次调用（2~5 秒）。而实测技能只有 2 个、
触发场景明确，显式触发词已够用。等技能数量到十几个、
触发词开始互相干扰时再考虑——那时**才**有做 LLM 选择的收益。
"""

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"

# 指代触发短语：用户不点名技能，而用泛指说法引用（如「按技能来分析」）
_REFERENCE_PATTERNS = ("按技能", "按照技能", "用技能", "使用技能", "技能步骤")


def _parse_alias(description: str) -> str:
    """从 description 截取别名。**仅用于未声明 triggers 的旧技能**（兼容路径）。"""
    for sep in ("——", "—", "：", ":", "-"):
        if sep in description:
            return description.split(sep, 1)[0].strip()
    return description.strip()


def _parse_triggers(raw: str) -> list[str]:
    """解析 `triggers: a, b, c` 形式的触发词列表。

    分隔符同时接受中文逗号与顿号——两种写法在中文文档里都常见，
    只认一种会让用户写了却不生效（而这是静默的）。
    """
    if not raw:
        return []
    parts = re.split(r"[,，、;；]", raw)
    return [p.strip() for p in parts if p.strip()]


def _load_skills(skills_dir=None) -> list:
    """扫描 src/skills/<name>/SKILL.md，返回 skill 字典列表。

    单个技能文件损坏只记日志跳过，不中断其余技能加载——
    一个 SKILL.md 写错不该让所有技能一起失效。
    """
    root = Path(skills_dir) if skills_dir else _SKILLS_DIR
    skills = []
    if not root.is_dir():
        return skills
    for md in sorted(root.glob("*/SKILL.md")):
        try:
            text = md.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("技能文件读取失败 %s: %s", md, exc)
            continue
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", text, re.S)
        if not match:
            logger.warning("技能缺少 frontmatter，已跳过: %s", md)
            continue
        header, body = match.group(1), match.group(2).strip()
        fields = {}
        for line in header.splitlines():
            if ":" in line:
                key, _, value = line.partition(":")
                fields[key.strip().lower()] = value.strip()
        name = fields.get("name", md.parent.name)
        description = fields.get("description", "")
        if not name or not description:
            logger.warning("技能缺少 name/description，已跳过: %s", md)
            continue

        triggers = _parse_triggers(fields.get("triggers", ""))
        if not triggers:
            # 兼容路径：旧技能没写 triggers 时用 description 别名
            alias = _parse_alias(description)
            triggers = [alias] if alias else []

        skills.append({
            "name": name,
            "description": description,
            "triggers": triggers,
            # 保留 alias 键：既有测试与调用方依赖它（取首个触发词）
            "alias": triggers[0] if triggers else "",
            "body": body,
        })
    return skills


def _match_skills(question: str, skills=None) -> list:
    """三层匹配：点名（name）> 触发词（triggers）> 泛指指代。

    ★ 泛指时只回落到"全部技能"，且**仅在没有任何技能被点名/触发时**。
    原实现在遍历中无条件 append 全部技能，导致
    「按技能分析投标文件」这种"既泛指又点名"的问法会把无关技能也注入。
    """
    q = (question or "").strip()
    if not q:
        return []
    loaded = skills if skills is not None else _load_skills()

    matched = []
    for skill in loaded:
        if skill["name"] and skill["name"] in q:
            matched.append(skill)
            continue
        if any(t and t in q for t in skill.get("triggers") or []):
            matched.append(skill)
    if matched:
        return matched

    # 没点名也没触发任何技能时，才考虑"泛指"回落
    if any(pattern in q for pattern in _REFERENCE_PATTERNS):
        return list(loaded)
    return []


def list_skills(skills=None) -> list[dict]:
    """列出技能名与触发词（供诊断与前端展示"有哪些技能可用"）。

    为什么需要：技能是隐式生效的，出问题时（该触发的没触发）
    需要能直接看到"注册了哪些技能、各自的触发词是什么"。
    """
    loaded = skills if skills is not None else _load_skills()
    return [
        {"name": s["name"], "triggers": list(s.get("triggers") or []),
         "description": s["description"]}
        for s in loaded
    ]

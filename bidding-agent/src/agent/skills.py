"""技能加载与匹配：SKILL.md 是上下文注入（无额外工具往返）。

约定见 docs/开发文档.md §8.4：frontmatter 声明 name 与 description，
alias 取 description 中破折号/冒号前半段作为中文触发短语。
"""

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"

# 指代触发短语：用户不点名技能，而用泛指说法引用（如「按技能来分析」）
_REFERENCE_PATTERNS = ("按技能", "按照技能", "用技能", "使用技能", "技能步骤")


def _parse_alias(description: str) -> str:
    """alias = description 中破折号/冒号前半段，作为中文触发短语。"""
    for sep in ("——", "—", "：", ":", "-"):
        if sep in description:
            return description.split(sep, 1)[0].strip()
    return description.strip()


def _load_skills(skills_dir=None) -> list:
    """扫描 src/skills/<name>/SKILL.md，返回 [{name, description, alias, body}]。

    单个技能文件损坏只记日志跳过，不中断其余技能加载。
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
        skills.append({
            "name": name,
            "description": description,
            "alias": _parse_alias(description),
            "body": body,
        })
    return skills


def _match_skills(question: str, skills=None) -> list:
    """三层匹配：点名（name）> 别名（alias）> 指代短语。命中任意层即注入该技能。"""
    q = (question or "").strip()
    if not q:
        return []
    loaded = skills if skills is not None else _load_skills()
    matched = []
    for skill in loaded:
        if skill["name"] and skill["name"] in q:
            matched.append(skill)
            continue
        if skill["alias"] and skill["alias"] in q:
            matched.append(skill)
            continue
        if any(pattern in q for pattern in _REFERENCE_PATTERNS):
            matched.append(skill)
    return matched

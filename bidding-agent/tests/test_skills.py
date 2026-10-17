"""Skills 触发机制单测。

★ 为什么重做触发机制（实测依据）
------------------------------
原实现用「description 中破折号前半段」当触发短语，实测两个问题：

    '深度分析招投标问题'     → 匹配不到 bid-analysis（别名是「投标文件分析」）
    '帮我看看这个招标文件'    → 匹配不到（同上）

即用户稍微换个说法就命中不了——而**技能没生效是静默的**：
模型不会说"我没拿到技能"，它只是按普通方式回答，没人会发现。

故改为 frontmatter 显式声明 `triggers`，并保留旧行为的兼容回落。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent import skills as S  # noqa: E402


def _write_skill(root: Path, name: str, frontmatter: str, body: str = "步骤内容"):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(
        f"---\n{frontmatter}\n---\n{body}\n", encoding="utf-8"
    )


@pytest.fixture
def skills_dir(tmp_path):
    """临时技能目录，避免依赖仓库里真实的 SKILL.md（它们会被改动）。"""
    _write_skill(
        tmp_path, "alpha",
        "name: alpha\ntriggers: 甲触发, 甲备选\ndescription: 甲技能——做甲事",
    )
    _write_skill(
        tmp_path, "beta",
        "name: beta\ntriggers: 乙触发\ndescription: 乙技能——做乙事",
    )
    return tmp_path


class TestLoadSkills:
    def test_loads_triggers_from_frontmatter(self, skills_dir):
        loaded = S._load_skills(skills_dir)
        by_name = {s["name"]: s for s in loaded}
        assert by_name["alpha"]["triggers"] == ["甲触发", "甲备选"]
        assert by_name["beta"]["triggers"] == ["乙触发"]

    def test_accepts_chinese_separators(self, tmp_path):
        """★ 中文逗号与顿号都要认。

        只认半角逗号时，用户写了中文逗号会**静默失效**——
        技能加载成功但触发词为空，表现为"技能不生效"，很难查。
        """
        _write_skill(tmp_path, "x", "name: x\ntriggers: 甲，乙、丙;丁\ndescription: d")
        loaded = S._load_skills(tmp_path)
        assert loaded[0]["triggers"] == ["甲", "乙", "丙", "丁"]

    def test_keeps_alias_for_backward_compat(self, skills_dir):
        """`alias` 键保留（既有调用方与测试依赖它），取首个触发词。"""
        loaded = S._load_skills(skills_dir)
        by_name = {s["name"]: s for s in loaded}
        assert by_name["alpha"]["alias"] == "甲触发"

    def test_falls_back_to_description_when_no_triggers(self, tmp_path):
        """★ 兼容路径：旧技能没写 triggers 时，仍从 description 截取别名。

        不做这个回落的话，**已有的 SKILL.md 会因为忘写 triggers 而全部失效**，
        且失效是静默的。
        """
        _write_skill(tmp_path, "old", "name: old\ndescription: 旧式技能——按描述截取")
        loaded = S._load_skills(tmp_path)
        assert loaded[0]["triggers"] == ["旧式技能"]
        assert loaded[0]["alias"] == "旧式技能"

    def test_skips_skill_without_frontmatter(self, tmp_path):
        d = tmp_path / "broken"
        d.mkdir()
        (d / "SKILL.md").write_text("没有 frontmatter 的内容", encoding="utf-8")
        assert S._load_skills(tmp_path) == []

    def test_skips_skill_missing_description(self, tmp_path):
        """缺 description 会被跳过。

        注意 `name` 缺失**不算错**——它会回落到目录名（既有设计），
        故这里只测 description 缺失这一种真正会跳过的情况。
        """
        _write_skill(tmp_path, "no-desc", "name: no-desc")
        assert S._load_skills(tmp_path) == []

    def test_falls_back_to_dirname_when_name_missing(self, tmp_path):
        """`name` 缺失时用目录名——这是既有的容错设计，不该被当成错误。"""
        _write_skill(tmp_path, "from-dir", "description: 只有描述")
        loaded = S._load_skills(tmp_path)
        assert [s["name"] for s in loaded] == ["from-dir"]

    def test_one_broken_skill_does_not_break_others(self, tmp_path):
        """一个 SKILL.md 写错不该让所有技能一起失效。"""
        _write_skill(tmp_path, "good", "name: good\ntriggers: 好\ndescription: d")
        bad = tmp_path / "bad"
        bad.mkdir()
        (bad / "SKILL.md").write_text("坏的", encoding="utf-8")
        loaded = S._load_skills(tmp_path)
        assert [s["name"] for s in loaded] == ["good"]

    def test_missing_dir_returns_empty(self, tmp_path):
        assert S._load_skills(tmp_path / "不存在") == []


class TestMatchSkills:
    def test_matches_by_name(self, skills_dir):
        loaded = S._load_skills(skills_dir)
        assert [s["name"] for s in S._match_skills("用 alpha 做点事", loaded)] == ["alpha"]

    def test_matches_by_trigger(self, skills_dir):
        loaded = S._load_skills(skills_dir)
        assert [s["name"] for s in S._match_skills("帮我甲触发一下", loaded)] == ["alpha"]

    def test_matches_any_of_multiple_triggers(self, skills_dir):
        loaded = S._load_skills(skills_dir)
        assert [s["name"] for s in S._match_skills("甲备选", loaded)] == ["alpha"]

    def test_no_match_returns_empty(self, skills_dir):
        loaded = S._load_skills(skills_dir)
        assert S._match_skills("完全无关的问题", loaded) == []

    def test_empty_question(self, skills_dir):
        loaded = S._load_skills(skills_dir)
        assert S._match_skills("", loaded) == []
        assert S._match_skills("   ", loaded) == []

    def test_reference_phrase_loads_all(self, skills_dir):
        """泛指时加载全部——用户说了「按技能来」但没点名，全给是合理的。"""
        loaded = S._load_skills(skills_dir)
        got = [s["name"] for s in S._match_skills("按技能来分析", loaded)]
        assert got == ["alpha", "beta"]

    def test_named_skill_wins_over_reference_phrase(self, skills_dir):
        """★ 回归哨兵：「按技能分析投标文件」这类"既泛指又点名"的问法，
        不该把无关技能也注入。

        原实现在遍历中无条件 append 全部技能，导致点名时也全给——
        无关技能的步骤混进 system 会干扰模型。
        """
        loaded = S._load_skills(skills_dir)
        got = [s["name"] for s in S._match_skills("按技能甲触发一下", loaded)]
        assert got == ["alpha"], f"不该注入无关技能：{got}"

    def test_multiple_skills_can_match(self, skills_dir):
        loaded = S._load_skills(skills_dir)
        got = [s["name"] for s in S._match_skills("甲触发和乙触发都要", loaded)]
        assert got == ["alpha", "beta"]

    def test_empty_triggers_do_not_match_everything(self, tmp_path):
        """★ 触发词为空时不能匹配所有问题。

        若用 `"" in q`（恒为 True）判断，一个 triggers 为空的技能会
        在**每次**问答里被注入。
        """
        _write_skill(tmp_path, "empty", "name: empty\ntriggers: \ndescription: ——")
        loaded = S._load_skills(tmp_path)
        assert S._match_skills("随便什么问题", loaded) == []


class TestListSkills:
    def test_lists_name_triggers_description(self, skills_dir):
        items = S.list_skills(S._load_skills(skills_dir))
        assert {i["name"] for i in items} == {"alpha", "beta"}
        alpha = next(i for i in items if i["name"] == "alpha")
        assert alpha["triggers"] == ["甲触发", "甲备选"]
        assert "description" in alpha

    def test_empty_dir(self, tmp_path):
        assert S.list_skills(S._load_skills(tmp_path)) == []


class TestRealSkills:
    """对仓库里真实 SKILL.md 的契约检查。

    这些文件是**数据**，改坏了不会让代码报错，只会让技能静默失效。
    """

    #: 本次新增的 3 个业务 skill，与各自的触发场景
    #:
    #: 为什么逐个钉住触发词：技能是**隐式生效**的——
    #: 触发词写漏了不会报错，只是用户换个说法就命中不了（`docs/30` 记的第 7 类缺陷）。
    NEW_SKILLS = {
        "regulation-citation-audit": "这个招标文件要求本市注册满三年，合规吗？",
        "legal-update-monitor": "2026 年最新出台了哪些招投标法规？",
        "competition-landscape": "安徽智能化工程主要有哪些供应商在竞争？",
    }

    def test_real_skills_load_with_triggers(self):
        loaded = S._load_skills()
        assert loaded, "仓库里至少应有一个技能"
        for s in loaded:
            assert s["triggers"], f"{s['name']} 没有触发词，将永远无法被触发"

    def test_real_skills_have_expected_names(self):
        names = {s["name"] for s in S._load_skills()}
        assert {"bid-analysis", "debug-answer"} <= names

    def test_new_business_skills_present(self):
        names = {s["name"] for s in S._load_skills()}
        for name in self.NEW_SKILLS:
            assert name in names, f"缺少技能 {name}"

    @pytest.mark.parametrize("skill_name,question", sorted(NEW_SKILLS.items()))
    def test_new_skill_is_triggered(self, skill_name, question):
        loaded = S._load_skills()
        got = [s["name"] for s in S._match_skills(question, loaded)]
        assert skill_name in got, f"{question} 应触发 {skill_name}，实际 {got}"

    def test_new_skills_have_multiple_triggers(self):
        """★ 触发词不能只有一个。

        用户问法多样，单个触发词意味着绝大多数说法都命中不了。
        实测教训：`bid-analysis` 原只有「投标文件分析」一个说法可用，
        「帮我看看这个招标文件」就命中不了。
        """
        loaded = {s["name"]: s for s in S._load_skills()}
        for name in self.NEW_SKILLS:
            assert len(loaded[name]["triggers"]) >= 5, (
                f"{name} 触发词只有 {len(loaded[name]['triggers'])} 个，太少"
            )

    def test_new_skills_have_substantive_body(self):
        """技能正文必须有实质步骤——空壳技能只占 prompt 预算。

        门槛 300 字：3 个新技能的正文都在 700~1200 字，
        低于 300 字基本说明只写了个标题。
        """
        loaded = {s["name"]: s for s in S._load_skills()}
        for name in self.NEW_SKILLS:
            body = loaded[name]["body"]
            assert len(body) >= 300, f"{name} 正文仅 {len(body)} 字，过于单薄"

    def test_bid_analysis_triggered_by_common_phrasings(self):
        """★ 实测中命中不了的两种说法，现在必须命中。"""
        loaded = S._load_skills()
        for q in ("帮我看看这个招标文件", "这份投标文件有什么风险"):
            got = [s["name"] for s in S._match_skills(q, loaded)]
            assert "bid-analysis" in got, f"{q} 应触发 bid-analysis，实际 {got}"

    def test_unrelated_question_triggers_nothing(self):
        """回归哨兵：技能不该被无关问题激活——
        注入无关技能的步骤会干扰正常回答。"""
        loaded = S._load_skills()
        assert S._match_skills("投标保证金的退还时限是多久？", loaded) == []

    def test_no_trigger_word_collision_across_skills(self):
        """★ 触发词不应跨技能重复。

        重复会让一个问题同时激活两个技能，注入两份步骤——
        模型要在冲突的流程间选，反而更容易乱。
        （技能变多后这条会越来越重要。）
        """
        loaded = S._load_skills()
        owner: dict[str, str] = {}
        collisions: list[str] = []
        for s in loaded:
            for t in s["triggers"]:
                if t in owner and owner[t] != s["name"]:
                    collisions.append(f"{t!r}: {owner[t]} 与 {s['name']}")
                owner[t] = s["name"]
        assert not collisions, f"触发词冲突：{collisions}"

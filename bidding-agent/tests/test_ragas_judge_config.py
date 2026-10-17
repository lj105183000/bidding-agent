"""ragas 裁判 LLM 的配置测试（避免自评偏差）。

★ 背景：这不是"换个模型名"那么简单
============================
原实现的裁判是 **DeepSeek**，而**被评测的生成方也是 DeepSeek**
（`LLM_PROVIDER=deepseek`）——裁判与考生同源，属典型的**自评偏差**
（self-preference bias）：同族模型倾向给同族输出更高分，指标系统性偏高。

换成智谱 GLM 后裁判与生成不同家，偏差显著减小。本文件锁住三件事：

    ① 裁判走智谱端点、默认用 GLM
    ② **同源自检**能发现"裁判与生成又同家"（用户改 `LLM_PROVIDER` 就会发生）
    ③ 模型名可被环境变量覆盖（模型会下线，硬编码早晚会坏）

★ 不真调 API
==========
本文件只验证**配置解析**与**构建出的对象属性**，不调真实模型——
测试不该消耗额度也不该依赖网络。真调验证在
`logs/verify_ragas_judge.py` 里单独做过。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "eval"
for p in (str(ROOT), str(EVAL_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import ragas_eval as R  # noqa: E402


class TestJudgeUsesZhipu:
    """① 裁判必须走智谱，不能再用 DeepSeek（与生成方分离才能降偏差）。"""

    def test_default_model_is_glm(self):
        assert R.DEFAULT_JUDGE_MODEL == "glm-5.3-flash"

    def test_builds_against_zhipu_endpoint(self):
        from src.clients.zai_client import ZHIPU_BASE_URL

        llm = R.build_evaluator_llm()
        base = str(getattr(llm, "openai_api_base", "") or "")
        assert "bigmodel.cn" in base or base == ZHIPU_BASE_URL, (
            f"裁判必须走智谱端点，实得 {base!r}——"
            f"若仍指向 DeepSeek 就是自评偏差"
        )

    def test_model_name_matches_default(self, monkeypatch):
        monkeypatch.delenv(R.ENV_JUDGE_MODEL, raising=False)
        assert R.build_evaluator_llm().model_name == R.DEFAULT_JUDGE_MODEL

    def test_temperature_zero(self):
        """裁判要确定性：temperature=0 才有可复现的评分。"""
        llm = R.build_evaluator_llm()
        assert llm.temperature == 0.0

    def test_both_http_clients_pinned(self):
        """★ 两个客户端都要钉死 `trust_env=False`。

        ragas 只走**异步**路径，而 `ChatOpenAI` 的异步用的是独立的
        `http_async_client` 字段。只配同步客户端的话，真实链路会回落到
        `trust_env=True` 的默认客户端，代理残留场景（WinError 10061）照样炸。
        （文件头把这条列为"红线"，故用测试锁住。）
        """
        llm = R.build_evaluator_llm()
        assert llm.http_client is not None, "同步客户端未注入"
        assert llm.http_async_client is not None, "异步客户端未注入（ragas 实际走这条）"


class TestBiasSelfCheck:
    """② 同源自检——避免"换了模型但偏差又回来了"。

    换成 GLM 只在"生成方是 DeepSeek"时消除偏差。用户若把 `LLM_PROVIDER`
    改成 `zhipu`，裁判与生成**又变成同家**，而**不会有任何提示**，
    指标悄悄变得不可信。故显式检查。
    """

    def test_detects_same_provider(self, monkeypatch):
        from src.config import settings

        monkeypatch.setattr(settings, "llm_provider", "zhipu", raising=False)
        same, gen, judge = R.judge_provider_matches_generator()
        assert same is True, "生成方也是 zhipu 时必须报同源"
        assert gen == "zhipu" and judge == "zhipu"

    def test_detects_different_provider(self, monkeypatch):
        from src.config import settings

        monkeypatch.setattr(settings, "llm_provider", "deepseek", raising=False)
        same, gen, judge = R.judge_provider_matches_generator()
        assert same is False, "生成 deepseek / 裁判 zhipu 属不同源"
        assert gen == "deepseek" and judge == "zhipu"

    def test_case_insensitive_and_stripped(self, monkeypatch):
        """`.env` 是手写的，`" Zhipu "` 这种写法不该漏判。"""
        from src.config import settings

        monkeypatch.setattr(settings, "llm_provider", "  Zhipu  ", raising=False)
        same, _gen, _judge = R.judge_provider_matches_generator()
        assert same is True

    def test_returns_generator_and_judge_names(self, monkeypatch):
        """返回值要能直接用于提示文案（两个名字都在）。"""
        from src.config import settings

        monkeypatch.setattr(settings, "llm_provider", "deepseek", raising=False)
        result = R.judge_provider_matches_generator()
        assert isinstance(result, tuple) and len(result) == 3
        assert all(isinstance(x, str) for x in result[1:])


class TestModelNameOverridable:
    """③ 模型名可覆盖（模型会下线，硬编码早晚会坏）。

    本项目刚吃过这个亏：DeepSeek 的 `deepseek-chat` 退役、独立推理模型名被取消。
    故裁判模型名不能硬编码到代码里。
    """

    def test_env_overrides_default(self, monkeypatch):
        monkeypatch.setenv(R.ENV_JUDGE_MODEL, "glm-4.7-flashx")
        assert R.build_evaluator_llm().model_name == "glm-4.7-flashx"

    def test_falls_back_when_env_empty(self, monkeypatch):
        monkeypatch.setenv(R.ENV_JUDGE_MODEL, "   ")
        assert R.build_evaluator_llm().model_name == R.DEFAULT_JUDGE_MODEL

    def test_falls_back_when_env_absent(self, monkeypatch):
        monkeypatch.delenv(R.ENV_JUDGE_MODEL, raising=False)
        assert R.build_evaluator_llm().model_name == R.DEFAULT_JUDGE_MODEL

    def test_helper_reads_env(self, monkeypatch):
        monkeypatch.setenv(R.ENV_JUDGE_MODEL, "custom-model")
        assert R._judge_model_name() == "custom-model"
        monkeypatch.delenv(R.ENV_JUDGE_MODEL, raising=False)
        assert R._judge_model_name() == R.DEFAULT_JUDGE_MODEL

    def test_env_var_name_is_documented_constant(self):
        """环境变量名要有常量，避免各处写字符串字面量写歪。"""
        assert R.ENV_JUDGE_MODEL == "RAGAS_JUDGE_MODEL"


class TestDoesNotImportAtModuleLoad:
    """裁判依赖仍是**延迟导入**。

    本模块（`ragas_eval.py`）的导入曾经全部延迟——文件头写明原因：
    ragas + langchain 是重依赖，且在模块顶层导入会改变 `httpx` 客户端
    的初始化时机（触发 trust_env 相关的 WinError 10061）。
    故新增的 `_judge_model_name` 里也是延迟 `import os`。

    这条测试防止日后有人"顺手"把 import 提到文件顶部。
    """

    def test_no_toplevel_langchain_import(self):
        src = (EVAL_DIR / "ragas_eval.py").read_text(encoding="utf-8")
        head = src.split("def ", 1)[0]  # 到第一个函数定义之前
        assert "from langchain_openai import" not in head, (
            "langchain_openai 不该在模块顶层导入（会改变导入时序）"
        )
        assert "import httpx" not in head, "httpx 不该在模块顶层导入"


class TestRetryAttempts:
    """判分失败重试——**换 GLM 裁判后必需**，不是保险起见。

    ★ 为什么需要（实测数据，不是猜的）
    =============================
    `glm-5.3-flash` 当 ragas 裁判时**单次成功率只有 4/6 ≈ 0.67**，
    失败信息是 `faithfulness: 裁判未给出分数（NaN/Inf）`。

    根因不是 JSON 解析，而是 ragas 的第一步"从回答抽取陈述"返回空列表
    （`ragas/metrics/_faithfulness.py` L191/L210 → `np.nan`）。
    **同一份输入重试就能成功**——故是随机性，重试有效。

    实测加 3 次尝试后：单次 0.67 → **8/8 全部成功**。

    对照（同一条用例）：
        DeepSeek（原裁判）3/3   glm-5.3-flash 4/6
        glm-4.7-flashx 0/4     glm-4.6 0/3      glm-4.7 0/3
    即智谱家族在 ragas 的结构化输出上明显弱——但为了消除自评偏差
    （裁判必须与生成方不同家）只能用它，故用重试补可靠性。
    """

    def test_default_attempts_at_least_three(self):
        """p=0.67 时 3 次尝试才有 ≈0.96 的成功率。默认不能低于 3。"""
        assert R.DEFAULT_JUDGE_ATTEMPTS >= 3

    def test_attempts_default(self, monkeypatch):
        monkeypatch.delenv(R.ENV_JUDGE_ATTEMPTS, raising=False)
        assert R._judge_attempts() == R.DEFAULT_JUDGE_ATTEMPTS

    def test_attempts_env_override(self, monkeypatch):
        monkeypatch.setenv(R.ENV_JUDGE_ATTEMPTS, "5")
        assert R._judge_attempts() == 5

    def test_attempts_never_below_one(self, monkeypatch):
        """★ 配成 0 或负数时必须回落，不能变成"一次都不试"。

        `range(1, 0 + 1)` 是空循环 → `scores` 保持空 dict → **每一题都判分失败**，
        而且看起来像"裁判全坏"。这类配置错误要静默兜住。
        """
        for bad in ("0", "-3", "abc", ""):
            monkeypatch.setenv(R.ENV_JUDGE_ATTEMPTS, bad)
            assert R._judge_attempts() >= 1, f"{bad!r} 应回落到至少 1 次"

    def test_attempts_is_runtime_not_frozen(self, monkeypatch):
        """★ 必须是**运行期**解析。

        第一版评分逻辑里直接用模块常量 `JUDGE_ATTEMPTS`，而它在**导入时**冻结
        ——环境变量改了不生效，且没有任何提示。
        """
        monkeypatch.setenv(R.ENV_JUDGE_ATTEMPTS, "7")
        assert R._judge_attempts() == 7, "读的必须是环境变量，不是导入时的常量值"

    def test_no_logger_reference_in_source(self):
        """★ 重试分支不能引用 `logger`。

        本模块**没有** logging 配置（全文用 print）。实测踩到：第一版写了
        `logger.warning`，而本文件从未定义 logger → 一旦真的触发重试就
        `NameError`，把"判分失败"升级成"整轮崩溃"——**只在失败路径上炸**，
        正常跑永远发现不了。
        """
        src = (EVAL_DIR / "ragas_eval.py").read_text(encoding="utf-8")
        code_lines = [
            ln for ln in src.splitlines() if not ln.strip().startswith("#")
        ]
        assert "logger." not in "\n".join(code_lines), (
            "代码里引用了 logger，但本模块没有定义它——失败路径会 NameError"
        )

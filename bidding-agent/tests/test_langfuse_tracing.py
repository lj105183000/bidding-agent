"""Langfuse 集成的测试。

★ 这是本项目**第一个向外发数据**的模块
=================================
此前的可观测性（health / logs / metrics / 阶段耗时 / LLM 追踪）全部自持：
不开容器、不连外部服务、不发任何数据出机器。引入 Langfuse 意味着多一个
**网络依赖**，且 **prompt 与回答会离开本机**。

故本文件的重点不是"上报格式对不对"，而是**边界条件**：

    ① 未配置 → 完全 no-op（不 import SDK、不建客户端、不联网）
    ② SDK 未安装 → 同样 no-op（它是 optional-dependency）
    ③ 任何上报路径都不抛异常（观测不能搞挂业务）
    ④ 失败后进入冷却（上游挂了不该让每个请求都白等超时）
    ⑤ 变量名兼容 `LANGFUSE_BASE_URL` 与 `LANGFUSE_HOST`
       —— 实测用户 `.env` 里写的是 `BASE_URL`，只读 `HOST` 会导致
          **静默发到错误的地方**（数据去了默认云端而用户以为是自建）

全部用 monkeypatch 控制环境变量与 SDK 可用性，**不联网、不发真实数据**。
"""

from __future__ import annotations

import pytest

from src import langfuse_tracing as lt
from src.monitoring_trace import LLMCall, LLMTraceRecorder


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """清掉所有 Langfuse 环境变量，让每个用例从"未配置"开始。

    ★ 必须清：开发者本机 `.env` 里**可能真的配了 key**（实测用户就配了），
    不清会导致测试往外发真实数据。
    """
    for name in (lt.ENV_PUBLIC_KEY, lt.ENV_SECRET_KEY, *lt.ENV_HOST_ALIASES):
        monkeypatch.delenv(name, raising=False)
    yield


class TestHostResolution:
    """★ 变量名兼容——这是实测踩到的真 bug。

    用户 `.env` 里写的是 `LANGFUSE_BASE_URL=https://jp.cloud.langfuse.com`，
    而第一版只读 `LANGFUSE_HOST` → 取不到 → 传 `None` 给 SDK
    → **SDK 回落到默认云端**。配置被静默忽略，数据发到了别处。
    """

    def test_base_url_is_read(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_BASE_URL", "https://jp.cloud.langfuse.com")
        assert lt.resolve_host() == "https://jp.cloud.langfuse.com"

    def test_host_alias_also_read(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_HOST", "http://localhost:3000")
        assert lt.resolve_host() == "http://localhost:3000"

    def test_base_url_wins_when_both_present(self, monkeypatch):
        """两个都在时以 BASE_URL 为准（它是用户已有的写法）。"""
        monkeypatch.setenv("LANGFUSE_BASE_URL", "https://a.example")
        monkeypatch.setenv("LANGFUSE_HOST", "https://b.example")
        assert lt.resolve_host() == "https://a.example"

    def test_env_host_order_declares_base_url_first(self):
        assert lt.ENV_HOST_ALIASES[0] == "LANGFUSE_BASE_URL"

    def test_empty_returns_empty_not_a_default(self, monkeypatch):
        """★ 未配置时返回空串，**不编造默认地址**。

        若这里硬编码 `https://cloud.langfuse.com`，用户配了自建实例却因为
        变量名写错而取不到时，数据会被**静默发到默认云端**——那正是上面那个 bug。
        返回空串则交给 SDK 自己决定，且 `status()` 会如实报出 host_env 为空。
        """
        assert lt.resolve_host() == ""

    def test_strips_quotes_and_spaces(self, monkeypatch):
        """`.env` 里写 `LANGFUSE_BASE_URL="https://x"` 时要去掉引号。"""
        monkeypatch.setenv("LANGFUSE_BASE_URL", '  "https://x.example"  ')
        assert lt.resolve_host() == "https://x.example"


class TestDisabledByDefault:
    """① 未配置时完全 no-op。"""

    def test_not_configured_without_keys(self):
        assert lt.is_configured() is False

    def test_configured_needs_both_keys(self, monkeypatch):
        monkeypatch.setenv(lt.ENV_PUBLIC_KEY, "pk")
        assert lt.is_configured() is False, "只有 public key 不算配置完成"
        monkeypatch.setenv(lt.ENV_SECRET_KEY, "sk")
        assert lt.is_configured() is True

    def test_blank_keys_not_configured(self, monkeypatch):
        monkeypatch.setenv(lt.ENV_PUBLIC_KEY, "   ")
        monkeypatch.setenv(lt.ENV_SECRET_KEY, "   ")
        assert lt.is_configured() is False

    def test_recorder_falls_back_to_base_class(self, monkeypatch):
        """未配置时返回**基类**（纯本地记录），调用方无需分支。"""
        rec = lt.build_recorder()
        assert type(rec).__name__ == "LLMTraceRecorder"
        assert not hasattr(rec, "tagger")

    def test_configured_returns_langfuse_recorder(self, monkeypatch):
        monkeypatch.setenv(lt.ENV_PUBLIC_KEY, "pk")
        monkeypatch.setenv(lt.ENV_SECRET_KEY, "sk")
        monkeypatch.setattr(lt, "_sdk_available", lambda: True)
        # ★ 必须挡住真实客户端：否则 `Langfuse(...)` 会真的建起来并在后台
        # 往云端导出（实测打出 `Failed to export spans batch code: 401`）——
        # 用假 key 污染用户真实账号，且测试变得依赖网络。
        monkeypatch.setattr(
            lt.LangfuseTagger, "_get_client", lambda self: None
        )
        rec = lt.build_recorder()
        assert type(rec).__name__ == "LangfuseTraceRecorder"
        assert hasattr(rec, "tagger")

    def test_all_methods_noop_when_unconfigured(self):
        """★ 未配置时四个方法都必须是 no-op 且不抛。

        这是"默认关闭"的实际保证——`_chat_events` 与 `react_loop` 会无条件
        调用它们，不能要求调用方先判断是否配置。
        """
        t = lt.LangfuseTagger()
        assert t.available() is False
        t.begin("chat", metadata={"a": 1})  # 不应抛
        t.push_llm_call(LLMCall(kind="decision", elapsed_ms=1))
        t.push_tool({"name": "x", "ms": 1})
        t.finish(output_summary={"a": 1})


class TestSdkMissing:
    """② SDK 未安装时同样 no-op（它是 optional-dependency）。"""

    def test_recorder_falls_back_when_sdk_missing(self, monkeypatch):
        monkeypatch.setenv(lt.ENV_PUBLIC_KEY, "pk")
        monkeypatch.setenv(lt.ENV_SECRET_KEY, "sk")
        monkeypatch.setattr(lt, "_sdk_available", lambda: False)
        rec = lt.build_recorder()
        assert type(rec).__name__ == "LLMTraceRecorder", (
            "SDK 未装时应降级为本地记录，而不是抛 ImportError"
        )

    def test_tagger_unavailable_when_sdk_missing(self, monkeypatch):
        monkeypatch.setenv(lt.ENV_PUBLIC_KEY, "pk")
        monkeypatch.setenv(lt.ENV_SECRET_KEY, "sk")
        monkeypatch.setattr(lt, "_sdk_available", lambda: False)
        assert lt.LangfuseTagger().available() is False

    def test_status_reports_missing_sdk(self, monkeypatch):
        monkeypatch.setenv(lt.ENV_PUBLIC_KEY, "pk")
        monkeypatch.setenv(lt.ENV_SECRET_KEY, "sk")
        monkeypatch.setattr(lt, "_sdk_available", lambda: False)
        st = lt.status()
        assert st["enabled"] is False
        assert st["sdk_installed"] is False
        assert "langfuse" in st["reason"]


class TestNeverRaises:
    """③ 任何上报路径都不抛异常（观测不能搞挂业务）。"""

    def test_push_tolerates_broken_client(self, monkeypatch):
        monkeypatch.setenv(lt.ENV_PUBLIC_KEY, "pk")
        monkeypatch.setenv(lt.ENV_SECRET_KEY, "sk")
        monkeypatch.setattr(lt, "_sdk_available", lambda: True)

        class Boom:
            def start_observation(self, **kw):
                raise RuntimeError("SDK 炸了")

            def create_event(self, **kw):
                raise RuntimeError("SDK 炸了")

            def create_trace_id(self):
                return "tid"

        t = lt.LangfuseTagger()
        monkeypatch.setattr(t, "_get_client", lambda: Boom())
        t.begin("chat")  # 不应抛
        t.push_llm_call(LLMCall(kind="decision", elapsed_ms=1))  # 不应抛
        t.push_tool({"name": "x", "ms": 1})  # 不应抛
        t.finish()  # 不应抛

    def test_get_client_failure_enters_cooldown(self, monkeypatch):
        """客户端建不出来 → 进冷却，而不是每个请求都重试一次。"""
        monkeypatch.setenv(lt.ENV_PUBLIC_KEY, "pk")
        monkeypatch.setenv(lt.ENV_SECRET_KEY, "sk")
        monkeypatch.setattr(lt, "_sdk_available", lambda: True)

        import builtins

        real_import = builtins.__import__

        def fake_import(name, *a, **k):
            if name == "langfuse":
                raise RuntimeError("导入期炸")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        t = lt.LangfuseTagger()
        assert t._get_client() is None  # 不应抛
        assert t.available() is False, "失败后应进入冷却"


class TestRecorderSubclass:
    """`LangfuseTraceRecorder` 对外必须与基类**完全一致**。

    这样"关掉 Langfuse"只是换个工厂返回值，调用方无需任何类型分支。
    """

    def test_local_recording_still_works(self, monkeypatch):
        """★ 上报失败或未配置时，**本地数据必须照旧完整**。

        本地 `llm_trace` 是耗时面板与 `/api/metrics` 的数据源；
        不能因为"上报失败"而丢本地记录。
        """
        monkeypatch.setenv(lt.ENV_PUBLIC_KEY, "pk")
        monkeypatch.setenv(lt.ENV_SECRET_KEY, "sk")

        rec = lt.LangfuseTraceRecorder(tagger=lt.LangfuseTagger())  # 未配置 host → 不上报
        rec.record(LLMCall(kind="decision", elapsed_ms=100, total_tokens=10))
        rec.record(LLMCall(kind="stream", elapsed_ms=200, total_tokens=20))

        s = rec.summary()
        assert s["call_count"] == 2
        assert s["total_ms"] == 300
        assert s["total_tokens"] == 30

    def test_is_a_llm_trace_recorder(self):
        rec = lt.LangfuseTraceRecorder(tagger=lt.LangfuseTagger())
        assert isinstance(rec, LLMTraceRecorder), (
            "必须仍是 LLMTraceRecorder（agent 与客户端都按这个类型用）"
        )

    def test_push_tool_does_not_raise_when_disabled(self):
        rec = lt.LangfuseTraceRecorder(tagger=lt.LangfuseTagger())
        rec.push_tool({"name": "search_knowledge_base", "ms": 10, "ok": True})
        rec.push_tool({})  # 畸形输入也不该抛


class TestUsageDetails:
    """usage_details 的构造：**只放真实取到的字段，不补 0**。

    Langfuse 用这些值算成本。补 0 会让"没拿到 usage"看起来像"那次没花 token"，
    把成本曲线压低——比缺数据更糟。
    """

    def test_all_fields_present(self):
        call = LLMCall(kind="decision", elapsed_ms=1, prompt_tokens=100,
                       completion_tokens=20, total_tokens=120, cache_hit_tokens=80)
        d = lt._usage_details(call)
        assert d == {"input": 100, "output": 20, "total": 120, "input_cached_tokens": 80}

    def test_missing_fields_omitted_not_zeroed(self):
        call = LLMCall(kind="decision", elapsed_ms=1)
        assert lt._usage_details(call) is None, "全缺时应返回 None，而不是 {'input': 0}"

    def test_partial_fields_kept(self):
        call = LLMCall(kind="decision", elapsed_ms=1, total_tokens=5)
        assert lt._usage_details(call) == {"total": 5}


class TestStatus:
    def test_disabled_reports_reason(self):
        st = lt.status()
        assert st["enabled"] is False
        assert st["configured"] is False
        assert lt.ENV_PUBLIC_KEY in st["reason"]

    def test_reports_which_env_var_provided_host(self, monkeypatch):
        monkeypatch.setenv(lt.ENV_PUBLIC_KEY, "pk")
        monkeypatch.setenv(lt.ENV_SECRET_KEY, "sk")
        monkeypatch.setenv("LANGFUSE_BASE_URL", "https://jp.example")
        st = lt.status()
        assert st["host"] == "https://jp.example"
        assert st["host_env"] == "LANGFUSE_BASE_URL"


class TestNoRealNetworkInTests:
    """★ 元测试：本文件**不得**真的连 Langfuse。

    实测踩到：`test_configured_returns_langfuse_recorder` 建了真实客户端，
    SDK 后台线程拿着假 key 往云端导出，打出
    `Failed to export spans batch code: 401` —— 那是**真实网络请求**。

    危害有两个，都不可接受：
        ① 用无效凭据污染用户的真实 Langfuse 账号（会有失败记录）
        ② 测试依赖网络，CI 上可能慢或超时

    故这条测试锁住"建客户端必须被挡住"这个约束。
    """

    def test_building_client_is_blocked_in_tests(self, monkeypatch):
        monkeypatch.setenv(lt.ENV_PUBLIC_KEY, "pk")
        monkeypatch.setenv(lt.ENV_SECRET_KEY, "sk")
        monkeypatch.setattr(lt, "_sdk_available", lambda: True)
        monkeypatch.setattr(lt.LangfuseTagger, "_get_client", lambda self: None)
        t = lt.LangfuseTagger()
        assert t._get_client() is None, "测试里不该真的建 Langfuse 客户端"
        # 后续所有上报都因 client=None 而早退，不会联网
        t.begin("chat")
        t.push_llm_call(LLMCall(kind="decision", elapsed_ms=1))
        t.finish()


# ===========================================================================
# 下面是与"局限与未解决"对应的补充测试
#
#   ② LANGFUSE_TRACING_ENABLED 总开关
#   ③ flush 策略（开关 + 超时上界）
#   ④ 正文上报（默认关）
#   ⑤ 强制脱敏
#   ⑥ /api/metrics 接入
#   ⑦ 成本回填
#   ⑧ 采样率
#   ① 自检入口（不联网的那一半）
#
# 全部离线：用假客户端 / 假 httpx，绝不在测试里发真实数据（见
# `TestNoRealNetworkInTests`）。
# ===========================================================================


class FakeClient:
    """够用的假 Langfuse 客户端：只记下调用参数。"""

    def __init__(self, flush_seconds: float = 0.0) -> None:
        self.observations: list[dict] = []
        self.events: list[dict] = []
        self.flush_calls = 0
        self._flush_seconds = flush_seconds

    def create_trace_id(self) -> str:
        return "trace-0001"

    def create_event(self, **kw):
        self.events.append(kw)
        return self

    def start_observation(self, **kw):
        self.observations.append(kw)
        return self

    def end(self, **kw):
        return self

    def flush(self) -> None:
        self.flush_calls += 1
        if self._flush_seconds:
            import time as _t

            _t.sleep(self._flush_seconds)


@pytest.fixture
def fake_client(monkeypatch):
    """把"建客户端"整体换成假客户端（不 import SDK、不联网）。"""
    client = FakeClient()
    monkeypatch.setattr(lt, "build_client", lambda: client)
    monkeypatch.setattr(lt, "_sdk_available", lambda: True)
    monkeypatch.setattr(lt, "_CLIENTS", {})
    monkeypatch.setenv(lt.ENV_PUBLIC_KEY, "pk")
    monkeypatch.setenv(lt.ENV_SECRET_KEY, "sk")
    lt.reset_stats()
    yield client
    lt.reset_stats()


class TestEnvParsing:
    """环境变量解析：开关的"写法兼容"是这一版的核心修复点。"""

    @pytest.mark.parametrize("raw", ["false", "False", "FALSE", "0", "no", "off", "disabled", "否", "关闭"])
    def test_tracing_enabled_falsey_forms(self, monkeypatch, raw):
        """★ SDK 只把字面量 `"false"` 当关；写 `0`/`no`/`off` 时它**照旧上报**。

        用户写这些写法时期望的是"别发数据"，故本模块自己解析，并把结果显式
        传给 SDK（见 `build_client`），两边判定永远一致。
        """
        monkeypatch.setenv(lt.ENV_TRACING_ENABLED, raw)
        assert lt.tracing_enabled() is False

    @pytest.mark.parametrize("raw", ["true", "1", "yes", "on", "是"])
    def test_tracing_enabled_truthy_forms(self, monkeypatch, raw):
        monkeypatch.setenv(lt.ENV_TRACING_ENABLED, raw)
        assert lt.tracing_enabled() is True

    def test_tracing_enabled_default_is_on(self):
        assert lt.tracing_enabled() is True

    def test_unknown_value_keeps_default(self, monkeypatch):
        monkeypatch.setenv(lt.ENV_TRACING_ENABLED, "maybe")
        assert lt.tracing_enabled() is True

    @pytest.mark.parametrize("raw,expected", [("0.5", 0.5), ("1", 1.0), ("0", 0.0), ("-1", 0.0), ("2", 1.0)])
    def test_sample_rate_parsing_and_clamping(self, monkeypatch, raw, expected):
        monkeypatch.setenv(lt.ENV_SAMPLE_RATE, raw)
        assert lt.sample_rate() == expected

    def test_sample_rate_invalid_falls_back(self, monkeypatch):
        """★ 非法采样率不能把客户端建崩。

        SDK 自己会 `float(...)` 后抛 ValueError → 客户端建不起来 → **整条
        上报链路静默失效**。本模块先收敛成合法值，宁可退回 1.0 也不能哑掉。
        """
        monkeypatch.setenv(lt.ENV_SAMPLE_RATE, "abc")
        assert lt.sample_rate() == 1.0

    def test_sample_rate_default(self):
        assert lt.sample_rate() == 1.0

    def test_capture_default_off(self):
        """正文上报**默认关**（招投标场景含敏感信息）。"""
        assert lt.capture_enabled() is False
        assert lt.capture_limit() == lt.DEFAULT_MAX_CAPTURE_CHARS

    def test_capture_limit_override(self, monkeypatch):
        monkeypatch.setenv(lt.ENV_CAPTURE_CONTENT, "true")
        monkeypatch.setenv(lt.ENV_MAX_CAPTURE_CHARS, "128")
        assert lt.capture_enabled() is True
        assert lt.capture_limit() == 128

    def test_capture_limit_never_negative(self, monkeypatch):
        monkeypatch.setenv(lt.ENV_MAX_CAPTURE_CHARS, "-5")
        assert lt.capture_limit() == 0

    def test_flush_policy_defaults(self):
        assert lt.flush_on_finish() is False
        assert lt.flush_timeout_s() == lt.DEFAULT_FLUSH_TIMEOUT_S

    def test_flush_timeout_has_floor(self, monkeypatch):
        """超时必须 >0：写成 0 会让每次 flush 都立刻判超时，等于永远不等。"""
        monkeypatch.setenv(lt.ENV_FLUSH_ON_FINISH, "true")
        monkeypatch.setenv(lt.ENV_FLUSH_TIMEOUT, "0")
        assert lt.flush_timeout_s() >= 0.1
        assert lt.flush_on_finish() is True


class TestEnabledGate:
    """② 总开关必须能单独关掉一切，且 status() 要说明原因。"""

    def test_is_enabled_requires_switch(self, monkeypatch):
        monkeypatch.setenv(lt.ENV_PUBLIC_KEY, "pk")
        monkeypatch.setenv(lt.ENV_SECRET_KEY, "sk")
        monkeypatch.setattr(lt, "_sdk_available", lambda: True)
        assert lt.is_enabled() is True
        monkeypatch.setenv(lt.ENV_TRACING_ENABLED, "0")
        assert lt.is_enabled() is False, "总开关关掉后不许再上报"

    def test_recorder_falls_back_when_switched_off(self, monkeypatch):
        monkeypatch.setenv(lt.ENV_PUBLIC_KEY, "pk")
        monkeypatch.setenv(lt.ENV_SECRET_KEY, "sk")
        monkeypatch.setattr(lt, "_sdk_available", lambda: True)
        monkeypatch.setenv(lt.ENV_TRACING_ENABLED, "false")
        assert type(lt.build_recorder()).__name__ == "LLMTraceRecorder"

    def test_build_client_returns_none_when_switched_off(self, monkeypatch):
        """★ 开关关掉时**连客户端都不建**（不 import SDK、不起后台线程）。"""
        monkeypatch.setenv(lt.ENV_PUBLIC_KEY, "pk")
        monkeypatch.setenv(lt.ENV_SECRET_KEY, "sk")
        monkeypatch.setenv(lt.ENV_TRACING_ENABLED, "off")
        monkeypatch.setattr(lt, "_sdk_available", lambda: True)
        assert lt.build_client() is None

    def test_status_reports_switch(self, monkeypatch):
        monkeypatch.setenv(lt.ENV_PUBLIC_KEY, "pk")
        monkeypatch.setenv(lt.ENV_SECRET_KEY, "sk")
        monkeypatch.setenv(lt.ENV_TRACING_ENABLED, "false")
        monkeypatch.setattr(lt, "_sdk_available", lambda: True)
        st = lt.status()
        assert st["enabled"] is False
        assert st["tracing_enabled"] is False
        assert lt.ENV_TRACING_ENABLED in st["reason"]

    def test_status_exposes_all_knobs(self, monkeypatch):
        monkeypatch.setenv(lt.ENV_SAMPLE_RATE, "0.25")
        monkeypatch.setenv(lt.ENV_CAPTURE_CONTENT, "true")
        monkeypatch.setenv(lt.ENV_FLUSH_ON_FINISH, "true")
        st = lt.status()
        for key in ("sample_rate", "capture_content", "max_capture_chars",
                    "flush_on_finish", "flush_timeout_s", "sdk_version"):
            assert key in st, f"status() 缺少开关 {key}"
        assert st["sample_rate"] == 0.25
        assert st["capture_content"] is True


class TestSampling:
    """⑧ 采样（按 trace，不按 observation）。"""

    def test_zero_rate_samples_out_everything(self, fake_client, monkeypatch):
        monkeypatch.setenv(lt.ENV_SAMPLE_RATE, "0")
        t = lt.LangfuseTagger()
        t.begin("chat")
        assert t.sampled() is False
        assert t.available() is False
        t.push_llm_call(LLMCall(kind="stream", elapsed_ms=1))
        t.finish()
        assert fake_client.observations == []
        assert fake_client.events == []
        assert lt.stats()["traces_sampled_out"] == 1

    def test_full_rate_always_samples(self, fake_client):
        t = lt.LangfuseTagger()
        t.begin("chat")
        assert t.sampled() is True
        assert fake_client.events and fake_client.events[0]["name"] == "chat"

    def test_sampling_is_trace_level(self, fake_client, monkeypatch):
        """★ 一条 trace 要么整棵树都记，要么一条都不记。

        按 observation 采样会出现"有工具调用但没有对应 LLM 调用"的残缺调用树，
        比没有数据更容易误导排查。
        """
        monkeypatch.setenv(lt.ENV_SAMPLE_RATE, "0")
        t = lt.LangfuseTagger()
        t.begin("chat")
        for _ in range(5):
            t.push_llm_call(LLMCall(kind="decision", elapsed_ms=1))
            t.push_tool({"name": "x", "ok": True})
        assert fake_client.observations == []

    def test_partial_rate_is_actually_random(self, fake_client, monkeypatch):
        monkeypatch.setenv(lt.ENV_SAMPLE_RATE, "0.5")
        hit = 0
        for _ in range(400):
            t = lt.LangfuseTagger()
            t.begin("chat")
            hit += 1 if t.sampled() else 0
        assert 130 < hit < 270, f"0.5 采样率下 400 次命中 {hit} 次，偏离过大"

    def test_metadata_records_sample_rate(self, fake_client, monkeypatch):
        """trace 元数据里要写采样率：事后看一条 trace 才知道"为什么没有正文"。"""
        monkeypatch.setenv(lt.ENV_SAMPLE_RATE, "1")
        monkeypatch.setenv(lt.ENV_CAPTURE_CONTENT, "true")
        lt.LangfuseTagger().begin("chat", metadata={"session_id": "s"})
        meta = fake_client.events[0]["metadata"]
        assert meta["sample_rate"] == 1.0
        assert meta["capture_content"] is True
        assert meta["session_id"] == "s"


class TestRedaction:
    """⑤ 脱敏：**没有开关**，一旦发正文就必须执行。"""

    @pytest.mark.parametrize(
        "text,masked,mark",
        [
            ("电话 13812345678", "13812345678", "[手机号]"),
            ("身份证 340104199001011234", "340104199001011234", "[身份证号]"),
            ("信用代码 91340100MA2TXXXXXX", "91340100MA2TXXXXXX", "[统一社会信用代码]"),
            ("邮箱 zhangsan@example.com", "zhangsan@example.com", "[邮箱]"),
            ("预算 1,250,000.00 元", "1,250,000.00", "[金额]"),
            ("金额 500万元", "500万元", "[金额]"),
            ("项目编号：2026BFA0001", "2026BFA0001", "[编号]"),
            ("采购人：安徽智云信息技术有限公司", "安徽智云信息技术有限公司", "[企业名称]"),
            ("联系人：张三", "张三", "[姓名]"),
        ],
    )
    def test_each_rule_masks(self, text, masked, mark):
        out = lt.redact(text)
        assert masked not in out, f"{masked!r} 未被脱敏"
        assert mark in out

    def test_is_idempotent(self):
        once = lt.redact("预算 500万元，电话 13812345678")
        twice = lt.redact(once)
        assert once == twice, "二次脱敏不该改变结果（mask 钩子会对同一份数据再跑一次）"

    def test_redact_payload_recurses(self):
        payload = {
            "prompt": "联系人：张三，预算 500万元",
            "meta": {"company": "安徽智云信息技术有限公司"},
            "list": ["电话 13812345678", 42, None],
        }
        out = lt.redact_payload(payload)
        blob = str(out)
        assert "张三" not in blob
        assert "500万元" not in blob
        assert "安徽智云信息技术有限公司" not in blob
        assert "13812345678" not in blob
        assert out["list"][1] == 42, "非字符串原样保留"

    def test_empty_text_passthrough(self):
        assert lt.redact("") == ""

    def test_counter_tracks_hits(self):
        lt.reset_stats()
        lt.redact("预算 500万元，电话 13812345678")
        assert lt.stats()["redactions"] == 2


class TestCaptureContent:
    """④ 正文上报：默认关；开了也必须脱敏 + 截断。"""

    def _call(self, **kw):
        base = dict(kind="stream", provider="deepseek", model="deepseek-chat", elapsed_ms=1)
        base.update(kw)
        return LLMCall(**base)

    def test_off_by_default_no_content_sent(self, fake_client):
        t = lt.LangfuseTagger()
        t.begin("chat")
        t.push_llm_call(self._call(prompt_text="user: 绝密ABC", response_text="回答XYZ"))
        obs = fake_client.observations[0]
        # `_push` 只在有值时传字段，故用 get()：缺键 = 没发这一类数据
        assert obs.get("input") is None, "默认不该发 prompt"
        assert "response" not in (obs.get("output") or {}), "默认不该发回答正文"
        assert obs["output"]["chars"] == 0  # 计数仍然上报

    def test_on_sends_redacted_content(self, fake_client, monkeypatch):
        monkeypatch.setenv(lt.ENV_CAPTURE_CONTENT, "true")
        t = lt.LangfuseTagger()
        t.begin("chat")
        t.push_llm_call(self._call(
            prompt_text="user: 采购人：安徽智云信息技术有限公司，预算 500万元",
            response_text="建议报价 480万元",
            content_chars=len("建议报价 480万元"),
        ))
        obs = fake_client.observations[0]
        blob = str(obs.get("input")) + str(obs.get("output"))
        assert "安徽智云信息技术有限公司" not in blob
        assert "500万元" not in blob
        assert "[企业名称]" in blob and "[金额]" in blob
        assert obs["output"]["chars"] == len("建议报价 480万元")

    def test_clips_long_text_keeping_head_and_tail(self, fake_client, monkeypatch):
        monkeypatch.setenv(lt.ENV_CAPTURE_CONTENT, "true")
        monkeypatch.setenv(lt.ENV_MAX_CAPTURE_CHARS, "100")
        long_prompt = "HEAD" + "x" * 5000 + "TAIL"
        t = lt.LangfuseTagger()
        t.begin("chat")
        t.push_llm_call(self._call(prompt_text=long_prompt))
        sent = fake_client.observations[0]["input"]["prompt"]
        assert len(sent) <= 100 + 40, "截断后不应超上限太多"
        assert sent.startswith("HEAD")
        assert sent.endswith("TAIL"), "★ 尾部必须保留（用户最后问的那句话在这里）"
        assert "省略" in sent

    def test_captured_chars_counter(self, fake_client, monkeypatch):
        monkeypatch.setenv(lt.ENV_CAPTURE_CONTENT, "true")
        t = lt.LangfuseTagger()
        t.begin("chat")
        t.push_llm_call(self._call(prompt_text="abc", response_text="de"))
        assert lt.stats()["captured_chars"] == 5

    def test_tool_io_never_captured(self, fake_client, monkeypatch):
        """工具出入参不采集：常含整段招标文件原文，体积大且最敏感。"""
        monkeypatch.setenv(lt.ENV_CAPTURE_CONTENT, "true")
        t = lt.LangfuseTagger()
        t.begin("chat")
        t.push_tool({"name": "search", "ok": True, "ms": 3, "text": "整段原文"})
        obs = fake_client.observations[0]
        assert obs.get("input") is None
        assert "整段原文" not in str(obs.get("output"))


class TestCostEstimation:
    """⑦ 成本回填：不依赖 Langfuse 侧价目表。"""

    def test_known_model_cost(self):
        cost = lt.estimate_cost("deepseek-chat", {"input": 1_000_000, "output": 1_000_000})
        assert cost is not None
        assert cost["input"] == pytest.approx(0.27, abs=1e-6)
        assert cost["output"] == pytest.approx(1.10, abs=1e-6)
        assert cost["total"] == pytest.approx(1.37, abs=1e-6)

    def test_cache_hit_is_cheaper(self):
        """★ 缓存命中要按折扣价算：DeepSeek 的 input 是**含**命中的总量。"""
        no_cache = lt.estimate_cost("deepseek-chat", {"input": 1_000_000, "output": 0})
        with_cache = lt.estimate_cost(
            "deepseek-chat", {"input": 1_000_000, "output": 0, "input_cached_tokens": 1_000_000}
        )
        assert with_cache["total"] < no_cache["total"]
        assert with_cache["total"] == pytest.approx(0.07, abs=1e-6)

    def test_model_prefix_match(self):
        """带日期后缀的模型名（deepseek-chat-0324）也要能匹配到单价。"""
        assert lt.estimate_cost("deepseek-chat-0324", {"input": 100, "output": 0}) is not None

    def test_unknown_model_returns_none(self):
        """算不出来就**不发**：发 0 会让控制台显示"这次没花钱"。"""
        assert lt.estimate_cost("some-unknown-model", {"input": 1000, "output": 100}) is None

    def test_no_usage_returns_none(self):
        assert lt.estimate_cost("deepseek-chat", None) is None
        assert lt.estimate_cost("deepseek-chat", {}) is None

    def test_local_model_is_zero_not_missing(self):
        """本地 Ollama：显式 0 比"缺数据"准确（确实没有 API 费用）。"""
        cost = lt.estimate_cost("qwen2.5:7b", {"input": 10_000, "output": 5_000})
        assert cost == {"input": 0.0, "output": 0.0, "total": 0.0}

    def test_env_price_table_overrides_and_merges(self, monkeypatch):
        monkeypatch.setenv(lt.ENV_PRICE_TABLE, '{"deepseek-chat": {"input": 1.0, "output": 2.0}}')
        cost = lt.estimate_cost("deepseek-chat", {"input": 1_000_000, "output": 0})
        assert cost["input"] == pytest.approx(1.0)
        # 没覆盖的模型仍走内置表（合并而不是替换）
        assert lt.estimate_cost("qwen2.5:7b", {"input": 10, "output": 0}) is not None

    def test_broken_price_table_does_not_raise(self, monkeypatch):
        monkeypatch.setenv(lt.ENV_PRICE_TABLE, "{不是 JSON")
        assert lt.estimate_cost("deepseek-chat", {"input": 1000, "output": 0}) is not None

    def test_cost_attached_to_observation(self, fake_client):
        t = lt.LangfuseTagger()
        t.begin("chat")
        t.push_llm_call(LLMCall(kind="decision", model="deepseek-chat", elapsed_ms=1,
                                prompt_tokens=1000, completion_tokens=100))
        obs = fake_client.observations[0]
        assert obs["cost_details"]["total"] > 0
        assert obs["metadata"]["cost_usd"] > 0
        assert obs["metadata"]["cost_source"] == "builtin"

    def test_no_cost_when_model_unknown(self, fake_client):
        t = lt.LangfuseTagger()
        t.begin("chat")
        t.push_llm_call(LLMCall(kind="decision", model="unknown-llm", elapsed_ms=1,
                                prompt_tokens=1000, completion_tokens=100))
        obs = fake_client.observations[0]
        assert obs.get("cost_details") is None, "算不出成本就不发（发 0 = 谎报没花钱）"
        assert "cost_usd" not in (obs.get("metadata") or {})


class TestFlushPolicy:
    """③ flush：默认不 flush，开了必须真的 flush。"""

    def test_finish_does_not_flush_by_default(self, fake_client):
        t = lt.LangfuseTagger()
        t.begin("chat")
        t.finish()
        assert fake_client.flush_calls == 0, "默认不该在收尾路径上等 flush"

    def test_finish_flushes_when_enabled(self, fake_client, monkeypatch):
        monkeypatch.setenv(lt.ENV_FLUSH_ON_FINISH, "true")
        t = lt.LangfuseTagger()
        t.begin("chat")
        t.finish()
        assert fake_client.flush_calls == 1

    def test_safe_flush_counts_timeout(self, monkeypatch):
        """★ 挂起的 flush 必须被**超时切断**，不能拖住请求收尾。"""
        lt.reset_stats()
        slow = FakeClient(flush_seconds=1.0)
        import time as _t

        t0 = _t.monotonic()
        ok = lt._safe_flush(slow, 0.1, source="test")
        elapsed = _t.monotonic() - t0
        assert ok is False
        assert elapsed < 0.6, f"超时未生效，等了 {elapsed:.2f}s"
        assert lt.stats()["flush_timeouts"] == 1

    def test_safe_flush_returns_true_when_fast(self):
        lt.reset_stats()
        assert lt._safe_flush(FakeClient(), 1.0, source="test") is True
        assert lt.stats()["flush_timeouts"] == 0

    def test_safe_flush_tolerates_broken_client(self):
        class Boom:
            def flush(self):
                raise RuntimeError("炸")

        assert lt._safe_flush(Boom(), 1.0, source="test") is True, "flush 抛异常也该算已尝试（不能把收尾卡住）"

    def test_module_flush_reports_summary(self, monkeypatch):
        # 整个替换 `_CLIENTS`（不是往里塞一条）：别的用例可能已经注册过客户端
        monkeypatch.setattr(lt, "_CLIENTS", {("pk", "url"): FakeClient()})
        out = lt.flush(0.5)
        assert out["clients"] == 1 and out["completed"] == 1

    def test_flush_client_on_explicit_client(self):
        assert lt.flush_client(FakeClient(), 0.5) is True


class TestMetricsSection:
    """⑥ /api/metrics 接入。"""

    def test_metrics_contains_status_and_counters(self, monkeypatch):
        monkeypatch.setenv(lt.ENV_PUBLIC_KEY, "pk")
        monkeypatch.setenv(lt.ENV_SECRET_KEY, "sk")
        m = lt.metrics()
        assert "counters" in m
        assert m["enabled"] in (True, False)
        for key in ("spans_pushed", "spans_failed", "traces_started", "traces_sampled_out",
                    "flush_timeouts", "cooldowns_entered", "captured_chars", "redactions"):
            assert key in m["counters"], f"counters 缺少 {key}"

    def test_metrics_never_raises(self, monkeypatch):
        def boom():
            raise RuntimeError("状态读取炸了")

        monkeypatch.setattr(lt, "status", boom)
        out = lt.metrics()
        assert out["enabled"] is False
        assert "error" in out

    def test_counters_move_with_activity(self, fake_client):
        t = lt.LangfuseTagger()
        t.begin("chat")
        t.push_llm_call(LLMCall(kind="decision", model="deepseek-chat", elapsed_ms=1))
        t.push_tool({"name": "x", "ok": True})
        c = lt.metrics()["counters"]
        assert c["traces_started"] == 1
        assert c["spans_pushed"] == 2

    def test_component_metrics_includes_langfuse(self):
        """`/api/metrics` 的 components 段里必须有一节 langfuse。"""
        from src import monitoring as mon

        section = mon.component_metrics(None, None).get("langfuse")
        assert isinstance(section, dict)
        assert "enabled" in section and "counters" in section

    def test_prometheus_renders_langfuse_lines(self):
        from src import monitoring as mon

        text = mon.render_prometheus(None, None)
        assert "bidding_langfuse_enabled" in text
        assert "bidding_langfuse_spans_pushed_total" in text


class TestSelfTest:
    """① 自检：默认只做**只读**检查（不往账号里写数据）。"""

    def test_not_configured(self):
        r = lt.self_test()
        assert r["ok"] is False
        assert any(lt.ENV_PUBLIC_KEY in x for x in r["reasons"])

    def test_switch_off_is_reported(self, monkeypatch):
        monkeypatch.setenv(lt.ENV_PUBLIC_KEY, "pk")
        monkeypatch.setenv(lt.ENV_SECRET_KEY, "sk")
        monkeypatch.setenv(lt.ENV_TRACING_ENABLED, "0")
        monkeypatch.setattr(lt, "_sdk_available", lambda: True)
        r = lt.self_test()
        assert r["ok"] is False
        assert any(lt.ENV_TRACING_ENABLED in x for x in r["reasons"])

    def test_bad_credentials_reported(self, monkeypatch):
        import httpx

        monkeypatch.setenv(lt.ENV_PUBLIC_KEY, "pk")
        monkeypatch.setenv(lt.ENV_SECRET_KEY, "sk")
        monkeypatch.setenv("LANGFUSE_BASE_URL", "https://example.invalid")
        monkeypatch.setattr(lt, "_sdk_available", lambda: True)

        class Resp:
            status_code = 401

        monkeypatch.setattr(httpx, "get", lambda *a, **k: Resp())
        r = lt.self_test()
        assert r["ok"] is False and r["auth_ok"] is False
        assert any("凭据被拒绝" in x for x in r["reasons"])

    def test_readonly_success_path(self, monkeypatch):
        import httpx

        monkeypatch.setenv(lt.ENV_PUBLIC_KEY, "pk")
        monkeypatch.setenv(lt.ENV_SECRET_KEY, "sk")
        monkeypatch.setenv("LANGFUSE_BASE_URL", "https://example.invalid")
        monkeypatch.setattr(lt, "_sdk_available", lambda: True)

        class Resp:
            status_code = 200

            @staticmethod
            def json():
                return {"data": [{"id": "p1", "name": "My Project"}]}

        monkeypatch.setattr(httpx, "get", lambda *a, **k: Resp())
        r = lt.self_test(send=False)
        assert r["ok"] is True
        assert r["auth_ok"] is True
        assert r["project"] == "My Project"
        assert r["sent"] is False, "默认不许发送任何数据"
        assert "auth" in r["checked"]

    def test_network_failure_is_reported_not_raised(self, monkeypatch):
        import httpx

        monkeypatch.setenv(lt.ENV_PUBLIC_KEY, "pk")
        monkeypatch.setenv(lt.ENV_SECRET_KEY, "sk")
        monkeypatch.setenv("LANGFUSE_BASE_URL", "https://example.invalid")
        monkeypatch.setattr(lt, "_sdk_available", lambda: True)

        def boom(*a, **k):
            raise OSError("网络不通")

        monkeypatch.setattr(httpx, "get", boom)
        r = lt.self_test()
        assert r["ok"] is False
        assert any("无法访问" in x for x in r["reasons"])

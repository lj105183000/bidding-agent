"""Exa MCP 客户端测试。

★ 全部用假 payload，不打真实网络
-----------------------------
Exa 托管服务是**免费限速**的（实测 5 次调用内就触限），
测试若打真实网络会：① 随机失败 ② 消耗额度 ③ 让 CI 不稳定。
故本文件只测**解析、错误识别、冷却、线程安全**这些本地逻辑。
真实连通性由 `logs/verify_exa_integration.py` 手工验证。

★ 为什么这些测试值得写
-------------------
本轮开发中踩了 3 个只有测试才能钉住的坑：
    ① 限速文案被当成搜索结果（伪装成一条结果，看起来像"重复"）
    ② 解析在 try 之外 → 限速异常绕过冷却，`available()` 恒为 True
    ③ 同一结果在 Exa 返回文本里出现两次，需要去重
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mcp import web_search_exa as M  # noqa: E402


def _text_payload(text: str, *, is_error: bool = False) -> dict:
    """构造一个 MCP tools/call 响应。"""
    return {
        "jsonrpc": "2.0",
        "id": 2,
        "result": {
            "content": [{"type": "text", "text": text}],
            "isError": is_error,
        },
    }


#: Exa 实际的排版文本格式（实测抄下来的）
FORMATTED = """Title: 招采信息
URL: http://www.ahtba.org.cn/site/trade/affiche/gotoTradeList
Published: N/A
Author: N/A
Highlights:
招采信息_安徽省招标投标信息网
邮箱：2609994999@qq.com

---

Title: 招标采购 - 安徽建工集团
URL: https://cg.aceg.com.cn/
Published: N/A
Author: N/A
Highlights:
- 【报名进行中】建工安装电梯公司加装电梯项目

---

Title: 招采详情
URL: https://jkv2.ahtba.org.cn/site/trade/affiche/detail/50326411
Published: N/A
Highlights:
项目区域：池州市
"""


class TestParseSseJson:
    """MCP 的 streamable HTTP 用 SSE 包装，直接 response.json() 会抛异常。"""

    def test_parses_sse_wrapped(self):
        raw = 'event: message\ndata: {"jsonrpc":"2.0","id":2,"result":{"ok":1}}\n\n'
        assert M._parse_sse_json(raw) == {"jsonrpc": "2.0", "id": 2, "result": {"ok": 1}}

    def test_parses_bare_json(self):
        assert M._parse_sse_json('{"a":1}') == {"a": 1}

    def test_parses_data_without_event_line(self):
        assert M._parse_sse_json('data: {"b":2}\n\n') == {"b": 2}

    def test_returns_none_on_garbage(self):
        assert M._parse_sse_json("") is None
        assert M._parse_sse_json("not json at all") is None

    def test_takes_first_valid_block(self):
        raw = 'event: ping\ndata: {"first":1}\n\nevent: message\ndata: {"second":2}\n\n'
        assert M._parse_sse_json(raw) == {"first": 1}


class TestRateLimitDetection:
    """★ 坑①：限速文案是**正常返回**，不是 JSON-RPC error，也不是 HTTP 429。

    Exa 免费限速时返回：
        "You've hit Exa's free MCP rate limit. ... create your own Exa API key."

    若不识别，它会被"兜底：整段作为一条结果"当成一条搜索结果——
    标题"Exa 搜索结果"、正文是限速提示。更糟的是连续调用拿到同一段文本，
    **看起来像"结果重复"，把排查引向解析逻辑**（实测因此白查一轮）。
    """

    @pytest.mark.parametrize(
        "text",
        [
            "You've hit Exa's free MCP rate limit. To continue using without limits,"
            " create your own Exa API key.",
            "Rate limit exceeded",
            "请创建 Exa API key 后重试",
        ],
    )
    def test_detects_rate_limit_text(self, text):
        with pytest.raises(M.ExaMCPError):
            M.ExaMCPClient._tool_text(_text_payload(text))

    def test_normal_text_passes(self):
        assert "招采信息" in M.ExaMCPClient._tool_text(_text_payload(FORMATTED))

    def test_is_error_flag_raises(self):
        with pytest.raises(M.ExaMCPError):
            M.ExaMCPClient._tool_text(_text_payload("boom", is_error=True))


class TestParseResults:
    def test_parses_formatted_text(self):
        client = M.ExaMCPClient()
        items = client._extract_results(_text_payload(FORMATTED))
        assert len(items) == 3
        assert items[0]["title"] == "招采信息"
        assert items[0]["url"].startswith("http://www.ahtba.org.cn")
        assert "安徽省招标投标信息网" in items[0]["content"]
        assert items[1]["title"] == "招标采购 - 安徽建工集团"

    def test_parses_json_array(self):
        payload = _text_payload(json.dumps([
            {"title": "A", "url": "https://a.example", "content": "内容A"},
            {"title": "B", "url": "https://b.example", "content": "内容B"},
        ]))
        items = M.ExaMCPClient()._extract_results(payload)
        assert [i["title"] for i in items] == ["A", "B"]

    def test_parses_json_object_with_results(self):
        payload = _text_payload(json.dumps(
            {"results": [{"title": "X", "url": "https://x.example", "text": "正文"}]}
        ))
        items = M.ExaMCPClient()._extract_results(payload)
        assert items[0]["title"] == "X"
        assert items[0]["content"] == "正文"

    def test_falls_back_to_whole_text(self):
        """解析不出结构化结果时不能丢内容——宁可给一条无标题的结果。"""
        payload = _text_payload("这是一段没有 Title/URL 的普通文本，但含 https://a.example 链接")
        items = M.ExaMCPClient()._extract_results(payload)
        assert len(items) == 1
        assert items[0]["url"] == "https://a.example"
        assert "普通文本" in items[0]["content"]

    def test_empty_text_gives_empty_list(self):
        assert M.ExaMCPClient()._extract_results(_text_payload("")) == []

    def test_normalize_item_requires_title_or_url(self):
        assert M.ExaMCPClient._normalize_item({"content": "只有正文"}) is None
        assert M.ExaMCPClient._normalize_item({"title": "T"}) is not None
        assert M.ExaMCPClient._normalize_item({"url": "https://a"}) is not None

    def test_normalize_item_accepts_alternative_keys(self):
        item = M.ExaMCPClient._normalize_item(
            {"name": "n", "link": "https://l", "snippet": "s"}
        )
        assert item == {"title": "n", "url": "https://l", "content": "s"}


class TestDedupe:
    """★ 坑③：同一结果在 Exa 返回文本里可能出现两次。"""

    def test_dedupes_by_title_and_url(self):
        items = [
            {"title": "A", "url": "https://a", "content": "短"},
            {"title": "A", "url": "https://a", "content": "更长的正文内容在这里"},
            {"title": "B", "url": "https://b", "content": "B 的正文"},
        ]
        out = M.ExaMCPClient._dedupe(items)
        assert len(out) == 2
        # 保留正文更长的那条
        assert out[0]["content"] == "更长的正文内容在这里"
        assert out[1]["title"] == "B"

    def test_preserves_first_seen_order(self):
        items = [
            {"title": "B", "url": "https://b", "content": "1"},
            {"title": "A", "url": "https://a", "content": "2"},
        ]
        assert [i["title"] for i in M.ExaMCPClient._dedupe(items)] == ["B", "A"]

    def test_drops_items_without_title_and_url(self):
        assert M.ExaMCPClient._dedupe([{"content": "x"}]) == []

    def test_tolerates_non_dict(self):
        assert M.ExaMCPClient._dedupe(["str", None, {"title": "T"}])  # type: ignore[list-item]

    def test_exact_duplicates_collapse(self):
        same = {"title": "A", "url": "https://a", "content": "同"}
        assert len(M.ExaMCPClient._dedupe([same, dict(same)])) == 1


class TestCooldown:
    """★ 坑②：解析在 try 之外 → 限速异常绕过冷却。

    `search()` 第一版写成：

        try:
            payload = self._call_tool(...)      # try 只护到网络调用
        except ExaMCPError:
            self._enter_cooldown(); raise
        return {"results": self._extract_results(payload), ...}   # ← 异常从这里逃出

    而**限速是在解析阶段被识别的**，于是 `_enter_cooldown` 永不执行，
    `available()` 恒为 True —— 表现为"限速后每个问题都重新等一次超时"。
    """

    def test_rate_limit_enters_cooldown(self):
        c = M.ExaMCPClient()
        c._call_tool = lambda name, args: _text_payload("rate limit hit")  # type: ignore[method-assign]
        with pytest.raises(M.ExaMCPError):
            c.search("q")
        assert c.available() is False, "限速后必须进入冷却"

    def test_cooldown_blocks_network_call(self):
        """冷却期内不该再发请求——否则每次都要白等一次超时。"""
        c = M.ExaMCPClient()
        calls = {"n": 0}

        def fake(name, args):
            calls["n"] += 1
            return _text_payload("rate limit hit")

        c._call_tool = fake  # type: ignore[method-assign]
        with pytest.raises(M.ExaMCPError):
            c.search("q1")
        assert calls["n"] == 1
        with pytest.raises(M.ExaMCPError):
            c.search("q2")
        assert calls["n"] == 1, "冷却期内仍发出了网络请求"

    def test_exception_in_extract_also_cools(self):
        """★ 回归哨兵：解析阶段抛错也必须触发冷却（就是上面那个 bug）。"""
        c = M.ExaMCPClient()
        c._call_tool = lambda name, args: {"result": {"content": [], "isError": True}}  # type: ignore[method-assign]
        with pytest.raises(M.ExaMCPError):
            c.search("q")
        assert c.available() is False

    def test_cooldown_can_be_bypassed_by_new_instance(self):
        """冷却状态是实例级的——测试与诊断需要新建实例来绕开。"""
        a = M.ExaMCPClient()
        b = M.ExaMCPClient()
        a._enter_cooldown()
        assert a.available() is False
        assert b.available() is True

    def test_empty_query_does_not_enter_cooldown(self):
        """空查询是调用方错误，不该污染冷却状态。"""
        c = M.ExaMCPClient()
        with pytest.raises(M.ExaMCPError):
            c.search("")
        assert c.available() is True


class TestSearchContract:
    """`rag_tools.search_exa` 约定的返回形状不能变。"""

    @staticmethod
    def _client(payload: dict) -> M.ExaMCPClient:
        c = M.ExaMCPClient()
        c._call_tool = lambda name, args: payload  # type: ignore[method-assign]
        return c

    def test_returns_results_and_answer_keys(self):
        c = self._client(_text_payload(FORMATTED))
        out = c.search("问题", num_results=5)
        assert set(out) >= {"results", "answer"}
        assert isinstance(out["results"], list)
        assert out["answer"] == "", "Exa 不返回 answer，约定为空串"

    def test_results_have_title_url_content(self):
        c = self._client(_text_payload(FORMATTED))
        for item in c.search("问题")["results"]:
            assert set(item) == {"title", "url", "content"}

    def test_clamps_num_results(self):
        seen = {}

        def fake(name, args):
            seen.update(args)
            return _text_payload(FORMATTED)

        c = M.ExaMCPClient()
        c._call_tool = fake  # type: ignore[method-assign]
        c.search("q", num_results=999)
        assert seen["numResults"] == M.MAX_RESULTS
        c._call_tool = fake  # type: ignore[method-assign]
        c.search("q", num_results=0)
        assert seen["numResults"] == 1

    def test_uses_web_search_exa_tool_name(self):
        seen = {}

        def fake(name, args):
            seen["name"] = name
            return _text_payload(FORMATTED)

        c = M.ExaMCPClient()
        c._call_tool = fake  # type: ignore[method-assign]
        c.search("q")
        assert seen["name"] == "web_search_exa"


class TestFetch:
    def test_fetch_extracts_content(self):
        payload = _text_payload(json.dumps(
            {"results": [{"url": "https://a", "content": "全文正文"}]}
        ))
        c = M.ExaMCPClient()
        c._call_tool = lambda name, args: payload  # type: ignore[method-assign]
        out = c.fetch(["https://a"])
        assert out["results"][0]["content"] == "全文正文"

    def test_fetch_rejects_empty_urls(self):
        with pytest.raises(M.ExaMCPError):
            M.ExaMCPClient().fetch([])


class TestConfig:
    def test_default_url(self, monkeypatch):
        monkeypatch.delenv("EXA_MCP_URL", raising=False)
        assert M.ExaMCPClient()._url == M.DEFAULT_MCP_URL

    def test_url_from_env(self, monkeypatch):
        monkeypatch.setenv("EXA_MCP_URL", "https://self.hosted/mcp")
        assert M.ExaMCPClient()._url == "https://self.hosted/mcp"

    def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("EXA_API_KEY", "k123")
        headers = M.ExaMCPClient()._headers(with_session=False)
        assert headers["x-api-key"] == "k123"

    def test_no_api_key_header_when_unset(self, monkeypatch):
        monkeypatch.delenv("EXA_API_KEY", raising=False)
        headers = M.ExaMCPClient()._headers(with_session=False)
        assert "x-api-key" not in headers

    def test_session_header_included_when_set(self):
        c = M.ExaMCPClient()
        c._session_id = "sess-1"
        assert c._headers()["mcp-session-id"] == "sess-1"

    def test_timeout_below_global_read_timeout(self):
        """★ 单次调用超时必须小于全局 60s，否则会拖住整轮问答。

        项目对单次搜索的预算是 `web_search.SEARCH_TIMEOUT_SECONDS = 25`。
        """
        from src.http_client import HTTP_TIMEOUT
        from src.web_search import SEARCH_TIMEOUT_SECONDS

        assert M.CALL_TIMEOUT_SECONDS <= SEARCH_TIMEOUT_SECONDS
        assert M.CALL_TIMEOUT_SECONDS < float(HTTP_TIMEOUT.read)


class TestResultCache:
    """结果缓存。

    ★ 为什么 Exa 比 Tavily/智谱**更**需要缓存
    -------------------------------------
    Tavily 与智谱用付费 key，重复调用只是多花钱；而 Exa 走 `mcp.exa.ai`
    的**免费限速**服务——实测 5 次调用内就触限（docs/38）。没有缓存意味着
    同一轮问答里模型重复用同一句话调 `search_exa` 会直接把额度打光，
    然后进入 300 秒冷却，该数据源彻底不可用。
    """

    @staticmethod
    def _client(text: str = FORMATTED) -> tuple[M.ExaMCPClient, dict]:
        calls = {"n": 0}

        def fake(name, args):
            calls["n"] += 1
            return _text_payload(text)

        c = M.ExaMCPClient()
        c._call_tool = fake  # type: ignore[method-assign]
        return c, calls

    def test_second_call_hits_cache(self):
        c, calls = self._client()
        c.search("招标公告", num_results=5)
        c.search("招标公告", num_results=5)
        assert calls["n"] == 1, "第二次相同查询不该再调底层"
        assert c.cache_info()["hits"] == 1

    def test_strip_and_case_variants_share_entry(self):
        """`q` 已 strip 过，键再 lower —— 空格与大小写变体应命中同一项。"""
        c, calls = self._client()
        c.search("招标公告", num_results=5)
        c.search("  招标公告  ", num_results=5)
        c.search("招标公告".upper(), num_results=5)
        assert calls["n"] == 1

    def test_limit_is_part_of_key(self):
        """★ 与 Tavily/智谱**不同**：这里的条数直接决定返回几条。

        键里不含 `limit` 的话，先查 3 条再查 5 条会命中同一项，
        **拿到 3 条却以为要的是 5 条**——不报错，只静默少给数据。
        """
        c, calls = self._client()
        c.search(" q ", num_results=5)
        c.search("q", num_results=3)
        assert calls["n"] == 2, "不同 limit 必须是不同的缓存项"

    def test_empty_result_content_is_cached_too(self):
        """正常返回但解析出空列表也算成功——不该反复重试消耗额度。"""
        c, calls = self._client(text="")
        c.search("q")
        c.search("q")
        assert calls["n"] == 1

    def test_failure_not_cached(self):
        """★ 失败结果**绝不进缓存**。

        Exa 是限速服务，把"限速/超时"缓存 10 分钟等于把故障固化，
        反而更难恢复（与 `web_search` 的取舍一致）。
        """
        calls = {"n": 0}

        def boom(name, args):
            calls["n"] += 1
            raise RuntimeError("网络炸了")

        c = M.ExaMCPClient()
        c._call_tool = boom  # type: ignore[method-assign]
        with pytest.raises(M.ExaMCPError):
            c.search("q")
        assert c.cache_info()["size"] == 0, "失败被缓存了"

        # ★ 必须清冷却才能验证"第二次会真的再试一次"。
        # 不清的话第二次被 300 秒冷却拦住，`calls` 仍是 1 —— 那验证的是
        # "冷却生效"而不是"失败没进缓存"（实测被这一点绊过）。
        c.clear_cooldown()
        with pytest.raises(M.ExaMCPError):
            c.search("q")
        assert calls["n"] == 2, "失败后应真实重试，而不是被缓存挡住"
        assert c.cache_info()["size"] == 0

    def test_hit_returns_deep_copy(self):
        """★ 命中必须返回深拷贝。

        `rag_tools._rerank_web_results` 会往结果里写 `score` / `rerank_score`，
        共享同一份对象会把改写带进缓存，后续命中就拿到被污染的数据。
        """
        c, _ = self._client()
        r1 = c.search("q", num_results=5)
        baseline = len(r1["results"])
        original = r1["results"][0]["content"]
        # 三种改写都要验证：改字段值、改嵌套、改列表结构
        r1["results"][0]["score"] = 999.0
        r1["results"][0]["content"] = "被改过"
        r1["results"].append({"title": "注入", "url": "", "content": ""})
        assert len(r1["results"]) == baseline + 1

        r2 = c.search("q", num_results=5)
        assert r2["results"][0]["content"] == original, "缓存被改写污染"
        assert r2["results"][0].get("score") is None
        assert len(r2["results"]) == baseline, "对返回列表的追加污染了缓存"

    def test_ttl_expiry(self, monkeypatch):
        monkeypatch.setattr(M, "CACHE_TTL_SECONDS", 0.05)
        c, calls = self._client()
        c.search("q")
        c.search("q")
        assert calls["n"] == 1
        import time

        time.sleep(0.15)
        c.search("q")
        assert calls["n"] == 2, "TTL 过后应重新请求"

    def test_capacity_bounded(self, monkeypatch):
        """容量有界——无界容器在长跑服务里就是内存泄漏。"""
        monkeypatch.setattr(M, "CACHE_MAX_ITEMS", 5)
        c, _ = self._client()
        for i in range(30):
            c.search(f"q{i}")
        assert c.cache_info()["size"] <= 5

    def test_clear_cache(self):
        c, calls = self._client()
        c.search("q")
        c.clear_cache()
        info = c.cache_info()
        assert info["size"] == 0 and info["hits"] == 0 and info["misses"] == 0
        c.search("q")
        assert calls["n"] == 2

    def test_cache_info_shape(self):
        info = M.ExaMCPClient().cache_info()
        for key in ("size", "max_items", "ttl_seconds", "hits", "misses", "hit_rate"):
            assert key in info
        assert info["hit_rate"] == 0.0

    def test_instances_are_isolated(self):
        """不同实例的缓存互不影响（测试与多实例场景的前提）。"""
        a, ca = self._client()
        b, cb = self._client()
        a.search("q")
        b.search("q")
        assert ca["n"] == 1 and cb["n"] == 1


class TestThreadSafety:
    def test_concurrent_search_does_not_crash(self):
        """工具层是 4 线程并行执行，单例会被并发调用。"""
        c = M.ExaMCPClient()
        c._call_tool = lambda name, args: _text_payload(FORMATTED)  # type: ignore[method-assign]
        errors: list[Exception] = []

        def worker():
            try:
                c.search("q", num_results=3)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, f"并发调用出错: {errors[:2]}"

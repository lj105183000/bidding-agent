"""缓存状态与清除端点的测试。

★ 为什么用 TestClient 而不是纯函数测试
--------------------------------
这两个端点的价值就在于"**把四处分散的缓存汇总成一个视图**"，
若绕开应用单测各段，就验证不了"汇总是否真的接上了"。
（`docs/48` 那次的教训：抽纯函数能避开重依赖，但也可能漏掉装配错误。）

导入 `api.server` 会拉起 Agent/RAG，较慢，但本项目既有测试已经这么做
（`test_api_server.py`），保持一致。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    import api.server as srv

    with TestClient(srv.app) as c:
        yield c


class TestCacheStatus:
    def test_returns_all_four_caches(self, client):
        """★ 四个缓存都要出现——这是本端点的全部价值。

        项目里有四种缓存（检索 / Tavily / 智谱 / Exa），此前没有任何地方
        能一眼看到它们的状态。若某一段因为 import 失败而缺失，
        "汇总视图"就名不副实。
        """
        r = client.get("/api/cache")
        assert r.status_code == 200
        d = r.json()
        for key in ("retrieval", "web_search_tavily", "web_search_zhipu", "web_search_exa"):
            assert key in d, f"缺 {key} 段"

    def test_no_section_reports_error(self, client):
        """正常情况下每段都应有真实数据，而不是 error。"""
        d = client.get("/api/cache").json()
        for key, val in d.items():
            assert "error" not in val, f"{key} 取数失败: {val.get('error')}"

    def test_exa_section_has_hit_rate(self, client):
        """Exa 是本次新增统计的，应给出命中率。"""
        exa = client.get("/api/cache").json()["web_search_exa"]
        assert "hit_rate" in exa
        assert "ttl_seconds" in exa
        assert exa["ttl_seconds"] > 0

    def test_tavily_and_zhipu_report_ttl(self, client):
        d = client.get("/api/cache").json()
        for key in ("web_search_tavily", "web_search_zhipu"):
            assert d[key]["ttl_seconds"] == 600.0, f"{key} 的 TTL 应为 600s"

    def test_missing_hit_rate_is_null_not_zero(self, client):
        """★ 取不到命中率要返回 None，**不编造 0**。

        `web_search` / `web_search_zhipu` 没有命中计数（本轮只给 Exa 加了）。
        返回 0 会让人以为"缓存一次都没命中"，而实际是"没统计"。
        """
        d = client.get("/api/cache").json()
        assert d["web_search_tavily"]["hit_rate"] is None
        assert d["web_search_zhipu"]["hit_rate"] is None


class TestCacheClear:
    def test_clear_all(self, client):
        r = client.delete("/api/cache?scope=all")
        assert r.status_code == 200
        d = r.json()
        assert d["ok"] is True
        assert set(d["cleared"]) >= {"retrieval", "web_search_exa"}

    def test_clear_retrieval_only(self, client):
        d = client.delete("/api/cache?scope=retrieval").json()
        assert d["cleared"] == ["retrieval"]

    def test_clear_web_only(self, client):
        d = client.delete("/api/cache?scope=web").json()
        assert "retrieval" not in d["cleared"]
        assert "web_search_exa" in d["cleared"]

    def test_unknown_scope_clears_nothing(self, client):
        """未知 scope 不该被当作 all —— 静默清空全部是危险行为。"""
        d = client.delete("/api/cache?scope=banana").json()
        assert d["cleared"] == []

    def test_clear_then_exa_size_is_zero(self, client):
        """端到端：清空后 Exa 缓存确实为 0。"""
        from src.mcp.web_search_exa import exa_search_client

        exa_search_client._cache[(("x"), 3)] = (1e18, {"results": []})
        assert client.get("/api/cache").json()["web_search_exa"]["size"] >= 1
        client.delete("/api/cache?scope=web")
        assert client.get("/api/cache").json()["web_search_exa"]["size"] == 0

"""前端静态托管（同源部署）的测试。

★ 为什么这些用例值得写
==================
这条兜底路由有**两个只会在部署后才暴露**的坑，dev server 都会替我们掩盖：

    ① `/chat` 这类前端路由在服务器上没有对应文件 → 不兜底就 404，
       用户刷新子页面直接白屏（dev server 自带兜底，本地永远看不到）
    ② 兜底路由若注册在 `/api/*` 之前 → 把接口全部吃掉，
       表现为"前端正常、接口全 404"，且错误信息毫无线索
另外缓存头也不能错：`index.html` 一旦被缓存，发版后用户会一直请求
已经不存在的旧 chunk。
"""

import importlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import api.server as server_mod


@pytest.fixture()
def spa_client(tmp_path, monkeypatch):
    """用临时目录冒充前端构建产物，重载模块以走"已挂载"分支。"""
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><div id=root></div>", encoding="utf-8")
    (dist / "assets" / "index-abc123.js").write_text("console.log(1)", encoding="utf-8")
    (dist / "logo.svg").write_text("<svg/>", encoding="utf-8")

    monkeypatch.setenv("FRONTEND_DIST", str(dist))
    reloaded = importlib.reload(server_mod)
    with TestClient(reloaded.app) as client:
        yield client, dist
    monkeypatch.delenv("FRONTEND_DIST", raising=False)
    importlib.reload(server_mod)


def test_spa_route_returns_index_html(spa_client):
    """★ 前端路由（服务器上无此文件）必须回 index.html，而不是 404。"""
    client, _dist = spa_client
    for path in ("/", "/chat", "/analysis", "/qualification/deep/link"):
        resp = client.get(path)
        assert resp.status_code == 200, f"{path} 未兜底（刷新子页面会白屏）"
        assert "<div id=root>" in resp.text


def test_index_html_is_not_cached(spa_client):
    """index.html 不能被缓存：否则发版后用户一直请求已删除的旧 chunk。"""
    client, _dist = spa_client
    assert "no-cache" in client.get("/chat").headers.get("cache-control", "")


def test_real_files_are_served(spa_client):
    """真实存在的文件（logo/favicon 等）按文件返回，不走兜底。"""
    client, _dist = spa_client
    resp = client.get("/logo.svg")
    assert resp.status_code == 200
    assert "<svg/>" in resp.text


def test_hashed_assets_are_served(spa_client):
    """带哈希的资源由 StaticFiles 提供。"""
    client, _dist = spa_client
    resp = client.get("/assets/index-abc123.js")
    assert resp.status_code == 200
    assert "console.log" in resp.text


def test_api_404_is_not_swallowed_by_spa_fallback(spa_client):
    """★ `/api/*` 不存在的路径必须返回 404，不能被 SPA 兜底吞掉。

    否则前端会把"接口 404"当成"页面正常"，错误被伪装成成功。
    """
    client, _dist = spa_client
    resp = client.get("/api/definitely-not-a-route")
    assert resp.status_code == 404


def test_existing_api_routes_still_work(spa_client):
    """兜底路由不能把已注册的接口吃掉（注册顺序的回归）。"""
    client, _dist = spa_client
    resp = client.get("/api/health/live")
    assert resp.status_code in (200, 503), "健康检查被 SPA 兜底吃了"


def test_not_mounted_when_dist_missing(tmp_path, monkeypatch):
    """dist 不存在时完全不挂载（开发时不该看到过期页面）。"""
    monkeypatch.setenv("FRONTEND_DIST", str(tmp_path / "nope"))
    reloaded = importlib.reload(server_mod)
    assert reloaded._FRONTEND_READY is False
    with TestClient(reloaded.app) as client:
        assert client.get("/chat").status_code == 404
    monkeypatch.delenv("FRONTEND_DIST", raising=False)
    importlib.reload(server_mod)


def test_explicit_off_disables_mounting(tmp_path, monkeypatch):
    """`FRONTEND_DIST=off` 显式关闭（部署时不想要 SPA 兜底时用）。"""
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("x", encoding="utf-8")
    monkeypatch.setenv("FRONTEND_DIST", "off")
    reloaded = importlib.reload(server_mod)
    assert reloaded._FRONTEND_READY is False
    monkeypatch.delenv("FRONTEND_DIST", raising=False)
    importlib.reload(server_mod)


def test_path_traversal_is_not_served(spa_client):
    """路径穿越不能读到 dist 之外的文件（回 index.html 即可）。"""
    client, dist = spa_client
    secret = dist.parent / "secret.txt"
    secret.write_text("TOP-SECRET", encoding="utf-8")
    resp = client.get("/../secret.txt")
    assert "TOP-SECRET" not in resp.text


class TestDefaultPathResolution:
    """★ 默认路径解析的回归测试（真实事故：写死相对层级 → 静默不托管）。

    事故经过：第一版写死 `parents[1] / "frontend" / "dist"`，而本机前端在
    **工程根的上一级**（`bidding-agent/frontend`；工程根是
    `bidding-agent/bidding-agent`）。解析结果指向一个不存在的目录 →
    因为"dist 不存在就完全不挂载"的策略，**静默不生效**：用户按说明打开
    8001 看到 404，现象就是"前端打不开"，而日志里只有一行 INFO。

    原有用例全部通过 `FRONTEND_DIST` 环境变量指定目录，**没有覆盖**默认解析
    这条路径 —— 这正是当初漏掉它的原因。
    """

    def test_candidates_cover_both_layouts(self, monkeypatch):
        """候选里必须同时含"与后端同目录"和"工程根上一级"两种前端布局。"""
        import importlib

        from api import server as server_mod

        monkeypatch.delenv("FRONTEND_DIST", raising=False)
        reloaded = importlib.reload(server_mod)
        try:
            names = [str(p).replace("\\", "/") for p in reloaded._FRONTEND_CANDIDATES]
            assert any(p.endswith("bidding-agent/frontend/dist") for p in names), (
                f"缺少『工程根上一级』布局的候选 —— 本机前端就在这里：{names}"
            )
            assert len(names) >= 2, f"应探测多个候选而不是写死一个：{names}"
        finally:
            monkeypatch.delenv("FRONTEND_DIST", raising=False)
            importlib.reload(server_mod)

    def test_env_override_short_circuits(self, tmp_path, monkeypatch):
        """显式指定 FRONTEND_DIST 时只认真实值，不再探测其它候选。"""
        import importlib

        from api import server as server_mod

        target = tmp_path / "custom-dist"
        (target / "assets").mkdir(parents=True)
        (target / "index.html").write_text("<div id=root></div>", encoding="utf-8")
        monkeypatch.setenv("FRONTEND_DIST", str(target))
        reloaded = importlib.reload(server_mod)
        try:
            assert reloaded._FRONTEND_READY is True
            assert reloaded._FRONTEND_DIST == target
            assert reloaded._FRONTEND_CANDIDATES == [target], "指定了就不该再猜"
        finally:
            monkeypatch.delenv("FRONTEND_DIST", raising=False)
            importlib.reload(server_mod)

    def test_warning_lists_probed_paths(self, tmp_path, monkeypatch, caplog):
        """找不到时必须在日志里列出**探测过的路径** —— 否则『配置错』与
        『没构建』在日志里长得一样，只能靠猜。"""
        import importlib
        import logging

        from api import server as server_mod

        monkeypatch.setenv("FRONTEND_DIST", str(tmp_path / "does-not-exist"))
        with caplog.at_level(logging.WARNING, logger="api.server"):
            reloaded = importlib.reload(server_mod)
            try:
                assert reloaded._FRONTEND_READY is False
            finally:
                monkeypatch.delenv("FRONTEND_DIST", raising=False)
                importlib.reload(server_mod)
        messages = " ".join(record.getMessage() for record in caplog.records)
        assert "does-not-exist" in messages, "告警里要带上探测过的路径"

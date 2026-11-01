"""数据质量扫描的规则注册表与聚合逻辑测试。

不连任何后端：用 fake `QualityContext` / 自定义规则验证
“单条失败只 skip”“overall 汇总”“规则资产必须已登记”。
"""

from src.datahub import quality as Q
from src.datahub.registry import ASSET_INDEX


def test_rule_names_unique():
    names = [rule.name for rule in Q.RULES]
    assert len(names) == len(set(names))


def test_rule_assets_are_registered():
    for rule in Q.RULES:
        assert rule.asset == "catalog" or rule.asset in ASSET_INDEX, (
            f"规则 {rule.name} 引用了未登记资产 {rule.asset}"
        )


def test_rule_severity_is_valid():
    for rule in Q.RULES:
        assert rule.severity in {"warn", "fail"}


def _context_that_fails() -> Q.QualityContext:
    def boom(*args, **kwargs):
        raise RuntimeError("backend down")

    return Q.QualityContext(
        pg_rows=boom,
        neo4j_rows=boom,
        qdrant_count=boom,
        qdrant_exists=boom,
    )


def test_scan_never_raises_when_backend_down():
    report = Q.run_quality_scan(persist=False, context=_context_that_fails())
    assert report["overall"] in {"degraded", "fail"}
    assert report["summary"]["pass"] == 0
    assert report["summary"]["skip"] >= 1
    assert len(report["checks"]) == len(Q.RULES)


def _rule(name: str, status: str):
    def check(ctx):
        if status == "boom":
            raise RuntimeError("boom")
        return {"status": status, "value": 1, "threshold": 1, "message": name}

    return Q.QualityRule(name, "catalog", "fail", name, check)


def test_scan_aggregates_statuses(monkeypatch):
    monkeypatch.setattr(
        Q,
        "RULES",
        (
            _rule("a.pass", "pass"),
            _rule("b.warn", "warn"),
            _rule("c.boom", "boom"),
        ),
    )
    report = Q.run_quality_scan(persist=False, context=Q.QualityContext())
    assert report["summary"] == {"pass": 1, "warn": 1, "fail": 0, "skip": 1}
    assert report["overall"] == "degraded"
    assert report["checks"][2]["status"] == "skip"
    assert "RuntimeError" in report["checks"][2]["message"]


def test_scan_fail_overrides_warn(monkeypatch):
    monkeypatch.setattr(
        Q, "RULES", (_rule("a.fail", "fail"), _rule("b.warn", "warn"))
    )
    report = Q.run_quality_scan(persist=False, context=Q.QualityContext())
    assert report["overall"] == "fail"


def test_scan_persists_when_requested(monkeypatch):
    calls = {}

    def fake_save(report, source="manual"):
        calls["source"] = source
        calls["overall"] = report["overall"]
        return 7

    monkeypatch.setattr(
        "src.datahub.quality_store.save_report", fake_save, raising=False
    )
    monkeypatch.setattr(Q, "RULES", (_rule("a.pass", "pass"),))
    report = Q.run_quality_scan(source="scheduler", persist=True)
    assert report["report_id"] == 7
    assert calls == {"source": "scheduler", "overall": "ok"}


def test_latest_report_delegates(monkeypatch):
    monkeypatch.setattr(
        "src.datahub.quality_store.latest_report",
        lambda: {"overall": "ok", "checks": []},
        raising=False,
    )
    assert Q.latest_report()["overall"] == "ok"

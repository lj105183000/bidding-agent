"""CLI 入口的安全守卫测试。

`python main.py all` 是“最容易顺手执行”的破坏性入口：
它会 ingest（删集合重建）+ pg（DROP 重建）。因此要求显式 `--force`。
本文件只测分发与参数透传，不启动 uvicorn、不跑真实构建。
"""

import sys

import pytest

import main as M


def test_all_requires_force(monkeypatch):
    calls: list[tuple] = []
    monkeypatch.setattr(sys, "argv", ["main.py", "all"])
    monkeypatch.setattr(M, "setup_logging", lambda: None)
    monkeypatch.setattr(M, "_ingest", lambda args: calls.append(("ingest", args)) or 0)
    monkeypatch.setattr(M, "_run_builder", lambda *args: calls.append(("builder", args)) or 0)

    with pytest.raises(SystemExit) as exc:
        M.main()
    assert exc.value.code == 1
    assert calls == []


def test_all_passes_force_to_pg_and_strips_it_from_ingest(monkeypatch):
    calls: list[tuple] = []
    monkeypatch.setattr(sys, "argv", ["main.py", "all", "--force"])
    monkeypatch.setattr(M, "setup_logging", lambda: None)

    def fake_ingest(args):
        calls.append(("ingest", list(args)))
        return 0

    def fake_builder(script, extra, title):
        calls.append((script, list(extra)))
        return 0

    monkeypatch.setattr(M, "_ingest", fake_ingest)
    monkeypatch.setattr(M, "_run_builder", fake_builder)

    with pytest.raises(SystemExit) as exc:
        M.main()
    assert exc.value.code == 0
    assert calls[0] == ("ingest", [])
    assert calls[-1] == ("to_postgresql.py", ["--force"])


def test_pg_passthrough(monkeypatch):
    calls: list[tuple] = []
    monkeypatch.setattr(sys, "argv", ["main.py", "pg", "--force", "--dry-run"])
    monkeypatch.setattr(M, "setup_logging", lambda: None)
    monkeypatch.setattr(
        M,
        "_run_builder",
        lambda script, extra, title: calls.append((script, list(extra))) or 0,
    )

    with pytest.raises(SystemExit) as exc:
        M.main()
    assert exc.value.code == 0
    assert calls == [("to_postgresql.py", ["--force", "--dry-run"])]


def test_unknown_command_mentions_force(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["main.py", "nope"])
    monkeypatch.setattr(M, "setup_logging", lambda: None)

    with pytest.raises(SystemExit) as exc:
        M.main()
    assert exc.value.code == 1
    output = capsys.readouterr().out
    assert "--force" in output

"""batch/提交batch任务.py 的提交/续用/下载逻辑。

脚本名带中文，用 importlib 按路径加载——直接 import 会因为模块名不是合法标识符而失败。

不打网络：本脚本是对着平台 API 的薄封装，可测的是它自己的判断——什么时候**不该**重新
提交（重复提交 = 8751 次真实计费）、终态怎么分支、结果按什么名字落盘。用文件内局部
fake client（§9.3 约定），只实现被调到的三个方法。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "batch" / "提交batch任务.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("batch_submit", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


submit_mod = _load_module()


# ---- fake client ----


class _Counts:
    def __init__(self, total=0, completed=0, failed=0):
        self.total, self.completed, self.failed = total, completed, failed


class _Batch:
    def __init__(self, batch_id="batch_1", status="completed", output_file_id="file_out",
                 error_file_id=None, counts=None, statuses=None):
        self.id = batch_id
        self.status = status
        self.output_file_id = output_file_id
        self.error_file_id = error_file_id
        self.request_counts = counts or _Counts()
        # 多次 retrieve 依次返回不同状态，用来模拟 validating → in_progress → completed
        self._statuses = list(statuses or [])
        self.retrieve_calls = 0

    def next_status(self):
        if self._statuses:
            self.status = self._statuses.pop(0)
        return self


class _FileObject:
    def __init__(self, file_id="file_in"):
        self.id = file_id


class _Content:
    def __init__(self, payload: bytes):
        self.content = payload


class _FakeFiles:
    def __init__(self):
        self.uploaded: list[str] = []
        self.deleted: list[str] = []

    def create(self, *, file, purpose):
        self.uploaded.append(getattr(file, "name", "?"))
        return _FileObject(f"file_{len(self.uploaded)}")

    def content(self, file_id, **kwargs):
        return _Content(f'{{"custom_id":"request-1","file":"{file_id}"}}\n'.encode("utf-8"))

    def delete(self, file_id):
        self.deleted.append(file_id)


class _FakeBatches:
    def __init__(self, batch=None):
        self.created: list[dict] = []
        self.batch = batch or _Batch()
        self.retrieved: list[str] = []

    def create(self, **kwargs):
        self.created.append(kwargs)
        return self.batch

    def retrieve(self, batch_id, **kwargs):
        self.retrieved.append(batch_id)
        return self.batch.next_status()


class _FakeClient:
    def __init__(self, batch=None):
        self.files = _FakeFiles()
        self.batches = _FakeBatches(batch)


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    """make_client() 会校验密钥存在，测试里给个占位值。"""
    monkeypatch.setenv("ZHIPU_API_KEY", "test-key")


def _src(tmp_path: Path, name="batch_requests.jsonl") -> Path:
    path = tmp_path / name
    path.write_text(
        json.dumps({"custom_id": "request-1", "method": "POST",
                    "url": "/v4/chat/completions",
                    "body": {"model": "glm-4-flashx-250414", "messages": []}},
                   ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


# ---- 状态文件 ----


def test_state_roundtrip(tmp_path):
    path = tmp_path / "job.json"
    submit_mod.save_state(path, {"batch_id": "batch_1", "endpoint": "/v4/chat/completions"})
    assert submit_mod.load_state(path)["batch_id"] == "batch_1"


def test_load_state_of_missing_file_is_empty(tmp_path):
    assert submit_mod.load_state(tmp_path / "nope.json") == {}


def test_load_state_of_broken_json_is_empty(tmp_path):
    """状态文件坏了不该让整件事卡死——当作没有记录，走"重新提交"的显式分支。"""
    path = tmp_path / "job.json"
    path.write_text("{坏了", encoding="utf-8")
    assert submit_mod.load_state(path) == {}


def test_read_model_from_requests(tmp_path):
    assert submit_mod.read_model_from_requests(_src(tmp_path)) == "glm-4-flashx-250414"


def test_read_model_from_requests_tolerates_garbage(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text("不是 JSON\n", encoding="utf-8")
    assert submit_mod.read_model_from_requests(path) is None


# ---- 描述与终止态 ----


def test_describe_includes_progress():
    line = submit_mod.describe(_Batch(status="in_progress", counts=_Counts(10, 4, 1)))
    assert "in_progress" in line and "4/10" in line and "1" in line


def test_describe_without_counts_is_just_status():
    """平台刚建完任务时可能还没有 request_counts，此时只报状态，别显示 0/0 误导人。"""
    batch = _Batch(status="validating")
    batch.request_counts = None
    assert submit_mod.describe(batch) == "validating"


def test_wait_returns_on_terminal_status(tmp_path):
    batch = _Batch(status="validating", statuses=["in_progress", "completed"])
    client = _FakeClient(batch)
    done = submit_mod.wait_for(client, "batch_1", timeout=5, interval=0)
    assert done.status == "completed"


def test_wait_gives_up_on_timeout_but_reports_batch_id(capsys):
    """超时只是本脚本放弃等待，任务在平台上照跑，batch_id 已落盘。"""
    client = _FakeClient(_Batch(status="in_progress"))
    batch = submit_mod.wait_for(client, "batch_42", timeout=0, interval=0)
    out = capsys.readouterr().out
    assert batch.status == "in_progress"
    assert "batch_42" in out


def test_wait_returns_on_failed_status():
    client = _FakeClient(_Batch(status="failed", statuses=["failed"]))
    assert submit_mod.wait_for(client, "batch_1", timeout=5, interval=0).status == "failed"


# ---- 下载 ----


def test_fetch_results_downloads_output(tmp_path):
    client = _FakeClient()
    dest, err = tmp_path / "out.jsonl", tmp_path / "err.jsonl"
    assert submit_mod.fetch_results(client, _Batch(), dest, err) == 0
    assert "file_out" in dest.read_text(encoding="utf-8")
    assert not err.exists()


def test_fetch_results_downloads_error_file_when_present(tmp_path):
    """失败请求走 error_file_id 单独返回，不混在结果里——不下载就等于悄悄丢数据。"""
    client = _FakeClient()
    dest, err = tmp_path / "out.jsonl", tmp_path / "err.jsonl"
    assert submit_mod.fetch_results(client, _Batch(error_file_id="file_err"), dest, err) == 0
    assert err.read_text(encoding="utf-8")


def test_fetch_results_refuses_failed_batch(tmp_path, capsys):
    client = _FakeClient()
    assert submit_mod.fetch_results(client, _Batch(status="failed"), tmp_path / "o.jsonl",
                                    tmp_path / "e.jsonl") == 1
    assert "failed" in capsys.readouterr().out


def test_fetch_results_refuses_batch_without_output(tmp_path, capsys):
    client = _FakeClient()
    assert submit_mod.fetch_results(client, _Batch(output_file_id=None), tmp_path / "o.jsonl",
                                    tmp_path / "e.jsonl") == 1
    assert "output_file_id" in capsys.readouterr().out


# ---- main ----


def test_main_uploads_and_creates(tmp_path, monkeypatch, capsys):
    client = _FakeClient(_Batch(batch_id="batch_new", status="completed"))
    monkeypatch.setattr(submit_mod, "make_client", lambda: client)
    src, state = _src(tmp_path), tmp_path / "job.json"

    code = submit_mod.main(["--src", str(src), "-o", str(tmp_path / "out.jsonl"),
                            "--state", str(state), "--interval", "0"])
    assert code == 0
    assert len(client.files.uploaded) == 1
    assert client.batches.created[0]["endpoint"] == submit_mod.DEFAULT_ENDPOINT
    assert submit_mod.load_state(state)["batch_id"] == "batch_new"
    assert (tmp_path / "out.jsonl").exists()


def test_main_upload_uses_batch_purpose(tmp_path, monkeypatch):
    """purpose 必须是 batch，否则建任务时会被拒。"""
    seen = {}

    class _Files(_FakeFiles):
        def create(self, *, file, purpose):
            seen["purpose"] = purpose
            return super().create(file=file, purpose=purpose)

    client = _FakeClient()
    client.files = _Files()
    monkeypatch.setattr(submit_mod, "make_client", lambda: client)

    submit_mod.main(["--src", str(_src(tmp_path)), "-o", str(tmp_path / "o.jsonl"),
                     "--state", str(tmp_path / "job.json"), "--interval", "0"])
    assert seen["purpose"] == "batch"


def test_main_reuses_recorded_batch_instead_of_resubmitting(tmp_path, monkeypatch, capsys):
    """重复提交 = 8751 次真实计费，默认必须续用已有任务。"""
    client = _FakeClient(_Batch(batch_id="batch_old", status="completed"))
    monkeypatch.setattr(submit_mod, "make_client", lambda: client)
    state = tmp_path / "job.json"
    submit_mod.save_state(state, {"batch_id": "batch_old"})

    code = submit_mod.main(["--src", str(_src(tmp_path)), "-o", str(tmp_path / "o.jsonl"),
                            "--state", str(state), "--interval", "0"])
    assert code == 0
    assert client.batches.created == []          # 没有重新建任务
    assert client.files.uploaded == []           # 也没有重新上传
    assert client.batches.retrieved == ["batch_old"]
    assert "续用任务" in capsys.readouterr().out


def test_main_new_flag_resubmits(tmp_path, monkeypatch, capsys):
    client = _FakeClient(_Batch(batch_id="batch_new", status="completed"))
    monkeypatch.setattr(submit_mod, "make_client", lambda: client)
    state = tmp_path / "job.json"
    submit_mod.save_state(state, {"batch_id": "batch_old"})

    code = submit_mod.main(["--src", str(_src(tmp_path)), "-o", str(tmp_path / "o.jsonl"),
                            "--state", str(state), "--new", "--interval", "0"])
    assert code == 0
    assert len(client.batches.created) == 1
    assert "重复计费" in capsys.readouterr().out


def test_main_no_wait_skips_polling(tmp_path, monkeypatch):
    client = _FakeClient(_Batch(batch_id="batch_new", status="in_progress"))
    monkeypatch.setattr(submit_mod, "make_client", lambda: client)

    code = submit_mod.main(["--src", str(_src(tmp_path)), "-o", str(tmp_path / "o.jsonl"),
                            "--state", str(tmp_path / "job.json"), "--no-wait"])
    assert code == 0
    assert client.batches.retrieved == []


def test_main_status_only_reports(tmp_path, monkeypatch, capsys):
    client = _FakeClient(_Batch(batch_id="batch_1", status="in_progress",
                                counts=_Counts(8751, 120, 0)))
    monkeypatch.setattr(submit_mod, "make_client", lambda: client)

    code = submit_mod.main(["--batch-id", "batch_1", "--state", str(tmp_path / "job.json"),
                            "--status"])
    assert code == 0
    out = capsys.readouterr().out
    assert "in_progress" in out and "120/8751" in out
    assert not (tmp_path / "out.jsonl").exists()


def test_main_batch_id_skips_upload(tmp_path, monkeypatch):
    client = _FakeClient(_Batch(batch_id="batch_given", status="completed"))
    monkeypatch.setattr(submit_mod, "make_client", lambda: client)

    code = submit_mod.main(["--batch-id", "batch_given", "-o", str(tmp_path / "o.jsonl"),
                            "--state", str(tmp_path / "job.json"), "--interval", "0"])
    assert code == 0
    assert client.files.uploaded == []


def test_main_missing_source_returns_error(tmp_path, monkeypatch, capsys):
    client = _FakeClient()
    monkeypatch.setattr(submit_mod, "make_client", lambda: client)
    code = submit_mod.main(["--src", str(tmp_path / "nope.jsonl"),
                            "--state", str(tmp_path / "job.json")])
    assert code == 1
    assert "编JSONL" in capsys.readouterr().out


def test_main_without_api_key_returns_error(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("ZHIPU_API_KEY", raising=False)
    monkeypatch.setattr(submit_mod, "_force_utf8", lambda: None)
    code = submit_mod.main(["--src", str(_src(tmp_path)), "--state", str(tmp_path / "job.json")])
    assert code == 1
    assert "ZHIPU_API_KEY" in capsys.readouterr().out


def test_main_reports_platform_error_without_traceback(tmp_path, monkeypatch, capsys):
    """建任务失败（如未实名认证）要打成人话，不是堆栈。"""
    class _Boom(_FakeBatches):
        def create(self, **kwargs):
            raise RuntimeError('{"error":{"code":"1000","message":"请先完成实名认证"}}')

    client = _FakeClient()
    client.batches = _Boom()
    monkeypatch.setattr(submit_mod, "make_client", lambda: client)

    code = submit_mod.main(["--src", str(_src(tmp_path)), "-o", str(tmp_path / "o.jsonl"),
                            "--state", str(tmp_path / "job.json")])
    assert code == 1
    assert "实名认证" in capsys.readouterr().out

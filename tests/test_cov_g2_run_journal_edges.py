"""Run journal edges: secrets, mail, ledgers, recorded models and locks."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from spacr import run_journal as rj


def test_an_unreadable_ledger_is_not_noted():
    run = types.SimpleNamespace(_ledgers=[])
    rj.Run._note_ledger(run, object())
    assert run._ledgers == []


def test_a_failed_secret_write_removes_its_temporary(tmp_path, monkeypatch):
    path = tmp_path / "secrets.json"

    def broken(*a, **k):
        raise ValueError("cannot encode")

    monkeypatch.setattr(rj.json, "dump", broken)
    with pytest.raises(ValueError):
        rj._write_notify_secret_file({"a": "1"}, path)
    assert list(tmp_path.iterdir()) == []


def test_a_secret_temporary_already_gone_is_not_an_error(tmp_path, monkeypatch):
    path = tmp_path / "secrets.json"

    def vanish(source, target):
        Path(source).unlink()
        raise OSError("disk went away")

    monkeypatch.setattr(rj.os, "replace", vanish)
    with pytest.raises(OSError, match="disk went away"):
        rj._write_notify_secret_file({"a": "1"}, path)


def test_an_empty_keyring_answer_reads_the_secret_file(tmp_path, monkeypatch):
    ring = types.SimpleNamespace(get_password=lambda service, name: "")
    monkeypatch.setattr(rj, "_notify_keyring", lambda: ring)
    monkeypatch.setattr(rj, "_notify_secrets_path", lambda: tmp_path / "s.json")
    rj._write_notify_secret_file({"slack_webhook": "from-file"})
    assert rj._load_notify_secret("slack_webhook") == "from-file"


def test_no_preferences_means_no_notification_config(monkeypatch):
    monkeypatch.setitem(sys.modules, "spacr.qt.preferences", None)
    assert rj._notification_config() is None


class _Server:
    def __init__(self, *args, **kwargs):
        self.calls = []
        _Server.last = self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def starttls(self, context=None):
        self.calls.append("starttls")

    def login(self, user, password):
        self.calls.append("login")

    def send_message(self, mail):
        self.calls.append("send")


def test_mail_over_starttls_without_a_login(monkeypatch):
    import smtplib

    monkeypatch.setattr(smtplib, "SMTP", _Server)
    rj._notify_by_email({"title": "t", "body": "b", "failed": False},
                        {"smtp_host": "mail.example", "email_to": "a@b.c",
                         "smtp_security": "starttls"}, "")
    assert _Server.last.calls == ["starttls", "send"]


def test_a_crashing_notification_thread_keeps_no_results(monkeypatch):
    def broken(message, notify):
        raise RuntimeError("no network stack")

    monkeypatch.setattr(rj, "_send_notification", broken)
    thread = rj._dispatch_notification({"title": "t"}, {})
    thread.join(5)
    assert thread.results == {}


def test_ledger_listening_is_idempotent_and_survives_no_errors_module(
        monkeypatch):
    from spacr import errors

    monkeypatch.setattr(errors, "_FINALIZE_LISTENERS", [])
    rj._listen_for_ledgers()
    rj._listen_for_ledgers()
    assert errors._FINALIZE_LISTENERS == [rj._note_ledger_on_the_open_run]
    monkeypatch.setitem(sys.modules, "spacr.errors", None)
    rj._listen_for_ledgers()


def test_unhashable_lock_files_are_left_out(tmp_path, monkeypatch):
    model = tmp_path / "model.pth"
    model.write_bytes(b"weights")
    monkeypatch.setattr(rj, "hash_file", lambda path, full=True: "")
    assert rj._lock_files({"model_path": str(model)}, ()) == {}


def test_recorded_models_skip_unreadable_runs(tmp_path, monkeypatch):
    blocked = tmp_path / "not-a-folder"
    blocked.write_text("x")
    monkeypatch.setattr(rj, "runs_root", lambda: blocked)
    assert rj._recorded_models("classify", tmp_path) == {}
    root = tmp_path / "runs"
    for name, models in (("c", {}), ("b", {}), ("a", None)):
        (root / name).mkdir(parents=True)
        (root / name / "manifest.json").write_text(json.dumps(
            {"app_key": "classify", "model_files": models or {}}))
    monkeypatch.setattr(rj, "runs_root", lambda: root)
    real = rj.load_run_settings

    def settings(folder):
        if Path(folder).name == "a":
            raise ValueError("unreadable settings")
        if Path(folder).name == "c":
            return {"src": str(tmp_path / "elsewhere")}
        return {"src": str(tmp_path)}

    monkeypatch.setattr(rj, "load_run_settings", settings)
    assert rj._recorded_models("classify", tmp_path) == {}
    assert real


def test_an_unchanged_locked_file_is_no_deviation(tmp_path):
    model = tmp_path / "model.pth"
    model.write_bytes(b"weights")
    digest = rj.hash_file(model, full=True)
    record = {"settings": {}, "files": {str(model): digest}}
    assert not [d for d in rj._lock_deviations(record, {})
                if d["key"].startswith("file:")]


def test_unreadable_first_seen_lines_are_skipped():
    record = {"lock_id": "lock-seen-edge"}
    rj._seen_log(record).parent.mkdir(parents=True, exist_ok=True)
    rj._seen_log(record).write_text('not json\n{"key": "a"}\n')
    deviations = [{"key": "k", "locked": 1, "now": 2}]
    rj._note_first_seen(record, deviations)
    assert deviations[0]["first_seen_utc"]


def _locked_run(tmp_path, models, files=None):
    run = types.SimpleNamespace(
        _analysis_lock_record={"models": models, "files": files or {}},
        _analysis_lock={"deviations": [], "uncovered_models": []},
        model_files={"m": {"sha256": "abc"}})
    return run


def test_a_model_among_the_locked_files_needs_no_check(tmp_path, monkeypatch):
    path = tmp_path / "m.pth"
    judged = []
    monkeypatch.setattr(rj, "_set_run_lock_verdict",
                        lambda run, verdict: judged.append(verdict))
    run = _locked_run(tmp_path, {"other": {"sha256": "zzz"}},
                      {str(path.resolve()): "abc"})
    rj._check_model_for_run(run, "m", path)
    assert judged == []


def test_a_model_check_that_breaks_is_only_logged(tmp_path, caplog):
    run = _locked_run(tmp_path, {})
    run.model_files = None
    rj._check_model_for_run(run, "m", tmp_path / "m.pth")
    assert "could not be checked" in caplog.text

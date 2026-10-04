"""Run-finished notifications: where secrets live and how each channel refuses.

No message leaves the machine: SMTP, HTTP and the desktop notifier are
stood in for, and the keyring is a stand-in object.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from spacr import run_journal as rj


def test_the_secret_file_lives_under_dot_spacr():
    path = rj.unsandboxed_notify_secrets_path()
    assert path.name == "notification_secrets.json"
    assert path.parent.name == ".spacr"


@pytest.mark.parametrize("backend, usable", [
    (types.SimpleNamespace(priority=0), False),
    (type("Fail", (), {"__module__": "keyring.backends.fail", "priority": 1})(),
     False),
    (types.SimpleNamespace(priority=5), True),
])
def test_an_unusable_keyring_counts_as_none(monkeypatch, backend, usable):
    keyring = types.ModuleType("keyring")
    keyring.get_keyring = lambda: backend
    monkeypatch.setitem(sys.modules, "keyring", keyring)
    assert (rj.unsandboxed_notify_keyring() is keyring) is usable


def test_a_broken_keyring_counts_as_none(monkeypatch):
    keyring = types.ModuleType("keyring")

    def broken():
        raise RuntimeError("no D-Bus")

    keyring.get_keyring = broken
    monkeypatch.setitem(sys.modules, "keyring", keyring)
    assert rj.unsandboxed_notify_keyring() is None


def test_a_secret_file_that_is_not_an_object_or_not_json_is_empty(tmp_path):
    path = tmp_path / "s.json"
    path.write_text("[1, 2]")
    assert rj._read_notify_secret_file(path) == {}
    path.write_text("{broken")
    assert rj._read_notify_secret_file(path) == {}


def test_emptying_the_file_removes_it_even_when_it_is_already_gone(tmp_path):
    rj._write_notify_secret_file({}, tmp_path / "missing.json")
    assert not (tmp_path / "missing.json").exists()


def test_a_failed_secret_write_leaves_no_temporary_file(tmp_path, monkeypatch):
    def refuse(*_args):
        raise OSError("read-only")

    monkeypatch.setattr(rj.os, "replace", refuse)
    with pytest.raises(OSError):
        rj._write_notify_secret_file({"slack_webhook": "x"}, tmp_path / "s.json")
    assert list(tmp_path.iterdir()) == []


class _Ring:
    def __init__(self, refuse=False):
        self.values, self.refuse = {}, refuse

    def set_password(self, service, name, value):
        if self.refuse:
            raise RuntimeError("locked")
        self.values[name] = value

    def get_password(self, service, name):
        if self.refuse:
            raise RuntimeError("locked")
        return self.values.get(name)

    def delete_password(self, service, name):
        raise RuntimeError("not there")


def test_secrets_move_from_the_file_into_a_keyring(monkeypatch):
    rj._write_notify_secret_file({"slack_webhook": "https://hooks/1"})
    ring = _Ring()
    monkeypatch.setattr(rj, "_notify_keyring", lambda: ring)
    assert rj._store_notify_secret("slack_webhook", "https://hooks/2") == "keyring"
    assert rj._read_notify_secret_file() == {}
    assert rj._load_notify_secret("slack_webhook") == "https://hooks/2"
    assert rj._store_notify_secret("slack_webhook", "") == "forgotten"


def test_a_refusing_keyring_falls_back_to_the_file(monkeypatch):
    monkeypatch.setattr(rj, "_notify_keyring", lambda: _Ring(refuse=True))
    assert rj._store_notify_secret("ntfy_topic", "lab-runs") == "file"
    assert rj._load_notify_secret("ntfy_topic") == "lab-runs"
    rj._store_notify_secret("ntfy_topic", "")


def test_notification_settings_unreadable_are_none(monkeypatch):
    from spacr.qt import preferences

    def broken():
        raise RuntimeError("settings store locked")

    monkeypatch.setattr(preferences, "_run_notification_config", broken)
    assert rj._notification_config() is None


def _run(tmp_path, **fields):
    run = types.SimpleNamespace(
        status="finished", app_key="mask", dir=tmp_path / "run1", start_ts=0.0,
        end_ts=65.0, error_traceback="", settings={}, _ledgers=[], stages=[],
        run_warnings=[], output_hashes={})
    run.__dict__.update(fields)
    return run


def test_the_summary_counts_stages_metrics_warnings_and_outputs(tmp_path):
    run = _run(tmp_path, stages=[
        {"id": "seg", "state": "done", "metrics": {"objects": 12, "ok": True,
                                                    "mean": 1.5, "name": "x"}},
        {"label": "measure", "state": "failed", "metrics": {}}],
        run_warnings=["low contrast"], output_hashes={"a.csv": "h"},
        settings={"dst": ["/out/one", "/out/two"]})
    lines = rj._run_qc_summary(run)
    assert "Stages: 1 done, 1 failed" in lines
    assert "seg objects: 12" in lines and "seg mean: 1.5" in lines
    assert "Warnings: 1" in lines and "Output files recorded: 1" in lines
    assert rj._notify_output_pointer(run) == "/out/one"
    assert rj._notify_output_pointer(_run(tmp_path, settings={"dst": []})) == ""


def test_a_failed_run_without_a_traceback_has_no_error_line(tmp_path):
    message = rj._run_notification_message(_run(tmp_path, status="failed"))
    assert message["failed"] and "Error:" not in message["body"]


def test_an_http_error_status_is_raised(monkeypatch):
    import urllib.request

    class _Reply:
        status = 500

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(urllib.request, "urlopen", lambda r, timeout: _Reply())
    with pytest.raises(RuntimeError, match="answered 500"):
        rj._notify_http_post("https://hooks.example/x", b"{}", {})


def test_email_needs_a_server_and_uses_ssl_or_plain(monkeypatch):
    import smtplib

    message = {"title": "spaCR run finished: mask", "body": "ok", "failed": False}
    with pytest.raises(ValueError, match="SMTP server"):
        rj._notify_by_email(message, {"email_to": "a@b"}, "")
    sent = []

    class _Server:
        def __init__(self, host, port, timeout=None, context=None):
            sent.append((type(self).__name__, host, port))

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def starttls(self, context=None):
            sent.append("starttls")

        def login(self, user, password):
            sent.append("login")

        def send_message(self, mail):
            sent.append(mail["To"])

    monkeypatch.setattr(smtplib, "SMTP_SSL", type("SMTP_SSL", (_Server,), {}))
    monkeypatch.setattr(smtplib, "SMTP", type("SMTP", (_Server,), {}))
    rj._notify_by_email(message, {"smtp_host": "mail", "email_to": "a@b",
                                  "smtp_security": "ssl"}, "")
    rj._notify_by_email(message, {"smtp_host": "mail", "email_to": "a@b; c@d",
                                  "smtp_security": "plain"}, "")
    assert sent == [("SMTP_SSL", "mail", 587), "a@b",
                    ("SMTP", "mail", 587), "a@b, c@d"]


def test_slack_and_ntfy_need_their_address(monkeypatch):
    message = {"title": "Lauf fertig: Maske ✓", "body": "ok", "failed": True}
    with pytest.raises(ValueError, match="Slack webhook"):
        rj._notify_by_slack(message, "")
    with pytest.raises(ValueError, match="ntfy topic"):
        rj._notify_by_ntfy(message, {}, "", "")
    posted = []
    monkeypatch.setattr(rj, "_notify_http_post",
                        lambda url, data, headers: posted.append(headers))
    rj._notify_by_ntfy(message, {}, "lab", "")
    assert posted[0]["Title"].startswith("=?utf-8?")


def test_the_desktop_notification_uses_the_platform_tool(monkeypatch):
    import shutil
    import subprocess

    ran = []
    monkeypatch.setattr(subprocess, "run", lambda args, **k: ran.append(args[0]))
    monkeypatch.setattr(rj.sys, "platform", "linux")
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/notify-send")
    rj._desktop_os_notify("t", "b")
    monkeypatch.setattr(rj.sys, "platform", "darwin")
    rj._desktop_os_notify('a "quoted" title', "two\nlines")
    assert ran == ["/usr/bin/notify-send", "osascript"]
    monkeypatch.setattr(rj.sys, "platform", "linux")
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="no desktop notification"):
        rj._desktop_os_notify("t", "b")


def test_without_a_qt_notifier_the_os_one_is_used(monkeypatch):
    shown = []
    monkeypatch.setattr(rj, "_DESKTOP_NOTIFIER", [None])
    monkeypatch.setattr(rj, "_desktop_os_notify",
                        lambda title, body: shown.append(title))
    rj._notify_by_desktop({"title": "done", "body": "", "failed": False})
    assert shown == ["done"]

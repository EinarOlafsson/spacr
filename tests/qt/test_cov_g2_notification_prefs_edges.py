"""Run-notification preferences: unreadable values, secrets and test sends."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings  # noqa: E402

from spacr import run_journal  # noqa: E402
from spacr.qt import preferences as p  # noqa: E402


@pytest.fixture
def store(tmp_path, monkeypatch):
    path = tmp_path / "notify.ini"
    monkeypatch.setattr(p, "_settings",
                        lambda: QSettings(str(path), QSettings.IniFormat))
    return p._settings


def test_unreadable_notification_values_fall_back(store):
    settings = store()
    settings.setValue("notify/when", "sometimes")
    settings.setValue("notify/smtp_security", "rot13")
    settings.setValue("notify/smtp_port", 70000)
    settings.setValue("notify/min_minutes", "soon")
    settings.sync()
    values = p._get_run_notifications()
    for key in ("when", "smtp_security", "smtp_port", "min_minutes"):
        assert values[key] == p._NOTIFY_DEFAULTS[key]


def test_saved_secret_names_read_from_a_list_and_empty_secrets_are_skipped(
        store, monkeypatch):
    stored = []
    monkeypatch.setattr(run_journal, "_store_notify_secret",
                        lambda name, value: stored.append(name))
    store().setValue(p._KEY_NOTIFY_SAVED_SECRETS, ["a", "b"])
    assert p._saved_notification_secrets() == frozenset({"a", "b"})
    p._set_run_notifications({}, {"ntfy_topic": "", "slack_webhook": "x"})
    assert stored == ["slack_webhook"]


def test_a_nan_tooltip_delay_uses_the_default():
    assert p._clamped_tooltip_delay(float("nan")) == p._TOOLTIP_DELAY_DEFAULT


def _page(**values):
    texts = []
    page = SimpleNamespace(
        test_result=SimpleNamespace(setText=texts.append),
        send_test=SimpleNamespace(setEnabled=lambda on: None),
        values=lambda: dict({name: False for name in (
            "desktop", "email", "slack", "ntfy", "teams", "webhook")}, **values),
        secrets=lambda: {}, _mark_saved_secrets=lambda: None,
        _thread=None, _timer=None)
    return page, texts


def test_a_test_with_no_channel_on_asks_for_one():
    page, texts = _page()
    assert p._NotificationsPage._send_test(page) is None
    assert "at least one way" in texts[-1]


def test_a_running_or_missing_test_thread_is_not_reported():
    page, texts = _page()
    assert p._NotificationsPage._test_finished(page) is False
    page._thread = SimpleNamespace(is_alive=lambda: True)
    assert p._NotificationsPage._test_finished(page) is False
    page._thread = SimpleNamespace(is_alive=lambda: False,
                                   results={"slack": "ValueError"})
    assert p._NotificationsPage._test_finished(page) is True
    assert texts[-1].startswith("Not sent: slack")


def test_a_failed_forget_is_reported(monkeypatch):
    def broken():
        raise OSError("keyring locked")

    monkeypatch.setattr(p, "_forget_run_notification_secrets", broken)
    page, texts = _page()
    p._NotificationsPage._forget(page)
    assert "Could not forget" in texts[-1]

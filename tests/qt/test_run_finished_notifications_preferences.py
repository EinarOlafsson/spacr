"""Run-finished notifications are configured in Preferences, as an alpha.

Pinned here:

* with Show alpha features off (a fresh install) the Notifications tab is
  not built, and a configuration saved earlier sends nothing -- the gate
  hides it, so it does not fire;
* with it on, the tab carries every registered control, Save stores the
  preferences and puts a typed secret in the secret store rather than the
  preference file, and the stored configuration reaches a closing run;
* notifications are off until they are switched on;
* the Send a test button reports what got through, and the app's desktop
  relay carries a message from a worker thread to the GUI thread.
"""
from __future__ import annotations

import os
import threading

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings                              # noqa: E402
from PySide6.QtWidgets import (QDialogButtonBox, QLineEdit,       # noqa: E402
                               QPushButton, QTabWidget, QWidget)

from spacr import run_journal                                     # noqa: E402
from spacr.settings import ALPHA_FEATURES                         # noqa: E402

WIDGETS = ALPHA_FEATURES[577]["widgets"]


@pytest.fixture
def prefs(tmp_path, monkeypatch, qt_theme_applied):
    from spacr.qt import preferences

    path = tmp_path / "notify.ini"
    monkeypatch.setattr(preferences, "_settings",
                        lambda: QSettings(str(path), QSettings.IniFormat))
    return preferences


@pytest.fixture
def secret_file(tmp_path, monkeypatch):
    path = tmp_path / "secrets" / "notification_secrets.json"
    monkeypatch.setattr(run_journal, "_notify_secrets_path", lambda: path)
    monkeypatch.setattr(run_journal, "_notify_keyring", lambda: None)
    return path


@pytest.fixture
def dispatched(monkeypatch):
    calls = []
    monkeypatch.setattr(run_journal, "_dispatch_notification",
                        lambda message, config: calls.append(
                            (message, config)))
    return calls


def _dialog(qtbot, prefs):
    dlg = prefs.PreferencesDialog()
    qtbot.addWidget(dlg)
    return dlg


def _tab_titles(dlg):
    tabs = dlg.findChild(QTabWidget, "PreferencesTabs")
    return [tabs.tabText(i) for i in range(tabs.count())]


def _configured(prefs, secret_file):
    prefs._set_run_notifications(
        {"enabled": True, "min_minutes": 0, "desktop": False, "ntfy": True},
        {"ntfy_topic": "lab-topic-7c1"})


def test_notifications_are_off_on_a_fresh_install(prefs, secret_file):
    prefs._set_show_alpha_features(True)
    assert prefs._get_run_notifications()["enabled"] is False
    assert prefs._run_notification_config() is None
    assert run_journal._notification_config() is None


def test_hidden_the_tab_is_not_built_and_nothing_fires(qtbot, prefs,
                                                       secret_file,
                                                       dispatched):
    _configured(prefs, secret_file)
    dlg = _dialog(qtbot, prefs)
    assert "Notifications" not in _tab_titles(dlg)
    for name in WIDGETS:
        assert dlg.findChild(QWidget, name) is None, name
    assert prefs._run_notification_config() is None
    with run_journal.open_run("mask", {}):
        pass
    assert dispatched == []


def test_shown_the_tab_saves_and_the_run_is_announced(qtbot, prefs,
                                                      secret_file,
                                                      dispatched, tmp_path):
    prefs._set_show_alpha_features(True)
    dlg = _dialog(qtbot, prefs)
    assert "Notifications" in _tab_titles(dlg)
    for name in WIDGETS:
        assert dlg.findChild(QWidget, name) is not None, name
    assert not dlg.findChild(QWidget, "NotifyRunsEnabled").isChecked()

    dlg.findChild(QWidget, "NotifyRunsEnabled").setChecked(True)
    dlg.findChild(QWidget, "NotifyRunsMinMinutes").setValue(0)
    dlg.findChild(QWidget, "NotifyDesktop").setChecked(False)
    dlg.findChild(QWidget, "NotifySlack").setChecked(True)
    webhook = dlg.findChild(QLineEdit, "NotifySlackWebhook")
    assert webhook.echoMode() == QLineEdit.Password
    webhook.setText("https://hooks.slack.com/services/T0/B0/zz9secret")
    dlg.findChild(QDialogButtonBox).button(QDialogButtonBox.Save).click()

    assert "zz9secret" not in (tmp_path / "notify.ini").read_text()
    assert "zz9secret" in secret_file.read_text()
    assert prefs._saved_notification_secrets() == {"slack_webhook"}
    config = prefs._run_notification_config()
    assert config["slack"] is True and config["desktop"] is False
    assert config["email"] is False and config["ntfy"] is False

    with run_journal.open_run("mask", {"src": "/data/p1"}):
        pass
    assert len(dispatched) == 1
    message, sent_config = dispatched[0]
    assert message["title"] == "spaCR run finished: mask"
    assert "Output: /data/p1" in message["body"]
    assert "zz9secret" not in repr(sent_config)

    prefs._set_show_alpha_features(False)
    with run_journal.open_run("mask", {}):
        pass
    assert len(dispatched) == 1


def test_reopened_a_saved_secret_is_not_shown_and_reset_keeps_it(
        qtbot, prefs, secret_file):
    prefs._set_show_alpha_features(True)
    _configured(prefs, secret_file)
    dlg = _dialog(qtbot, prefs)
    topic = dlg.findChild(QLineEdit, "NotifyNtfyTopic")
    assert topic.text() == ""
    assert topic.placeholderText() == "Saved; type to replace"
    assert dlg.findChild(QLineEdit, "NotifySmtpPassword"
                         ).placeholderText() == "Not saved"
    assert dlg.findChild(QWidget, "NotifyRunsEnabled").isChecked()
    dlg.findChild(QPushButton, "PreferencesReset").click()
    assert not dlg.findChild(QWidget, "NotifyRunsEnabled").isChecked()
    assert dlg.findChild(QWidget, "NotifyRunsMinMinutes").value() == 5
    assert run_journal._load_notify_secret("ntfy_topic") == "lab-topic-7c1"

    dlg.findChild(QPushButton, "NotifyForgetSecrets").click()
    assert run_journal._load_notify_secret("ntfy_topic") == ""
    assert not secret_file.exists()
    assert topic.placeholderText() == "Not saved"


def test_send_a_test_reports_what_got_through(qtbot, prefs, secret_file,
                                              monkeypatch):
    shown = []
    monkeypatch.setattr(run_journal, "_DESKTOP_NOTIFIER",
                        [lambda title, body, failed: shown.append(title)])
    prefs._set_show_alpha_features(True)
    dlg = _dialog(qtbot, prefs)
    dlg.findChild(QWidget, "NotifyDesktop").setChecked(True)
    dlg.findChild(QWidget, "NotifySlack").setChecked(True)
    dlg.findChild(QLineEdit, "NotifySlackWebhook").setText(
        "ftp://not-a-webhook/secretpart")
    dlg.findChild(QPushButton, "NotifySendTest").click()
    label = dlg.findChild(QWidget, "NotifyTestResult")
    qtbot.waitUntil(lambda: "Sent:" in label.text(), timeout=10000)
    assert shown == ["spaCR test notification"]
    assert "Sent: desktop" in label.text()
    assert "Not sent: slack (ValueError" in label.text()
    assert "secretpart" not in label.text()
    assert prefs._saved_notification_secrets() == frozenset()


def test_the_desktop_relay_reaches_the_gui_thread(qtbot, monkeypatch):
    from PySide6.QtWidgets import QApplication, QSystemTrayIcon

    from spacr.qt.preferences import _install_run_notifier

    monkeypatch.setattr(run_journal, "_DESKTOP_NOTIFIER", [None])
    monkeypatch.setattr(QSystemTrayIcon, "isSystemTrayAvailable",
                        staticmethod(lambda: False))
    shown = []
    monkeypatch.setattr(run_journal, "_desktop_os_notify",
                        lambda title, body: shown.append((title, body)))
    relay = _install_run_notifier(QApplication.instance())
    try:
        assert run_journal._DESKTOP_NOTIFIER[0] is not None
        worker = threading.Thread(target=run_journal._notify_by_desktop,
                                  args=({"title": "t", "body": "b",
                                         "failed": True},))
        worker.start()
        worker.join(10)
        qtbot.waitUntil(lambda: shown == [("t", "b")], timeout=10000)
    finally:
        relay.setParent(None)
        relay.deleteLater()

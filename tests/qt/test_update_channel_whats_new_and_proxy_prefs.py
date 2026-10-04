"""Preferences carry the update channel, proxy and CA; an update shows What's new."""
import json

import pytest

pytest.importorskip("PySide6")

from spacr import updater
from spacr.qt import preferences as P


@pytest.fixture(autouse=True)
def _network_sandbox(tmp_path, monkeypatch):
    monkeypatch.setenv("SPACR_NETWORK_CONFIG", str(tmp_path / "network.json"))
    for key in (updater._PROXY_VARIABLES + updater._CA_VARIABLES
                + updater._NO_PROXY_VARIABLES):
        monkeypatch.delenv(key, raising=False)
    yield
    updater._undo_network_exports()


def _save(dlg):
    from PySide6.QtWidgets import QDialogButtonBox
    dlg.findChild(QDialogButtonBox).button(QDialogButtonBox.Save).click()


def test_preferences_save_the_channel_proxy_and_bundle(qtbot, tmp_path):
    import os

    from PySide6.QtWidgets import QComboBox, QFormLayout, QLineEdit

    bundle = tmp_path / "corp.pem"
    bundle.write_text("x")
    from spacr.qt.widgets.hint_bar import HintBar

    dlg = P.PreferencesDialog(None)
    qtbot.addWidget(dlg)
    dlg.show()
    qtbot.waitExposed(dlg)
    channel = dlg.findChild(QComboBox, "UpdateChannel")
    proxy = dlg.findChild(QLineEdit, "NetworkProxy")
    ca = dlg.findChild(QLineEdit, "NetworkCaBundle")
    assert channel.currentData() == "stable"
    bar = dlg.window().findChild(HintBar)
    for widget, default in ((channel, "Default stable."), (proxy, "Default empty."),
                            (ca, "Default empty.")):
        labels = [f.labelForField(widget) for f in dlg.findChildren(QFormLayout)]
        label = next(x for x in labels if x is not None)
        tip = bar.explains(label)
        assert tip.endswith(default) and len(tip) <= 600
    channel.setCurrentIndex(channel.findData("nightly"))
    proxy.setText("http://proxy.example.org:3128")
    ca.setText(str(bundle))
    _save(dlg)
    assert P._get_update_channel() == "nightly"
    saved = json.loads((tmp_path / "network.json").read_text())
    assert saved == {"proxy": "http://proxy.example.org:3128", "ca_bundle": str(bundle)}
    assert os.environ["HTTPS_PROXY"] == "http://proxy.example.org:3128"
    assert os.environ["SSL_CERT_FILE"] == str(bundle)


def test_the_running_version_is_remembered_and_an_update_is_noticed():
    assert P._note_running_version("1.5.1.2") is None
    assert P._note_running_version("1.5.1.2") is None
    assert P._note_running_version("1.5.1.3") == "1.5.1.2"
    assert P._previous_version() == "1.5.1.2"


@pytest.fixture
def win(qtbot, qt_theme_applied):
    from spacr.qt.app import MainWindow
    w = MainWindow()
    qtbot.addWidget(w)
    return w


def test_nightly_channel_checks_prereleases(win, qtbot, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    asked = []
    monkeypatch.setattr(updater, "_check_on_channel",
                        lambda channel: asked.append(channel) or updater.UpdateInfo(
                            "9.9.9", "9.9.9", None))
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    P._set_update_channel("nightly")
    win._check_for_updates()
    qtbot.waitUntil(lambda: bool(asked), timeout=5000)
    assert asked == ["nightly"]


def test_whats_new_shows_the_notes_between_the_two_versions(win, qtbot, monkeypatch):
    monkeypatch.setattr(updater, "_installed_version", lambda: "1.5.1.3")
    monkeypatch.setattr(updater, "_bundled_release_notes", lambda: [
        {"tag": "v1.5.1.3", "name": "spaCR 1.5.1.3", "body": "Third"},
        {"tag": "v1.5.1.2", "name": "spaCR 1.5.1.2", "body": "Second"},
        {"tag": "v1.5.1.0", "name": "spaCR 1.5.1.0", "body": "Zero"}])
    monkeypatch.setattr(P, "get_refresh_news", lambda: False)
    P._note_running_version("1.5.1.1")
    win._maybe_show_whats_new()
    dialog = win._whats_new_dialog
    from PySide6.QtWidgets import QTextBrowser
    text = dialog.findChild(QTextBrowser, "WhatsNewNotes").toPlainText()
    assert "Third" in text and "Second" in text and "Zero" not in text
    dialog.close()


def test_whats_new_uses_fetched_releases_when_online(win, qtbot, monkeypatch):
    monkeypatch.setattr(updater, "_installed_version", lambda: "1.5.1.3")
    monkeypatch.setattr(updater, "_bundled_release_notes", lambda: [])
    monkeypatch.setattr(updater, "fetch_release_notes", lambda: [
        {"tag": "v1.5.1.3", "name": "spaCR 1.5.1.3", "body": "From GitHub"}])
    monkeypatch.setattr(P, "get_refresh_news", lambda: True)
    P._note_running_version("1.5.1.2")
    win._maybe_show_whats_new()
    qtbot.waitUntil(lambda: getattr(win, "_whats_new_dialog", None) is not None,
                    timeout=5000)
    from PySide6.QtWidgets import QTextBrowser
    text = win._whats_new_dialog.findChild(QTextBrowser, "WhatsNewNotes").toPlainText()
    assert "From GitHub" in text
    win._whats_new_dialog.close()


def test_a_first_launch_shows_nothing(win, monkeypatch):
    monkeypatch.setattr(updater, "_installed_version", lambda: "1.5.1.3")
    win._maybe_show_whats_new()
    assert getattr(win, "_whats_new_dialog", None) is None

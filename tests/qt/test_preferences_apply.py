"""Apply previews real settings without dismissing Preferences."""

from __future__ import annotations

import pytest

from PySide6.QtCore import QSettings, Qt
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import (
    QComboBox, QDialogButtonBox, QDoubleSpinBox, QLineEdit, QMessageBox, QSlider, QWidget,
)

from spacr.qt import preferences
from spacr import updater


@pytest.fixture
def private_preferences(tmp_path, monkeypatch):
    path = tmp_path / "preferences.ini"
    monkeypatch.setattr(preferences, "_settings",
                        lambda: QSettings(str(path), QSettings.IniFormat))
    monkeypatch.setenv("SPACR_NETWORK_CONFIG", str(tmp_path / "network.json"))
    monkeypatch.setenv("SPACR_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("SPACR_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setattr(preferences, "_SAFE_MODE", False)
    preferences.set_language("en")
    preferences.set_theme_choice("dark")
    preferences.set_ambient_animation("none")
    yield
    updater._undo_network_exports()


def _store_values():
    store = preferences._settings()
    return {key: store.value(key) for key in store.allKeys()}


def _dialog(qtbot):
    class Owner(QWidget):
        refreshes = 0

        def refresh_theme(self):
            self.refreshes += 1

    owner = Owner()
    qtbot.addWidget(owner)
    owner.show()
    dialog = preferences.PreferencesDialog(owner)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitExposed(dialog)
    accepted = QSignalSpy(dialog.accepted)
    theme = next(combo for combo in dialog.findChildren(QComboBox)
                 if combo.findData("glass") >= 0)
    theme.setCurrentIndex(theme.findData("light"))
    dialog.findChild(QSlider, "PaneOpacity").setValue(63)
    return dialog, owner, accepted


def _apply(dialog, qtbot):
    box = dialog.findChild(QDialogButtonBox)
    box.button(QDialogButtonBox.Apply).click()
    qtbot.waitUntil(lambda: dialog._apply_confirmation is not None)
    question = dialog._apply_confirmation
    assert isinstance(question, QMessageBox)
    assert question.isVisible()
    assert dialog.isVisible()
    assert preferences.get_theme_choice() == "light"
    assert preferences.get_pane_opacity() == pytest.approx(0.63)
    assert question.parentWidget() is dialog
    return question


def _answer(question, text, qtbot):
    button = next(button for button in question.buttons()
                  if button.text() == text)
    button.click()
    qtbot.waitUntil(lambda: not question.isVisible())


def test_apply_keep_persists_and_keeps_preferences_open(
        private_preferences, qtbot, qapp):
    dialog, owner, accepted = _dialog(qtbot)
    question = _apply(dialog, qtbot)
    assert qapp._spacr_preferences_style_signature[0] == "light"
    _answer(question, "Keep", qtbot)
    assert dialog.isVisible()
    assert accepted.count() == 0
    assert owner.refreshes == 1
    assert preferences.get_theme_choice() == "light"
    dialog.reject()
    assert preferences.get_theme_choice() == "light"


def test_database_queue_budget_apply_revert_and_keep(private_preferences, qtbot):
    from spacr.measure import _measure_write_queue_budget

    dialog, _owner, _accepted = _dialog(qtbot)
    control = dialog.findChild(QDoubleSpinBox, 'DatabaseWriteQueueGiB')
    assert control.value() == 1.0
    control.setValue(0.0)
    _answer(_apply(dialog, qtbot), 'Revert', qtbot)
    assert preferences.get_database_write_queue_gib() == 1.0
    control.setValue(0.25)
    _answer(_apply(dialog, qtbot), 'Keep', qtbot)
    assert preferences.get_database_write_queue_gib() == 0.25
    assert _measure_write_queue_budget({}) == 0.25
    assert _measure_write_queue_budget({'database_write_queue_gib': 0}) == 0.0
    dialog.reject()
    assert preferences.get_theme_choice() == "light"


@pytest.mark.parametrize("answer", ["Revert", "Escape", "close"])
def test_revert_restores_exact_store_and_live_theme(
        private_preferences, qtbot, qapp, answer):
    dialog, owner, accepted = _dialog(qtbot)
    before = _store_values()
    question = _apply(dialog, qtbot)
    preferences._settings().setValue("unrelated/concurrent", "retained")
    if answer == "Escape":
        qtbot.keyClick(question, Qt.Key_Escape)
    elif answer == "close":
        question.close()
    else:
        _answer(question, answer, qtbot)
    qtbot.waitUntil(lambda: dialog._apply_confirmation is None)
    assert _store_values() == {**before, "unrelated/concurrent": "retained"}
    assert preferences.get_theme_choice() == "dark"
    assert qapp._spacr_preferences_style_signature[0] == "dark"
    assert dialog.isVisible()
    assert accepted.count() == 0
    assert owner.refreshes == 2
    assert dialog.findChild(QDialogButtonBox).button(
        QDialogButtonBox.Apply).isEnabled()


def test_repeated_apply_reverts_to_the_last_kept_values(
        private_preferences, qtbot):
    dialog, _owner, accepted = _dialog(qtbot)
    _answer(_apply(dialog, qtbot), "Keep", qtbot)
    before = _store_values()
    dialog.findChild(QSlider, "PaneOpacity").setValue(29)
    dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Apply).click()
    assert preferences.get_pane_opacity() == pytest.approx(0.29)
    _answer(dialog._apply_confirmation, "Revert", qtbot)
    assert _store_values() == before
    assert dialog.isVisible()
    assert accepted.count() == 0


@pytest.mark.parametrize("existing", [False, True])
def test_revert_restores_network_file_and_exports(
        private_preferences, qtbot, monkeypatch, existing):
    monkeypatch.setenv("HTTPS_PROXY", "http://original.example:3128")
    path = updater._network_config_path()
    original = b'{ "proxy": "http://saved.example:3128", "ca_bundle": "" }\n'
    if existing:
        path.write_bytes(original)
    updater._apply_network_settings()
    before_proxy = updater._effective_network()["proxy"]
    dialog, _owner, _accepted = _dialog(qtbot)
    dialog.findChild(QLineEdit, "NetworkProxy").setText(
        "http://preview.example:3128")
    question = _apply(dialog, qtbot)
    assert updater._effective_network()["proxy"] == "http://preview.example:3128"
    _answer(question, "Revert", qtbot)
    assert path.exists() is existing
    if existing:
        assert path.read_bytes() == original
    assert updater._effective_network()["proxy"] == before_proxy
    import os
    assert os.environ["HTTPS_PROXY"] == before_proxy


def test_notification_secrets_are_only_written_after_keep(
        private_preferences, qtbot, monkeypatch):
    from spacr import run_journal
    preferences._set_show_alpha_features(True)
    written = []
    monkeypatch.setattr(run_journal, "_store_notify_secret",
                        lambda name, value: written.append((name, value)))
    dialog, _owner, _accepted = _dialog(qtbot)
    secret = dialog.findChild(QLineEdit, "NotifyWebhookToken")
    assert secret is not None
    secret.setText("test-replacement")
    _answer(_apply(dialog, qtbot), "Revert", qtbot)
    assert written == []
    _answer(_apply(dialog, qtbot), "Keep", qtbot)
    assert written == [("webhook_token", "test-replacement")]


def test_save_after_keep_still_closes_without_a_second_confirmation(
        private_preferences, qtbot):
    dialog, _owner, accepted = _dialog(qtbot)
    _answer(_apply(dialog, qtbot), "Keep", qtbot)
    dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Save).click()
    assert not dialog.isVisible()
    assert accepted.count() == 1
    assert dialog._apply_confirmation is None
    assert preferences.get_theme_choice() == "light"


def test_original_save_still_closes_directly(private_preferences, qtbot):
    dialog, _owner, accepted = _dialog(qtbot)
    dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Save).click()
    assert not dialog.isVisible()
    assert accepted.count() == 1
    assert dialog._apply_confirmation is None
    assert preferences.get_theme_choice() == "light"


def test_closing_preferences_during_preview_reverts(private_preferences, qtbot):
    dialog, _owner, accepted = _dialog(qtbot)
    before = _store_values()
    _apply(dialog, qtbot)
    dialog.reject()
    assert _store_values() == before
    assert accepted.count() == 0
    assert dialog._apply_confirmation is None


def test_partial_apply_failure_restores_the_previous_settings(
        private_preferences, qtbot, monkeypatch):
    dialog, _owner, accepted = _dialog(qtbot)
    before = _store_values()
    messages = []

    def fail_after_earlier_writes(_value):
        raise RuntimeError("controlled apply failure")

    monkeypatch.setattr(preferences, "set_ambient_density",
                        fail_after_earlier_writes)
    monkeypatch.setattr(QMessageBox, "warning",
                        lambda *_args: messages.append(_args[-1]))
    dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Apply).click()
    assert _store_values() == before
    assert preferences.get_theme_choice() == "dark"
    assert dialog.isVisible()
    assert accepted.count() == 0
    assert dialog._apply_confirmation is None
    assert messages == [
        "Could not apply settings. Previous settings restored."]

"""Reset to defaults, and what "default" means.

Two things, and the second is the one that bit: a fresh install already
resolved to Follow system / blobs / spaCR, but a config that had drifted
away from them had no way back short of deleting the file. The button is
the way back.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QComboBox, QDialogButtonBox, QPushButton


@pytest.fixture()
def private_store(monkeypatch):
    """Point preferences at a throwaway INI.

    Not optional. These tests write preferences, and `preferences._settings`
    otherwise resolves to the real `~/.config/spacr/qt.conf` -- which is
    exactly how a diagnostic script once left a user's ambient backdrop
    switched off and their app looking flat and grey on next launch.
    """
    from spacr.qt import preferences

    path = os.path.join(tempfile.mkdtemp(prefix="spacr-prefs-"), "user.ini")
    monkeypatch.setattr(
        preferences, "_settings",
        lambda: QSettings(path, QSettings.IniFormat))
    return path


# ---------------------------------------------------------------------------
# 1. What a fresh install resolves to
# ---------------------------------------------------------------------------

def test_a_fresh_install_is_dark(private_store, qapp):
    from spacr.qt import preferences

    assert preferences.get_theme() == "dark"


def test_a_fresh_install_animates_blobs_in_the_spacr_palette(private_store,
                                                             qapp):
    from spacr.qt import preferences

    assert preferences.get_ambient_enabled() is True
    assert preferences.get_ambient_animation() == "blobs"
    assert preferences.get_ambient_palette() == "spacr"


# ---------------------------------------------------------------------------
# 2. The button
# ---------------------------------------------------------------------------

def _dialog(qtbot):
    from spacr.qt.preferences import PreferencesDialog

    dialog = PreferencesDialog()
    qtbot.addWidget(dialog)
    dialog.resize(900, 700)
    dialog.show()
    qtbot.waitExposed(dialog)
    return dialog


def test_the_reset_button_sits_left_of_cancel(private_store, qtbot):
    """`ResetRole`, which is what puts it away from the two that close.

    Asserted on rendered x positions rather than on the role, because the
    role is the mechanism and the position is the request.
    """
    dialog = _dialog(qtbot)
    box = dialog.findChild(QDialogButtonBox)
    reset = dialog.findChild(QPushButton, "PreferencesReset")
    assert reset is not None, "the Reset to defaults button is missing"

    ordered = sorted(box.findChildren(QPushButton),
                     key=lambda b: b.mapTo(dialog, b.rect().topLeft()).x())
    texts = [b.text() for b in ordered]
    assert texts[0] == reset.text(), (
        f"Reset is not the leftmost button: {texts}")
    assert texts.index(reset.text()) < texts.index("Cancel"), (
        f"Reset must sit left of Cancel: {texts}")


def test_reset_restores_the_three_the_user_named(private_store, qtbot):
    from spacr.qt import preferences

    preferences.set_theme_choice("light")
    preferences.set_ambient_animation("aurora")
    preferences.set_ambient_palette("ocean")

    dialog = _dialog(qtbot)
    dialog.findChild(QPushButton, "PreferencesReset").click()

    chosen = [combo.currentData() for combo in dialog.findChildren(QComboBox)]
    theme_combo = next(combo for combo in dialog.findChildren(QComboBox)
                       if combo.findData("glass") >= 0)
    assert theme_combo.currentData() == "dark", (
        "the theme did not go back to Dark")
    assert "blobs" in chosen, "the animation did not go back to blobs"
    assert "spacr" in chosen, "the palette did not go back to spaCR"
    assert "aurora" not in chosen and "ocean" not in chosen


def test_reset_writes_nothing_until_save(private_store, qtbot):
    """So Cancel still walks away from a reset the user did not mean.

    The button changes the controls; Save is what persists them. Anything
    else makes Reset an irreversible action wearing a dialog that has a
    Cancel button on it.
    """
    from spacr.qt import preferences

    preferences.set_theme_choice("dark")
    preferences.set_ambient_animation("aurora")
    preferences.set_pane_opacity(1.0)

    dialog = _dialog(qtbot)
    dialog.findChild(QPushButton, "PreferencesReset").click()

    assert preferences.get_theme_choice() == "dark"
    assert preferences.get_ambient_animation() == "aurora"
    assert preferences.get_pane_opacity() == 1.0


def test_reset_leaves_the_settings_accessor_restored(private_store, qtbot):
    """The reset reads defaults by pointing `_settings` at an empty store.

    If it failed to put the real one back, every later read and write in
    the process would go to a temporary file in /tmp -- the preferences
    would appear to save and be gone next launch.
    """
    from spacr.qt import preferences

    before = preferences._settings
    dialog = _dialog(qtbot)
    dialog.findChild(QPushButton, "PreferencesReset").click()
    assert preferences._settings is before

    preferences.set_pane_opacity(0.85)
    assert preferences.get_pane_opacity() == 0.85


# ---------------------------------------------------------------------------
# 3. The Logging tab's switches (instruction 294)
# ---------------------------------------------------------------------------

_LEVEL_NAMES = ("Debug", "Info", "Warning", "Error", "Critical")


def _switch(dialog, kind, name):
    from spacr.qt.widgets.toggle import Toggle

    return dialog.findChild(Toggle, f"Log{kind}Level{name}")


def _on(dialog, kind):
    return {name for name in _LEVEL_NAMES
            if _switch(dialog, kind, name).isChecked()}


def _verbose_switch(dialog):
    from spacr.qt.widgets.toggle import Toggle

    return next(t for t in dialog.findChildren(Toggle)
                if t.text() == "Enable verbose logging")


def _drift_the_logging_switches():
    import logging

    from spacr.qt import preferences

    preferences.set_verbose_logging(False)
    preferences.set_log_levels([logging.DEBUG, logging.ERROR],
                               [logging.DEBUG])


def test_reset_puts_the_logging_switches_back(private_store, qtbot):
    """Reset used to leave every Logging switch where the user had it."""
    _drift_the_logging_switches()
    dialog = _dialog(qtbot)
    assert _on(dialog, "File") == {"Debug", "Error"}

    dialog.findChild(QPushButton, "PreferencesReset").click()

    assert _verbose_switch(dialog).isChecked() is True
    assert _on(dialog, "File") == set(_LEVEL_NAMES)
    assert not _switch(dialog, "File", "Debug").isEnabled()
    assert _on(dialog, "Console") == {"Warning", "Error", "Critical"}


def test_after_reset_verbose_off_gives_back_the_default_debug_choice(
        private_store, qtbot):
    """Not the drifted one: DEBUG off, which is what a fresh install keeps."""
    _drift_the_logging_switches()
    dialog = _dialog(qtbot)

    dialog.findChild(QPushButton, "PreferencesReset").click()
    _verbose_switch(dialog).setChecked(False)

    assert _switch(dialog, "File", "Debug").isEnabled()
    assert _on(dialog, "File") == {"Info", "Warning", "Error", "Critical"}
    assert not _switch(dialog, "Console", "Debug").isChecked()


def test_reset_then_save_stores_the_default_levels(private_store, qtbot):
    import logging

    from spacr.logging_util import DEFAULT_CONSOLE_LEVELS, DEFAULT_FILE_LEVELS
    from spacr.qt import preferences

    _drift_the_logging_switches()
    dialog = _dialog(qtbot)
    dialog.findChild(QPushButton, "PreferencesReset").click()
    dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Save).click()

    assert preferences._chosen_log_file_levels() == DEFAULT_FILE_LEVELS
    assert preferences.get_log_console_levels() == DEFAULT_CONSOLE_LEVELS
    assert logging.DEBUG in preferences.get_log_file_levels()

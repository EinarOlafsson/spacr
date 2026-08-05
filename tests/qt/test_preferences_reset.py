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

def test_a_fresh_install_follows_the_system_theme(private_store, qapp):
    from spacr.qt import preferences

    assert preferences.get_theme() == "system"


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

    preferences.set_theme_choice("dark")
    preferences.set_ambient_animation("aurora")
    preferences.set_ambient_palette("ocean")

    dialog = _dialog(qtbot)
    dialog.findChild(QPushButton, "PreferencesReset").click()

    chosen = [combo.currentData() for combo in dialog.findChildren(QComboBox)]
    assert "system" in chosen, "the theme did not go back to Follow system"
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

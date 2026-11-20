"""Item 1001: Preferences chooses the view settings forms open on.

Essentials by default, so nothing changes until the user picks All. With All
chosen, a module with no remembered choice opens on All; a module's own
switch is still remembered; changing the preference again reaches every
module.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtCore import QSettings  # noqa: E402

from spacr.qt import preferences  # noqa: E402
from spacr.qt.settings_search import (ALL, ESSENTIALS, disclosure_for,  # noqa: E402
                                      forget_disclosure, remember_disclosure)

pytestmark = pytest.mark.qt


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A preference store of its own, and no remembered module views."""
    path = tmp_path / "prefs.ini"
    monkeypatch.setattr(preferences, "_settings",
                        lambda: QSettings(str(path), QSettings.IniFormat))
    forget_disclosure()
    yield path
    forget_disclosure()


def test_the_default_is_essentials_as_before(store):
    assert preferences._get_settings_view_default() == "essentials"
    assert disclosure_for("mask") == ESSENTIALS


def test_all_reaches_every_module_without_a_choice(store):
    assert preferences._set_settings_view_default("all") is True
    assert preferences._get_settings_view_default() == "all"
    assert disclosure_for("mask") == ALL and disclosure_for("ops") == ALL
    assert preferences._set_settings_view_default("all") is False, "no change"


def test_a_module_switch_is_still_remembered_until_the_default_changes(store):
    preferences._set_settings_view_default("all")
    remember_disclosure("mask", ESSENTIALS)
    from spacr.qt.prefs import _flush_stores

    _flush_stores()
    assert disclosure_for("mask") == ESSENTIALS
    assert disclosure_for("measure") == ALL
    preferences._set_settings_view_default("essentials")
    remember_disclosure("measure", ALL)
    _flush_stores()
    preferences._set_settings_view_default("all")
    assert disclosure_for("mask") == ALL, "a new default reaches every module"
    preferences._set_settings_view_default("essentials")
    assert disclosure_for("measure") == ESSENTIALS


def test_an_unknown_value_reads_as_essentials(store):
    preferences._settings().setValue(preferences._KEY_SETTINGS_VIEW_DEFAULT, "most")
    assert preferences._get_settings_view_default() == "essentials"


def test_the_preferences_row_reads_and_writes_it(qtbot, store):
    """Save in Preferences stores the choice; a new dialog shows it."""
    from PySide6.QtWidgets import QComboBox, QDialogButtonBox

    dialog = preferences.PreferencesDialog()
    qtbot.addWidget(dialog)
    combo = dialog.findChild(QComboBox, "SettingsViewDefault")
    assert combo is not None and combo.currentData() == "essentials"
    assert [combo.itemData(i) for i in range(combo.count())] == ["essentials", "all"]
    combo.setCurrentIndex(combo.findData("all"))
    boxes = dialog.findChildren(QDialogButtonBox)
    next(box.button(QDialogButtonBox.Save) for box in boxes
         if box.button(QDialogButtonBox.Save) is not None).click()
    assert preferences._get_settings_view_default() == "all"
    assert disclosure_for("mask") == ALL
    again = preferences.PreferencesDialog()
    qtbot.addWidget(again)
    assert again.findChild(QComboBox, "SettingsViewDefault").currentData() == "all"

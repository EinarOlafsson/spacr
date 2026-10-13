"""The Home tutorial capture finds the performance selector it films.

Item 286, bullet 7. ``tools/tutorials/authoring/tools/capture_all_modules.py``
opens the real Preferences dialog, looks up the level selector and the note
under it by object name, and raises "Performance level selector or
explanation is not visible" when either is missing. Measured 2026-09-19: it
looked for ``PerformanceLevelNote`` while the dialog named that label
``SpacrModeNote``, after the "spaCR mode" control the five-level selector
replaced, so the capture could not run. The dialog now uses the selector's
name, and this test reads the names out of the capture script itself so the
two cannot drift apart again unnoticed.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QComboBox, QLabel, QTabWidget

CAPTURE = (Path(__file__).resolve().parents[2] / "tools" / "tutorials"
           / "authoring" / "tools" / "capture_all_modules.py")


@pytest.fixture(autouse=True)
def _isolated_qsettings(monkeypatch, qt_theme_applied, tmp_path):
    from spacr.qt import preferences as prefs
    store = QSettings(str(tmp_path / "prefs.ini"), QSettings.IniFormat)
    monkeypatch.setattr(prefs, "_settings", lambda: store)
    return store


def _lookups():
    text = CAPTURE.read_text(encoding="utf-8")
    return sorted(set(re.findall(
        r'preferences\.findChild\(\s*(QComboBox|QLabel|QTabWidget)\s*,\s*'
        r'"([A-Za-z]+)"\s*\)', text)))


def test_the_capture_still_looks_the_selector_up_by_name():
    assert ("QComboBox", "PerformanceLevel") in _lookups()
    assert ("QLabel", "PerformanceLevelNote") in _lookups()


def test_every_name_the_capture_looks_up_is_on_the_real_dialog(qtbot):
    from spacr.qt.preferences import PreferencesDialog

    dialog = PreferencesDialog()
    qtbot.addWidget(dialog)
    kinds = {"QComboBox": QComboBox, "QLabel": QLabel,
             "QTabWidget": QTabWidget}
    missing = [f"{kind} {name}" for kind, name in _lookups()
               if dialog.findChild(kinds[kind], name) is None]
    assert not missing, (
        f"capture_all_modules.py looks for {missing}, which the "
        "Preferences dialog does not build")


def test_the_note_is_visible_beside_the_selector_once_its_tab_is_shown(
        qtbot):
    """What the capture does: pick the selector's tab, show, measure both."""
    from spacr.qt.preferences import PreferencesDialog

    dialog = PreferencesDialog()
    qtbot.addWidget(dialog)
    tabs = dialog.findChild(QTabWidget, "PreferencesTabs")
    level = dialog.findChild(QComboBox, "PerformanceLevel")
    note = dialog.findChild(QLabel, "PerformanceLevelNote")
    for index in range(tabs.count()):
        if tabs.widget(index).findChild(QComboBox, "PerformanceLevel") is level:
            tabs.setCurrentIndex(index)
            break
    assert tabs.widget(tabs.currentIndex()).isAncestorOf(note)
    dialog.resize(1200, 1200)
    dialog.show()
    qtbot.waitExposed(dialog)
    assert level.isVisible() and note.isVisible()
    assert note.text().strip(), "the note under the selector is empty"
    assert note.width() > 0 and note.height() > 0

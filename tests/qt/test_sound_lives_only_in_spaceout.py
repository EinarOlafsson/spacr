"""Sound belongs to spaceout mode; ordinary spaCR has none, and no Sound tab.

Maintainer, 2026-09-21: "transfer the shound tab in preferences to only be
visible in spaceout mode. in normal spacr sound should be of by default and
there should be no sound tab in preferences."

Spaceout is process-local: the ``spaceout`` launcher calls
:func:`spacr.qt.theme.enable_spaceout`, nothing stores it. So:

* ordinary spaCR builds no Sound tab and plays nothing, WHATEVER IS STORED
  -- a switch the user cannot see must not be able to make a noise;
* the stored sound values are never erased, so the next spaceout launch
  finds them as the user left them;
* the tab is decided each time Preferences is built, so a process that
  turns spaceout on or off gets the matching dialog the next time it opens.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

from PySide6.QtCore import QSettings                            # noqa: E402
from PySide6.QtWidgets import (QDialogButtonBox, QTabWidget,     # noqa: E402
                               QWidget)

from spacr.qt import preferences as prefs                       # noqa: E402
from spacr.qt import theme                                      # noqa: E402


@pytest.fixture
def store(monkeypatch, tmp_path, qt_theme_applied):
    settings = QSettings(str(tmp_path / "prefs.ini"), QSettings.IniFormat)
    monkeypatch.setattr(prefs, "_settings", lambda: settings)
    return settings


@pytest.fixture
def spaceout(monkeypatch):
    """Switch spaceout on or off for the rest of the test."""
    state = {"on": False}
    monkeypatch.setattr(theme, "spaceout_enabled", lambda: state["on"])

    def switch(on: bool):
        state["on"] = bool(on)
    return switch


def _dialog(qtbot):
    dialog = prefs.PreferencesDialog()
    qtbot.addWidget(dialog)
    return dialog


def _titles(dialog):
    tabs = dialog.findChild(QTabWidget, "PreferencesTabs")
    return [tabs.tabText(i) for i in range(tabs.count())]


def test_ordinary_spacr_has_no_sound_tab(store, spaceout, qtbot):
    dialog = _dialog(qtbot)
    assert "Sound" not in _titles(dialog)
    assert dialog.findChild(QWidget, "PreferencesTabSound") is None
    assert dialog.findChild(QWidget, "SoundEnabled") is None


def test_spaceout_has_the_sound_tab_last(store, spaceout, qtbot):
    spaceout(True)
    dialog = _dialog(qtbot)
    assert _titles(dialog)[-1] == "Sound"
    assert dialog.findChild(QWidget, "SoundEnabled") is not None


def test_the_tab_follows_the_mode_the_next_time_preferences_opens(
        store, spaceout, qtbot):
    assert "Sound" not in _titles(_dialog(qtbot))
    spaceout(True)
    assert "Sound" in _titles(_dialog(qtbot))
    spaceout(False)
    assert "Sound" not in _titles(_dialog(qtbot))


def test_sound_is_off_by_default_in_both_modes(store, spaceout):
    assert prefs.get_sound_enabled() is False
    spaceout(True)
    assert prefs.get_sound_enabled() is False


def test_sound_switched_on_in_spaceout_is_silent_in_ordinary_spacr(
        store, spaceout):
    spaceout(True)
    prefs.set_sound_enabled(True)
    assert prefs.get_sound_enabled() is True
    spaceout(False)
    assert prefs.sound_is_offered() is False
    assert prefs.get_sound_enabled() is False
    assert prefs.get_saved_sound_enabled() is True, (
        "the stored switch was erased; spaceout would lose it")
    spaceout(True)
    assert prefs.get_sound_enabled() is True


def test_ordinary_spacr_builds_no_sound_engine_even_with_sound_stored_on(
        store, spaceout, qapp, monkeypatch):
    from spacr.qt import sound

    made = []
    monkeypatch.setattr(sound, "_create_engine",
                        lambda app: made.append(app))
    monkeypatch.setattr(sound, "_ENGINE", None)
    store.setValue("sound/enabled", True)
    assert sound.apply_sound_preferences(qapp) is None
    assert made == []


def test_saving_preferences_in_ordinary_spacr_leaves_the_sound_values(
        store, spaceout, qtbot, monkeypatch):
    """Save writes every control on the dialog; there is no sound control
    in ordinary spaCR, so nothing may write the sound keys."""
    monkeypatch.setattr(prefs, "apply_preferences_to_app", lambda *a: None)
    store.setValue("sound/enabled", True)
    store.setValue("sound/volume", 0.8)
    store.setValue("sound/event/hover", True)
    dialog = _dialog(qtbot)
    dialog.findChild(QDialogButtonBox).button(
        QDialogButtonBox.Save).click()
    assert prefs.get_saved_sound_enabled() is True
    assert prefs.get_sound_volume() == pytest.approx(0.8)
    assert prefs.get_sound_event_enabled("hover") is True


def test_a_process_that_never_imported_the_theme_offers_no_sound(
        monkeypatch):
    """Asked on paths kept free of QtGui; not importing the theme module
    means spaceout cannot have been switched on."""
    import sys

    monkeypatch.delitem(sys.modules, "spacr.qt.theme")
    assert prefs.sound_is_offered() is False

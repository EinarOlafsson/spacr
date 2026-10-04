"""Every start reopens the last session; unsaved settings survive a crash.

An ordinary start reopens the module, settings and folder that were on screen
when spaCR last closed or crashed, unless Preferences turns it off or the
start asked to open fresh. Unsaved settings of every open module are
autosaved as drafts; a clean close drops them, so drafts found at start-up
are offered back once.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

from spacr import restart_state                                  # noqa: E402


@pytest.fixture(autouse=True)
def _own_state_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SPACR_HOME", str(tmp_path / "spacr_home"))
    import spacr.qt.app as app_mod

    monkeypatch.setattr(app_mod, "_OPEN_FRESH", [False])
    yield
    restart_state.discard()
    restart_state._clear_drafts()


def _window(qtbot, **kwargs):
    from spacr.qt.app import MainWindow

    window = MainWindow(**kwargs)
    qtbot.addWidget(window)
    return window


def test_the_session_record_keeps_module_settings_and_folder():
    assert restart_state._save_session(
        "regression", {"src": "/data/plate1", "fdr_alpha": 0.01})
    state = restart_state._last_session()
    assert state["module"] == "regression"
    assert state["folder"] == "/data/plate1"
    assert state["settings"]["fdr_alpha"] == 0.01
    assert restart_state._last_session() is not None


def test_drafts_are_taken_once():
    assert restart_state._save_drafts({"regression": {"fdr_alpha": 0.2}})
    assert restart_state._take_drafts() == {"regression": {"fdr_alpha": 0.2}}
    assert restart_state._take_drafts() == {}


def test_an_empty_or_broken_store_reads_as_nothing(monkeypatch):
    assert restart_state._last_session() is None
    monkeypatch.setattr(restart_state, "_store",
                        lambda: (_ for _ in ()).throw(OSError("gone")))
    assert restart_state._last_session() is None
    assert restart_state._save_session("mask", {}) is False


def test_an_ordinary_start_reopens_the_last_module_with_its_settings(qtbot):
    restart_state._save_session("regression", {"fdr_alpha": 0.01})
    window = _window(qtbot)

    screen = window._screens.get("regression")
    assert screen is not None
    assert screen._settings_model.collect()["fdr_alpha"] == 0.01
    button = window.findChild(object, "OpenFreshButton")
    assert button is not None and button.text()


def test_the_session_is_reopened_on_every_start_not_once(qtbot):
    restart_state._save_session("regression", {"fdr_alpha": 0.01})
    _window(qtbot)
    second = _window(qtbot)
    assert "regression" in second._screens


def test_the_preference_switch_turns_reopening_off(qtbot):
    from spacr.qt.preferences import _get_restore_session, _set_restore_session

    assert _get_restore_session() is True
    _set_restore_session(False)
    restart_state._save_session("regression", {"fdr_alpha": 0.01})
    window = _window(qtbot)
    assert "regression" not in window._screens
    assert window._stack.currentWidget() is window._startup


def test_fresh_on_the_command_line_skips_the_session(qtbot, monkeypatch):
    import spacr.qt.app as app_mod

    monkeypatch.setattr(app_mod, "_OPEN_FRESH", [True])
    restart_state._save_session("regression", {"fdr_alpha": 0.01})
    window = _window(qtbot)
    assert "regression" not in window._screens


def test_open_fresh_puts_the_defaults_back_and_goes_home(qtbot):
    restart_state._save_session("regression", {"fdr_alpha": 0.01})
    window = _window(qtbot)
    screen = window._screens["regression"]
    default = window._fresh_settings[1]["fdr_alpha"]
    assert default != 0.01

    window._open_fresh()
    assert screen._settings_model.collect()["fdr_alpha"] == default
    assert window._stack.currentWidget() is window._startup
    assert window._open_fresh_button is None


def test_a_forced_restart_record_still_wins(qtbot):
    restart_state._save_session("regression", {"fdr_alpha": 0.01})
    restart_state.save(module="mask", settings={})
    window = _window(qtbot)
    assert window._stack.currentWidget() is window._screens.get("mask")


def test_a_clean_close_records_the_session_and_drops_drafts(qtbot):
    window = _window(qtbot, initial_app="regression")
    window._screens["regression"].apply_settings_dict({"fdr_alpha": 0.03})
    assert window._autosave_settings() is True
    assert window._autosave_settings() is False
    window._end_session_cleanly()

    assert restart_state._take_drafts() == {}
    state = restart_state._last_session()
    assert state["module"] == "regression"
    assert state["settings"]["fdr_alpha"] == 0.03


def test_the_autosave_timer_runs_in_the_window(qtbot):
    window = _window(qtbot)
    timer = window._settings_autosave_timer
    assert timer.isActive()
    assert timer.interval() == window._SETTINGS_AUTOSAVE_MS


def test_drafts_from_a_crash_are_offered_and_restored(qtbot):
    from PySide6.QtWidgets import QMessageBox

    restart_state._save_drafts({"regression": {"fdr_alpha": 0.04}})
    window = _window(qtbot, initial_app="mask")
    qtbot.waitUntil(lambda: getattr(window, "_drafts_box", None) is not None)
    box = window._drafts_box
    assert "regression" in box.text()
    assert box.isVisible()
    assert restart_state._take_drafts() == {}

    restore = next(b for b in box.buttons()
                   if box.buttonRole(b) == QMessageBox.ButtonRole.AcceptRole)
    restore.click()
    values = window._screens["regression"]._settings_model.collect()
    assert values["fdr_alpha"] == 0.04
    assert window._stack.currentWidget() is window._screens["mask"]


def test_a_draft_equal_to_the_reopened_session_is_not_offered(qtbot):
    restart_state._save_session("regression", {"fdr_alpha": 0.01})
    restart_state._save_drafts({"regression": {"fdr_alpha": 0.01}})
    window = _window(qtbot)
    qtbot.wait(50)
    assert getattr(window, "_drafts_box", None) is None


def test_the_preferences_dialog_shows_and_saves_the_session_switch(
        qtbot, monkeypatch):
    from PySide6.QtWidgets import QDialogButtonBox, QWidget

    from spacr.qt import preferences as prefs

    monkeypatch.setattr(prefs, "apply_preferences_to_app", lambda *args: None)
    dialog = prefs.PreferencesDialog()
    qtbot.addWidget(dialog)
    switch = dialog.findChild(QWidget, "RestoreLastSession")
    assert switch is not None and switch.isChecked() is True
    switch.setChecked(False)
    dialog.findChild(QDialogButtonBox).accepted.emit()
    assert prefs._get_restore_session() is False

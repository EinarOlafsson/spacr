"""Normal launches open Home; explicit restarts restore UI and drafts survive crashes.

spaCR no longer remembers or reopens the last session: nothing is written
under ``session/last`` and Preferences has no switch for it.
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
    monkeypatch.setattr(app_mod, "_RESUME_RESTART", [False])
    yield
    restart_state._flush_writes()
    restart_state.discard()
    restart_state._clear_drafts()


def _window(qtbot, **kwargs):
    from spacr.qt.app import MainWindow

    window = MainWindow(**kwargs)
    qtbot.addWidget(window)
    return window


def test_the_last_session_feature_is_gone():
    from spacr.qt import preferences as prefs
    from spacr.qt.app import MainWindow

    for name in ("_last_session", "_save_session", "_SESSION_KEY",
                 "_folder_of"):
        assert not hasattr(restart_state, name), name
    for name in ("_get_restore_session", "_set_restore_session",
                 "_KEY_RESTORE_SESSION", "DEFAULT_RESTORE_SESSION"):
        assert not hasattr(prefs, name), name
    for name in ("_resume_last_session", "_offer_open_fresh", "_open_fresh",
                 "_drop_open_fresh", "_OPEN_FRESH_SHOWN_MS"):
        assert not hasattr(MainWindow, name), name


def test_drafts_are_taken_once():
    assert restart_state._save_drafts({"regression": {"fdr_alpha": 0.2}})
    assert restart_state._take_drafts() == {"regression": {"fdr_alpha": 0.2}}
    assert restart_state._take_drafts() == {}


def test_an_empty_or_broken_store_reads_as_nothing(monkeypatch):
    assert restart_state._take_drafts() == {}
    monkeypatch.setattr(restart_state, "_store",
                        lambda: (_ for _ in ()).throw(OSError("gone")))
    assert restart_state._take_drafts() == {}
    assert restart_state._save_drafts({"mask": {"src": "/data"}}) is False


@pytest.mark.parametrize('navigator', [False, True])
def test_an_ordinary_start_always_opens_selected_home(qtbot, navigator):
    from spacr.qt.preferences import set_easy_mode
    from spacr.qt.widgets.home import _EasyHomePage

    set_easy_mode(navigator)
    restart_state.save(module='mask', settings={'src': '/stale/restart'})
    window = _window(qtbot)
    assert window._stack.currentWidget() is window._startup
    assert isinstance(window._startup, _EasyHomePage) is navigator
    assert window.findChild(object, 'OpenFreshButton') is None


def test_closed_module_does_not_reopen_on_next_launch(qtbot):
    previous = _window(qtbot, initial_app='regression')
    previous._screens['regression'].apply_settings_dict({'fdr_alpha': 0.03})
    previous._end_session_cleanly()
    previous.close()
    second = _window(qtbot)
    assert second._stack.currentWidget() is second._startup
    assert 'regression' not in second._screens


def test_explicit_module_request_still_wins(qtbot, monkeypatch):
    import spacr.qt.app as app_mod

    restart_state.save(module='mask', settings={})
    app_mod._RESUME_RESTART[0] = True
    window = _window(qtbot, initial_app='measure')
    assert window._stack.currentWidget() is window._screens['measure']
    assert app_mod._RESUME_RESTART[0] is False


def test_an_explicit_restart_restores_once_then_next_launch_opens_home(qtbot):
    import spacr.qt.app as app_mod

    app_mod._RESUME_RESTART[0] = True
    restart_state.save(module="mask", settings={})
    window = _window(qtbot)
    assert window._stack.currentWidget() is window._screens.get("mask")
    assert restart_state.peek() is None and app_mod._RESUME_RESTART[0] is False
    next_window = _window(qtbot)
    assert next_window._stack.currentWidget() is next_window._startup


def test_a_clean_close_drops_drafts_and_records_no_session(qtbot):
    window = _window(qtbot, initial_app="regression")
    window._screens["regression"].apply_settings_dict({"fdr_alpha": 0.03})
    assert window._autosave_settings() is True
    assert window._autosave_settings() is False
    window._end_session_cleanly()

    assert restart_state._take_drafts() == {}
    assert not restart_state._store().contains("session/last")


def test_the_autosave_timer_runs_in_the_window(qtbot):
    window = _window(qtbot)
    timer = window._settings_autosave_timer
    assert timer.isActive()
    assert timer.interval() == window._SETTINGS_AUTOSAVE_MS


def test_a_failed_autosave_retries_unchanged_settings(qtbot, monkeypatch):
    window = _window(qtbot, initial_app="regression")
    window._screens["regression"].apply_settings_dict({"fdr_alpha": 0.08})
    checkpoint = window._last_autosave
    calls = []
    original = restart_state._save_drafts

    def attempt(*args):
        calls.append(1)
        if len(calls) == 1:
            return False
        return original(*args)

    monkeypatch.setattr(restart_state, "_save_drafts", attempt)
    assert window._autosave_settings() is True
    assert restart_state._flush_writes()
    assert window._last_autosave == checkpoint
    assert window._autosave_settings() is True
    assert restart_state._flush_writes()
    assert len(calls) == 2
    assert window._autosave_settings() is False
    assert len(calls) == 2
    assert restart_state._take_drafts()["regression"]["fdr_alpha"] == 0.08


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


def test_a_draft_from_previous_session_is_still_offered_on_home(qtbot):
    restart_state._save_drafts({"regression": {"fdr_alpha": 0.01}})
    window = _window(qtbot)
    qtbot.waitUntil(lambda: getattr(window, "_drafts_box", None) is not None)
    assert window._stack.currentWidget() is window._startup


def test_preferences_have_no_restore_session_switch(qtbot, monkeypatch, tmp_path):
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QDialogButtonBox, QWidget

    from spacr.qt import preferences as prefs

    path = str(tmp_path / 'preferences.ini')
    monkeypatch.setattr(prefs, '_settings', lambda: QSettings(path, QSettings.IniFormat))
    monkeypatch.setattr(prefs, 'apply_preferences_to_app', lambda *args: None)
    dialog = prefs.PreferencesDialog()
    qtbot.addWidget(dialog)
    assert dialog.findChild(QWidget, 'RestoreLastSession') is None
    dialog.findChild(QDialogButtonBox).accepted.emit()
    assert not QSettings(path, QSettings.IniFormat).contains(
        'prefs/restore_last_session')


def test_autosave_writes_on_the_writer_thread_not_the_gui_thread(qtbot, monkeypatch):
    """A slow disk holds the writer, never the window (lesson 13: 1.2-4.4 s)."""
    import threading

    window = _window(qtbot, initial_app="regression")
    window._screens["regression"].apply_settings_dict({"fdr_alpha": 0.021})
    threads = []
    original = restart_state._save_drafts

    def recorded(*args):
        threads.append(threading.current_thread())
        return original(*args)

    monkeypatch.setattr(restart_state, "_save_drafts", recorded)
    assert window._autosave_settings() is True
    assert restart_state._flush_writes()
    assert threads and threading.main_thread() not in threads
    assert {thread.name for thread in threads} == {"spacr-settings-writer"}
    assert restart_state._take_drafts()["regression"]["fdr_alpha"] == 0.021


def test_rapid_autosaves_store_only_the_newest_snapshot(qtbot, monkeypatch):
    import threading

    window = _window(qtbot, initial_app="regression")
    started, release = threading.Event(), threading.Event()
    written = []
    original = restart_state._save_drafts

    def held(drafts):
        written.append(drafts["regression"]["fdr_alpha"])
        started.set()
        release.wait(10)
        return original(drafts)

    monkeypatch.setattr(restart_state, "_save_drafts", held)
    try:
        window._screens["regression"].apply_settings_dict({"fdr_alpha": 0.011})
        assert window._autosave_settings() is True
        assert started.wait(10)
        for value in (0.012, 0.013, 0.014):
            window._screens["regression"].apply_settings_dict({"fdr_alpha": value})
            assert window._autosave_settings() is True
    finally:
        release.set()
    assert restart_state._flush_writes()
    assert written == [0.011, 0.014]
    assert restart_state._take_drafts()["regression"]["fdr_alpha"] == 0.014


def test_a_clean_close_waits_for_the_last_queued_snapshot(qtbot, monkeypatch):
    """The exit flush stores the newest snapshot before the drafts are dropped."""
    import threading

    window = _window(qtbot, initial_app="regression")
    release = threading.Event()
    order = []
    save_drafts = restart_state._save_drafts
    clear = restart_state._clear_drafts

    def slow_drafts(drafts):
        release.wait(10)
        order.append(("drafts", threading.current_thread().name,
                      drafts["regression"]["fdr_alpha"]))
        return save_drafts(drafts)

    def recorded_clear():
        order.append(("clear", threading.current_thread().name, None))
        return clear()

    monkeypatch.setattr(restart_state, "_save_drafts", slow_drafts)
    monkeypatch.setattr(restart_state, "_clear_drafts", recorded_clear)
    window._screens["regression"].apply_settings_dict({"fdr_alpha": 0.044})
    assert window._autosave_settings() is True
    threading.Timer(0.2, release.set).start()
    window._end_session_cleanly()
    assert order[0] == ("drafts", "spacr-settings-writer", 0.044)
    assert order[-1][0] == "clear"
    assert restart_state._take_drafts() == {}


def test_a_write_that_raises_is_logged_and_the_snapshot_retried(qtbot, monkeypatch, caplog):
    window = _window(qtbot, initial_app="regression")
    window._screens["regression"].apply_settings_dict({"fdr_alpha": 0.055})
    checkpoint = window._last_autosave
    calls = []
    original = restart_state._save_drafts

    def flaky(drafts):
        calls.append(1)
        if len(calls) == 1:
            raise OSError("disk went away")
        return original(drafts)

    monkeypatch.setattr(restart_state, "_save_drafts", flaky)
    with caplog.at_level("WARNING", logger="spacr.restart_state"):
        assert window._autosave_settings() is True
        assert restart_state._flush_writes()
    assert "disk went away" in caplog.text
    assert window._last_autosave == checkpoint
    assert window._autosave_settings() is True
    assert restart_state._flush_writes()
    assert len(calls) == 2
    assert restart_state._take_drafts()["regression"]["fdr_alpha"] == 0.055


@pytest.mark.parametrize('args,target,resume,fresh', [
    ([], None, False, False),
    (['--resume-restart'], None, True, False),
    (['--resume-restart', 'measure'], 'measure', True, False),
    (['--fresh', '--resume-restart'], None, False, True),
    (['measure', '--fresh'], 'measure', False, True),
])
def test_startup_flags_never_become_module_names(monkeypatch, args, target, resume, fresh):
    from spacr.qt import app

    assert app._startup_target(args) == target
    assert app._RESUME_RESTART[0] is resume
    assert app._OPEN_FRESH[0] is fresh


def test_navigator_launcher_enables_navigator_without_overriding_restart(monkeypatch):
    from spacr.qt import app
    from spacr.qt.preferences import get_easy_mode, set_easy_mode

    monkeypatch.setattr(app.sys, 'argv', ['spacr-navigator'])
    app._RESUME_RESTART[0] = True
    set_easy_mode(False)
    assert app._navigator_launch_target(None) is None
    assert get_easy_mode() is True
    app._RESUME_RESTART[0] = False
    assert app._navigator_launch_target(None) == '__home__'

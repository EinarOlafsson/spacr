"""Normal launches open Home; explicit restarts restore UI and drafts survive crashes."""
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


@pytest.mark.parametrize('navigator', [False, True])
@pytest.mark.parametrize('legacy_restore', [False, True])
def test_an_ordinary_start_always_opens_selected_home(qtbot, navigator, legacy_restore):
    from spacr.qt.preferences import _set_restore_session, set_easy_mode
    from spacr.qt.widgets.home import _EasyHomePage

    set_easy_mode(navigator)
    _set_restore_session(legacy_restore)
    restart_state._save_session('regression', {'fdr_alpha': 0.01})
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


def test_explicit_module_request_still_wins(qtbot, monkeypatch):
    import spacr.qt.app as app_mod

    restart_state._save_session('regression', {'fdr_alpha': 0.01})
    restart_state.save(module='mask', settings={})
    app_mod._RESUME_RESTART[0] = True
    window = _window(qtbot, initial_app='measure')
    assert window._stack.currentWidget() is window._screens['measure']
    assert app_mod._RESUME_RESTART[0] is False


def test_fresh_on_the_command_line_skips_the_session(qtbot, monkeypatch):
    import spacr.qt.app as app_mod

    monkeypatch.setattr(app_mod, "_OPEN_FRESH", [True])
    restart_state._save_session("regression", {"fdr_alpha": 0.01})
    window = _window(qtbot)
    assert "regression" not in window._screens


def test_open_fresh_puts_the_defaults_back_and_goes_home(qtbot):
    restart_state._save_session("regression", {"fdr_alpha": 0.01})
    window = _window(qtbot)
    window._resume_last_session()
    screen = window._screens["regression"]
    default = window._fresh_settings[1]["fdr_alpha"]
    assert default != 0.01

    window._open_fresh()
    assert screen._settings_model.collect()["fdr_alpha"] == default
    assert window._stack.currentWidget() is window._startup
    assert window._open_fresh_button is None


def test_an_explicit_restart_restores_once_then_next_launch_opens_home(qtbot):
    import spacr.qt.app as app_mod

    app_mod._RESUME_RESTART[0] = True
    restart_state._save_session("regression", {"fdr_alpha": 0.01})
    restart_state.save(module="mask", settings={})
    window = _window(qtbot)
    assert window._stack.currentWidget() is window._screens.get("mask")
    assert restart_state.peek() is None and app_mod._RESUME_RESTART[0] is False
    next_window = _window(qtbot)
    assert next_window._stack.currentWidget() is next_window._startup


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


@pytest.mark.parametrize("failed", ["session", "drafts", "both"])
def test_a_failed_autosave_retries_unchanged_settings(qtbot, monkeypatch, failed):
    window = _window(qtbot, initial_app="regression")
    window._screens["regression"].apply_settings_dict({"fdr_alpha": 0.08})
    checkpoint = window._last_autosave
    counts = {"session": 0, "drafts": 0}

    def save(kind, original):
        def attempt(*args):
            counts[kind] += 1
            if counts[kind] == 1 and failed in {kind, "both"}:
                return False
            return original(*args)
        return attempt

    for kind, name in (("session", "_save_session"), ("drafts", "_save_drafts")):
        monkeypatch.setattr(restart_state, name, save(kind, getattr(restart_state, name)))
    assert window._autosave_settings() is True
    assert restart_state._flush_writes()
    assert window._last_autosave == checkpoint
    assert window._autosave_settings() is True
    assert restart_state._flush_writes()
    assert counts == {"session": 2, "drafts": 2}
    assert window._autosave_settings() is False
    assert counts == {"session": 2, "drafts": 2}
    assert restart_state._last_session()["settings"]["fdr_alpha"] == 0.08
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
    restart_state._save_session("regression", {"fdr_alpha": 0.01})
    restart_state._save_drafts({"regression": {"fdr_alpha": 0.01}})
    window = _window(qtbot)
    qtbot.waitUntil(lambda: getattr(window, "_drafts_box", None) is not None)
    assert window._stack.currentWidget() is window._startup


@pytest.mark.parametrize('saved', [False, True])
def test_preferences_omit_legacy_restore_switch_without_changing_saved_value(
        qtbot, monkeypatch, tmp_path, saved):
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QDialogButtonBox, QWidget

    from spacr.qt import preferences as prefs

    path = str(tmp_path / 'preferences.ini')
    monkeypatch.setattr(prefs, '_settings', lambda: QSettings(path, QSettings.IniFormat))
    monkeypatch.setattr(prefs, 'apply_preferences_to_app', lambda *args: None)
    prefs._set_restore_session(saved)
    dialog = prefs.PreferencesDialog()
    qtbot.addWidget(dialog)
    assert dialog.findChild(QWidget, 'RestoreLastSession') is None
    dialog.findChild(QDialogButtonBox).accepted.emit()
    assert prefs._get_restore_session() is saved
    reopened = prefs.PreferencesDialog()
    qtbot.addWidget(reopened)
    assert reopened.findChild(QWidget, 'RestoreLastSession') is None
    reopened.reject()
    assert prefs._get_restore_session() is saved


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
    assert restart_state._last_session()["settings"]["fdr_alpha"] == 0.014
    assert restart_state._take_drafts()["regression"]["fdr_alpha"] == 0.014


def test_a_clean_close_waits_for_the_last_queued_snapshot(qtbot, monkeypatch):
    """The exit flush stores the newest snapshot before the drafts are dropped."""
    import threading

    window = _window(qtbot, initial_app="regression")
    release = threading.Event()
    order = []
    save_session = restart_state._save_session
    clear = restart_state._clear_drafts

    def slow_session(*args):
        release.wait(10)
        order.append(("session", threading.current_thread().name, args[1]["fdr_alpha"]))
        return save_session(*args)

    def recorded_clear():
        order.append(("clear", threading.current_thread().name, None))
        return clear()

    monkeypatch.setattr(restart_state, "_save_session", slow_session)
    monkeypatch.setattr(restart_state, "_clear_drafts", recorded_clear)
    window._screens["regression"].apply_settings_dict({"fdr_alpha": 0.044})
    assert window._autosave_settings() is True
    threading.Timer(0.2, release.set).start()
    window._end_session_cleanly()
    assert order[0] == ("session", "spacr-settings-writer", 0.044)
    assert order[-1][0] == "clear"
    assert restart_state._take_drafts() == {}
    assert restart_state._last_session()["settings"]["fdr_alpha"] == 0.044


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


def test_selected_fold_settings_changes_are_autosaved_even_when_host_is_unchanged(qtbot):
    from types import SimpleNamespace

    from PySide6.QtWidgets import QTabWidget, QWidget

    window = _window(qtbot, initial_app='regression')
    host = window._screens['regression']
    pages = QTabWidget()
    qtbot.addWidget(pages)
    pages.addTab(QWidget(), 'Host')
    values = {'src': '/experiment/ops', 'round': 1}
    page = QWidget()
    page.app_key = 'ops'
    page._settings_model = SimpleNamespace(collect=lambda: dict(values))
    pages.addTab(page, 'OPS')
    pages.setCurrentWidget(page)
    host._fold_pages = pages
    assert window._autosave_settings()
    assert restart_state._flush_writes()
    values['round'] = 2
    assert window._autosave_settings()
    assert restart_state._flush_writes()
    assert restart_state._last_session()['module'] == 'ops'
    assert restart_state._last_session()['settings']['round'] == 2

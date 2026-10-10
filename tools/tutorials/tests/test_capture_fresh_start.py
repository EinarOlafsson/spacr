"""Every tutorial recording starts fresh (647/648 session restore).

spaCR reopens the last module, settings and folder on an ordinary start
(Preferences -> Session, default on) and offers autosaved settings back after
a crash. A recording must show what a viewer gets on a fresh start, so every
capture path starts the app as ``spacr --fresh`` with no remembered session
and no crash drafts, and capture refuses a window that started in restore
mode. The switch stays at its default (on) so Preferences scenes show the
default truthfully.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

TUTORIALS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TUTORIALS))
import capture_policy  # noqa: E402
from capture_policy import (  # noqa: E402
    DRAFTS_KEY,
    RESTORE_SESSION_KEY,
    SESSION_KEY,
    force_fresh_start,
    force_fresh_start_in_profiles,
    fresh_argv,
    launcher_accepts_fresh,
    verify_fresh_start,
    verify_profile_starts_fresh,
)

from spacr import restart_state  # noqa: E402


@pytest.fixture(autouse=True)
def _private_store(tmp_path, monkeypatch):
    """A private preference store and an ordinary (not fresh) start flag."""
    from PySide6.QtCore import QSettings

    from spacr.qt import app as gui
    from spacr.qt import preferences as prefs

    monkeypatch.setenv("SPACR_HOME", str(tmp_path / "spacr_home"))
    store = QSettings(str(tmp_path / "qt.conf"), QSettings.IniFormat)
    monkeypatch.setattr(prefs, "_settings", lambda: store)
    monkeypatch.setattr(gui, "_OPEN_FRESH", [False])
    yield store
    restart_state.discard()


def _restore_mode(store):
    """A profile that would reopen Regression and offer crash drafts."""
    restart_state._save_session("regression", {"fdr_alpha": 0.01})
    restart_state._save_drafts({"regression": {"fdr_alpha": 0.04}})
    assert store.contains(SESSION_KEY) and store.contains(DRAFTS_KEY)


def _window(qtbot):
    from spacr.qt.app import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    qtbot.wait(50)                       # the drafts offer runs on the next turn
    return window


def test_force_fresh_start_sets_the_flag_keeps_the_default_and_drops_drafts(_private_store):
    from spacr.qt import app as gui
    from spacr.qt.preferences import _get_restore_session, _set_restore_session

    _restore_mode(_private_store)
    _set_restore_session(False)              # an earlier profile turned it off
    assert force_fresh_start() is True
    assert gui._OPEN_FRESH[0] is True
    assert _get_restore_session() is True    # the default, shown truthfully
    assert not _private_store.contains(RESTORE_SESSION_KEY)
    assert not _private_store.contains(SESSION_KEY)
    assert not _private_store.contains(DRAFTS_KEY)


def test_configure_appearance_starts_every_recording_fresh(_private_store):
    from spacr.qt import app as gui

    from spacr.qt.preferences import _get_restore_session

    capture_policy.configure_appearance()
    assert gui._OPEN_FRESH[0] is True
    assert _get_restore_session() is True


def test_a_fresh_recording_window_opens_home_with_no_drafts_offer(qtbot, _private_store):
    from spacr.qt.preferences import _get_restore_session

    _restore_mode(_private_store)
    force_fresh_start()
    assert _get_restore_session() is True    # restore on, yet nothing reopens
    window = _window(qtbot)
    assert "regression" not in window._screens
    assert window._stack.currentWidget() is window._startup
    assert getattr(window, "_drafts_box", None) is None
    assert verify_fresh_start(window) is True


def test_capture_refuses_a_window_that_started_in_restore_mode(qtbot, _private_store):
    _restore_mode(_private_store)
    window = _window(qtbot)              # an ordinary start: reopens Regression
    assert "regression" in window._screens
    with pytest.raises(RuntimeError, match="restore mode"):
        verify_fresh_start(window)


def test_fresh_wins_over_the_default_restore_switch_even_with_a_session(qtbot, _private_store):
    """--fresh alone keeps a stored session from reopening (switch on)."""
    from spacr.qt import app as gui

    restart_state._save_session("regression", {"fdr_alpha": 0.01})
    gui._OPEN_FRESH[0] = True
    window = _window(qtbot)
    assert "regression" not in window._screens
    assert verify_fresh_start(window) is True


def test_capture_refuses_a_start_without_fresh_even_with_nothing_to_reopen(_private_store):
    with pytest.raises(RuntimeError, match="restore mode"):
        verify_fresh_start()


class _Window:
    _restored_session_settings = ("", {})
    _drafts_box = None
    _pending_drafts = {}


def test_capture_refuses_a_reopened_session_or_a_crash_offer(_private_store):
    force_fresh_start()
    assert verify_fresh_start(_Window()) is True
    reopened = _Window()
    reopened._restored_session_settings = ("measure", {"src": "x"})
    with pytest.raises(RuntimeError, match="reopened the last session"):
        verify_fresh_start(reopened)
    offered = _Window()
    offered._pending_drafts = {"measure": {"src": "x"}}
    with pytest.raises(RuntimeError, match="unsaved settings from a crash"):
        verify_fresh_start(offered)


def test_every_frame_check_includes_the_fresh_start_check():
    source = (TUTORIALS / "capture_policy.py").read_text()
    body = source[source.index("def verify_appearance("):source.index("def exclude_special_backdrops(")]
    assert "verify_fresh_start(window)" in body
    body = source[source.index("def configure_appearance("):source.index("def verify_appearance(")]
    assert "force_fresh_start()" in body


def test_profiles_are_written_with_the_default_switch_and_no_session(tmp_path):
    from PySide6.QtCore import QSettings

    home = tmp_path / "config"
    store = QSettings(str(home / "spacr" / "qt.conf"), QSettings.IniFormat)
    store.setValue(SESSION_KEY, '{"module": "measure"}')
    store.setValue(DRAFTS_KEY, '{"modules": {}}')
    store.setValue("prefs/theme", "dark")
    store.setValue(RESTORE_SESSION_KEY, False)
    store.sync()
    with pytest.raises(RuntimeError, match="crash drafts"):
        verify_profile_starts_fresh(home)
    assert force_fresh_start_in_profiles([home]) == [home / "spacr" / "qt.conf"]
    store = QSettings(str(home / "spacr" / "qt.conf"), QSettings.IniFormat)
    assert not store.contains(RESTORE_SESSION_KEY)          # default (on)
    assert not store.contains(SESSION_KEY) and not store.contains(DRAFTS_KEY)
    assert store.value("prefs/theme") == "dark"
    assert verify_profile_starts_fresh(home) == home / "spacr" / "qt.conf"


def test_the_profile_check_reads_a_real_session_record(tmp_path, _private_store, monkeypatch):
    from PySide6.QtCore import QSettings

    from spacr.qt import preferences as prefs

    home = tmp_path / "profile"
    store = QSettings(str(home / "spacr" / "qt.conf"), QSettings.IniFormat)
    monkeypatch.setattr(prefs, "_settings", lambda: store)
    restart_state._save_session("", {})               # closed on Home
    assert verify_profile_starts_fresh(home)
    restart_state._save_session("measure", {"src": "/data/plate"})
    with pytest.raises(RuntimeError, match="reopen the last session"):
        verify_profile_starts_fresh(home)
    assert verify_profile_starts_fresh(tmp_path / "brand-new")


def test_the_shell_launcher_profile_call_clears_the_session(tmp_path):
    from PySide6.QtCore import QSettings

    store = QSettings(str(tmp_path / "spacr" / "qt.conf"), QSettings.IniFormat)
    store.setValue(SESSION_KEY, '{"module": "measure"}')
    store.sync()
    assert capture_policy.main(["--force-alpha-off", str(tmp_path)]) == 0
    store = QSettings(str(tmp_path / "spacr" / "qt.conf"), QSettings.IniFormat)
    assert not store.contains(SESSION_KEY)
    assert not store.contains(RESTORE_SESSION_KEY)
    script = (TUTORIALS / "run_neutral_capture.sh").read_text()
    assert "capture_policy.py" in script and "--force-alpha-off" in script


def test_fresh_argv_and_the_installed_launcher_probe(tmp_path):
    assert fresh_argv() == ["--fresh"]
    assert fresh_argv(["--no-setup", "--fresh"]) == ["--fresh", "--no-setup"]
    new = tmp_path / "new_app.py"
    new.write_text("_OPEN_FRESH = [False]\n")
    old = tmp_path / "old_app.py"
    old.write_text("def main(argv=None): ...\n")
    assert launcher_accepts_fresh(new) is True
    assert launcher_accepts_fresh(old) is False
    assert launcher_accepts_fresh(tmp_path / "missing.py") is False
    import spacr.qt.app as gui
    assert launcher_accepts_fresh(gui.__file__) is True


_RECORDERS = sorted(
    path for path in [*TUTORIALS.glob("*.py"), *(TUTORIALS / "authoring" / "tools").glob("*.py")]
    if re.search(r"\bMainWindow\(\)", path.read_text()))


def test_the_recorder_list_is_not_empty():
    names = {path.name for path in _RECORDERS}
    assert {"capture_refresh.py", "capture_suggest_segment.py",
            "capture_all_modules.py", "capture_workflow_modules.py"} <= names


@pytest.mark.parametrize("path", _RECORDERS, ids=lambda p: p.name)
def test_every_recorder_starts_fresh_and_checks_its_window(path):
    """Fresh before the window is built; refused after if it restored."""
    source = path.read_text()
    built = re.search(r"\bwindow = [\w.]*MainWindow\(\)", source).start()
    before = source[:built]
    assert "force_fresh_start()" in before or "configure_appearance(" in before, path.name
    assert "verify_fresh_start(window)" in source[built:], path.name


def test_the_openings_recorder_checks_the_fresh_start():
    source = (TUTORIALS / "capture_openings.py").read_text()
    body = source[source.index("def record_openings("):source.index("def _open(")]
    assert body.index("verify_fresh_start(window)") < body.index("go_home()\n    capture(")


def test_installed_app_recordings_start_fresh():
    pip = (TUTORIALS / "capture_pip_installation.py").read_text()
    assert "_set_restore_session" not in pip       # the default stays on
    assert "fresh_argv(" in pip and "launcher_accepts_fresh(" in pip
    assert "verify_profile_starts_fresh(" in pip
    terminal = (TUTORIALS / "capture_terminal_install.py").read_text()
    body = terminal[terminal.index("    def gui(self"):]
    assert body.index("verify_profile_starts_fresh(") < body.index("self.send(command")


# 651/652: "What's new" after an update never appears in a recording.

def test_force_fresh_start_marks_the_running_version_seen(_private_store):
    from spacr.qt.preferences import _note_running_version
    from spacr.updater import _installed_version

    _private_store.setValue(capture_policy.WHATS_NEW_SEEN_KEY, "0.0.1")
    force_fresh_start()
    assert capture_policy.running_version() == _installed_version()
    if _installed_version() != "unknown":
        assert _private_store.value(capture_policy.WHATS_NEW_SEEN_KEY) == _installed_version()
    assert _note_running_version(_installed_version()) is None    # no dialog


def test_profiles_mark_the_version_seen(tmp_path):
    from PySide6.QtCore import QSettings

    store = QSettings(str(tmp_path / "spacr" / "qt.conf"), QSettings.IniFormat)
    store.setValue(capture_policy.WHATS_NEW_SEEN_KEY, "0.0.1")
    store.setValue(capture_policy.WHATS_NEW_PREVIOUS_KEY, "0.0.0")
    store.sync()
    force_fresh_start_in_profiles([tmp_path], version="9.9.9")
    store = QSettings(str(tmp_path / "spacr" / "qt.conf"), QSettings.IniFormat)
    assert store.value(capture_policy.WHATS_NEW_SEEN_KEY) == "9.9.9"
    assert not store.contains(capture_policy.WHATS_NEW_PREVIOUS_KEY)


def test_a_fresh_window_shows_no_whats_new(qtbot, _private_store):
    _private_store.setValue(capture_policy.WHATS_NEW_SEEN_KEY, "0.0.1")
    force_fresh_start()
    window = _window(qtbot)
    window._maybe_show_whats_new()
    qtbot.wait(50)
    assert getattr(window, "_whats_new_dialog", None) is None
    assert verify_fresh_start(window) is True


def test_capture_refuses_an_open_whats_new_dialog(qtbot, _private_store):
    from PySide6.QtWidgets import QDialog

    force_fresh_start()
    dialog = QDialog()
    dialog.setObjectName(capture_policy.WHATS_NEW_DIALOG)
    qtbot.addWidget(dialog)
    dialog.show()
    with pytest.raises(RuntimeError, match="What's new"):
        verify_fresh_start()
    holder = _Window()
    holder._whats_new_dialog = dialog
    with pytest.raises(RuntimeError, match="What's new"):
        capture_policy.verify_no_whats_new(holder)
    dialog.hide()
    assert verify_fresh_start(holder) is True


def test_capture_refuses_a_whats_new_window_title():
    capture_policy.refuse_whats_new_titles(["spaCR", "Set spaCR up — spaCR"])
    with pytest.raises(RuntimeError, match="What's new"):
        capture_policy.refuse_whats_new_titles(["spaCR", "What's new in spaCR — spaCR"])


def test_installed_app_recordings_never_record_whats_new():
    pip = (TUTORIALS / "capture_pip_installation.py").read_text()
    assert "updates/last_seen_version" in pip
    body = pip[pip.index("    def snapshot(name):"):]
    assert body.index("refuse_whats_new_titles(") < body.index("grabWindow(0)")
    terminal = (TUTORIALS / "capture_terminal_install.py").read_text()
    body = terminal[terminal.index("    def shot(self"):]
    # The grab is ffmpeg x11grab since 0fbbd1e47 (was ImageMagick import).
    assert body.index("refuse_whats_new_titles(") < body.index("'x11grab'")

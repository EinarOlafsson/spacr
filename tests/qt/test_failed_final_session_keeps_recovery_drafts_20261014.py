"""Closing the window must not discard the last recoverable settings on error."""
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QTimer

from spacr import restart_state
from spacr.qt.app import MainWindow


@pytest.mark.parametrize("outcome", ["failed", "raised", "home"])
def test_failed_final_session_preserves_saved_drafts(qtbot, tmp_path, monkeypatch, outcome):
    monkeypatch.setenv("SPACR_HOME", str(tmp_path / "settings"))
    drafts = {"regression": {"fdr_alpha": 0.031}, "mask": {"src": "/saved/plate"}}
    assert restart_state._save_drafts(drafts)
    timer = QTimer()
    timer.start(30000)

    def save(module, settings):
        if outcome == "raised":
            raise OSError("The final settings write failed")
        return False

    monkeypatch.setattr(restart_state, "_save_session", save)
    window = SimpleNamespace(
        _settings_autosave_timer=timer,
        _current_session=lambda: ("" if outcome == "home" else "regression", {}),
    )
    MainWindow._end_session_cleanly(window)
    assert not timer.isActive()
    assert restart_state._take_drafts() == ({} if outcome == "home" else drafts)


@pytest.mark.parametrize("action", ["autosave", "close"])
@pytest.mark.parametrize("error", [OSError, ValueError])
def test_failed_settings_collection_keeps_the_previous_recovery_state(
        qtbot, tmp_path, monkeypatch, action, error):
    monkeypatch.setenv("SPACR_HOME", str(tmp_path / "settings"))
    settings = {"fdr_alpha": 0.037}
    drafts = {"regression": settings, "mask": {"src": "/saved/plate"}}
    assert restart_state._save_session("regression", settings)
    assert restart_state._save_drafts(drafts)

    def collect():
        raise error("The settings could not be collected")

    screen = SimpleNamespace(app_key="regression", _settings_model=SimpleNamespace(collect=collect))
    timer = QTimer()
    timer.start(30000)
    window = SimpleNamespace(
        _settings_autosave_timer=timer, _last_autosave="previous checkpoint",
        _collect_screen=MainWindow._collect_screen,
        _stack=SimpleNamespace(currentWidget=lambda: screen),
        _open_app_screens=lambda: {"regression": screen},
    )
    window._current_session = lambda: MainWindow._current_session(window)
    if action == "autosave":
        assert MainWindow._autosave_settings(window) is False
        assert timer.isActive()
        assert window._last_autosave == "previous checkpoint"
    else:
        MainWindow._end_session_cleanly(window)
        assert not timer.isActive()
    assert restart_state._last_session()["settings"] == settings
    assert restart_state._take_drafts() == drafts
    assert MainWindow._collect_screen(screen) == {}

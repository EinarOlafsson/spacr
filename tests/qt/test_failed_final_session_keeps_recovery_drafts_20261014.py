"""A failed autosave keeps the last recoverable drafts; a clean close drops them."""
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QTimer

from spacr import restart_state
from spacr.qt.app import MainWindow


def test_a_clean_close_stops_the_autosave_and_drops_the_drafts(qtbot, tmp_path, monkeypatch):
    monkeypatch.setenv("SPACR_HOME", str(tmp_path / "settings"))
    drafts = {"regression": {"fdr_alpha": 0.031}, "mask": {"src": "/saved/plate"}}
    assert restart_state._save_drafts(drafts)
    timer = QTimer()
    timer.start(30000)
    window = SimpleNamespace(_settings_autosave_timer=timer)
    MainWindow._end_session_cleanly(window)
    assert not timer.isActive()
    assert restart_state._take_drafts() == {}


@pytest.mark.parametrize("error", [OSError, ValueError])
def test_failed_settings_collection_keeps_the_previous_recovery_state(
        qtbot, tmp_path, monkeypatch, error):
    monkeypatch.setenv("SPACR_HOME", str(tmp_path / "settings"))
    drafts = {"regression": {"fdr_alpha": 0.037}, "mask": {"src": "/saved/plate"}}
    assert restart_state._save_drafts(drafts)

    def collect():
        raise error("The settings could not be collected")

    screen = SimpleNamespace(app_key="regression", _settings_model=SimpleNamespace(collect=collect))
    timer = QTimer()
    timer.start(30000)
    window = SimpleNamespace(
        _settings_autosave_timer=timer, _last_autosave="previous checkpoint",
        _collect_screen=MainWindow._collect_screen,
        _open_app_screens=lambda: {"regression": screen},
    )
    assert MainWindow._autosave_settings(window) is False
    assert timer.isActive()
    assert window._last_autosave == "previous checkpoint"
    assert restart_state._take_drafts() == drafts
    assert MainWindow._collect_screen(screen) == {}

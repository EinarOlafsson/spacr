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

"""Real Qt INI write errors remain retryable, including safe-mode writes."""
import json

import pytest
from PySide6.QtCore import QSettings

from spacr import restart_state
from spacr.qt.preferences import _DefaultsForReadingRealForWriting
from tests.qt.test_session_restore_and_settings_autosave import _own_state_dir, _window


def _writer(real, safe_mode):
    return _DefaultsForReadingRealForWriting(real) if safe_mode else real


@pytest.mark.parametrize("safe_mode", [False, True])
@pytest.mark.parametrize("operation", ["session", "drafts", "clear"])
def test_native_ini_errors_are_reported_by_every_draft_writer(
        tmp_path, monkeypatch, safe_mode, operation):
    blocked = tmp_path / "blocked"
    blocked.write_text("a file cannot hold an INI child")
    ini = blocked / "settings.ini"
    real = QSettings(str(ini), QSettings.IniFormat)
    real.setValue("sentinel", "keep")
    monkeypatch.setattr(restart_state, "_store", lambda: _writer(real, safe_mode))
    if operation == "session":
        result = restart_state._save_session("regression", {"fdr_alpha": 0.08})
    elif operation == "drafts":
        result = restart_state._save_drafts({"regression": {"fdr_alpha": 0.08}})
    else:
        result = restart_state._clear_drafts()
    assert real.status() == QSettings.AccessError
    assert not ini.exists()
    assert result is False


@pytest.mark.parametrize("safe_mode", [False, True])
def test_autosave_recovers_when_the_same_native_ini_path_becomes_writable(
        qtbot, tmp_path, monkeypatch, safe_mode):
    window = _window(qtbot, initial_app="regression")
    window._screens["regression"].apply_settings_dict({"fdr_alpha": 0.08})
    checkpoint = window._last_autosave
    blocked = tmp_path / "blocked"
    blocked.write_text("a file cannot hold an INI child")
    ini = blocked / "settings.ini"
    stores = []

    def store():
        real = QSettings(str(ini), QSettings.IniFormat)
        stores.append(real)
        return _writer(real, safe_mode)

    monkeypatch.setattr(restart_state, "_store", store)
    assert window._autosave_settings() is False
    assert all(real.status() == QSettings.AccessError for real in stores)
    assert window._last_autosave == checkpoint and not ini.exists()
    blocked.unlink()
    blocked.mkdir()
    assert window._autosave_settings() is True
    assert ini.is_file()
    durable = QSettings(str(ini), QSettings.IniFormat)
    durable.sync()
    session = json.loads(durable.value(restart_state._SESSION_KEY))
    drafts = json.loads(durable.value(restart_state._DRAFTS_KEY))
    assert session["settings"]["fdr_alpha"] == 0.08
    assert drafts["modules"]["regression"]["fdr_alpha"] == 0.08
    assert window._autosave_settings() is False

"""The resource sampler has its own persisted, three-state preference."""
from __future__ import annotations

import sys

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QComboBox, QDialogButtonBox


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    from spacr.qt import preferences

    store = QSettings(str(tmp_path / "preferences.ini"), QSettings.IniFormat)
    monkeypatch.setattr(preferences, "_settings", lambda: store)
    return preferences, store


def test_performance_logging_defaults_to_summary_and_rejects_unknown_modes(
        isolated_store):
    preferences, store = isolated_store

    assert preferences.get_performance_logging() == "summary"
    store.setValue(preferences._KEY_PERFORMANCE_LOG, "LOUD")
    assert preferences.get_performance_logging() == "summary"

    with pytest.raises(ValueError, match="Unknown performance-logging level"):
        preferences.set_performance_logging("verbose")


def test_the_dialog_saves_performance_logging_without_enabling_the_profiler(
        isolated_store, qtbot):
    preferences, _store = isolated_store
    preferences.set_verbose_logging(False)
    profile_before = sys.getprofile()

    dialog = preferences.PreferencesDialog()
    qtbot.addWidget(dialog)
    combo = dialog.findChild(QComboBox, "PerformanceLogging")

    assert combo is not None
    assert [combo.itemData(index) for index in range(combo.count())] == [
        "off", "summary", "detailed"]
    assert combo.currentData() == "summary"

    combo.setCurrentIndex(combo.findData("detailed"))
    buttons = dialog.findChild(QDialogButtonBox)
    buttons.button(QDialogButtonBox.Save).click()

    assert preferences.get_performance_logging() == "detailed"
    assert preferences.get_verbose_logging() is False
    assert sys.getprofile() is profile_before


def test_off_mode_starts_no_sampler_thread(isolated_store, tmp_path):
    preferences, _store = isolated_store
    from spacr.resource_log import ResourceSampler

    preferences.set_performance_logging("off")
    sampler = ResourceSampler(
        tmp_path / "resources.jsonl",
        level=preferences.get_performance_logging(),
    )

    assert sampler.start() is False
    assert sampler.is_running() is False
    assert not (tmp_path / "resources.jsonl").exists()

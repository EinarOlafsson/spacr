"""Timepose v1 in Timelapse is alpha (item 711): hidden until Show alpha features is on."""
from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings                              # noqa: E402


@pytest.fixture
def prefs(tmp_path, monkeypatch):
    from spacr.qt import preferences

    path = tmp_path / "alpha.ini"
    monkeypatch.setattr(preferences, "_settings",
                        lambda: QSettings(str(path), QSettings.IniFormat))
    return preferences


def test_the_choice_model_and_settings_are_gated(prefs):
    assert not prefs._is_alpha_visible("choices", "timelapse_mode", "timepose_v1")
    assert not prefs._is_alpha_visible("models", "timepose_v1")
    assert not prefs._is_alpha_visible("settings", "timepose_invert")
    assert prefs._is_alpha_visible("choices", "timelapse_mode", "trackastra")
    prefs._set_show_alpha_features(True)
    assert prefs._is_alpha_visible("choices", "timelapse_mode", "timepose_v1")
    assert prefs._is_alpha_visible("models", "timepose_v1")


def test_the_preview_note_follows_the_switch(qtbot, prefs):
    from spacr.qt.preferences import _apply_alpha_widgets
    from spacr.qt.widgets import timelapse_preview as TP
    from PySide6.QtWidgets import QLabel

    panel = TP.TimelapsePreviewPanel(threaded=False)
    qtbot.addWidget(panel)
    note = panel.findChild(QLabel, "TimelapseTimeposeNote")
    assert note is not None and note.isHidden()
    prefs._set_show_alpha_features(True)
    _apply_alpha_widgets(panel)
    assert not note.isHidden()
    assert "timepose_v1" in note.text()

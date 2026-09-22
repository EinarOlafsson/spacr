"""Zoom survives closing windows and failed theme refreshes without leaking fonts."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import shiboken6
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QMainWindow, QWidget

from spacr.qt import live_zoom as zoom
from spacr.qt import preferences


@pytest.fixture
def gesture(qapp, monkeypatch):
    monkeypatch.setattr(preferences, 'set_font_scale', Mock())
    monkeypatch.setattr(preferences, 'get_font_scale', lambda: 1.0)
    monkeypatch.setattr(preferences, 'apply_preferences_to_app', Mock())
    active = zoom.LiveZoomFilter()
    yield active
    active.settle()


def test_point_sized_font_scales_from_its_baseline_and_returns_to_it(qapp):
    base = QFont('Sans Serif')
    base.setPointSizeF(12.0)
    grown = zoom._scaled_font(base, 1.25, base)
    assert grown.pointSizeF() == 15.0
    assert base.pointSizeF() == 12.0
    assert zoom._scaled_font(base, 1.25, grown) is None
    restored = zoom._scaled_font(base, 1.0, grown)
    assert restored.pointSizeF() == 12.0
    assert zoom._scaled_font(base, 0.001, grown).pointSizeF() == zoom._MIN_PT


def test_validity_fallback_handles_both_live_and_deleted_widgets(qtbot, monkeypatch):
    owner = QWidget()
    qtbot.addWidget(owner)
    widget = QWidget(owner)
    monkeypatch.setattr(shiboken6, 'isValid', Mock(side_effect=ImportError('unavailable')))
    assert zoom._alive(widget)
    shiboken6.delete(widget)
    assert not zoom._alive(widget)


@pytest.mark.parametrize('problem', [RuntimeError('closed'), AttributeError('no status bar')])
def test_unavailable_status_bar_does_not_abort_a_gesture(gesture, qtbot, problem):
    window = QMainWindow()
    qtbot.addWidget(window)
    window.statusBar = Mock(side_effect=problem)
    gesture._window = window
    gesture._announce()
    window.statusBar.assert_called_once()


def test_status_reports_round_the_percentage_and_accept_windows_without_a_bar(gesture, qtbot):
    window = QMainWindow()
    qtbot.addWidget(window)
    gesture._window = window
    gesture._live_scale = 0.7999999999999998
    gesture._announce()
    assert window.statusBar().currentMessage() == 'Text size 80 %'
    dialog = QWidget(window)
    gesture._window = dialog
    gesture._announce()
    shiboken6.delete(dialog)
    gesture._announce()


def test_snapshot_handles_a_window_disappearing_during_lookup(gesture):
    watched = SimpleNamespace(window=Mock(side_effect=RuntimeError('closed')))
    gesture._begin(watched)
    assert gesture._window is None
    gesture._begin(object())
    assert gesture._window is None


def test_absent_application_leaves_snapshot_and_installation_empty(gesture, monkeypatch):
    monkeypatch.setattr(zoom, 'QApplication', SimpleNamespace(instance=lambda: None))
    gesture._begin(object())
    assert gesture._baseline == []
    assert zoom.install_live_zoom() is None


@pytest.mark.parametrize('window_fails', [False, True])
def test_failed_application_restyle_still_restores_font_and_refreshes_window(
        gesture, qtbot, monkeypatch, caplog, window_fails):
    label = QWidget()
    qtbot.addWidget(label)
    font = QFont('Monospace')
    font.setPixelSize(12)
    label.setFont(font)
    window = QWidget()
    qtbot.addWidget(window)
    window.refresh_theme = Mock(side_effect=RuntimeError('window closing') if window_fails else None)
    monkeypatch.setattr(preferences, 'apply_preferences_to_app', Mock(side_effect=RuntimeError('theme failed')))
    gesture._base_scale = 1.0
    gesture._live_scale = 1.5
    gesture._baseline = [(label, QFont(label.font()), True)]
    gesture._window = window
    gesture._apply()
    assert label.font().pixelSize() == 18
    gesture.settle()
    assert label.font().pixelSize() == 12
    assert gesture._baseline == [] and gesture._touched == set()
    assert gesture._window is None
    preferences.set_font_scale.assert_called_once_with(1.5)
    window.refresh_theme.assert_called_once()
    assert 'could not apply the font scale' in caplog.text


def test_invalid_zero_baseline_does_not_mutate_widgets(gesture, qtbot):
    widget = QWidget()
    qtbot.addWidget(widget)
    before = QFont(widget.font())
    gesture._baseline = [(widget, before, False)]
    gesture._base_scale = 0
    gesture._apply()
    assert widget.font() == before and not gesture._touched
    gesture._baseline = []

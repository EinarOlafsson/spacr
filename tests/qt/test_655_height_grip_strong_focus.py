"""Shared panel resize grips honor the existing stronger-focus preference."""

import pytest
from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication, QPushButton, QVBoxLayout, QWidget

from spacr.qt import preferences, theme
from spacr.qt.widgets.height_grip import HeightGrip


@pytest.fixture
def grip(qtbot, monkeypatch, tmp_path, qapp, activate_focus_window):
    store = QSettings(str(tmp_path / "grip-focus.ini"), QSettings.IniFormat)
    monkeypatch.setattr(preferences, "_settings", lambda: store)
    monkeypatch.setattr(preferences, "_SAFE_MODE", False)
    preferences.set_strong_focus(False)
    palette = qapp.palette()
    theme._forget_painting_theme()
    host = QWidget()
    qtbot.addWidget(host)
    layout = QVBoxLayout(host)
    before = QPushButton("Before", host)
    target = QWidget(host)
    handle = HeightGrip(target, 100, 400, host)
    after = QPushButton("After", host)
    for widget in (before, target, handle, after):
        layout.addWidget(widget)
    handle.resize_target(200)
    host.resize(400, 340)
    host.show()
    activate_focus_window(host)
    QApplication.processEvents()
    yield host, handle, before, after
    qapp.setPalette(palette)
    theme._forget_painting_theme()


@pytest.mark.parametrize("scheme", ["dark", "light", "high_contrast"])
def test_real_focus_tracks_stronger_outlines_without_resizing(grip, qtbot, scheme):
    host, handle, before, after = grip
    preferences.set_theme_choice(scheme)
    theme.apply_qpalette(QApplication.instance(), scheme)
    host.setStyleSheet(theme.stylesheet())
    before.setFocus(Qt.TabFocusReason)
    QApplication.processEvents()
    plain = handle.grab().toImage()
    changed = []
    handle.height_changed.connect(changed.append)
    qtbot.keyClick(before, Qt.Key_Tab)
    assert handle.hasFocus()
    ordinary = handle.grab().toImage()
    assert ordinary != plain
    geometry, target_height = handle.geometry(), handle.target_height()
    preferences.set_strong_focus(True)
    host.setStyleSheet(theme.stylesheet())
    QApplication.processEvents()
    strong = handle.grab().toImage()
    assert strong != ordinary
    colour = QColor(theme.focus_ring_colour(scheme))
    assert [strong.pixelColor(strong.width() // 2, y)
            for y in range(theme.FOCUS_RING_PX)] == [colour] * theme.FOCUS_RING_PX
    assert handle.geometry() == geometry
    assert handle.target_height() == target_height
    assert changed == []
    preferences.set_strong_focus(False)
    host.setStyleSheet(theme.stylesheet())
    QApplication.processEvents()
    assert handle.hasFocus() and handle.grab().toImage() == ordinary
    qtbot.keyClick(handle, Qt.Key_Tab)
    assert after.hasFocus()
    assert handle.grab().toImage() == plain


def test_focused_paint_uses_cached_preference(grip, monkeypatch):
    host, handle, _, _ = grip
    preferences.set_strong_focus(True)
    host.setStyleSheet(theme.stylesheet())
    handle.setFocus(Qt.TabFocusReason)
    QApplication.processEvents()
    from spacr.qt.widgets import figure_grid_view

    def refuse():
        raise AssertionError("Paint must not reread stored focus preferences")

    monkeypatch.setattr(figure_grid_view, "_strong_focus_style", refuse)
    assert not handle.grab().isNull()

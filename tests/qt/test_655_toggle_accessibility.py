"""Keyboard focus and reduced motion reach the custom-painted switches."""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QAbstractAnimation, Qt
from PySide6.QtGui import QColor
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QPushButton, QVBoxLayout, QWidget

from spacr.qt import preferences, theme
from spacr.qt.widgets.toggle import Toggle


@pytest.fixture
def switches(qtbot, monkeypatch, qt_theme_applied, activate_focus_window):
    monkeypatch.setattr(preferences, "get_reduce_motion", lambda: False)
    host = QWidget()
    qtbot.addWidget(host)
    layout = QVBoxLayout(host)
    before = QPushButton("Before")
    switch = Toggle("Enable preview")
    after = QPushButton("After")
    for widget in (before, switch, after):
        layout.addWidget(widget)
    switch.setFixedSize(180, 36)
    host.show()
    activate_focus_window(host)
    before.setFocus(Qt.TabFocusReason)
    qtbot.waitUntil(before.hasFocus)
    yield before, switch, after


@pytest.mark.parametrize("caption", ["", "Enable preview"])
@pytest.mark.parametrize("lent", [False, True])
def test_keyboard_focus_remains_visible_when_the_field_paints_the_switch(
        switches, qtbot, caption, lent):
    before, switch, after = switches
    switch.setText(caption)
    if lent:
        assert switch._spacr_static_picture() is not None
        switch._spacr_lend_static(switch._spacr_take_back_static)
    unfocused = switch.grab().toImage()
    QTest.keyClick(before, Qt.Key_Tab)
    qtbot.waitUntil(switch.hasFocus)
    focused = switch.grab().toImage()
    assert focused != unfocused, "the focused switch has no visible focus mark"
    QTest.keyClick(switch, Qt.Key_Tab)
    qtbot.waitUntil(after.hasFocus)
    assert switch.grab().toImage() == unfocused


@pytest.mark.parametrize("theme_name", ["dark", "light", "high_contrast"])
def test_stronger_focus_keeps_the_theme_three_pixel_border(switches, theme_name):
    before, switch, _after = switches
    switch.setStyleSheet(theme.focus_ring_qss(theme_name))
    QTest.keyClick(before, Qt.Key_Tab)
    assert switch.hasFocus()
    image = switch.grab().toImage()
    expected = QColor(theme.focus_ring_colour(theme_name))
    x = image.width() // 2
    assert [image.pixelColor(x, y) for y in range(theme.FOCUS_RING_PX)] == [
        expected] * theme.FOCUS_RING_PX


@pytest.mark.parametrize("checked", [False, True])
def test_reduce_motion_sets_the_final_knob_position_without_animation(
        switches, monkeypatch, checked):
    _before, switch, _after = switches
    monkeypatch.setattr(preferences, "get_reduce_motion", lambda: True)
    switch.setChecked(not checked)
    switch.setChecked(checked)
    assert switch._anim.state() == QAbstractAnimation.Stopped
    expected = switch._maximum_knob_x() if checked else switch._minimum_knob_x()
    assert switch._knob_pos == float(expected)


def test_keyboard_space_changes_the_switch_once_and_respects_reduce_motion(
        switches, monkeypatch):
    before, switch, _after = switches
    monkeypatch.setattr(preferences, "get_reduce_motion", lambda: True)
    changes = []
    switch.toggled.connect(changes.append)
    QTest.keyClick(before, Qt.Key_Tab)
    QTest.keyClick(switch, Qt.Key_Space)
    assert changes == [True]
    assert switch.isChecked()
    assert switch._anim.state() == QAbstractAnimation.Stopped

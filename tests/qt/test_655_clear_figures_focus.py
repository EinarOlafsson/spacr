"""Keyboard focus is visible on the figure queue's destructive text control."""

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QAccessible, QColor, QPalette
from PySide6.QtWidgets import QPushButton, QVBoxLayout, QWidget

from spacr.qt.widgets.figure_queue import _ClearFiguresLabel


@pytest.fixture
def clear_control(qtbot, activate_focus_window):
    host = QWidget()
    layout = QVBoxLayout(host)
    before = QPushButton("Before")
    label = _ClearFiguresLabel()
    after = QPushButton("After")
    for widget in (before, label, after):
        layout.addWidget(widget)
    host.resize(320, 160)
    qtbot.addWidget(host)
    host.show()
    activate_focus_window(host)
    qtbot.wait(40)
    before.setFocus(Qt.TabFocusReason)
    qtbot.waitUntil(before.hasFocus)
    yield before, label, after


def test_existing_tab_route_and_accessible_text_remain(clear_control, qtbot):
    before, label, after = clear_control
    qtbot.keyClick(before, Qt.Key_Tab)
    assert label.hasFocus()
    interface = QAccessible.queryAccessibleInterface(label)
    assert interface.text(QAccessible.Name) == label.text()
    qtbot.keyClick(label, Qt.Key_Tab)
    assert after.hasFocus()


@pytest.mark.parametrize("key", [Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space])
def test_existing_keys_emit_once_and_keep_the_click_flash(clear_control, qtbot, key):
    before, label, _ = clear_control
    fired = []
    label.clicked.connect(lambda: fired.append(True))
    qtbot.keyClick(before, Qt.Key_Tab)
    qtbot.keyClick(label, key)
    assert fired == [True]
    assert label._flash.active


@pytest.mark.parametrize("highlight", ["#0088dd", "#ee11aa"])
def test_rendered_focus_is_visible_uses_palette_and_leaves_no_trace(
        clear_control, qtbot, highlight):
    before, label, after = clear_control
    palette = label.palette()
    palette.setColor(QPalette.Highlight, QColor(highlight))
    label.setPalette(palette)
    resting_style = label.styleSheet()
    resting = label.grab().toImage()
    qtbot.keyClick(before, Qt.Key_Tab)
    assert label.hasFocus()
    focused = label.grab().toImage()
    assert focused != resting
    assert any(focused.pixelColor(x, 1) == QColor(highlight)
               for x in range(2, focused.width() - 2))
    assert label.styleSheet() == resting_style
    assert not label._flash.active
    assert focused.copy(4, 4, focused.width() - 8, focused.height() - 8) == (
        resting.copy(4, 4, resting.width() - 8, resting.height() - 8))
    qtbot.keyClick(label, Qt.Key_Tab)
    assert after.hasFocus()
    assert label.grab().toImage() == resting


def test_flash_preserves_focus_marker(clear_control, qtbot, request):
    before, label, _ = clear_control
    palette = label.palette()
    palette.setColor(QPalette.Highlight, QColor("#ee11aa"))
    label.setPalette(palette)
    qtbot.keyClick(before, Qt.Key_Tab)
    focused = label.grab().toImage()
    before_colour = label.palette().color(QPalette.Highlight)
    qtbot.keyClick(label, Qt.Key_Space)
    assert label._flash.active
    assert label.hasFocus()
    flashing = label.grab().toImage()
    current_colour = label.palette().color(QPalette.Highlight)
    request.node.user_properties.extend([
        ("highlight_before_flash", before_colour.name()),
        ("highlight_after_restyle", current_colour.name()),
    ])
    assert focused != flashing
    assert any(flashing.pixelColor(x, 1) == current_colour
               for x in range(2, flashing.width() - 2))

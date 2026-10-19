"""The fold button's verdict dot and the AI label's focus ring are native layers.

Both were drawn by a Python ``paintEvent`` that every frame of the window's
animated field re-entered, with or without anything to draw. They are now
drawn once into a pixmap shown by a plain label child. The pixels must be
the ones the ``paintEvent`` drew over the widget, and a repaint must not
enter Python for them. Counted and compared, not timed.
"""
from __future__ import annotations

import sys

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QRectF, Qt  # noqa: E402
from PySide6.QtGui import QColor, QImage, QPainter, QPalette, QPen  # noqa: E402

from spacr.qt.widgets.ai_toggle_label import AiToggleLabel  # noqa: E402
from spacr.qt.widgets.fold_strip import FoldButton  # noqa: E402

pytestmark = pytest.mark.qt


def _grab(widget):
    return widget.grab().toImage().convertToFormat(
        QImage.Format_ARGB32_Premultiplied)


def _python_entered_by(call, *modules) -> list:
    entered = []
    files = {module.__file__ for module in modules}

    def profile(frame, event, arg):
        if event == "call" and frame.f_code.co_filename in files:
            entered.append(frame.f_code.co_name)

    sys.setprofile(profile)
    try:
        call()
    finally:
        sys.setprofile(None)
    return entered


@pytest.mark.parametrize("level", ["pass", "check", "fail"])
def test_the_verdict_dot_is_the_painted_dot(qtbot, level):
    from spacr.qt.widgets import fold_strip

    button = FoldButton("regression_diagnostics")
    qtbot.addWidget(button)
    button.resize(44, 40)
    button.show()
    qtbot.waitExposed(button)
    button.set_verdict(level, "detail")
    layer = button._verdict_layer
    assert layer.isVisible()

    layer.hide()
    expected = _grab(button)
    painter = QPainter(expected)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    size = max(6.0, button.height() * 0.22)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(button._verdict_ink(level)))
    painter.drawEllipse(QRectF(button.width() - size - 3.0, 3.0, size, size))
    painter.end()
    layer.show()
    assert _grab(button) == expected

    assert _python_entered_by(button.repaint, fold_strip) == []
    button.set_verdict("")
    assert not layer.isVisible()


def test_the_dot_follows_a_resize(qtbot):
    button = FoldButton("regression_diagnostics")
    qtbot.addWidget(button)
    button.resize(44, 40)
    button.show()
    qtbot.waitExposed(button)
    button.set_verdict("fail")
    button.resize(60, 56)
    qtbot.waitUntil(lambda: button._verdict_layer.size() == button.size(),
                    timeout=2000)
    assert button._verdict_layer.geometry() == button.rect()
    assert button._verdict_layer.isVisible()


def test_the_focus_ring_is_the_painted_ring(qtbot):
    from spacr.qt.widgets import ai_toggle_label

    label = AiToggleLabel()
    qtbot.addWidget(label)
    label.resize(60, 28)
    label.show()
    qtbot.waitExposed(label)
    label.setFocus()
    qtbot.waitUntil(label.hasFocus, timeout=2000)
    layer = label._focus_layer
    assert layer.isVisible()

    layer.hide()
    expected = _grab(label)
    painter = QPainter(expected)
    width, colour = label._strong_focus_ring or (
        2, label.palette().color(QPalette.Highlight))
    painter.setPen(QPen(QColor(colour), width))
    painter.setBrush(Qt.NoBrush)
    painter.drawRect(label.rect().adjusted(1, 1, -1, -1))
    painter.end()
    layer.show()
    assert _grab(label) == expected

    assert _python_entered_by(label.repaint, ai_toggle_label) == []
    label.clearFocus()
    qtbot.waitUntil(lambda: not layer.isVisible(), timeout=2000)

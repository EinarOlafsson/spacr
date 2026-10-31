"""The composer's Protocols control is a dot: white off, accent blue on.

The gray circle behind it appears only while the pointer is over it or it is
pressed, and the dot stays centred and inside that circle at every font
scale.
"""
from __future__ import annotations

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QCursor
from PySide6.QtWidgets import QWidget

from spacr.qt import preferences
from spacr.qt.theme import palette_for
from spacr.qt.widgets.home import _NavigatorProtocolsButton

pytestmark = pytest.mark.qt


@pytest.fixture
def button(qtbot, qt_theme_applied):
    host = QWidget()
    qtbot.addWidget(host)
    host.resize(240, 120)
    widget = _NavigatorProtocolsButton(host)
    widget.move(180, 60)
    QCursor.setPos(QPoint(0, 0))
    host.show()
    qtbot.waitExposed(host)
    qtbot.mouseMove(host, QPoint(10, 10))
    qtbot.waitUntil(lambda: not widget.underMouse(), timeout=2000)
    yield widget
    host.close()


def _pixel(widget, point):
    image = widget.grab().toImage()
    ratio = image.devicePixelRatio()
    return image.pixelColor(int(point.x() * ratio), int(point.y() * ratio))


def _ring_point(widget):
    centre = widget.rect().center()
    dot = widget._dot_rect()
    circle = min(widget.width(), widget.height()) - 6
    offset = int((dot.width() / 2 + circle / 2) / 2)
    return centre + QPoint(offset, 0)


def test_the_inactive_dot_is_white_with_no_circle(button):
    assert not button.isChecked()
    centre = _pixel(button, button.rect().center())
    assert (centre.red(), centre.green(), centre.blue()) == (255, 255, 255)
    assert centre.alpha() == 255
    assert _pixel(button, _ring_point(button)).alpha() == 0
    assert button._hover_circle_color() is None


def test_the_active_dot_is_the_accent_blue(button):
    button.setChecked(True)
    accent = QColor(palette_for('dark')['accent'])
    assert button._dot_color() == accent
    centre = _pixel(button, button.rect().center())
    assert (centre.red(), centre.green(), centre.blue()) == (
        accent.red(), accent.green(), accent.blue())
    assert _pixel(button, _ring_point(button)).alpha() == 0


def test_the_gray_circle_shows_on_press_and_hover_only(button, qtbot):
    button.setDown(True)
    pressed = _pixel(button, _ring_point(button))
    assert pressed.alpha() == 255 and pressed.name() == '#303030'
    button.setDown(False)
    assert _pixel(button, _ring_point(button)).alpha() == 0
    qtbot.mouseMove(button, button.rect().center())
    qtbot.waitUntil(button.underMouse, timeout=2000)
    hovered = _pixel(button, _ring_point(button))
    assert hovered.alpha() == 255 and hovered.name() == '#363636'


@pytest.mark.parametrize('scale', [1.0, 1.5])
def test_the_dot_stays_centred_inside_the_circle(button, scale, monkeypatch):
    previous = preferences.get_font_scale()
    preferences.set_font_scale(scale)
    try:
        dot = button._dot_rect()
        circle = min(button.width(), button.height()) - 6
        assert dot.center() == button.rect().toRectF().center()
        assert 0 < dot.width() == dot.height() <= circle / 2
    finally:
        preferences.set_font_scale(previous)

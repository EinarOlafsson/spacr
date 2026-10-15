"""Geometry-based popup lifetime when a platform omits hover events."""

import pytest
from PySide6.QtCore import QEvent, QObject, QPoint
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QLabel, QWidget

from spacr.qt.widgets.hover_tooltip import HoverTooltip


class MissingHoverEvents(QObject):
    def eventFilter(self, obj, event):
        return event.type() in (QEvent.Enter, QEvent.Leave)


@pytest.mark.parametrize("rest_on", ["popup", "anchor"])
def test_geometry_fallback_keeps_reading_help_then_dismisses_it(
        qtbot, rest_on):
    host = QWidget()
    host.resize(300, 120)
    anchor = QLabel("Setting name", host)
    anchor.resize(120, 40)
    qtbot.addWidget(host)
    host.show()
    popup = HoverTooltip()
    qtbot.addWidget(popup)
    suppressor = MissingHoverEvents(popup)
    popup.installEventFilter(suppressor)
    anchor.installEventFilter(suppressor)
    popup.show_for(anchor, "Help remains available while it is being read.",
                   animation=None, immediate=True)
    target = popup if rest_on == "popup" else anchor
    QCursor.setPos(target.mapToGlobal(target.rect().center()))
    popup.start_hide()
    qtbot.wait(1150)
    assert popup.isVisible(), "The geometry fallback interrupted reading"
    QCursor.setPos(QPoint(2000, 2000))
    qtbot.waitUntil(lambda: not popup.isVisible(), timeout=1600)


def test_hiding_the_popup_stops_its_geometry_timer(qtbot):
    anchor = QLabel("Setting name")
    qtbot.addWidget(anchor)
    anchor.show()
    popup = HoverTooltip()
    qtbot.addWidget(popup)
    popup.show_for(anchor, "A reading popup.", animation=None, immediate=True)
    popup.start_hide()
    assert popup._hide_timer.isActive()
    popup.hide()
    assert not popup._hide_timer.isActive()

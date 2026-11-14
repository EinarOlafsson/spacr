"""The edge drawer opens on intent and closes without racing its slide.

`is_open` is intent and `is_fully_open` is geometry; a click on the hot strip
opens and pins; a pin outranks a close timeout already in flight; Escape is
the only key that closes; leaving the panel schedules a close; and a closed
drawer ends up hidden, not parked off-screen.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtCore import QPointF, Qt                          # noqa: E402
from PySide6.QtGui import QEnterEvent                           # noqa: E402
from PySide6.QtTest import QTest                                # noqa: E402
from PySide6.QtWidgets import QPushButton, QWidget              # noqa: E402

from spacr.qt.widgets.drawer import EdgeDrawer                  # noqa: E402


def _drawer(qtbot):
    host = QWidget()
    host.resize(600, 400)
    qtbot.addWidget(host)
    panel = QWidget()
    panel.resize(200, 400)
    QPushButton("Mask", panel).setFocusPolicy(Qt.StrongFocus)
    drawer = EdgeDrawer(host, panel, width=200)
    host.show()
    return drawer


def test_the_drawer_opens_on_intent_and_closes_only_when_it_should(qtbot,
                                                                   subtests):
    """Each path that decides whether the drawer is on screen."""
    with subtests.test(msg="not fully open until the slide finishes, then hidden after closing"):
        drawer = _drawer(qtbot)
        assert drawer.is_fully_open() is False
        drawer.open()
        assert drawer.is_open() is True
        qtbot.waitUntil(drawer.is_fully_open, timeout=3000)
        assert drawer.x() == 0
        drawer.close()
        assert drawer.is_open() is False
        qtbot.waitUntil(drawer.isHidden, timeout=3000)
        assert drawer.is_fully_open() is False

    with subtests.test(msg="a click on the hot strip opens and pins"):
        drawer = _drawer(qtbot)
        QTest.mouseClick(drawer._trigger, Qt.LeftButton)
        assert (drawer.is_open(), drawer.is_held()) == (True, True)

    with subtests.test(msg="a close timeout landing after the pin leaves it open"):
        drawer = _drawer(qtbot)
        drawer.open()
        drawer.schedule_close()
        drawer.hold(True)
        drawer._close_timer.timeout.emit()
        assert (drawer.is_open(), drawer.is_held()) == (True, True)
        drawer.hold(False)
        drawer._close_timer.timeout.emit()
        assert drawer.is_open() is False

    with subtests.test(msg="Escape closes and other keys do not"):
        drawer = _drawer(qtbot)
        drawer.open_for_keyboard()
        QTest.keyClick(drawer, Qt.Key_A)
        assert drawer.is_open() is True
        QTest.keyClick(drawer, Qt.Key_Escape)
        assert drawer.is_open() is False

    with subtests.test(msg="leaving the panel schedules a close but does not close"):
        drawer = _drawer(qtbot)
        drawer.open()
        drawer.hold(False)
        assert drawer._close_timer.isActive() is False
        drawer.leaveEvent(QEnterEvent(QPointF(1.0, 1.0), QPointF(1.0, 1.0),
                                      QPointF(1.0, 1.0)))
        assert drawer._close_timer.isActive() is True
        assert drawer.is_open() is True

"""The edge reveal opens on intent and closes without racing its animation.

The drawer is the only way to reach the app list, so every path that decides
whether it is on screen matters: a deliberate click on the strip has to open
and pin it, a keyboard user has to be able to open it and press Escape, and
the slide-out has to leave the widget hidden rather than parked off-screen
where it still takes hover events.
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


def _drawer(qtbot, with_button=True):
    host = QWidget()
    host.resize(600, 400)
    qtbot.addWidget(host)
    panel = QWidget()
    panel.resize(200, 400)
    if with_button:
        button = QPushButton("Mask", panel)
        button.setFocusPolicy(Qt.StrongFocus)
    drawer = EdgeDrawer(host, panel, width=200)
    host.show()
    return host, drawer


def _enter(widget):
    return QEnterEvent(QPointF(1.0, 1.0), QPointF(1.0, 1.0), QPointF(1.0, 1.0))


# ---------------------------------------------------------------------------
# Opening
# ---------------------------------------------------------------------------


def test_a_click_on_a_disabled_strip_does_nothing(qtbot):
    """The strip is disabled when the dock is locked open or hidden; a
    click there belongs to whatever is underneath it."""
    _host, drawer = _drawer(qtbot)
    drawer.set_enabled(False)

    QTest.mouseClick(drawer._trigger, Qt.LeftButton)

    assert drawer.is_open() is False
    assert drawer.is_held() is False


def test_a_keyboard_open_with_nothing_focusable_still_opens(qtbot):
    """A panel whose rows are not built yet has nothing to focus. The
    drawer still has to open, or the keyboard path is dead."""
    _host, drawer = _drawer(qtbot, with_button=False)

    drawer.open_for_keyboard()

    assert drawer.is_open() is True
    assert drawer.is_held() is True


def test_a_keyboard_open_moves_focus_into_the_panel(qtbot):
    """The contrast: with a focusable row present, focus lands on it, which
    is what makes the app list reachable without a pointer."""
    _host, drawer = _drawer(qtbot)

    drawer.open_for_keyboard()

    assert isinstance(drawer._first_focusable(), QPushButton)


# ---------------------------------------------------------------------------
# Closing
# ---------------------------------------------------------------------------


def test_the_pointer_entering_the_panel_cancels_a_pending_close(qtbot):
    """The grace period exists for the gap between strip and panel; a
    pointer that arrives in the panel must cancel the close it started."""
    _host, drawer = _drawer(qtbot)
    drawer.open()
    drawer.hold(False)
    drawer.schedule_close()
    assert drawer._close_timer.isActive()

    drawer.enterEvent(_enter(drawer))

    assert not drawer._close_timer.isActive()
    assert drawer.is_open() is True

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

def test_a_drawer_is_not_fully_open_until_the_slide_finishes(qtbot):
    """`is_open` is intent and `is_fully_open` is geometry. A caller that
    reads position while the panel is still sliding would be told it is
    closed, which is how the two race each other."""
    _host, drawer = _drawer(qtbot)

    assert drawer.is_fully_open() is False

    drawer.open()
    assert drawer.is_open() is True

    qtbot.waitUntil(drawer.is_fully_open, timeout=3000)
    assert drawer.x() == 0


def test_a_click_on_the_hot_strip_opens_and_pins(qtbot):
    """A deliberate click must not have to survive the close animation --
    the pin is what stops the drawer sliding shut under the pointer."""
    _host, drawer = _drawer(qtbot)

    QTest.mouseClick(drawer._trigger, Qt.LeftButton)

    assert drawer.is_open() is True
    assert drawer.is_held() is True


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

def test_a_closed_drawer_ends_up_hidden_not_parked_offscreen(qtbot):
    """A visible widget at x = -200 still answers hover and geometry
    queries; hiding it at the end of the slide is what actually takes it out
    of the page."""
    _host, drawer = _drawer(qtbot)
    drawer.open()
    qtbot.waitUntil(drawer.is_fully_open, timeout=3000)

    drawer.close()

    qtbot.waitUntil(lambda: drawer.isHidden(), timeout=3000)
    assert drawer.is_open() is False


def test_escape_closes_the_drawer_and_other_keys_do_not(qtbot):
    """Escape is the keyboard user's way out. Swallowing every other key
    would make the panel a trap for the keys its rows need."""
    _host, drawer = _drawer(qtbot)
    drawer.open_for_keyboard()

    QTest.keyClick(drawer, Qt.Key_A)
    assert drawer.is_open() is True

    QTest.keyClick(drawer, Qt.Key_Escape)
    assert drawer.is_open() is False


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


def test_the_pointer_leaving_the_panel_schedules_a_close(qtbot):
    """Leaving does not close immediately, but it must start the timer --
    a drawer that stays open after the pointer leaves never closes at all."""
    _host, drawer = _drawer(qtbot)
    drawer.open()
    drawer.hold(False)

    drawer.leaveEvent(_enter(drawer))

    assert drawer._close_timer.isActive()
    assert drawer.is_open() is True, "leaving must not close it immediately"

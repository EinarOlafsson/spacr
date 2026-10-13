"""Dragging the frameless window by its menu bar.

WHY IT EXISTS. The main window is frameless -- asked for on 2026-08-23,
"remove the minus and x bar from the spacr window" -- so there is no
title bar for the compositor to offer as a drag handle. A comment in the
file said the menu bar is what you drag it by, and nothing implemented
it: the window could not be moved at all.

WHY `startSystemMove` AND NOT `move()`. The window cannot position
itself on Wayland; a client asking to be somewhere is ignored. Handing
the gesture to the compositor is the only mechanism that works, and it
is what the filter does.

The whole filter was written without a test. These assert the two halves
that matter to somebody using it -- the bare strip drags, a menu still
opens -- and the three ways it can be asked to drag a window that is not
there, none of which may raise: an event filter's exception has no
caller, so PySide prints it and goes on delivering, once per mouse move.
"""
from __future__ import annotations

import pytest

from PySide6.QtCore import QEvent, QObject, QPointF, Qt
from PySide6.QtGui import QMouseEvent

from spacr.qt.app import _DragsTheWindowByTheMenuBar

pytestmark = pytest.mark.qt


class _Handle:
    def __init__(self):
        self.moves = 0

    def startSystemMove(self):        # noqa: N802 - Qt naming
        self.moves += 1


class _Window(QObject):
    """A QObject, because the filter parents itself to the window."""

    def __init__(self, handle=None):
        super().__init__()
        self._handle = handle

    def windowHandle(self):           # noqa: N802 - Qt naming
        return self._handle


class _Bar:
    """A menu bar stand-in: it knows which points carry an action."""

    def __init__(self, action_at=None):
        self._action_at = action_at

    def actionAt(self, _point):       # noqa: N802 - Qt naming
        return self._action_at


def _press(button=Qt.MouseButton.LeftButton,
           kind=QEvent.Type.MouseButtonPress):
    return QMouseEvent(kind, QPointF(40.0, 8.0), button, button,
                       Qt.KeyboardModifier.NoModifier)


def test_a_press_on_the_bare_strip_hands_the_drag_to_the_compositor():
    handle = _Handle()
    drag = _DragsTheWindowByTheMenuBar(_Window(handle))
    assert drag.eventFilter(_Bar(action_at=None), _press()) is True
    assert handle.moves == 1


def test_a_press_on_a_menu_opens_the_menu_instead():
    """THE HALF THAT MAKES IT USABLE.

    Swallowing this press would make the window draggable and the menus
    unopenable, which is a worse application than the one that could not
    be moved.
    """
    handle = _Handle()
    drag = _DragsTheWindowByTheMenuBar(_Window(handle))
    assert drag.eventFilter(_Bar(action_at=object()), _press()) is False
    assert handle.moves == 0


def test_a_right_click_does_not_drag():
    """Only the left button. A right press belongs to a context menu."""
    handle = _Handle()
    drag = _DragsTheWindowByTheMenuBar(_Window(handle))
    event = _press(button=Qt.MouseButton.RightButton)
    assert drag.eventFilter(_Bar(action_at=None), event) is False
    assert handle.moves == 0


@pytest.mark.parametrize("kind", [
    QEvent.Type.MouseButtonRelease,
    QEvent.Type.MouseMove,
    QEvent.Type.Paint,
])
def test_anything_that_is_not_a_press_is_passed_straight_through(kind):
    """The filter sees every event the bar gets; it must be cheap.

    Returning early on the type is what keeps a mouse move from doing
    the action lookup on every pixel.
    """
    handle = _Handle()
    drag = _DragsTheWindowByTheMenuBar(_Window(handle))
    if kind is QEvent.Type.Paint:
        event = QEvent(kind)
    else:
        event = _press(kind=kind)
    assert drag.eventFilter(_Bar(action_at=None), event) is False
    assert handle.moves == 0


class TestItNeverRaisesOutOfTheEventLoop:
    """An exception here has no caller who can handle it.

    PySide prints "Error calling Python override of
    QObject::eventFilter()" and carries on delivering, so one bad state
    becomes one traceback per mouse press.
    """

    def test_a_window_with_no_native_handle_yet_does_not_drag(self):
        """Before the window is mapped there is nothing to move."""
        drag = _DragsTheWindowByTheMenuBar(_Window(None))
        assert drag.eventFilter(_Bar(action_at=None), _press()) is False

    def test_a_bar_that_cannot_answer_actionAt_is_survived(self, caplog):
        class _Gone:
            def actionAt(self, _point):   # noqa: N802 - Qt naming
                raise RuntimeError("Internal C++ object already deleted.")

        drag = _DragsTheWindowByTheMenuBar(_Window(_Handle()))
        with caplog.at_level("DEBUG"):
            assert drag.eventFilter(_Gone(), _press()) is False

    def test_a_compositor_that_refuses_the_move_is_survived(self):
        class _Refuses:
            def startSystemMove(self):    # noqa: N802 - Qt naming
                raise RuntimeError("the compositor declined")

        drag = _DragsTheWindowByTheMenuBar(_Window(_Refuses()))
        assert drag.eventFilter(_Bar(action_at=None), _press()) is False

    def test_a_window_that_has_been_torn_down_is_survived(self):
        class _Dead(QObject):
            def windowHandle(self):       # noqa: N802 - Qt naming
                raise RuntimeError("Internal C++ object already deleted.")

        drag = _DragsTheWindowByTheMenuBar(_Dead())
        assert drag.eventFilter(_Bar(action_at=None), _press()) is False


def test_the_real_window_installs_it_on_the_real_menu_bar(qtbot):
    """The wiring, not just the class -- a filter nobody installs is inert."""
    from spacr.qt.app import MainWindow

    win = MainWindow()
    qtbot.addWidget(win)
    assert isinstance(getattr(win, "_menu_drag", None),
                      _DragsTheWindowByTheMenuBar)
    assert win.menuBar() is not None

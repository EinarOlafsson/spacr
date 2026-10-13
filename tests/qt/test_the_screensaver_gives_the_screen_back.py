"""``spacr.qt.screensaver``: leaving it, and never taking the app with it.

The module docstring explains why this is a window of its own rather than a
takeover of the main one: hiding the real widgets and restoring them means
remembering what was visible, what had focus, which docks were open and where
the splitters were -- and getting any of it wrong leaves the user's layout
rearranged by something meant to be a screensaver.

That design choice puts the whole burden on two behaviours, and neither was
tested: ANY key or click gives the screen back, and closing stops the backdrop
before the window goes. A canvas destroyed with its timer still running is the
crash this module exists not to cause.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtCore import QEvent, QPointF, Qt                       # noqa: E402
from PySide6.QtGui import QKeyEvent, QMouseEvent                    # noqa: E402
from PySide6.QtWidgets import QWidget                               # noqa: E402

from spacr.qt.screensaver import Screensaver, show_screensaver      # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture
def saver(qapp):
    """A screensaver, closed at the end if the test has not closed it already.

    NOT registered with ``qtbot``. ``Screensaver`` sets WA_DeleteOnClose
    deliberately -- it is a throwaway window, which is the whole reason it is
    a window rather than a takeover of the main one -- so pytest-qt's own
    teardown close lands on a freed C++ object and reports the previous test
    as not torn down. The fixture closes it once, tolerantly, and Qt frees it.
    """
    made = Screensaver()
    yield made
    try:
        if not made.isHidden():
            made.close()
    except RuntimeError:
        pass
    qapp.processEvents()


def _key(key=Qt.Key_A):
    return QKeyEvent(QEvent.Type.KeyPress, key, Qt.NoModifier)


def _click(button=Qt.LeftButton):
    # QPointF, not QPoint: the QPoint overload is deprecated in Qt 6 and
    # emits a warning the suite's own hygiene checks would rather not carry.
    return QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(4.0, 4.0),
                       button, button, Qt.NoModifier)


# ---------------------------------------------------------------------------
# giving the screen back
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", [Qt.Key_A, Qt.Key_Escape, Qt.Key_Shift,
                                 Qt.Key_F1, Qt.Key_Space])
def test_any_key_closes_it(saver, qapp, key):
    """"Any key" includes the modifiers.

    Somebody reaching for their keyboard to get their screen back may well
    press Shift first. A screensaver that ignores it looks broken, and the
    next thing they try is the power button.
    """
    saver.show()
    qapp.processEvents()

    saver.keyPressEvent(_key(key))

    assert not saver.isVisible()


@pytest.mark.parametrize("button", [Qt.LeftButton, Qt.RightButton,
                                    Qt.MiddleButton])
def test_any_click_closes_it(saver, qapp, button):
    saver.show()
    qapp.processEvents()

    saver.mousePressEvent(_click(button))

    assert not saver.isVisible()


def test_the_event_is_accepted_so_nothing_behind_it_sees_the_keystroke(saver):
    """The key that dismisses the screensaver must not also do something.

    Unaccepted, it propagates to whatever is underneath -- and the Escape
    somebody pressed to wake the screen cancels their running job as well.
    """
    event = _key(Qt.Key_Escape)

    saver.keyPressEvent(event)

    assert event.isAccepted()


# ---------------------------------------------------------------------------
# stopping the backdrop first
# ---------------------------------------------------------------------------

def test_closing_pauses_the_backdrop_before_the_window_goes(saver):
    """`pause` is the documented way to make a canvas give its threads back."""
    paused = []

    class _Backdrop:
        def pause(self):
            paused.append(True)

    saver._backdrop = _Backdrop()

    saver.close()

    assert paused == [True], "the backdrop was never paused"


def test_a_backdrop_that_cannot_be_paused_does_not_stop_the_close(saver):
    """A screensaver that refuses to close is worse than a leaked thread.

    The user is holding a key down trying to get their screen back; an
    exception out of closeEvent leaves the window up.
    """
    class _Stuck:
        def pause(self):
            raise RuntimeError("the render thread is already gone")

    saver._backdrop = _Stuck()

    saver.close()

    assert not saver.isVisible()


def test_a_backdrop_with_no_pause_is_left_alone(saver):
    """The backdrop is whatever could be built, and not all of them pause."""
    saver._backdrop = object()

    saver.close()

    assert not saver.isVisible()


def test_no_backdrop_at_all_still_closes(saver):
    """Building one can fail, and the window still has to be dismissible."""
    saver._backdrop = None

    saver.close()

    assert not saver.isVisible()


# ---------------------------------------------------------------------------
# opening it
# ---------------------------------------------------------------------------

def test_it_opens_on_the_parent_s_screen_and_takes_focus(qtbot):
    """Focus is set explicitly, and the comment says why.

    Without it the key meant to close the screensaver goes to whatever had
    focus before, and the screensaver stays up -- which is the one failure
    that cannot be recovered from inside the application.
    """
    parent = QWidget()
    qtbot.addWidget(parent)
    parent.show()
    qtbot.waitExposed(parent)

    saver = show_screensaver(parent)

    assert saver is not None
    qtbot.addWidget(saver)
    try:
        assert saver.isVisible()
        assert saver.hasFocus() or saver.focusWidget() is not None
    finally:
        saver.close()


def test_a_screensaver_that_cannot_be_opened_returns_none_rather_than_raising(
        monkeypatch):
    """It is started from a timer, where an exception has nowhere to go."""
    from spacr.qt import screensaver as module

    class _Refuses:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("no window system here")

    monkeypatch.setattr(module, "Screensaver", _Refuses)

    assert module.show_screensaver(None) is None

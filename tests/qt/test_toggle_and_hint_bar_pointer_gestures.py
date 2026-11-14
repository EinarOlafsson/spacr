"""The pointer gestures a switch and the help strip's resize grip must decline.

`Toggle` only answers to the left button, ignores a release it never saw the
press for, treats a twitch under three pixels as a tap and snaps a drag that
lets go on its own side back home.  The `HintBar` grip only starts a drag on a
left press and only commits a height when the left button is released during
a drag; everything else is left unaccepted for the parent.
"""
from __future__ import annotations

from contextlib import contextmanager

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt          # noqa: E402
from PySide6.QtGui import QMouseEvent                           # noqa: E402
from PySide6.QtTest import QTest                                # noqa: E402
from PySide6.QtWidgets import QApplication                      # noqa: E402

from spacr.qt.widgets.hint_bar import HintBar                   # noqa: E402
from spacr.qt.widgets.toggle import Toggle                      # noqa: E402

pytestmark = pytest.mark.qt


@contextmanager
def _at_design_scale():
    """Keep saved and active zoom local to designed-coordinate checks."""
    from spacr.qt import gui_scale, preferences

    settings = preferences._settings()
    keys = ("prefs/font_scale", "prefs/gui_scale")
    previous = {key: settings.value(key) for key in keys
                if settings.contains(key)}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(gui_scale, "_SCALE", 1.0)
        try:
            preferences.set_font_scale(1.0)
            preferences.set_gui_scale(1.0)
            yield
        finally:
            for key in keys:
                if key in previous:
                    settings.setValue(key, previous[key])
                else:
                    settings.remove(key)


def _toggle(qtbot):
    widget = Toggle()
    qtbot.addWidget(widget)
    widget.resize(60, 24)
    widget.show()
    qtbot.waitExposed(widget)
    return widget


def test_the_switch_flips_only_for_a_left_tap_or_a_drag_across(qtbot, qapp,
                                                               subtests):
    """Right clicks, foreign releases, twitches and same-side drags leave the switch off."""
    with _at_design_scale():
        with subtests.test(msg="a right click neither flips nor arms the drag"):
            toggle = _toggle(qtbot)
            QTest.mouseClick(toggle, Qt.RightButton, pos=QPoint(
                int(toggle._minimum_knob_x() + 6), toggle.height() // 2))
            assert toggle.isChecked() is False
            assert toggle._mouse_pressed is False

        with subtests.test(msg="a release from a press elsewhere is ignored"):
            toggle = _toggle(qtbot)
            QTest.mouseRelease(toggle, Qt.LeftButton, pos=QPoint(
                int(toggle._maximum_knob_x()), toggle.height() // 2))
            assert toggle.isChecked() is False
            assert toggle._knob_pos == float(toggle._minimum_knob_x())

        with subtests.test(msg="a twitch under three pixels is not a drag; past it the knob follows"):
            toggle = _toggle(qtbot)
            resting = toggle._knob_pos
            start, middle = int(toggle._minimum_knob_x() + 6), toggle.height() // 2
            QTest.mousePress(toggle, Qt.LeftButton, pos=QPoint(start, middle))
            QTest.mouseMove(toggle, QPoint(start + 2, middle))
            assert (toggle._dragging, toggle._knob_pos) == (False, resting)
            QTest.mouseMove(toggle, QPoint(start + 20, middle))
            assert toggle._dragging is True
            assert toggle._knob_pos > resting
            QTest.mouseRelease(toggle, Qt.LeftButton, pos=QPoint(start + 20, middle))

        with subtests.test(msg="a drag letting go on the same side snaps the knob back"):
            toggle = _toggle(qtbot)
            y = toggle.height() // 2
            start_x = toggle._minimum_knob_x() + toggle._knob_d // 2
            QTest.mousePress(toggle, Qt.LeftButton, pos=QPoint(start_x, y))
            QTest.mouseMove(toggle, pos=QPoint(start_x + 5, y))
            assert toggle._dragging is True
            assert toggle._knob_pos > toggle._minimum_knob_x()
            QTest.mouseRelease(toggle, Qt.LeftButton, pos=QPoint(start_x + 5, y))
            assert toggle.isChecked() is False
            qtbot.waitUntil(
                lambda: toggle._knob_pos == float(toggle._minimum_knob_x()),
                timeout=2000)


def _mouse(kind, y, button, buttons=None):
    """A real mouse event at global height ``y``."""
    held = button if buttons is None else buttons
    return QMouseEvent(kind, QPointF(2, 2), QPointF(2, y), button, held,
                       Qt.NoModifier)


def test_the_hint_bar_grip_drags_only_on_the_left_button(qtbot, subtests):
    """Only a left press starts a drag and only a left release during one commits."""
    press, move, release = (QEvent.Type.MouseButtonPress, QEvent.Type.MouseMove,
                            QEvent.Type.MouseButtonRelease)

    def bar():
        strip = HintBar("Hover something.")
        qtbot.addWidget(strip)
        committed = []
        strip.helpHeightCommitted.connect(committed.append)
        return strip, strip._resize_handle, committed

    for case, event in (
            ("a right press does not start a drag",
             _mouse(press, 300, Qt.RightButton)),
            ("a move with no drag under way", _mouse(move, 10, Qt.NoButton,
                                                     Qt.LeftButton)),
            ("a left release with no drag", _mouse(release, 300, Qt.LeftButton,
                                                   Qt.NoButton)),
    ):
        with subtests.test(msg=case):
            strip, handle, committed = bar()
            height = strip.height()
            QApplication.sendEvent(handle, event)
            assert event.isAccepted() is False
            assert handle._drag is None
            assert strip.height() == height
            assert strip._manual_height is None
            assert committed == []

    with subtests.test(msg="another button released mid-drag does not commit; the left one does"):
        strip, handle, committed = bar()
        start = _mouse(press, 300, Qt.LeftButton)
        QApplication.sendEvent(handle, start)
        assert start.isAccepted() is True
        assert handle._drag == (300.0, strip.height())
        other = _mouse(release, 300, Qt.RightButton, Qt.LeftButton)
        QApplication.sendEvent(handle, other)
        assert other.isAccepted() is False
        assert handle._drag == (300.0, strip.height())
        assert committed == []
        left = _mouse(release, 300, Qt.LeftButton, Qt.NoButton)
        QApplication.sendEvent(handle, left)
        assert left.isAccepted() is True
        assert handle._drag is None
        assert committed == [strip.height()]

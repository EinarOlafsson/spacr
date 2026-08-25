"""A switch only answers to the left button, and never strands its knob.

``Toggle`` takes over the mouse handling a QCheckBox would normally do, so
every event it declines to handle has to be handed back to Qt rather than
swallowed. The cases below are the ones a user produces by accident: a
right-click, a pointer passing over the switch, a release that belongs to a
press somewhere else, and a drag that lets go on the side it started from.
"""
from __future__ import annotations

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest

from spacr.qt.widgets.toggle import Toggle


@pytest.fixture
def toggle(qtbot):
    """A shown switch, wide enough for the track and a drag."""
    widget = Toggle()
    qtbot.addWidget(widget)
    widget.resize(60, 24)
    widget.show()
    qtbot.waitExposed(widget)
    return widget


def _knob_centre(widget, x):
    return QPoint(int(x), widget.height() // 2)


def test_a_right_click_does_not_flip_the_switch(toggle):
    """The right button belongs to the context menu, not to the switch."""
    start = toggle.isChecked()

    QTest.mouseClick(toggle, Qt.RightButton,
                     pos=_knob_centre(toggle, toggle._minimum_knob_x() + 6))

    assert toggle.isChecked() is start
    assert toggle._mouse_pressed is False, (
        "a right press must not arm the drag state the left button uses")


def test_a_pointer_passing_over_the_switch_does_not_drag_the_knob(toggle):
    """A move with no button held is a hover, and hovering moves nothing."""
    resting = toggle._knob_pos

    QTest.mouseMove(toggle, pos=_knob_centre(toggle, toggle._maximum_knob_x()))

    assert toggle._knob_pos == resting
    assert toggle._dragging is False


def test_a_release_from_a_press_elsewhere_is_ignored(toggle):
    """A left release the switch never saw the press for changes nothing.

    Releasing over a widget that was not the one pressed is ordinary mouse
    behaviour -- press on a neighbouring control, slide across, let go. It
    must not count as a tap on the switch.
    """
    start = toggle.isChecked()

    QTest.mouseRelease(toggle, Qt.LeftButton,
                       pos=_knob_centre(toggle, toggle._maximum_knob_x()))

    assert toggle.isChecked() is start
    assert toggle._knob_pos == float(toggle._minimum_knob_x())


def test_a_drag_that_lets_go_on_the_same_side_snaps_the_knob_back(toggle,
                                                                  qtbot):
    """A short drag that does not cross the middle returns the knob home.

    The state does not change, so nothing else would move the knob: without
    an explicit re-animation it would sit wherever the finger let go, showing
    a half-open switch for a setting that is still off.
    """
    assert toggle.isChecked() is False
    y = toggle.height() // 2
    start_x = toggle._minimum_knob_x() + toggle._knob_d // 2

    QTest.mousePress(toggle, Qt.LeftButton, pos=QPoint(start_x, y))
    QTest.mouseMove(toggle, pos=QPoint(start_x + 5, y))
    assert toggle._dragging is True
    assert toggle._knob_pos > toggle._minimum_knob_x(), (
        "the drag has to have moved the knob for the snap-back to mean "
        "anything")

    QTest.mouseRelease(toggle, Qt.LeftButton, pos=QPoint(start_x + 5, y))

    assert toggle.isChecked() is False
    qtbot.waitUntil(
        lambda: toggle._knob_pos == float(toggle._minimum_knob_x()),
        timeout=2000)


def test_a_disabled_switch_is_drawn_dimmer_than_a_live_one(qtbot):
    """A switch that cannot be changed says so by fading, label and all.

    The state colour and the track fill are both painted at reduced alpha when
    the widget is disabled. Drawn at full strength it would look like a
    setting the user can still reach.
    """
    live, dead = Toggle("Timelapse"), Toggle("Timelapse")
    for widget in (live, dead):
        qtbot.addWidget(widget)
        widget.resize(120, 24)
        widget.setChecked(True)
    dead.setEnabled(False)

    knob_x = live._maximum_knob_x() + live._knob_d // 2
    y = live.height() // 2
    live_pixel = live.grab().toImage().pixelColor(knob_x, y)
    dead_pixel = dead.grab().toImage().pixelColor(knob_x, y)

    assert live_pixel != dead_pixel, (
        "the disabled switch painted its knob at full strength")


def test_a_labelled_switch_asks_for_room_for_the_track_and_the_text(qtbot):
    """The size hint covers the switch as well as the words beside it.

    A QCheckBox sizes itself for its indicator, which this widget does not
    draw; without the track's own width the label would be laid out over the
    switch.
    """
    from PySide6.QtWidgets import QCheckBox

    toggle = Toggle("Timelapse")
    qtbot.addWidget(toggle)
    plain = QCheckBox("Timelapse")
    qtbot.addWidget(plain)

    assert toggle.sizeHint().width() >= (
        plain.sizeHint().width() + toggle._track_x + toggle._track_w
        + toggle._label_gap)
    assert toggle.sizeHint().height() >= toggle._track_h

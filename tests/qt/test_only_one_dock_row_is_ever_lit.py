"""At most one dock row is accent-coloured, and it is the one under the pointer.

Reported 2026-09-05: "for some reason run compare and run history are always
blue in the dock". Read off the screen recording the maintainer took: Run
History accent-coloured while Database Browser directly above and Report
directly below are white, and the pointer is at the other end of the column.

THE CAUSE WAS `:hover`. The rows were coloured by a
`QPushButton#SidebarItem:hover` rule, and Qt drives that pseudo-state from
`WA_UnderMouse` -- which sticks when the widget under the pointer is replaced
without the pointer moving. Clicking a dock row does exactly that: the stack
swaps a whole screen in underneath it and no Leave is ever delivered. Both
rows the maintainer named are ones they had opened.

So the dock lights rows itself, from one pass over all of them, and the
stylesheet keys on a property rather than on Qt's belief about the pointer.
It is the same lesson as the hover blink before it: where Qt's per-widget
state can be stale, the container has to be the single source of truth.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QPushButton


@pytest.fixture
def dock(qtbot, qt_theme_applied):
    from spacr.qt.app import MainWindow

    win = MainWindow()
    qtbot.addWidget(win)
    win.resize(1440, 2000)
    win.show()
    return win._sidebar


def _rows(dock):
    return [r for r in dock.findChildren(QPushButton)
            if r.objectName() == "SidebarItem"]


def _lit(dock):
    return [str(r.property("navKey")) for r in _rows(dock)
            if r.property("hovered")]


def test_nothing_is_lit_at_rest(dock):
    assert _lit(dock) == []


def test_hovering_lights_exactly_one_row(dock):
    dock._on_row_hovered("run_history", True)
    assert _lit(dock) == ["run_history"]


def test_moving_on_hands_the_ink_over(dock):
    """The defect was two rows lit at once, which this is the direct test of."""
    dock._on_row_hovered("run_history", True)
    dock._on_row_hovered("mask", True)
    assert _lit(dock) == ["mask"], (
        "the previous row kept its ink -- that is the reported bug")


def test_a_row_that_never_gets_its_leave_is_still_cleared(dock):
    """The exact sequence that stuck.

    A click swaps the screen out from under the pointer, so the row is never
    told the pointer left. Only the dock's own Leave arrives, and it must be
    enough on its own.
    """
    dock._on_row_hovered("run_compare", True)
    assert _lit(dock) == ["run_compare"]
    dock.leaveEvent(QEvent(QEvent.Type.Leave))
    assert _lit(dock) == [], (
        "a row stayed lit after the pointer left the dock")


def test_the_stylesheet_does_not_colour_on_hover(dock):
    """`:hover` is what stuck. It must not come back.

    Asserted on the dock's own sheet rather than the application one,
    because that is where the rule lived.
    """
    sheet = dock.styleSheet()
    assert 'SidebarItem[hovered="true"]' in sheet, (
        "the property-driven rule is gone; rows will never light")
    assert "SidebarItem:hover" not in sheet, (
        "the `:hover` colour rule is back, and with it the stuck ink")

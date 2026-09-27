"""A plate drag held past the left or top edge scrolls back toward the
first well, and a scroll tick that arrives after the drag ended does
nothing.

The right/bottom edges are pinned in test_the_plate_viewer_is_one_locked_grid;
these are the other two directions, and the timer firing late -- once the
press has been let go -- which must stop the timer rather than move the
plate under a pointer that is no longer dragging.
"""
from __future__ import annotations

import pytest
from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QApplication

from spacr.qt.widgets.plate_map_picker import PlateMapPicker


@pytest.fixture
def big(qtbot, qt_theme_applied):
    widget = PlateMapPicker(layout=1536)
    qtbot.addWidget(widget)
    widget.resize(500, 400)
    widget.show()
    QApplication.processEvents()
    return widget


def _scrolled_to_the_end(picker):
    horizontal = picker._area.horizontalScrollBar()
    vertical = picker._area.verticalScrollBar()
    horizontal.setValue(horizontal.maximum())
    vertical.setValue(vertical.maximum())
    QApplication.processEvents()
    return horizontal, vertical


def _visible_well(picker):
    view = picker._view_rect_global()
    for (row, column), well in picker._wells.items():
        centre = well.mapToGlobal(well.rect().center())
        if view.adjusted(80, 80, -80, -80).contains(centre):
            return row, column, centre
    raise AssertionError("no well in the middle of the view")


def test_holding_past_the_left_edge_scrolls_back_left(big):
    horizontal, _vertical = _scrolled_to_the_end(big)
    start = horizontal.value()
    row, column, centre = _visible_well(big)
    view = big._view_rect_global()
    big.begin_drag(row, column)

    big.drag_to(QPoint(view.left() - 30, centre.y()))

    assert big._scroll_step[0] < 0
    assert big._autoscroll.isActive()
    for _ in range(10):
        big._autoscroll_tick()
    assert horizontal.value() < start
    assert min(c for _r, c in big.selection()) < column


def test_holding_above_the_top_edge_scrolls_back_up(big):
    _horizontal, vertical = _scrolled_to_the_end(big)
    start = vertical.value()
    row, column, centre = _visible_well(big)
    view = big._view_rect_global()
    big.begin_drag(row, column)

    big.drag_to(QPoint(centre.x(), view.top() - 30))

    assert big._scroll_step[1] < 0
    for _ in range(10):
        big._autoscroll_tick()
    assert vertical.value() < start


def test_a_tick_after_the_drag_ended_stops_the_timer_and_leaves_the_plate(big):
    view = big._view_rect_global()
    big.begin_drag(2, 1)
    big.drag_to(QPoint(view.right() + 30, view.center().y()))
    assert big._autoscroll.isActive()
    big.finish_drag()
    before = big._area.horizontalScrollBar().value()
    chosen = set(big.selection())

    big._autoscroll_tick()

    assert not big._autoscroll.isActive()
    assert big._area.horizontalScrollBar().value() == before
    assert set(big.selection()) == chosen

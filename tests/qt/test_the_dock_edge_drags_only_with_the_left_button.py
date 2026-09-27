"""The dock's resize edge and its width at their edges.

Pinned here, each as what the user sees or gets:

* the edge redraws its line when the pointer enters and leaves it;
* only a left press starts a drag: a right press, or a move or release with
  no press before it, leaves the dock's width alone and stores nothing;
* a left press released where it began is a click, not a drag, and stores
  nothing;
* a stored width that cannot be read gives the fitting width, and a width
  that cannot be stored is still applied;
* a row whose style is already gone still takes the hover mark.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QEnterEvent, QMouseEvent

from spacr.qt import preferences as P
from spacr.qt.widgets.dock import Dock, DockEdge

ROWS = [
    ("__home__", "Home", "Back to the tiles", "Core"),
    ("mask", "Generate Masks", "Segment objects", "Core"),
    ("measure", "Measure", "Extract features", "Core"),
]


@pytest.fixture
def stored(monkeypatch):
    """Record what the dock would store, instead of storing it."""
    widths = []
    monkeypatch.setattr(P, "get_dock_width", lambda: 0)
    monkeypatch.setattr(P, "set_dock_width", widths.append)
    return widths


@pytest.fixture
def dock(qtbot, stored):
    bar = Dock(ROWS, icon_for=lambda key: None)
    qtbot.addWidget(bar)
    return bar


@pytest.fixture
def edge(qtbot, dock):
    grip = DockEdge(dock)
    qtbot.addWidget(grip)
    return grip


def _mouse(kind, x, button):
    held = Qt.NoButton if kind == QEvent.Type.MouseButtonRelease else button
    return QMouseEvent(kind, QPointF(2, 2), QPointF(x, 2), button, held,
                       Qt.NoModifier)


def test_the_line_is_redrawn_as_the_pointer_comes_and_goes(edge,
                                                           monkeypatch):
    redraws = []
    monkeypatch.setattr(edge, "update", lambda: redraws.append(True))
    point = QPointF(1, 1)
    edge.enterEvent(QEnterEvent(point, point, point))
    edge.leaveEvent(QEvent(QEvent.Type.Leave))
    assert len(redraws) == 2


def test_a_right_press_does_not_start_a_drag(edge, dock, stored):
    width = dock.width()
    edge.mousePressEvent(_mouse(QEvent.Type.MouseButtonPress, 100,
                                Qt.RightButton))
    edge.mouseMoveEvent(_mouse(QEvent.Type.MouseMove, 180, Qt.RightButton))
    edge.mouseReleaseEvent(_mouse(QEvent.Type.MouseButtonRelease, 180,
                                  Qt.RightButton))
    assert edge._pressed_x is None
    assert dock.width() == width
    assert stored == []


def test_a_left_click_without_moving_stores_nothing(edge, dock, stored):
    width = dock.width()
    edge.mousePressEvent(_mouse(QEvent.Type.MouseButtonPress, 100,
                                Qt.LeftButton))
    assert edge._pressed_x == 100
    edge.mouseReleaseEvent(_mouse(QEvent.Type.MouseButtonRelease, 100,
                                  Qt.LeftButton))
    assert edge._pressed_x is None
    assert dock.width() == width
    assert stored == []


def test_a_stored_width_that_cannot_be_read_fits_the_names(dock,
                                                           monkeypatch):
    def unreadable():
        raise OSError("settings locked")

    monkeypatch.setattr(P, "get_dock_width", unreadable)
    assert dock.column_width() == dock.fitting_width()


def test_a_width_that_cannot_be_stored_is_still_applied(dock, monkeypatch):
    def unwritable(_width):
        raise OSError("settings locked")

    monkeypatch.setattr(P, "set_dock_width", unwritable)
    applied = dock.set_column_width(dock.fitting_width() + 40)
    assert dock.width() == applied
    assert applied == dock.clamp_width(dock.fitting_width() + 40)


def test_a_row_whose_style_is_gone_still_takes_the_hover_mark(dock,
                                                             monkeypatch):
    row = dock.rows()[1]
    monkeypatch.setattr(row, "style", lambda: None)
    dock._light_only(row.key)
    assert row.property("hovered") is True
    assert not dock.rows()[0].property("hovered")

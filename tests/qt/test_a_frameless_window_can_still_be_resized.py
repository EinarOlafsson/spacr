"""Dropping the frame dropped the resize grips with it.

The main window and every settings popup are frameless, so the edges the
window manager used to draw -- and the corner grip a user reaches for --
are not there. They could be moved and not resized.

The edges do it now, and the drag is handed to the window manager through
``startSystemResize`` rather than computed here: a hand-rolled resize
works until the pointer outruns the events, and then the window walks
away from the cursor.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QDialog, QLabel, QVBoxLayout

from spacr.qt.widgets import glass


@pytest.fixture
def dialog(qtbot, qt_theme_applied, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    made = QDialog()
    qtbot.addWidget(made)
    column = QVBoxLayout(made)
    column.addWidget(QLabel("something to read"))
    made.resize(400, 300)
    glass.glass(made)
    return made


def test_a_glassed_popup_carries_the_resizer(dialog):
    assert getattr(dialog, "_spacr_resizer", None) is not None


def test_installing_it_twice_leaves_one(dialog):
    """A dialog shown, closed and shown again must not collect filters."""
    first = dialog._spacr_resizer

    assert glass.let_the_user_resize(dialog) is False
    assert dialog._spacr_resizer is first


@pytest.mark.parametrize("point,expected", [
    ((1, 100), Qt.Edge.LeftEdge),
    ((399, 100), Qt.Edge.RightEdge),
    ((200, 1), Qt.Edge.TopEdge),
    ((200, 299), Qt.Edge.BottomEdge),
    ((1, 1), Qt.Edge.LeftEdge | Qt.Edge.TopEdge),
    ((399, 299), Qt.Edge.RightEdge | Qt.Edge.BottomEdge),
])
def test_every_edge_and_corner_is_a_grab(dialog, point, expected):
    assert glass._edges_at(dialog, QPoint(*point)) == expected


def test_the_middle_is_not_a_grab(dialog):
    """A band wide enough to swallow clicks on controls would be worse
    than no resize at all."""
    assert glass._edges_at(dialog, QPoint(200, 150)) == Qt.Edge(0)


@pytest.mark.parametrize("edges,shape", [
    (Qt.Edge.LeftEdge, Qt.CursorShape.SizeHorCursor),
    (Qt.Edge.TopEdge, Qt.CursorShape.SizeVerCursor),
    (Qt.Edge.LeftEdge | Qt.Edge.TopEdge, Qt.CursorShape.SizeFDiagCursor),
    (Qt.Edge.RightEdge | Qt.Edge.TopEdge, Qt.CursorShape.SizeBDiagCursor),
])
def test_the_pointer_says_which_way_it_will_move(edges, shape):
    """A resize nobody can see is one nobody finds."""
    assert glass._cursor_for(edges) == shape


def test_the_main_window_is_resizable_too(qtbot, qt_theme_applied):
    from spacr.qt.app import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)

    assert getattr(window, "_spacr_resizer", None) is not None

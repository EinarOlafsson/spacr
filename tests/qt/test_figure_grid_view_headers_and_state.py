"""The figure grid: section headings, saved arrangements and placement.

A run's heading folds its figures only on Return/Enter/Space or a left release
inside the heading; every other key or button belongs to Qt, and a press
dragged off the heading is taken back.  A saved arrangement is applied as far
as it can be read (folds even without a tile width, a numeric width through
the setter, nothing from a non-mapping).  A figure wider than the room left
starts a new row, scrolling to a vanished section reports failure, and
``clear`` discards every figure tile even after the layout lost it.

One general test per behaviour family, each walking a case table with pytest's
``subtests`` fixture.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtGui import QPixmap  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QWidget  # noqa: E402

from spacr.qt.widgets import figure_grid_view as fgv  # noqa: E402

pytestmark = pytest.mark.qt

_KEY = ("run one", 0)


def _pixmap(width, height):
    pixmap = QPixmap(width, height)
    pixmap.fill(Qt.white)
    return pixmap


class _RecordingView(QWidget):
    """An ancestor answering ``toggle_section`` that records each fold request."""

    def __init__(self):
        super().__init__()
        self.toggled = []

    def toggle_section(self, header):  # noqa: D102 - see class doc
        self.toggled.append(getattr(header, "section_key", None))
        return True


def _grid(qtbot, width=1200, height=800):
    grid = fgv.FigureGridView()
    qtbot.addWidget(grid)
    grid.resize(width, height)
    return grid


def test_section_header_folds_on_fold_keys_and_left_releases_inside_it(qtbot, subtests):
    """Fold keys and an in-heading left release fold; other keys, buttons and dragged-off releases do not."""
    def gesture_keys(*keys):
        return lambda header: [QTest.keyClick(header, key) for key in keys]

    def click(button):
        return lambda header: QTest.mouseClick(header, button, pos=QPoint(5, 5))

    def drag_off(header):
        QTest.mousePress(header, Qt.LeftButton, pos=QPoint(5, 5))
        QTest.mouseRelease(header, Qt.LeftButton, pos=QPoint(400, 300))

    cases = {
        "letter_and_arrow_fall_through": (gesture_keys(Qt.Key_A, Qt.Key_Down), []),
        "space_folds": (gesture_keys(Qt.Key_Space), [_KEY]),
        "return_and_enter_fold": (gesture_keys(Qt.Key_Return, Qt.Key_Enter),
                                  [_KEY, _KEY]),
        "right_button_ignored": (click(Qt.RightButton), []),
        "middle_button_ignored": (click(Qt.MiddleButton), []),
        "left_click_folds": (click(Qt.LeftButton), [_KEY]),
        "release_dragged_off_cancels": (drag_off, []),
    }
    for case_id, (gesture, toggled) in cases.items():
        with subtests.test(msg=case_id):
            view = _RecordingView()
            qtbot.addWidget(view)
            header = fgv._SectionHeader("run one", _KEY, view)
            header.resize(200, 24)
            gesture(header)
            assert view.toggled == toggled
            assert header.text() == "run one"


def test_grid_applies_what_it_can_read_from_a_saved_arrangement(qtbot, subtests):
    """Non-mappings apply nothing; folds restore without a width; numeric widths go through the setter."""
    for case_id, state in (("none", None), ("list", [1, 2, 3]),
                           ("string", "cell_width=400")):
        with subtests.test(msg=f"not_a_mapping_{case_id}"):
            grid = _grid(qtbot)
            before = grid._target
            assert grid.apply_workspace_state(state) is False
            assert grid._target == before
    with subtests.test(msg="folds_without_width_then_width_then_zero"):
        grid = _grid(qtbot)
        grid.set_target_cell_width(300)
        assert grid.apply_workspace_state({"collapsed": [list(_KEY)]}) is True
        assert grid._target == 300
        assert grid.is_section_collapsed(*_KEY) is True
        assert grid.apply_workspace_state({"cell_width": 420, "collapsed": []}) is True
        assert grid._target == 420
        assert grid.is_section_collapsed(*_KEY) is False
        assert grid.apply_workspace_state({"cell_width": 0}) is False
        assert grid._target == 420
    with subtests.test(msg="numeric_width_alone"):
        grid = _grid(qtbot)
        assert grid.apply_workspace_state({"cell_width": 512}) is True
        assert grid._target == 512


def test_grid_places_scrolls_and_clears_figure_tiles(qtbot, monkeypatch, subtests):
    """Wide figures wrap to a new row, vanished sections cannot be scrolled to, and clear discards orphaned tiles."""
    with subtests.test(msg="wide_figure_starts_next_row"):
        monkeypatch.setattr(fgv, "cells_across", lambda *args, **kwargs: 3)
        monkeypatch.setattr(fgv, "cell_span", lambda aspect: 2)
        grid = _grid(qtbot, 1400, 900)
        grid.set_figures([_pixmap(1200, 400), _pixmap(1200, 400)])
        positions = [grid._grid.getItemPosition(i)
                     for i in range(grid._grid.count())]
        assert positions == [(0, 0, 1, 2), (1, 0, 1, 2)]
        monkeypatch.undo()
    with subtests.test(msg="scroll_to_missing_section_fails"):
        grid = _grid(qtbot, 1400, 900)
        grid.set_figures([_pixmap(400, 400)], sections=[("run one", 0, 1)])
        assert grid._scroll_section_to_top(("run two", 7)) is False
    with subtests.test(msg="clear_discards_tiles_the_layout_lost"):
        grid = _grid(qtbot, 400, 300)
        grid.set_figures([_pixmap(24, 24), _pixmap(24, 24)], titles=["a", "b"])
        figures = list(grid._cells)
        while grid._grid.count():
            grid._grid.takeAt(0)
        discarded = []
        monkeypatch.setattr(fgv.FigureGridView, "_discard",
                            staticmethod(discarded.append))
        grid.clear()
        assert [id(d) for d in discarded] == [id(c) for c in figures]
        assert grid._cells == []

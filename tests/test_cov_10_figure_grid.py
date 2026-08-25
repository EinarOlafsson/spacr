"""The search-figure grid when the container, or the search, has no shape.

Two things decide where a figure lands: the parameters that were searched,
and how much room the container has. The branches here are the ones where
one of the two supplies nothing -- a viewport too narrow for a single cell, a
set of figures with no parameters behind them, a boolean on an axis that
cannot be compared with a number, and a coordinate missing the parameter its
caption names. Each has to produce a readable grid rather than a division by
zero or a comparison that raises.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

from spacr.qt.widgets.figure_grid import (             # noqa: E402
    SearchFigureGrid, axis_layout, cell_caption, reflow_shape,
)


def _png(tmp_path, name):
    from PySide6.QtGui import QImage
    image = QImage(8, 6, QImage.Format_RGB32)
    image.fill(0xFF3366)
    path = tmp_path / name
    assert image.save(str(path), "PNG")
    return str(path)


def test_a_container_with_no_width_still_offers_one_column():
    """A grid docked to nothing, or measured before its first layout, has
    zero usable width. Every candidate column is then narrower than a pixel,
    and the answer has to be one column of one-pixel cells rather than no
    layout at all."""
    columns, rows, cell_width = reflow_shape(3, width=0, height=100)
    assert (columns, rows) == (1, 3)
    assert cell_width == 1


def test_no_figures_is_no_grid():
    """Zero figures must not become a one-by-zero grid that a caller then
    divides by."""
    assert reflow_shape(0, width=800, height=600) == (0, 0, 0)


def test_a_boolean_on_an_axis_sorts_as_text_not_as_a_number():
    """``True`` is an int in Python, so a numeric sort would order it
    between 0 and 2 and put the axis in an order the user cannot read.
    Booleans belong with the other named values."""
    coordinates = [{"invert": True}, {"invert": False}]
    _rows, values, cells = axis_layout(coordinates, ["invert"])
    assert values == [False, True]
    assert cells == [(0, 1), (0, 0)]


def test_a_caption_names_only_the_parameters_a_figure_actually_has():
    """A resumed search can hold trials from before a parameter was added.
    Naming a parameter the trial does not carry would print a value it never
    used."""
    assert cell_caption({"lr": 0.5}, ["lr", "added_later"]) == "lr=0.5"


def test_figures_with_no_parameters_reflow_to_fit_the_container(tmp_path,
                                                                qtbot):
    """Without axes there is nothing a position can claim, so the cells are
    free to reflow. They must fill the width rather than stay on one row that
    scrolls sideways."""
    grid = SearchFigureGrid()
    qtbot.addWidget(grid)
    grid.resize(400, 400)
    for i in range(6):
        grid.add_figure(_png(tmp_path, f"f{i}.png"), {"trial": i})
    grid.relayout()

    assert grid.count() == 6
    assert grid.parameters() == []
    positions = [(grid._grid.getItemPosition(i)[0],
                  grid._grid.getItemPosition(i)[1])
                 for i in range(grid._grid.count())]
    assert len(positions) == 6
    assert max(row for row, _col in positions) >= 1


def test_a_cell_index_outside_the_grid_has_no_file(tmp_path, qtbot):
    """Asking for the path of a figure that is not there must give an empty
    string, not the last figure's file under another figure's name."""
    grid = SearchFigureGrid()
    qtbot.addWidget(grid)
    grid.add_figure(_png(tmp_path, "only.png"), {"trial": 0})
    assert grid.figure_path(0).endswith("only.png")
    assert grid.figure_path(-1) == ""
    assert grid.figure_path(7) == ""

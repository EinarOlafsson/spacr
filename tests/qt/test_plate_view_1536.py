"""The Qt plate grid draws a 1536-well plate, rows ``AA``…``AF`` included.

The screen takes its geometry from the ``plate_layout`` it is handed rather
than from a literal, so it needed no change to draw 32 x 48 — but "needed no
change" is a claim, and an untested claim about a widget that paints 1536
rectangles and 32 row letters is not worth much. This file is the claim,
tested: the right grid, the right row letters past Z, a click that lands on
``AF48``, and a paint pass that survives the cell size a 1536 plate implies.

Runs offscreen. The plate is built by :func:`spacr.plate_qc.plate_layout`
from identifiers composed by :mod:`spacr.schema`, so the row letters under
test are the ones the database agrees with.
"""
from __future__ import annotations

import pandas as pd
import pytest

from PySide6.QtGui import QPixmap

from spacr import plate_qc as pqc
from spacr.qt.screens.plate_view import PlateGridWidget


N_ROWS, N_COLS = 32, 48


@pytest.fixture
def layout_1536():
    """A full 1536-well layout, one object per well, value ``row*100+col``."""
    frame = pd.DataFrame({
        "prc": [f"plate1_r{r}_c{c}"
                for r in range(1, N_ROWS + 1) for c in range(1, N_COLS + 1)],
        "value": [float(r * 100 + c)
                  for r in range(1, N_ROWS + 1) for c in range(1, N_COLS + 1)],
    })
    layout = pqc.plate_layout(frame, "value", grouping="mean")
    assert layout.attrs["plate_format"] == 1536
    return layout


@pytest.fixture
def grid(layout_1536, qtbot):
    widget = PlateGridWidget()
    qtbot.addWidget(widget)
    widget.resize(1200, 800)
    widget.set_plate(layout_1536, vmin=101.0, vmax=3248.0)
    return widget


def test_the_grid_is_32_by_48(grid):
    assert grid.grid_size() == (N_ROWS, N_COLS)
    assert grid.has_plate()
    assert grid.well_value(N_ROWS, N_COLS) == 32 * 100 + 48
    assert grid.well_count(N_ROWS, N_COLS) == 1


def test_the_row_letters_run_past_z(grid):
    """Row 27 is ``AA`` and row 32 is ``AF`` — the labels the widget paints."""
    assert pqc.row_label(1) == "A"
    assert pqc.row_label(26) == "Z"
    assert pqc.row_label(27) == "AA"
    assert pqc.row_label(N_ROWS) == "AF"
    labels = [pqc.row_label(r) for r in range(1, N_ROWS + 1)]
    assert len(set(labels)) == N_ROWS
    assert all(l.isalpha() and l.isupper() for l in labels)


def test_a_click_in_the_far_corner_lands_on_af48(grid):
    """The hit test must scale with the grid, not with a 384-well cell size."""
    rect = grid.cell_rect(N_ROWS, N_COLS)
    hit = grid.well_at(rect.center().toPoint())
    assert hit == (N_ROWS, N_COLS)
    assert pqc.well_id(*hit) == "AF48"

    assert grid.well_at(grid.cell_rect(1, 1).center().toPoint()) == (1, 1)
    assert grid.well_at(grid.cell_rect(27, 1).center().toPoint()) == (27, 1)


def test_painting_1536_wells_does_not_fall_over(grid):
    grid.render(QPixmap(grid.size()))
    grid.select(N_ROWS, N_COLS)
    assert grid.selected_well() == (N_ROWS, N_COLS)
    grid.render(QPixmap(grid.size()))


def test_a_gap_in_a_1536_plate_stays_a_gap(layout_1536, qtbot):
    """An unpipetted well past row P is blank, not a coloured zero."""
    trimmed = layout_1536[~((layout_1536["row_index"] == 30)
                            & (layout_1536["column_index"] == 40))]
    trimmed.attrs = dict(layout_1536.attrs)

    widget = PlateGridWidget()
    qtbot.addWidget(widget)
    widget.resize(1200, 800)
    widget.set_plate(trimmed, vmin=101.0, vmax=3248.0)

    assert widget.grid_size() == (N_ROWS, N_COLS)     # the grid is still whole
    assert widget.well_value(30, 40) is None
    assert widget.well_count(30, 40) == 0
    assert widget.well_value(30, 41) == 30 * 100 + 41
    widget.render(QPixmap(widget.size()))

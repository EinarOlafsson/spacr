"""``facet_grid`` terminates even when asked for a ceiling it cannot meet.

Instruction 288. The trimming loop shrinks the grid until rows x columns
fits under ``max_panels``. When neither axis can lose another level it
breaks, and that break was marked ``# pragma: no cover - unreachable
while max_panels >= 1``.

The reason is accurate: at one row and one column the product is 1, so
the loop is only still running if the ceiling is below 1, and no caller
in spaCR passes one. ``max_panels`` is a documented keyword on a public
function, though, and without the break a ceiling of 0 is an INFINITE
LOOP -- a frozen window with no error and nothing in the log.

So it is driven rather than deleted. The cost of being wrong about
"nobody passes 0" is not a wrong picture, it is a hang.
"""
from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets.graph_spec import (
    MAX_PANELS, SCATTER, GraphSpec, facet_grid,
)


def _frame():
    return pd.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0],
        "y": [1.0, 2.0, 3.0, 4.0],
        "row": ["a", "a", "b", "b"],
        "col": ["p", "q", "p", "q"],
    })


def _spec():
    return GraphSpec(x="x", y="y", kind=SCATTER,
                     facet_row="row", facet_col="col")


@pytest.mark.timeout(30)
def test_a_ceiling_of_zero_stops_rather_than_spinning():
    """THE ARM. Without the break this never returns."""
    grid = facet_grid(_frame(), _spec(), max_panels=0)

    # One row and one column is as small as the trimming can make it; the
    # break is what stops it trying to go further.
    assert len(grid.row_levels) == 1
    assert len(grid.col_levels) == 1
    assert len(grid.panels) == 1


@pytest.mark.timeout(30)
def test_a_ceiling_of_one_also_terminates():
    """The boundary the pragma's reason names.

    At exactly 1 the loop condition goes false rather than the break
    firing, so this pins the two sides of that boundary apart.
    """
    grid = facet_grid(_frame(), _spec(), max_panels=1)
    assert len(grid.panels) == 1


def test_an_ordinary_ceiling_keeps_the_whole_grid():
    """So the tests above are not passing because trimming eats every
    grid regardless of the ceiling."""
    grid = facet_grid(_frame(), _spec(), max_panels=MAX_PANELS)
    assert len(grid.row_levels) == 2
    assert len(grid.col_levels) == 2
    assert len(grid.panels) == 4


def test_the_trim_takes_columns_before_rows():
    """The documented preference, pinned because the break sits inside
    the same loop and a rewrite could quietly reverse it.

    A grid is read down the page, so losing a column costs less than
    losing a row.
    """
    frame = pd.DataFrame({
        "x": [1.0] * 9, "y": [1.0] * 9,
        "row": ["a", "b", "c"] * 3,
        "col": ["p"] * 3 + ["q"] * 3 + ["r"] * 3,
    })
    spec = GraphSpec(x="x", y="y", kind=SCATTER,
                     facet_row="row", facet_col="col")

    grid = facet_grid(frame, spec, max_panels=6)

    assert len(grid.row_levels) == 3, "a row was trimmed before a column"
    assert len(grid.col_levels) == 2

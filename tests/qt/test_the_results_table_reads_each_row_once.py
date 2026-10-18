"""The coefficient table reads each frame row once, not once per cell.

``ResultsTable.set_frame`` built a whole row Series for every CELL, which
for lesson 13's 759 x 30 coefficient table was 22,770 row constructions and
over a second of the GUI thread on every redraw. It now builds each row
once and shows exactly the values it showed before.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets.fast_plots import ResultsTable

pytestmark = pytest.mark.qt


def _old_texts(frame):
    from spacr.qt.widgets.sortable_table import table_item

    return [[table_item(frame.iloc[row][name]).text()
             for name in frame.columns] for row in range(len(frame))]


def _texts(table):
    grid = table.table
    return [[grid.item(row, column).text()
             for column in range(grid.columnCount())]
            for row in range(grid.rowCount())]


FRAMES = {
    "mixed": pd.DataFrame({
        "feature": ["gene[a]", "gene[b]", None],
        "coef": [0.5, np.nan, -1.25e-7],
        "n": [3, 4, 5],
        "hit": [True, False, True],
    }),
    "numeric": pd.DataFrame({"n": [1, 2, 3], "p": [0.1, 0.2, 0.3]}),
    "duplicated": pd.DataFrame([[1, "x", 2.5], [2, "y", 3.5]],
                               columns=["a", "b", "a"]),
}


@pytest.mark.parametrize("name", sorted(FRAMES))
def test_each_row_is_read_once_and_shows_the_same_values(
        qtbot, monkeypatch, name):
    frame = FRAMES[name]
    expected = _old_texts(frame)
    reads = []
    real = pd.DataFrame._ixs

    def counted(self, i, axis=0):
        if self is frame and axis == 0:
            reads.append(i)
        return real(self, i, axis=axis)

    monkeypatch.setattr(pd.DataFrame, "_ixs", counted)
    table = ResultsTable()
    qtbot.addWidget(table)

    assert table.set_frame(frame) == len(frame)

    assert sorted(reads) == list(range(len(frame)))
    table.table.setSortingEnabled(False)
    assert _texts(table) == expected

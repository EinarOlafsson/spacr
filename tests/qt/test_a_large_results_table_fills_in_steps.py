"""A large coefficient table fills over several event-loop turns.

Lesson 13's finished regression filled a 759 x 30 table in one go on the
GUI thread, a few hundred milliseconds of it after every other cost of the
result had been cut. Regression's tables now fill a few thousand cells per
turn; anything that reads the rows finishes the fill first, and the result
is the table a one-go fill makes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt

from spacr.qt.widgets.fast_plots import ResultsTable

pytestmark = pytest.mark.qt


def _frame(rows=200, columns=30):
    data = {f"c{i}": np.arange(rows) * (i + 1) * 0.5 for i in range(columns)}
    data["c0"] = [f"gene_fraction:gene[G{i}]" for i in range(rows)]
    return pd.DataFrame(data).rename(columns={"c0": "feature"})


def _texts(table):
    grid = table.table
    return [[grid.item(r, c).text() for c in range(grid.columnCount())]
            for r in range(grid.rowCount())]


def _table(qtbot, stepped):
    table = ResultsTable()
    table.fill_in_steps = stepped
    qtbot.addWidget(table)
    return table


def test_the_fill_spreads_over_turns_and_matches_a_one_go_fill(qtbot):
    whole = _table(qtbot, False)
    whole.set_frame(_frame())
    stepped = _table(qtbot, True)
    turns = []
    real = ResultsTable._fill_a_step

    def counted(self):
        turns.append(1)
        return real(self)

    ResultsTable._fill_a_step = counted
    try:
        assert stepped.set_frame(_frame()) == 200
        assert stepped._filling is not None, "the fill ran in one go"
        qtbot.waitUntil(lambda: stepped._filling is None, timeout=10000)
    finally:
        ResultsTable._fill_a_step = real

    assert len(turns) >= 3
    assert _texts(stepped) == _texts(whole)
    assert stepped._count.text() == whole._count.text()
    assert stepped.table.isSortingEnabled()


def test_reading_a_cell_finishes_the_fill(qtbot):
    table = _table(qtbot, True)
    table.set_frame(_frame())
    assert table._filling is not None

    item = table.table.item(199, 1)

    assert item is not None and item.text() == "199.0"
    assert table._filling is None


def test_selecting_keys_mid_fill_sees_every_row(qtbot):
    table = _table(qtbot, True)
    table.set_frame(_frame())

    found = table.select_keys(["gene_fraction:gene[G199]"])

    assert found == 1
    assert table.selected_keys() == ["gene_fraction:gene[G199]"]


def test_a_new_frame_replaces_a_fill_still_running(qtbot):
    table = _table(qtbot, True)
    table.set_frame(_frame())
    table.set_frame(_frame(rows=3, columns=4))
    qtbot.wait(50)

    assert table.table.rowCount() == 3
    assert table.table.item(2, 1).text() == "2.0"


def test_a_small_or_unflagged_table_fills_at_once(qtbot):
    small = _table(qtbot, True)
    small.set_frame(_frame(rows=10, columns=5))
    plain = _table(qtbot, False)
    plain.set_frame(_frame())

    assert small._filling is None and plain._filling is None
    assert plain.table.item(199, 1) is not None
    assert plain.table.item(0, 0).data(Qt.UserRole) is not None

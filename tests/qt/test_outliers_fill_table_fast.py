"""Refilling the Outliers object table is cheap, keeps every item alive and shows the new values.

The wall-clock budgets hold in an uninstrumented process; under a coverage
tracer every Python line of the fill is traced, so there the timings measure
the tracer and only the behaviour is asserted.
"""
import sys
import time

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QTableWidget

from spacr.qt.screens.outliers import _fill_table
from spacr.qt.widgets.sortable_table import install_sorting


#: Whether the wall-clock budgets apply: not under a line tracer such as coverage.
TIMED = sys.gettrace() is None


def _frame(rows, columns, seed):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.random((rows, columns)), columns=[f"f{i}" for i in range(columns)])


def test_a_large_refill_takes_under_a_second_and_shows_the_new_values(qtbot):
    table = QTableWidget()
    qtbot.addWidget(table)
    table.resize(1000, 500)
    table.show()
    install_sorting(table)
    first, second = _frame(500, 100, 1), _frame(500, 100, 2)
    _fill_table(table, first)
    kept = table.item(3, 4)
    started = time.perf_counter()
    _fill_table(table, second)
    elapsed = time.perf_counter() - started
    assert elapsed < 1.0 or not TIMED, f"refilling 50 000 cells took {elapsed:.2f} s"
    assert table.rowCount() == 500 and table.columnCount() == 100
    assert table.item(3, 4) is kept                                   # rewritten in place, none deleted
    assert table.item(499, 99).text() == f"{second.iloc[499, 99]:.6g}"
    assert table.horizontalHeaderItem(7).text() == "f7"


def test_a_smaller_frame_shrinks_the_table_and_a_bigger_one_grows_it(qtbot):
    table = QTableWidget()
    qtbot.addWidget(table)
    install_sorting(table)
    _fill_table(table, _frame(20, 6, 3))
    _fill_table(table, _frame(5, 3, 4))
    assert (table.rowCount(), table.columnCount()) == (5, 3)
    big = _frame(30, 9, 5)
    _fill_table(table, big)
    assert (table.rowCount(), table.columnCount()) == (30, 9)
    assert table.item(29, 8).text() == f"{big.iloc[29, 8]:.6g}"


def test_shrinking_from_every_feature_to_one_is_cheap_and_regrowing_reuses_the_kept_items(qtbot):
    """An all-features scan (many columns) followed by a one-feature scan must not wipe every surplus cell."""
    table = QTableWidget()
    qtbot.addWidget(table)
    table.resize(1000, 500)
    table.show()
    install_sorting(table)
    wide, narrow = _frame(500, 280, 6), _frame(500, 3, 7)
    _fill_table(table, wide)
    started = time.perf_counter()
    _fill_table(table, narrow)
    shrink = time.perf_counter() - started
    assert shrink < 1.0 or not TIMED, f"shrinking 140 000 cells took {shrink:.2f} s"
    assert (table.rowCount(), table.columnCount()) == (500, 3)
    assert table.item(499, 2).text() == f"{narrow.iloc[499, 2]:.6g}"
    started = time.perf_counter()
    _fill_table(table, wide)
    regrow = time.perf_counter() - started
    assert regrow < 3.0 or not TIMED, f"regrowing 140 000 cells took {regrow:.2f} s"
    assert table.item(250, 279).text() == f"{wide.iloc[250, 279]:.6g}"

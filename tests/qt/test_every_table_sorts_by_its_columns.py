"""One sorting contract, measured on the drawn table.

Every table in the application answers a header click the same way: the
first click sorts descending, the second ascending, the third puts the rows
back in the order the table was built in. The assertions here read the rows
Qt has laid out, and click with a real mouse event on the header, because a
model call proves nothing about what the user sees.
"""
from __future__ import annotations

import sys
import time

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QTableWidget, QTreeWidget  # noqa: E402

from spacr.qt.widgets.sortable_table import (install_sorting,  # noqa: E402
                                             table_item, tree_item)


def click_header(view, column, app):
    """Click a column header the way a mouse does, then let Qt catch up."""
    header = (view.header() if isinstance(view, QTreeWidget)
              else view.horizontalHeader())
    x = header.sectionViewportPosition(column) + header.sectionSize(column) // 2
    QTest.mouseClick(header.viewport(), Qt.LeftButton, Qt.NoModifier,
                     QPoint(x, header.height() // 2))
    app.processEvents()


def drawn_column(view, column):
    """The column as drawn, top row first, skipping hidden rows."""
    if isinstance(view, QTreeWidget):
        return [view.topLevelItem(row).text(column)
                for row in range(view.topLevelItemCount())]
    return [(view.item(row, column).text() if view.item(row, column) else "")
            for row in range(view.rowCount())
            if not view.isRowHidden(row)]


def build_table(values, qapp, headers=("value", "name")):
    table = QTableWidget(0, len(headers))
    table.setHorizontalHeaderLabels(list(headers))
    install_sorting(table)
    table.setRowCount(len(values))
    for row, value in enumerate(values):
        table.setItem(row, 0, table_item(value))
        table.setItem(row, 1, table_item(f"row{row}"))
    table.resize(400, 300)
    table.show()
    qapp.processEvents()
    return table


def test_a_numeric_column_sorts_as_numbers_not_as_words(qapp):
    """2, 9, 10 sorts 10, 9, 2 -- the failure a lexical sort hides."""
    table = build_table([2, 9, 10], qapp)
    click_header(table, 0, qapp)
    assert drawn_column(table, 0) == ["10", "9", "2"]
    table.deleteLater()


def test_the_first_click_is_descending_the_second_ascending(qapp):
    table = build_table([2, 9, 10], qapp)
    header = table.horizontalHeader()

    click_header(table, 0, qapp)
    assert header.sortIndicatorOrder() == Qt.DescendingOrder
    assert drawn_column(table, 0) == ["10", "9", "2"]

    click_header(table, 0, qapp)
    assert header.sortIndicatorOrder() == Qt.AscendingOrder
    assert drawn_column(table, 0) == ["2", "9", "10"]
    table.deleteLater()


def test_the_third_click_gives_the_table_its_own_order_back(qapp):
    """Built out of order on purpose: the table's own order has to be a
    THIRD answer, not the ascending one wearing a different hat."""
    table = build_table([9, 2, 10], qapp)
    click_header(table, 0, qapp)
    assert drawn_column(table, 0) == ["10", "9", "2"]
    click_header(table, 0, qapp)
    assert drawn_column(table, 0) == ["2", "9", "10"]
    click_header(table, 0, qapp)
    assert drawn_column(table, 0) == ["9", "2", "10"]
    assert drawn_column(table, 1) == ["row0", "row1", "row2"]
    # And the cycle starts over rather than dead-ending.
    click_header(table, 0, qapp)
    assert drawn_column(table, 0) == ["10", "9", "2"]
    table.deleteLater()


def test_a_fresh_column_starts_descending_too(qapp):
    """Not only column 0: moving to another column restarts the cycle."""
    table = build_table([2, 9, 10], qapp)
    click_header(table, 0, qapp)
    click_header(table, 1, qapp)
    assert table.horizontalHeader().sortIndicatorOrder() == Qt.DescendingOrder
    assert drawn_column(table, 1) == ["row2", "row1", "row0"]
    table.deleteLater()


def test_scientific_notation_is_a_number_not_a_word(qapp):
    table = build_table(["1e-2", "3e-9", "2e-5"], qapp)
    click_header(table, 0, qapp)
    assert drawn_column(table, 0) == ["1e-2", "2e-5", "3e-9"]
    table.deleteLater()


def test_blanks_and_nan_land_last_whichever_way_it_is_sorted(qapp):
    import math

    table = build_table([2, float("nan"), 10, None, 9], qapp)
    click_header(table, 0, qapp)
    assert drawn_column(table, 0)[:3] == ["10", "9", "2"]
    assert drawn_column(table, 0)[3:] == ["", ""]
    click_header(table, 0, qapp)
    assert drawn_column(table, 0)[:3] == ["2", "9", "10"]
    assert drawn_column(table, 0)[3:] == ["", ""]
    assert math.isnan(float("nan"))
    table.deleteLater()


def test_a_word_column_still_sorts_alphabetically(qapp):
    table = build_table(["beta", "alpha", "gamma"], qapp)
    click_header(table, 0, qapp)
    assert drawn_column(table, 0) == ["gamma", "beta", "alpha"]
    click_header(table, 0, qapp)
    assert drawn_column(table, 0) == ["alpha", "beta", "gamma"]
    table.deleteLater()


def test_a_tree_sorts_the_same_way(qapp):
    tree = QTreeWidget()
    tree.setColumnCount(2)
    tree.setHeaderLabels(["value", "name"])
    install_sorting(tree)
    # Built out of order on purpose, so "the tree's own order" is a third
    # answer and not the ascending one wearing a different hat.
    for row, value in enumerate([9, 2, 10]):
        tree.addTopLevelItem(tree_item([str(value), f"row{row}"]))
    tree.resize(400, 300)
    tree.show()
    qapp.processEvents()

    click_header(tree, 0, qapp)
    assert drawn_column(tree, 0) == ["10", "9", "2"]
    click_header(tree, 0, qapp)
    assert drawn_column(tree, 0) == ["2", "9", "10"]
    click_header(tree, 0, qapp)
    assert drawn_column(tree, 0) == ["9", "2", "10"], "the tree's own order"
    assert drawn_column(tree, 1) == ["row0", "row1", "row2"]
    tree.deleteLater()


def test_a_refill_does_not_scramble_the_rows_of_a_sorted_table(qapp):
    """Qt re-sorts on every write into the sorted column. That moves the row
    out from under the loop filling it, and the rest of the row lands on a
    different one."""
    table = build_table([2, 9, 10], qapp)
    click_header(table, 0, qapp)
    assert drawn_column(table, 0) == ["10", "9", "2"]

    table.setRowCount(0)
    values = [5, 1, 7, 3]
    table.setRowCount(len(values))
    for row, value in enumerate(values):
        table.setItem(row, 0, table_item(value))
        table.setItem(row, 1, table_item(f"row{row}"))
    qapp.processEvents()

    # The pairing survived, and the sort the user chose came back.
    pairs = list(zip(drawn_column(table, 0), drawn_column(table, 1)))
    assert pairs == [("7", "row2"), ("5", "row0"), ("3", "row3"),
                     ("1", "row1")]
    table.deleteLater()


@pytest.mark.skipif(
    sys.gettrace() is not None,
    reason="wall-clock sorting budget requires an uninstrumented process",
)
def test_fifty_thousand_rows_sort_without_the_window_going_away(qapp):
    values = [(i * 7919) % 50000 for i in range(50000)]
    table = QTableWidget(0, 3)
    table.setHorizontalHeaderLabels(["value", "name", "extra"])
    install_sorting(table)
    table.setRowCount(len(values))
    for row, value in enumerate(values):
        table.setItem(row, 0, table_item(value))
        table.setItem(row, 1, table_item(f"g{row}"))
        table.setItem(row, 2, table_item(value / 3.0))
    table.resize(600, 400)
    table.show()
    qapp.processEvents()

    started = time.perf_counter()
    click_header(table, 0, qapp)
    elapsed = time.perf_counter() - started

    assert drawn_column(table, 0)[:3] == ["49999", "49998", "49997"]
    # A click that blocks the GUI thread for a second reads as a hang.
    assert elapsed < 2.5, f"sorting 50k rows took {elapsed:.2f}s"
    table.deleteLater()


def test_editing_a_cell_re_reads_the_number_it_sorts_on(qapp):
    """An editable column is edited, and the sort has to follow the edit."""
    table = build_table([3, 9, 10], qapp)
    table.item(0, 0).setText("20")
    qapp.processEvents()

    click_header(table, 0, qapp)
    assert drawn_column(table, 0) == ["20", "10", "9"]
    table.deleteLater()


def test_a_cell_can_carry_its_own_key_when_its_text_is_not_a_number(qapp):
    """A size reads "900 KB" and belongs under "12 MB"."""
    table = QTableWidget(0, 1)
    table.setHorizontalHeaderLabels(["size"])
    install_sorting(table)
    table.setRowCount(3)
    for row, (text, key) in enumerate(
            [("900 KB", 921600), ("12 MB", 12582912), ("4 KB", 4096)]):
        table.setItem(row, 0, table_item(text, key=key))
    table.resize(300, 200)
    table.show()
    qapp.processEvents()

    click_header(table, 0, qapp)
    assert drawn_column(table, 0) == ["12 MB", "900 KB", "4 KB"]
    table.deleteLater()


def test_a_gene_name_is_not_a_number(qapp):
    """"TP53" must not sort as 53, or a gene column sorts by its digits."""
    from spacr.qt.widgets.sortable_table import numeric_value

    assert numeric_value("TP53") is None
    assert numeric_value("gene10") is None
    assert numeric_value("A01") is None
    assert numeric_value("3.2 s") == 3.2
    assert numeric_value("1,234") == 1234.0

"""Sortable table cells, tree rows and the installer that wires a view.

:func:`spacr.qt.widgets.sortable_table.table_item` and :func:`tree_item` build
cells that sort numbers as numbers, keep an explicit key, and put missing
values last whichever way the column is sorted.  :func:`install_sorting` gives
a view the header contract (descending first, clearable indicator, natural
order on start) and suspends the sort while a fill is running.  Each test walks
a case table of comparisons or views and asserts the exact order or header
state.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtCore import Qt                                       # noqa: E402
from PySide6.QtGui import QStandardItem, QStandardItemModel         # noqa: E402
from PySide6.QtWidgets import (QHeaderView, QListView, QTableView,  # noqa: E402
                               QTableWidget, QTableWidgetItem, QTreeWidget,
                               QTreeWidgetItem)

from spacr.qt.widgets import sortable_table as st                   # noqa: E402
from spacr.qt.widgets.sortable_table import (                       # noqa: E402
    _STATE_ATTR, install_sorting, numeric_value, SORT_KEY_ROLE,
    SortableProxyModel, table_item, tree_item)

pytestmark = pytest.mark.qt


def _table(qapp, columns=3, rows=3):
    """A sorted table of ``row + column`` numbers, its fill already settled."""
    table = QTableWidget(0, columns)
    table.setHorizontalHeaderLabels([f"c{column}" for column in range(columns)])
    install_sorting(table)
    table.setRowCount(rows)
    for row in range(rows):
        for column in range(columns):
            table.setItem(row, column, table_item(row + column))
    qapp.processEvents()
    return table


def _tree_column(tree, column=0):
    """The text of one column of a tree, top to bottom."""
    return [tree.topLevelItem(row).text(column)
            for row in range(tree.topLevelItemCount())]


def test_tree_rows_compare_on_the_sorted_column_with_holes_last(
        qtbot, monkeypatch, subtests):
    """``tree_item`` ``<``: numbers before words, explicit keys win, plain Qt
    rows on the other side, and blanks last in both directions."""
    for case, left, right, expected in [
            ("words, no tree", ["apple"], ["banana"], True),
            ("words reversed", ["banana"], ["apple"], False),
            ("hole after a number", [""], ["5"], False),
            ("number before a hole", ["5"], [""], True),
            ("blank vs dash", [""], ["—"], False),
            ("dash vs blank", ["—"], [""], False),
            ("number before a word", ["5"], ["apple"], True),
            ("word after a number", ["apple"], ["5"], False),
    ]:
        with subtests.test(msg=case):
            assert (tree_item(left) < tree_item(right)) is expected

    for case, left, right, expected in [
            ("ours below a plain item", "5", "9", True),
            ("ours above a plain item", "9", "5", False),
    ]:
        with subtests.test(msg=case):
            assert (tree_item([left]) < QTreeWidgetItem([right])) is expected

    with subtests.test(msg="an explicit key beats the cell text"):
        item = tree_item(["2 h 5 m"])
        item.setData(0, SORT_KEY_ROLE, 125.0)
        assert (item < tree_item(["30"])) is False
        assert (tree_item(["30"]) < item) is True

    with subtests.test(msg="a tree with a header sorts holes last"):
        tree = QTreeWidget()
        qtbot.addWidget(tree)
        tree.setColumnCount(2)
        tree.setHeaderLabels(["value", "name"])
        install_sorting(tree)
        for index, value in enumerate(["5", "", "1", "—"]):
            tree.addTopLevelItem(tree_item([value, f"row{index}"]))
        tree.sortItems(0, Qt.AscendingOrder)
        assert _tree_column(tree) == ["1", "5", "", "—"]
        tree.sortItems(0, Qt.DescendingOrder)
        assert _tree_column(tree) == ["5", "1", "", "—"]

    with subtests.test(msg="an unprimed row against a plain Qt row"):
        tree = QTreeWidget()
        qtbot.addWidget(tree)
        ours, plain = tree_item(["2"]), QTreeWidgetItem(["10"])
        tree.addTopLevelItems([ours, plain])
        assert getattr(ours, "_sort_primed", None) is None
        monkeypatch.setattr(st, "_PRIMED", (False, object()))
        assert ours < plain
        ours.setText(0, "20")
        assert not ours < plain
        assert _tree_column(tree) == ["20", "10"]
        monkeypatch.undo()

    with subtests.test(msg="a retired sort state touches nothing"):
        import shiboken6
        tree = QTreeWidget()
        qtbot.addWidget(tree)
        tree.addTopLevelItems([tree_item(["10"]), tree_item(["2"])])
        install_sorting(tree)
        state = getattr(tree, _STATE_ATTR)
        shiboken6.delete(state)
        before = _tree_column(tree)
        st._hold_sorting_for_fill(tree)
        st._keep_sorting_held(tree)
        assert _tree_column(tree) == before == ["10", "2"]
        assert not shiboken6.isValid(state)


def test_table_cells_sort_on_their_key_and_the_fill_guard_puts_it_back(
        qapp, qtbot, subtests):
    """``table_item`` keys and the fill guard: a resume restores the user's
    sort unless it was cleared or its column went away."""
    with subtests.test(msg="a percent sign does not make a word a number"):
        assert numeric_value("42%") == 42.0
        assert numeric_value("abc%") is None
        assert (table_item("abc%") < table_item("zzz%")) is True

    with subtests.test(msg="editing the text keeps the explicit key"):
        item = table_item("2 h 5 m", key=125)
        item.setText("3 h")
        assert item.text() == "3 h"
        assert (table_item(100) < item) is True
        assert (item < table_item(130)) is True
        assert (item < table_item(4)) is False

    with subtests.test(msg="a failed trial sorts below a scored one"):
        from spacr.qt.screens.hyperparam import _NumericTableItem
        assert _NumericTableItem("0.812", 0.812) < QTableWidgetItem("-")
        assert table_item("0.812", key=0.812) < table_item("-")
        assert not table_item("-") < table_item("0.812", key=0.812)

    with subtests.test(msg="the guard stands down while a resume runs"):
        table = _table(qapp)
        state = getattr(table, _STATE_ATTR)
        header = table.horizontalHeader()
        header.setSortIndicator(1, Qt.AscendingOrder)
        state._resuming = True
        table.insertRow(0)
        assert state._suspended is None
        assert header.sortIndicatorSection() == 1
        state._resuming = False
        table.insertRow(0)
        assert state._suspended == (1, Qt.AscendingOrder)
        assert header.sortIndicatorSection() == -1
        qapp.processEvents()
        assert header.sortIndicatorSection() == 1
        table.deleteLater()

    with subtests.test(msg="a late resume leaves a cleared sort cleared"):
        table = _table(qapp)
        state = getattr(table, _STATE_ATTR)
        header = table.horizontalHeader()
        header.setSortIndicator(1, Qt.DescendingOrder)
        table.insertRow(0)
        qapp.processEvents()
        assert header.sortIndicatorSection() == 1
        header.setSortIndicator(-1, Qt.AscendingOrder)
        state.resume_after_fill()
        assert header.sortIndicatorSection() == -1
        table.deleteLater()

    for case, narrow_to, expected in [("column kept", None, 2),
                                      ("column gone", 1, -1)]:
        with subtests.test(msg=case):
            table = _table(qapp)
            table.horizontalHeader().setSortIndicator(2, Qt.DescendingOrder)
            table.insertRow(0)
            assert table.horizontalHeader().sortIndicatorSection() == -1
            if narrow_to is not None:
                table.setColumnCount(narrow_to)
            qapp.processEvents()
            assert table.horizontalHeader().sortIndicatorSection() == expected
            table.deleteLater()

    with subtests.test(msg="retiring a table with empty cells"):
        table = QTableWidget(2, 2)
        qtbot.addWidget(table)
        table.setItem(0, 0, table_item("kept"))
        assert table.item(1, 1) is None
        st._retire_table_cells(table)
        assert table.rowCount() == 0
        assert table.item(0, 0) is None
        spare = st._spare_items(table)
        assert [cell.text() for cell in spare] == [""]


def test_the_sweep_table_reads_its_gene_from_a_sortable_cell(qtbot):
    """``SweepPanel.selected_gene`` reads a ``table_item`` gene cell and
    answers None for a blank one before any sweep has run."""
    from spacr.qt.widgets.sweep_panel import SweepPanel

    panel = SweepPanel(threaded=False)
    qtbot.addWidget(panel)
    assert panel._result is None
    panel.table.setSortingEnabled(False)
    panel.table.insertRow(0)
    panel.table.setItem(0, 0, table_item("gene"))
    panel.table.setItem(0, 1, table_item("223550"))
    panel.table.setSortingEnabled(True)
    panel.table.selectRow(0)
    assert panel.selected_gene() == "223550"

    panel.table.setSortingEnabled(False)
    panel.table.setItem(0, 1, table_item("  "))
    panel.table.setSortingEnabled(True)
    panel.table.selectRow(0)
    assert panel.table.selectedIndexes()
    assert panel.selected_gene() is None


def test_install_sorting_states_the_header_contract_on_every_view(
        qapp, subtests):
    """``install_sorting`` on no view, a list, a table, a tree, a model view,
    a proxy view, a second install and an old Qt header."""
    descending = Qt.DescendingOrder.value

    with subtests.test(msg="no view, and a view with no header"):
        assert install_sorting(None) is None
        listing = QListView()
        assert install_sorting(listing) is listing
        assert getattr(listing, _STATE_ATTR, None) is None
        table = QTableWidget(0, 1)
        assert install_sorting(table) is table
        assert getattr(table, _STATE_ATTR, None) is not None
        table.deleteLater()

    with subtests.test(msg="a second install restates the header"):
        table = QTableWidget(0, 2)
        table.setHorizontalHeaderLabels(["a", "b"])
        install_sorting(table)
        state = getattr(table, _STATE_ATTR)
        table.model().blockSignals(True)
        table.horizontalHeaderItem(0).setData(
            Qt.InitialSortOrderRole, Qt.AscendingOrder.value)
        table.model().blockSignals(False)
        assert install_sorting(table) is table
        assert [table.horizontalHeaderItem(c).data(Qt.InitialSortOrderRole)
                for c in range(2)] == [descending, descending]
        assert getattr(table, _STATE_ATTR) is state
        table.deleteLater()

    with subtests.test(msg="trees with and without a header item"):
        tree = QTreeWidget()
        tree.setColumnCount(2)
        install_sorting(tree)
        assert [tree.headerItem().data(c, Qt.InitialSortOrderRole)
                for c in range(2)] == [descending] * 2

        class _HeadlessTree(QTreeWidget):
            """A tree with nowhere to write the answer."""

            def headerItem(self):
                return None

        headless = _HeadlessTree()
        headless.setColumnCount(2)
        assert install_sorting(headless) is headless
        assert isinstance(getattr(headless, _STATE_ATTR, None),
                          type(getattr(tree, _STATE_ATTR)))
        assert headless.header().sortIndicatorSection() == -1
        tree.deleteLater()
        headless.deleteLater()

    with subtests.test(msg="model views are wrapped in the proxy once"):
        source = QStandardItemModel()
        for value in ["3", "1", "2"]:
            source.appendRow(QStandardItem(value))
        view = QTableView()
        view.setModel(source)
        install_sorting(view)
        assert isinstance(view.model(), SortableProxyModel)
        assert view.model().sourceModel() is source
        already = QTableView()
        proxy = SortableProxyModel(already)
        proxy.setSourceModel(source)
        already.setModel(proxy)
        install_sorting(already)
        assert already.model() is proxy
        view.deleteLater()
        already.deleteLater()

    with subtests.test(msg="a view with no model gets the header contract"):
        empty = QTableView()
        assert install_sorting(empty) is empty
        assert empty.model() is None
        assert empty.isSortingEnabled()
        assert empty.horizontalHeader().isSortIndicatorShown()
        assert empty.horizontalHeader().sortIndicatorSection() == -1
        empty.deleteLater()

    with subtests.test(msg="an older Qt with no clearable indicator"):
        class _OldHeader(QHeaderView):
            @property
            def setSortIndicatorClearable(self):
                raise AttributeError("Qt older than 6.1")

        table = QTableWidget(0, 2)
        table.setHorizontalHeader(_OldHeader(Qt.Horizontal, table))
        assert install_sorting(table) is table
        assert table.isSortingEnabled()
        assert table.horizontalHeader().isSortIndicatorShown()
        assert table.horizontalHeader().sortIndicatorSection() == -1
        assert table.horizontalHeader().isSortIndicatorClearable() is False
        modern = QTableWidget(0, 2)
        install_sorting(modern)
        assert modern.horizontalHeader().isSortIndicatorClearable() is True
        table.deleteLater()
        modern.deleteLater()

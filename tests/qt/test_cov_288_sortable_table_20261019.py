"""Sortable table helpers on tables and trees with gaps in them."""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QTableWidget, QTableWidgetItem, QTreeWidget

from spacr.qt.widgets import sortable_table as st

pytestmark = pytest.mark.qt


def test_text_set_before_the_sort_state_exists_is_stored_and_not_an_error(qapp):
    item = st.SortableTableItem.__new__(st.SortableTableItem)
    QTableWidgetItem.__init__(item)
    assert not hasattr(item, "_sort_serial")
    item.setData(Qt.DisplayRole, "late")
    assert item.text() == "late"


def test_retiring_a_table_with_empty_cells_keeps_the_cells_that_exist(qtbot):
    table = QTableWidget(2, 2)
    qtbot.addWidget(table)
    table.setItem(0, 0, st.table_item("kept"))
    assert table.item(1, 1) is None
    st._retire_table_cells(table)
    assert table.rowCount() == 0
    assert table.item(0, 0) is None
    spare = st._spare_items(table)
    assert len(spare) == 1 and spare[0].text() == ""


def test_retiring_a_table_with_headers_takes_only_the_headers_it_has(qtbot):
    table = QTableWidget(2, 2)
    qtbot.addWidget(table)
    table.setHorizontalHeaderItem(0, QTableWidgetItem("first"))
    table.setVerticalHeaderItem(1, QTableWidgetItem("row"))
    st._retire_table_cells(table, headers=True)
    assert table.horizontalHeaderItem(0) is None
    assert table.verticalHeaderItem(1) is None
    assert len(table.__dict__["_spacr_spare_headers"]) == 2


def test_children_of_an_item_outside_any_tree_grow_and_shrink(qtbot):
    parent = st.tree_item(["root"])
    assert parent.treeWidget() is None
    children = st._reuse_tree_items(parent, 3)
    assert len(children) == 3 and parent.childCount() == 3
    children = st._reuse_tree_items(parent, 1)
    assert len(children) == 1 and parent.childCount() == 1


def test_children_of_an_item_in_a_tree_are_wiped_before_reuse(qtbot):
    tree = QTreeWidget()
    qtbot.addWidget(tree)
    tree.setColumnCount(2)
    root = st.tree_item(["root", "x"])
    tree.addTopLevelItem(root)
    first = st._reuse_tree_items(root, 2)
    first[0].setText(0, "stale")
    again = st._reuse_tree_items(root, 2)
    assert [c.text(0) for c in again] == ["", ""]

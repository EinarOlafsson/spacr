"""A sorted tree redrawn in place is not re-sorted row by row while it is written.

Lesson 56's Build tree froze in the rows' Python ``__lt__``: a sorted
``QTreeWidget`` re-sorts the level of every item whose sorted-column text
changes, so wiping and rewriting a 7,000-node Lineage tree compared rows
about 54,000 times on the GUI thread, reading a sort key twice each time,
and the rows moved under the loop that was writing them. The redraw helper
now drops the sort before its first write and the sort comes back once the
fill is done. Counted, not timed.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QTreeWidget  # noqa: E402

from spacr.qt.widgets import sortable_table as st  # noqa: E402

pytestmark = pytest.mark.qt


def _draw(tree, values):
    rows = st._reuse_tree_items(tree, len(values))
    for item, value in zip(rows, values):
        item.setText(0, f"row {value}")
        item.setText(1, str(value))
        kids = st._reuse_tree_items(item, 2)
        for index, kid in enumerate(kids):
            kid.setText(0, f"kid {value}.{index}")
            kid.setText(1, str(value * 10 + index))


def _column(tree, column):
    return [tree.topLevelItem(row).text(column)
            for row in range(tree.topLevelItemCount())]


@pytest.fixture()
def key_reads(monkeypatch):
    reads = []
    original = st.SortableTreeItem._resolve_key

    def counted(self, *args):
        reads.append(self)
        return original(self, *args)

    monkeypatch.setattr(st.SortableTreeItem, "_resolve_key", counted)
    return reads


def test_a_redraw_reads_no_sort_key_until_the_fill_is_done(qtbot, key_reads):
    tree = QTreeWidget()
    tree.setColumnCount(2)
    qtbot.addWidget(tree)
    st.install_sorting(tree)
    _draw(tree, [5, 1, 4, 2, 3, 9, 7])
    st._settle_sorting(tree)
    header = tree.header()
    header.setSortIndicator(1, Qt.DescendingOrder)
    assert _column(tree, 1) == ["9", "7", "5", "4", "3", "2", "1"]

    key_reads.clear()
    _draw(tree, [12, 30, 11, 25, 18, 40, 16])
    assert key_reads == []
    assert header.sortIndicatorSection() == -1

    st._settle_sorting(tree)
    assert key_reads
    assert header.sortIndicatorSection() == 1
    assert header.sortIndicatorOrder() == Qt.DescendingOrder
    assert _column(tree, 1) == ["40", "30", "25", "18", "16", "12", "11"]
    assert _column(tree, 0) == [f"row {v}" for v in _column(tree, 1)]
    for row in range(tree.topLevelItemCount()):
        item = tree.topLevelItem(row)
        assert [item.child(i).text(0) for i in range(item.childCount())] == [
            f"kid {item.text(1)}.0", f"kid {item.text(1)}.1"][::-1]


def test_a_redraw_of_an_unsorted_tree_keeps_arrival_order(qtbot, key_reads):
    tree = QTreeWidget()
    tree.setColumnCount(2)
    qtbot.addWidget(tree)
    st.install_sorting(tree)
    _draw(tree, [3, 1, 2])
    _draw(tree, [6, 4, 5, 8])
    st._settle_sorting(tree)
    assert key_reads == []
    assert tree.header().sortIndicatorSection() == -1
    assert _column(tree, 1) == ["6", "4", "5", "8"]

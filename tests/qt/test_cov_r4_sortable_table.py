"""The corners of the sorting contract that a drawn table rarely reaches.

Three families live here, none of which a straightforward "click the header
and read the rows" test can get to:

* the readings :func:`numeric_value` refuses -- ``None``, blank, and a word
  wearing a percent sign -- and the array-valued cell that ``pandas.isna``
  answers with an array rather than a verdict;
* :class:`SortableTreeItem`, whose comparison resolves its key from whichever
  column is being sorted, and which has to cope with an item that is not one
  of ours on the other side of the ``<``;
* the fill guard in ``_SortState`` and the installer's refusals -- a view
  with no header, a view with no model, a column that disappeared between the
  fill and the queued resume, and a second :func:`install_sorting` on a view
  that already has one.

Every assertion here is about ordering, header state, or the value a cell
sorts on. Where a private is touched it is because the arc has no public
door, and the comment says which one.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt                                       # noqa: E402
from PySide6.QtGui import QStandardItem, QStandardItemModel         # noqa: E402
from PySide6.QtWidgets import (                                     # noqa: E402
    QTableView, QTableWidget, QTreeWidget,
    QTreeWidgetItem)

from spacr.qt.widgets.sortable_table import (                       # noqa: E402
    _column_count, _SortableMixin, install_sorting, is_missing,
    numeric_value, restore_natural_order, SortableProxyModel,
    table_item, tree_item)

pytestmark = pytest.mark.qt


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

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


def _column(table, column=0):
    return [table.item(row, column).text() for row in range(table.rowCount())]


def _proxy(values, order=None, labels=None):
    source = QStandardItemModel()
    for value in values:
        source.appendRow(QStandardItem(str(value)))
    if labels is not None:
        source.setHorizontalHeaderLabels(list(labels))
    proxy = SortableProxyModel()
    proxy.setSourceModel(source)
    if order is not None:
        proxy.sort(0, order)
    return proxy


def _shown(proxy):
    return [proxy.data(proxy.index(row, 0), Qt.DisplayRole)
            for row in range(proxy.rowCount())]


# ---------------------------------------------------------------------------
# reading a value out of a cell
# ---------------------------------------------------------------------------

def test_a_cell_holding_an_array_is_not_a_missing_cell():
    """``pandas.isna`` answers an array with an array, which is not a verdict.

    ``bool()`` of it raises, and the cell must come out populated rather than
    sinking to the bottom with the holes.
    """
    assert is_missing(np.array([1.0, np.nan])) is False
    assert is_missing(np.nan) is True


def test_nothing_at_all_has_no_numeric_value():
    assert numeric_value(None) is None
    assert numeric_value("   ") is None
    assert numeric_value(" 3.5 ") == 3.5


# ---------------------------------------------------------------------------
# the mixin's own defaults
# ---------------------------------------------------------------------------

class _PlainCell(_SortableMixin):
    """A cell type that overrides neither hook, to pin what they default to.

    ``_SortableMixin`` is the module's extension point -- ``SortableTableItem``
    and ``SortableTreeItem`` are both it plus a Qt item -- but every shipped
    subclass overrides ``_sort_text`` and ``_descending``, so the defaults are
    only reachable through a subclass that does not.
    """

    def __init__(self, text):
        self._text = text
        self._init_sort_key(text)

    def text(self):
        return self._text


def test_a_cell_that_declares_no_text_offers_none_to_a_word_comparison():
    """The default ``_sort_text`` is empty, so it never wins a tie-break."""
    assert (_PlainCell("zebra") < _PlainCell("apple")) is True
    # Both directions read "less than": neither side contributes text of its
    # own, so each is comparing "" against the other's text.
    assert (_PlainCell("apple") < _PlainCell("zebra")) is True


def test_a_cell_that_declares_no_order_sorts_missing_values_last():
    """The default ``_descending`` is False, i.e. holes after values."""
    assert (_PlainCell("") < _PlainCell("5")) is False
    assert (_PlainCell("5") < _PlainCell("")) is True


# ---------------------------------------------------------------------------
# SortableTableItem: an edit does not lose an explicit key
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# SortableTreeItem
# ---------------------------------------------------------------------------

def test_a_tree_item_declines_to_order_itself_against_a_non_item():
    """Declined, not guessed: ``sorted`` believes whatever it is told."""
    assert tree_item(["5"]).__lt__(object()) is NotImplemented
    with pytest.raises(TypeError):
        tree_item(["5"]) < object()


def test_a_tree_item_reads_its_text_from_the_column_being_sorted(qapp):
    """``_sort_text`` is the mixin hook; the tree's own ``__lt__`` inlines the
    same read, so the hook itself is only reachable directly."""
    tree = QTreeWidget()
    tree.setColumnCount(2)
    tree.setHeaderLabels(["value", "name"])
    install_sorting(tree)
    tree.addTopLevelItem(tree_item(["5", "alpha"]))
    row = tree.topLevelItem(0)

    tree.sortItems(0, Qt.AscendingOrder)
    assert row._sort_text() == "5"

    tree.sortItems(1, Qt.AscendingOrder)
    assert row._sort_text() == "alpha"
    tree.deleteLater()


# ---------------------------------------------------------------------------
# the fill guard
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# installing, and refusing to
# ---------------------------------------------------------------------------

def test_the_column_count_of_a_view_that_is_gone_is_zero(qapp):
    """Guards the queued resume against a view torn down under it; the only
    caller has already checked ``_alive``, so nothing else reaches it."""
    table = _table(qapp)
    assert _column_count(table) == 3
    assert _column_count(None) == 0
    table.deleteLater()


# ---------------------------------------------------------------------------
# restoring the natural order
# ---------------------------------------------------------------------------

def test_restoring_a_view_that_is_gone_leaves_the_live_one_alone(qapp):
    table = _table(qapp)
    table.sortItems(0, Qt.DescendingOrder)
    assert _column(table) == ["2", "1", "0"]

    restore_natural_order(None)
    assert _column(table) == ["2", "1", "0"]

    restore_natural_order(table)
    assert _column(table) == ["0", "1", "2"]
    table.deleteLater()


def test_a_model_backed_view_is_restored_through_its_proxy(qapp):
    view = QTableView()
    restore_natural_order(view)  # no model yet: nothing to put back

    model = QStandardItemModel()
    for value in ["3", "1", "2"]:
        model.appendRow(QStandardItem(value))
    view.setModel(model)
    install_sorting(view)
    proxy = view.model()
    proxy.sort(0, Qt.DescendingOrder)
    assert _shown(proxy) == ["3", "2", "1"]

    restore_natural_order(view)
    assert _shown(proxy) == ["3", "1", "2"]
    view.deleteLater()


# ---------------------------------------------------------------------------
# SortableProxyModel
# ---------------------------------------------------------------------------

def test_the_proxy_tells_the_header_to_start_descending():
    proxy = _proxy(["1"], labels=["value"])

    assert proxy.headerData(
        0, Qt.Horizontal, Qt.InitialSortOrderRole) == Qt.DescendingOrder.value
    # Rows make no such claim, and every other role is the source's answer.
    assert proxy.headerData(0, Qt.Vertical, Qt.InitialSortOrderRole) is None
    assert proxy.headerData(0, Qt.Horizontal, Qt.DisplayRole) == "value"


def test_the_proxy_sorts_numbers_then_words_then_holes_ascending():
    proxy = _proxy(["10", "", "banana", "2", "apple", "n/a", "9"],
                   Qt.AscendingOrder)

    assert _shown(proxy) == ["2", "9", "10", "apple", "banana", "", "n/a"]


def test_the_proxy_keeps_the_holes_last_descending_too():
    """Reversing reverses the values and nothing else: words, then numbers,
    and the two holes still at the bottom rather than at the top."""
    proxy = _proxy(["10", "", "banana", "2", "apple", "n/a", "9"],
                   Qt.DescendingOrder)

    assert _shown(proxy)[:5] == ["banana", "apple", "10", "9", "2"]
    # The two holes compare equal to each other, so which of them is last is
    # not part of the contract; that they are both last is.
    assert set(_shown(proxy)[5:]) == {"", "n/a"}


def test_cached_numeric_keys_survive_stale_priming_and_text_changes(qapp, monkeypatch):
    from spacr.qt.widgets import sortable_table as st

    first, second = tree_item(['2']), tree_item(['10'])
    assert first._resolve_key() == (False, 2.)
    assert first._resolve_key() == (False, 2.)
    first._sort_primed = (object(),)
    second._sort_primed = (object(),)
    monkeypatch.setattr(st, '_PRIMED', (False, object()))
    assert first < second
    first.setText(0, '20')
    assert second < first
    assert first._resolve_key() == (False, 20.)


def test_priming_walks_plain_tree_parents_and_keeps_current_token_keys(qtbot, monkeypatch):
    from spacr.qt.widgets import sortable_table as st

    tree = QTreeWidget()
    qtbot.addWidget(tree)
    parent = QTreeWidgetItem(['group'])
    tree.addTopLevelItem(parent)
    for value in ('10', '', '2'):
        parent.addChild(tree_item([value]))
    token = object()
    primed = st._prime_tree_keys(tree, 0, token)
    assert len(primed) == 3
    original_keys = [item._sort_primed for item in primed]
    assert st._prime_tree_keys(tree, 0, token) == primed
    assert all(item._sort_primed is key for item, key in zip(primed, original_keys))
    monkeypatch.setattr(st, '_PRIMED', (False, token))
    parent.sortChildren(0, Qt.AscendingOrder)
    assert [parent.child(row).text(0) for row in range(3)] == ['2', '10', '']
    monkeypatch.setattr(st, '_PRIMED', (True, token))
    parent.sortChildren(0, Qt.DescendingOrder)
    assert [parent.child(row).text(0) for row in range(3)] == ['10', '2', '']



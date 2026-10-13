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
    QHeaderView, QListView, QTableView, QTableWidget, QTreeWidget,
    QTreeWidgetItem)

from spacr.qt.widgets.sortable_table import (                       # noqa: E402
    _column_count, _SortableMixin, _STATE_ATTR, install_sorting, is_missing,
    numeric_value, restore_natural_order, SORT_KEY_ROLE, SortableProxyModel,
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


def _tree_column(tree, column=0):
    return [tree.topLevelItem(row).text(column)
            for row in range(tree.topLevelItemCount())]


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


def test_a_percent_sign_does_not_make_a_word_into_a_number():
    """"42%" is forty-two; "abc%" is still a word and sorts as one."""
    assert numeric_value("42%") == 42.0
    assert numeric_value("abc%") is None
    # ... and so orders lexically against another word, not against numbers.
    assert (table_item("abc%") < table_item("zzz%")) is True


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

def test_editing_the_text_keeps_the_explicit_sort_key():
    """A duration shown as "2 h 5 m" sorts on the seconds it was given.

    Re-typing the display text must not throw that away for whatever the new
    text happens to parse as.
    """
    item = table_item("2 h 5 m", key=125)
    item.setText("3 h")

    assert item.text() == "3 h"
    assert (table_item(100) < item) is True
    assert (item < table_item(130)) is True
    # Not the 3 the new text starts with.
    assert (item < table_item(4)) is False


# ---------------------------------------------------------------------------
# SortableTreeItem
# ---------------------------------------------------------------------------

def test_a_tree_item_with_no_tree_sorts_on_its_first_column():
    """A row built but not yet added still compares, on column 0."""
    assert (tree_item(["apple"]) < tree_item(["banana"])) is True
    assert (tree_item(["banana"]) < tree_item(["apple"])) is False


def test_a_tree_item_with_no_tree_still_puts_missing_values_last():
    """With no header to ask, the order is the ascending one: holes last."""
    assert (tree_item([""]) < tree_item(["5"])) is False
    assert (tree_item(["5"]) < tree_item([""])) is True


def test_two_missing_tree_cells_are_neither_before_nor_after_each_other():
    blank, dash = tree_item([""]), tree_item(["—"])

    assert (blank < dash) is False
    assert (dash < blank) is False


def test_a_number_sorts_before_a_word_in_a_mixed_tree_column():
    assert (tree_item(["5"]) < tree_item(["apple"])) is True
    assert (tree_item(["apple"]) < tree_item(["5"])) is False


def test_an_explicit_key_beats_the_text_of_a_tree_cell():
    item = tree_item(["2 h 5 m"])
    item.setData(0, SORT_KEY_ROLE, 125.0)

    assert (item < tree_item(["30"])) is False
    assert (tree_item(["30"]) < item) is True


def test_a_tree_item_compares_against_a_plain_qt_item():
    """A tree can hold rows nobody routed through :func:`tree_item`."""
    assert (tree_item(["5"]) < QTreeWidgetItem(["9"])) is True
    assert (tree_item(["9"]) < QTreeWidgetItem(["5"])) is False


def test_a_tree_item_declines_to_order_itself_against_a_non_item():
    """Declined, not guessed: ``sorted`` believes whatever it is told."""
    assert tree_item(["5"]).__lt__(object()) is NotImplemented
    with pytest.raises(TypeError):
        tree_item(["5"]) < object()


def test_a_tree_puts_missing_values_last_whichever_way_it_is_sorted(qapp):
    """The header is present here, so ``_descending`` reads a real order."""
    tree = QTreeWidget()
    tree.setColumnCount(2)
    tree.setHeaderLabels(["value", "name"])
    install_sorting(tree)
    for index, value in enumerate(["5", "", "1", "—"]):
        tree.addTopLevelItem(tree_item([value, f"row{index}"]))

    tree.sortItems(0, Qt.AscendingOrder)
    assert _tree_column(tree) == ["1", "5", "", "—"]

    tree.sortItems(0, Qt.DescendingOrder)
    assert _tree_column(tree) == ["5", "1", "", "—"]
    tree.deleteLater()


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

def test_the_fill_guard_stands_down_while_the_sort_is_being_put_back(qapp):
    """``_resuming`` is set only inside ``resume_after_fill``'s own restore.

    It is the re-entrancy guard: a row signal that arrives while the sort is
    being re-applied must not capture the half-restored state as the thing to
    restore. Set here directly because the guard is, by construction, only
    true inside a call that is not itself emitting row signals.
    """
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


def test_a_late_resume_does_not_undo_a_sort_the_user_has_cleared(qapp):
    """The third click clears the indicator; a stray callback leaves it clear."""
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


def test_a_column_that_went_away_during_a_fill_is_not_sorted_again(qapp):
    """The resume is queued; the table can be narrower by the time it runs."""
    keeps = _table(qapp)
    keeps.horizontalHeader().setSortIndicator(2, Qt.DescendingOrder)
    keeps.insertRow(0)
    assert keeps.horizontalHeader().sortIndicatorSection() == -1
    qapp.processEvents()
    assert keeps.horizontalHeader().sortIndicatorSection() == 2

    loses = _table(qapp)
    loses.horizontalHeader().setSortIndicator(2, Qt.DescendingOrder)
    loses.insertRow(0)
    loses.setColumnCount(1)
    qapp.processEvents()
    assert loses.horizontalHeader().sortIndicatorSection() == -1
    keeps.deleteLater()
    loses.deleteLater()


# ---------------------------------------------------------------------------
# installing, and refusing to
# ---------------------------------------------------------------------------

def test_a_view_with_no_header_is_handed_back_unchanged(qapp):
    """A list has no columns to click, and a gone view has nothing at all."""
    assert install_sorting(None) is None

    listing = QListView()
    assert install_sorting(listing) is listing
    assert getattr(listing, _STATE_ATTR, None) is None

    table = QTableWidget(0, 1)
    assert getattr(install_sorting(table), _STATE_ATTR, None) is not None
    table.deleteLater()


def test_the_column_count_of_a_view_that_is_gone_is_zero(qapp):
    """Guards the queued resume against a view torn down under it; the only
    caller has already checked ``_alive``, so nothing else reaches it."""
    table = _table(qapp)
    assert _column_count(table) == 3
    assert _column_count(None) == 0
    table.deleteLater()


def test_a_second_install_restates_the_header_contract(qapp):
    table = QTableWidget(0, 2)
    table.setHorizontalHeaderLabels(["a", "b"])
    install_sorting(table)
    state = getattr(table, _STATE_ATTR)

    # Blocked, or the model's own headerDataChanged would restate it for us
    # before the second install could be asked to.
    table.model().blockSignals(True)
    table.horizontalHeaderItem(0).setData(
        Qt.InitialSortOrderRole, Qt.AscendingOrder.value)
    table.model().blockSignals(False)
    assert table.horizontalHeaderItem(0).data(
        Qt.InitialSortOrderRole) == Qt.AscendingOrder.value

    assert install_sorting(table) is table
    assert table.horizontalHeaderItem(0).data(
        Qt.InitialSortOrderRole) == Qt.DescendingOrder.value
    # The same state object: a second one would wire every model signal twice.
    assert getattr(table, _STATE_ATTR) is state
    table.deleteLater()


def test_a_tree_that_reports_no_header_item_still_installs(qapp):
    """``headerItem()`` is where the descending-first answer is written."""
    tree = QTreeWidget()
    tree.setColumnCount(2)
    install_sorting(tree)
    assert [tree.headerItem().data(column, Qt.InitialSortOrderRole)
            for column in range(2)] == [Qt.DescendingOrder.value] * 2

    class _HeadlessTree(QTreeWidget):
        """A tree with nowhere to write the answer."""

        def headerItem(self):
            return None

    headless = _HeadlessTree()
    headless.setColumnCount(2)
    assert install_sorting(headless) is headless
    assert getattr(headless, _STATE_ATTR, None) is not None
    assert headless.header().sortIndicatorSection() == -1
    tree.deleteLater()
    headless.deleteLater()


def test_an_older_qt_with_no_clearable_indicator_still_installs(qapp):
    """``setSortIndicatorClearable`` arrived in Qt 6.1; the rest works without."""

    class _OldHeader(QHeaderView):
        @property
        def setSortIndicatorClearable(self):
            raise AttributeError("Qt older than 6.1")

    table = QTableWidget(0, 2)
    header = _OldHeader(Qt.Horizontal, table)
    table.setHorizontalHeader(header)

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


def test_a_model_backed_view_is_wrapped_in_the_proxy_exactly_once(qapp):
    view = QTableView()
    source = QStandardItemModel()
    for value in ["3", "1", "2"]:
        source.appendRow(QStandardItem(value))
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


def test_a_view_with_no_model_still_gets_the_header_contract(qapp):
    empty = QTableView()
    assert install_sorting(empty) is empty
    assert empty.model() is None
    assert empty.isSortingEnabled()
    assert empty.horizontalHeader().isSortIndicatorShown()
    assert empty.horizontalHeader().sortIndicatorSection() == -1

    filled = QTableView()
    model = QStandardItemModel()
    model.appendRow(QStandardItem("1"))
    filled.setModel(model)
    install_sorting(filled)
    assert isinstance(filled.model(), SortableProxyModel)
    empty.deleteLater()
    filled.deleteLater()


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

"""``SortableProxyModel.lessThan``, and the one rule the whole module exists for.

Missing values follow populated ones **in both directions**. That is not how a
comparator normally behaves -- reversing the order usually reverses
everything -- and it is the point: a column of p-values with four blanks in it
should show the four blanks at the bottom whichever way the user clicked,
because a blank is not a small number and it is not a large one either.

The model-backed proxy is the half of this module a view gets when it is
driven by a QAbstractItemModel rather than by QTableWidgetItems, and none of
its comparison had ever run. The item-backed half is
:class:`SortableTreeItem.__lt__`, whose peer-type paths are covered here too.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtCore import Qt                                       # noqa: E402
from PySide6.QtGui import QStandardItem, QStandardItemModel         # noqa: E402

from spacr.qt.widgets.sortable_table import (                       # noqa: E402
    SortableProxyModel, SortableTreeItem, sort_key_of, tree_item)

pytestmark = pytest.mark.qt


def _proxy(values, order=Qt.AscendingOrder):
    """A proxy over one column of ``values``, sorted in ``order``."""
    model = QStandardItemModel()
    for value in values:
        model.appendRow(QStandardItem(str(value)))
    proxy = SortableProxyModel()
    proxy.setSourceModel(model)
    proxy.sort(0, order)
    return proxy


def _shown(proxy):
    return [proxy.data(proxy.index(row, 0), Qt.DisplayRole)
            for row in range(proxy.rowCount())]


# ---------------------------------------------------------------------------
# the rule
# ---------------------------------------------------------------------------

def test_blanks_sink_to_the_bottom_ascending():
    proxy = _proxy(["3", "", "1", "", "2"], Qt.AscendingOrder)

    assert _shown(proxy)[:3] == ["1", "2", "3"]
    assert _shown(proxy)[3:] == ["", ""]


def test_blanks_sink_to_the_bottom_descending_too():
    """The same end of the table, which is the whole contract.

    A comparator that simply reversed would put the blanks on top, and a
    reader scanning for the largest coefficient would find two empty rows
    above it.
    """
    proxy = _proxy(["3", "", "1", "", "2"], Qt.DescendingOrder)

    assert _shown(proxy)[:3] == ["3", "2", "1"]
    assert _shown(proxy)[3:] == ["", ""]


def test_two_missing_values_do_not_reorder_each_other():
    """``lessThan`` answers False both ways, so their input order stands.

    Returning True either way makes the comparison inconsistent, and Qt's sort
    is free to produce a different arrangement on each run -- a table that
    reshuffles its blank rows every time the user clicks.
    """
    model = QStandardItemModel()
    for text in ("", "nan"):
        model.appendRow(QStandardItem(text))
    proxy = SortableProxyModel()
    proxy.setSourceModel(model)

    left = proxy.sourceModel().index(0, 0)
    right = proxy.sourceModel().index(1, 0)

    assert proxy.lessThan(left, right) is False
    assert proxy.lessThan(right, left) is False


# ---------------------------------------------------------------------------
# what "sorts numerically" means
# ---------------------------------------------------------------------------

def test_numbers_sort_as_numbers_and_not_as_text():
    """The failure this replaces: "10" before "9" in a string sort."""
    proxy = _proxy(["9", "10", "100", "2"], Qt.AscendingOrder)

    assert _shown(proxy) == ["2", "9", "10", "100"]


def test_scientific_notation_sorts_with_the_plain_numbers():
    """p-values arrive formatted, and 1e-9 is smaller than 0.04."""
    proxy = _proxy(["0.04", "1e-9", "0.5", "3.2e-3"], Qt.AscendingOrder)

    assert _shown(proxy) == ["1e-9", "3.2e-3", "0.04", "0.5"]


def test_text_that_is_not_a_number_sorts_case_insensitively():
    """Gene names, and a case-sensitive sort puts every lowercase name last."""
    proxy = _proxy(["beta", "Alpha", "gamma", "Delta"], Qt.AscendingOrder)

    assert _shown(proxy) == ["Alpha", "beta", "Delta", "gamma"]


def test_a_number_sorts_before_text_that_is_not_a_number():
    """A mixed column is a column with a problem, and the numbers are the
    part the reader can still use, so they come first."""
    proxy = _proxy(["7", "n/a-ish", "2"], Qt.AscendingOrder)

    shown = _shown(proxy)
    assert shown[:2] == ["2", "7"]
    assert shown[2] == "n/a-ish"


# ---------------------------------------------------------------------------
# the item-backed comparison's peer types
# ---------------------------------------------------------------------------

def test_a_tree_item_compares_against_a_plain_item_through_its_text(qtbot):
    """Qt mixes item classes in one tree more often than it looks.

    A plain QTreeWidgetItem beside a SortableTreeItem still has ``text``, so
    the comparison reads its column rather than refusing -- otherwise one
    unsorted row makes the whole tree fall back to Qt's string ordering.
    """
    from PySide6.QtWidgets import QTreeWidgetItem

    mine = tree_item("2")
    theirs = QTreeWidgetItem(["10"])

    assert (mine < theirs) is True


def test_a_peer_with_no_text_at_all_is_declined_rather_than_guessed_at():
    """``NotImplemented`` lets Python try the reflected operation.

    Returning False instead would silently claim an ordering against an object
    this class knows nothing about.
    """
    mine = tree_item("2")

    assert mine.__lt__(object()) is NotImplemented


def test_an_item_with_no_tree_is_not_in_descending_order(qtbot):
    """``_descending`` reads the header, and a detached item has none.

    Items are built before they are inserted -- ``tree_item(...)`` then
    ``addTopLevelItem`` -- so a comparison during construction must not
    dereference a view that is not there yet.
    """
    item = tree_item("")
    assert item.treeWidget() is None

    assert item._descending() is False


# ---------------------------------------------------------------------------
# the key itself
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", ["", "  ", None, float("nan")])
def test_the_values_that_count_as_missing(value):
    missing, number = sort_key_of(value)

    assert missing is True
    assert number is None


@pytest.mark.parametrize("value,expected", [("1", 1.0), ("-2.5", -2.5),
                                            ("1e3", 1000.0), ("  4 ", 4.0)])
def test_the_values_that_read_as_numbers(value, expected):
    missing, number = sort_key_of(value)

    assert missing is False
    assert number == pytest.approx(expected)

"""One sorting contract for every table and tree in the application.

Thirty-odd screens build a table, and a sort behaviour written at each site
drifts: half of them could not sort at all, the ones that could sorted
ascending first because that is Qt's default, and the numeric columns among
them sorted "10" before "9". This module holds the behaviour once so that
every table answers a header click the same way.

THE CONTRACT
------------

* the first click on a column sorts it **descending** -- a coefficient table
  and a hit list are read from the top, and the top is where the largest
  effect and the smallest q-value belong;
* the second click sorts it **ascending**;
* the third click clears the sort and puts the rows back in the order the
  table was built in;
* the fourth click starts the cycle again.

Numbers sort as numbers, whatever they were formatted as: ``0.5`` sorts
between ``0.1`` and ``2``, and ``1.2e-07`` is a number and not a word.
Missing values -- blank, ``NaN``, ``pandas.NA``, ``NaT`` -- sort to the
BOTTOM in every direction, so a column of results never opens on its holes.

USING IT
--------

Build the cells with :func:`table_item` or :func:`tree_item` and hand the
view to :func:`install_sorting`::

    self.table = QTableWidget(0, 3)
    install_sorting(self.table)
    ...
    self.table.setItem(row, 0, table_item(coefficient))

:func:`install_sorting` also keeps a repopulation from scrambling the rows.
Qt re-sorts a sorted table on every ``setItem`` into the sorted column, which
moves rows out from under the loop that is filling them; the helper drops the
sort for the duration of a fill and puts it back when the fill settles.
"""
from __future__ import annotations

import re
from typing import Optional

import shiboken6
from PySide6.QtCore import QObject, QSortFilterProxyModel, Qt, QTimer
from PySide6.QtWidgets import (QTableView, QTableWidget, QTableWidgetItem,
                               QTreeView, QTreeWidget, QTreeWidgetItem)

__all__ = [
    "SORT_KEY_ROLE",
    "SortableTableItem",
    "SortableTreeItem",
    "install_sorting",
    "is_missing",
    "numeric_value",
    "sort_key_of",
    "sorts_as_missing",
    "SortableProxyModel",
    "table_item",
    "tree_item",
]

#: A cell may carry its own sort key here when the text is not the thing to
#: sort on -- a duration shown as "2 h 5 m", a date shown as "yesterday".
SORT_KEY_ROLE = Qt.UserRole + 90

# The natural order is the order the cells were created in. A serial handed
# out at construction records it without a second pass over the table, and a
# table is filled row by row, so serials increase down a column whichever way
# the loop that fills it is nested.
_SERIAL = 0

# Set only while this module is putting a table back into its natural order.
# Sorting is synchronous and on the GUI thread, so a module-level flag is
# read by exactly the comparisons of the sort that set it.
_RESTORING = False

_STATE_ATTR = "_spacr_sort_state"

#: A comma between digits is a thousands separator and nothing else.
_THOUSANDS = re.compile(r"(?<=\d),(?=\d\d\d(\D|$))")


def is_missing(value) -> bool:
    """True for ``None``, ``NaN``, pandas' ``NA`` and ``NaT``.

    ``pandas.isna`` rather than ``value != value``: ``pd.NA != pd.NA`` is
    ``pd.NA``, not ``True``, and ``bool(pd.NA)`` raises -- so the naive test
    reports the one sentinel it was written for as present.
    """
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        import pandas as _pd

        return bool(_pd.isna(value))
    except (ImportError, TypeError, ValueError):
        # isna on an array-like returns an array; a cell holding one is not
        # missing.
        return False


def numeric_value(text) -> Optional[float]:
    """The number a cell stands for, or ``None`` when it is a word.

    Deliberately strict. A column of gene names must not sort as numbers, so
    a leading letter disqualifies the cell -- "TP53" is a gene and not 53.
    What does count as a number is what the screens actually print: plain
    digits, scientific notation, a thousands separator between digits, a
    trailing percent, and a trailing unit separated by a space ("3.2 s",
    "12 MB"). A cell whose text is none of those can still sort numerically
    by carrying its own key -- see :data:`SORT_KEY_ROLE`.
    """
    if text is None:
        return None
    if isinstance(text, bool):
        return None
    if isinstance(text, (int, float)):
        number = float(text)
        return None if number != number else number
    stripped = str(text).strip()
    if not stripped:
        return None
    for candidate in _numeric_candidates(stripped):
        try:
            number = float(candidate)
        except (TypeError, ValueError):
            continue
        return None if number != number else number
    return None


def _numeric_candidates(text: str):
    """The readings of ``text`` that could be a number, best first."""
    yield text
    # 1,234,567 -- a separator only between digits, never a decimal comma,
    # which would turn "1,5" into fifteen thousand.
    without_separators = _THOUSANDS.sub("", text)
    if without_separators != text:
        yield without_separators
        text = without_separators
    if text.endswith("%"):
        yield text[:-1].strip()
        return
    # "3.2 s", "12 MB", "0.4 µm" -- the unit is a word after a space, not a
    # letter glued to digits, which is what an identifier looks like.
    head, sep, tail = text.rpartition(" ")
    if sep and head and tail and not tail[0].isdigit():
        yield head.strip().rstrip("%")


#: How a screen writes "there is nothing here". A cell reading any of these
#: sorts with the holes rather than between the numbers, and keeps the text
#: it was given: an em dash is what the column picker draws for a column with
#: no declaration, and it should still say so while sorting to the bottom.
_PLACEHOLDERS = frozenset(
    {"nan", "na", "n/a", "none", "null", "<na>", "nat", "-", "\u2014",
     "\u2013", "?"})


def sorts_as_missing(value) -> bool:
    """True when ``value`` is a hole, however the screen chose to write it."""
    if is_missing(value):
        return True
    return isinstance(value, str) and value.strip().casefold() in _PLACEHOLDERS


def sort_key_of(value):
    """``(missing, number)`` for ``value``: the pair every comparison uses."""
    if sorts_as_missing(value):
        return True, None
    return False, numeric_value(value)


class _SortableMixin:
    """The comparison. See the module docstring for the contract."""

    def _init_sort_key(self, value, key=None) -> None:
        global _SERIAL
        _SERIAL += 1
        self._sort_serial = _SERIAL
        if key is not None:
            self._sort_missing = False
            self._sort_number = float(key)
        else:
            self._sort_missing, self._sort_number = sort_key_of(value)

    # -- the numbers a comparison needs -----------------------------------

    def _resolve_key(self):
        """The key this cell sorts on. Settled once, refreshed on an edit."""
        return self._sort_missing, self._sort_number

    def _sort_text(self) -> str:
        return ""

    def _descending(self) -> bool:
        return False

    def __lt__(self, other):
        if _RESTORING:
            return getattr(self, "_sort_serial", 0) < getattr(
                other, "_sort_serial", 0)
        mine_missing, mine = self._resolve_key()
        if isinstance(other, _SortableMixin):
            theirs_missing, theirs = other._resolve_key()
        else:
            peer = getattr(other, "text", None)
            if not callable(peer):
                # Declined, not guessed. ``sorted`` believes every answer it
                # is given, so a cell that guessed would produce an order
                # nobody could account for; NotImplemented lets Python raise.
                return NotImplemented
            theirs_missing, theirs = sort_key_of(peer())
        if mine_missing or theirs_missing:
            if mine_missing and theirs_missing:
                return False
            # Missing goes last in BOTH directions, so it has to be the
            # largest when Qt is ordering by "<" and the smallest when Qt is
            # ordering by ">".
            return mine_missing if self._descending() else theirs_missing
        if mine is not None and theirs is not None:
            return mine < theirs
        if mine is None and theirs is None:
            return self._sort_text().casefold() < str(other.text()).casefold()
        # A number and a word in one column: numbers first, always the same
        # way round, so the order never depends on which cell Qt asked.
        return mine is not None


class SortableTableItem(_SortableMixin, QTableWidgetItem):
    """A table cell that sorts by what it means, not by how it reads.

    :param value: what the cell holds. Missing values render as an empty
        cell -- ``str(float('nan'))`` is the word "nan", which is how a hole
        in a frame ends up printed as data.
    :param key: an explicit sort key, for a cell whose text is not sortable
        on its own.
    """

    def __init__(self, value="", key=None):
        QTableWidgetItem.__init__(
            self, "" if is_missing(value) else str(value))
        self._init_sort_key(value, key=key)
        if key is not None:
            self.setData(SORT_KEY_ROLE, float(key))

    def setData(self, role, value):  # noqa: N802 - Qt spelling
        """Re-read the sort key when the cell's text changes.

        An editable column is edited: the user types 20 over 3, Qt writes it
        through here, and a key settled at construction would keep sorting
        the cell as a 3. Only the two roles that are the text count, so an
        explicit key or a colour costs nothing.
        """
        QTableWidgetItem.setData(self, role, value)
        if role in (Qt.DisplayRole, Qt.EditRole) and hasattr(
                self, "_sort_serial"):
            override = self.data(SORT_KEY_ROLE)
            serial = self._sort_serial
            if override is None:
                self._init_sort_key(self.text())
            else:
                self._init_sort_key(None, key=override)
            # The row keeps the place it was built in: an edit is not a new
            # row, and the natural order must not shuffle under one.
            self._sort_serial = serial

    def _sort_text(self) -> str:
        return self.text() or ""

    def _descending(self) -> bool:
        view = self.tableWidget()
        if view is None:
            return False
        return view.horizontalHeader().sortIndicatorOrder() == Qt.DescendingOrder


class SortableTreeItem(_SortableMixin, QTreeWidgetItem):
    """A tree row that sorts by what its cells mean. See
    :class:`SortableTableItem`; the difference is that a tree row carries
    every column, so the key is resolved for the column being sorted on."""

    def __init__(self, *args, **kwargs):
        QTreeWidgetItem.__init__(self, *args, **kwargs)
        global _SERIAL
        _SERIAL += 1
        self._sort_serial = _SERIAL

    def _sort_column(self) -> int:
        view = self.treeWidget()
        if view is None:
            return 0
        column = view.sortColumn()
        return 0 if column < 0 else column

    def _resolve_key(self):
        """The key for the column being sorted on, read once per text.

        A tree row carries every column, so the key cannot be settled when
        the row is built. Cached against the text it was read from, because
        a sort asks for it O(n log n) times and the text does not move.
        """
        column = self._sort_column()
        text = self.text(column)
        cached = getattr(self, "_sort_cache", None)
        if cached is not None and cached[0] == column and cached[1] == text:
            return cached[2], cached[3]
        override = self.data(column, SORT_KEY_ROLE)
        if override is not None:
            missing, number = False, float(override)
        else:
            missing, number = sort_key_of(text)
        self._sort_cache = (column, text, missing, number)
        return missing, number

    def _sort_text(self) -> str:
        return self.text(self._sort_column()) or ""

    def _descending(self) -> bool:
        view = self.treeWidget()
        if view is None:
            return False
        return view.header().sortIndicatorOrder() == Qt.DescendingOrder

    def __lt__(self, other):
        if _RESTORING:
            return getattr(self, "_sort_serial", 0) < getattr(
                other, "_sort_serial", 0)
        column = self._sort_column()
        mine_missing, mine = self._resolve_key()
        if isinstance(other, SortableTreeItem):
            theirs_missing, theirs = other._resolve_key()
        else:
            peer = getattr(other, "text", None)
            if not callable(peer):
                return NotImplemented
            theirs_missing, theirs = sort_key_of(peer(column))
        if mine_missing or theirs_missing:
            if mine_missing and theirs_missing:
                return False
            return mine_missing if self._descending() else theirs_missing
        if mine is not None and theirs is not None:
            return mine < theirs
        if mine is None and theirs is None:
            return (self.text(column) or "").casefold() < (
                other.text(column) or "").casefold()
        return mine is not None


def table_item(value="", key=None) -> SortableTableItem:
    """A :class:`SortableTableItem`. The one way to build a table cell."""
    return SortableTableItem(value, key=key)


def tree_item(*args, **kwargs) -> SortableTreeItem:
    """A :class:`SortableTreeItem`, taking what ``QTreeWidgetItem`` takes."""
    return SortableTreeItem(*args, **kwargs)


class _SortState(QObject):
    """Per-view bookkeeping: the fill guard and the natural-order restore."""

    def __init__(self, view):
        super().__init__(view)
        self._view = view
        self._suspended = None
        self._resuming = False
        self._stamping = False

    # -- keeping a fill from scrambling the rows ---------------------------

    def suspend_for_fill(self, *args) -> None:
        """Drop the sort while rows are being written, put it back after.

        Qt re-sorts a sorted table on every ``setItem`` into the sorted
        column: the row moves, and the loop that is filling it writes the
        rest of the row into whatever landed at that index. Clearing the sort
        indicator stops that -- the model only re-sorts on a write to the
        section the indicator names.
        """
        header = _header(self._view)
        if header is None or self._resuming:
            return
        if self._suspended is None:
            self._suspended = (header.sortIndicatorSection(),
                               header.sortIndicatorOrder())
            if self._suspended[0] >= 0:
                header.setSortIndicator(-1, Qt.AscendingOrder)
        QTimer.singleShot(0, self.resume_after_fill)

    def resume_after_fill(self) -> None:
        """Re-apply the sort the user had chosen, once the fill has settled."""
        if self._suspended is None or not _alive(self._view):
            return
        section, order = self._suspended
        self._suspended = None
        header = _header(self._view)
        if header is None or section < 0:
            return
        if section >= _column_count(self._view):
            return
        self._resuming = True
        try:
            header.setSortIndicator(section, order)
        finally:
            self._resuming = False

    # -- the header's own contract ----------------------------------------

    def stamp_initial_order(self, *args) -> None:
        """Tell the header that a fresh column starts descending.

        ``QHeaderView`` asks the model for ``Qt.InitialSortOrderRole`` when
        the click lands on a column that is not the sorted one, and uses
        ascending when nothing answers. Answering is the whole of
        descending-first: Qt does the rest, including the flip on the second
        click.
        """
        if self._stamping or not _alive(self._view):
            return
        self._stamping = True
        try:
            _stamp_initial_order(self._view)
        finally:
            self._stamping = False

    def on_indicator_changed(self, section: int, order) -> None:
        """A cleared indicator means "put it back the way it was"."""
        if section >= 0 or self._resuming or not _alive(self._view):
            return
        restore_natural_order(self._view)


def _alive(obj) -> bool:
    """True while ``obj``'s C++ half still exists.

    A table takes its model down with it, and the model signals its own
    teardown -- so the handlers wired here are called once more with the
    view already half destroyed. Touching it then raises out of the event
    loop, which pytest-qt reports as an error in whatever test happened to
    be starting.
    """
    return obj is not None and shiboken6.isValid(obj)


def _header(view):
    if not _alive(view):
        return None
    if isinstance(view, (QTreeView, QTreeWidget)):
        return view.header()
    if hasattr(view, "horizontalHeader"):
        return view.horizontalHeader()
    return None


def _column_count(view) -> int:
    if not _alive(view):
        return 0
    model = view.model()
    return 0 if model is None else model.columnCount()


def _stamp_initial_order(view) -> None:
    """Put ``Qt.InitialSortOrderRole`` on every column of ``view``."""
    descending = Qt.DescendingOrder.value
    if isinstance(view, QTableWidget):
        for column in range(view.columnCount()):
            item = view.horizontalHeaderItem(column)
            if item is None:
                item = QTableWidgetItem(str(column + 1))
                view.setHorizontalHeaderItem(column, item)
            item.setData(Qt.InitialSortOrderRole, descending)
    elif isinstance(view, QTreeWidget):
        item = view.headerItem()
        if item is not None:
            for column in range(view.columnCount()):
                item.setData(column, Qt.InitialSortOrderRole, descending)


def restore_natural_order(view) -> None:
    """Put the rows back in the order the table was built in.

    The third click on a column. Qt clears its own indicator and stops there,
    leaving the rows in whatever order the second click left them, which
    reads as a click that did nothing.
    """
    global _RESTORING
    if not _alive(view):
        return
    model = view.model()
    if model is None:
        return
    if not isinstance(view, (QTableWidget, QTreeWidget)):
        # A proxy holds its source's order and hands it back when the sort
        # column goes away. Asked through the model rather than through
        # ``sortByColumn``, which declines a negative column.
        model.sort(-1, Qt.AscendingOrder)
        return
    _RESTORING = True
    try:
        model.sort(0, Qt.AscendingOrder)
    finally:
        _RESTORING = False


def install_sorting(view):
    """Give ``view`` the whole sorting contract. Returns ``view``.

    Accepts a ``QTableWidget``, a ``QTreeWidget`` or a ``QTableView``. A
    ``QTableView`` whose model cannot sort is given a
    :class:`SortableProxyModel`, so call this straight after ``setModel`` and
    take the selection model from the view afterwards.
    """
    header = _header(view)
    if header is None:
        return view
    if getattr(view, _STATE_ATTR, None) is not None:
        # Idempotent: a second call would wire the model signals twice and
        # run every fill guard twice with it.
        getattr(view, _STATE_ATTR).stamp_initial_order()
        return view

    if isinstance(view, QTableView) and not isinstance(view, QTableWidget):
        _wrap_in_proxy(view)

    view.setSortingEnabled(True)
    header.setSectionsClickable(True)
    header.setSortIndicatorShown(True)
    if hasattr(header, "setSortIndicatorClearable"):
        # The third click. Without it Qt flips between two orders forever and
        # the order the table was built in is unreachable.
        header.setSortIndicatorClearable(True)
    # A fresh view already points its indicator at column 0, so the first
    # click there would read as a flip and give ascending.
    header.setSortIndicator(-1, Qt.AscendingOrder)

    state = _SortState(view)
    setattr(view, _STATE_ATTR, state)
    state.stamp_initial_order()

    model = view.model()
    if model is not None:
        model.columnsInserted.connect(state.stamp_initial_order)
        model.modelReset.connect(state.stamp_initial_order)
        model.headerDataChanged.connect(state.stamp_initial_order)
        if isinstance(view, (QTableWidget, QTreeWidget)):
            model.rowsInserted.connect(state.suspend_for_fill)
            model.rowsRemoved.connect(state.suspend_for_fill)
    header.sortIndicatorChanged.connect(state.on_indicator_changed)
    return view


def _wrap_in_proxy(view) -> None:
    model = view.model()
    if model is None or isinstance(model, QSortFilterProxyModel):
        return
    proxy = SortableProxyModel(view)
    proxy.setSourceModel(model)
    view.setModel(proxy)


class SortableProxyModel(QSortFilterProxyModel):
    """Sorts a model's rows by what its cells mean.

    The same comparison as :class:`SortableTableItem`, for the views backed
    by a model rather than by items: numbers as numbers, missing last in
    both directions, and a header that asks for descending on the first
    click.
    """

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.InitialSortOrderRole and orientation == Qt.Horizontal:
            return Qt.DescendingOrder.value
        return super().headerData(section, orientation, role)

    def lessThan(self, left, right):
        source = self.sourceModel()
        left_value = source.data(left, Qt.DisplayRole)
        right_value = source.data(right, Qt.DisplayRole)
        mine_missing, mine = sort_key_of(left_value)
        theirs_missing, theirs = sort_key_of(right_value)
        if mine_missing or theirs_missing:
            if mine_missing and theirs_missing:
                return False
            descending = self.sortOrder() == Qt.DescendingOrder
            return mine_missing if descending else theirs_missing
        if mine is not None and theirs is not None:
            return mine < theirs
        if mine is None and theirs is None:
            return str(left_value or "").casefold() < str(
                right_value or "").casefold()
        return mine is not None

"""Consistent three-state sorting for Qt tables and trees.

The sorting cycle is:

* the first click on a column sorts it **descending**;
* the second click sorts it **ascending**;
* the third click clears the sort and restores the original row order;
* the fourth click starts the cycle again.

Formatted numeric values, including scientific notation, are sorted
numerically. Missing values such as blank cells, ``NaN``, ``pandas.NA``, and
``NaT`` sort after populated values in both directions.

Build cells with :func:`table_item` or :func:`tree_item` and pass the view to
:func:`install_sorting`::

    self.table = QTableWidget(0, 3)
    install_sorting(self.table)
    ...
    self.table.setItem(row, 0, table_item(coefficient))

:func:`install_sorting` temporarily suspends sorting while a view is
repopulated, preventing Qt from moving partially populated rows, and restores
the selected ordering after the update completes.
"""
from __future__ import annotations

import re
from typing import Optional

import shiboken6
from PySide6.QtCore import (
    QModelIndex,
    QObject,
    QSortFilterProxyModel,
    Qt,
    QTimer,
    Slot,
)
from PySide6.QtWidgets import (
    QListWidget,
    QListWidgetItem,
    QTableView,
    QTableWidget,
    QTableWidgetItem,
    QTreeView,
    QTreeWidget,
    QTreeWidgetItem,
)

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

_SERIAL = 0

_DISPLAY_ROLE = Qt.ItemDataRole.DisplayRole
_EDIT_ROLE = Qt.ItemDataRole.EditRole
_ROLE_TYPE = type(_DISPLAY_ROLE)
_TEXT_ROLE_VALUES = (_DISPLAY_ROLE.value, _EDIT_ROLE.value)

_RESTORING = False

_STATE_ATTR = "_spacr_sort_state"

#: A comma between digits is a thousands separator and nothing else.
_THOUSANDS = re.compile(r"(?<=\d),(?=\d\d\d(\D|$))")


def is_missing(value) -> bool:
    """Return whether ``value`` is empty or a recognized missing sentinel.

    This includes ``None``, blank strings, ``NaN``, pandas ``NA``, and
    ``NaT``. After the ``None`` and string checks, the result of
    ``pandas.isna`` is converted to ``bool``. A single-element array containing
    a missing value can therefore return ``True``; ambiguous multi-element
    results return ``False``.

    :param value: a cell value of any type.
    """
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, float):
        return value != value
    if isinstance(value, int):
        return False
    if getattr(value, "ndim", None) == 0:
        kind = getattr(getattr(value, "dtype", None), "kind", "")
        if kind in ("i", "u", "b"):
            return False
        if kind == "f":
            return bool(value != value)
    try:
        import pandas as _pd

        return bool(_pd.isna(value))
    except (ImportError, TypeError, ValueError):
        return False


def numeric_value(text) -> Optional[float]:
    """Return the numeric value represented by a cell, if one is present.

    Accepted formats include ordinary numbers, scientific notation, comma
    thousands separators, percentages, and a unit separated from the value by
    a space, such as ``"3.2 s"``. Alphanumeric identifiers such as ``"TP53"``
    return ``None``. Use :data:`SORT_KEY_ROLE` to provide an explicit numeric
    key for other display formats.

    :param text: cell text or number; ``None``, booleans, blanks and NaN give
        ``None``.
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
    without_separators = _THOUSANDS.sub("", text)
    if without_separators != text:
        yield without_separators
        text = without_separators
    if text.endswith("%"):
        yield text[:-1].strip()
        return
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
    """Return whether ``value`` should sort as missing table data.

    :param value: a cell value; missing per :func:`is_missing`, or a
        placeholder string such as ``"n/a"``, ``"-"`` or ``"?"``
        (case-insensitive), sorts as missing.
    """
    if is_missing(value):
        return True
    return isinstance(value, str) and value.strip().casefold() in _PLACEHOLDERS


def sort_key_of(value):
    """Return the ``(is_missing, numeric_value)`` comparison key.

    :param value: a cell value; see :func:`sorts_as_missing` and
        :func:`numeric_value`.
    """
    if sorts_as_missing(value):
        return True, None
    return False, numeric_value(value)


class _SortableMixin:
    """The comparison. See the module docstring for the contract."""

    def _init_sort_key(self, value, key=None) -> None:
        """Remember the value to sort by, and the row's arrival order.

        The serial is what "natural order" restores to, so it has to be taken
        when the row is BUILT rather than derived later from anything on screen.
        """
        global _SERIAL
        _SERIAL += 1
        self._sort_serial = _SERIAL
        if key is not None:
            self._sort_missing = False
            self._sort_number = float(key)
        else:
            self._sort_missing, self._sort_number = sort_key_of(value)


    def _resolve_key(self):
        """The key this cell sorts on. Settled once, refreshed on an edit."""
        return self._sort_missing, self._sort_number

    def _sort_text(self) -> str:
        """The text this cell sorts as. Subclasses that show text override it."""
        return ""

    def _descending(self) -> bool:
        """Whether the view is currently ordering downward.

        Read because MISSING VALUES GO LAST IN BOTH DIRECTIONS, which means a
        missing cell has to compare as the largest when Qt orders by "<" and the
        smallest when it orders by ">".
        """
        return False

    def __lt__(self, other):
        """Order two cells: numbers before words, missing values last.

        Returns ``NotImplemented`` for a peer with no text rather than guessing.
        ``sorted`` believes every answer it is given, so a cell that guessed
        would produce an order nobody could account for; declining lets Python
        raise instead.

        While the natural order is being restored this compares by ARRIVAL
        SERIAL, which is the only record of the order the rows came in.
        """
        if _RESTORING:
            return getattr(self, "_sort_serial", 0) < getattr(
                other, "_sort_serial", 0)
        mine_missing, mine = self._resolve_key()
        if isinstance(other, _SortableMixin):
            theirs_missing, theirs = other._resolve_key()
        else:
            peer = getattr(other, "text", None)
            if not callable(peer):
                return NotImplemented
            theirs_missing, theirs = sort_key_of(peer())
        if mine_missing or theirs_missing:
            if mine_missing and theirs_missing:
                return False
            return mine_missing if self._descending() else theirs_missing
        if mine is not None and theirs is not None:
            return mine < theirs
        if mine is None and theirs is None:
            return self._sort_text().casefold() < str(other.text()).casefold()
        return mine is not None


class SortableTableItem(_SortableMixin, QTableWidgetItem):
    """Table item with semantic numeric and missing-value sorting.

    :param value: Cell value. Missing values are rendered as empty cells.
    :param key: Optional explicit numeric sort key for display text that
        cannot be parsed directly.
    """

    def __init__(self, value="", key=None):
        """Create a table cell that sorts on a value rather than on its text.

        :param value: what the cell shows; a missing value renders blank.
        :param key: an explicit sort key, for a cell whose text does not order
            the way the value does -- a formatted duration, say. ``None``
            derives one from ``value``.
        """
        QTableWidgetItem.__init__(
            self, "" if is_missing(value) else str(value))
        self._init_sort_key(value, key=key)
        if key is not None:
            self.setData(SORT_KEY_ROLE, float(key))

    def setData(self, role, value):  # noqa: N802 - Qt spelling
        """Update the sort key when displayed or edited text changes.

        Non-text roles do not modify the key, and editing does not change the
        item's position in the table's unsorted order.

        :param role: the Qt item data role; only ``DisplayRole`` and
            ``EditRole`` refresh the sort key, which then comes from
            :data:`SORT_KEY_ROLE` when set and from the text otherwise.
        :param value: the data stored for ``role``.
        """
        QTableWidgetItem.setData(self, role, value)
        if role is not _DISPLAY_ROLE and role is not _EDIT_ROLE and (
                type(role) is _ROLE_TYPE
                or getattr(role, "value", role) not in _TEXT_ROLE_VALUES):
            return
        if hasattr(self, "_sort_serial"):
            override = self.data(SORT_KEY_ROLE)
            serial = self._sort_serial
            if override is None:
                self._init_sort_key(self.text())
            else:
                self._init_sort_key(None, key=override)
            self._sort_serial = serial

    def _sort_text(self) -> str:
        """Return the text this cell falls back to when it has no numeric key.

        :returns: the cell's text, never ``None``.
        """
        return self.text() or ""

    def _descending(self) -> bool:
        """Report whether the table is currently sorted descending.

        Used to keep missing values at the bottom in both directions rather
        than letting them flip to the top on a reverse.

        :returns: ``True`` when the header's indicator says descending;
            ``False`` for a cell not yet in a table.
        """
        view = self.tableWidget()
        if view is None:
            return False
        return view.horizontalHeader().sortIndicatorOrder() == Qt.DescendingOrder


class SortableTreeItem(_SortableMixin, QTreeWidgetItem):
    """Tree item with semantic sorting for the selected column.

    See :class:`SortableTableItem`. A tree item contains multiple columns, so
    its comparison key is resolved from the column currently being sorted.
    """

    def __init__(self, *args, **kwargs):
        """Create a tree row that sorts on its values and remembers its insertion order.

        The serial is what restores the original order when sorting is turned
        off -- a tree has no unsorted model to fall back to.

        :param args: passed through to ``QTreeWidgetItem``.
        :param kwargs: passed through to ``QTreeWidgetItem``.
        """
        QTreeWidgetItem.__init__(self, *args, **kwargs)
        global _SERIAL
        _SERIAL += 1
        self._sort_serial = _SERIAL

    def _sort_column(self) -> int:
        """Return the column the tree is sorting on.

        :returns: the sort column, or ``0`` for a row not yet in a tree or a
            tree with no indicator set.
        """
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
        """Return the text this row falls back to when it has no numeric key.

        :returns: the sort column's text, never ``None``.
        """
        return self.text(self._sort_column()) or ""

    def _descending(self) -> bool:
        """Report whether the tree is currently sorted descending.

        :returns: ``True`` when the header's indicator says descending;
            ``False`` for a row not yet in a tree.
        """
        view = self.treeWidget()
        if view is None:
            return False
        return view.header().sortIndicatorOrder() == Qt.DescendingOrder

    def __lt__(self, other):
        """Order two rows by value, keeping missing values last in both directions.

        While the original order is being restored this compares insertion
        serials instead, which is what lets "no sort" mean the order the rows
        arrived in.

        :param other: the row to compare against; a plain ``QTreeWidgetItem`` is
            compared on its text, and anything without one is declined so Python
            can try the reflected comparison.
        :returns: ``True`` when this row sorts first.
        """
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
    """Return a :class:`SortableTableItem` for ``value`` and optional key."""
    return SortableTableItem(value, key=key)


_SPARE_ATTR = "_spacr_spare_items"


def _spare_items(view) -> list:
    """The items ``view`` has taken out of its rows and keeps for reuse.

    :param view: a list, table or tree widget.
    :returns: the view's spare list, created empty the first time.
    """
    return view.__dict__.setdefault(_SPARE_ATTR, [])


def _wipe(view, item, columns: int = 1) -> None:
    """Make an item that is still in ``view`` look newly built.

    Every data role it holds, in every column, is cleared through the view's
    own model (which reports exactly the roles an item carries), its flags
    go back to a new item's, and it is shown and unselected. A sortable item
    also takes a fresh arrival serial, so "natural order" follows the order
    rows are written in again.

    :param view: the list, table or tree the item is in.
    :param item: the item to wipe.
    :param columns: columns to clear, for a tree item.
    """
    model = view.model()
    tree = isinstance(view, QTreeWidget)
    for column in range(columns if tree else 1):
        index = view.indexFromItem(item, column) if tree else view.indexFromItem(item)
        for role in list(model.itemData(index)):
            if tree:
                item.setData(column, role, None)
            else:
                item.setData(role, None)
    item.setFlags(_DEFAULT_FLAGS.get(_base_kind(item), item.flags()))
    item.setSelected(False)
    if hasattr(item, "setHidden"):
        item.setHidden(False)
    if isinstance(item, _SortableMixin):
        global _SERIAL
        _SERIAL += 1
        item._sort_serial = _SERIAL
        item._sort_cache = None
        if isinstance(item, SortableTableItem):
            item._sort_missing, item._sort_number = sort_key_of("")


def _base_kind(item) -> str:
    """Which Qt item class ``item`` is: ``list``, ``table`` or ``tree``."""
    if isinstance(item, QTableWidgetItem):
        return "table"
    if isinstance(item, QTreeWidgetItem):
        return "tree"
    return "list"


_DEFAULT_FLAGS: dict = {}


def _remember_default_flags() -> None:
    """Note the flags Qt gives a new list, table and tree item."""
    if not _DEFAULT_FLAGS:
        _DEFAULT_FLAGS["list"] = QListWidgetItem().flags()
        _DEFAULT_FLAGS["table"] = QTableWidgetItem().flags()
        _DEFAULT_FLAGS["tree"] = QTreeWidgetItem().flags()


def _reuse_list_items(listing: QListWidget, count: int,
                      make=QListWidgetItem) -> list:
    """The first ``count`` rows of ``listing``, as wiped items to rewrite.

    For a redraw that used to ``clear()`` the list and add new items. No
    item is deleted: surplus rows are taken out and kept for a later redraw,
    because deleting Python-made items while assistive technology may still
    hold wrappers of them is how lessons 39, 42 and 44 segfaulted. Selection
    and the current row are cleared, as ``clear()`` cleared them.

    :param listing: the list to redraw.
    :param count: how many rows it should have.
    :param make: builds a new item when no spare one is left.
    :returns: its items in row order; set every attribute each one needs.
    """
    _remember_default_flags()
    spare = _spare_items(listing)
    listing.clearSelection()
    listing.setCurrentItem(None)
    while listing.count() > count:
        last = listing.count() - 1
        _wipe(listing, listing.item(last))
        spare.append(listing.takeItem(last))
    for row in range(listing.count()):
        _wipe(listing, listing.item(row))
    while listing.count() < count:
        listing.addItem(spare.pop() if spare else make())
    return [listing.item(row) for row in range(count)]


def _retire_table_cells(table: QTableWidget, *, headers: bool = False) -> None:
    """Empty ``table`` the way ``setRowCount(0)`` did, deleting no item.

    The cells are wiped and kept for :func:`_spare_table_item`; with
    ``headers`` the header items are taken out too, as ``clear()`` removed
    them, and the row and column counts are left alone as ``clear()`` left
    them. Without it the table ends with no rows.

    :param table: the table to empty.
    :param headers: behave like ``clear()`` rather than ``setRowCount(0)``.
    """
    _remember_default_flags()
    spare = _spare_items(table)
    sorting = table.isSortingEnabled()
    table.setSortingEnabled(False)
    try:
        for row in range(table.rowCount()):
            for column in range(table.columnCount()):
                item = table.item(row, column)
                if item is not None:
                    _wipe(table, item)
                    spare.append(table.takeItem(row, column))
        if headers:
            keep = table.__dict__.setdefault("_spacr_spare_headers", [])
            for column in range(table.columnCount()):
                item = table.takeHorizontalHeaderItem(column)
                if item is not None:
                    keep.append(item)
            for row in range(table.rowCount()):
                item = table.takeVerticalHeaderItem(row)
                if item is not None:
                    keep.append(item)
        else:
            table.setRowCount(0)
    finally:
        table.setSortingEnabled(sorting)


def _spare_table_item(table: QTableWidget, value="", make=None):
    """A cell for ``table``: a kept one rewritten, or a new one.

    :param table: the table the cell is for.
    :param value: what the cell should show.
    :param make: builds a new item from ``value`` when no spare one is
        left; :func:`table_item` by default.
    :returns: an item not yet in the table.
    """
    spare = _spare_items(table)
    if not spare:
        return (make or table_item)(value)
    item = spare.pop()
    item.setText("" if is_missing(value) else str(value))
    if isinstance(item, SortableTableItem):
        item._init_sort_key(value)
    return item


def _reuse_tree_items(parent, count: int, make=None) -> list:
    """The first ``count`` children of ``parent``, as wiped items to rewrite.

    ``parent`` is a tree (its top-level items) or a tree item (its
    children). Surplus children are taken out with their own children and
    kept; call this again on each returned item for its children, which
    wipes or trims what it carried before. Deletes no item.

    :param parent: a ``QTreeWidget`` or a ``QTreeWidgetItem`` in one.
    :param count: how many children it should have.
    :param make: builds a new item when no spare one is left;
        :func:`tree_item` by default.
    :returns: the children in order; set every attribute each one needs.
    """
    _remember_default_flags()
    tree = parent if isinstance(parent, QTreeWidget) else parent.treeWidget()
    columns = max(1, tree.columnCount()) if tree is not None else 1
    spare = _spare_items(tree) if tree is not None else []
    top = isinstance(parent, QTreeWidget)
    size = parent.topLevelItemCount if top else parent.childCount
    child = parent.topLevelItem if top else parent.child
    take = parent.takeTopLevelItem if top else parent.takeChild
    add = parent.addTopLevelItem if top else parent.addChild
    if top:
        parent.clearSelection()
        parent.setCurrentItem(None)
    while size() > count:
        last = size() - 1
        if tree is not None:
            _wipe(tree, child(last), columns)
        spare.append(take(last))
    for row in range(size()):
        if tree is not None:
            _wipe(tree, child(row), columns)
    while size() < count:
        add(spare.pop() if spare else (make or tree_item)())
    return [child(row) for row in range(count)]


def tree_item(*args, **kwargs) -> SortableTreeItem:
    """Return a :class:`SortableTreeItem` using ``QTreeWidgetItem`` arguments."""
    return SortableTreeItem(*args, **kwargs)


class _SortState(QObject):
    """Per-view bookkeeping: the fill guard and the natural-order restore.

    :param view: the tree this state belongs to, and its QObject parent --
        so the state cannot outlive the view whose sort order it restores.
    """

    def __init__(self, view):
        """Take the view as parent and start with nothing suspended."""
        super().__init__(view)
        self._view = view
        self._suspended = None
        self._resuming = False
        self._stamping = False
        self._resume_timer = QTimer(self)
        self._resume_timer.setSingleShot(True)
        self._resume_timer.timeout.connect(self.resume_after_fill)


    @Slot(QModelIndex, int, int)
    def suspend_for_fill(self, *args) -> None:
        """Drop the sort while rows are being written, put it back after.

        Qt re-sorts a sorted table on every ``setItem`` into the sorted
        column: the row moves, and the loop that is filling it writes the
        rest of the row into whatever landed at that index. Clearing the sort
        indicator stops that -- the model only re-sorts on a write to the
        section the indicator names.
        """
        view = getattr(self, "_view", None)
        header = _header(view)
        if header is None or getattr(self, "_resuming", True):
            return
        if getattr(self, "_suspended", None) is None:
            self._suspended = (header.sortIndicatorSection(),
                               header.sortIndicatorOrder())
            if self._suspended[0] >= 0:
                header.setSortIndicator(-1, Qt.AscendingOrder)
        self._resume_timer.start(0)

    @Slot()
    def resume_after_fill(self) -> None:
        """Re-apply the sort the user had chosen, once the fill has settled."""
        suspended = getattr(self, "_suspended", None)
        view = getattr(self, "_view", None)
        if suspended is None or not _alive(view):
            return
        section, order = suspended
        self._suspended = None
        header = _header(view)
        if header is None or section < 0:
            return
        if section >= _column_count(view):
            return
        self._resuming = True
        try:
            header.setSortIndicator(section, order)
        finally:
            self._resuming = False


    @Slot()
    @Slot(QModelIndex, int, int)
    @Slot(Qt.Orientation, int, int)
    def stamp_initial_order(self, *args) -> None:
        """Tell the header that a fresh column starts descending.

        ``QHeaderView`` asks the model for ``Qt.InitialSortOrderRole`` when
        the click lands on a column that is not the sorted one, and uses
        ascending when nothing answers. Answering is the whole of
        descending-first: Qt does the rest, including the flip on the second
        click.
        """
        view = getattr(self, "_view", None)
        if getattr(self, "_stamping", True) or not _alive(view):
            return
        self._stamping = True
        try:
            _stamp_initial_order(view)
        finally:
            self._stamping = False

    @Slot(int, Qt.SortOrder)
    def on_indicator_changed(self, section: int, order) -> None:
        """A cleared indicator means "put it back the way it was"."""
        view = getattr(self, "_view", None)
        if (section >= 0 or getattr(self, "_resuming", True)
                or not _alive(view)):
            return
        restore_natural_order(view)


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
    """Find a view's header, whichever kind of view it is.

    :param view: the table or tree.
    :returns: the header, or ``None`` for a view that has none or whose C++
        half is gone.
    """
    if not _alive(view):
        return None
    if isinstance(view, (QTreeView, QTreeWidget)):
        return view.header()
    if hasattr(view, "horizontalHeader"):
        return view.horizontalHeader()
    return None


def _column_count(view) -> int:
    """How many columns a view's model has.

    :param view: the table or tree.
    :returns: the count, and ``0`` for a view with no model or one whose
        C++ half is gone.
    """
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


def _settle_sorting(view) -> None:
    """Re-apply a sort suspended for a fill now, instead of on the next turn.

    For code that fills a table and then reads or selects rows in the same
    call: the deferred resume would otherwise run after it.

    :param view: a view :func:`install_sorting` was called on.
    """
    state = getattr(view, _STATE_ATTR, None)
    if state is None or not _alive(state):
        return
    state._resume_timer.stop()
    state.resume_after_fill()


def restore_natural_order(view) -> None:
    """Restore rows to the order in which the view was populated.

    This completes the third state of the sorting cycle after Qt clears the
    header's sort indicator.
    """
    global _RESTORING
    if not _alive(view):
        return
    model = view.model()
    if model is None:
        return
    if not isinstance(view, (QTableWidget, QTreeWidget)):
        model.sort(-1, Qt.AscendingOrder)
        return
    _RESTORING = True
    try:
        model.sort(0, Qt.AscendingOrder)
    finally:
        _RESTORING = False


def install_sorting(view):
    """Install three-state semantic sorting on ``view`` and return it.

    Accepts a ``QTableWidget``, ``QTreeWidget``, or ``QTableView``. A
    ``QTableView`` with a non-proxy model is wrapped in
    :class:`SortableProxyModel`. An existing ``QSortFilterProxyModel`` is
    preserved with its own comparison behavior; a view without a model is
    not wrapped. Call this function immediately after ``setModel`` and
    obtain the view's selection model afterwards.

    :param view: the view to make sortable; a view with no header is
        returned untouched, and one already installed only has its initial
        order re-stamped.
    """
    header = _header(view)
    if header is None:
        return view
    if getattr(view, _STATE_ATTR, None) is not None:
        getattr(view, _STATE_ATTR).stamp_initial_order()
        return view

    if isinstance(view, QTableView) and not isinstance(view, QTableWidget):
        _wrap_in_proxy(view)

    view.setSortingEnabled(True)
    header.setSectionsClickable(True)
    header.setSortIndicatorShown(True)
    if hasattr(header, "setSortIndicatorClearable"):
        header.setSortIndicatorClearable(True)
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
    """Put a sorting proxy over a view's model, unless one is already there.

    Wrapping twice would sort a sorted view, so an existing proxy is left
    alone -- which is what makes installing sorting idempotent.

    :param view: the view to wrap.
    """
    model = view.model()
    if model is None or isinstance(model, QSortFilterProxyModel):
        return
    proxy = SortableProxyModel(view)
    proxy.setSourceModel(model)
    view.setModel(proxy)


class SortableProxyModel(QSortFilterProxyModel):
    """Proxy model providing semantic sorting for model-backed views.

    Numeric values sort numerically, missing values follow populated values
    in either direction, and the initial header order is descending. This is
    the model-backed equivalent of :class:`SortableTableItem`.
    """

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        """One header label, taken from the source model.

        :param section: the row or column number.
        :param orientation: which header.
        :param role: the Qt display role.
        :returns: the label, or None.
        """
        if role == Qt.InitialSortOrderRole and orientation == Qt.Horizontal:
            return Qt.DescendingOrder.value
        return super().headerData(section, orientation, role)

    def lessThan(self, left, right):
        """Order two cells by their values rather than by their text.

        A column of numbers rendered as text sorts 10 before 9 under a string
        comparison; this compares the values instead.

        :param left: the left cell's index.
        :param right: the right cell's index.
        :returns: True when left sorts first.
        """
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

"""
Database Browser — a read-only query panel for a spaCR ``measurements.db``.

The screen answers questions such as "how many cells are in plate1?" or
"what does the ``png_list`` table contain?" without a terminal and
``sqlite3 .../measurements/measurements.db``. Unless edit mode is armed,
the GUI has no way to write to the file.

Layout::

    ┌───────────────────────────────────────────────────────────────────┐
    │ /data/plate1/measurements/measurements.db   [DB…] [Run folder…]   │
    │ ☐ Edit mode   Read-only — enable editing in Preferences first.    │
    ├──────────────┬────────────────────────────────────────────────────┤
    │ Tables       │ Columns [search…]      (120 of 512 columns)        │
    │  cell   40   │ ┌────────────────────────────────────────────────┐ │
    │  nucleus     │ │ prc          cell_area   cell_channel_1_mean…  │ │
    │  png_list    │ │ plate1_A01_1 1204.5      3311.2                │ │
    │              │ └────────────────────────────────────────────────┘ │
    │              │ showing 100 of ≈412 003 rows (estimate) [Load more]│
    ├──────────────┴────────────────────────────────────────────────────┤
    │ Filter: [column ▾] [op ▾] [value]  ☐ raw SQL  [Apply] [Clear]     │
    │                                              [Export filtered CSV]│
    └───────────────────────────────────────────────────────────────────┘

Design notes that matter for real spaCR databases:

* Never ``SELECT *`` the whole table. Measurement tables run to
  hundreds of thousands of rows and hundreds of feature columns. The
  first chunk (100 rows) is fetched and painted on its own; the rest
  arrives as the user scrolls, through ``canFetchMore`` / ``fetchMore``.
* **Keyset paging, not OFFSET.** Each chunk asks for ``rowid > <last
  rowid seen>`` and ``ORDER BY rowid``. ``LIMIT ? OFFSET ?`` makes
  SQLite walk (and throw away) every skipped row, so chunk 500 of a
  400 k-row table would cost 500× chunk 1. Only tables with no usable key
  (views, composite-primary-key ``WITHOUT ROWID`` tables) fall back to
  ``OFFSET``, and those are small by construction.
* **The count never blocks the first paint.** ``SELECT COUNT(*)`` is a
  full scan. The first chunk is painted against ``max(rowid)``, which
  is O(1), and that number is *always* rendered as "≈… (estimate)".
  The exact ``COUNT(*)`` follows on its own job and replaces it.
* **Off the GUI thread.** Every chunk, count and export goes through
  :func:`spacr.qt.bridge.make_thread`, the same helper the pipeline
  screens use, and each worker opens (and closes) its **own** sqlite
  connection — ``sqlite3`` objects are not shareable across threads.
  Jobs queue and run **one at a time**; see :meth:`DbBrowserScreen._run_job`
  for why two ``PipelineWorker``\\ s must not overlap.
* **Cancellable.** Switching table or database bumps a load token.
  Results tagged with a stale token are dropped, so an in-flight load
  can never paint the previous table's rows into the new view, and a
  job that has not started yet is dropped without spending a thread.
  Nothing is ever killed mid-flight.
* **Read-only by default, structurally.** Browsing connections are
  opened with the ``file:…?mode=ro`` URI *and* ``PRAGMA query_only =
  ON``. A write is rejected by SQLite itself, not by an application-level
  check. Editing is a separate, opt-in path (see below) with its own
  connection.
* **No string-formatted values.** Everything the user types is bound as
  a ``?`` parameter. Identifiers (table + column names) never come from
  free text — they are matched against the live schema and only then
  double-quoted.
* **No modal dialogs on any error path.** Problems land in an inline
  status label; a headless run can never block on a message box. The
  single exception is the edit-mode confirmation, which is injectable
  (:attr:`DbBrowserScreen.confirm_edit_mode`).

Editing (opt-in, and guarded five ways)
---------------------------------------

An UPDATE against ``measurements.db`` is unrecoverable, so every guard
below has to fail open to *read-only*:

1. :func:`spacr.qt.preferences.get_db_browser_editable` must be on —
   it is off by default and lives in Preferences, not on this screen.
2. The database must have been chosen explicitly in this session
   (``set_database(..., explicit=True)``).
3. The user must tick "Edit mode" *and* confirm; ticking alone does
   nothing.
4. The row must be addressable by ``rowid`` or a primary key. Without
   one, the edit is refused — an UPDATE matching on column values can
   hit many rows, which on a measurements table is silent mass
   corruption. The write also probes ``COUNT(*)`` for the row address
   first and rolls back unless ``rowcount == 1``.
5. The typed text must be coercible to the column's declared type.
   SQLite stores ``'abc'`` in an INTEGER column without error;
   :func:`coerce_for_column` refuses instead.

Loading a different database always resets edit mode to off.

Since unified-store step S6 every read goes through
:class:`spacr.store_browse.StoreDb` and :mod:`spacr.store`, and this module
opens no database itself. A spaCR store project is paged by keyset with
:meth:`spacr.store.Store.page`, reads only the visible columns of a wide
table, counts from its registry and sorts and filters in DuckDB; an older
measurements.db is paged on ``rowid`` as described above but is now never
written. Edit mode applies to store projects only, whose edits are edit rows
in the project's edit log (``spacr.store.edit``); until that exists every
project is read-only.
"""
from __future__ import annotations

import os
import re
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from ..widgets.fold_strip import FoldStrip

import pandas as pd
from PySide6.QtCore import (
    QAbstractTableModel,
    QItemSelection,
    QItemSelectionModel,
    QModelIndex,
    Qt,
    Signal,
    QTimer,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSpinBox,
    QStyle,
    QTableView,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from spacr import store as _store
from spacr.store_browse import READ_ERRORS, StoreDb

from ...selection import (
    OBJECT_KEY_COLUMNS,
    DataFilter,
    Selection,
    with_object_type,
)
from ..bridge import make_thread, _wait_serving_the_interface
from ..i18n import tr
from ..linked_selection import LinkedView
from ..preferences import get_db_browser_editable
from ..theme import SPACING, active_palette
from ..widgets import Divider
from ..widgets.collapsible_splitter import CollapsibleSplitter
from ..widgets.measurements_example import (
    EXAMPLE_TABLE, install_test_data_button,
)
from ..widgets.sortable_table import _reuse_list_items
from ..widgets.toggle import Toggle
from .app_screen import ModuleHeader

__all__ = [
    "DB_FILENAME",
    "DEFAULT_PAGE_SIZE",
    "DbBrowserScreen",
    "EditRefused",
    "OPERATORS",
    "PreviewModel",
    "build_where",
    "coerce_for_column",
    "column_affinity",
    "quote_ident",
    "resolve_db_path",
    "validate_raw_predicate",
]


DB_FILENAME = "measurements.db"
_MEASUREMENTS_SUBDIR = "measurements"

#: Rows fetched per chunk. Big enough to fill a window, small enough
#: that a 500-column table still paints instantly.
DEFAULT_PAGE_SIZE = 100
#: (min, max) the "Rows / fetch" spin box allows.
PAGE_SIZE_RANGE = (25, 1000)

#: Above this many *visible* columns the view keeps a fixed column width
#: instead of measuring content. ``resizeColumnsToContents`` walks every
#: cell of every column, which is fine for a 30-column table and painful
#: for a 500-column feature table.
AUTOSIZE_MAX_COLUMNS = 60



def resolve_db_path(path: str) -> str:
    """Return the absolute path of the project ``path`` refers to.

    Resolved by :func:`spacr.store.locate`: a spaCR store project (its
    folder, its ``spacr.duckdb``, or a run folder whose ``measurements``
    folder is one) gives the project folder, and wins over an older
    ``measurements.db`` beside it; otherwise the database file itself, or
    ``<src>/measurements/measurements.db`` (``<src>/measurements.db`` as a
    fallback) of a run ``src`` folder. A ``measurements.db`` path in a
    store project names that store even when the file does not exist.

    :param path: file or folder chosen by the user.
    :returns: absolute path to the store project folder or the database file.
    :raises ValueError: when ``path`` is empty.
    :raises FileNotFoundError: when nothing resolvable exists there.
    """
    if path is None or not str(path).strip():
        raise ValueError(
            "No database selected — choose a measurements.db or a run folder.")
    p = os.path.abspath(os.path.expanduser(str(path).strip()))
    from spacr.measurement_store import exists as _measurements_exist
    if os.path.isfile(p) or (os.path.basename(p) == DB_FILENAME
                             and _measurements_exist(p)):
        return _store.locate(p)
    if os.path.isdir(p):
        try:
            return _store.locate(p)
        except FileNotFoundError:
            pass
        raise FileNotFoundError(tr(
            "No {filename} under {folder}; expected {expected}. Choose a "
            "measurements.db file from a completed run, or select the run "
            "folder containing measurements/measurements.db.",
            filename=DB_FILENAME,
            folder=p,
            expected=os.path.join(p, _MEASUREMENTS_SUBDIR, DB_FILENAME)))
    raise FileNotFoundError(tr(
        "No such file or folder: {path}. Check that the drive is "
        "connected, then choose an existing measurements.db file or its "
        "run folder.",
        path=p))


def quote_ident(name: str) -> str:
    """Double-quote a SQL identifier, escaping embedded quotes.

    Only ever called with a name that has already been matched against
    the live schema; the quoting is an additional safeguard for identifiers
    that are legal but awkward (``cell_channel_1 (raw)``).

    :param name: schema-validated table or column identifier.
    :returns: identifier surrounded by SQL double quotes.
    """
    return '"' + str(name).replace('"', '""') + '"'


_INT_RE = re.compile(r"^[+-]?\d+$")
_FLOAT_RE = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")


def _coerce_value(text: Any) -> Any:
    """Turn a text field into int/float when it clearly is one, else str.

    Bound parameters carry no affinity, so ``WHERE cell_area > '1000'``
    would still work via SQLite's numeric-affinity coercion — but binding
    a real number makes equality behave the way users expect and keeps
    the comparison honest for TEXT columns holding digits.
    """
    s = str(text).strip()
    if _INT_RE.match(s):
        return int(s)
    if _FLOAT_RE.match(s):
        return float(s)
    return str(text)


def _like_escape(value: Any) -> str:
    """Escape LIKE wildcards so a literal ``%`` or ``_`` stays literal."""
    s = str(value)
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


#: operator label -> (SQL template, argument count, value transform)
#:
#: The template is a *fixed* string; ``{col}`` is filled with a quoted,
#: schema-validated identifier and every user value goes in as ``?``.
OPERATORS: Dict[str, Tuple[str, int, Optional[Callable[[Any], Any]]]] = {
    "=":           ("{col} = ?", 1, _coerce_value),
    "!=":          ("{col} <> ?", 1, _coerce_value),
    "<":           ("{col} < ?", 1, _coerce_value),
    "<=":          ("{col} <= ?", 1, _coerce_value),
    ">":           ("{col} > ?", 1, _coerce_value),
    ">=":          ("{col} >= ?", 1, _coerce_value),
    "contains":    ("{col} LIKE ? ESCAPE '\\'", 1,
                    lambda v: f"%{_like_escape(v)}%"),
    "starts with": ("{col} LIKE ? ESCAPE '\\'", 1,
                    lambda v: f"{_like_escape(v)}%"),
    "ends with":   ("{col} LIKE ? ESCAPE '\\'", 1,
                    lambda v: f"%{_like_escape(v)}"),
    "is null":     ("{col} IS NULL", 0, None),
    "is not null": ("{col} IS NOT NULL", 0, None),
}


def build_where(column: str, op: str, value: Any,
                columns: Sequence[str]) -> Tuple[str, tuple]:
    """Build a ``(where_sql, params)`` pair from the structured filter row.

    :param column: column name; must appear in ``columns``.
    :param op: one of :data:`OPERATORS`.
    :param value: raw text from the value field — bound, never formatted.
    :param columns: the table's real column list, from the schema.
    :raises ValueError: for an unknown column or operator.
    """
    if column not in set(columns):
        raise ValueError(tr(
            "Unknown column {column!r} in this table. Reload the table and "
            "choose a column from the filter list.",
            column=column))
    spec = OPERATORS.get(op)
    if spec is None:
        raise ValueError(tr(
            "Unknown operator {op!r}. Choose an operator from the filter "
            "list, such as = or contains.",
            op=op))
    template, nargs, transform = spec
    sql = template.format(col=quote_ident(column))
    if nargs == 0:
        return sql, ()
    return sql, (transform(value) if transform else value,)


_FORBIDDEN_RAW = re.compile(
    r"(?:^|[^A-Za-z_])"
    r"(insert|update|delete|drop|alter|create|replace|attach|detach|"
    r"pragma|vacuum|reindex|begin|commit|rollback)"
    r"(?:[^A-Za-z_]|$)",
    re.IGNORECASE,
)


def validate_raw_predicate(text: str) -> str:
    """Return ``text`` stripped, or raise if it is not a lone predicate.

    The raw box accepts a hand-written WHERE predicate (``cell_area > 1000
    AND well LIKE 'A%'``). Statement separators, comments and DDL/DML
    keywords are refused so it stays a *predicate*; the read-only
    connection plus ``PRAGMA query_only`` prevent any write.

    :param text: raw WHERE-clause fragment entered by the user.
    :returns: stripped predicate after the safety checks.
    :raises ValueError: when the fragment is empty or contains statement
        syntax, comments, or a forbidden SQL keyword.
    """
    t = "" if text is None else str(text).strip()
    if not t:
        raise ValueError("Raw filter is empty — type a condition or untick "
                         "'raw SQL'.")
    if ";" in t:
        raise ValueError(tr(
            "';' is not allowed in a filter. Remove the semicolon and enter "
            "one condition, such as cell_area > 100."))
    if "--" in t or "/*" in t:
        raise ValueError(tr(
            "SQL comments are not allowed in the filter. Remove comment text "
            "and enter only the condition to apply."))
    m = _FORBIDDEN_RAW.search(t)
    if m:
        raise ValueError(tr(
            "{keyword} is not allowed; filters only read. Enter a condition "
            "such as cell_area > 100, or turn off raw SQL and use the "
            "column/operator controls.",
            keyword=m.group(1).upper()))
    return t



class EditRefused(Exception):
    """An edit was rejected *before* anything was written.

    Distinct from :class:`sqlite3.Error` so the screen can tell "we
    refused this" (a guard fired, nothing happened) from "SQLite
    refused this" (a constraint, a locked file).
    """


def column_affinity(decl_type: Optional[str]) -> str:
    """Return SQLite's type affinity for a declared column type.

    The five rules are the ones in the SQLite file-format spec, in
    order: ``INT`` → INTEGER, ``CHAR``/``CLOB``/``TEXT`` → TEXT,
    ``BLOB`` or no type at all → BLOB, ``REAL``/``FLOA``/``DOUB`` →
    REAL, otherwise NUMERIC. Knowing the affinity is what lets
    :func:`coerce_for_column` refuse a value SQLite would otherwise
    store with the wrong type.

    :param decl_type: declared SQLite column type, or ``None``/an empty string
        when untyped.
    :returns: ``INTEGER``, ``TEXT``, ``BLOB``, ``REAL``, or ``NUMERIC``.
    """
    t = str(decl_type or "").upper()
    if "INT" in t:
        return "INTEGER"
    if "CHAR" in t or "CLOB" in t or "TEXT" in t:
        return "TEXT"
    if "BLOB" in t or not t.strip():
        return "BLOB"
    if "REAL" in t or "FLOA" in t or "DOUB" in t:
        return "REAL"
    return "NUMERIC"


def coerce_for_column(text: Any, decl_type: Optional[str],
                      column: str = "value") -> Any:
    """Return the value to bind for ``text`` in a column of ``decl_type``.

    SQLite has no static typing: binding the string ``"abc"`` to an
    INTEGER column stores the *text* "abc" there, and every downstream
    ``pandas.read_sql`` then gets an object column where it expected a
    number. This function refuses instead.

    * empty text → ``None`` (SQL NULL). A NOT NULL column then raises
      from SQLite itself and the write is rolled back.
    * INTEGER affinity → ``int``, or :class:`ValueError`.
    * REAL affinity → ``float``, or :class:`ValueError`.
    * TEXT affinity → ``str``, always.
    * NUMERIC affinity / no declared type → ``int``, else ``float``,
      else ``str`` (which is exactly what SQLite itself would store).
    * anything declared BLOB → refused; binary is not editable as text.

    :param text: editor value to convert before binding it to SQLite.
    :param decl_type: declared type of the destination column.
    :param column: destination column name used in actionable error messages.
    :raises ValueError: when ``text`` cannot be represented in the
        column's type.
    """
    decl = str(decl_type or "").strip()
    label = decl or "untyped"
    s = "" if text is None else str(text)
    if "BLOB" in decl.upper():
        raise ValueError(tr(
            "{column!r} is declared {label}; binary values cannot be edited "
            "as text. Leave this value unchanged, or use a database editor "
            "that supports binary fields.",
            column=column,
            label=label))
    if not s.strip():
        return None
    stripped = s.strip()
    affinity = column_affinity(decl)
    if affinity == "INTEGER":
        if _INT_RE.match(stripped):
            return int(stripped)
        raise ValueError(tr(
            "{value!r} is not a whole number, and {column!r} is declared "
            "{label}. Enter a whole number without a decimal point, such as "
            "1.",
            value=s,
            column=column,
            label=label))
    if affinity == "REAL":
        if _FLOAT_RE.match(stripped):
            return float(stripped)
        raise ValueError(tr(
            "{value!r} is not a number, and {column!r} is declared {label}. "
            "Enter a number such as 1.5, using a dot as the decimal "
            "separator.",
            value=s,
            column=column,
            label=label))
    if affinity == "TEXT":
        return s
    if _INT_RE.match(stripped):
        return int(stripped)
    if _FLOAT_RE.match(stripped):
        return float(stripped)
    return s


def _capture_result(fn: Callable[[], Any], payload: Dict[str, Any]) -> None:
    """Run ``fn`` and leave its return value in ``payload``.

    ``PipelineWorker`` calls its function as ``fn(settings)`` and its
    ``finished`` signal only carries a success flag, so a job's actual
    result has to travel in the settings dict it was handed. Runs on the
    worker thread; touches nothing but ``payload``.
    """
    payload["result"] = fn()


_TEXT_ROLES = frozenset({int(Qt.DisplayRole.value), int(Qt.ToolTipRole.value)})
_EDIT_ROLE = int(Qt.EditRole.value)
_COLUMN_SEARCH_DELAY_MS = 150
_WIDTH_SAMPLE_ROWS = 500
_WIDTH_CANDIDATES = 3


def _display_text(value: Any) -> str:
    """How a stored value reads in a preview cell."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _sort_key(value: Any) -> tuple:
    """Total order over a sqlite column: NULLs, then numbers, then text."""
    if value is None:
        return (0, 0.0, "")
    if isinstance(value, (int, float)):
        return (1, float(value), "")
    return (2, 0.0, str(value))


class PreviewModel(QAbstractTableModel):
    """Holds the rows fetched so far plus the column-visibility mask.

    Besides holding the rows, it handles two things:

    * **Incremental fetch.** ``canFetchMore``/``fetchMore`` are Qt's
      incremental-loading interface; the view calls
      them when the user scrolls to the bottom and the model asks the
      screen for another chunk. While a chunk is in flight
      ``canFetchMore`` is False, so a scroll cannot queue ten of them.
    * **Row identity.** Every row carries the key tuple it was fetched
      with (``rowid``, or the primary key). That is what makes an edit
      addressable — and its absence is what makes an edit refusable.

    Chunks are fetched with *every* column, and the column search only
    changes which of them are mapped into the view. Typing in the search
    box therefore never re-queries the database — which is what makes it
    usable on a table with 500 feature columns.

    :param parent: optional Qt owner responsible for the model's lifetime.
    """

    def __init__(self, parent=None):
        """Build an empty preview."""
        super().__init__(parent)
        self._columns: List[str] = []
        self._rows: List[tuple] = []
        self._keys: List[Optional[tuple]] = []
        self._visible: List[int] = []
        self._filter: str = ""
        self._row_offset: int = 0
        self._editable: bool = False
        self._more: bool = False
        self._fetch_hook: Optional[Callable[[], Any]] = None
        self._commit_hook: Optional[Callable[[int, str, Any], bool]] = None
        self._shown: List[Optional[tuple]] = []
        self._widest: Dict[int, List[str]] = {}
        self._cell_flags = Qt.ItemIsSelectable | Qt.ItemIsEnabled


    def set_page(self, columns: Sequence[str], rows: Sequence[Sequence[Any]],
                 row_offset: int = 0,
                 keys: Optional[Sequence[Optional[tuple]]] = None) -> None:
        """Replace the model contents, keeping the current column search.

        :param columns: schema-order column names represented by every row.
        :param rows: complete values for the newly loaded page.
        :param row_offset: zero-based database offset used for row headings.
        :param keys: stable row addresses, or ``None`` for uneditable rows.
        """
        self.beginResetModel()
        self._columns = list(columns)
        self._rows = [tuple(r) for r in rows]
        self._keys = (list(keys) if keys is not None
                      else [None] * len(self._rows))
        self._row_offset = int(row_offset)
        self._shown = [None] * len(self._rows)
        self._widest = {}
        self._recompute_visible()
        self.endResetModel()

    def append_rows(self, rows: Sequence[Sequence[Any]],
                    keys: Optional[Sequence[Optional[tuple]]] = None) -> int:
        """Add a fetched chunk to the end and return how many rows landed.

        :param rows: complete schema-order rows to append.
        :param keys: stable row addresses paired with ``rows``; ``None``
            supplies an uneditable ``None`` key for every appended row.
        """
        new = [tuple(r) for r in rows]
        if not new:
            return 0
        first = len(self._rows)
        self.beginInsertRows(QModelIndex(), first, first + len(new) - 1)
        self._rows.extend(new)
        self._shown.extend([None] * len(new))
        self._keys.extend(list(keys) if keys is not None
                          else [None] * len(new))
        if first < _WIDTH_SAMPLE_ROWS:
            self._widest = {}
        self.endInsertRows()
        return len(new)

    def clear(self) -> None:
        """Drop every row and column."""
        self.set_page([], [], 0)

    def all_columns(self) -> List[str]:
        """Every column the table has, filtered or not.

        :returns: the column names.
        """
        return list(self._columns)

    def rows(self) -> List[tuple]:
        """The raw rows loaded so far, with every column (search-independent)."""
        return list(self._rows)

    def row_key(self, row: int) -> Optional[tuple]:
        """Return the key tuple addressing ``row``, or ``None``.

        :param row: zero-based model row index.
        """
        if 0 <= row < len(self._keys):
            return self._keys[row]
        return None

    def value(self, row: int, column: str) -> Any:
        """Return the stored, unformatted value at ``row`` / ``column``.

        :param row: zero-based model row index.
        :param column: schema column name.
        """
        if not (0 <= row < len(self._rows)) or column not in self._columns:
            return None
        return self._rows[row][self._columns.index(column)]

    def set_value(self, row: int, column: str, value: Any) -> bool:
        """Write a value into the in-memory page after a real UPDATE.

        :param row: zero-based model row index.
        :param column: schema column whose displayed value changed.
        :param value: already-coerced value written to the database.
        :returns: whether the addressed cell exists and was updated.
        """
        if not (0 <= row < len(self._rows)) or column not in self._columns:
            return False
        col = self._columns.index(column)
        current = list(self._rows[row])
        current[col] = value
        self._rows[row] = tuple(current)
        self._shown[row] = None
        self._widest.pop(col, None)
        if col in self._visible:
            idx = self.index(row, self._visible.index(col))
            self.dataChanged.emit(idx, idx, [Qt.DisplayRole, Qt.EditRole])
        return True

    def visible_columns(self) -> List[str]:
        """The columns the filter is currently letting through.

        :returns: the visible column names.
        """
        return [self._columns[i] for i in self._visible]

    def column_filter(self) -> str:
        """The text currently filtering the columns.

        :returns: the filter text.
        """
        return self._filter

    def set_column_filter(self, text: str) -> None:
        """Show only columns whose name contains ``text`` (case-insensitive).

        An empty string restores every column.

        :param text: case-insensitive substring required in visible names.
        """
        self._filter = "" if text is None else str(text)
        self.beginResetModel()
        self._recompute_visible()
        self.endResetModel()

    def _recompute_visible(self) -> None:
        """Re-apply the column filter and tell the view what moved."""
        needle = self._filter.strip().lower()
        if not needle:
            self._visible = list(range(len(self._columns)))
        else:
            self._visible = [i for i, c in enumerate(self._columns)
                             if needle in str(c).lower()]


    def set_fetch_hook(self, hook: Optional[Callable[[], Any]]) -> None:
        """Set what :meth:`fetchMore` calls to ask for the next chunk.

        :param hook: zero-argument fetch request, or ``None`` to disconnect it.
        """
        self._fetch_hook = hook

    def set_more(self, more: bool) -> None:
        """Say whether another chunk can be fetched right now.

        :param more: true only while an unfetched chunk remains available.
        """
        self._more = bool(more)

    def canFetchMore(self, parent=QModelIndex()) -> bool:  # noqa: N802, B008
        """Return whether the root model can request another chunk.

        :param parent: Qt parent index; valid child indexes cannot fetch.
        """
        return (not parent.isValid()) and self._more

    def fetchMore(self, parent=QModelIndex()) -> None:  # noqa: N802, B008
        """Request the next chunk through the configured fetch hook.

        :param parent: Qt parent index; valid child indexes are ignored.
        """
        if parent.isValid() or not self._more or self._fetch_hook is None:
            return
        self._fetch_hook()


    def set_commit_hook(self,
                        hook: Optional[Callable[[int, str, Any], bool]]) -> None:
        """Set what :meth:`setData` calls to actually write a cell.

        :param hook: ``(row, column, value)`` writer, or ``None`` to disable it.
        """
        self._commit_hook = hook

    def set_editable(self, editable: bool) -> None:
        """Turn cell editing on or off and repaint the item flags.

        :param editable: whether valid cells advertise Qt's editable flag.
        """
        editable = bool(editable)
        if editable == self._editable:
            return
        self.beginResetModel()
        self._editable = editable
        self._cell_flags = Qt.ItemIsSelectable | Qt.ItemIsEnabled
        if editable:
            self._cell_flags |= Qt.ItemIsEditable
        self.endResetModel()

    def is_editable(self) -> bool:
        """Whether this preview allows edits.

        Off unless edit mode is armed. The browser reads a project's real
        measurements, and an accidental edit there is a silent change to data
        a run already produced.

        :returns: True when editing is enabled.
        """
        return self._editable


    def rowCount(self, parent=QModelIndex()) -> int:  # noqa: N802, B008
        """Return loaded rows for the root model and zero for children.

        :param parent: Qt parent index whose child-row count is requested.
        """
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent=QModelIndex()) -> int:  # noqa: N802, B008
        """Return visible columns for the root model and zero for children.

        :param parent: Qt parent index whose child-column count is requested.
        """
        return 0 if parent.isValid() else len(self._visible)

    def flags(self, index):
        """Return Qt item flags for ``index``, including armed editability.

        :param index: model index whose interaction flags are requested.
        """
        if not index.isValid():
            return Qt.NoItemFlags
        return self._cell_flags

    def data(self, index, role=Qt.DisplayRole):
        """Return the display, tooltip, or exact editor value for a cell.

        :param index: model index identifying the requested cell.
        :param role: Qt data role; unsupported roles return ``None``.
        """
        if role in _TEXT_ROLES:
            row = index.row()
            if row < 0:
                return None
            try:
                shown = self._shown[row]
                if shown is None:
                    shown = self._shown[row] = tuple(
                        _display_text(v) for v in self._rows[row])
                return shown[self._visible[index.column()]]
            except IndexError:
                return None
        if role != _EDIT_ROLE or not index.isValid():
            return None
        try:
            value = self._rows[index.row()][self._visible[index.column()]]
        except IndexError:
            return None
        if value is None:
            return ""
        if isinstance(value, bytes):
            return f"<{len(value)} bytes>"
        return repr(value) if isinstance(value, float) else str(value)

    def setData(self, index, value, role=Qt.EditRole) -> bool:  # noqa: N802
        """Commit an editor value through the guarded screen callback.

        :param index: valid editable model index.
        :param value: raw editor value passed to the commit hook.
        :param role: Qt role, which must be :attr:`Qt.EditRole`.
        :returns: whether the commit hook accepted and wrote the change.
        """
        if (role != Qt.EditRole or not index.isValid() or not self._editable
                or self._commit_hook is None):
            return False
        try:
            column = self._columns[self._visible[index.column()]]
        except IndexError:
            return False
        return bool(self._commit_hook(index.row(), column, value))

    def headerData(self, section, orientation, role=Qt.DisplayRole):  # noqa: N802
        """Return a column name or one-based absolute row number.

        :param section: zero-based visible column or row-header section.
        :param orientation: horizontal for names, vertical for row numbers.
        :param role: Qt data role; only :attr:`Qt.DisplayRole` is served.
        """
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            if 0 <= section < len(self._visible):
                return self._columns[self._visible[section]]
            return None
        return str(self._row_offset + section + 1)

    def sort(self, column: int, order=Qt.AscendingOrder) -> None:
        """Sort the loaded rows in memory.

        The screen only enables view sorting once the *whole* table is
        loaded, so this never sorts a partial result and calls it the
        table's order. Keys travel with their rows, so an edit after a
        sort still addresses the row it looks like it addresses.

        :param column: zero-based visible column to sort by.
        :param order: Qt sort order, defaulting to
            :attr:`Qt.AscendingOrder`.
        """
        if not (0 <= column < len(self._visible)) or not self._rows:
            return
        idx = self._visible[column]
        order_ = sorted(range(len(self._rows)),
                        key=lambda i: _sort_key(self._rows[i][idx]),
                        reverse=order == Qt.DescendingOrder)
        self.beginResetModel()
        self._rows = [self._rows[i] for i in order_]
        self._keys = [self._keys[i] for i in order_]
        self._shown = [self._shown[i] for i in order_]
        self._widest = {}
        self.endResetModel()

    def _widest_texts(self, column: int) -> List[str]:
        """The longest display strings of visible ``column`` in the sample."""
        try:
            col = self._visible[column]
        except IndexError:
            return []
        cached = self._widest.get(col)
        if cached is None:
            texts = {_display_text(row[col])
                     for row in self._rows[:_WIDTH_SAMPLE_ROWS]}
            cached = sorted(texts, key=len, reverse=True)[:_WIDTH_CANDIDATES]
            self._widest[col] = cached
        return cached



def _export_format(path: str, chosen: str = "") -> str:
    """The export format a chosen file name or dialog filter names.

    :param path: the file name.
    :param chosen: the file dialog's selected filter, if any.
    :returns: ``"csv"``, ``"parquet"`` or ``"sqlite"``.
    """
    suffix = os.path.splitext(str(path))[1].lower()
    if suffix in (".parquet", ".pq"):
        return "parquet"
    if suffix in (".db", ".sqlite", ".sqlite3"):
        return "sqlite"
    if suffix == ".csv":
        return "csv"
    text = str(chosen or "").lower()
    if "parquet" in text:
        return "parquet"
    if "sqlite" in text:
        return "sqlite"
    return "csv"


def _job_lane(kind: str) -> str:
    """Which worker lane a job of ``kind`` runs on.

    A ``COUNT(*)`` is a full scan of the table, seconds on a large
    measurement table, so it runs on a lane of its own: the next page a
    scroll asks for never waits behind it. SQLite serves concurrent
    read-only connections, and every job opens its own.
    """
    return "count" if kind == "count" else "main"


class DbBrowserScreen(LinkedView, QWidget):
    """Browser for a spaCR measurement project — read-only by default.

    The project is a spaCR store project or an older ``measurements.db``;
    both open through :class:`spacr.store_browse.StoreDb`.

    Joined to the shared selection as ``"db_browser"``, in both directions:

    * selecting rows publishes them, so a row picked out here is highlighted
      on the plate heatmap and in the UMAP;
    * a selection published elsewhere selects and scrolls to the same rows;
    * a filter published elsewhere hides rows, which a selection never does.

    All three act on the rows already fetched and none of them re-queries:
    the shared filter applies to the rows in memory, and the row-count label
    states how much of the table those rows are. (The column search can
    re-query a store project, see :meth:`set_column_filter`.)

    :param parent: optional Qt owner responsible for the screen's lifetime.
    :param threaded: run queries on a worker thread (the default). Tests
        pass ``False`` to get deterministic, synchronous behaviour.
    :ivar last_error: text of the most recent failure, ``""`` when the
        last operation succeeded. Errors are *only* ever reported here
        and in the inline status label — never in a modal dialog.
    :ivar confirm_edit_mode: callable taking the confirmation text and
        returning a bool. Replace it to arm edit mode without a dialog
        (every test does). The default opens the one QMessageBox this
        screen owns.
    :ivar auto_count: when True (the default) the exact row count follows
        the first chunk on its own job. Set False to browse a very large
        table on the estimate alone (``max(rowid)`` of an older file; a
        store project's count comes from its registry).
    """

    #: emitted with the resolved path whenever a database opens
    database_opened = Signal(str)
    #: emitted after every query / export job settles (ok or not)
    job_finished = Signal(bool)
    #: emitted whenever edit mode is armed or disarmed
    edit_mode_changed = Signal(bool)
    #: internal relay: (job id, ok). See :meth:`_run_job` — this is what
    #: drags a worker-thread completion back onto the GUI thread.
    _job_settled = Signal(int, bool)
    #: internal relay: the id of a job whose thread has exited. Same
    #: reason as :attr:`_job_settled` — ``QThread.finished`` is emitted
    #: in the worker thread, and ``self._jobs`` must only ever be
    #: mutated on the GUI thread. An *id* rather than the QThread
    #: itself: by the time a queued delivery lands, Qt's own
    #: ``deleteLater`` may already have taken the C++ object away.
    _thread_retired = Signal(int)

    def __init__(self, parent=None, threaded: bool = True):
        """Build the browser: the table list, the preview and the controls.

        :param parent: parent widget.
        """
        super().__init__(parent)
        self.app_key = "db_browser"
        self._threaded = bool(threaded)
        self._db: Optional[StoreDb] = None
        #: The columns a store table reads per page (None reads them all).
        self._fetch_columns: Optional[List[str]] = None
        self._table: str = ""
        self._all_columns: List[str] = []
        self._where: Optional[str] = None
        self._params: tuple = ()
        self._filter_label: str = ""

        self._token: int = 0
        #: ``(column, descending)`` while a header sort is active, else None.
        #: The sort happens in SQL over the WHOLE table, so it is correct
        #: however little of the table the view has loaded.
        self._sort: Optional[Tuple[str, bool]] = None
        #: A column a seed asked to show that no chunk has arrived for yet.
        #: The first chunk is what defines the column set, so a seed handled
        #: before it lands has nothing to scroll to; it is consumed once the
        #: columns exist. Synchronous loads never set it.
        self._seed_column: Optional[str] = None
        self._loaded: int = 0
        self._last_key: Optional[tuple] = None
        self._exhausted: bool = False
        self._exact_count: Optional[int] = None
        self._estimate: Optional[int] = None
        self.auto_count: bool = True

        self._export_busy: bool = False
        self._load_jobs: int = 0
        self._chunk_jobs: int = 0
        self._jobs: Dict[int, tuple] = {}
        self._thread = None
        self._worker = None
        self._count_thread = None
        self._pending: Dict[int, tuple] = {}
        self._queue: List[tuple] = []
        self._next_job_id: int = 0
        self._job_settled.connect(self._on_job_settled)
        self._thread_retired.connect(self._retire_job)
        self.last_error: str = ""

        self._edit_mode: bool = False
        self._edit_path: str = ""
        self._explicit_path: str = ""
        self._suppress_edit_signal: bool = False
        self.last_edit_sql: str = ""
        self.confirm_edit_mode: Callable[[str], bool] = self._default_confirm

        self._syncing_selection: bool = False
        #: Model rows the shared filter hides, by row index.
        self._linked_hidden: set = set()
        #: Appended to the status line while the shared filter is narrowing
        #: what this table shows, or explaining why it could not be applied.
        self._linked_filter_note: str = ""

        self._build_ui()
        from ..dnd import install_dropzone
        from ..dnd_handlers import DatabaseDropHandler
        install_dropzone(self, DatabaseDropHandler(), self)
        self._set_status(
            "Choose a measurements.db, or a run folder containing "
            "measurements/measurements.db.")
        self._update_controls()
        self.link_selection("db_browser")
        from .settings_model import retarget_field_tooltips
        retarget_field_tooltips(self)


    def _build_ui(self) -> None:
        """Lay out the table list, the preview grid and the action row."""
        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACING["lg"], SPACING["lg"],
                                 SPACING["lg"], SPACING["lg"])
        outer.setSpacing(SPACING["md"])

        header = ModuleHeader(
            "Database Browser",
            description="Browse, filter and export tables from "
                        "measurements.db",
            app_key="db_browser",
        )
        self._header = header
        outer.addWidget(header)

        subtitle = QLabel(
            tr("Read-only by default. spaCR reads the project through its "
               "store, so nothing here can modify your measurements."))
        subtitle.setObjectName("Muted")
        subtitle.setWordWrap(True)
        outer.addWidget(subtitle)

        outer.addWidget(Divider())

        self._path_edit = QLineEdit(self)
        self._path_edit.setPlaceholderText("measurements.db")
        self._path_edit.setToolTip("…/measurements/measurements.db  — or a run folder")
        self._path_edit.setClearButtonEnabled(True)
        self._path_edit.returnPressed.connect(self._on_open_typed_path)
        self._btn_pick_db = QPushButton("Choose database…", self)
        self._btn_pick_db.clicked.connect(self._pick_database)
        self._btn_pick_src = QPushButton("Choose run folder…", self)
        self._btn_pick_src.clicked.connect(self._pick_run_folder)
        self._btn_open = QPushButton("Open", self)
        self._btn_open.clicked.connect(self._on_open_typed_path)
        from ..theme import stack_when_narrow

        src_buttons = QHBoxLayout()
        src_buttons.setSpacing(SPACING["sm"])
        src_buttons.addWidget(self._btn_pick_db)
        src_buttons.addWidget(self._btn_pick_src)
        src_buttons.addWidget(self._btn_open)
        install_test_data_button(
            self, src_buttons, self._open_the_example,
            say=lambda message: self._set_status(message, error=True))
        outer.addWidget(stack_when_narrow(self._path_edit, src_buttons,
                                          SPACING["sm"], self))

        edit_row = QHBoxLayout()
        edit_row.setSpacing(SPACING["sm"])
        self._edit_check = Toggle("Edit mode", self)
        self._edit_check.setToolTip(
            "Off by default. Ticking this asks for confirmation before "
            "spaCR opens a read-write connection; every change is one "
            "UPDATE scoped to one row, and there is no undo.")
        self._edit_check.toggled.connect(self._on_edit_toggled)
        self._edit_note = QLabel("", self)
        self._edit_note.setObjectName("Muted")
        self._edit_note.setWordWrap(True)
        edit_row.addWidget(self._edit_check)
        edit_row.addWidget(self._edit_note, 1)
        outer.addLayout(edit_row)

        split = CollapsibleSplitter(Qt.Horizontal, self,
                                    persist_key="db_browser::body")

        self._table_list = QListWidget()
        self._table_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table_list.currentItemChanged.connect(self._on_table_selected)
        split.add_section(self._table_list, "Tables",
                          persist_key="db_browser/Tables", stretch=0,
                          extent=220)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(SPACING["xs"])

        col_row = QHBoxLayout()
        col_row.setSpacing(SPACING["sm"])
        self._col_search = QLineEdit(right)
        self._col_search.setPlaceholderText(
            "Search columns…  e.g. 'percentile' or 'channel_1'")
        self._col_search.setClearButtonEnabled(True)
        self._column_search_timer = QTimer(self)
        self._column_search_timer.setSingleShot(True)
        self._column_search_timer.setInterval(_COLUMN_SEARCH_DELAY_MS)
        self._column_search_timer.timeout.connect(self._apply_column_search)
        self._col_search.textChanged.connect(self._on_column_search_typed)
        self._col_search.editingFinished.connect(self._flush_column_search)
        self._col_count_label = QLabel("", right)
        self._col_count_label.setObjectName("Muted")
        col_row.addWidget(self._col_search, 1)
        col_row.addWidget(self._col_count_label)
        right_layout.addLayout(col_row)

        self._model = PreviewModel(self)
        self._model.set_fetch_hook(self.fetch_more)
        self._model.set_commit_hook(self.edit_cell)
        self._view = QTableView(right)
        self._view.setObjectName("DbBrowserPreview")
        self._view.setModel(self._model)
        self._view.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._view.setSelectionBehavior(QAbstractItemView.SelectItems)
        self._view.setAlternatingRowColors(True)
        self._view.setSortingEnabled(False)
        header = self._view.horizontalHeader()
        header.setSectionsClickable(True)
        header.sectionClicked.connect(self._on_header_clicked)
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setStretchLastSection(False)
        header.setDefaultSectionSize(150)
        self._view.selectionModel().selectionChanged.connect(
            self._on_view_selection_changed)
        self._model.modelReset.connect(self._apply_linked_filter)
        self._model.rowsInserted.connect(self._apply_linked_filter)
        right_layout.addWidget(self._view, 1)

        page_row = QHBoxLayout()
        page_row.setSpacing(SPACING["sm"])
        self._btn_more = QPushButton("Load more", right)
        self._btn_more.setToolTip(
            "Fetch the next chunk. Scrolling to the bottom does this "
            "automatically.")
        self._btn_more.clicked.connect(self.fetch_more)
        self._rows_label = QLabel("", right)
        self._rows_label.setObjectName("Muted")
        self._sort_note = QLabel("", right)
        self._sort_note.setObjectName("Muted")
        self._sort_note.setWordWrap(True)
        self._page_size_box = QSpinBox(right)
        self._page_size_box.setRange(*PAGE_SIZE_RANGE)
        self._page_size_box.setSingleStep(25)
        self._page_size_box.setValue(DEFAULT_PAGE_SIZE)
        self._page_size_box.setToolTip("(int) Rows fetched per chunk.")
        self._page_size_box.valueChanged.connect(self._on_page_size_changed)
        page_row.addWidget(self._btn_more)
        page_row.addWidget(self._rows_label)
        page_row.addWidget(self._sort_note, 1)
        page_row.addWidget(QLabel("Rows / fetch", right))
        page_row.addWidget(self._page_size_box)
        right_layout.addLayout(page_row)

        split.add_section(right, "Table preview",
                          persist_key="db_browser/Table preview")
        outer.addWidget(split, 1)
        self._body_splitter = split

        filt_row = QHBoxLayout()
        filt_row.setSpacing(SPACING["sm"])
        filt_row.addWidget(QLabel("Filter", self))
        self._filter_col = QComboBox(self)
        self._filter_col.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self._filter_col.setMinimumWidth(180)
        self._filter_op = QComboBox(self)
        self._filter_op.addItems(list(OPERATORS.keys()))
        self._filter_op.setAccessibleName(tr("Comparison"))
        self._filter_value = QLineEdit(self)
        self._filter_value.setPlaceholderText("value")
        self._filter_value.returnPressed.connect(self.apply_filter)
        self._raw_toggle = Toggle("raw SQL", self)
        self._raw_toggle.setToolTip(
            "Type a WHERE predicate yourself, e.g. "
            "cell_area > 1000 AND well LIKE 'A%'")
        self._raw_edit = QLineEdit(self)
        self._raw_edit.setPlaceholderText("cell_area > 1000 AND well LIKE 'A%'")
        self._raw_edit.setVisible(False)
        self._raw_edit.returnPressed.connect(self.apply_filter)
        self._btn_apply = QPushButton("Apply filter", self)
        self._btn_apply.clicked.connect(self.apply_filter)
        self._btn_clear = QPushButton("Clear", self)
        self._btn_clear.clicked.connect(self.clear_filter)
        self._btn_export = QPushButton(tr("Export filtered rows…"), self)
        self._btn_export.clicked.connect(self._pick_export_path)
        self._btn_merge = QPushButton(tr("Merge tables…"), self)
        self._btn_merge.setToolTip(tr(
            "Stack or join tables from this and other projects, and save "
            "the result as a new table."))
        self._btn_merge.clicked.connect(self.open_merge)
        self._merge_dialog: Optional[_MergeDialog] = None
        for w in (self._filter_col, self._filter_op, self._filter_value,
                  self._raw_edit):
            w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        filt_row.addWidget(self._filter_col, 2)
        filt_row.addWidget(self._filter_op, 1)
        filt_row.addWidget(self._filter_value, 2)
        filt_row.addWidget(self._raw_edit, 4)
        outer.addLayout(filt_row)

        from ..widgets.flow import FlowHost, FlowLayout

        action_host = FlowHost(self)
        action_row = FlowLayout(action_host, spacing=SPACING["sm"])
        for w in (self._raw_toggle, self._btn_apply, self._btn_clear,
                  self._btn_export, self._btn_merge):
            action_row.addWidget(w)
        outer.addWidget(action_host)

        self._sql_label = QLabel("", self)
        self._sql_label.setObjectName("Muted")
        self._sql_label.setWordWrap(True)
        self._sql_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        outer.addWidget(self._sql_label)

        self._status = QLabel("", self)
        self._status.setObjectName("Muted")
        self._status.setWordWrap(True)
        self._status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        outer.addWidget(self._status)

        self._filter_op.currentTextChanged.connect(
            lambda *_: self._update_controls())
        self._raw_toggle.toggled.connect(lambda *_: self._update_controls())


    def _set_status(self, text: str, error: bool = False) -> None:
        """Report inline. Deliberately never a QMessageBox — a modal dialog
        would hang a headless run (and did, in MakeMasksScreen)."""
        self.last_error = text if error else ""
        palette = active_palette()
        colour = palette["error"] if error else palette["fg_muted"]
        self._status.setStyleSheet(f"color: {colour};")
        self._status.setText(text)

    def status_text(self) -> str:
        """Current inline status message (test/introspection helper)."""
        return self._status.text()

    def sql_text(self) -> str:
        """The statement shown to the user before it runs (or ``''``)."""
        return self._sql_label.text()


    def _pick_database(self) -> None:
        """Ask for a database file and open it."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open measurements database", "",
            "spaCR projects (spacr.duckdb *.db *.sqlite *.sqlite3);;"
            "All files (*)")
        if path:
            self.set_database(path)

    def _pick_run_folder(self) -> None:
        """Ask for a run folder and open the database inside it."""
        path = QFileDialog.getExistingDirectory(self, "Choose a run folder", "")
        if path:
            self.set_database(path)

    def _on_open_typed_path(self) -> None:
        """Open whatever path the user typed."""
        self.set_database(self._path_edit.text())

    def _open_the_example(self, _folder, database) -> None:
        """Open the example plate's database, browse-only.

        Opened with ``explicit=False``: the example is spaCR's choice rather
        than the user's, and :meth:`set_database` keeps such a database out
        of edit mode, so a demonstration cannot write into the cached copy
        every other screen's example reads.

        The per-cell table is selected rather than the first one listed,
        which alphabetically is an empty annotation log.

        :param _folder: the example plate folder.
        :param database: its ``measurements/measurements.db``.
        """
        if not self.set_database(str(database), explicit=False):
            return
        names = self.tables()
        if EXAMPLE_TABLE in names:
            self._table_list.setCurrentRow(names.index(EXAMPLE_TABLE))

    def set_database(self, path: str, explicit: bool = True) -> bool:
        """Open ``path`` read-only and list its tables.

        Accepts an older database file, a spaCR store project, or a run
        ``src`` folder holding either. Any problem (missing file, not a
        database, unreadable) is reported in the status label and returns
        ``False`` — this never raises.

        Always resets edit mode to off: a database the user armed for
        editing is *that* database, never the next one.

        :param path: database file, store project, or run ``src`` directory
            holding one; resolved by :func:`resolve_db_path` and opened
            through :class:`spacr.store_browse.StoreDb`.
        :param explicit: True when the user chose this database
            themselves. Pass False when spaCR opens one on their behalf
            (a remembered path, a folder handed over by another screen);
            such a database can be browsed but never edited.
        :returns: True when a database was opened.
        """
        self.disable_edit_mode(quiet=True)
        self._token += 1
        self._model.set_more(False)
        self._model.clear()
        _reuse_list_items(self._table_list, 0)
        self._db = None
        self._table = ""
        self._all_columns = []
        self._explicit_path = ""
        self._reset_load_state()
        self._update_rows_label()
        self._where, self._params, self._filter_label = None, (), ""
        self._filter_col.clear()
        self._sql_label.setText("")
        try:
            db = StoreDb(resolve_db_path(path))
        except Exception as e:
            self._set_status(self._humanise(e, path), error=True)
            self._update_controls()
            return False
        self._db = db
        if explicit:
            self._explicit_path = db.path
        self._path_edit.setText(db.path)
        try:
            tables = db.tables(refresh=True)
        except Exception as e:
            self._set_status(self._humanise(e, path), error=True)
            self._update_controls()
            return False
        for item, name in zip(
                _reuse_list_items(self._table_list, len(tables)), tables):
            item.setText(name)
        if not tables:
            self._set_status(
                f"Opened {db.path} — but it has no tables.", error=False)
            self._update_controls()
            self.database_opened.emit(db.path)
            return True
        self._set_status(
            f"Opened {db.path} (read-only) — {len(tables)} "
            f"table{'s' if len(tables) != 1 else ''}.")
        self.database_opened.emit(db.path)
        self._table_list.setCurrentRow(0)
        self._update_controls()
        return True

    @staticmethod
    def _humanise(exc: Exception, path: str) -> str:
        """Turn a store, SQLite or OS error into one line a biologist can act on."""
        msg = str(exc).strip() or exc.__class__.__name__
        kind = type(exc).__name__
        if kind in ("DatabaseError", "Error") and "not a database" in msg.lower():
            return (f"{path} is not a SQLite database "
                    f"(sqlite said: {msg}).")
        if kind == "OperationalError" and "unable to open" in msg.lower():
            return f"Could not open {path} — {msg}."
        return msg

    def database_path(self) -> str:
        """Path of the open database, or ``''``."""
        return self._db.path if self._db is not None else ""

    def tables(self) -> List[str]:
        """Table names currently listed in the sidebar."""
        return [self._table_list.item(i).text()
                for i in range(self._table_list.count())]


    def _on_table_selected(self, current, _previous=None) -> None:
        """Load the first page of the selected table."""
        if current is None:
            return
        self.select_table(current.text())

    def select_table(self, name: str) -> bool:
        """Make ``name`` the previewed table and reset load and filter state.

        :param name: live table or view name from the open database.
        :returns: whether the schema was read and the table load was started.
        """
        if self._db is None:
            self._set_status("No database open.", error=True)
            return False
        try:
            known = self._db.tables()
        except Exception as e:
            self._set_status(self._humanise(e, name), error=True)
            return False
        if name not in known:
            self._set_status(tr(
                "No table named {table!r} in {database}. Reload the database and "
                "choose a table from its table list, or open the database that "
                "contains the required table.",
                table=name,
                database=os.path.basename(self._db.path)), error=True)
            return False
        try:
            self._all_columns = self._db.columns(name)
            self._fetch_columns = self._db.fetch_columns(
                name, self._col_search.text())
        except Exception as e:
            self._set_status(self._humanise(e, name), error=True)
            return False
        self._table = name
        self._clear_sort()
        self._where, self._params, self._filter_label = None, (), ""
        self._raw_edit.clear()
        self._filter_value.clear()
        self._filter_col.blockSignals(True)
        self._filter_col.clear()
        self._filter_col.addItems(self._all_columns)
        self._filter_col.blockSignals(False)
        for i in range(self._table_list.count()):
            if self._table_list.item(i).text() == name:
                if self._table_list.currentRow() != i:
                    self._table_list.blockSignals(True)
                    self._table_list.setCurrentRow(i)
                    self._table_list.blockSignals(False)
                break
        self.refresh()
        return True

    def current_table(self) -> str:
        """Which table the browser is showing.

        :returns: the table's name, or ``""`` when none is open.
        """
        return self._table


    def set_column_filter(self, text: str) -> None:
        """Narrow the displayed columns to those containing ``text``.

        On an older database this is purely a view operation — it is not
        queried again, even on a table with hundreds of feature columns. A
        store project reads only the columns it shows, so when the search
        asks for columns not yet read the visible rows are read again with
        them. An empty string restores every column.

        :param text: case-insensitive substring required in visible names.
        """
        text = text or ""
        if self._col_search.text() != text:
            self._col_search.setText(text)
        self._column_search_timer.stop()
        self._model.set_column_filter(text)
        self._update_column_count()
        self._autosize_columns()
        if self._db is not None and self._table and self._db.is_store():
            wanted = self._db.fetch_columns(self._table, text)
            if wanted != self._fetch_columns:
                self._fetch_columns = wanted
                self.refresh()

    def _on_column_search_typed(self, _text: str = "") -> None:
        """Wait for a pause in typing before filtering the columns."""
        self._column_search_timer.start()

    def _flush_column_search(self) -> None:
        """Apply a typed column search that is still waiting for its pause."""
        if self._column_search_timer.isActive():
            self._column_search_timer.stop()
            self._apply_column_search()

    def _apply_column_search(self) -> None:
        """Filter the columns by what the search box holds now."""
        text = self._col_search.text()
        if text != self._model.column_filter():
            self.set_column_filter(text)

    def visible_columns(self) -> List[str]:
        """Column names currently shown in the preview."""
        self._flush_column_search()
        return self._model.visible_columns()

    def preview_columns(self) -> List[str]:
        """Every column of the loaded rows, ignoring the column search."""
        return self._model.all_columns()

    def preview_rows(self) -> List[tuple]:
        """Rows loaded so far, as tuples in schema-column order."""
        return self._model.rows()

    def _autosize_columns(self) -> None:
        """Fit columns to their content, but only when there aren't many.

        A 500-column feature table keeps the fixed default width — the
        user narrows it with the column search first.
        """
        count = self._model.columnCount()
        if not 0 < count <= AUTOSIZE_MAX_COLUMNS:
            return
        view = self._view
        header = view.horizontalHeader()
        metrics = view.fontMetrics()
        pad = None
        for column in range(count):
            texts = self._model._widest_texts(column)
            width = header.sectionSizeHint(column)
            if texts:
                if pad is None:
                    pad = self._cell_padding(texts[0], column)
                width = max(width, pad + max(
                    metrics.horizontalAdvance(t) for t in texts))
            if header.sectionSize(column) != width:
                header.resizeSection(column, width)

    def _cell_padding(self, text: str, column: int) -> int:
        """What a cell adds around its text, read off one real cell."""
        rows = self._model.rowCount()
        if rows:
            hint = self._view.sizeHintForIndex(self._model.index(0, column))
            first = self._model.data(self._model.index(0, column))
            if first is not None:
                extra = 1 if self._view.showGrid() else 0
                return max(0, hint.width() - self._view.fontMetrics(
                ).horizontalAdvance(str(first))) + extra
        return 2 * (self._view.style().pixelMetric(
            QStyle.PM_FocusFrameHMargin, None, self._view) + 1) + 1

    def _update_column_count(self) -> None:
        """Say how many columns the filter is letting through."""
        total = len(self._model.all_columns())
        if self._fetch_columns is not None and self._all_columns:
            total = len(self._all_columns)
        shown = len(self._model.visible_columns())
        if not total:
            self._col_count_label.setText("")
        elif shown == total:
            self._col_count_label.setText(f"{total} columns")
        else:
            self._col_count_label.setText(f"{shown} of {total} columns")


    def page_size(self) -> int:
        """Rows fetched per chunk."""
        return int(self._page_size_box.value())

    def loaded_rows(self) -> int:
        """How many rows have been fetched so far."""
        return self._loaded

    def row_count(self) -> int:
        """Best known total for the current table + filter.

        Exact once ``COUNT(*)`` has landed or the table has been read to
        the end; otherwise the ``max(rowid)`` estimate, or the number of
        rows loaded so far. Ask :meth:`row_count_is_estimate` before
        quoting it as a fact.
        """
        if self._exact_count is not None:
            return self._exact_count
        if self._estimate is not None:
            return self._estimate
        return self._loaded

    def row_count_is_estimate(self) -> bool:
        """True while :meth:`row_count` is not known to be exact."""
        return self._exact_count is None and not self._exhausted

    def is_fully_loaded(self) -> bool:
        """True when every row of the current table + filter is in memory."""
        return self._exhausted

    def apply_seed(self, seed: Dict[str, Any]) -> None:
        """Open a database (and optionally a table) another screen sent here.

        The generic hand-off seam ``MainWindow._on_train_requested`` looks
        for. Everything is optional and anything unusable is ignored rather
        than raised: this is a convenience jump, so the screen still opens
        when a seed is stale, possibly on a different table.

        :param seed: ``db_path``, and optionally ``table`` and ``column``.
        """
        path = seed.get("db_path") or seed.get("path")
        if path and not self.set_database(str(path)):
            return
        table = seed.get("table")
        if table:
            try:
                tables = self._db.tables() if self._db else []
            except Exception:
                tables = []
            if table in tables:
                self.select_table(table)
        column = seed.get("column")
        if column:
            if not self._scroll_column_into_view(column):
                if not self.visible_columns():
                    self._seed_column = str(column)

    def _scroll_column_into_view(self, column) -> bool:
        """Bring ``column`` into view. False when this table has no such column.

        :param column: column name to reveal.
        :returns: whether the column was found and scrolled to.
        """
        try:
            section = self.visible_columns().index(str(column))
        except (ValueError, AttributeError):
            return False
        self._view.scrollTo(self._model.index(0, section))
        return True

    def _clear_sort(self) -> None:
        """Forget the sort. Called when the TABLE changes, not on refresh.

        A column name is meaningless in a different table, and carrying one
        across would put an unknown column in an ORDER BY -- rejected by
        check_columns, so the table would simply fail to load.
        """
        self._sort = None
        header = self._view.horizontalHeader()
        header.setSortIndicatorShown(False)
        header.setSortIndicator(-1, Qt.AscendingOrder)

    def _reset_load_state(self) -> None:
        """Forget the current page, count and cursor.

        Called when the TABLE changes: a row count belonging to the previous
        table would be shown against this one until the new count arrives.
        """
        self._loaded = 0
        self._last_key = None
        self._exhausted = False
        self._exact_count = None
        self._estimate = None

    def _on_page_size_changed(self, _value: int) -> None:
        """Reload with a different number of rows per page.

        :param _value: the new page size; re-read from the widget.
        """
        self.refresh()

    def refresh(self) -> None:
        """Abandon any load in flight and read the first chunk again."""
        if self._db is None or not self._table:
            return
        self._token += 1
        self._db.cancel()
        self._db.sync()
        self._reset_load_state()
        self._model.set_more(False)
        self._model.clear()
        self._view.setSortingEnabled(False)
        header = self._view.horizontalHeader()
        if self._sort is None:
            header.setSortIndicatorShown(False)
            header.setSortIndicator(-1, Qt.AscendingOrder)
        else:
            try:
                section = self._model.visible_columns().index(self._sort[0])
            except ValueError:
                section = -1
            header.setSortIndicatorShown(section >= 0)
            header.setSortIndicator(
                section,
                Qt.DescendingOrder if self._sort[1] else Qt.AscendingOrder)
        self._update_rows_label()
        self._update_sort_state()
        self._fetch_chunk(self._token, first=True)

    def fetch_more(self) -> bool:
        """Fetch the next chunk. Called by the view when it scrolls to the end.

        Returns False when there is nothing to fetch or a chunk is
        already in flight — a scroll must never queue ten of them.
        """
        if self._db is None or not self._table or self._exhausted:
            return False
        if self._chunk_jobs:
            return False
        self._fetch_chunk(self._token, first=False)
        return True

    def _fetch_chunk(self, token: int, first: bool) -> None:
        """Read the next page on a worker."""
        db, table = self._db, self._table
        where, params = self._where, self._params
        limit = self.page_size()
        after = None if first else self._last_key
        loaded = 0 if first else self._loaded
        order_by = self._sort
        columns = self._fetch_columns
        want_estimate = first and not where

        def _job() -> Dict[str, Any]:
            """Read one chunk of the table. Off the GUI thread."""
            cols, rows, keys = db.chunk(table, columns=columns, where=where,
                                        params=params, limit=limit,
                                        after=after, loaded=loaded,
                                        order_by=order_by)
            exact = db.registry_count(table) if want_estimate else None
            estimate = (db.estimate_count(table)
                        if want_estimate and exact is None else None)
            return {"token": token, "first": first, "columns": cols,
                    "rows": rows, "keys": keys, "limit": limit,
                    "estimate": estimate, "exact": exact}

        self._model.set_more(False)
        self._run_job(_job, self._apply_chunk, kind="chunk", token=token)

    def _apply_chunk(self, result: Dict[str, Any]) -> None:
        """Paint a chunk — unless the user has moved on since it was asked for."""
        if not result or result.get("token") != self._token:
            return
        first = bool(result.get("first"))
        columns = result.get("columns", [])
        rows = result.get("rows", [])
        keys = result.get("keys", [])
        limit = int(result.get("limit", self.page_size()))
        if first:
            self._model.set_page(columns, rows, row_offset=0, keys=keys)
            self._estimate = result.get("estimate")
            if result.get("exact") is not None:
                self._exact_count = int(result["exact"])
        else:
            self._model.append_rows(rows, keys)
        self._loaded += len(rows)
        if keys and keys[-1] is not None:
            self._last_key = keys[-1]
        if len(rows) < limit or (self._exact_count is not None
                                 and self._loaded >= self._exact_count):
            self._exhausted = True
            if len(rows) < limit:
                self._exact_count = self._loaded
        self._model.set_more(not self._exhausted)
        if first:
            self._update_column_count()
            self._autosize_columns()
            if self._seed_column is not None:
                pending, self._seed_column = self._seed_column, None
                self._scroll_column_into_view(pending)
        self._update_rows_label()
        self._update_sort_state()
        self._report_table_status()
        if (first and not self._exhausted and self.auto_count
                and self._exact_count is None):
            self._start_count(self._token)

    def refresh_count(self) -> None:
        """Replace the estimate with an exact ``COUNT(*)``."""
        if self._db is None or not self._table:
            return
        self._start_count(self._token)

    def _start_count(self, token: int) -> None:
        """Count the table's rows on a worker.

        SEPARATE FROM THE PAGE FETCH because a COUNT on a large table is slow
        and the first page is not: waiting for the count to show any rows
        would make a fast read feel like a hang.
        """
        db, table = self._db, self._table
        where, params = self._where, self._params

        def _job() -> Dict[str, Any]:
            """Count the matching rows, carrying the token that ordered it.

            The token comes back with the answer so a slow count from a query the
            user has since changed can be recognised and dropped.
            """
            return {"token": token, "count": db.count(table, where, params)}

        self._run_job(_job, self._apply_count, kind="count", token=token)

    def _apply_count(self, result: Dict[str, Any]) -> None:
        """Show a row count that has come back.

        :param result: the finished count.
        """
        if not result or result.get("token") != self._token:
            return
        self._exact_count = int(result.get("count", 0))
        if self._loaded >= self._exact_count:
            self._exhausted = True
            self._model.set_more(False)
        self._update_rows_label()
        self._update_sort_state()
        self._report_table_status()

    def _count_text(self) -> str:
        """How many rows there are — never guessing without saying so."""
        if self._exact_count is not None:
            return f"{self._exact_count:,} rows"
        if self._estimate is not None:
            return f"≈{self._estimate:,} rows (estimate)"
        return "an unknown number of rows (counting…)"

    def _update_rows_label(self) -> None:
        """Say which rows are on screen, and out of how many."""
        if self._db is None or not self._table:
            self._rows_label.setText("")
            return
        self._rows_label.setText(
            f"showing {self._loaded:,} of {self._count_text()}")

    def _update_sort_state(self) -> None:
        """Describe the sort. Click-to-sort is always available.

        It used to be switched on only once ``self._exhausted`` -- the whole
        table in memory -- because Qt's own ``setSortingEnabled`` sorts the
        MODEL, and sorting the rows fetched so far and presenting that as the
        table's order is the one option worse than not sorting at all. On a
        400 k-row measurement table that moment never arrives interactively,
        so the feature was effectively absent from the tables that most need
        it.

        Sorting in SQL removes the trade-off: the ORDER BY runs over the
        whole table whatever the view has loaded, so the first row shown is
        the first row of the sorted table and not the smallest of the first
        hundred. Qt's model sort stays OFF -- the header click is handled by
        :meth:`_on_header_clicked` instead.
        """
        self._view.setSortingEnabled(False)
        if self._sort is None:
            self._sort_note.setText(
                "Click a column header to sort the whole table.")
            return
        column, descending = self._sort
        note = (
            f"sorted by {column} {'descending' if descending else 'ascending'}"
            " — whole table, in SQL. Click again to reverse, a third time to "
            "clear.")
        if self._db is not None and self._db.is_store():
            note += " " + tr("Rows where this column is empty are left out "
                             "while it is sorted.")
        self._sort_note.setText(note)

    def _on_header_clicked(self, section: int) -> None:
        """Cycle the clicked column: descending, ascending, unsorted.

        The same cycle every other table in the application follows -- see
        :mod:`spacr.qt.widgets.sortable_table`. Descending first because a
        measurement table is read from the top and the top is where the
        largest value belongs; ascending second; and a third click back to
        the table's own order.

        That third state matters more here than anywhere else: sorting forces
        OFFSET paging, which gets slower the further the user scrolls, so
        there has to be a way back to the keyset-paged order without
        reloading the screen.
        """
        self._flush_column_search()
        columns = self._model.visible_columns()
        if not (0 <= section < len(columns)):
            return
        column = columns[section]

        if self._sort is None or self._sort[0] != column:
            self._sort = (column, True)
        elif self._sort[1]:
            self._sort = (column, False)
        else:
            self._sort = None

        header = self._view.horizontalHeader()
        if self._sort is None:
            header.setSortIndicatorShown(False)
        else:
            header.setSortIndicatorShown(True)
            header.setSortIndicator(
                section, Qt.DescendingOrder if self._sort[1] else Qt.AscendingOrder)

        self.refresh()

    def _report_table_status(self) -> None:
        """Say what the current table is doing: loading, counted, or failed."""
        bits = [f"{self._table}: {self._count_text()}",
                f"{len(self._model.all_columns())} columns"]
        if self._filter_label:
            bits.append(f"filter: {self._filter_label}")
        bits.append("edit mode" if self._edit_mode else "read-only")
        self._set_status(" · ".join(bits) + self._linked_filter_note)


    def _linked_frame(self, columns: Sequence[str]) -> pd.DataFrame:
        """The loaded rows as a frame, indexed by model row number.

        Only the columns asked for, and only those this table actually has:
        a measurement table is 500 columns wide and a shared filter names
        two of them. Building the whole page as a frame on every filter
        change would make a slider drag cost more than the query did.

        The positional index is load-bearing — it is what turns the filtered
        frame back into the row numbers to hide.

        Stamped with the table it came from, because the browser knows and
        the frame does not: without it, selecting nucleus 1 here published
        the same key as selecting pathogen 1, and the views that key was
        supposed to reach landed on whichever of the two they held. A table
        that is not one spaCR keys objects by — ``png_list``, a summary, a
        user's own — is left alone by ``with_object_type``, which is the
        right answer for something that has no object type.
        """
        available = self._model.all_columns()
        at = {c: i for i, c in enumerate(available)}
        wanted = [c for c in dict.fromkeys(columns) if c in at]
        rows = self._model.rows()
        frame = pd.DataFrame(
            {c: [row[at[c]] for row in rows] for c in wanted},
            index=range(len(rows)), columns=wanted)
        return with_object_type(frame, self._table)

    def _apply_linked_filter(self, *_args) -> None:
        """Hide the rows the shared filter excludes.

        Hiding rather than dropping: the model still holds every fetched row,
        so an edit made before the filter arrived still addresses the row it
        was made against, and clearing the filter costs no re-query.

        Degrades to hiding nothing when the filter names a column this table
        does not have — a filter carried over from the ``cell`` table to
        ``png_list`` is the common case, and an empty table is a worse answer
        than a complete one, PROVIDED the status line says which it is.
        """
        total = self._model.rowCount()
        hidden: set = set()
        note = ""
        try:
            data_filter = self.link.filter
        except Exception:
            data_filter = DataFilter()
        if total and not data_filter.is_empty:
            try:
                frame = self._linked_frame(
                    [c.column for c in data_filter.clauses])
                kept = {int(i) for i in self.linked_visible(frame).index}
                hidden = {row for row in range(total) if row not in kept}
                note = (f" · filtered: {data_filter.describe()} "
                        f"({total - len(hidden)} of {total} loaded rows)")
            except Exception as exc:
                hidden = set()
                note = f" · filter ignored ({exc.__class__.__name__})"
        previous, self._linked_hidden = self._linked_hidden, hidden
        if hidden or previous:
            for row in range(total):
                self._view.setRowHidden(row, row in hidden)
        changed = note != self._linked_filter_note
        self._linked_filter_note = note
        if changed and self._db is not None and self._table:
            self._report_table_status()

    def on_linked_filter_changed(self, data_filter: DataFilter) -> None:
        """Apply a newly published shared filter to the loaded rows.

        :param data_filter: current process-wide filter; the linked-view state
            already owns it, so this callback only refreshes row visibility.
        """
        self._apply_linked_filter()

    def hidden_rows(self) -> List[int]:
        """Model rows the shared filter is hiding, ascending."""
        return sorted(self._linked_hidden)

    def visible_rows(self) -> List[int]:
        """Model rows the shared filter keeps on screen, ascending."""
        return [row for row in range(self._model.rowCount())
                if row not in self._linked_hidden]


    def rows_for_selection(self, selection: Selection) -> List[int]:
        """The loaded rows ``selection`` names, ascending.

        Empty — not an error — for a table with no object identity in it
        (``png_list`` keyed on a path, a summary, a view). A shared selection
        does not identify those rows.

        :param selection: shared object identity selection to match.
        """
        if not selection.is_active or not self._model.rowCount():
            return []
        try:
            mask = selection.mask_for(self._linked_frame(OBJECT_KEY_COLUMNS))
        except Exception:
            return []
        return [row for row, keep in enumerate(mask) if keep]

    def selected_rows(self) -> List[int]:
        """The rows the user currently has selected, ascending."""
        model = self._view.selectionModel()
        if model is None:
            return []
        return sorted({index.row() for index in model.selectedIndexes()})

    def select_rows(self, rows: Sequence[int]) -> List[int]:
        """Select ``rows`` as a user would, publishing them to every view.

        :param rows: zero-based model row indices requested by the caller.
        :returns: the rows that were in range and got selected.
        """
        model = self._view.selectionModel()
        columns = self._model.columnCount()
        total = self._model.rowCount()
        wanted = sorted({int(r) for r in rows if 0 <= int(r) < total})
        if model is None or not columns:
            return []
        model.clearSelection()
        if wanted:
            model.select(self._selection_block(wanted),
                         QItemSelectionModel.Select)
        return wanted

    def _selection_block(self, rows: Sequence[int]) -> QItemSelection:
        """One whole-row-per-range selection covering ``rows``."""
        block = QItemSelection()
        last = self._model.columnCount() - 1
        for row in rows:
            block.select(self._model.index(row, 0),
                         self._model.index(row, last))
        return block

    def _on_view_selection_changed(self, *_args) -> None:
        """Publish what the user picked out.

        An *empty* selection is deliberately not published. Qt clears the
        view selection on every model reset — a new chunk, a column search, a
        sort — and publishing that as "the user selected nothing" would wipe
        a lasso drawn in the UMAP every time this screen loaded a page.
        Returning to the resting state is :meth:`clear_linked_selection`'s
        job, not a side effect of scrolling.
        """
        if self._syncing_selection:
            return
        rows = self.selected_rows()
        if not rows:
            return
        try:
            self.publish_selection(
                self._linked_frame(OBJECT_KEY_COLUMNS).iloc[rows])
        except Exception:
            return

    def on_linked_selection_changed(self, selection: Selection) -> None:
        """Select and scroll to the rows somebody else picked.

        Nothing is hidden: rows the selection does not name stay exactly
        where they are. Only the shared *filter* removes rows from view.

        :param selection: process-wide object selection published elsewhere.
        """
        model = self._view.selectionModel()
        if model is None or not self._model.columnCount():
            return
        rows = self.rows_for_selection(selection)
        self._syncing_selection = True
        try:
            model.clearSelection()
            if rows:
                model.select(self._selection_block(rows),
                             QItemSelectionModel.Select)
                self._view.scrollTo(self._model.index(rows[0], 0),
                                    QAbstractItemView.PositionAtCenter)
        finally:
            self._syncing_selection = False


    def apply_filter(self) -> bool:
        """Read the filter row, validate it, and reload from the first chunk.

        Returns False (and reports inline) when the filter is malformed.
        """
        if self._db is None or not self._table:
            self._set_status("Open a database and pick a table first.",
                             error=True)
            return False
        try:
            where, params, label = self._collect_filter()
        except Exception as e:
            self._set_status(f"Filter error: {e}", error=True)
            return False
        if where:
            try:
                self._db.validate_where(self._table, where, params)
            except Exception as e:
                self._set_status(f"Filter error: {e}", error=True)
                return False
        self._where, self._params, self._filter_label = where, params, label
        self.refresh()
        return True

    def clear_filter(self) -> None:
        """Drop the WHERE clause and reload."""
        self._where, self._params, self._filter_label = None, (), ""
        self._filter_value.clear()
        self._raw_edit.clear()
        self.refresh()

    def _collect_filter(self) -> Tuple[Optional[str], tuple, str]:
        """Return ``(where, params, human_label)`` from the filter widgets."""
        if self._raw_toggle.isChecked():
            raw = validate_raw_predicate(self._raw_edit.text())
            return raw, (), raw
        column = self._filter_col.currentText()
        op = self._filter_op.currentText()
        _, nargs, _ = OPERATORS[op]
        value = self._filter_value.text()
        if nargs and not value.strip():
            return None, (), ""
        where, params = build_where(column, op, value, self._all_columns)
        label = f"{column} {op}" + (f" {value}" if nargs else "")
        return where, params, label

    def set_filter(self, column: str, op: str, value: Any = "") -> bool:
        """Programmatic equivalent of filling in the filter row and applying.

        :param column: live schema column to filter.
        :param op: structured operator label from :data:`OPERATORS`.
        :param value: raw filter value; operators such as ``is null`` ignore it.
        :returns: whether validation succeeded and the reload was started.
        """
        self._raw_toggle.setChecked(False)
        idx = self._filter_col.findText(column)
        if idx < 0:
            self._set_status(
                f"Filter error: Unknown column {column!r} — "
                f"not in this table's schema.", error=True)
            return False
        self._filter_col.setCurrentIndex(idx)
        op_idx = self._filter_op.findText(op)
        if op_idx < 0:
            self._set_status(
                f"Filter error: Unknown operator {op!r}.", error=True)
            return False
        self._filter_op.setCurrentIndex(op_idx)
        self._filter_value.setText("" if value is None else str(value))
        return self.apply_filter()

    def set_raw_filter(self, predicate: str) -> bool:
        """Programmatic equivalent of the raw-SQL box and Apply.

        :param predicate: lone read-only WHERE-clause fragment.
        :returns: whether validation succeeded and the reload was started.
        """
        self._raw_toggle.setChecked(True)
        self._raw_edit.setText(predicate or "")
        return self.apply_filter()

    def where_clause(self) -> Optional[str]:
        """The active WHERE fragment, or None."""
        return self._where


    def editing_allowed_by_preference(self) -> bool:
        """Whether Preferences permits edit mode at all (read fresh)."""
        return bool(get_db_browser_editable())

    def edit_mode_enabled(self) -> bool:
        """True when edit mode is armed for the open database."""
        return self._edit_mode

    def _edit_block_reason(self) -> str:
        """Why edit mode cannot be armed right now — ``''`` when it can."""
        if self._db is None:
            return ("Open a database before turning on edit mode.")
        if not self.editing_allowed_by_preference():
            return ("Editing is turned off. Preferences → Database Browser → "
                    "'Allow editing in the Database Browser' has to be on "
                    "first — spaCR keeps this browser read-only by default "
                    "because there is no undo for an edited measurements.db.")
        if self._explicit_path != self._db.path:
            return (f"{os.path.basename(self._db.path)} was opened for you, "
                    f"not chosen by you. Pick it with 'Choose database…' "
                    f"before editing it.")
        if not self._db.can_edit():
            if not self._db.is_store():
                return tr(
                    "{name} is an older measurements.db, which the Database "
                    "Browser opens read-only. Convert the run to a spaCR store "
                    "project to record edits.",
                    name=os.path.basename(self._db.path))
            return tr(
                "This spaCR store project is read-only in the Database Browser "
                "for now: cell edits are recorded in the project's edit log, "
                "which this version of spaCR does not have yet.")
        return ""

    def _confirmation_text(self) -> str:
        """The words the user has to agree to before edit mode arms."""
        return tr(
            "Edit mode records changes to {path}.\n\n"
            "Each change is one edit row in the project's edit log, for one "
            "cell of one row, with your name and the time. The measured "
            "values in the tables are never overwritten; the newest edit is "
            "what spaCR shows, and an edit is undone by recording the old "
            "value again.\n\nTurn edit mode on?",
            path=self._db.path)

    def _default_confirm(self, message: str) -> bool:
        """Ask, out loud, before arming edit mode.

        The one dialog this screen owns, and it is deliberate: arming
        edit mode is the single action here that can destroy a
        measurements file, so it must not be possible to do by accident.
        Headless callers (and every test) replace
        :attr:`confirm_edit_mode` instead, so no automated run blocks.
        """
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Enable edit mode?")
        box.setText("Edit this database?")
        box.setInformativeText(message)
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
        box.setDefaultButton(QMessageBox.Cancel)
        return box.exec() == QMessageBox.Yes

    def _on_edit_toggled(self, checked: bool) -> None:
        """The checkbox *asks*; it does not decide."""
        if self._suppress_edit_signal:
            return
        if not checked:
            self.disable_edit_mode()
            return
        if not self.enable_edit_mode():
            self._set_edit_check(False)

    def _set_edit_check(self, checked: bool) -> None:
        """Turn editing on or off, and say so.

        :param checked: True to allow edits.
        """
        self._suppress_edit_signal = True
        try:
            self._edit_check.setChecked(bool(checked))
        finally:
            self._suppress_edit_signal = False

    def enable_edit_mode(self) -> bool:
        """Arm edit mode for the open database, if every guard allows it.

        Requires the Preferences opt-in, a database the user chose in
        this session, and an explicit confirmation. Returns False and
        explains inline otherwise; never raises.
        """
        reason = self._edit_block_reason()
        if reason:
            self._set_status(reason, error=True)
            self._set_edit_check(False)
            return False
        try:
            agreed = bool(self.confirm_edit_mode(self._confirmation_text()))
        except Exception as e:
            self._set_status(f"Edit mode not enabled: {e}", error=True)
            self._set_edit_check(False)
            return False
        if not agreed:
            self._set_status("Edit mode not enabled — still read-only.")
            self._set_edit_check(False)
            return False
        self._edit_mode = True
        self._edit_path = self._db.path
        self._set_edit_check(True)
        self._update_controls()
        self._set_status(
            tr("Edit mode is ON for {path}. Every change is one edit row "
               "for one cell of one row.", path=self._db.path))
        self.edit_mode_changed.emit(True)
        return True

    def disable_edit_mode(self, quiet: bool = False) -> None:
        """Drop back to read-only. Safe to call when already read-only.

        :param quiet: suppress the read-only status message when true.
        """
        was_on = self._edit_mode
        self._edit_mode = False
        self._edit_path = ""
        self._set_edit_check(False)
        self._update_controls()
        if was_on and not quiet:
            self._set_status("Edit mode is off — back to read-only.")
        if was_on:
            self.edit_mode_changed.emit(False)

    def edit_cell(self, row: int, column: str, text: Any) -> bool:
        """Write one cell of one row. Returns False, with a reason, if refused.

        This is the only path in the screen that can write, and it is
        also what :meth:`PreviewModel.setData` calls when a cell editor
        closes.

        :param row: zero-based loaded model row to address.
        :param column: schema column whose value the user edited.
        :param text: raw editor value to coerce against the declared type.
        :returns: whether exactly one database row was updated or already held
            the requested value.
        """
        if not self._edit_mode:
            self._set_status(
                "This database is open read-only — tick 'Edit mode' to "
                "change values.", error=True)
            return False
        if self._db is None or not self._table:
            self._set_status("Open a database and pick a table first.",
                             error=True)
            return False
        if self._edit_path != self._db.path:
            self._set_status(
                "Edit mode was armed for a different database — turning it "
                "off.", error=True)
            self.disable_edit_mode(quiet=True)
            return False
        _kind, key_columns = self._db.row_key(self._table)
        key = self._model.row_key(row)
        if not key_columns or key is None:
            self._set_status(tr(
                "{table!r} has no rowid and no primary key, so spaCR cannot limit"
                " an edit to one row. Choose a table with a rowid or primary key;"
                " keep this table in read-only mode.",
                table=self._table), error=True)
            return False
        if len(key) != len(key_columns):
            self._set_status(tr(
                "Row address is incomplete; the edit was refused. Reload the "
                "table, select the row again, and retry the edit."), error=True)
            return False
        types = self._db.column_types(self._table)
        if column not in types:
            self._set_status(
                f"Cannot edit {column!r}: it is not a column of "
                f"{self._table!r}.", error=True)
            return False
        try:
            value = coerce_for_column(text, types[column], column)
        except ValueError as e:
            self._set_status(f"Edit refused: {e}", error=True)
            return False
        if value == self._model.value(row, column):
            self._set_status(f"{column} is already that value — nothing "
                             f"written.")
            return True
        self._show_pending_sql(
            f"store.edit {self._table}.{column} where "
            f"{', '.join(key_columns)} = ?", (value,) + tuple(key))
        try:
            self._db.edit_cell(self._table, column, value, key)
        except EditRefused as e:
            self._set_status(f"Edit refused: {e}", error=True)
            return False
        except (*READ_ERRORS, ValueError) as e:
            self._set_status(f"Edit failed: {e}", error=True)
            return False
        self._model.set_value(row, column, value)
        self._set_status(tr(
            "Recorded 1 edit of {table} — {column} = {value}.",
            table=self._table, column=column, value=repr(value)))
        return True

    def _show_pending_sql(self, sql: str, params: Sequence[Any]) -> None:
        """Put the exact statement on screen *before* it runs."""
        self.last_edit_sql = sql
        self._sql_label.setText(f"About to run:  {sql}   ← {list(params)!r}")

    def pending_edit_sql(self) -> str:
        """The last statement shown to the user (test/introspection helper)."""
        return self.last_edit_sql

    def _table_is_editable(self) -> bool:
        """True when the current table offers a unique row address.

        Cheap after the first call — :meth:`spacr.store_browse.StoreDb.row_key`
        caches.
        """
        if self._db is None or not self._table:
            return False
        return bool(self._db.row_key(self._table)[1])

    def _update_edit_ui(self) -> None:
        """Keep the checkbox, the note and the view's edit triggers honest."""
        allowed = self.editing_allowed_by_preference()
        self._edit_check.setEnabled(self._db is not None and allowed)
        writable = self._edit_mode and self._table_is_editable()
        self._model.set_editable(writable)
        self._view.setEditTriggers(
            (QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
            if writable else QAbstractItemView.NoEditTriggers)
        if self._edit_mode and not writable:
            self._edit_note.setText(
                f"EDIT MODE is on, but {self._table or 'this table'} has no "
                f"row key — spaCR cannot address one row of it, so it stays "
                f"read-only.")
            return
        if self._edit_mode:
            self._edit_note.setText(tr(
                "EDIT MODE — each change is recorded as one edit row in the "
                "edit log of {name}; the measured tables are not overwritten.",
                name=os.path.basename(self._edit_path)))
            return
        if not allowed:
            self._edit_note.setText(
                "Read-only. Editing is off in Preferences → Database Browser.")
        elif self._db is None:
            self._edit_note.setText(
                "Read-only. Open a database you choose yourself to edit it.")
        elif not self._db.can_edit():
            self._edit_note.setText(tr(
                "Read-only. This project does not take edits in the Database "
                "Browser."))
        elif self._explicit_path != self._db.path:
            self._edit_note.setText(
                "Read-only — this database was opened for you. Re-open it "
                "with 'Choose database…' to edit it.")
        else:
            self._edit_note.setText(
                "Read-only. Tick 'Edit mode' to write to this file; you will "
                "be asked to confirm.")


    def _pick_export_path(self) -> None:
        """Ask where to write the exported rows, and in which format."""
        default = f"{self._table or 'export'}.csv"
        path, chosen = QFileDialog.getSaveFileName(
            self, "Export filtered rows", default,
            "CSV files (*.csv);;Parquet files (*.parquet);;"
            "SQLite databases (*.db);;All files (*)")
        if path:
            self.export_rows(path, _export_format(path, chosen))

    def open_merge(self) -> "_MergeDialog":
        """Show the Merge tables window, with the open project listed first.

        :returns: the window.
        """
        if self._merge_dialog is None:
            self._merge_dialog = _MergeDialog(self)
        elif self.database_path() and not self._merge_dialog.holds(
                self.database_path()):
            self._merge_dialog.add_source(self.database_path())
        self._merge_dialog.show()
        self._merge_dialog.raise_()
        return self._merge_dialog

    def export_csv(self, out_path: str) -> bool:
        """Write the current table + filter + visible columns to ``out_path``.

        Runs off the GUI thread and streams the result, so a
        400 000-row export neither freezes the window nor loads the table
        into memory. Reports inline on failure and returns ``False``.

        :param out_path: destination CSV path selected by the user.
        :returns: whether the export job was accepted for execution.
        """
        return self.export_rows(out_path, "csv")

    def export_rows(self, out_path: str, fmt: str = "csv") -> bool:
        """Write the current table, filter, sort and visible columns to a file.

        Streamed off the GUI thread through :func:`spacr.store.export`:
        CSV, Parquet (the stored types kept) or SQLite, where a table of
        2,000 columns or more is split into one table per feature family,
        each repeating the row keys. Reports inline on failure.

        :param out_path: the file to write.
        :param fmt: ``"csv"``, ``"parquet"`` or ``"sqlite"``.
        :returns: whether the export job was accepted for execution.
        """
        if self._db is None or not self._table:
            self._set_status("Open a database and pick a table first.",
                             error=True)
            return False
        if fmt not in ("csv", "parquet", "sqlite"):
            self._set_status(tr("Unknown export format {fmt}; choose CSV, "
                                "Parquet or SQLite.", fmt=fmt), error=True)
            return False
        if self._export_busy:
            self._set_status("An export is already running…", error=True)
            return False
        db, table = self._db, self._table
        where, params, order_by = self._where, self._params, self._sort
        self._flush_column_search()
        columns = self._model.visible_columns() or self._all_columns
        unknown = [c for c in columns if c not in set(self._all_columns)]
        if unknown:
            self._set_status(tr(
                "Unknown column(s) for {table!r}: {columns}. Reload the table and"
                " select columns shown in its column list.",
                table=table,
                columns=', '.join(map(repr, unknown))), error=True)
            return False

        def _job() -> Dict[str, Any]:
            """Write the file. Off the GUI thread.

            :returns: the rows written, the path and the column count.
            """
            n = db.export(out_path, table, columns=columns, where=where,
                          params=params, fmt=fmt, order_by=order_by)
            return {"exported": n, "path": out_path,
                    "columns": len(columns)}

        def _done(result: Dict[str, Any]) -> None:
            """Say how many rows were exported, and where.

            :param result: what the export job returned.
            """
            self._set_status(
                f"Exported {result['exported']:,} row"
                f"{'s' if result['exported'] != 1 else ''} × "
                f"{result['columns']} columns → {result['path']}")

        return self._run_job(_job, _done, kind="export")


    def _acquire(self, kind: str) -> None:
        """Take the database handle for a job, refusing a second one."""
        if kind == "export":
            self._export_busy = True
            return
        self._load_jobs += 1
        if kind == "chunk":
            self._chunk_jobs += 1

    def _release(self, kind: str) -> None:
        """Give the database handle back."""
        if kind == "export":
            self._export_busy = False
            return
        self._load_jobs = max(0, self._load_jobs - 1)
        if kind == "chunk":
            self._chunk_jobs = max(0, self._chunk_jobs - 1)

    def _run_job(self, fn: Callable[[], Any],
                 on_done: Callable[[Any], None],
                 kind: str = "chunk",
                 token: Optional[int] = None) -> bool:
        """Queue ``fn`` to run off the GUI thread; ``on_done`` takes its result.

        Uses :func:`spacr.qt.bridge.make_thread` — the same QThread +
        worker pairing the pipeline screens use — so there is exactly one
        threading idiom in the Qt layer. ``PipelineWorker`` calls
        ``fn(settings)``; we pass a private dict and let
        :func:`_capture_result` drop the return value in, since the
        worker's ``finished`` signal only carries a success flag.

        ``fn`` runs on a worker thread, so it reads through its own store
        cursor or SQLite connection — every
        :class:`spacr.store_browse.StoreDb` read does.

        **One worker per lane.** Jobs queue here and :meth:`_pump` starts
        them one by one on their lane: page reads, edits and exports on the
        main lane, ``COUNT(*)`` on its own (see :func:`_job_lane`), so a
        scroll is never held behind a full-table count. Output from each
        worker is routed by thread, so two lanes running at once keep their
        own ``print`` output. The jobs are read-only housekeeping, so they
        write no run-journal record and stay off Home's list of runs.

        Cancellation stays cooperative. A queued job carries the load
        token it was created for; if the user has moved on before it
        starts, :meth:`_pump` drops it without spending a thread. A job
        that is *already running* is never killed — it finishes, retires
        its own thread, and its result is discarded by the token check in
        :meth:`_apply_chunk`.

        With ``threaded=False`` (tests) the call runs inline and the same
        signals fire, so both paths behave identically from the outside.

        :param kind: ``"chunk"``, ``"count"`` or ``"export"`` — decides
            which busy counter the job holds.
        :param token: the load token this job belongs to, or ``None`` for
            a job (like an export) that no table switch invalidates.
        :returns: for the synchronous path, whether the job succeeded; for
            the threaded path, ``True`` once the job has been queued.
        """
        if not self._threaded:
            self._acquire(kind)
            ok = True
            try:
                result = fn()
            except Exception as e:
                self._release(kind)
                self._on_job_error(e)
                ok = False
            else:
                self._release(kind)
                try:
                    on_done(result)
                except Exception as e:
                    self._on_job_error(e)
                    ok = False
            self._update_controls()
            self.job_finished.emit(ok)
            return ok

        self._acquire(kind)
        self._queue.append((fn, on_done, kind, token))
        self._update_controls()
        self._pump()
        return True

    def _pump(self) -> None:
        """Start the next queued job, if no worker is running.

        Called after every enqueue and after every thread retires, so the
        queue drains without a timer. Jobs whose load token has been
        superseded are dropped here rather than started — a ``COUNT(*)``
        for a table the user has already left is pure waste.
        """
        for lane in ("main", "count"):
            if self._lane_thread(lane) is not None:
                continue
            index = 0
            while index < len(self._queue):
                fn, on_done, kind, token = self._queue[index]
                if _job_lane(kind) != lane:
                    index += 1
                    continue
                del self._queue[index]
                if token is not None and token != self._token:
                    self._release(kind)
                    continue
                self._start_job(fn, on_done, kind)
                break
        self._update_controls()

    def _lane_thread(self, lane: str):
        """The thread running on ``lane`` (``"main"`` or ``"count"``), if any."""
        return self._thread if lane == "main" else self._count_thread

    def _start_job(self, fn: Callable[[], Any],
                   on_done: Callable[[Any], None], kind: str) -> None:
        """Hand one queued job to a QThread.

        ``PipelineWorker.finished`` is emitted *in the worker thread*, and
        PySide6 invokes a plain closure connected to it directly, on that
        same thread — so a completion handler wired that way would touch
        widgets off the GUI thread (undefined behaviour, and it corrupts
        pytest-qt's ``waitSignal`` state into the bargain). The two tiny
        lambdas below are the only things that run there, and all they do
        is re-emit a signal, which is safe from any thread. Their
        receivers are *bound methods* of this widget, so Qt queues the
        calls onto the GUI thread where every other widget call lives.
        """
        box: Dict[str, Any] = {}
        thread, worker = make_thread(
            partial(_capture_result, fn), box, journal=False,
            user_visible=False, capture_figures=False)
        self._next_job_id += 1
        job_id = self._next_job_id
        self._jobs[job_id] = (thread, worker)
        if _job_lane(kind) == "count":
            self._count_thread = thread
        else:
            self._thread, self._worker = thread, worker
        self._pending[job_id] = (box, on_done, kind)
        worker.error.connect(self._on_worker_error_text)
        worker.finished.connect(
            lambda ok, jid=job_id: self._job_settled.emit(jid, bool(ok)))
        thread.finished.connect(self._retire_finished_jobs)
        self._update_controls()
        thread.start()

    def queued_jobs(self) -> int:
        """How many jobs are waiting for the worker to free up."""
        return len(self._queue)

    def _on_job_settled(self, job_id: int, ok: bool) -> None:
        """Finish one job by id. Always on the GUI thread.

        Bookkeeping (releasing the busy counter, retiring the pending
        entry) happens for *every* job, including one the user abandoned
        — otherwise a cancelled load would leave the screen permanently
        "busy". Whether the result is painted is a separate decision,
        made by :meth:`_apply_chunk` from the load token it carries.
        """
        entry = self._pending.pop(job_id, None)
        if entry is None:
            return
        box, on_done, kind = entry
        self._release(kind)
        ok = bool(ok)
        if ok and on_done is not None:
            try:
                on_done(box.get("result"))
            except Exception as e:
                self._on_job_error(e)
                ok = False
        self._update_controls()
        self.job_finished.emit(ok)

    def _retire_finished_jobs(self) -> None:
        """Retire every job whose QThread has stopped. GUI thread only.

        It sweeps rather than naming a sender: by the time this runs the
        emitting QThread may be exactly what is gone —
        ``thread.finished -> thread.deleteLater`` is connected first — and
        ``QObject.sender()`` is null for a queued call whose emitter was
        destroyed.
        """
        from ..bridge import thread_has_stopped

        for job_id, entry in list(self._jobs.items()):
            if thread_has_stopped(entry[0]):
                self._retire_job(job_id)

    def _retire_job(self, job_id: int) -> None:
        """Release *this* job's refs once its own event loop has exited.

        Releasing by job id matters: a plain "clear the refs" slot would
        drop the references of whichever job happens to be current when a
        previous thread finishes, and a QThread garbage-collected while it
        is still running takes the whole process down with it.
        """
        entry = self._jobs.pop(job_id, None)
        if entry is not None and entry[0] is self._thread:
            self._thread = None
            self._worker = None
        if entry is not None and entry[0] is self._count_thread:
            self._count_thread = None
        self._pump()

    def active_jobs(self) -> int:
        """How many query/export threads are still winding down."""
        return len(self._jobs)

    def _on_worker_error_text(self, tb: str) -> None:
        """Turn a worker traceback into one inline line (never a dialog)."""
        line = ""
        for candidate in reversed(str(tb).strip().splitlines()):
            if candidate.strip():
                line = candidate.strip()
                break
        self._set_status(f"Query failed: {line}", error=True)

    def _on_job_error(self, exc: Exception) -> None:
        """Report a failed background read without closing the browser.

        :param exc: what went wrong.
        """
        self._set_status(f"Query failed: {exc}", error=True)

    def is_busy(self) -> bool:
        """True while any query, count or export job is queued or running."""
        return self._export_busy or self._load_jobs > 0


    def _update_controls(self) -> None:
        """Enable each control only when it has something to act on."""
        has_db = self._db is not None
        has_table = has_db and bool(self._table)
        raw = self._raw_toggle.isChecked()
        self._raw_edit.setVisible(raw)
        self._filter_col.setVisible(not raw)
        self._filter_op.setVisible(not raw)
        needs_value = (not raw
                       and OPERATORS.get(self._filter_op.currentText(),
                                          ("", 1, None))[1] > 0)
        self._filter_value.setVisible(not raw)
        self._filter_value.setEnabled(needs_value)
        for w in (self._btn_apply, self._btn_clear, self._btn_export):
            w.setEnabled(has_table and not self._export_busy)
        self._btn_more.setEnabled(
            has_table and not self._exhausted and not self._chunk_jobs)
        self._table_list.setEnabled(has_db)
        self._update_edit_ui()


    def closeEvent(self, event):  # noqa: N802
        """Let every in-flight query thread finish before the widget dies.

        Destroying a QThread that is still running aborts the process, so
        the screen waits briefly for running threads. Jobs that have not
        started yet are dropped — nothing has been spawned for them.

        The shared link outlives this screen, so let go of it too.

        :param event: Qt close event forwarded after worker cleanup.
        """
        try:
            self.unlink_selection()
        except (RuntimeError, TypeError):
            pass
        for _fn, _on_done, kind, _token in self._queue:
            self._release(kind)
        self._queue.clear()
        for thread, _worker in list(self._jobs.values()):
            try:
                if thread.isRunning():
                    thread.quit()
                    _wait_serving_the_interface(thread, 5000)
            except RuntimeError:
                pass
        super().closeEvent(event)



_MERGE_PREVIEW_ROWS = 200


class _MergeDialog(QDialog):
    """Merge tables from one or more projects into a new table.

    Pick the projects (the open one is listed first), stack one table from
    each or join object tables on their keys, preview, then save the result
    as a new table in a store project. The work is
    :func:`spacr.plate_measurements.merge_projects` and
    :func:`spacr.plate_measurements.save_merge`, run on the browser job
    queue; a project label keeps apart two projects that hold a plate of
    the same name.

    :param screen: the Database Browser that owns the dialog.
    """

    def __init__(self, screen: "DbBrowserScreen"):
        """Lay out the projects, the way to merge, the preview and saving."""
        super().__init__(screen)
        self._screen = screen
        self._paths: List[str] = []
        self._preview: Optional[pd.DataFrame] = None
        self._previewed: Optional[tuple] = None
        self.last_result: Optional[Dict[str, Any]] = None
        self.setWindowTitle(tr("Merge tables"))
        self.setObjectName("DbMergeDialog")
        self.resize(900, 760)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACING["lg"], SPACING["lg"],
                                 SPACING["lg"], SPACING["lg"])
        outer.setSpacing(SPACING["sm"])

        intro = QLabel(tr(
            "Put tables from one or more projects together. Each row keeps "
            "the label of its project, so two projects with a plate of the "
            "same name stay apart."), self)
        intro.setObjectName("Muted")
        intro.setWordWrap(True)
        outer.addWidget(intro)

        outer.addWidget(QLabel(tr("Projects"), self))
        self._sources = QTableWidget(0, 2, self)
        self._sources.setHorizontalHeaderLabels([tr("Label"), tr("Project")])
        self._sources.horizontalHeader().setStretchLastSection(True)
        self._sources.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Interactive)
        self._sources.setColumnWidth(0, 160)
        self._sources.verticalHeader().setVisible(False)
        self._sources.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._sources.setMaximumHeight(150)
        self._sources.itemChanged.connect(self._on_settings_changed)
        outer.addWidget(self._sources)

        source_row = QHBoxLayout()
        source_row.setSpacing(SPACING["sm"])
        self._btn_add = QPushButton(tr("Add project…"), self)
        self._btn_add.setToolTip(tr(
            "A store project folder, a run folder, or a measurements.db."))
        self._btn_add.clicked.connect(self._pick_source)
        self._btn_remove = QPushButton(tr("Remove"), self)
        self._btn_remove.clicked.connect(self._remove_selected)
        source_row.addWidget(self._btn_add)
        source_row.addWidget(self._btn_remove)
        source_row.addStretch(1)
        outer.addLayout(source_row)

        way_row = QHBoxLayout()
        way_row.setSpacing(SPACING["md"])
        self._stack = QRadioButton(tr("Stack rows: one table from each project"), self)
        self._join = QRadioButton(tr("Join on object keys: one row per cell"), self)
        self._stack.setChecked(True)
        self._stack.toggled.connect(self._on_way_changed)
        way_row.addWidget(self._stack)
        way_row.addWidget(self._join)
        way_row.addStretch(1)
        outer.addLayout(way_row)

        table_row = QHBoxLayout()
        table_row.setSpacing(SPACING["sm"])
        self._table_label = QLabel(tr("Table"), self)
        self._table_box = QComboBox(self)
        self._table_box.setMinimumWidth(200)
        self._table_box.currentTextChanged.connect(self._on_settings_changed)
        self._anchor_label = QLabel(tr("One row per"), self)
        self._anchor_box = QComboBox(self)
        self._anchor_box.currentTextChanged.connect(self._on_settings_changed)
        table_row.addWidget(self._table_label)
        table_row.addWidget(self._table_box)
        table_row.addWidget(self._anchor_label)
        table_row.addWidget(self._anchor_box)
        table_row.addStretch(1)
        outer.addLayout(table_row)
        self._join_tables = QListWidget(self)
        self._join_tables.setMaximumHeight(110)
        self._join_tables.itemChanged.connect(self._on_settings_changed)
        outer.addWidget(self._join_tables)

        column_row = QHBoxLayout()
        column_row.setSpacing(SPACING["sm"])
        self._union = Toggle(tr("Keep columns only some projects have"), self)
        self._union.setChecked(True)
        self._union.toggled.connect(self._on_settings_changed)
        self._columns_edit = QLineEdit(self)
        self._columns_edit.setPlaceholderText(
            tr("Only columns containing…  e.g. 'area' (keys always stay)"))
        self._columns_edit.setClearButtonEnabled(True)
        self._columns_edit.textChanged.connect(self._on_columns_typed)
        column_row.addWidget(self._union)
        column_row.addWidget(self._columns_edit, 1)
        outer.addLayout(column_row)

        preview_row = QHBoxLayout()
        self._btn_preview = QPushButton(tr("Preview"), self)
        self._btn_preview.clicked.connect(self.preview)
        preview_row.addWidget(self._btn_preview)
        preview_row.addStretch(1)
        outer.addLayout(preview_row)

        self._notes = QLabel("", self)
        self._notes.setObjectName("MergeNotes")
        self._notes.setWordWrap(True)
        self._notes.setTextInteractionFlags(Qt.TextSelectableByMouse)
        outer.addWidget(self._notes)

        self._model = PreviewModel(self)
        self._view = QTableView(self)
        self._view.setObjectName("DbMergePreview")
        self._view.setModel(self._model)
        self._view.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._view.setAlternatingRowColors(True)
        self._view.horizontalHeader().setDefaultSectionSize(130)
        outer.addWidget(self._view, 1)

        save_row = QHBoxLayout()
        save_row.setSpacing(SPACING["sm"])
        save_row.addWidget(QLabel(tr("Save as"), self))
        self._name_edit = QLineEdit(self)
        self._name_edit.setPlaceholderText(tr("new table name"))
        self._name_edit.textChanged.connect(lambda *_: self._update_controls())
        save_row.addWidget(self._name_edit, 1)
        save_row.addWidget(QLabel(tr("in"), self))
        self._target_box = QComboBox(self)
        self._target_box.setMinimumWidth(260)
        self._target_box.activated.connect(self._on_target_chosen)
        save_row.addWidget(self._target_box, 2)
        self._btn_save = QPushButton(tr("Save merged table"), self)
        self._btn_save.clicked.connect(self.save)
        save_row.addWidget(self._btn_save)
        outer.addLayout(save_row)

        self._status = QLabel("", self)
        self._status.setObjectName("Muted")
        self._status.setWordWrap(True)
        self._status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        outer.addWidget(self._status)

        if screen.database_path():
            self.add_source(screen.database_path())
        self._on_way_changed()

    def add_source(self, path: str, label: str = "") -> bool:
        """Add a project to the merge.

        :param path: a store project, a run folder or a measurements.db.
        :param label: the label its rows carry; blank takes the default.
        :returns: True when the project opened and was added.
        """
        from spacr.multi_database import source_labels

        try:
            located = _store.locate(resolve_db_path(str(path)))
        except Exception as e:
            self._say(tr("Could not open {path}: {error}", path=path, error=e),
                      error=True)
            return False
        if os.path.realpath(located) in {os.path.realpath(p) for p in self._paths}:
            self._say(tr("{path} is already in the merge.", path=located),
                      error=True)
            return False
        before = source_labels(self._paths) if self._paths else ()
        self._paths.append(located)
        defaults = source_labels(self._paths)
        self._sources.blockSignals(True)
        for index, old in enumerate(before):
            item = self._sources.item(index, 0)
            if item is not None and item.text().strip() == old:
                item.setText(defaults[index])
        row = self._sources.rowCount()
        self._sources.insertRow(row)
        self._sources.setItem(row, 0, QTableWidgetItem(label or defaults[-1]))
        place = QTableWidgetItem(located)
        place.setFlags(place.flags() & ~Qt.ItemIsEditable)
        place.setToolTip(located)
        self._sources.setItem(row, 1, place)
        self._sources.blockSignals(False)
        self._refresh_tables()
        self._say(tr("{count} project(s) in the merge.", count=len(self._paths)))
        return True

    def set_label(self, row: int, label: str) -> None:
        """Set the label of the project in ``row``.

        :param row: the project row, from 0.
        :param label: the label its rows carry.
        """
        self._sources.item(row, 0).setText(label)

    def labels(self) -> List[str]:
        """The label of each project, in order."""
        return [self._sources.item(row, 0).text().strip()
                for row in range(self._sources.rowCount())]

    def sources(self) -> List[str]:
        """The projects in the merge, in order."""
        return list(self._paths)

    def holds(self, path: str) -> bool:
        """Whether the project at ``path`` is already in the merge.

        :param path: what :meth:`add_source` accepts.
        """
        try:
            located = _store.locate(resolve_db_path(str(path)))
        except Exception:
            return False
        return os.path.realpath(located) in {os.path.realpath(p)
                                             for p in self._paths}

    def set_way(self, how: str) -> None:
        """Choose how to merge.

        :param how: ``'stack'`` or ``'join'``.
        """
        (self._join if how == "join" else self._stack).setChecked(True)

    def way(self) -> str:
        """``'stack'`` or ``'join'``."""
        return "join" if self._join.isChecked() else "stack"

    def set_tables(self, tables: Sequence[str], anchor: str = "") -> None:
        """Choose the table to stack, or the tables to join.

        :param tables: one table for a stack; the object tables for a join.
        :param anchor: for a join, the table one row stands for.
        """
        if self.way() == "stack":
            self._table_box.setCurrentText(str(tables[0]))
            return
        wanted = set(tables)
        for index in range(self._join_tables.count()):
            item = self._join_tables.item(index)
            item.setCheckState(Qt.Checked if item.text() in wanted
                               else Qt.Unchecked)
        if anchor:
            self._anchor_box.setCurrentText(anchor)

    def set_column_filter(self, text: str) -> None:
        """Keep only columns whose name contains ``text`` (keys always stay).

        :param text: the search text; empty keeps every column.
        """
        self._columns_edit.setText(text)

    def set_target(self, path: str) -> None:
        """Choose the store project the merged table is saved in.

        :param path: a store project folder, or a new or empty folder.
        """
        index = self._target_box.findData(path)
        if index < 0:
            self._target_box.insertItem(self._target_box.count() - 1, path, path)
            index = self._target_box.findData(path)
        self._target_box.setCurrentIndex(index)
        self._update_controls()

    def set_name(self, name: str) -> None:
        """Name the new table.

        :param name: letters, digits and _.
        """
        self._name_edit.setText(name)

    def notes(self) -> str:
        """What the last preview or save said, as shown."""
        return self._notes.text()

    def status_text(self) -> str:
        """The status line."""
        return self._status.text()

    def preview_frame(self) -> Optional[pd.DataFrame]:
        """The rows of the last preview, or None."""
        return self._preview

    def preview_columns(self) -> List[str]:
        """The preview columns the column search shows."""
        return self._model.visible_columns()

    def _say(self, text: str, error: bool = False) -> None:
        """Show ``text`` on the status line."""
        self._status.setText(text)
        self._status.setProperty("error", bool(error))
        self._status.style().unpolish(self._status)
        self._status.style().polish(self._status)

    def _pick_source(self) -> None:
        """Ask for a project folder or a measurements.db and add it."""
        path = QFileDialog.getExistingDirectory(
            self, tr("Choose a project or run folder"), "")
        if path:
            self.add_source(path)

    def _remove_selected(self) -> None:
        """Drop the selected projects from the merge."""
        rows = sorted({index.row() for index in
                       self._sources.selectionModel().selectedRows()},
                      reverse=True)
        for row in rows:
            self._sources.removeRow(row)
            del self._paths[row]
        self._refresh_tables()

    def _tables_of(self, path: str) -> List[str]:
        """The tables of one project, without the key tables of the store."""
        from spacr import measurement_store

        return measurement_store.tables(path)

    def _refresh_tables(self) -> None:
        """List the tables every project holds, for stacking and for joining."""
        from spacr.merge_tables import OBJECT_TABLES
        from spacr.object_roles import ONE_ROW_PER_CELL

        shared: Optional[set] = None
        for path in self._paths:
            try:
                found = set(self._tables_of(path))
            except Exception:
                found = set()
            shared = found if shared is None else shared & found
        shared = shared or set()
        current = self._table_box.currentText()
        self._table_box.blockSignals(True)
        self._table_box.clear()
        self._table_box.addItems(sorted(shared))
        if current in shared:
            self._table_box.setCurrentText(current)
        elif "cell" in shared:
            self._table_box.setCurrentText("cell")
        self._table_box.blockSignals(False)
        objects = [t for t in OBJECT_TABLES if t in shared]
        ticked = {self._join_tables.item(i).text()
                  for i in range(self._join_tables.count())
                  if self._join_tables.item(i).checkState() == Qt.Checked}
        self._join_tables.blockSignals(True)
        self._join_tables.clear()
        for name in objects:
            item = QListWidgetItem(name, self._join_tables)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if (name in ticked or not ticked)
                               else Qt.Unchecked)
        self._join_tables.blockSignals(False)
        anchor = self._anchor_box.currentText()
        self._anchor_box.blockSignals(True)
        self._anchor_box.clear()
        self._anchor_box.addItems([t for t in ONE_ROW_PER_CELL if t in shared])
        if anchor:
            self._anchor_box.setCurrentText(anchor)
        self._anchor_box.blockSignals(False)
        self._refresh_targets()
        self._on_settings_changed()

    def _refresh_targets(self) -> None:
        """Offer the store projects of the merge, and another folder."""
        chosen = self._target_box.currentData()
        extra = [self._target_box.itemData(i)
                 for i in range(self._target_box.count())
                 if self._target_box.itemData(i)
                 and self._target_box.itemData(i) not in self._paths]
        self._target_box.blockSignals(True)
        self._target_box.clear()
        for path in self._paths:
            if os.path.isdir(path):
                self._target_box.addItem(path, path)
        for path in extra:
            self._target_box.addItem(path, path)
        self._target_box.addItem(tr("Another folder…"), "")
        index = self._target_box.findData(chosen) if chosen else 0
        self._target_box.setCurrentIndex(max(0, index))
        self._target_box.blockSignals(False)

    def _on_target_chosen(self, index: int) -> None:
        """Ask for a folder when "Another folder…" is chosen."""
        if self._target_box.itemData(index):
            self._update_controls()
            return
        path = QFileDialog.getExistingDirectory(
            self, tr("Choose a store project or an empty folder"), "")
        if path:
            self.set_target(path)

    def _on_way_changed(self, *_args) -> None:
        """Show the table picker that fits the chosen way of merging."""
        join = self.way() == "join"
        self._table_label.setVisible(not join)
        self._table_box.setVisible(not join)
        self._anchor_label.setVisible(join)
        self._anchor_box.setVisible(join)
        self._join_tables.setVisible(join)
        self._on_settings_changed()

    def _on_columns_typed(self, text: str) -> None:
        """Narrow the preview columns as the user types."""
        self._model.set_column_filter(text)
        self._update_controls()

    def _on_settings_changed(self, *_args) -> None:
        """A changed setting needs a new preview before saving."""
        if self._previewed is not None and self._previewed != self._settings():
            self._say(tr("The settings changed; preview again before saving."))
        self._update_controls()

    def _chosen_tables(self) -> Tuple[List[str], str]:
        """The tables to merge, and the anchor of a join."""
        if self.way() == "stack":
            name = self._table_box.currentText()
            return ([name] if name else []), ""
        anchor = self._anchor_box.currentText()
        ticked = [self._join_tables.item(i).text()
                  for i in range(self._join_tables.count())
                  if self._join_tables.item(i).checkState() == Qt.Checked]
        return [t for t in dict.fromkeys([anchor, *ticked]) if t], anchor

    def _settings(self) -> tuple:
        """Everything a preview depends on, to tell when it is out of date."""
        tables, anchor = self._chosen_tables()
        return (tuple(self._paths), tuple(self.labels()), self.way(),
                tuple(tables), anchor, self._union.isChecked())

    def _keep(self) -> Optional[List[str]]:
        """The columns the column search keeps, or None for all of them."""
        if not self._columns_edit.text().strip():
            return None
        return self._model.visible_columns()

    def _update_controls(self) -> None:
        """Enable each action when it has what it needs."""
        busy = self._screen._export_busy
        tables, _anchor = self._chosen_tables()
        ready = bool(self._paths) and bool(tables) and not busy
        self._btn_preview.setEnabled(ready)
        fresh = self._previewed is not None and self._previewed == self._settings()
        self._btn_save.setEnabled(
            ready and fresh and bool(self._name_edit.text().strip())
            and bool(self._target_box.currentData()))
        self._btn_remove.setEnabled(bool(self._paths) and not busy)

    def preview(self) -> bool:
        """Merge the first rows of each project and show them.

        A stack reads the first rows of each project; a join merges the
        tables whole, because a child row cannot be rolled up without all
        of its siblings. Runs on the browser job queue.

        :returns: whether the preview job was accepted.
        """
        from spacr.multi_database import MergeRefused, describe_merge
        from spacr.plate_measurements import merge_projects

        tables, anchor = self._chosen_tables()
        if not self._paths or not tables:
            self._say(tr("Add a project and choose a table first."), error=True)
            return False
        settings = self._settings()
        paths, labels, how = list(self._paths), self.labels(), self.way()
        columns = "union" if self._union.isChecked() else "common"

        def _job() -> Dict[str, Any]:
            """Merge the preview rows. Off the GUI thread."""
            try:
                frame = merge_projects(
                    paths, tables if how == "join" else tables[0], how=how,
                    anchor=anchor or None, labels=labels, columns=columns,
                    limit_per_source=_MERGE_PREVIEW_ROWS if how == "stack" else None)
            except MergeRefused as e:
                return {"refused": str(e)}
            held = None
            if how == "stack":
                plan = describe_merge([_store.locate(p) for p in paths], tables[0])
                held = {label: source.rows
                        for label, source in zip(labels, plan.sources)}
            return {"frame": frame, "held": held}

        def _done(result: Dict[str, Any]) -> None:
            """Show the preview and what the merge will do."""
            if "refused" in result:
                self._preview, self._previewed = None, None
                self._model.clear()
                self._notes.setText("")
                self._say(result["refused"], error=True)
                self._update_controls()
                return
            from spacr.multi_database import SCREEN_COLUMN, SOURCE_COLUMN

            frame = result["frame"]
            self._preview, self._previewed = frame, settings
            shown = frame.groupby(SOURCE_COLUMN, sort=False).head(
                _MERGE_PREVIEW_ROWS) if SOURCE_COLUMN in frame.columns else frame
            first = [c for c in (SOURCE_COLUMN, SCREEN_COLUMN) if c in shown.columns]
            shown = shown[first + [c for c in shown.columns if c not in first]]
            self._model.set_page(list(shown.columns),
                                 list(shown.itertuples(index=False, name=None)))
            self._model.set_column_filter(self._columns_edit.text())
            lines = list(frame.attrs.get("summary", ()))
            if result["held"] is not None and lines:
                per = ", ".join(f"{label} {count:,}"
                                for label, count in result["held"].items())
                lines[0] = tr(
                    "Showing the first {shown} rows of each project. The "
                    "merged table will hold {total:,} rows: {per}.",
                    shown=_MERGE_PREVIEW_ROWS,
                    total=sum(result["held"].values()), per=per)
            keys = ", ".join(frame.attrs.get("key_columns", ()))
            if keys:
                lines.append(tr("Each row is named by {keys}.", keys=keys))
            self._notes.setText("\n".join(lines))
            self._say(tr("Preview ready. Name the table and save it."))
            self._update_controls()

        self._say(tr("Merging the preview…"))
        return self._screen._run_job(_job, _done, kind="export")

    def save(self) -> bool:
        """Save the previewed merge as a new table in the chosen store project.

        The browser then opens that project at the new table.

        :returns: whether the save job was accepted.
        """
        from spacr.multi_database import MergeRefused
        from spacr.plate_measurements import save_merge

        if self._previewed is None or self._previewed != self._settings():
            self._say(tr("Preview the merge before saving it."), error=True)
            return False
        target = self._target_box.currentData()
        name = self._name_edit.text().strip()
        if not target or not name:
            self._say(tr("Name the table and choose where to save it."),
                      error=True)
            return False
        tables, anchor = self._chosen_tables()
        paths, labels, how = list(self._paths), self.labels(), self.way()
        columns = "union" if self._union.isChecked() else "common"
        keep = self._keep()

        def _job() -> Dict[str, Any]:
            """Write the merged table. Off the GUI thread."""
            try:
                return save_merge(target, name, paths,
                                  tables if how == "join" else tables[0],
                                  how=how, anchor=anchor or None,
                                  labels=labels, columns=columns, keep=keep)
            except MergeRefused as e:
                return {"refused": str(e)}

        def _done(result: Dict[str, Any]) -> None:
            """Say what was saved and open it in the browser."""
            if "refused" in result:
                self._say(result["refused"], error=True)
                return
            self.last_result = result
            self._notes.setText("\n".join(result["summary"]))
            self._say(tr("Saved {rows:,} rows as {table}.",
                         rows=result["rows"], table=result["table"]))
            if self._screen.set_database(result["project"]):
                self._screen.select_table(result["table"])

        self._say(tr("Saving {name}…", name=name))
        return self._screen._run_job(_job, _done, kind="export")


HOST_KEY = "db_browser"

#: Registry keys of the modules folded into Database Browser, in strip
#: order. Both are READS OF THE SAME TABLES the browser is already
#: pointed at -- lineage walks the containment columns, tabulate pivots
#: and counts them -- so each is a second view of the open database
#: rather than a separate errand.
FOLDED_APPS: Tuple[str, ...] = ('lineage', 'tabulate')


def _build_lineage(host_window: Optional[QWidget] = None) -> QWidget:
    """Lineage, as the window builds it."""
    from .map_barcodes import build_registered_screen

    return build_registered_screen("lineage", host_window)


def _build_tabulate(host_window: Optional[QWidget] = None) -> QWidget:
    """Tabulate, as the window builds it."""
    from .map_barcodes import build_registered_screen

    return build_registered_screen("tabulate", host_window)


#: One builder per folded module. :func:`install_folds` walks
#: :data:`FOLDED_APPS` and looks each key up here, so the strip's order
#: and the strip's contents cannot disagree.
BUILDERS: Dict[str, Callable[[Optional[QWidget]], QWidget]] = {
    "lineage": _build_lineage,
    "tabulate": _build_tabulate,
}


def install_folds(screen: QWidget) -> Optional["FoldStrip"]:
    """Put db_browser's fold strip on ``screen``'s masthead.

    Reached by the one pass over the stack that serves every host --
    see :data:`spacr.qt.screens.map_barcodes.FOLD_HOST_MODULES`.
    """
    from .map_barcodes import install_fold_strip

    return install_fold_strip(screen, HOST_KEY, FOLDED_APPS, BUILDERS)

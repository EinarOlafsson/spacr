"""
Database Browser — a read-only query panel for a spaCR ``measurements.db``.

Answering "how many cells are in plate1?" or "what does the ``png_list``
table actually contain?" used to mean dropping to a terminal and typing
``sqlite3 .../measurements/measurements.db``. This screen puts the same
questions one click away, without ever giving the GUI a way to write to
the file.

Layout::

    ┌───────────────────────────────────────────────────────────────────┐
    │ /data/plate1/measurements/measurements.db   [DB…] [Run folder…]   │
    │ Read-only (mode=ro) — nothing here can modify your measurements.  │
    ├──────────────┬────────────────────────────────────────────────────┤
    │ Tables       │ Columns [search…]      (120 of 512 columns)        │
    │  cell   40   │ ┌────────────────────────────────────────────────┐ │
    │  nucleus     │ │ prc          cell_area   cell_channel_1_mean…  │ │
    │  png_list    │ │ plate1_A01_1 1204.5      3311.2                │ │
    │              │ └────────────────────────────────────────────────┘ │
    │              │ [◀ Prev]  rows 1–100 of 412 003  [Next ▶]  [100 ▾] │
    ├──────────────┴────────────────────────────────────────────────────┤
    │ Filter: [column ▾] [op ▾] [value]  ☐ raw SQL  [Apply] [Clear]     │
    │                                              [Export filtered CSV]│
    └───────────────────────────────────────────────────────────────────┘

Design notes that matter for real spaCR databases:

* **Never ``SELECT *`` the whole table.** Measurement tables run to
  hundreds of thousands of rows and hundreds of feature columns. Every
  preview is ``LIMIT ? OFFSET ?`` with a bound page size, and the total
  is a separate ``COUNT(*)``.
* **Off the GUI thread.** Both the page query and the export go through
  :func:`spacr.qt.bridge.make_thread`, the same helper the pipeline
  screens use, so a slow ``COUNT(*)`` never freezes the window.
* **Read-only, structurally.** Connections are opened with the
  ``file:…?mode=ro`` URI *and* ``PRAGMA query_only = ON``. A write is
  rejected by SQLite itself, not by a check we could forget.
* **No string-formatted values.** Everything the user types is bound as
  a ``?`` parameter. Identifiers (table + column names) never come from
  free text — they are matched against the live schema and only then
  double-quoted.
* **No modal dialogs on any error path.** Problems land in an inline
  status label; a headless run can never block on a message box.
"""
from __future__ import annotations

import contextlib
import csv
import os
import re
import sqlite3
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from urllib.parse import quote as _urlquote

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from ..bridge import make_thread
from ..theme import PALETTE, SPACING
from ..widgets import Divider

__all__ = [
    "DB_FILENAME",
    "DEFAULT_PAGE_SIZE",
    "DbBrowserScreen",
    "OPERATORS",
    "PreviewModel",
    "ReadOnlyDb",
    "build_where",
    "quote_ident",
    "resolve_db_path",
    "validate_raw_predicate",
]


DB_FILENAME = "measurements.db"
_MEASUREMENTS_SUBDIR = "measurements"

#: Rows fetched per preview page. Big enough to be useful, small enough
#: that a 500-column table still renders instantly.
DEFAULT_PAGE_SIZE = 100
#: (min, max) the "Rows / page" spin box allows.
PAGE_SIZE_RANGE = (25, 1000)

#: Above this many *visible* columns the view keeps a fixed column width
#: instead of measuring content. ``resizeColumnsToContents`` walks every
#: cell of every column, which is fine for a 30-column table and painful
#: for a 500-column feature table.
AUTOSIZE_MAX_COLUMNS = 60


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def resolve_db_path(path: str) -> str:
    """Return the absolute path of the sqlite file ``path`` refers to.

    Accepts either the database file itself or a run ``src`` folder, in
    which case ``<src>/measurements/measurements.db`` is used — the same
    layout every other spaCR module assumes (see ``spacr.measure`` and
    ``spacr.ml``). ``<src>/measurements.db`` is accepted as a fallback so
    pointing at the ``measurements`` folder itself also works.

    :param path: file or folder chosen by the user.
    :returns: absolute path to an existing file.
    :raises ValueError: when ``path`` is empty.
    :raises FileNotFoundError: when nothing resolvable exists there.
    """
    if path is None or not str(path).strip():
        raise ValueError(
            "No database selected — choose a measurements.db or a run folder.")
    p = os.path.abspath(os.path.expanduser(str(path).strip()))
    if os.path.isfile(p):
        return p
    if os.path.isdir(p):
        for candidate in (
            os.path.join(p, _MEASUREMENTS_SUBDIR, DB_FILENAME),
            os.path.join(p, DB_FILENAME),
        ):
            if os.path.isfile(candidate):
                return candidate
        raise FileNotFoundError(
            f"No {DB_FILENAME} under {p} — expected "
            f"{os.path.join(p, _MEASUREMENTS_SUBDIR, DB_FILENAME)}")
    raise FileNotFoundError(f"No such file or folder: {p}")


def _read_only_uri(path: str) -> str:
    """Return the ``file:…?mode=ro`` URI SQLite needs for a read-only open."""
    # Percent-escape everything a URI would otherwise treat as syntax
    # ('?', '#', '%') while leaving path separators and the Windows drive
    # colon alone.
    return "file:" + _urlquote(str(path).replace("\\", "/"), safe="/:") + "?mode=ro"


# ---------------------------------------------------------------------------
# SQL construction — identifiers validated, values always bound
# ---------------------------------------------------------------------------

def quote_ident(name: str) -> str:
    """Double-quote a SQL identifier, escaping embedded quotes.

    Only ever called with a name that has already been matched against
    the live schema; the quoting is belt-and-braces for identifiers that
    are legal but awkward (``cell_channel_1 (raw)``).
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
        raise ValueError(
            f"Unknown column {column!r} — not in this table's schema.")
    spec = OPERATORS.get(op)
    if spec is None:
        raise ValueError(f"Unknown operator {op!r}.")
    template, nargs, transform = spec
    sql = template.format(col=quote_ident(column))
    if nargs == 0:
        return sql, ()
    return sql, (transform(value) if transform else value,)


# Statements that have no business inside a WHERE clause. The connection
# is read-only anyway, so this is a second line of defence — it mostly
# stops a user pasting a whole script into the predicate box and being
# confused by the error.
_FORBIDDEN_RAW = re.compile(
    r"(?:^|[^A-Za-z_])"
    r"(insert|update|delete|drop|alter|create|replace|attach|detach|"
    r"pragma|vacuum|reindex|begin|commit|rollback)"
    r"(?:[^A-Za-z_]|$)",
    re.IGNORECASE,
)


def validate_raw_predicate(text: str) -> str:
    """Return ``text`` stripped, or raise if it is not a lone predicate.

    The raw box is an explicit power-user escape hatch (``cell_area > 1000
    AND well LIKE 'A%'``). We refuse statement separators, comments and
    DDL/DML keywords so it stays a *predicate*; the read-only connection
    plus ``PRAGMA query_only`` guarantee the rest.
    """
    t = "" if text is None else str(text).strip()
    if not t:
        raise ValueError("Raw filter is empty — type a condition or untick "
                         "'raw SQL'.")
    if ";" in t:
        raise ValueError("';' is not allowed — the filter is one condition, "
                         "not a script.")
    if "--" in t or "/*" in t:
        raise ValueError("SQL comments are not allowed in the filter.")
    m = _FORBIDDEN_RAW.search(t)
    if m:
        raise ValueError(
            f"{m.group(1).upper()} is not allowed — this browser is read-only.")
    return t


# ---------------------------------------------------------------------------
# Read-only sqlite access
# ---------------------------------------------------------------------------

class ReadOnlyDb:
    """A read-only handle on a sqlite database.

    Every method opens (and closes) its own connection. That costs a few
    microseconds and buys thread-safety for free: the preview query runs
    on a worker thread while the GUI thread may still be listing tables,
    and ``sqlite3`` objects are not shareable across threads.

    :param path: database file, or a run folder — see :func:`resolve_db_path`.
    :raises sqlite3.DatabaseError: when the file is not a sqlite database.
    """

    def __init__(self, path: str):
        self.path = resolve_db_path(path)
        self.uri = _read_only_uri(self.path)
        #: SQL of the most recent statement — handy in tests + bug reports.
        self.last_sql: str = ""
        self._tables: Optional[List[str]] = None
        self._rowid_ok: Dict[str, bool] = {}
        # Probe now so "that file isn't a database" surfaces at open time
        # rather than three clicks later.
        with self._con() as con:
            con.execute("SELECT name FROM sqlite_master LIMIT 1").fetchall()

    # -- connections -------------------------------------------------------

    def connect(self) -> sqlite3.Connection:
        """Return a fresh read-only connection. The caller closes it."""
        con = sqlite3.connect(self.uri, uri=True)
        # mode=ro already refuses writes; query_only also blocks anything
        # that would try to change the *schema* through a temp attachment.
        con.execute("PRAGMA query_only = ON")
        return con

    @contextlib.contextmanager
    def _con(self):
        con = self.connect()
        try:
            yield con
        finally:
            con.close()

    def _execute(self, con: sqlite3.Connection, sql: str, params: Sequence = ()):
        self.last_sql = sql
        return con.execute(sql, tuple(params))

    # -- schema ------------------------------------------------------------

    def tables(self, refresh: bool = False) -> List[str]:
        """Return the user tables + views, alphabetically."""
        if self._tables is None or refresh:
            with self._con() as con:
                rows = self._execute(con,
                    "SELECT name FROM sqlite_master "
                    "WHERE type IN ('table', 'view') "
                    "AND name NOT LIKE 'sqlite_%' ORDER BY name").fetchall()
            self._tables = [r[0] for r in rows]
        return list(self._tables)

    def check_table(self, table: str) -> str:
        """Return ``table`` if the schema really has it, else raise.

        This is the gate that keeps identifiers out of the "user input"
        category: a name that isn't in ``sqlite_master`` never reaches SQL.
        """
        if table not in self.tables():
            raise ValueError(f"No table named {table!r} in {os.path.basename(self.path)}.")
        return table

    def columns(self, table: str) -> List[str]:
        """Return the column names of ``table`` in declaration order."""
        self.check_table(table)
        with self._con() as con:
            # PRAGMA takes no bound parameters; `table` is schema-validated
            # above and quoted here.
            rows = self._execute(
                con, f"PRAGMA table_info({quote_ident(table)})").fetchall()
        return [r[1] for r in rows]

    def check_columns(self, table: str, columns: Optional[Sequence[str]]) -> List[str]:
        """Return the requested columns, validated against the schema.

        ``None`` means "all of them".
        """
        real = self.columns(table)
        if columns is None:
            return real
        known = set(real)
        chosen = [c for c in columns if c in known]
        unknown = [c for c in columns if c not in known]
        if unknown:
            raise ValueError(
                f"Unknown column(s) for {table!r}: {', '.join(map(repr, unknown))}")
        return chosen or real

    def _has_rowid(self, table: str) -> bool:
        """True when ``table`` exposes ``rowid`` (i.e. isn't WITHOUT ROWID).

        Paging without a stable order is a lie, so the preview orders by
        ``rowid`` when it can. That is the table's own primary index, so
        it costs nothing on a 500 k-row measurement table.
        """
        if table in self._rowid_ok:
            return self._rowid_ok[table]
        ok = True
        try:
            with self._con() as con:
                self._execute(
                    con, f"SELECT rowid FROM {quote_ident(table)} LIMIT 1").fetchall()
        except sqlite3.Error:
            ok = False
        self._rowid_ok[table] = ok
        return ok

    # -- queries -----------------------------------------------------------

    def select_sql(self, table: str, columns: Sequence[str],
                   where: Optional[str] = None,
                   paged: bool = True) -> str:
        """Build the preview SELECT. Exposed so tests can assert on it."""
        col_sql = ", ".join(quote_ident(c) for c in columns)
        sql = f"SELECT {col_sql} FROM {quote_ident(table)}"
        if where:
            sql += f" WHERE {where}"
        if self._has_rowid(table):
            sql += " ORDER BY rowid"
        if paged:
            sql += " LIMIT ? OFFSET ?"
        return sql

    def validate_where(self, table: str, where: str,
                       params: Sequence = ()) -> None:
        """Let SQLite parse ``where`` and raise if it is malformed.

        Prepared with ``LIMIT 0``, which SQLite short-circuits before it
        touches a single row — so a typo costs a parse, not a full scan
        of a 400 000-row measurement table.

        :raises sqlite3.Error: with SQLite's own message (``no such
            column: cell_are``, ``near ">": syntax error``, …).
        """
        self.check_table(table)
        sql = f"SELECT 1 FROM {quote_ident(table)} WHERE {where} LIMIT 0"
        with self._con() as con:
            self._execute(con, sql, params).fetchall()

    def count(self, table: str, where: Optional[str] = None,
              params: Sequence = ()) -> int:
        """Return ``COUNT(*)`` for the table under the current filter."""
        self.check_table(table)
        sql = f"SELECT COUNT(*) FROM {quote_ident(table)}"
        if where:
            sql += f" WHERE {where}"
        with self._con() as con:
            row = self._execute(con, sql, params).fetchone()
        return int(row[0]) if row else 0

    def page(self, table: str, limit: int = DEFAULT_PAGE_SIZE, offset: int = 0,
             where: Optional[str] = None, params: Sequence = (),
             columns: Optional[Sequence[str]] = None
             ) -> Tuple[List[str], List[tuple]]:
        """Return ``(columns, rows)`` for one page.

        Always ``LIMIT ? OFFSET ?`` — the full table is never materialised,
        and the page size itself is a bound parameter.
        """
        self.check_table(table)
        cols = self.check_columns(table, columns)
        sql = self.select_sql(table, cols, where, paged=True)
        args = tuple(params) + (int(max(1, limit)), int(max(0, offset)))
        with self._con() as con:
            rows = self._execute(con, sql, args).fetchall()
        return cols, rows

    def export_csv(self, out_path: str, table: str,
                   columns: Optional[Sequence[str]] = None,
                   where: Optional[str] = None, params: Sequence = (),
                   chunk: int = 5000) -> int:
        """Stream the filtered result to ``out_path`` as CSV.

        Rows are pulled in ``chunk``-sized batches and written straight
        out, so exporting a 400 k-row table costs a constant amount of
        memory.

        :returns: number of data rows written (header excluded).
        """
        self.check_table(table)
        cols = self.check_columns(table, columns)
        sql = self.select_sql(table, cols, where, paged=False)
        written = 0
        out_dir = os.path.dirname(os.path.abspath(out_path))
        if out_dir and not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        with self._con() as con:
            cur = self._execute(con, sql, params)
            with open(out_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(cols)
                while True:
                    batch = cur.fetchmany(chunk)
                    if not batch:
                        break
                    writer.writerows(batch)
                    written += len(batch)
        return written


# ---------------------------------------------------------------------------
# Table model
# ---------------------------------------------------------------------------

class PreviewModel(QAbstractTableModel):
    """Holds one page of rows plus the column-visibility mask.

    The page is fetched with *every* column, and the column search only
    changes which of them are mapped into the view. Typing in the search
    box therefore never re-queries the database — which is what makes it
    usable on a table with 500 feature columns.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._columns: List[str] = []
        self._rows: List[tuple] = []
        self._visible: List[int] = []
        self._filter: str = ""
        self._row_offset: int = 0

    # -- data ---------------------------------------------------------------

    def set_page(self, columns: Sequence[str], rows: Sequence[Sequence[Any]],
                 row_offset: int = 0) -> None:
        """Replace the model contents, keeping the current column search."""
        self.beginResetModel()
        self._columns = list(columns)
        self._rows = [tuple(r) for r in rows]
        self._row_offset = int(row_offset)
        self._recompute_visible()
        self.endResetModel()

    def clear(self) -> None:
        self.set_page([], [], 0)

    def all_columns(self) -> List[str]:
        return list(self._columns)

    def rows(self) -> List[tuple]:
        """The raw page rows, with every column (search-independent)."""
        return list(self._rows)

    def visible_columns(self) -> List[str]:
        return [self._columns[i] for i in self._visible]

    def column_filter(self) -> str:
        return self._filter

    def set_column_filter(self, text: str) -> None:
        """Show only columns whose name contains ``text`` (case-insensitive).

        An empty string restores every column.
        """
        self._filter = "" if text is None else str(text)
        self.beginResetModel()
        self._recompute_visible()
        self.endResetModel()

    def _recompute_visible(self) -> None:
        needle = self._filter.strip().lower()
        if not needle:
            self._visible = list(range(len(self._columns)))
        else:
            self._visible = [i for i, c in enumerate(self._columns)
                             if needle in str(c).lower()]

    # -- QAbstractTableModel ------------------------------------------------

    def rowCount(self, parent=QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent=QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._visible)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or role not in (Qt.DisplayRole, Qt.ToolTipRole):
            return None
        try:
            value = self._rows[index.row()][self._visible[index.column()]]
        except IndexError:
            return None
        if value is None:
            return ""
        if isinstance(value, bytes):
            return f"<{len(value)} bytes>"
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    def headerData(self, section, orientation, role=Qt.DisplayRole):  # noqa: N802
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            if 0 <= section < len(self._visible):
                return self._columns[self._visible[section]]
            return None
        return str(self._row_offset + section + 1)


# ---------------------------------------------------------------------------
# Screen
# ---------------------------------------------------------------------------

class DbBrowserScreen(QWidget):
    """Read-only browser for a spaCR measurements database.

    :param threaded: run queries on a worker thread (the default). Tests
        pass ``False`` to get deterministic, synchronous behaviour.
    :ivar last_error: text of the most recent failure, ``""`` when the
        last operation succeeded. Errors are *only* ever reported here
        and in the inline status label — never in a modal dialog.
    """

    #: emitted with the resolved path whenever a database opens
    database_opened = Signal(str)
    #: emitted after every query / export job settles (ok or not)
    job_finished = Signal(bool)

    def __init__(self, parent=None, threaded: bool = True):
        super().__init__(parent)
        self._threaded = bool(threaded)
        self._db: Optional[ReadOnlyDb] = None
        self._table: str = ""
        self._all_columns: List[str] = []
        self._page_index: int = 0
        self._row_count: int = 0
        self._where: Optional[str] = None
        self._params: tuple = ()
        self._filter_label: str = ""
        self._busy: bool = False
        # Every (QThread, PipelineWorker) pair that has been started and
        # whose event loop has not yet exited. This is an ownership list,
        # not a convenience: PySide6 destroys the C++ QThread as soon as
        # the last Python reference goes, and destroying a *running*
        # QThread aborts the process. A single `self._thread` slot is not
        # enough, because `worker.finished` (which lets the next job
        # start) fires strictly before `thread.finished` (which retires
        # the old one) — so two jobs legitimately overlap for a moment.
        self._jobs: List[tuple] = []
        self._thread = None     # most recent thread, for introspection
        self._worker = None
        self.last_error: str = ""

        self._build_ui()
        self._set_status(
            "Choose a measurements.db, or a run folder containing "
            "measurements/measurements.db.")
        self._update_controls()

    # -- construction ------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACING["lg"], SPACING["lg"],
                                 SPACING["lg"], SPACING["lg"])
        outer.setSpacing(SPACING["md"])

        title = QLabel("Database Browser")
        title.setObjectName("DisplayHeading")
        outer.addWidget(title)

        subtitle = QLabel(
            "Read-only. spaCR opens the file with mode=ro and "
            "PRAGMA query_only, so nothing here can modify your measurements.")
        subtitle.setObjectName("Muted")
        subtitle.setWordWrap(True)
        outer.addWidget(subtitle)

        outer.addWidget(Divider())

        # ── Source row ────────────────────────────────────────────────
        src_row = QHBoxLayout()
        src_row.setSpacing(SPACING["sm"])
        self._path_edit = QLineEdit(self)
        self._path_edit.setPlaceholderText(
            "…/measurements/measurements.db  — or a run folder")
        self._path_edit.setClearButtonEnabled(True)
        self._path_edit.returnPressed.connect(self._on_open_typed_path)
        self._btn_pick_db = QPushButton("Choose database…", self)
        self._btn_pick_db.clicked.connect(self._pick_database)
        self._btn_pick_src = QPushButton("Choose run folder…", self)
        self._btn_pick_src.clicked.connect(self._pick_run_folder)
        self._btn_open = QPushButton("Open", self)
        self._btn_open.clicked.connect(self._on_open_typed_path)
        src_row.addWidget(self._path_edit, 1)
        src_row.addWidget(self._btn_pick_db)
        src_row.addWidget(self._btn_pick_src)
        src_row.addWidget(self._btn_open)
        outer.addLayout(src_row)

        # ── Body splitter: tables | preview ───────────────────────────
        split = QSplitter(Qt.Horizontal, self)

        left = QWidget(split)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(SPACING["xs"])
        left_layout.addWidget(QLabel("Tables"))
        self._table_list = QListWidget(left)
        self._table_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table_list.currentItemChanged.connect(self._on_table_selected)
        left_layout.addWidget(self._table_list, 1)
        split.addWidget(left)

        right = QWidget(split)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(SPACING["xs"])

        col_row = QHBoxLayout()
        col_row.setSpacing(SPACING["sm"])
        self._col_search = QLineEdit(right)
        self._col_search.setPlaceholderText(
            "Search columns…  e.g. 'percentile' or 'channel_1'")
        self._col_search.setClearButtonEnabled(True)
        self._col_search.textChanged.connect(self.set_column_filter)
        self._col_count_label = QLabel("", right)
        self._col_count_label.setObjectName("Muted")
        col_row.addWidget(self._col_search, 1)
        col_row.addWidget(self._col_count_label)
        right_layout.addLayout(col_row)

        self._model = PreviewModel(self)
        self._view = QTableView(right)
        self._view.setModel(self._model)
        # Read-only in the UI as well as on the connection.
        self._view.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._view.setSelectionBehavior(QAbstractItemView.SelectItems)
        self._view.setAlternatingRowColors(True)
        header = self._view.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        # No stretch-last-section: with feature columns the last one would
        # balloon to fill the window while its neighbours stay clipped.
        header.setStretchLastSection(False)
        header.setDefaultSectionSize(150)
        right_layout.addWidget(self._view, 1)

        page_row = QHBoxLayout()
        page_row.setSpacing(SPACING["sm"])
        self._btn_prev = QPushButton("◀ Prev", right)
        self._btn_prev.clicked.connect(self.prev_page)
        self._btn_next = QPushButton("Next ▶", right)
        self._btn_next.clicked.connect(self.next_page)
        self._page_label = QLabel("", right)
        self._page_label.setObjectName("Muted")
        self._page_size_box = QSpinBox(right)
        self._page_size_box.setRange(*PAGE_SIZE_RANGE)
        self._page_size_box.setSingleStep(25)
        self._page_size_box.setValue(DEFAULT_PAGE_SIZE)
        self._page_size_box.setToolTip("(int) Rows fetched per page.")
        self._page_size_box.valueChanged.connect(self._on_page_size_changed)
        page_row.addWidget(self._btn_prev)
        page_row.addWidget(self._btn_next)
        page_row.addWidget(self._page_label, 1)
        page_row.addWidget(QLabel("Rows / page", right))
        page_row.addWidget(self._page_size_box)
        right_layout.addLayout(page_row)

        split.addWidget(right)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([220, 900])
        outer.addWidget(split, 1)

        # ── Filter + export row ───────────────────────────────────────
        filt_row = QHBoxLayout()
        filt_row.setSpacing(SPACING["sm"])
        filt_row.addWidget(QLabel("Filter", self))
        self._filter_col = QComboBox(self)
        self._filter_col.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self._filter_col.setMinimumWidth(180)
        self._filter_op = QComboBox(self)
        self._filter_op.addItems(list(OPERATORS.keys()))
        self._filter_value = QLineEdit(self)
        self._filter_value.setPlaceholderText("value")
        self._filter_value.returnPressed.connect(self.apply_filter)
        self._raw_toggle = QCheckBox("raw SQL", self)
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
        self._btn_export = QPushButton("Export filtered CSV…", self)
        self._btn_export.clicked.connect(self._pick_export_path)
        for w in (self._filter_col, self._filter_op, self._filter_value,
                  self._raw_edit):
            w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        filt_row.addWidget(self._filter_col, 2)
        filt_row.addWidget(self._filter_op, 1)
        filt_row.addWidget(self._filter_value, 2)
        filt_row.addWidget(self._raw_edit, 4)
        filt_row.addWidget(self._raw_toggle)
        filt_row.addWidget(self._btn_apply)
        filt_row.addWidget(self._btn_clear)
        filt_row.addWidget(self._btn_export)
        outer.addLayout(filt_row)

        self._status = QLabel("", self)
        self._status.setObjectName("Muted")
        self._status.setWordWrap(True)
        self._status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        outer.addWidget(self._status)

        # Wire the enablement-affecting signals last, once every widget
        # _update_controls touches actually exists. Both signals carry an
        # argument the slot doesn't want, hence the *_ lambdas.
        self._filter_op.currentTextChanged.connect(
            lambda *_: self._update_controls())
        self._raw_toggle.toggled.connect(lambda *_: self._update_controls())

    # -- status ------------------------------------------------------------

    def _set_status(self, text: str, error: bool = False) -> None:
        """Report inline. Deliberately never a QMessageBox — a modal dialog
        would hang a headless run (and did, in MakeMasksScreen)."""
        self.last_error = text if error else ""
        colour = PALETTE["error"] if error else PALETTE["fg_muted"]
        self._status.setStyleSheet(f"color: {colour};")
        self._status.setText(text)

    def status_text(self) -> str:
        """Current inline status message (test/introspection helper)."""
        return self._status.text()

    # -- database selection ------------------------------------------------

    def _pick_database(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open measurements database", "",
            "SQLite databases (*.db *.sqlite *.sqlite3);;All files (*)")
        if path:
            self.set_database(path)

    def _pick_run_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose a run folder", "")
        if path:
            self.set_database(path)

    def _on_open_typed_path(self) -> None:
        self.set_database(self._path_edit.text())

    def set_database(self, path: str) -> bool:
        """Open ``path`` read-only and list its tables.

        Accepts the database file or a run ``src`` folder. Any problem
        (missing file, not a sqlite database, unreadable) is reported in
        the status label and returns ``False`` — this never raises.

        :returns: True when a database was opened.
        """
        self._model.clear()
        self._table_list.clear()
        self._db = None
        self._table = ""
        self._all_columns = []
        self._page_index = 0
        self._row_count = 0
        self._where, self._params, self._filter_label = None, (), ""
        self._filter_col.clear()
        try:
            db = ReadOnlyDb(path)
        except Exception as e:
            self._set_status(self._humanise(e, path), error=True)
            self._update_controls()
            return False
        self._db = db
        self._path_edit.setText(db.path)
        try:
            tables = db.tables(refresh=True)
        except Exception as e:
            self._set_status(self._humanise(e, path), error=True)
            self._update_controls()
            return False
        for name in tables:
            self._table_list.addItem(QListWidgetItem(name))
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
        # Selecting row 0 fires _on_table_selected, which loads the preview.
        self._table_list.setCurrentRow(0)
        self._update_controls()
        return True

    @staticmethod
    def _humanise(exc: Exception, path: str) -> str:
        """Turn a sqlite/OS error into one line a biologist can act on."""
        msg = str(exc).strip() or exc.__class__.__name__
        if isinstance(exc, sqlite3.DatabaseError) and "not a database" in msg.lower():
            return (f"{path} is not a SQLite database "
                    f"(sqlite said: {msg}).")
        if isinstance(exc, sqlite3.OperationalError) and "unable to open" in msg.lower():
            return f"Could not open {path} — {msg}."
        return msg

    def database_path(self) -> str:
        """Path of the open database, or ``''``."""
        return self._db.path if self._db is not None else ""

    def tables(self) -> List[str]:
        """Table names currently listed in the sidebar."""
        return [self._table_list.item(i).text()
                for i in range(self._table_list.count())]

    # -- table selection ---------------------------------------------------

    def _on_table_selected(self, current, _previous=None) -> None:
        if current is None:
            return
        self.select_table(current.text())

    def select_table(self, name: str) -> bool:
        """Make ``name`` the previewed table; resets paging and the filter."""
        if self._db is None:
            self._set_status("No database open.", error=True)
            return False
        try:
            self._all_columns = self._db.columns(name)
        except Exception as e:
            self._set_status(self._humanise(e, name), error=True)
            return False
        self._table = name
        self._page_index = 0
        self._where, self._params, self._filter_label = None, (), ""
        self._raw_edit.clear()
        self._filter_value.clear()
        self._filter_col.blockSignals(True)
        self._filter_col.clear()
        self._filter_col.addItems(self._all_columns)
        self._filter_col.blockSignals(False)
        # Keep the current selection in the table list in sync when this
        # was called programmatically.
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
        return self._table

    # -- column search -----------------------------------------------------

    def set_column_filter(self, text: str) -> None:
        """Narrow the displayed columns to those containing ``text``.

        Purely a view operation — no re-query — so it stays instant on a
        table with hundreds of feature columns. An empty string restores
        every column.
        """
        if self._col_search.text() != (text or ""):
            self._col_search.setText(text or "")
            return   # textChanged re-enters with the same value
        self._model.set_column_filter(text or "")
        self._update_column_count()
        self._autosize_columns()

    def visible_columns(self) -> List[str]:
        """Column names currently shown in the preview."""
        return self._model.visible_columns()

    def preview_columns(self) -> List[str]:
        """Every column of the previewed page, ignoring the column search."""
        return self._model.all_columns()

    def preview_rows(self) -> List[tuple]:
        """Rows of the page on screen, as tuples in schema-column order."""
        return self._model.rows()

    def _autosize_columns(self) -> None:
        """Fit columns to their content, but only when there aren't many.

        A 500-column feature table keeps the fixed default width — the
        user narrows it with the column search first.
        """
        if 0 < self._model.columnCount() <= AUTOSIZE_MAX_COLUMNS:
            self._view.resizeColumnsToContents()

    def _update_column_count(self) -> None:
        total = len(self._model.all_columns())
        shown = len(self._model.visible_columns())
        if not total:
            self._col_count_label.setText("")
        elif shown == total:
            self._col_count_label.setText(f"{total} columns")
        else:
            self._col_count_label.setText(f"{shown} of {total} columns")

    # -- paging ------------------------------------------------------------

    def page_size(self) -> int:
        return int(self._page_size_box.value())

    def page_index(self) -> int:
        """Zero-based index of the page on screen."""
        return self._page_index

    def row_count(self) -> int:
        """``COUNT(*)`` for the current table + filter."""
        return self._row_count

    def _on_page_size_changed(self, _value: int) -> None:
        self._page_index = 0
        self.refresh()

    def next_page(self) -> None:
        if (self._page_index + 1) * self.page_size() >= self._row_count:
            return
        self._page_index += 1
        self.refresh()

    def prev_page(self) -> None:
        if self._page_index <= 0:
            return
        self._page_index -= 1
        self.refresh()

    # -- filtering ---------------------------------------------------------

    def apply_filter(self) -> bool:
        """Read the filter row, validate it, and reload from page 1.

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
        self._page_index = 0
        self.refresh()
        return True

    def clear_filter(self) -> None:
        """Drop the WHERE clause and reload."""
        self._where, self._params, self._filter_label = None, (), ""
        self._filter_value.clear()
        self._raw_edit.clear()
        self._page_index = 0
        self.refresh()

    def _collect_filter(self) -> Tuple[Optional[str], tuple, str]:
        """Return ``(where, params, human_label)`` from the filter widgets."""
        if self._raw_toggle.isChecked():
            raw = validate_raw_predicate(self._raw_edit.text())
            return raw, (), raw
        column = self._filter_col.currentText()
        op = self._filter_op.currentText()
        if not column:
            return None, (), ""
        _, nargs, _ = OPERATORS[op]
        value = self._filter_value.text()
        if nargs and not value.strip():
            # Nothing typed: treat as "no filter" rather than an error.
            return None, (), ""
        where, params = build_where(column, op, value, self._all_columns)
        label = f"{column} {op}" + (f" {value}" if nargs else "")
        return where, params, label

    def set_filter(self, column: str, op: str, value: Any = "") -> bool:
        """Programmatic equivalent of filling in the filter row + Apply."""
        self._raw_toggle.setChecked(False)
        idx = self._filter_col.findText(column)
        if idx < 0:
            self._set_status(
                f"Filter error: Unknown column {column!r} — "
                f"not in this table's schema.", error=True)
            return False
        self._filter_col.setCurrentIndex(idx)
        op_idx = self._filter_op.findText(op)
        if op_idx >= 0:
            self._filter_op.setCurrentIndex(op_idx)
        self._filter_value.setText("" if value is None else str(value))
        return self.apply_filter()

    def set_raw_filter(self, predicate: str) -> bool:
        """Programmatic equivalent of the raw-SQL box + Apply."""
        self._raw_toggle.setChecked(True)
        self._raw_edit.setText(predicate or "")
        return self.apply_filter()

    def where_clause(self) -> Optional[str]:
        """The active WHERE fragment, or None."""
        return self._where

    # -- refresh (off the GUI thread) --------------------------------------

    def refresh(self) -> None:
        """Re-run ``COUNT(*)`` + the current page, off the GUI thread."""
        if self._db is None or not self._table:
            return
        if self._busy:
            self._set_status("A query is already running…")
            return
        db, table = self._db, self._table
        where, params = self._where, self._params
        limit, offset = self.page_size(), self._page_index * self.page_size()

        def _job() -> Dict[str, Any]:
            total = db.count(table, where, params)
            cols, rows = db.page(table, limit=limit, offset=offset,
                                 where=where, params=params)
            return {"count": total, "columns": cols, "rows": rows,
                    "offset": offset}

        self._run_job(_job, self._apply_page_result)

    def _apply_page_result(self, result: Dict[str, Any]) -> None:
        self._row_count = int(result.get("count", 0))
        columns = result.get("columns", [])
        rows = result.get("rows", [])
        offset = int(result.get("offset", 0))
        self._model.set_page(columns, rows, row_offset=offset)
        self._update_column_count()
        self._autosize_columns()
        self._update_page_label(offset, len(rows))
        bits = [f"{self._table}: {self._row_count:,} row"
                f"{'s' if self._row_count != 1 else ''}",
                f"{len(columns)} columns"]
        if self._filter_label:
            bits.append(f"filter: {self._filter_label}")
        self._set_status(" · ".join(bits) + "  (read-only)")

    def _update_page_label(self, offset: int, n_rows: int) -> None:
        if self._row_count == 0:
            self._page_label.setText("no rows")
            return
        first = offset + 1 if n_rows else offset
        last = offset + n_rows
        pages = max(1, (self._row_count + self.page_size() - 1) // self.page_size())
        self._page_label.setText(
            f"rows {first:,}–{last:,} of {self._row_count:,}   "
            f"(page {self._page_index + 1} of {pages:,})")

    # -- export ------------------------------------------------------------

    def _pick_export_path(self) -> None:
        default = f"{self._table or 'export'}.csv"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export filtered rows to CSV", default,
            "CSV files (*.csv);;All files (*)")
        if path:
            self.export_csv(path)

    def export_csv(self, out_path: str) -> bool:
        """Write the current table + filter + visible columns to ``out_path``.

        Runs off the GUI thread and streams the result, so a
        400 000-row export neither freezes the window nor loads the table
        into memory. Reports inline on failure and returns ``False``.
        """
        if self._db is None or not self._table:
            self._set_status("Open a database and pick a table first.",
                             error=True)
            return False
        if self._busy:
            self._set_status("A query is already running…", error=True)
            return False
        db, table = self._db, self._table
        where, params = self._where, self._params
        # Honour the column search: what you filtered down to is what you get.
        columns = self._model.visible_columns() or self._all_columns

        def _job() -> Dict[str, Any]:
            n = db.export_csv(out_path, table, columns=columns,
                              where=where, params=params)
            return {"exported": n, "path": out_path,
                    "columns": len(columns)}

        def _done(result: Dict[str, Any]) -> None:
            self._set_status(
                f"Exported {result['exported']:,} row"
                f"{'s' if result['exported'] != 1 else ''} × "
                f"{result['columns']} columns → {result['path']}")

        return self._run_job(_job, _done)

    # -- job plumbing ------------------------------------------------------

    def _run_job(self, fn: Callable[[], Any],
                 on_done: Callable[[Any], None]) -> bool:
        """Run ``fn`` off the GUI thread and hand its result to ``on_done``.

        Uses :func:`spacr.qt.bridge.make_thread` — the same QThread +
        worker pairing the pipeline screens use — so there is exactly one
        threading idiom in the Qt layer. ``PipelineWorker`` calls
        ``fn(settings)``; we pass a private dict and let the closure drop
        its return value in, since the worker's ``finished`` signal only
        carries a success flag.

        With ``threaded=False`` (tests) the call runs inline and the same
        signals fire, so both paths behave identically from the outside.

        :returns: for the synchronous path, whether the job succeeded; for
            the threaded path, ``True`` once the job has been started.
        """
        if not self._threaded:
            ok = True
            try:
                on_done(fn())
            except Exception as e:
                self._on_job_error(e)
                ok = False
            self._update_controls()
            self.job_finished.emit(ok)
            return ok

        box: Dict[str, Any] = {}

        def _job(payload: Dict[str, Any]) -> None:
            payload["result"] = fn()

        thread, worker = make_thread(_job, box)
        # Strong references: PySide6 will not keep the worker alive through
        # the started→run connection alone, and a collected worker means the
        # thread spins forever without ever calling run(). Same fix as
        # AppScreen._on_run — but held per-job, keyed by identity.
        self._jobs.append((thread, worker))
        self._thread, self._worker = thread, worker
        worker.error.connect(self._on_worker_error_text)

        def _finished(ok: bool) -> None:
            self._busy = False
            if ok:
                try:
                    on_done(box.get("result"))
                except Exception as e:      # pragma: no cover - defensive
                    self._on_job_error(e)
                    ok = False
            self._update_controls()
            self.job_finished.emit(ok)

        worker.finished.connect(_finished)
        thread.finished.connect(lambda t=thread: self._retire_job(t))
        self._busy = True
        self._update_controls()
        thread.start()
        return True

    def _retire_job(self, thread) -> None:
        """Release *this* job's refs once its own event loop has exited.

        Matching by identity matters: a plain "clear the refs" slot would
        drop the references of whichever job happens to be current when a
        previous thread finishes, and a QThread garbage-collected while it
        is still running takes the whole process down with it.
        """
        self._jobs = [(t, w) for (t, w) in self._jobs if t is not thread]
        if self._thread is thread:
            self._thread = None
            self._worker = None

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
        self._set_status(f"Query failed: {exc}", error=True)

    def is_busy(self) -> bool:
        return self._busy

    # -- enablement --------------------------------------------------------

    def _update_controls(self) -> None:
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
            w.setEnabled(has_table and not self._busy)
        self._btn_prev.setEnabled(
            has_table and not self._busy and self._page_index > 0)
        self._btn_next.setEnabled(
            has_table and not self._busy
            and (self._page_index + 1) * self.page_size() < self._row_count)
        self._table_list.setEnabled(has_db and not self._busy)

    # -- shutdown ----------------------------------------------------------

    def closeEvent(self, event):  # noqa: N802
        """Let every in-flight query thread finish before the widget dies.

        Destroying a QThread that is still running aborts the process, so
        we wait (briefly) rather than hope.
        """
        for thread, _worker in list(self._jobs):
            try:
                if thread.isRunning():
                    thread.quit()
                    thread.wait(5000)
            except RuntimeError:
                pass
        super().closeEvent(event)

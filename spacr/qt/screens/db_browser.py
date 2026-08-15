"""
Database Browser — a read-only query panel for a spaCR ``measurements.db``.

Answering "how many cells are in plate1?" or "what does the ``png_list``
table actually contain?" used to mean dropping to a terminal and typing
``sqlite3 .../measurements/measurements.db``. This screen puts the same
questions one click away, and — unless you deliberately arm edit mode —
without ever giving the GUI a way to write to the file.

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

* **Never ``SELECT *`` the whole table.** Measurement tables run to
  hundreds of thousands of rows and hundreds of feature columns. The
  first chunk (100 rows) is fetched and painted on its own; the rest
  arrives as the user scrolls, through ``canFetchMore`` / ``fetchMore``.
* **Keyset paging, not OFFSET.** Each chunk asks for ``rowid > <last
  rowid seen>`` and ``ORDER BY rowid``. ``LIMIT ? OFFSET ?`` makes
  SQLite walk (and throw away) every skipped row, so chunk 500 of a
  400 k-row table would cost 500× chunk 1 — the "fast" version would
  get slower the further you scrolled. Only tables with no usable key
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
  ON``. A write is rejected by SQLite itself, not by a check we could
  forget. Editing is a separate, opt-in path (see below) with its own
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
   SQLite will cheerfully store ``'abc'`` in an INTEGER column;
   :func:`coerce_for_column` refuses instead.

Loading a different database always resets edit mode to off.
"""
from __future__ import annotations

import contextlib
import csv
import os
import re
import sqlite3
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from urllib.parse import quote as _urlquote

from spacr.database_concurrency import (
    connect as connect_database,
    transaction,
)

import pandas as pd

from PySide6.QtCore import (
    QAbstractTableModel,
    QItemSelection,
    QItemSelectionModel,
    QModelIndex,
    Qt,
    Signal,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableView,
    QVBoxLayout,
    QWidget,
)
from ..widgets.toggle import Toggle

from ...selection import (OBJECT_KEY_COLUMNS, DataFilter, Selection,
                          with_object_type)
from ..bridge import make_thread
from ..linked_selection import LinkedView
from ..preferences import get_db_browser_editable
from ..theme import SPACING, active_palette
from ..widgets import Divider

__all__ = [
    "DB_FILENAME",
    "DEFAULT_PAGE_SIZE",
    "DbBrowserScreen",
    "EditRefused",
    "OPERATORS",
    "PreviewModel",
    "ReadOnlyDb",
    "WritableDb",
    "build_update",
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


# Statements that have no business inside a WHERE clause. The browsing
# connection is read-only anyway, so this is a second line of defence —
# it mostly stops a user pasting a whole script into the predicate box
# and being confused by the error.
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
            f"{m.group(1).upper()} is not allowed — filters only read.")
    return t


# ---------------------------------------------------------------------------
# Editing: types, coercion, and the one statement we are willing to run
# ---------------------------------------------------------------------------

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

    :raises ValueError: when ``text`` cannot be represented in the
        column's type.
    """
    decl = str(decl_type or "").strip()
    label = decl or "untyped"
    s = "" if text is None else str(text)
    if "BLOB" in decl.upper():
        raise ValueError(
            f"{column!r} is declared {label} — binary values cannot be "
            f"edited as text.")
    if not s.strip():
        return None
    stripped = s.strip()
    affinity = column_affinity(decl)
    if affinity == "INTEGER":
        if _INT_RE.match(stripped):
            return int(stripped)
        raise ValueError(
            f"{s!r} is not a whole number, and {column!r} is declared "
            f"{label}.")
    if affinity == "REAL":
        if _FLOAT_RE.match(stripped):
            return float(stripped)
        raise ValueError(
            f"{s!r} is not a number, and {column!r} is declared {label}.")
    if affinity == "TEXT":
        return s
    # NUMERIC, or a column with no declared type: mirror SQLite's own
    # behaviour — store a number when it is one, text otherwise.
    if _INT_RE.match(stripped):
        return int(stripped)
    if _FLOAT_RE.match(stripped):
        return float(stripped)
    return s


def build_update(table: str, column: str,
                 key_columns: Sequence[str]) -> str:
    """Return the one statement this screen is ever willing to run.

    ``UPDATE "t" SET "c" = ? WHERE "rowid" = ?`` — one column, one row
    address, everything bound. Exposed so the screen can show the user
    the exact SQL before it runs, and so tests can assert on it without
    a database.

    :raises EditRefused: when there is no row address at all. An UPDATE
        without a unique key would match on values and could rewrite
        thousands of rows.
    """
    if not key_columns:
        raise EditRefused(
            f"{table!r} has no rowid and no primary key — an UPDATE could "
            f"not be limited to one row, so spaCR will not run one.")
    where = " AND ".join(f"{quote_ident(k)} = ?" for k in key_columns)
    return (f"UPDATE {quote_ident(table)} SET {quote_ident(column)} = ? "
            f"WHERE {where}")


# ---------------------------------------------------------------------------
# Read-only sqlite access
# ---------------------------------------------------------------------------

class ReadOnlyDb:
    """A read-only handle on a sqlite database.

    Every method opens (and closes) its own connection. That costs a few
    microseconds and buys thread-safety for free: a chunk query runs on a
    worker thread while the GUI thread may still be listing tables, and
    ``sqlite3`` objects are not shareable across threads.

    :param path: database file, or a run folder — see :func:`resolve_db_path`.
    :raises sqlite3.DatabaseError: when the file is not a sqlite database.
    """

    def __init__(self, path: str):
        self.path = resolve_db_path(path)
        self.uri = _read_only_uri(self.path)
        #: SQL of the most recent statement — handy in tests + bug reports.
        self.last_sql: str = ""
        self._tables: Optional[List[str]] = None
        self._row_keys: Dict[str, Tuple[str, List[str]]] = {}
        self._types: Dict[str, Dict[str, str]] = {}
        # Probe now so "that file isn't a database" surfaces at open time
        # rather than three clicks later.
        with self._con() as con:
            con.execute("SELECT name FROM sqlite_master LIMIT 1").fetchall()

    # -- connections -------------------------------------------------------

    def connect(self) -> sqlite3.Connection:
        """Return a fresh read-only connection. The caller closes it."""
        return connect_database(self.path, readonly=True)

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

    def table_info(self, table: str) -> List[tuple]:
        """Raw ``PRAGMA table_info`` rows for ``table``."""
        self.check_table(table)
        with self._con() as con:
            # PRAGMA takes no bound parameters; `table` is schema-validated
            # above and quoted here.
            return self._execute(
                con, f"PRAGMA table_info({quote_ident(table)})").fetchall()

    def columns(self, table: str) -> List[str]:
        """Return the column names of ``table`` in declaration order."""
        return [r[1] for r in self.table_info(table)]

    def column_types(self, table: str) -> Dict[str, str]:
        """Return ``{column: declared type}`` — ``''`` for untyped columns.

        Cached: the edit path asks for this on every keystroke-committed
        cell, and the schema cannot change under a read-only connection.
        """
        if table not in self._types:
            self._types[table] = {r[1]: str(r[2] or "")
                                  for r in self.table_info(table)}
        return dict(self._types[table])

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

    def row_key(self, table: str) -> Tuple[str, List[str]]:
        """Return how one row of ``table`` can be addressed uniquely.

        ``("rowid", ["rowid"])`` for an ordinary table, ``("pk", [...])``
        for a ``WITHOUT ROWID`` table (its declared primary key), and
        ``("", [])`` for anything with neither — a view, or a table
        created ``WITHOUT ROWID`` with no primary key.

        Two things hang off this: paging (keyset needs an ordered key)
        and editing (an UPDATE without a unique address is refused).
        """
        if table in self._row_keys:
            return self._row_keys[table]
        key: Tuple[str, List[str]]
        # A view has no intrinsic row address. Probing ``SELECT _rowid_`` is
        # not portable discovery: newer SQLite releases may accept that
        # expression on a view and return NULL, which made CI arm editing for
        # a result set that no UPDATE can address uniquely. Determine the
        # schema object kind first; only a real table gets the rowid probe.
        self.check_table(table)
        with self._con() as con:
            object_row = self._execute(
                con,
                "SELECT type FROM sqlite_master WHERE name = ? LIMIT 1",
                (table,),
            ).fetchone()
        if object_row is None or str(object_row[0]).lower() != "table":
            key = ("", [])
            self._row_keys[table] = key
            return key
        # SQLite identifiers are case-insensitive, and a table that DECLARES a
        # column named `rowid` makes the bare name resolve to that column
        # rather than to the implicit row id. png_list declares `rowID`, so
        # this probe used to SUCCEED there and hand back 'r1' -- after which
        # editing one cell issued `UPDATE png_list SET c = ? WHERE rowid =
        # 'r1'` and rewrote every crop in that plate row, and keyset paging
        # ordered a TEXT column as if it were the row id. Ask for an alias the
        # table does not shadow.
        from ...predictions import _rowid_alias
        alias = _rowid_alias([str(r[1]) for r in self.table_info(table)])
        try:
            with self._con() as con:
                self._execute(
                    con,
                    f"SELECT {alias} FROM {quote_ident(table)} LIMIT 1").fetchall()
            key = ("rowid", [alias])
        except sqlite3.Error:
            pk = [r for r in self.table_info(table) if int(r[5] or 0) > 0]
            pk.sort(key=lambda r: int(r[5]))
            key = ("pk", [r[1] for r in pk]) if pk else ("", [])
        self._row_keys[table] = key
        return key

    # -- queries -----------------------------------------------------------

    def select_sql(self, table: str, columns: Sequence[str],
                   where: Optional[str] = None) -> str:
        """Build the unbounded SELECT used by the CSV export."""
        col_sql = ", ".join(quote_ident(c) for c in columns)
        sql = f"SELECT {col_sql} FROM {quote_ident(table)}"
        if where:
            sql += f" WHERE {where}"
        key_kind, key_cols = self.row_key(table)
        if key_kind == "rowid":
            # key_cols[0], not the literal "rowid" -- png_list declares a
            # rowID column that shadows the bare name.
            sql += f" ORDER BY {quote_ident(key_cols[0])}"
        return sql

    def chunk_sql(self, table: str, columns: Sequence[str],
                  key_columns: Sequence[str], where: Optional[str] = None,
                  after: bool = False, use_offset: bool = False,
                  order_by: Optional[Tuple[str, bool]] = None) -> str:
        """Build the paged chunk SELECT. Exposed so tests can read it.

        ``after`` says whether a ``key > ?`` clause is wanted (i.e. this
        is not the first chunk). ``use_offset`` is the fallback for
        tables that have no single-column key.

        ``order_by`` is ``(column, descending)`` when the user has clicked a
        column header. It sorts **in SQL, over the whole table**, which is
        the only way to sort a table the view has only partly loaded --
        sorting the rows fetched so far and presenting that as the table's
        order is the one option worse than not sorting at all.

        Two details that are not optional:

        * the key column is appended as a **tiebreak**. Without it, rows
          sharing a value come back in whatever order SQLite happens to
          produce, and that order is free to differ between two chunks of
          the same scroll -- so a row could be shown twice and another not
          at all.
        * paging falls back to ``OFFSET``. Keyset paging needs the ordering
          column to be the one being compared, and ``(value, rowid) > (?, ?)``
          over an arbitrary user-chosen column is a different and much more
          delicate query. OFFSET makes deep scrolling of a sorted view
          progressively slower, which is a real cost and the reason it is
          not the default; it is bounded here because a sort is an explicit
          act on a table the user is looking at.
        """
        col_sql = ", ".join(quote_ident(c)
                            for c in list(key_columns) + list(columns))
        sql = f"SELECT {col_sql} FROM {quote_ident(table)}"
        clauses = []
        if where:
            clauses.append(f"({where})")
        if after and not use_offset and order_by is None:
            clauses.append(f"{quote_ident(key_columns[0])} > ?")
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        if order_by is not None:
            column, descending = order_by
            terms = [f"{quote_ident(column)} {'DESC' if descending else 'ASC'}"]
            if key_columns:
                terms.append(f"{quote_ident(key_columns[0])} ASC")
            sql += " ORDER BY " + ", ".join(terms)
        elif key_columns and not use_offset:
            sql += f" ORDER BY {quote_ident(key_columns[0])}"
        sql += " LIMIT ?"
        if use_offset or order_by is not None:
            sql += " OFFSET ?"
        return sql

    def chunk(self, table: str, columns: Optional[Sequence[str]] = None,
              where: Optional[str] = None, params: Sequence = (),
              limit: int = DEFAULT_PAGE_SIZE,
              after: Optional[tuple] = None,
              loaded: int = 0,
              order_by: Optional[Tuple[str, bool]] = None
              ) -> Tuple[List[str], List[tuple], List[Optional[tuple]]]:
        """Return ``(columns, rows, keys)`` for the next ``limit`` rows.

        ``after`` is the key tuple of the last row already loaded;
        ``None`` asks for the first chunk. ``keys[i]`` addresses
        ``rows[i]`` uniquely (or is ``None`` when the table has no key,
        in which case the caller must refuse to edit it).

        :param loaded: only used by the ``OFFSET`` fallback for tables
            with no single-column key.
        """
        self.check_table(table)
        cols = self.check_columns(table, columns)
        _kind, key_cols = self.row_key(table)
        use_offset = len(key_cols) != 1
        if order_by is not None:
            # Validated against the table's real columns, not merely quoted:
            # this string reaches an ORDER BY clause, and check_columns is
            # the same gate every other column name in this class goes
            # through.
            self.check_columns(table, [order_by[0]])
        sql = self.chunk_sql(table, cols, key_cols, where,
                             after=after is not None, use_offset=use_offset,
                             order_by=order_by)
        args: List[Any] = list(params)
        if after is not None and not use_offset and order_by is None:
            args.append(after[0])
        args.append(int(max(1, limit)))
        if use_offset or order_by is not None:
            args.append(int(max(0, loaded)))
        with self._con() as con:
            raw = self._execute(con, sql, args).fetchall()
        n = len(key_cols)
        keys: List[Optional[tuple]] = ([tuple(r[:n]) for r in raw] if n
                                       else [None] * len(raw))
        rows = [tuple(r[n:]) for r in raw]
        return cols, rows, keys

    def estimate_count(self, table: str) -> Optional[int]:
        """Return an O(1) *estimate* of the row count, or ``None``.

        ``max(rowid)`` is answered from the right-hand edge of the
        table's b-tree without scanning, which is the whole point: a
        real ``COUNT(*)`` on a 400 k-row measurement table takes long
        enough to be felt. It is only an estimate — deleted rows leave
        gaps — so every caller must label it as one.

        ``None`` when the table has no rowid, or is empty.
        """
        self.check_table(table)
        key_kind, key_cols = self.row_key(table)
        if key_kind != "rowid":
            return None
        with self._con() as con:
            row = self._execute(
                con,
                f"SELECT max({quote_ident(key_cols[0])}) FROM {quote_ident(table)}"
            ).fetchone()
        if not row or row[0] is None:
            return None
        return int(row[0])

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
        """Return the exact ``COUNT(*)`` under the current filter.

        This one *is* a full scan — never call it on the path that
        paints the first chunk.
        """
        self.check_table(table)
        sql = f"SELECT COUNT(*) FROM {quote_ident(table)}"
        if where:
            sql += f" WHERE {where}"
        with self._con() as con:
            row = self._execute(con, sql, params).fetchone()
        return int(row[0]) if row else 0

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
        sql = self.select_sql(table, cols, where)
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
# Read-write sqlite access — the opt-in edit path, and nothing else
# ---------------------------------------------------------------------------

class WritableDb:
    """A read-write handle used only by an armed edit mode.

    Deliberately tiny: it knows how to write **one cell of one row** and
    nothing else. There is no ``execute()``, no DDL, no multi-row
    update — the class simply has no method that could touch more than
    a single row.

    :param path: database file, or a run folder — see :func:`resolve_db_path`.
    """

    def __init__(self, path: str):
        self.path = resolve_db_path(path)
        #: SQL of the last statement that ran (or was about to).
        self.last_sql: str = ""

    def connect(self) -> sqlite3.Connection:
        """Return a fresh read-write connection in autocommit mode.

        ``isolation_level = None`` hands transaction control back to us,
        so the BEGIN/COMMIT/ROLLBACK in :meth:`update_cell` are the real
        ones and do not fight Python's implicit transaction handling
        (which differs between 3.11 and 3.12+).
        """
        return connect_database(self.path)

    def update_cell(self, table: str, column: str, value: Any,
                    key_columns: Sequence[str],
                    key_values: Sequence[Any]) -> str:
        """Set one column of one row. Returns the SQL that ran.

        The guards, in order:

        1. ``table`` must be a real table in ``sqlite_master`` — a view
           has no row to update.
        2. ``column`` and every key column must exist in its schema.
        3. the row address must match **exactly one** row (probed with
           ``COUNT(*)`` before anything is written);
        4. the UPDATE itself must report ``rowcount == 1``, or the
           transaction is rolled back.

        :raises EditRefused: when any guard fires — nothing was written.
        :raises sqlite3.Error: when SQLite refuses the write itself
            (constraint violation, read-only file, locked database).
        """
        if not key_columns:
            raise EditRefused(
                f"{table!r} has no rowid and no primary key — an UPDATE "
                f"could not be limited to one row, so spaCR will not run one.")
        if len(key_columns) != len(key_values):
            raise EditRefused(
                "Row address is incomplete — refusing to guess which row "
                "you meant.")
        sql = build_update(table, column, key_columns)
        self.last_sql = sql
        con = self.connect()
        try:
            # Validation and update share one write transaction. This closes
            # the former check-then-write window where another connection
            # could remove or replace the addressed row between COUNT and
            # UPDATE.
            with transaction(con):
                real = con.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' "
                    "AND name = ?", (table,)).fetchone()
                if real is None:
                    raise EditRefused(
                        f"{table!r} is not an editable table in "
                        f"{os.path.basename(self.path)} — views and virtual "
                        f"tables have no single row to update.")
                known = {r[1] for r in con.execute(
                    f"PRAGMA table_info({quote_ident(table)})").fetchall()}
                if column not in known:
                    raise EditRefused(
                        f"{table!r} has no column {column!r}.")
                # All three implicit row-id spellings are legal here, not just
                # "rowid": row_key() returns one the table does not shadow.
                implicit = {"rowid", "oid", "_rowid_"}
                missing = [
                    key for key in key_columns
                    if key.lower() not in implicit and key not in known
                ]
                if missing:
                    raise EditRefused(
                        f"{table!r} has no column(s) "
                        f"{', '.join(map(repr, missing))} to address a row by.")
                where = " AND ".join(
                    f"{quote_ident(key)} = ?" for key in key_columns)
                n = int(con.execute(
                    f"SELECT COUNT(*) FROM {quote_ident(table)} WHERE {where}",
                    tuple(key_values)).fetchone()[0])
                if n != 1:
                    raise EditRefused(
                        f"that row address matches {n} rows, not 1 — refusing "
                        f"to write. Reload the table and try again.")
                cur = con.execute(sql, (value,) + tuple(key_values))
                if cur.rowcount != 1:
                    raise EditRefused(
                        f"the UPDATE would have touched {cur.rowcount} rows, "
                        f"not 1 — rolled back.")
        finally:
            con.close()
        return sql


# ---------------------------------------------------------------------------
# Table model
# ---------------------------------------------------------------------------

def _capture_result(fn: Callable[[], Any], payload: Dict[str, Any]) -> None:
    """Run ``fn`` and leave its return value in ``payload``.

    ``PipelineWorker`` calls its function as ``fn(settings)`` and its
    ``finished`` signal only carries a success flag, so a job's actual
    result has to travel in the settings dict it was handed. Runs on the
    worker thread; touches nothing but ``payload``.
    """
    payload["result"] = fn()


def _sort_key(value: Any) -> tuple:
    """Total order over a sqlite column: NULLs, then numbers, then text."""
    if value is None:
        return (0, 0.0, "")
    if isinstance(value, (int, float)):
        return (1, float(value), "")
    return (2, 0.0, str(value))


class PreviewModel(QAbstractTableModel):
    """Holds the rows fetched so far plus the column-visibility mask.

    Two jobs beyond the obvious one:

    * **Incremental fetch.** ``canFetchMore``/``fetchMore`` are the Qt
      way to say "there is more where that came from"; the view calls
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
    """

    def __init__(self, parent=None):
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

    # -- data ---------------------------------------------------------------

    def set_page(self, columns: Sequence[str], rows: Sequence[Sequence[Any]],
                 row_offset: int = 0,
                 keys: Optional[Sequence[Optional[tuple]]] = None) -> None:
        """Replace the model contents, keeping the current column search."""
        self.beginResetModel()
        self._columns = list(columns)
        self._rows = [tuple(r) for r in rows]
        self._keys = (list(keys) if keys is not None
                      else [None] * len(self._rows))
        self._row_offset = int(row_offset)
        self._recompute_visible()
        self.endResetModel()

    def append_rows(self, rows: Sequence[Sequence[Any]],
                    keys: Optional[Sequence[Optional[tuple]]] = None) -> int:
        """Add a fetched chunk to the end. Returns how many rows landed."""
        new = [tuple(r) for r in rows]
        if not new:
            return 0
        first = len(self._rows)
        self.beginInsertRows(QModelIndex(), first, first + len(new) - 1)
        self._rows.extend(new)
        self._keys.extend(list(keys) if keys is not None
                          else [None] * len(new))
        self.endInsertRows()
        return len(new)

    def clear(self) -> None:
        self.set_page([], [], 0)

    def all_columns(self) -> List[str]:
        return list(self._columns)

    def rows(self) -> List[tuple]:
        """The raw rows loaded so far, with every column (search-independent)."""
        return list(self._rows)

    def row_key(self, row: int) -> Optional[tuple]:
        """The key tuple addressing ``row``, or ``None`` when it has none."""
        if 0 <= row < len(self._keys):
            return self._keys[row]
        return None

    def value(self, row: int, column: str) -> Any:
        """The stored (unformatted) value at ``row`` / ``column``."""
        if not (0 <= row < len(self._rows)) or column not in self._columns:
            return None
        return self._rows[row][self._columns.index(column)]

    def set_value(self, row: int, column: str, value: Any) -> bool:
        """Write a value back into the in-memory page after a real UPDATE."""
        if not (0 <= row < len(self._rows)) or column not in self._columns:
            return False
        col = self._columns.index(column)
        current = list(self._rows[row])
        current[col] = value
        self._rows[row] = tuple(current)
        if col in self._visible:
            idx = self.index(row, self._visible.index(col))
            self.dataChanged.emit(idx, idx, [Qt.DisplayRole, Qt.EditRole])
        return True

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

    # -- incremental fetching -----------------------------------------------

    def set_fetch_hook(self, hook: Optional[Callable[[], Any]]) -> None:
        """Set what :meth:`fetchMore` calls to ask for the next chunk."""
        self._fetch_hook = hook

    def set_more(self, more: bool) -> None:
        """Say whether another chunk can be fetched right now."""
        self._more = bool(more)

    def canFetchMore(self, parent=QModelIndex()) -> bool:  # noqa: N802
        return (not parent.isValid()) and self._more

    def fetchMore(self, parent=QModelIndex()) -> None:  # noqa: N802
        if parent.isValid() or not self._more or self._fetch_hook is None:
            return
        self._fetch_hook()

    # -- editing ------------------------------------------------------------

    def set_commit_hook(self,
                        hook: Optional[Callable[[int, str, Any], bool]]) -> None:
        """Set what :meth:`setData` calls to actually write a cell."""
        self._commit_hook = hook

    def set_editable(self, editable: bool) -> None:
        """Turn cell editing on or off (and tell the view to repaint flags)."""
        editable = bool(editable)
        if editable == self._editable:
            return
        self.beginResetModel()
        self._editable = editable
        self.endResetModel()

    def is_editable(self) -> bool:
        return self._editable

    # -- QAbstractTableModel ------------------------------------------------

    def rowCount(self, parent=QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent=QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._visible)

    def flags(self, index):
        base = super().flags(index)
        if index.isValid() and self._editable:
            base |= Qt.ItemIsEditable
        return base

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or role not in (Qt.DisplayRole, Qt.ToolTipRole,
                                               Qt.EditRole):
            return None
        try:
            value = self._rows[index.row()][self._visible[index.column()]]
        except IndexError:
            return None
        if value is None:
            return ""
        if isinstance(value, bytes):
            return f"<{len(value)} bytes>"
        if role == Qt.EditRole:
            # The editor must start from the *exact* stored value, or a
            # cell the user opens and closes without touching would be
            # written back rounded.
            return repr(value) if isinstance(value, float) else str(value)
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    def setData(self, index, value, role=Qt.EditRole) -> bool:  # noqa: N802
        if (role != Qt.EditRole or not index.isValid() or not self._editable
                or self._commit_hook is None):
            return False
        try:
            column = self._columns[self._visible[index.column()]]
        except IndexError:
            return False
        return bool(self._commit_hook(index.row(), column, value))

    def headerData(self, section, orientation, role=Qt.DisplayRole):  # noqa: N802
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
        self.endResetModel()


# ---------------------------------------------------------------------------
# Screen
# ---------------------------------------------------------------------------

class DbBrowserScreen(LinkedView, QWidget):
    """Browser for a spaCR measurements database — read-only by default.

    Joined to the shared selection as ``"db_browser"``, in both directions:

    * selecting rows publishes them, so a row picked out here lights up on
      the plate heatmap and in the UMAP;
    * a selection published elsewhere selects and scrolls to the same rows;
    * a filter published elsewhere HIDES rows, which a selection never does.

    All three act on the rows already fetched. Nothing here re-queries: the
    shared filter is a lens over the page in memory, and the row-count label
    keeps saying how much of the table that page is.

    :param threaded: run queries on a worker thread (the default). Tests
        pass ``False`` to get deterministic, synchronous behaviour.
    :ivar last_error: text of the most recent failure, ``""`` when the
        last operation succeeded. Errors are *only* ever reported here
        and in the inline status label — never in a modal dialog.
    :ivar confirm_edit_mode: callable taking the confirmation text and
        returning a bool. Replace it to arm edit mode without a dialog
        (every test does). The default opens the one QMessageBox this
        screen owns.
    :ivar auto_count: when True (the default) the exact ``COUNT(*)``
        follows the first chunk on its own job. Set False to browse a
        very large table on the ``max(rowid)`` estimate alone.
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
        super().__init__(parent)
        self._threaded = bool(threaded)
        self._db: Optional[ReadOnlyDb] = None
        self._table: str = ""
        self._all_columns: List[str] = []
        self._where: Optional[str] = None
        self._params: tuple = ()
        self._filter_label: str = ""

        # -- incremental load state ---------------------------------------
        # Every load carries a token. A result whose token is stale (the
        # user switched table or database while it was in flight) is
        # dropped instead of painted — that race is the reason async
        # loading can otherwise feel *worse* than synchronous loading.
        self._token: int = 0
        #: ``(column, descending)`` while a header sort is active, else None.
        #: The sort happens in SQL over the WHOLE table, so it is correct
        #: however little of the table the view has loaded.
        self._sort: Optional[Tuple[str, bool]] = None
        self._loaded: int = 0
        self._last_key: Optional[tuple] = None
        self._exhausted: bool = False
        self._exact_count: Optional[int] = None
        self._estimate: Optional[int] = None
        self.auto_count: bool = True

        # -- job state -----------------------------------------------------
        self._export_busy: bool = False
        self._load_jobs: int = 0
        self._chunk_jobs: int = 0
        # job id -> (QThread, PipelineWorker) for every job that has been
        # started and whose event loop has not yet exited. This is an
        # ownership table, not a convenience: PySide6 destroys the C++
        # QThread as soon as the last Python reference goes, and
        # destroying a *running* QThread aborts the process. A single
        # `self._thread` slot is not enough, because `worker.finished`
        # (which lets the next job start) fires strictly before
        # `thread.finished` (which retires the old one) — so two jobs
        # legitimately overlap for a moment.
        self._jobs: Dict[int, tuple] = {}
        self._thread = None     # most recent thread, for introspection
        self._worker = None
        # job id -> (result box, completion callback, kind). Keyed by id
        # rather than FIFO because loads legitimately overlap: a chunk
        # the user abandoned can settle *after* the one that replaced it.
        self._pending: Dict[int, tuple] = {}
        # Jobs waiting for the single worker slot: (fn, on_done, kind,
        # token). See _run_job for why only one runs at a time.
        self._queue: List[tuple] = []
        self._next_job_id: int = 0
        self._job_settled.connect(self._on_job_settled)
        self._thread_retired.connect(self._retire_job)
        self.last_error: str = ""

        # -- edit state ----------------------------------------------------
        self._edit_mode: bool = False
        self._edit_path: str = ""
        self._explicit_path: str = ""
        self._suppress_edit_signal: bool = False
        self.last_edit_sql: str = ""
        self.confirm_edit_mode: Callable[[str], bool] = self._default_confirm

        # -- linked selection state ----------------------------------------
        # True while an incoming selection is being written into the view.
        # Echo suppression stops this screen hearing its *own* publications,
        # but not itself re-publishing what it was just told: the round trip
        # would replace the shared selection with the part of it this page
        # happens to have loaded, quietly narrowing a lasso to one chunk.
        self._syncing_selection: bool = False
        #: Model rows the shared filter hides, by row index.
        self._linked_hidden: set = set()
        #: Appended to the status line while the shared filter is narrowing
        #: what this table shows, or explaining why it could not be applied.
        self._linked_filter_note: str = ""

        self._build_ui()
        # Match the pipeline screens: the database file, its measurements/
        # folder, and the enclosing run folder can all be dropped anywhere on
        # this screen.
        from ..dnd import install_dropzone
        from ..dnd_handlers import DatabaseDropHandler
        install_dropzone(self, DatabaseDropHandler(), self)
        self._set_status(
            "Choose a measurements.db, or a run folder containing "
            "measurements/measurements.db.")
        self._update_controls()
        # After the UI: both hooks paint into the view, and a filter can
        # already be set by the time this screen opens.
        self.link_selection("db_browser")
        # Hover help belongs on a setting's NAME, not on the field the user
        # is about to type into (instruction 113). One post-pass rather than
        # a convention every hand-built row has to remember.
        from .settings_model import retarget_field_tooltips
        retarget_field_tooltips(self)

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
            "Read-only by default. spaCR opens the file with mode=ro and "
            "PRAGMA query_only, so nothing here can modify your measurements "
            "until you deliberately turn on edit mode.")
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

        # ── Edit-mode row ─────────────────────────────────────────────
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
        self._model.set_fetch_hook(self.fetch_more)
        self._model.set_commit_hook(self.edit_cell)
        self._view = QTableView(right)
        self._view.setModel(self._model)
        # Read-only in the UI as well as on the connection, until edit
        # mode says otherwise.
        self._view.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._view.setSelectionBehavior(QAbstractItemView.SelectItems)
        self._view.setAlternatingRowColors(True)
        # Sorting stays off while a table is partially loaded — see
        # _update_sort_state().
        self._view.setSortingEnabled(False)
        header = self._view.horizontalHeader()
        # Our own handler, not Qt's model sort: the sort runs in SQL over the
        # whole table, so it is right however little of the table is loaded.
        header.setSectionsClickable(True)
        header.sectionClicked.connect(self._on_header_clicked)
        header.setSectionResizeMode(QHeaderView.Interactive)
        # No stretch-last-section: with feature columns the last one would
        # balloon to fill the window while its neighbours stay clipped.
        header.setStretchLastSection(False)
        header.setDefaultSectionSize(150)
        # Publish whatever the user picks out, and re-hide the filtered rows
        # whenever the row set moves under the view. `setRowHidden` is
        # positional and Qt clears it on a model reset, so both signals are
        # needed: `modelReset` for a new page, a column search or a sort, and
        # `rowsInserted` for the chunks that arrive as the user scrolls.
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

    def set_database(self, path: str, explicit: bool = True) -> bool:
        """Open ``path`` read-only and list its tables.

        Accepts the database file or a run ``src`` folder. Any problem
        (missing file, not a sqlite database, unreadable) is reported in
        the status label and returns ``False`` — this never raises.

        Always resets edit mode to off: a database the user armed for
        editing is *that* database, never the next one.

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
        self._table_list.clear()
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
            db = ReadOnlyDb(path)
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
        """Make ``name`` the previewed table; resets the load and the filter."""
        if self._db is None:
            self._set_status("No database open.", error=True)
            return False
        try:
            self._all_columns = self._db.columns(name)
        except Exception as e:
            self._set_status(self._humanise(e, name), error=True)
            return False
        self._table = name
        # A sort column from the previous table would land in this table's
        # ORDER BY, where check_columns rejects it -- so the table would fail
        # to load rather than merely come back unsorted.
        self._clear_sort()
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

    # -- loading -----------------------------------------------------------

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
        than raised: this is a convenience jump, and a screen that refuses to
        open because a seed was stale is worse than one that opens on the
        wrong table.

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
            # Scroll the column into view rather than sorting by it: arriving
            # on a re-sorted table would hide which rows were just annotated,
            # which is the thing the user came here to look at.
            try:
                section = self.visible_columns().index(str(column))
            except (ValueError, AttributeError):
                return
            self._view.scrollTo(self._model.index(0, section))

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
        self._loaded = 0
        self._last_key = None
        self._exhausted = False
        self._exact_count = None
        self._estimate = None

    def _on_page_size_changed(self, _value: int) -> None:
        self.refresh()

    def refresh(self) -> None:
        """Abandon any load in flight and read the first chunk again."""
        if self._db is None or not self._table:
            return
        self._token += 1
        self._reset_load_state()
        self._model.set_more(False)
        self._model.clear()
        self._view.setSortingEnabled(False)
        header = self._view.horizontalHeader()
        if self._sort is None:
            header.setSortIndicatorShown(False)
            header.setSortIndicator(-1, Qt.AscendingOrder)
        else:
            # `refresh()` is how a header click re-reads the table, so it must
            # not wipe the indicator that click just set.
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
        db, table = self._db, self._table
        where, params = self._where, self._params
        limit = self.page_size()
        after = None if first else self._last_key
        loaded = 0 if first else self._loaded
        order_by = self._sort
        # max(rowid) is only an estimate of the *table* size; with a
        # filter in play it says nothing, so we don't pretend it does.
        want_estimate = first and not where

        def _job() -> Dict[str, Any]:
            cols, rows, keys = db.chunk(table, where=where, params=params,
                                        limit=limit, after=after,
                                        loaded=loaded, order_by=order_by)
            estimate = db.estimate_count(table) if want_estimate else None
            return {"token": token, "first": first, "columns": cols,
                    "rows": rows, "keys": keys, "limit": limit,
                    "estimate": estimate}

        self._model.set_more(False)
        self._run_job(_job, self._apply_chunk, kind="chunk", token=token)

    def _apply_chunk(self, result: Dict[str, Any]) -> None:
        """Paint a chunk — unless the user has moved on since it was asked for."""
        if not result or result.get("token") != self._token:
            return                      # cancelled: a stale load's rows
        first = bool(result.get("first"))
        columns = result.get("columns", [])
        rows = result.get("rows", [])
        keys = result.get("keys", [])
        limit = int(result.get("limit", self.page_size()))
        if first:
            self._model.set_page(columns, rows, row_offset=0, keys=keys)
            self._estimate = result.get("estimate")
        else:
            self._model.append_rows(rows, keys)
        self._loaded += len(rows)
        if keys and keys[-1] is not None:
            self._last_key = keys[-1]
        if len(rows) < limit:
            # A short chunk is the end of the table — and it makes the
            # count exact for free, no COUNT(*) needed.
            self._exhausted = True
            self._exact_count = self._loaded
        self._model.set_more(not self._exhausted)
        if first:
            self._update_column_count()
            self._autosize_columns()
        self._update_rows_label()
        self._update_sort_state()
        self._report_table_status()
        if first and not self._exhausted and self.auto_count:
            self._start_count(self._token)

    def refresh_count(self) -> None:
        """Replace the estimate with an exact ``COUNT(*)``."""
        if self._db is None or not self._table:
            return
        self._start_count(self._token)

    def _start_count(self, token: int) -> None:
        db, table = self._db, self._table
        where, params = self._where, self._params

        def _job() -> Dict[str, Any]:
            return {"token": token, "count": db.count(table, where, params)}

        self._run_job(_job, self._apply_count, kind="count", token=token)

    def _apply_count(self, result: Dict[str, Any]) -> None:
        if not result or result.get("token") != self._token:
            return                      # cancelled
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
        # Never Qt's own: it would reorder the loaded slice underneath the
        # SQL order and the two would disagree.
        self._view.setSortingEnabled(False)
        if self._sort is None:
            self._sort_note.setText(
                "Click a column header to sort the whole table.")
            return
        column, descending = self._sort
        self._sort_note.setText(
            f"sorted by {column} {'descending' if descending else 'ascending'}"
            " — whole table, in SQL. Click again to reverse, a third time to "
            "clear.")

    def _on_header_clicked(self, section: int) -> None:
        """Cycle the clicked column: ascending, descending, unsorted.

        A third state matters here. Sorting forces OFFSET paging, which gets
        slower the further the user scrolls, so there has to be a way back to
        the table's natural keyset-paged order without reloading the screen.
        """
        columns = self._model.visible_columns()
        if not (0 <= section < len(columns)):
            return
        column = columns[section]

        if self._sort is None or self._sort[0] != column:
            self._sort = (column, False)
        elif not self._sort[1]:
            self._sort = (column, True)
        else:
            self._sort = None

        header = self._view.horizontalHeader()
        if self._sort is None:
            header.setSortIndicatorShown(False)
        else:
            header.setSortIndicatorShown(True)
            header.setSortIndicator(
                section, Qt.DescendingOrder if self._sort[1] else Qt.AscendingOrder)

        # Reload from the top: rows already loaded are the wrong ones now,
        # not merely in the wrong order.
        self.refresh()

    def _report_table_status(self) -> None:
        bits = [f"{self._table}: {self._count_text()}",
                f"{len(self._model.all_columns())} columns"]
        if self._filter_label:
            bits.append(f"filter: {self._filter_label}")
        bits.append("edit mode" if self._edit_mode else "read-only")
        # A table quietly showing two thirds of its rows is how a count gets
        # reported as the whole population.
        self._set_status(" · ".join(bits) + self._linked_filter_note)

    # -- the shared filter and selection -----------------------------------

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
            # Skipped entirely while nothing is filtered, and this runs once
            # per chunk: a walk over every loaded row on each of the 4 000
            # chunks of a 400 k-row table is the difference between scrolling
            # and not scrolling.
            for row in range(total):
                self._view.setRowHidden(row, row in hidden)
        changed = note != self._linked_filter_note
        self._linked_filter_note = note
        if changed and self._db is not None and self._table:
            self._report_table_status()

    def on_linked_filter_changed(self, data_filter: DataFilter) -> None:
        self._apply_linked_filter()

    def hidden_rows(self) -> List[int]:
        """Model rows the shared filter is hiding, ascending."""
        return sorted(self._linked_hidden)

    def visible_rows(self) -> List[int]:
        """Model rows the shared filter keeps on screen, ascending."""
        return [row for row in range(self._model.rowCount())
                if row not in self._linked_hidden]

    # -- selection ---------------------------------------------------------

    def rows_for_selection(self, selection: Selection) -> List[int]:
        """The loaded rows ``selection`` names, ascending.

        Empty — not an error — for a table with no object identity in it
        (``png_list`` keyed on a path, a summary, a view). A shared selection
        simply has nothing to say about those rows.
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
            # No object identity in this table; selecting a row in it is a
            # local act, not something the other views can follow.
            return

    def on_linked_selection_changed(self, selection: Selection) -> None:
        """Select and scroll to the rows somebody else picked.

        Nothing is hidden: rows the selection does not name stay exactly
        where they are. Only the shared *filter* removes rows from view.
        """
        model = self._view.selectionModel()
        if model is None or not self._model.columnCount():
            return
        rows = self.rows_for_selection(selection)
        # Guarded, not merely echo-suppressed: this screen would otherwise
        # re-publish what it was just told, replacing a selection of ninety
        # thousand objects with the hundred of them this page has loaded.
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

    # -- filtering ---------------------------------------------------------

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

    # -- edit mode ---------------------------------------------------------

    def editing_allowed_by_preference(self) -> bool:
        """Whether Preferences permits edit mode at all (read fresh)."""
        return bool(get_db_browser_editable())

    def edit_mode_enabled(self) -> bool:
        """True when this screen currently holds a licence to write."""
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
        return ""

    def _confirmation_text(self) -> str:
        """The words the user has to agree to before edit mode arms."""
        table = self._table or "<table>"
        # No table (a database with none) or no key (a view) still gets a
        # statement to look at — the rowid shape, which is what an
        # editable table would use.
        key_columns = self._db.row_key(self._table)[1] if self._table else []
        statement = build_update(table, "<column>", key_columns or ["rowid"])
        return (
            f"Edit mode opens {self._db.path} read-write.\n\n"
            f"Changes are written straight into the file. There is no undo "
            f"and spaCR keeps no backup.\n\n"
            f"Every change runs exactly one statement, scoped to one row:\n"
            f"    {statement}\n\n"
            f"Turn edit mode on?")

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
            f"Edit mode is ON for {self._db.path}. Every change is one "
            f"UPDATE against one row, and there is no undo.")
        self.edit_mode_changed.emit(True)
        return True

    def disable_edit_mode(self, quiet: bool = False) -> None:
        """Drop back to read-only. Safe to call when already read-only."""
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
            # Belt and braces: edit mode is armed for one file only.
            self._set_status(
                "Edit mode was armed for a different database — turning it "
                "off.", error=True)
            self.disable_edit_mode(quiet=True)
            return False
        _kind, key_columns = self._db.row_key(self._table)
        key = self._model.row_key(row)
        if not key_columns or key is None:
            self._set_status(
                f"Cannot edit {self._table!r}: it has no rowid and no primary "
                f"key, so an UPDATE could not be limited to the row you "
                f"clicked. Editing is refused rather than risk rewriting "
                f"many rows.", error=True)
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
        sql = build_update(self._table, column, key_columns)
        self._show_pending_sql(sql, (value,) + tuple(key))
        try:
            WritableDb(self._edit_path).update_cell(
                self._table, column, value, key_columns, key)
        except EditRefused as e:
            self._set_status(f"Edit refused: {e}", error=True)
            return False
        except (sqlite3.Error, OSError, ValueError) as e:
            self._set_status(f"Edit failed: {e}", error=True)
            return False
        self._model.set_value(row, column, value)
        self._set_status(
            f"Updated 1 row of {self._table} — {column} = {value!r}.")
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

        Cheap after the first call — :meth:`ReadOnlyDb.row_key` caches.
        """
        if self._db is None or not self._table:
            return False
        return bool(self._db.row_key(self._table)[1])

    def _update_edit_ui(self) -> None:
        """Keep the checkbox, the note and the view's edit triggers honest."""
        allowed = self.editing_allowed_by_preference()
        self._edit_check.setEnabled(self._db is not None and allowed)
        # A table with no rowid and no primary key stays read-only even in
        # edit mode: offering a cell editor that always refuses would be a
        # lie told twice.
        writable = self._edit_mode and self._table_is_editable()
        self._model.set_editable(writable)
        self._view.setEditTriggers(
            (QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
            if writable else QAbstractItemView.NoEditTriggers)
        if self._edit_mode and not writable:
            self._edit_note.setText(
                f"EDIT MODE is on, but {self._table or 'this table'} has no "
                f"rowid and no primary key — spaCR cannot address one row of "
                f"it, so it stays read-only.")
            return
        if self._edit_mode:
            self._edit_note.setText(
                f"EDIT MODE — writes go straight into "
                f"{os.path.basename(self._edit_path)}. One UPDATE per cell, "
                f"scoped to one row, no undo.")
            return
        if not allowed:
            self._edit_note.setText(
                "Read-only. Editing is off in Preferences → Database Browser.")
        elif self._db is None:
            self._edit_note.setText(
                "Read-only. Open a database you choose yourself to edit it.")
        elif self._explicit_path != self._db.path:
            self._edit_note.setText(
                "Read-only — this database was opened for you. Re-open it "
                "with 'Choose database…' to edit it.")
        else:
            self._edit_note.setText(
                "Read-only. Tick 'Edit mode' to write to this file; you will "
                "be asked to confirm.")

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
        if self._export_busy:
            self._set_status("An export is already running…", error=True)
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

        return self._run_job(_job, _done, kind="export")

    # -- job plumbing ------------------------------------------------------

    def _acquire(self, kind: str) -> None:
        if kind == "export":
            self._export_busy = True
            return
        self._load_jobs += 1
        if kind == "chunk":
            self._chunk_jobs += 1

    def _release(self, kind: str) -> None:
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

        ``fn`` runs on a worker thread and must therefore open its own
        sqlite connection — every :class:`ReadOnlyDb` method does.

        **One worker at a time.** Jobs queue here and :meth:`_pump` starts
        them one by one. Chunked loading naturally wants to overlap (an
        abandoned chunk, its replacement, and a ``COUNT(*)`` can all be
        outstanding at once) but ``PipelineWorker.run`` swaps *global*
        process state around the call it makes — ``sys.stdout``,
        ``sys.stderr`` and ``matplotlib.pyplot.show`` are saved on entry
        and restored on exit. Two of those running at once interleave the
        swaps and restore each other's shims, and the loser is left
        permanently pointing at a dead redirector: every ``print`` in the
        process after that goes nowhere. (Measured, not theorised — a
        bare overlapping ``make_thread`` loop loses its own stdout.)
        Serialising costs nothing here: these jobs are I/O on one sqlite
        file, which does not go faster in parallel.

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
        if self._thread is not None:
            return
        while self._queue:
            fn, on_done, kind, token = self._queue.pop(0)
            if token is not None and token != self._token:
                self._release(kind)
                continue
            self._start_job(fn, on_done, kind)
            return
        self._update_controls()

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
        thread, worker = make_thread(partial(_capture_result, fn), box)
        # make_thread deliberately does not connect worker.deleteLater:
        # Python owns this worker, and _retire_job releases its last strong
        # reference on the GUI thread after the event loop exits. Do not try
        # to "defensively" disconnect a slot that is absent — PySide emits a
        # RuntimeWarning for every job, and signal mutation during native
        # teardown is precisely the lifecycle race this ownership scheme
        # avoids. See make_thread's ownership contract.
        # Strong references: PySide6 will not keep the worker alive through
        # the started→run connection alone, and a collected worker means the
        # thread spins forever without ever calling run(). A QThread that
        # loses its last Python reference while running takes the process
        # down with it, so the pair is held until `thread.finished` says
        # the event loop has exited. Same fix as AppScreen._on_run — but
        # held per-job, keyed by job id.
        self._next_job_id += 1
        job_id = self._next_job_id
        self._jobs[job_id] = (thread, worker)
        self._thread, self._worker = thread, worker
        self._pending[job_id] = (box, on_done, kind)
        worker.error.connect(self._on_worker_error_text)
        worker.finished.connect(
            lambda ok, jid=job_id: self._job_settled.emit(jid, bool(ok)))
        # A BOUND METHOD, not a closure — and the contrast with the line
        # above is the whole point. ``worker`` is moveToThread'd, so a
        # closure on ITS signal runs on the worker thread and re-emitting a
        # Signal is the only safe thing to do from one. ``thread`` is the
        # opposite case: the QThread object is GUI-affine, so PySide6 makes
        # it the receiver for a closure, and ``make_thread`` connects
        # ``thread.finished -> thread.deleteLater`` FIRST. Slots run in
        # connection order, so the DeferredDelete is posted ahead of the
        # closure's metacall and Qt discards queued events for a destroyed
        # receiver: the job was never retired and ``active_jobs()`` never
        # returned to zero.
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
        self._set_status(f"Query failed: {exc}", error=True)

    def is_busy(self) -> bool:
        """True while any query, count or export job is queued or running."""
        return self._export_busy or self._load_jobs > 0

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
            w.setEnabled(has_table and not self._export_busy)
        self._btn_more.setEnabled(
            has_table and not self._exhausted and not self._chunk_jobs)
        # The table list stays live during a load on purpose: switching
        # table mid-load has to be possible, and the token check makes it
        # safe.
        self._table_list.setEnabled(has_db)
        self._update_edit_ui()

    # -- shutdown ----------------------------------------------------------

    def closeEvent(self, event):  # noqa: N802
        """Let every in-flight query thread finish before the widget dies.

        Destroying a QThread that is still running aborts the process, so
        we wait (briefly) rather than hope. Jobs that have not started yet
        are simply dropped — nothing has been spawned for them.

        The shared link outlives this screen, so let go of it too.
        """
        try:
            self.unlink_selection()
        except (RuntimeError, TypeError):
            # The process-wide link's C++ side is gone (interpreter teardown).
            pass
        for _fn, _on_done, kind, _token in self._queue:
            self._release(kind)
        self._queue.clear()
        for thread, _worker in list(self._jobs.values()):
            try:
                if thread.isRunning():
                    thread.quit()
                    thread.wait(5000)
            except RuntimeError:
                pass
        super().closeEvent(event)

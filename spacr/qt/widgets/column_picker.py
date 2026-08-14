"""SQL column picker — the "SQL" button that sits beside every column field.

Anywhere spaCR asks for the *name of a database column* — the annotation
column, the measurement used to prefilter crops, the heatmap feature, the
regression's dependent variable — the field has historically been a bare
text box. Typing into it is blind: nothing tells you what the database
already contains, so ``annotate`` and ``annotaet`` are equally acceptable
and equally silent. The second one starts a brand new annotation pass, and
two passes that should have been one then look like two annotators who
agree on nothing.

This module adds the missing half of that interaction:

``ColumnPickerButton``
    a small ``SQL`` button that can be dropped beside an existing field.

``ColumnPickerDialog``
    the panel it opens — the tables in the database, the columns of the
    selected table with their declared type, and a name box that says, in
    words, what will happen to the name you typed: *used*, *created*, or
    *refused*.

``attach_column_picker``
    a one-liner that wires the two onto a ``QLineEdit``/``QComboBox`` a
    screen has already laid out. It re-uses the field's own slot in its
    parent layout, so a host adopts the picker without restructuring
    anything.

Three properties this file is built around:

* **Read-only.** Opening the picker reads column names and nothing else.
  The connection is ``file:…?mode=ro`` plus ``PRAGMA query_only = ON`` —
  the same pair :mod:`spacr.qt.screens.db_browser` and
  :mod:`spacr.agreement` use — so SQLite itself refuses a write. The
  dialog never creates a column: it returns a *name*, and the write path
  that owns creation (``annotate_engine.ensure_annotation_column``, which
  does it lazily with ``ALTER TABLE``) stays where it is.

* **Cheap on open.** ``PRAGMA table_info`` is free; ``SELECT COUNT(*)``
  over a 400 k-row measurement table is not. Opening the dialog runs no
  count at all. The per-table row figure comes from ``max(_rowid_)`` and
  is labelled an estimate; a per-column non-null count happens only when
  the user asks for it by name. (``_rowid_``, not ``rowid`` — see
  :meth:`SchemaReader.estimate_rows`, where the difference was a full
  table scan followed by a ``ValueError``.)

* **Off the GUI thread.** Cheap is not free, and opening the picker is
  four sequential sqlite round trips — open, list tables, read one
  table's columns, estimate its rows — every one of which used to happen
  inside ``__init__``, before the modal appeared. Measured cold on a
  383 MB measurements.db that is 45 ms, and on a 1 500-table schema
  87 ms, entirely between the click and any window. The button now
  builds the dialog with ``threaded=True`` and the reads arrive from a
  :class:`~spacr.qt.job_runner.JobRunner`. The default is still the
  synchronous mode, deliberately; :class:`ColumnPickerDialog` says why.

* **No modal errors.** A missing database, a file that isn't SQLite, a
  database with no tables, a name SQLite would refuse — every one of them
  is reported in a banner inside the dialog. The dialog itself is the one
  deliberate modal, and it is injectable (:meth:`ColumnPickerButton.
  set_dialog_runner`) so a headless test drives it without ever entering
  an event loop.
"""
from __future__ import annotations

import difflib
import os
import re
import sqlite3
import threading
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from urllib.parse import quote as _urlquote

from PySide6.QtCore import QItemSelectionModel, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLayout,
    QLineEdit,
    QListWidget,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)
from .toggle import Toggle

__all__ = [
    "ColumnPickerButton",
    "ColumnPickerDialog",
    "SchemaReader",
    "attach_column_picker",
    "chip_editor",
    "field_is_list",
    "field_text",
    "field_values",
    "near_miss",
    "read_schema",
    "read_table",
    "resolve_db_path",
    "set_field_text",
    "set_field_values",
    "validate_column_name",
]

DB_FILENAME = "measurements.db"
_MEASUREMENTS_SUBDIR = "measurements"

#: SQLite's three spellings of the implicit row id, least likely to be
#: shadowed first. See :meth:`SchemaReader.estimate_rows` for why the
#: order matters and what the bare ``rowid`` cost.
ROWID_ALIASES: Tuple[str, ...] = ("_rowid_", "oid", "rowid")

#: Levenshtein-ish cutoff for "this is probably a typo". 0.6 is the value
#: :mod:`spacr.cli` already uses for the same job on setting names, and the
#: two should agree — a user who has seen "Did you mean 'nucleus_area'?" on
#: the command line should get the same judgement in the GUI.
NEAR_MISS_CUTOFF = 0.6


# ---------------------------------------------------------------------------
# Path resolution + read-only schema access
# ---------------------------------------------------------------------------

def resolve_db_path(path: str) -> str:
    """Turn whatever a host screen holds into a path to a database file.

    Hosts variously hold ``src`` (a run folder), ``src/measurements`` or the
    database itself, so all three resolve here rather than in each caller.

    :param path: file, run folder, or measurements folder.
    :returns: absolute path to a ``.db`` file (existence not guaranteed —
        a file path is returned as given so the caller can report it).
    :raises ValueError: when ``path`` is empty.
    """
    text = "" if path is None else str(path).strip()
    if not text:
        raise ValueError("No database selected.")
    p = os.path.abspath(os.path.expanduser(text))
    if os.path.isdir(p):
        for candidate in (
            os.path.join(p, _MEASUREMENTS_SUBDIR, DB_FILENAME),
            os.path.join(p, DB_FILENAME),
        ):
            if os.path.isfile(candidate):
                return candidate
        return os.path.join(p, _MEASUREMENTS_SUBDIR, DB_FILENAME)
    return p


def _read_only_uri(path: str) -> str:
    """Return the ``file:…?mode=ro`` URI SQLite needs for a read-only open."""
    return "file:" + _urlquote(str(path).replace("\\", "/"), safe="/:") + "?mode=ro"


class SchemaReader:
    """Read-only access to a database's *schema* — names and types only.

    Every method opens and closes its own connection, which keeps the
    object cheap to hold and impossible to leave a transaction open on.
    ``mode=ro`` makes SQLite refuse writes; ``PRAGMA query_only`` refuses
    them a second time, including schema changes smuggled in through a
    temp attachment.

    Every method may be called from a worker thread — the dialog reads
    its schema off the GUI thread — and each one opens its own
    connection, so nothing is shared across threads except
    :attr:`executed`, which is guarded.

    :param path: database file or run folder (see :func:`resolve_db_path`).
    :ivar executed: every statement this reader has run, in order — the
        hook a test uses to prove that opening the picker costs no
        ``COUNT(*)``.
    """

    def __init__(self, path: str):
        self.path = resolve_db_path(path)
        self.uri = _read_only_uri(self.path)
        self.executed: List[str] = []
        # `executed` is appended to on whichever thread runs the query and
        # read from the GUI thread by `ColumnPickerDialog.executed_sql`.
        # list.append is atomic under the GIL, but the *snapshot* the
        # assertion hook hands out must not be taken mid-append, or a test
        # that asserts "opening cost no COUNT(*)" could be reading a list
        # that is one element ahead of the statement it is about.
        self._log_lock = threading.Lock()

    # -- plumbing ----------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.uri, uri=True, timeout=30)
        con.execute("PRAGMA query_only = ON")
        return con

    def _fetch(self, sql: str, params: Sequence = ()) -> List[tuple]:
        with self._log_lock:
            self.executed.append(sql)
        con = self._connect()
        try:
            return list(con.execute(sql, tuple(params)).fetchall())
        finally:
            con.close()

    def executed_sql(self) -> List[str]:
        """Return a consistent snapshot of :attr:`executed`."""
        with self._log_lock:
            return list(self.executed)

    # -- schema ------------------------------------------------------------

    def probe(self) -> None:
        """Open the file once so "that isn't a database" surfaces early.

        :raises sqlite3.Error: when the file is not a SQLite database or
            cannot be opened.
        """
        self._fetch("SELECT name FROM sqlite_master LIMIT 1")

    def tables(self) -> List[str]:
        """Return the user tables and views, alphabetically."""
        rows = self._fetch(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view') "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name")
        return [r[0] for r in rows]

    def column_info(self, table: str) -> List[Tuple[str, str]]:
        """Return ``[(column name, declared type), …]`` for ``table``.

        ``PRAGMA table_info`` reads the stored schema text; it never
        touches a data page, so this stays free on a huge table.

        :raises sqlite3.Error: for a view whose base table has been
            dropped — SQLite resolves the view here and says so.
        """
        rows = self._fetch(f"PRAGMA table_info({quote_ident(table)})")
        return [(str(r[1]), str(r[2] or "")) for r in rows]

    def estimate_rows(self, table: str) -> Optional[int]:
        """Return an O(1) *estimate* of the row count, or ``None``.

        ``max(_rowid_)`` is answered from the right-hand edge of the
        b-tree without a scan. Deleted rows leave gaps, so it is an
        estimate and every caller must label it one. ``None`` for a view,
        a WITHOUT ROWID table, or an empty table.

        **It is spelt ``_rowid_``, and that is load-bearing.** Every
        spaCR object table declares a column called ``rowID`` — the plate
        row, ``'r1'``, ``'r2'`` — and SQLite identifiers are
        case-insensitive, so a bare ``rowid`` resolves to *that column*
        rather than to the row id. The version this replaces asked for
        ``max(rowid)`` and got two things wrong at once on every real
        measurements database: it scanned the whole table to take the
        maximum of a text column (measured: 21 ms warm, 41 ms cold, on a
        200 000-row table — precisely the ``COUNT(*)`` cost this method
        exists to avoid), and then ``int('r16')`` raised ``ValueError``,
        which is not a ``sqlite3.Error`` and so escaped the caller's
        handler into the Qt event loop. :mod:`spacr.predictions`,
        :mod:`spacr.foreign` and :mod:`spacr.data_manager` all carry a
        comment about this shadowing; this method had not got the memo.

        The remaining spellings are tried in turn for the pathological
        table that declares ``_rowid_`` as well, and a value that will not
        convert is treated as a shadowed column rather than as an answer.
        """
        for alias in ROWID_ALIASES:
            try:
                rows = self._fetch(
                    f"SELECT max({alias}) FROM {quote_ident(table)}")
            except sqlite3.Error:
                continue
            if not rows or rows[0][0] is None:
                return None
            try:
                return int(rows[0][0])
            except (TypeError, ValueError):
                continue
        return None

    def count_non_null(self, table: str, column: str) -> int:
        """Return how many rows have a value in ``column``.

        This one is a real scan — it exists only behind an explicit
        button, never on the path that opens the dialog.
        """
        sql = (f"SELECT COUNT({quote_ident(column)}) "
               f"FROM {quote_ident(table)}")
        rows = self._fetch(sql)
        return int(rows[0][0]) if rows else 0


def quote_ident(name: str) -> str:
    """Double-quote a SQL identifier, escaping embedded quotes.

    Only ever called with a name taken from the live schema; the quoting
    keeps legal-but-awkward names (``cell_channel_1 (raw)``) working.
    """
    return '"' + str(name).replace('"', '""') + '"'


def open_reader(path: Any) -> Tuple[Optional[SchemaReader], str]:
    """Return ``(reader, message)`` — exactly one of the two is set.

    Turning every failure into a sentence here is what lets the dialog
    render problems inline instead of raising into a modal.
    """
    try:
        resolved = resolve_db_path(path)
    except ValueError:
        return None, ("No database selected — point this module at a run "
                      "folder (its 'src' setting) or at a measurements.db.")
    if not os.path.isfile(resolved):
        return None, (f"No database at {resolved} — run Measure first, or "
                      f"pick a folder that already has one.")
    reader = SchemaReader(resolved)
    try:
        reader.probe()
    except sqlite3.OperationalError as exc:
        # "unable to open database file" — a permission problem, a stale
        # network mount. Saying "not a database" here would send the user
        # looking for the wrong fault.
        return None, f"Cannot open {os.path.basename(resolved)}: {exc}"
    except sqlite3.DatabaseError as exc:
        return None, f"{os.path.basename(resolved)} is not a SQLite database ({exc})."
    return reader, ""


# ---------------------------------------------------------------------------
# Name validation
# ---------------------------------------------------------------------------

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

#: SQLite's reserved words. A column called ``index`` or ``group`` is legal
#: *if* every statement that ever touches it remembers to quote it, and the
#: moment one doesn't — a pandas ``read_sql`` f-string, a downstream script —
#: it fails somewhere far away from here. Refusing the name up front is the
#: kinder failure.
SQLITE_KEYWORDS = frozenset("""
abort action add after all alter always analyze and as asc attach autoincrement
before begin between by cascade case cast check collate column commit conflict
constraint create cross current current_date current_time current_timestamp
database default deferrable deferred delete desc detach distinct do drop each
else end escape except exclude exclusive exists explain fail filter first
following for foreign from full generated glob group groups having if ignore
immediate in index indexed initially inner insert instead intersect into is
isnull join key last left like limit match materialized natural no not nothing
notnull null nulls of offset on or order others outer over partition preceding
primary query raise range recursive references regexp reindex release rename
replace restrict returning right rollback row rows savepoint select set table
temp temporary then ties to transaction trigger unbounded union unique update
using vacuum values view virtual when where window with without
""".split())


def validate_column_name(name: str) -> str:
    """Return why ``name`` cannot be a new column, or ``""`` if it can.

    The point is to fail here, with a sentence, rather than three screens
    later inside an ``ALTER TABLE`` the user never sees.

    :param name: candidate column name as typed.
    :returns: an explanation, or the empty string when the name is fine.
    """
    text = "" if name is None else str(name)
    if not text.strip():
        return "Type a column name, or pick one from the list above."
    if text != text.strip():
        return ("The name has leading or trailing spaces — SQLite would keep "
                "them, and every later lookup would have to guess. Trim them.")
    if any(ch.isspace() for ch in text):
        return (f"'{text}' contains a space. A column name with a space has to "
                f"be quoted in every statement that touches it — use "
                f"underscores instead, e.g. '{text.replace(' ', '_')}'.")
    if len(text) > 128:
        return (f"'{text[:24]}…' is {len(text)} characters. Keep a column name "
                f"under 128 so it stays readable in every table header.")
    if text[0].isdigit():
        return (f"'{text}' starts with a digit, which SQLite only accepts "
                f"quoted. Start with a letter or an underscore.")
    if not _IDENT_RE.match(text):
        bad = sorted({ch for ch in text if not (ch.isalnum() or ch == "_")})
        shown = " ".join(repr(ch) for ch in bad)
        return (f"'{text}' contains {shown}, which SQLite only accepts inside "
                f"quotes. Use letters, digits and underscores only.")
    if text.lower().startswith("sqlite_"):
        return (f"'{text}' uses the 'sqlite_' prefix, which SQLite reserves "
                f"for its own objects. Pick another name.")
    if text.lower() in SQLITE_KEYWORDS:
        return (f"'{text}' is a reserved SQLite keyword, so it would need "
                f"quoting everywhere it is used. Pick another name — "
                f"'{text}_id' or '{text}_value' both work.")
    return ""


def find_existing(name: str, columns: Sequence[str]) -> str:
    """Return the column matching ``name``, or ``""``.

    Case-insensitively: SQLite resolves identifiers without regard to
    case, so ``Annotate`` *is* ``annotate`` and adding it would fail with
    "duplicate column name" rather than create a second column.
    """
    target = str(name or "").strip().lower()
    if not target:
        return ""
    for col in columns:
        if str(col).lower() == target:
            return str(col)
    return ""


def near_miss(name: str, columns: Sequence[str],
              cutoff: float = NEAR_MISS_CUTOFF) -> str:
    """Return the existing column ``name`` most resembles, or ``""``.

    A new name one keystroke away from an existing one is almost never a
    new column; it is the old one, misspelt. Uses
    :func:`difflib.get_close_matches` with the same cutoff
    :mod:`spacr.cli` applies to mistyped setting names.

    :param name: the candidate new column.
    :param columns: the columns the table already has.
    :returns: the closest existing column, or ``""`` when ``name`` is
        either already a column or genuinely unlike all of them.
    """
    text = str(name or "").strip()
    pool = [str(c) for c in columns]
    if not text or not pool or find_existing(text, pool):
        return ""
    matches = difflib.get_close_matches(text, pool, n=1, cutoff=cutoff)
    if matches:
        return matches[0]
    # get_close_matches compares whole strings, so a long shared prefix with
    # a short tail ('annotate' vs 'annotate_pass_two_of_the_second_batch')
    # scores below the cutoff. Those are near-misses too.
    lower = text.lower()
    for col in pool:
        c = col.lower()
        if len(lower) >= 4 and (c.startswith(lower) or lower.startswith(c)):
            return col
    return ""


# ---------------------------------------------------------------------------
# Worker-safe reads — everything opening the dialog costs, off the GUI thread
# ---------------------------------------------------------------------------

def read_table(reader: Optional[SchemaReader],
               table: str) -> Dict[str, Any]:
    """Read one table's columns and row estimate. Touches no widget.

    Returns plain data so the same result can come back from a worker
    thread or from an inline call, and the painting code does not have to
    know which it was.

    :returns: ``{"table", "columns", "rows", "error"}``. ``error`` is the
        sentence to put in the banner; an empty ``columns`` list with no
        ``error`` means the table honestly has none.
    """
    payload: Dict[str, Any] = {"table": str(table or ""), "columns": [],
                               "rows": None, "error": ""}
    if reader is None or not payload["table"]:
        return payload
    try:
        payload["columns"] = reader.column_info(payload["table"])
    except sqlite3.Error as exc:
        payload["error"] = (
            f"Cannot list the columns of '{payload['table']}': {exc}. A view "
            f"whose table has been dropped does this — pick another table.")
        return payload
    if payload["columns"]:
        payload["rows"] = reader.estimate_rows(payload["table"])
    return payload


def read_schema(db_path: Any, reader: Optional[SchemaReader] = None,
                preferred: str = "") -> Dict[str, Any]:
    """Everything opening the picker needs, in one worker-thread call.

    Opens the database, lists its tables, picks the one to show and reads
    that table — the four synchronous sqlite round trips that used to sit
    between the click on ``SQL`` and the dialog appearing.

    :param db_path: file, run folder, or measurements folder.
    :param reader: use this reader instead of opening ``db_path``.
    :param preferred: table to select if the database has it.
    :returns: ``{"reader", "error", "tables", "schema_error", "table"}``
        plus the keys :func:`read_table` returns for the chosen table.
    """
    error = ""
    if reader is None:
        reader, error = open_reader(db_path)
    payload: Dict[str, Any] = {
        "reader": reader, "error": error, "tables": [], "schema_error": "",
        "table": "", "columns": [], "rows": None,
    }
    if reader is None:
        return payload
    try:
        payload["tables"] = reader.tables()
    except sqlite3.Error as exc:
        payload["schema_error"] = f"Cannot read the schema: {exc}"
        return payload
    if not payload["tables"]:
        return payload
    names = payload["tables"]
    payload.update(read_table(
        reader, preferred if preferred in names else names[0]))
    return payload


# ---------------------------------------------------------------------------
# The dialog
# ---------------------------------------------------------------------------

#: Outcomes of the name box, in the order the dialog checks them.
ACTION_USE = "use"            # the name is an existing column
ACTION_CREATE = "create"      # the name is new and safe
ACTION_CONFIRM = "confirm"    # the name is new but looks like a typo
ACTION_INVALID = "invalid"    # SQLite would refuse the name
ACTION_UNCHECKED = "unchecked"  # no database to check against


class ColumnPickerDialog(QDialog):
    """Browse a database's columns and settle on one name.

    The dialog answers one question — "what should this field say?" — and
    is explicit about the consequence of the answer: an existing column
    will be *used*, a new one will be *created* by whoever owns the write
    path. It creates nothing itself.

    **Two modes, and the default is the synchronous one.**

    ``threaded=False`` (the default) reads the schema inside
    ``__init__``: when the constructor returns, the tables are listed, a
    table is selected, its columns are in the tree and the name box has
    been judged. That is what every programmatic caller and the ~60 tests
    in ``tests/qt/test_column_picker.py`` are written against — they
    construct a dialog and assert on its contents on the next line, with
    no event loop anywhere (the suite's autouse fixture makes
    ``QDialog.exec`` raise). Defaulting to the asynchronous mode would
    turn every one of those into a race, so it is opt-in.

    ``threaded=True`` is what the ``SQL`` button uses
    (:meth:`ColumnPickerButton.make_dialog`), and it is the real
    user-facing path. The window appears immediately saying it is reading
    the schema, and the tables, columns and row estimate arrive from a
    :class:`~spacr.qt.job_runner.JobRunner`. That matters because opening
    the picker is four sequential sqlite round trips against a file that
    is usually cold and often on a network mount — measured at 45 ms on a
    383 MB measurements.db with nothing cached and 87 ms on a
    1 500-table schema, all of it dead time between the click and the
    window.

    Both modes run the *same* reads in the same order through the same
    :func:`read_schema`; the runner is simply constructed unthreaded in
    the first, which makes it call its job inline.

    :param db_path: database file or run folder; may be empty.
    :param table: table to preselect (e.g. ``png_list`` for annotations).
    :param current: the field's current value, prefilled into the name box.
    :param allow_new: when False, only existing columns are accepted.
    :param reader: an already-built :class:`SchemaReader` (or a stand-in);
        injecting one is how tests exercise schema edge cases. Its reads
        are still threaded when ``threaded`` is set.
    :param threaded: read the schema on a worker thread. See above.
    :param multi: let the user select several columns at once and return
        every one of them (:meth:`chosen_columns`). For fields that hold a
        *list* of columns — ``exclude``, ``annotation_columns`` — where one
        name per press meant reopening the dialog, and re-reading the
        schema, once per column the user wanted.
    """

    def __init__(self, db_path: Any = "", table: Optional[str] = None,
                 current: str = "", parent: Optional[QWidget] = None,
                 allow_new: bool = True,
                 reader: Optional[SchemaReader] = None,
                 threaded: bool = False,
                 multi: bool = False):
        super().__init__(parent)
        from ..job_runner import JobRunner

        self._multi = bool(multi)
        self.setWindowTitle("Pick database columns" if self._multi
                            else "Pick a database column")
        self.setObjectName("ColumnPickerDialog")
        self.setMinimumWidth(560)

        self._allow_new = bool(allow_new)
        self._preferred_table = str(table or "")
        self._columns: List[Tuple[str, str]] = []
        self._action = ACTION_UNCHECKED
        self._near = ""
        self._threaded = bool(threaded)
        self._reader: Optional[SchemaReader] = None
        self._open_error = ""
        self._jobs = JobRunner(self, threaded=self._threaded,
                               app_key="column picker")

        self._build_ui()
        # Unthreaded, `submit` calls its job inline and `_apply_schema`
        # has run by the time this returns — which is the whole of the
        # default mode's contract.
        self._jobs.submit(
            lambda p=db_path, r=reader, t=self._preferred_table:
                read_schema(p, r, t),
            self._apply_schema)
        self._name.setText(str(current or ""))
        self._evaluate()

    # -- construction ------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setSpacing(8)

        self._banner = QLabel(self)
        self._banner.setObjectName("ColumnPickerBanner")
        self._banner.setWordWrap(True)
        self._banner.setVisible(False)
        outer.addWidget(self._banner)

        self._source = QLabel(self)
        self._source.setObjectName("ColumnPickerSource")
        self._source.setWordWrap(True)
        # Threaded, nothing has been opened yet and saying "No database
        # open." would be a lie the user reads for the whole of the load.
        self._source.setText("Reading the schema…" if self._threaded
                             else "No database open.")
        outer.addWidget(self._source)

        body = QHBoxLayout()
        body.setSpacing(8)

        left = QVBoxLayout()
        left.setSpacing(4)
        left.addWidget(QLabel("Tables", self))
        self._tables = QListWidget(self)
        self._tables.setSelectionMode(QAbstractItemView.SingleSelection)
        self._tables.currentTextChanged.connect(self._on_table_changed)
        self._tables.setMinimumWidth(160)
        left.addWidget(self._tables, 1)
        body.addLayout(left, 0)

        right = QVBoxLayout()
        right.setSpacing(4)
        self._columns_label = QLabel("Columns", self)
        right.addWidget(self._columns_label)
        self._filter = QLineEdit(self)
        self._filter.setPlaceholderText("Filter columns…")
        self._filter.setClearButtonEnabled(True)
        self._filter.textChanged.connect(self._apply_filter)
        right.addWidget(self._filter)
        self._column_tree = QTreeWidget(self)
        self._column_tree.setColumnCount(3)
        self._column_tree.setHeaderLabels(["Column", "Type", "Non-null"])
        self._column_tree.setRootIsDecorated(False)
        self._column_tree.setUniformRowHeights(True)
        self._column_tree.setSelectionMode(
            QAbstractItemView.ExtendedSelection if self._multi
            else QAbstractItemView.SingleSelection)
        self._column_tree.currentItemChanged.connect(self._on_column_changed)
        self._column_tree.itemDoubleClicked.connect(self._on_column_activated)
        if self._multi:
            # The selection, not the name box, is the answer in this mode,
            # so the verdict has to follow it.
            self._column_tree.itemSelectionChanged.connect(self._evaluate)
        self._columns_label.setText(self._columns_heading(""))
        header = self._column_tree.header()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        right.addWidget(self._column_tree, 1)

        summary_row = QHBoxLayout()
        summary_row.setSpacing(8)
        self._summary = QLabel("", self)
        self._summary.setObjectName("ColumnPickerSummary")
        summary_row.addWidget(self._summary, 1)
        self._count_btn = QPushButton("Count non-null", self)
        self._count_btn.setToolTip(
            "Count the rows that actually have a value in the selected "
            "column. This one reads the whole table, so it is not run "
            "when the dialog opens.")
        self._count_btn.setEnabled(False)
        self._count_btn.clicked.connect(self._count_selected)
        summary_row.addWidget(self._count_btn, 0)
        right.addLayout(summary_row)

        body.addLayout(right, 1)
        outer.addLayout(body, 1)

        form = QFormLayout()
        self._name = QLineEdit(self)
        self._name.setPlaceholderText("column name")
        self._name.textChanged.connect(self._evaluate)
        form.addRow("Column", self._name)
        outer.addLayout(form)

        self._status = QLabel("", self)
        self._status.setObjectName("ColumnPickerStatus")
        self._status.setWordWrap(True)
        outer.addWidget(self._status)

        self._confirm = Toggle(
            "Create it anyway — the new name is deliberate", self)
        self._confirm.setVisible(False)
        self._confirm.toggled.connect(lambda _on: self._evaluate())
        outer.addWidget(self._confirm)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        outer.addWidget(self._buttons)

    # -- loading -----------------------------------------------------------

    def _set_banner(self, text: str) -> None:
        self._banner.setText(text or "")
        self._banner.setVisible(bool(text))

    def _apply_schema(self, payload) -> None:
        """Install a :func:`read_schema` result. GUI thread only.

        The one place the tables list is populated, in either mode.
        """
        payload = payload or {}
        self._reader = payload.get("reader")
        self._open_error = payload.get("error") or ""
        self._source.setText(
            f"Reading {self._reader.path} (read-only)" if self._reader
            else "No database open.")
        if self._reader is None:
            self._set_banner(self._open_error)
            self._evaluate()
            return
        if payload.get("schema_error"):
            self._set_banner(payload["schema_error"])
            self._evaluate()
            return
        names = payload.get("tables") or []
        if not names:
            self._set_banner(
                f"{os.path.basename(self._reader.path)} has no tables yet — "
                f"nothing has been written to it. Run Measure first.")
            self._evaluate()
            return
        self._tables.addItems(names)
        wanted = payload.get("table") or names[0]
        # The columns for `wanted` are already in `payload` — the worker
        # read them in the same trip. Letting `setCurrentRow` fire
        # `currentTextChanged` would send `_on_table_changed` off to read
        # them a second time, and in threaded mode that is a second job
        # racing the one that just landed. Select it quietly and paint
        # from what we were handed.
        blocked = self._tables.blockSignals(True)
        try:
            self._tables.setCurrentRow(names.index(wanted))
        finally:
            self._tables.blockSignals(blocked)
        self._paint_table(payload)
        self._evaluate()

    def _on_table_changed(self, name: str) -> None:
        """The user picked another table. Read it the way we were told to."""
        if not self._threaded:
            self._load_columns(name)
            self._evaluate()
            return
        # Supersede: clicking down the table list must not leave the tree
        # showing whichever read happened to finish last.
        self._jobs.cancel()
        self._begin_table(name)
        self._jobs.submit(
            lambda r=self._reader, t=name: read_table(r, t),
            self._apply_table)

    def _columns_heading(self, table: str) -> str:
        """Heading over the column tree.

        Multi-select is invisible otherwise: a tree looks exactly the same
        whether it takes one row or twenty, so the heading has to say which.
        """
        base = f"Columns in {table}" if table else "Columns"
        if not self._multi:
            return base
        return f"{base} — pick as many as you need (ctrl/shift-click)"

    def _begin_table(self, table: str) -> None:
        """Say a table is being read, without claiming to know anything."""
        self._column_tree.clear()
        self._columns = []
        self._count_btn.setEnabled(False)
        if table and self._reader is not None:
            self._columns_label.setText(self._columns_heading(table))
            self._summary.setText("reading…")
        else:
            self._summary.setText("")

    def _apply_table(self, payload) -> None:
        """Install a :func:`read_table` result. GUI thread only."""
        self._paint_table(payload)
        self._evaluate()

    def _load_columns(self, table: str) -> None:
        """Read and paint one table's columns, blocking. GUI thread.

        The unthreaded path, and the one a host may call directly.
        """
        self._paint_table(read_table(self._reader, table))

    def _paint_table(self, payload) -> None:
        """Put a :func:`read_table` result on screen. GUI thread only."""
        payload = payload or {}
        table = payload.get("table") or ""
        self._column_tree.clear()
        self._columns = []
        self._summary.setText("")
        self._count_btn.setEnabled(False)
        if not table or self._reader is None:
            return
        self._columns_label.setText(self._columns_heading(table))
        if payload.get("error"):
            self._set_banner(payload["error"])
            return
        info = payload.get("columns") or []
        if not info:
            self._set_banner(
                f"'{table}' reports no columns, so there is nothing to pick "
                f"from it. Pick another table.")
            return
        self._set_banner("")
        self._columns = info
        for col_name, decl in info:
            QTreeWidgetItem(self._column_tree, [col_name, decl or "—", ""])
        rows = payload.get("rows")
        shown = f"≈ {rows:,} rows (estimate)" if rows is not None else "row count unknown"
        self._summary.setText(f"{len(info)} columns · {shown}")
        self._apply_filter(self._filter.text())

    def _apply_filter(self, text: str) -> None:
        needle = str(text or "").strip().lower()
        for i in range(self._column_tree.topLevelItemCount()):
            item = self._column_tree.topLevelItem(i)
            item.setHidden(bool(needle) and needle not in item.text(0).lower())

    # -- interaction -------------------------------------------------------

    def _on_column_changed(self, current, _previous=None) -> None:
        self._count_btn.setEnabled(current is not None)
        if current is not None:
            self._name.setText(current.text(0))

    def _on_column_activated(self, item, _column: int = 0) -> None:
        if item is not None:
            self._name.setText(item.text(0))
            if self._buttons.button(QDialogButtonBox.Ok).isEnabled():
                self.accept()

    def _count_selected(self) -> None:
        item = self._column_tree.currentItem()
        table = self.chosen_table()
        if item is None or self._reader is None or not table:
            return
        try:
            n = self._reader.count_non_null(table, item.text(0))
        except sqlite3.Error as exc:
            self._set_banner(f"Could not count '{item.text(0)}': {exc}")
            return
        item.setText(2, f"{n:,}")

    # -- the verdict -------------------------------------------------------

    def _evaluate(self, *_args) -> None:
        """Recompute what the typed name means and say so, in words."""
        name = self._name.text()
        table = self.chosen_table() or "the table"
        existing = [c for c, _t in self._columns]

        # Multi-select: once more than one row is highlighted the name box is
        # no longer the answer, so judging the name would report on one column
        # out of several. Every selected row is an existing column by
        # construction, so the verdict is simply "use them".
        picked = self._selected_column_names()
        if self._multi and len(picked) > 1:
            self._near = ""
            self._action = ACTION_USE
            shown = ", ".join(picked[:6])
            more = "" if len(picked) <= 6 else f", and {len(picked) - 6} more"
            self._status.setText(
                f"{len(picked)} columns selected in {table} — all of them "
                f"will be added: {shown}{more}.")
            self._confirm.setVisible(False)
            self._sync_ok()
            return
        self._near = ""

        if self._reader is None or not existing:
            self._action = (ACTION_UNCHECKED if str(name).strip()
                            else ACTION_INVALID)
            self._status.setText(
                f"'{name.strip()}' cannot be checked — no database columns are "
                f"loaded. It will be used exactly as typed."
                if self._action == ACTION_UNCHECKED
                else "Type a column name.")
            self._confirm.setVisible(False)
            self._sync_ok()
            return

        match = find_existing(name, existing)
        if match:
            self._action = ACTION_USE
            extra = ("" if match == name.strip()
                     else f" (SQLite ignores case, so this is '{match}')")
            self._status.setText(
                f"'{match}' already exists in {table}{extra} — it will be used "
                f"as it is, and nothing new is created.")
            self._confirm.setVisible(False)
            self._sync_ok()
            return

        if not self._allow_new:
            self._action = ACTION_INVALID
            self._status.setText(
                f"'{name.strip()}' is not a column of {table}. This field only "
                f"accepts a column that already exists — pick one above.")
            self._confirm.setVisible(False)
            self._sync_ok()
            return

        problem = validate_column_name(name)
        if problem:
            self._action = ACTION_INVALID
            self._status.setText(problem)
            self._confirm.setVisible(False)
            self._sync_ok()
            return

        self._near = near_miss(name, existing)
        if self._near:
            self._confirm.setVisible(True)
            if self._confirm.isChecked():
                self._action = ACTION_CREATE
                self._status.setText(
                    f"'{name.strip()}' will be created in {table}, alongside "
                    f"'{self._near}'.")
            else:
                self._action = ACTION_CONFIRM
                self._status.setText(
                    f"'{name.strip()}' is new, but {table} already has "
                    f"'{self._near}'. Did you mean '{self._near}'? One "
                    f"mistyped character here splits your work across two "
                    f"near-identical columns that then look like two "
                    f"annotators who agree on nothing. Tick the box below to "
                    f"create '{name.strip()}' anyway.")
            self._sync_ok()
            return

        self._confirm.setVisible(False)
        self._action = ACTION_CREATE
        self._status.setText(
            f"'{name.strip()}' is not in {table} yet — it will be created "
            f"the first time spaCR writes to it.")
        self._sync_ok()

    def _sync_ok(self) -> None:
        ok = self._buttons.button(QDialogButtonBox.Ok)
        ok.setEnabled(self._action in (ACTION_USE, ACTION_CREATE,
                                       ACTION_UNCHECKED))

    # -- public API --------------------------------------------------------

    def action(self) -> str:
        """Return what OK would do: ``use``/``create``/``confirm``/
        ``invalid``/``unchecked``."""
        return self._action

    def status_text(self) -> str:
        """Return the sentence under the name box."""
        return self._status.text()

    def banner_text(self) -> str:
        """Return the inline problem banner (``""`` when there is none)."""
        return self._banner.text()

    def confirm_offered(self) -> bool:
        """Return whether the "create it anyway" box is being offered."""
        return not self._confirm.isHidden()

    def near_miss_column(self) -> str:
        """Return the existing column the typed name resembles, or ``""``."""
        return self._near

    def confirm_box(self) -> Toggle:
        """Return the "create it anyway" checkbox (visible only on a near-miss)."""
        return self._confirm

    def name_edit(self) -> QLineEdit:
        """Return the name box, for hosts that want to prefill or focus it."""
        return self._name

    def _selected_column_names(self) -> List[str]:
        """The highlighted rows of the column tree, in table order."""
        tree = getattr(self, "_column_tree", None)
        if tree is None:
            return []
        return [tree.topLevelItem(i).text(0)
                for i in range(tree.topLevelItemCount())
                if tree.topLevelItem(i).isSelected()]

    def is_multi(self) -> bool:
        """Whether this dialog returns several columns."""
        return self._multi

    def chosen_column(self) -> str:
        """Return the name currently in the box, trimmed."""
        return self._name.text().strip()

    def chosen_columns(self) -> List[str]:
        """Every column OK would hand back, in order and without repeats.

        Always non-empty when :meth:`chosen_column` is — a single-select
        dialog is the one-element case of this, so a caller can use this
        alone. In multi mode the highlighted rows are the answer, with the
        typed name added when it is not one of them (that is how a column
        that does not exist yet is named).
        """
        picked = self._selected_column_names() if self._multi else []
        typed = self.chosen_column()
        if typed and typed not in picked:
            # Only one row selected: the tree filled the name box from it, so
            # `typed` IS that row and appending it would not add anything.
            picked = picked + [typed]
        return list(dict.fromkeys(picked))

    def select_columns(self, names: Sequence[str]) -> List[str]:
        """Highlight several columns at once. Returns the ones that existed."""
        tree = self._column_tree
        wanted = [str(n) for n in names]
        tree.clearSelection()
        found: List[str] = []
        last = None
        for i in range(tree.topLevelItemCount()):
            item = tree.topLevelItem(i)
            if item.text(0) in wanted:
                item.setSelected(True)
                found.append(item.text(0))
                last = item
        if last is not None:
            # setCurrentItem would clear the selection we just made; the
            # current item only matters for the name box and Count non-null.
            tree.setCurrentItem(last, 0, QItemSelectionModel.NoUpdate)
            self._name.setText(last.text(0))
        self._evaluate()
        return found

    def chosen_table(self) -> str:
        """Return the selected table name, or ``""``."""
        item = self._tables.currentItem()
        return item.text() if item is not None else ""

    def table_names(self) -> List[str]:
        """Return the tables listed in the dialog."""
        return [self._tables.item(i).text() for i in range(self._tables.count())]

    def column_names(self) -> List[str]:
        """Return the columns of the selected table."""
        return [c for c, _t in self._columns]

    def visible_column_names(self) -> List[str]:
        """Return the columns not hidden by the filter box."""
        tree = self._column_tree
        return [tree.topLevelItem(i).text(0)
                for i in range(tree.topLevelItemCount())
                if not tree.topLevelItem(i).isHidden()]

    def executed_sql(self) -> List[str]:
        """Return every statement the dialog's reader has run.

        The assertion hook for "what did opening this cost". Threaded,
        the statements are appended on a worker thread, so the snapshot
        is taken under the reader's lock — see
        :meth:`SchemaReader.executed_sql`. It is still only complete once
        the load is (:meth:`is_busy` says when).
        """
        if self._reader is None:
            return []
        return self._reader.executed_sql()

    def is_busy(self) -> bool:
        """True while a schema or table read has not been delivered."""
        return self._jobs.is_busy()

    def active_jobs(self) -> int:
        """How many reader threads are still winding down."""
        return self._jobs.active_jobs()

    def select_table(self, name: str) -> bool:
        """Select ``name`` in the table list. Returns False if absent."""
        items = self._tables.findItems(str(name), Qt.MatchExactly)
        if not items:
            return False
        self._tables.setCurrentItem(items[0])
        return True

    def select_column(self, name: str) -> bool:
        """Select ``name`` in the column list (and fill the name box)."""
        matches = self._column_tree.findItems(str(name), Qt.MatchExactly, 0)
        if not matches:
            return False
        self._column_tree.setCurrentItem(matches[0])
        self._name.setText(matches[0].text(0))
        return True

    def set_name(self, text: str) -> None:
        """Type ``text`` into the name box (as if the user had)."""
        self._name.setText(str(text))

    def is_accept_enabled(self) -> bool:
        """Return whether OK is currently clickable."""
        return self._buttons.button(QDialogButtonBox.Ok).isEnabled()

    def accept(self) -> None:  # noqa: D102 - Qt override
        # Belt and braces: the button is already disabled in these states,
        # but Enter in the name box would otherwise bypass it.
        if self._action not in (ACTION_USE, ACTION_CREATE, ACTION_UNCHECKED):
            return
        super().accept()

    def done(self, result: int) -> None:  # noqa: D102 - Qt override
        """Close, and let no reader thread outlive the dialog.

        ``done`` rather than ``closeEvent`` because it is the one funnel:
        ``accept``, ``reject`` and the window's close button all arrive
        here, and a dialog dismissed halfway through its first read is
        the ordinary case — the user clicked ``SQL`` on the wrong field.
        ``JobRunner.shutdown`` drops the results and waits briefly so
        nothing destroys a running QThread.
        """
        self._jobs.shutdown()
        super().done(result)


# ---------------------------------------------------------------------------
# The button + the one-line attachment
# ---------------------------------------------------------------------------

class ColumnPickerButton(QToolButton):
    """The small ``SQL`` button that opens a :class:`ColumnPickerDialog`.

    :param db_path_getter: callable returning the current database path or
        run folder — a callable, not a string, because the path usually
        depends on a ``src`` field the user is still editing. A plain
        string is accepted and wrapped.
    :param table: table to preselect in the dialog.
    :param field: the ``QLineEdit``/``QComboBox``/chip-strip editor a picked
        name is written into; may be ``None`` for a button that only emits
        :attr:`picked`.
    :param multi: this field holds a *list* of columns — names are appended
        rather than replacing what is there, and the dialog lets the user
        select any number of columns in one visit instead of one per press.
        ``None`` auto-detects (:func:`field_is_list`).
    """

    #: Emitted once per chosen column name, after the dialog is accepted.
    picked = Signal(str)
    #: Emitted once with every chosen name, after the dialog is accepted.
    picked_many = Signal(list)

    def __init__(self, db_path_getter: Any = "", table: Optional[str] = None,
                 field: Optional[QWidget] = None, parent: Optional[QWidget] = None,
                 text: str = "SQL", allow_new: bool = True,
                 multi: Optional[bool] = None):
        super().__init__(parent)
        self._getter = (db_path_getter if callable(db_path_getter)
                        else (lambda v=db_path_getter: v))
        self.table = table
        self.field = field
        self.allow_new = bool(allow_new)
        self.multi = multi
        self._runner: Callable[[ColumnPickerDialog], int] = lambda d: d.exec()

        self.setObjectName("ColumnPickerButton")
        self.setText(text)
        self.setCursor(Qt.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setToolTip(
            "Show the columns this database already has, and pick or name "
            "one. Opening this reads the schema only — nothing is written.")
        self.clicked.connect(self.open_picker)

    # -- wiring ------------------------------------------------------------

    def db_path(self) -> str:
        """Return the path the getter currently reports (``""`` on error)."""
        try:
            value = self._getter()
        except Exception:
            return ""
        return "" if value is None else str(value)

    def set_dialog_runner(self, runner: Callable[["ColumnPickerDialog"], int]) -> None:
        """Replace how the dialog is run.

        The default enters a modal event loop. Tests inject a runner that
        inspects the dialog and returns ``QDialog.Accepted``/``Rejected``
        without ever blocking — which is also how a host could swap in a
        non-modal presentation later.
        """
        self._runner = runner

    def current_text(self) -> str:
        """Return the host field's current text (``""`` when unbound)."""
        return field_text(self.field)

    def make_dialog(self) -> ColumnPickerDialog:
        """Build the dialog, wired to the current path — but do not run it.

        ``threaded=True``: this is the user-facing path, and the dialog
        it builds is run modally by :meth:`open_picker`, so every
        millisecond the constructor spends in sqlite is a millisecond
        between the click and any window at all. See
        :class:`ColumnPickerDialog` for the two modes and why the
        *default* is the other one.
        """
        current = self.current_text()
        multi = self.is_multi()
        if multi:
            current = ""
        return ColumnPickerDialog(
            db_path=self.db_path(), table=self.table, current=current,
            parent=self.window(), allow_new=self.allow_new, threaded=True,
            multi=multi)

    def is_multi(self) -> bool:
        """Whether this field holds a list of columns rather than one.

        Explicit ``multi`` wins. Otherwise the *field* decides: a chip-strip
        editor is list-valued whether or not it currently holds anything,
        which a text inspection cannot tell (an empty one reads ``""``, and
        an empty list field is exactly when the user most wants to add
        several at once). Text fields keep the old shape test.
        """
        if self.multi is not None:
            return bool(self.multi)
        return field_is_list(self.field) or _looks_like_list(self.current_text())

    def open_picker(self) -> str:
        """Open the picker; on OK, write every chosen name into the field.

        :returns: the first chosen column name, or ``""`` when cancelled.
            :attr:`picked_many` carries the whole selection.
        """
        dialog = self.make_dialog()
        try:
            result = self._runner(dialog)
            if result != QDialog.Accepted:
                return ""
            names = dialog.chosen_columns()
        finally:
            dialog.deleteLater()
        if not names:
            return ""
        multi = self.is_multi()
        set_field_values(self.field, names, append=multi)
        for name in names:
            self.picked.emit(name)
        self.picked_many.emit(list(names))
        return names[0]


def _looks_like_list(text: str) -> bool:
    """Return True when a field's text holds several column names.

    ``annotation_columns`` renders as ``['annotate', 'annotate_2']`` and
    ``measurement`` as ``a, b``; overwriting either with a single name
    would quietly drop the rest.
    """
    t = str(text or "").strip()
    return t.startswith("[") or ("," in t)


def chip_editor(field: Optional[QWidget]) -> Optional[QWidget]:
    """Return ``field`` when it is a chip-strip list editor, else ``None``.

    Duck-typed on purpose: the editor lives in
    :mod:`spacr.qt.screens.settings_model` and importing a *screen* from a
    *widget* would invert the dependency (and, in practice, cycle). The test
    is "not a text field, but speaks ``get_value``/``set_value``" — the
    settings screen's ``_ScalarEdit`` and ``_ListEdit`` speak those too, but
    they are ``QLineEdit`` subclasses and are caught by the branch above.
    """
    if field is None or isinstance(field, (QLineEdit, QComboBox)):
        return None
    if callable(getattr(field, "get_value", None)) and \
            callable(getattr(field, "set_value", None)):
        return field
    return None


def field_is_list(field: Optional[QWidget]) -> bool:
    """Whether ``field`` holds a list of names rather than a single one."""
    return chip_editor(field) is not None


def field_values(field: Optional[QWidget]) -> List[str]:
    """Return the column names ``field`` currently holds, in order."""
    editor = chip_editor(field)
    if editor is not None:
        value = editor.get_value()
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return [str(v).strip() for v in value if str(v).strip()]
        return [str(value).strip()] if str(value).strip() else []
    text = field_text(field).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        return [p.strip().strip("'\"") for p in text[1:-1].split(",")
                if p.strip()]
    return [p.strip() for p in text.split(",") if p.strip()]


def field_text(field: Optional[QWidget]) -> str:
    """Return the text of a line edit / combo box / chip strip."""
    if isinstance(field, QComboBox):
        return field.currentText()
    if isinstance(field, QLineEdit):
        return field.text()
    editor = chip_editor(field)
    if editor is not None:
        return ", ".join(field_values(editor))
    return ""


def set_field_values(field: Optional[QWidget], names: Sequence[str],
                     append: bool = False) -> bool:
    """Write every name in ``names`` into ``field``. False when it could not.

    The multi-column half of :func:`set_field_text`. A chip strip is set
    from a real list — no punctuation round trip — and anything else keeps
    its own list style through :func:`_appended`.
    """
    wanted = [str(n).strip() for n in names if str(n).strip()]
    if field is None or not wanted:
        return False
    editor = chip_editor(field)
    if editor is not None:
        existing = field_values(editor) if append else []
        editor.set_value(list(dict.fromkeys(existing + wanted)))
        return True
    if not append:
        # Replacing a single-valued field with several names would lose all
        # but one silently; keep them all, in the field's own style.
        ok = set_field_text(field, wanted[0], append=False)
        for name in wanted[1:]:
            ok = set_field_text(field, name, append=True) and ok
        return ok
    ok = True
    for name in wanted:
        ok = set_field_text(field, name, append=True) and ok
    return ok


def set_field_text(field: Optional[QWidget], name: str,
                   append: bool = False) -> bool:
    """Write ``name`` into ``field``. Returns False when it could not.

    :param field: ``QLineEdit`` (including the settings screen's
        ``_ScalarEdit``/``_ListEdit`` subclasses) or ``QComboBox``.
    :param name: the column name to write.
    :param append: add to the existing value instead of replacing it,
        keeping the field's own list style (``['a', 'b']`` or ``a, b``).
    """
    if field is None:
        return False
    if append:
        name = _appended(field_text(field), name)
    if isinstance(field, QComboBox):
        if field.isEditable():
            field.setEditText(name)
            return True
        idx = field.findText(name)
        if idx < 0:
            field.addItem(name)
            idx = field.findText(name)
        field.setCurrentIndex(idx)
        return True
    if isinstance(field, QLineEdit):
        field.setText(name)
        return True
    return False


def _appended(existing: str, name: str) -> str:
    """Return ``existing`` with ``name`` added, in whichever style it uses."""
    current = str(existing or "").strip()
    if not current:
        return name
    if current.startswith("[") and current.endswith("]"):
        inner = current[1:-1].strip()
        parts = [p.strip() for p in inner.split(",") if p.strip()]
        if any(p.strip("'\"") == name for p in parts):
            return current
        parts.append(repr(name))
        return "[" + ", ".join(parts) + "]"
    parts = [p.strip() for p in current.split(",") if p.strip()]
    if name in parts:
        return current
    parts.append(name)
    return ", ".join(parts)


def _find_layout_with(layout: Optional[QLayout],
                      widget: QWidget) -> Optional[QLayout]:
    """Return the (possibly nested) layout that directly holds ``widget``."""
    if layout is None:
        return None
    if layout.indexOf(widget) >= 0:
        return layout
    for i in range(layout.count()):
        child = layout.itemAt(i)
        found = _find_layout_with(child.layout() if child else None, widget)
        if found is not None:
            return found
    return None


def attach_column_picker(field: QWidget, db_path_getter: Any,
                         table: Optional[str] = None, *,
                         text: str = "SQL", allow_new: bool = True,
                         multi: Optional[bool] = None,
                         tooltip: Optional[str] = None,
                         layout: Optional[QLayout] = None,
                         on_pick: Optional[Callable[[str], None]] = None
                         ) -> ColumnPickerButton:
    """Put a ``SQL`` button beside ``field`` and wire it to the picker.

    Designed to be a single line a host screen adds after it has already
    laid the field out::

        attach_column_picker(self._ann_col,
                             lambda: self._src_edit.text(), "png_list")

    The field keeps its place in its parent layout: the button and the
    field are wrapped in a small horizontal box that takes over the
    field's original slot, so a ``QFormLayout`` row keeps its label and a
    caller that stored a reference to the field keeps using it unchanged.
    When the field is not in a layout yet the button is simply returned
    unplaced, and the caller positions it.

    :param field: the ``QLineEdit``/``QComboBox`` the user types into.
    :param db_path_getter: callable (or string) giving the database path
        or run folder. Called each time the button is pressed, so a path
        the user edits later is picked up.
    :param table: table to preselect — ``"png_list"`` for annotation
        columns, ``"cell"``/``"object"`` for measurements.
    :param text: button label.
    :param allow_new: allow naming a column that does not exist yet.
    :param multi: append rather than replace (``None`` auto-detects a
        list-valued field).
    :param tooltip: override the button's tooltip.
    :param layout: the layout holding ``field``, for the case where the
        host builds a ``QFormLayout`` before installing it on a widget
        (so ``field.parentWidget()`` is still ``None``). Normally
        unnecessary — the field's own parent is searched.
    :param on_pick: extra callback invoked with the chosen name.
    :returns: the created :class:`ColumnPickerButton`.
    """
    button = ColumnPickerButton(db_path_getter, table=table, field=field,
                                text=text, allow_new=allow_new, multi=multi)
    if tooltip:
        button.setToolTip(tooltip)
    if on_pick is not None:
        button.picked.connect(on_pick)

    parent = field.parentWidget()
    host = _find_layout_with(layout if layout is not None
                             else (parent.layout() if parent else None), field)
    if host is None:
        button.setParent(parent)
        return button

    wrapper = QWidget(parent)
    wrapper.setObjectName("ColumnPickerRow")
    old = host.replaceWidget(field, wrapper)
    if old is not None:
        del old
    row = QHBoxLayout(wrapper)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(4)
    row.addWidget(field, 1)
    row.addWidget(button, 0)
    # Visibility is deliberately left alone. Qt's own reparent-into-layout
    # path handles it: QWidget::setParent clears WA_WState_ExplicitShowHide
    # when it hides an already-shown widget, so the field reappears with the
    # wrapper, and QLayout::addChildWidget queues a show for a wrapper added
    # under a parent that is visible right now. Forcing visibility here would
    # set the explicit-show flag and break a collapsed Section's expand.
    return button

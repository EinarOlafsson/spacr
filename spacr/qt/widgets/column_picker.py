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
  count at all. The per-table row figure comes from ``max(rowid)`` and is
  labelled an estimate; a per-column non-null count happens only when the
  user asks for it by name.

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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from urllib.parse import quote as _urlquote

from PySide6.QtCore import Qt, Signal
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
    "near_miss",
    "resolve_db_path",
    "validate_column_name",
]

DB_FILENAME = "measurements.db"
_MEASUREMENTS_SUBDIR = "measurements"

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

    :param path: database file or run folder (see :func:`resolve_db_path`).
    :ivar executed: every statement this reader has run, in order — the
        hook a test uses to prove that opening the picker costs no
        ``COUNT(*)``.
    """

    def __init__(self, path: str):
        self.path = resolve_db_path(path)
        self.uri = _read_only_uri(self.path)
        self.executed: List[str] = []

    # -- plumbing ----------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.uri, uri=True)
        con.execute("PRAGMA query_only = ON")
        return con

    def _fetch(self, sql: str, params: Sequence = ()) -> List[tuple]:
        self.executed.append(sql)
        con = self._connect()
        try:
            return list(con.execute(sql, tuple(params)).fetchall())
        finally:
            con.close()

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

        ``max(rowid)`` is answered from the right-hand edge of the b-tree
        without a scan. Deleted rows leave gaps, so it is an estimate and
        every caller must label it one. ``None`` for a view, a WITHOUT
        ROWID table, or an empty table.
        """
        try:
            rows = self._fetch(f"SELECT max(rowid) FROM {quote_ident(table)}")
        except sqlite3.Error:
            return None
        if not rows or rows[0][0] is None:
            return None
        return int(rows[0][0])

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

    :param db_path: database file or run folder; may be empty.
    :param table: table to preselect (e.g. ``png_list`` for annotations).
    :param current: the field's current value, prefilled into the name box.
    :param allow_new: when False, only existing columns are accepted.
    :param reader: an already-built :class:`SchemaReader` (or a stand-in);
        injecting one is how tests exercise schema edge cases.
    """

    def __init__(self, db_path: Any = "", table: Optional[str] = None,
                 current: str = "", parent: Optional[QWidget] = None,
                 allow_new: bool = True,
                 reader: Optional[SchemaReader] = None):
        super().__init__(parent)
        self.setWindowTitle("Pick a database column")
        self.setObjectName("ColumnPickerDialog")
        self.setMinimumWidth(560)

        self._allow_new = bool(allow_new)
        self._preferred_table = str(table or "")
        self._columns: List[Tuple[str, str]] = []
        self._action = ACTION_UNCHECKED
        self._near = ""

        if reader is None:
            self._reader, self._open_error = open_reader(db_path)
        else:
            self._reader, self._open_error = reader, ""

        self._build_ui()
        self._load_tables()
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
        self._source.setText(
            f"Reading {self._reader.path} (read-only)" if self._reader
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
        self._column_tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self._column_tree.currentItemChanged.connect(self._on_column_changed)
        self._column_tree.itemDoubleClicked.connect(self._on_column_activated)
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

    def _load_tables(self) -> None:
        if self._reader is None:
            self._set_banner(self._open_error)
            return
        try:
            names = self._reader.tables()
        except sqlite3.Error as exc:
            self._set_banner(f"Cannot read the schema: {exc}")
            return
        if not names:
            self._set_banner(
                f"{os.path.basename(self._reader.path)} has no tables yet — "
                f"nothing has been written to it. Run Measure first.")
            return
        self._tables.addItems(names)
        wanted = self._preferred_table if self._preferred_table in names else names[0]
        self._tables.setCurrentRow(names.index(wanted))

    def _on_table_changed(self, name: str) -> None:
        self._load_columns(name)
        self._evaluate()

    def _load_columns(self, table: str) -> None:
        self._column_tree.clear()
        self._columns = []
        self._summary.setText("")
        self._count_btn.setEnabled(False)
        if not table or self._reader is None:
            return
        self._columns_label.setText(f"Columns in {table}")
        try:
            info = self._reader.column_info(table)
        except sqlite3.Error as exc:
            self._set_banner(
                f"Cannot list the columns of '{table}': {exc}. A view whose "
                f"table has been dropped does this — pick another table.")
            return
        if not info:
            self._set_banner(
                f"'{table}' reports no columns, so there is nothing to pick "
                f"from it. Pick another table.")
            return
        self._set_banner("")
        self._columns = info
        for col_name, decl in info:
            QTreeWidgetItem(self._column_tree, [col_name, decl or "—", ""])
        rows = self._reader.estimate_rows(table)
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

    def chosen_column(self) -> str:
        """Return the name currently in the box, trimmed."""
        return self._name.text().strip()

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
        """Return every statement the dialog's reader has run."""
        return list(self._reader.executed) if self._reader is not None else []

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
    :param field: the ``QLineEdit``/``QComboBox`` a picked name is written
        into; may be ``None`` for a button that only emits :attr:`picked`.
    :param multi: append to the field instead of replacing it (for fields
        that hold a *list* of columns). ``None`` auto-detects.
    """

    #: Emitted with the chosen column name after the dialog is accepted.
    picked = Signal(str)

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
        """Build the dialog, wired to the current path — but do not run it."""
        current = self.current_text()
        if self.multi or (self.multi is None and _looks_like_list(current)):
            current = ""
        return ColumnPickerDialog(
            db_path=self.db_path(), table=self.table, current=current,
            parent=self.window(), allow_new=self.allow_new)

    def open_picker(self) -> str:
        """Open the picker; on OK, write the name into the field.

        :returns: the chosen column name, or ``""`` when cancelled.
        """
        dialog = self.make_dialog()
        try:
            result = self._runner(dialog)
            if result != QDialog.Accepted:
                return ""
            name = dialog.chosen_column()
        finally:
            dialog.deleteLater()
        if not name:
            return ""
        multi = self.multi
        if multi is None:
            multi = _looks_like_list(self.current_text())
        set_field_text(self.field, name, append=bool(multi))
        self.picked.emit(name)
        return name


def _looks_like_list(text: str) -> bool:
    """Return True when a field's text holds several column names.

    ``annotation_columns`` renders as ``['annotate', 'annotate_2']`` and
    ``measurement`` as ``a, b``; overwriting either with a single name
    would quietly drop the rest.
    """
    t = str(text or "").strip()
    return t.startswith("[") or ("," in t)


def field_text(field: Optional[QWidget]) -> str:
    """Return the text of a line edit / combo box, ``""`` for anything else."""
    if isinstance(field, QComboBox):
        return field.currentText()
    if isinstance(field, QLineEdit):
        return field.text()
    return ""


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

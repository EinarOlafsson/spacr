"""A settings widget for "one or more files": browse, drop, repeat.

Several spaCR settings are lists of input paths -- ``score_data``,
``count_data``, ``metadata_files``, ``grna_csv`` and friends. They used to
render as a chip strip that the user typed absolute paths into, one character
at a time, which is unusable for the four plate CSVs a screen actually has and
silently accepted a path that did not exist.

:class:`FilePathListWidget` replaces that with the two gestures people expect:

* **Add files...** opens a file dialog with multi-selection enabled. Pressing
  it again *appends*, so several sources can be gathered from different
  folders in several trips rather than one impossible single selection.
* **Dropping** files or folders anywhere on the widget adds them. A dropped
  folder contributes its matching files, one directory level deep, sorted.

Everything else follows from those: duplicates are refused, order is editable
because a regression that reads ``plate1..plate4`` cares about it, missing
paths are marked instead of being discovered at run time, and the value is a
plain ``list[str]`` so the CLI, the settings CSV and the Qt panel all agree.
"""

from __future__ import annotations

import os
import re
from typing import Any, Iterable, List, Sequence

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDragMoveEvent, QDropEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


def _pair_tokens(path: str) -> set[str]:
    stem = os.path.splitext(os.path.basename(os.fspath(path)))[0].casefold()
    stem = re.sub(r"plate[\s_-]*0*(\d+)", r"plate\1", stem)
    generic = {"score", "scores", "count", "counts", "result", "results",
               "unique", "combinations", "csv", "maxvit", "xgb"}
    return {token for token in re.findall(r"[a-z]+\d+|\d+|[a-z]+", stem)
            if token not in generic and len(token) > 1}


#: What a measurements DATABASE is called, by extension.
#:
#: Databases are told apart from tables by extension and never by header.
#: :func:`side_for_header` opens a file as text, and a sqlite file read as
#: CSV yields a header carrying neither a gRNA nor a count -- which scores it
#: as a SCORE table. A binary database silently filed as the regression's
#: response is exactly the failure this column exists to remove.
DATABASE_EXTENSIONS = (".db", ".sqlite", ".sqlite3")

# Words every plate's database shares, so they cannot tell two plates apart.
# 'measurements' is here because spaCR's own layout calls every plate's
# database <plate>/measurements/measurements.db.
_DATABASE_GENERIC = frozenset({
    "measurements", "measurement", "database", "sqlite", "data", "merged",
    "results", "result", "analysis"})


def is_database_path(path) -> bool:
    """``True`` when ``path`` names a measurements database by extension.

    One rule, in one place, for the widget that files a dropped path into a
    column and for the drop handler that catches the same file when it lands
    on the screen around the widget. Two copies would disagree the first time
    somebody's database was called ``plate1.sqlite``.
    """
    return os.path.splitext(os.fspath(path))[1].lower() in DATABASE_EXTENSIONS


def _database_tokens(path, *, depth: int = 2) -> set[str]:
    """Pairing tokens for a database, its parent folders included.

    ``_pair_tokens`` reads the basename, which is all a CSV has. A database
    has less: spaCR writes every plate's to ``<plate>/measurements/
    measurements.db``, so basename tokens are identical for every plate,
    every candidate ties, and the tie rule leaves them all unpaired. The
    plate is named by the FOLDER, exactly as ``multi_database._label_for``
    already assumes when it disambiguates two databases for a legend.
    """
    full = os.fspath(path)
    tokens = _pair_tokens(full) - _DATABASE_GENERIC
    parent = os.path.dirname(full)
    for _ in range(max(0, int(depth))):
        name = os.path.basename(parent)
        if not name:
            break
        folder = _pair_tokens(name) - _DATABASE_GENERIC
        tokens |= folder
        parent = os.path.dirname(parent)
        if folder:
            # THE NEAREST FOLDER THAT SAYS ANYTHING IS ENOUGH. Climbing
            # further collects whatever the tree above happens to be called,
            # and one project folder with a number in its name would invent a
            # plate token for every database underneath it. ``measurements``
            # and its friends are skipped because they say nothing about
            # WHICH plate -- which is the only reason to look up at all.
            break
    return tokens


def _plate_label(tokens) -> str:
    """The longest ``plate...`` token, which is what names the row."""
    plates = sorted((token for token in tokens if token.startswith("plate")),
                    key=len, reverse=True)
    return plates[0] if plates else ""


def _best_unique(left: set, tokens: Sequence[set], unused: set):
    """Index of the single best token match in ``unused``, or ``None``.

    The rule instruction 107 settled, kept in one function because all three
    columns now obey it: a match must overlap at all, and must beat the
    runner-up outright. A TIE LEAVES THE ROW UNPAIRED rather than guessing,
    because a confident wrong pairing is the failure that has no symptom.
    """
    ranked = sorted(((len(left & tokens[index]), index) for index in unused),
                    reverse=True)
    if ranked and ranked[0][0] > 0 and (
            len(ranked) == 1 or ranked[0][0] > ranked[1][0]):
        return ranked[0][1]
    return None


def suggest_file_pairs(scores: Sequence[str], counts: Sequence[str], *,
                       databases: Sequence[str] = ()) -> list[dict]:
    """Propose visible score/count/database rows by filename tokens.

    A proposal is never authoritative: the editable table is the contract the
    user confirms. Unique best matches are used; ties remain unpaired.

    ``databases`` is keyword-only and defaults to nothing, so the two-argument
    call every existing caller makes still means what it did. A database is
    matched against BOTH cells of a row it may join -- a plate is named by its
    score CSV as often as by its count CSV -- and one that matches nothing is
    listed on its own row rather than being dropped or guessed onto row 0.
    """
    unused = set(range(len(counts)))
    count_tokens = [_pair_tokens(path) for path in counts]
    rows = []
    for score in scores:
        left = _pair_tokens(score)
        match = _best_unique(left, count_tokens, unused)
        if match is not None:
            unused.remove(match)
        common = left & (count_tokens[match] if match is not None else set())
        rows.append({"plate": _plate_label(common),
                     "score": os.fspath(score),
                     "count": os.fspath(counts[match]) if match is not None else None,
                     "database": None})
    for index in sorted(unused):
        rows.append({"plate": "", "score": None,
                     "count": os.fspath(counts[index]), "database": None})
    return _attach_databases(rows, databases)


def _attach_databases(rows: list[dict], databases: Sequence[str]) -> list[dict]:
    """Fill each row's ``database`` cell by token, appending what is left over."""
    paths = [os.fspath(path) for path in (databases or [])]
    tokens = [_database_tokens(path) for path in paths]
    unused = set(range(len(paths)))
    for row in rows:
        left = set()
        for side in ("score", "count"):
            if row.get(side):
                left |= _pair_tokens(row[side])
        if row.get("plate"):
            left.add(row["plate"])
        match = _best_unique(left, tokens, unused) if unused else None
        if match is None:
            row.setdefault("database", None)
            continue
        unused.remove(match)
        row["database"] = paths[match]
        if not row.get("plate"):
            row["plate"] = _plate_label(left & tokens[match])
    for index in sorted(unused):
        rows.append({"plate": _plate_label(tokens[index]), "score": None,
                     "count": None, "database": paths[index]})
    return rows


def side_for_header(path) -> str:
    """``'count'`` when the file's header names a gRNA and a count, else
    ``'score'``.

    Read from the header rather than the filename: a count export carries a
    gRNA name and a count, a score export carries neither, and that is true
    whatever the file is called.

    Module-level because two screens ask the same question of the same files.
    Regression asks it through :class:`PairedFileTableWidget`; Parameter Sweep
    holds its two sides in separate list widgets and asks it through
    ``spacr.qt.dnd_handlers.SweepInputsDropHandler``. A second copy of this
    rule would drift, and the direction it would drift in is silent: a count
    table filed as a score is not an error, it is a wrong regression.
    """
    import csv as _csv
    try:
        with open(path, newline="", encoding="utf-8",
                  errors="replace") as handle:
            header = {str(name).strip().lower()
                      for name in next(_csv.reader(handle), [])}
    except (OSError, _csv.Error):
        # csv.Error is 'line contains NUL': a BINARY file was asked this
        # question. It used to escape from inside Qt's drop dispatch, where
        # an exception is a crash rather than an error dialog -- so dropping
        # a measurements database on the input table killed the window.
        # Databases are now routed by extension before they get here
        # (:func:`is_database_path`); this catch is for the next binary.
        return "score"
    return ("count" if {"grna", "grna_name"} & header and "count" in header
            else "score")


class PairedFileTableWidget(QWidget):
    """Editable one-row-per-plate score / count / database input contract.

    A row is ONE PLATE: its score CSV, its count CSV, and the measurements
    database that plate's per-object tables live in. All three columns are
    filled BY ADDITION -- every arrival re-proposes the whole table from
    filename tokens -- so databases dropped in the opposite order to the CSVs
    still land on the right plates. That is instruction 107's rule, and the
    third column obeys it rather than keeping a list of its own.

    A plate with NO database is legal and is not an error: the regression is
    fitted on scores and counts. The database is what makes that plate's
    measurements available downstream, so its absence disables the plate
    there instead of failing the run.
    """

    value_changed = Signal()

    _EMPTY_STATUS = ("Drop score and count CSVs here. Drop a measurements "
                     "database on a plate row to attach it to that plate.")

    def __init__(self, value=None, parent=None):
        super().__init__(parent)
        self._scores: list[str] = []
        self._counts: list[str] = []
        self._databases: list[str] = []
        # Databases the user placed on a row BY HAND, anchored to that row's
        # identity rather than to its index. Every addition re-proposes the
        # whole table, so without this an explicit attachment would be undone
        # by the next CSV drop -- silently, which is the same class of bug as
        # pairing by list position.
        self._pinned: dict[str, dict] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.table = QTableWidget(0, 5, self)
        self.table.setHorizontalHeaderLabels(
            ["Plate / proposal", "Score CSV", "Count CSV",
             "Measurements DB", "Plate rule"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.itemChanged.connect(lambda *_: self.value_changed.emit())
        layout.addWidget(self.table)
        # What the table just did, in words. A database attached to "the
        # first row without one" that says nothing is a file the user
        # believes is on another plate.
        self.status = QLabel(self._EMPTY_STATUS, self)
        self.status.setWordWrap(True)
        self.status.setProperty("role", "hint")
        layout.addWidget(self.status)
        buttons = QHBoxLayout()
        add_scores = QPushButton("Add score CSVs…", self)
        add_counts = QPushButton("Add count CSVs…", self)
        add_databases = QPushButton("Add measurements DBs…", self)
        add_row = QPushButton("Add empty pair", self)
        up = QPushButton("↑", self)
        down = QPushButton("↓", self)
        remove = QPushButton("Remove", self)
        add_scores.clicked.connect(lambda: self._pick("score"))
        add_counts.clicked.connect(lambda: self._pick("count"))
        add_databases.clicked.connect(lambda: self._pick("database"))
        add_row.clicked.connect(lambda: self._append_row({}))
        up.clicked.connect(lambda: self._move(-1))
        down.clicked.connect(lambda: self._move(1))
        remove.clicked.connect(self._remove)
        for button in (add_scores, add_counts, add_databases, add_row, up,
                       down, remove):
            buttons.addWidget(button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        # The table is the drop target the user aims at, so the widget takes
        # drops and the table does not swallow them first.
        self.setAcceptDrops(True)
        self.table.setAcceptDrops(False)
        self.set_value(value)

    #: Column index of each side in the table, so a drop lands where the user
    #: aimed it rather than in whichever input the router reached first.
    SIDE_COLUMNS = {"score": 1, "count": 2, "database": 3}

    # The read-only column that reports how the plate id was resolved. It
    # moved right when the database column was inserted; it is named here so
    # that the next column to arrive moves one number, not four.
    RULE_COLUMN = 4

    def add_paths_for_side(self, paths, side: str = "score") -> int:
        """Add files to one column of the table and re-propose the pairing.

        The drop targets the user asked for -- "a dependent variable square I
        can drop into and an independent variable square I can drop into" --
        resolve to this. It is also what the drop router calls once it has
        decided which side a file belongs to from its header, so dropping a
        count table anywhere on the widget still fills the count column.

        ``'database'`` is the third side. It goes through this same method,
        and therefore through the same whole-table re-proposal, rather than
        through an adder of its own that would keep a private list and pair
        by the order things arrived.

        Returns how many files were added.
        """
        if side not in self.SIDE_COLUMNS:
            raise ValueError(
                f"side must be 'score', 'count' or 'database'; got {side!r}.")
        incoming = [os.fspath(p) for p in (paths or [])]
        if not incoming:
            return 0
        self._rebuild_sides()
        target = {"score": self._scores, "count": self._counts,
                  "database": self._databases}[side]
        added = 0
        for path in incoming:
            if path not in target:
                target.append(path)
                added += 1
        if added:
            # Re-propose so a count dropped after its score lands on the same
            # row: the pairing is by filename token, not by drop order.
            self._repropose()
            self.value_changed.emit()
        return added

    def _rebuild_sides(self) -> None:
        """Refill the three flat lists from the table, which is the state."""
        current = self.get_value()
        self._scores = list(dict.fromkeys(
            row["score"] for row in current if row.get("score")))
        self._counts = list(dict.fromkeys(
            row["count"] for row in current if row.get("count")))
        self._databases = list(dict.fromkeys(
            row["database"] for row in current if row.get("database")))

    def _repropose(self) -> None:
        """Rebuild every row from tokens, then honour the manual attachments."""
        rows = suggest_file_pairs(self._scores, self._counts,
                                  databases=self._databases)
        self.set_value(self._apply_pinned(rows))

    def _apply_pinned(self, rows: list[dict]) -> list[dict]:
        """Move each pinned database back onto the row the user chose."""
        for database, anchor in list(self._pinned.items()):
            current = next((index for index, row in enumerate(rows)
                            if row.get("database") == database), None)
            if current is None:
                # The user removed it from the table; the pin goes with it.
                self._pinned.pop(database, None)
                continue
            target = self._row_for_anchor(rows, anchor)
            if target is None or target == current:
                continue
            rows[current]["database"] = rows[target].get("database")
            rows[target]["database"] = database
        return [row for row in rows
                if row.get("score") or row.get("count") or row.get("database")]

    @staticmethod
    def _row_for_anchor(rows: list[dict], anchor: dict):
        """The row a pin names, by its files first and its plate label last."""
        for key in ("score", "count", "plate"):
            value = anchor.get(key)
            if not value:
                continue
            for index, row in enumerate(rows):
                if row.get(key) == value:
                    return index
        return None

    def _anchor_for_row(self, row: int) -> dict:
        return {key: self._cell(row, column) or None
                for key, column in (("plate", 0),
                                    ("score", self.SIDE_COLUMNS["score"]),
                                    ("count", self.SIDE_COLUMNS["count"]))}

    def _cell(self, row: int, column: int) -> str:
        item = self.table.item(row, column)
        return item.text().strip() if item else ""

    def _side_for_header(self, path) -> str:
        """Which column ``path`` belongs in. See :func:`side_for_header`."""
        return side_for_header(path)

    def _side_for_path(self, path) -> str:
        """Which column ``path`` belongs in, databases decided by extension."""
        return ("database" if is_database_path(path)
                else self._side_for_header(path))

    # ------------------------------------------------------- the third column

    def attach_database(self, path, row=None) -> str:
        """Attach a measurements database to one plate row, and say which.

        ``row`` is the row the user aimed the drop at -- an EXPLICIT
        assignment, which is remembered and survives the re-proposal that the
        next dropped CSV triggers.

        With no row, the database is offered to the token pairing first, so
        ``plate2/measurements/measurements.db`` finds plate 2 wherever that
        row happens to be. Only when nothing in its path names a plate does
        it fall back to the first row that has no database -- and the
        returned sentence, also shown under the table, NAMES that row. A
        database attached to row 0 in silence is a plate's measurements
        quietly credited to another plate.

        Returns the sentence, so a caller with a console logs the same words
        the user is reading.
        """
        # Stripped, because the table stores text and hands it back stripped:
        # a path that is not equal to its own strip is one this widget could
        # write and then never find again.
        database = os.fspath(path).strip()
        if not database:
            raise ValueError("attach_database needs a path to a database.")
        if row is not None:
            row = int(row)
            if not 0 <= row < self.table.rowCount():
                raise IndexError(
                    f"row {row} is not in a table of {self.table.rowCount()} "
                    "rows.")
            replaced = self._place_database(row, database)
            message = self._describe_row(row, database, "attached to")
            if replaced:
                message += f" It replaced {os.path.basename(replaced)}."
            self.value_changed.emit()
            self._refresh_status(message)
            return message

        already = self._row_of_database(database)
        # After this the path IS somewhere in the table: the side lists are
        # rebuilt from the table itself, so a path that was not already on a
        # row comes back on one of its own.
        self.add_paths_for_side([database], "database")
        index = self._row_of_database(database)
        if already is not None:
            message = self._describe_row(index, database, "is already on")
            self._refresh_status(message)
            return message
        if self._cell(index, self.SIDE_COLUMNS["score"]) or \
                self._cell(index, self.SIDE_COLUMNS["count"]):
            message = self._describe_row(index, database, "paired by filename with")
            self._refresh_status(message)
            return message
        target = self._first_row_without_database(exclude=index)
        if target is None:
            message = (f"{os.path.basename(database)} is on row {index + 1} "
                       "of its own: no plate row is waiting for a database. "
                       "It will pair with a plate when that plate's CSVs "
                       "arrive.")
            self._refresh_status(message)
            return message
        self.table.blockSignals(True)
        self.table.removeRow(index)
        self.table.blockSignals(False)
        # Asked again rather than adjusted by hand: removing a row shifts
        # every index after it, and an off-by-one here attaches a database to
        # the wrong plate, which is the one mistake this column cannot make.
        target = self._first_row_without_database()
        self._place_database(target, database)
        message = self._describe_row(target, database, "attached to")
        message += (" It is the first row with no database: nothing in the "
                    "file's path named a plate.")
        self.value_changed.emit()
        self._refresh_status(message)
        return message

    def missing_databases(self) -> list:
        """Rows whose database is not on disk: ``(row number, plate, path)``.

        Row numbers are 1-based, the way the table numbers them.

        Checked as soon as the path is attached and restated in the status
        line, because the alternative is a run that reads its inputs, fits
        nothing for four minutes, and then fails on a path the panel could
        have flagged the moment the settings were loaded. A settings file is
        routinely written on one machine and run on another, which is the
        case that produces this.
        """
        missing = []
        for index, row in enumerate(self.get_value(), start=1):
            database = row.get("database")
            if database and not os.path.exists(database):
                missing.append((index, row.get("plate") or "", database))
        return missing

    def _row_of_database(self, database: str):
        column = self.SIDE_COLUMNS["database"]
        for index in range(self.table.rowCount()):
            if self._cell(index, column) == database:
                return index
        return None

    def _first_row_without_database(self, exclude=None):
        column = self.SIDE_COLUMNS["database"]
        for index in range(self.table.rowCount()):
            if index == exclude:
                continue
            if not self._cell(index, column):
                return index
        return None

    def _place_database(self, row: int, database: str) -> str:
        """Write ``database`` into ``row`` and pin it there. Returns what it
        replaced, if anything."""
        column = self.SIDE_COLUMNS["database"]
        replaced = self._cell(row, column)
        self.table.blockSignals(True)
        self.table.setItem(row, column, self._database_item(database))
        self.table.blockSignals(False)
        if replaced and replaced != database:
            self._pinned.pop(replaced, None)
        self._pinned[database] = self._anchor_for_row(row)
        return replaced if replaced != database else ""

    @staticmethod
    def _database_item(value: str) -> QTableWidgetItem:
        item = QTableWidgetItem(str(value))
        if value and not os.path.exists(str(value)):
            # Marked, not discarded: the path may be right and the disk
            # merely not mounted yet, and a silently emptied cell is worse
            # than a red one.
            item.setForeground(Qt.red)
            item.setToolTip(f"{value}\n\nThis database is not on disk right "
                            "now, so this plate has no measurements to join.")
        return item

    def _describe_row(self, row: int, database: str, verb: str) -> str:
        plate = self._cell(row, 0)
        label = f"{plate} (row {row + 1})" if plate else f"row {row + 1}"
        return f"{os.path.basename(database)} {verb} {label}."

    def _refresh_status(self, message: str = "") -> None:
        rows = self.get_value()
        parts = [message] if message else []
        attached = [row for row in rows if row.get("database")]
        if rows:
            noun = "row" if len(rows) == 1 else "rows"
            parts.append(f"{len(attached)} of {len(rows)} plate {noun} "
                         "carry a measurements database.")
        missing = self.missing_databases()
        if missing:
            named = "; ".join(
                f"{plate or f'row {number}'}: {path}"
                for number, plate, path in missing)
            parts.append(f"NOT ON DISK — {named}. Fix or clear these before "
                         "the run: they are read after it starts.")
        self.status.setText(" ".join(parts) or self._EMPTY_STATUS)

    # ---------------------------------------------------------- drag / drop

    @staticmethod
    def _dropped(event):
        mime = event.mimeData()
        if not mime.hasUrls():
            return []
        return [url.toLocalFile() for url in mime.urls() if url.isLocalFile()]

    def dragEnterEvent(self, event):  # noqa: N802 - Qt name
        if self._dropped(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):  # noqa: N802 - Qt name
        if self._dropped(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):  # noqa: N802 - Qt name
        """Route each dropped file to the column, and row, it belongs in.

        A drop over the score or count column goes there regardless of what
        the file looks like -- the user aimed it. A drop anywhere else is
        sorted by header, so dropping the whole set at once still fills both
        columns correctly.

        A DATABASE is decided by its extension and never by aim: a ``.db``
        dropped on the score column is a mis-aim, not a request to fit the
        regression on a sqlite file, and a CSV dropped on the database column
        is likewise still a CSV. What aim adds for a database is the ROW --
        dropping it on a plate's row attaches it to THAT plate, which is the
        one thing the token pairing cannot know when the file is called
        ``measurements.db`` like everybody else's.
        """
        paths = self._dropped(event)
        if not paths:
            event.ignore()
            return
        position = event.position().toPoint() if hasattr(event, "position") \
            else event.pos()
        # VIEWPORT coordinates, which is what columnAt and rowAt document
        # themselves to take. Mapping into the table itself instead offsets
        # every answer by the header: one whole row down, and one vertical
        # header's width across. Nobody noticed while only the column was
        # read -- columns are wide -- but a row aimed at is a row missed.
        local = self.table.viewport().mapFrom(self, position)
        inside = self.table.viewport().rect().contains(local)
        column = self.table.columnAt(local.x()) if inside else -1
        row = self.table.rowAt(local.y()) if inside else -1
        aimed = None
        for side, index in self.SIDE_COLUMNS.items():
            if column == index and inside:
                aimed = side
                break
        for path in paths:
            natural = self._side_for_path(path)
            if natural == "database":
                self.attach_database(
                    path, row if aimed == "database" and row >= 0 else None)
            else:
                side = aimed if aimed in ("score", "count") else natural
                self.add_paths_for_side([path], side)
        event.acceptProposedAction()

    def _pick(self, side: str) -> None:
        if side == "database":
            title, filters = ("Add measurements databases",
                              "Databases (*.db *.sqlite *.sqlite3)")
        else:
            title, filters = f"Add {side} CSVs", "Tables (*.csv *.tsv *.txt)"
        paths, _ = QFileDialog.getOpenFileNames(self, title, "", filters)
        if not paths:
            return
        # Through the same seam a drop uses, so the picker cannot pair by the
        # order the file dialog happened to return.
        self.add_paths_for_side(paths, side)

    def _append_row(self, row: dict) -> None:
        index = self.table.rowCount()
        self.table.insertRow(index)
        values = (row.get("plate") or "", row.get("score") or "",
                  row.get("count") or "", row.get("database") or "",
                  row.get("rule") or "resolved at run")
        for column, value in enumerate(values):
            if column == self.SIDE_COLUMNS["database"]:
                item = self._database_item(value)
            else:
                item = QTableWidgetItem(str(value))
            if column == self.RULE_COLUMN:
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(index, column, item)

    def set_value(self, value: Any) -> None:
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        for row in value or []:
            if isinstance(row, dict):
                self._append_row(row)
        self.table.blockSignals(False)
        self._refresh_status()

    def get_value(self) -> list[dict]:
        rows = []
        for index in range(self.table.rowCount()):
            score = self._cell(index, self.SIDE_COLUMNS["score"])
            count = self._cell(index, self.SIDE_COLUMNS["count"])
            database = self._cell(index, self.SIDE_COLUMNS["database"])
            if score or count or database:
                rows.append({"plate": self._cell(index, 0) or None,
                             "score": score or None, "count": count or None,
                             "database": database or None})
        return rows

    def _move(self, offset: int) -> None:
        row = self.table.currentRow()
        target = row + offset
        if row < 0 or not 0 <= target < self.table.rowCount():
            return
        values = self.get_value()
        values[row], values[target] = values[target], values[row]
        self.set_value(values)
        self.table.selectRow(target)
        self.value_changed.emit()

    def _remove(self) -> None:
        rows = sorted({index.row() for index in self.table.selectedIndexes()},
                      reverse=True)
        for row in rows:
            # A removed row takes its database's pin with it, or the next
            # addition would put a file back that the user just deleted.
            self._pinned.pop(self._cell(row, self.SIDE_COLUMNS["database"]),
                             None)
            self.table.removeRow(row)
        if rows:
            self._refresh_status()
            self.value_changed.emit()

#: Extension groups offered in the dialog, by the kind of input a setting wants.
FILE_KIND_FILTERS: dict[str, str] = {
    "table": "Tables (*.csv *.tsv *.txt *.xlsx *.parquet);;CSV (*.csv);;"
             "Excel (*.xlsx);;All files (*)",
    "csv": "CSV (*.csv *.tsv *.txt);;All files (*)",
    "image": "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp *.czi *.lif *.nd2);;"
             "All files (*)",
    "model": "Models (*.pth *.pt *.ckpt *.h5 *.joblib *.pkl);;All files (*)",
    "sequencing": "Reads (*.fastq *.fq *.fastq.gz *.fq.gz);;All files (*)",
    "any": "All files (*)",
}

#: Extensions a dropped *folder* contributes, per kind. A folder dropped on a
#: CSV setting must not add its PNGs.
_KIND_EXTENSIONS: dict[str, tuple[str, ...]] = {
    "table": (".csv", ".tsv", ".txt", ".xlsx", ".parquet"),
    "csv": (".csv", ".tsv", ".txt"),
    "image": (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".czi",
              ".lif", ".nd2"),
    "model": (".pth", ".pt", ".ckpt", ".h5", ".joblib", ".pkl"),
    "sequencing": (".fastq", ".fq", ".gz"),
    "any": (),
}


class FilePathListWidget(QWidget):
    """An ordered, de-duplicated list of input paths with picker and drop.

    ``single=True`` is the same control for a setting that names exactly ONE
    file. It keeps the file dialog and the drop target -- which is the whole
    reason these settings stopped being text boxes -- but the value it holds
    and returns is a plain ``str``, choosing again REPLACES rather than
    appends, and the reorder buttons are gone because one path has no order.

    That distinction is not cosmetic. ``grna_csv``, ``row_csv`` and
    ``column_csv`` are declared ``str`` and go straight to ``pd.read_csv``,
    so rendering them as a list turned a working default into
    ``['/path/to/barcodes_row.csv']`` the moment the screen was opened and
    saved. Every run from that file was then refused by the pre-flight --
    "column_csv=[...] is a list, but str is expected" -- against a value the
    user had never typed and could not correct from the panel that wrote it.
    """

    value_changed = Signal()

    def __init__(
        self,
        value: Any = None,
        *,
        kind: str = "table",
        title: str = "Choose input files",
        allow_folders: bool = True,
        single: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self._kind = kind if kind in FILE_KIND_FILTERS else "any"
        self._title = title
        self._single = bool(single)
        # A folder is not a file, and expanding one into "all the CSVs in
        # here" cannot mean anything for a setting that names one of them.
        self._allow_folders = bool(allow_folders) and not self._single
        self._last_directory = ""

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        self._list = QListWidget(self)
        self._list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._list.setAlternatingRowColors(True)
        self._list.setMinimumHeight(48 if self._single else 96)
        self._list.setUniformItemSizes(True)
        # The list itself must not swallow the drop before the widget sees it.
        self._list.setAcceptDrops(False)
        self._list.setDragDropMode(QAbstractItemView.NoDragDrop)
        outer.addWidget(self._list)

        self._hint = QLabel(self._empty_hint(), self)
        self._hint.setWordWrap(True)
        self._hint.setProperty("role", "hint")
        outer.addWidget(self._hint)

        row = QHBoxLayout()
        row.setSpacing(4)
        self._add_files_button = QPushButton(
            "Choose file…" if self._single else "Add files…", self)
        self._add_files_button.setToolTip(
            "Select the file this setting names. Choosing again replaces it."
            if self._single else
            "Select one or more files. Press again to add more from another "
            "folder — each press appends to the list.")
        self._add_files_button.clicked.connect(self.pick_files)
        row.addWidget(self._add_files_button)

        if self._allow_folders:
            self._add_folder_button = QPushButton("Add folder…", self)
            self._add_folder_button.setToolTip(
                "Add every matching file directly inside a folder.")
            self._add_folder_button.clicked.connect(self.pick_folder)
            row.addWidget(self._add_folder_button)
        else:
            self._add_folder_button = None

        # One path has no order, so the two buttons that reorder the list are
        # not built at all rather than built and left doing nothing.
        if self._single:
            self._up_button = self._down_button = None
        else:
            self._up_button = QPushButton("↑", self)
            self._up_button.setToolTip(
                "Move the selected file earlier in the list")
            self._up_button.setMaximumWidth(30)
            self._up_button.clicked.connect(lambda: self._move_selected(-1))
            row.addWidget(self._up_button)

            self._down_button = QPushButton("↓", self)
            self._down_button.setToolTip(
                "Move the selected file later in the list")
            self._down_button.setMaximumWidth(30)
            self._down_button.clicked.connect(lambda: self._move_selected(1))
            row.addWidget(self._down_button)

        self._remove_button = QPushButton("Remove", self)
        self._remove_button.clicked.connect(self.remove_selected)
        row.addWidget(self._remove_button)

        self._clear_button = QPushButton("Clear", self)
        self._clear_button.clicked.connect(self.clear)
        row.addWidget(self._clear_button)
        row.addStretch(1)
        outer.addLayout(row)

        self.setAcceptDrops(True)
        self.set_value(value)

    # ------------------------------------------------------------------ value

    def set_value(self, value: Any) -> None:
        """Replace the contents. Accepts None, a str, or any iterable.

        A single-file widget keeps only the last of whatever it is given.
        That is what loads a settings file written while these keys were
        wrongly rendered as lists: ``['/x/barcodes_row.csv']`` comes back as
        ``/x/barcodes_row.csv`` rather than carrying the wrong shape forward.
        """
        self._list.clear()
        paths = self._coerce(value)
        for path in (paths[-1:] if self._single else paths):
            self._append(path)
        self._refresh_hint()

    def paths(self) -> List[str]:
        """Every path currently listed, in order -- always a list."""
        return [self._list.item(row).data(Qt.UserRole)
                for row in range(self._list.count())]

    def get_value(self) -> Any:
        """The setting's value: a ``list[str]``, or a ``str`` when single.

        The shape is the SETTING's, not the widget's. A key declared ``str``
        that came back as a one-element list rewrote the user's settings file
        on open and reached ``pd.read_csv`` as a list.
        """
        listed = self.paths()
        if not self._single:
            return listed
        return listed[0] if listed else ""

    # A settings CSV written before this widget existed can hold the literal
    # placeholder 'list of paths'; it is not a path and must not become one.
    _PLACEHOLDERS = {"list of paths", "none", "", "[]"}

    @classmethod
    def _coerce(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, (str, bytes, os.PathLike)):
            value = [value]
        out: List[str] = []
        for item in value:
            if item is None:
                continue
            text = os.fspath(item) if isinstance(item, os.PathLike) else str(item)
            text = text.strip().strip('"').strip("'")
            if text.lower() in cls._PLACEHOLDERS:
                continue
            out.append(text)
        return out

    # ------------------------------------------------------------------- edit

    def add_paths(self, paths: Iterable[Any]) -> int:
        """Append ``paths``, expanding folders. Returns how many were added.

        When the setting names ONE file this REPLACES what is there. A second
        choice is a correction, and a control that appended left the run
        reading a file the user believed they had swapped out.
        """
        incoming = self._coerce(paths)
        if self._single:
            if not incoming:
                return 0
            chosen = os.path.abspath(os.path.expanduser(incoming[-1]))
            if chosen == (self.paths() or [None])[0]:
                return 0
            self._list.clear()
            self._append(chosen)
            self._refresh_hint()
            self.value_changed.emit()
            return 1
        added = 0
        for raw in incoming:
            expanded = os.path.abspath(os.path.expanduser(raw))
            if os.path.isdir(expanded):
                for member in self._folder_members(expanded):
                    added += int(self._append(member))
            else:
                added += int(self._append(expanded))
        if added:
            self._refresh_hint()
            self.value_changed.emit()
        return added

    def _folder_members(self, folder: str) -> List[str]:
        """Matching files one level inside ``folder``, sorted for stable order."""
        extensions = _KIND_EXTENSIONS.get(self._kind, ())
        try:
            names = sorted(os.listdir(folder))
        except OSError:
            return []
        members = []
        for name in names:
            full = os.path.join(folder, name)
            if not os.path.isfile(full):
                continue
            if extensions and not name.lower().endswith(extensions):
                continue
            members.append(full)
        return members

    def _append(self, path: str) -> bool:
        """Add one path unless it is already listed. Returns True if added."""
        resolved = os.path.abspath(os.path.expanduser(str(path)))
        if resolved in set(self.paths()):
            return False
        item = QListWidgetItem(self._display_text(resolved))
        item.setData(Qt.UserRole, resolved)
        if os.path.exists(resolved):
            item.setToolTip(resolved)
        else:
            # Marked, not dropped: a settings file may legitimately be edited
            # on one machine and run on another, and silently discarding the
            # path would leave the user staring at an empty list.
            item.setToolTip(f"{resolved}\n\nThis path does not exist right now.")
            item.setForeground(Qt.red)
        self._list.addItem(item)
        return True

    @staticmethod
    def _display_text(path: str) -> str:
        """Basename plus enough parent to tell four plate CSVs apart."""
        parent = os.path.basename(os.path.dirname(path))
        name = os.path.basename(path)
        return f"{parent}/{name}" if parent else name

    def remove_selected(self) -> None:
        rows = sorted((self._list.row(item) for item in self._list.selectedItems()),
                      reverse=True)
        for row in rows:
            self._list.takeItem(row)
        if rows:
            self._refresh_hint()
            self.value_changed.emit()

    def clear(self) -> None:
        if self._list.count():
            self._list.clear()
            self._refresh_hint()
            self.value_changed.emit()

    def _move_selected(self, offset: int) -> None:
        """Move the single selected row by ``offset``, keeping it selected."""
        items = self._list.selectedItems()
        if len(items) != 1:
            return
        row = self._list.row(items[0])
        target = row + offset
        if not 0 <= target < self._list.count():
            return
        item = self._list.takeItem(row)
        self._list.insertItem(target, item)
        self._list.setCurrentRow(target)
        self.value_changed.emit()

    def _empty_hint(self) -> str:
        if self._single:
            return "Drop one file here, or use Choose file…"
        return "Drop files or folders here, or use Add files…"

    def _refresh_hint(self) -> None:
        count = self._list.count()
        missing = sum(
            1 for row in range(count)
            if not os.path.exists(self._list.item(row).data(Qt.UserRole))
        )
        if not count:
            self._hint.setText(self._empty_hint())
        elif missing:
            self._hint.setText(
                f"{count} file{'s' if count != 1 else ''} selected — "
                f"{missing} not found (shown in red)")
        else:
            self._hint.setText(
                f"{count} file{'s' if count != 1 else ''} selected")

    # -------------------------------------------------------------- drag/drop

    @staticmethod
    def _urls(event) -> List[str]:
        mime = event.mimeData()
        if not mime.hasUrls():
            return []
        return [url.toLocalFile() for url in mime.urls() if url.isLocalFile()]

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if self._urls(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:  # noqa: N802
        if self._urls(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        paths = self._urls(event)
        if not paths:
            event.ignore()
            return
        self.add_paths(paths)
        event.acceptProposedAction()

    # ----------------------------------------------------------------- picker

    def pick_files(self) -> int:
        """Open the file dialog and take what is chosen.

        Multi-select, except when the setting names one file -- a dialog that
        lets you pick four when three of them will be discarded is a control
        lying about what it does.
        """
        if self._single:
            path, _selected = QFileDialog.getOpenFileName(
                self, self._title, self._start_directory(),
                FILE_KIND_FILTERS[self._kind])
            paths = [path] if path else []
        else:
            paths, _selected = QFileDialog.getOpenFileNames(
                self, self._title, self._start_directory(),
                FILE_KIND_FILTERS[self._kind])
        if not paths:
            return 0
        self._last_directory = os.path.dirname(paths[0])
        return self.add_paths(paths)

    def pick_folder(self) -> int:
        folder = QFileDialog.getExistingDirectory(
            self, f"{self._title} — choose a folder", self._start_directory())
        if not folder:
            return 0
        self._last_directory = folder
        return self.add_paths([folder])

    def _start_directory(self) -> str:
        """Reopen where the user last was, or beside the last file added."""
        if self._last_directory and os.path.isdir(self._last_directory):
            return self._last_directory
        values = self.paths()
        if values:
            parent = os.path.dirname(values[-1])
            if os.path.isdir(parent):
                return parent
        return ""


__all__ = ["DATABASE_EXTENSIONS", "FilePathListWidget", "FILE_KIND_FILTERS",
           "PairedFileTableWidget", "is_database_path", "side_for_header",
           "suggest_file_pairs"]

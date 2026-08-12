"""Column/value editor for excluding rows from UMAP input data.

Everything this widget offers the user comes out of a measurements
database, and both reads used to happen on the GUI thread:

* ``discover_columns`` opens every database the ``src`` setting resolves
  to and runs ``sqlite_master`` plus a ``PRAGMA table_info`` per table.
  It is driven from ``SettingsWidgets._refresh_contextual_widgets``, so
  it runs every time ``src`` or ``tables`` is set.
* ``distinct_values`` runs one ``SELECT DISTINCT … LIMIT 501`` per
  ``(database, table)`` pair the chosen column appears in — and the
  columns a user actually excludes on (``plateID``, ``columnID``,
  ``rowID``) hold a handful of distinct values across the whole table,
  so the LIMIT is never reached and SQLite scans every row. Measured on
  a 200 000-row × 8-table measurements.db with a warm page cache:
  **196 ms per column**, and the column combo is editable, so
  ``currentTextChanged`` fires on every keystroke. Eight quick edits
  froze the window for **894 ms** in one unbroken block; ``set_source``
  itself cost 183 ms and a single deliberate column choice 220 ms.
  Threaded, the same three are 29 ms, 5 ms and 2 ms.

So both go through a :class:`~spacr.qt.job_runner.JobRunner` owned by the
editor, and the value reads are additionally debounced. Three rules hold
this together:

* ``_value_cache`` and ``_column_sources`` are GUI-thread state. The
  worker returns plain data and the completion handler — which Qt runs on
  the GUI thread — is the only thing that writes them.
* Schema discovery and value reads get **separate** runners.
  :meth:`JobRunner.cancel` works by generation, so one runner cannot
  supersede a value read without also dropping the schema read that the
  value read depends on.
* A debounce timer coalesces keystrokes, and :meth:`JobRunner.cancel`
  drops whatever an earlier keystroke started, so the last edit wins
  rather than the slowest read.
"""

from __future__ import annotations

import ast
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ...row_exclusions import normalize_row_exclusions

#: How long a keystroke in the editable column combo waits before its
#: values are read. Long enough that typing "columnID" costs one query
#: rather than eight, short enough that a deliberate choice still feels
#: immediate.
DEBOUNCE_MS = 150

#: Stop after this many distinct values. A dropdown is not a way to read
#: 400 000 identifiers, and the LIMIT keeps the scan bounded on the
#: high-cardinality columns (``prcfo``) where it can actually help.
VALUE_LIMIT = 500


def _quote(name: str) -> str:
    """Double-quote a SQL identifier, escaping embedded quotes."""
    return '"' + str(name).replace('"', '""') + '"'


def source_paths(source) -> list[Path]:
    """Return the ``measurements.db`` files ``source`` resolves to.

    Accepts a path, a list of paths, or the repr of a list (which is what
    a settings text field holds after a multi-folder drop), and tolerates
    a run folder, its ``measurements`` subfolder, or the database itself.
    Paths that do not exist are dropped rather than reported: this feeds
    a dropdown, and a half-typed ``src`` is not an error.
    """
    if isinstance(source, str):
        text = source.strip()
        if text.startswith(("[", "(")):
            try:
                source = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                source = text
    values = source if isinstance(source, (list, tuple)) else [source]
    paths: list[Path] = []
    for value in values:
        if not value:
            continue
        path = Path(str(value)).expanduser()
        if path.is_file() and path.suffix.lower() == ".db":
            candidate = path
        elif path.name == "measurements":
            candidate = path / "measurements.db"
        else:
            candidate = path / "measurements" / "measurements.db"
        if candidate.is_file():
            paths.append(candidate)
    return paths


def discover_columns(source, tables=None) -> dict[str, list[tuple[Path, str]]]:
    """Map every column name to the ``(database, table)`` pairs holding it.

    Pure and Qt-free so it can run on a worker thread; the result is
    plain data the GUI thread installs.

    :param source: anything :func:`source_paths` accepts.
    :param tables: restrict to these table names; falsy means all of them.
    """
    wanted = set(tables or ())
    found: dict[str, list[tuple[Path, str]]] = {}
    for db_path in source_paths(source):
        try:
            connection = sqlite3.connect(str(db_path), timeout=30)
        except sqlite3.Error:
            continue
        try:
            available = [
                row[0] for row in connection.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type='table' ORDER BY name")
            ]
            for table in available:
                if wanted and table not in wanted:
                    continue
                for info in connection.execute(
                        f"PRAGMA table_info({_quote(table)})"):
                    found.setdefault(str(info[1]), []).append((db_path, table))
        except sqlite3.Error:
            continue
        finally:
            connection.close()
    return found


def distinct_values(sources: Iterable[tuple[Path, str]], column: str,
                    limit: int = VALUE_LIMIT) -> list[Any]:
    """Return the distinct values of ``column`` across ``sources``, sorted.

    The slow half of this widget, and the reason it has a worker thread.
    Pure and Qt-free — hand it the ``(database, table)`` pairs
    :func:`discover_columns` found and it touches nothing else.

    Every connection is closed in a ``finally``. The version this
    replaces used ``with sqlite3.connect(...) as connection``, which
    commits but does **not** close, so a session of typing in the column
    box leaked one file descriptor onto a multi-hundred-megabyte database
    per keystroke.
    """
    if not column:
        return []
    values: list[Any] = []
    seen: set[str] = set()
    quoted_column = _quote(column)
    for db_path, table in sources:
        try:
            connection = sqlite3.connect(str(db_path), timeout=30)
        except sqlite3.Error:
            continue
        try:
            rows = connection.execute(
                f"SELECT DISTINCT {quoted_column} FROM {_quote(table)} "
                f"WHERE {quoted_column} IS NOT NULL LIMIT {int(limit) + 1}")
            for (value,) in rows:
                key = str(value)
                if key not in seen:
                    seen.add(key)
                    values.append(value)
                if len(values) >= limit:
                    break
        except sqlite3.Error:
            continue
        finally:
            connection.close()
        if len(values) >= limit:
            break
    values.sort(key=lambda value: str(value))
    return values


class _CheckableValueCombo(QComboBox):
    """A compact dropdown that keeps multiple checked values."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setEditable(True)
        self.lineEdit().setReadOnly(True)
        self.lineEdit().setPlaceholderText("Choose values…")
        self.setModel(QStandardItemModel(self))
        self.view().pressed.connect(self._toggle_index)

    def _toggle_index(self, index) -> None:
        item = self.model().itemFromIndex(index)
        state = item.checkState()
        item.setCheckState(
            Qt.Unchecked if state == Qt.Checked else Qt.Checked)
        self._refresh_text()

    def set_options(self, options, selected=()) -> None:
        selected_text = {str(value) for value in selected}
        all_values = list(options)
        existing = {str(value) for value in all_values}
        all_values.extend(
            value for value in selected if str(value) not in existing)
        model = self.model()
        model.clear()
        for value in all_values:
            item = QStandardItem(str(value))
            item.setData(value, Qt.UserRole)
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
            item.setCheckState(
                Qt.Checked if str(value) in selected_text else Qt.Unchecked)
            model.appendRow(item)
        self._refresh_text()

    def checked_values(self) -> list[Any]:
        model = self.model()
        return [
            model.item(row).data(Qt.UserRole)
            for row in range(model.rowCount())
            if model.item(row).checkState() == Qt.Checked
        ]

    def _refresh_text(self) -> None:
        self.lineEdit().setText(
            ", ".join(str(value) for value in self.checked_values()))


class _ExclusionRuleRow(QWidget):
    """One editable ``column is one of values`` rule."""

    column_changed = Signal(str)
    remove_requested = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        self.column = QComboBox(self)
        self.column.setEditable(True)
        self.column.setMinimumWidth(150)
        self.column.setToolTip("Database column whose matching rows are removed.")
        self.column.currentTextChanged.connect(self.column_changed.emit)
        row.addWidget(self.column, 2)

        self.values = _CheckableValueCombo(self)
        self.values.setMinimumWidth(180)
        self.values.setToolTip(
            "Check one or more values. A row matching any checked value is "
            "excluded.")
        row.addWidget(self.values, 3)

        remove = QToolButton(self)
        remove.setText("×")
        remove.setToolTip("Remove this exclusion rule")
        remove.clicked.connect(lambda: self.remove_requested.emit(self))
        row.addWidget(remove)

    def set_columns(self, columns) -> None:
        current = self.column.currentText()
        self.column.blockSignals(True)
        self.column.clear()
        self.column.addItems([str(column) for column in columns])
        if current:
            index = self.column.findText(current)
            if index < 0:
                self.column.addItem(current)
                index = self.column.count() - 1
            self.column.setCurrentIndex(index)
        self.column.blockSignals(False)


class RowExclusionEditor(QWidget):
    """Add one or more UMAP row exclusions by choosing columns and values.

    The database reads behind the two dropdowns run on worker threads;
    see the module docstring for the measured reason. Nothing about the
    widget's public contract changed — :meth:`set_source` still takes a
    source and returns, it just no longer waits for sqlite before it does.

    :param value: initial rules, in any form
        :func:`spacr.row_exclusions.normalize_row_exclusions` accepts.
    :param parent: Qt parent.
    :param threaded: ``False`` runs both reads inline, in the same order,
        emitting the same signals — a caller that must have the values
        the instant :meth:`set_source` returns can ask for it. Keyword
        only, and defaulted, because the shipped call site
        (``SettingsWidgets._widget_for``) passes ``value`` and ``parent``
        and nothing else.
    :param debounce_ms: how long a column edit waits before it is read.
        ``0`` reads on the next event-loop turn without coalescing.
    """

    #: Emitted after a background read has been applied to the widget.
    #: Carries True for a schema read, False for a value read.
    loaded = Signal(bool)

    def __init__(self, value=None, parent=None, *, threaded: bool = True,
                 debounce_ms: int = DEBOUNCE_MS):
        super().__init__(parent)
        from ..job_runner import JobRunner

        self._rows: list[_ExclusionRuleRow] = []
        self._column_sources: dict[str, list[tuple[Path, str]]] = {}
        self._value_cache: dict[str, list[Any]] = {}
        #: row -> the column whose values it is still waiting for.
        self._pending: dict[_ExclusionRuleRow, str] = {}
        self._threaded = bool(threaded)

        # Two runners, not one. `JobRunner.cancel` abandons *everything*
        # that runner has in flight, and superseding a keystroke's value
        # read must not also abandon the schema read that tells us which
        # databases that column even lives in.
        self._schema_jobs = JobRunner(self, threaded=self._threaded,
                                      app_key="exclusion schema")
        self._value_jobs = JobRunner(self, threaded=self._threaded,
                                     app_key="exclusion values")

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(max(0, int(debounce_ms)))
        # A bound method of a GUI-thread QObject, per job_runner's rules.
        self._debounce.timeout.connect(self._run_pending_loads)

        self._outer = QVBoxLayout(self)
        self._outer.setContentsMargins(0, 0, 0, 0)
        self._outer.setSpacing(4)
        self._rows_layout = QVBoxLayout()
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(4)
        self._outer.addLayout(self._rows_layout)

        add = QToolButton(self)
        add.setText("+ Add exclusion")
        add.setToolTip("Exclude values from another database column")
        add.clicked.connect(lambda: self._add_row())
        self._outer.addWidget(add, 0, Qt.AlignLeft)
        self.set_value(value)

    def get_value(self) -> dict[str, list[Any]] | None:
        rules: dict[str, list[Any]] = {}
        for row in self._rows:
            column = row.column.currentText().strip()
            values = row.values.checked_values()
            if column and values:
                rules.setdefault(column, []).extend(values)
        return normalize_row_exclusions(rules) or None

    def set_value(self, value) -> None:
        rules = normalize_row_exclusions(value)
        self._clear_rows()
        if rules:
            for column, values in rules.items():
                self._add_row(column, values)
        else:
            self._add_row()

    def set_source(self, source, tables=None) -> None:
        """Populate column/value choices from a dropped measurements DB.

        Returns as soon as the schema read is dispatched. The dropdowns
        keep whatever they are showing until the worker delivers — a
        list that is half a second stale beats a frozen window, and on
        the first call they are showing nothing anyway.
        """
        # Everything cached describes the *previous* source, and any read
        # still in flight would repopulate it. Drop both.
        self._value_cache.clear()
        self._pending.clear()
        self._debounce.stop()
        self._value_jobs.cancel()
        self._schema_jobs.cancel()
        self._schema_jobs.submit(
            lambda s=source, t=tables: discover_columns(s, t),
            self._apply_columns)

    def _apply_columns(self, found) -> None:
        """Install a worker's schema read. GUI thread only."""
        self._column_sources = dict(found or {})
        columns = list(self._column_sources)
        for row in self._rows:
            row.set_columns(columns)
            self._refresh_values(row)
        self.loaded.emit(True)

    def _add_row(self, column: str = "", values=()) -> None:
        row = _ExclusionRuleRow(self)
        row.remove_requested.connect(self._remove_row)
        row.column_changed.connect(
            lambda _text, r=row: self._refresh_values(r, preserve=False))
        self._rows.append(row)
        self._rows_layout.addWidget(row)
        row.set_columns(self._column_sources)
        if column:
            index = row.column.findText(column)
            if index < 0:
                row.column.addItem(column)
                index = row.column.count() - 1
            row.column.setCurrentIndex(index)
        self._refresh_values(row, selected=values)

    def _remove_row(self, row) -> None:
        if row in self._rows:
            self._rows.remove(row)
        self._pending.pop(row, None)
        row.setParent(None)
        row.deleteLater()
        if not self._rows:
            self._add_row()

    def _clear_rows(self) -> None:
        for row in self._rows:
            self._pending.pop(row, None)
            row.setParent(None)
            row.deleteLater()
        self._rows.clear()

    # -- values ------------------------------------------------------------

    def _refresh_values(self, row, selected=(), preserve: bool = True) -> None:
        """Show ``row``'s values, reading them in the background if needed.

        A cached column is applied here and now — the cache is GUI-thread
        state and reading it costs nothing, so a column the user comes
        back to fills in without a round trip through a thread.
        """
        column = row.column.currentText().strip()
        if preserve and not selected:
            selected = row.values.checked_values()
        cached = self._value_cache.get(column)
        if not column or cached is not None:
            self._pending.pop(row, None)
            row.values.set_options(cached or (), selected)
            return
        # Not read yet. Show the selection alone rather than the previous
        # column's values, which are now wrong, and queue the read.
        row.values.set_options((), selected)
        self._pending[row] = column
        if self._threaded:
            self._debounce.start()
        else:
            # `threaded=False` promises the values are there when the
            # call that asked for them returns, so there is nothing to
            # coalesce and nothing to wait a timer out for.
            self._run_pending_loads()

    def _run_pending_loads(self) -> None:
        """Read every column the queued rows are waiting for, in one job.

        One job for all of them, because the queue is drained after a
        debounce and by then several rows may want values; and because
        ``cancel`` then means exactly "abandon the previous keystroke".
        """
        pending = dict(self._pending)
        self._pending.clear()
        wanted = {column for column in pending.values()
                  if column and column not in self._value_cache}
        if not wanted:
            return
        request = {column: list(self._column_sources.get(column, ()))
                   for column in sorted(wanted)}
        # Supersede: a read started by an earlier keystroke is for a
        # column the user has already moved off. Its thread is asked to
        # stop and its result is dropped on arrival.
        self._value_jobs.cancel()
        self._value_jobs.submit(
            lambda req=request: {column: distinct_values(sources, column)
                                 for column, sources in req.items()},
            self._apply_values)

    def _apply_values(self, payload) -> None:
        """Install a worker's value read. GUI thread only."""
        if not payload:
            return
        self._value_cache.update(payload)
        for row in self._rows:
            column = row.column.currentText().strip()
            if column in payload:
                row.values.set_options(payload[column],
                                       row.values.checked_values())
        self.loaded.emit(False)

    # -- background state --------------------------------------------------

    def is_busy(self) -> bool:
        """True while a read is queued, running, or undelivered."""
        return bool(self._debounce.isActive() or self._pending
                    or self._schema_jobs.is_busy()
                    or self._value_jobs.is_busy())

    def active_jobs(self) -> int:
        """How many worker threads are still winding down."""
        return (self._schema_jobs.active_jobs()
                + self._value_jobs.active_jobs())

    def shutdown(self) -> None:
        """Stop reading and let no worker outlive the widget.

        Public because a host that owns this editor inside a larger
        screen can call it directly — Qt delivers a close event to the
        window, not to every widget inside it, so a screen that wants
        the reads stopped the moment the user navigates away has to say
        so. Idempotent.
        """
        self._debounce.stop()
        self._pending.clear()
        self._schema_jobs.shutdown()
        self._value_jobs.shutdown()

    def closeEvent(self, event):        # noqa: N802 - Qt override
        """Closing mid-read must not leave a thread behind.

        Qt aborts the process when a running QThread is destroyed, and a
        worker that delivers into a widget on its way out is a
        use-after-free. ``JobRunner.shutdown`` handles both.

        Not the only line of defence, deliberately, because it is not
        always reached: this editor is a child inside a settings panel,
        and navigating away from that panel destroys it without any
        close event. What covers that case is the runner itself — the
        QThreads are unparented and retire themselves, and
        ``JobRunner._relay`` catches the ``RuntimeError`` PySide6 raises
        when a worker settles after its runner's C++ half has gone.
        """
        self.shutdown()
        super().closeEvent(event)

    # -- kept for callers that predate the module-level readers ------------

    @staticmethod
    def _source_paths(source) -> list[Path]:
        """Deprecated alias for :func:`source_paths`."""
        return source_paths(source)

    @classmethod
    def _discover_columns(cls, source, tables=None):
        """Deprecated alias for :func:`discover_columns`."""
        return discover_columns(source, tables)

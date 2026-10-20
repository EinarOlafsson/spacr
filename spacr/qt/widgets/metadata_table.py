"""
Editable metadata table for image ingestion.

When a dataset's plate / well / field / channel assignment is *inferred*
— from a container file's internal structure or from a folder layout —
the guess is rarely perfect. This widget shows those inferred rows in an
editable :class:`QTableWidget` so the user can correct wells, split
conditions, relabel channels, etc. before committing, then writes a
``filename_map.csv`` the pipeline consumes.

Rows are the plain dicts produced by :mod:`spacr.qt.ingest_preview`
(``original / plate / well / field / channel / time / canonical``). The
``canonical`` column is recomputed live from the editable columns so the
generated filenames always reflect the user's edits.

:class:`MetadataTablePanel` is the reusable widget (embeddable / testable
without an event loop); :class:`MetadataTableDialog` wraps it with
Apply / Cancel buttons for the drop flow.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..ingest_preview import ROW_COLUMNS, _yokogawa_name, rows_to_mappings
from .sortable_table import install_sorting, table_item
from .. import _after

_EDITABLE = {"plate", "well", "field", "channel", "time"}
_INT_COLS = {"field", "channel", "time"}
_HEADERS = ["Source", "Plate", "Well", "Field", "Channel", "Time", "Filename"]


#: Rows :meth:`MetadataTablePanel._append_later` adds per event-loop turn.
_ROWS_PER_TURN = 25


class MetadataTablePanel(QWidget):
    """An editable grid of ingestion metadata rows.

    :param rows: initial preview rows (see :mod:`spacr.qt.ingest_preview`).
    :param parent: optional Qt parent.
    """

    def __init__(self, rows: Optional[List[Dict[str, Any]]] = None,
                 parent: Optional[QWidget] = None):
        """Build the metadata review table.

        :param rows: the extracted metadata to show; ``None`` starts empty.
        :param parent: parent widget, or ``None``.
        """
        super().__init__(parent)
        self._table = QTableWidget(0, len(ROW_COLUMNS), self)
        install_sorting(self._table)
        self._table.setHorizontalHeaderLabels(_HEADERS)
        self._table.verticalHeader().setVisible(False)
        self._table.setAlternatingRowColors(True)
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)
        for i in range(1, len(ROW_COLUMNS)):
            hdr.setSectionResizeMode(i, QHeaderView.ResizeToContents)

        self._summary = QLabel("", self)
        try:
            from ..theme import active_palette
            self._summary.setStyleSheet(
                f"color: {active_palette()['fg_muted']};")
        except Exception:
            pass

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._summary)
        lay.addWidget(self._table)

        self._guard = False
        self._pending: List[Dict[str, Any]] = []
        self._table.itemChanged.connect(self._on_item_changed)
        self.set_rows(rows or [])

    def set_rows(self, rows: List[Dict[str, Any]]) -> None:
        """Replace the table contents with ``rows``.

        :param rows: one dict per row, keyed by column; missing keys show
            blank.
        """
        self._pending = []
        self._guard = True
        try:
            with self._filling():
                self._table.setRowCount(0)
                for r in rows:
                    self._append_row(r)
        finally:
            self._guard = False
        self._refresh_summary()

    def _append_later(self, rows: List[Dict[str, Any]]) -> None:
        """Add ``rows`` a chunk per event-loop turn, after those shown.

        A dropped plate plans 200 rows, and building all of their cells in
        the turn that opens the table held the GUI thread for that long.
        Anything that reads the table first takes the rest at once
        (:meth:`_flush_pending`), so no reader sees half a plan.

        :param rows: rows to add after the current ones.
        """
        if not rows:
            return
        idle = not self._pending
        self._pending.extend(rows)
        if idle:
            _after(0, self, self._drain_pending)

    def _drain_pending(self) -> None:
        """Add the next chunk of :meth:`_append_later` rows."""
        if not self._pending:
            return
        chunk = self._pending[:_ROWS_PER_TURN]
        del self._pending[:_ROWS_PER_TURN]
        self._add_rows(chunk)
        if self._pending:
            _after(0, self, self._drain_pending)
        else:
            self._refresh_summary()

    def _flush_pending(self) -> None:
        """Add every row :meth:`_append_later` has still to add, now."""
        if self._pending:
            chunk, self._pending = self._pending, []
            self._add_rows(chunk)
            self._refresh_summary()

    def _add_rows(self, rows: List[Dict[str, Any]]) -> None:
        """Append ``rows`` without treating them as edits."""
        self._guard = True
        try:
            with self._filling():
                for r in rows:
                    self._append_row(r)
        finally:
            self._guard = False

    @contextmanager
    def _filling(self):
        """Hold off sorting, column fitting and repaints while rows go in.

        With sorting on and columns sized to their contents, Qt re-sorted and
        re-measured every column for each of the 1,400 cells of a 200-row
        plan: about 130 ms on the GUI thread when a folder was dropped. The
        columns are fitted once, afterwards, to the same contents.
        """
        table = self._table
        header = table.horizontalHeader()
        modes = [header.sectionResizeMode(i) for i in range(header.count())]
        sorting = table.isSortingEnabled()
        updates = table.updatesEnabled()
        table.setUpdatesEnabled(False)
        table.setSortingEnabled(False)
        for i, mode in enumerate(modes):
            if mode == QHeaderView.ResizeToContents:
                header.setSectionResizeMode(i, QHeaderView.Interactive)
        try:
            yield
        finally:
            for i, mode in enumerate(modes):
                header.setSectionResizeMode(i, mode)
            table.setSortingEnabled(sorting)
            table.setUpdatesEnabled(updates)

    def _append_row(self, r: Dict[str, Any]) -> None:
        """Add one metadata row.

        Only the columns a user may correct are editable; the rest are shown
        greyed, because they are derived and typing over one would produce a
        filename that does not follow from the values beside it.

        :param r: the row's values by column key; missing keys render blank.
        """
        row = self._table.rowCount()
        self._table.insertRow(row)
        for col, key in enumerate(ROW_COLUMNS):
            val = r.get(key, "")
            item = table_item("" if val is None else str(val))
            if key in _EDITABLE:
                item.setFlags(item.flags() | Qt.ItemIsEditable)
            else:
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                item.setForeground(Qt.gray)
            if key in _INT_COLS:
                item.setTextAlignment(Qt.AlignCenter)
            self._table.setItem(row, col, item)

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        """Coerce an edited cell and rebuild the filename it feeds.

        Integer columns are clamped to at least 1, and unparseable input reverts
        to 1 rather than being left as text a later stage would choke on. The
        re-entrancy guard is what stops the rewrite from re-triggering this.

        :param item: the edited cell; edits to a read-only column are ignored.
        """
        if self._guard:
            return
        col = item.column()
        key = ROW_COLUMNS[col]
        if key not in _EDITABLE:
            return
        self._guard = True
        try:
            if key in _INT_COLS:
                try:
                    n = max(1, int(float(item.text())))
                except (ValueError, TypeError):
                    n = 1
                item.setText(str(n))
            self._recompute_canonical(item.row())
        finally:
            self._guard = False
        self._refresh_summary()

    def _recompute_canonical(self, row: int) -> None:
        """Rebuild the read-only Filename cell from the editable columns."""
        get = lambda k: (self._table.item(row, ROW_COLUMNS.index(k))
                         or table_item("")).text()
        plate = get("plate") or "plate1"
        well = get("well") or f"{plate}_A01"
        if not well.startswith(plate + "_") and "_" not in well:
            well = f"{plate}_{well}"
        try:
            field = max(1, int(float(get("field") or 1)))
            channel = max(1, int(float(get("channel") or 1)))
            time = max(1, int(float(get("time") or 1)))
        except (ValueError, TypeError):
            field = channel = time = 1
        name = _yokogawa_name(well, time, field, channel)
        cell = self._table.item(row, ROW_COLUMNS.index("canonical"))
        if cell is not None:
            cell.setText(name)

    def rows(self) -> List[Dict[str, Any]]:
        """Return the current (possibly edited) rows as dicts."""
        self._flush_pending()
        out: List[Dict[str, Any]] = []
        for row in range(self._table.rowCount()):
            rec: Dict[str, Any] = {}
            for col, key in enumerate(ROW_COLUMNS):
                item = self._table.item(row, col)
                text = item.text() if item is not None else ""
                if key in _INT_COLS:
                    try:
                        rec[key] = max(1, int(float(text)))
                    except (ValueError, TypeError):
                        rec[key] = 1
                else:
                    rec[key] = text
            out.append(rec)
        return out

    def write_filename_map(self, dst: Any) -> Path:
        """Write the current rows to ``dst`` as a ``filename_map.csv``.

        :param dst: path of the CSV to write, converted to
            :class:`~pathlib.Path`; its parent folders are created.
        :returns: the path written.
        """
        from ..folder_metadata import save_filename_map
        mappings = rows_to_mappings(self.rows())
        return save_filename_map(Path(dst), mappings)

    def _refresh_summary(self) -> None:
        """Restate what the table now holds, above it."""
        from ..ingest_preview import summarize_rows
        self._summary.setText("Review & edit the extracted metadata — "
                              + summarize_rows(self.rows()))


class MetadataTableDialog(QDialog):
    """Modal wrapper around :class:`MetadataTablePanel` for the drop flow.

    :param rows: preview rows to edit.
    :param dst: where the ``filename_map.csv`` is written on Apply.
    :param on_apply: optional callback invoked with the written CSV path.
    :param parent: parent widget; ownership only.
    """

    def __init__(self, rows: List[Dict[str, Any]], dst: Any,
                 on_apply: Optional[Callable[[Path], None]] = None,
                 parent: Optional[QWidget] = None):
        """Wrap the metadata table in a modal review dialog.

        :param rows: the extracted metadata to review.
        :param dst: where ``filename_map.csv`` is written on Apply.
        :param on_apply: called with the written path once it lands.
        :param parent: parent widget, or ``None``.
        """
        super().__init__(parent)
        self.setWindowTitle("Review extracted image metadata")
        self.setModal(True)
        self.resize(760, 460)
        self._dst = Path(dst)
        self._on_apply = on_apply
        self._written: Optional[Path] = None

        self.panel = MetadataTablePanel(rows, self)
        info = QLabel(
            "Edit the plate / well / field / channel / time columns as needed. "
            "The Filename column updates live. Apply writes a filename_map.csv "
            "next to your data — the pipeline uses it to name every extracted "
            "image.", self)
        info.setWordWrap(True)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Cancel, Qt.Horizontal, self)
        self._apply_btn = QPushButton("Apply && write filename_map.csv", self)
        buttons.addButton(self._apply_btn, QDialogButtonBox.AcceptRole)
        buttons.rejected.connect(self.reject)
        self._apply_btn.clicked.connect(self._apply)

        lay = QVBoxLayout(self)
        lay.addWidget(info)
        lay.addWidget(self.panel)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(buttons)
        lay.addLayout(row)

    @property
    def written_path(self) -> Optional[Path]:
        """Path of the CSV written on Apply, or None if cancelled."""
        return self._written

    def _apply(self) -> None:
        """Write ``filename_map.csv`` and close.

        A failure leaves the written path unset rather than raising, so the
        caller sees "nothing was written" instead of an exception out of a
        dialog's button.
        """
        try:
            self._written = self.panel.write_filename_map(self._dst)
            if self._on_apply is not None:
                self._on_apply(self._written)
        except Exception:
            self._written = None
        self.accept()

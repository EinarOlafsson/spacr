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

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from PySide6.QtCore import Qt
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

# Columns the user may edit; "original" and "canonical" are read-only.
_EDITABLE = {"plate", "well", "field", "channel", "time"}
_INT_COLS = {"field", "channel", "time"}
_HEADERS = ["Source", "Plate", "Well", "Field", "Channel", "Time", "Filename"]


class MetadataTablePanel(QWidget):
    """An editable grid of ingestion metadata rows.

    :param rows: initial preview rows (see :mod:`spacr.qt.ingest_preview`).
    :param parent: optional Qt parent.
    """

    def __init__(self, rows: Optional[List[Dict[str, Any]]] = None,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._table = QTableWidget(0, len(ROW_COLUMNS), self)
        self._table.setHorizontalHeaderLabels(_HEADERS)
        self._table.verticalHeader().setVisible(False)
        self._table.setAlternatingRowColors(True)
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)
        for i in range(1, len(ROW_COLUMNS)):
            hdr.setSectionResizeMode(i, QHeaderView.ResizeToContents)

        self._summary = QLabel("", self)
        try:
            from ..theme import PALETTE
            self._summary.setStyleSheet(f"color: {PALETTE['fg_muted']};")
        except Exception:
            pass

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._summary)
        lay.addWidget(self._table)

        self._guard = False  # re-entrancy guard while we rewrite cells
        self._table.itemChanged.connect(self._on_item_changed)
        self.set_rows(rows or [])

    # ------------------------------------------------------------------
    # Population
    # ------------------------------------------------------------------
    def set_rows(self, rows: List[Dict[str, Any]]) -> None:
        """Replace the table contents with ``rows``."""
        self._guard = True
        try:
            self._table.setRowCount(0)
            for r in rows:
                self._append_row(r)
        finally:
            self._guard = False
        self._refresh_summary()

    def _append_row(self, r: Dict[str, Any]) -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)
        for col, key in enumerate(ROW_COLUMNS):
            val = r.get(key, "")
            item = QTableWidgetItem("" if val is None else str(val))
            if key in _EDITABLE:
                item.setFlags(item.flags() | Qt.ItemIsEditable)
            else:
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                item.setForeground(Qt.gray)
            if key in _INT_COLS:
                item.setTextAlignment(Qt.AlignCenter)
            self._table.setItem(row, col, item)

    # ------------------------------------------------------------------
    # Editing
    # ------------------------------------------------------------------
    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if self._guard:
            return
        col = item.column()
        key = ROW_COLUMNS[col]
        if key not in _EDITABLE:
            return
        self._guard = True
        try:
            # Coerce integer columns; revert bad input to 1.
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
                         or QTableWidgetItem("")).text()
        plate = get("plate") or "plate1"
        well = get("well") or f"{plate}_A01"
        # Keep the plate prefix on the well token, matching convert_to_yokogawa.
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

    # ------------------------------------------------------------------
    # Read-back
    # ------------------------------------------------------------------
    def rows(self) -> List[Dict[str, Any]]:
        """Return the current (possibly edited) rows as dicts."""
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

        :returns: the path written.
        """
        from ..folder_metadata import save_filename_map
        mappings = rows_to_mappings(self.rows())
        return save_filename_map(Path(dst), mappings)

    def _refresh_summary(self) -> None:
        from ..ingest_preview import summarize_rows
        self._summary.setText("Review & edit the extracted metadata — "
                              + summarize_rows(self.rows()))


class MetadataTableDialog(QDialog):
    """Modal wrapper around :class:`MetadataTablePanel` for the drop flow.

    :param rows: preview rows to edit.
    :param dst: where the ``filename_map.csv`` is written on Apply.
    :param on_apply: optional callback invoked with the written CSV path.
    """

    def __init__(self, rows: List[Dict[str, Any]], dst: Any,
                 on_apply: Optional[Callable[[Path], None]] = None,
                 parent: Optional[QWidget] = None):
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
        try:
            self._written = self.panel.write_filename_map(self._dst)
            if self._on_apply is not None:
                self._on_apply(self._written)
        except Exception:
            self._written = None
        self.accept()

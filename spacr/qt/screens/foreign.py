"""
Import Project — somebody else's images, masks and measurements, reviewed
column by column before any of it becomes a spaCR database.

:mod:`spacr.foreign` does the work; this screen exists for the one step
that cannot be automated. Their ``Area`` is in µm², spaCR's ``cell_area``
is in px², and the difference between an import that is right and one that
is plausibly wrong by a factor of several hundred is whether a human
looked at the mapping table. So the mapping table is the middle of the
screen, it is **editable**, and every edit re-runs the conflict and unit
checks in place.

Layout::

    ┌──────────────────────────────────────────────────────────────────┐
    │ Images       /data/theirs/images                   [Choose…]     │
    │ Masks   [cell ▾] /data/theirs/cell_masks   [Choose…]  [Add]      │
    │   cell -> /data/theirs/cell_masks                     [Remove]   │
    │ Table        /data/theirs/results.csv              [Choose…]     │
    │ µm per pixel [0.65]   On conflict [Refuse ▾]      [Preview]      │
    ├──────────────────────────────────────────────────────────────────┤
    │ their column   →  target            transform unit_in unit_out   │
    │ Area (um^2)       foreign_area_um2  area      um^2   px^2        │
    │ MeanIntensity     foreign_meanint…  identity                     │
    │ …                                     (target/transform/units    │
    │                                        are editable in place)    │
    ├──────────────────────────────────────────────────────────────────┤
    │ NOT MAPPED (1): Notes                                            │
    │ CONFLICTS (1): [shadows_spacr] 'cell_area' -> 'foreign_cell_area'│
    ├──────────────────────────────────────────────────────────────────┤
    │ /data/imported          [Save mapping…] [Load mapping…] [Import] │
    └──────────────────────────────────────────────────────────────────┘

Design notes:

* **Preview writes nothing.** :func:`spacr.foreign.plan_import` scans,
  pairs and verifies; ``Import`` is a separate press, and it stays
  disabled while the plan has a blocking problem — a target that collides
  with a spaCR feature name, a measurement table with no object-label
  column, a field whose masks do not pair.
* **Editing is instant and local.** A cell edit calls
  :meth:`spacr.foreign.ImportPlan.with_column_maps`, which re-resolves the
  columns without touching the disk, so the unmapped list, the conflict
  list and the Import button update as you type.
* **Off the GUI thread.** Scanning a plate of TIFFs and reading every
  label image takes seconds; the import takes minutes. Both go through
  :func:`spacr.qt.bridge.make_thread`, and the completion handler is
  reached through a *bound method* (:attr:`ForeignScreen._job_settled`)
  rather than a closure, because ``PipelineWorker.finished`` is emitted in
  the worker thread and a closure connected to it would build widget
  children there. Tests pass ``threaded=False``.
* **No modal dialogs on any error path.** Everything lands in the inline
  status label and the report pane. A QMessageBox would hang a headless
  run.
"""
from __future__ import annotations

import os
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, Signal
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
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from ... import foreign as fgn
from ..bridge import make_thread
from ..theme import SPACING, active_palette
from ..widgets import Divider

__all__ = [
    "ForeignScreen",
    "ColumnMapModel",
    "MAP_COLUMNS",
    "OBJECT_CHOICES",
    "CONFLICT_CHOICES",
]


#: The mapping table's columns: ``(ColumnMap field, header, editable)``.
#: ``source`` is their column name and is never editable — renaming it
#: would silently point the mapping at a column that does not exist.
MAP_COLUMNS: Tuple[Tuple[str, str, bool], ...] = (
    ("source", "Their column", False),
    ("target", "spaCR column", True),
    ("transform", "Transform", True),
    ("unit_in", "Unit in", True),
    ("unit_out", "Unit out", True),
    ("note", "Note", True),
)

#: Mask classes, in the order their planes are appended to a merged array.
OBJECT_CHOICES: Tuple[str, ...] = ("cell", "nucleus", "pathogen", "organelle")

#: What to do about a target that collides with a spaCR name.
CONFLICT_CHOICES: Tuple[Tuple[str, str], ...] = (
    ("Refuse the import", "refuse"),
    ("Rename to the foreign prefix", "rename"),
)


class ColumnMapModel(QAbstractTableModel):
    """Editable table over a list of :class:`spacr.foreign.ColumnMap`.

    A model rather than a QTableWidget because a CellProfiler export has
    several hundred columns and building that many QTableWidgetItems
    freezes the window.

    An edit emits :attr:`mapping_edited` *as well as* ``dataChanged``, and
    the screen listens to the former. Hanging the live re-resolve off
    ``dataChanged`` looks equivalent and is not: re-resolving updates the
    per-row status, updating the status refreshes the tooltips, and
    refreshing tooltips emits ``dataChanged`` again — one keystroke
    recursed until the interpreter ran out of stack.
    """

    #: Emitted when a cell's value actually changed. Never on a tooltip
    #: or status refresh, which is the whole point of it existing.
    mapping_edited = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._maps: List[fgn.ColumnMap] = []
        self._status: Dict[str, str] = {}

    # -- content -----------------------------------------------------------

    def set_maps(self, maps: Optional[List[fgn.ColumnMap]]) -> None:
        """Replace the whole mapping."""
        self.beginResetModel()
        self._maps = list(maps or [])
        self.endResetModel()

    def set_status(self, status: Optional[Dict[str, str]]) -> None:
        """Attach a per-source status (``mapped`` / ``uncalibrated`` / …).

        Shown as the row's tooltip, so a user can see *why* a column was
        renamed without leaving the table.
        """
        self._status = dict(status or {})
        if self._maps:
            self.dataChanged.emit(
                self.index(0, 0),
                self.index(len(self._maps) - 1, len(MAP_COLUMNS) - 1),
                [Qt.ToolTipRole])

    def maps(self) -> List[fgn.ColumnMap]:
        """The mapping as it currently stands, including every edit."""
        return list(self._maps)

    def map_at(self, row: int) -> Optional[fgn.ColumnMap]:
        """One row's mapping, or None when ``row`` is out of range."""
        if 0 <= row < len(self._maps):
            return self._maps[row]
        return None

    def row_of(self, source: str) -> int:
        """The row index for a source column name, or -1."""
        for index, mapping in enumerate(self._maps):
            if mapping.source == str(source):
                return index
        return -1

    # -- QAbstractTableModel ----------------------------------------------

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._maps)

    def columnCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(MAP_COLUMNS)

    def flags(self, index):
        base = Qt.ItemIsEnabled | Qt.ItemIsSelectable
        if not index.isValid():
            return base
        if MAP_COLUMNS[index.column()][2]:
            return base | Qt.ItemIsEditable
        return base

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        mapping = self._maps[index.row()]
        key = MAP_COLUMNS[index.column()][0]
        if role in (Qt.DisplayRole, Qt.EditRole):
            return str(getattr(mapping, key) or "")
        if role == Qt.ToolTipRole:
            status = self._status.get(mapping.source, "")
            return (f"{mapping.source}  ->  {mapping.target or '(unmapped)'}"
                    + (f"\n{status}" if status else ""))
        return None

    def setData(self, index, value, role=Qt.EditRole) -> bool:
        if not index.isValid() or role != Qt.EditRole:
            return False
        key, _label, editable = MAP_COLUMNS[index.column()]
        if not editable:
            return False
        current = self._maps[index.row()]
        text = "" if value is None else str(value).strip()
        if str(getattr(current, key) or "") == text:
            return False
        fields = current.to_row()
        fields[key] = text
        self._maps[index.row()] = fgn.ColumnMap.from_row(fields)
        self.dataChanged.emit(index, index, [Qt.DisplayRole, Qt.EditRole])
        self.mapping_edited.emit()
        return True

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return MAP_COLUMNS[section][1]
        return str(section + 1)


class ForeignScreen(QWidget):
    """Pick their files, review the mapping, import, read the summary.

    :param parent: parent widget.
    :param threaded: when False every job runs inline on the calling
        thread. Tests use it so assertions are exact; the app leaves it
        True so a long import does not freeze the window.
    """

    #: Emitted with True/False when a preview or an import settles.
    job_finished = Signal(bool)
    #: Internal relay so the completion handler runs on the GUI thread.
    _job_settled = Signal(bool)
    #: ``(done, total, item)`` — emitted from the worker thread.
    _progress = Signal(int, int, str)

    def __init__(self, parent=None, threaded: bool = True):
        super().__init__(parent)
        self._threaded = bool(threaded)
        self._plan: Optional[fgn.ImportPlan] = None
        self._result: Optional[fgn.ImportResult] = None
        self._masks: Dict[str, str] = {}
        self._busy = False
        self._jobs: List[tuple] = []
        self._pending: List[Tuple[Dict[str, Any], Callable[[Any], None]]] = []
        self._thread = None
        self._worker = None
        self.last_error: str = ""

        self._job_settled.connect(self._on_job_settled)
        self._progress.connect(self._on_progress)
        self._build_ui()
        self._set_status(
            "Choose their images, their mask folder(s) and their measurement "
            "table, then Preview. Nothing is written until you press Import.")
        self._update_controls()

    # -- construction ------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACING["lg"], SPACING["lg"],
                                 SPACING["lg"], SPACING["lg"])
        outer.setSpacing(SPACING["md"])

        title = QLabel("Import Project")
        title.setObjectName("DisplayHeading")
        outer.addWidget(title)

        subtitle = QLabel(
            "Turn somebody else's images, label masks and measurement table "
            "into a spaCR project: Yokogawa-named TIFFs, merged arrays, and a "
            "measurements.db carrying their columns next to spaCR's "
            "plateID/rowID/columnID/fieldID/object_label. Their column names "
            "are mapped onto spaCR's here, in a table you edit — an inferred "
            "mapping is a proposal, never something that runs unread.")
        subtitle.setObjectName("Muted")
        subtitle.setWordWrap(True)
        outer.addWidget(subtitle)

        outer.addWidget(Divider())

        # ── Images ────────────────────────────────────────────────────
        img_row = QHBoxLayout()
        img_row.setSpacing(SPACING["sm"])
        self._images_edit = QLineEdit(self)
        self._images_edit.setPlaceholderText(
            "…/their_images  — the folder of their raw image files")
        self._images_edit.setClearButtonEnabled(True)
        self._images_edit.textChanged.connect(self._on_input_changed)
        self._btn_pick_images = QPushButton("Choose…", self)
        self._btn_pick_images.clicked.connect(self._pick_images)
        img_row.addWidget(QLabel("Images"))
        img_row.addWidget(self._images_edit, 1)
        img_row.addWidget(self._btn_pick_images)
        outer.addLayout(img_row)

        # ── Masks ─────────────────────────────────────────────────────
        mask_row = QHBoxLayout()
        mask_row.setSpacing(SPACING["sm"])
        self._object_box = QComboBox(self)
        for name in OBJECT_CHOICES:
            self._object_box.addItem(name, name)
        self._mask_edit = QLineEdit(self)
        self._mask_edit.setPlaceholderText(
            "…/their_cell_masks  — one folder of label images per class")
        self._mask_edit.setClearButtonEnabled(True)
        self._btn_pick_mask = QPushButton("Choose…", self)
        self._btn_pick_mask.clicked.connect(self._pick_mask)
        self._btn_add_mask = QPushButton("Add", self)
        self._btn_add_mask.clicked.connect(self._add_from_fields)
        self._btn_remove_mask = QPushButton("Remove", self)
        self._btn_remove_mask.clicked.connect(self._remove_selected_mask)
        mask_row.addWidget(QLabel("Masks"))
        mask_row.addWidget(self._object_box)
        mask_row.addWidget(self._mask_edit, 1)
        mask_row.addWidget(self._btn_pick_mask)
        mask_row.addWidget(self._btn_add_mask)
        mask_row.addWidget(self._btn_remove_mask)
        outer.addLayout(mask_row)

        self._mask_list = QListWidget(self)
        self._mask_list.setMaximumHeight(76)
        outer.addWidget(self._mask_list)

        # ── Measurement table ─────────────────────────────────────────
        table_row = QHBoxLayout()
        table_row.setSpacing(SPACING["sm"])
        self._table_edit = QLineEdit(self)
        self._table_edit.setPlaceholderText(
            "…/results.csv  — CSV, TSV, Excel, Parquet or SQLite")
        self._table_edit.setClearButtonEnabled(True)
        self._table_edit.textChanged.connect(self._on_input_changed)
        self._btn_pick_table = QPushButton("Choose…", self)
        self._btn_pick_table.clicked.connect(self._pick_table)
        table_row.addWidget(QLabel("Measurements"))
        table_row.addWidget(self._table_edit, 1)
        table_row.addWidget(self._btn_pick_table)
        outer.addLayout(table_row)

        # ── Options ───────────────────────────────────────────────────
        opt_row = QHBoxLayout()
        opt_row.setSpacing(SPACING["sm"])
        self._scale_edit = QLineEdit(self)
        self._scale_edit.setPlaceholderText(
            "µm per pixel — leave blank if unknown")
        self._scale_edit.textChanged.connect(self._on_scale_changed)
        self._conflict_box = QComboBox(self)
        for label, value in CONFLICT_CHOICES:
            self._conflict_box.addItem(label, value)
        self._conflict_box.currentIndexChanged.connect(self._on_scale_changed)
        self._btn_preview = QPushButton("Preview", self)
        self._btn_preview.clicked.connect(self.preview)
        opt_row.addWidget(QLabel("µm/pixel"))
        opt_row.addWidget(self._scale_edit, 1)
        opt_row.addWidget(QLabel("On name conflict"))
        opt_row.addWidget(self._conflict_box, 1)
        opt_row.addWidget(self._btn_preview)
        outer.addLayout(opt_row)

        # ── The mapping table ─────────────────────────────────────────
        self._model = ColumnMapModel(self)
        self._model.mapping_edited.connect(self._on_mapping_edited)
        self._table = QTableView(self)
        self._table.setModel(self._model)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setAlternatingRowColors(True)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)
        self._table.verticalHeader().setVisible(False)
        outer.addWidget(self._table, 1)

        # ── The report ────────────────────────────────────────────────
        self._report = QPlainTextEdit(self)
        self._report.setReadOnly(True)
        self._report.setMaximumHeight(190)
        self._report.setPlaceholderText(
            "The columns that could not be mapped, the ones that collide with "
            "a spaCR name, the ones with no pixel size to convert them, and "
            "how many measurement rows found an object in the masks.")
        outer.addWidget(self._report)

        # ── Destination + actions ─────────────────────────────────────
        dst_row = QHBoxLayout()
        dst_row.setSpacing(SPACING["sm"])
        self._dst_edit = QLineEdit(self)
        self._dst_edit.setPlaceholderText(
            "…/imported  — a NEW folder; their originals are never touched")
        self._dst_edit.setClearButtonEnabled(True)
        self._btn_pick_dst = QPushButton("Choose…", self)
        self._btn_pick_dst.clicked.connect(self._pick_destination)
        self._btn_save_map = QPushButton("Save mapping…", self)
        self._btn_save_map.clicked.connect(self._pick_save_mapping)
        self._btn_load_map = QPushButton("Load mapping…", self)
        self._btn_load_map.clicked.connect(self._pick_load_mapping)
        self._btn_import = QPushButton("Import", self)
        self._btn_import.setObjectName("PrimaryButton")
        self._btn_import.clicked.connect(self.run_import)
        dst_row.addWidget(QLabel("Destination"))
        dst_row.addWidget(self._dst_edit, 1)
        dst_row.addWidget(self._btn_pick_dst)
        dst_row.addWidget(self._btn_save_map)
        dst_row.addWidget(self._btn_load_map)
        dst_row.addWidget(self._btn_import)
        outer.addLayout(dst_row)

        self._progress_bar = QProgressBar(self)
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(False)
        outer.addWidget(self._progress_bar)

        self._status = QLabel("", self)
        self._status.setObjectName("Muted")
        self._status.setWordWrap(True)
        outer.addWidget(self._status)

    # -- inline reporting --------------------------------------------------

    def _set_status(self, text: str, error: bool = False) -> None:
        """Report inline. Deliberately never a QMessageBox — a modal dialog
        would hang a headless run (and did, in MakeMasksScreen)."""
        self.last_error = text if error else ""
        palette = active_palette()
        colour = palette["error"] if error else palette["fg_muted"]
        self._status.setStyleSheet(f"color: {colour};")
        self._status.setText(text)

    def status_text(self) -> str:
        """Current inline status message."""
        return self._status.text()

    def report_text(self) -> str:
        """Whatever is in the report pane."""
        return self._report.toPlainText()

    def _set_report(self, text: str) -> None:
        self._report.setPlainText(text or "")

    # -- configuration -----------------------------------------------------

    def set_images(self, path: str) -> None:
        """Point the screen at their image folder without opening a dialog."""
        self._images_edit.setText(str(path or ""))
        if path and not self._dst_edit.text().strip():
            normalised = os.path.normpath(str(path))
            self._dst_edit.setText(normalised + "_spacr")

    def images_path(self) -> str:
        """The image folder currently typed in."""
        return self._images_edit.text().strip()

    def add_mask_folder(self, object_type: str, path: str) -> bool:
        """Register a mask folder for one object class.

        :returns: False when the class is unknown or the path is blank,
            with the reason inline — never an exception into a GUI slot.
        """
        name = str(object_type or "").strip()
        folder = str(path or "").strip()
        if name not in OBJECT_CHOICES:
            self._set_status(
                f"{name!r} is not a spaCR mask class; expected one of "
                f"{', '.join(OBJECT_CHOICES)}.", error=True)
            return False
        if not folder:
            self._set_status("Choose a mask folder before adding it.",
                             error=True)
            return False
        self._masks[name] = folder
        self._refresh_mask_list()
        self._on_input_changed()
        return True

    def remove_mask_folder(self, object_type: str) -> bool:
        """Forget one object class's mask folder."""
        if str(object_type) not in self._masks:
            return False
        del self._masks[str(object_type)]
        self._refresh_mask_list()
        self._on_input_changed()
        return True

    def mask_folders(self) -> Dict[str, str]:
        """``{object_type: folder}``, in spaCR's mask-plane order."""
        return {name: self._masks[name] for name in OBJECT_CHOICES
                if name in self._masks}

    def _refresh_mask_list(self) -> None:
        self._mask_list.clear()
        for name, folder in self.mask_folders().items():
            item = QListWidgetItem(f"{name}  →  {folder}")
            item.setData(Qt.UserRole, name)
            self._mask_list.addItem(item)

    def _add_from_fields(self) -> None:
        self.add_mask_folder(str(self._object_box.currentData()),
                             self._mask_edit.text().strip())

    def _remove_selected_mask(self) -> None:
        item = self._mask_list.currentItem()
        if item is None:
            self._set_status("Select a mask folder in the list to remove it.",
                             error=True)
            return
        self.remove_mask_folder(str(item.data(Qt.UserRole)))

    def set_measurements(self, path: str) -> None:
        """Point the screen at their measurement table."""
        self._table_edit.setText(str(path or ""))

    def measurements_path(self) -> str:
        """The measurement table currently typed in."""
        return self._table_edit.text().strip()

    def set_destination(self, path: str) -> None:
        """Set the destination project root."""
        self._dst_edit.setText(str(path or ""))
        self._update_controls()

    def destination_path(self) -> str:
        """The destination currently typed in."""
        return self._dst_edit.text().strip()

    def set_pixel_size(self, value: Any) -> None:
        """Set the µm-per-pixel scale; None or '' means unknown."""
        self._scale_edit.setText("" if value in (None, "") else str(value))

    def pixel_size(self) -> Optional[float]:
        """The scale as a number, or None when blank or unparseable.

        None is a real answer, not a failure: a column that needs a pixel
        size and does not have one is reported uncalibrated rather than
        scaled by 1.0.
        """
        text = self._scale_edit.text().strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    def set_on_conflict(self, value: str) -> None:
        """Choose refuse-or-rename for colliding targets."""
        index = self._conflict_box.findData(value)
        if index < 0:
            raise ValueError(f"Unknown on_conflict: {value!r}")
        self._conflict_box.setCurrentIndex(index)

    def on_conflict(self) -> str:
        """The selected collision policy."""
        return str(self._conflict_box.currentData())

    def _on_input_changed(self, *_args) -> None:
        """Any input change invalidates the plan on screen.

        A mapping table that no longer belongs to the files above it is
        worse than an empty one: it is a table the user believes.
        """
        if self._plan is not None:
            self._plan = None
            self._model.set_maps(None)
            self._set_report("")
            self._set_status("Inputs changed — press Preview again.")
        self._update_controls()

    def _on_scale_changed(self, *_args) -> None:
        """Re-resolve in place when the pixel size or the policy changes.

        No folder is rescanned: :meth:`spacr.foreign.ImportPlan.with_column_maps`
        redoes only the unit arithmetic and the collision checks, which is
        exactly what a new pixel size can change.
        """
        if self._plan is None:
            self._update_controls()
            return
        self._plan = self._plan.with_column_maps(
            self._model.maps(), um_per_px=self.pixel_size(),
            on_conflict=self.on_conflict())
        self._refresh_report()

    # -- pickers -----------------------------------------------------------

    def _pick_images(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose their images")
        if path:
            self.set_images(path)

    def _pick_mask(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose a mask folder")
        if path:
            self._mask_edit.setText(path)

    def _pick_table(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self, "Choose their measurement table", "",
            "Tables (*.csv *.tsv *.txt *.xlsx *.xls *.parquet *.db *.sqlite);;"
            "All files (*)")
        if path:
            self.set_measurements(path)

    def _pick_destination(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose destination")
        if path:
            self.set_destination(path)

    def _pick_save_mapping(self) -> None:
        path, _filter = QFileDialog.getSaveFileName(
            self, "Save the column mapping", fgn.COLUMN_MAP_FILENAME,
            "CSV (*.csv);;All files (*)")
        if path:
            self.save_mapping(path)

    def _pick_load_mapping(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self, "Load a column mapping", "", "CSV (*.csv);;All files (*)")
        if path:
            self.load_mapping(path)

    # -- preview -----------------------------------------------------------

    def preview(self) -> bool:
        """Scan, pair and verify. Writes nothing.

        :returns: True when the job was started (or, unthreaded,
            completed); False when the inputs are unusable, with the
            reason in the inline status label.
        """
        images = self.images_path()
        if not images:
            self._set_status("Choose their image folder first.", error=True)
            return False
        if not os.path.isdir(images):
            self._set_status(f"Not a folder: {images}", error=True)
            return False
        masks = self.mask_folders()
        if not masks:
            self._set_status(
                "Add at least one mask folder — a spaCR project without "
                "object masks has nothing to measure and nothing to crop.",
                error=True)
            return False
        missing = [f"{name}: {folder}" for name, folder in masks.items()
                   if not os.path.isdir(folder)]
        if missing:
            self._set_status(f"Not a folder — {'; '.join(missing)}", error=True)
            return False
        table = self.measurements_path()
        if not table:
            self._set_status("Choose their measurement table first.",
                             error=True)
            return False
        if not os.path.isfile(table):
            self._set_status(f"Not a file: {table}", error=True)
            return False

        scale = self.pixel_size()
        policy = self.on_conflict()

        def _job():
            return fgn.plan_import(images, masks, table, um_per_px=scale,
                                   on_conflict=policy)

        self._set_status(f"Scanning {images} and reading the masks…")
        return self._run_job(_job, self._on_plan_ready)

    def _on_plan_ready(self, plan: Optional[fgn.ImportPlan]) -> None:
        """Show the plan and its inferred mapping. Always on the GUI thread."""
        self._plan = plan
        if plan is None:
            self._model.set_maps(None)
            self._set_report("")
            self._set_status("The preview produced no plan.", error=True)
            self._update_controls()
            return
        self._model.set_maps(plan.column_maps)
        self._refresh_report()

    def _refresh_report(self) -> None:
        """Redraw the report pane and the status line from the current plan."""
        plan = self._plan
        if plan is None:
            self._set_report("")
            self._update_controls()
            return
        self._model.set_status(
            {r.source: f"{r.status}: {r.reason}" if r.reason else r.status
             for r in plan.resolved})
        self._set_report(fgn.format_plan(plan))
        blocking = len(plan.blocking_conflicts)
        if blocking or plan.errors or not plan.images.ok:
            self._set_status(
                f"{blocking} column conflict(s) and "
                f"{len(plan.errors) + len(plan.images.errors)} other blocking "
                f"problem(s) — nothing can be imported until they are fixed. "
                f"See the report below.", error=True)
        else:
            self._set_status(
                f"{len(plan.stems)} field(s) ready. "
                f"{plan.join.rows_matched}/{plan.join.rows_total} measurement "
                f"row(s) matched an object; {len(plan.unmapped)} column(s) "
                f"unmapped, {len(plan.uncalibrated)} uncalibrated. Nothing "
                f"has been written yet.")
        self._update_controls()

    def plan(self) -> Optional[fgn.ImportPlan]:
        """The plan currently on screen, or None."""
        return self._plan

    def result(self) -> Optional[fgn.ImportResult]:
        """The result of the last import, or None."""
        return self._result

    def column_maps(self) -> List[fgn.ColumnMap]:
        """The mapping as edited — exactly what Import would apply."""
        return self._model.maps()

    def mapping_row_count(self) -> int:
        """Rows in the mapping table."""
        return self._model.rowCount()

    def set_mapping_value(self, row: int, key: str, value: str) -> bool:
        """Edit one cell the way the delegate would.

        The screen's own edit path, exposed so a test drives the same code
        an editor widget does rather than reaching into the model.
        """
        columns = [name for name, _label, _editable in MAP_COLUMNS]
        if key not in columns:
            return False
        return self._model.setData(self._model.index(row, columns.index(key)),
                                   value, Qt.EditRole)

    def unmapped_columns(self) -> List[str]:
        """Their columns with no mapping, by name."""
        return list(self._plan.unmapped) if self._plan is not None else []

    def conflict_lines(self) -> List[str]:
        """One line per conflict, blocking or not."""
        if self._plan is None:
            return []
        return [str(c) for c in self._plan.conflicts]

    def _on_mapping_edited(self, *_args) -> None:
        """A cell changed: re-resolve without touching the disk."""
        if self._plan is None:
            return
        self._plan = self._plan.with_column_maps(
            self._model.maps(), um_per_px=self.pixel_size(),
            on_conflict=self.on_conflict())
        self._refresh_report()

    # -- the mapping file --------------------------------------------------

    def save_mapping(self, path: str) -> bool:
        """Write the mapping on screen to a reviewable CSV."""
        if not self._model.rowCount():
            self._set_status("There is no mapping to save — press Preview "
                             "first.", error=True)
            return False
        try:
            written = fgn.save_column_map(self._model.maps(), str(path))
        except Exception as exc:
            self._set_status(f"Could not save the mapping: {exc}", error=True)
            return False
        self._set_status(f"Mapping saved to {written}. Edit it and load it "
                         f"back, or hand it to someone else to check.")
        return True

    def load_mapping(self, path: str) -> bool:
        """Read a mapping back and apply it to the plan on screen."""
        try:
            maps = fgn.load_column_map(str(path))
        except Exception as exc:
            self._set_status(f"Could not load the mapping: {exc}", error=True)
            return False
        self._model.set_maps(maps)
        if self._plan is not None:
            self._plan = self._plan.with_column_maps(
                maps, um_per_px=self.pixel_size(),
                on_conflict=self.on_conflict())
            self._refresh_report()
        else:
            self._set_status(f"Loaded {len(maps)} mapping row(s) from {path}. "
                             f"Press Preview to check them against the data.")
        self._update_controls()
        return True

    # -- import ------------------------------------------------------------

    def run_import(self) -> bool:
        """Build the project. Off the GUI thread unless ``threaded=False``.

        :returns: True when the job was started, False when it was refused
            — with the reason inline.
        """
        if self._plan is None:
            self._set_status("Press Preview first — there is nothing to "
                             "import yet.", error=True)
            return False
        if not self._plan.ok:
            self._set_status(
                "This plan has blocking problems; fix them and preview "
                "again. Nothing was written.", error=True)
            return False
        dst = self.destination_path()
        if not dst:
            self._set_status("Choose a destination folder first.", error=True)
            return False

        plan = self._plan
        emit = self._progress.emit

        def _job():
            return fgn.run_import(plan, dst, progress=emit)

        self._progress_bar.setVisible(True)
        self._progress_bar.setRange(0, 7)
        self._progress_bar.setValue(0)
        self._set_status(f"Importing {len(plan.stems)} field(s) into {dst}…")
        return self._run_job(_job, self._on_result_ready)

    def _on_progress(self, done: int, total: int, item: str) -> None:
        """Progress from the worker thread. Always on the GUI thread."""
        self._progress_bar.setRange(0, max(int(total), 1))
        self._progress_bar.setValue(int(done))
        self._set_status(f"Importing {done}/{total} — {item}")

    def _on_result_ready(self, result: Optional[fgn.ImportResult]) -> None:
        """Show the import summary. Always on the GUI thread."""
        self._result = result
        self._progress_bar.setVisible(False)
        if result is None:
            self._set_status("The import produced no result.", error=True)
            self._update_controls()
            return
        self._set_report(result.summary())
        if result.is_complete:
            self._set_status(
                f"Imported {result.n_fields} field(s) and {result.n_rows} "
                f"measurement row(s) into {result.dst}.")
        else:
            self._set_status(
                f"Imported {result.n_fields} field(s), but the run is "
                f"INCOMPLETE — see the summary. The database is stamped.",
                error=True)
        self._update_controls()

    # -- controls ----------------------------------------------------------

    def _update_controls(self) -> None:
        idle = not self._busy
        for widget in (self._btn_pick_images, self._btn_pick_mask,
                       self._btn_add_mask, self._btn_remove_mask,
                       self._btn_pick_table, self._btn_pick_dst,
                       self._btn_preview, self._images_edit, self._mask_edit,
                       self._table_edit, self._dst_edit, self._scale_edit,
                       self._object_box, self._conflict_box, self._table,
                       self._mask_list):
            widget.setEnabled(idle)
        has_map = self._model.rowCount() > 0
        self._btn_save_map.setEnabled(idle and has_map)
        self._btn_load_map.setEnabled(idle)
        self._btn_import.setEnabled(
            idle and self._plan is not None and self._plan.ok)

    def can_import(self) -> bool:
        """True when the Import button is live."""
        return self._btn_import.isEnabled()

    # -- job plumbing ------------------------------------------------------

    def _run_job(self, fn: Callable[[], Any],
                 on_done: Callable[[Any], None]) -> bool:
        """Run ``fn`` off the GUI thread and hand its result to ``on_done``.

        The same idiom as ``PlateViewScreen._run_job`` and
        ``ConvertScreen._run_job``, and for the same reason:
        ``PipelineWorker.finished`` is emitted *in the worker thread*, and
        PySide6 invokes a plain closure connected to it directly, on that
        thread. The completion handlers here reset a table model and fill
        a QPlainTextEdit, and building a QTextDocument's children off the
        GUI thread is undefined behaviour. So ``finished`` is chained
        through :attr:`_job_settled` into a *bound method* of this widget,
        which has GUI-thread affinity — Qt then queues the call.

        With ``threaded=False`` the call runs inline and the same signals
        fire, so both paths behave identically from outside.
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
        thread, worker = make_thread(partial(self._capture, fn), box)
        # Strong references: PySide6 will not keep the worker alive through
        # the started→run connection alone, and a QThread garbage-collected
        # while still running takes the process down with it. The worker is
        # deliberately never deleteLater'd — see bridge.make_thread.
        self._jobs.append((thread, worker))
        self._thread, self._worker = thread, worker
        self._pending.append((box, on_done))
        worker.error.connect(self._on_worker_error_text)
        worker.finished.connect(self._job_settled)
        thread.finished.connect(lambda t=thread: self._retire_job(t))
        self._busy = True
        self._update_controls()
        thread.start()
        return True

    @staticmethod
    def _capture(fn: Callable[[], Any], payload: Dict[str, Any]) -> None:
        """Run ``fn`` in the worker thread and stash its result in ``payload``.

        A named method rather than a closure: this body executes on a
        QThread, where coverage cannot see it, and a nested function would
        be untestable except by running the thread. This one can be called
        directly.
        """
        payload["result"] = fn()

    def _on_job_settled(self, ok: bool) -> None:
        """Finish the oldest in-flight job. Always on the GUI thread."""
        self._busy = False
        box, on_done = self._pending.pop(0) if self._pending else ({}, None)
        ok = bool(ok)
        if ok and on_done is not None:
            try:
                on_done(box.get("result"))
            except Exception as e:
                self._on_job_error(e)
                ok = False
        self._update_controls()
        self.job_finished.emit(ok)

    def _retire_job(self, thread) -> None:
        """Release *this* job's refs once its own event loop has exited."""
        self._jobs = [(t, w) for (t, w) in self._jobs if t is not thread]
        if self._thread is thread:
            self._thread = None
            self._worker = None

    def active_jobs(self) -> int:
        """How many worker threads are still winding down."""
        return len(self._jobs)

    def is_busy(self) -> bool:
        """True while a preview or an import is in flight."""
        return self._busy

    def _on_job_error(self, exc: Exception) -> None:
        self._busy = False
        self._progress_bar.setVisible(False)
        self._set_status(str(exc) or exc.__class__.__name__, error=True)

    def _on_worker_error_text(self, text: str) -> None:
        line = (text or "").strip().splitlines()[-1] if text else "unknown error"
        self._busy = False
        self._progress_bar.setVisible(False)
        self._set_status(f"Import failed: {line}", error=True)

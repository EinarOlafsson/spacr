"""
Import Images — a pile of files from any microscope, read before it is written.

:mod:`spacr.image_import` does the work; this screen exists for the step the
instruction that asked for it calls "the single most important requirement":
**showing the proposal**. spaCR's import path before this asked the user to
pick a filename convention from a closed list — ``cellvoyager``, ``cq1``, or
write your own regular expression — and gave them no way to see whether it
worked until masks came out wrong. Of ten real acquisition layouts, two
parsed and eight recovered nothing.

So the middle of this screen is the parse, one row per file, **with the
filename beside the fields it was parsed into**. A field read as a channel is
visible there in a second and invisible everywhere else.

Layout::

    ┌──────────────────────────────────────────────────────────────────┐
    │ Images   /data/raw                                  [Choose…]    │
    │ Sample [400] ☑ Read inside each file                [Scan]       │
    ├──────────────────────────────────────────────────────────────────┤
    │ file                     plate   well  field channel z   t       │
    │ A01/fld1_DAPI.tif        plate1  A01   1     1       1   1       │
    │ A01/fld1_GFP.tif         plate1  A01   1     2       1   1       │
    ├──────────────────────────────────────────────────────────────────┤
    │ token  value   is channel   (editable — one answer per value)    │
    │ 0      DAPI    1                                                 │
    │ 0      GFP     2                                                 │
    ├──────────────────────────────────────────────────────────────────┤
    │ 8 files, 4 axes resolved …  ! what is still unanswered           │
    ├──────────────────────────────────────────────────────────────────┤
    │ /data/raw_spacr  plate1  ☑link ☐tiles  [Save…][Load…][Import]    │
    └──────────────────────────────────────────────────────────────────┘

Design notes:

* **Scan writes nothing.** :func:`spacr.image_import.plan_import` walks a
  sample of the tree and reads each file's own axis metadata; ``Import`` is a
  separate press, and it stays disabled while the plan has a problem —
  because every problem the plan states is a way for the result to be quietly
  wrong, and the import is the irreversible half.
* **The proposal table is not editable, and the answer table is.** A wrong
  guess is never wrong for one file: it is wrong for a token position, in
  every name that has one. So the correction is made where the mistake lives,
  one answer fixing every file that shares it, rather than by retyping eight
  thousand rows. That is also the only correction
  :meth:`spacr.image_import.ImportPlan.with_mapping` can apply without
  re-walking the disk, which is what makes editing instant.
* **Everything the plan could not place is on screen**, with its values, and
  counted. Silence about unparsed files is how a plate imports with a third
  of its fields missing.
* **Off the GUI thread.** A scan reads every file's header; an import of a
  400-plate archive takes minutes. Both go through
  :func:`spacr.qt.bridge.make_thread`, and the completion handler is reached
  through a *bound method* (:attr:`ImageImportScreen._job_settled`) rather
  than a closure, because ``PipelineWorker.finished`` is emitted in the
  worker thread and a closure connected to it would build widget children
  there. Tests pass ``threaded=False``.
* **No modal dialogs on any error path.** Everything lands in the inline
  status label and the report pane, as in :mod:`spacr.qt.screens.foreign` —
  a QMessageBox would hang a headless run.
"""
from __future__ import annotations

import os
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

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
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from ... import image_import as imp
from ..bridge import make_thread
from ..theme import SPACING, active_palette
from ..widgets import Divider
from ..widgets.sortable_table import install_sorting

__all__ = [
    "ANSWER_COLUMNS",
    "TILE_POLICIES",
    "AnswerModel",
    "ImageImportScreen",
    "ProposalModel",
]

#: What to do with a field that arrives as several tiles, as
#: ``(label, value)``. STITCHING IS FIRST AND IS THE DEFAULT, decided by the
#: maintainer on 2026-09-02: "tiles be stitched at import with the option to
#: not stitch but stitch by default." The other two are the answers that
#: existed before there was a stitcher, and each loses something a user
#: should have to choose: keeping tiles as fields discards the fact that they
#: are one field, and skipping them discards the images.
TILE_POLICIES: Tuple[Tuple[str, str], ...] = (
    ("Stitch into one field", "stitch"),
    ("Keep each tile as a field", "fields"),
    ("Skip tiled images", "skip"),
)

#: The answer table's columns. Only the last is editable: a token's POSITION
#: and the VALUE seen there are facts about the folder, and letting either be
#: typed over would point an answer at something that is not in the names.
ANSWER_COLUMNS: Tuple[Tuple[str, bool], ...] = (
    ("Token", False),
    ("Value", False),
    ("Is channel", True),
)


class ProposalModel(QAbstractTableModel):
    """Read-only table over :meth:`spacr.image_import.ImportPlan.rows`.

    A model rather than a QTableWidget because a plate is tens of thousands
    of files and building that many QTableWidgetItems freezes the window.

    Its header and its cells both come from the plan, so this screen and the
    text table :meth:`spacr.image_import.ImportPlan.table` prints cannot
    disagree about what was inferred — one is the other with padding.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._header: List[str] = []
        self._rows: List[List[str]] = []

    def set_plan(self, plan: Optional["imp.ImportPlan"]) -> None:
        """Show ``plan``'s parse, or clear the table for None."""
        self.beginResetModel()
        if plan is None:
            self._header, self._rows = [], []
        else:
            self._header = plan.columns()
            self._rows = plan.rows()
        self.endResetModel()

    def headers(self) -> List[str]:
        """The columns currently shown — the axes this folder actually has."""
        return list(self._header)

    def row(self, index: int) -> List[str]:
        """One row's cells, or an empty list when ``index`` is out of range."""
        return list(self._rows[index]) if 0 <= index < len(self._rows) else []

    def value_at(self, index: int, column: str) -> str:
        """One cell, addressed by column NAME.

        The columns depend on the folder — a tiled tree has a ``tile`` column
        and a flat one does not — so a caller that wants the channel must ask
        for "channel" rather than for column 4.
        """
        if column not in self._header:
            return ""
        return self.row(index)[self._header.index(column)] if self.row(index) else ""

    # -- QAbstractTableModel ----------------------------------------------

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._header)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        if role in (Qt.DisplayRole, Qt.EditRole):
            return self._rows[index.row()][index.column()]
        if role == Qt.ToolTipRole:
            cells = self._rows[index.row()]
            return "\n".join(f"{name}: {value or '—'}"
                             for name, value in zip(self._header, cells))
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return self._header[section]
        return str(section + 1)


class AnswerModel(QAbstractTableModel):
    """Editable table of the tokens inference could not name.

    ONE ROW PER VALUE, not per token: the question "what does ``DAPI`` mean"
    has a different answer from "what does ``GFP`` mean", and a single cell
    holding both would have to be parsed back out of a string the user typed.

    An edit emits :attr:`answers_edited` *as well as* ``dataChanged``, and the
    screen listens to the former — re-resolving the plan refreshes this
    table's tooltips, and a live re-resolve hung off ``dataChanged`` would
    recurse.
    """

    #: Emitted when a cell's value actually changed.
    answers_edited = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        #: ``(token position, value, answer as typed)``.
        self._rows: List[List[str]] = []

    def set_plan(self, plan: Optional["imp.ImportPlan"]) -> None:
        """Ask, for every token position the names could not place, what its
        values mean — carrying forward any answer already given."""
        self.beginResetModel()
        self._rows = []
        if plan is not None:
            answered = plan.mapping
            for position, values in sorted(plan.layout.unplaced.items()):
                for value in sorted(set(values)):
                    given = answered.get(position, {}).get(value, "")
                    self._rows.append([str(position), str(value),
                                       "" if given == "" else str(given)])
        self.endResetModel()

    def mapping(self) -> Dict[int, Dict[str, int]]:
        """The answers as :meth:`spacr.image_import.ImportPlan.with_mapping`
        wants them, skipping every row still blank or not yet a number.

        A HALF-TYPED ANSWER IS NOT AN ANSWER. ``"1"`` arrives one keystroke
        after ``""``, and treating a blank or a stray letter as zero would
        write a channel nobody asked for.
        """
        out: Dict[int, Dict[str, int]] = {}
        for position, value, answer in self._rows:
            text = str(answer).strip()
            if not text.isdigit():
                continue
            out.setdefault(int(position), {})[value] = int(text)
        return out

    def answers(self) -> List[List[str]]:
        """Every row as it stands, answered or not."""
        return [list(row) for row in self._rows]

    def set_answer(self, row: int, text: str) -> bool:
        """Type into one row's answer cell, the way the editor would."""
        return self.setData(self.index(row, 2), text, Qt.EditRole)

    # -- QAbstractTableModel ----------------------------------------------

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(ANSWER_COLUMNS)

    def flags(self, index):
        base = Qt.ItemIsEnabled | Qt.ItemIsSelectable
        if index.isValid() and ANSWER_COLUMNS[index.column()][1]:
            return base | Qt.ItemIsEditable
        return base

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        if role in (Qt.DisplayRole, Qt.EditRole):
            return self._rows[index.row()][index.column()]
        if role == Qt.ToolTipRole:
            _position, value, _answer = self._rows[index.row()]
            return (f"{value!r} varies across the folder and no convention "
                    f"says what it is. Give it a channel number, or the "
                    f"images that differ only by it cannot be told apart.")
        return None

    def setData(self, index, value, role=Qt.EditRole) -> bool:
        if not index.isValid() or role != Qt.EditRole:
            return False
        if not ANSWER_COLUMNS[index.column()][1]:
            return False
        text = "" if value is None else str(value).strip()
        if self._rows[index.row()][index.column()] == text:
            return False
        self._rows[index.row()][index.column()] = text
        self.dataChanged.emit(index, index, [Qt.DisplayRole, Qt.EditRole])
        self.answers_edited.emit()
        return True

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return ANSWER_COLUMNS[section][0]
        return str(section + 1)


class ImageImportScreen(QWidget):
    """Choose a folder, read the proposal, answer what it asks, import.

    :param parent: parent widget.
    :param threaded: when False every job runs inline on the calling thread.
        Tests use it so assertions are exact; the app leaves it True so a
        long scan does not freeze the window.
    """

    #: Emitted with True/False when a scan or an import settles.
    job_finished = Signal(bool)
    #: Internal relay so the completion handler runs on the GUI thread.
    _job_settled = Signal(bool)

    def __init__(self, parent=None, threaded: bool = True):
        super().__init__(parent)
        self._threaded = bool(threaded)
        self._plan: Optional["imp.ImportPlan"] = None
        self._result: Optional["imp.ImportResult"] = None
        self._busy = False
        self._jobs: List[tuple] = []
        self._pending: List[Tuple[Dict[str, Any], Callable[[Any], None]]] = []
        self._thread = None
        self._worker = None
        self.last_error: str = ""

        self._job_settled.connect(self._on_job_settled)
        self._build_ui()
        # A DROPPED FOLDER IS THE GESTURE THIS MODULE IS FOR. The handler
        # also takes a saved plan, so last week's answers arrive the same
        # way this week's images do.
        from ..dnd import install_dropzone
        from ..dnd_handlers import get_handler
        install_dropzone(self, get_handler("import_images"), self)
        self._set_status(
            "Choose the folder your images are in and press Scan. spaCR works "
            "out the naming from the folder itself — nothing is written until "
            "you press Import.")
        self._update_controls()

    # -- construction ------------------------------------------------------

    def _build_ui(self) -> None:
        # ITS OWN REGISTRY KEY -- what `install_folds_on` and the drop
        # handlers dispatch on.
        self.app_key = "import_images"
        # IMPORTED HERE, NOT AT MODULE LEVEL: `app_screen` imports the screen
        # registry, which reaches this module, and a top-level import would
        # close that circle at startup rather than at first build.
        from .app_screen import ModuleHeader

        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACING["lg"], SPACING["lg"],
                                 SPACING["lg"], SPACING["lg"])
        outer.setSpacing(SPACING["md"])

        self._header = ModuleHeader(
            "Import Images",
            description="Read a folder of images from any microscope into a "
                        "spaCR project",
            app_key="import_images")
        outer.addWidget(self._header)

        subtitle = QLabel(
            "Point spaCR at your images however they are named and arranged — "
            "Yokogawa, CQ1, Opera Phenix, ImageXpress, OME-TIFFs, a folder per "
            "well or a folder per dye. The naming is worked out from what "
            "varies across the folder rather than chosen from a list, and the "
            "parse is shown below, one row per file, before anything is "
            "written.")
        subtitle.setObjectName("Muted")
        subtitle.setWordWrap(True)
        outer.addWidget(subtitle)

        outer.addWidget(Divider())

        # ── Source ────────────────────────────────────────────────────
        src_row = QHBoxLayout()
        src_row.setSpacing(SPACING["sm"])
        self._root_edit = QLineEdit(self)
        self._root_edit.setPlaceholderText(
            "…/raw_images  — the folder your microscope wrote, however deep")
        self._root_edit.setClearButtonEnabled(True)
        self._root_edit.textChanged.connect(self._on_input_changed)
        self._btn_pick_root = QPushButton("Choose…", self)
        self._btn_pick_root.clicked.connect(self._pick_root)
        src_row.addWidget(QLabel("Images"))
        src_row.addWidget(self._root_edit, 1)
        src_row.addWidget(self._btn_pick_root)
        outer.addLayout(src_row)

        # ── Scan options ──────────────────────────────────────────────
        opt_row = QHBoxLayout()
        opt_row.setSpacing(SPACING["sm"])
        self._sample_box = QSpinBox(self)
        self._sample_box.setRange(10, 1000000)
        self._sample_box.setValue(400)
        self._sample_box.setToolTip(
            "How many files to read before deciding what the names mean. A "
            "sample, so a 400-plate archive is as quick to inspect as one "
            "plate: the convention is the same in the first hundred files as "
            "in the last hundred thousand.")
        self._inside_box = QCheckBox("Read inside each file", self)
        self._inside_box.setChecked(True)
        self._inside_box.setToolTip(
            "Open each file for its own axis metadata. A channel that lives "
            "as a page INSIDE an OME-TIFF is invisible without this, and it "
            "is the only thing that can tell a Z-stack from a timelapse when "
            "their filenames are identical. Turn it off for a fast first "
            "look at a very large archive.")
        self._btn_scan = QPushButton("Scan", self)
        self._btn_scan.clicked.connect(self.scan)
        opt_row.addWidget(QLabel("Sample"))
        opt_row.addWidget(self._sample_box)
        opt_row.addWidget(self._inside_box)
        opt_row.addStretch(1)
        opt_row.addWidget(self._btn_scan)
        outer.addLayout(opt_row)

        # ── The proposal ──────────────────────────────────────────────
        self._model = ProposalModel(self)
        self._table = QTableView(self)
        self._table.setModel(self._model)
        # After setModel: the helper puts a sorting proxy over the model,
        # which replaces the view's model and its selection model.
        install_sorting(self._table)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setAlternatingRowColors(True)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)
        self._table.verticalHeader().setVisible(False)
        outer.addWidget(self._table, 1)

        # ── The questions ─────────────────────────────────────────────
        self._answers = AnswerModel(self)
        self._answers.answers_edited.connect(self._on_answer_edited)
        self._answer_table = QTableView(self)
        self._answer_table.setModel(self._answers)
        install_sorting(self._answer_table)
        self._answer_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._answer_table.setMaximumHeight(120)
        self._answer_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)
        self._answer_table.verticalHeader().setVisible(False)
        outer.addWidget(self._answer_table)

        # ── The report ────────────────────────────────────────────────
        self._report = QPlainTextEdit(self)
        self._report.setReadOnly(True)
        self._report.setMaximumHeight(170)
        self._report.setPlaceholderText(
            "How many files were read, how many distinct wells, fields and "
            "channels they hold, what could not be placed, and what was named "
            "unlike the rest and therefore not interpreted.")
        outer.addWidget(self._report)

        # ── Destination + actions ─────────────────────────────────────
        dst_row = QHBoxLayout()
        dst_row.setSpacing(SPACING["sm"])
        self._dst_edit = QLineEdit(self)
        self._dst_edit.setPlaceholderText(
            "…/raw_images_spacr  — a NEW folder; your originals are linked, "
            "never moved")
        self._dst_edit.setClearButtonEnabled(True)
        self._btn_pick_dst = QPushButton("Choose…", self)
        self._btn_pick_dst.clicked.connect(self._pick_destination)
        self._plate_edit = QLineEdit("plate1", self)
        self._plate_edit.setMaximumWidth(110)
        self._plate_edit.setToolTip(
            "The plate name written into every filename. spaCR groups by it, "
            "so two acquisitions imported under one name are one plate.")
        dst_row.addWidget(QLabel("Destination"))
        dst_row.addWidget(self._dst_edit, 1)
        dst_row.addWidget(self._btn_pick_dst)
        dst_row.addWidget(QLabel("Plate"))
        dst_row.addWidget(self._plate_edit)
        outer.addLayout(dst_row)

        act_row = QHBoxLayout()
        act_row.setSpacing(SPACING["sm"])
        self._link_box = QCheckBox("Link instead of copy", self)
        self._link_box.setChecked(True)
        self._link_box.setToolTip(
            "Symlink each image under its spaCR name rather than duplicating "
            "it. Nothing about renaming requires copying bytes, and a 300 GB "
            "plate would otherwise cost 600 GB to import. Untick it if the "
            "project has to survive the originals moving, or if it is going "
            "somewhere symlinks do not work.")
        self._tiles_box = QComboBox(self)
        for label, value in TILE_POLICIES:
            self._tiles_box.addItem(label, value)
        self._tiles_box.currentIndexChanged.connect(self._on_tiles_changed)
        self._tiles_box.setToolTip(
            "What to do when a field arrives as several tiles. Stitching is "
            "the default and is what makes spaCR's filename enough: the "
            "convention has no tile slot, so four tiles of one field would "
            "share one name — and a stitched field IS one image with one "
            "name. Keeping the tiles as fields discards the fact that they "
            "are one field, which anything measuring per field then gets "
            "wrong; skipping them loses them, with a reason.")
        self._btn_save_plan = QPushButton("Save plan…", self)
        self._btn_save_plan.clicked.connect(self._pick_save_plan)
        self._btn_load_plan = QPushButton("Load plan…", self)
        self._btn_load_plan.clicked.connect(self._pick_load_plan)
        self._btn_import = QPushButton("Import", self)
        self._btn_import.setObjectName("PrimaryButton")
        self._btn_import.clicked.connect(self.run_import)
        act_row.addWidget(self._link_box)
        act_row.addWidget(QLabel("Tiles"))
        act_row.addWidget(self._tiles_box)
        act_row.addStretch(1)
        act_row.addWidget(self._btn_save_plan)
        act_row.addWidget(self._btn_load_plan)
        act_row.addWidget(self._btn_import)
        outer.addLayout(act_row)

        self._progress_bar = QProgressBar(self)
        self._progress_bar.setRange(0, 0)
        self._progress_bar.setVisible(False)
        outer.addWidget(self._progress_bar)

        self._status = QLabel("", self)
        self._status.setObjectName("Muted")
        self._status.setWordWrap(True)
        outer.addWidget(self._status)

    # -- inline reporting --------------------------------------------------

    def _set_status(self, text: str, error: bool = False) -> None:
        """Report inline. Deliberately never a QMessageBox — a modal dialog
        would hang a headless run."""
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

    def set_root(self, path: str) -> None:
        """Point the screen at a folder of images without opening a dialog.

        Fills the destination with ``<folder>_spacr`` when it is still empty,
        because a destination INSIDE the folder being read is the
        ``consolidate`` bug this module replaces: the second run imports the
        first run's output.
        """
        self._root_edit.setText(str(path or ""))
        if path and not self._dst_edit.text().strip():
            self._dst_edit.setText(os.path.normpath(str(path)) + "_spacr")

    def root_path(self) -> str:
        """The image folder currently typed in."""
        return self._root_edit.text().strip()

    def set_destination(self, path: str) -> None:
        """Where the project will be written."""
        self._dst_edit.setText(str(path or ""))

    def destination_path(self) -> str:
        """The destination currently typed in."""
        return self._dst_edit.text().strip()

    def set_plate_name(self, name: str) -> None:
        """The plate name every written filename carries."""
        self._plate_edit.setText(str(name or ""))

    def plate_name(self) -> str:
        """The plate name, falling back to ``plate1`` when it is blank."""
        return self._plate_edit.text().strip() or "plate1"

    def set_sample(self, count: int) -> None:
        """How many files the scan reads before deciding."""
        self._sample_box.setValue(int(count))

    def sample(self) -> int:
        """The sample size."""
        return int(self._sample_box.value())

    def set_read_inside(self, on: bool) -> None:
        """Whether the scan opens each file for its own axis metadata."""
        self._inside_box.setChecked(bool(on))

    def read_inside(self) -> bool:
        """Whether files are opened during the scan."""
        return self._inside_box.isChecked()

    def set_link(self, on: bool) -> None:
        """Whether the import symlinks rather than copies."""
        self._link_box.setChecked(bool(on))

    def link(self) -> bool:
        """Whether the import will symlink."""
        return self._link_box.isChecked()

    def set_tile_policy(self, policy: str) -> bool:
        """Choose what happens to a field that arrives as several tiles.

        :param policy: one of the values in :data:`TILE_POLICIES`.
        :returns: False for a policy that is not offered, with the reason
            inline rather than as an exception into a GUI slot.
        """
        index = self._tiles_box.findData(str(policy))
        if index < 0:
            self._set_status(
                f"{policy!r} is not a way to treat tiles; expected one of "
                f"{', '.join(value for _label, value in TILE_POLICIES)}.",
                error=True)
            return False
        self._tiles_box.setCurrentIndex(index)
        return True

    def tile_policy(self) -> str:
        """What will happen to tiled fields: stitch, fields or skip."""
        return str(self._tiles_box.currentData())

    def set_tiles_as_fields(self, on: bool) -> None:
        """Whether each tile is given a field number of its own.

        The older two-state spelling of :meth:`set_tile_policy`, kept because
        it is what a caller who has not heard of stitching means: off is now
        the DEFAULT policy rather than the skip it used to be, since a
        stitched field is what it always wanted.
        """
        self.set_tile_policy("fields" if on else "stitch")

    def tiles_as_fields(self) -> bool:
        """Whether tiles will be written as fields."""
        return self.tile_policy() == "fields"

    def stitch_tiles(self) -> bool:
        """Whether tiles will be assembled into the field they came from."""
        return self.tile_policy() == "stitch"

    # -- pickers -----------------------------------------------------------

    def _pick_root(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose your images")
        if path:
            self.set_root(path)

    def _pick_destination(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose destination")
        if path:
            self.set_destination(path)

    def _pick_save_plan(self) -> None:
        path, _filter = QFileDialog.getSaveFileName(
            self, "Save the import plan", "import_plan.json",
            "JSON (*.json);;All files (*)")
        if path:
            self.save_plan(path)

    def _pick_load_plan(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self, "Load an import plan", "", "JSON (*.json);;All files (*)")
        if path:
            self.load_plan(path)

    # -- scan --------------------------------------------------------------

    def scan(self) -> bool:
        """Work out what the folder holds. Writes nothing.

        :returns: True when the job was started (or, unthreaded, completed);
            False when the inputs are unusable, with the reason inline.
        """
        root = self.root_path()
        if not root:
            self._set_status("Choose the folder your images are in first.",
                             error=True)
            return False
        if not os.path.isdir(root):
            self._set_status(f"Not a folder: {root}", error=True)
            return False

        sample = self.sample()
        inside = self.read_inside()

        def _job():
            """Read the folder in the worker thread. Writes nothing.

            The settings are read out of the widgets BEFORE this closes over
            them: a job must not touch a QLineEdit from another thread, and
            a user who edits the sample size mid-scan would otherwise change
            what the running scan is doing.
            """
            return imp.plan_import(root, sample=sample, read_files=inside)

        self._progress_bar.setVisible(True)
        self._set_status(f"Reading {root}…")
        return self._run_job(_job, self._on_plan_ready)

    def _on_plan_ready(self, plan: Optional["imp.ImportPlan"]) -> None:
        """Show the proposal and its questions. Always on the GUI thread."""
        self._progress_bar.setVisible(False)
        self._plan = plan
        if plan is None:
            self._model.set_plan(None)
            self._answers.set_plan(None)
            self._set_report("")
            self._set_status("The scan produced no plan.", error=True)
            self._update_controls()
            return
        self._model.set_plan(plan)
        self._answers.set_plan(plan)
        self._refresh_report()

    def _on_answer_edited(self, *_args) -> None:
        """An answer changed: re-resolve in memory, touching no disk.

        :meth:`spacr.image_import.ImportPlan.with_mapping` returns a NEW plan
        rather than mutating this one, so a half-typed answer costs nothing
        and the folder is never re-walked.
        """
        if self._plan is None:
            return
        self._plan = imp.ImportPlan(layout=self._plan.layout,
                                    inside=self._plan.inside,
                                    mapping=self._answers.mapping())
        self._model.set_plan(self._plan)
        self._refresh_report()

    def _refresh_report(self) -> None:
        """Redraw the report pane and the status line from the current plan."""
        plan = self._plan
        if plan is None:
            self._set_report("")
            self._update_controls()
            return

        lines = [f"{len(plan.files)} image(s) under {plan.root}",
                 plan.layout.summary()]
        if plan.layout.skipped:
            lines.append("")
            lines.append(f"NOT INTERPRETED ({len(plan.layout.skipped)} named "
                         f"unlike the rest):")
            lines += [f"  {rel}" for rel in plan.layout.skipped[:10]]
            if len(plan.layout.skipped) > 10:
                lines.append(f"  ... and {len(plan.layout.skipped) - 10} more")
        problems = plan.problems()
        tiled = self._tiled_images()
        if problems or tiled:
            lines.append("")
            lines += [f"  ! {issue}" for issue in problems]
            if tiled:
                lines.append(f"  ! {tiled} image(s) are tiles of a field. "
                             f"{self._tile_plan(tiled)}")
        self._set_report("\n".join(lines))

        counts = plan.counts()
        if problems:
            self._set_status(
                f"{len(problems)} question(s) left — answer them below, or "
                f"nothing can be imported. Nothing has been written.",
                error=True)
        else:
            shape = "  ".join(f"{axis} {counts[axis]}" for axis in imp.AXES
                              if counts.get(axis))
            self._set_status(
                f"{len(plan.files)} image(s) ready — {shape}. Check a few "
                f"filenames against their fields above, then press Import. "
                f"Nothing has been written yet.")
        self._update_controls()

    def _tile_plan(self, tiled: int) -> str:
        """What the current policy will do with ``tiled`` tile images.

        Said in the report BEFORE the import, because each of the three
        choices loses something different and only one of them is
        recoverable by pressing the button again.
        """
        policy = self.tile_policy()
        if policy == "stitch":
            fields = len({canonical for canonical in (
                imp.canonical_name(entry, plate=self.plate_name())
                for entry in self._plan.files.values() if "tile" in entry)})
            return (f"They will be assembled into {fields} field(s). Any "
                    f"whose tiles do not correlate is butt-joined and said "
                    f"so in the summary.")
        if policy == "fields":
            return ("Each will be written as a field of its own, which "
                    "discards the fact that they are one field.")
        return ("They will be SKIPPED: spaCR's filename has no tile slot, so "
                "they would share one name. Stitch them instead to keep "
                "them.")

    def _tiled_images(self) -> int:
        """How many of the images on screen are tiles of some field."""
        if self._plan is None:
            return 0
        return sum(1 for entry in self._plan.files.values() if "tile" in entry)

    def _tiles_at_risk(self) -> int:
        """How many images the current tile policy would DROP.

        Zero unless the policy is to skip them: stitching writes every tile
        into a field and keeping them as fields writes every tile as one.
        Counted BEFORE the import rather than reported after it -- the whole
        point of the plan is that a loss is visible while it can still be
        prevented.
        """
        return self._tiled_images() if self.tile_policy() == "skip" else 0

    def _on_tiles_changed(self, *_args) -> None:
        """The tile policy changed: it changes what the report has to say."""
        if self._plan is not None:
            self._refresh_report()

    # -- what the screen is showing ----------------------------------------

    def plan(self) -> Optional["imp.ImportPlan"]:
        """The plan currently on screen, or None."""
        return self._plan

    def result(self) -> Optional["imp.ImportResult"]:
        """The result of the last import, or None."""
        return self._result

    def proposal_row_count(self) -> int:
        """Rows in the proposal table — one per image the scan read."""
        return self._model.rowCount()

    def proposal_columns(self) -> List[str]:
        """The proposal's columns: the axes this folder actually has."""
        return self._model.headers()

    def proposal_value(self, row: int, column: str) -> str:
        """One parsed field, by row and column name."""
        return self._model.value_at(row, column)

    def question_count(self) -> int:
        """How many values are waiting for an answer."""
        return self._answers.rowCount()

    def questions(self) -> List[List[str]]:
        """``[token, value, answer]`` for each unnamed value."""
        return self._answers.answers()

    def answer_question(self, row: int, channel: str) -> bool:
        """Answer one row of the question table, as an editor would."""
        return self._answers.set_answer(row, channel)

    def problems(self) -> List[str]:
        """Everything that would make this import wrong, in plain sentences."""
        return self._plan.problems() if self._plan is not None else []

    # -- the plan file -----------------------------------------------------

    def save_plan(self, path: str) -> bool:
        """Write the plan on screen where a later run can load it back."""
        if self._plan is None:
            self._set_status("There is no plan to save — press Scan first.",
                             error=True)
            return False
        try:
            written = imp.save_plan(self._plan, str(path))
        except OSError as exc:
            self._set_status(f"Could not save the plan: {exc}", error=True)
            return False
        self._set_status(
            f"Plan saved to {written}. Next week's plate imports in one press "
            f"— and `spacr.image_import.load_plan` reads the same file with "
            f"no GUI at all.")
        return True

    def load_plan(self, path: str) -> bool:
        """Read a saved plan back and show it.

        The folder is re-read and only the saved ANSWERS are reused, which is
        the point of loading one: this week's plate may have more images than
        last week's, and replaying a stale file table would silently import
        last week's.
        """
        try:
            plan = imp.load_plan(str(path))
        except (OSError, ValueError, KeyError) as exc:
            self._set_status(f"Could not load the plan: {exc}", error=True)
            return False
        self.set_root(str(plan.root))
        self._on_plan_ready(plan)
        return True

    # -- import ------------------------------------------------------------

    def run_import(self) -> bool:
        """Write the project. Off the GUI thread unless ``threaded=False``.

        :returns: True when the job was started, False when it was refused —
            with the reason inline.
        """
        if self._plan is None:
            self._set_status("Press Scan first — there is nothing to import "
                             "yet.", error=True)
            return False
        problems = self._plan.problems()
        if problems:
            self._set_status(
                f"This plan still has {len(problems)} problem(s); answer them "
                f"and try again. Nothing was written.", error=True)
            return False
        dst = self.destination_path()
        if not dst:
            self._set_status("Choose a destination folder first.", error=True)
            return False
        inside = self._is_inside(dst, self.root_path())
        if inside:
            self._set_status(
                f"{dst} is inside the folder being imported. A destination "
                f"under the source is read by the next scan as more images — "
                f"the bug that made `consolidate` double a plate on every "
                f"run. Choose a folder beside it instead.", error=True)
            return False

        plan = self._plan
        link = self.link()
        plate = self.plate_name()
        tiles = self.tiles_as_fields()
        stitch = self.stitch_tiles()

        def _job():
            """Write the project in the worker thread.

            Every argument is a plain value captured above, for the same
            reason the scan's is: this runs off the GUI thread, and reading
            a checkbox from there is undefined behaviour rather than a
            stale answer.
            """
            return imp.apply_import(plan, dst, link=link, plate=plate,
                                    tiles_as_fields=tiles, stitch_tiles=stitch)

        self._progress_bar.setVisible(True)
        self._set_status(f"Importing {len(plan.files)} image(s) into {dst}…")
        return self._run_job(_job, self._on_result_ready)

    @staticmethod
    def _is_inside(destination: str, root: str) -> bool:
        """Whether ``destination`` sits under ``root`` — the same folder
        included. Symlinks are resolved first, because ``dst`` pointing at
        the source through a link is the same trap wearing a different name.
        """
        if not destination or not root:
            return False
        try:
            dst = os.path.realpath(destination)
            src = os.path.realpath(root)
        except OSError:                                  # pragma: no cover
            return False
        return dst == src or dst.startswith(src + os.sep)

    def _on_result_ready(self, result: Optional["imp.ImportResult"]) -> None:
        """Show the import summary. Always on the GUI thread."""
        self._result = result
        self._progress_bar.setVisible(False)
        if result is None:
            self._set_status("The import produced no result.", error=True)
            self._update_controls()
            return
        self._set_report(result.summary())
        made = (f"Imported {result.written} image(s)"
                + (f", {result.stitched} of them stitched from their tiles"
                   if result.stitched else "")
                + f" into {result.destination}")
        if result.skipped:
            self._set_status(
                f"{made}, but {len(result.skipped)} source image(s) were NOT "
                f"written — see the summary for each one and why.", error=True)
        elif result.unverified:
            self._set_status(
                f"{made}. {len(result.unverified)} field(s) had nothing to "
                f"correlate, so their tiles were butt-joined and the seams "
                f"are unverified — check those fields before measuring.",
                error=True)
        else:
            self._set_status(
                f"{made}. Point Mask at that folder — it is a spaCR project "
                f"now, and no convention has to be chosen.")
        self._update_controls()

    # -- controls ----------------------------------------------------------

    def _on_input_changed(self, *_args) -> None:
        """The folder changed: whatever is on screen describes the old one."""
        if self._plan is not None:
            self._plan = None
            self._result = None
            self._model.set_plan(None)
            self._answers.set_plan(None)
            self._set_report("")
            self._set_status("The folder changed — press Scan again.")
        self._update_controls()

    def _update_controls(self) -> None:
        idle = not self._busy
        for widget in (self._btn_pick_root, self._btn_pick_dst, self._btn_scan,
                       self._root_edit, self._dst_edit, self._plate_edit,
                       self._sample_box, self._inside_box, self._link_box,
                       self._tiles_box, self._table, self._answer_table):
            widget.setEnabled(idle)
        self._btn_save_plan.setEnabled(idle and self._plan is not None)
        self._btn_load_plan.setEnabled(idle)
        self._btn_import.setEnabled(
            idle and self._plan is not None and not self._plan.problems())

    def can_import(self) -> bool:
        """True when the Import button is live."""
        return self._btn_import.isEnabled()

    def is_busy(self) -> bool:
        """True while a scan or an import is in flight."""
        return self._busy

    def active_jobs(self) -> int:
        """How many worker threads are still winding down."""
        return len(self._jobs)

    # -- job plumbing ------------------------------------------------------

    def _run_job(self, fn: Callable[[], Any],
                 on_done: Callable[[Any], None]) -> bool:
        """Run ``fn`` off the GUI thread and hand its result to ``on_done``.

        The same idiom as :meth:`spacr.qt.screens.foreign.ForeignScreen._run_job`,
        and for the same reason: ``PipelineWorker.finished`` is emitted *in
        the worker thread*, and PySide6 invokes a plain closure connected to
        it directly, on that thread. The completion handlers here reset a
        table model and fill a QPlainTextEdit, and building a QTextDocument's
        children off the GUI thread is undefined behaviour. So ``finished``
        is chained through :attr:`_job_settled` into a *bound method* of this
        widget, which has GUI-thread affinity — Qt then queues the call.

        With ``threaded=False`` the call runs inline and the same signals
        fire, so both paths behave identically from outside.
        """
        if not self._threaded:
            ok = True
            try:
                on_done(fn())
            except Exception as e:                       # noqa: BLE001
                self._on_job_error(e)
                ok = False
            self._update_controls()
            self.job_finished.emit(ok)
            return ok

        box: Dict[str, Any] = {}
        thread, worker = make_thread(partial(self._capture, fn), box)
        # Strong references: PySide6 will not keep the worker alive through
        # the started→run connection alone, and a QThread garbage-collected
        # while still running takes the process down with it.
        self._jobs.append((thread, worker))
        self._thread, self._worker = thread, worker
        self._pending.append((box, on_done))
        worker.error.connect(self._on_worker_error_text)
        worker.finished.connect(self._job_settled)
        thread.finished.connect(self._retire_finished_jobs)
        self._busy = True
        self._update_controls()
        thread.start()
        return True

    @staticmethod
    def _capture(fn: Callable[[], Any], payload: Dict[str, Any]) -> None:
        """Run ``fn`` in the worker thread and stash its result in ``payload``.

        A named method rather than a closure: this body executes on a
        QThread, where coverage cannot see it, and a nested function would be
        untestable except by running the thread. This one can be called
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
            except Exception as e:                       # noqa: BLE001
                self._on_job_error(e)
                ok = False
        self._update_controls()
        self.job_finished.emit(ok)

    def _retire_finished_jobs(self) -> None:
        """Retire every job whose QThread has stopped. GUI thread only.

        A BOUND METHOD, not a closure — the rule ``make_thread`` states and
        then relies on for its own ``handle.retire``. With a closure PySide6
        makes the QThread itself the receiver, and ``make_thread`` connects
        ``thread.finished -> thread.deleteLater`` FIRST; slots run in
        connection order, so the DeferredDelete is posted ahead of the
        closure's metacall and Qt discards queued events for a destroyed
        receiver.

        It sweeps rather than naming a sender for the same reason: by the
        time this runs, the emitter may be exactly what is gone, and
        ``QObject.sender()`` is null for a queued call whose emitter was
        destroyed.
        """
        from ..bridge import thread_has_stopped

        for thread, _worker in list(self._jobs):
            if thread_has_stopped(thread):
                self._retire_job(thread)

    def _retire_job(self, thread) -> None:
        """Release *this* job's refs once its own event loop has exited."""
        self._jobs = [(t, w) for (t, w) in self._jobs if t is not thread]
        if self._thread is thread:
            self._thread = None
            self._worker = None

    def _on_job_error(self, exc: Exception) -> None:
        self._busy = False
        self._progress_bar.setVisible(False)
        self._set_status(str(exc) or exc.__class__.__name__, error=True)

    def _on_worker_error_text(self, text: str) -> None:
        line = (text or "").strip().splitlines()[-1] if text else "unknown error"
        self._busy = False
        self._progress_bar.setVisible(False)
        self._set_status(f"Import failed: {line}", error=True)

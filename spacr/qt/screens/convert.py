"""
Format Converter — vendor microscopy files into Yokogawa TIFFs, with the
mapping on screen before anything is written.

The screen exists because the conversion step is where a screen silently
goes wrong. Rename 384 wells' worth of ND2 into
``plate1_A01_T0001F001L01A01Z01C01.tif`` and the filenames stop carrying
any trace of where they came from; get the well assignment wrong and
nobody finds out until the hit list is being followed up, weeks later.
So this screen does two things :func:`spacr.io.convert_to_yokogawa` never
did: it shows the source → target table *before* writing, and it emits a
map file that turns every converted name back into the original one.

Layout::

    ┌───────────────────────────────────────────────────────────────────┐
    │ /data/run1                                    [Choose source…]    │
    │ Layout [auto ▾]  Z [keep every plane ▾]  Plate names [plate1 ▾]   │
    │ /data/run1_yokogawa                      [Choose destination…]    │
    │                                     [Preview]        [Convert]    │
    ├───────────────────────────────────────────────────────────────────┤
    │ source              target                       plate well  fld  │
    │ run1/wt/f01_C1.tif  plate1_A01_T0001F001…C01.tif plate1 A01  1    │
    │ run1/wt/f01_C2.tif  plate1_A01_T0001F001…C02.tif plate1 A01  1    │
    │ …                                                                 │
    ├───────────────────────────────────────────────────────────────────┤
    │ 20 file(s) would be written from 20 source(s).                    │
    │ 1 plate(s), 1 well(s), 2 channel id(s).                           │
    ├───────────────────────────────────────────────────────────────────┤
    │ Previewed 20 output file(s). Nothing has been written.            │
    └───────────────────────────────────────────────────────────────────┘

Design notes:

* **The preview is the product.** :func:`spacr.convert.scan` and
  :func:`spacr.convert.plan` write nothing at all; ``Convert`` is a
  separate press. A plan with a blocking error (two sources colliding on
  one output name) leaves ``Convert`` disabled — the fix is upstream, in
  the folder layout, not in a "yes, overwrite" button.
* **Everything heavy is in** :mod:`spacr.convert`, which imports neither
  torch nor cellpose, so this stays a view and the logic is testable
  headless.
* **Off the GUI thread.** Scanning a plate's worth of ND2 headers takes
  seconds; converting takes minutes. Both go through
  :func:`spacr.qt.bridge.make_thread`, and the completion handler is
  reached through a *bound method* (:attr:`ConvertScreen._job_settled`)
  rather than a closure, because ``PipelineWorker.finished`` is emitted
  in the worker thread and a closure connected to it would build widget
  children there. Tests pass ``threaded=False``.
* **No modal dialogs on any error path.** A missing folder, an absent
  ``nd2reader``, a name collision — all of it lands in the inline status
  label and the summary pane. A QMessageBox would hang a headless run.
"""
from __future__ import annotations

import os
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from ... import convert as cvt
from ..bridge import make_thread
from ..theme import SPACING, active_palette
from ..widgets import Divider

__all__ = [
    "ConvertScreen",
    "PlanTableModel",
    "LAYOUT_CHOICES",
    "Z_CHOICES",
    "PLATE_NAME_CHOICES",
]


#: Source layouts, as ``(label, value)``. The labels spell out what the
#: folder tree has to look like — "auto" is right almost always, and the
#: explicit ones exist for the trees it guesses wrong.
LAYOUT_CHOICES: Tuple[Tuple[str, str], ...] = (
    ("Detect automatically", "auto"),
    ("src/<plate>/<well>/images", "plate_well"),
    ("src/<well>/images", "well"),
    ("images directly in src", "flat"),
)

#: Z handling. The default keeps every plane; both lossy options say so
#: in the label, because the whole point is that projection is a choice
#: somebody made rather than something that happened to their data.
Z_CHOICES: Tuple[Tuple[str, str], ...] = (
    ("Keep every plane (one file per Z)", cvt.Z_KEEP),
    ("Max-project Z (planes are discarded)", cvt.Z_MAX),
    ("First plane only (planes are discarded)", cvt.Z_FIRST),
)

#: How plate folders are named in the output.
PLATE_NAME_CHOICES: Tuple[Tuple[str, str], ...] = (
    ("plate1, plate2, …", "index"),
    ("keep the folder name", "name"),
)

#: Preview columns, in display order, with their headers.
PREVIEW_COLUMNS: Tuple[Tuple[str, str], ...] = (
    ("source", "Source"),
    ("target", "Target"),
    ("plate", "Plate"),
    ("well", "Well"),
    ("field", "Field"),
    ("channel", "Channel"),
    ("z", "Z"),
    ("t", "T"),
    ("source_well", "From well"),
    ("source_field", "From field"),
    ("source_channel", "From channel"),
    ("z_handling", "Z handling"),
    ("status", "Status"),
)


class PlanTableModel(QAbstractTableModel):
    """Read-only table model over a :meth:`ConversionPlan.to_frame` frame.

    A model rather than a QTableWidget because the preview for a full
    plate is tens of thousands of rows and populating that many
    QTableWidgetItems freezes the window for seconds.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._frame: pd.DataFrame = pd.DataFrame(
            columns=[key for key, _label in PREVIEW_COLUMNS])
        self._columns: List[Tuple[str, str]] = list(PREVIEW_COLUMNS)

    def set_frame(self, frame: Optional[pd.DataFrame]) -> None:
        """Replace the displayed frame, keeping only the known columns."""
        self.beginResetModel()
        if frame is None or not len(frame):
            self._frame = pd.DataFrame(
                columns=[key for key, _label in PREVIEW_COLUMNS])
            self._columns = list(PREVIEW_COLUMNS)
        else:
            self._columns = [(key, label) for key, label in PREVIEW_COLUMNS
                             if key in frame.columns]
            self._frame = frame
        self.endResetModel()

    def frame(self) -> pd.DataFrame:
        """The frame currently displayed."""
        return self._frame

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else int(len(self._frame))

    def columnCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._columns)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or role not in (Qt.DisplayRole, Qt.ToolTipRole):
            return None
        key = self._columns[index.column()][0]
        value = self._frame.iloc[index.row()][key]
        if key == "source" and role == Qt.DisplayRole:
            # The full path is the tooltip; the cell shows enough to
            # recognise the file without a 200-pixel column of prefix.
            return os.path.basename(str(value))
        return "" if value is None else str(value)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return self._columns[section][1]
        return str(section + 1)


class ConvertScreen(QWidget):
    """Pick a source tree, review the mapping, convert, read the summary.

    :param parent: parent widget.
    :param threaded: when False every job runs inline on the calling
        thread. Tests use it so assertions are exact; the app leaves it
        True so a 40-minute conversion does not freeze the window.
    """

    #: Emitted with True/False when a scan or a conversion settles.
    job_finished = Signal(bool)
    #: Internal relay so the completion handler runs on the GUI thread.
    _job_settled = Signal(bool)
    #: ``(done, total, item)`` — emitted from the worker thread.
    _progress = Signal(int, int, str)

    def __init__(self, parent=None, threaded: bool = True):
        super().__init__(parent)
        self._threaded = bool(threaded)
        self._plan: Optional[cvt.ConversionPlan] = None
        self._result: Optional[cvt.ConversionResult] = None
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
            "Choose a folder of microscope files, then Preview. Nothing is "
            "written until you press Convert.")
        self._update_controls()

    # -- construction ------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACING["lg"], SPACING["lg"],
                                 SPACING["lg"], SPACING["lg"])
        outer.setSpacing(SPACING["md"])

        title = QLabel("Format Converter")
        title.setObjectName("DisplayHeading")
        outer.addWidget(title)

        subtitle = QLabel(
            "ND2 / CZI / LIF / OME-TIFF / TIFF / PNG into Yokogawa-named "
            "TIFFs that Mask and Measure read directly. The mapping is shown "
            "before anything is written, and a conversion_map.csv in the "
            "destination records which original file every converted name "
            "came from.")
        subtitle.setObjectName("Muted")
        subtitle.setWordWrap(True)
        outer.addWidget(subtitle)

        outer.addWidget(Divider())

        # ── Source row ────────────────────────────────────────────────
        src_row = QHBoxLayout()
        src_row.setSpacing(SPACING["sm"])
        self._src_edit = QLineEdit(self)
        self._src_edit.setPlaceholderText(
            "…/run1  — a folder of images, or <plate>/<well>/ folders")
        self._src_edit.setClearButtonEnabled(True)
        self._src_edit.returnPressed.connect(self.preview)
        self._btn_pick_src = QPushButton("Choose source…", self)
        self._btn_pick_src.clicked.connect(self._pick_source)
        src_row.addWidget(QLabel("Source"))
        src_row.addWidget(self._src_edit, 1)
        src_row.addWidget(self._btn_pick_src)
        outer.addLayout(src_row)

        # ── Options row ───────────────────────────────────────────────
        opt_row = QHBoxLayout()
        opt_row.setSpacing(SPACING["sm"])
        self._layout_box = QComboBox(self)
        for label, value in LAYOUT_CHOICES:
            self._layout_box.addItem(label, value)
        self._z_box = QComboBox(self)
        for label, value in Z_CHOICES:
            self._z_box.addItem(label, value)
        self._plate_box = QComboBox(self)
        for label, value in PLATE_NAME_CHOICES:
            self._plate_box.addItem(label, value)
        for box in (self._layout_box, self._z_box, self._plate_box):
            box.currentIndexChanged.connect(self._on_option_changed)
        opt_row.addWidget(QLabel("Layout"))
        opt_row.addWidget(self._layout_box, 1)
        opt_row.addWidget(QLabel("Z"))
        opt_row.addWidget(self._z_box, 1)
        opt_row.addWidget(QLabel("Plate names"))
        opt_row.addWidget(self._plate_box, 1)
        outer.addLayout(opt_row)

        # ── Destination row ───────────────────────────────────────────
        dst_row = QHBoxLayout()
        dst_row.setSpacing(SPACING["sm"])
        self._dst_edit = QLineEdit(self)
        self._dst_edit.setPlaceholderText(
            "…/run1_yokogawa  — a NEW folder; the originals are never touched")
        self._dst_edit.setClearButtonEnabled(True)
        self._btn_pick_dst = QPushButton("Choose destination…", self)
        self._btn_pick_dst.clicked.connect(self._pick_destination)
        self._btn_preview = QPushButton("Preview", self)
        self._btn_preview.clicked.connect(self.preview)
        self._btn_convert = QPushButton("Convert", self)
        self._btn_convert.setObjectName("PrimaryButton")
        self._btn_convert.clicked.connect(self.run_convert)
        dst_row.addWidget(QLabel("Destination"))
        dst_row.addWidget(self._dst_edit, 1)
        dst_row.addWidget(self._btn_pick_dst)
        dst_row.addWidget(self._btn_preview)
        dst_row.addWidget(self._btn_convert)
        outer.addLayout(dst_row)

        # ── Preview table ─────────────────────────────────────────────
        self._model = PlanTableModel(self)
        self._table = QTableView(self)
        self._table.setModel(self._model)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)
        self._table.verticalHeader().setVisible(False)
        outer.addWidget(self._table, 1)

        # ── Summary + progress + status ───────────────────────────────
        self._summary = QPlainTextEdit(self)
        self._summary.setReadOnly(True)
        self._summary.setMaximumHeight(140)
        self._summary.setPlaceholderText(
            "The plan summary and, after a run, what was converted and what "
            "was skipped.")
        outer.addWidget(self._summary)

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
        """Current inline status message (test/introspection helper)."""
        return self._status.text()

    def summary_text(self) -> str:
        """Whatever is in the summary pane."""
        return self._summary.toPlainText()

    def _set_summary(self, text: str) -> None:
        self._summary.setPlainText(text or "")

    # -- configuration -----------------------------------------------------

    def set_source(self, path: str) -> None:
        """Point the screen at a source folder without opening a dialog."""
        self._src_edit.setText(str(path or ""))
        if path and not self._dst_edit.text().strip():
            self._dst_edit.setText(
                os.path.join(os.path.dirname(os.path.normpath(str(path))),
                             os.path.basename(os.path.normpath(str(path)))
                             + "_yokogawa"))
        self._on_option_changed()

    def source_path(self) -> str:
        """The source folder currently typed in."""
        return self._src_edit.text().strip()

    def set_destination(self, path: str) -> None:
        """Set the destination folder."""
        self._dst_edit.setText(str(path or ""))
        self._update_controls()

    def destination_path(self) -> str:
        """The destination folder currently typed in."""
        return self._dst_edit.text().strip()

    def _set_combo(self, box: QComboBox, value: str, what: str) -> None:
        index = box.findData(value)
        if index < 0:
            raise ValueError(f"Unknown {what}: {value!r}")
        box.setCurrentIndex(index)

    def set_layout_mode(self, value: str) -> None:
        """Choose the source layout (see :data:`LAYOUT_CHOICES`)."""
        self._set_combo(self._layout_box, value, "layout")

    def layout_mode(self) -> str:
        """The selected source layout."""
        return str(self._layout_box.currentData())

    def set_z_handling(self, value: str) -> None:
        """Choose how z planes are treated (see :data:`Z_CHOICES`)."""
        self._set_combo(self._z_box, value, "z_handling")

    def z_handling(self) -> str:
        """The selected z handling."""
        return str(self._z_box.currentData())

    def set_plate_naming(self, value: str) -> None:
        """Choose how output plates are named."""
        self._set_combo(self._plate_box, value, "plate_naming")

    def plate_naming(self) -> str:
        """The selected plate naming scheme."""
        return str(self._plate_box.currentData())

    def _on_option_changed(self, *_args) -> None:
        """Any option change invalidates the plan on screen.

        A preview that no longer matches the settings above it is worse
        than no preview: it is a table the user believes.
        """
        if self._plan is not None:
            self._plan = None
            self._model.set_frame(None)
            self._set_summary("")
            self._set_status("Settings changed — press Preview again.")
        self._update_controls()

    # -- pickers -----------------------------------------------------------

    def _pick_source(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose source folder")
        if path:
            self.set_source(path)

    def _pick_destination(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose destination folder")
        if path:
            self.set_destination(path)

    # -- preview -----------------------------------------------------------

    def preview(self) -> bool:
        """Scan the source and build the plan. Writes nothing.

        :returns: True when the scan was started (or, unthreaded,
            completed) — False when the source is unusable, with the
            reason in the inline status label.
        """
        src = self.source_path()
        if not src:
            self._set_status("Choose a source folder first.", error=True)
            return False
        if not os.path.isdir(src):
            self._set_status(f"Not a folder: {src}", error=True)
            return False

        layout = self.layout_mode()
        z_handling = self.z_handling()
        plate_naming = self.plate_naming()

        def _job():
            sources = cvt.scan(src, layout=layout)
            return cvt.plan(sources, z_handling=z_handling,
                            plate_naming=plate_naming)

        self._set_status(f"Scanning {src}…")
        return self._run_job(_job, self._on_plan_ready)

    def _on_plan_ready(self, plan: Optional[cvt.ConversionPlan]) -> None:
        """Show the plan. Always on the GUI thread."""
        self._plan = plan
        if plan is None:
            self._model.set_frame(None)
            self._set_summary("")
            self._set_status("Scan produced no plan.", error=True)
            self._update_controls()
            return
        self._model.set_frame(plan.to_frame())
        self._set_summary(plan.summary())
        if not plan.ok:
            self._set_status(
                f"{len(plan.errors)} blocking problem(s) — nothing can be "
                f"converted until they are fixed. See the summary below.",
                error=True)
        elif not len(plan):
            self._set_status("No readable images were found in that folder.",
                             error=True)
        else:
            skipped = len(plan.unreadable)
            tail = f" {skipped} source(s) cannot be read." if skipped else ""
            self._set_status(
                f"Previewed {len(plan)} output file(s) from {plan.n_sources} "
                f"source(s). Nothing has been written yet.{tail}")
        self._update_controls()

    def plan(self) -> Optional[cvt.ConversionPlan]:
        """The plan currently on screen, or None."""
        return self._plan

    def result(self) -> Optional[cvt.ConversionResult]:
        """The result of the last conversion, or None."""
        return self._result

    def preview_row_count(self) -> int:
        """Rows in the preview table."""
        return self._model.rowCount()

    def preview_value(self, row: int, column: str) -> str:
        """One preview cell by column name (test/introspection helper)."""
        frame = self._model.frame()
        if row < 0 or row >= len(frame) or column not in frame.columns:
            return ""
        return str(frame.iloc[row][column])

    def preview_targets(self) -> List[str]:
        """Every target filename in the preview, in table order."""
        frame = self._model.frame()
        if "target" not in frame.columns:
            return []
        return [str(v) for v in frame["target"].tolist()]

    # -- convert -----------------------------------------------------------

    def run_convert(self) -> bool:
        """Convert the previewed plan into the destination folder.

        :returns: True when the job was started, False when it was
            refused — with the reason inline.
        """
        if self._plan is None:
            self._set_status("Press Preview first — there is nothing to "
                             "convert yet.", error=True)
            return False
        if not self._plan.ok:
            self._set_status(
                "This plan has blocking problems; fix them and preview "
                "again. Nothing was written.", error=True)
            return False
        if not len(self._plan):
            self._set_status("The plan is empty — nothing to convert.",
                             error=True)
            return False
        dst = self.destination_path()
        if not dst:
            self._set_status("Choose a destination folder first.", error=True)
            return False

        plan = self._plan
        emit = self._progress.emit

        def _job():
            return cvt.convert(plan, dst, progress=emit)

        self._progress_bar.setVisible(True)
        self._progress_bar.setRange(0, max(plan.n_sources, 1))
        self._progress_bar.setValue(0)
        self._set_status(f"Converting {len(plan)} file(s) into {dst}…")
        return self._run_job(_job, self._on_result_ready)

    def _on_progress(self, done: int, total: int, item: str) -> None:
        """Progress from the worker thread. Always on the GUI thread."""
        self._progress_bar.setRange(0, max(int(total), 1))
        self._progress_bar.setValue(int(done))
        self._set_status(f"Converting {done}/{total} — {item}")

    def _on_result_ready(self, result: Optional[cvt.ConversionResult]) -> None:
        """Show the conversion summary. Always on the GUI thread."""
        self._result = result
        self._progress_bar.setVisible(False)
        if result is None:
            self._set_status("The conversion produced no result.", error=True)
            self._update_controls()
            return
        self._set_summary(result.summary())
        if result.is_complete:
            self._set_status(
                f"Converted {result.n_written} file(s) into {result.dst}. "
                f"Map: {os.path.basename(result.map_path)}")
        else:
            self._set_status(
                f"Converted {result.n_written} file(s), skipped "
                f"{result.n_skipped} — see the summary. The map file is "
                f"stamped incomplete.", error=True)
        self._update_controls()

    # -- controls ----------------------------------------------------------

    def _update_controls(self) -> None:
        idle = not self._busy
        has_plan = self._plan is not None and self._plan.ok and len(self._plan) > 0
        for widget in (self._btn_pick_src, self._btn_pick_dst,
                       self._btn_preview, self._src_edit, self._dst_edit,
                       self._layout_box, self._z_box, self._plate_box):
            widget.setEnabled(idle)
        self._btn_convert.setEnabled(idle and has_plan)

    def can_convert(self) -> bool:
        """True when the Convert button is live."""
        return self._btn_convert.isEnabled()

    # -- job plumbing ------------------------------------------------------

    def _run_job(self, fn: Callable[[], Any],
                 on_done: Callable[[Any], None]) -> bool:
        """Run ``fn`` off the GUI thread and hand its result to ``on_done``.

        The same idiom as ``PlateViewScreen._run_job``, and for the same
        reason: ``PipelineWorker.finished`` is emitted *in the worker
        thread*, and PySide6 invokes a plain closure connected to it
        directly, on that thread. The completion handlers here fill a
        QPlainTextEdit and reset a table model, and building a
        QTextDocument's children off the GUI thread is undefined
        behaviour. So ``finished`` is chained through
        :attr:`_job_settled` into a *bound method* of this widget, which
        has GUI-thread affinity — Qt then queues the call.

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
        # while still running takes the process down with it.
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

        A named method rather than the closure ``PlateViewScreen`` uses,
        for one reason: this body executes on a QThread, where coverage
        cannot see it, and a nested function would be untestable except
        by running the thread. This one can be called directly.
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
        """True while a scan or conversion is in flight."""
        return self._busy

    def _on_job_error(self, exc: Exception) -> None:
        self._busy = False
        self._progress_bar.setVisible(False)
        self._set_status(str(exc) or exc.__class__.__name__, error=True)

    def _on_worker_error_text(self, text: str) -> None:
        line = (text or "").strip().splitlines()[-1] if text else "unknown error"
        self._busy = False
        self._progress_bar.setVisible(False)
        self._set_status(f"Conversion failed: {line}", error=True)

"""
Model Compare — two Cellpose models, the same three fields, one table.

Picking a segmentation model in spaCR means running a plate, looking at
montages, changing one number and running it again. This screen makes the
question small enough to answer in a minute: choose a folder, take three
fields, configure two models, press Compare. The masks come back side by side
over the same image with the object counts, the ARI and the split/merge
attribution underneath.

Layout::

    ┌──────────────────────────────────────────────────────────────────────┐
    │ /data/plate1/1                             [Choose folder…] [Load]   │
    │ Fields [3]                                             [Compare]     │
    ├───────────────────────────────┬──────────────────────────────────────┤
    │ Model A                       │ Model B                              │
    │ model     [cpsam         ]    │ model     [cpsam            ]        │
    │ diameter  [30.0]              │ diameter  [60.0]                     │
    │ flow / cellprob / min size …  │ …                                    │
    ├───────────────────────────────┴──────────────────────────────────────┤
    │ ! B: diam_mean=17 is ignored — use diameter=, that one rescales.     │
    ├──────────────────────────────────────────────────────────────────────┤
    │ parameter          A       B        reaches the model?               │
    │ diameter           30.0    60.0     yes  ←  varied                   │
    │ diam_mean          –       17       no   ←  ignored by Cellpose 4    │
    ├──────────────────────────────────────────────────────────────────────┤
    │ field  A obj  B obj  Δ   ARI   matched  splits  merges  only B …     │
    ├───────────────────────────────┬──────────────────────────────────────┤
    │  [ image + A masks ]          │  [ image + B masks ]                 │
    └───────────────────────────────┴──────────────────────────────────────┘

Design notes:

* **The resolved-parameter table is not decoration.** Cellpose 4 accepts
  ``model_type``, ``diam_mean``, ``nchan``, ``channels`` and ``rescale`` and
  then ignores every one of them, and it resolves all the pre-SAM model names
  to ``cpsam``. A comparison that differs only in one of those reports "no
  difference", which reads as "the models are equivalent" rather than "you
  changed nothing". So the screen shows what actually reached each model,
  marks what was dropped, and says so in a banner above the numbers.
* **Off the GUI thread.** Segmentation is minutes, not milliseconds, so the run
  goes through :func:`spacr.qt.bridge.make_thread` like every other spaCR job.
  Tests pass ``threaded=False``, which runs the same code inline.
* **No modal dialogs on any error path.** A folder with no images, a model that
  will not load, a field of the wrong shape — all of it lands in the inline
  status label. A QMessageBox hangs a headless run.
* **The preview colours by correspondence, not by label id.** Label 3 in A has
  nothing to do with label 3 in B, so a shared random palette would invite
  exactly the wrong comparison. Objects that have a partner in the other mask
  are drawn in one colour and objects that do not in another, which is the
  comparison rather than a decoration of it.
* **Neither model is called correct.** The screen renders
  :mod:`spacr.model_compare`'s directional wording as-is.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ... import model_compare as mc
from ..bridge import make_thread
from ..theme import PALETTE, SPACING
from ..widgets import Divider

__all__ = ["ModelCompareScreen", "FIELD_RANGE", "PREVIEW_PX"]


#: How many fields the screen will compare. Three is the default because three
#: is what a human actually looks at; the ceiling keeps an accidental "all of
#: them" from turning an interactive question into an overnight run.
FIELD_RANGE = (1, 25)

#: Edge of each mask preview, in pixels.
PREVIEW_PX = 360

#: RGB for objects that have a partner in the other mask, and for those that do
#: not. Neither colour means "right": unmatched is a fact about the pair.
COLOUR_MATCHED = (46, 196, 182)
COLOUR_UNMATCHED = (235, 165, 60)

_ROW_HEADERS = ("field", "A objects", "B objects", "Δ", "ARI", "matched",
                "mean IoU", "splits", "merges", "only B", "only A",
                "A qc", "B qc")

_PARAM_HEADERS = ("parameter", "A", "B", "reaches the model?")


def _cell(text: str) -> QTableWidgetItem:
    """A read-only table cell."""
    item = QTableWidgetItem(text)
    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
    return item


def parse_extra(text: str) -> Dict[str, Any]:
    """Parse a ``key=value, key=value`` line into eval keyword arguments.

    Numbers and booleans are coerced so ``diam_mean=17`` arrives as a number
    rather than the string ``"17"`` — the report compares values, and ``"17"``
    and ``17`` would look like a difference where there is none.

    :param text: the raw line; blank returns ``{}``.
    :returns: the parsed mapping.
    :raises ValueError: on a fragment with no ``=`` in it, naming the fragment.
    """
    out: Dict[str, Any] = {}
    for chunk in str(text or "").replace(";", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(
                f"{chunk!r} is not a key=value pair — write e.g. "
                f"diam_mean=17, channels=[2,0]")
        key, _, raw = chunk.partition("=")
        out[key.strip()] = _coerce(raw.strip())
    return out


def _coerce(raw: str) -> Any:
    """Turn a settings fragment into a number, bool, None or string."""
    low = raw.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null", ""):
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


class _ModelPanel(QGroupBox):
    """The settings for one side of the comparison.

    Only the arguments Cellpose 4 actually reads get a widget. Anything else a
    user wants to try goes in the free-text ``extra`` line, where the report
    will pick it up and tell them whether it does anything.
    """

    def __init__(self, title: str, parent: Optional[QWidget] = None,
                 diameter: float = 30.0):
        super().__init__(title, parent)
        form = QFormLayout(self)
        form.setContentsMargins(SPACING["sm"], SPACING["md"],
                                SPACING["sm"], SPACING["sm"])
        form.setSpacing(SPACING["xs"])

        self.model_edit = QLineEdit(mc.DEFAULT_MODEL, self)
        self.model_edit.setToolTip(
            "(str) 'cpsam', or a path to a custom Cellpose checkpoint. Every "
            "pre-SAM name (cyto, cyto2, cyto3, nuclei…) resolves to cpsam.")
        form.addRow("model", self.model_edit)

        self.diameter_box = QDoubleSpinBox(self)
        self.diameter_box.setRange(0.0, 1000.0)
        self.diameter_box.setDecimals(1)
        self.diameter_box.setValue(float(diameter))
        self.diameter_box.setSpecialValueText("native")
        self.diameter_box.setToolTip(
            "(float) Expected object diameter in px. The one size argument "
            "Cellpose 4 still acts on: it rescales the image by 30/diameter. "
            "0 = 'native' leaves the image alone.")
        form.addRow("diameter", self.diameter_box)

        self.flow_box = QDoubleSpinBox(self)
        self.flow_box.setRange(0.0, 10.0)
        self.flow_box.setDecimals(2)
        self.flow_box.setSingleStep(0.1)
        self.flow_box.setValue(0.4)
        self.flow_box.setToolTip("(float) Flow-error cutoff.")
        form.addRow("flow_threshold", self.flow_box)

        self.cellprob_box = QDoubleSpinBox(self)
        self.cellprob_box.setRange(-12.0, 12.0)
        self.cellprob_box.setDecimals(2)
        self.cellprob_box.setSingleStep(0.5)
        self.cellprob_box.setValue(0.0)
        self.cellprob_box.setToolTip("(float) Mask-probability cutoff.")
        form.addRow("cellprob_threshold", self.cellprob_box)

        self.min_size_box = QSpinBox(self)
        self.min_size_box.setRange(0, 100000)
        self.min_size_box.setValue(15)
        self.min_size_box.setToolTip("(int) Objects smaller than this are dropped.")
        form.addRow("min_size", self.min_size_box)

        self.normalize_box = QCheckBox("normalize", self)
        self.normalize_box.setChecked(True)
        self.normalize_box.setToolTip(
            "(bool) Percentile-normalise each image inside Cellpose.")
        form.addRow("", self.normalize_box)

        self.resample_box = QCheckBox("resample", self)
        self.resample_box.setChecked(True)
        self.resample_box.setToolTip(
            "(bool) Run the dynamics at full resolution.")
        form.addRow("", self.resample_box)

        self.extra_edit = QLineEdit("", self)
        self.extra_edit.setPlaceholderText("diam_mean=17, channels=[2,0]")
        self.extra_edit.setToolTip(
            "(str) Any other eval keyword, as key=value pairs. Arguments "
            "Cellpose 4 ignores are reported rather than passed on.")
        form.addRow("extra", self.extra_edit)

    def config(self, name: str) -> mc.ModelConfig:
        """Build a :class:`spacr.model_compare.ModelConfig` from the widgets.

        :param name: the label this side carries into the report.
        :raises ValueError: when the ``extra`` line does not parse.
        """
        diameter = float(self.diameter_box.value())
        return mc.ModelConfig(
            name=name,
            model=self.model_edit.text().strip() or mc.DEFAULT_MODEL,
            diameter=diameter if diameter > 0 else None,
            flow_threshold=float(self.flow_box.value()),
            cellprob_threshold=float(self.cellprob_box.value()),
            normalize=bool(self.normalize_box.isChecked()),
            resample=bool(self.resample_box.isChecked()),
            min_size=int(self.min_size_box.value()),
            extra=parse_extra(self.extra_edit.text()),
        )

    def set_enabled(self, enabled: bool) -> None:
        for widget in (self.model_edit, self.diameter_box, self.flow_box,
                       self.cellprob_box, self.min_size_box,
                       self.normalize_box, self.resample_box, self.extra_edit):
            widget.setEnabled(enabled)


class ModelCompareScreen(QWidget):
    """Compare two segmentation models on the same handful of fields.

    :param parent: Qt parent.
    :param threaded: run the segmentation on a worker thread (the default).
        Tests pass ``False`` for deterministic, synchronous behaviour.
    :ivar last_error: text of the most recent failure, ``""`` when the last
        operation succeeded. Errors are only ever reported here and in the
        inline status label — never in a modal dialog.
    """

    #: emitted with the resolved folder whenever fields load
    fields_loaded = Signal(str, int)
    #: emitted after every comparison settles (ok or not)
    job_finished = Signal(bool)

    def __init__(self, parent=None, threaded: bool = True):
        super().__init__(parent)
        self._threaded = bool(threaded)
        self._folder: str = ""
        self._field_names: List[str] = []
        self._images: List[np.ndarray] = []
        self._report: Optional[mc.ComparisonReport] = None
        self._segment_fn: Optional[Callable] = None
        self._busy = False
        # Ownership list for in-flight (QThread, worker) pairs — a QThread
        # collected while still running takes the process down with it. Same
        # idiom as AgreementScreen._jobs.
        self._jobs: List[tuple] = []
        self.last_error: str = ""

        self._build_ui()
        self._set_status(
            "Choose a folder of fields, configure both models, then Compare. "
            "Neither model is treated as ground truth.")
        self._update_controls()

    # -- construction ------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACING["lg"], SPACING["lg"],
                                 SPACING["lg"], SPACING["lg"])
        outer.setSpacing(SPACING["md"])

        title = QLabel("Model Compare")
        title.setObjectName("DisplayHeading")
        outer.addWidget(title)

        subtitle = QLabel(
            "Run two Cellpose models over the same fields and see what "
            "changed: object counts, the background-excluded ARI, and whether "
            "the extra objects are new cells or fragments of old ones.")
        subtitle.setObjectName("Muted")
        subtitle.setWordWrap(True)
        outer.addWidget(subtitle)

        outer.addWidget(Divider())

        # ── source row ────────────────────────────────────────────────
        src_row = QHBoxLayout()
        src_row.setSpacing(SPACING["sm"])
        self._path_edit = QLineEdit(self)
        self._path_edit.setPlaceholderText(
            "…/plate1/1  — a folder of .tif / .png / .npy / .npz fields")
        self._path_edit.setClearButtonEnabled(True)
        self._path_edit.returnPressed.connect(self._on_open_typed_path)
        self._btn_pick = QPushButton("Choose folder…", self)
        self._btn_pick.clicked.connect(self._pick_folder)
        self._btn_load = QPushButton("Load", self)
        self._btn_load.clicked.connect(self._on_open_typed_path)
        src_row.addWidget(self._path_edit, 1)
        src_row.addWidget(self._btn_pick)
        src_row.addWidget(self._btn_load)
        src_row.addWidget(QLabel("Fields", self))
        self._fields_box = QSpinBox(self)
        self._fields_box.setRange(*FIELD_RANGE)
        self._fields_box.setValue(mc.DEFAULT_N_FIELDS)
        self._fields_box.setToolTip(
            "(int) How many fields to segment with each model.")
        self._fields_box.valueChanged.connect(lambda *_: self._reload())
        src_row.addWidget(self._fields_box)
        outer.addLayout(src_row)

        # ── the two model panels ──────────────────────────────────────
        panels = QHBoxLayout()
        panels.setSpacing(SPACING["md"])
        self._panel_a = _ModelPanel("Model A", self, diameter=30.0)
        self._panel_b = _ModelPanel("Model B", self, diameter=60.0)
        panels.addWidget(self._panel_a, 1)
        panels.addWidget(self._panel_b, 1)
        outer.addLayout(panels)

        run_row = QHBoxLayout()
        run_row.addStretch(1)
        self._btn_compare = QPushButton("Compare", self)
        self._btn_compare.setObjectName("PrimaryButton")
        self._btn_compare.clicked.connect(self.compare)
        run_row.addWidget(self._btn_compare)
        outer.addLayout(run_row)

        # ── warnings ──────────────────────────────────────────────────
        self._warnings = QLabel("", self)
        self._warnings.setWordWrap(True)
        self._warnings.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._warnings.setVisible(False)
        outer.addWidget(self._warnings)

        # ── resolved parameters ───────────────────────────────────────
        outer.addWidget(QLabel("Parameters that reached each model", self))
        self._param_table = QTableWidget(0, len(_PARAM_HEADERS), self)
        self._param_table.setHorizontalHeaderLabels(list(_PARAM_HEADERS))
        self._prepare_table(self._param_table)
        self._param_table.setMaximumHeight(200)
        outer.addWidget(self._param_table)

        # ── per-field metrics ─────────────────────────────────────────
        outer.addWidget(QLabel("Per-field comparison", self))
        self._row_table = QTableWidget(0, len(_ROW_HEADERS), self)
        self._row_table.setHorizontalHeaderLabels(list(_ROW_HEADERS))
        self._prepare_table(self._row_table)
        self._row_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._row_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._row_table.currentCellChanged.connect(
            lambda row, *_: self.select_field(row))
        outer.addWidget(self._row_table, 1)

        self._summary = QLabel("", self)
        self._summary.setWordWrap(True)
        self._summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        outer.addWidget(self._summary)

        outer.addWidget(Divider())

        # ── side-by-side masks ────────────────────────────────────────
        preview = QSplitter(Qt.Horizontal, self)
        self._preview_a, self._caption_a = self._build_preview(preview, "A")
        self._preview_b, self._caption_b = self._build_preview(preview, "B")
        preview.setSizes([600, 600])
        outer.addWidget(preview, 1)

        self._status = QLabel("", self)
        self._status.setObjectName("Muted")
        self._status.setWordWrap(True)
        self._status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        outer.addWidget(self._status)

    def _build_preview(self, parent: QSplitter, side: str):
        holder = QWidget(parent)
        layout = QVBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACING["xs"])
        caption = QLabel(f"Model {side}", holder)
        caption.setObjectName("Caption")
        caption.setWordWrap(True)
        layout.addWidget(caption)
        canvas = QLabel("", holder)
        canvas.setAlignment(Qt.AlignCenter)
        canvas.setMinimumSize(PREVIEW_PX, PREVIEW_PX)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        canvas.setStyleSheet(f"background: {PALETTE['bg']};")
        layout.addWidget(canvas, 1)
        parent.addWidget(holder)
        return canvas, caption

    @staticmethod
    def _prepare_table(table: QTableWidget) -> None:
        """Common read-only look for every result table."""
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setAlternatingRowColors(True)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        table.horizontalHeader().setStretchLastSection(True)

    # -- status ------------------------------------------------------------

    def _set_status(self, text: str, error: bool = False) -> None:
        """Report inline. Deliberately never a QMessageBox — a modal dialog
        would hang a headless run (and did, in MakeMasksScreen)."""
        self.last_error = text if error else ""
        colour = PALETTE["error"] if error else PALETTE["fg_muted"]
        self._status.setStyleSheet(f"color: {colour};")
        self._status.setText(text)

    def status_text(self) -> str:
        """Current inline status message (test/introspection helper)."""
        return self._status.text()

    def summary_text(self) -> str:
        """The aggregate summary line, or ``''`` before a report exists."""
        return self._summary.text()

    def warning_text(self) -> str:
        """The banner above the numbers, or ``''`` when there is nothing to say."""
        return self._warnings.text()

    # -- source ------------------------------------------------------------

    def _pick_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Choose a folder of fields", self._folder or os.getcwd())
        if path:
            self.set_source(path)

    def _on_open_typed_path(self) -> None:
        self.set_source(self._path_edit.text())

    def configure(self, model_a: str = "", model_b: str = "",
                  folder: str = "", n_fields: int = 0) -> bool:
        """Preload the screen with two models and a folder.

        The public route for another screen to hand this one a comparison —
        the Model Zoo's "compare the two selected" uses it. A caller reaching
        into ``_panel_a.model_edit`` would break the moment either panel is
        restructured.

        Every argument is optional; an empty one leaves that control alone.
        ``n_fields`` is applied BEFORE the folder so the reload the folder
        triggers already uses the right count, rather than loading N fields and
        immediately reloading with a different N.

        :param model_a: model name or checkpoint path for the left panel.
        :param model_b: same, for the right panel.
        :param folder: directory of fields to compare on.
        :param n_fields: how many fields; 0 keeps the current value.
        :returns: what :meth:`set_source` returned, or True when no folder was
            given.
        """
        if n_fields:
            self._fields_box.blockSignals(True)
            self._fields_box.setValue(int(n_fields))
            self._fields_box.blockSignals(False)
        if model_a:
            self._panel_a.model_edit.setText(str(model_a))
        if model_b:
            self._panel_b.model_edit.setText(str(model_b))
        if folder:
            return self.set_source(str(folder))
        return True

    def set_source(self, folder: str) -> bool:
        """Load the first N fields out of ``folder``.

        Every failure here is a normal state — a mistyped path, a folder of
        CSVs, an empty plate — so it lands in the status label and returns
        False. This never raises and never opens a dialog.

        :param folder: a directory of ``.tif`` / ``.png`` / ``.npy`` / ``.npz``
            fields.
        :returns: True when at least one field loaded.
        """
        self._clear_results()
        self._folder = ""
        self._field_names = []
        self._images = []
        try:
            names, images = mc.load_fields(folder,
                                           n_fields=int(self._fields_box.value()))
        except Exception as e:
            self._set_status(str(e) or e.__class__.__name__, error=True)
            self._update_controls()
            return False

        self._folder = os.fspath(folder)
        self._field_names = names
        self._images = images
        self._path_edit.setText(self._folder)
        self.fields_loaded.emit(self._folder, len(images))
        self._set_status(
            f"Loaded {len(images)} field(s) from {self._folder}: "
            f"{', '.join(names)}. Configure both models and press Compare.")
        self._update_controls()
        return True

    def _reload(self) -> None:
        """Re-read the folder after the field count changed."""
        if self._folder:
            self.set_source(self._folder)

    def source_folder(self) -> str:
        """The loaded folder, or ``''``."""
        return self._folder

    def field_names(self) -> List[str]:
        """The loaded field names, in order."""
        return list(self._field_names)

    def set_segment_fn(self, fn: Optional[Callable]) -> None:
        """Override the segmentation backend.

        The Model Zoo (and every test in this file) hands in its own callable
        rather than loading Cellpose; None restores
        :func:`spacr.model_compare.segment_with_cellpose`.

        :param fn: ``fn(images, config) -> masks``, or None.
        """
        self._segment_fn = fn

    # -- run ---------------------------------------------------------------

    def model_configs(self):
        """``(config_a, config_b)`` as currently configured.

        :raises ValueError: when either ``extra`` line does not parse.
        """
        return (self._panel_a.config("A"), self._panel_b.config("B"))

    def compare(self) -> bool:
        """Segment every loaded field with both models and fill the tables.

        :returns: for the synchronous path, whether a report was produced; for
            the threaded path, True once the job has started.
        """
        if not self._images:
            self._set_status("Load a folder of fields first.", error=True)
            return False
        if self._busy:
            self._set_status("A comparison is already running…", error=True)
            return False
        try:
            config_a, config_b = self.model_configs()
        except ValueError as e:
            self._set_status(str(e), error=True)
            return False

        images = list(self._images)
        names = list(self._field_names)
        segment_fn = self._segment_fn

        def _job() -> mc.ComparisonReport:
            return mc.compare_models(images, config_a, config_b,
                                     field_names=names, segment_fn=segment_fn)

        self._set_status(
            f"Segmenting {len(images)} field(s) with {config_a.name} and "
            f"{config_b.name}…")
        return self._run_job(_job, self._apply_result)

    def _apply_result(self, report: mc.ComparisonReport) -> None:
        self._report = report
        self._fill_param_table(report)
        self._fill_row_table(report)
        self._fill_warnings(report)
        self._summary.setText(report.summary)
        if report.comparisons:
            self._row_table.setCurrentCell(0, 0)
            self.select_field(0)
        self._set_status(
            f"Compared {report.n_fields} field(s): "
            f"{report.total_objects_a} vs {report.total_objects_b} "
            f"{report.object_type}(s) "
            f"({report.object_count_delta:+d}), mean ARI "
            f"{mc._fmt(report.mean_ari)}. "
            f"{report.model_a.name} took {report.seconds_a:.1f}s, "
            f"{report.model_b.name} {report.seconds_b:.1f}s.")

    def report(self) -> Optional[mc.ComparisonReport]:
        """The most recent :class:`~spacr.model_compare.ComparisonReport`."""
        return self._report

    # -- rendering ---------------------------------------------------------

    def _clear_results(self) -> None:
        self._report = None
        self._param_table.setRowCount(0)
        self._row_table.setRowCount(0)
        self._summary.setText("")
        self._warnings.setText("")
        self._warnings.setVisible(False)
        for canvas, caption, side in ((self._preview_a, self._caption_a, "A"),
                                      (self._preview_b, self._caption_b, "B")):
            canvas.setPixmap(QPixmap())
            canvas.setText("")
            caption.setText(f"Model {side}")

    def _fill_warnings(self, report: mc.ComparisonReport) -> None:
        if not report.warnings:
            self._warnings.setText("")
            self._warnings.setVisible(False)
            return
        colour = PALETTE["warning"]
        self._warnings.setStyleSheet(f"color: {colour};")
        self._warnings.setText("<br>".join(f"! {w}" for w in report.warnings))
        self._warnings.setVisible(True)

    def _fill_param_table(self, report: mc.ComparisonReport) -> None:
        """Show every parameter, honoured and ignored, with which is which.

        This is the table that stops a run from silently comparing a model with
        itself: an argument Cellpose 4 drops is listed with "no — ignored by
        Cellpose 4" beside it rather than being invisible.
        """
        honoured_a = report.model_a.honoured_parameters()
        honoured_b = report.model_b.honoured_parameters()
        ignored_a = report.model_a.ignored_parameters()
        ignored_b = report.model_b.ignored_parameters()

        rows: List[List[str]] = []
        for key in sorted(set(honoured_a) | set(honoured_b)):
            a, b = honoured_a.get(key), honoured_b.get(key)
            note = "yes — varied by this run" if a != b else "yes"
            rows.append([key, mc._value(a), mc._value(b), note])
        for key in sorted(set(ignored_a) | set(ignored_b)):
            reason = mc.IGNORED_ARGUMENTS.get(
                key, "every pre-SAM name resolves to cpsam: Cellpose 4 "
                     "ships one model")
            # 'model' appears in both halves — what was asked for and what will
            # load. Two rows with the same label would read as a contradiction.
            label = "model (requested)" if key == "model" else key
            rows.append([label, mc._value(ignored_a.get(key)),
                         mc._value(ignored_b.get(key)),
                         f"no — {reason}"])

        table = self._param_table
        table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, text in enumerate(row):
                item = _cell(text)
                if row[3].startswith("no"):
                    item.setForeground(_brush(PALETTE["warning"]))
                    item.setToolTip(row[3])
                elif row[3].endswith("varied by this run"):
                    item.setForeground(_brush(PALETTE["accent"]))
                table.setItem(r, c, item)
        table.resizeColumnsToContents()

    def parameter_rows(self) -> List[List[str]]:
        """The resolved-parameter table as plain strings."""
        return _table_rows(self._param_table)

    def _fill_row_table(self, report: mc.ComparisonReport) -> None:
        table = self._row_table
        table.blockSignals(True)
        table.setRowCount(len(report.comparisons))
        for r, c in enumerate(report.comparisons):
            cells = (
                c.field,
                f"{c.n_objects_a}",
                f"{c.n_objects_b}",
                f"{c.object_count_delta:+d}",
                mc._fmt(c.ari),
                mc._fmt(c.iou_matched_fraction, pct=True),
                mc._fmt(c.mean_matched_iou),
                f"{c.split_events}",
                f"{c.merge_events}",
                f"{c.new_objects_b}",
                f"{c.missing_objects_a}",
                c.qc_a.severity if c.qc_a else "-",
                c.qc_b.severity if c.qc_b else "-",
            )
            for col, text in enumerate(cells):
                item = _cell(text)
                item.setToolTip(c.note)
                table.setItem(r, col, item)
        table.blockSignals(False)
        table.resizeColumnsToContents()

    def metric_rows(self) -> List[List[str]]:
        """The per-field table as plain strings."""
        return _table_rows(self._row_table)

    # -- preview -----------------------------------------------------------

    def select_field(self, row: int) -> bool:
        """Draw field ``row``'s two masks side by side over the same image.

        :param row: index into the per-field table.
        :returns: True when both panels rendered.
        """
        report = self._report
        if report is None or not (0 <= row < len(report.comparisons)):
            return False
        comparison = report.comparisons[row]
        if row >= len(report.masks_a) or row >= len(report.masks_b):
            self._caption_a.setText(
                "Masks were not kept for this run, so there is nothing to draw.")
            return False

        image = report.images[row] if row < len(report.images) else None
        matched_a = {m[0] for m in comparison.matches}
        matched_b = {m[1] for m in comparison.matches}
        ok_a = self._draw(self._preview_a, image, report.masks_a[row], matched_a)
        ok_b = self._draw(self._preview_b, image, report.masks_b[row], matched_b)
        self._caption_a.setText(
            f"{report.model_a.name} — {comparison.field}: "
            f"{comparison.n_objects_a} object(s), "
            f"{len(matched_a)} with a partner in B")
        self._caption_b.setText(
            f"{report.model_b.name} — {comparison.field}: "
            f"{comparison.n_objects_b} object(s), "
            f"{len(matched_b)} with a partner in A")
        return ok_a and ok_b

    def _draw(self, canvas: QLabel, image: Optional[np.ndarray],
              mask: np.ndarray, matched: set) -> bool:
        """Compose one panel and put it on ``canvas``."""
        composed = compose_overlay(image, mask, matched)
        if composed is None:
            canvas.setPixmap(QPixmap())
            canvas.setText("Nothing to draw for this field.")
            return False
        height, width = composed.shape[:2]
        qimage = QImage(composed.tobytes(), width, height, 3 * width,
                        QImage.Format_RGB888).copy()
        canvas.setText("")
        canvas.setPixmap(QPixmap.fromImage(qimage).scaled(
            max(PREVIEW_PX, canvas.width()), max(PREVIEW_PX, canvas.height()),
            Qt.KeepAspectRatio, Qt.SmoothTransformation))
        return True

    def preview_sizes(self):
        """``(a, b)`` pixmap sizes — ``(0, 0)`` for a panel with no image."""
        out = []
        for canvas in (self._preview_a, self._preview_b):
            pixmap = canvas.pixmap()
            out.append((0, 0) if pixmap is None or pixmap.isNull()
                       else (pixmap.width(), pixmap.height()))
        return tuple(out)

    def preview_captions(self):
        """``(a, b)`` caption strings under the two panels."""
        return (self._caption_a.text(), self._caption_b.text())

    # -- job plumbing ------------------------------------------------------

    def _run_job(self, fn: Callable[[], Any],
                 on_done: Callable[[Any], None]) -> bool:
        """Run ``fn`` off the GUI thread and hand its result to ``on_done``.

        Mirrors ``AgreementScreen._run_job``: one threading idiom for the whole
        Qt layer, and ``threaded=False`` runs inline while firing the same
        signals, so both paths behave identically from outside.
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

        def _job(payload: Dict[str, Any]) -> None:
            payload["result"] = fn()

        thread, worker = make_thread(_job, box)
        self._jobs.append((thread, worker))
        worker.error.connect(self._on_worker_error_text)

        def _finished(ok: bool) -> None:
            self._busy = False
            if ok:
                try:
                    on_done(box.get("result"))
                except Exception as e:
                    self._on_job_error(e)
                    ok = False
            self._update_controls()
            self.job_finished.emit(ok)

        worker.finished.connect(_finished)
        thread.finished.connect(lambda t=thread: self._retire_job(t))
        self._busy = True
        self._update_controls()
        thread.start()
        return True

    def _retire_job(self, thread) -> None:
        """Release *this* job's refs once its own event loop has exited."""
        self._jobs = [(t, w) for (t, w) in self._jobs if t is not thread]

    def active_jobs(self) -> int:
        """How many comparison threads are still winding down."""
        return len(self._jobs)

    def is_busy(self) -> bool:
        return self._busy

    def _on_worker_error_text(self, tb: str) -> None:
        """Turn a worker traceback into one inline line (never a dialog)."""
        line = ""
        for candidate in reversed(str(tb).strip().splitlines()):
            if candidate.strip():
                line = candidate.strip()
                break
        self._clear_results()
        self._set_status(f"Comparison failed: {line}", error=True)

    def _on_job_error(self, exc: Exception) -> None:
        self._clear_results()
        self._set_status(f"Comparison failed: {exc}", error=True)

    # -- enablement --------------------------------------------------------

    def _update_controls(self) -> None:
        loaded = bool(self._images)
        self._btn_compare.setEnabled(loaded and not self._busy)
        self._btn_load.setEnabled(not self._busy)
        self._btn_pick.setEnabled(not self._busy)
        self._fields_box.setEnabled(not self._busy)
        self._panel_a.set_enabled(not self._busy)
        self._panel_b.set_enabled(not self._busy)

    # -- shutdown ----------------------------------------------------------

    def closeEvent(self, event):  # noqa: N802
        """Let every in-flight comparison thread finish before the widget dies."""
        for thread, _worker in list(self._jobs):
            try:
                if thread.isRunning():
                    thread.quit()
                    thread.wait(5000)
            except RuntimeError:
                pass
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# drawing helpers — plain numpy, so they are testable without a widget
# ---------------------------------------------------------------------------

def to_display_gray(image: Optional[np.ndarray], shape) -> np.ndarray:
    """Reduce any field to a uint8 grayscale of ``shape``, for a backdrop.

    Multi-channel fields are collapsed to their max projection rather than a
    mean: a nucleus channel averaged with an empty channel is a dim nucleus.
    Contrast is a 1-99.9 percentile stretch, the same window
    :func:`spacr.qt.mask_engine.normalize_uint16` uses. A missing or
    wrong-shaped image comes back black rather than raising — the masks are
    what the screen is really showing.

    :param image: the field, or None.
    :param shape: ``(height, width)`` the mask expects.
    :returns: a uint8 array of ``shape``.
    """
    height, width = int(shape[0]), int(shape[1])
    if image is None:
        return np.zeros((height, width), dtype=np.uint8)
    array = np.asarray(image)
    while array.ndim > 2:
        array = array.max(axis=-1)
    if array.shape[:2] != (height, width):
        return np.zeros((height, width), dtype=np.uint8)
    array = array.astype(np.float64)
    if not np.isfinite(array).all():
        array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    low, high = np.percentile(array, 1.0), np.percentile(array, 99.9)
    if high <= low:
        high = low + 1.0
    return (np.clip((array - low) / (high - low), 0, 1) * 255).astype(np.uint8)


def compose_overlay(image: Optional[np.ndarray], mask: Any,
                    matched: Optional[Sequence[int]] = None,
                    alpha: float = 0.45) -> Optional[np.ndarray]:
    """Blend a mask over its image, colouring by correspondence.

    Objects listed in ``matched`` (they have a partner in the other model's
    mask) are drawn in one colour and the rest in another. Colouring by label id
    would be worse than useless here: label 3 in A has nothing to do with label
    3 in B, so a shared palette invites the eye to compare things that are not
    the same object.

    :param image: the field, or None for a black backdrop.
    :param mask: a 2-D label image.
    :param matched: labels of ``mask`` that have a partner; None colours
        everything as unmatched.
    :param alpha: overlay strength.
    :returns: a uint8 ``(H, W, 3)`` RGB array, or None when the mask is not a
        2-D label image.
    """
    labels = np.squeeze(np.asarray(mask))
    if labels.ndim != 2 or labels.size == 0:
        return None
    labels = np.rint(labels).astype(np.int64)
    if labels.min() < 0:
        return None

    gray = to_display_gray(image, labels.shape)
    rgb = np.stack((gray,) * 3, axis=-1).astype(np.float32)

    top = int(labels.max())
    palette = np.zeros((top + 1, 3), dtype=np.float32)
    palette[1:] = np.array(COLOUR_UNMATCHED, dtype=np.float32)
    for label in (matched or ()):
        index = int(label)
        if 1 <= index <= top:
            palette[index] = COLOUR_MATCHED
    palette[0] = 0.0

    coloured = palette[labels]
    foreground = labels > 0
    out = np.where(foreground[..., None],
                   rgb * (1.0 - alpha) + coloured * alpha, rgb)
    return np.clip(out, 0, 255).astype(np.uint8)


def _brush(colour: str):
    """A QBrush for a palette colour string."""
    from PySide6.QtGui import QBrush, QColor

    return QBrush(QColor(colour))


def _table_rows(table: QTableWidget) -> List[List[str]]:
    """Every cell of ``table`` as plain strings."""
    return [[(table.item(r, c).text() if table.item(r, c) else "")
             for c in range(table.columnCount())]
            for r in range(table.rowCount())]

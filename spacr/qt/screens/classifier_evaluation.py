"""Classifier Evaluation workbench for out-of-fold result bundles.

Discovery and CSV/JSON parsing run away from the GUI thread.  The screen only
renders already-loaded data, so a large OOF prediction table cannot freeze the
application while it is being read.

``C8`` — the confusion matrix is a query, not a picture
-------------------------------------------------------

Every cell of the matrix is clickable, and clicking one asks
:func:`spacr.qt.linked_selection.open_objects` for exactly the crops that cell
counted — so "43 uninfected called infected" stops being a number and becomes
43 images you can look at. The analysis behind it is in :mod:`spacr.confusion`,
which has no Qt in it; this file is the table, the two lists and the buttons.

Two lists, not one, per cell. The model being **sure** and wrong is evidence
against the *annotation*; the model being **unsure** and wrong is evidence
about the *boundary*. They are different diagnoses with different fixes, so
they are opened separately and never blended into one confidence-sorted list
where the distinction disappears somewhere in the middle of a scroll.

And before either: where did the errors come from. A cell broken down per well
and per plate answers the question that makes re-labelling worth doing at all
— 43 errors from 20 wells is the model's problem, 43 errors from one well is
the bench's.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDesktopServices, QDragEnterEvent, QDropEvent
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ... import confusion as cx
from ...classifier_evaluation import (
    find_evaluation_bundles,
    load_evaluation_bundle,
)
from ..bridge import make_thread
from ..iconset import icon
from ..i18n import tr
from ..linked_selection import (DEFAULT_OPEN_KIND, has_object_opener,
                                open_objects)
from ..theme import SPACING, active_palette
from ..widgets import Divider

LOG = logging.getLogger(__name__)

__all__ = [
    "ClassifierEvaluationScreen",
    "APP_KEY",
    "APP_NAME",
    "APP_SECTION",
    "APP_INTRO",
    "LINK_SOURCE",
]

#: What this screen calls itself when it routes objects somewhere, so the
#: receiving grid's header can say where the crops came from.
LINK_SOURCE = "classifier_evaluation"

#: How many objects a cell list shows before it stops adding rows. The list is
#: a preview of an ordering, not the work surface — the work surface is the
#: crop grid the button opens, and rendering ninety thousand QListWidgetItems
#: to look at the top twelve is how a click becomes a two-second freeze.
LIST_PREVIEW = 200

APP_KEY = "classifier_evaluation"
APP_NAME = "Classifier Evaluation"
APP_SECTION = "Results & QC"
APP_INTRO = (
    "Inspect held-out predictions, grouped or nested cross-validation, "
    "calibration, confusion matrices, per-plate performance and leakage checks."
)


class _DropPathEdit(QLineEdit):
    """A path field that accepts one dropped folder or manifest."""

    path_dropped = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        """Accept local file/folder URLs."""
        if any(url.isLocalFile() for url in event.mimeData().urls()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        """Place the first dropped local path in the field."""
        paths = [
            url.toLocalFile()
            for url in event.mimeData().urls()
            if url.isLocalFile()
        ]
        if paths:
            self.setText(paths[0])
            self.path_dropped.emit(paths[0])
            event.acceptProposedAction()
        else:
            event.ignore()


def _item(value: Any) -> QTableWidgetItem:
    """Return a read-only table item."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        text = ""
    elif isinstance(value, float):
        text = f"{value:.5g}"
    else:
        text = str(value)
    item = QTableWidgetItem(text)
    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
    return item


class ClassifierEvaluationScreen(QWidget):
    """Browse and inspect classifier evaluation bundles.

    :param parent: Qt parent.
    :param threaded: discover and parse bundles in a worker thread. Tests may
        pass ``False`` to make refresh deterministic.
    :ivar last_error: latest non-fatal source or parsing error.
    :ivar bundles: manifests found by the most recent scan.
    """

    evaluation_loaded = Signal(str)

    def __init__(self, parent=None, threaded: bool = True):
        super().__init__(parent)
        self._threaded = bool(threaded)
        self._busy = False
        self._jobs: List[tuple] = []
        self._pending_error = ""
        self._pending_bundles: Optional[List[Path]] = None
        self._pending_bundle: Optional[Dict[str, Any]] = None
        self.bundles: List[Path] = []
        self.bundle: Optional[Dict[str, Any]] = None
        self.last_error = ""
        #: the confusion cell currently being inspected, or ``None``. Held so
        #: that moving the confidence threshold re-splits *this* cell rather
        #: than needing the user to click it again.
        self._cell: Optional[cx.ConfusionCell] = None
        #: the counts matrix the cells are read out of, for tests and for the
        #: ranking line.
        self._counts = pd.DataFrame()
        self._build_ui()
        self._set_status(
            tr("Choose or drop a classifier run folder, then select Scan.")
        )

    def _build_ui(self) -> None:
        """Construct source controls and tabbed evaluation views."""
        outer = QVBoxLayout(self)
        outer.setContentsMargins(
            SPACING["lg"], SPACING["lg"], SPACING["lg"], SPACING["lg"],
        )
        outer.setSpacing(SPACING["md"])

        title = QLabel(tr(APP_NAME), self)
        title.setObjectName("DisplayHeading")
        outer.addWidget(title)
        intro = QLabel(tr(APP_INTRO), self)
        intro.setObjectName("Muted")
        intro.setWordWrap(True)
        outer.addWidget(intro)
        outer.addWidget(Divider())

        source_row = QHBoxLayout()
        source_label = QLabel(tr("Results folder"), self)
        self._source = _DropPathEdit(self)
        self._source.setPlaceholderText(
            tr("Drop a run/evaluation folder or evaluation_manifest.json")
        )
        self._source.returnPressed.connect(self.scan)
        self._source.path_dropped.connect(lambda _path: self.scan())
        self._browse = QPushButton(tr("Browse…"), self)
        self._browse.setIcon(icon("folder"))
        self._browse.clicked.connect(self._choose_source)
        self._scan = QPushButton(tr("Scan"), self)
        self._scan.setObjectName("PrimaryButton")
        self._scan.setIcon(icon("redo"))
        self._scan.clicked.connect(self.scan)
        source_row.addWidget(source_label)
        source_row.addWidget(self._source, 1)
        source_row.addWidget(self._browse)
        source_row.addWidget(self._scan)
        outer.addLayout(source_row)

        bundle_row = QHBoxLayout()
        bundle_row.addWidget(QLabel(tr("Evaluation run"), self))
        self._bundle_choice = QComboBox(self)
        self._bundle_choice.currentIndexChanged.connect(
            self._load_selected_bundle,
        )
        self._open_folder = QPushButton(tr("Open folder"), self)
        self._open_folder.setIcon(icon("folder"))
        self._open_folder.clicked.connect(self._open_current_folder)
        self._copy_path = QPushButton(tr("Copy path"), self)
        self._copy_path.clicked.connect(self._copy_current_path)
        bundle_row.addWidget(self._bundle_choice, 1)
        bundle_row.addWidget(self._open_folder)
        bundle_row.addWidget(self._copy_path)
        outer.addLayout(bundle_row)

        self._tabs = QTabWidget(self)
        self._overview = QPlainTextEdit(self)
        self._overview.setReadOnly(True)
        confusion_page = self._build_confusion_page()
        self._per_plate = self._table()
        self._calibration = self._table()
        predictions_page = QWidget(self)
        predictions_layout = QVBoxLayout(predictions_page)
        predictions_layout.setContentsMargins(0, 0, 0, 0)
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel(tr("Filter predictions"), predictions_page))
        self._prediction_filter = QLineEdit(predictions_page)
        self._prediction_filter.setPlaceholderText(
            tr("Plate, well, class, filename…")
        )
        self._prediction_filter.setClearButtonEnabled(True)
        self._prediction_filter.textChanged.connect(
            self._render_predictions,
        )
        filter_row.addWidget(self._prediction_filter, 1)
        predictions_layout.addLayout(filter_row)
        self._predictions = self._table()
        predictions_layout.addWidget(self._predictions, 1)
        self._leakage = QPlainTextEdit(self)
        self._leakage.setReadOnly(True)
        for label, widget in (
            (tr("Summary"), self._overview),
            (tr("Confusion matrix"), confusion_page),
            (tr("Per-plate metrics"), self._per_plate),
            (tr("Calibration"), self._calibration),
            (tr("Predictions"), predictions_page),
            (tr("Leakage audit"), self._leakage),
        ):
            self._tabs.addTab(widget, label)
        outer.addWidget(self._tabs, 1)

        self._status = QLabel("", self)
        self._status.setObjectName("Muted")
        self._status.setWordWrap(True)
        outer.addWidget(self._status)
        self._clear_bundle()

    def _table(self) -> QTableWidget:
        """Return a read-only, row-selecting data table."""
        table = QTableWidget(0, 0, self)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setAlternatingRowColors(True)
        table.verticalHeader().setVisible(False)
        return table

    # ------------------------------------------------------------------
    # C8 — the confusion matrix as a set of live queries
    # ------------------------------------------------------------------
    def _build_confusion_page(self) -> QWidget:
        """The matrix, the ranking in words, and the clicked cell's two lists.

        Selection is per *cell*, not per row: a confusion is a (true,
        predicted) pair and selecting whole rows would make "click the cell
        with 43 in it" ambiguous between the four cells beside it.
        """
        page = QWidget(self)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACING["sm"])

        self._confusion_ranking = QLabel("", page)
        self._confusion_ranking.setWordWrap(True)
        layout.addWidget(self._confusion_ranking)

        split = QSplitter(Qt.Vertical, page)

        self._confusion = self._table()
        self._confusion.setSelectionBehavior(QAbstractItemView.SelectItems)
        self._confusion.setSelectionMode(QAbstractItemView.SingleSelection)
        self._confusion.cellClicked.connect(self._on_confusion_cell)
        split.addWidget(self._confusion)

        inspector = QWidget(page)
        column = QVBoxLayout(inspector)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(SPACING["sm"])

        controls = QHBoxLayout()
        controls.addWidget(QLabel(tr("The model counts as sure at"), inspector))
        # Where "sure" starts is a property of the assay, not of arithmetic —
        # see `spacr.confusion.confidence_threshold`. Exposed rather than
        # baked in, because the person reading the crops is the one who can
        # tell whether 0.75 is where their model stops guessing.
        self._threshold = QDoubleSpinBox(inspector)
        self._threshold.setRange(0.0, 1.0)
        self._threshold.setSingleStep(0.05)
        self._threshold.setDecimals(2)
        self._threshold.setValue(0.75)
        self._threshold.setToolTip(tr(
            "Errors at or above this confidence suggest a wrong label; below "
            "it, a boundary the model has not learned."))
        self._threshold.valueChanged.connect(self._on_threshold_changed)
        controls.addWidget(self._threshold)
        controls.addStretch(1)
        column.addLayout(controls)

        self._cell_summary = QLabel("", inspector)
        self._cell_summary.setWordWrap(True)
        column.addWidget(self._cell_summary)

        self._cell_breakdown = QLabel("", inspector)
        self._cell_breakdown.setWordWrap(True)
        self._cell_breakdown.setObjectName("Muted")
        column.addWidget(self._cell_breakdown)

        lists = QHBoxLayout()
        lists.setSpacing(SPACING["md"])
        self._high_head, self._high_list, self._high_open = self._error_column(
            inspector, tr("Sure and wrong — suspect the label"),
            tr("Open these crops in Annotate, most confident first. If the "
               "model is this certain and disagrees, look at the annotation."),
            self._open_high)
        lists.addLayout(self._high_head)
        self._low_head, self._low_list, self._low_open = self._error_column(
            inspector, tr("Unsure and wrong — suspect the boundary"),
            tr("Open these crops in Annotate, least confident first. These sat "
               "on the decision boundary; the label is probably right."),
            self._open_low)
        lists.addLayout(self._low_head)
        column.addLayout(lists, 1)

        split.addWidget(inspector)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)
        layout.addWidget(split, 1)
        return page

    def _error_column(self, parent: QWidget, title: str, tip: str,
                      on_open) -> Tuple[QVBoxLayout, QListWidget, QPushButton]:
        """One of the two lists: a heading, the objects, and one button."""
        box = QVBoxLayout()
        box.setSpacing(4)
        heading = QLabel(title, parent)
        heading.setWordWrap(True)
        box.addWidget(heading)
        listing = QListWidget(parent)
        listing.setSelectionMode(QAbstractItemView.NoSelection)
        listing.setToolTip(tip)
        box.addWidget(listing, 1)
        button = QPushButton(tr("Open in Annotate"), parent)
        button.setToolTip(tip)
        button.setEnabled(False)
        button.clicked.connect(on_open)
        box.addWidget(button)
        # `heading` is kept on the layout only for the caller's convenience;
        # the layout itself is what gets added, so nothing here is orphaned.
        return box, listing, button

    def _choose_source(self) -> None:
        """Choose a folder containing one or more evaluation bundles."""
        path = QFileDialog.getExistingDirectory(
            self, tr("Choose classifier results folder"),
            self._source.text().strip(),
        )
        if path:
            self._source.setText(path)
            self.scan()

    def scan(self) -> None:
        """Discover evaluation bundles without blocking the GUI thread."""
        if self._busy:
            return
        source = self._source.text().strip()
        if not source:
            self._set_status(tr("Choose a results folder first."), error=True)
            return
        self._start_busy(tr("Searching for classifier evaluations…"))
        self._pending_bundles = None
        self._pending_bundle = None
        self._pending_error = ""

        def _work(_settings):
            try:
                self._pending_bundles = find_evaluation_bundles(source)
            except Exception as exc:
                self._pending_error = f"{type(exc).__name__}: {exc}"
                LOG.exception("Classifier-evaluation scan failed")

        if not self._threaded:
            _work({})
            self._finish_scan(not bool(self._pending_error))
            return
        thread, worker = make_thread(
            _work, {}, app_key="classifier_evaluation_scan", journal=False,
        )
        self._keep_job(thread, worker, self._finish_scan)

    def _finish_scan(self, ok: bool) -> None:
        """Populate the run selector after discovery."""
        self._finish_busy()
        self.last_error = self._pending_error
        self.bundles = list(self._pending_bundles or []) if ok else []
        self._bundle_choice.blockSignals(True)
        self._bundle_choice.clear()
        root = Path(self._source.text().strip()).expanduser()
        for manifest in self.bundles:
            try:
                label = str(manifest.parent.relative_to(
                    root if root.is_dir() else root.parent,
                ))
            except ValueError:
                label = str(manifest.parent)
            self._bundle_choice.addItem(label or manifest.parent.name, manifest)
        self._bundle_choice.blockSignals(False)
        if self.last_error:
            self._clear_bundle()
            self._set_status(
                f"{tr('Could not scan classifier evaluations')}: "
                f"{self.last_error}",
                error=True,
            )
        elif not self.bundles:
            self._clear_bundle()
            self._set_status(
                tr("No evaluation_manifest.json files were found."),
                error=True,
            )
        else:
            self._set_status(
                f"{tr('Found')} {len(self.bundles)} "
                f"{tr('classifier evaluation(s).')}"
            )
            self._bundle_choice.setCurrentIndex(0)
            self._load_selected_bundle()

    def _load_selected_bundle(self, *_args) -> None:
        """Parse the selected bundle in a worker."""
        if self._busy:
            return
        manifest = self._bundle_choice.currentData()
        if not manifest:
            return
        self._start_busy(tr("Loading evaluation tables…"))
        self._pending_bundle = None
        self._pending_error = ""

        def _work(_settings):
            try:
                self._pending_bundle = load_evaluation_bundle(manifest)
            except Exception as exc:
                self._pending_error = f"{type(exc).__name__}: {exc}"
                LOG.exception("Classifier-evaluation load failed")

        if not self._threaded:
            _work({})
            self._finish_load(not bool(self._pending_error))
            return
        thread, worker = make_thread(
            _work, {}, app_key="classifier_evaluation_load", journal=False,
        )
        self._keep_job(thread, worker, self._finish_load)

    def _finish_load(self, ok: bool) -> None:
        """Render a loaded bundle on the GUI thread."""
        self._finish_busy()
        self.last_error = self._pending_error
        self.bundle = self._pending_bundle if ok else None
        if self.bundle is None:
            self._clear_bundle()
            self._set_status(
                f"{tr('Could not load classifier evaluation')}: "
                f"{self.last_error or tr('worker failed')}",
                error=True,
            )
            return
        summary = dict(self.bundle.get("summary") or {})
        manifest = dict(self.bundle.get("manifest") or {})
        self._overview.setPlainText(json.dumps({
            "results": str(self.bundle["path"]),
            "summary": summary,
            "bundle_warnings": manifest.get("warnings") or [],
        }, indent=2, sort_keys=True, default=str))
        self._render_confusion(summary)
        self._render_frame(self._per_plate, self.bundle["per_plate"])
        self._render_frame(self._calibration, self.bundle["calibration"])
        self._leakage.setPlainText(json.dumps(
            self.bundle.get("leakage") or {},
            indent=2, sort_keys=True, default=str,
        ))
        self._render_predictions()
        self._open_folder.setEnabled(True)
        self._copy_path.setEnabled(True)
        count = len(self.bundle["predictions"])
        passed = bool((self.bundle.get("leakage") or {}).get("passed", False))
        self._set_status(
            f"{tr('Loaded')} {count} {tr('held-out prediction(s)')} · "
            f"{tr('leakage audit')}: "
            f"{tr('passed') if passed else tr('review required')}."
        )
        self.evaluation_loaded.emit(str(self.bundle["path"]))

    def _render_frame(
        self,
        table: QTableWidget,
        frame: pd.DataFrame,
        *,
        include_index: bool = False,
        row_limit: Optional[int] = None,
    ) -> None:
        """Render a dataframe without mutating it."""
        shown = frame if row_limit is None else frame.head(row_limit)
        columns = list(shown.columns)
        if include_index:
            columns = [shown.index.name or "true_class", *columns]
        table.setSortingEnabled(False)
        table.clear()
        table.setColumnCount(len(columns))
        table.setHorizontalHeaderLabels([str(column) for column in columns])
        table.setRowCount(len(shown))
        for row, (index, values) in enumerate(shown.iterrows()):
            offset = 0
            if include_index:
                table.setItem(row, 0, _item(index))
                offset = 1
            for column, value in enumerate(values):
                table.setItem(row, column + offset, _item(value))
        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        if columns:
            header.setSectionResizeMode(len(columns) - 1, QHeaderView.Stretch)
        table.setSortingEnabled(True)

    def _render_predictions(self, *_args) -> None:
        """Filter and render at most 2,000 OOF rows."""
        if self.bundle is None:
            self._render_frame(self._predictions, pd.DataFrame())
            return
        frame = self.bundle["predictions"]
        terms = [
            term.casefold()
            for term in self._prediction_filter.text().split()
            if term
        ]
        if terms and not frame.empty:
            text = frame.astype(str).agg(" ".join, axis=1).str.casefold()
            mask = pd.Series(True, index=frame.index)
            for term in terms:
                mask &= text.str.contains(term, regex=False, na=False)
            frame = frame.loc[mask]
        self._render_frame(self._predictions, frame, row_limit=2000)

    # ------------------------------------------------------------------
    # C8 — rendering and routing
    # ------------------------------------------------------------------
    def _render_confusion(self, summary: Dict[str, Any]) -> None:
        """Draw the matrix in COUNTS, and say the ranking out loud.

        Counts, not the normalised shares the bundle also carries, because
        this table is now a set of buttons: "43" is what tells you whether a
        cell is worth opening, while "0.12" needs the row total to interpret.
        The share is on every cell's tooltip, so nothing is lost.

        Recomputed from the prediction table rather than read from
        ``confusion_counts.csv``. The two are the same number today, but only
        one of them can also list the objects — and a matrix that disagreed
        with the crops it opens is worse than no matrix.
        """
        predictions = self._predictions_frame()
        classes = summary.get("classes") or None
        try:
            counts = cx.confusion_counts(predictions, classes)
        except cx.ConfusionError as exc:
            LOG.info("cannot build a clickable confusion matrix: %s", exc)
            self._render_frame(self._confusion,
                               self.bundle["confusion_normalized"],
                               include_index=True)
            self._confusion_ranking.setText(
                f"{tr('This bundle cannot be broken down by object')}: {exc}")
            self._clear_cell()
            return
        self._counts = counts
        self._render_frame(self._confusion, counts, include_index=True)
        shares = counts.to_numpy(dtype=float)
        row_totals = shares.sum(axis=1)
        for row in range(self._confusion.rowCount()):
            for column in range(1, self._confusion.columnCount()):
                item = self._confusion.item(row, column)
                if item is None:
                    continue
                total = row_totals[row] if row < len(row_totals) else 0.0
                value = shares[row, column - 1] if row < len(shares) else 0.0
                item.setToolTip(
                    f"{int(value)} object(s) · {value / total:.1%} of this "
                    f"row" if total else f"{int(value)} object(s)")
        n_classes = max(2, len(counts.index))
        blocked = self._threshold.blockSignals(True)
        try:
            self._threshold.setValue(cx.confidence_threshold(n_classes))
        finally:
            self._threshold.blockSignals(blocked)
        self._confusion_ranking.setText(cx.describe_confusions(counts))
        self._clear_cell()

    def _predictions_frame(self) -> pd.DataFrame:
        """The loaded out-of-fold table, or an empty frame."""
        if self.bundle is None:
            return pd.DataFrame()
        frame = self.bundle.get("predictions")
        return frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()

    def _on_confusion_cell(self, row: int, column: int) -> None:
        """A cell was clicked: resolve it to (true, predicted) and inspect it.

        Read out of the widget rather than out of the frame by position. The
        table is sortable, so visual row 0 is not model row 0 after a header
        click — and a cell inspector that showed the wrong class after a sort
        would be worse than one that showed nothing.
        """
        if column <= 0:
            self._set_status(tr(
                "Click a cell rather than the class name: a confusion is a "
                "(true, predicted) pair."))
            return
        true_item = self._confusion.item(row, 0)
        header = self._confusion.horizontalHeaderItem(column)
        if true_item is None or header is None:
            return
        self.show_cell(true_item.text(), header.text())

    def show_cell(self, true_class: str, predicted_class: str) -> None:
        """Inspect one confusion cell. The seam a test (or a link) goes through.

        :returns: nothing; the two lists, the breakdown and the buttons are
            the result.
        """
        predictions = self._predictions_frame()
        if predictions.empty:
            self._clear_cell()
            return
        try:
            cell = cx.ConfusionCell.build(
                predictions, true_class, predicted_class,
                threshold=float(self._threshold.value()))
        except cx.ConfusionError as exc:
            self._clear_cell()
            self._cell_summary.setText(str(exc))
            return
        self._cell = cell
        self._cell_summary.setText(
            f"{cell.reason()} — {cell.describe()}")
        self._cell_breakdown.setText(self._describe_origin(cell))
        self._fill_list(self._high_list, cell.high, cell.threshold)
        self._fill_list(self._low_list, cell.low, cell.threshold)
        openable = has_object_opener(DEFAULT_OPEN_KIND)
        for button, frame, label in (
            (self._high_open, cell.high, tr("sure and wrong")),
            (self._low_open, cell.low, tr("unsure and wrong")),
        ):
            button.setEnabled(openable and not frame.empty)
            button.setText(
                f"{tr('Open')} {len(frame)} {label}"
                if not frame.empty else tr("Nothing to open"))
            if not openable:
                button.setToolTip(tr(
                    "Open the Annotate screen first — it is what shows crops."))

    def _describe_origin(self, cell: "cx.ConfusionCell") -> str:
        """Per-well and per-plate breakdown of one cell, both lines.

        Both levels, always, because they answer different questions and the
        interesting case is when they disagree: concentrated in one well but
        spread over plates is a pipetting error, concentrated in one plate but
        spread over its wells is an imaging session.
        """
        lines = []
        for level in ("well", "plate"):
            try:
                lines.append(f"{level}: {cx.describe_breakdown(cell.rows, level)}")
            except cx.ConfusionError as exc:
                lines.append(f"{level}: {exc}")
        return "\n".join(lines)

    def _fill_list(self, listing: QListWidget, frame: pd.DataFrame,
                   threshold: float) -> None:
        """Show the head of one ordered list, and say when it is a head."""
        listing.clear()
        if frame.empty:
            listing.addItem(QListWidgetItem(tr("(none)")))
            return
        try:
            column = cx.object_key_column(frame)
        except cx.ConfusionError:
            column = None
        shown = frame.head(LIST_PREVIEW)
        for _index, values in shown.iterrows():
            name = (str(values.get("basename") or values.get(column or "", ""))
                    if column else "")
            confidence = values.get(cx.CONFIDENCE_COLUMN)
            text = f"{float(confidence):.3f}  {name}" if pd.notna(
                confidence) else f"    ?    {name}"
            item = QListWidgetItem(text)
            item.setToolTip(str(values.get(column or "", name)))
            listing.addItem(item)
        if len(frame) > len(shown):
            listing.addItem(QListWidgetItem(
                f"… {len(frame) - len(shown)} {tr('more; open them to see all')}"))

    def _on_threshold_changed(self, _value: float) -> None:
        """Re-split the open cell where "sure" now starts."""
        if self._cell is not None:
            self.show_cell(self._cell.true_class, self._cell.predicted_class)

    def _open_high(self) -> Any:
        return self.open_cell("high")

    def _open_low(self) -> Any:
        return self.open_cell("low")

    def open_cell(self, which: str) -> Any:
        """Route one half of the open cell to whatever shows crops.

        Nothing here imports Annotate: the request travels through
        :func:`spacr.qt.linked_selection.open_objects`, so a second
        destination added later needs no change in this file.

        The per-key confidences ride along in ``context`` so the receiver can
        show *why* this order, and the threshold so it can say where the split
        was made.

        :returns: whatever the opener returned, or ``None`` when there was
            nothing to open or nowhere to open it.
        """
        cell = self._cell
        if cell is None:
            return None
        try:
            keys = cell.keys(which)
        except cx.ConfusionError as exc:
            self._set_status(str(exc), error=True)
            return None
        if not len(keys):
            self._set_status(tr("That list is empty."))
            return None
        frame = cell.high if which == "high" else cell.low
        scores = {}
        try:
            column = cx.object_key_column(frame)
            scores = {
                str(k): float(v)
                for k, v in zip(frame[column], frame[cx.CONFIDENCE_COLUMN])
                if pd.notna(v)
            }
        except Exception:
            LOG.debug("no per-key confidences to send with the request",
                      exc_info=True)
        try:
            return open_objects(
                keys, reason=cell.reason(which), source=LINK_SOURCE,
                context={"scores": scores, "threshold": cell.threshold,
                         "true_class": cell.true_class,
                         "predicted_class": cell.predicted_class,
                         "which": which})
        except Exception as exc:
            LOG.exception("Could not open a confusion cell")
            self._set_status(f"{tr('Could not open those crops')}: {exc}",
                             error=True)
            return None

    def _clear_cell(self) -> None:
        """No cell is open: say so rather than showing the last one's lists."""
        self._cell = None
        self._cell_summary.setText(tr(
            "Click a cell of the matrix to see the objects it counted."))
        self._cell_breakdown.setText("")
        for listing in (self._high_list, self._low_list):
            listing.clear()
        for button in (self._high_open, self._low_open):
            button.setEnabled(False)
            button.setText(tr("Open in Annotate"))

    def _start_busy(self, text: str) -> None:
        """Disable source actions while a worker is active."""
        self._busy = True
        self._scan.setEnabled(False)
        self._browse.setEnabled(False)
        self._bundle_choice.setEnabled(False)
        self._set_status(text)

    def _finish_busy(self) -> None:
        """Restore source actions after worker completion."""
        self._busy = False
        self._scan.setEnabled(True)
        self._browse.setEnabled(True)
        self._bundle_choice.setEnabled(True)

    def _keep_job(self, thread, worker, callback) -> None:
        """Retain a worker pair until its QThread exits."""
        pair = (thread, worker)
        self._jobs.append(pair)
        worker.finished.connect(callback)
        thread.finished.connect(self._retire_jobs)
        thread.start()

    def _retire_jobs(self) -> None:
        """Drop ownership of pairs whose QThread has stopped.

        A bare ``isRunning()`` filter leaked every pair: by the time this
        queued slot runs, ``thread.finished -> deleteLater`` has reaped the
        QThread's C++ half and ``isRunning()`` raises ``RuntimeError`` out
        of the slot, so the assignment never happens. See
        :func:`spacr.qt.bridge.prune_job_pairs`.
        """
        from ..bridge import prune_job_pairs

        self._jobs = prune_job_pairs(self._jobs, self.sender())

    def closeEvent(self, event) -> None:
        """Drain scan/load workers before Qt destroys this screen.

        A job left running here outlives its owner but stays in the
        process-wide run registry, which ``MainWindow.closeEvent`` reads to
        decide whether the application may quit.
        """
        from ..bridge import drain_thread

        for thread, worker in list(self._jobs):
            if worker is not None:
                try:
                    worker.request_cancel("classifier-evaluation closed")
                except Exception:
                    pass
            drain_thread(thread, worker, timeout_ms=3000)
        self._jobs.clear()
        super().closeEvent(event)

    def _clear_bundle(self) -> None:
        """Clear all views and disable bundle actions."""
        self.bundle = None
        self._counts = pd.DataFrame()
        self._overview.clear()
        self._leakage.clear()
        for table in (
            self._confusion, self._per_plate, self._calibration,
            self._predictions,
        ):
            self._render_frame(table, pd.DataFrame())
        self._confusion_ranking.setText("")
        self._clear_cell()
        self._open_folder.setEnabled(False)
        self._copy_path.setEnabled(False)

    def _open_current_folder(self) -> None:
        """Open the loaded evaluation folder."""
        if self.bundle is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(
                str(Path(self.bundle["path"]).parent.resolve())
            ))

    def _copy_current_path(self) -> None:
        """Copy the loaded manifest path."""
        if self.bundle is not None:
            QApplication.clipboard().setText(str(self.bundle["path"]))
            self._set_status(tr("Evaluation path copied."))

    def _set_status(self, text: str, *, error: bool = False) -> None:
        """Show an inline status; failures are never silent or modal."""
        self._status.setText(str(text))
        palette = active_palette()
        self._status.setStyleSheet(
            f"color: {palette['error']};" if error else ""
        )

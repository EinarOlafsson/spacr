"""Classifier Evaluation workbench for out-of-fold result bundles.

Discovery and CSV/JSON parsing run away from the GUI thread.  The screen only
renders already-loaded data, so a large OOF prediction table cannot freeze the
application while it is being read.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDesktopServices, QDragEnterEvent, QDropEvent
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...classifier_evaluation import (
    find_evaluation_bundles,
    load_evaluation_bundle,
)
from ..bridge import make_thread
from ..iconset import icon
from ..i18n import tr
from ..theme import SPACING, active_palette
from ..widgets import Divider

LOG = logging.getLogger(__name__)

__all__ = [
    "ClassifierEvaluationScreen",
    "APP_KEY",
    "APP_NAME",
    "APP_SECTION",
    "APP_INTRO",
]

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
        self._confusion = self._table()
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
            (tr("Confusion matrix"), self._confusion),
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
        self._render_frame(
            self._confusion, self.bundle["confusion_normalized"],
            include_index=True,
        )
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
        self._overview.clear()
        self._leakage.clear()
        for table in (
            self._confusion, self._per_plate, self._calibration,
            self._predictions,
        ):
            self._render_frame(table, pd.DataFrame())
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

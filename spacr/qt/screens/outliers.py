"""The Outliers screen — which objects are wrong, and which *wells* are wrong.

A thin surface over :mod:`spacr.qt.widgets.outlier_model`, which is where all
the statistics and all the arguments live. The screen's only jobs are to load a
table, collect the four decisions the engine needs (features, method,
threshold, transform), run it off the GUI thread, and show what came back
without recomputing a single number of its own.

Assembles what already exists:

* :func:`spacr.qt.screens.graph_builder.read_table` /
  :func:`~spacr.qt.screens.graph_builder.table_names` — the same CSV/SQLite
  reader the Graph Builder and PCA load through, rather than a third one that
  reads ``measurements.db`` slightly differently;
* :class:`spacr.qt.widgets.pca_view.FeaturePicker` — the same tick list PCA
  uses, offering the same continuous columns by the same classifier, so a user
  who ticked eight features for a decomposition does not meet a different list
  here;
* :class:`spacr.qt.job_runner.JobRunner` — an MCD fit on 200,000 objects is
  seconds of work and must not be seconds of a frozen window.

Two tables, because there are two answers
-----------------------------------------
The **Objects** tab is the per-object flags; the **Wells** tab is the
across-well pass. They are separate tabs rather than one merged view because
they answer different questions and routinely disagree: a well shifted as a
whole flags almost none of its individual objects and is nevertheless the
loudest point among wells. The engine's ``report()`` is shown verbatim in the
third tab, caveats included — the share flagged is not interpretable without
the sentence saying whether a symmetric fence on skewed data produced it.

The object table shows the worst rows first and caps how many it draws; the
number of flagged objects in the header comes from the result, never from the
number of rows the table happens to be showing. Nothing about the *export* is
capped.

Nothing is deleted here
-----------------------
The export writes the whole table with the flag columns added, and separately
the flagged rows on their own. There is no "remove outliers" button: dropping
objects is a decision about the analysis, made in the analysis, and a screen
that offered it as a single click would make it the default.

:func:`register` is **not** called at import; read its docstring.
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional

import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QHBoxLayout, QHeaderView, QLabel, QPlainTextEdit,
    QPushButton, QSpinBox, QSplitter, QTabWidget, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..job_runner import JobRunner
from ..theme import (RADIUS, SPACING, pane_surface, register_widget_qss)

#: The control column's object name, and what the QSS block below keys off.
CONTROLS_OBJECT = "OutlierControls"


def _outliers_qss(palette: dict, opacity=None) -> str:
    """This screen's QSS block, appended to every generated stylesheet.

    The control column is a named ``QWidget`` and had no rule of its own,
    so it fell through to the blanket ``QWidget {{ background-color: bg }}``
    -- the WINDOW colour, not a surface, which no page-opacity setting can
    reach. It is a page surface now, the same one the Graph Builder's and
    the Trellis's shelves take, and the feature picker inside it is a
    transparent display that shows it through rather than a second panel.
    """
    return f"""
QWidget#{CONTROLS_OBJECT} {{
    background: {pane_surface("surface_alt", palette.get("theme"), opacity)};
    border-radius: {RADIUS["md"]}px;
}}
"""


# `replace=True`: reachable through the screens package and by direct
# import, and a second import must refresh the block rather than raise.
register_widget_qss("Outliers", _outliers_qss, replace=True)
from ..widgets.outlier_model import (
    DEFAULT_ALPHA, DEFAULT_IQR_C, DEFAULT_MAD_K, DEFAULT_MIN_WELL_OBJECTS,
    METHOD_IQR, METHOD_MAD, METHOD_MAHALANOBIS, TRANSFORM_LOG10,
    TRANSFORM_NONE, OutlierSpec, detect_outliers,
)
from ..widgets.pca_view import FeaturePicker
from .graph_builder import read_table, table_names
from .app_screen import ModuleHeader

LOG = logging.getLogger("spacr.qt.screens.outliers")

__all__ = ["OutliersScreen", "make_outliers_screen", "register",
           "APP_KEY", "APP_NAME", "APP_DESCRIPTION", "APP_INTRO",
           "APP_CLI_NOTE", "APP_NAME_TRANSLATIONS", "MAX_TABLE_ROWS"]

#: The registry key. Chosen once and never renamed.
APP_KEY = "outliers"

#: Object rows drawn in the table, worst score first. A QTableWidget with
#: 200,000 rows in it costs a second of layout and answers no question the
#: first few hundred do not — the counts in the header and everything the
#: export writes are over the whole table regardless.
MAX_TABLE_ROWS = 500

#: The method menu, in the order it is offered: the safest default first.
_METHOD_LABELS = (
    (METHOD_MAD, "MAD — modified z against the median"),
    (METHOD_IQR, "IQR — Tukey's fence on the quartiles"),
    (METHOD_MAHALANOBIS, "Mahalanobis — robust multivariate (MCD)"),
)

#: What the one threshold spinbox means under each method: label, tooltip,
#: range, step, decimals and default.
_THRESHOLD_FIELDS = {
    METHOD_MAD: ("k (robust SDs)",
                 "Flag beyond this many robust SDs from the median. 3.5 is "
                 "the usual choice and is stricter than the 3 a mean/SD rule "
                 "would use, because a robust scale is not inflated by the "
                 "points being tested.",
                 0.1, 20.0, 0.1, 2, DEFAULT_MAD_K),
    METHOD_IQR: ("c (× IQR)",
                 "Flag outside Q1 − c·IQR / Q3 + c·IQR. 1.5 is the box "
                 "plot's whisker; 3.0 is the conventional 'far out' fence.",
                 0.1, 20.0, 0.1, 2, DEFAULT_IQR_C),
    METHOD_MAHALANOBIS: ("α (per object)",
                         "Expected share of CLEAN objects flagged. 0.001 "
                         "means one in a thousand — about 200 false flags "
                         "over 200,000 objects, which the report states.",
                         1e-6, 0.2, 0.0005, 6, DEFAULT_ALPHA),
}


class OutliersScreen(QWidget):
    """Load a measurement table, flag the bad objects, and name the bad wells.

    :param threaded: ``False`` runs the table read and the scan inline, in the
        same order and through the same signals, so a test can drive the whole
        screen synchronously without the behaviour diverging.
    """

    #: A scan finished. Carries the
    #: :class:`~spacr.qt.widgets.outlier_model.OutlierResult`.
    scanned = Signal(object)
    #: A read or a scan failed. Carries the message the engine wrote.
    failed = Signal(str)

    def __init__(self, parent=None, *, threaded: bool = True):
        super().__init__(parent)
        self.setObjectName("OutliersScreen")
        self._frame: Optional[pd.DataFrame] = None
        self._path: Optional[str] = None
        self._result = None
        self._objects: Optional[pd.DataFrame] = None
        self._jobs = JobRunner(self, threaded=threaded, app_key=APP_KEY)
        self._jobs.job_failed.connect(self._on_job_failed)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACING["md"], SPACING["md"],
                                 SPACING["md"], SPACING["md"])
        outer.setSpacing(SPACING["sm"])

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(SPACING["sm"])
        header = ModuleHeader(
            APP_NAME,
            description=APP_DESCRIPTION,
            instruction="Load a table, choose the rule and the columns, then scan.",
        )
        self._header = header
        head.addWidget(header)

        self._source = QLabel("no table loaded", self)
        self._source.setObjectName("OutlierSourceLabel")
        head.addWidget(self._source, 1)

        self._table_picker = QComboBox(self)
        self._table_picker.setObjectName("OutlierTablePicker")
        self._table_picker.setToolTip("Which table of the database to scan")
        self._table_picker.setVisible(False)
        self._table_picker.currentTextChanged.connect(self._on_table_picked)
        head.addWidget(self._table_picker)

        load = QPushButton("Load table…", self)
        load.setObjectName("PrimaryButton")
        load.setToolTip("A measurements.db, or a CSV of measurements")
        load.clicked.connect(self.choose_table)
        head.addWidget(load)

        self._export = QPushButton("Export…", self)
        self._export.setToolTip(
            "Write the whole table with the flag columns added, the flagged "
            "rows on their own, and the per-well summary")
        self._export.clicked.connect(self.export_csv)
        self._export.setEnabled(False)
        head.addWidget(self._export)
        outer.addLayout(head)

        body = QSplitter(Qt.Horizontal, self)
        body.setChildrenCollapsible(False)

        self.tabs = QTabWidget(self)
        self.tabs.setObjectName("OutlierTabs")
        self.object_table = self._make_table("OutlierObjectTable")
        self.tabs.addTab(self.object_table, "Objects")
        self.well_table = self._make_table("OutlierWellTable")
        self.tabs.addTab(self.well_table, "Wells")
        self.report = QPlainTextEdit(self)
        self.report.setObjectName("OutlierReport")
        self.report.setReadOnly(True)
        self.report.setPlaceholderText(
            "Load a table and press Scan. Everything printed here — the "
            "counts, the thresholds and the caveats — comes from "
            "spacr.qt.widgets.outlier_model, which is also the headless "
            "entry point.")
        self.tabs.addTab(self.report, "Report")
        body.addWidget(self.tabs)

        body.addWidget(self._build_controls())
        body.setStretchFactor(0, 1)
        body.setStretchFactor(1, 0)
        outer.addWidget(body, 1)
        # Drop anywhere on this screen: the path is resolved through spaCR's
        # project layout, so the plate folder finds what this screen reads.
        from ..dnd import install_for
        install_for(self, "outliers")

    # -- construction ------------------------------------------------------
    def _make_table(self, name: str) -> QTableWidget:
        """A read-only results grid. Two of them, built the same way."""
        table = QTableWidget(self)
        table.setObjectName(name)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setAlternatingRowColors(True)
        table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)
        table.verticalHeader().setVisible(False)
        return table

    def _build_controls(self) -> QWidget:
        """The right-hand column: what to test, how, and where the line is."""
        panel = QWidget(self)
        panel.setObjectName(CONTROLS_OBJECT)
        panel.setMaximumWidth(360)
        layout = QVBoxLayout(panel)
        # Room for the panel's own rounded surface: the column sits ON a
        # page surface now rather than straight on the window.
        layout.setContentsMargins(SPACING["sm"], SPACING["sm"],
                                  SPACING["sm"], SPACING["sm"])
        layout.setSpacing(SPACING["sm"])

        self.features = FeaturePicker(panel)
        layout.addWidget(self.features, 1)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(SPACING["xs"])

        self.method = QComboBox(panel)
        self.method.setObjectName("OutlierMethod")
        for key, label in _METHOD_LABELS:
            self.method.addItem(label, key)
        self.method.setToolTip(
            "MAD and IQR test one feature at a time. Mahalanobis tests the "
            "features together and is the only one that can see a combination "
            "going wrong while each column on its own looks ordinary.")
        self.method.currentIndexChanged.connect(self._on_method_changed)
        form.addRow("Method", self.method)

        self.threshold = QDoubleSpinBox(panel)
        self.threshold.setObjectName("OutlierThreshold")
        self.threshold_label = QLabel("k (robust SDs)", panel)
        form.addRow(self.threshold_label, self.threshold)

        self.transform = QComboBox(panel)
        self.transform.setObjectName("OutlierTransform")
        self.transform.addItem("none — values as measured", TRANSFORM_NONE)
        self.transform.addItem("log10", TRANSFORM_LOG10)
        self.transform.setToolTip(
            "A symmetric fence on a right-skewed measurement flags a large "
            "share of the right tail whatever the data. log10 makes the rule "
            "mean what it looks like — and refuses on zero or negative "
            "values rather than inventing a pseudocount.")
        form.addRow("Transform", self.transform)

        self.per_well = QCheckBox("Score wells too", panel)
        self.per_well.setObjectName("OutlierPerWell")
        self.per_well.setChecked(True)
        self.per_well.setToolTip(
            "Reduce each well to the median of each feature and run the same "
            "rule across wells. A whole bad well flags almost none of its "
            "individual objects, so this is a different test rather than a "
            "summary of the one above.")
        form.addRow("", self.per_well)

        self.min_well = QSpinBox(panel)
        self.min_well.setObjectName("OutlierMinWellObjects")
        self.min_well.setRange(1, 100_000)
        self.min_well.setValue(DEFAULT_MIN_WELL_OBJECTS)
        self.min_well.setToolTip(
            "A well with fewer objects than this is reported 'not scored' "
            "rather than compared or dropped — a median over a handful of "
            "objects moves more than any plate effect worth finding.")
        form.addRow("Min objects / well", self.min_well)
        layout.addLayout(form)

        self.scan_button = QPushButton("Scan", panel)
        self.scan_button.setObjectName("PrimaryButton")
        self.scan_button.setToolTip("Run the test over the loaded table")
        self.scan_button.clicked.connect(self.scan)
        self.scan_button.setEnabled(False)
        layout.addWidget(self.scan_button)

        self._on_method_changed()
        return panel

    def _on_method_changed(self, *_args) -> None:
        """Re-point the single threshold spinbox at the chosen method's number.

        One control rather than three, because only one of ``k``, ``c`` and
        ``alpha`` is in play at a time and three greyed-out boxes teach nobody
        which one matters.
        """
        label, tip, low, high, step, decimals, default = \
            _THRESHOLD_FIELDS[self.current_method()]
        self.threshold_label.setText(label)
        self.threshold.setToolTip(tip)
        self.threshold.setDecimals(decimals)
        self.threshold.setRange(low, high)
        self.threshold.setSingleStep(step)
        self.threshold.setValue(default)

    # -- state -------------------------------------------------------------
    def current_method(self) -> str:
        """The :data:`spacr.qt.widgets.outlier_model.METHODS` member picked."""
        return self.method.currentData() or METHOD_MAD

    def spec(self) -> OutlierSpec:
        """The controls as an :class:`OutlierSpec`. The screen's whole state.

        :raises OutlierError: from the spec itself on a combination it refuses
            — the screen never invents a value the engine would not accept.
        """
        method = self.current_method()
        value = float(self.threshold.value())
        return OutlierSpec(
            features=self.features.selected(),
            method=method,
            k=value if method == METHOD_MAD else DEFAULT_MAD_K,
            c=value if method == METHOD_IQR else DEFAULT_IQR_C,
            alpha=value if method == METHOD_MAHALANOBIS else DEFAULT_ALPHA,
            transform=self.transform.currentData() or TRANSFORM_NONE,
            min_well_objects=int(self.min_well.value()),
            per_well=bool(self.per_well.isChecked()))

    @property
    def result(self):
        """The last :class:`OutlierResult`, or ``None``."""
        return self._result

    @property
    def frame(self) -> Optional[pd.DataFrame]:
        """The table being scanned, unmodified."""
        return self._frame

    def objects_frame(self) -> Optional[pd.DataFrame]:
        """The loaded table with the flag columns added, or ``None``."""
        return self._objects

    # -- data --------------------------------------------------------------
    def set_frame(self, frame: pd.DataFrame, *, label: str = "",
                  scan: bool = True) -> None:
        """Scan ``frame``. The one call a host needs.

        :param scan: ``False`` loads the table and fills the feature picker
            without running anything, for a caller that wants to set the
            method first.
        """
        self._frame = frame
        self._result = None
        self._objects = None
        self._export.setEnabled(False)
        self.features.set_frame(frame)
        self.scan_button.setEnabled(True)
        self._source.setText(
            label or f"{len(frame):,} rows × {len(frame.columns)} columns")
        if scan:
            self.scan()

    def choose_table(self) -> None:
        """Ask for a file, then :meth:`load_path` it."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open a measurement table", "",
            "Measurements (*.db *.sqlite *.csv *.tsv);;All files (*)")
        if path:
            self.load_path(path)

    def load_path(self, path: str, table: Optional[str] = None) -> None:
        """Load a CSV or one table of a SQLite measurement database.

        The read runs on a worker thread through
        :class:`spacr.qt.job_runner.JobRunner`; listing the table names stays
        inline because the picker has to be populated before the read is
        dispatched, to know which table to read.
        """
        self._path = path
        names: List[str] = []
        if not str(path).lower().endswith((".csv", ".tsv", ".txt")):
            try:
                names = table_names(path)
            except Exception as exc:
                LOG.info("could not list tables in %s", path, exc_info=True)
                # Same seam as a failed job, so a host that only listens for
                # `failed` hears about an unreadable file too.
                self._report_failure(
                    f"could not read {os.path.basename(path)}: {exc}")
                return
        self._table_picker.blockSignals(True)
        self._table_picker.clear()
        self._table_picker.addItems(names)
        self._table_picker.setVisible(bool(names))
        if table and table in names:
            self._table_picker.setCurrentText(table)
        self._table_picker.blockSignals(False)
        chosen = table or (self._table_picker.currentText() or None)
        # A second load supersedes the first, so switching table twice does
        # not deliver the frames in whatever order the reads happen to finish.
        self._jobs.cancel()
        self._source.setText(
            f"loading {os.path.basename(path)}"
            + (f" · {chosen}" if chosen else "") + "…")
        self._jobs.submit(
            lambda p=path, t=chosen: (t, read_table(p, t)),
            self._on_frame_loaded)

    def _on_frame_loaded(self, payload) -> None:
        """Hand a worker-read frame to the screen. GUI thread only."""
        chosen, frame = payload
        path = self._path or ""
        suffix = f" · {chosen}" if chosen else ""
        self.set_frame(
            frame,
            label=f"{os.path.basename(path)}{suffix} · {len(frame):,} rows "
                  f"× {len(frame.columns)} columns")

    def _on_table_picked(self, name: str) -> None:
        if self._path and name:
            self.load_path(self._path, table=name)

    def _report_failure(self, message: str) -> None:
        """Say it in the status line and tell anyone listening. Never a modal —
        a dialog nobody can dismiss is how a headless run hangs.

        The engine's refusals are written to be read by the user
        ("``cell_area`` has 412 non-positive values..."), so the message is
        shown verbatim rather than replaced with a house apology.
        """
        LOG.info("outlier screen: %s", message)
        self._source.setText(message)
        self.failed.emit(message)

    def _on_job_failed(self, message: str) -> None:
        """A worker raised. Clear the result before saying so.

        Whatever is on screen was computed from a run that has now failed, so
        leaving the export button live would offer a file of stale flags under
        the name of the table that could not be read.
        """
        self._result = None
        self._objects = None
        self._export.setEnabled(False)
        self.report.setPlainText(message)
        self._report_failure(message)

    # -- the scan ----------------------------------------------------------
    def scan(self) -> None:
        """Run the engine over the loaded table, off the GUI thread."""
        frame = self._frame
        if frame is None:
            self._source.setText("Load a table first.")
            return
        try:
            spec = self.spec()
        except ValueError as exc:
            self._on_job_failed(str(exc))
            return
        self._source.setText(f"scanning {len(frame):,} objects — "
                             f"{spec.describe()}…")
        self._jobs.submit(lambda f=frame, s=spec: detect_outliers(f, s),
                          self._on_scanned)

    def _on_scanned(self, result) -> None:
        """Show a finished scan. GUI thread only."""
        self._result = result
        frame = self._frame
        if frame is None:  # pragma: no cover - the frame cannot vanish mid-job
            return
        self._objects = result.object_frame(frame)
        self._fill_object_table(result)
        self._fill_well_table(result)
        self.report.setPlainText(result.report())
        self._export.setEnabled(True)
        self.tabs.setTabText(
            0, f"Objects ({result.n_flagged:,})")
        self.tabs.setTabText(
            1, f"Wells ({len(result.flagged_wells())})"
            if result.has_wells else "Wells")
        self._source.setText(result.headline())
        self.scanned.emit(result)

    def _fill_object_table(self, result) -> None:
        """Worst rows first, capped, with the counts taken from the result."""
        frame = self._objects
        if frame is None:  # pragma: no cover - set immediately before
            return
        names = dict(result.column_names)
        columns = [c for c in result.well_keys if c in frame.columns]
        columns += [c for c in result.features if c in frame.columns]
        columns += [names[key] for key in ("outlier", "score", "reason")
                    if names.get(key) in frame.columns]
        shown = frame.loc[:, columns]
        # Sort by score, NaN last: an unscorable object is not the most
        # interesting row on the screen, but it must not disappear either.
        order = shown[names["score"]].sort_values(
            ascending=False, na_position="last").index
        _fill_table(self.object_table, shown.loc[order].head(MAX_TABLE_ROWS))

    def _fill_well_table(self, result) -> None:
        """Every well, flagged first, then unscored, then the rest."""
        wells = result.well_frame()
        if wells.empty:
            _fill_table(self.well_table, wells)
            return
        wells = wells.sort_values(
            ["well_outlier", "well_outlier_score"],
            ascending=[False, False], na_position="last")
        _fill_table(self.well_table, wells.head(MAX_TABLE_ROWS))

    def active_jobs(self) -> int:
        """How many worker threads are still winding down."""
        return self._jobs.active_jobs()

    def is_busy(self) -> bool:
        """True while a read or a scan is in flight."""
        return self._jobs.is_busy()

    # -- export ------------------------------------------------------------
    def export_csv(self) -> None:
        """Write the flags out: the whole table, the flagged rows, the wells.

        Three files, because they have three different row meanings and one
        sheet mixing them would have to be unpicked before anyone could use it.
        The first is the *whole* table with the columns added — the "write
        columns" case, and the one that keeps every object — and the second is
        the same rows filtered down to the flagged ones for a quick look. The
        engine's own ``filtered()`` is what produces it, so the file and the
        screen cannot disagree about what was flagged.
        """
        result = self._result
        if result is None or self._objects is None:
            self._source.setText("Nothing to export — run a Scan first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export the outlier scan", "outliers.csv", "CSV (*.csv)")
        if not path:
            return
        stem = path[:-4] if path.lower().endswith(".csv") else path
        try:
            self._objects.to_csv(f"{stem}_objects.csv", index=False)
            flagged = self._objects.loc[result.flags]
            flagged.to_csv(f"{stem}_flagged.csv", index=False)
            if result.has_wells:
                result.well_frame().to_csv(f"{stem}_wells.csv", index=False)
            with open(f"{stem}_report.txt", "w", encoding="utf-8") as handle:
                handle.write(result.report() + "\n")
        except OSError as exc:
            LOG.info("could not export the outlier scan", exc_info=True)
            self._source.setText(f"could not write those files: {exc}")
            return
        self._source.setText(
            f"wrote {os.path.basename(stem)}_objects / _flagged"
            + (" / _wells" if result.has_wells else "")
            + " .csv and _report.txt")

    def closeEvent(self, event):  # noqa: N802 - Qt name
        # Abandon an in-flight read or scan rather than let it outlive the
        # screen: Qt aborts the process if a running QThread is destroyed, and
        # a worker that delivers into a closed widget is a use-after-free.
        self._jobs.shutdown()
        super().closeEvent(event)


def _fill_table(table: QTableWidget, frame: pd.DataFrame) -> None:
    """Put ``frame`` into ``table`` as read-only text.

    Floats are formatted to six significant figures — a QTableWidget showing
    ``0.30000000000000004`` teaches nobody anything — and everything else is
    ``str``. Booleans render as ``yes`` / ``no``, because ``True`` in a column
    called ``outlier`` reads as a header rather than a value at a glance.
    """
    table.clear()
    table.setRowCount(len(frame))
    table.setColumnCount(len(frame.columns))
    table.setHorizontalHeaderLabels([str(c) for c in frame.columns])
    for column, name in enumerate(frame.columns):
        values = frame[name].tolist()
        for row, value in enumerate(values):
            table.setItem(row, column, QTableWidgetItem(_cell(value)))


def _cell(value) -> str:
    """One frame value as the string a table cell should show."""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if value != value:                    # NaN, without importing math
            return ""
        return f"{value:.6g}"
    return "" if value is None else str(value)


def make_outliers_screen(app_key: Optional[str] = None) -> QWidget:
    """Factory handed to :func:`spacr.qt.app.register_app`."""
    return OutliersScreen()


APP_NAME = "Outliers"
APP_DESCRIPTION = ("Robust per-object and per-well outlier detection — MAD, "
                   "Tukey and MCD Mahalanobis")
APP_INTRO = (
    "Finds the objects that are wrong and, separately, the wells that are "
    "wrong — which is usually the one that matters, and which per-object "
    "flags are nearly blind to: a well shifted as a whole flags almost none "
    "of its individual cells. Nothing is estimated from a mean or an SD, "
    "because the outliers would move both. Pick features, pick a rule — a "
    "modified z against the median, Tukey's fence, or a robust multivariate "
    "distance whose threshold is a stated false-positive rate — and the flags "
    "arrive as added columns. No row is ever dropped.")
APP_CLI_NOTE = (
    "Outliers is an interactive QC surface: the feature list, the method and "
    "the threshold are the feature; run it in the GUI (spacr-qt). Headless, "
    "spacr.qt.widgets.outlier_model.detect_outliers() computes exactly the "
    "same object flags, well scores and report with no Qt involved.")
#: The display name in the nine non-English UI languages, in
#: `spacr.qt.i18n.LANGUAGES` order (sv, de, es, zh_CN, pt, hi, ko, is, fr).
APP_NAME_TRANSLATIONS = (
    "Avvikare", "Ausreißer", "Valores atípicos", "离群值",
    "Valores atípicos", "आउटलायर", "이상치", "Frávik",
    "Valeurs aberrantes")


def register() -> bool:
    """Put Outliers in the app registry, through the public seam. Idempotent.

    Everything after the section is a table this key would otherwise need a
    hand-edit in: the screen header and blurb, the "no headless run" sentence,
    the API doc link and the nine translations of the display name.
    :func:`spacr.qt.app.register_app` distributes them from this one call.

    ``SECTION_EXPLORE`` rather than ``SECTION_RESULTS``, and the reasoning is
    worth writing down because the first instinct is the other one. This screen
    looks like QC — it is about whether to believe what a run produced — but
    what it *does* is what the Gate Editor and the Feature Explorer do, and it
    sits beside them: you pick features, you move a threshold, you watch a
    distribution answer, and what comes out is a **column** the user then
    filters or gates on. Results & QC holds the screens that hand back a
    verdict; this one hands back a question with the evidence attached, and
    :meth:`~spacr.qt.widgets.outlier_model.OutlierResult.filtered` is the only
    place a row is ever dropped and it has to be called on purpose.

    The cap made the choice concrete rather than academic:
    ``MAX_APPS_PER_SECTION`` is 13 and Results & QC was already at 12 with
    Control Charts — the campaign-level verdict — arriving in the same batch.
    That is the section's honest occupant, and this one had somewhere honest
    to go. Both were checked against the cap before the placement, not after.

    **Not called at import.** ``app.py`` imports ``spacr.qt.widgets`` before
    ``register_app`` exists, so nothing reachable from the top of that file can
    register during its import, and a registration that happens later is one
    that some importer's snapshot of ``APPS`` predates. The one place a
    registration is visible to everybody is
    :data:`spacr.qt.SELF_REGISTERING_MODULES`, which
    :func:`spacr.qt.run` runs after ``spacr.qt.app`` is fully executed and
    before ``MainWindow.__init__`` reads the registry. Turning this screen on
    is therefore one row there::

        "spacr.qt.screens.outliers",

    and nothing else: the strings above travel with the registration.

    :returns: ``True`` if this call is what registered it. Safe to call
        again — a module imported from two paths, or a test that re-imports
        it, must not raise on the duplicate key.
    """
    from ..app import APPS, SECTION_EXPLORE, STAGE_ALPHA, register_app
    if any(row[0] == APP_KEY for row in APPS):
        return False
    register_app(APP_KEY, APP_NAME, APP_DESCRIPTION, SECTION_EXPLORE,
                 factory=make_outliers_screen, stage=STAGE_ALPHA,
                 intro=APP_INTRO, cli_note=APP_CLI_NOTE,
                 api_module="qt/screens/outliers",
                 translations=APP_NAME_TRANSLATIONS)
    return True

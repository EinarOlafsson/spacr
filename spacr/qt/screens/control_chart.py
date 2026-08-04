"""B9 — Control Charts: is the control still the control, plate after plate?

A screening campaign is dozens of plates run over weeks, and the one assumption
holding the whole analysis up is that the controls are the same thing every
time. Hit calling, plate normalisation and Z' all measure against them, so when
the controls drift, everything downstream is already wrong and nothing says so
— the analysis normalises to whatever controls it is handed.

This screen is the picture that says so. One control value per plate along run
order, limits estimated from a stated baseline and applied forward, every
Nelson rule that fires marked on the plate it fired on and named in words
underneath.

Everything numeric comes from :mod:`spacr.qt.widgets.control_chart`, which has
no Qt in it and carries the argument for every decision — most of all the one
the module turns on, that sigma comes from the average moving range over d2 and
never from the standard deviation of the series, because the SD is inflated by
exactly the drift the chart exists to detect. This file draws that result and
nothing else: no statistic is computed here, so the chart on screen and the
chart in a report cannot disagree.

Assembles:

* :func:`spacr.qt.screens.graph_builder.read_table` /
  :func:`~spacr.qt.screens.graph_builder.table_names` — the same measurement
  loader every Explore screen uses, so a ``measurements.db`` opens here the way
  it opens there;
* :class:`spacr.qt.job_runner.JobRunner` — the read *and* the chart run off the
  GUI thread; a campaign table is object rows, and grouping a few hundred
  thousand of them per redraw is a frozen window;
* the owned-timer matplotlib canvas from
  :mod:`spacr.qt.widgets.graph_builder`, imported rather than copied because
  two copies of a segfault fix is one copy too many.

:func:`register` is not called at import; read its docstring.
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QFileDialog, QFormLayout,
    QHBoxLayout, QHeaderView, QLabel, QListWidget, QListWidgetItem,
    QPlainTextEdit, QPushButton, QSpinBox, QSplitter, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..job_runner import JobRunner
from ..theme import (RADIUS, SPACING, active_palette, pane_surface,
                     register_widget_qss)

#: The control column's object name, and what the QSS block below keys off.
CONTROLS_OBJECT = "ControlChartControls"

#: The column under the chart — the report and the violations table. It is
#: the third of the page's three regions and the one that had no panel.
OUTPUT_OBJECT = "ControlChartOutput"


def _control_chart_qss(palette: dict, opacity=None) -> str:
    """This screen's QSS block, appended to every generated stylesheet.

    The control column is a named ``QWidget`` and had no rule of its own,
    so it fell through to the blanket ``QWidget {{ background-color: bg }}``
    -- the WINDOW colour, not a surface, which no page-opacity setting can
    reach. It is a page surface now, the same one the Graph Builder's and
    the Trellis's shelves take.

    The output column under the chart was the region left over. It was an
    ANONYMOUS ``QWidget``, so ``clear_container_surfaces`` tagged it
    transparent as scaffolding -- and the report and the violations table
    inside it are both ``QAbstractScrollArea``, which that sweep tags by
    type as well. Three transparent things stacked: the whole lower right
    of the page measured 1.000, the backdrop arriving untouched, which
    over a dark window is the black rectangle that was reported.

    The panel goes on the column rather than on the two widgets, which is
    the treatment Classifier Evaluation's and Run History's tab panes
    take: the container is the surface, and a read-only display sitting on
    it shows it through instead of painting an opaque rectangle over the
    thing that was just made translucent.
    """
    surface = pane_surface("surface_alt", palette.get("theme"), opacity)
    return f"""
QWidget#{CONTROLS_OBJECT}, QWidget#{OUTPUT_OBJECT} {{
    background: {surface};
    border-radius: {RADIUS["md"]}px;
}}
"""


# `replace=True`: reachable through the screens package and by direct
# import, and a second import must refresh the block rather than raise.
register_widget_qss("ControlChart", _control_chart_qss, replace=True)
# `_canvas_class` is the owned-timer FigureCanvas fix: matplotlib schedules its
# idle draw on a static QTimer that is not owned by the canvas and can fire
# after Qt has deleted it, which is a segfault on close. Imported from the one
# place that has it rather than copied.
from ..widgets.graph_builder import (_canvas_class, _page_surface_axes,
                                     categorical_colours)
from ..widgets.control_chart import (
    ESTIMATOR_AUTO, ESTIMATOR_LABELS, ESTIMATORS, RULES_ALL, RULES_DEFAULT,
    RULES_LIMITS_ONLY, RULE_DETECTS, RULE_NAMES, DEFAULT_BASELINE,
    MIN_BASELINE, ControlChartError, ControlChartResult, ControlChartSpec,
    candidate_key_columns, candidate_value_columns, control_chart,
    zprime_chart,
)
from .graph_builder import read_table, table_names
from .app_screen import ModuleHeader

LOG = logging.getLogger("spacr.qt.screens.control_chart")

__all__ = ["ControlChartCanvas", "ControlChartScreen",
           "make_control_chart_screen", "register", "RULE_SETS",
           "APP_KEY", "APP_NAME", "APP_DESCRIPTION", "APP_INTRO",
           "APP_CLI_NOTE", "APP_NAME_TRANSLATIONS"]

#: The registry key. Chosen once and never renamed.
APP_KEY = "control_chart"

#: The rule sets offered, with the consequence of each in the label rather than
#: in a manual nobody opens. The default is first.
RULE_SETS: Tuple[Tuple[str, Tuple[int, ...]], ...] = (
    ("Western Electric (1, 2, 5, 6) — the usual four", RULES_DEFAULT),
    ("Nelson (all eight) — most sensitive, most false alarms", RULES_ALL),
    ("Limits only (rule 1) — nothing but 3 sigma", RULES_LIMITS_ONLY),
)

#: Column names worth guessing at, best first, when a table is first loaded.
#: A guess the user can see and change beats an empty form.
_PLATE_GUESSES = ("plateID", "plate_id", "plate", "PlateID", "barcode")
_ORDER_GUESSES = ("run_date", "date", "run_order", "order", "timepoint",
                  "day", "acquisition_date", "timestamp")
_CONTROL_GUESSES = ("well_type", "condition", "control", "treatment",
                    "sample_type", "gene")

#: How many x tick labels a chart draws before it starts thinning them. Past
#: this the labels overlap into a grey smear and stop being labels.
_MAX_TICKS = 30


class ControlChartCanvas(QWidget):
    """The chart: zones, limits, points, and the violating plates marked.

    Draws a :class:`~spacr.qt.widgets.control_chart.ControlChartResult` and
    computes nothing. The zones are filled from the result's **per-point**
    limit arrays rather than from a single pair of numbers, so a campaign whose
    plates carry different numbers of control wells draws the stepped limits
    that are actually in force rather than an average that is in force nowhere.
    """

    #: Emitted after every draw with the result that was drawn (or ``None``).
    rendered = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ControlChartCanvas")
        self._result: Optional[ControlChartResult] = None
        self._message = "no chart yet"

        from matplotlib.figure import Figure
        palette = active_palette()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        # No `facecolor` and no inline `background:` -- the canvas paints
        # the page panel in its own `paintEvent` under a transparent figure
        # patch, and either of those would put the opaque rectangle back.
        self.figure = Figure(figsize=(8.0, 4.2))
        self.canvas = _canvas_class()(self.figure)
        self.canvas.setMinimumHeight(260)
        outer.addWidget(self.canvas, 1)

    @property
    def result(self) -> Optional[ControlChartResult]:
        """The result currently drawn, or ``None``."""
        return self._result

    def set_result(self, result: Optional[ControlChartResult], *,
                   message: str = "") -> None:
        """Draw ``result``; ``None`` draws ``message`` on an empty axis."""
        self._result = result
        self._message = message or "no chart yet"
        self.render_now()

    def render_now(self) -> None:
        """Redraw from the held result. Idempotent."""
        palette = active_palette()
        self.figure.clear()
        # `clear()` restores the rc facecolor and its alpha with it.
        self.figure.patch.set_alpha(0.0)
        ax = self.figure.add_subplot(111)
        _page_surface_axes(ax, palette)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(palette["border"])
            ax.spines[side].set_linewidth(0.8)
        ax.tick_params(colors=palette["fg_muted"], labelsize=8, length=3)

        result = self._result
        if result is None or not len(result):
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(0.5, 0.5, self._message, ha="center", va="center",
                    color=palette["fg_muted"], fontsize=10, wrap=True,
                    transform=ax.transAxes)
            self.figure.tight_layout(pad=0.6)
            self.canvas.draw_idle()
            self.rendered.emit(None)
            return

        x = np.arange(len(result))
        centre = np.full(len(result), result.centre)
        sigma = result.sigma_at

        # The zones, outermost first so the inner ones sit on top. Three bands
        # rather than two lines because rules 5 and 6 are statements about the
        # 2- and 1-sigma bands, and a reader cannot check them against a chart
        # that only draws the 3-sigma limit.
        if not result.degenerate:
            for k, alpha in ((3, 0.10), (2, 0.16), (1, 0.24)):
                ax.fill_between(x, centre - k * sigma, centre + k * sigma,
                                color=palette["accent"], alpha=alpha,
                                linewidth=0.0, zorder=0)
            ax.plot(x, result.upper, color=palette["error"], linewidth=1.2,
                    linestyle="--", zorder=2, label="±3σ")
            ax.plot(x, result.lower, color=palette["error"], linewidth=1.2,
                    linestyle="--", zorder=2)
        ax.plot(x, centre, color=palette["fg_dim"], linewidth=1.2, zorder=2,
                label="centre")

        # Where Phase I ends. The limits are a statement about the points to
        # the left of this line and a test of the points to the right, and a
        # chart that does not show the boundary invites reading the baseline
        # as evidence for itself.
        if result.baseline.size and int(result.baseline.max()) < len(result) - 1:
            ax.axvline(float(result.baseline.max()) + 0.5,
                       color=palette["fg_muted"], linewidth=1.0,
                       linestyle=":", zorder=2)
            ax.text(float(result.baseline.max()) + 0.6, 0.98,
                    "baseline ends", transform=ax.get_xaxis_transform(),
                    va="top", fontsize=7, color=palette["fg_muted"])

        ax.plot(x, result.values, color=palette["fg"], linewidth=1.0,
                marker="o", markersize=3.4, zorder=3,
                markerfacecolor=palette["surface"])

        # One overplotted marker per rule, so a plate that trips three rules
        # carries three marks and the legend says which. Colour-coded by rule
        # number through the fixed eight-hue series — eight rules, eight hues.
        series = categorical_colours()
        marks: Dict[int, List[int]] = {}
        for violation in result.violations:
            marks.setdefault(violation.rule, []).extend(violation.points)
        for offset, (rule, points) in enumerate(sorted(marks.items())):
            index = np.unique(np.asarray(points, dtype=int))
            ax.scatter(index, result.values[index],
                       s=90 + 34 * offset, facecolors="none",
                       edgecolors=series[(rule - 1) % len(series)],
                       linewidths=1.6, zorder=4,
                       label=f"rule {rule} — {RULE_NAMES[rule]}")

        ax.set_xlim(-0.6, len(result) - 0.4)
        step = max(1, int(np.ceil(len(result) / _MAX_TICKS)))
        ax.set_xticks(x[::step])
        ax.set_xticklabels([result.plates[i] for i in x[::step]],
                           rotation=45, ha="right", fontsize=7)
        ax.set_ylabel(result.value_column, color=palette["fg_dim"], fontsize=9)
        ax.set_xlabel(
            (f"run order — {result.order_column}" if result.order_column
             else f"run order — INFERRED from {result.plate_column}"),
            color=palette["fg_dim"], fontsize=9)
        ax.grid(True, axis="y", color=palette["border_soft"], linewidth=0.6,
                alpha=0.5)
        ax.set_axisbelow(True)
        if marks or not result.degenerate:
            ax.legend(loc="best", fontsize=7, frameon=False,
                      labelcolor=palette["fg_dim"])
        self.figure.tight_layout(pad=0.6)
        self.canvas.draw_idle()
        self.rendered.emit(result)

    def closeEvent(self, event):  # noqa: N802 - Qt name
        cancel = getattr(self.canvas, "cancel_pending_draw", None)
        if callable(cancel):
            cancel()
        super().closeEvent(event)


class ControlChartScreen(QWidget):
    """A table, the columns that say what a control is, and the chart.

    :param threaded: ``False`` runs the table read and the chart inline, so a
        test drives the screen synchronously without the behaviour diverging.
    """

    #: Emitted whenever a chart is refused, with the engine's message. The
    #: message is also shown inline — a refusal that only logs is a blank
    #: canvas the user has no explanation for.
    failed = Signal(str)

    def __init__(self, parent=None, *, threaded: bool = True):
        super().__init__(parent)
        self.setObjectName("ControlChartScreen")
        self._frame: Optional[pd.DataFrame] = None
        self._path: Optional[str] = None
        self._result: Optional[ControlChartResult] = None
        self._loading = False
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
            instruction="Load a table, name the plate column and the "
                        "measurement, then read the chart.",
        )
        self._header = header
        head.addWidget(header)

        self._source = QLabel("no table loaded", self)
        self._source.setObjectName("ControlChartSourceLabel")
        head.addWidget(self._source, 1)

        self._table_picker = QComboBox(self)
        self._table_picker.setObjectName("ControlChartTablePicker")
        self._table_picker.setToolTip("Which table of the database to chart")
        self._table_picker.setVisible(False)
        self._table_picker.currentTextChanged.connect(self._on_table_picked)
        head.addWidget(self._table_picker)

        load = QPushButton("Load table…", self)
        load.setObjectName("PrimaryButton")
        load.setToolTip("A measurements.db, or a CSV of per-well values")
        load.clicked.connect(self.choose_table)
        head.addWidget(load)

        export = QPushButton("Export points…", self)
        export.setToolTip(
            "Every plate with its value, its limits, its z and the rules that "
            "fired on it, as CSV")
        export.clicked.connect(self.choose_export)
        head.addWidget(export)
        outer.addLayout(head)

        body = QSplitter(Qt.Horizontal, self)
        body.setChildrenCollapsible(False)
        body.addWidget(self._build_controls())

        right = QSplitter(Qt.Vertical, self)
        right.setChildrenCollapsible(False)
        self.canvas = ControlChartCanvas(self)
        right.addWidget(self.canvas)

        lower = QWidget(self)
        # Named, so it is a panel rather than scaffolding the container
        # sweep tags transparent -- see `_control_chart_qss`.
        lower.setObjectName(OUTPUT_OBJECT)
        lower_layout = QVBoxLayout(lower)
        # Room for the column's own rounded surface around the report and
        # the violations table, which show it through.
        lower_layout.setContentsMargins(SPACING["sm"], SPACING["sm"],
                                        SPACING["sm"], SPACING["sm"])
        lower_layout.setSpacing(SPACING["xs"])
        self.report = QPlainTextEdit(lower)
        self.report.setObjectName("ControlChartReport")
        self.report.setReadOnly(True)
        self.report.setMinimumHeight(90)
        self.report.setPlainText("Load a table and pick a plate column and a "
                                 "measurement.")
        lower_layout.addWidget(self.report, 1)

        self.violations = QTableWidget(0, 4, lower)
        self.violations.setObjectName("ControlChartViolations")
        self.violations.setHorizontalHeaderLabels(
            ["Rule", "Plates", "What it detects", "In words"])
        self.violations.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.violations.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.violations.verticalHeader().setVisible(False)
        self.violations.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.Stretch)
        self.violations.setMinimumHeight(90)
        lower_layout.addWidget(self.violations, 1)
        right.addWidget(lower)
        right.setStretchFactor(0, 3)
        right.setStretchFactor(1, 2)

        body.addWidget(right)
        body.setStretchFactor(0, 0)
        body.setStretchFactor(1, 1)
        outer.addWidget(body, 1)
        # Drop anywhere on this screen: the path is resolved through spaCR's
        # project layout, so the plate folder finds what this screen reads.
        from ..dnd import install_for
        install_for(self, "control_chart")

    # -- the form ---------------------------------------------------------
    def _build_controls(self) -> QWidget:
        """The left-hand column: what a plate is, what the control is, and the
        three statistical choices that change the answer."""
        panel = QWidget(self)
        panel.setObjectName(CONTROLS_OBJECT)
        panel.setMaximumWidth(330)
        form = QFormLayout(panel)
        # Room for the panel's own rounded surface: the column sits ON a
        # page surface now rather than straight on the window.
        form.setContentsMargins(SPACING["sm"], SPACING["sm"],
                                SPACING["sm"], SPACING["sm"])
        form.setSpacing(SPACING["xs"])

        self._plate = QComboBox(panel)
        self._plate.setObjectName("ControlChartPlate")
        self._plate.setToolTip("One point on the chart per level of this column")
        self._plate.currentTextChanged.connect(self._on_control_changed)
        form.addRow("Plate", self._plate)

        self._order = QComboBox(panel)
        self._order.setObjectName("ControlChartOrder")
        self._order.setToolTip(
            "The run order or date. Leave empty and the order is inferred "
            "from the plate id — which every run-based rule then rests on.")
        self._order.currentTextChanged.connect(self._on_control_changed)
        form.addRow("Run order", self._order)

        self._value = QComboBox(panel)
        self._value.setObjectName("ControlChartValue")
        self._value.setToolTip("The measurement to chart")
        self._value.currentTextChanged.connect(self._on_control_changed)
        form.addRow("Measurement", self._value)

        self._control_column = QComboBox(panel)
        self._control_column.setObjectName("ControlChartControlColumn")
        self._control_column.setToolTip(
            "The column saying what each well is. Leave empty when the table "
            "is already only the control.")
        self._control_column.currentTextChanged.connect(self._on_control_column)
        form.addRow("Control column", self._control_column)

        self._levels = QListWidget(panel)
        self._levels.setObjectName("ControlChartLevels")
        self._levels.setSelectionMode(QAbstractItemView.MultiSelection)
        self._levels.setMaximumHeight(96)
        self._levels.setToolTip("Which level(s) are the control being charted")
        self._levels.itemSelectionChanged.connect(self._on_control_changed)
        form.addRow("Control is", self._levels)

        self._estimator = QComboBox(panel)
        self._estimator.setObjectName("ControlChartEstimator")
        for key in ESTIMATORS:
            self._estimator.addItem(f"{key} — {ESTIMATOR_LABELS[key]}", key)
        self._estimator.setCurrentIndex(ESTIMATORS.index(ESTIMATOR_AUTO))
        self._estimator.setToolTip(
            "Where sigma comes from. Never the standard deviation of the "
            "series — that is inflated by the drift the chart is looking for.")
        self._estimator.currentIndexChanged.connect(self._on_control_changed)
        form.addRow("Sigma from", self._estimator)

        self._rules = QComboBox(panel)
        self._rules.setObjectName("ControlChartRules")
        for label, rules in RULE_SETS:
            self._rules.addItem(label, rules)
        self._rules.setToolTip(
            "More rules is more sensitivity and more false alarms; the report "
            "says how many to expect over this campaign.")
        self._rules.currentIndexChanged.connect(self._on_control_changed)
        form.addRow("Rules", self._rules)

        self._baseline = QSpinBox(panel)
        self._baseline.setObjectName("ControlChartBaseline")
        self._baseline.setRange(MIN_BASELINE, 500)
        self._baseline.setValue(DEFAULT_BASELINE)
        self._baseline.setToolTip(
            "Phase I: how many plates from the start the limits are estimated "
            "from. They are then applied forward to everything after.")
        self._baseline.valueChanged.connect(self._on_control_changed)
        form.addRow("Baseline plates", self._baseline)

        self._reestimate = QCheckBox("Re-estimate without flagged plates",
                                     panel)
        self._reestimate.setObjectName("ControlChartReestimate")
        self._reestimate.setToolTip(
            "When the baseline itself trips a rule, drop those plates and "
            "estimate once more. One pass, never iterated.")
        self._reestimate.toggled.connect(self._on_control_changed)
        form.addRow("", self._reestimate)

        self._zprime = QCheckBox("Chart Z' instead", panel)
        self._zprime.setObjectName("ControlChartZPrime")
        self._zprime.setToolTip(
            "Needs a positive and a negative level selected below. Charts the "
            "per-plate assay window rather than one control's level.")
        self._zprime.toggled.connect(self._on_control_changed)
        form.addRow("", self._zprime)

        self._positive = QComboBox(panel)
        self._positive.setObjectName("ControlChartPositive")
        self._positive.currentTextChanged.connect(self._on_control_changed)
        form.addRow("Positive control", self._positive)

        self._negative = QComboBox(panel)
        self._negative.setObjectName("ControlChartNegative")
        self._negative.currentTextChanged.connect(self._on_control_changed)
        form.addRow("Negative control", self._negative)
        return panel

    # -- data -------------------------------------------------------------
    def set_frame(self, frame: pd.DataFrame, *, label: str = "") -> None:
        """Chart ``frame``. The one call a host needs."""
        self._frame = frame
        self._loading = True
        try:
            self._refill_pickers(frame)
        finally:
            self._loading = False
        self._source.setText(
            label or f"{len(frame):,} rows × {len(frame.columns)} columns")
        self.recompute()

    def _refill_pickers(self, frame: pd.DataFrame) -> None:
        """Offer the columns and guess the obvious ones.

        A guess the user can see and correct beats an empty form: on a spaCR
        measurement table ``plateID`` is the plate column essentially always,
        and starting the screen with a drawn chart is what makes the pickers
        legible as *changes to* something rather than as a questionnaire.
        """
        keys = list(candidate_key_columns(frame))
        values = list(candidate_value_columns(frame))
        columns = [str(c) for c in frame.columns]
        if not values:
            # The classifier offers *continuous* columns, and a control that
            # never moved is not continuous — which is exactly the table a user
            # opens this screen to find out about. Falling back to every
            # numeric column keeps the degenerate case reachable; falling back
            # to every column would offer the plate id as a measurement.
            values = [name for name in columns
                      if pd.api.types.is_numeric_dtype(frame[name])]

        def fill(box: QComboBox, options: List[str], *, blank: bool,
                 guesses: Tuple[str, ...] = ()) -> None:
            previous = box.currentText()
            box.blockSignals(True)
            box.clear()
            if blank:
                box.addItem("")
            box.addItems(options)
            chosen = ""
            if previous in options:
                chosen = previous
            else:
                for guess in guesses:
                    if guess in options:
                        chosen = guess
                        break
                if not chosen and options and not blank:
                    chosen = options[0]
            box.setCurrentText(chosen)
            box.blockSignals(False)

        fill(self._plate, keys or columns, blank=False, guesses=_PLATE_GUESSES)
        fill(self._order, [c for c in columns], blank=True,
             guesses=_ORDER_GUESSES)
        fill(self._value, values or columns, blank=False)
        fill(self._control_column, keys, blank=True, guesses=_CONTROL_GUESSES)
        self._refill_levels(frame)

    def _refill_levels(self, frame: pd.DataFrame) -> None:
        """The distinct levels of the control column, for the three pickers."""
        column = self._control_column.currentText()
        wanted = self._selected_levels()
        self._levels.blockSignals(True)
        self._levels.clear()
        levels: List[str] = []
        if column and column in frame.columns:
            levels = sorted({str(v) for v in frame[column].dropna().unique()})
            for level in levels[:200]:
                item = QListWidgetItem(level, self._levels)
                item.setSelected(level in wanted)
        self._levels.blockSignals(False)
        for box in (self._positive, self._negative):
            previous = box.currentText()
            box.blockSignals(True)
            box.clear()
            box.addItem("")
            box.addItems(levels)
            if previous in levels:
                box.setCurrentText(previous)
            box.blockSignals(False)

    def _selected_levels(self) -> Tuple[str, ...]:
        return tuple(item.text() for item in self._levels.selectedItems())

    def _on_control_column(self, _text: str) -> None:
        if self._frame is not None:
            self._refill_levels(self._frame)
        self._on_control_changed()

    def _on_control_changed(self, *_args) -> None:
        if not self._loading:
            self.recompute()

    # -- the chart --------------------------------------------------------
    def spec(self) -> ControlChartSpec:
        """The spec the form describes.

        :raises ControlChartError: for a form that cannot mean anything — the
            same refusals the engine makes, at the same point, with the same
            messages.
        """
        rules = self._rules.currentData() or RULES_DEFAULT
        column = self._control_column.currentText() or None
        levels = self._selected_levels() if column else ()
        positive = self._positive.currentText()
        negative = self._negative.currentText()
        if self._zprime.isChecked() and column and not levels:
            # Z' does not chart one control's level, so an empty tick list is
            # not a missing answer here — the two named controls are.
            levels = tuple(x for x in (positive, negative) if x)
        return ControlChartSpec(
            value=self._value.currentText(),
            plate=self._plate.currentText(),
            order=self._order.currentText() or None,
            control_column=column if levels else None,
            control_levels=levels,
            positive_levels=(positive,) if positive else (),
            negative_levels=(negative,) if negative else (),
            estimator=self._estimator.currentData() or ESTIMATOR_AUTO,
            rules=tuple(rules),
            baseline_n=int(self._baseline.value()),
            reestimate=bool(self._reestimate.isChecked()))

    def recompute(self) -> None:
        """Rebuild the chart from the form, off the GUI thread.

        A second request supersedes the first, so dragging the baseline spinner
        does not deliver the charts in whatever order the workers finish and
        leave the picture disagreeing with the form.
        """
        frame = self._frame
        if frame is None:
            return
        try:
            spec = self.spec()
        except ControlChartError as exc:
            self._show_refusal(str(exc))
            return
        zprime = bool(self._zprime.isChecked())
        self._jobs.cancel()
        self._jobs.submit(
            lambda f=frame, s=spec, z=zprime: (
                zprime_chart(f, s) if z else control_chart(f, s)),
            self._on_result)

    def _on_result(self, result: ControlChartResult) -> None:
        """Show a worker-computed chart. GUI thread only."""
        self._result = result
        self.canvas.set_result(result)
        self.report.setPlainText(result.report())
        self._fill_violations(result)

    def _fill_violations(self, result: ControlChartResult) -> None:
        self.violations.setRowCount(len(result.violations))
        for row, violation in enumerate(result.violations):
            for column, text in enumerate((
                    f"{violation.rule} — {RULE_NAMES[violation.rule]}",
                    ", ".join(violation.plates),
                    RULE_DETECTS[violation.rule],
                    violation.describe())):
                self.violations.setItem(row, column, QTableWidgetItem(text))
        self.violations.resizeColumnsToContents()

    def _show_refusal(self, message: str) -> None:
        """A refusal is a sentence on screen, never a traceback or a modal."""
        self._result = None
        self.canvas.set_result(None, message=message)
        self.report.setPlainText(message)
        self.violations.setRowCount(0)
        self.failed.emit(message)

    def _on_job_failed(self, message: str) -> None:
        LOG.info("control chart refused: %s", message)
        self._show_refusal(message)

    @property
    def result(self) -> Optional[ControlChartResult]:
        """The chart currently drawn, or ``None`` if the last attempt was
        refused."""
        return self._result

    # -- loading ----------------------------------------------------------
    def choose_table(self) -> None:
        """Ask for a measurement table and load it."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open a measurement table", "",
            "Measurements (*.db *.sqlite *.csv *.tsv);;All files (*)")
        if path:
            self.load_path(path)

    def load_path(self, path: str, table: Optional[str] = None) -> None:
        """Read a CSV or one table of a measurement database, off the GUI
        thread."""
        self._path = path
        names: List[str] = []
        if not str(path).lower().endswith((".csv", ".tsv", ".txt")):
            try:
                names = table_names(path)
            except Exception as exc:
                LOG.info("could not list tables in %s", path, exc_info=True)
                self._source.setText(
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
        self._jobs.cancel()
        self._source.setText(
            f"loading {os.path.basename(path)}"
            + (f" · {chosen}" if chosen else "") + "…")
        self._jobs.submit(
            lambda p=path, t=chosen: (t, read_table(p, t)),
            self._on_frame_loaded)

    def _on_frame_loaded(self, payload) -> None:
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

    def active_jobs(self) -> int:
        """How many worker threads are still winding down."""
        return self._jobs.active_jobs()

    def is_busy(self) -> bool:
        """True while a read or a chart is in flight."""
        return self._jobs.is_busy()

    # -- export -----------------------------------------------------------
    def choose_export(self) -> None:
        """Ask where to write the per-plate table and write it."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Export the chart's points", "control_chart.csv",
            "CSV (*.csv);;All files (*)")
        if path:
            self.export_points(path)

    def export_points(self, path: str) -> Optional[str]:
        """Write one row per plate — value, limits, z, rules fired — as CSV."""
        if self._result is None:
            self._source.setText("Nothing charted yet.")
            return None
        self._result.points_frame().to_csv(path, index=False)
        self._source.setText(f"points written to {os.path.basename(path)}")
        return path

    def closeEvent(self, event):  # noqa: N802 - Qt name
        # Abandon in-flight work rather than let it outlive the screen: Qt
        # aborts the process if a running QThread is destroyed, and a worker
        # delivering into a closed widget is a use-after-free.
        self._jobs.shutdown()
        self.canvas.close()
        super().closeEvent(event)


def make_control_chart_screen(app_key: Optional[str] = None) -> QWidget:
    """Factory handed to :func:`spacr.qt.app.register_app`."""
    return ControlChartScreen()


APP_NAME = "Control Charts"
APP_DESCRIPTION = "Track a control plate by plate and see drift before it ruins a screen"
APP_INTRO = (
    "A campaign's controls are supposed to be the same thing every time, and "
    "when they stop being the same, hit calling and normalisation are already "
    "wrong. Pick the plate column, the run order and the control, and the "
    "chart puts limits round it: an individuals / moving-range chart when a "
    "plate has one control well, X-bar/S when it has several, and a robust "
    "variant when one bad plate would drag the classical limits out. Sigma "
    "comes from short-term variation, never from the spread of the whole "
    "series — that one is inflated by exactly the drift you are looking for. "
    "Limits are estimated from a stated baseline and applied forward, and "
    "every Nelson rule that fires is marked on the plate and named in words, "
    "along with how many false alarms the rule set you chose is worth over a "
    "campaign this long.")
APP_CLI_NOTE = (
    "Control Charts is a picture you read: the zones, the marked plates and "
    "the rule list are the feature. Run it in the GUI (spacr-qt). Headless, "
    "spacr.qt.widgets.control_chart.control_chart(frame, spec) returns the "
    "same limits, the same violations and the same report text with no Qt "
    "involved, so a QC gate in a script can refuse a campaign on it.")
#: The display name in the nine non-English UI languages, in
#: `spacr.qt.i18n.LANGUAGES` order (sv, de, es, zh_CN, pt, hi, ko, is, fr).
APP_NAME_TRANSLATIONS = (
    "Styrdiagram", "Regelkarten", "Gráficos de control",
    "控制图", "Cartas de controlo", "कंट्रोल चार्ट", "관리도",
    "Stýririt", "Cartes de contrôle")


def register() -> bool:
    """Put Control Charts in the app registry. Idempotent.

    Called from :data:`spacr.qt.SELF_REGISTERING_MODULES`, which
    :func:`spacr.qt.run` runs after ``spacr.qt.app`` is fully executed and
    before ``MainWindow.__init__`` reads the registry.

    Everything after ``SECTION_RESULTS`` is a table this key would otherwise
    need a hand-edit in: the screen header and blurb, the "no headless run"
    sentence, the API doc link and the nine translations of the display name.
    :func:`spacr.qt.app.register_app` distributes them from this one call.

    :returns: ``True`` if this call is what registered it. Safe to call again:
        a module imported twice, or a test that re-imports it, must not raise
        on the duplicate key.
    """
    from ..app import APPS, SECTION_RESULTS, STAGE_ALPHA, register_app
    if any(row[0] == APP_KEY for row in APPS):
        return False
    register_app(APP_KEY, APP_NAME, APP_DESCRIPTION, SECTION_RESULTS,
                 factory=make_control_chart_screen, stage=STAGE_ALPHA,
                 intro=APP_INTRO, cli_note=APP_CLI_NOTE,
                 api_module="qt/screens/control_chart",
                 translations=APP_NAME_TRANSLATIONS)
    return True

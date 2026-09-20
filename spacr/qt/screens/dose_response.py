"""Dose–Response — a concentration series, a 4PL curve, and an honest EC50.

The statistics are all in :mod:`spacr.qt.widgets.dose_response` and there is
none in here. This module is the surface: pick the concentration column, the
response column and (optionally) a grouping column, and get one curve per
gene or compound with its EC50 and confidence interval, drawn on the log axis
a dilution series is read on.

Three decisions shape the screen, and all three follow from the engine's
central claim — that the useful answer is sometimes *"this experiment does not
locate the EC50"*, and a screen that cannot render that sentence would undo
the module underneath it.

**Refusals and one-sided bounds are rows, not silence.** A plate where three
compounds fit and the fourth is cytotoxic at the top dose has four rows. The
cytotoxic one says ``refused`` and carries the engine's message; a compound
whose midpoint sits past the highest dose says ``unbounded`` and carries
``EC50 > 30 µM``, with an empty EC50 cell. Dropping either from the table
would turn "we checked and the answer is no" into "no data", which is the one
reading the numbers cannot survive.

**Every number on this screen comes out of the engine.** The table is
:meth:`~spacr.qt.widgets.dose_response.DoseResponseSet.table`, the text is
:meth:`~spacr.qt.widgets.dose_response.DoseResponseResult.report`, and the
plotted line is
:meth:`~spacr.qt.widgets.dose_response.DoseResponseResult.curve`. Nothing is
recomputed here, so the figure, the exported CSV and the sentence a user
pastes into a methods section cannot drift apart.

**The Curve picker offers the asymmetric model and does not choose it.**
Four-parameter is what Fit uses until someone says otherwise, and a
five-parameter fit says in its own report whether the fifth parameter earned
itself. A screen that picked the better-fitting model per group would hand
back a plate whose EC50s came from two different models, compared as though
they had not.

**The fit runs off the GUI thread** through
:class:`spacr.qt.job_runner.JobRunner`, like every other read and compute in
the Qt layer. A profile-likelihood interval on a 96-compound plate is
seconds, not milliseconds, and ``threaded=False`` runs the identical code
inline so a test drives the same path the shipped screen does.

:func:`register` is **not** called at import; read its docstring.
"""
from __future__ import annotations

from dataclasses import replace

import logging
import os
from typing import List, Optional

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QFileDialog, QHBoxLayout, QLabel,
    QLineEdit, QPlainTextEdit, QPushButton, QSplitter, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..job_runner import JobRunner
from ..theme import SPACING, active_palette, mark_surface
from ..widgets.dose_response import (
    CI_PROFILE, CI_WALD, MODEL_4PL, MODEL_5PL, STATUS_FITTED, STATUS_REFUSED,
    STATUS_UNBOUNDED, DoseResponseError, DoseResponseResult, DoseResponseSet,
    DoseResponseSpec, candidate_concentration_columns,
    candidate_response_columns, fit_frame,
)
from ..widgets.graph_builder import (_canvas_class, _page_surface_axes,
                                     categorical_colours)
from ..widgets.graph_spec import CATEGORICAL, column_kinds
from .graph_builder import read_table, table_names
from .app_screen import ModuleHeader
from ..i18n import set_translatable_text, tr
from ..widgets.dose_response import (PERCENT_COLUMN, PlateSpec,
                                     normalise_to_controls, pool_frame,
                                     selectivity_index, SYNERGY_BLISS,
                                     SYNERGY_LOEWE, bliss_surface,
                                     checkerboard_from_frame,
                                     fit_dose_response, loewe_surface)

LOG = logging.getLogger("spacr.qt.screens.dose_response")

__all__ = ["DoseResponseScreen", "make_dose_response_screen", "register",
           "APP_KEY", "APP_NAME", "APP_DESCRIPTION", "APP_INTRO",
           "APP_CLI_NOTE", "APP_NAME_TRANSLATIONS", "NO_GROUP",
           "TABLE_COLUMNS", "NOTE_WIDTH"]

#: Characters of a refusal message the grid cell shows before eliding. The
#: engine's messages are paragraphs on purpose; the whole text is on the
#: cell's tooltip and in the report pane.
NOTE_WIDTH = 90

#: The registry key. Chosen once and never renamed.
APP_KEY = "dose_response"

#: What the grouping picker calls "fit the whole table as one curve".
NO_GROUP = "(one curve for the whole table)"

#: The results grid, as ``(engine key, header)``. A subset of
#: :meth:`~spacr.qt.widgets.dose_response.DoseResponseSet.table`'s columns —
#: the ones that fit on screen — in the order a reader wants them: what it is,
#: whether to believe it, the number, then the diagnostics.
TABLE_COLUMNS = (
    ("group", "Group"),
    ("status", "Status"),
    ("n", "n"),
    ("concentrations", "Doses"),
    ("ec50", "EC50"),
    ("ec50_low", "CI low"),
    ("ec50_high", "CI high"),
    ("hill", "Hill"),
    ("r_squared", "R²"),
    ("lack_of_fit_p", "Lack-of-fit p"),
    ("note", "Note"),
)
from ..widgets.toggle import Toggle
from ..widgets.sortable_table import install_sorting, table_item
from ..app_catalog import declared_app, register_declared

#: Substrings that make a column the first guess for the dose axis. A
#: convenience for the common column names, not a classifier — nothing is
#: fitted until the user presses Fit.
_CONCENTRATION_HINTS = ("conc", "dose", "µm", "um", "nm", "mm", "molar")

#: What the plate and control pickers call "not chosen". An existing caption,
#: so the Plates row adds nothing a translator has not already seen.
_NO_COLUMN = "(none)"

#: The picker entries whose caption follows the language. Every other entry
#: is a column name or a control level, which is data and never translated.
_SENTINELS = (NO_GROUP, _NO_COLUMN)

#: Substrings that make a control-column level the first guess for each
#: control. A convenience, like `_CONCENTRATION_HINTS`: nothing is normalised
#: until a plate and a control column have both been chosen.
_POSITIVE_HINTS = ("pos", "kill", "max")
_NEGATIVE_HINTS = ("neg", "vehicle", "dmso", "mock")


def _fit_with_plates(frame, spec, plate_spec, plate_column=None,
                     host_column=None, second_dose=None,
                     synergy_model=SYNERGY_BLISS):
    """Fit ``frame``, normalising and pooling across plates as asked.

    :param frame: the loaded table.
    :param spec: the fit the pickers describe.
    :param plate_spec: which plate and control columns to normalise by, or
        ``None`` to fit the raw response exactly as before.
    :param plate_column: the replicate column to pool each group across, or
        ``None`` for no pooling. It does not need controls: every plate is
        fitted on its own scale, so an EC50 is comparable across plates even
        when their raw signals are not.
    :param host_column: a second readout from the same wells -- host-cell
        count, viability -- or ``None``. Each group is fitted on it too, and
        its host EC50 divided by the response EC50 is the selectivity index.
    :param second_dose: a second compound's dose column, for a checkerboard,
        or ``None``. With one, the curves, pooling and host readout use only
        the wells where it is zero, and each group's combination wells are
        scored against ``synergy_model``.
    :param synergy_model: :data:`SYNERGY_BLISS` or :data:`SYNERGY_LOEWE`.
    :returns: ``(result set, plate reports, pooled fits, selectivity,
        synergy)``. The
        reports are empty when nothing was normalised; the pooled fits map
        each group to its :class:`~spacr.qt.widgets.dose_response.PooledFit`,
        or to the engine's sentence when pooling was refused; selectivity
        maps each group to its
        :class:`~spacr.qt.widgets.dose_response.SelectivityIndex`; synergy
        maps each group to its interaction surface, or to the engine's
        sentence when the table is not a checkerboard it can score.

    RAW RESPONSES ARE NOT COMPARABLE ACROSS PLATES, which is the engine's
    argument for normalising and the screen's for offering it: two plates read
    on different days differ in absolute signal by more than most compounds
    move it. Each plate is scaled by its own controls, so positive reads 100
    and negative reads 0, and a plate without a usable pair is left out of the
    fit and says why instead of being scaled by someone else's controls.
    """
    reports = ()
    fitted_frame, fitted_spec = frame, spec
    if plate_spec is not None:
        fitted_frame, reports = normalise_to_controls(
            frame, plate_spec, response=spec.response)
        fitted_spec = replace(spec, response=PERCENT_COLUMN)
    # THE CURVES ARE THE FIRST COMPOUND ALONE when a second one is named.
    # Combination wells are not a dose series of either agent, and fitting them
    # into one would describe neither; the synergy surface is where they count.
    single = _alone(fitted_frame, second_dose)
    result = fit_frame(single, fitted_spec)
    pooled = (_pool_each_group(single, fitted_spec, plate_column)
              if plate_column else {})
    selectivity = {}
    if host_column:
        # THE HOST READOUT IS FITTED RAW, on the table as loaded. Plate
        # normalisation scales the RESPONSE by that response's own controls;
        # a host readout has different controls, or none, and an EC50 does not
        # need them -- it is a concentration, not a percentage.
        host = fit_frame(_alone(frame, second_dose),
                         replace(spec, response=host_column))
        for fit in result:
            host_fit = host.get(fit.group)
            selectivity[fit.group] = selectivity_index(
                fit.result, None if host_fit is None else host_fit.result)
    synergy = (_score_each_group(fitted_frame, fitted_spec, second_dose,
                                 synergy_model)
               if second_dose else {})
    return result, reports, pooled, selectivity, synergy


def _alone(frame, column):
    """The rows where ``column`` is zero, or every row when there is none.

    :param frame: the table.
    :param column: a second compound's dose column, or ``None``.
    :returns: the single-agent rows of the first compound, vehicles included.
    """
    if not column:
        return frame
    return frame[pd.to_numeric(frame[column], errors="coerce") == 0]


def _score_each_group(frame, spec, second_dose, model):
    """One interaction surface per group of a two-compound checkerboard.

    :param frame: the table -- normalised, when it was.
    :param spec: the grid's spec; its concentration is the first compound.
    :param second_dose: the second compound's dose column.
    :param model: :data:`SYNERGY_BLISS` or :data:`SYNERGY_LOEWE`.
    :returns: group -> surface, or group -> the refusal sentence.

    BOTH MODELS PREDICT THE COMBINATION FROM THE SINGLE AGENTS, so each
    compound's alone-axis is fitted first from the board's own wells, and a
    board missing either axis is refused with the engine's reason rather than
    scored against itself.
    """
    if spec.group is None:
        levels = [("", frame)]
    else:
        levels = [(str(level), rows) for level, rows in
                  frame.groupby(frame[spec.group].astype(str), sort=False)]
    one_curve = replace(spec, group=None)
    surface_for = loewe_surface if model == SYNERGY_LOEWE else bliss_surface
    scored = {}
    for level, rows in levels:
        try:
            board = checkerboard_from_frame(
                rows, dose_a=spec.concentration, dose_b=second_dose,
                response=spec.response)
            fit_a = fit_dose_response(*board.a_alone, one_curve,
                                      group=f"{spec.concentration} alone")
            fit_b = fit_dose_response(*board.b_alone, one_curve,
                                      group=f"{second_dose} alone")
            scored[level] = surface_for(board.dose_a, board.dose_b,
                                        board.response, fit_a=fit_a,
                                        fit_b=fit_b)
        except DoseResponseError as refusal:
            scored[level] = str(refusal)
    return scored


def _excess_grid(surface):
    """The surface itself, as rows of text: first compound down, second across.

    :param surface: an interaction surface.
    :returns: one header line and one line per dose of the first compound.

    THE SURFACE IS THE RESULT AND A SINGLE INDEX IS NOT, which is the engine's
    own argument: synergy that lives at one corner of the board and
    antagonism at another average to nothing in one number.
    """
    width = 8
    head = " " * width + "".join(f"{_format(dose):>{width}}"
                                 for dose in surface.dose_b)
    rows = [head]
    for i, dose in enumerate(surface.dose_a):
        cells = "".join(
            f"{value:>+{width}.2f}" if np.isfinite(value) else f"{'—':>{width}}"
            for value in surface.excess[i])
        rows.append(f"{_format(dose):>{width}}{cells}")
    return rows


def _pool_each_group(frame, spec, plate):
    """One pooled EC50 per group, with plate as a random effect.

    :param frame: the table the grid was fitted on -- normalised, when it was.
    :param spec: the grid's spec; each group is pooled as one curve.
    :param plate: the replicate column.
    :returns: group -> pooled fit, or group -> the refusal sentence.

    `pool_frame` fits every row of a plate as ONE curve, so a table holding
    several compounds is split by group first; pooling the whole plate would
    average a dozen compounds into one meaningless EC50. A group seen on a
    single plate is not pooled at all: one plate is one fit, already in the
    grid, and a "pooled" number over one replicate would claim a
    reproducibility nobody measured.
    """
    if spec.group is None:
        levels = [("", frame)]
    else:
        levels = [(str(level), rows) for level, rows in
                  frame.groupby(frame[spec.group].astype(str), sort=False)]
    one_curve = replace(spec, group=None)
    pooled = {}
    for level, rows in levels:
        if rows[plate].astype(str).nunique() < 2:
            continue
        try:
            pooled[level] = pool_frame(rows, one_curve, plate=plate)
        except DoseResponseError as refusal:
            pooled[level] = str(refusal)
    return pooled


#: How a status reads in the grid. The engine's words, spelled for a human.
_STATUS_LABELS = {
    STATUS_FITTED: "fitted",
    STATUS_UNBOUNDED: "unbounded",
    STATUS_REFUSED: "refused",
}

#: Captions this screen shows through a variable, so the runtime catalog
#: generator cannot find them at a literal call site and imports this set
#: instead -- the same arrangement as `_GENE_TILE_UI_SOURCES`. Without it the
#: results-grid headers and the status words were in no catalog at all, and
#: every language showed them in English. "n", "EC50", "Hill" and "R²" are
#: symbols rather than words and are left out on purpose; `tr` passes them
#: through unchanged.
_DOSE_RESPONSE_UI_SOURCES = frozenset({
    NO_GROUP,
    "Group", "Status", "Doses", "CI low", "CI high", "Lack-of-fit p",
    "fitted", "unbounded", "refused",
})



def _format(value) -> str:
    """One cell of the results grid, as text.

    ``NaN`` becomes an em dash rather than the string ``nan``: a blank EC50 on
    an unbounded row is a deliberate absence, and ``nan`` reads as a bug.
    """
    if value is None:
        return "—"
    if isinstance(value, float):
        if not np.isfinite(value):
            return "—"
        return f"{value:.4g}"
    return str(value)


class DoseResponseScreen(QWidget):
    """Load a concentration series, fit a 4PL per group, and read the EC50s.

    :param parent: the usual Qt parent.
    :param threaded: ``False`` runs the table read and the fit inline,
        emitting the same signals in the same order, so a test drives the
        screen synchronously without the behaviour diverging.
    """

    def __init__(self, parent=None, *, threaded: bool = True):
        """Build the screen: the curve canvas beside the fit table and report.

        :param parent: parent widget, or ``None``.
        :param threaded: read and fit on a worker thread. Set ``False`` in tests
            so ``fit`` finishes before it returns.
        """
        super().__init__(parent)
        self.setObjectName("DoseResponseScreen")
        self._frame: Optional[pd.DataFrame] = None
        self._path: Optional[str] = None
        self._set: Optional[DoseResponseSet] = None
        #: What each plate's controls said on the last normalised fit, in the
        #: order the plates appear. Empty when the fit read the raw response.
        self._plate_reports = ()
        #: Group -> pooled fit (or the refusal sentence) from the last fit
        #: that had a plate column. Empty when nothing was pooled.
        self._pooled = {}
        #: Group -> selectivity index from the last fit that had a host
        #: readout. Empty when none was chosen.
        self._selectivity = {}
        #: Group -> interaction surface (or the refusal sentence) from the
        #: last fit that named a second compound. Empty when none was.
        self._synergy = {}
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
            instruction="Load a table, pick the concentration and response "
                        "columns, then fit.",
        )
        self._header = header
        head.addWidget(header)

        self._source = QLabel("no table loaded", self)
        self._source.setObjectName("DoseResponseSourceLabel")
        head.addWidget(self._source, 1)

        self._table_picker = QComboBox(self)
        self._table_picker.setObjectName("DoseResponseTablePicker")
        self._table_picker.setToolTip("Which table of the database to fit")
        self._table_picker.setVisible(False)
        self._table_picker.setProperty("i18nSkipItems", True)
        self._table_picker.currentTextChanged.connect(self._on_table_picked)
        head.addWidget(self._table_picker)

        load = QPushButton("Load table…", self)
        load.setObjectName("PrimaryButton")
        load.setToolTip("A measurements.db, or a CSV of dose and response")
        load.clicked.connect(self.choose_table)
        head.addWidget(load)
        outer.addLayout(head)

        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(SPACING["sm"])

        controls.addWidget(QLabel("Concentration", self))
        self.concentration_picker = QComboBox(self)
        self.concentration_picker.setObjectName("DoseResponseConcentration")
        self.concentration_picker.setToolTip(
            "The dose column. A concentration of 0 is read as a vehicle "
            "control: it is excluded from the fit and reported as a "
            "reference, never fed to log10.")
        controls.addWidget(self.concentration_picker)

        controls.addWidget(QLabel("Response", self))
        self.response_picker = QComboBox(self)
        self.response_picker.setObjectName("DoseResponseResponse")
        controls.addWidget(self.response_picker)

        controls.addWidget(QLabel("One curve per", self))
        self.group_picker = QComboBox(self)
        self.group_picker.setObjectName("DoseResponseGroup")
        self.group_picker.setToolTip("A gene or compound column, or nothing")
        controls.addWidget(self.group_picker)

        controls.addWidget(QLabel("Unit", self))
        self.unit_edit = QLineEdit(self)
        self.unit_edit.setObjectName("DoseResponseUnit")
        self.unit_edit.setPlaceholderText("µM")
        self.unit_edit.setMaximumWidth(70)
        self.unit_edit.setToolTip(
            "Cosmetic only — it never enters the arithmetic")
        controls.addWidget(self.unit_edit)

        controls.addWidget(QLabel("Interval", self))
        self.ci_picker = QComboBox(self)
        self.ci_picker.setObjectName("DoseResponseCI")
        self.ci_picker.addItem("Profile likelihood (can decline to close)",
                               CI_PROFILE)
        self.ci_picker.addItem("Wald (symmetric, always finite)", CI_WALD)
        self.ci_picker.setToolTip(
            "The Wald interval is finite even when the data does not "
            "determine the EC50. The profile interval can report an open "
            "side, which is why it is the default.")
        controls.addWidget(self.ci_picker)

        controls.addWidget(QLabel("Curve", self))
        self.model_picker = QComboBox(self)
        self.model_picker.setObjectName("DoseResponseModel")
        self.model_picker.addItem("Four-parameter logistic (symmetric)",
                                  MODEL_4PL)
        self.model_picker.addItem("Five-parameter logistic (asymmetric)",
                                  MODEL_5PL)
        self.model_picker.setToolTip(
            "A 4PL is symmetric about its midpoint. Real asymmetry exists, "
            "and a 4PL absorbs it into a displaced EC50; the 5PL fits it with "
            "one more parameter, needs one more concentration, and says "
            "whether that parameter earned itself.")
        controls.addWidget(self.model_picker)

        self.force_check = Toggle("Fit non-monotone data", self)
        self.force_check.setObjectName("DoseResponseForce")
        self.force_check.setToolTip(
            "A bell-shaped series is refused by default: it is not a 4PL, "
            "and the usual cause is cytotoxicity at the top dose. Ticking "
            "this fits it anyway and keeps the warning on the result.")
        controls.addWidget(self.force_check)

        # "FIT CURVE", NOT "FIT". The bare word is also zoom-to-fit in the
        # ortho view, comparison grid and layer viewer, and a reviewed
        # translation is keyed by its English source, so one string could
        # never be translated right for both meanings.
        self.fit_button = QPushButton("Fit curve", self)
        self.fit_button.setObjectName("PrimaryButton")
        self.fit_button.clicked.connect(self.fit)
        self.fit_button.setEnabled(False)
        controls.addWidget(self.fit_button)
        controls.addStretch(1)
        outer.addLayout(controls)

        # THE PLATES ROW. Every caption on it already exists elsewhere in the
        # application, so it adds no string a translator has not seen. Both
        # column pickers start at "(none)", which keeps the default fit
        # exactly the raw-response fit it always was.
        plates = QHBoxLayout()
        plates.setContentsMargins(0, 0, 0, 0)
        plates.setSpacing(SPACING["sm"])
        plates.addWidget(QLabel("Plate", self))
        self.plate_picker = QComboBox(self)
        self.plate_picker.setObjectName("DoseResponsePlate")
        plates.addWidget(self.plate_picker)
        plates.addWidget(QLabel("Controls", self))
        self.control_picker = QComboBox(self)
        self.control_picker.setObjectName("DoseResponseControl")
        self.control_picker.currentIndexChanged.connect(
            lambda _index: self._on_control_picked(
                (self.control_picker.currentData() or self.control_picker.currentText())))
        plates.addWidget(self.control_picker)
        plates.addWidget(QLabel("Positive control wells", self))
        self.positive_picker = QComboBox(self)
        self.positive_picker.setObjectName("DoseResponsePositive")
        plates.addWidget(self.positive_picker)
        plates.addWidget(QLabel("Negative control wells", self))
        self.negative_picker = QComboBox(self)
        self.negative_picker.setObjectName("DoseResponseNegative")
        plates.addWidget(self.negative_picker)
        plates.addStretch(1)
        outer.addLayout(plates)

        # THE HOST READOUT. A second column from the same wells; with it, each
        # group's host EC50 over its response EC50 is the selectivity index --
        # the number that decides whether an anti-parasitic compound is worth
        # anything, because killing the parasite at 1 uM means nothing if the
        # host monolayer dies at 1.2.
        hosts = QHBoxLayout()
        hosts.setContentsMargins(0, 0, 0, 0)
        hosts.setSpacing(SPACING["sm"])
        hosts.addWidget(QLabel("Host response", self))
        self.host_picker = QComboBox(self)
        self.host_picker.setObjectName("DoseResponseHost")
        self.host_picker.setToolTip(
            "A second readout from the same wells, such as the host-cell "
            "count. Each group's host EC50 divided by its response EC50 is "
            "the selectivity index.")
        hosts.addWidget(self.host_picker)
        hosts.addStretch(1)
        outer.addLayout(hosts)

        # THE SECOND COMPOUND. Naming its dose column turns the table into a
        # checkerboard: the curves use the first compound alone, and the
        # combination wells are scored against Bliss or Loewe.
        combos = QHBoxLayout()
        combos.setContentsMargins(0, 0, 0, 0)
        combos.setSpacing(SPACING["sm"])
        combos.addWidget(QLabel("Second compound", self))
        self.second_dose_picker = QComboBox(self)
        self.second_dose_picker.setObjectName("DoseResponseSecondDose")
        self.second_dose_picker.setToolTip(
            "A second dose column, for a two-compound checkerboard. The "
            "curves then use only the wells without it, and each group's "
            "combination wells are scored against the chosen model.")
        combos.addWidget(self.second_dose_picker)
        combos.addWidget(QLabel("Model", self))
        self.synergy_picker = QComboBox(self)
        self.synergy_picker.setObjectName("DoseResponseSynergyModel")
        self.synergy_picker.addItem("Bliss independence", SYNERGY_BLISS)
        self.synergy_picker.addItem("Loewe additivity", SYNERGY_LOEWE)
        combos.addWidget(self.synergy_picker)
        combos.addStretch(1)
        outer.addLayout(combos)

        body = QSplitter(Qt.Horizontal, self)
        body.setChildrenCollapsible(False)

        from matplotlib.figure import Figure
        palette = active_palette()
        self._figure = Figure(figsize=(6.5, 4.6))
        self.canvas = _canvas_class()(self._figure)
        self.canvas.setObjectName("DoseResponseCanvas")
        body.addWidget(self.canvas)

        side = QSplitter(Qt.Vertical, self)
        side.setChildrenCollapsible(False)
        self.table = QTableWidget(0, len(TABLE_COLUMNS), self)
        install_sorting(self.table)
        self.table.setObjectName("DoseResponseTable")
        self.table.setHorizontalHeaderLabels(
            [tr(header) for _key, header in TABLE_COLUMNS])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.itemSelectionChanged.connect(self._on_row_selected)
        side.addWidget(self.table)

        self.report = QPlainTextEdit(self)
        self.report.setObjectName("DoseResponseReport")
        self.report.setReadOnly(True)
        self.report.setPlaceholderText(
            "Pick a concentration column and a response column, then Fit.")
        side.addWidget(self.report)
        mark_surface(self.table, self.report)
        side.setStretchFactor(0, 1)
        side.setStretchFactor(1, 1)

        body.addWidget(side)
        body.setStretchFactor(0, 3)
        body.setStretchFactor(1, 2)
        outer.addWidget(body, 1)
        from ..dnd import install_for
        install_for(self, "dose_response")
        from .settings_model import retarget_field_tooltips
        retarget_field_tooltips(self)

    def set_frame(self, frame: pd.DataFrame, *, label: str = "") -> None:
        """Offer ``frame``'s columns and wait to be told which ones to fit.

        The one call a host needs. It deliberately does **not** fit: which
        column is the dose is not guessable from a measurement table, and a
        curve through the wrong pair of columns is worse than an empty axis.
        """
        self._frame = frame
        self._set = None
        doses = candidate_concentration_columns(frame)
        responses = candidate_response_columns(frame)
        kinds = column_kinds(frame)
        groups = [name for name, kind in sorted(kinds.items())
                  if kind == CATEGORICAL]

        self._refill(self.concentration_picker, doses,
                     prefer=_CONCENTRATION_HINTS)
        self._refill(self.response_picker, responses)
        self._refill(self.group_picker, [NO_GROUP] + groups)
        self._refill(self.plate_picker,
                     [_NO_COLUMN] + [str(name) for name in frame.columns])
        self._refill(self.control_picker, [_NO_COLUMN] + groups)
        self._on_control_picked((self.control_picker.currentData() or self.control_picker.currentText()))
        self._refill(self.host_picker, [_NO_COLUMN] + list(responses))
        self._refill(self.second_dose_picker, [_NO_COLUMN] + list(doses))
        self._plate_reports = ()
        self._pooled = {}
        self._selectivity = {}
        self._synergy = {}
        self.fit_button.setEnabled(bool(doses and responses))
        self.table.setRowCount(0)
        self.report.setPlainText("")
        self._draw(None)
        if not doses:
            self.report.setPlainText(tr(
                "No column of this table has at least four distinct positive "
                "values, so none of them can be a dilution series. A "
                "dose–response needs the concentration itself, not a log "
                "dose and not a plate coordinate."))
        if label:
            self._source.setText(label)
        else:
            set_translatable_text(
                self._source, "{rows} rows × {columns} columns",
                rows=f"{len(frame):,}", columns=len(frame.columns))

    @staticmethod
    def _refill(picker: QComboBox, values, prefer=()) -> None:
        """Replace a picker's items, keeping the choice the user already made.

        ``prefer`` is a list of substrings; the first item whose name contains
        one wins the initial selection. It is a convenience and never more
        than that — the screen refuses to guess hard enough to fit anything
        without being asked, because a curve through the wrong pair of columns
        is worse than an empty axis.

        Every entry carries its value as item data, and handlers read
        ``currentData()``. The language pass rewrites a dropdown's item text by
        exact catalog match, so a column called ``gene`` or an untouched
        "(none)" read back as text reached the fit translated -- "gène" and
        "(Aucune)" on a French screen, neither of which the table has. Only
        the sentinels in ``_SENTINELS`` are offered to that pass; a column name
        is recorded as an empty source, which it leaves alone. An entry
        added without data is read by its caption, as before.
        """
        options = list(values)
        previous = picker.currentData() or picker.currentText()
        picker.blockSignals(True)
        picker.clear()
        for value in options:
            picker.addItem(tr(value) if value in _SENTINELS else value, value)
        picker._spacr_i18n_item_sources = [
            value if value in _SENTINELS else "" for value in options]
        if previous and previous in options:
            picker.setCurrentIndex(options.index(previous))
        elif prefer:
            for index, name in enumerate(options):
                if any(hint in str(name).lower() for hint in prefer):
                    picker.setCurrentIndex(index)
                    break
        picker.blockSignals(False)

    def _on_control_picked(self, name: str) -> None:
        """Offer the chosen control column's levels as the two controls.

        :param name: the control column, or "(none)".
        """
        levels = []
        frame = self._frame
        if frame is not None and name not in ("", _NO_COLUMN) \
                and name in frame.columns:
            levels = sorted({str(value) for value in frame[name].dropna()})
        self._refill(self.positive_picker, levels, prefer=_POSITIVE_HINTS)
        self._refill(self.negative_picker, levels, prefer=_NEGATIVE_HINTS)

    def _plate_spec(self) -> Optional[PlateSpec]:
        """The plate normalisation the pickers describe, or ``None``.

        :returns: ``None`` while either column picker reads "(none)".
        :raises DoseResponseError: when the columns are chosen but a control
            level is not, with the engine's sentence saying which.
        """
        plate = (self.plate_picker.currentData() or self.plate_picker.currentText())
        control = (self.control_picker.currentData() or self.control_picker.currentText())
        if plate in ("", _NO_COLUMN) or control in ("", _NO_COLUMN):
            return None
        positive = (self.positive_picker.currentData() or self.positive_picker.currentText())
        negative = (self.negative_picker.currentData() or self.negative_picker.currentText())
        return PlateSpec(plate=plate, control=control,
                         positive=(positive,) if positive else (),
                         negative=(negative,) if negative else ())

    def _with_plates(self, text: str) -> str:
        """Prefix ``text`` with plate verdicts, pooled EC50s and selectivity.

        Both lead because they decide what the curve below them means: a
        refused plate is not in the fit, and a pooled EC50 is the number a
        reader should quote when there are replicates -- they should meet both
        before a single plate's curve.

        :param text: the report the pane would otherwise show.
        :returns: the report, with plate and pooled lines first when there are
            any.
        """
        lines = []
        for report in self._plate_reports:
            row = report.summary_row()
            zprime = row["zprime"]
            shown = "—" if zprime is None or not np.isfinite(zprime) \
                else f"{zprime:.2f}"
            if report.usable:
                line = tr("{plate}: usable, Z′ {zprime}",
                          plate=row["plate"], zprime=shown)
            else:
                line = tr("{plate}: refused, Z′ {zprime}",
                          plate=row["plate"], zprime=shown)
            if not report.usable and row["note"]:
                line += f" — {row['note']}"
            lines.append(line)
        if lines and self._pooled:
            lines.append("")
        for group, pooled in self._pooled.items():
            name = group or tr("all rows")
            if isinstance(pooled, str):
                lines.append(tr("{name}: not pooled — {reason}",
                                name=name, reason=pooled))
                continue
            row = pooled.summary_row()
            if row["status"] != STATUS_FITTED:
                lines.append(tr("{name}: not pooled — {reason}",
                                name=name, reason=row["note"]))
                continue
            i_squared = row["i_squared"]
            spread = "—" if not np.isfinite(i_squared) else f"{i_squared:.0%}"
            unit = f" {row['unit']}" if row["unit"] else ""
            line = tr("{name}: pooled EC50 {ec50}{unit} ({low}–{high}) across "
                      "{used} of {plates} plates, I² {spread}",
                      name=name, ec50=_format(row["ec50"]), unit=unit,
                      low=_format(row["ec50_low"]),
                      high=_format(row["ec50_high"]),
                      used=row["n_used"], plates=row["n_plates"],
                      spread=spread)
            if not pooled.reproducible:
                line = tr("{line}, plates disagree", line=line)
            if row["note"]:
                line += f" — {row['note']}"
            lines.append(line)
        if lines and self._selectivity:
            lines.append("")
        for group, index in self._selectivity.items():
            name = group or tr("all rows")
            row = index.summary_row()
            if row["status"] == STATUS_REFUSED:
                lines.append(tr("{name}: no selectivity index — {reason}",
                                name=name, reason=row["note"]))
                continue
            line = tr("{name}: selectivity index {index} ({low}–{high}), "
                      "host EC50 {host} over response EC50 {response}",
                      name=name, index=_format(row["selectivity_index"]),
                      low=_format(row["si_low"]), high=_format(row["si_high"]),
                      host=_format(row["host_ec50"]),
                      response=_format(row["pathogen_ec50"]))
            if row["note"]:
                line += f" — {row['note']}"
            lines.append(line)
        for group, surface in self._synergy.items():
            if lines and lines[-1] != "":
                lines.append("")
            name = group or tr("all rows")
            if isinstance(surface, str):
                lines.append(tr("{name}: no synergy surface — {reason}",
                                name=name, reason=surface))
                continue
            summary = surface.summary()
            model = "Bliss" if surface.model == SYNERGY_BLISS else "Loewe"
            note = f" — {summary['note']}" if summary.get("note") else ""
            if not summary["n_cells"]:
                lines.append(tr("{name}: no combination well could be scored "
                                "against {model}", name=name, model=model)
                             + note)
                continue
            lines.append(tr(
                "{name}: {model} excess over {cells} combination wells, max "
                "{max} at {dose_a} + {dose_b}, min {min}; {synergistic} "
                "synergistic, {antagonistic} antagonistic",
                name=name, model=model, cells=summary["n_cells"],
                max=f"{summary['max_excess']:+.2f}",
                dose_a=_format(summary["max_at_dose_a"]),
                dose_b=_format(summary["max_at_dose_b"]),
                min=f"{summary['min_excess']:+.2f}",
                synergistic=summary["synergistic_cells"],
                antagonistic=summary["antagonistic_cells"]) + note)
            lines.extend(_excess_grid(surface))
        if not lines:
            return text
        return "\n".join(lines) + "\n\n" + text

    def choose_table(self) -> None:
        """Ask for a file and load it."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open a dose–response table", "",
            "Measurements (*.db *.sqlite *.csv *.tsv);;All files (*)")
        if path:
            self.load_path(path)

    def load_path(self, path: str, table: Optional[str] = None) -> None:
        """Load a CSV or one table of a SQLite measurement database.

        The read runs on a worker thread through :class:`JobRunner`; listing
        the table names stays inline because the picker has to be populated
        before the read is dispatched, to know which table to read. The same
        shape :class:`spacr.qt.screens.trellis.TrellisScreen` uses, for the
        same reasons.
        """
        self._path = path
        names: List[str] = []
        if not str(path).lower().endswith((".csv", ".tsv", ".txt")):
            try:
                names = table_names(path)
            except Exception as exc:
                LOG.info("could not list tables in %s", path, exc_info=True)
                set_translatable_text(
                    self._source, "could not read {name}: {reason}",
                    name=os.path.basename(path), reason=exc)
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
        set_translatable_text(
            self._source, "loading {name}…",
            name=os.path.basename(path) + (f" · {chosen}" if chosen else ""))
        self._jobs.submit(
            lambda p=path, t=chosen: (t, read_table(p, t)),
            self._on_frame_loaded)

    def _on_frame_loaded(self, payload) -> None:
        """Hand a worker-read frame to the pickers. GUI thread only."""
        chosen, frame = payload
        path = self._path or ""
        suffix = f" · {chosen}" if chosen else ""
        self.set_frame(frame)
        set_translatable_text(
            self._source, "{name} · {rows} rows × {columns} columns",
            name=f"{os.path.basename(path)}{suffix}",
            rows=f"{len(frame):,}", columns=len(frame.columns))

    def _on_table_picked(self, name: str) -> None:
        """Reload the current database at a newly chosen table.

        :param name: the table to read; a blank one, or no loaded path, does
            nothing.
        """
        if self._path and name:
            self.load_path(self._path, table=name)

    def _on_job_failed(self, message: str) -> None:
        """Every worker failure lands in the report pane; no modal dialogs."""
        LOG.info("dose–response job failed: %s", message)
        self._source.setText(message)
        self.report.setPlainText(message)

    def spec(self) -> DoseResponseSpec:
        """The spec the controls currently describe."""
        group = (self.group_picker.currentData() or self.group_picker.currentText())
        return DoseResponseSpec(
            concentration=(self.concentration_picker.currentData() or self.concentration_picker.currentText()),
            response=(self.response_picker.currentData() or self.response_picker.currentText()),
            group=None if group in ("", NO_GROUP) else group,
            ci_method=self.ci_picker.currentData() or CI_PROFILE,
            unit=self.unit_edit.text().strip(),
            allow_non_monotone=self.force_check.isChecked(),
            model=self.model_picker.currentData() or MODEL_4PL)

    def fit(self) -> None:
        """Fit every group, off the GUI thread."""
        if self._frame is None:
            return
        try:
            spec = self.spec()
            plate_spec = self._plate_spec()
            plate_column = (self.plate_picker.currentData() or self.plate_picker.currentText())
            plate_column = (None if plate_column in ("", _NO_COLUMN)
                            else plate_column)
            host_column = (self.host_picker.currentData() or self.host_picker.currentText())
            host_column = (None if host_column in ("", _NO_COLUMN)
                           else host_column)
            second_dose = (self.second_dose_picker.currentData() or self.second_dose_picker.currentText())
            second_dose = (None if second_dose in ("", _NO_COLUMN)
                           else second_dose)
            synergy_model = self.synergy_picker.currentData() or SYNERGY_BLISS
        except DoseResponseError as exc:
            self.report.setPlainText(str(exc))
            return
        frame = self._frame
        self._jobs.cancel()
        self.report.setPlainText("fitting…")
        self._jobs.submit(
            lambda: _fit_with_plates(frame, spec, plate_spec, plate_column,
                                     host_column, second_dose,
                                     synergy_model),
            self._on_fitted)

    def _on_fitted(self, result) -> None:
        """Fill the grid from the engine's table. GUI thread only.

        :param result: the fit, or ``(fit, plate reports, pooled fits,
            selectivity, synergy)`` as the fitting job hands it back.
        """
        reports, pooled, selectivity, synergy = (), {}, {}, {}
        if isinstance(result, tuple):
            result, reports, *rest = result
            pooled = rest[0] if rest else {}
            selectivity = rest[1] if len(rest) > 1 else {}
            synergy = rest[2] if len(rest) > 2 else {}
        self._plate_reports = tuple(reports)
        self._pooled = dict(pooled)
        self._selectivity = dict(selectivity)
        self._synergy = dict(synergy)
        self._set = result
        rows = result.table()
        self.table.setRowCount(len(rows))
        for row in range(len(rows)):
            record = rows.iloc[row]
            status = str(record["status"])
            for column, (key, _header) in enumerate(TABLE_COLUMNS):
                value = record[key]
                if key == "status":
                    value = tr(_STATUS_LABELS.get(status, status))
                if key == "group" and not str(value):
                    value = tr("all rows")
                text = _format(value)
                if key == "note" and len(text) > NOTE_WIDTH:
                    text = text[:NOTE_WIDTH].rstrip() + "…"
                item = table_item(text)
                if status != STATUS_FITTED:
                    item.setToolTip(str(record["note"]))
                if column == 0:
                    item.setData(Qt.UserRole, row)
                self.table.setItem(row, column, item)
        self.table.resizeColumnsToContents()
        if len(rows):
            self.table.clearSelection()
            self.table.selectRow(0)
        else:
            self.report.setPlainText(self._with_plates(result.report()))
            self._draw(None)

    def _on_row_selected(self) -> None:
        """Show the curve for the selected row.

        The row carries the index of the fit it was built from rather than being
        identified by its position: the table sorts, so the top row is not
        always the first curve.
        """
        rows = {index.row() for index in self.table.selectedIndexes()}
        if not rows or self._set is None:
            return
        item = self.table.item(sorted(rows)[0], 0)
        fit = None if item is None else item.data(Qt.UserRole)
        self.show_group(sorted(rows)[0] if fit is None else int(fit))

    def show_group(self, index: int) -> None:
        """Draw and describe the ``index``-th curve of the last fit."""
        if self._set is None or not 0 <= index < len(self._set.fits):
            return
        fit = self._set.fits[index]
        if fit.result is not None:
            self.report.setPlainText(self._with_plates(fit.result.report()))
        else:
            self.report.setPlainText(self._with_plates(
                tr("{name}: REFUSED", name=fit.group or tr("all rows"))
                + f"\n\n{fit.error}"))
        self._draw(index)

    def _draw(self, selected: Optional[int]) -> None:
        """Points, curves, and the selected group's EC50 with its interval.

        Every curve that fitted is drawn, so a plate reads as one picture; the
        EC50 marker belongs to the selected row only, because twenty-four
        vertical lines and twenty-four shaded bands is not a figure. An
        unbounded EC50 gets an arrow at the edge of the tested range instead
        of a line inside it — the drawing has to make the same distinction the
        numbers do.
        """
        palette = active_palette()
        self._figure.clear()
        self._figure.patch.set_alpha(0.0)
        axes = self._figure.add_subplot(111)
        _page_surface_axes(axes, palette)
        axes.grid(True, color=palette["border_soft"], linewidth=0.6, alpha=0.5)
        axes.set_axisbelow(True)
        for side in ("top", "right"):
            axes.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axes.spines[side].set_color(palette["border"])
            axes.spines[side].set_linewidth(0.8)
        axes.tick_params(colors=palette["fg_muted"], labelsize=8, length=3)

        if self._set is None or not self._set.results():
            axes.set_xlabel(tr("concentration"), color=palette["fg_muted"],
                            fontsize=9)
            axes.set_ylabel(tr("response"), color=palette["fg_muted"],
                            fontsize=9)
            self.canvas.draw_idle()
            return

        colours = categorical_colours()
        spec = self._set.spec
        for index, fit in enumerate(self._set.fits):
            result = fit.result
            if result is None:
                continue
            colour = colours[index % len(colours)]
            focused = (selected is None or index == selected)
            axes.plot(result.dose, result.response, "o", color=colour,
                      markersize=4, alpha=0.9 if focused else 0.25,
                      label=(fit.group or tr("all rows")))
            x, y = result.curve()
            axes.plot(x, y, "-", color=colour, linewidth=1.8 if focused else 0.9,
                      alpha=1.0 if focused else 0.3)
        axes.set_xscale("log")

        if selected is not None and 0 <= selected < len(self._set.fits):
            chosen = self._set.fits[selected].result
            if chosen is not None:
                limits = axes.get_xlim()
                self._draw_ec50(axes, chosen, colours[selected % len(colours)],
                                palette)
                axes.set_xlim(limits)

        unit = f" ({spec.unit})" if spec.unit else ""
        axes.set_xlabel(f"{spec.concentration or tr('concentration')}{unit}",
                        color=palette["fg_muted"], fontsize=9)
        axes.set_ylabel(spec.response or tr("response"),
                        color=palette["fg_muted"], fontsize=9)
        if len(self._set.results()) > 1:
            legend = axes.legend(fontsize=7, frameon=False, loc="best")
            for text in legend.get_texts():
                text.set_color(palette["fg_muted"])
        self._figure.tight_layout()
        self.canvas.draw_idle()

    def _draw_ec50(self, axes, result: DoseResponseResult, colour: str,
                   palette) -> None:
        """The EC50 marker: a line and a band, or an arrow and a ``>``."""
        if result.ec50_bounded and result.ec50 is not None:
            if result.ec50_low is not None and result.ec50_high is not None:
                axes.axvspan(result.ec50_low, result.ec50_high, color=colour,
                             alpha=0.15, linewidth=0)
            axes.axvline(result.ec50, color=colour, linestyle="--",
                         linewidth=1.2)
            axes.annotate(f"EC50 {result.ec50:.3g}",
                          xy=(result.ec50, 0.02),
                          xycoords=("data", "axes fraction"),
                          color=palette["fg"], fontsize=8,
                          ha="left", va="bottom",
                          xytext=(4, 0), textcoords="offset points")
            return
        edge = (result.dose_max if result.bound_direction != "below"
                else result.dose_min)
        symbol = ">" if result.bound_direction != "below" else "<"
        axes.axvline(edge, color=palette["warning"], linestyle=":",
                     linewidth=1.4)
        axes.annotate(f"EC50 {symbol} {edge:.3g}", xy=(edge, 0.02),
                      xycoords=("data", "axes fraction"),
                      color=palette["warning"], fontsize=8,
                      ha="right" if symbol == ">" else "left", va="bottom",
                      xytext=(-4 if symbol == ">" else 4, 0),
                      textcoords="offset points")

    def result_set(self) -> Optional[DoseResponseSet]:
        """The last fit, or ``None``. What a test and an exporter both read."""
        return self._set

    def active_jobs(self) -> int:
        """Worker threads still winding down."""
        return self._jobs.active_jobs()

    def is_busy(self) -> bool:
        """Whether a read or a fit is in flight."""
        return self._jobs.is_busy()

    def closeEvent(self, event):  # noqa: N802 - Qt name
        """Stop background work and unlink before going away.

        :param event: the Qt close event.
        """
        self._jobs.shutdown()
        cancel = getattr(self.canvas, "cancel_pending_draw", None)
        if cancel is not None:
            cancel()
        super().closeEvent(event)


def make_dose_response_screen(app_key: Optional[str] = None) -> QWidget:
    """Factory handed to :func:`spacr.qt.app.register_app`."""
    return DoseResponseScreen()


_ROW = declared_app(APP_KEY)
APP_NAME = _ROW.name
APP_DESCRIPTION = _ROW.desc
APP_INTRO = _ROW.intro
APP_CLI_NOTE = _ROW.cli_note
APP_NAME_TRANSLATIONS = _ROW.translations


def register() -> bool:
    """Put Dose–Response in the app registry. Idempotent.

    Called from :data:`spacr.qt.SELF_REGISTERING_MODULES`, which
    :func:`spacr.qt.run` runs after ``spacr.qt.app`` is fully executed and
    before ``MainWindow.__init__`` reads the registry — the position the
    docstring there explains. Not called at import, so importing this module
    to reach :class:`DoseResponseScreen` from a test or a notebook does not
    mutate process-wide state.

    The row itself -- the key, the name, the blurb, the section, the "no
    headless run" sentence, the API doc link and the nine translations of the
    display name -- is declared in :mod:`spacr.qt.app_catalog`.
    :func:`spacr.qt.app.register_app` distributes those into the four tables
    each used to need a hand-edit in, and this function's whole job is to name
    which row. That is what lets the app be registered without importing this
    module at all: the launch reads the table, and the screen is imported when
    somebody opens it.

    ``SECTION_DESIGN``, which is not the obvious answer and is the right one.
    Design is "everything that happens before the microscope: power, sample
    size, plate layout, controls and replicates", and it already holds Power /
    Design. A dose–response series is the *other* pre-experiment calculation a
    screening lab runs: nobody fits an EC50 to admire it, they fit it to pick
    the concentration the actual screen will use, exactly as they run a power
    calculation to pick n. The output of this screen is an input to the next
    experiment, which is what the section means.

    The alternative reading — Explore, "ask the numbers a question you did not
    plan for" — is the weaker one *because* a concentration series is planned:
    the doses were chosen in advance and the curve is the thing the experiment
    was for. The cap made the choice concrete: ``MAX_APPS_PER_SECTION`` is 13,
    Explore stood at 12 before this batch, and Outliers — which really is an
    open question asked of a finished table — is the one that belongs there.

    :returns: ``True`` if this call is what registered it.
    """
    return register_declared(__name__) is not None

"""Everything a finished regression produced, in one place and fast.

The module used to answer a run with a stack of matplotlib pictures, the last
of which -- the volcano -- cost ~115 ms per redraw and made the whole window
lag. Worse, the numbers behind it were only in a CSV, so "is EAF1 a hit" meant
leaving the application.

This panel is what a finished run opens into:

    Volcano        every dot clickable, drawn by Qt, 3.6 ms
    Table          the coefficients, sortable, filterable, wired to the plots
    p-values       the histogram that says whether a correction means anything
    Q-Q            calibration, with the inflation figure
    Controls       the assay window
    Guide support  which genes rest on one guide, as a plot and a table
    Gene           the tile for whatever was last clicked

EVERY PLOT WHOSE MARKS ARE COEFFICIENTS IS CLICKABLE, not only the volcano --
instruction 124 F. Clicking a point selects its row; selecting a row marks
that point on every plot that drew it. They are views of one table, which is
the thing that was missing.

Joined on the KEY, never on a position. The Q-Q is sorted by p, the control
panel is split into groups and the agreement plot has one point per gene from
a frame with one row per guide, so a mark's position is not its row in any of
them. Measured on the real screen: the second point on the Q-Q is the term
``gene_fraction:gene[244480]`` (p = 2.9e-13, the strongest hit in the screen)
while its drawing position names ``fraction:grna[000000_10]``, a control guide
with p = 0.81. That is what a positional join would have selected, and nothing
about it would have looked wrong.
"""

from __future__ import annotations

import os
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QPushButton, QSplitter, QTabWidget,
    QVBoxLayout, QWidget,
)

#: Files a regression writes, best first. ``results.csv`` is the full
#: coefficient table; the gene/grna splits are views of it.
RESULT_FILENAMES = ("results.csv", "results_grna.csv", "results_gene.csv")

#: How far below the folder the user pointed at a results table is looked for.
#: ``src`` is the screen folder and a table lands at
#: ``results/<plate>/<regression_type>/list/results.csv``, which is four; the
#: rest is headroom. A bound is the difference between "no results here" and
#: walking a home directory.
MAX_SEARCH_DEPTH = 6

#: Stop after this many candidates. Only the newest is shown and the rest are
#: a count in the status line, so there is nothing to gain from finding a
#: thousand of them.
MAX_CANDIDATES = 200

#: Every spelling of the p-value column this panel can be handed, best first.
#:
#: spaCR's own ``process_model_coefficients`` writes ``p_value``. A table that
#: came through a statsmodels summary carries ``P>|t|`` for OLS and ``P>|z|``
#: for a GLM or a Poisson fit -- the ``t value``/``z value`` beside it is the
#: STATISTIC, not the p -- and an R-shaped export spells it ``Pr(>|t|)``.
#: Matching is done on a normalised form, so case and separators do not count.
P_VALUE_COLUMNS = (
    "p_value", "p", "pvalue", "p-value", "p_val", "pval",
    "P>|t|", "P>|z|", "Pr(>|t|)", "Pr(>|z|)",
)

#: The ranking the penalised backends report instead. ``lasso`` and
#: ``elasticnet`` have no frequentist p-value at all -- see
#: :data:`spacr.hits.NO_P_VALUE_TYPES` -- so a bootstrap selection frequency
#: is what orders their coefficients.
SELECTION_COLUMNS = ("selection_frequency", "selection_freq",
                     "proportion_selected")

#: Test statistics. Present without a p-value they are worth naming in the
#: status line, because "z value" is the thing a user will point at and ask
#: why the histogram is empty.
STATISTIC_COLUMNS = ("z value", "t value", "z_value", "t_value", "z", "t",
                     "statistic", "wald")


def _normalise(name) -> str:
    """A column name reduced to what actually distinguishes it."""
    return "".join(str(name).lower().split()).replace("_", "").replace("-", "")


def _match_column(frame, wanted) -> Optional[str]:
    """The first column of ``frame`` matching one of ``wanted``, in order."""
    if frame is None:
        return None
    columns = list(getattr(frame, "columns", ()))
    for name in wanted:
        if name in columns:
            return name
    lookup = {}
    for column in columns:
        lookup.setdefault(_normalise(column), column)
    for name in wanted:
        found = lookup.get(_normalise(name))
        if found is not None:
            return found
    return None


def find_results_tables(path, *, max_depth: int = MAX_SEARCH_DEPTH,
                        limit: int = MAX_CANDIDATES) -> list:
    """Every results CSV under ``path``, NEWEST RUN FIRST.

    The first table in a sorted walk is not this run's results. ``glm`` sorts
    before ``ols`` and ``2025`` before ``2026``, so a folder holding more than
    one run answered with whichever one happened to win the alphabet -- which
    is how a user watching a run finish reads last month's screen and never
    finds out.

    ORDERED BY RUN, NOT BY FILE. ``perform_regression`` writes ``results.csv``,
    then ``results_gene.csv``, then ``results_grna.csv``, milliseconds apart
    into one folder -- so ranking individual files by modification time hands
    back the guide split, which on the real screen is 823 rows of a 1,213-row
    fit. The run folder's newest table dates the RUN; inside it,
    :data:`RESULT_FILENAMES` order decides, and the full coefficient table
    comes first because the gene and guide files are views of it.
    """
    if not path:
        return []
    try:
        root = os.path.abspath(os.path.expanduser(os.fspath(path)))
    except TypeError:
        return []
    if os.path.isfile(root):
        return [root] if root.lower().endswith(".csv") else []
    if not os.path.isdir(root):
        return []
    runs = []
    total = 0
    for folder, subdirs, files in os.walk(root, followlinks=False):
        depth = folder[len(root):].count(os.sep)
        if depth >= max_depth:
            subdirs[:] = []
        subdirs.sort()
        present = set(files)
        found = []
        newest = None
        for order, name in enumerate(RESULT_FILENAMES):
            if name not in present:
                continue
            candidate = os.path.join(folder, name)
            try:
                stamp = os.path.getmtime(candidate)
            except OSError:            # vanished between listing and stat
                continue
            found.append((order, candidate))
            newest = stamp if newest is None else max(newest, stamp)
        if found:
            found.sort()
            runs.append((-newest, folder, [c for _, c in found]))
            total += len(found)
        if total >= limit:
            break
    runs.sort()
    return [candidate for _, _, group in runs for candidate in group]


def find_results_table(path) -> Optional[str]:
    """The results CSV for ``path``: a file, a run folder, or a parent of one.

    Those are the three things a user actually has to hand when they want to
    look at a regression again, so all three are accepted. Where a parent
    holds several, the most recently modified wins -- see
    :func:`find_results_tables`.
    """
    tables = find_results_tables(path)
    return tables[0] if tables else None


#: What the run says when a fit is not identifiable. Repeated on the Summary
#: tab because statsmodels prints a full table of standard errors and P
#: values regardless, and it looks exactly like a summary of a well-posed
#: fit -- which a reader may paste into a methods section from one click
#: away.
UNIDENTIFIABLE_WARNING = (
    "THIS FIT IS NOT IDENTIFIABLE: {wells} analysed observations are being "
    "used to estimate {params} parameters.\n"
    "Every standard error and P value below is one arbitrary solution out of "
    "infinitely many; refitting the same data can give different numbers and "
    "neither set is wrong.\n"
    "Set inference='nonparametric' to test each guide as a plate-blocked "
    "marginal association, which stays valid at any width.\n")


def summary_text(model, regression_type=None) -> str:
    """The statsmodels summary for ``model``, or why there is none.

    VERBATIM, not rebuilt. The point of asking for the statsmodels summary is
    to get the statsmodels summary; a re-implementation would differ from
    every textbook and every other tool a reader compares it against.

    :returns: the summary text, always a string -- a backend without one
        comes back as a sentence naming the backend rather than as an
        exception or an empty tab, both of which read as a bug.
    """
    if model is None:
        return ("No summary: this panel was opened from a results table on "
                "disk rather than from a run, so the fitted model is not "
                "here. Re-run to see it.")

    summary = getattr(model, "summary", None)
    if not callable(summary):
        named = f" ({regression_type})" if regression_type else ""
        return (f"No summary: this backend{named} is not a statsmodels fit, "
                f"so it has none. The sklearn-backed types (lasso, ridge, "
                f"elasticnet, hinge) report coefficients without standard "
                f"errors, which is why they are ranked by bootstrap "
                f"selection frequency instead -- see the Coefficients tab.")

    try:
        text = str(summary())
    except Exception as error:                                   # noqa: BLE001
        return (f"No summary: statsmodels could not render one for this fit "
                f"({type(error).__name__}: {error}).")

    warning = _identifiability_warning(model)
    return f"{warning}\n{text}" if warning else text


def _identifiability_warning(model) -> str:
    """The run's own not-identifiable warning, or "".

    Read off the model rather than off the settings, so the tab cannot
    disagree with the table it is printed above.
    """
    try:
        observations = int(getattr(model, "nobs", 0))
        params = len(getattr(model, "params", ()))
    except Exception:                                            # noqa: BLE001
        return ""
    if observations and params and params >= observations:
        return UNIDENTIFIABLE_WARNING.format(wells=observations,
                                             params=params)
    return ""


def backend_of(path) -> Optional[str]:
    """The regression type a results path was written under, if it says.

    ``perform_regression`` writes to ``results/<screen>/<regression_type>/``,
    so the folder names the backend. This is the only way the panel can tell a
    lasso table from an OLS one: spacr.ml writes an OLS-style ``p_value``
    into BOTH, and on the penalised branch that number is computed ignoring
    the penalty and means nothing at all.
    """
    if not path:
        return None
    try:
        from ...hits import NO_P_VALUE_TYPES
    except Exception:                  # pragma: no cover - hits unavailable
        return None
    parts = {part.strip().lower()
             for part in str(path).replace("\\", "/").split("/")}
    for name in NO_P_VALUE_TYPES:
        if name in parts:
            return name
    return None


class RegressionResultsPanel(QWidget):
    """Volcano, table and diagnostics for one finished regression."""

    #: Emitted with the results CSV whenever a new one is loaded.
    loaded = Signal(str)

    #: Emitted with a settings dict when the user asks, from the plot, for
    #: the same screen through a different model. The panel does not start
    #: the run itself -- it has no worker, no console and no Stop button --
    #: so whichever screen owns those does.
    refit_requested = Signal(object)

    def __init__(self, parent=None, external_volcano: bool = False):
        """
        :param external_volcano: build the volcano but do not place it. The
            caller takes it and gives it the room it deserves -- it is the
            graph the maintainer asked to be interactive, and a thumbnail
            above its own table is not that. All the wiring stays here, so
            the key join, the redraw and the surviving selection work the
            same wherever the widget ends up.
        """
        super().__init__(parent)
        from .fast_plots import (ControlSeparation, GuideAgreementPlot,
                                 InfluencePlot, PValueHistogram, QQPlot,
                                 ResidualPlot, ResultsTable,
                                 ScaleLocationPlot, VolcanoPlot)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        header = QHBoxLayout()
        # A WAY IN THAT DOES NOT REQUIRE STARTING A RUN. The panel used to be
        # filled from exactly one place -- successful run completion -- so a
        # user whose results were already on disk, or whose run finished while
        # the settings pointed somewhere else, had no way to open them at all
        # and no reason given for the empty table.
        self._load_button = QPushButton("Load results…")
        self._load_button.setToolTip(
            "Choose a regression results folder, or a parent of one. The most "
            "recently written results table under it is loaded.")
        self._load_button.clicked.connect(self.browse_for_results)
        header.addWidget(self._load_button)
        self._source = QLabel("No regression loaded.")
        self._source.setWordWrap(True)
        header.addWidget(self._source, 1)
        header.addWidget(QLabel("colour by"))
        self._colour_by = QComboBox()
        self._colour_by.setMinimumWidth(140)
        self._colour_by.currentIndexChanged.connect(self._redraw_volcano)
        header.addWidget(self._colour_by)
        layout.addLayout(header)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)

        # Volcano and table share a splitter: the two views of one table
        # belong beside each other, and the divider is the user's to move.
        self.volcano = VolcanoPlot()
        self.table = ResultsTable()
        self.external_volcano = bool(external_volcano)
        self.table.setMinimumHeight(150)
        if self.external_volcano:
            # The table is the panel; the graph is the caller's to place.
            self.tabs.addTab(self.table, "Coefficients")
        else:
            split = QSplitter(Qt.Vertical)
            split.setChildrenCollapsible(False)
            split.addWidget(self.volcano)
            split.addWidget(self.table)
            split.setStretchFactor(0, 3)
            split.setStretchFactor(1, 2)
            # Floors, not preferences. Without them the panel's share of the
            # window is divided by the widgets' own size hints and BOTH end up
            # too short to read -- a volcano with no room for its axes and a
            # table showing its header and one row, which is what this looked
            # like on the real screen before the numbers were put in.
            self.volcano.setMinimumHeight(240)
            split.setSizes([340, 220])
            self.tabs.addTab(split, "Volcano")

        self.p_values = PValueHistogram()
        self.qq = QQPlot()
        self.controls = ControlSeparation()
        self.tabs.addTab(self.p_values, "p-values")
        self.tabs.addTab(self.qq, "Q-Q")
        self.tabs.addTab(self.controls, "Controls")

        # THE STATSMODELS SUMMARY. Monospace, read-only and SELECTABLE: the
        # reason to want it is usually to paste a number into a methods
        # section, and a summary you cannot select is a summary you retype.
        from PySide6.QtGui import QFontDatabase
        from PySide6.QtWidgets import QPlainTextEdit

        self._summary = QPlainTextEdit()
        self._summary.setReadOnly(True)
        self._summary.setLineWrapMode(QPlainTextEdit.NoWrap)
        self._summary.setFont(
            QFontDatabase.systemFont(QFontDatabase.FixedFont))
        self._summary.setPlainText(
            "Run a regression to see its summary.")
        # ADDED AFTER THE DIAGNOSTICS, not before them. Q-Q, Controls,
        # Residuals, Scale-location and Influence are one group that reads in
        # order, and a test asserts they sit together; dropping the Summary
        # into the middle of it split the group.

        # THE RESIDUAL DIAGNOSTICS, LIVE. "in the tabs Q-Q and Controls, there
        # should be Tabs like residuals showing the residuals and regression
        # controll graphs like that."
        #
        # Every tab so far is drawn from the COEFFICIENT table, which is one
        # row per guide and says nothing about how well the model fitted the
        # WELLS it was given. These three are the well-level half, and they
        # are the three the QC report leads with: is the mean right, is the
        # variance flat, is the answer resting on one well. They come from
        # `spacr.regression_qc` -- the same arrays the saved report draws --
        # rather than being recomputed here, because a live panel that named
        # different influential wells than the PDF beside it would be worse
        # than no live panel.
        #
        # THEY ARE ALWAYS TABS, even before there is a model to fill them,
        # and each one SAYS why it is empty. A diagnostic that appears only
        # once it happens to be computable is one nobody knows to look for.
        self.residuals = ResidualPlot()
        self.scale_location = ScaleLocationPlot()
        self.influence = InfluencePlot()
        self.tabs.addTab(self.residuals, "Residuals")
        self.tabs.addTab(self.scale_location, "Scale-location")
        self.tabs.addTab(self.influence, "Influence")
        # The statsmodels summary closes the diagnostic group: it is the
        # model-level readout the panels above it are pictures of.
        self.tabs.addTab(self._summary, "Summary")
        self.tabs.setTabToolTip(
            self.tabs.indexOf(self.residuals),
            "Residual against fitted, one point per well. A horizontal band "
            "is a well-specified mean; a curve or a funnel is not.")
        self.tabs.setTabToolTip(
            self.tabs.indexOf(self.scale_location),
            "The spread of the residuals with the sign taken out. A rising "
            "trend means the standard errors, and so every p-value on the "
            "volcano, depend on the fitted value.")
        self.tabs.setTabToolTip(
            self.tabs.indexOf(self.influence),
            "Leverage against standardised residual. Which wells are moving "
            "the coefficients on their own.")

        # GUIDE SUPPORT. The one thing the volcano structurally cannot show:
        # a gene carried by a single surviving guide and a gene whose guides
        # agree are the same dot, ranked by the same number, and only one of
        # them is independent evidence.
        #
        # Plot above table, the same arrangement as the volcano tab and for
        # the same reason: the picture says which genes rest on one guide at a
        # glance, and the numbers behind any one of them are a click away.
        self.agreement = GuideAgreementPlot()
        self.support = ResultsTable()
        agreement_split = QSplitter(Qt.Vertical)
        agreement_split.setChildrenCollapsible(False)
        agreement_split.addWidget(self.agreement)
        agreement_split.addWidget(self.support)
        agreement_split.setStretchFactor(0, 3)
        agreement_split.setStretchFactor(1, 2)
        self.agreement.setMinimumHeight(240)
        self.support.setMinimumHeight(150)
        agreement_split.setSizes([340, 220])
        self.tabs.addTab(agreement_split, "Guide support")

        # THE GENE TILE. Instruction 121. The volcano answers "which guides
        # moved" and structurally cannot answer "what IS 411710", which is the
        # question the user has the instant they click one. The frame is
        # reached through a callable rather than stored, so a newly loaded
        # regression is never answered from the previous one.
        from .gene_tile import GeneTilePanel
        self.gene = GeneTilePanel(frame_provider=lambda: self._frame)
        if not self.external_volcano:
            self.tabs.addTab(self.gene, "Gene")
        # When the volcano is external the tile goes WITH IT, not behind a
        # tab: "when a gene is clicked a tile should appear with all the
        # information on that gene" -- appear, beside the point that was
        # clicked. A tile the user has to go and find is a tile they will not
        # look at. The caller places it, exactly as it places the volcano.

        # THE TWO DIRECTIONS OF THE SAME LINK, JOINED ON THE KEY.
        #
        # Not on a position. The table is sorted by whatever column the user
        # clicked last and filtered by whatever is in the search box; the
        # volcano is drawn in input order and, since it stopped plotting the
        # nuisance terms, does not even hold the same number of rows. Two
        # frames in two orders joined by index highlight the wrong guide --
        # silently, and in the one direction nobody questions, because a point
        # did light up.
        #
        # `feature` is the key: 1,213 rows and 1,213 distinct values on the
        # real screen. It is checked, not assumed -- see _key_column.
        #
        # EVERY PLOT WHOSE POINTS ARE COEFFICIENTS, not just the volcano.
        # Instruction 124 F: "id like to be able to presson the datapoints of
        # all graphs where data is represented as genes and grnas, e.g. like
        # the Q-Q plots". A Q-Q point IS a coefficient; so is a dot in the
        # control panel and a gene in the agreement plot. They all reach the
        # table by the same one-line route, which is why they cannot disagree
        # about what a click means.
        for plot in self._keyed_plots():
            plot.key_selected.connect(self.table.select_key)
        # A BAR IS NOT A POINT. The histogram is the one mark here that stands
        # for many rows, so it narrows the table to them rather than pretending
        # to pick one -- see PValueHistogram.select_bin. When a bar happens to
        # hold exactly ONE coefficient there is nothing to guess between, so it
        # selects it like any other point and takes the same route as the rest.
        self.p_values.key_selected.connect(self.table.select_key)
        self.p_values.keys_selected.connect(self._show_keys)
        self.table.key_selected.connect(self._select_key)
        # ON THE TABLE, NOT THE VOLCANO, and one connection rather than two:
        # table.key_selected is the funnel BOTH directions already pass
        # through -- volcano.key_selected -> table.select_key -> selection
        # change -> re-emit. Connecting the volcano as well would build the
        # tile twice for every click.
        self.table.key_selected.connect(self.gene.show_feature)

        self._frame = None
        self._path = None
        self._selected_key = None
        self._status = "No regression loaded."
        self._ranking = (None, None)
        #: The settings that produced the table on screen, when they are
        #: known. BORN HERE rather than on first use: an attribute a menu
        #: handler reads, created only by a code path that may not have run,
        #: is the `_significance` crash that took the panel down at launch.
        self._run_settings = None
        #: ``(kind, name)`` -- what effects are measured FROM. Born here for
        #: the same reason as everything else in this block: `_redraw_volcano`
        #: reads it and is reachable from a signal connected above.
        self._baseline = (None, None)
        #: The one TAGM/LOPIT compartment picked out against grey, or None.
        self._compartment = None
        #: How the effect-size cut is measured, and how wide.
        self._threshold_method = "mad"
        self._threshold_multiplier = 3.0
        #: Why a colour-by option is present but useless, or "".
        self._colour_by_note = ""
        #: None, "gene" or "grna" -- which rows the volcano draws.
        self._level = None
        #: The fitted model behind the table, when a run in this session
        #: produced it. BORN HERE for the same reason as everything else in
        #: this block.
        self._model = None

        # The diagnostics start out saying what they are waiting for. An
        # empty plot with no sentence is indistinguishable from a broken one.
        self.clear_diagnostics()

        # RE-FITTING IS OFFERED FROM THE PLOT, under its own heading, because
        # that is where it was asked for: "right click on the regression plot
        # and choose a different regression". It is separated from the
        # restyling entries above it for the reason `offer_refit` gives.
        self.volcano.offer_refit(self.ask_refit)
        self._offer_levels()
        self._offer_baselines()
        self._offer_compartments()

    # -------------------------------------------------------------- re-fitting

    def set_run_settings(self, settings) -> None:
        """Remember the settings that produced the table now on screen.

        Called by the screen when a run finishes, because the run's own
        settings are better than anything read back off disk: the saved copy
        under ``settings/`` is overwritten by every later run of the same
        screen, so on a second run it describes the wrong one.
        """
        self._run_settings = dict(settings) if settings else None

    def ask_refit(self) -> bool:
        """Offer another model for the same data, and ask for the run.

        :returns: True if a re-fit was asked for.

        The panel goes no further than emitting. It has no worker, no
        console and no Stop button, and a widget that started a background
        fit with none of those would be a run the user cannot watch or stop.
        """
        from ...refit import refit_settings, settings_of_run

        base = self._run_settings or settings_of_run(self._path)
        try:
            refit_settings(base or {})
        except ValueError as error:
            # No settings, or no count data in them. SAID ON THE PANEL and
            # BEFORE the dialog opens: a form whose only content is a
            # disabled button and an error is worse than a sentence, and the
            # user right-clicked a graph -- a traceback is not an answer to
            # that either.
            self.say(str(error))
            return False

        from .refit_dialog import ask_refit as ask

        answer = ask(base, self)
        if answer is None:
            return False
        settings, notes = answer
        if notes:
            self.say("Re-fitting: " + "; ".join(notes))
        self.refit_requested.emit(settings)
        return True

    # ------------------------------------------------------------ diagnostics

    #: What the diagnostic tabs say before a run in this session has produced
    #: a model. Not "no data": a user looking at a table they loaded off disk
    #: needs to know that these three tabs are empty for a REASON they can act
    #: on, and what the action is.
    NO_MODEL_MESSAGE = (
        "Residual diagnostics are computed from the fitted model, and only a "
        "run in this session hands one over -- a results table read from disk "
        "is one row per guide and carries nothing about the wells. Run the "
        "regression here, or re-fit from the volcano's right-click menu, and "
        "these fill in. The same panels are already on disk beside the "
        "results, under regression_qc/.")

    def diagnostic_plots(self) -> tuple:
        """The three well-level tabs, in the order they are shown.

        Public because it is how a caller asks the panel to restyle, export or
        interrogate them WITHOUT reaching past the panel into its widgets --
        which is how one screen ended up depending on another's private
        surface.
        """
        return (self.residuals, self.scale_location, self.influence)

    def clear_diagnostics(self, reason: str = "") -> None:
        """Empty the three well-level tabs and SAY why they are empty.

        :param reason: the specific reason, when there is one. The default is
            :data:`NO_MODEL_MESSAGE` -- "no run in this session has fitted
            anything yet", which is the ordinary case and still an answer.
        """
        self._model = None
        for plot in self.diagnostic_plots():
            plot._reset_scene()
            plot.set_status(reason or self.NO_MODEL_MESSAGE)

    def set_summary(self, model, regression_type=None) -> bool:
        """Fill the Summary tab from the fitted model.

        :returns: True when a summary was rendered.

        Rendered from `model.summary()` verbatim rather than rebuilt: the
        point of asking for the statsmodels summary is to get the statsmodels
        summary, and a re-implementation would differ from every textbook and
        every other tool the reader compares it against.
        """
        text = summary_text(model, regression_type)
        self._summary.setPlainText(text)
        return not text.startswith("No summary")

    def set_diagnostics(self, model, regression_type=None) -> bool:
        """Fill the residual tabs from the model a run just fitted.

        :param model: the fitted model, straight off ``perform_regression``'s
            return payload. ``None`` clears the tabs and says why.
        :returns: True when the tabs were filled.

        EVERY NUMBER COMES FROM :mod:`spacr.regression_qc`. The residuals, the
        standardisation, the leverage and Cook's distance are the arrays that
        module already computes for the report it writes to disk -- so the
        live tab and the saved PDF cannot disagree about which well is
        influential, which they would within a week if this panel did its own
        arithmetic.

        NEVER RAISES INTO THE GUI. A model class that cannot be diagnosed --
        the penalised backends keep no design matrix -- puts its reason on the
        three tabs instead, because "this fit cannot answer that" and "this
        tab is broken" must not look the same.
        """
        if model is None:
            self.clear_diagnostics()
            return False
        try:
            from ...regression_qc import (PanelUnavailable, context_from_model,
                                          cooks_distance)
        except Exception as error:                               # noqa: BLE001
            self.clear_diagnostics(
                f"Could not load the diagnostics module: {error}")
            return False
        try:
            ctx = context_from_model(model, coef_df=self._frame,
                                     regression_type=regression_type)
        except PanelUnavailable as error:
            self.clear_diagnostics(str(error))
            return False
        except Exception as error:                               # noqa: BLE001
            self.clear_diagnostics(
                f"The diagnostics could not be computed for this fit "
                f"({type(error).__name__}: {error}).")
            return False

        self._model = model
        labels = list(ctx.labels) if ctx.labels is not None else []
        reason = (ctx.standardisation.reason
                  if ctx.standardisation is not None else "")
        self.residuals.set_residuals(ctx.fitted, ctx.resid, labels=labels)
        self.scale_location.set_scale_location(
            ctx.fitted, ctx.std_resid, labels=labels, reason=reason or "")
        self.influence.set_influence(
            ctx.leverage, ctx.std_resid,
            cooks_distance(ctx.std_resid, ctx.leverage, ctx.p),
            labels=labels, n_params=ctx.p, reason=reason or "")
        return True

    def _keyed_plots(self) -> tuple:
        """Every plot whose marks are individual coefficients or genes.

        The volcano, the Q-Q, the control panel and the guide-agreement plot.
        The p-value histogram is deliberately NOT here: its marks are bins of
        many rows, so it cannot select one and does not pretend to. The
        residual plot is not here either -- its points are wells, not
        coefficients, and there is no key for a well.
        """
        return (self.volcano, self.qq, self.controls, self.agreement)

    def _show_keys(self, keys) -> None:
        """A set of coefficients was chosen on a plot: narrow the table."""
        self.table.show_keys(list(keys))

    # ------------------------------------------------------------------ load

    def results_frame(self):
        """The coefficient table on screen, or ``None``.

        Public because two callers outside this panel need to know whether
        there is anything to draw -- the publication sheet and the figure grid
        -- and both were reaching in for ``_frame`` with ``getattr``. A
        ``getattr`` for a private attribute is not encapsulation, it is the
        same coupling with the failure mode hidden: rename the attribute and
        both callers silently decide there are no results.
        """
        return self._frame

    def status_text(self) -> str:
        """Whatever the panel last had to say, success or failure.

        The header label is the only place the user learns why a table is
        empty, so it is worth reading back in a test rather than reaching into
        a private widget.
        """
        return self._status

    def say(self, text: str, detail: str = "") -> None:
        """Put a sentence where the user will read it.

        Public because the panel is not the only thing that can fail to fill
        it: the caller that decides WHICH folder to hand over can come up with
        nothing at all, and that has to reach the same header rather than
        being logged at debug level and dropped.

        :param detail: the tooltip; the long form, when there is one.
        """
        self._status = text
        self._source.setText(text)
        self._source.setToolTip(detail or text)

    def browse_for_results(self) -> bool:
        """Ask for a results folder and load it. False if nothing was chosen.

        Companion to :meth:`load`: the panel is otherwise only ever filled by
        a run finishing, which is no use to a user who has results and no run.
        """
        from PySide6.QtWidgets import QFileDialog

        start = ""
        if self._path:
            start = os.path.dirname(str(self._path))
        folder = QFileDialog.getExistingDirectory(
            self, "Choose a regression results folder", start)
        if not folder:
            return False
        return self.load(folder)

    def load(self, path) -> bool:
        """Load a results CSV, a run folder, or a parent of one.

        EVERY WAY THIS FAILS SAYS SO. It used to return False five different
        ways and leave the user looking at a table with columns and no rows,
        which is indistinguishable from a run that produced nothing.
        """
        import pandas as pd

        if not path:
            self.say("Nothing was handed to the results panel, so there is "
                      "no folder to search. Use “Load results…”.")
            return False
        searched = os.path.abspath(os.path.expanduser(os.fspath(path)))
        if not os.path.exists(searched):
            self.say(f"{searched} does not exist, so there is nothing to "
                      f"load from it.")
            return False

        tables = find_results_tables(searched)
        if not tables:
            self.say(
                f"Searched {searched} and found none of "
                f"{', '.join(RESULT_FILENAMES)} in it or in any folder up to "
                f"{MAX_SEARCH_DEPTH} deep.")
            return False

        found = tables[0]
        try:
            frame = pd.read_csv(found)
        except Exception as error:  # noqa: BLE001 - report, do not raise
            self.say(f"Could not read {found}: {error}")
            return False
        if not self.set_frame(frame, source=found):
            # set_frame said why -- an empty table is not the same failure as
            # a missing one -- but the folder it came from is worth adding.
            self.say(f"{self._status} ({found}, found under {searched})")
            return False
        runs = {os.path.dirname(table) for table in tables}
        if len(runs) > 1:
            self.say(
                f"{self._status} — newest of {len(runs)} runs under "
                f"{searched}",
                detail="\n".join(tables[:20]))
        elif len(tables) > 1:
            self.say(f"{self._status} — the newest run under {searched} "
                      f"wrote {len(tables)} tables; this is the full one.",
                      detail="\n".join(tables[:20]))
        return True

    def set_frame(self, frame, source: str = "") -> bool:
        """Show an already-loaded coefficient table."""
        import pandas as pd

        if frame is None or not len(frame):
            self.say("The results table is empty: it has columns but no "
                      "rows, so the fit produced no coefficients.")
            return False
        self._frame = frame
        self._path = source
        self._ranking = self._rank_by(frame, source)

        # A NEW TABLE IS A NEW FIT, so the old fit's residuals have to go. The
        # caller that HAS a model calls `set_diagnostics` immediately after
        # this; the caller that loaded a CSV has none, and leaving the last
        # run's residuals on the tabs would describe a fit the user is no
        # longer looking at -- with nothing on screen saying so.
        self.clear_diagnostics()

        # WHICH SETTINGS PRODUCED THIS TABLE. Read from beside the table, and
        # REPLACED rather than kept: a new table is a new experiment, and
        # carrying the last run's settings over would offer to re-fit a
        # screen the panel is no longer showing. A live run overrides this by
        # calling `set_run_settings` afterwards, which is better still --
        # the shared settings/ copy on disk is overwritten by every later run.
        from ...refit import settings_of_run

        try:
            self._run_settings = settings_of_run(source) if source else None
        except Exception:                                        # noqa: BLE001
            self._run_settings = None

        # Offer every column that could sensibly colour the points, without
        # guessing: a column with one value per point is not a category.
        self._colour_by.blockSignals(True)
        self._colour_by.clear()
        self._colour_by.addItem("nothing", None)
        skipped = []
        cap = max(40, len(frame) // 20)
        for name in frame.columns:
            if frame[name].dtype == object or str(frame[name].dtype) == "category":
                distinct = frame[name].nunique(dropna=True)
                if 1 < distinct <= cap:
                    self._colour_by.addItem(f"{name} ({distinct})", name)
                elif name in self.ALWAYS_OFFERED:
                    # OFFERED ANYWAY, AND THE COUNT SAYS WHY IT IS USELESS.
                    # A `condition` column with ONE value is not a boring
                    # column, it is a FINDING: it means the negative/positive
                    # control names matched no feature, so nothing got
                    # labelled. Dropping it silently hides that, and the
                    # maintainer reported exactly this as "the color by
                    # doesn't include condition".
                    self._colour_by.addItem(f"{name} ({distinct})", name)
                    skipped.append(
                        f"{name} has {distinct} distinct value"
                        f"{'' if distinct == 1 else 's'}"
                        + (" -- the control names matched no feature"
                           if distinct == 1 and name == "condition" else ""))
        # LOPIT IS NOT A COLUMN, so the walk above cannot see it. It is
        # joined from the bundled TAGM table, and only offered when this
        # screen actually has compartments in it.
        try:
            from ...localisation import present

            compartments = present(frame)
        except Exception:                                        # noqa: BLE001
            compartments = []
        if compartments:
            self._colour_by.addItem(
                f"LOPIT localisation ({len(compartments)})", self.LOPIT_KEY)

        # 'condition' is what a screen labels its controls with, so it is the
        # colouring a reader wants first.
        preferred = self._colour_by.findData("condition")
        self._colour_by.setCurrentIndex(preferred if preferred >= 0 else 0)
        self._colour_by.blockSignals(False)
        self._colour_by_note = "; ".join(skipped)

        # A new table is a new experiment; carrying the old selection over
        # would ring a point that means something else now.
        #
        # EVERY PLOT, not just the volcano. Each one re-marks `_selected_key`
        # at the end of its own draw so a restyle does not lose the user's
        # place -- which means a plot whose selection is NOT cleared here
        # cheerfully re-rings the new run at the old key. Caught by exporting
        # the control panel after a reload and finding a ring still on it.
        self._selected_key = None
        for plot in self._keyed_plots():
            plot.clear_highlight()
        self.p_values.clear_highlight()

        # THE COMPARTMENT MENU IS BUILT FROM THE TABLE, so it has to be
        # rebuilt when the table changes. Built once in __init__ it is built
        # from no frame at all, which is an empty submenu that never appears
        # -- and a new screen would otherwise be offered the last one's
        # compartments.
        self._compartment = None
        self._level = None
        self._offer_levels()
        self._offer_thresholds()
        self._offer_compartments()

        self._redraw_volcano()
        kind, column = self._ranking
        # "significant only" cuts on `value <= alpha`. That is the right way
        # round for a p-value and exactly backwards for a selection frequency,
        # where the interesting rows are the HIGH ones -- so on a penalised
        # backend the checkbox is taken away rather than left to hide every
        # feature the bootstrap kept.
        self.table.configure(significance_filter=(kind == "p-value"))
        # The table prefers a CORRECTED column when there is one, and that is
        # the right cut; the detected column is only needed when the p-value
        # is spelled some way the table would not recognise.
        significance = None
        if kind == "p-value" and not any(
                name in frame.columns
                for name in ("q_value", "adjusted_p_value", "p_value")):
            significance = column
        self.table.set_frame(
            frame, key_column=self._key_column(frame),
            significance_column=significance)
        self._show_significance(frame, kind, column, source)
        if self._colour_by_note:
            # SAID, not swallowed. A colouring the user expected and cannot
            # find is a bug report; the same colouring listed with the reason
            # it is useless is an answer.
            self.say(f"{self._status} Colouring: {self._colour_by_note}.")
        self._draw_controls(frame)
        self._draw_guide_support(frame)
        self.loaded.emit(source or "")
        return True

    @staticmethod
    def _gene_terms(frame) -> dict:
        """``{gene id: the gene-level term that names it}``.

        The bridge between the two things a screen calls a gene. The support
        table is indexed by the bare id (``244480``); the coefficient table,
        the volcano and the gene tile all join on the design-matrix term
        (``gene_fraction:gene[244480]``). Without this the agreement plot
        would be a second key space that nothing else can resolve, and
        clicking a gene there would select nothing anywhere.
        """
        if frame is None or "feature" not in getattr(frame, "columns", ()):
            return {}
        try:
            from ...hits import gene_of
        except Exception:              # pragma: no cover - hits unavailable
            return {}
        terms = {}
        for feature in frame["feature"].astype(str):
            if not feature.startswith("gene_fraction:gene"):
                continue
            gene = gene_of(feature)
            if gene is not None:
                terms.setdefault(str(gene), feature)
        return terms

    def _draw_guide_support(self, frame) -> None:
        """Per-gene guide agreement, ordered by gene p."""
        try:
            from ...guide_concordance import guide_support
        except Exception:  # pragma: no cover - module unavailable
            return
        try:
            support = guide_support(frame)
        except Exception:  # pragma: no cover - odd table shape
            self.support.set_frame(None)
            self.agreement.set_support(None)
            return
        if support is None or not len(support):
            self.support.set_frame(None)
            self.agreement.set_support(None)
            return
        table = support.reset_index()
        # The term each gene is called in the coefficient table, put in the
        # table as a column so the support rows and the agreement points join
        # on the SAME key every other view here uses.
        terms = self._gene_terms(frame)
        table.insert(0, "feature",
                     [terms.get(str(gene)) for gene in table["gene"]])
        # A verdict column, because "n_guides=1" is a fact and "this hit rests
        # on one guide" is what the reader needs to take from it.
        def verdict(row):
            if row["single_guide"]:
                return "single guide -- gene p IS that guide's p"
            if row["concordance"] < 0.6:
                return "guides disagree in direction"
            if row["n_guides_significant"] == 0:
                return "agreement is the evidence"
            return "supported"
        table["verdict"] = table.apply(verdict, axis=1)
        # A gene with no gene-level term has no key. Offering `feature` as the
        # key column anyway would make every such row unselectable AND make
        # the column non-unique on None; the table checks, so say nothing and
        # let it fall back.
        usable = table["feature"].notna().all() and table["feature"].is_unique
        self.support.set_frame(table,
                               key_column="feature" if usable else None)
        self.agreement.set_support(table, keys=table["feature"])

    def ranking(self):
        """``(kind, column)`` for the table on screen.

        ``kind`` is ``"p-value"``, ``"selection-frequency"`` or ``None``.
        The panel is handed coefficient tables from every backend spaCR can
        fit and they do not agree on this: OLS reports ``P>|t|``, a GLM or a
        Poisson fit ``P>|z|``, spaCR's own writer ``p_value``, and the
        penalised backends report no p-value whatsoever.
        """
        return self._ranking

    @staticmethod
    def _rank_by(frame, source: str = ""):
        """What orders this table, and which column carries it.

        THE FOLDER OVERRULES THE COLUMNS on the penalised backends. spacr.ml
        writes an OLS-style ``p_value`` into a lasso ``results.csv`` -- it is
        computed as though there were no penalty, which is why
        :data:`spacr.hits.NO_P_VALUE_TYPES` exists -- so a panel that trusts
        the column draws a p-value histogram of a quantity that is not a
        p-value, and a q-value would be a correction applied to it.
        """
        selection = _match_column(frame, SELECTION_COLUMNS)
        backend = backend_of(source)
        if backend is not None:
            return ("selection-frequency", selection)
        if selection is not None:
            return ("selection-frequency", selection)
        p_column = _match_column(frame, P_VALUE_COLUMNS)
        if p_column is not None:
            return ("p-value", p_column)
        return (None, None)

    @staticmethod
    def _p_column(frame) -> Optional[str]:
        """The p-value column, however this backend spelled it."""
        return _match_column(frame, P_VALUE_COLUMNS)

    def _show_significance(self, frame, kind, column, source: str = "") -> None:
        """Fill the p-value and Q-Q tabs, or say why they are empty.

        An empty histogram with no caption reads as a broken panel. A
        penalised fit has no p-value to draw and never will, and a table
        carrying only a ``z value`` has a statistic rather than a test -- both
        are answers, and both used to be silence.
        """
        rows = len(frame)
        where = source or "this table"
        if kind == "p-value":
            # THE KEYS GO IN WITH THE VALUES, IN FRAME ORDER. Both are taken
            # positionally out of the same frame, so they stay aligned however
            # the plot reorders them afterwards -- and the Q-Q reorders them
            # completely, which is the whole point of handing them over rather
            # than letting the plot infer a row from a drawing position.
            keys = self._keys_for(frame)
            self.p_values.set_p_values(frame[column], keys=keys)
            self.qq.set_p_values(frame[column], keys=keys)
            self.say(f"{rows} coefficients from {where}, ranked by "
                      f"“{column}”.")
            return

        backend = backend_of(source)
        if kind == "selection-frequency":
            named = f"“{column}”" if column else "bootstrap selection frequency"
            which = f"{backend} " if backend else ""
            reason = (f"A {which}fit is ranked by bootstrap selection "
                      f"frequency ({named}) and has no p-value: it is a "
                      f"selection method, not a hypothesis test.")
        else:
            statistic = _match_column(frame, STATISTIC_COLUMNS)
            carries = (f"It carries “{statistic}”, which is a test statistic "
                       f"and not a p-value. " if statistic else "")
            reason = (f"No p-value column in this table. {carries}"
                      f"Looked for {', '.join(P_VALUE_COLUMNS[:4])} and the "
                      f"rest of the usual spellings.")
        self.p_values.set_p_values([])
        self.p_values.set_status(reason)
        self.qq.set_p_values([])
        self.qq.set_status(reason)
        self.say(f"{rows} coefficients from {where}. {reason}")

    @classmethod
    def _keys_for(cls, frame):
        """The identifier of every row, in frame order, or ``None``.

        ``None`` is a real answer: a table with no column that names a row
        uniquely has nothing to join on, and a plot given no keys stays
        readable and simply does not claim its points are clickable. That
        beats handing over ``gene``, which repeats across a gene's guides and
        would select an arbitrary one of them.
        """
        column = cls._key_column(frame)
        return None if column is None else frame[column]

    @staticmethod
    def _effect_column(frame) -> str:
        for name in ("coefficient", "coef", "effect", "estimate"):
            if name in frame.columns:
                return name
        return "coefficient"

    #: Columns offered for colouring EVEN WHEN the generic filter would drop
    #: them. A screen's own control annotation is worth seeing whatever it
    #: contains -- a single value means the control names matched nothing,
    #: which the user needs to know rather than to be shielded from.
    ALWAYS_OFFERED = ("condition",)

    #: Which rows the volcano shows. `None` is both, which is what a screen
    #: IS, so it is the default -- a panel that opened already filtered would
    #: be reporting a subset as the result.
    LEVELS = ((None, "genes and guides"), ("gene", "genes only"),
              ("grna", "guides only"))

    #: Sentinel for the derived LOPIT colouring, which is not a frame column.
    LOPIT_KEY = "\0lopit"

    #: What the volcano can measure its effects from, and what each says.
    BASELINES = (
        (None, "zero (no dose-response)"),
        ("controls", "the non-targeting controls"),
    )

    def _offer_baselines(self) -> None:
        """Put the baselines on the volcano's right-click menu."""
        chosen = self._baseline[0]
        self.volcano.offer_baselines([
            (label, (lambda k=kind: self.set_baseline(k)), kind == chosen)
            for kind, label in self.BASELINES])

    def level_counts(self) -> dict:
        """``{None: n, "gene": n, "grna": n}`` for the table on screen.

        Counted rather than assumed, and put in the MENU: "genes only" that
        silently draws 400 of 1,213 points is a filter a user applies without
        knowing what they gave up.
        """
        frame = self._frame
        if frame is None or "feature" not in getattr(frame, "columns", ()):
            return {None: 0, "gene": 0, "grna": 0}
        from ...hits import guide_of

        guides = frame["feature"].map(lambda f: guide_of(str(f)) is not None)
        return {None: int(len(frame)),
                "grna": int(guides.sum()),
                # A row that is not a guide and IS in the tested family is a
                # gene term. Counted as the complement rather than by a second
                # parse, so the two cannot disagree about a row.
                "gene": int((~guides).sum())}

    def _offer_levels(self) -> None:
        """Put genes / guides / both on the volcano's right-click menu."""
        counts = self.level_counts()
        self.volcano.offer_levels([
            (f"{label} ({counts.get(key, 0)})",
             (lambda k=key: self.set_level(k)), key == self._level)
            for key, label in self.LEVELS])

    def set_level(self, level) -> None:
        """Draw only genes, only guides, or both.

        A FILTER ON THE PLOT, not a mode of the run. The coefficient table
        already carries both -- `feature` is `gene_fraction:gene[...]` or
        `fraction:grna[...]` -- so this needs no re-fit and no second table,
        which is why it is better here than in the settings where it used to
        live.
        """
        self._level = level
        self._offer_levels()
        self._redraw_volcano()
        counts = self.level_counts()
        if level:
            self.say(f"Volcano: {counts.get(level, 0)} of {counts[None]} "
                     f"coefficients — "
                     f"{'genes' if level == 'gene' else 'guides'} only.")

    def _level_mask(self, frame):
        """Rows of ``frame`` at the chosen level, or None for all of them."""
        if not self._level or "feature" not in getattr(frame, "columns", ()):
            return None
        from ...hits import guide_of

        is_guide = frame["feature"].map(
            lambda f: guide_of(str(f)) is not None)
        return is_guide if self._level == "grna" else ~is_guide

    def _offer_thresholds(self) -> None:
        """Put the effect-size cut on the volcano's right-click menu.

        Asked for 2026-08-17: the multiplier and the mode, on the plot,
        because the settings-panel controls for them GREY OUT under
        `inference='nonparametric'` -- correctly, since the permutation path
        does not use a control-spread cut -- and the maintainer could not
        find them.
        """
        from ...thresholds import METHODS

        self.volcano.offer_thresholds(
            [(f"{name} — {METHODS[name][1].split(' --')[0]}",
              (lambda n=name: self.set_threshold_method(n)),
              name == self._threshold_method) for name in METHODS],
            multiplier=self._threshold_multiplier,
            on_multiplier=self.set_threshold_multiplier)

    def set_threshold_method(self, method) -> None:
        """Measure the effect-size cut a different way, and redraw."""
        self._threshold_method = method
        self._offer_thresholds()
        self._redraw_volcano()
        self.say(self._threshold_sentence())

    def set_threshold_multiplier(self, multiplier) -> None:
        """How many spreads wide the cut is."""
        self._threshold_multiplier = float(multiplier)
        self._offer_thresholds()
        self._redraw_volcano()
        self.say(self._threshold_sentence())

    def _threshold_sentence(self) -> str:
        """What the cut is, and what drew it. Never a bare number."""
        from ...thresholds import coefficient_threshold, describe

        frame = self._frame
        if frame is None or "condition" not in getattr(frame, "columns", ()):
            return "No control coefficients, so no effect-size cut."
        controls = frame.loc[
            frame["condition"].astype(str).str.lower().isin(("nc", "control")),
            self._effect_column(frame)]
        value, rule = coefficient_threshold(
            controls, self._threshold_method, self._threshold_multiplier)
        if value is None:
            return f"No effect-size cut: {rule}."
        return (f"Effect-size cut {value:.3g} — {rule}. "
                f"{describe(self._threshold_method)}.")

    def _offer_compartments(self) -> None:
        """Put this screen's own compartments on the volcano's menu.

        NOT ALL 27 IN THE REFERENCE TABLE. A menu offering 22 choices that
        would colour nothing is a menu where a choice that colours nothing is
        indistinguishable from a broken one.
        """
        from ...localisation import present

        try:
            names = present(self._frame) if self._frame is not None else []
        except Exception:                                        # noqa: BLE001
            names = []
        options = [("none (up / down)", lambda: self.set_compartment(None),
                    self._compartment is None)]
        options += [(name, (lambda n=name: self.set_compartment(n)),
                     name == self._compartment) for name in names]
        self.volcano.offer_compartments(options if names else [])

    def set_compartment(self, name) -> None:
        """Colour one TAGM/LOPIT compartment against grey, or none.

        ONE. "Everything is grey except what the sentence is about" -- and a
        27-colour volcano is both what that rule forbids and, measured, the
        version whose legend cost 40 ms of a 49 ms redraw.
        """
        self._compartment = name
        self._offer_compartments()
        self._redraw_volcano()
        if name:
            from ...localisation import mask

            found = int(mask(self._frame, name).sum()) if self._frame is not None else 0
            self.say(f"{found} of {len(self._frame)} coefficients are "
                     f"annotated {name} in the TAGM/LOPIT table; the rest are "
                     f"grey."
                     if found else
                     f"No coefficient in this screen is annotated {name}, so "
                     f"nothing is picked out.")

    def set_baseline(self, kind, name=None) -> None:
        """Measure every effect from ``kind`` -- see :mod:`spacr.baseline`.

        The interactive volcano only. The saved figures take their own
        baseline argument, so a user who moved it here and then exported gets
        a picture and a caption that agree.
        """
        self._baseline = (kind, name)
        self._offer_baselines()
        self._redraw_volcano()
        if self._frame is not None:
            from ...baseline import resolve

            chosen = resolve(self._frame, kind or "zero",
                             column=self._effect_column(self._frame),
                             name=name)
            # THE REASON, WHEN THERE IS ONE. A request that could not be
            # honoured silently falling back to zero is a user who believes
            # they are reading control-relative effects and is not.
            self.say(chosen.sentence
                     + (f" Asked for the {kind} baseline, but "
                        f"{chosen.reason}." if chosen.reason else ""))

    def _redraw_volcano(self) -> None:
        if self._frame is None:
            return
        kind, column = self._ranking
        # A volcano's y-axis IS -log10(p). Where there is no p-value the axis
        # has nothing to be, so the plot is left empty on purpose and says
        # why -- rather than plotting the OLS-style number a penalised fit
        # carries, which would look exactly like a volcano and be one of a
        # quantity nobody tested.
        p_column = column if kind == "p-value" else "\0no p-value"
        # MEASURED FROM WHATEVER THE USER CHOSE, on a copy. The run's own
        # table is not shifted under the coefficient table beside it.
        from ...baseline import apply as apply_baseline
        from ...baseline import resolve as resolve_baseline

        effect_column = self._effect_column(self._frame)
        baseline_kind, baseline_name = self._baseline
        baseline = resolve_baseline(self._frame, baseline_kind or "zero",
                                    column=effect_column, name=baseline_name)
        frame = apply_baseline(self._frame, baseline, column=effect_column)

        # THE LOPIT OPTION IS DERIVED, so it is materialised onto the copy
        # rather than looked up as a column. On the copy, not the run's own
        # table -- the same rule the baseline follows.
        category = self._colour_by.currentData()
        if category == self.LOPIT_KEY:
            from ...localisation import of as compartments_of

            frame = frame.copy()
            frame["localisation"] = compartments_of(frame).replace(
                "", "unannotated")
            category = "localisation"

        mask = self._level_mask(frame)
        if mask is not None:
            frame = frame.loc[mask]

        self.volcano.set_results(
            frame,
            effect=effect_column,
            p_column=p_column,
            label_column="feature" if "feature" in frame.columns
            else frame.columns[0],
            category_column=category,
            key_column=self._key_column(frame),
            compartment=self._compartment,
        )
        if kind != "p-value":
            self.volcano.set_status(
                "No p-value in this table, so there is no -log10(p) to plot. "
                "The coefficients are in the Coefficients tab.")
        # THE SELECTION SURVIVES A SETTINGS CHANGE. Changing the colouring
        # redraws from scratch; without this the ring the user was reading
        # disappears and they have to find their guide again.
        if self._selected_key is not None:
            self.volcano.highlight_key(self._selected_key)

    def _draw_controls(self, frame) -> None:
        """Split the effects by the control labels the fit assigned."""
        if "condition" not in frame.columns:
            self.controls.set_groups({})
            return
        effect = self._effect_column(frame)
        if effect not in frame.columns:
            self.controls.set_groups({})
            return
        groups, keys = {}, {}
        key_column = self._key_column(frame)
        names = {"nc": "negative", "pc": "positive", "control": "control",
                 "other": "other"}
        for key, label in names.items():
            rows = frame[frame["condition"].astype(str) == key]
            if len(rows):
                groups[label] = rows[effect].to_numpy()
                # Sliced out of the frame WITH their values, so a dot's row
                # travels with it into a group that is drawn in a different
                # order from the table.
                if key_column is not None:
                    keys[label] = rows[key_column].astype(str).tolist()
        self.controls.set_groups(groups, keys=keys or None)

    @staticmethod
    def _key_column(frame) -> Optional[str]:
        """The column that names a row uniquely, or ``None`` if none does.

        Checked rather than assumed. ``feature`` is the design-matrix term
        name and is one-to-one with the row on every table this module writes,
        but a frame arriving from somewhere else may not carry it -- and
        ``gene`` and ``grna`` are NOT keys, because a gene has several guides
        and therefore several rows. Joining on one of those selects an
        arbitrary member of the group.
        """
        if frame is None:
            return None
        for name in ("feature",):
            if name in frame.columns and frame[name].is_unique:
                return name
        return None

    def _select_key(self, key: str) -> None:
        """A row was picked: mark it on EVERY plot that drew it.

        The link runs both ways on all of them, not just the volcano. A guide
        found in the table should light up in the Q-Q as well -- that is how
        a user answers "is my hit the one lifting off the diagonal", which is
        the question the Q-Q exists for and could not previously be asked.

        Each plot answers for itself, and a False is a real answer: a
        coefficient with an unusable p-value is on no plot, a nuisance term is
        off the volcano on purpose, and a guide is not a point on the
        per-gene agreement plot at all.
        """
        self._selected_key = str(key)
        for plot in self._keyed_plots():
            plot.note_selection(key, plot.highlight_key(key))
        # The histogram has no point to ring, but it can outline the bar the
        # coefficient falls in, which is the honest equivalent. No note goes
        # with it: a bar is a hundred rows, and printing one row's name beside
        # it would read as a claim that the bar IS that row.
        self.p_values.highlight_key(key)

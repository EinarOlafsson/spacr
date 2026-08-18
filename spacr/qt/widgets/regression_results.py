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

ONE GENE/GUIDE FILTER FOR ALL OF THEM -- instruction 128 L. `set_level` is
read by every draw path, not by the volcano alone, because a filter that
reaches four of six tabs is worse than one that reaches none: the two then
disagree on screen at the same time and nothing says which is which. Filtering
the family also CHANGES A DIAGNOSTIC rather than merely narrowing it -- on a
table of 200 null gene terms and 600 enriched guide terms the Q-Q reports
inflation at the median of 2.90, 0.97 and 4.07 for the whole fit, the genes
and the guides, three answers to "is this screen calibrated" from one plot --
so the family is written into the tab label and into the plot's own title.
The three well-level tabs are NOT filtered, because a residual is one WELL and
a well is neither a gene nor a guide; they say so rather than staying quiet.

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
        from .fast_plots import (ControlSeparation, EffectDistribution,
                                 EffectRankPlot, GuideAgreementPlot,
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
        # THE SAME GESTURE ON THE TABLE AS ON THE PLOT. Instruction 128 L:
        # "i should be able to right click on the coeffisients table and only
        # see grna or genes and this should also filer the subsequent
        # data/graphs in the subsequent tabs". Wired to the SAME
        # :meth:`set_level`, so the two entry points cannot end up with two
        # opinions about which rows the panel is showing.
        self.table.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.table.customContextMenuRequested.connect(self._level_menu_at)
        if self.external_volcano:
            # The table is the panel; the graph is the caller's to place.
            self._volcano_tab, self._volcano_tab_name = self.table, "Coefficients"
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
            self._volcano_tab, self._volcano_tab_name = split, "Volcano"
            self.tabs.addTab(split, "Volcano")

        # THE TWO EFFECT PANELS, which the sheet has drawn since it existed
        # and the screen had no twin of -- instruction 129 B, where the gap
        # was measured at exactly these two. They sit directly after the
        # volcano because they read in the same order the sheet does: the
        # result, then HOW BIG it is and how sure, then whether the model was
        # entitled to say it.
        #
        # A volcano ranks by significance and cannot answer either question.
        # On the TSG101 screen its top guide by p is not its top guide by
        # effect, and the strongest effect in the screen (4.37) has q = 3e-05
        # while the third strongest (-4.22) has q = 0.063 -- one is called and
        # the other is not, and only a ranked list with intervals shows that
        # they are the same size.
        self.effect_rank = EffectRankPlot()
        self.effect_distribution = EffectDistribution()
        self.tabs.addTab(self.effect_rank, "Effect rank")
        self.tabs.addTab(self.effect_distribution, "Effect distribution")
        self.tabs.setTabToolTip(
            self.tabs.indexOf(self.effect_rank),
            "Every coefficient ranked by the size of its effect, as a dot "
            "with its confidence interval. The volcano ranks by significance, "
            "which is a different order.")
        self.tabs.setTabToolTip(
            self.tabs.indexOf(self.effect_distribution),
            "Where this screen's effects sit, and how wide the null under "
            "them is. σ is a MAD, which the outliers a screen is looking for "
            "do not inflate.")

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

        # THE HOMOGENEITY VERDICT, BESIDE THE PICTURE. Instruction 128 M: the
        # Scale-location tab shipped the plot and never asked the question the
        # plot exists to answer -- is the residual spread constant across the
        # fitted range? A reader who cannot answer that from a scatter of 610
        # points (and most cannot) has no way to know that every standard
        # error in the Summary tab, and so every p-value in the coefficient
        # table, is optimistic.
        #
        # A LABEL, NOT A STATUS LINE. The plot's own status is overwritten by
        # whatever was last clicked (`FastPlot.note_selection`), and a verdict
        # that disappears when the user interacts with the panel is a verdict
        # they will not have read.
        self.homogeneity = QLabel(self.NO_HOMOGENEITY_VERDICT)
        self.homogeneity.setWordWrap(True)
        self.homogeneity.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._scale_location_tab = QWidget()
        spread = QVBoxLayout(self._scale_location_tab)
        spread.setContentsMargins(0, 0, 0, 0)
        spread.setSpacing(4)
        spread.addWidget(self.scale_location, 1)
        spread.addWidget(self.homogeneity)
        self.tabs.addTab(self._scale_location_tab, "Scale-location")
        self.tabs.addTab(self.influence, "Influence")
        # The statsmodels summary closes the diagnostic group: it is the
        # model-level readout the panels above it are pictures of.
        self.tabs.addTab(self._summary, "Summary")
        self.tabs.setTabToolTip(
            self.tabs.indexOf(self.residuals),
            "Residual against fitted, one point per well. A horizontal band "
            "is a well-specified mean; a curve or a funnel is not.")
        self.tabs.setTabToolTip(
            self.tabs.indexOf(self._scale_location_tab),
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
        self._support_tab = agreement_split
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
        # The effect distribution is the OTHER histogram, and it takes the
        # same route for the same reason: its marks are bars of many
        # coefficients, so it narrows the table rather than guessing which of
        # them the user meant.
        self.effect_distribution.key_selected.connect(self.table.select_key)
        self.effect_distribution.keys_selected.connect(self._show_keys)
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
        #: "raw" or "adjusted" -- which p-value the volcano's y-axis is.
        self._p_value_kind = "raw"
        self._p_value_note = ""
        #: How the effect-size cut is measured, and how wide.
        self._threshold_method = "mad"
        self._threshold_multiplier = 3.0
        #: Why a colour-by option is present but useless, or "".
        self._colour_by_note = ""
        #: None, "gene" or "grna" -- which rows EVERY tab draws. One piece of
        #: state, read by every draw path: see :meth:`refresh_views`.
        # "grna", NOT None. The coefficient table holds BOTH levels, so
        # None drew a gene once per guide PLUS once for itself: on the real
        # screen `225160` is four rows -- grna[225160_1], [_2], [_3] and
        # gene[225160] -- all labelled 225160. Reported as "GRA14 and 225160
        # occur in the top right side of the graph 4 times each which is
        # obviously wrong". It is.
        #
        # It looks fine on the RAW axis because the four have different p
        # values and spread vertically; on the ADJUSTED axis Benjamini-
        # Hochberg ties pull them to one height and they stack into a single
        # spot. The adjusted axis did not create it, it made it visible --
        # which is why it was reported as an adjusted-p bug, and why I looked
        # in the wrong place three times.
        #
        # Guides rather than genes: the guide is the unit the screen
        # measures, and a permutation run reports guides ONLY -- so this is
        # the level on which the two inference modes agree. Instruction 128 R
        # is the real repair: fit the two levels SEPARATELY.
        self._level = "grna"
        #: The fitted model behind the table, when a run in this session
        #: produced it. BORN HERE for the same reason as everything else in
        #: this block.
        self._model = None
        #: The :class:`spacr.regression_qc.RegressionQCContext` the residual
        #: tabs were drawn from, kept so the homogeneity verdict is read off
        #: the SAME arrays the saved report used rather than rebuilt.
        self._qc_context = None
        #: regression_qc's own scale-location statistics, verbatim.
        self._homogeneity = {}
        #: The constant-spread verdict now on screen.
        self._homogeneity_text = self.NO_HOMOGENEITY_VERDICT

        # The diagnostics start out saying what they are waiting for. An
        # empty plot with no sentence is indistinguishable from a broken one.
        self.clear_diagnostics()
        for plot in (self.effect_rank, self.effect_distribution):
            plot.set_status(self.NO_EFFECTS_YET)

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

    #: What the Scale-location tab's verdict says before a fit reaches it.
    #: A blank space under a plot reads as "nothing to report", which is the
    #: one thing this panel must never say by accident.
    NO_HOMOGENEITY_VERDICT = (
        "No constant-spread verdict yet: it is computed from the fitted "
        "model, and only a run in this session hands one over.")

    #: What to reach for when the spread is not constant. NAMED, because
    #: "the standard errors are optimistic" without the remedy is a warning
    #: a reader can only act on by guessing. ``cov_type`` is already a spaCR
    #: setting -- see :data:`spacr.regression_spec` -- so this is a re-fit,
    #: not a feature request.
    HC3_FIX = (
        "The fix: re-fit with cov_type='HC3', a sandwich "
        "(heteroscedasticity-consistent) covariance estimator that is valid "
        "when the spread is not constant. It is already a supported spaCR "
        "setting for ols, wls, glm, poisson, quasi_binomial, logit and "
        "probit, and it changes the standard errors and every p-value while "
        "leaving the coefficients exactly where they are. Right-click the "
        "volcano and choose “Re-fit with another model…”.")

    #: What :func:`spacr.regression_qc._panel_scale_location` found, in a
    #: sentence. KEYED ON THAT MODULE'S OWN VERDICT STRINGS so the live tab
    #: and the saved PDF cannot describe one fit two ways: the statistics come
    #: from :func:`spacr.regression_qc.draw_panel` and only the wording is
    #: this panel's.
    HOMOGENEITY_FINDINGS = {
        "no detectable trend in spread":
            "CONSTANT SPREAD. The residual spread does not change across the "
            "fitted range.",
        "variance grows with the fit":
            "SPREAD GROWS WITH THE FITTED VALUE.",
        "variance shrinks with the fit":
            "SPREAD SHRINKS WITH THE FITTED VALUE.",
        "spread differs across the fit, but not monotonically":
            "SPREAD IS NOT CONSTANT across the fitted range -- a funnel "
            "rather than a slope, which a rank correlation on its own "
            "reports as nothing at all.",
    }

    #: What the finding DOES TO THE TABLE, which is the half that changes
    #: what a reader does. Only reached when the fit's standard errors are the
    #: classical ones; a fit that already used a sandwich estimator gets
    #: :data:`ALREADY_ROBUST` instead, because telling its reader their errors
    #: are optimistic would be false.
    HOMOGENEITY_CONSEQUENCES = {
        "no detectable trend in spread":
            "The ordinary standard errors in the Summary tab -- and so every "
            "p-value in the coefficient table and on the volcano -- are the "
            "right ones.",
        "variance grows with the fit":
            "The standard errors above are OPTIMISTIC, so every p-value in "
            "the coefficient table is smaller than it should be and the hits "
            "at the top of the volcano gain most from the error.",
        "variance shrinks with the fit":
            "The standard errors above are wrong in both directions -- "
            "conservative where the fit is small, optimistic where it is "
            "large -- so the p-values are not comparable across the range.",
        "spread differs across the fit, but not monotonically":
            "The standard errors above are wrong by an amount that depends "
            "on the fitted value, so the p-values are not comparable across "
            "the range.",
    }

    #: The consequence for a fit that was ALREADY given a sandwich estimator.
    #: The picture looks the same and means something else: the spread really
    #: does vary, and the errors already account for it.
    ALREADY_ROBUST = (
        "This fit was given cov_type={used!r}, so the standard errors in the "
        "Summary tab and every p-value in the table are ALREADY robust to "
        "this. The plot is describing the data, not warning about the table.")

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
        self._qc_context = None
        for plot in self.diagnostic_plots():
            plot._reset_scene()
            plot.set_status(reason or self.NO_MODEL_MESSAGE)
        # THE VERDICT GOES WITH THE PICTURE IT JUDGED. Left behind, it would
        # be a sentence about the previous fit sitting under an empty plot,
        # which is the worst of both: authoritative and about nothing.
        self._homogeneity = {}
        self._homogeneity_text = (
            f"No constant-spread verdict: {reason}" if reason
            else self.NO_HOMOGENEITY_VERDICT)
        self.homogeneity.setText(self._homogeneity_text)

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
        self._qc_context = ctx
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
        self.judge_homogeneity(ctx)
        # The three plots above each wrote their own headline, which clears
        # whatever note was on them -- so the "these are wells, not
        # coefficients" sentence is put back last or it is not there at all.
        self._note_the_diagnostics()
        return True

    def judge_homogeneity(self, ctx=None) -> str:
        """Say whether the residual spread is constant, and what to do if not.

        :param ctx: a :class:`spacr.regression_qc.RegressionQCContext`.
            ``None`` re-judges the context the diagnostics were last drawn
            from, so a caller does not have to keep a copy of it to ask again.
        :returns: the verdict now on screen.

        NOT ONE NUMBER RE-DERIVED HERE. The statistics come from
        :func:`spacr.regression_qc.draw_panel` -- Spearman's rho on
        sqrt|standardised residual| against fitted, a Brown-Forsythe test
        across quartiles of the fitted value, and the quartile SD ratio --
        drawn into a throwaway axes purely to get the dict it returns. That
        module is 3,444 lines and already computes them for the PDF the run
        writes; a second implementation here would disagree with it about the
        same fit inside a week, and the reader would have no way to tell which
        of the two was wrong.

        TWO STATISTICS, NOT ONE, and that is regression_qc's decision rather
        than this panel's: a rank correlation is exactly zero for a SYMMETRIC
        funnel -- wide at both ends, narrow in the middle -- which is what a
        mis-specified link produces, and a panel reporting "no trend in
        spread" over one would be a confident wrong answer.
        """
        from matplotlib.figure import Figure

        from ...regression_qc import PanelUnavailable, draw_panel

        ctx = self._qc_context if ctx is None else ctx
        if ctx is None:
            self._homogeneity = {}
            self._homogeneity_text = self.NO_HOMOGENEITY_VERDICT
            self.homogeneity.setText(self._homogeneity_text)
            return self._homogeneity_text
        figure = Figure()
        try:
            stats = draw_panel("scale_location", ctx, figure.add_subplot(111))
        except PanelUnavailable as error:
            # A real answer about the fit -- a quantile or hinge fit has no
            # error scale, so it has no standardised residual to judge -- and
            # it must not read like a broken panel.
            self._homogeneity = {}
            self._homogeneity_text = f"No constant-spread verdict: {error}"
        except Exception as error:                               # noqa: BLE001
            self._homogeneity = {}
            self._homogeneity_text = (
                f"No constant-spread verdict: the test could not be computed "
                f"for this fit ({type(error).__name__}: {error}).")
        else:
            self._homogeneity = dict(stats)
            self._homogeneity_text = self._homogeneity_sentence(stats)
        finally:
            figure.clf()
        self.homogeneity.setText(self._homogeneity_text)
        index = self.tabs.indexOf(self._scale_location_tab)
        if index >= 0:
            self.tabs.setTabToolTip(index, self._homogeneity_text)
        return self._homogeneity_text

    def homogeneity_verdict(self) -> str:
        """Whether the residual spread is constant, in the words on screen.

        Public because it is the one sentence on the Scale-location tab that
        changes what a reader does, and a test that reads it back has to be
        able to do so without reaching into a label.
        """
        return self._homogeneity_text

    def homogeneity_stats(self) -> dict:
        """The numbers behind the verdict -- ``regression_qc``'s own dict.

        ``spearman_rho``, ``spearman_p``, ``levene_p``,
        ``quartile_sd_ratio``, ``verdict`` and ``n_points``, exactly as
        :func:`spacr.regression_qc.draw_panel` returned them for the saved
        scale-location panel. Empty when there is no verdict.
        """
        return dict(self._homogeneity)

    def _homogeneity_sentence(self, stats) -> str:
        """regression_qc's finding, what it costs, the fix, and the numbers."""
        verdict = str(stats.get("verdict", ""))
        finding = self.HOMOGENEITY_FINDINGS.get(
            verdict,
            f"The constant-spread test returned “{verdict}”, which this "
            f"panel has no sentence for; the numbers are below.")
        numbers = (
            f"Spearman rho = {stats.get('spearman_rho', float('nan')):+.2f} "
            f"(p = {stats.get('spearman_p', float('nan')):.2g}), "
            f"Brown-Forsythe p = {stats.get('levene_p', float('nan')):.2g}, "
            f"max/min quartile SD = "
            f"{stats.get('quartile_sd_ratio', float('nan')):.2f}, over "
            f"{stats.get('n_points', 0)} wells. These are the same numbers "
            f"the saved regression_qc/scale_location panel prints.")
        return " ".join(part for part in
                        (finding, self._consequence(verdict), numbers) if part)

    def _consequence(self, verdict) -> str:
        """What the finding costs the table, and the fix when there is one.

        THE FIT'S OWN COVARIANCE IS READ FIRST. A model already given
        ``cov_type='HC3'`` has robust standard errors, so "the errors above
        are optimistic" would be simply false about it and "re-fit with HC3"
        would be advice to repeat what was already done. Read off the model
        rather than off the settings, because the settings copy on disk is
        overwritten by every later run of the same screen.
        """
        used = str(getattr(self._model, "cov_type", "") or "")
        robust = bool(used) and used.lower() not in ("nonrobust", "none")
        flat = verdict == "no detectable trend in spread"
        if flat or not robust:
            consequence = self.HOMOGENEITY_CONSEQUENCES.get(verdict, "")
        else:
            consequence = self.ALREADY_ROBUST.format(used=used)
        if flat or robust:
            return consequence
        return f"{consequence} {self.HC3_FIX}".strip()

    def _keyed_plots(self) -> tuple:
        """Every plot whose marks are individual coefficients or genes.

        The volcano, the effect ranking, the Q-Q, the control panel and the
        guide-agreement plot. The two HISTOGRAMS are deliberately not here:
        their marks are bins of many rows, so neither can select one and
        neither pretends to. The residual plot is not here either -- its
        points are wells, not coefficients, and there is no key for a well.
        """
        return (self.volcano, self.effect_rank, self.qq, self.controls,
                self.agreement)

    def _show_keys(self, keys) -> None:
        """A set of coefficients was chosen on a plot: narrow the table."""
        self.table.show_keys(list(keys))

    # ------------------------------------------------------------------ load

    def results_frame(self):
        """The RUN's coefficient table, whole, or ``None``.

        Not what the tabs are currently drawing: the gene/guide filter is a
        view and never an edit, so a caller exporting or re-plotting the
        results gets the fit rather than whatever the user last right-clicked.
        :meth:`filtered_frame` is the other one.

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
        for histogram in (self.p_values, self.effect_distribution):
            histogram.clear_highlight()

        # THE COMPARTMENT MENU IS BUILT FROM THE TABLE, so it has to be
        # rebuilt when the table changes. Built once in __init__ it is built
        # from no frame at all, which is an empty submenu that never appears
        # -- and a new screen would otherwise be offered the last one's
        # compartments.
        self._compartment = None
        # THE DEFAULT LEVEL IS READ OFF THE TABLE, not asserted.
        #
        # Defaulting to "grna" unconditionally is what fixed the four-fold
        # duplication -- a gene drawn once per guide -- and it is right for
        # the table a mixed or hierarchical run writes, which carries both
        # levels. It is WRONG for the table instruction 128 R produces: the
        # separate gene fit writes `results_gene.csv`, whose every row is a
        # gene term, and a guide filter over it selects nothing. The panel
        # then draws an empty volcano with a full coefficient table beside
        # it, which reads as a broken plot rather than as an empty filter.
        self._level = self._default_level()
        self._offer_levels()
        self._offer_p_values()
        self._offer_thresholds()
        self._offer_compartments()

        # EVERY TAB THROUGH ONE PATH. A new table and a change of the
        # gene/guide filter draw the panel with the same method, so the two
        # cannot leave it in two different states -- which is the whole of
        # instruction 128 L.
        self.refresh_views()
        if self._colour_by_note:
            # SAID, not swallowed. A colouring the user expected and cannot
            # find is a bug report; the same colouring listed with the reason
            # it is useless is an answer.
            self.say(f"{self._status} Colouring: {self._colour_by_note}.")
        self.loaded.emit(source or "")
        return True

    def refresh_views(self) -> None:
        """Draw EVERY tab from the coefficient table at the chosen level.

        ONE PIECE OF STATE, READ SIX TIMES. `_level` used to reach the volcano
        and nothing else, so "genes only" left the coefficient table, the
        p-value histogram, the Q-Q, the control panel and the guide support
        showing the whole fit -- five tabs disagreeing with the sixth, at the
        same time, with nothing on screen saying which was which. That is
        worse than no filter at all: a reader who trusts the volcano and reads
        the inflation figure off the Q-Q beside it has combined two different
        multiple-testing families and cannot tell.

        Public because it is also what a caller does after changing something
        the panel does not own -- and because a private redraw that four
        methods have to remember to call is how the fifth one forgets.
        """
        frame = self._frame
        if frame is None:
            # THE LABELS STILL GO ON. A panel with no table yet is still a
            # panel whose filter is set to something, and tab labels that
            # disagreed with `level()` for as long as it took a run to finish
            # would be the exact failure this method exists to prevent.
            self._say_which_family()
            return
        shown = self.filtered_frame()
        kind, column = self._ranking
        self._redraw_volcano()
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
                name in shown.columns
                for name in ("q_value", "adjusted_p_value", "p_value")):
            significance = column
        self.table.set_frame(
            shown, key_column=self._key_column(shown),
            significance_column=significance)
        self._show_significance(shown, kind, column, self._path or "")
        self._draw_effects(shown, kind)
        self._draw_controls(shown)
        # THE FULL TABLE, on purpose. A gene's concordance is how its GUIDES
        # agree, so the guide rows are what the number is made of -- see
        # `_draw_guide_support`, which narrows which genes are LISTED without
        # touching how any of them was computed.
        self._draw_guide_support(frame)
        self._say_which_family()

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
        # THE FILTER NARROWS WHICH GENES ARE LISTED, never how one was
        # measured. "genes only" keeps the genes the fit gave a gene-level
        # term -- the ones whose dot is on the filtered volcano -- and drops
        # the genes that exist here only as a bundle of guides. "guides only"
        # drops nothing, because every row of this table IS a gene's guides;
        # that is stated on the tab rather than left to look like a filter
        # that failed to fire.
        if self._level == "gene":
            table = table[table["feature"].notna()].reset_index(drop=True)
        if not len(table):
            self.support.set_frame(None)
            self.agreement.set_support(None)
            self.agreement.set_status(
                "No gene in this table was fitted a gene-level term, so "
                "“genes only” leaves nothing to draw here. "
                + self.GUIDE_SUPPORT_NEEDS_BOTH)
            return
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

    #: Which rows EVERY tab shows. `None` is both, which is what a screen
    #: IS, so it is the default -- a panel that opened already filtered would
    #: be reporting a subset as the result.
    LEVELS = ((None, "genes and guides"), ("gene", "genes only"),
              ("grna", "guides only"))

    #: The level, as it is said in a sentence.
    LEVEL_NAMES = {None: "genes and guides", "gene": "genes only",
                   "grna": "guides only"}

    #: The level, as it is appended to a tab. Short, because it goes on five
    #: tab labels at once and a tab bar that wraps has hidden the filter it
    #: was put there to advertise.
    LEVEL_SUFFIXES = {None: "", "gene": " (genes)", "grna": " (guides)"}

    #: The level, as it is appended to a PLOT's title. Longer than the tab
    #: suffix: the title has the room, and a reader looking at a picture is
    #: the reader most likely to forget which filter is on.
    LEVEL_TITLES = {None: "", "gene": " — genes only",
                    "grna": " — guides only"}

    #: Why the three well-level tabs do not follow the gene/guide filter.
    #: SAID, not left to be noticed: a filter that reaches five tabs and
    #: silently skips three is the same failure as one that reaches four of
    #: six, only quieter.
    DIAGNOSTICS_ARE_WHOLE_FIT = (
        "NOT FILTERED: one point here is one WELL, and a well is neither a "
        "gene nor a guide. These describe the whole fit whatever the "
        "coefficient tabs are showing.")

    #: Why the guide-support tab is narrowed differently from the rest.
    GUIDE_SUPPORT_NEEDS_BOTH = (
        "Computed from the whole table, always: a gene's concordance is how "
        "its own guides agree, so the guide rows are what the number is made "
        "of and a gene-only table has none of them. The filter narrows which "
        "genes are LISTED, never how one was measured.")

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

    def _default_level(self):
        """The level to open a new table at.

        Guides when the table has any -- the guide is the unit the screen
        measures, and drawing a gene once per guide is the duplication this
        default exists to prevent. Genes when it has no guide terms at all,
        which is exactly what the separate gene fit writes. Whole fit when it
        has neither, so an unrecognised table is shown rather than hidden.
        """
        frame = self._frame
        if frame is None or "feature" not in getattr(frame, "columns", ()):
            return None
        from ...hits import guide_of, tested_family

        features = frame["feature"].map(str)
        if features.map(lambda f: guide_of(f) is not None).any():
            return "grna"
        # `tested_family`, NOT `level_counts()["gene"]` and NOT `gene_of`.
        #
        # `level_counts()["gene"]` is the COMPLEMENT of the guides, so it
        # counts `Intercept` and every row/column term as a gene -- the right
        # number for the menu label, because "genes only" is what the mask
        # actually selects, and the wrong test here.
        #
        # `gene_of` does not draw the line either: it parses the bracketed
        # token, so `rowID[T.r03]` comes back as the "gene" r03. The one
        # statement of which coefficients are hypotheses lives in `hits`, and
        # asking it is what stops this becoming a second copy that drifts.
        if tested_family(features).any():
            return "gene"
        return None

    def _offer_levels(self) -> None:
        """Put genes / guides / both on the volcano's right-click menu."""
        counts = self.level_counts()
        self.volcano.offer_levels([
            (f"{label} ({counts.get(key, 0)})",
             (lambda k=key: self.set_level(k)), key == self._level)
            for key, label in self.LEVELS])

    def build_level_menu(self):
        """The genes / guides / both menu, with this table's counts on it.

        Public for the reason :meth:`spacr.qt.widgets.fast_plots.FastPlot.build_style_menu`
        is: it is how a test reads what the user is offered without
        synthesising a right-click, and how a second surface offers the SAME
        three entries instead of growing a copy that drifts from this one.

        The counts are in the labels because "genes only" that silently draws
        300 of 1,200 rows is a filter a user applies without knowing what they
        gave up -- the same rule the volcano's menu already follows.
        """
        from PySide6.QtWidgets import QMenu

        menu = QMenu(self)
        heading = menu.addAction("Show, on every tab")
        heading.setEnabled(False)
        menu.addSeparator()
        counts = self.level_counts()
        for key, label in self.LEVELS:
            action = menu.addAction(f"{label} ({counts.get(key, 0)})")
            action.setCheckable(True)
            action.setChecked(key == self._level)
            action.triggered.connect(
                lambda _checked=False, chosen=key: self.set_level(chosen))
        return menu

    def _level_menu_at(self, position) -> None:
        """Right-click on the coefficients table: the same three choices.

        Instruction 128 L, in the maintainer's words: "i should be able to
        right click on the coeffisients table and only see grna or genes and
        this should also filer the subsequent data/graphs in the subsequent
        tabs". It sets the same `_level` the volcano's menu does, so the two
        gestures cannot disagree about what the panel is showing.
        """
        self.build_level_menu().exec(
            self.table.table.viewport().mapToGlobal(position))

    def level(self):
        """``None``, ``"gene"`` or ``"grna"`` -- which family every tab shows.

        Public because it is the one piece of state that decides what SIX
        tabs are drawing, and a caller that has to read it off a private
        attribute is a caller that will one day set it there too.
        """
        return self._level

    def filtered_frame(self):
        """The coefficient table at the chosen level, or ``None``.

        This is what every tab is drawn from -- see :meth:`refresh_views`.
        :meth:`results_frame` is the RUN's table and stays whole: the filter
        is a view, not an edit, and a caller exporting the results must get
        the fit rather than whatever the user last right-clicked.
        """
        frame = self._frame
        if frame is None:
            return None
        mask = self._level_mask(frame)
        return frame if mask is None else frame.loc[mask]

    def family_note(self) -> str:
        """Which multiple-testing family the p-value and Q-Q tabs are drawing.

        THE ONE THING FILTERING A Q-Q CHANGES THAT FILTERING A VOLCANO DOES
        NOT. A volcano of the guides is the same dots with some removed; a
        Q-Q of the guides is a DIFFERENT DIAGNOSTIC -- the expected quantiles
        are recomputed over 900 tests instead of 1,200, so the diagonal moves,
        the inflation figure at the median is a different number, and the
        excess in the histogram's first bin is this family's excess and not
        the run's. A reader who does not know which one is on screen cannot
        use either.
        """
        counts = self.level_counts()
        total = counts.get(None, 0)
        if not self._level:
            return (f"Every tab covers the whole fit: all {total} "
                    f"coefficients, one multiple-testing family.")
        family = "genes" if self._level == "gene" else "guides"
        return (f"{family} only — {counts.get(self._level, 0)} of {total} "
                f"coefficients. The p-value histogram and the Q-Q are "
                f"therefore a DIFFERENT multiple-testing family from the "
                f"whole fit: the inflation figure and the excess in the "
                f"first bin are this family's, not the run's.")

    def set_level(self, level) -> None:
        """Draw only genes, only guides, or both -- ON EVERY TAB.

        A FILTER ON THE PANEL, not a mode of the run. The coefficient table
        already carries both -- `feature` is `gene_fraction:gene[...]` or
        `fraction:grna[...]` -- so this needs no re-fit and no second table,
        which is why it is better here than in the settings where it used to
        live.

        ONE PIECE OF STATE. Reached from the volcano's right-click menu and
        from the coefficients table's, and read by every draw path, because a
        filter that reaches four of six tabs is worse than one that reaches
        none: the two then disagree on screen at the same time and nothing
        says which is which.
        """
        self._level = level
        self._offer_levels()
        self.refresh_views()
        counts = self.level_counts()
        shown = counts.get(level, counts.get(None, 0))
        self.say(f"{shown} of {counts.get(None, 0)} coefficients — "
                 f"{self.LEVEL_NAMES.get(level, 'genes and guides')}, on "
                 f"every tab. {self.family_note()}",
                 detail=f"{self.family_note()}\n\n"
                        f"{self.DIAGNOSTICS_ARE_WHOLE_FIT}\n\n"
                        f"{self.GUIDE_SUPPORT_NEEDS_BOTH}")

    def _filtered_surfaces(self) -> tuple:
        """``[(tab widget, tab name, plot, plot title)]`` the filter narrows.

        The family is written into BOTH the tab and the plot's own title: a
        tab label survives a click where a plot's status line does not, and
        the title is what a reader is looking at when they read the picture.
        """
        return (
            (self._volcano_tab, self._volcano_tab_name, self.volcano,
             "Volcano"),
            (self.effect_rank, "Effect rank", self.effect_rank,
             "Effect rank"),
            (self.effect_distribution, "Effect distribution",
             self.effect_distribution, "Effect distribution"),
            (self.p_values, "p-values", self.p_values,
             "p-value distribution"),
            (self.qq, "Q-Q", self.qq, "p-value Q-Q"),
            (self.controls, "Controls", self.controls, "Control separation"),
            (self._support_tab, "Guide support", self.agreement,
             "Guide agreement"),
        )

    def _say_which_family(self) -> None:
        """Write the level onto every tab and every plot that follows it."""
        suffix = self.LEVEL_SUFFIXES.get(self._level, "")
        note = self.family_note()
        for widget, name, plot, title in self._filtered_surfaces():
            index = self.tabs.indexOf(widget)
            if index >= 0:
                self.tabs.setTabText(index, f"{name}{suffix}")
                self.tabs.setTabToolTip(index, note)
            # THE TITLE, NOT THE STATUS LINE. A plot's status is overwritten
            # by whatever was last clicked -- `note_selection` does exactly
            # that -- so a family written there is gone the moment the reader
            # uses the panel. A title is not.
            plot.plot.setTitle(
                f"{title}{self.LEVEL_TITLES.get(self._level, '')}")
        support = self.tabs.indexOf(self._support_tab)
        if support >= 0:
            self.tabs.setTabToolTip(
                support, f"{note}\n\n{self.GUIDE_SUPPORT_NEEDS_BOTH}")
        self._note_the_diagnostics()

    def _note_the_diagnostics(self) -> None:
        """Say on the well-level tabs that the filter does not reach them.

        They are one point per WELL. There is no gene/guide split to make and
        pretending to make one would be worse than the silence it replaces --
        but silence is what let five tabs narrow while three did not, with
        nothing on screen distinguishing "unfiltered on purpose" from
        "forgot to filter".
        """
        note = self.DIAGNOSTICS_ARE_WHOLE_FIT if self._level else ""
        for plot in self.diagnostic_plots():
            plot.set_status_note(note)

    def _level_mask(self, frame):
        """Rows of ``frame`` at the chosen level, or None for all of them."""
        if not self._level or "feature" not in getattr(frame, "columns", ()):
            return None
        from ...hits import guide_of

        is_guide = frame["feature"].map(
            lambda f: guide_of(str(f)) is not None)
        return is_guide if self._level == "grna" else ~is_guide

    def _offer_p_values(self) -> None:
        """Offer raw vs adjusted p-values, when there is a choice.

        Asked for 2026-08-17. Offered ONLY when a corrected column exists:
        a menu entry for "adjusted" on a run with no correction would promise
        a number that is not there, and `multiple_testing_method='none'`
        writes a q_value equal to the p_value, which is not the same thing as
        having corrected.
        """
        frame = self._frame
        corrected = (_match_column(frame, ("q_value", "adjusted_p_value"))
                     if frame is not None else None)
        if corrected is None:
            self.volcano.offer_p_values([])
            return
        method = ""
        if "multiple_testing_method" in getattr(frame, "columns", ()):
            values = frame["multiple_testing_method"].dropna().unique()
            method = str(values[0]) if len(values) else ""
        if method.lower() in ("none", "nan", ""):
            # The column is there and it is the raw p under another name.
            self.volcano.offer_p_values([])
            self._p_value_note = (
                f"No correction was applied, so {corrected!r} equals the raw "
                f"p-value; there is nothing to switch between.")
            return
        self._p_value_note = ""
        self.volcano.offer_p_values([
            (f"raw p-value", lambda: self.set_p_value_kind("raw"),
             self._p_value_kind == "raw"),
            (f"adjusted ({method})",
             lambda: self.set_p_value_kind("adjusted"),
             self._p_value_kind == "adjusted"),
        ])

    def set_p_value_kind(self, kind) -> None:
        """Draw the volcano against the raw or the adjusted p-value."""
        self._p_value_kind = kind
        self._offer_p_values()
        self._redraw_volcano()
        self.say(f"Volcano y-axis: {kind} p-value."
                 + (f" {self._p_value_note}" if self._p_value_note else ""))

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

    def _current_threshold(self):
        """The effect-size cut for the table on screen, or None.

        Split from :meth:`_threshold_sentence` so the number the plot draws
        and the number the sentence quotes come from ONE call -- a panel
        whose line and caption disagreed would be worse than one with no
        line at all.
        """
        from ...thresholds import coefficient_threshold

        controls = self._control_effects()
        if controls is None:
            return None
        value, _rule = coefficient_threshold(
            controls, self._threshold_method, self._threshold_multiplier)
        return value

    def _control_effects(self):
        """The control guides' effects, or ``None`` when there are none to take.

        THE EFFECT COLUMN IS CHECKED, NOT ASSUMED, and that is a crash rather
        than tidiness. :meth:`_effect_column` answers ``"coefficient"`` for a
        table carrying none of the four spellings -- it is a default, not a
        finding -- and ``frame.loc[mask, name]`` on an absent column raises
        KeyError. That escaped through :meth:`refresh_views` into
        :meth:`set_frame`, so a table with a ``condition`` column and an
        unrecognised effect column took the WHOLE PANEL down at load time
        instead of opening and saying which column it could not find.
        """
        frame = self._frame
        if frame is None or "condition" not in getattr(frame, "columns", ()):
            return None
        effect = self._effect_column(frame)
        if effect not in frame.columns:
            return None
        return frame.loc[
            frame["condition"].astype(str).str.lower().isin(("nc", "control")),
            effect]

    def _threshold_sentence(self) -> str:
        """What the cut is, and what drew it. Never a bare number."""
        from ...thresholds import coefficient_threshold, describe

        controls = self._control_effects()
        if controls is None:
            return "No control coefficients, so no effect-size cut."
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
        if kind == "p-value" and self._p_value_kind == "adjusted":
            corrected = _match_column(self._frame,
                                      ("q_value", "adjusted_p_value"))
            if corrected is not None:
                p_column = corrected
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
            # THE CUT THE MENU COMPUTED, actually drawn.
            #
            # Reported 2026-08-17: "the coefficient threshold still dosnt
            # work". It did not: the seven methods, the multiplier and the
            # status sentence all landed, and the NUMBER was never handed to
            # the plot -- `set_results`'s `effect_threshold` defaults to None,
            # so every method redrew the same volcano with no line and only
            # the sentence changed. A feature whose every visible part works
            # except the one that draws it.
            effect_threshold=self._current_threshold(),
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

    #: What the two effect tabs say before a coefficient table reaches them.
    #: A blank plot behind a tab nobody has opened is indistinguishable from a
    #: broken one, which is the failure instruction 129 B names for an ABSENT
    #: tab and which an empty present one commits just as quietly.
    NO_EFFECTS_YET = (
        "No coefficient table yet. Both effect tabs are drawn from the fitted "
        "effects: run a regression, or open one with “Load results…”.")

    #: Why these two tabs cannot be drawn from the table that IS loaded.
    #: Named rather than shrugged at, because it is a specific and checkable
    #: fact about the file: a frame with no fitted-effect column is not a
    #: coefficient table, and no amount of re-running will make it one.
    NO_EFFECT_COLUMN = (
        "No fitted-effect column in this table, so there is nothing to rank "
        "and nothing to distribute. Looked for coefficient, coef, effect and "
        "estimate. Every other tab here is drawn from the same column, so a "
        "table without it is not a regression result.")

    def _draw_effects(self, frame, kind=None) -> None:
        """Fill the Effect rank and Effect distribution tabs, or say why not.

        :param frame: the coefficient table at the chosen gene/guide level.
        :param kind: what orders this table -- :meth:`ranking`'s first item.
            ``"p-value"`` lets the ranking colour its dots by the corrected
            p; anything else says THIS TABLE HAS NO SIGNIFICANCE, which is not
            the same instruction as "go and find one". A penalised fit carries
            an OLS-style ``p_value`` computed as though there were no penalty,
            and a plot left to search would colour its dots by a number nobody
            tested -- the identical trap :meth:`_rank_by` documents.

        A TAB THAT CANNOT BE FILLED SAYS WHY, IN THE TAB. Instruction 129 B:
        an absent tab is indistinguishable from a bug, and a present empty one
        with no sentence is indistinguishable from a broken one.

        THE NUISANCE TERMS COME OUT OF BOTH, and not for the reason one
        expects. :class:`EffectRankPlot` drops them itself; the distribution
        is handed values rather than a frame, so the drop happens here. It is
        the FAMILY that requires it rather than the axis -- measured on the
        TSG101 screen, the intercept's coefficient of 0.190 ranks 547 of 1,213
        and moves σ (MAD) by 0.08% -- but its q-value is NaN, so it can never
        be called, and a covariate among the hypotheses is a different
        experiment from the one the q-values describe.
        """
        from .fast_plots import NO_SIGNIFICANCE

        effect = self._effect_column(frame)
        if effect not in getattr(frame, "columns", ()):
            self.effect_rank.set_results(None)
            self.effect_rank.set_status(self.NO_EFFECT_COLUMN)
            self.effect_distribution.set_effects([])
            self.effect_distribution.set_status(self.NO_EFFECT_COLUMN)
            return
        self.effect_rank.set_results(
            frame, effect=effect, key_column=self._key_column(frame),
            significance_column=(None if kind == "p-value"
                                 else NO_SIGNIFICANCE))
        tested = self._tested_mask(frame)
        family = frame if tested is None else frame.loc[tested]
        self.effect_distribution.set_effects(
            family[effect], keys=self._keys_for(family),
            untested=0 if tested is None else int((~tested).sum()))

    @staticmethod
    def _tested_mask(frame):
        """Rows of ``frame`` that are hypotheses, or ``None`` if it cannot say.

        Via :func:`spacr.hits.tested_family`, which is the repository's single
        statement of where the covariates end and the hypotheses begin. A
        second copy of that rule is how the volcano, the sheet and this panel
        would come to draw three different families.
        """
        if "feature" not in getattr(frame, "columns", ()):
            return None
        try:
            from ...hits import tested_family
        except Exception:              # pragma: no cover - hits unavailable
            return None
        return tested_family(frame["feature"])

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
        # A histogram has no point to ring, but it can outline the bar the
        # coefficient falls in, which is the honest equivalent. No note goes
        # with it: a bar is a hundred rows, and printing one row's name beside
        # it would read as a claim that the bar IS that row.
        for histogram in (self.p_values, self.effect_distribution):
            histogram.highlight_key(key)

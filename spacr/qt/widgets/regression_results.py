"""Inspect coefficient tables and diagnostics from a finished regression.

The results panel combines a clickable volcano plot, sortable coefficient
table, p-value histogram, Q-Q calibration view, assay controls, guide-support
view, model summary, and annotation for the selected gene. Selecting a
coefficient in any linked plot or table highlights the same feature in the
other views.

Gene and gRNA filters are applied to coefficient-level views, including their
calibration diagnostics. Well-level residual and influence diagnostics remain
unfiltered because wells do not belong to either coefficient family. Plot
selections are joined by feature keys rather than drawing positions, so
sorting or aggregation does not change which result is selected.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Dict, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QPushButton, QSplitter, QTabWidget,
    QVBoxLayout, QWidget,
)

LOG = logging.getLogger(__name__)

#: How the effect-size cut is measured, and how wide, on a run nobody has
#: told otherwise. Named so :meth:`RegressionResultsPanel.set_frame`'s reset
#: and the constructor cannot drift: they used to be two literals, and the
#: reset did not exist at all -- so run B inherited run A's cut.
DEFAULT_THRESHOLD_METHOD = "mad"
DEFAULT_THRESHOLD_MULTIPLIER = 3.0

#: Files a regression writes, best first. ``results.csv`` is the full
#: coefficient table; the gene/grna splits are views of it.
RESULT_FILENAMES = ("results.csv", "results_grna.csv", "results_gene.csv")

#: How far below the selected folder to search for a results table. Current
#: runs write ``results/<kind>[_n]/results.csv``; the additional depth keeps
#: older layouts discoverable without walking an entire home directory.
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
    """Find regression result tables beneath a path.

    Parameters
    ----------
    path : path-like
        Results CSV, run directory, or parent directory to search. A CSV is
        returned directly; missing paths and non-CSV files return no matches.
    max_depth : int, default=MAX_SEARCH_DEPTH
        Maximum directory depth to descend below ``path``. The root directory
        is still inspected when this value is zero.
    limit : int, default=MAX_CANDIDATES
        Stop walking after at least this many candidate tables have been
        collected. All recognised tables in the final run directory are kept,
        so the returned count can be slightly larger than this value.

    Returns
    -------
    list of str
        Recognised result-table paths. Run directories are ordered by the
        newest modification time among their tables. Within a run,
        ``results.csv`` precedes the gene and gRNA views according to
        :data:`RESULT_FILENAMES`.
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


def _summary_filenames() -> tuple:
    """Every name a run's summary may be on disk under, newest first.

    Read from :mod:`spacr.ml`, which is the module that WRITES it -- one
    vocabulary, so a rename there cannot leave this reader hunting for a file
    nobody makes any more. Private because it is an implementation detail of
    :func:`find_summary_file`, which is the thing a caller wants.
    """
    try:
        from ...ml import SUMMARY_FILENAMES
    except Exception:                  # pragma: no cover - ml unavailable
        # Named rather than guessed: without the writer there is nothing to
        # agree with, and inventing the list here would be the second source
        # of truth this indirection exists to avoid.
        return ()
    return tuple(SUMMARY_FILENAMES)


def find_summary_file(path) -> Optional[str]:
    """Find the model summary associated with a regression result.

    Parameters
    ----------
    path : path-like
        Results CSV, run directory, or parent directory containing a
        discoverable results table.

    Returns
    -------
    str or None
        Path to the first supported summary file, or ``None`` when no summary
        is found. Both the current and legacy summary filenames are accepted.
    """
    names = _summary_filenames()
    if not path or not names:
        return None
    try:
        root = os.path.abspath(os.path.expanduser(os.fspath(path)))
    except TypeError:
        return None
    folders = []
    if os.path.isfile(root):
        folders.append(os.path.dirname(root))
    elif os.path.isdir(root):
        folders.append(root)
    # A PARENT OF A RUN FOLDER IS ALSO A LEGAL ANSWER, and it is the one
    # `load` accepts, so the summary is looked for beside the table that was
    # actually chosen rather than only in the folder the user typed.
    table = find_results_table(root)
    if table:
        folder = os.path.dirname(table)
        if folder not in folders:
            folders.append(folder)
    for folder in folders:
        for name in names:
            candidate = os.path.join(folder, name)
            if os.path.isfile(candidate):
                return candidate
    return None


#: What the run says when a fit is not identifiable. Repeated on the Summary
#: tab because statsmodels prints a full table of standard errors and P
#: values regardless, and it looks exactly like a summary of a well-posed
#: fit -- which a reader may paste into a methods section from one click
#: away.
UNIDENTIFIABLE_WARNING = (
    "THIS FIT IS SATURATED OR NOT IDENTIFIABLE: {wells} analysed "
    "observations are being "
    "used to estimate {params} parameters.\n"
    "With no residual degrees of freedom, and possible rank deficiency, "
    "individual coefficients, standard errors and P values cannot be "
    "interpreted reliably.\n"
    "Set inference='nonparametric' to test each guide as a plate-blocked "
    "marginal association without fitting every guide coefficient at once, "
    "or use inference='auto' to let spaCR choose.\n")


#: Prefixed to a summary that was READ rather than rendered. A reader pasting
#: this into a methods section is entitled to know it is the run's own text,
#: recovered from the run folder and not recomputed here -- and if the folder
#: were ever pointed at the wrong run, this line is what would show it.
#: The first line of a summary spaCR wrote itself, and the section heading it
#: files the statsmodels text under. Both spelled by
#: :func:`spacr.regression_summary.format_run_summary`; matched rather than
#: re-derived, so a rename there shows up as the spaCR summary going missing
#: from this tab rather than as a wrong-looking one.
SPACR_SUMMARY_HEADING = "spaCR RUN SUMMARY"
VERBATIM_HEADING = "THE STATSMODELS SUMMARY"

SUMMARY_FROM_DISK = (
    "Read from {path}: this is the run's own summary, written when it was "
    "fitted, not recomputed here.")

#: Why there is no fitted model, when nothing more specific is known. Used
#: ONLY where it has been checked -- see :func:`summary_text`.
NO_MODEL_FROM_DISK = (
    "this panel was opened from a results table on disk rather than from a "
    "run, so the fitted model is not here")
NO_MODEL_AT_ALL = "no run in this session has fitted anything yet"

# Canonical results tables share the run folder's remembered plot state.
_CANONICAL_TABLE = "results.csv"


def _why_no_model(path, reason) -> str:
    """Return the verified reason that no fitted model is available.

    Prefer an explicit reason. Otherwise distinguish a table loaded from disk
    from a session in which no model has been fitted.
    """
    if reason:
        return str(reason)
    return NO_MODEL_FROM_DISK if path else NO_MODEL_AT_ALL


def summary_text(model, regression_type=None, *, path=None,
                 reason: str = "") -> str:
    """Return a model summary or an explanation of its absence.

    Parameters
    ----------
    model : object or None
        Fitted model. Objects with a callable ``summary`` method use that
        method; ``None`` triggers lookup of a summary saved with the run.
    regression_type : str or None, optional
        Backend name included when the fitted object has no statsmodels-style
        summary.
    path : path-like or None, optional
        Results CSV, run directory, or parent directory. When ``model`` is
        ``None``, the function looks beside the selected results table for a
        summary written during fitting.
    reason : str, optional
        Known reason that no live model is available. When omitted, the
        explanation is limited to what can be inferred from ``path``.

    Returns
    -------
    str
        Saved or live summary text. If no summary is available, a message
        names the relevant backend, path, or read failure instead of returning
        an empty string.

    Notes
    -----
    A live statsmodels summary is returned from the fitted model rather than
    reconstructed. When the run summary is available on disk, it is included
    with the model summary.
    """
    if model is None:
        found = find_summary_file(path)
        if found:
            try:
                with open(found, "r", encoding="utf-8", errors="replace") as f:
                    text = f.read().strip()
            except OSError as error:
                return (f"No summary: {_why_no_model(path, reason)}, and the "
                        f"summary this run wrote could not be read "
                        f"({error}).")
            if text:
                return f"{SUMMARY_FROM_DISK.format(path=found)}\n\n{text}"
            return (f"No summary: {_why_no_model(path, reason)}, and the "
                    f"summary file this run wrote ({found}) is empty.")
        why = _why_no_model(path, reason)
        if path:
            names = _summary_filenames()
            looked = os.path.dirname(str(path)) if os.path.isfile(
                str(path)) else str(path)
            return (f"No summary: {why}, and this run wrote none — looked "
                    f"for {' or '.join(names) if names else 'a summary file'} "
                    f"in {looked}. Re-running with this build writes one.")
        return f"No summary: {why}."

    summary = getattr(model, "summary", None)
    if not callable(summary):
        named = f" ({regression_type})" if regression_type else ""
        return _with_spacr_summary(
            path,
            f"this backend{named} is not a statsmodels fit, so it has none. "
            f"Lasso, elastic net, and group lasso are ranked by bootstrap "
            f"selection frequency because they do not report frequentist "
            f"p-values. Ridge and hinge uncertainty is reported in the "
            f"Coefficients tab.",
            missing=True)

    try:
        text = str(summary())
    except Exception as error:                                   # noqa: BLE001
        return _with_spacr_summary(
            path,
            f"statsmodels could not render one for this fit "
            f"({type(error).__name__}: {error}).",
            missing=True)

    warning = _identifiability_warning(model)
    return _with_spacr_summary(path, f"{warning}\n{text}" if warning else text)


def _with_spacr_summary(path, statsmodels_text: str, *,
                        missing: bool = False) -> str:
    """Prepend the spaCR run summary to the statsmodels summary.

    The saved summary's embedded statsmodels tail is removed so the live text
    is not duplicated. When ``missing`` is true, the returned second section
    explains why no statsmodels summary is available.
    """
    spacr = _spacr_summary_text(path)
    if not spacr:
        # NOTHING AT ALL, and the tab says exactly that. "No summary" is the
        # sentinel every caller tests for; the qualified wording below is
        # only honest when there IS a spaCR summary above it.
        return f"No summary: {statsmodels_text}" if missing else statsmodels_text
    body = (f"No statsmodels summary: {statsmodels_text}" if missing
            else statsmodels_text)
    return f"{spacr}\n\n{VERBATIM_HEADING}\n{body}"


def _spacr_summary_text(path) -> str:
    """spaCR's own summary for a run, without its verbatim tail. "" if none.

    The file already carries the statsmodels text as its last section (see
    `write_run_summary`), so that tail is trimmed rather than printed twice --
    the live fit is the one on screen and is the one to keep.
    """
    found = find_summary_file(path)
    if not found:
        return ""
    try:
        with open(found, "r", encoding="utf-8", errors="replace") as handle:
            text = handle.read().strip()
    except OSError:
        return ""
    # A file that is ONLY statsmodels text is a run from before spaCR wrote
    # its own summary. Nothing to put in front.
    if not text or not text.startswith(SPACR_SUMMARY_HEADING):
        return ""
    cut = text.find(VERBATIM_HEADING)
    return text[:cut].rstrip() if cut > 0 else text


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

    ``perform_regression`` writes to ``results/<kind>[_n]/``. The folder name
    is therefore the only evidence available when a table does not record its
    backend directly.
    """
    if not path:
        return None
    try:
        from ...hits import NO_P_VALUE_TYPES
    except Exception:                  # pragma: no cover - hits unavailable
        return None
    parts = {
        re.sub(r"_\d+$", "", part.strip().lower())
        for part in str(path).replace("\\", "/").split("/")
    }
    for name in NO_P_VALUE_TYPES:
        if name in parts:
            return name
    return None


class RegressionResultsPanel(QWidget):
    """Volcano, table and diagnostics for one finished regression."""

    #: Emitted with the results CSV whenever a new one is loaded.
    loaded = Signal(str)
    #: An asynchronous load finished. ``True`` when the run is on screen.
    #: Separate from :attr:`loaded`, which carries a path and predates the
    #: worker: a caller needs to know SUCCESS, and it arrives later than the
    #: call that started it.
    load_finished = Signal(bool)

    #: What the run label says when the table on screen came from no run --
    #: a bare CSV, or a frame handed straight in. NOT blank: an empty label
    #: reads as "the run has no name" rather than "this table has no run".
    NO_RUN_NAMED = "No run"

    #: Emitted with a settings dict when the user asks, from the plot, for
    #: the same screen through a different model. The panel does not start
    #: the run itself -- it has no worker, no console and no Stop button --
    #: so whichever screen owns those does.
    refit_requested = Signal(object)

    def __init__(self, parent=None, external_volcano: bool = False):
        """Initialize the regression results panel.

        Parameters
        ----------
        parent : QWidget or None, optional
            Parent widget.
        external_volcano : bool, default=False
            Build and wire the interactive volcano plot without adding it to
            this panel's layout. Use this when a host places the volcano in a
            larger external view; selection and redraw behavior are unchanged.
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
        # WHICH RUN IS ON SCREEN, SAID WHERE THE RESULTS ARE. Instruction 157:
        # the loaded mark lived in the Runs tab and the coefficients lived
        # here, so the only way to notice the two had diverged was to compare
        # two views -- and the maintainer did exactly that ("even if the ols
        # model is marked as loaded i still see the mixed results"). A panel
        # that names its own run makes the disagreement visible in the view
        # that is wrong, rather than in the one that is right.
        self._run_label = QLabel(self.NO_RUN_NAMED)
        self._run_label.setObjectName("resultsRunName")
        header.addWidget(self._run_label)
        # THE CONTROL BELONGS ON THE FIGURE IT CHANGES. Asked for 2026-08-19:
        # "the color by in results can be removed as its alos in the right
        # click for the volcano graph the only place it is used i think" --
        # and it is: the volcano's own menu already offers "Colour by a
        # column…" and "Colour by localisation", and the volcano is the only
        # thing this combo redraws.
        #
        # HIDDEN, NOT DELETED, and the difference matters. `_redraw_volcano`
        # reads `currentData()`, `_restore_plot_state` writes it back, and a
        # saved run carries a `colour_by` key -- so the object stays and
        # keeps answering, while the header loses a duplicate. Deleting it
        # would have meant unpicking a saved-state field for a cosmetic win.
        self._colour_by_label = QLabel("colour by")
        self._colour_by_label.setVisible(False)
        header.addWidget(self._colour_by_label)
        self._colour_by = QComboBox()
        self._colour_by.setMinimumWidth(140)
        self._colour_by.currentIndexChanged.connect(self._redraw_volcano)
        self._colour_by.setVisible(False)
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
            # WHICH MODEL DREW THIS (189 B), directly under the graph and not
            # three tabs away. Two volcanoes can be correctly identical --
            # glm and quasi_binomial share every coefficient, because
            # dispersion moves the standard errors and not the point
            # estimates -- and without a label that is indistinguishable from
            # a bug. It was, and it is what "all the plots look the same no
            # matter which regression type i do" was about.
            self._model_line = QLabel("")
            self._model_line.setObjectName("Muted")
            self._model_line.setWordWrap(True)
            self._model_line.setVisible(False)
            split.addWidget(self._model_line)
            split.addWidget(self.table)
            split.setStretchFactor(0, 3)
            split.setStretchFactor(1, 2)
            # Floors, not preferences. Without them the panel's share of the
            # window is divided by the widgets' own size hints and BOTH end up
            # too short to read -- a volcano with no room for its axes and a
            # table showing its header and one row, which is what this looked
            # like on the real screen before the numbers were put in.
            self.volcano.setMinimumHeight(240)
            split.setSizes([340, 20, 220])
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
        # 168 D: "The Summary tab shows the verdict expanded and each
        # section collapsed, with the section headings as the outline."
        # A drop-in for the QPlainTextEdit that was here -- same
        # setPlainText/toPlainText -- so nothing that fills or reads it
        # changes, and text with no spaCR headings (the statsmodels summary)
        # is still shown whole.
        from .folding_summary import FoldingSummaryView

        self._summary = FoldingSummaryView()
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
        # WHERE THE ANNOTATED CELLS LAND (215). An independent check: the
        # annotation is made from sequencing fractions plus a phenotype
        # call, and this asks where the cell sits among the CONTROLS using
        # every measurement at once. Agreement between two routes that
        # different is worth more than either alone.
        #
        # Nothing is computed until the tab is opened and its button
        # pressed. A UMAP over a screen's cells is seconds of work, and a
        # tab that embedded on construction would charge that to every user
        # who never looks at it.
        try:
            from .annotation_umap_tab import AnnotationUmapTab

            self.annotation_umap = AnnotationUmapTab(self)
            self.tabs.addTab(self.annotation_umap, "Annotation check")
        except Exception:                                    # noqa: BLE001
            # A panel that cannot build its optional tab is still a panel.
            self.annotation_umap = None

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
        #
        # `GenePanel` and not `GeneTilePanel`: the tile alone says WHICH gene
        # a dot is and what THIS screen measured about it, both read out of
        # the frame already on screen. The panel adds the other half the
        # instruction asks for -- product, DeepTMHMM topology with each
        # segment's coordinates, hyperLOPIT compartment, the published
        # fitness screens and the stage expression -- out of
        # `spacr.annotation`, and it loads those five CSVs on a worker
        # thread. Cold that read is 360 ms, and 360 ms inside a mouse press
        # is a plot that reads as broken.
        from .gene_panel import GenePanel
        self.gene = GenePanel(frame_provider=lambda: self._frame)
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
            plot.key_selected.connect(self._select_from_a_plot)
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
        #: Per-run plot state, keyed by the run's source path.
        #:
        #: Only the visible run keeps a live volcano widget; other runs retain
        #: lightweight state so switching runs does not multiply redraw cost.
        self._plot_states: Dict[str, dict] = {}
        self._status = "No regression loaded."
        self._ranking = (None, None)
        #: Settings that produced the table on screen, when known.
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
        self._threshold_method = DEFAULT_THRESHOLD_METHOD
        self._threshold_multiplier = DEFAULT_THRESHOLD_MULTIPLIER
        #: Why a colour-by option is present but useless, or "".
        self._colour_by_note = ""
        # LOADING A RUN GOES OFF THE GUI THREAD (instruction 159). `load`
        # walked the folder, read the CSV and rebuilt every view inline, and
        # this file contained no JobRunner at all -- so a big table or a deep
        # folder stopped the window with no spinner and no cancel, which is
        # indistinguishable from a crash. It is the same defect 154 A fixed
        # for the merge, and this is the same machinery rather than a second
        # one.
        from ..job_runner import JobRunner

        self._loading = False
        self._load_jobs = JobRunner(self, threaded=True, app_key="results")
        self._load_jobs.job_failed.connect(self._on_load_job_failed)
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
        #: Fitted model behind the table when the current process produced it.
        self._model = None
        #: What went wrong drawing the diagnostics, verbatim, or "". Kept so
        #: an absent summary can name the error that caused it rather than
        #: telling the "opened from disk" story regardless.
        self._diagnostics_error = ""
        #: Whether the table came directly from a run in the current process.
        #: This distinguishes disk-loaded results from a live run that failed
        #: to return a fitted model.
        self._from_live_run = False
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

        # HOVER HELP BELONGS ON THE SETTING'S NAME, NOT ON THE CONTROL
        # (instruction 113, restated across every module 2026-08-19: "the
        # tooltip should only be visable when hovering the mouse over the
        # setting name text, and not when hovering over the field, checkbox,
        # or whatever the setting controlls"). One post-pass rather than a
        # convention every hand-built row has to remember -- which is what
        # `tests/test_tooltips_are_on_the_setting_not_the_field.py` exists to
        # catch, and did catch this screen.
        from ..screens.settings_model import retarget_field_tooltips
        retarget_field_tooltips(self)

    # -------------------------------------------------------------- re-fitting

    def set_run_settings(self, settings) -> None:
        """Remember the settings that produced the table now on screen.

        Called by the screen when a run finishes, because the run's own
        settings are better than anything read back off disk: the saved copy
        under ``settings/`` is overwritten by every later run of the same
        screen, so on a second run it describes the wrong one.
        """
        self._run_settings = dict(settings) if settings else None
        # ONLY THE LIVE PATH CALLS THIS, so it is also the answer to "did
        # this table come from a run in this session or off the disk" -- a
        # question the Summary tab used to answer with a guess.
        self._from_live_run = True
        # The reason for an absent summary has just changed, so the sentence
        # on the tab has to. Only when there is no model to render: a summary
        # already on screen is the run's own and is not rewritten.
        if self._model is None:
            self.set_summary(None)

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

    #: Message shown when diagnostics have no fitted model. It distinguishes
    #: disk-loaded coefficient tables from a live fit and names the available
    #: actions.
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

    #: Live-tile key -> the attribute holding the tab page that shows it.
    #: THE GRID'S KEYS, not the tab labels: the grid photographs panels by
    #: key and a label is a display string somebody will translate or
    #: shorten. Where the page is a splitter carrying the plot plus a table
    #: (`Volcano`, `Guide support`) or a plot plus its spread
    #: (`Scale-location`), the SPLITTER is named -- `setCurrentWidget` only
    #: accepts a widget the tab bar itself owns, so naming the inner plot
    #: would be a silent no-op, which is the whole failure this map exists
    #: to end.
    _PANEL_TABS = {
        "regression": "_volcano_tab",
        "effect_rank": "effect_rank",
        "effect_distribution": "effect_distribution",
        "p_values": "p_values",
        "qq": "qq",
        "controls": "controls",
        "agreement": "_support_tab",
        "residuals": "residuals",
        "scale_location": "_scale_location_tab",
        "influence": "influence",
        "gene": "gene",
        "annotation_check": "annotation_umap",
    }

    def show_panel(self, key: str) -> bool:
        """Raise the tab that holds the live panel named ``key``.

        :param key: a key from :meth:`FigureGridView.set_live_tiles`.
        :returns: ``True`` when the corresponding tab exists and was raised;
            otherwise ``False``. The available tabs depend on the fitted
            model and whether the volcano is displayed outside this panel.
        """
        attribute = self._PANEL_TABS.get(str(key))
        if attribute is None:
            return False
        page = getattr(self, attribute, None)
        if page is None:
            return False
        try:
            # -1 when the widget exists but was never added, which is the
            # ordinary state of the volcano and gene tabs on a screen that
            # shows the volcano outside the panel.
            if self.tabs.indexOf(page) < 0:
                return False
            self.tabs.setCurrentWidget(page)
        except (RuntimeError, TypeError):             # pragma: no cover
            return False
        return True

    def clear_diagnostics(self, reason: str = "") -> None:
        """Empty the three well-level tabs and SAY why they are empty.

        :param reason: the specific reason, when there is one. The default is
            :data:`NO_MODEL_MESSAGE` -- "no run in this session has fitted
            anything yet", which is the ordinary case and still an answer.

        THIS DROPS THE MODEL, and that is what it is for: it is called when a
        NEW TABLE arrives, and the previous fit has nothing to say about it.
        A failure to DRAW the diagnostics is a different event and goes
        through :meth:`_clear_diagnostic_views`, which leaves the fit alone --
        a failure in the view must not destroy the thing being viewed.
        """
        self._model = None
        self._diagnostics_error = ""
        self._clear_diagnostic_views(reason)

    def _clear_diagnostic_views(self, reason: str = "") -> None:
        """Empty the three tabs and say why, WITHOUT discarding the fit."""
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
        if model is None:
            # THE MODEL THIS PANEL ALREADY HAS. A caller handing over a
            # payload whose `model` key is empty is saying "the run returned
            # none", not "throw away the one you were given" -- and the fit
            # kept by `set_diagnostics` is the same fit this table came from.
            model = self._model
        text = summary_text(model, regression_type, path=self._path,
                            reason=self._no_model_reason())
        self._summary.setPlainText(text)
        self._name_the_model(model, regression_type)
        return not text.startswith("No summary")

    def _name_the_model(self, model, regression_type=None) -> str:
        """Update the model identity shown below the volcano plot.

        The label and run summary share
        (:func:`spacr.regression_summary.model_identity_line`), so the caption
        and Summary tab describe the same fitted model.

        Returns
        -------
        str
            Rendered identity text, or an empty string when unavailable.
        """
        label = getattr(self, "_model_line", None)
        if label is None:
            return ""
        try:
            from ...regression_summary import model_identity_line

            said = model_identity_line(regression_type,
                                       self._run_settings or {}, model)
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not name the model", exc_info=True)
            said = ""
        label.setText(said)
        # HIDDEN WHEN IT HAS NOTHING TO SAY. An empty muted strip under the
        # graph is a row of pixels that means nothing, and a caption that
        # guessed would be worse than no caption.
        label.setVisible(bool(said))
        return said

    #: Reason reported when a live run returns no fitted model.
    NO_MODEL_FROM_THIS_RUN = (
        "this run came back without a fitted model, so there is none to "
        "summarise")

    def _no_model_reason(self) -> str:
        """Return a verified reason that this panel has no fitted model.

        Return an empty string when :func:`summary_text` should infer the
        reason from the available path instead.
        """
        if self._model is not None:
            return ""
        if self._diagnostics_error:
            # Defensive rather than expected: `set_diagnostics` now stores the
            # fit BEFORE it tries to draw anything, so a diagnostics failure
            # no longer takes the model with it. If some other path ever
            # loses one, the tab says which error did it rather than telling
            # the disk story again.
            return (f"the diagnostics failed and the fit was not kept "
                    f"({self._diagnostics_error})")
        if self._from_live_run:
            return self.NO_MODEL_FROM_THIS_RUN
        return ""

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

        # THE FIT IS STORED BEFORE ANYTHING IS DRAWN FROM IT.
        #
        # It used to be assigned at the bottom, after the context had been
        # built, so every early return below threw the model away -- and the
        # Summary tab then explained its absence with "this panel was opened
        # from a results table on disk", which for a live run whose
        # diagnostics could not be built is simply untrue. A failure in the
        # VIEW must not destroy the thing being viewed: the model is the fit,
        # the diagnostics are one way of looking at it, and the summary is
        # another that works perfectly well when this one does not.
        self._model = model
        self._diagnostics_error = ""
        try:
            from ...regression_qc import (PanelUnavailable, context_from_model,
                                          cooks_distance)
        except Exception as error:                               # noqa: BLE001
            self._diagnostics_error = f"could not load the diagnostics module: {error}"
            self._clear_diagnostic_views(
                f"Could not load the diagnostics module: {error}")
            return False
        try:
            ctx = context_from_model(model, coef_df=self._frame,
                                     regression_type=regression_type)
        except PanelUnavailable as error:
            self._diagnostics_error = str(error)
            self._clear_diagnostic_views(str(error))
            return False
        except Exception as error:                               # noqa: BLE001
            self._diagnostics_error = (
                f"{type(error).__name__}: {error}")
            self._clear_diagnostic_views(
                f"The diagnostics could not be computed for this fit "
                f"({type(error).__name__}: {error}).")
            return False

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
        """Return the complete coefficient table for the displayed run.

        The table is unaffected by the gene/guide view filter. Use
        :meth:`filtered_frame` for the rows currently shown by the result tabs.
        """
        return self._frame

    def run_folder(self) -> str:
        """Return the folder for the regression run shown by this panel.

        Live results may store a directory while results loaded from disk may
        store a table path; both resolve to the containing run folder. Return
        an empty string for an unassociated CSV or an in-memory frame.
        """
        return self._folder_of(self._path)

    @staticmethod
    def _folder_of(source) -> str:
        """The run folder behind ``source``, whatever shape it arrived in.

        A run opened off disk gives the CSV; a live run gives the directory
        ``perform_regression`` wrote. ONE ANSWER FOR BOTH, because they are
        the same run -- and while they were two answers they were also two
        keys, so a run looked at live and then returned to from the Runs tab
        was two runs to everything keyed on this.
        """
        path = str(source or "").strip()
        if not path:
            return ""
        path = os.path.abspath(os.path.expanduser(path))
        if os.path.isdir(path):
            return path
        if os.path.isfile(path):
            return os.path.dirname(path)
        # A PATH THAT NO LONGER EXISTS IS STILL EVIDENCE. A run folder that
        # was deleted (146) or moved should name the run it named yesterday
        # rather than reading as "this table came from nowhere" -- and the
        # state keyed on it has to be reachable to be forgotten. The two
        # shapes are told apart by the only thing left to read: a results
        # table is `results.csv` and a run folder is `ols_3`, so a suffix
        # means a file and no suffix means the folder.
        if os.path.splitext(os.path.basename(path))[1]:
            return os.path.dirname(path)
        return path

    def run_name(self) -> str:
        """The run on screen, as the name the Runs tab calls it.

        ``results/<kind>_<n>`` is the folder a run writes, so the basename is
        ``ols_3`` -- which is what the Runs table shows, what the figure grid
        heads its section with and what the montage names. One vocabulary
        prevents three views from calling the same run three different things.
        """
        folder = self.run_folder()
        return os.path.basename(folder.rstrip(os.sep)) if folder else ""

    def _name_the_run(self) -> None:
        """Put the run's name in the header, beside its table.

        LABELLED, not bare. `ols_4` on its own between a status sentence and
        a "colour by" menu is a word with no job; "Run: ols_4" is the answer
        to the question the user is actually asking of this header.
        """
        name = self.run_name()
        self._run_label.setText(f"Run: {name}" if name else self.NO_RUN_NAMED)
        self._run_label.setToolTip(
            self.run_folder() if name else
            "The table on screen was not read from a run folder.")

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

    def closeEvent(self, event):                 # noqa: N802 - Qt's spelling
        """Stop the results loader before closing the widget.

        Qt requires a running ``QThread`` to outlive every object that owns it.
        The bounded shutdown cancels the load and waits briefly; a slower
        worker is transferred to :func:`spacr.qt.bridge.drain_thread` so that
        closing the screen neither terminates the worker nor blocks on it.

        :param event: Qt close event passed to the parent implementation.
        """
        try:
            self._load_jobs.shutdown()
        except Exception:                                        # noqa: BLE001
            LOG.debug("could not stop the results loader", exc_info=True)
        super().closeEvent(event)

    def is_loading(self) -> bool:
        """Whether a run is being read right now."""
        return bool(self._loading)

    def start_load(self, path) -> bool:
        """Start loading a regression run outside the GUI thread.

        File discovery and CSV reads run in a worker. Only the resulting data
        is returned to the GUI thread.

        :param path: Regression-results directory to load.
        :returns: whether a load was started. ``False`` when one already is --
            a second click must not read the same folder twice.
        """
        if self._loading:
            return False
        if not path:
            self.say("Nothing was handed to the results panel, so there is "
                     "no folder to search. Use “Load results…”.")
            return False
        self._loading = True
        self.say(f"Reading the run in {os.path.basename(str(path)) or path}…")
        started = self._load_jobs.submit(
            lambda: self._read_run(path),
            self._finish_load)
        if not started:                      # pragma: no cover - JobRunner
            self._loading = False            # always returns True today
        return bool(started)

    @staticmethod
    def _read_run(path):
        """THE WORKER HALF. Touches no widget; returns what the GUI needs.

        Returns a dict rather than raising, because a JobRunner job that
        raises loses the detail on the way back across the thread boundary --
        the same reason `_merge_worker` returns its outcome.
        """
        import pandas as pd

        searched = os.path.abspath(os.path.expanduser(os.fspath(path)))
        if not os.path.exists(searched):
            return {"error": f"{searched} does not exist, so there is "
                             f"nothing to load from it."}
        tables = find_results_tables(searched)
        if not tables:
            return {"error": (
                f"Searched {searched} and found none of "
                f"{', '.join(RESULT_FILENAMES)} in it or in any folder up to "
                f"{MAX_SEARCH_DEPTH} deep.")}
        found = tables[0]
        try:
            frame = pd.read_csv(found)
        except Exception as error:  # noqa: BLE001 - report, do not raise
            return {"error": f"Could not read {found}: {error}"}
        return {"frame": frame, "found": found, "searched": searched,
                "tables": tables}

    def _finish_load(self, outcome) -> bool:
        """THE GUI HALF: everything that touches a widget."""
        self._loading = False
        ok = False
        if not isinstance(outcome, dict):
            self.say("The run loader came back with nothing, which is a bug "
                     "in the loader rather than in the run.")
        elif outcome.get("error"):
            self.say(str(outcome["error"]))
        else:
            ok = self._apply_loaded_run(
                outcome["frame"], outcome["found"], outcome["searched"],
                outcome["tables"])
        # ALWAYS, on both endings. A caller waiting on this must not be left
        # waiting by a failure -- that is a spinner nothing clears.
        self.load_finished.emit(bool(ok))
        return ok

    def _on_load_job_failed(self, message: str) -> None:
        self._loading = False
        self.say(f"The run could not be read: {message}")
        self.load_finished.emit(False)

    def load(self, path) -> bool:
        """Load a results CSV, a run folder, or a parent of one.

        THE SYNCHRONOUS ENTRY POINT, kept for tests and headless callers.
        The GUI goes through :meth:`start_load`; both end in
        :meth:`_apply_loaded_run`, so the two cannot drift -- which is the
        rule `start_merge` and `merge` already follow.

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
        return self._apply_loaded_run(frame, found, searched, tables)

    def _apply_loaded_run(self, frame, found, searched, tables) -> bool:
        """Put a read run on screen. THE ONE ENDING BOTH LOAD PATHS SHARE.

        Split out so `load` (synchronous) and `start_load` (on a worker)
        cannot drift -- the rule `merge` and `start_merge` already follow, and
        the reason the run loader was still on the GUI thread long after the
        merge had been moved off it is that nobody had made them one path.
        """
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

    def _say_if_no_p_values(self, frame) -> None:
        """Explain a table whose coefficients carry no significance.

        Distinguishes the two ways it happens, because they call for
        different things from the reader: a mixed fit's guide rows are BLUPs
        and never had a p value, and that is expected; anything else with no
        finite p value is a fit that failed to produce one, and that is not.
        """
        import pandas as pd

        if "p_value" not in frame.columns:
            return
        values = pd.to_numeric(frame["p_value"], errors="coerce")
        if values.notna().any():
            return
        kinds = set()
        if "term_type" in frame.columns:
            kinds = {str(k) for k in frame["term_type"].dropna().unique()}
        if kinds and kinds <= {"random_effect_blup"}:
            self.say(
                f"These {len(frame):,} row(s) are BLUPs, not estimates. A "
                f"mixed model makes the guide a RANDOM effect, so each guide "
                f"gets a shrunken prediction and no p value -- which is why "
                f"the volcano is empty, its vertical axis being the p value. "
                f"The coefficients are in the table beside it. For per-guide "
                f"significance, fit at guide level with a fixed-effect "
                f"model, or use inference='nonparametric', which tests each "
                f"guide on its own.")
        else:
            self.say(
                f"None of these {len(frame):,} coefficient(s) carries a "
                f"p value, so nothing can be plotted against significance. "
                f"That is a fit that did not produce one rather than a fit "
                f"with nothing to say -- the coefficients themselves are in "
                f"the table.")

    def set_frame(self, frame, source: str = "") -> bool:
        """Show an already-loaded coefficient table."""
        import pandas as pd

        if frame is None or not len(frame):
            self.say("The results table is empty: it has columns but no "
                      "rows, so the fit produced no coefficients.")
            return False
        # ROWS BUT NO P VALUES IS NOT AN EMPTY TABLE, and saying nothing
        # about it produced the report "with guides i see nothing in the
        # graph" (2026-08-21) against a run that had worked perfectly.
        #
        # A MIXED MODEL MAKES THE GUIDE A RANDOM EFFECT. Each guide gets a
        # shrunken BLUP -- a prediction -- and a BLUP has no p value, so a
        # volcano, whose vertical axis IS the p value, has nothing to draw.
        # The run already says this in the console; the panel the user is
        # looking at did not.
        self._say_if_no_p_values(frame)
        # THE OUTGOING RUN KEEPS WHAT THE USER BUILT ON IT. Saved BEFORE
        # anything is replaced, because every line below this one resets a
        # piece of it -- and saved against the OLD path, which is the key the
        # user will come back through.
        self._remember_plot_state()
        self._frame = frame
        self._path = source
        # NAMED THE MOMENT THE TABLE CHANGES, not at the end: every early
        # return below this line would otherwise leave the header naming the
        # previous run over the new one's coefficients.
        self._name_the_run()
        self._ranking = self._rank_by(frame, source)

        # A NEW TABLE IS A NEW FIT, so the old fit's residuals have to go. The
        # caller that HAS a model calls `set_diagnostics` immediately after
        # this; the caller that loaded a CSV has none, and leaving the last
        # run's residuals on the tabs would describe a fit the user is no
        # longer looking at -- with nothing on screen saying so.
        self._from_live_run = False
        self.clear_diagnostics()

        # AND THE SUMMARY IS THE SAME KIND OF STALENESS. It was left
        # untouched here, so a table opened from disk sat under the PREVIOUS
        # run's statsmodels output with nothing saying whose it was. Refreshed
        # from the new table's own folder: `perform_regression` writes the
        # summary beside `results.csv`, so a run re-opened from disk shows the
        # text it showed while it was running -- byte for byte, because it is
        # the same bytes. A live run overrides this a moment later with the
        # model itself, which is better still.
        self.set_summary(None)

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
        # AND THE TABLE'S OWN ROW, because the table is what re-establishes
        # a selection. Clearing `_selected_key` and the rings was not enough:
        # rebuilding the table leaves a row highlighted, that row re-emits
        # `key_selected`, and the panel comes back from the reset holding a
        # key -- on the new run, but one nobody chose, and a mark nobody
        # chose is exactly what this reset exists to prevent. Blocked, so the
        # clear itself does not emit a THIRD time.
        try:
            blocked = self.table.table.blockSignals(True)
            self.table.table.clearSelection()
            self.table.table.setCurrentCell(-1, -1)
            self.table.table.blockSignals(blocked)
        except (RuntimeError, AttributeError):   # pragma: no cover - no table
            pass
        for plot in self._keyed_plots():
            plot.clear_highlight()
        for histogram in (self.p_values, self.effect_distribution):
            histogram.clear_highlight()
        # AND THE GENE TILE, for the same reason and it is the worst offender:
        # a plot re-rings a point, but the tile keeps a whole paragraph about
        # a gene from the previous screen, with this screen's effect nowhere
        # near it and nothing on it saying which run it came from.
        self.gene.clear()
        # Warm the annotation for THIS screen's genes, off the GUI thread.
        # One join covers the whole table -- 400 genes cost the same 21 ms as
        # one -- so every click afterwards is a dictionary lookup.
        self.gene.warm_for(frame)

        # THE COMPARTMENT MENU IS BUILT FROM THE TABLE, so it has to be
        # rebuilt when the table changes. Built once in __init__ it is built
        # from no frame at all, which is an empty submenu that never appears
        # -- and a new screen would otherwise be offered the last one's
        # compartments.
        self._compartment = None
        # AND SO DO THE EFFECT CUT AND THE AXIS WINDOW, which they did not.
        # Measured on the real panel: after typing an x-range of (-1.5, 1.5)
        # and a 2-spread cut on run A, opening run B drew run B inside run
        # A's window with run A's cut -- a picture nobody chose, with nothing
        # on screen saying where it came from. The other four resets in this
        # block exist for exactly that reason; these two were missing.
        #
        # A run RETURNED TO gets its own back at the end of this method
        # (`_restore_plot_state`), so the reset costs nothing a user built.
        self._threshold_method = DEFAULT_THRESHOLD_METHOD
        self._threshold_multiplier = DEFAULT_THRESHOLD_MULTIPLIER
        self._p_value_kind = "raw"
        try:
            self.volcano.auto_range_axes()
        except Exception:                                        # noqa: BLE001
            LOG.debug("could not release the previous run's axis limits",
                      exc_info=True)
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
        self._mark_the_level_on_the_plots()
        # BOTH FITS ARE ANNOUNCED ON LOAD. A run at level='both' writes two
        # tables and the panel opens on one; until this line nothing said the
        # other existed, and a user who ran glm reported "it only runs once"
        # about a run that had written both.
        note = self.both_levels_note()
        if note:
            self.say(f"{self._status} {note}")
        if self._colour_by_note:
            # SAID, not swallowed. A colouring the user expected and cannot
            # find is a bug report; the same colouring listed with the reason
            # it is useless is an answer.
            self.say(f"{self._status} Colouring: {self._colour_by_note}.")
        # AND A RUN COME BACK TO GETS ITS PLOT BACK. Last, so it wins over
        # every default the lines above chose from the table -- which is the
        # whole point: those defaults are right the FIRST time a run is
        # opened and wrong every time after, because by then the user has
        # said what they want to look at.
        self._restore_plot_state(source)
        self.loaded.emit(source or "")
        return True

    # ----------------------------------------------- a run owns its plot

    def plot_state(self) -> dict:
        """What the user has built on the plot, as data.

        The design: "every regression run should have its own
        interactive volcano plot." A run does not have a volcano today -- the
        screen has one and a run borrows it -- so opening run B destroyed the
        level, the colouring, the axis limits, the effect cut and the
        selection a user had chosen on run A.

        EVERYTHING HERE BELONGS TO THE RUN AND NOT TO THE WIDGET. The colour
        column is stored by NAME rather than by index, because the combo box
        is rebuilt from each table's own columns and index 3 is a different
        column in the next run.
        """
        # THE PLOT'S OWN ANSWER, through its public method (116's last
        # private coupling). `getattr(volcano, "_pinned")` was reaching into
        # another module's attribute for a question that module can answer.
        reader = getattr(self.volcano, "pinned_limits", None)
        pinned = reader() if callable(reader) else (
            getattr(self.volcano, "_pinned", None) or {})
        return {
            "level": self._level,
            "colour_by": self._colour_by.currentData(),
            "baseline": tuple(self._baseline),
            "compartment": self._compartment,
            "p_value_kind": self._p_value_kind,
            "threshold_method": self._threshold_method,
            "threshold_multiplier": self._threshold_multiplier,
            "selected_key": self._selected_key,
            # ONLY WHAT THE USER PINNED. Storing the view range would freeze
            # an auto-ranged plot at whatever it happened to show, so a run
            # returned to would stop following its own data.
            "x_limits": pinned.get("x"),
            "y_limits": pinned.get("y"),
        }

    def apply_plot_state(self, state) -> bool:
        """Put a saved :meth:`plot_state` back. Returns whether it applied.

        ONE REDRAW, not one per setting. Every public setter here redraws,
        so restoring nine of them through nine setters would draw the panel
        nine times on every run switch -- and the intermediate frames are of
        combinations the user never chose.
        """
        if not isinstance(state, dict) or self._frame is None:
            return False
        if "level" in state:
            self._level = state["level"]
        if "baseline" in state:
            kind, name = tuple(state["baseline"])
            self._baseline = (kind, name)
        for field in ("compartment", "p_value_kind", "threshold_method",
                      "threshold_multiplier"):
            if field in state:
                setattr(self, f"_{field}", state[field])
        colour = state.get("colour_by")
        if colour is not None:
            index = self._colour_by.findData(colour)
            if index >= 0:
                # Blocked: the redraw below covers it, and letting the combo
                # fire here would draw the panel against a level that has
                # not been re-offered yet.
                blocked = self._colour_by.blockSignals(True)
                self._colour_by.setCurrentIndex(index)
                self._colour_by.blockSignals(blocked)
        self._offer_levels()
        self._offer_p_values()
        self._offer_thresholds()
        self._offer_compartments()
        self._offer_baselines()
        self.refresh_views()
        self._mark_the_level_on_the_plots()
        limits = (state.get("x_limits"), state.get("y_limits"))
        if any(limit is not None for limit in limits):
            try:
                self.volcano.set_axis_limits(x=limits[0], y=limits[1])
            except Exception:                                    # noqa: BLE001
                LOG.debug("could not restore the run's axis limits",
                          exc_info=True)
        key = state.get("selected_key")
        if key:
            # THE ROW MAY NOT BE THERE. A saved selection is a feature NAME,
            # and the level restored above may filter it out -- a gene picked
            # at level=None is not in the guide table. Missing is not an
            # error; it is a row the user cannot currently see.
            try:
                self._select_key(str(key))
            except Exception:                                    # noqa: BLE001
                LOG.debug("could not restore the run's selection",
                          exc_info=True)
        return True

    def remembered_runs(self) -> tuple:
        """Return run paths whose plot state has been stored.

        The currently displayed run is omitted because its state remains
        live in the widgets until the panel switches away from it.
        """
        return tuple(self._plot_states)

    # -- State contributed to a saved workspace ------------------------------

    def workspace_state(self) -> dict:
        """Return saved view state for every run opened in this panel.

        The currently displayed run is recorded before the stored run states
        are copied into the returned workspace mapping.
        """
        self._remember_plot_state()
        return {
            "path": str(self._path or ""),
            "level": self._level,
            "runs": {key: dict(state) for key, state in self._plot_states.items()},
        }

    def apply_workspace_state(self, state) -> bool:
        """Put the remembered views back, and reopen the run that was open.

        Returns whether anything was put back. THE STORE IS MERGED, NOT
        REPLACED: restoring a workspace into a session that already has runs
        open must not silently drop the views the user built since. A run in
        both is taken from the document, which is what the user asked to
        restore.
        """
        if not isinstance(state, dict):
            return False
        runs = state.get("runs")
        if isinstance(runs, dict):
            for key, remembered in runs.items():
                if isinstance(remembered, dict):
                    self._plot_states[str(key)] = dict(remembered)
        path = str(state.get("path") or "")
        # LOADED, not just assigned. `_path` without the table behind it is a
        # panel claiming to show a run it has not read, and every diagnostic
        # tab would draw the previous run's numbers under the new run's name.
        if path and os.path.exists(path):
            return bool(self.load(path))
        return bool(runs)

    def forget_plot_state(self, source) -> bool:
        """Drop one run's remembered plot. The design deletes a run.

        :param source: the run's folder, or any path inside it -- the CSV a
            caller happens to be holding answers the same as the folder.
        :returns: whether there was anything to drop.

        A deleted run must take its state with it, or a later run written
        into the same folder inherits the deleted one's level and colouring
        and there is nothing on screen saying where they came from.
        """
        return self._plot_states.pop(self._plot_state_key(source),
                                     None) is not None

    def forget_run(self, source) -> bool:
        """A run was deleted: drop its view, and clear the panel if it is IT.

        THE ORDER IS THE WHOLE POINT, and the failure sequence is explicit
        before 146 existed: "deleting the run CURRENTLY ON SCREEN must also
        clear the panel, or leaving that run re-saves the state that was just
        forgotten." `set_frame` calls `_remember_plot_state` on the way out of
        a run, so forgetting first and clearing second files the deleted run
        again, under the same key, with the state it was just relieved of.

        So the panel lets go of the run FIRST -- `_path` and `_frame` cleared
        by hand rather than through `set_frame`, which is a route INTO a run
        and re-saves on the way -- and forgets afterwards.

        :param source: the run's folder, or any path inside it.
        :returns: whether anything was dropped: a state, the table, or both.
        """
        key = self._plot_state_key(source)
        was_showing = bool(key) and self._plot_state_key(self._path) == key
        if was_showing:
            self._path = ""
            self._frame = None
            self.clear_diagnostics(
                "the run this was describing has been deleted")
            self.say("The run this was showing has been deleted. "
                     "Pick another in the Runs tab.")
        return bool(self.forget_plot_state(source)) or was_showing

    @classmethod
    def _plot_state_key(cls, source) -> str:
        """Return the key used to store a results table's plot state.

        A live run path and its canonical ``results.csv`` file share the run
        folder as their key. Alternative tables in the same folder include
        their file name, which keeps gene- and guide-level plot selections
        independent. Sources without a resolvable folder use their string
        representation.
        """
        folder = cls._folder_of(source)
        if not folder:
            return str(source or "")
        name = os.path.basename(str(source or ""))
        if name and name != os.path.basename(folder) and name != _CANONICAL_TABLE:
            return f"{folder}::{name}"
        return folder

    def _remember_plot_state(self) -> None:
        """Save the run on screen, if it has a path to be saved under."""
        key = self._plot_state_key(self._path)
        if key and self._frame is not None:
            self._plot_states[key] = self.plot_state()

    def _restore_plot_state(self, source: str) -> bool:
        """Put back what this run was left looking like, if anything."""
        state = self._plot_states.get(self._plot_state_key(source))
        return self.apply_plot_state(state) if state else False

    def _mark_the_level_on_the_plots(self) -> None:
        """Put the level sentence where the dots are, not only in a header.

        A status line at the top of a panel is read once, on load. The
        question "am I looking at guides or genes" is asked every time the
        user comes back to the tab, and the answer belongs beside the marks
        it describes.
        """
        # THE VOLCANO ONLY, and deliberately. The diagnostics carry the
        # numbers they exist for -- the inflation factor, the control
        # medians, how many genes rest on one guide -- and writing the level
        # sentence over those would trade a panel's whole content for
        # something the header already says. The volcano is the plot the
        # report was about and the one a user looks at first.
        #
        # THROUGH `_offer_levels`, NOT `set_status_note`. The click slot is
        # rewritten by every click, so the sentence was gone the first time
        # the plot was used; `offer_levels`' own note slot is durable. One
        # call keeps the control, its counts and its sentence in step, which
        # is why this is a delegation and not a second copy of the sentence.
        self._offer_levels()

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

    def both_levels_note(self) -> str:
        """One sentence naming the level shown and the one that is not.

        THE RUN FITS TWICE AND THE PANEL SHOWS ONE. The design splits
        `level='both'` into a guide fit and a gene fit -- two tables, two
        multiple-testing families -- and the panel opens on guides so a gene
        is not drawn once per guide. Both of those are right.

        Previously, nothing said so. In a representative GLM run, both fits
        completed -- ``results_grna.csv`` had 15 rows and ``results_gene.csv``
        had 5 -- while half of
        it was invisible with nothing on screen naming the other half, so "it
        only runs once" is the honest reading from the user's side.

        Empty when there is nothing to say: one level in the table, or no
        filter on. A note that fires every time is a note nobody reads.
        """
        counts = self.level_counts()
        total = counts.get(None, 0)
        if not self._level or not total:
            return ""
        shown = counts.get(self._level, 0)
        other = "gene" if self._level == "grna" else "grna"
        if not counts.get(other, 0):
            return ""
        return (f"{self.LEVEL_NAMES.get(self._level, 'everything')}: "
                f"{shown} of {total} coefficients. The "
                f"{'gene' if other == 'gene' else 'guide'} fit is in this run "
                f"too — switch with Level.")

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
        """Add gene, guide, and combined level choices to the volcano plot.

        The level summary uses the plot's persistent option-note slot so point
        selections cannot replace the explanation of hidden coefficients.
        """
        counts = self.level_counts()
        self.volcano.offer_levels(
            [(f"{label} ({counts.get(key, 0)})",
              (lambda k=key: self.set_level(k)), key == self._level)
             for key, label in self.LEVELS],
            note=self.both_levels_note())

    def build_level_menu(self):
        """Build the coefficient-level menu with row counts.

        Returns
        -------
        PySide6.QtWidgets.QMenu
            Menu entries for genes, guides, and both levels. Each label reports
            how many rows the corresponding view contains.
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
        """Open the coefficient-level menu for the results table.

        The table and volcano menus update the same level state so every
        downstream table and plot uses the same gene/guide filter.
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
        self._mark_the_level_on_the_plots()
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
        """Offer raw and adjusted p-values when correction was performed.

        A copied ``q_value`` from ``multiple_testing_method='none'`` does not
        count as an adjusted value and does not enable the choice.
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
        """Add effect-size threshold controls to the volcano plot menu."""
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
        from ...localisation import ALL as ALL_COMPARTMENTS

        options = [("none (up / down)", lambda: self.set_compartment(None),
                    self._compartment is None)]
        # ALL, asked for on 2026-08-20. Second, so the one-at-a-time reading
        # the house style prefers is still what a user lands on first.
        options.append(("all localisations",
                        lambda: self.set_compartment(ALL_COMPARTMENTS),
                        self._compartment == ALL_COMPARTMENTS))
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
        from ...localisation import ALL as ALL_COMPARTMENTS

        if name == ALL_COMPARTMENTS:
            # ITS OWN SENTENCE. `mask` takes ONE compartment, so the branch
            # below would hand it the sentinel and report "0 annotated
            # \x00all-localisations" -- a number about nothing, printed
            # confidently.
            from ...localisation import of as compartment_of

            annotated = 0
            total = 0
            if self._frame is not None:
                names = compartment_of(self._frame)
                total = len(self._frame)
                annotated = int((names.astype(str) != "").sum()) if len(names) else 0
            self.say(
                f"{annotated} of {total} coefficients carry a TAGM/LOPIT "
                f"localisation and each is coloured by it; the rest are one "
                f"colour marked 'elsewhere'."
                if annotated else
                "No coefficient in this screen carries a TAGM/LOPIT "
                "localisation, so there is nothing to colour by.")
        elif name:
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

    #: Message shown in effect tabs before a coefficient table is available.
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
        """Fill the effect-rank and effect-distribution tabs.

        Parameters
        ----------
        frame : pandas.DataFrame
            Coefficient table at the selected gene or guide level.
        kind : str or None, optional
            Ranking type returned by :meth:`ranking`. Only ``"p-value"``
            permits significance coloring; penalized rankings must not reuse
            inferential p-values computed under a different model.

        Notes
        -----
        Unsupported plots display their reason in the tab. Nuisance terms are
        excluded from both views because they are not part of the multiple-
        testing family represented by the reported q-values.
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

    def _select_from_a_plot(self, key: str) -> None:
        """Select a clicked coefficient, changing level when required.

        Gene-only plots can be activated while the panel is filtered to
        guides. In that case the level changes first and selection is applied
        on the next event-loop turn, after the view rebuild and signal dispatch
        finish. Clicks whose row is already visible do not change the filter.
        """
        if self._reachable(key):
            self.table.select_key(key)
            return
        from ...hits import guide_of
        from PySide6.QtCore import QTimer

        self.set_level("grna" if guide_of(str(key)) else "gene")
        # A single shot rather than a direct call: see the docstring. The
        # bound method keeps the panel alive for the one turn it needs.
        QTimer.singleShot(0, lambda k=str(key): self.table.select_key(k))
        self.say(f"Showing {self.LEVEL_NAMES.get(self._level, 'everything')} "
                 f"so the point you clicked has a row.")

    def _reachable(self, key) -> bool:
        """Whether ``key`` has a row in the table at the current level.

        True when there is no filter, no frame or no `feature` column -- in
        each of those the table is not hiding anything and an unfound key is
        somebody else's problem to report.
        """
        if not key or not self._level or self._frame is None:
            return True
        if "feature" not in getattr(self._frame, "columns", ()):
            return True
        mask = self._level_mask(self._frame)
        if mask is None:
            return True
        features = self._frame["feature"].astype(str)
        # A KEY THE TABLE DOES NOT HOLD AT ANY LEVEL IS REACHABLE, which
        # reads oddly and is right: moving the filter cannot produce a row
        # that does not exist, so the only thing it would achieve is
        # rearranging the panel around a click nobody can honour. The plot
        # that emitted it reports the miss; that is its job, not this one's.
        if str(key) not in set(features):
            return True
        return str(key) in set(features[mask])

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

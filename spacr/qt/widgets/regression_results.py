"""Everything a finished regression produced, in one place and fast.

The module used to answer a run with a stack of matplotlib pictures, the last
of which -- the volcano -- cost ~115 ms per redraw and made the whole window
lag. Worse, the numbers behind it were only in a CSV, so "is EAF1 a hit" meant
leaving the application.

This panel is what a finished run opens into:

    Volcano       every dot clickable, drawn by Qt, 3.6 ms
    Table         the coefficients, sortable, filterable, wired to the volcano
    p-values      the histogram that says whether a correction means anything
    Q-Q           calibration, with the inflation figure
    Controls      the assay window

Clicking a point selects its row; selecting a row highlights its point. They
are two views of one table, which is the thing that was missing.
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
        from .fast_plots import (ControlSeparation, PValueHistogram, QQPlot,
                                 ResultsTable, VolcanoPlot)

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

        # GUIDE SUPPORT. The one thing the volcano structurally cannot show:
        # a gene carried by a single surviving guide and a gene whose guides
        # agree are the same dot, ranked by the same number, and only one of
        # them is independent evidence.
        self.support = ResultsTable()
        self.tabs.addTab(self.support, "Guide support")

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
        self.volcano.key_selected.connect(self.table.select_key)
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

    # ------------------------------------------------------------------ load

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

        # Offer every column that could sensibly colour the points, without
        # guessing: a column with one value per point is not a category.
        self._colour_by.blockSignals(True)
        self._colour_by.clear()
        self._colour_by.addItem("nothing", None)
        for name in frame.columns:
            if frame[name].dtype == object or str(frame[name].dtype) == "category":
                distinct = frame[name].nunique(dropna=True)
                if 1 < distinct <= max(40, len(frame) // 20):
                    self._colour_by.addItem(f"{name} ({distinct})", name)
        # 'condition' is the compartment column the screen actually uses.
        preferred = self._colour_by.findData("condition")
        self._colour_by.setCurrentIndex(preferred if preferred >= 0 else 0)
        self._colour_by.blockSignals(False)

        # A new table is a new experiment; carrying the old selection over
        # would ring a point that means something else now.
        self._selected_key = None
        self.volcano.clear_highlight()

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
        self._draw_controls(frame)
        self._draw_guide_support(frame)
        self.loaded.emit(source or "")
        return True

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
            return
        if support is None or not len(support):
            self.support.set_frame(None)
            return
        table = support.reset_index()
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
        self.support.set_frame(table)

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
            self.p_values.set_p_values(frame[column])
            self.qq.set_p_values(frame[column])
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

    @staticmethod
    def _effect_column(frame) -> str:
        for name in ("coefficient", "coef", "effect", "estimate"):
            if name in frame.columns:
                return name
        return "coefficient"

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
        self.volcano.set_results(
            self._frame,
            effect=self._effect_column(self._frame),
            p_column=p_column,
            label_column="feature" if "feature" in self._frame.columns
            else self._frame.columns[0],
            category_column=self._colour_by.currentData(),
            key_column=self._key_column(self._frame),
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
        groups = {}
        names = {"nc": "negative", "pc": "positive", "control": "control",
                 "other": "other"}
        for key, label in names.items():
            rows = frame[frame["condition"].astype(str) == key]
            if len(rows):
                groups[label] = rows[effect].to_numpy()
        self.controls.set_groups(groups)

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
        """A row was picked: ring its point, and say which point that is."""
        self._selected_key = str(key)
        found = self.volcano.highlight_key(key)
        if found:
            self.volcano.set_status(f"{key}")
            return
        # Honest about the miss. The commonest cause is a coefficient with an
        # unusable p-value, which is drawn nowhere; the next is a nuisance
        # term, which the volcano leaves off on purpose.
        self.volcano.set_status(
            f"{key} is in the table but not on this plot.")

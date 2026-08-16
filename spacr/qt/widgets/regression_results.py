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
    QComboBox, QHBoxLayout, QLabel, QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

#: Files a regression writes, best first. ``results.csv`` is the full
#: coefficient table; the gene/grna splits are views of it.
RESULT_FILENAMES = ("results.csv", "results_grna.csv", "results_gene.csv")


def find_results_table(path) -> Optional[str]:
    """The results CSV for ``path``: a file, a run folder, or a parent of one.

    Those are the three things a user actually has to hand when they want to
    look at a regression again, so all three are accepted.
    """
    if not path:
        return None
    path = os.path.abspath(os.path.expanduser(os.fspath(path)))
    if os.path.isfile(path):
        return path if path.lower().endswith(".csv") else None
    if not os.path.isdir(path):
        return None
    for name in RESULT_FILENAMES:
        candidate = os.path.join(path, name)
        if os.path.isfile(candidate):
            return candidate
    try:
        entries = sorted(os.listdir(path))
    except OSError:
        return None
    for entry in entries:
        child = os.path.join(path, entry)
        if os.path.isdir(child):
            found = find_results_table(child)
            if found:
                return found
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

        self._frame = None
        self._path = None
        self._selected_key = None

    # ------------------------------------------------------------------ load

    def load(self, path) -> bool:
        """Load a results CSV, a run folder, or a parent of one."""
        import pandas as pd

        found = find_results_table(path)
        if not found:
            self._source.setText(f"No results table under {path}")
            return False
        try:
            frame = pd.read_csv(found)
        except Exception as error:  # noqa: BLE001 - report, do not raise
            self._source.setText(f"Could not read {found}: {error}")
            return False
        return self.set_frame(frame, source=found)

    def set_frame(self, frame, source: str = "") -> bool:
        """Show an already-loaded coefficient table."""
        import pandas as pd

        if frame is None or not len(frame):
            self._source.setText("The results table is empty.")
            return False
        self._frame = frame
        self._path = source
        self._source.setText(source or f"{len(frame)} coefficients")

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
        self.table.set_frame(frame, key_column=self._key_column(frame))

        p_column = self._p_column(frame)
        if p_column:
            self.p_values.set_p_values(frame[p_column])
            self.qq.set_p_values(frame[p_column])
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

    @staticmethod
    def _p_column(frame) -> Optional[str]:
        for name in ("p_value", "p", "pvalue"):
            if name in frame.columns:
                return name
        return None

    @staticmethod
    def _effect_column(frame) -> str:
        for name in ("coefficient", "coef", "effect", "estimate"):
            if name in frame.columns:
                return name
        return "coefficient"

    def _redraw_volcano(self) -> None:
        if self._frame is None:
            return
        self.volcano.set_results(
            self._frame,
            effect=self._effect_column(self._frame),
            p_column=self._p_column(self._frame) or "p_value",
            label_column="feature" if "feature" in self._frame.columns
            else self._frame.columns[0],
            category_column=self._colour_by.currentData(),
            key_column=self._key_column(self._frame),
        )
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

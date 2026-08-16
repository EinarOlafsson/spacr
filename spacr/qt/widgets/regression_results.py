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

    def __init__(self, parent=None):
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
        split = QSplitter(Qt.Vertical)
        split.setChildrenCollapsible(False)
        split.addWidget(self.volcano)
        split.addWidget(self.table)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        self.tabs.addTab(split, "Volcano")

        self.p_values = PValueHistogram()
        self.qq = QQPlot()
        self.controls = ControlSeparation()
        self.tabs.addTab(self.p_values, "p-values")
        self.tabs.addTab(self.qq, "Q-Q")
        self.tabs.addTab(self.controls, "Controls")

        # The two directions of the same link.
        self.volcano.point_clicked.connect(self.table.select_frame_row)
        self.table.row_selected.connect(self._highlight_point)

        self._frame = None
        self._path = None

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

        self._redraw_volcano()
        self.table.set_frame(frame)

        p_column = self._p_column(frame)
        if p_column:
            self.p_values.set_p_values(frame[p_column])
            self.qq.set_p_values(frame[p_column])
        self._draw_controls(frame)
        self.loaded.emit(source or "")
        return True

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
        )

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

    def _highlight_point(self, index: int) -> None:
        """Say which point a selected row is, in the volcano's status line."""
        text = self.volcano._describe(index)
        if text:
            self.volcano.set_status(f"row {index}:  {text}")

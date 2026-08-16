"""The sweep's runs, as a tab beside the regression's results.

Instruction 116, as corrected by the maintainer on 2026-08-16:

    the parameter search is the main module setup plus one extra tab for the
    runs (the maintainer's exact words are in instruction 116).

Which is a smaller job than the plan it superseded, and a better one. The
parameter search was a bespoke screen carrying its own copies of the table,
the figure queue and the results panel; every fix to the shared module screen
had to be made again there, and this repository is already paying for that
kind of drift -- 155 field tooltips existed because 28 hand-built screens each
re-implemented a row.

So the runs go where the results go: the left half of the module screen's
figure splitter, as a second tab. Picking a run swaps the figures on the
right, which is the substance of the request and the only part that did not
change.

DELIBERATELY NOT THE SHAPE THE RESULTS GET. A results table is hundreds of
findings read ALONGSIDE the figure describing them, so it is on screen at the
same time as the plot. A runs table is a short list scanned top to bottom,
and picking one replaces everything to its right -- so it is a tab. One is
reading within a run, the other is navigating between them.
"""

from __future__ import annotations

import os
from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

#: What the sweep writes its table to, inside the destination folder.
RESULTS_FILENAME = "sweep_results.csv"

#: The columns worth reading first, in this order, when the table has them.
#: Settings, then WHAT WENT IN, then what came out: a hit count means little
#: without the size of the design behind it, and two trials differing only by
#: a filtration cutoff can be fitted on completely different data.
PREFERRED_COLUMNS = (
    "trial_id", "status", "regression_type", "inference", "analysis_unit",
    "agg_type", "transform", "multiple_testing_method", "fdr_alpha",
    "fraction_threshold", "min_cell_count",
    "n_wells", "n_guides", "n_cells", "n_rows_fitted",
    "n_results", "n_below_alpha", "positive_rank", "positive_percentile",
    "r_squared", "genomic_inflation", "seconds", "error_type",
)


def ordered_columns(frame) -> list:
    """``PREFERRED_COLUMNS`` that this frame has, then everything else.

    Ordering, not filtering. A column nobody thought to list is still worth
    seeing -- it is in the CSV, and hiding it means the user has to leave the
    application to read their own results.
    """
    if frame is None:
        return []
    have = list(frame.columns)
    first = [name for name in PREFERRED_COLUMNS if name in have]
    return first + [name for name in have if name not in first]


class SweepRunsPanel(QWidget):
    """One row per trial, sorted and filterable, wired to the figure grid.

    :ivar trial_activated: emitted with the selected trial's row, as a dict.
    :ivar loaded: emitted with the number of trials read.
    """

    trial_activated = Signal(dict)
    loaded = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        from .fast_plots import ResultsTable

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        header = QHBoxLayout()
        self._status = QLabel("No sweep loaded.")
        self._status.setWordWrap(True)
        header.addWidget(self._status, 1)
        self._reload = QPushButton("Reload")
        self._reload.setToolTip(
            "Re-read the results table. A sweep writes each trial as it "
            "finishes, so a running sweep can be watched.")
        self._reload.clicked.connect(self.reload)
        header.addWidget(self._reload)
        layout.addLayout(header)

        # The same table widget the results use: it already sorts numerically,
        # filters, and copies as TSV, and a second implementation of those is
        # a second set of bugs.
        self.table = ResultsTable()
        # Its own words. The coefficient table's "type a gene, a guide" and
        # "significant only" belong to a table of findings; over a list of
        # trials the first is wrong and the second cannot do anything.
        self.table.configure(
            placeholder="Filter runs — a model, a cutoff, anything in the row",
            significance_filter=False)
        self.table.table.itemSelectionChanged.connect(self._on_selection)
        layout.addWidget(self.table, 1)

        self._frame = None
        self._folder = ""

    # ------------------------------------------------------------------ load

    def load(self, folder) -> bool:
        """Read ``sweep_results.csv`` from a sweep's destination folder."""
        import pandas as pd

        if not folder:
            return False
        folder = os.path.abspath(os.path.expanduser(os.fspath(folder)))
        path = folder if folder.lower().endswith(".csv") else os.path.join(
            folder, RESULTS_FILENAME)
        if not os.path.isfile(path):
            self._status.setText(f"No results table at {path} yet.")
            return False
        try:
            frame = pd.read_csv(path)
        except Exception as error:  # noqa: BLE001 - report, do not raise
            self._status.setText(f"Could not read {path}: {error}")
            return False
        self._folder = folder
        return self.set_frame(frame, source=path)

    def reload(self) -> bool:
        return self.load(self._folder) if self._folder else False

    def set_frame(self, frame, source: str = "") -> bool:
        if frame is None or not len(frame):
            self._status.setText("The sweep has recorded no trials yet.")
            self.table.set_frame(None)
            return False
        self._frame = frame[ordered_columns(frame)]
        self.table.set_frame(self._frame, key_column=(
            "trial_id" if "trial_id" in frame.columns else None))
        failed = 0
        if "status" in frame.columns:
            failed = int((frame["status"].astype(str) != "ok").sum())
        note = f"{len(frame)} trials"
        if failed:
            # Said out loud rather than left to be noticed. A sweep whose
            # trials mostly failed still writes a full-looking table.
            note += f", {failed} of which did not produce a regression"
        self._status.setText(f"{note}. {source}" if source else note)
        self.loaded.emit(len(frame))
        return True

    # ------------------------------------------------------------- selection

    def selected_trial(self) -> Optional[dict]:
        """The selected row as a dict, or ``None``.

        Read back through the FRAME, not off the table's cells: the table
        holds display strings, and a trial re-run from `"0.05"` instead of
        0.05 is not the trial that was recorded.
        """
        if self._frame is None:
            return None
        items = self.table.table.selectedItems()
        if not items:
            return None
        index = items[0].data(0x0100)          # Qt.UserRole: the frame row
        if index is None or not 0 <= int(index) < len(self._frame):
            return None
        return self._frame.iloc[int(index)].to_dict()

    def _on_selection(self) -> None:
        record = self.selected_trial()
        if record is not None:
            self.trial_activated.emit(record)

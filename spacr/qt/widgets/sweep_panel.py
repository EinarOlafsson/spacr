"""One button: every gene against every measurement.

Instruction 175. The engine is :mod:`spacr.gene_measurement_sweep`; this is
the thin panel over it, and it is thin on purpose -- the three corrections
that make the answer trustworthy (identifiers excluded, Benjamini-Hochberg
across the grid, circularity reported per row) live in the engine so a
settings CSV, a macro and this button cannot disagree about them.

IT RUNS OFF THE GUI THREAD, and the reason is today's crash rather than
politeness: a regression built Qt widgets on its own worker and the process
segfaulted somewhere else entirely. So the worker here touches NO widget and
returns plain data, and the figure is rendered by matplotlib's Agg canvas
which has no Qt object to own.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, Optional

import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox,
                               QDoubleSpinBox, QFileDialog, QHBoxLayout,
                               QLabel, QPushButton, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

LOG = logging.getLogger("spacr.qt.sweep_panel")

__all__ = ["SweepPanel", "sweep_inputs"]


def sweep_inputs(cells, counts, *, score_column: str = "pred", scores=None):
    """``(wells, fractions, plates, scores)`` from a merged frame and counts.

    THE PLATE NAMES ARE CANONICALISED ON BOTH SIDES. A score CSV of the real
    screen says `pplate1` where its measurements database says `plate1`, so an
    un-canonicalised join matches no well at all -- and the resulting all-NaN
    circularity column reads as "nothing here is circular", which is the most
    confident possible way to say nothing. Instruction 145.
    """
    from ... import schema
    from ...multi_database import normalise_plate_ids

    frame = normalise_plate_ids(cells.copy())
    if "prc" not in frame.columns:
        frame["prc"] = [schema.compose_prc(p, r, c) for p, r, c in
                        zip(frame["plateID"], frame["rowID"],
                            frame["columnID"])]
    numeric = [c for c in frame.columns
               if pd.api.types.is_numeric_dtype(frame[c])]
    wells = frame.groupby("prc", observed=True)[numeric].mean()
    plates = wells.index.to_series().str.split("_").str[0]

    fractions = counts.pivot_table(index="prc", columns="grna",
                                   values="fraction", aggfunc="sum",
                                   fill_value=0.0)
    found = None
    if score_column in frame.columns:
        found = frame.groupby("prc", observed=True)[score_column].mean()
    elif scores is not None and len(scores):
        # The scores live in the run's own score CSVs, not in the measurement
        # tables -- and THEY say `pplate1` where the databases say `plate1`.
        offered = normalise_plate_ids(pd.DataFrame(scores).copy())
        if "prc" not in offered.columns and "plateID" in offered.columns:
            offered["prc"] = [schema.compose_prc(p, r, c) for p, r, c in
                              zip(offered["plateID"], offered["rowID"],
                                  offered["columnID"])]
        column = score_column if score_column in offered.columns else None
        if column and "prc" in offered.columns:
            found = offered.groupby("prc", observed=True)[column].mean()
    if found is not None:
        found = found.reindex(wells.index)
    return wells, fractions, plates, found


class SweepPanel(QWidget):
    """The button, the table, and the picture."""

    finished = Signal(object)

    def __init__(self, cells_provider: Optional[Callable] = None,
                 counts_provider: Optional[Callable] = None,
                 parent: Optional[QWidget] = None, *, threaded: bool = True,
                 scores_provider: Optional[Callable] = None):
        super().__init__(parent)
        from ..job_runner import JobRunner

        self._cells_provider = cells_provider
        self._counts_provider = counts_provider
        #: The per-object scores, so circularity can be computed at all. The
        #: merged measurements frame has no `pred` column -- it is the
        #: measurement tables -- so without this the column is NaN and the
        #: panel says so rather than showing zeros.
        self._scores_provider = scores_provider
        self._result = None
        self._jobs = JobRunner(self, threaded=bool(threaded),
                               app_key="sweep every gene")
        self._jobs.job_failed.connect(self._failed)

        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        self.run_button = QPushButton("Sweep every gene against every measurement")
        self.run_button.setToolTip(
            "Every guide against every measurement, blocked on plate and "
            "corrected across the whole grid. Identifier columns are left "
            "out, and every row says how far the classification score "
            "already tracks that measurement — a measurement the score "
            "predicts cannot corroborate anything derived from it.")
        self.run_button.clicked.connect(self.start)
        row.addWidget(self.run_button)

        self.level = QComboBox()
        for value, label in (("gene", "genes"), ("guide", "guides"),
                             ("both", "genes and guides")):
            self.level.addItem(label, value)
        self.level.setToolTip(
            "A gene's fraction in a well is the SUM of its guides', which is "
            "the same rule the regression applies — so 'does this gene move "
            "this measurement' is not a different arithmetic from the fit "
            "that found the gene.")
        row.addWidget(self.level)

        row.addWidget(QLabel("q <"))
        self.alpha = QDoubleSpinBox()
        self.alpha.setDecimals(3)
        self.alpha.setRange(0.001, 0.5)
        self.alpha.setSingleStep(0.01)
        self.alpha.setValue(0.05)
        self.alpha.valueChanged.connect(self._refill)
        row.addWidget(self.alpha)

        self.hide_circular = QCheckBox("hide what the score already tracks")
        self.hide_circular.setToolTip(
            "Leaves out measurements the classification score predicts. They "
            "are not wrong, they just cannot corroborate a result derived "
            "from that score.")
        self.hide_circular.setChecked(True)
        self.hide_circular.stateChanged.connect(self._refill)
        row.addWidget(self.hide_circular)

        row.addStretch(1)
        self.save_button = QPushButton("Save table…")
        self.save_button.clicked.connect(self.save)
        self.save_button.setEnabled(False)
        row.addWidget(self.save_button)
        layout.addLayout(row)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(
            ["level", "gene / guide", "measurement", "effect", "q",
             "circularity", "wells", "effective"])
        self.table.setSortingEnabled(True)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        layout.addWidget(self.table, 1)

    # ------------------------------------------------------------- running

    def start(self, *_args) -> bool:
        """Run the sweep. Returns whether one was started."""
        if self._cells_provider is None or self._counts_provider is None:
            self.status.setText(
                "Nothing to sweep: this panel has no measurements and no "
                "counts wired to it.")
            return False
        try:
            cells = self._cells_provider()
            counts = self._counts_provider()
        except Exception as error:                       # noqa: BLE001
            self.status.setText(f"Could not read the inputs: {error}")
            return False
        if cells is None or not len(cells) or counts is None or not len(counts):
            self.status.setText(
                "Nothing to sweep: merge the measurement databases first, and "
                "attach the count CSVs the run was fitted on.")
            return False

        self.run_button.setEnabled(False)
        self.status.setText("Sweeping…")
        # THE WORKER TOUCHES NO WIDGET. It returns the result object and the
        # GUI thread does the rest -- which is the rule a regression broke
        # today by building Qt widgets on its own worker.
        scores = None
        if self._scores_provider is not None:
            try:
                scores = self._scores_provider()
            except Exception:                            # noqa: BLE001
                LOG.debug("could not read the scores", exc_info=True)
        level = str(self.level.currentData() or "gene")
        return bool(self._jobs.submit(
            lambda: self._work(cells, counts, scores, level), self._done))

    @staticmethod
    def _work(cells, counts, scores=None, level="gene"):
        from ...gene_measurement_sweep import sweep

        wells, fractions, plates, found = sweep_inputs(cells, counts,
                                                       scores=scores)
        return sweep(wells, fractions, blocks=plates, scores=found,
                     level=level)

    def _done(self, result) -> None:
        self.run_button.setEnabled(True)
        self._result = result
        if result is None:
            self.status.setText("The sweep returned nothing.")
            return
        self.save_button.setEnabled(True)
        self.status.setText(result.describe())
        self._refill()
        self.finished.emit(result)

    def _failed(self, message: str) -> None:
        self.run_button.setEnabled(True)
        self.status.setText(f"The sweep did not finish: {message}")

    # -------------------------------------------------------------- the view

    def rows(self) -> pd.DataFrame:
        """What the table is showing, as a frame."""
        if self._result is None:
            return pd.DataFrame()
        # A CIRCULARITY BAR THE RESULT CANNOT HONOUR IS REFUSED, not silently
        # applied to a column of NaN -- which returns nothing and looks like
        # an answer.
        bar = 0.15 if (self.hide_circular.isChecked()
                       and self._result.circularity_known) else 1.0
        return self._result.survivors(alpha=float(self.alpha.value()),
                                      max_circularity=bar)

    def _refill(self, *_args) -> None:
        keep = self.rows()
        self.table.setSortingEnabled(False)
        self.table.setRowCount(0)
        shown = keep.head(2000)
        self.table.setRowCount(len(shown))
        for row, (_index, entry) in enumerate(shown.iterrows()):
            values = [str(entry.get("level", "guide")), str(entry["guide"]),
                      str(entry["measurement"]),
                      f"{entry['effect']:+.3f}", f"{entry['q']:.2e}",
                      ("—" if pd.isna(entry["circularity"])
                       else f"{entry['circularity']:.2f}"),
                      str(int(entry["n_wells"])),
                      f"{entry['effective_wells']:.0f}"]
            for column, text in enumerate(values):
                self.table.setItem(row, column, QTableWidgetItem(text))
        self.table.setSortingEnabled(True)
        if len(keep) > len(shown):
            self.status.setText(
                f"{self.status.text()}  Showing the first {len(shown):,} of "
                f"{len(keep):,} — save the table for all of them.")

    def figure(self, path: Optional[str] = None):
        """The picture of what survived, or ``None``."""
        if self._result is None:
            return None
        from ...gene_measurement_sweep import plot_sweep

        bar = 0.15 if (self.hide_circular.isChecked()
                       and self._result.circularity_known) else 1.0
        # The picture follows the level the panel is showing, so a "both"
        # sweep does not draw a gene and its own guides as if they were
        # independent agreement.
        chosen = str(self.level.currentData() or "gene")
        return plot_sweep(self._result, path=path,
                          alpha=float(self.alpha.value()),
                          max_circularity=bar,
                          level=None if chosen == "guide" else "gene")

    def save(self, *_args) -> str:
        """Write the whole table -- not the page on screen."""
        if self._result is None:
            return ""
        path, _filter = QFileDialog.getSaveFileName(
            self, "Save the sweep", "gene_measurement_sweep.csv",
            "CSV (*.csv)")
        if not path:
            return ""
        self._result.table.to_csv(path, index=False)
        figure_path = os.path.splitext(path)[0] + ".png"
        try:
            self.figure(path=figure_path)
        except Exception:                                # noqa: BLE001
            LOG.debug("could not draw the sweep figure", exc_info=True)
        self.status.setText(f"Saved {len(self._result.table):,} rows to {path}")
        return path

    def closeEvent(self, event):                         # noqa: N802 - Qt name
        try:
            self._jobs.shutdown()
        finally:
            super().closeEvent(event)

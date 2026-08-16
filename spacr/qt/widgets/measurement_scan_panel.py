"""Which measurement has genes with clear effect sizes.

Instruction 122 part 3, as asked for:

    "doing a sweep on these screen data of which measurements have genes with
     an effect size. so instead of a parameter search a search for which
     measurement has genes with clear effect sizes (one or several)"

The pure logic is :mod:`spacr.measurement_scan`, which had no caller. This is
the thin renderer over it, and it lives beside the sweep's runs for the reason
the instruction gives: structurally this IS the parameter search with a
different thing varying -- the settings are held fixed and the DEPENDENT
VARIABLE is swept.

WHAT THIS PANEL EXISTS TO SHOW, and the reason a plain table would be wrong:

    A MEASUREMENT SCAN IS A MULTIPLE-TESTING PROBLEM ACROSS MEASUREMENTS.

spaCR measures hundreds of features per object. Scan 500 for "genes with a
clear effect" and some look clear by chance, and they look exactly as
convincing as the real ones -- because the per-measurement FDR was computed
WITHIN each measurement and knows nothing about the other 499. So both
numbers are on every row, and the across-scan one is what the verdict column
reads. A panel that showed only the within-run q-value would have rebuilt the
exact trap the module exists to close.

Measured on plate1 of the tsg101 screen with the gene labels permuted, so no
effect can exist: the within-run correction fired on 83.5% of those scans and
the across-scan correction on 5.0%. That gap is the feature.

Ranked by EFFECT SIZE, not by p-value: with two screens' worth of wells a
trivial effect is significant, and "clear effect sizes" is what was asked for.
"""

from __future__ import annotations

import os
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)

#: Columns worth reading first, in this order. Effect size leads because it is
#: the primary sort and the thing that was asked for; the two corrections sit
#: beside each other so the gap between them is visible without scrolling.
PREFERRED_COLUMNS = (
    "measurement", "effect_size", "top_gene", "across_scan_q", "within_run_q",
    "verdict", "coefficient", "p_value", "within_run_hits", "n_wells",
    "n_genes", "measurement_p",
)

#: What a row's two corrections mean together, in words. The middle one is the
#: single most useful thing this feature can say and the easiest to hide.
VERDICT_SURVIVES = "clear effect"
VERDICT_WITHIN_ONLY = "would pass alone — not across the scan"
VERDICT_NEITHER = "no effect"


def verdict_for(row) -> str:
    """One phrase per measurement, from BOTH corrections."""
    if getattr(row, "survives_across_scan", False):
        return VERDICT_SURVIVES
    if getattr(row, "survives_within_run", False):
        return VERDICT_WITHIN_ONLY
    return VERDICT_NEITHER


def ordered_columns(frame) -> list:
    """:data:`PREFERRED_COLUMNS` this frame has, then everything else.

    Ordering, not filtering -- the columns nobody thought to list are still
    the user's own numbers.
    """
    if frame is None:
        return []
    have = list(frame.columns)
    first = [name for name in PREFERRED_COLUMNS if name in have]
    return first + [name for name in have if name not in first]


class MeasurementScanPanel(QWidget):
    """The scan's result table, and the two numbers behind every row.

    :ivar measurement_selected: emitted with the measurement name of the
        selected row, so a host can draw it.
    :ivar scanned: emitted with the number of measurements scanned.
    """

    measurement_selected = Signal(str)
    scanned = Signal(int)

    def __init__(self, frame_provider=None, parent=None):
        """
        :param frame_provider: called with no arguments for the well-level
            frame to scan. A callable rather than a stored frame, so the panel
            cannot go on scanning the previous run's data after a new one is
            loaded.
        """
        super().__init__(parent)
        from .fast_plots import ResultsTable

        self._frame_provider = frame_provider
        self._result = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        top = QHBoxLayout()
        self._run = QPushButton("Scan measurements")
        self._run.setToolTip(
            "Hold the model fixed and sweep the dependent variable: which "
            "measurement has genes with a clear effect. Corrected across the "
            "scan, not only within each measurement.")
        self._run.clicked.connect(self.run_scan)
        top.addWidget(self._run)

        top.addWidget(QLabel("rank by"))
        self._rank = QComboBox()
        # Effect size first, because that is what was asked for and because
        # with enough wells a trivial effect is significant.
        self._rank.addItem("effect size", "effect_size")
        self._rank.addItem("across-scan q", "across_scan_q")
        self._rank.addItem("within-run q", "within_run_q")
        self._rank.currentIndexChanged.connect(self._resort)
        top.addWidget(self._rank)
        top.addStretch(1)
        layout.addLayout(top)

        self._status = QLabel("No scan yet.")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        self.table = ResultsTable()
        self.table.configure(
            placeholder="Filter measurements — a channel, a shape, anything",
            significance_filter=False)
        self.table.table.itemSelectionChanged.connect(self._on_selection)
        layout.addWidget(self.table, 1)

    # -------------------------------------------------------------- running

    def set_frame_provider(self, provider) -> None:
        self._frame_provider = provider

    def run_scan(self, **kwargs) -> bool:
        """Scan whatever the provider is holding. Returns whether it ran."""
        frame = None
        if callable(self._frame_provider):
            try:
                frame = self._frame_provider()
            except Exception as error:  # noqa: BLE001 - report, do not raise
                self._status.setText(f"Could not read the data: {error}")
                return False
        if frame is None or not len(frame):
            self._status.setText(
                "Nothing to scan. Load a run whose wells carry both the gene "
                "assignment and the measurements.")
            return False
        return self.scan(frame, **kwargs)

    def scan(self, frame, **kwargs) -> bool:
        """Scan ``frame`` and show the result."""
        from ...measurement_scan import ScanRefused, scan_measurements

        try:
            result = scan_measurements(frame, **kwargs)
        except ScanRefused as refusal:
            # A refusal is an ANSWER and it says what to do about it. Shown
            # in full rather than summarised: "the scan failed" would send the
            # user looking for a bug in the software.
            self._status.setText(str(refusal))
            self.table.set_frame(None)
            self._result = None
            return False
        except Exception as error:  # noqa: BLE001 - report, do not raise
            self._status.setText(f"The scan did not finish: {error}")
            self.table.set_frame(None)
            self._result = None
            return False
        return self.set_result(result)

    def set_result(self, result) -> bool:
        """Show an already-computed :class:`ScanResult`."""
        self._result = result
        table = result.frame()
        if not len(table):
            self._status.setText("No measurement could be scanned.\n"
                                 + result.describe())
            self.table.set_frame(None)
            return False

        # BOTH CORRECTIONS, IN WORDS, ON EVERY ROW. A measurement that passes
        # within its own run and fails across the scan is the single most
        # important thing this feature can tell a user, and it is invisible in
        # two columns of small numbers.
        table = table.copy()
        table["verdict"] = [verdict_for(row) for row in result.rows]
        table = table.loc[table.index]           # keep the frame's own order
        self.table.set_frame(table[ordered_columns(table)],
                             key_column="measurement")
        self._status.setText(self._summary(result))
        self.scanned.emit(len(result.rows))
        return True

    @staticmethod
    def _summary(result) -> str:
        """The header. Leads with the gap between the two corrections."""
        survivors = len(result.surviving())
        within = sum(1 for row in result.rows if row.survives_within_run)
        text = [
            f"{len(result.rows)} measurements scanned. "
            f"{survivors} show a clear gene effect across the scan; "
            f"{within} would have been reported by a single-measurement run."
        ]
        if within > survivors:
            text.append(
                f"The {within - survivors} in between are the ones a "
                f"per-measurement analysis would have shown you as hits.")
        dropped = getattr(result, "genes_dropped", None)
        if dropped:
            text.append(
                f"{len(dropped)} gene(s) left out for having fewer than two "
                f"wells — a gene in one well has nothing corroborating it: "
                + ", ".join(sorted(dropped)[:6])
                + ("…" if len(dropped) > 6 else ""))
        if result.skipped:
            text.append(f"{len(result.skipped)} column(s) not scanned.")
        return "  ".join(text)

    # ------------------------------------------------------------ selection

    @property
    def result(self):
        return self._result

    def _resort(self) -> None:
        if self._result is None:
            return
        column = self._rank.currentData()
        table = self._result.frame().copy()
        table["verdict"] = [verdict_for(row) for row in self._result.rows]
        if column in table.columns:
            ascending = column != "effect_size"
            key = (lambda s: s.abs()) if column == "effect_size" else None
            table = table.sort_values(column, ascending=ascending, key=key,
                                      kind="stable").reset_index(drop=True)
        self.table.set_frame(table[ordered_columns(table)],
                             key_column="measurement")

    def _on_selection(self) -> None:
        key = None
        items = self.table.table.selectedItems()
        if items:
            key = self.table.key_for_row(items[0].data(Qt.UserRole))
        if key:
            self.measurement_selected.emit(str(key))

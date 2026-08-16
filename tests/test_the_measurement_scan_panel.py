"""The panel that answers "which measurement has genes with a clear effect".

Instruction 122 part 3. `spacr.measurement_scan` had no caller at all -- the
logic existed and nothing in the application could reach it.

WHAT THIS PANEL HAS TO GET RIGHT, and it is not the table:

    A MEASUREMENT SCAN IS A MULTIPLE-TESTING PROBLEM ACROSS MEASUREMENTS.

Scan 500 features for "genes with a clear effect" and some look clear by
chance, and they look exactly as convincing as the real ones -- because the
per-measurement FDR was computed WITHIN each measurement and knows nothing
about the other 499. Measured on plate1 of the tsg101 screen with the gene
labels permuted, so no effect can exist by construction: the within-run
correction fired on 83.5% of those scans, the across-scan correction on 5.0%.

So a panel that showed the within-run q-value as "the answer" would have
rebuilt the exact trap the module exists to close. Both numbers are on every
row and the VERDICT reads the across-scan one -- and the measurements in
between the two get a phrase of their own, because that set is the single
most useful thing this feature can say and the easiest to bury in a column
of small numbers.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


def _wells(n_measurements=30, seed=0, planted=4):
    """A well-level frame: two replicated genes, correlated measurements."""
    rng = np.random.default_rng(seed)
    genes = ["geneA"] * 14 + ["geneB"] * 14 + ["nc"] * 8
    n = len(genes)
    latent = rng.normal(size=n)
    frame = pd.DataFrame(
        {f"m{j}": latent * rng.uniform(.5, 1.0) + rng.normal(0, .5, n)
         for j in range(n_measurements)})
    frame["gene"] = genes
    for column in [f"m{j}" for j in range(planted)]:
        frame.loc[frame.gene == "geneA", column] += 6.0
    return frame


@pytest.fixture()
def panel(qtbot):
    from spacr.qt.widgets.measurement_scan_panel import MeasurementScanPanel

    widget = MeasurementScanPanel()
    qtbot.addWidget(widget)
    return widget


# --------------------------------------------------------------------------- #
#  It runs, and it ranks by effect size
# --------------------------------------------------------------------------- #

def test_it_scans_and_fills_the_table(panel):
    assert panel.scan(_wells(), gene_column="gene", block_columns=(),
                      control_genes=["nc"]) is True
    assert panel.table.table.rowCount() == 30


def test_effect_size_is_the_primary_sort(panel):
    """"clear effect sizes" is what was asked for. With enough wells a
    trivial effect is significant, so ranking on p would put noise on top."""
    panel.scan(_wells(), gene_column="gene", block_columns=(),
               control_genes=["nc"])

    columns = [panel.table.table.horizontalHeaderItem(i).text()
               for i in range(panel.table.table.columnCount())]
    assert columns[0] == "measurement"
    assert columns[1] == "effect_size", columns[:4]

    sizes = [abs(float(panel.table.table.item(r, 1).text()))
             for r in range(min(6, panel.table.table.rowCount()))]
    assert sizes == sorted(sizes, reverse=True), sizes


def test_the_planted_measurements_come_out_on_top(panel):
    panel.scan(_wells(planted=4), gene_column="gene", block_columns=(),
               control_genes=["nc"])

    top4 = {panel.table.table.item(r, 0).text() for r in range(4)}
    assert top4 == {"m0", "m1", "m2", "m3"}, top4


# --------------------------------------------------------------------------- #
#  Both corrections, and the gap between them
# --------------------------------------------------------------------------- #

def test_every_row_carries_both_corrections(panel):
    panel.scan(_wells(), gene_column="gene", block_columns=(),
               control_genes=["nc"])

    columns = [panel.table.table.horizontalHeaderItem(i).text()
               for i in range(panel.table.table.columnCount())]
    assert "across_scan_q" in columns
    assert "within_run_q" in columns
    assert columns.index("across_scan_q") < columns.index("within_run_q"), (
        "the within-run number is presented as the primary one")


def test_the_verdict_reads_the_across_scan_number(panel):
    """Not the within-run one. A panel whose verdict came from the per-
    measurement FDR would be the trap this module exists to close."""
    from spacr.qt.widgets.measurement_scan_panel import (VERDICT_SURVIVES,
                                                         verdict_for)

    panel.scan(_wells(), gene_column="gene", block_columns=(),
               control_genes=["nc"])

    for row in panel.result.rows:
        if verdict_for(row) == VERDICT_SURVIVES:
            assert row.survives_across_scan, row.measurement


def test_the_in_between_set_is_named_not_buried(panel):
    """A measurement that passes alone and fails across the scan is what a
    per-measurement analysis would have shown the user as a hit."""
    from spacr.qt.widgets.measurement_scan_panel import (VERDICT_WITHIN_ONLY,
                                                         verdict_for)

    # Weak, correlated signal: enough to pass within a run, not across a scan.
    rng = np.random.default_rng(7)
    frame = _wells(n_measurements=40, seed=4, planted=0)
    for column in [f"m{j}" for j in range(12)]:
        frame.loc[frame.gene == "geneA", column] += rng.normal(0.9, .1)

    panel.scan(frame, gene_column="gene", block_columns=(),
               control_genes=["nc"])

    verdicts = [verdict_for(row) for row in panel.result.rows]
    if VERDICT_WITHIN_ONLY in verdicts:
        assert VERDICT_WITHIN_ONLY in panel._status.text() or \
            "in between" in panel._status.text(), panel._status.text()
    # The header always states both counts, whichever way the data falls.
    assert "across the scan" in panel._status.text()
    assert "single-measurement run" in panel._status.text()


def test_dropped_single_well_genes_are_reported(panel):
    """They are dropped because a gene in one well corroborates nothing --
    and the user is told, because a gene missing with no explanation reads as
    a gene with no effect."""
    frame = _wells()
    frame.loc[frame.index[-1], "gene"] = "seen_once"

    panel.scan(frame, gene_column="gene", block_columns=(),
               control_genes=["nc"])

    assert "seen_once" in panel._status.text()
    assert "nothing corroborating it" in panel._status.text()


# --------------------------------------------------------------------------- #
#  Re-ranking, without re-fitting
# --------------------------------------------------------------------------- #

def test_ranking_by_q_reorders_without_a_refit(panel):
    panel.scan(_wells(), gene_column="gene", block_columns=(),
               control_genes=["nc"])
    before = panel.result

    index = panel._rank.findData("across_scan_q")
    panel._rank.setCurrentIndex(index)

    assert panel.result is before, "changing the sort re-ran the scan"
    qs = [float(panel.table.table.item(r, 3).text())
          for r in range(min(5, panel.table.table.rowCount()))]
    assert qs == sorted(qs), qs


# --------------------------------------------------------------------------- #
#  A refusal is an answer
# --------------------------------------------------------------------------- #

def test_a_refusal_is_shown_in_full(panel):
    """"the scan failed" would send the user looking for a bug in spaCR. The
    refusal says what is wrong with the data and what to do about it."""
    frame = pd.DataFrame({"m0": [1.0, 2.0, 3.0], "m1": [3.0, 2.0, 1.0]})

    assert panel.scan(frame, gene_column="gene", block_columns=()) is False
    assert "gene" in panel._status.text()
    assert panel.table.table.rowCount() == 0


def test_no_data_at_all_says_so_rather_than_showing_an_empty_table(panel):
    """An empty table reads as "no measurement has an effect", which is the
    opposite of "nothing was scanned"."""
    assert panel.run_scan() is False
    assert "Nothing to scan" in panel._status.text()


def test_a_provider_that_raises_does_not_take_the_panel_down(qtbot):
    from spacr.qt.widgets.measurement_scan_panel import MeasurementScanPanel

    def _boom():
        raise OSError("the database went away")

    widget = MeasurementScanPanel(frame_provider=_boom)
    qtbot.addWidget(widget)

    assert widget.run_scan() is False
    assert "went away" in widget._status.text()


def test_the_table_speaks_about_measurements(panel):
    """It reuses the coefficient table's widget and must not inherit its
    words -- "significant only" over a scan means the wrong correction."""
    assert "gene" not in panel.table._filter.placeholderText().lower()
    assert "measurement" in panel.table._filter.placeholderText().lower()
    assert not panel.table._only_hits.isVisibleTo(panel)


def test_selecting_a_row_names_its_measurement(panel):
    panel.scan(_wells(), gene_column="gene", block_columns=(),
               control_genes=["nc"])

    seen = []
    panel.measurement_selected.connect(seen.append)
    panel.table.table.selectRow(0)

    assert seen and seen[-1] == panel.table.table.item(0, 0).text()

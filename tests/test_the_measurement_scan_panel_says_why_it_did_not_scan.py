"""The Measurements tab's answers when the scan produced nothing.

The paths beside :mod:`tests.test_the_measurement_scan_panel`, which covers
the scan that worked. These cover the four ways it does not, and they matter
for one reason:

    AN EMPTY TABLE READS AS "NO MEASUREMENT HAS AN EFFECT".

That is the opposite of "nothing was scanned", and the two are one pixel apart
on screen. Every one of these paths therefore has to leave a sentence behind.

Written while wiring the measurement databases into this tab (instruction
130): the file was at 87% and every missing line was a thing the panel says
when something is wrong.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


def _wells(n_measurements=6, seed=0):
    rng = np.random.default_rng(seed)
    genes = ["geneA"] * 6 + ["geneB"] * 6 + ["nc"] * 6
    n = len(genes)
    frame = pd.DataFrame({f"m{j}": rng.normal(size=n)
                          for j in range(n_measurements)})
    frame["gene"] = genes
    return frame


@pytest.fixture()
def panel(qtbot):
    from spacr.qt.widgets.measurement_scan_panel import MeasurementScanPanel

    widget = MeasurementScanPanel()
    qtbot.addWidget(widget)
    return widget


def test_a_scan_of_flat_measurements_names_them_rather_than_showing_nothing(
        panel):
    """Every column constant: there are no rows to show and no effect to
    report, and the reason is per column. "No measurement could be scanned"
    plus the list is a diagnosis; an empty table is a conclusion, and the
    wrong one."""
    flat = pd.DataFrame({"gene": ["a"] * 4 + ["b"] * 4,
                         "m0": [1.0] * 8, "m1": [2.0] * 8})

    assert panel.scan(flat, gene_column="gene", block_columns=()) is False
    assert "No measurement could be scanned" in panel._status.text()
    assert "no variance" in panel._status.text()
    assert panel.table.table.rowCount() == 0


def test_a_scan_that_fails_outright_says_so_instead_of_raising(panel):
    """Not a refusal -- a refusal is about the data and says what to do. This
    is anything else, and the panel is a renderer: it must not take the window
    down with it."""
    assert panel.scan(_wells(), gene_column="gene", block_columns=(),
                      across_scan_method="not_a_method") is False
    assert "did not finish" in panel._status.text()
    assert "not_a_method" in panel._status.text()
    assert panel.result is None


def test_a_scan_names_the_columns_it_left_out(panel):
    """A measurement missing from the result with no explanation reads as a
    measurement with no effect."""
    frame = _wells()
    frame["m_flat"] = 3.0

    assert panel.scan(frame, gene_column="gene", block_columns=()) is True
    assert "1 column(s) not scanned" in panel._status.text()


def test_a_new_frame_provider_is_what_the_next_scan_reads(panel):
    """The provider is a callable so the panel cannot go on scanning the
    previous run's data. Replacing it has to replace the data."""
    panel.set_frame_provider(lambda: _wells())

    assert panel.run_scan(gene_column="gene", block_columns=()) is True
    assert panel.table.table.rowCount() == 6


def test_re_ranking_before_a_scan_does_nothing_at_all(panel):
    """The rank box is live from the start. Reading a result that is not there
    would take the tab down on a click that means nothing yet."""
    panel._rank.setCurrentIndex(1)

    assert panel.result is None
    assert panel.table.table.rowCount() == 0


def test_a_frame_that_is_not_there_orders_no_columns():
    """`ordered_columns` is called on whatever the panel was handed."""
    from spacr.qt.widgets.measurement_scan_panel import ordered_columns

    assert ordered_columns(None) == []

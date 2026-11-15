"""Plate QC keeps its promises on the inputs that sit just off the happy path.

Three behaviours in :mod:`spacr.plate_qc` decide whether a QC pass helps or
misleads, and each is reached only by input the common case never produces:

* a database column that *looks* numeric but holds nothing usable — offering
  it in a measurement picker gives the user an empty heatmap;
* a well value that is infinite — the median it poisons has to be reported as
  *undefined*, never as a difference of zero or a NaN;
* a ring profile capped at zero rings — the table empties while the
  outer-vs-interior test stays.
"""
from __future__ import annotations

import re
import sqlite3

import numpy as np
import pandas as pd
import pytest

from spacr import plate_qc as pq

# ---------------------------------------------------------------------------
# Synthetic plates
# ---------------------------------------------------------------------------

def _long_frame(n_rows=8, n_cols=12, *, edge_boost=0.0, plate="plate1",
                n_objects=4, seed=0, edge_value=None):
    """A long per-object frame for one plate, one row per object.

    ``edge_boost`` lifts the outer ring by a known fraction; ``edge_value``
    instead pins every outer-ring object to a fixed value (used to plant an
    infinity on the ring).
    """
    rng = np.random.default_rng(seed)
    records = []
    for r in range(1, n_rows + 1):
        for c in range(1, n_cols + 1):
            ring = min(r - 1, n_rows - r, c - 1, n_cols - c)
            for _ in range(n_objects):
                if ring == 0 and edge_value is not None:
                    value = float(edge_value)
                else:
                    mult = 1.0 + (edge_boost if ring == 0 else 0.0)
                    value = 100.0 * mult * float(rng.lognormal(0.0, 0.08))
                records.append({"prc": f"{plate}_r{r}_c{c}", "value": value})
    return pd.DataFrame(records)


def _measurements_db(tmp_path):
    """A ``measurements.db`` whose ``cell`` table holds three unusable columns.

    ``area`` is the one real measurement. ``all_null`` is declared REAL but
    was never written to; ``untyped`` has no SQLite declared type at all (so
    the affinity screen cannot reject it) and holds text; ``well_note`` is
    plainly TEXT.
    """
    db_path = tmp_path / "measurements" / "measurements.db"
    db_path.parent.mkdir(parents=True)
    con = sqlite3.connect(db_path)
    try:
        con.execute("CREATE TABLE cell (prc TEXT, area REAL, all_null REAL, "
                    "untyped, well_note TEXT, object_label INTEGER)")
        con.executemany(
            "INSERT INTO cell VALUES (?, ?, ?, ?, ?, ?)",
            [(f"plate1_r{r}_c{c}", 10.0 * r + c, None, "not a number",
              "text-not-a-number", r * 100 + c)
             for r in range(1, 9) for c in range(1, 13)])
        con.commit()
    finally:
        con.close()
    return str(db_path)


def _assert_no_nan_rendered(text):
    """A statistic may be printed as ``undefined``, never as ``nan``.

    Word-bounded on purpose: "Dominant" contains the three letters and is not
    a NaN.
    """
    assert not re.search(r"\bnan\b", text, re.IGNORECASE), text


# ---------------------------------------------------------------------------
# The measurement picker
# ---------------------------------------------------------------------------

def test_a_column_with_no_usable_numbers_is_not_offered_as_a_measurement(tmp_path):
    """``numeric_columns`` fills the measurement dropdown of the plate view.
    A column that is declared REAL but holds only NULLs, or that has no
    declared type at all and holds text, passes the cheap affinity screen —
    only reading a sample rejects it. If it were offered, the user would pick
    it and get a blank plate with no explanation of why.
    """
    db_path = _measurements_db(tmp_path)

    offered = pq.numeric_columns(db_path, "cell")

    # The columns that survive the sample: the measurement and the integer id.
    assert offered == ["area", "object_label"]
    # ... and the three that must not, each rejected for a different reason.
    assert "all_null" not in offered      # declared REAL, sampled nothing
    assert "untyped" not in offered       # no affinity to screen on, text data
    assert "well_note" not in offered     # plainly TEXT
    # The identifier column is excluded by name, not by its contents.
    assert "prc" not in offered
    # And every name offered is a real column of the table.
    assert set(offered) <= set(pq.table_columns(db_path, "cell"))


def test_a_sample_of_one_row_still_rejects_a_text_column(tmp_path):
    """The sample size is a user-facing cost control — a 500-column feature
    table is scanned once per column. Shrinking it must not make the screen
    credulous: one sampled row of ``'not a number'`` is already proof enough
    that the column cannot be plotted.
    """
    db_path = _measurements_db(tmp_path)

    offered = pq.numeric_columns(db_path, "cell", sample=1)

    assert offered == ["area", "object_label"]
    assert "untyped" not in offered


# ---------------------------------------------------------------------------
# A median that cannot be computed
# ---------------------------------------------------------------------------

def test_an_infinite_edge_median_is_reported_as_undefined_not_as_no_difference():
    """A divide-by-zero upstream puts ``inf`` in a feature column, and it
    reaches plate QC as a real well value. The outer-ring median is then not
    a number — but the ring *is* populated, so the comparison still runs. The
    difference between the rings has to come back as ``None`` and print as
    "undefined": a median difference of 0.0 here would read as "checked, no
    edge effect" for a plate nobody has actually measured.
    """
    poisoned = pq.detect_edge_effect(
        _long_frame(8, 12, edge_value=float("inf")), "value")

    assert poisoned.ok                       # both groups are populated
    assert (poisoned.n_edge_wells, poisoned.n_interior_wells) == (36, 60)
    assert poisoned.edge_median is None      # inf is not a number to report
    assert poisoned.interior_median == pytest.approx(100.0, abs=5.0)
    assert poisoned.median_difference is None
    assert poisoned.pct_difference is None
    # The rank test still has an answer — ranks survive an infinity.
    assert poisoned.cliffs_delta == pytest.approx(1.0)

    text = pq.format_edge_report(poisoned)
    _assert_no_nan_rendered(text)
    assert "median difference   undefined" in text
    assert "edge median         undefined" in text

    # The contrast: with finite values the same fields carry numbers, so the
    # "undefined" above is the infinity's doing and not a dead code path.
    finite = pq.detect_edge_effect(_long_frame(8, 12, edge_boost=0.30), "value")
    assert finite.edge_median == pytest.approx(130.0, abs=8.0)
    assert finite.median_difference == pytest.approx(30.0, abs=8.0)
    assert finite.pct_difference == pytest.approx(30.0, abs=8.0)
    _assert_no_nan_rendered(pq.format_edge_report(finite))


# ---------------------------------------------------------------------------
# The ring cap
# ---------------------------------------------------------------------------

def test_a_ring_cap_of_zero_reports_no_rings_and_keeps_the_edge_test():
    """Zero rings is what a caller passes to suppress the profile entirely
    (the plate-view summary does this when the panel is collapsed). It must
    empty the ring table without disabling the outer-vs-interior test the
    report is built around.
    """
    frame = _long_frame(8, 12, edge_boost=0.30)

    report = pq.detect_edge_effect(frame, "value", max_rings=0)

    assert report.rings == []
    assert report.ok and report.edge_detected
    assert report.pct_difference == pytest.approx(30.0, abs=8.0)
    text = pq.format_edge_report(report)
    assert "Ring profile" not in text
    _assert_no_nan_rendered(text)
    # With the profile asked for, the section is there — the absence above is
    # the cap's doing.
    assert "Ring profile" in pq.format_edge_report(
        pq.detect_edge_effect(frame, "value"))


# ---------------------------------------------------------------------------
# The measurement picker when nothing at all is plottable
# ---------------------------------------------------------------------------

def _one_measurement_db(path, values):
    """A ``cell`` table with one identifier and one REAL measurement column.

    ``values`` are written into ``area`` one row per well, so the same table
    shape can be built once full of numbers and once full of NULLs.
    """
    con = sqlite3.connect(path)
    try:
        con.execute("CREATE TABLE cell (prc TEXT, area REAL)")
        con.executemany(
            "INSERT INTO cell VALUES (?, ?)",
            [(f"plate1_r1_c{i + 1}", v) for i, v in enumerate(values)])
        con.commit()
    finally:
        con.close()
    return str(path)


def test_a_table_whose_only_measurement_is_empty_offers_nothing_to_plot(tmp_path):
    """A measure run that died after creating its tables leaves every feature
    column in place and every row NULL. ``numeric_columns`` is what fills the
    plate view's measurement dropdown, and the declared REAL affinity cannot
    tell the two cases apart — only reading the rows can. If the empty column
    were offered, the user would pick it, get a blank plate, and conclude the
    plate failed rather than the run. An empty list is what lets the caller
    say "this database holds no measurements yet".
    """
    empty = _one_measurement_db(tmp_path / "empty.db", [None] * 12)
    written = _one_measurement_db(tmp_path / "written.db",
                                  [10.0 * i for i in range(1, 13)])

    assert pq.numeric_columns(empty, "cell") == []
    # The identical table *with* numbers in it does offer the column, so the
    # empty list above is the NULLs' doing and not a query that never matches.
    assert pq.numeric_columns(written, "cell") == ["area"]
    # The column exists in both databases — its contents, not the schema,
    # decide whether it can be plotted.
    assert pq.table_columns(empty, "cell") == ["prc", "area"]
    assert pq.table_columns(empty, "cell") == pq.table_columns(written, "cell")

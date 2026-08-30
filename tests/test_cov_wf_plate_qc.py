"""Plate QC keeps its promises on the inputs that sit just off the happy path.

Four behaviours in :mod:`spacr.plate_qc` decide whether a QC pass helps or
misleads, and each is reached only by input the common case never produces:

* a database column that *looks* numeric but holds nothing usable — offering
  it in a measurement picker gives the user an empty heatmap;
* a forced plate format that the data already fits — the grid must be taken
  as given, without the "your geometry is too small" warning that belongs to
  the other case;
* a well value that is infinite — the median it poisons has to be reported as
  *undefined*, never as a difference of zero or a NaN;
* a ring profile capped below the number of rings on the plate — the cap has
  to be honoured rather than run to the core.
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
# Forced plate geometry
# ---------------------------------------------------------------------------

def test_a_forced_plate_format_that_fits_is_used_without_a_warning():
    """Forcing 96 on a 96-well plate is the ordinary case — a user who knows
    their format types it in so an under-pipetted plate is not mis-inferred
    as something smaller. The grid must come back exactly 8x12, and the
    "your forced geometry is smaller than the data" warning must stay away,
    or the warning stops meaning anything when it does appear.
    """
    fits = pq.plate_layout(_long_frame(8, 12), "value", plate_format=96)

    assert fits.attrs["plate_format"] == 96
    assert (fits.attrs["n_rows"], fits.attrs["n_cols"]) == (8, 12)
    assert not [n for n in fits.attrs["notes"] if "Forced" in n]
    assert pq.layout_matrix(fits).shape == (8, 12)

    # The same call on data that overflows the forced format *does* warn, and
    # grows the grid rather than dropping the wells that fall outside it.
    overflows = pq.plate_layout(_long_frame(16, 24), "value", plate_format=96)

    forced_notes = [n for n in overflows.attrs["notes"] if "Forced" in n]
    assert len(forced_notes) == 1
    assert "8x12" in forced_notes[0] and "16x24" in forced_notes[0]
    assert (overflows.attrs["n_rows"], overflows.attrs["n_cols"]) == (16, 24)
    assert pq.layout_matrix(overflows).shape == (16, 24)
    assert len(overflows) == 16 * 24


def test_a_half_filled_384_plate_forced_to_384_keeps_the_real_plate_edge():
    """An assay pipetted only into rows A-H, columns 1-12 of a 384 plate has
    a 96-plate extent, so inference calls it a 96 plate and treats row H and
    column 12 as the plate edge. They are not — they sit in the middle of the
    plastic, where no evaporation happens. Forcing 384 restores the nominal
    grid, and with it the ring geometry every edge statistic is computed on.
    """
    frame = _long_frame(8, 12)

    forced = pq.plate_layout(frame, "value", plate_format=384)
    inferred = pq.plate_layout(frame, "value")

    assert forced.attrs["plate_format"] == 384
    assert (forced.attrs["n_rows"], forced.attrs["n_cols"]) == (16, 24)
    assert not [n for n in forced.attrs["notes"] if "Forced" in n]
    assert len(forced) == 8 * 12                       # the wells that exist
    assert pq.layout_matrix(forced).shape == (16, 24)  # the grid they sit in

    # Row H, column 6 is interior plastic on the 384 grid ...
    row_h = forced[(forced["row_index"] == 8) & (forced["column_index"] == 6)]
    assert not bool(row_h["is_edge"].iloc[0])
    assert int(row_h["ring"].iloc[0]) == 5
    # ... and only row A / column 1 remain on the true outer ring.
    assert int(forced["is_edge"].sum()) == 12 + 8 - 1

    # The contrast: left to infer, the same frame calls row H an edge well.
    assert inferred.attrs["plate_format"] == 96
    inferred_h = inferred[(inferred["row_index"] == 8)
                          & (inferred["column_index"] == 6)]
    assert bool(inferred_h["is_edge"].iloc[0])


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


def test_an_infinite_well_leaves_its_ring_row_undefined_but_still_counted():
    """The ring table is what a user scans to see how far in the artefact
    reaches. A ring whose median is undefined must still show its well count
    and its rank statistics — dropping the row would silently shorten the
    profile and hide that ring 0 was measured at all.
    """
    report = pq.detect_edge_effect(
        _long_frame(8, 12, edge_value=float("inf")), "value")

    outer = report.rings[0]
    assert outer.ring == 0
    assert outer.n_wells == 36
    assert outer.median is None
    assert outer.delta is None and outer.pct is None
    assert outer.cliffs_delta == pytest.approx(1.0)
    # Ring 1 is untouched by the infinity and keeps its numbers.
    inner = report.rings[1]
    assert inner.ring == 1 and inner.n_wells == 28
    assert inner.median == pytest.approx(100.0, abs=5.0)


# ---------------------------------------------------------------------------
# The ring cap
# ---------------------------------------------------------------------------

def test_a_ring_profile_capped_at_one_ring_stops_at_the_outer_ring():
    """``max_rings`` is what keeps the ring table readable on a 1536 plate,
    where the profile would otherwise walk many rings inward. The cap has to
    bound the table even when the plate has more rings left to report and the
    core is still further in — a cap that were ignored would put a wall of
    rows in front of the one number the user came for.
    """
    frame = _long_frame(8, 12, edge_boost=0.30)

    capped = pq.detect_edge_effect(frame, "value", max_rings=1)
    uncapped = pq.detect_edge_effect(frame, "value")

    assert [r.ring for r in capped.rings] == [0]
    assert capped.rings[0].n_wells == 36
    assert capped.rings[0].pct == pytest.approx(30.0, abs=8.0)
    # The plate really does have a second ring; only the cap hid it.
    assert [r.ring for r in uncapped.rings] == [0, 1]
    # Capping the table changes nothing about the headline test itself.
    assert capped.edge_detected and uncapped.edge_detected
    assert capped.pct_difference == pytest.approx(uncapped.pct_difference)
    assert "ring  wells" in pq.format_edge_report(capped)


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

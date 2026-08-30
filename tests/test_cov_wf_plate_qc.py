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


# ---------------------------------------------------------------------------
# Forced geometry larger than anything on the plate
# ---------------------------------------------------------------------------

def test_a_1536_plate_pipetted_in_a_384_footprint_keeps_its_real_edge():
    """Screens routinely fill one 384-well quadrant footprint of a 1536 plate.
    Inference sees a 16x24 extent and calls it a 384 plate, which puts row P
    and column 24 on the outer ring — they are interior plastic, four rings
    deep, and nothing evaporates there. Forcing 1536 has to accept the grid as
    given without the "forced geometry is smaller than the data" warning: that
    warning is the signal the user typed the wrong format, and it means
    nothing if it also fires when the format fits.
    """
    frame = _long_frame(16, 24)

    forced = pq.plate_layout(frame, "value", plate_format=1536)
    inferred = pq.plate_layout(frame, "value")

    assert forced.attrs["plate_format"] == 1536
    assert (forced.attrs["n_rows"], forced.attrs["n_cols"]) == (32, 48)
    assert not [n for n in forced.attrs["notes"] if "Forced" in n]
    assert len(forced) == 16 * 24                      # the wells that exist
    assert pq.layout_matrix(forced).shape == (32, 48)  # the grid they sit in

    # P24 — the corner of the pipetted block — is 15 rings in on the real
    # plate, and only row A / column 1 stay on the true outer ring.
    corner = forced[(forced["row_index"] == 16) & (forced["column_index"] == 24)]
    assert int(corner["ring"].iloc[0]) == 15
    assert not bool(corner["is_edge"].iloc[0])
    assert int(forced["is_edge"].sum()) == 24 + 16 - 1

    # The contrast: left to infer, the same frame calls that corner an edge
    # well and hands the edge test a ring made of interior plastic.
    assert inferred.attrs["plate_format"] == 384
    inferred_corner = inferred[(inferred["row_index"] == 16)
                               & (inferred["column_index"] == 24)]
    assert bool(inferred_corner["is_edge"].iloc[0])
    assert int(inferred["is_edge"].sum()) == 2 * 16 + 2 * 24 - 4


# ---------------------------------------------------------------------------
# An interior median that cannot be computed
# ---------------------------------------------------------------------------

def _with_interior_infinities(n_rows=8, n_cols=12):
    """The 8x12 frame with every *interior* object set to ``inf``.

    The mirror image of ``_long_frame(edge_value=inf)``: here it is the group
    the edge is measured *against* that has no median.
    """
    frame = _long_frame(n_rows, n_cols)
    rc = frame["prc"].str.extract(r"_r(\d+)_c(\d+)$").astype(int)
    rows, cols = rc[0].to_numpy(), rc[1].to_numpy()
    ring = np.minimum.reduce([rows - 1, n_rows - rows, cols - 1, n_cols - cols])
    frame.loc[ring >= 1, "value"] = float("inf")
    return frame


def test_an_infinite_interior_median_leaves_the_difference_undefined():
    """The baseline the edge is compared against can be the poisoned one: a
    single overflowing feature in the middle of the plate is enough. The edge
    median is then a perfectly good number and the interior median is not, so
    the *difference* is unknown. Reporting it as the edge median alone — or as
    zero — would tell the user the ring matches an interior nobody measured.
    The percentage has to stay undefined with it, since there is no baseline
    to divide by.
    """
    report = pq.detect_edge_effect(_with_interior_infinities(), "value")

    assert report.ok                          # both groups are populated
    assert (report.n_edge_wells, report.n_interior_wells) == (36, 60)
    assert report.edge_median == pytest.approx(100.0, abs=5.0)
    assert report.interior_median is None     # inf is not a number to report
    assert report.median_difference is None
    assert report.pct_difference is None
    # The rank test still answers: every interior well outranks every edge one.
    assert report.cliffs_delta == pytest.approx(-1.0)

    text = pq.format_edge_report(report)
    _assert_no_nan_rendered(text)
    assert "interior median     undefined" in text
    assert "median difference   undefined" in text
    assert "undetermined amount" in text
    # And the edge median it *could* compute is still printed, so the report
    # says which half of the comparison failed.
    assert re.search(r"edge median\s+\d", text), text

    # The contrast: a finite interior gives both medians and a real difference.
    finite = pq.detect_edge_effect(_long_frame(8, 12, edge_boost=0.30), "value")
    assert finite.interior_median == pytest.approx(100.0, abs=5.0)
    assert finite.median_difference == pytest.approx(30.0, abs=8.0)


# ---------------------------------------------------------------------------
# How deep "the core" is
# ---------------------------------------------------------------------------

def test_a_deeper_core_reports_one_more_ring_before_the_profile_stops():
    """The ring profile stops where the rings become the core it compares
    against — past that point a ring would be tested against itself and every
    row after it would read as "no difference" for arithmetic reasons rather
    than biological ones. ``core_depth`` is therefore what decides how far the
    table runs: at depth 3 the user gets ring 2 as a real comparison against
    the innermost 12 wells, and at the default depth 2 that row is correctly
    withheld.
    """
    frame = _long_frame(8, 12, edge_boost=0.30)

    deep = pq.detect_edge_effect(frame, "value", core_depth=3)
    default = pq.detect_edge_effect(frame, "value")

    assert [r.ring for r in deep.rings] == [0, 1, 2]
    assert [r.n_wells for r in deep.rings] == [36, 28, 20]
    assert deep.rings[0].pct == pytest.approx(30.0, abs=8.0)
    # Rings 1 and 2 are ordinary interior wells: no boost was applied there.
    assert deep.rings[1].pct == pytest.approx(0.0, abs=5.0)
    assert deep.rings[2].pct == pytest.approx(0.0, abs=5.0)
    # The shallower default core swallows ring 2 rather than comparing it
    # against wells it contains.
    assert [r.ring for r in default.rings] == [0, 1]
    # Neither depth changes the headline test.
    assert deep.edge_detected and default.edge_detected
    assert deep.pct_difference == pytest.approx(default.pct_difference)


def test_a_core_deeper_than_the_plate_says_which_depth_it_fell_back_to():
    """Asking for a core 5 rings in on a 96-well plate asks for wells that do
    not exist — the deepest ring there is 3. Silently profiling against an
    empty or two-well core would put statistics in the table computed from
    almost nothing. The profile backs off to the deepest usable core and has
    to say so, because "vs core" means something different afterwards.
    """
    frame = _long_frame(8, 12, edge_boost=0.30)

    fallback = pq.detect_edge_effect(frame, "value", core_depth=5)

    notes = [n for n in fallback.notes if "core" in n]
    assert len(notes) == 1
    assert "at least 5" in notes[0] and "depth >= 3" in notes[0]
    # The table itself is still the full three rings outside that core.
    assert [r.ring for r in fallback.rings] == [0, 1, 2]
    assert [r.n_wells for r in fallback.rings] == [36, 28, 20]
    assert fallback.rings[0].pct == pytest.approx(30.0, abs=8.0)
    # A core the plate can supply produces no such note, so the note above is
    # the impossible depth's doing.
    assert not [n for n in pq.detect_edge_effect(
        frame, "value", core_depth=3).notes if "core" in n]

"""Plate QC — the edge-effect detector and the well grid behind it.

Everything here is synthetic and deterministic: a plate is built with a
*known* artefact baked into it, and the test asserts the module recovers
that number rather than merely "says something". The test that matters
most is :func:`test_a_clean_plate_is_never_flagged` — a detector that
always fires is worse than no detector, because it teaches the user to
ignore it.

The whole module is also pinned as dependency-light: importing it must
not drag in torch or cellpose, so the Qt screen can draw a plate without
paying a multi-second import.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from spacr.plate_qc import (
    DEFAULT_MIN_EFFECT,
    EdgeEffectReport,
    GROUPINGS,
    LAYOUT_COLUMNS,
    PLATE_FORMATS,
    colour_limits,
    detect_edge_effect,
    format_edge_report,
    infer_plate_format,
    layout_matrix,
    load_plate_frame,
    numeric_columns,
    parse_column_label,
    parse_row_label,
    plate_layout,
    plates_in,
    row_column_trends,
    row_label,
    table_columns,
    tables,
    well_id,
    write_layout_csv,
)


# ---------------------------------------------------------------------------
# Synthetic plates
# ---------------------------------------------------------------------------

def assert_no_nan_in_text(text):
    """No statistic may be rendered as ``nan``.

    Word-bounded on purpose: "dominant" contains the letters and is not
    a NaN.
    """
    assert not re.search(r"\bnan\b", text, re.IGNORECASE), text


def synth_plate(n_rows=16, n_cols=24, *, edge_boost=0.0, ring1_boost=0.0,
                col_slope=0.0, row_slope=0.0, n_objects=8, seed=0,
                plate="plate1", base=100.0, noise=0.12, thin_wells=(),
                thin_n=2):
    """A long per-object frame with a known artefact baked in.

    Noise is multiplicative log-normal — the shape per-well aggregates of
    object measurements actually have, and the reason the module tests
    ranks rather than means. ``exp(N(0, s))`` has median 1, so a well's
    median lands on its intended multiplier and an asserted "+30 %" is a
    real +30 %.

    :param edge_boost: fractional lift applied to ring 0.
    :param ring1_boost: fractional lift applied to ring 1.
    :param col_slope: fractional lift per column step (a gradient).
    :param row_slope: fractional lift per row step.
    :param thin_wells: ``(row, column)`` pairs given only ``thin_n``
        objects, for the ``min_count`` tests.
    """
    rng = np.random.default_rng(seed)
    thin = {tuple(w) for w in thin_wells}
    records = []
    for r in range(1, n_rows + 1):
        for c in range(1, n_cols + 1):
            ring = min(r - 1, n_rows - r, c - 1, n_cols - c)
            mult = 1.0
            if ring == 0:
                mult += edge_boost
            if ring == 1:
                mult += ring1_boost
            mult += col_slope * (c - 1) + row_slope * (r - 1)
            count = thin_n if (r, c) in thin else n_objects
            for _ in range(count):
                records.append({
                    "prc": f"{plate}_r{r}_c{c}",
                    "value": base * mult * float(rng.lognormal(0.0, noise)),
                })
    return pd.DataFrame(records)


@pytest.fixture
def measdb(tmp_path):
    """A real ``<src>/measurements/measurements.db`` with a 96-well plate."""
    src = tmp_path / "plate1"
    meas = src / "measurements"
    meas.mkdir(parents=True)
    db_path = meas / "measurements.db"
    frame = synth_plate(8, 12, edge_boost=0.35, n_objects=6, seed=11)
    frame["object_label"] = range(1, len(frame) + 1)
    frame["well_note"] = "text-not-a-number"

    con = sqlite3.connect(db_path)
    try:
        con.execute("CREATE TABLE cell (prc TEXT, value REAL, "
                    "object_label INTEGER, well_note TEXT)")
        con.executemany(
            "INSERT INTO cell VALUES (?, ?, ?, ?)",
            list(frame[["prc", "value", "object_label", "well_note"]]
                 .itertuples(index=False, name=None)))
        con.execute("CREATE TABLE notes (comment TEXT)")
        con.execute("INSERT INTO notes VALUES ('no numbers here')")
        con.commit()
    finally:
        con.close()
    return str(db_path)


# ---------------------------------------------------------------------------
# The headline: a known edge ring
# ---------------------------------------------------------------------------

def test_a_known_30_percent_edge_ring_is_detected_at_the_right_size():
    """A +30 % outer ring must come back as +30 %, not just as 'p < 0.05'."""
    df = synth_plate(edge_boost=0.30, seed=1)
    report = detect_edge_effect(df, "value")

    assert report.ok
    assert report.edge_detected
    assert report.dominant == "edge"
    # The number the user acts on, within sampling noise of 76 vs 308 wells.
    assert report.pct_difference == pytest.approx(30.0, abs=4.0)
    # Rank effect size, not just significance.
    assert report.cliffs_delta > 0.5
    assert report.p_value < 1e-6
    assert report.n_edge_wells == 76
    assert report.n_interior_wells == 308
    assert report.plate_format == 384


def test_the_effect_size_leads_the_p_value_in_the_text():
    """With 384 wells the p-value is the least informative number here."""
    report = detect_edge_effect(synth_plate(edge_boost=0.30, seed=1), "value")
    text = format_edge_report(report)
    assert "%" in text.splitlines()[3]        # the verdict line
    assert text.index("median difference") < text.index("Mann-Whitney p")
    assert "Cliff's delta" in text
    assert_no_nan_in_text(text)


def test_a_downward_edge_is_detected_too():
    """Evaporation concentrates; some assays read *lower* at the edge."""
    report = detect_edge_effect(synth_plate(edge_boost=-0.25, seed=4), "value")
    assert report.edge_detected
    assert report.pct_difference == pytest.approx(-25.0, abs=4.0)
    assert report.cliffs_delta < -0.5


# ---------------------------------------------------------------------------
# The test that matters most: no false alarms
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", list(range(12)))
def test_a_clean_plate_is_never_flagged(seed):
    """A detector that always fires teaches the user to ignore it.

    Twelve independent clean plates, all of which must come back quiet —
    both for the edge and for the gradient. Requiring an effect size as
    well as a p-value is what makes this hold: on 384 wells, ``p < 0.05``
    alone would fire on roughly one plate in twenty.
    """
    report = detect_edge_effect(synth_plate(seed=100 + seed), "value")
    assert report.ok
    assert not report.edge_detected, report.summary
    assert not report.gradient_detected, report.summary
    assert report.dominant == "none"
    assert abs(report.pct_difference) < 5.0
    assert abs(report.cliffs_delta) < DEFAULT_MIN_EFFECT


def test_the_effect_size_gate_is_what_stops_a_trivial_drift_being_a_hit():
    """A 1.7 % drift across 384 wells clears p and is still not a hit.

    This is the reason detection needs an effect size at all: the very
    same plate is "significant" the moment the gate is dropped, and a
    user shown that verdict would re-plate a screen over 1.7 %.
    """
    df = synth_plate(edge_boost=0.010, seed=26)
    report = detect_edge_effect(df, "value")
    assert report.p_value < 0.05                          # easy
    assert abs(report.cliffs_delta) < DEFAULT_MIN_EFFECT  # not
    assert not report.edge_detected
    assert abs(report.pct_difference) < 4.0
    assert detect_edge_effect(df, "value", min_effect=0.0).edge_detected


# ---------------------------------------------------------------------------
# Gradients are not edge effects
# ---------------------------------------------------------------------------

def test_a_pure_column_gradient_is_a_gradient_not_an_edge_effect():
    """A plate-reader gradient runs across the plate; evaporation runs around it.

    A linear column gradient leaves the outer ring straddling both the low
    and the high end, so its median matches the interior and the ring test
    correctly stays quiet while Spearman on the column index fires.
    """
    report = detect_edge_effect(synth_plate(col_slope=0.05, seed=2), "value")

    assert report.ok
    assert not report.edge_detected, report.summary
    assert report.gradient_detected
    assert report.dominant == "gradient"

    column = report.gradient("column")
    row = report.gradient("row")
    assert column.detected and column.spearman_rho > 0.8
    assert not row.detected
    assert column.pct_first_last > 50.0
    assert "gradient" in format_edge_report(report)


def test_a_row_gradient_is_reported_on_the_row_axis():
    report = detect_edge_effect(synth_plate(row_slope=0.06, seed=3), "value")
    assert report.gradient("row").detected
    assert not report.gradient("column").detected
    assert not report.edge_detected
    assert report.gradient("row").first_label == "A"
    assert report.gradient("row").last_label == "P"


def test_an_edge_ring_and_a_gradient_can_both_be_reported():
    """They are separate questions, so both answers must survive."""
    report = detect_edge_effect(
        synth_plate(edge_boost=0.35, col_slope=0.04, seed=5), "value")
    assert report.edge_detected
    assert report.gradient("column").detected
    assert report.dominant in ("edge", "gradient")


# ---------------------------------------------------------------------------
# Ring-by-ring
# ---------------------------------------------------------------------------

def test_a_two_ring_gradient_shows_up_ring_by_ring():
    """Evaporation is not a step function — ring 1 must be visible too."""
    df = synth_plate(edge_boost=0.40, ring1_boost=0.20, seed=6)
    report = detect_edge_effect(df, "value")

    rings = {r.ring: r for r in report.rings}
    assert set(rings) == {0, 1}
    assert rings[0].n_wells == 76
    assert rings[1].n_wells == 68
    assert rings[0].pct == pytest.approx(40.0, abs=5.0)
    assert rings[1].pct == pytest.approx(20.0, abs=5.0)
    # Monotonic inward decay, each ring individually significant.
    assert rings[0].pct > rings[1].pct > 5.0
    assert rings[0].cliffs_delta > rings[1].cliffs_delta > 0.3
    assert rings[0].p_value < 0.01 and rings[1].p_value < 0.01
    assert "Ring profile" in format_edge_report(report)


def test_the_ring_profile_never_compares_a_ring_against_itself():
    """Ring 1 is measured against a core that excludes ring 1."""
    report = detect_edge_effect(synth_plate(edge_boost=0.3, seed=6), "value",
                                core_depth=2)
    assert max(r.ring for r in report.rings) < 2


def test_a_plate_with_no_room_for_a_core_falls_back_and_says_so():
    """A 4x6 plate has almost nothing two wells in from every edge."""
    report = detect_edge_effect(synth_plate(4, 6, edge_boost=0.4, seed=8),
                                "value")
    assert report.ok
    assert report.rings
    assert any("core" in n for n in report.notes)


# ---------------------------------------------------------------------------
# Plate format inference
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_rows,n_cols,expected", [
    (8, 12, 96), (16, 24, 384), (32, 48, 1536),
])
def test_the_plate_format_is_inferred_from_the_labels(n_rows, n_cols, expected):
    """96 / 384 / 1536 come from the observed labels, never hard-coded."""
    frame = pd.DataFrame([
        {"well": f"{row_label(r)}{c:02d}", "value": 1.0 + r + c}
        for r in range(1, n_rows + 1) for c in range(1, n_cols + 1)])
    layout = plate_layout(frame, "value")
    assert layout.attrs["plate_format"] == expected
    assert (layout.attrs["n_rows"], layout.attrs["n_cols"]) == (n_rows, n_cols)
    assert len(layout) == expected
    assert any(f"{expected}-well" in n for n in layout.attrs["notes"])


def test_a_half_used_384_plate_does_not_get_a_fake_bottom_edge():
    """Rows A-H of a 384 plate: row H is interior, not the plate edge."""
    frame = synth_plate(8, 24, seed=9)
    layout = plate_layout(frame, "value")
    assert layout.attrs["plate_format"] == 384
    assert layout.attrs["n_rows"] == 16
    bottom = layout[layout["row_index"] == 8]
    # Only the two wells in columns 1 and 24 are on a real edge.
    assert int(bottom["is_edge"].sum()) == 2


def test_infer_plate_format_snaps_up_and_gives_up_honestly():
    assert infer_plate_format(8, 12) == (96, 8, 12)
    assert infer_plate_format(5, 9) == (96, 8, 12)
    assert infer_plate_format(1, 1) == (6, 2, 3)
    assert infer_plate_format(40, 60) == (None, 40, 60)
    assert [w for w, _ in PLATE_FORMATS] == sorted(w for w, _ in PLATE_FORMATS)


def test_a_non_standard_grid_is_labelled_as_such():
    frame = pd.DataFrame([{"prc": f"p_r{r}_c{c}", "value": 1.0}
                          for r in range(1, 41) for c in range(1, 61)])
    layout = plate_layout(frame, "value")
    assert layout.attrs["plate_format"] is None
    assert any("no standard plate format" in n for n in layout.attrs["notes"])


# ---------------------------------------------------------------------------
# min_count
# ---------------------------------------------------------------------------

def test_min_count_drops_low_n_wells_and_reports_how_many():
    """A heatmap silently missing wells looks exactly like data."""
    thin = [(1, 1), (2, 2), (3, 3), (8, 8), (16, 24)]
    df = synth_plate(edge_boost=0.3, seed=10, thin_wells=thin, thin_n=2)

    kept_all = plate_layout(df, "value", min_count=0)
    assert len(kept_all) == 384
    assert kept_all.attrs["n_dropped_min_count"] == 0

    filtered = plate_layout(df, "value", min_count=5)
    assert len(filtered) == 384 - len(thin)
    assert filtered.attrs["n_dropped_min_count"] == len(thin)
    assert any("dropped" in n and "not zero" in n
               for n in filtered.attrs["notes"])

    report = detect_edge_effect(df, "value", min_count=5)
    assert report.n_dropped_min_count == len(thin)
    assert report.n_wells == 384 - len(thin)
    assert f"{len(thin)} well(s) dropped" in format_edge_report(report)


def test_filtering_every_well_away_invents_no_grid():
    """An empty extent must not be inferred as a 2x3 plate."""
    layout = plate_layout(synth_plate(8, 12, seed=11), "value", min_count=999)
    assert len(layout) == 0
    assert layout.attrs["plate_format"] is None
    assert (layout.attrs["n_rows"], layout.attrs["n_cols"]) == (0, 0)
    assert layout.attrs["n_dropped_min_count"] == 96
    assert any("lower min_count" in n for n in layout.attrs["notes"])
    assert layout_matrix(layout).shape == (0, 0)

    report = detect_edge_effect(synth_plate(8, 12, seed=11), "value",
                                min_count=999)
    assert not report.ok
    assert report.n_wells == 0
    assert report.n_dropped_min_count == 96
    _no_nan_anywhere(report)


def test_a_dropped_well_is_blank_in_the_matrix_never_zero():
    """``generate_plate_heatmap`` fills missing wells with 0; this must not."""
    df = synth_plate(8, 12, seed=12, thin_wells=[(4, 5)], thin_n=1)
    grid = layout_matrix(plate_layout(df, "value", min_count=3))
    assert math.isnan(grid.iat[3, 4])
    assert grid.notna().sum().sum() == 95
    assert grid.shape == (8, 12)
    assert list(grid.index[:3]) == ["A", "B", "C"]


# ---------------------------------------------------------------------------
# Degenerate input explains itself
# ---------------------------------------------------------------------------

def _no_nan_anywhere(report: EdgeEffectReport):
    """Every float on the report is finite; undefined is spelled ``None``."""
    seen = []
    for value in vars(report).values():
        if isinstance(value, float):
            seen.append(value)
        elif isinstance(value, list):
            for item in value:
                seen += [v for v in vars(item).values()
                         if isinstance(v, float)] if hasattr(item, "__dict__") else []
    assert not any(math.isnan(v) for v in seen), seen


def test_an_empty_frame_explains_itself_instead_of_crashing():
    report = detect_edge_effect(pd.DataFrame(columns=["prc", "value"]), "value")
    assert not report.ok
    assert not report.edge_detected
    assert report.n_wells == 0
    assert report.p_value is None and report.cliffs_delta is None
    assert "empty" in report.summary.lower()
    _no_nan_anywhere(report)
    assert_no_nan_in_text(format_edge_report(report))


def test_a_single_well_says_it_has_nothing_to_compare():
    report = detect_edge_effect(
        pd.DataFrame([{"prc": "p_r1_c1", "value": 3.0}]), "value")
    assert not report.ok
    assert report.n_wells == 1
    assert "A01" in report.summary
    assert "compare" in report.summary
    _no_nan_anywhere(report)


def test_all_identical_values_are_explained_not_flagged():
    """No variation means nothing to attribute to the edge — not a hit."""
    frame = pd.DataFrame([{"prc": f"p_r{r}_c{c}", "value": 5.0}
                          for r in range(1, 9) for c in range(1, 13)])
    report = detect_edge_effect(frame, "value")
    assert report.ok
    assert not report.edge_detected
    assert not report.gradient_detected
    assert report.p_value == 1.0
    assert report.cliffs_delta == 0.0
    assert any("no variation" in n for n in report.notes)
    _no_nan_anywhere(report)
    assert_no_nan_in_text(format_edge_report(report))


def test_a_plate_with_no_interior_says_so():
    """A 2x3 plate is all edge — there is nothing to compare it against."""
    frame = pd.DataFrame([{"prc": f"p_r{r}_c{c}", "value": float(r + c)}
                          for r in range(1, 3) for c in range(1, 4)])
    report = detect_edge_effect(frame, "value")
    assert not report.ok
    assert report.n_interior_wells == 0
    assert "interior" in report.summary
    _no_nan_anywhere(report)


def test_a_zero_interior_median_reports_absolute_units_only():
    """'+300 % of nothing' is not a number to show a user."""
    frame = pd.DataFrame(
        [{"prc": f"p_r{r}_c{c}",
          "value": 4.0 if min(r - 1, 8 - r, c - 1, 12 - c) == 0 else 0.0}
         for r in range(1, 9) for c in range(1, 13)])
    report = detect_edge_effect(frame, "value")
    assert report.pct_difference is None
    assert report.median_difference == pytest.approx(4.0)
    assert any("absolute units" in n for n in report.notes)
    # The verdict falls back to absolute units rather than printing a
    # percentage of nothing, and the ring table says "undefined" outright.
    assert "in absolute units" in report.summary
    assert "undefined" in format_edge_report(report)
    assert_no_nan_in_text(format_edge_report(report))


def test_wells_whose_value_is_all_nan_are_left_blank_and_counted():
    frame = synth_plate(8, 12, seed=13)
    frame.loc[frame["prc"] == "plate1_r4_c5", "value"] = np.nan
    report = detect_edge_effect(frame, "value")
    assert report.n_wells == 95
    assert any("no usable" in n for n in report.notes)


# ---------------------------------------------------------------------------
# Well / row / column labels
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label,expected", [
    ("r3", 3), ("R3", 3), ("row3", 3), ("3", 3), (3, 3),
    ("C", 3), ("c", 3), ("AA", 27), ("AF", 32), ("P", 16),
    ("", None), (None, None), ("banana", None), ("r0", None), (float("nan"), None),
])
def test_parse_row_label(label, expected):
    assert parse_row_label(label) == expected


@pytest.mark.parametrize("label,expected", [
    ("c12", 12), ("C12", 12), ("column12", 12), ("12", 12), (12, 12),
    ("", None), (None, None), ("A", None), ("c0", None),
])
def test_parse_column_label(label, expected):
    assert parse_column_label(label) == expected


@pytest.mark.parametrize("row,col,expected", [
    (1, 1, "A01"), (3, 7, "C07"), (16, 24, "P24"), (32, 48, "AF48"),
])
def test_well_id(row, col, expected):
    assert well_id(row, col) == expected


@pytest.mark.parametrize("row,col,expected", [
    (0, 5, "?05"),          # no such row
    (3, 0, "C00"),          # no such column
    (-2, 3, "?03"),
])
def test_well_id_renders_an_impossible_position_rather_than_raising(
        row, col, expected):
    """A layout table has to render every cell it was handed.

    :func:`spacr.schema.well_id` is the definition and it *raises* for a
    position that is not a well — right for a key, wrong for a report. This
    wrapper keeps the ``'?'`` so a QC figure still draws.
    """
    assert well_id(row, col) == expected


def test_well_id_agrees_with_schema_on_every_real_well():
    """One definition: QC and the database must name a well the same way."""
    from spacr import schema

    for row in range(1, 33):
        for col in (1, 12, 24, 48):
            assert well_id(row, col) == schema.well_id(row, col)


def test_plate_formats_are_schemas_plate_formats():
    """Two copies of the plate geometry is two answers to "does c30 exist?"."""
    from spacr import schema

    assert dict(PLATE_FORMATS) == schema.PLATE_FORMATS
    assert [n for n, _ in PLATE_FORMATS] == sorted(schema.PLATE_FORMATS)


def test_wells_can_come_from_prc_from_row_column_or_from_a_well_column():
    """All three spellings must land on the same grid."""
    from_prc = plate_layout(
        pd.DataFrame([{"prc": "p1_r2_c3", "value": 7.0}]), "value")
    from_ids = plate_layout(
        pd.DataFrame([{"plateID": "p1", "rowID": "r2", "columnID": "c3",
                       "value": 7.0}]), "value")
    from_well = plate_layout(
        pd.DataFrame([{"plate": "p1", "well": "B03", "value": 7.0}]), "value")
    for layout in (from_prc, from_ids, from_well):
        assert layout.iloc[0]["well"] == "B03"
        assert (layout.iloc[0]["row_index"], layout.iloc[0]["column_index"]) == (2, 3)
        assert layout.iloc[0]["plateID"] == "p1"


def test_a_four_token_prc_still_finds_the_row_and_column():
    """``plateID_rowID_columnID_fieldID`` is written all over spaCR."""
    frame = pd.DataFrame([{"prc": f"plate1_r{r}_c{c}_f1", "value": 1.0}
                          for r in range(1, 9) for c in range(1, 13)])
    layout = plate_layout(frame, "value")
    assert len(layout) == 96
    assert layout.attrs["plate"] == "plate1"


def test_an_unreadable_prc_is_refused_with_an_explanation():
    frame = pd.DataFrame([{"prc": "one_two_three", "value": 1.0}])
    with pytest.raises(ValueError, match="plateID_rowID_columnID"):
        plate_layout(frame, "value")


def test_a_frame_with_no_well_identifier_is_refused():
    with pytest.raises(ValueError, match="No well identifier"):
        plate_layout(pd.DataFrame([{"value": 1.0}]), "value")


def test_unparseable_labels_are_skipped_and_counted():
    frame = pd.DataFrame([
        {"plateID": "p1", "rowID": "r1", "columnID": "c1", "value": 1.0},
        {"plateID": "p1", "rowID": "???", "columnID": "c2", "value": 2.0},
    ])
    layout = plate_layout(frame, "value")
    assert len(layout) == 1
    assert layout.attrs["n_unparsed_rows"] == 1
    assert any("could not be read" in n for n in layout.attrs["notes"])


# ---------------------------------------------------------------------------
# Plates + groupings
# ---------------------------------------------------------------------------

def test_multiple_plates_are_listed_and_the_first_is_used_by_default():
    frame = pd.concat([synth_plate(8, 12, seed=14, plate="plateA"),
                       synth_plate(8, 12, seed=15, plate="plateB")])
    assert plates_in(frame) == ["plateA", "plateB"]
    default = plate_layout(frame, "value")
    assert default.attrs["plate"] == "plateA"
    assert any("showing" in n for n in default.attrs["notes"])
    assert plate_layout(frame, "value", plate="plateB").attrs["plate"] == "plateB"


def test_asking_for_a_plate_that_is_not_there_says_which_ones_are():
    layout = plate_layout(synth_plate(4, 6, seed=16), "value", plate="nope")
    assert len(layout) == 0
    assert any("Present: plate1" in n for n in layout.attrs["notes"])


def test_plates_in_never_raises_on_junk():
    assert plates_in(pd.DataFrame()) == []
    assert plates_in(None) == []
    assert plates_in(pd.DataFrame([{"value": 1}])) == []


@pytest.mark.parametrize("grouping", GROUPINGS)
def test_every_grouping_produces_a_grid(grouping):
    layout = plate_layout(synth_plate(4, 6, seed=17), "value",
                          grouping=grouping)
    assert len(layout) == 24
    assert list(layout.columns) == list(LAYOUT_COLUMNS)
    assert layout["n"].min() == 8


def test_count_grouping_needs_no_value_column():
    layout = plate_layout(synth_plate(4, 6, seed=18), grouping="count")
    assert (layout["value"] == layout["n"]).all()
    assert layout.attrs["value_col"] is None


def test_an_unknown_grouping_or_missing_column_is_refused():
    df = synth_plate(4, 6, seed=19)
    with pytest.raises(ValueError, match="grouping must be"):
        plate_layout(df, "value", grouping="fancy")
    with pytest.raises(ValueError, match="No column 'absent'"):
        plate_layout(df, "absent")
    with pytest.raises(ValueError, match="needs a value column"):
        plate_layout(df, None, grouping="mean")


def test_a_layout_passed_back_in_is_returned_unchanged():
    """The Qt screen recomputes from a layout it already holds."""
    layout = plate_layout(synth_plate(8, 12, seed=20), "value")
    again = plate_layout(layout, "value")
    pd.testing.assert_frame_equal(layout, again)
    assert again.attrs["plate_format"] == 96


# ---------------------------------------------------------------------------
# Row / column trends
# ---------------------------------------------------------------------------

def test_row_column_trends_summarises_both_axes():
    trends = row_column_trends(synth_plate(8, 12, col_slope=0.05, seed=21),
                               "value")
    assert (trends["axis"] == "row").sum() == 8
    assert (trends["axis"] == "column").sum() == 12
    assert list(trends[trends["axis"] == "row"]["label"])[:3] == ["A", "B", "C"]
    assert (trends["n_wells"] > 0).all()
    assert (trends["n_objects"] == trends["n_wells"] * 8).all()
    columns = trends[trends["axis"] == "column"]
    assert columns["spearman_rho"].iloc[0] > 0.8
    assert columns["mean"].is_monotonic_increasing


def test_row_column_trends_on_an_empty_plate_is_empty_not_broken():
    trends = row_column_trends(pd.DataFrame(columns=["prc", "value"]), "value")
    assert len(trends) == 0
    assert "spearman_rho" in trends.columns


def test_row_column_trends_reports_thin_rows_honestly():
    """A row average over three wells is not the same claim as over twelve."""
    df = synth_plate(8, 12, seed=22,
                     thin_wells=[(1, c) for c in range(4, 13)], thin_n=1)
    trends = row_column_trends(df, "value", min_count=4)
    row_a = trends[(trends["axis"] == "row") & (trends["label"] == "A")].iloc[0]
    assert row_a["n_wells"] == 3


# ---------------------------------------------------------------------------
# Colour limits + CSV
# ---------------------------------------------------------------------------

def test_colour_limits_understands_the_same_spec_as_plot_py():
    layout = plate_layout(synth_plate(8, 12, seed=23), "value")
    values = layout["value"].to_numpy(float)
    assert colour_limits(layout, "all") == (float(values.min()), float(values.max()))
    lo, hi = colour_limits(layout, "allq")
    assert lo > values.min() and hi < values.max()
    assert colour_limits(layout, [10.0, 200.0]) == (10.0, 200.0)
    q_lo, q_hi = colour_limits(layout, [0.25, 0.75])
    assert q_lo == pytest.approx(float(np.quantile(values, 0.25)))
    assert q_hi == pytest.approx(float(np.quantile(values, 0.75)))


def test_colour_limits_never_returns_a_degenerate_range():
    empty = plate_layout(pd.DataFrame(columns=["prc", "value"]), "value")
    assert colour_limits(empty, "all") == (0.0, 1.0)
    flat = plate_layout(pd.DataFrame([{"prc": f"p_r{r}_c1", "value": 2.0}
                                      for r in range(1, 5)]), "value")
    lo, hi = colour_limits(flat, "all")
    assert hi > lo


def test_write_layout_csv_round_trips_the_grid(tmp_path):
    layout = plate_layout(synth_plate(8, 12, seed=24), "value", min_count=3)
    out = write_layout_csv(layout, str(tmp_path / "sub" / "wells.csv"))
    assert os.path.isfile(out)
    back = pd.read_csv(out)
    assert list(back.columns) == list(LAYOUT_COLUMNS)
    assert len(back) == len(layout)
    assert back["well"].iloc[0] == "A01"
    assert set(back["ring"]) <= set(layout["ring"])
    with pytest.raises(ValueError, match="No output path"):
        write_layout_csv(layout, "")


# ---------------------------------------------------------------------------
# Database access — read-only, narrow
# ---------------------------------------------------------------------------

def test_tables_and_numeric_columns_from_a_real_database(measdb):
    assert tables(measdb) == ["cell", "notes"]
    assert table_columns(measdb, "cell") == [
        "prc", "value", "object_label", "well_note"]
    numeric = numeric_columns(measdb, "cell")
    assert "value" in numeric
    assert "well_note" not in numeric      # text
    assert "prc" not in numeric            # an identifier, not a measurement
    assert numeric_columns(measdb, "notes") == []
    with pytest.raises(ValueError, match="no 'ghost' table"):
        table_columns(measdb, "ghost")


def test_load_plate_frame_reads_only_what_it_needs(measdb):
    frame = load_plate_frame(measdb, "cell", "value")
    assert list(frame.columns) == ["prc", "value"]
    assert len(frame) == 96 * 6
    assert len(load_plate_frame(measdb, "cell", "value", limit=10)) == 10
    with pytest.raises(ValueError, match="no column 'nope'"):
        load_plate_frame(measdb, "cell", "nope")
    with pytest.raises(ValueError, match="no well identifier"):
        load_plate_frame(measdb, "notes", "comment")


def test_the_database_is_opened_read_only(measdb):
    """A write must be refused by SQLite itself, not by a check we wrote."""
    before = hashlib.sha256(open(measdb, "rb").read()).hexdigest()
    report = detect_edge_effect(load_plate_frame(measdb, "cell", "value"),
                                "value")
    assert report.edge_detected
    from spacr.plate_qc import _connect
    con = _connect(measdb)
    try:
        with pytest.raises(sqlite3.OperationalError):
            con.execute("UPDATE cell SET value = 0")
    finally:
        con.close()
    assert hashlib.sha256(open(measdb, "rb").read()).hexdigest() == before
    assert sorted(os.listdir(os.path.dirname(measdb))) == ["measurements.db"]


def test_connect_says_which_file_is_missing(tmp_path):
    from spacr.plate_qc import _connect
    with pytest.raises(FileNotFoundError, match="ghost.db"):
        _connect(str(tmp_path / "ghost.db"))
    with pytest.raises(ValueError, match="No database path"):
        _connect("")


# ---------------------------------------------------------------------------
# Contract with spacr.plot — one heatmap, not two
# ---------------------------------------------------------------------------

def test_the_well_grid_matches_generate_plate_heatmap_well_for_well():
    """The anti-drift pin.

    ``spacr.plot.generate_plate_heatmap`` is the existing implementation;
    this module deliberately re-implements the aggregation so the Qt
    screen does not have to import torch to draw a plate. On a full 384
    plate — where the original's ``fillna(0)`` and hard-coded r1..r16 /
    c1..c27 grid make no difference — the two must agree exactly, and the
    colour limits with them. If this test ever fails, the two have drifted
    and one of them is now lying about a plate.
    """
    plot = pytest.importorskip("spacr.plot")
    df = synth_plate(edge_boost=0.2, seed=25)

    theirs, (their_min, their_max) = plot.generate_plate_heatmap(
        df.copy(), "plate1", "value", "mean", "all", 0)
    layout = plate_layout(df, "value", grouping="mean")
    mine = layout_matrix(layout)

    assert mine.shape == theirs.shape == (16, 24)
    np.testing.assert_allclose(mine.to_numpy(float), theirs.to_numpy(float))
    assert colour_limits(layout, "all") == (their_min, their_max)

    _, (their_qmin, their_qmax) = plot.generate_plate_heatmap(
        df.copy(), "plate1", "value", "mean", "allq", 0)
    assert colour_limits(layout, "allq") == pytest.approx(
        (their_qmin, their_qmax))


# ---------------------------------------------------------------------------
# Dependency weight
# ---------------------------------------------------------------------------

def test_importing_plate_qc_pulls_in_neither_torch_nor_cellpose():
    """Drawing a plate must not cost a multi-second torch import.

    Measured as "modules added by the import" in a fresh interpreter
    rather than "torch is absent", so a conftest or sitecustomize that
    pre-imports torch for unrelated reasons cannot make this pass or fail
    by accident.
    """
    code = (
        "import sys, json\n"
        "before = set(sys.modules)\n"
        "import spacr.plate_qc\n"
        "print(json.dumps(sorted(set(sys.modules) - before)))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr
    added = json.loads(proc.stdout.strip().splitlines()[-1])
    for heavy in ("torch", "cellpose", "torchvision", "cv2", "matplotlib",
                  "seaborn", "skimage", "PySide6"):
        assert not any(m == heavy or m.startswith(heavy + ".") for m in added), \
            f"importing spacr.plate_qc dragged in {heavy}"


def test_plate_qc_does_not_reference_torch_or_cellpose():
    import spacr.plate_qc as module
    source = open(module.__file__).read()
    assert "import torch" not in source
    assert "import cellpose" not in source
    assert "from cellpose" not in source

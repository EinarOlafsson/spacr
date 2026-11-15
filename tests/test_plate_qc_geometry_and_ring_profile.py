"""Plate QC lays wells on the real plate and says "undefined" when it must.

Two units of :mod:`spacr.plate_qc`:

* :func:`~spacr.plate_qc.plate_layout` -- unreadable labels leave an empty
  layout that counts them, an impossible plate format is refused by name, a
  forced geometry that is too small grows and says so, one that fits (or is
  larger than the pipetted footprint) is used silently and keeps the real
  plate edge; reversed colour limits are put back in order.
* :func:`~spacr.plate_qc.detect_edge_effect` -- the ring profile honours its
  cap and its core depth (and says which depth it fell back to), and an
  infinite edge or interior leaves the medians and difference undefined
  while the rank statistics still answer.

The synthetic plates are seeded, so every statistic is pinned exactly.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from spacr import plate_qc as pq

_PRC_NOTE = "Wells located from prc."
_INFERRED_96 = ("Inferred a 96-well plate (8x12) from the observed labels "
                "(rows to H, columns to 12).")


def _approx(value):
    return pytest.approx(value, rel=1e-9)


def _long_frame(n_rows=8, n_cols=12, *, edge_boost=0.0, plate="plate1",
                n_objects=4, seed=0, edge_value=None):
    """A long per-object frame for one plate, one row per object.

    ``edge_boost`` lifts the outer ring by a known fraction; ``edge_value``
    instead pins every outer-ring object to a fixed value.
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


def _with_interior_infinities(n_rows=8, n_cols=12):
    """The 8x12 frame with every interior object set to ``inf``."""
    frame = _long_frame(n_rows, n_cols)
    rc = frame["prc"].str.extract(r"_r(\d+)_c(\d+)$").astype(int)
    rows, cols = rc[0].to_numpy(), rc[1].to_numpy()
    ring = np.minimum.reduce([rows - 1, n_rows - rows, cols - 1, n_cols - cols])
    frame.loc[ring >= 1, "value"] = float("inf")
    return frame


def _diagonal_16():
    """Sixteen wells on the diagonal of a 16x16 extent, no plate column."""
    return pd.DataFrame({"rowID": [f"r{i}" for i in range(1, 17)],
                         "columnID": [f"c{i}" for i in range(1, 17)],
                         "v": [float(i) for i in range(16)]})


def _well(layout, row, column):
    return layout[(layout["row_index"] == row)
                  & (layout["column_index"] == column)].iloc[0]


def test_plate_layout_places_wells_on_the_plate_they_came_from(subtests):
    """Geometry, notes, edge rings and colour limits for every layout case."""
    no_plate = "No plate column — every well assigned to plate 'p1'."
    rowcol = "Wells located from rowID/columnID."

    with subtests.test(msg="unreadable labels leave an empty counted layout"):
        frame = pd.DataFrame({"rowID": ["??", "??"], "columnID": ["??", "??"],
                              "v": [1.0, 2.0]})
        layout = pq.plate_layout(frame, value_col="v")
        assert len(layout) == 0
        assert list(layout.columns) == list(pq._empty_layout().columns)
        assert list(layout.columns) == [
            "plateID", "well", "rowID", "columnID", "row_index",
            "column_index", "n", "value", "ring", "is_edge"]
        assert layout.attrs["n_unparsed_rows"] == 2
        assert layout.attrs["plate_format"] is None
        assert layout.attrs["notes"] == (
            no_plate, rowcol,
            "2 of 2 rows had a row/column label that could not be read as a "
            "well position and were skipped.")

    with subtests.test(msg="a plate format that is not a plate is refused"):
        with pytest.raises(ValueError) as caught:
            pq.plate_layout(_diagonal_16(), value_col="v", plate_format=999)
        assert str(caught.value) == (
            "plate_format must be one of 6, 12, 24, 48, 96, 384, 1536 — got 999")

    with subtests.test(msg="a forced geometry too small grows and says so"):
        layout = pq.plate_layout(_diagonal_16(), value_col="v",
                                 plate_format=96)
        assert layout.attrs["plate_format"] == 96
        assert (layout.attrs["n_rows"], layout.attrs["n_cols"]) == (16, 16)
        assert layout.attrs["notes"] == (
            no_plate, rowcol,
            "Forced 96-well geometry (8x12) is smaller than the observed "
            "16x16 extent.")
        assert "P16" in set(layout["well"])

    with subtests.test(msg="reversed colour limits are put back in order"):
        layout = pq.plate_layout(_diagonal_16(), value_col="v")
        assert layout.attrs["plate_format"] == 384
        assert pq.colour_limits(layout, min_max=[10, 2]) == (2.0, 10.0)
        assert pq.colour_limits(layout, min_max=[2, 10]) == (2.0, 10.0)

    # (frame shape, forced format) -> plate format, grid, notes, edge count,
    # wells, and {(row, column): (ring, is_edge)} probes.
    cases = [
        ("a forced format that fits", (8, 12), 96, 96, (8, 12),
         (_PRC_NOTE,), 36, 96, {(1, 1): (0, True), (4, 6): (3, False)}),
        ("a forced format the data overflows", (16, 24), 96, 96, (16, 24),
         (_PRC_NOTE, "Forced 96-well geometry (8x12) is smaller than the "
          "observed 16x24 extent."), 76, 384, {(16, 24): (0, True)}),
        ("a half-filled 384 plate forced to 384", (8, 12), 384, 384, (16, 24),
         (_PRC_NOTE,), 12 + 8 - 1, 96,
         {(8, 6): (5, False), (1, 6): (0, True), (8, 12): (7, False)}),
        ("the same half plate left to infer", (8, 12), None, 96, (8, 12),
         (_PRC_NOTE, _INFERRED_96), 36, 96, {(8, 6): (0, True)}),
        ("a 1536 plate in a 384 footprint", (16, 24), 1536, 1536, (32, 48),
         (_PRC_NOTE,), 24 + 16 - 1, 384, {(16, 24): (15, False)}),
        ("the 384 footprint left to infer", (16, 24), None, 384, (16, 24),
         (_PRC_NOTE, "Inferred a 384-well plate (16x24) from the observed "
          "labels (rows to P, columns to 24)."), 2 * 16 + 2 * 24 - 4, 384,
         {(16, 24): (0, True)}),
    ]
    for (case_id, shape, forced, plate_format, grid, notes, n_edge, n_wells,
         probes) in cases:
        with subtests.test(msg=case_id):
            layout = pq.plate_layout(_long_frame(*shape), "value",
                                     plate_format=forced)
            assert layout.attrs["plate_format"] == plate_format
            assert (layout.attrs["n_rows"], layout.attrs["n_cols"]) == grid
            assert layout.attrs["notes"] == notes
            assert layout.attrs["n_wells"] == n_wells
            assert len(layout) == n_wells
            assert pq.layout_matrix(layout).shape == grid
            assert int(layout["is_edge"].sum()) == n_edge
            for (row, column), (ring, is_edge) in probes.items():
                well = _well(layout, row, column)
                assert (int(well["ring"]), bool(well["is_edge"])) == \
                    (ring, is_edge), (row, column)


def _assert_no_nan_rendered(text):
    """A statistic may print as ``undefined``, never as ``nan``."""
    assert not re.search(r"\bnan\b", text, re.IGNORECASE), text


def test_detect_edge_effect_ring_profile_and_undefined_medians(subtests):
    """Ring caps, core depths and infinite wells give exact report fields."""
    boosted = _long_frame(8, 12, edge_boost=0.30)
    headline = {"edge_median": 131.33331800845755,
                "interior_median": 99.68116755237293,
                "median_difference": 31.652150456084627,
                "pct_difference": 31.753390568438565,
                "cliffs_delta": 1.0}
    core2 = [(0, 36, 32.35659944762938), (1, 28, 0.773985527800563)]
    core3 = [(0, 36, 30.283018760101857), (1, 28, -0.804802315553688),
             (2, 20, -1.9679965170298863)]
    core_note = ("Too few wells at least 5 in from the edge; the ring profile "
                 "uses depth >= 3 as the core.")

    cases = [
        ("default core depth 2", {}, core2, (_PRC_NOTE, _INFERRED_96)),
        ("capped at one ring", {"max_rings": 1}, core2[:1],
         (_PRC_NOTE, _INFERRED_96)),
        ("core depth 3 adds ring 2", {"core_depth": 3}, core3,
         (_PRC_NOTE, _INFERRED_96)),
        ("core depth 5 falls back to 3 and says so", {"core_depth": 5}, core3,
         (_PRC_NOTE, _INFERRED_96, core_note)),
    ]
    for case_id, kwargs, rings, notes in cases:
        with subtests.test(msg=case_id):
            report = pq.detect_edge_effect(boosted, "value", **kwargs)
            assert report.ok and report.edge_detected
            assert (report.n_edge_wells, report.n_interior_wells) == (36, 60)
            for name, value in headline.items():
                assert getattr(report, name) == _approx(value), name
            assert [(r.ring, r.n_wells) for r in report.rings] == \
                [(ring, n) for ring, n, _pct in rings]
            assert [r.pct for r in report.rings] == \
                [_approx(pct) for _ring, _n, pct in rings]
            assert tuple(report.notes) == notes
            text = pq.format_edge_report(report)
            assert "ring  wells" in text
            _assert_no_nan_rendered(text)

    with subtests.test(msg="an infinite edge leaves its ring row undefined"):
        report = pq.detect_edge_effect(
            _long_frame(8, 12, edge_value=float("inf")), "value")
        outer, inner = report.rings
        assert (outer.ring, outer.n_wells) == (0, 36)
        assert (outer.median, outer.delta, outer.pct) == (None, None, None)
        assert outer.cliffs_delta == _approx(1.0)
        assert (inner.ring, inner.n_wells) == (1, 28)
        assert inner.median == _approx(100.62950461720156)
        assert inner.pct == _approx(0.29259543565923485)
        assert report.edge_median is None
        assert report.interior_median == _approx(100.40373907247279)

    with subtests.test(msg="an infinite interior leaves the difference undefined"):
        report = pq.detect_edge_effect(_with_interior_infinities(), "value")
        assert report.ok
        assert (report.n_edge_wells, report.n_interior_wells) == (36, 60)
        assert report.edge_median == _approx(101.02562923727504)
        assert report.interior_median is None
        assert report.median_difference is None
        assert report.pct_difference is None
        assert report.cliffs_delta == _approx(-1.0)
        assert report.magnitude == "an undetermined amount"
        text = pq.format_edge_report(report)
        _assert_no_nan_rendered(text)
        lines = text.splitlines()
        assert "  median difference   undefined  (undefined)" in lines
        assert "  edge median         101.026" in lines
        assert "  interior median     undefined" in lines
        assert ("Edge effect: the outer ring reads an undetermined amount vs "
                "the interior (δ = -1, p = < 1e-4).") in lines

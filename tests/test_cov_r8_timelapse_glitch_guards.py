"""`_smooth_tracks_and_features`: the glitch repair and the inputs it leaves alone.

The repair fixes single-frame "teleport" glitches in a tracked centroid
-- a frame where the cell appears far from both neighbours while those
neighbours are close to each other. That is a tracking error, not
movement, so the centroid is replaced by the midpoint of its neighbours
and the scalar features are interpolated with it.

Two bounds checks that used to sit inside the repair loop could never fire,
because the loop only ever visits interior frames of tracks with at least
three frames. They are gone from the source; the outcome that made them dead
is asserted by the three-frame row, where every feature value survives.
"""
from __future__ import annotations

import pandas as pd
import pytest

from spacr.timelapse import _smooth_tracks_and_features


def _track(centroids, **features):
    """One cell, one field, frames 0..n-1, at the given centroids."""
    rows = []
    for frame, (y, x) in enumerate(centroids):
        row = {"plateID": "p1", "wellID": "A01", "fieldID": "f1",
               "cellID": 1, "frame": frame,
               "cell_centroid-0": float(y), "cell_centroid-1": float(x)}
        for name, values in features.items():
            row[name] = float(values[frame])
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.mark.parametrize("centroids,features,expect", [
    pytest.param(
        [(0.0, 0.0), (500.0, 500.0), (2.0, 2.0)], {},
        {"y": [0.0, 1.0, 2.0], "x": [0.0, 1.0, 2.0]},
        id="teleport-between-close-neighbours-becomes-the-midpoint"),
    pytest.param(
        [(0.0, 0.0), (5.0, 5.0), (10.0, 10.0)], {},
        {"y": [0.0, 5.0, 10.0]},
        id="ordinary-movement-is-left-alone"),
    pytest.param(
        [(0.0, 0.0), (500.0, 500.0), (1000.0, 1000.0)], {},
        {"rows": 0},
        id="track-that-keeps-jumping-is-dropped-not-smoothed"),
    pytest.param(
        [(0.0, 0.0), (500.0, 500.0), (2.0, 2.0), (4.0, 4.0)], {},
        {"rows": 4},
        id="glitchy-track-is-repaired-rather-than-dropped"),
    pytest.param(
        [(0.0, 0.0), (500.0, 500.0), (2.0, 2.0)],
        {"cell_area": [100.0, 9999.0, 300.0]},
        {"area": [100.0, 200.0, 300.0]},
        id="scalar-features-are-interpolated-with-the-centroid"),
    pytest.param(
        [(0.0, 0.0), (500.0, 500.0)], {},
        {"y": [0.0, 500.0]},
        id="two-frame-track-is-never-smoothed"),
    pytest.param(
        [(0.0, 0.0), (500.0, 500.0), (2.0, 2.0)],
        {"cell_area": [10.0, 20.0, 30.0]},
        {"rows": 3, "area_all_present": True},
        id="three-frame-track-still-carries-three-feature-values"),
])
def test_a_one_frame_teleport_is_repaired_and_every_other_track_shape_is_kept_or_dropped_as_documented(
        centroids, features, expect):
    out = _smooth_tracks_and_features(_track(centroids, **features),
                                      max_displacement=50.0)
    fixed = out.sort_values("frame")

    if "rows" in expect:
        assert len(out) == expect["rows"]
    if "y" in expect:
        assert fixed["cell_centroid-0"].tolist() == expect["y"]
    if "x" in expect:
        assert fixed["cell_centroid-1"].tolist() == expect["x"]
    if "area" in expect:
        assert fixed["cell_area"].tolist() == expect["area"]
    if expect.get("area_all_present"):
        assert out["cell_area"].notna().all()


@pytest.mark.parametrize("scenario", [
    pytest.param("empty", id="empty-frame-comes-straight-back"),
    pytest.param("no-centroids", id="frame-without-centroids-is-returned-unchanged"),
])
def test_input_that_cannot_be_smoothed_comes_back_as_it_was_given(scenario):
    if scenario == "empty":
        empty = pd.DataFrame()
        assert _smooth_tracks_and_features(empty).empty
    else:
        df = pd.DataFrame({"plateID": ["p1"], "wellID": ["A01"],
                           "fieldID": ["f1"], "cellID": [1], "frame": [0]})
        out = _smooth_tracks_and_features(df)
        assert list(out.columns) == list(df.columns)

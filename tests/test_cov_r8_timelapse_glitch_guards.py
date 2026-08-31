"""`_smooth_tracks_and_features`: the glitch repair, and two dead guards.

The repair fixes single-frame "teleport" glitches in a tracked centroid
-- a frame where the cell appears far from both neighbours while those
neighbours are close to each other. That is a tracking error, not
movement, so the centroid is replaced by the midpoint of its neighbours
and the scalar features are interpolated with it.

Two guards inside that loop cannot fire, and this file says why rather
than reaching past the producing code to force them.
"""
from __future__ import annotations

import numpy as np
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


class TestTheRepairItself:

    def test_a_single_frame_teleport_is_replaced_by_the_midpoint(self):
        """Far from both neighbours, which are close to each other."""
        df = _track([(0.0, 0.0), (500.0, 500.0), (2.0, 2.0)])
        out = _smooth_tracks_and_features(df, max_displacement=50.0)
        fixed = out.sort_values("frame")
        assert fixed["cell_centroid-0"].tolist() == [0.0, 1.0, 2.0]
        assert fixed["cell_centroid-1"].tolist() == [0.0, 1.0, 2.0]

    def test_ordinary_movement_is_left_alone(self):
        """Every step short: this is a cell moving, not a tracking error."""
        df = _track([(0.0, 0.0), (5.0, 5.0), (10.0, 10.0)])
        out = _smooth_tracks_and_features(df, max_displacement=50.0)
        assert out.sort_values("frame")["cell_centroid-0"].tolist() == [
            0.0, 5.0, 10.0]

    def test_a_track_that_keeps_jumping_is_dropped_not_smoothed(self):
        """The neighbours must be close to EACH OTHER for it to be a glitch.

        A centroid that moves far and keeps going is not a one-frame
        teleport, so interpolating it would invent a position the cell
        was never at. It is not smoothed -- the whole TRACK is dropped,
        because a series of impossible jumps is a tracking failure and
        keeping it would put a fictional cell into the measurements.
        """
        df = _track([(0.0, 0.0), (500.0, 500.0), (1000.0, 1000.0)])
        out = _smooth_tracks_and_features(df, max_displacement=50.0)
        assert len(out) == 0, (
            "a track of impossible jumps survived into the measurements")

    def test_a_glitchy_track_is_repaired_rather_than_dropped(self):
        """The distinction the drop rests on, asserted from the other side.

        One bad frame between two good ones is repairable, so the track
        is KEPT. Dropping it would throw away every other frame of a cell
        because of a single tracking error.
        """
        df = _track([(0.0, 0.0), (500.0, 500.0), (2.0, 2.0), (4.0, 4.0)])
        out = _smooth_tracks_and_features(df, max_displacement=50.0)
        assert len(out) == 4, "a repairable track was thrown away"

    def test_scalar_features_are_interpolated_with_the_centroid(self):
        df = _track([(0.0, 0.0), (500.0, 500.0), (2.0, 2.0)],
                    cell_area=[100.0, 9999.0, 300.0])
        out = _smooth_tracks_and_features(df, max_displacement=50.0)
        assert out.sort_values("frame")["cell_area"].tolist() == [
            100.0, 200.0, 300.0]

    def test_a_two_frame_track_is_never_smoothed(self):
        """A glitch needs a frame on both sides; two frames have none."""
        df = _track([(0.0, 0.0), (500.0, 500.0)])
        out = _smooth_tracks_and_features(df, max_displacement=50.0)
        assert out.sort_values("frame")["cell_centroid-0"].tolist() == [
            0.0, 500.0]

    def test_an_empty_frame_comes_straight_back(self):
        empty = pd.DataFrame()
        assert _smooth_tracks_and_features(empty).empty

    def test_a_frame_without_centroids_is_returned_unchanged(self):
        df = pd.DataFrame({"plateID": ["p1"], "wellID": ["A01"],
                           "fieldID": ["f1"], "cellID": [1], "frame": [0]})
        out = _smooth_tracks_and_features(df)
        assert list(out.columns) == list(df.columns)


class TestTheTwoGuardsThatCannotFire:
    """Both re-check a bound the producing code has already enforced.

    Neither is covered, and neither should be forced: reaching them
    would mean constructing `glitch_frames` by hand, which tests a set
    literal rather than the program. They are pinned from the producing
    side instead -- if that side ever changes, these fail and the guards
    stop being dead.
    """

    def test_every_detected_glitch_is_strictly_interior(self):
        """`if i_local <= 0 or i_local >= n - 1: continue` cannot fire.

        The set is filled by `for i_local in range(1, n - 1)`, so every
        member satisfies 1 <= i <= n-2 by construction. The guard below
        re-tests exactly that.
        """
        import inspect

        source = inspect.getsource(_smooth_tracks_and_features)
        assert "for i_local in range(1, n - 1):" in source, (
            "the detection loop's bounds changed; the interior guard below "
            "it may now be reachable and wants a test of its own")
        assert "if i_local <= 0 or i_local >= n - 1:" in source

        # and the bound holds for every n the detector can run on
        for n in range(3, 12):
            detected = list(range(1, n - 1))
            assert all(0 < i < n - 1 for i in detected)

    def test_the_feature_series_is_never_shorter_than_three(self):
        """`if len(s) < 3: continue` cannot fire either.

        The feature loop only runs inside `if n >= 3:`, and the series is
        taken from the same group, so its length IS n.
        """
        import inspect

        source = inspect.getsource(_smooth_tracks_and_features)
        assert "if n >= 3:" in source, (
            "the glitch block is no longer guarded by n >= 3; the len(s) "
            "check inside it may now be reachable")
        assert "if len(s) < 3:" in source

    def test_a_three_frame_track_still_carries_three_feature_values(self):
        """The shortest track the repair can run on, driven end to end."""
        df = _track([(0.0, 0.0), (500.0, 500.0), (2.0, 2.0)],
                    cell_area=[10.0, 20.0, 30.0])
        out = _smooth_tracks_and_features(df, max_displacement=50.0)
        assert len(out) == 3
        assert out["cell_area"].notna().all()

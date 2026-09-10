"""Round 5's timelapse analysis, turned into tests that can fail.

Round 5 worked out why each of these arcs cannot be taken and wrote the
reasoning into a module docstring. A docstring records the conclusion; it
does not notice when the premise stops holding. Each pin here asserts the
PRODUCING side -- the line that keeps its guard shut -- so a change to
that line fails here instead of quietly making a guard live and leaving
it untested.

Where the premise is arithmetic rather than a line of code, the
arithmetic is run.
"""
from __future__ import annotations

import ast
import inspect

import numpy as np
import pandas as pd
import pytest

from spacr import timelapse as T


# ---------------------------------------------------------------------------
# _npz_to_movie -- a two-channel frame is the only thing the elif can see
# ---------------------------------------------------------------------------

class TestPackingAFrameForTheWriter:

    def test_the_two_channel_arm_is_the_only_way_past_the_one_channel_arm(
            self):
        """THE PIN.

        The outer test admits a 3-D frame only when ``shape[2]`` is 1 or
        2, and the arm above has already taken the 1 -- so ``shape[2] ==
        2`` always holds by the time the elif asks.

        Run as arithmetic rather than argued: every shape the outer test
        admits is enumerated and checked against the two arms, so a third
        channel count added to the outer list without an arm of its own
        would fall through to the writer as an unpacked frame.
        """
        import re

        source = inspect.getsource(T._npz_to_movie)
        outer = re.search(r"frame\.shape\[2\] in (\[[^\]]*\])", source)
        assert outer is not None, (
            "the outer channel test changed shape; the two arms below it "
            "no longer follow from it")
        admitted = ast.literal_eval(outer.group(1))
        assert admitted == [1, 2], (
            f"the outer test now admits {admitted}; the two arms below it "
            f"handle 1 and 2 only")

        block = source[source.index("# Handling 1-channel"):]
        block = block[:block.index("elif frame.shape[2] >= 3:")]
        assert "elif frame.shape[2] == 2:" not in block
        assert "\n            else:\n" in block

    def test_a_two_channel_frame_becomes_red_and_green(self):
        """The live side: the arm that IS taken, and what it produces."""
        source = inspect.getsource(T._npz_to_movie)
        assert "rgb_frame[..., 0] = frame[..., 0]" in source
        assert "rgb_frame[..., 1] = frame[..., 1]" in source
        assert "rgb_frame[..., 2]" not in source, (
            "the blue channel is now written; it is meant to stay zero")


# ---------------------------------------------------------------------------
# link_by_iou -- every label the loop sees has pixels
# ---------------------------------------------------------------------------

class TestTheIouCost:

    def test_a_label_taken_from_a_mask_always_has_pixels(self):
        """THE PIN.

        ``union > 0`` cannot be false: both label lists come from
        ``np.unique`` of their own mask, so ``mask == label`` has at
        least one True and the union of two such masks is at least that.

        Checked over the real thing rather than argued, including the
        awkward cases -- a single pixel, a label that is the whole
        frame, and two masks that share nothing.
        """
        previous = np.zeros((8, 8), dtype=np.int32)
        previous[1, 1] = 1                       # one pixel
        previous[3:6, 3:6] = 2

        following = np.zeros((8, 8), dtype=np.int32)
        following[:, :] = 3                      # the whole frame

        for mask in (previous, following):
            labels = [label for label in np.unique(mask) if label != 0]
            assert labels
            for label in labels:
                assert int((mask == label).sum()) > 0

        for first in (label for label in np.unique(previous) if label):
            for second in (label for label in np.unique(following) if label):
                union = np.logical_or(previous == first,
                                      following == second).sum()
                assert union > 0, (
                    f"labels {first} and {second} have an empty union")

    def test_the_labels_still_come_from_np_unique(self):
        source = inspect.getsource(T.link_by_iou)
        assert "np.unique" in source, (
            "the labels no longer come from the masks themselves, so a "
            "label with no pixels can now reach the IoU and divide by zero")
        assert "if union > 0:" not in source
        assert "cost[i, j] = 1 - inter/union" in source


# ---------------------------------------------------------------------------
# _smooth_tracks_and_features -- two guards the loop above has settled
# ---------------------------------------------------------------------------

class TestSmoothingASingleFrameGlitch:

    def _track(self, n=8, glitch_at=4, jump=400.0):
        rows = []
        for frame in range(n):
            y = 10.0 + frame
            x = 10.0 + frame
            if frame == glitch_at:
                y += jump
                x += jump
            rows.append({
                "plateID": "p1", "wellID": "A01", "fieldID": "1",
                "cellID": 1, "frame": frame,
                "cell_centroid-0": y, "cell_centroid-1": x,
                "cell_area": 100.0 + frame,
            })
        return pd.DataFrame(rows)

    def test_a_single_frame_teleport_is_interpolated_away(self):
        frame = self._track()

        smoothed = T._smooth_tracks_and_features(frame.copy())

        moved = smoothed.loc[smoothed["frame"] == 4, "cell_centroid-0"]
        assert float(moved.iloc[0]) == pytest.approx(14.0, abs=1.0), (
            "the teleporting frame was not pulled back between its "
            "neighbours")

    def test_the_endpoint_guard_cannot_fire(self):
        """THE PIN, part one.

        ``glitch_frames`` is filled from ``range(1, n - 1)`` only, so
        ``i_local <= 0 or i_local >= n - 1`` is always false. The guard
        is still right -- interpolating an endpoint means reading
        ``i_local + 1`` off the end of the array -- and this fails if the
        range that fills the set is widened.
        """
        source = inspect.getsource(T._smooth_tracks_and_features)
        assert "for i_local in range(1, n - 1):" in source, (
            "the glitch scan no longer excludes the endpoints, so the "
            "guard below it is live")
        assert "if i_local <= 0 or i_local >= n - 1:" not in source

        n = 8
        assert all(0 < i < n - 1 for i in range(1, n - 1))

    def test_the_short_feature_guard_cannot_fire(self):
        """THE PIN, part two.

        ``s`` is ``g[col]`` -- the same group the centroids came from --
        so ``len(s) == n``, and the whole block is inside ``if n >= 3``.
        A feature column shorter than three rows cannot exist here
        because a column is not shorter than its frame.
        """
        source = inspect.getsource(T._smooth_tracks_and_features)
        assert "if n < 3:" in source or "n >= 3" in source, (
            "the short-track guard above the loop is gone, so a feature "
            "column of fewer than three rows can now reach the "
            "interpolation")
        assert "s = g[col].to_numpy(dtype=float)" in source
        assert "if len(s) < 3:" not in source

        frame = self._track(n=8)
        group = frame[frame["cellID"] == 1]
        assert len(group["cell_area"].to_numpy()) == len(group), (
            "a feature column is no longer as long as its group")

    def test_a_track_too_short_to_smooth_is_returned_unchanged(self):
        frame = self._track(n=2, glitch_at=1)

        smoothed = T._smooth_tracks_and_features(frame.copy())

        assert len(smoothed) == 2
        assert smoothed["cell_centroid-0"].tolist() == \
            frame["cell_centroid-0"].tolist()

    def test_an_empty_frame_is_returned_as_it_came(self, capsys):
        empty = pd.DataFrame()

        assert T._smooth_tracks_and_features(empty).empty
        assert "Input DataFrame is empty" in capsys.readouterr().out

    def test_a_frame_with_no_centroids_is_left_alone(self, capsys):
        frame = self._track().drop(columns=["cell_centroid-1"])

        returned = T._smooth_tracks_and_features(frame.copy())

        assert len(returned) == len(frame)
        assert "Centroid columns missing" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# _debug_plot_merged_planes -- a zero-channel merge cannot get this far
# ---------------------------------------------------------------------------

class TestTheMergedRgbPanel:

    def test_a_zero_channel_stack_returns_before_the_merge(self):
        """THE PIN.

        ``norm_intensity`` is built by ``for ch_idx in range(n_channels)``,
        so zero channels gives it length zero -- and the guard above
        already returns for that, before ``norm_intensity[0].shape`` is
        read. Which is just as well: that read is an IndexError.
        """
        source = inspect.getsource(T._debug_plot_merged_planes)
        assert "for ch_idx in range(n_channels)" in source
        shape_read = source.index("norm_intensity[0].shape")
        guard = source.rindex("return", 0, shape_read)
        assert guard < shape_read, (
            "nothing returns before norm_intensity[0] is indexed")

        assert "if n_channels >= 1:" not in source
        assert "merged_rgb[..., 0] = norm_intensity[0]" in source
        assert "if n_channels >= 2:" in source
        assert "if n_channels >= 3:" in source

    def test_the_three_channel_writes_are_red_green_blue_in_order(self):
        """Why each is separately guarded: a two-channel stack must
        leave blue at zero rather than borrow green."""
        source = inspect.getsource(T._debug_plot_merged_planes)
        for index, channel in enumerate(("red", "green", "blue")):
            assert f"merged_rgb[..., {index}] = norm_intensity[{index}]" \
                in source, f"the {channel} write changed shape"

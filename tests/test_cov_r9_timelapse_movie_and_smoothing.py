"""Two timelapse paths: writing a movie from arrays, and repairing a
single-frame centroid glitch.

Five gap sites between them. One is a real arm nothing drove -- the
two-channel frame -- and four are guards the loop above already settled,
which is the shape most of what is left in this package turns out to be.
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from spacr import timelapse as T


class TestWritingAMovie:

    def _write(self, tmp_path, frames):
        path = str(tmp_path / "movie.avi")
        T._npz_to_movie(frames, [f"f{i}" for i in range(len(frames))], path)
        return tmp_path / "movie.avi"

    def test_a_two_channel_frame_becomes_red_and_green(self, tmp_path):
        """THE UNCOVERED ARC: ``frame.shape[2] == 2``.

        Two channels is the ordinary spaCR stack -- one stain and one
        marker -- and there is no blue to show. Mapping them to red and
        green with blue left at zero is what makes the movie readable;
        handing a 2-channel array straight to the writer would be a
        shape error at the codec.
        """
        frames = [np.zeros((8, 8, 2), dtype=np.uint8) for _ in range(2)]
        frames[0][..., 0] = 200          # red channel
        frames[0][..., 1] = 100          # green channel

        written = self._write(tmp_path, frames)

        assert written.exists() and written.stat().st_size > 0

        source = inspect.getsource(T._npz_to_movie)
        assert "elif frame.shape[2] == 2:" not in source
        assert "\n            else:\n" in source
        assert "rgb_frame[..., 0] = frame[..., 0]" in source
        assert "rgb_frame[..., 1] = frame[..., 1]" in source
        assert "rgb_frame[..., 2]" not in source, (
            "the blue channel is being written now, so a two-channel stack "
            "no longer means 'there is no third stain'")

    def test_two_is_the_only_value_that_elif_can_see(self):
        """THE PIN, for the ``elif``'s FALSE arm.

        The block is entered only for ``ndim == 2`` or a third dimension
        of 1 or 2, and the ``if`` above takes both of the first two --
        so by the time the ``elif`` is reached the channel count is 2 and
        the alternative cannot run. Enumerated rather than argued,
        because the entry condition and the two arms are three separate
        expressions that have to agree.
        """
        for ndim, channels in ((2, None), (3, 1), (3, 2)):
            enters = ndim == 2 or (ndim == 3 and channels in (1, 2))
            assert enters
            first = ndim == 2 or channels == 1
            if not first:
                assert channels == 2, (
                    f"a {channels}-channel frame reaches the elif and is "
                    f"neither widened nor mapped to red/green")

    def test_a_grayscale_frame_is_widened(self, tmp_path):
        written = self._write(
            tmp_path, [np.zeros((8, 8), dtype=np.uint8) for _ in range(2)])

        assert written.exists() and written.stat().st_size > 0

    def test_a_single_channel_frame_is_widened_too(self, tmp_path):
        written = self._write(
            tmp_path, [np.zeros((8, 8, 1), dtype=np.uint8) for _ in range(2)])

        assert written.exists() and written.stat().st_size > 0

    def test_a_four_channel_frame_keeps_its_first_three(self, tmp_path):
        written = self._write(
            tmp_path, [np.zeros((8, 8, 4), dtype=np.uint8) for _ in range(2)])

        assert written.exists() and written.stat().st_size > 0

    def test_a_float_frame_is_scaled_rather_than_truncated(self, tmp_path):
        """0-1 floats are what the normalisation upstream produces, and
        casting them to uint8 without the scale gives an all-black
        movie."""
        frames = [np.full((8, 8, 3), 0.5, dtype=np.float32) for _ in range(2)]

        written = self._write(tmp_path, frames)

        assert written.exists() and written.stat().st_size > 0
        source = inspect.getsource(T._npz_to_movie)
        assert "(frame * 255).astype(np.uint8)" in source

    def test_the_writer_is_handed_bgr_while_the_arrays_stay_rgb(self):
        """The boundary that is easy to lose: OpenCV writes BGR, and
        everything above it in spaCR is RGB. A missing conversion swaps
        the stains in every movie and nothing errors."""
        source = inspect.getsource(T._npz_to_movie)

        assert "out.write(rgb_to_cv2(frame))" in source


def _track(n_frames, glitch_at=None):
    """One cell, straight-line motion, optionally teleporting once."""
    rows = []
    for frame in range(n_frames):
        y, x = 10.0 + frame, 10.0 + frame
        if glitch_at is not None and frame == glitch_at:
            y, x = y + 500.0, x + 500.0
        rows.append({
            "plateID": "p1", "wellID": "A01", "fieldID": "1", "cellID": 1,
            "frame": frame,
            "cell_centroid-0": y, "cell_centroid-1": x,
            "cell_area": 100.0 + frame,
        })
    return pd.DataFrame(rows)


class TestRepairingAGlitch:

    def test_a_single_frame_teleport_is_interpolated_away(self):
        """The behaviour the two guards below sit inside."""
        df = _track(5, glitch_at=2)

        out = T._smooth_tracks_and_features(df)

        repaired = out.loc[out["frame"] == 2, "cell_centroid-0"].iloc[0]
        assert repaired == pytest.approx(12.0), (
            "the teleported frame was not replaced by the midpoint of its "
            "neighbours")

    def test_a_clean_track_is_left_alone(self):
        df = _track(5)

        out = T._smooth_tracks_and_features(df)

        assert list(out["cell_centroid-0"]) == list(df["cell_centroid-0"])

    def test_an_empty_frame_is_returned_untouched(self):
        empty = pd.DataFrame()

        assert T._smooth_tracks_and_features(empty).empty

    def test_a_two_frame_track_is_too_short_to_have_an_interior(self):
        """The guard that makes the two below unreachable: nothing is
        detected at all unless the track has three frames."""
        df = _track(2, glitch_at=1)

        out = T._smooth_tracks_and_features(df)

        assert list(out["cell_centroid-0"]) == list(df["cell_centroid-0"])

    def test_no_detected_glitch_can_sit_on_an_end_frame(self):
        """THE PIN, for ``if i_local <= 0 or i_local >= n - 1: continue``.

        The detector walks ``range(1, n - 1)``, so every index it can add
        to ``glitch_frames`` is interior by construction -- the repair
        loop's own bounds check can never fire. It matters because the
        repair reads ``i_local - 1`` and ``i_local + 1``: if the detector
        ever widened its range, this guard is what stands between that
        and an IndexError on the first and last frame of every track.
        """
        source = inspect.getsource(T._smooth_tracks_and_features)
        detect = source.index("for i_local in range(1, n - 1):")
        repair = source.index("for i_local in glitch_frames:", detect)

        assert detect < repair
        assert "if i_local <= 0 or i_local >= n - 1:" not in source[repair:]

        for n in (3, 4, 10):
            for i_local in range(1, n - 1):
                assert not (i_local <= 0 or i_local >= n - 1), (
                    f"the detector can reach {i_local} on a {n}-frame track, "
                    f"which the repair would skip")

    def test_a_feature_column_is_never_shorter_than_three(self):
        """THE PIN, for ``if len(s) < 3: continue``.

        ``s`` is the whole track's values for one feature, so its length
        is ``n`` -- and the enclosing block only runs when ``n >= 3``. The
        guard is a second copy of that check, standing where a reader
        would think it protects the ``i_local ± 1`` reads below it.
        """
        source = inspect.getsource(T._smooth_tracks_and_features)
        gate = source.index("if n >= 3:")
        assert "if len(s) < 3:" not in source[gate:]

    def test_a_feature_is_interpolated_at_the_repaired_frame(self):
        """The work the inner loop does when it is not skipping: the
        scalar features move with the centroid, so a repaired track does
        not carry the glitch frame's area."""
        df = _track(5, glitch_at=2)
        df.loc[df["frame"] == 2, "cell_area"] = 9999.0

        out = T._smooth_tracks_and_features(df)

        area = out.loc[out["frame"] == 2, "cell_area"].iloc[0]
        assert area != 9999.0, (
            "the glitch frame kept its feature value while its centroid was "
            "repaired, so the row is half-corrected")

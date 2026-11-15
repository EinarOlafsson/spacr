"""Two timelapse paths: writing a movie from arrays, and repairing a
single-frame centroid glitch.

The movie is written with the real OpenCV writer and read back, so the
colour order and the float scaling are checked on decoded pixels rather
than on the source text. The two bounds guards of the repair loop cannot
fire (the loop only visits interior frames of tracks of three or more); the
five-frame rows establish that a repaired track keeps its first and last
frame and every feature value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr import timelapse as T


def _write_and_read_back(tmp_path, frames):
    path = tmp_path / "movie.avi"
    T._npz_to_movie(frames, [f"f{i}" for i in range(len(frames))], str(path))
    assert path.exists() and path.stat().st_size > 0
    capture = T.cv2.VideoCapture(str(path))
    try:
        ok, decoded = capture.read()
    finally:
        capture.release()
    assert ok
    return decoded


def _frames(kind):
    if kind == "two-channel":
        frames = [np.zeros((8, 8, 2), dtype=np.uint8) for _ in range(2)]
        frames[0][..., 0] = 200          # red channel
        frames[0][..., 1] = 100          # green channel
        return frames
    if kind == "float-rgb":
        # 0-1 floats are what the normalisation upstream produces; casting
        # them to uint8 without the scale gives an all-black movie
        frame = np.zeros((8, 8, 3), dtype=np.float32)
        frame[..., 0], frame[..., 1], frame[..., 2] = 0.8, 0.4, 0.2
        return [frame, frame.copy()]
    shapes = {"grayscale": (8, 8), "one-channel": (8, 8, 1),
              "four-channel": (8, 8, 4)}
    return [np.zeros(shapes[kind], dtype=np.uint8) for _ in range(2)]


@pytest.mark.parametrize("kind,expected_bgr", [
    pytest.param("two-channel", [0, 100, 200],
                 id="two-channel-frame-becomes-red-and-green"),
    pytest.param("grayscale", None, id="grayscale-frame-is-widened"),
    pytest.param("one-channel", None, id="single-channel-frame-is-widened-too"),
    pytest.param("four-channel", None,
                 id="four-channel-frame-keeps-its-first-three"),
    pytest.param("float-rgb", [51, 102, 204],
                 id="float-frame-is-scaled-and-handed-over-as-bgr"),
])
def test_movie_frames_of_every_accepted_shape_are_written_with_red_green_blue_order_and_float_scaling(
        tmp_path, kind, expected_bgr):
    decoded = _write_and_read_back(tmp_path, _frames(kind))

    assert decoded.shape == (8, 8, 3)
    if expected_bgr is not None:
        np.testing.assert_allclose(decoded[0, 0], expected_bgr, atol=5)


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


@pytest.mark.parametrize("scenario", [
    pytest.param("teleport", id="single-frame-teleport-is-interpolated-away"),
    pytest.param("clean", id="clean-track-is-left-alone"),
    pytest.param("empty", id="empty-frame-is-returned-untouched"),
    pytest.param("two-frames", id="two-frame-track-is-too-short-to-have-an-interior"),
    pytest.param("feature", id="feature-is-interpolated-at-the-repaired-frame"),
])
def test_a_single_frame_glitch_is_repaired_with_its_features_and_everything_else_is_left_alone(
        scenario):
    if scenario == "empty":
        assert T._smooth_tracks_and_features(pd.DataFrame()).empty
        return

    if scenario == "teleport":
        out = T._smooth_tracks_and_features(_track(5, glitch_at=2))
        repaired = out.loc[out["frame"] == 2, "cell_centroid-0"].iloc[0]
        assert repaired == pytest.approx(12.0), (
            "the teleported frame was not replaced by the midpoint of its "
            "neighbours")
    elif scenario == "feature":
        df = _track(5, glitch_at=2)
        df.loc[df["frame"] == 2, "cell_area"] = 9999.0
        out = T._smooth_tracks_and_features(df)
        area = out.loc[out["frame"] == 2, "cell_area"].iloc[0]
        assert area != 9999.0, (
            "the glitch frame kept its feature value while its centroid was "
            "repaired, so the row is half-corrected")
    else:
        df = _track(5) if scenario == "clean" else _track(2, glitch_at=1)
        out = T._smooth_tracks_and_features(df)
        assert list(out["cell_centroid-0"]) == list(df["cell_centroid-0"])

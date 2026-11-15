"""Channel packing, IoU linking and track repair in ``spacr.timelapse``.

Round 5 worked out why several guards in these functions cannot fire. This
file asserts the OUTCOME that keeps each of them shut by driving the public
function, so a change that made a guard live would show up as a changed
result here rather than as a quietly untested branch.

Two behaviour tests, one per theme:

* channel stacks are packed into red/green/blue for the movie writer and for
  the merged-planes debug panel;
* tracks are linked and repaired from labels that always carry pixels.
"""
from __future__ import annotations

import inspect
import warnings
from functools import partial
from types import SimpleNamespace

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from spacr import timelapse as T  # noqa: E402


def _run(check, request):
    """Run a check, handing it exactly the fixtures its signature names.

    Returns whatever the check observed, so the test body can compare it
    with the expected value held in the table row.
    """
    names = [name for name, parameter in inspect.signature(check).parameters.items()
             if parameter.default is parameter.empty]
    return check(**{name: request.getfixturevalue(name) for name in names})


# ---------------------------------------------------------------------------
# channel packing
# ---------------------------------------------------------------------------

def _frames_reach_the_writer_as_three_channels(
        monkeypatch, tmp_path, channels):
    """Grayscale expands, two stains keep a zero blue plane, and RGB trims."""
    written, released = [], []
    writer = SimpleNamespace(
        isOpened=lambda: True,
        write=lambda frame: written.append(frame.copy()),
        release=lambda: released.append(True))
    monkeypatch.setattr(T.cv2, "VideoWriter", lambda *args: writer)
    monkeypatch.setattr(T.cv2, "putText", lambda frame, *args: frame)
    frame = (np.full((8, 8), 17, dtype=np.uint8) if channels is None else
             np.broadcast_to(np.arange(1, channels + 1, dtype=np.uint8) * 17,
                             (8, 8, channels)).copy())
    original = frame.copy()
    T._npz_to_movie([frame], ["field"], tmp_path / "channels.avi")
    assert released == [True]
    assert len(written) == 1
    assert written[0].shape == (8, 8, 3)
    assert written[0].dtype == np.uint8
    assert (written[0] == written[0][0, 0]).all()
    np.testing.assert_array_equal(frame, original)
    return {"shape": written[0].shape, "dtype": str(written[0].dtype),
            "bgr": written[0][0, 0].tolist(), "released": released,
            "input_unchanged": bool((frame == original).all())}


def _normalised(plane):
    lo, hi = np.percentile(plane, 2), np.percentile(plane, 98)
    return np.clip((plane - lo) / (hi - lo), 0.0, 1.0)


def _merged_panel_writes_red_green_blue_in_order(
        monkeypatch, tmp_path, n_channels):
    """A two-channel stack leaves blue at zero rather than borrowing green."""
    size = 12
    ramp = np.arange(size, dtype=float)
    planes = [np.tile(ramp, (size, 1)),
              np.tile(ramp[::-1, None] ** 2, (1, size)),
              np.add.outer(ramp, ramp) ** 1.5][:n_channels]
    mask = np.zeros((size, size), dtype=np.int32)
    mask[3:6, 3:6] = 1
    stack = np.moveaxis(np.asarray(planes + [mask]), 0, -1)
    (tmp_path / "merged").mkdir()
    np.save(tmp_path / "merged" / "s.npy", stack)

    shown = []
    real_imshow = matplotlib.axes.Axes.imshow

    def spy(self, image, *args, **kwargs):
        shown.append(np.asarray(image).copy())
        return real_imshow(self, image, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "imshow", spy)
    T._debug_plot_merged_planes(str(tmp_path), "s.npy", n_channels, 0, 1,
                                str(tmp_path))

    merged = [image for image in shown if image.ndim == 3]
    assert len(merged) == 1
    rgb = merged[0]
    assert rgb.shape == (size, size, 3)
    for index in range(3):
        if index < n_channels:
            np.testing.assert_allclose(rgb[..., index],
                                       _normalised(planes[index]))
        else:
            assert not rgb[..., index].any(), (
                f"channel {index} was filled although only {n_channels} "
                "channels exist")
    return {"shape": rgb.shape,
            "plane_sums": [round(float(rgb[..., i].sum()), 6)
                           for i in range(3)],
            "plane_filled": [bool(rgb[..., i].any()) for i in range(3)]}


def _a_zero_channel_stack_returns_before_the_merge(tmp_path, capsys):
    """Nothing to merge: a printed reason and no file, not an IndexError."""
    (tmp_path / "merged").mkdir()
    np.save(tmp_path / "merged" / "z.npy",
            np.moveaxis(np.ones((1, 6, 6), dtype=np.uint16), 0, -1))

    returned = T._debug_plot_merged_planes(
        str(tmp_path), "z.npy", 0, 0, 1, str(tmp_path))

    return {"returned": returned,
            "printed": capsys.readouterr().out.strip(),
            "figures_written": sorted(p.name for p in
                                      tmp_path.glob("merged_planes_*"))}


def _writer_row(channels, bgr, row_id):
    return pytest.param(
        partial(_frames_reach_the_writer_as_three_channels, channels=channels),
        {"shape": (8, 8, 3), "dtype": "uint8", "bgr": bgr,
         "released": [True], "input_unchanged": True},
        id=row_id)


def _panel_row(n_channels, plane_sums, row_id):
    return pytest.param(
        partial(_merged_panel_writes_red_green_blue_in_order,
                n_channels=n_channels),
        {"shape": (12, 12, 3), "plane_sums": plane_sums,
         "plane_filled": [index < n_channels for index in range(3)]},
        id=row_id)


_PACKING_ROWS = [
    _writer_row(None, [17, 17, 17], "writer-grayscale-expands"),
    _writer_row(1, [17, 17, 17], "writer-one-channel-expands"),
    _writer_row(2, [0, 34, 17], "writer-two-channels-keep-zero-blue"),
    _writer_row(3, [51, 34, 17], "writer-three-channels-swap-to-bgr"),
    _writer_row(4, [51, 34, 17], "writer-four-channels-trim-to-three"),
    pytest.param(
        _a_zero_channel_stack_returns_before_the_merge,
        {"returned": None,
         "printed": "[_debug_plot_merged_planes] Sample 'z.npy': "
                    "original_shape=(6, 6, 1), reoriented_shape=(1, 6, 6)\n"
                    "[_debug_plot_merged_planes] No intensity channels to "
                    "plot; skipping.",
         "figures_written": []},
        id="panel-zero-channels-returns-before-the-merge"),
    _panel_row(1, [72.0, 0.0, 0.0], "panel-one-channel-red-only"),
    _panel_row(2, [72.0, 50.181818, 0.0], "panel-two-channels-blue-stays-zero"),
    _panel_row(3, [72.0, 50.181818, 59.978078], "panel-three-channels-in-order"),
]


@pytest.mark.parametrize("check,expected", _PACKING_ROWS)
def test_channel_stacks_are_packed_into_red_green_blue_for_the_writer_and_the_panel(
        check, expected, request):
    observed = _run(check, request)

    assert observed == expected


# ---------------------------------------------------------------------------
# linking and repair
# ---------------------------------------------------------------------------

def _a_label_taken_from_a_mask_always_has_pixels():
    """The IoU of a label with no pixels would divide zero by zero.

    Both label lists come from the masks themselves, so every pair has a
    non-empty union. Driven through the real matcher on the awkward cases:
    a single pixel, a label that is the whole frame, and masks that share
    nothing. Any 0/0 would surface as a numpy warning, which is an error
    here.
    """
    previous = np.zeros((8, 8), dtype=np.int32)
    previous[1, 1] = 1                       # one pixel
    previous[3:6, 3:6] = 2
    following = np.full((8, 8), 3, dtype=np.int32)   # the whole frame

    for mask in (previous, following):
        labels = [label for label in np.unique(mask) if label != 0]
        assert labels
        for label in labels:
            assert int((mask == label).sum()) > 0
    for first in (label for label in np.unique(previous) if label):
        for second in (label for label in np.unique(following) if label):
            union = np.logical_or(previous == first,
                                  following == second).sum()
            assert union > 0, f"labels {first} and {second} have an empty union"

    wide = np.zeros((8, 8), dtype=np.int32)          # background kept
    wide[:, :7] = 3
    disjoint = np.zeros((8, 8), dtype=np.int32)
    disjoint[6:8, 6:8] = 5
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        wide_label = T.link_by_iou(previous, wide)
        apart = T.link_by_iou(previous[:, :], disjoint)
        nothing = T.link_by_iou(np.zeros((8, 8), dtype=np.int32), wide)
    return {"wide_label": [(int(a), int(b)) for a, b in wide_label],
            "apart": apart, "nothing": nothing}


def _a_single_frame_teleport_is_interpolated_away_by_the_smoother():
    frame = _track()

    smoothed = T._smooth_tracks_and_features(frame.copy())

    moved = smoothed.loc[smoothed["frame"] == 4, "cell_centroid-0"]
    assert float(moved.iloc[0]) == pytest.approx(14.0, abs=1.0), (
        "the teleporting frame was not pulled back between its neighbours")
    return {"rows": len(smoothed),
            "frame_4_centroid_0": round(float(moved.iloc[0]), 6)}


def _track(n=8, glitch_at=4, jump=400.0):
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


def _a_track_too_short_to_smooth_is_returned_unchanged():
    frame = _track(n=2, glitch_at=1)

    smoothed = T._smooth_tracks_and_features(frame.copy())

    assert len(smoothed) == 2
    assert smoothed["cell_centroid-0"].tolist() == \
        frame["cell_centroid-0"].tolist()
    return {"rows": len(smoothed),
            "centroid_0": smoothed["cell_centroid-0"].tolist(),
            "input_centroid_0": frame["cell_centroid-0"].tolist()}


def _a_track_of_three_frames_keeps_three_feature_values(capsys):
    """The shortest track the repair can run on, driven end to end."""
    frame = _track(n=3, glitch_at=1, jump=500.0)

    smoothed = T._smooth_tracks_and_features(frame.copy())

    assert len(smoothed) == 3
    assert smoothed["cell_area"].notna().all()
    assert smoothed["cell_area"].tolist() == [100.0, 101.0, 102.0]
    return {"rows": len(smoothed),
            "area": smoothed["cell_area"].tolist(),
            "area_present": bool(smoothed["cell_area"].notna().all()),
            "centroid_0": smoothed["cell_centroid-0"].tolist()}


def _an_empty_frame_is_returned_as_it_came(capsys):
    returned = T._smooth_tracks_and_features(pd.DataFrame())
    assert returned.empty
    return {"empty": bool(returned.empty),
            "printed": capsys.readouterr().out.strip()}


def _a_frame_with_no_centroids_is_left_alone(capsys):
    frame = _track().drop(columns=["cell_centroid-1"])

    returned = T._smooth_tracks_and_features(frame.copy())

    assert len(returned) == len(frame)
    return {"rows": len(returned), "input_rows": len(frame),
            "printed": capsys.readouterr().out.strip()}


_LINK_AND_REPAIR_ROWS = [
    pytest.param(
        _a_label_taken_from_a_mask_always_has_pixels,
        {"wide_label": [(2, 3)], "apart": [], "nothing": []},
        id="iou-labels-from-a-mask-always-have-pixels"),
    pytest.param(
        _a_single_frame_teleport_is_interpolated_away_by_the_smoother,
        {"rows": 8, "frame_4_centroid_0": 14.0},
        id="smoothing-teleport-pulled-between-neighbours"),
    pytest.param(
        _a_track_too_short_to_smooth_is_returned_unchanged,
        {"rows": 2, "centroid_0": [10.0, 411.0],
         "input_centroid_0": [10.0, 411.0]},
        id="smoothing-two-frame-track-unchanged"),
    pytest.param(
        _a_track_of_three_frames_keeps_three_feature_values,
        {"rows": 3, "area": [100.0, 101.0, 102.0], "area_present": True,
         "centroid_0": [10.0, 11.0, 12.0]},
        id="smoothing-three-frame-track-keeps-its-features"),
    pytest.param(
        _an_empty_frame_is_returned_as_it_came,
        {"empty": True,
         "printed": "[_smooth_tracks_and_features] Input DataFrame is empty."},
        id="smoothing-empty-frame-comes-back"),
    pytest.param(
        _a_frame_with_no_centroids_is_left_alone,
        {"rows": 8, "input_rows": 8,
         "printed": "[_smooth_tracks_and_features] Centroid columns missing, "
                    "nothing to smooth."},
        id="smoothing-no-centroids-left-alone"),
]


@pytest.mark.parametrize("check,expected", _LINK_AND_REPAIR_ROWS)
def test_tracks_are_linked_and_repaired_without_relying_on_empty_labels_or_end_frames(
        check, expected, request):
    observed = _run(check, request)

    assert observed == expected

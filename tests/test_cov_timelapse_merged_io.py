"""
Coverage for the merged/*.npy I/O + regionprops helpers in spacr.timelapse.

Covers (spacr/timelapse.py lines ~2059-2617):

  * _compute_cell_mean_intensity_per_channel  — happy path + every guard
  * _summarise_child_features_per_parent      — empty-merge guard, default agg
  * _load_intensity_stack_from_merged         — both plane layouts + all skips
  * _load_masks_from_merged                   — cell/nucleus/pathogen plane maps
  * _compute_regionprops_stack                — geom-only vs geom+intensity

All tests are CPU-only, offline and operate on tiny (8x8) synthetic label /
intensity stacks written to tmp_path as real .npy files.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


H = W = 8


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _merged_dir(tmp_path):
    d = tmp_path / "merged"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _blob(h=H, w=W, label=1, box=(1, 4, 1, 4), dtype=np.int32):
    """Single rectangular label blob on an otherwise empty (h, w) canvas."""
    m = np.zeros((h, w), dtype=dtype)
    y0, y1, x0, x1 = box
    m[y0:y1, x0:x1] = label
    return m


def _write_merged(tmp_path, name, arr):
    """np.save `arr` under merged/<name> keeping the exact filename given."""
    d = _merged_dir(tmp_path)
    path = d / name
    if name.endswith(".npy"):
        np.save(path, arr)
    else:
        # np.save would append '.npy'; save then rename to keep the extension.
        tmp = d / (name + ".npy")
        np.save(tmp, arr)
        os.replace(tmp, path)
    return path


def _planes_first(n_channels, mask_planes, height=H, width=W):
    """Build a (planes, H, W) merged array: channel c is filled with (c+1)*100,
    mask plane k holds label k+1 in a distinct rectangle."""
    planes = []
    for c in range(n_channels):
        planes.append(np.full((height, width), (c + 1) * 100.0, dtype=np.float32))
    boxes = [(0, 3, 0, 3), (4, 7, 4, 7), (0, 3, 4, 7)]
    for k in range(mask_planes):
        planes.append(
            _blob(height, width, label=k + 1, box=boxes[k % len(boxes)],
                  dtype=np.int32).astype(np.float32)
        )
    return np.stack(planes, axis=0)


# ===========================================================================
# _compute_cell_mean_intensity_per_channel
# ===========================================================================

def _mask_and_intensity(T=3, n_labels=2, C=2):
    """(T, H, W) label stack + (T, H, W, C) intensity stack with known means."""
    masks = np.zeros((T, H, W), dtype=np.int32)
    inten = np.zeros((T, H, W, C), dtype=np.float32)
    for t in range(T):
        masks[t, 0:3, 0:3] = 1
        if n_labels > 1:
            masks[t, 5:8, 5:8] = 2
        for c in range(C):
            # label 1 region -> 10*(t+1)+c ; label 2 region -> 50*(t+1)+c
            inten[t, :, :, c] = 1.0
            inten[t, 0:3, 0:3, c] = 10.0 * (t + 1) + c
            inten[t, 5:8, 5:8, c] = 50.0 * (t + 1) + c
    return masks, inten


def test_cell_mean_intensity_per_channel_values_and_shape():
    from spacr.timelapse import _compute_cell_mean_intensity_per_channel
    masks, inten = _mask_and_intensity(T=3, n_labels=2, C=2)

    df = _compute_cell_mean_intensity_per_channel(masks, inten, channel_index=1)

    assert list(df.columns) == ["track_id", "cell_mean_intensity_ch1", "frame"]
    # 3 frames x 2 objects
    assert len(df) == 6
    assert sorted(df["frame"].unique().tolist()) == [0, 1, 2]
    assert sorted(df["track_id"].unique().tolist()) == [1, 2]

    row = df[(df["frame"] == 2) & (df["track_id"] == 2)].iloc[0]
    assert row["cell_mean_intensity_ch1"] == pytest.approx(50.0 * 3 + 1)
    row0 = df[(df["frame"] == 0) & (df["track_id"] == 1)].iloc[0]
    assert row0["cell_mean_intensity_ch1"] == pytest.approx(10.0 + 1)


def test_cell_mean_intensity_skips_empty_frames_only():
    """Frames with no labels are dropped, the rest still contribute rows."""
    from spacr.timelapse import _compute_cell_mean_intensity_per_channel
    masks, inten = _mask_and_intensity(T=3, n_labels=1, C=1)
    masks[1] = 0  # blank the middle frame

    df = _compute_cell_mean_intensity_per_channel(masks, inten, channel_index=0)

    assert sorted(df["frame"].unique().tolist()) == [0, 2]
    assert len(df) == 2


def test_cell_mean_intensity_none_stack_returns_empty_frame(capsys):
    from spacr.timelapse import _compute_cell_mean_intensity_per_channel
    masks, _ = _mask_and_intensity(T=2, C=1)

    df = _compute_cell_mean_intensity_per_channel(masks, None, channel_index=0)

    assert df.empty
    assert list(df.columns) == ["frame", "track_id", "cell_mean_intensity_ch0"]
    assert "intensity_stack is None" in capsys.readouterr().out


@pytest.mark.parametrize("bad_channel", [None, -1, 2, 99])
def test_cell_mean_intensity_invalid_channel_returns_empty_frame(bad_channel, capsys):
    from spacr.timelapse import _compute_cell_mean_intensity_per_channel
    masks, inten = _mask_and_intensity(T=2, C=2)

    df = _compute_cell_mean_intensity_per_channel(masks, inten, bad_channel)

    assert df.empty
    assert list(df.columns) == [
        "frame", "track_id", f"cell_mean_intensity_ch{bad_channel}"
    ]
    assert "invalid channel index" in capsys.readouterr().out


def test_cell_mean_intensity_all_frames_empty_returns_empty_frame(capsys):
    from spacr.timelapse import _compute_cell_mean_intensity_per_channel
    masks = np.zeros((4, H, W), dtype=np.int32)
    inten = np.ones((4, H, W, 1), dtype=np.float32)

    df = _compute_cell_mean_intensity_per_channel(masks, inten, channel_index=0)

    assert df.empty
    assert list(df.columns) == ["frame", "track_id", "cell_mean_intensity_ch0"]
    assert "no objects found in any of 4 frames" in capsys.readouterr().out


# ===========================================================================
# _summarise_child_features_per_parent — remaining branches
# ===========================================================================

def test_summarise_child_features_default_agg_is_mean_for_other_features():
    """A numeric feature matching none of area/intensity/dist is averaged."""
    from spacr.timelapse import _summarise_child_features_per_parent
    overlaps = pd.DataFrame(
        {"frame": [0, 0], "cell_id": [1, 1], "obj_id": [1, 2]}
    )
    props = pd.DataFrame(
        {
            "frame": [0, 0],
            "obj_id": [1, 2],
            "solidity": [0.5, 0.9],     # -> mean (default branch)
            "perimeter": [10.0, 30.0],  # -> mean (default branch)
        }
    )

    out = _summarise_child_features_per_parent(
        overlaps, props, "cell_id", "obj_id", "child_count"
    )

    row = out.loc[out["cell_id"] == 1].iloc[0]
    assert row["child_count"] == 2
    assert row["solidity"] == pytest.approx(0.7)
    assert row["perimeter"] == pytest.approx(20.0)


def test_summarise_child_features_empty_merge_result_returns_stub_frame():
    """Defensive branch: if the join yields nothing, a stub frame comes back.

    A real left-join can never empty a non-empty left frame, so the branch is
    exercised with a duck-typed overlaps object whose merge() returns empty.
    """
    from spacr.timelapse import _summarise_child_features_per_parent

    class _EmptyMergeOverlaps:
        empty = False

        def __init__(self):
            self.calls = []

        def merge(self, other, on=None, how=None):
            self.calls.append((tuple(on), how))
            return pd.DataFrame(columns=["frame", "obj_id"])

    overlaps = _EmptyMergeOverlaps()
    props = pd.DataFrame({"frame": [0], "obj_id": [1], "area": [10]})

    out = _summarise_child_features_per_parent(
        overlaps, props, "cell_id", "obj_id", "child_count"
    )

    assert overlaps.calls == [(("frame", "obj_id"), "left")]
    assert out.empty
    assert list(out.columns) == ["frame", "cell_id", "child_count"]


# ===========================================================================
# _load_intensity_stack_from_merged
# ===========================================================================

def test_load_intensity_stack_planes_first(tmp_path):
    from spacr.timelapse import _load_intensity_stack_from_merged
    names = ["p1_A01_1_t000.npy", "p1_A01_1_t001.npy"]
    for i, n in enumerate(names):
        arr = _planes_first(n_channels=2, mask_planes=1)
        arr[0] += i  # make the frames distinguishable
        _write_merged(tmp_path, n, arr)

    stack = _load_intensity_stack_from_merged(
        str(tmp_path), names, n_channels=2, height=H, width=W
    )

    assert stack.shape == (2, H, W, 2)
    assert stack.dtype == np.float32
    assert np.all(stack[0, :, :, 0] == 100.0)
    assert np.all(stack[1, :, :, 0] == 101.0)
    assert np.all(stack[:, :, :, 1] == 200.0)


def test_load_intensity_stack_planes_last_layout(tmp_path):
    """(H, W, planes) arrays are reoriented before slicing channels."""
    from spacr.timelapse import _load_intensity_stack_from_merged
    arr = np.moveaxis(_planes_first(n_channels=2, mask_planes=1), 0, -1)
    assert arr.shape == (H, W, 3)
    _write_merged(tmp_path, "p1_A01_1_t000.npy", arr)

    stack = _load_intensity_stack_from_merged(
        str(tmp_path), ["p1_A01_1_t000.npy"], n_channels=2, height=H, width=W
    )

    assert stack.shape == (1, H, W, 2)
    assert np.all(stack[0, :, :, 0] == 100.0)
    assert np.all(stack[0, :, :, 1] == 200.0)


def test_load_intensity_stack_custom_dtype_and_fewer_planes_than_channels(tmp_path):
    """planes < n_channels -> only the available channels are filled."""
    from spacr.timelapse import _load_intensity_stack_from_merged
    arr = _planes_first(n_channels=2, mask_planes=0)  # 2 planes only
    _write_merged(tmp_path, "p1_A01_1_t000.npy", arr)

    stack = _load_intensity_stack_from_merged(
        str(tmp_path), ["p1_A01_1_t000.npy"], n_channels=3,
        height=H, width=W, dtype=np.uint16,
    )

    assert stack.dtype == np.uint16
    assert stack.shape == (1, H, W, 3)
    assert np.all(stack[0, :, :, 0] == 100)
    assert np.all(stack[0, :, :, 1] == 200)
    assert np.all(stack[0, :, :, 2] == 0)  # third channel never written


def test_load_intensity_stack_no_merged_dir_returns_zero_channels(tmp_path):
    from spacr.timelapse import _load_intensity_stack_from_merged
    stack = _load_intensity_stack_from_merged(
        str(tmp_path), ["a.npy", "b.npy"], n_channels=2, height=H, width=W
    )
    assert stack.shape == (2, H, W, 0)
    assert stack.size == 0


@pytest.mark.parametrize("n_channels", [None, 0, -1])
def test_load_intensity_stack_non_positive_channels(tmp_path, n_channels):
    from spacr.timelapse import _load_intensity_stack_from_merged
    _write_merged(tmp_path, "p1_A01_1_t000.npy", _planes_first(2, 1))

    stack = _load_intensity_stack_from_merged(
        str(tmp_path), ["p1_A01_1_t000.npy"], n_channels=n_channels,
        height=H, width=W,
    )
    assert stack.shape == (1, H, W, 0)


def test_load_intensity_stack_missing_and_non3d_files_leave_zeros(tmp_path):
    from spacr.timelapse import _load_intensity_stack_from_merged
    _write_merged(tmp_path, "f0.npy", _planes_first(2, 1))
    _write_merged(tmp_path, "f2.npy", np.ones((H, W), dtype=np.float32))  # 2-D

    stack = _load_intensity_stack_from_merged(
        str(tmp_path), ["f0.npy", "missing.npy", "f2.npy"],
        n_channels=2, height=H, width=W,
    )

    assert stack.shape == (3, H, W, 2)
    assert np.all(stack[0, :, :, 0] == 100.0)
    assert np.all(stack[1] == 0)  # file not on disk
    assert np.all(stack[2] == 0)  # 2-D array rejected


def test_load_intensity_stack_alternate_filename_candidate(tmp_path):
    """Falls back to the literal filename when <base>.npy does not exist."""
    from spacr.timelapse import _load_intensity_stack_from_merged
    _write_merged(tmp_path, "f0.tif", _planes_first(2, 1))
    assert not (tmp_path / "merged" / "f0.npy").exists()

    stack = _load_intensity_stack_from_merged(
        str(tmp_path), ["f0.tif"], n_channels=2, height=H, width=W
    )
    assert np.all(stack[0, :, :, 1] == 200.0)


def test_load_intensity_stack_size_mismatch_is_skipped(tmp_path, capsys):
    from spacr.timelapse import _load_intensity_stack_from_merged
    _write_merged(tmp_path, "f0.npy", _planes_first(2, 1, height=6, width=6))

    stack = _load_intensity_stack_from_merged(
        str(tmp_path), ["f0.npy"], n_channels=2, height=H, width=W
    )

    assert np.all(stack == 0)
    out = capsys.readouterr().out
    assert "_load_intensity_stack_from_merged] Skipping f0.npy" in out
    assert "expected H=8, W=8" in out


def test_load_intensity_stack_zero_plane_array_is_skipped(tmp_path):
    """A (0, H, W) array reorients to planes=0 -> nothing to copy."""
    from spacr.timelapse import _load_intensity_stack_from_merged
    _write_merged(tmp_path, "f0.npy", np.zeros((0, H, W), dtype=np.float32))

    stack = _load_intensity_stack_from_merged(
        str(tmp_path), ["f0.npy"], n_channels=2, height=H, width=W
    )
    assert stack.shape == (1, H, W, 2)
    assert np.all(stack == 0)


def test_load_intensity_stack_reorient_valueerror_is_swallowed(tmp_path, monkeypatch):
    from spacr import timelapse as TL

    def _boom(arr, n_channels, **kwargs):
        raise ValueError("injected reorientation failure")

    monkeypatch.setattr(TL, "_reorient_merged_array", _boom)
    _write_merged(tmp_path, "f0.npy", _planes_first(2, 1))

    stack = TL._load_intensity_stack_from_merged(
        str(tmp_path), ["f0.npy"], n_channels=2, height=H, width=W
    )
    assert stack.shape == (1, H, W, 2)
    assert np.all(stack == 0)


# ===========================================================================
# _load_masks_from_merged
# ===========================================================================

def test_load_masks_cell_only_plane(tmp_path):
    from spacr.timelapse import _load_masks_from_merged
    _write_merged(tmp_path, "f0.npy", _planes_first(n_channels=2, mask_planes=1))

    cell, nuc, path = _load_masks_from_merged(
        str(tmp_path), ["f0.npy"], n_channels=2, height=H, width=W
    )

    assert cell.dtype == np.int32
    assert cell.shape == (1, H, W)
    assert set(np.unique(cell)) == {0, 1}
    assert cell[0, 0, 0] == 1 and cell[0, 7, 7] == 0
    assert not nuc.any()
    assert not path.any()


def test_load_masks_second_plane_is_nucleus_when_only_nucleus_requested(tmp_path):
    from spacr.timelapse import _load_masks_from_merged
    _write_merged(tmp_path, "f0.npy", _planes_first(n_channels=2, mask_planes=2))

    cell, nuc, path = _load_masks_from_merged(
        str(tmp_path), ["f0.npy"], n_channels=2, height=H, width=W,
        nucleus_chan=0, pathogen_chan=None,
    )

    assert set(np.unique(cell)) == {0, 1}
    assert set(np.unique(nuc)) == {0, 2}
    assert nuc[0, 5, 5] == 2
    assert not path.any()


def test_load_masks_second_plane_is_pathogen_when_only_pathogen_requested(tmp_path):
    from spacr.timelapse import _load_masks_from_merged
    _write_merged(tmp_path, "f0.npy", _planes_first(n_channels=2, mask_planes=2))

    cell, nuc, path = _load_masks_from_merged(
        str(tmp_path), ["f0.npy"], n_channels=2, height=H, width=W,
        nucleus_chan=None, pathogen_chan=1,
    )

    assert set(np.unique(cell)) == {0, 1}
    assert not nuc.any()
    assert set(np.unique(path)) == {0, 2}
    assert path[0, 5, 5] == 2


def test_load_masks_three_planes_nucleus_then_pathogen(tmp_path):
    from spacr.timelapse import _load_masks_from_merged
    _write_merged(tmp_path, "f0.npy", _planes_first(n_channels=2, mask_planes=3))

    cell, nuc, path = _load_masks_from_merged(
        str(tmp_path), ["f0.npy"], n_channels=2, height=H, width=W,
        nucleus_chan=0, pathogen_chan=1, dtype=np.uint16,
    )

    assert cell.dtype == nuc.dtype == path.dtype == np.uint16
    assert set(np.unique(cell)) == {0, 1}
    assert set(np.unique(nuc)) == {0, 2}
    assert set(np.unique(path)) == {0, 3}
    assert path[0, 0, 5] == 3


def test_load_masks_second_plane_ignored_when_no_child_channels(tmp_path):
    """nucleus_chan and pathogen_chan both None -> extra planes are dropped."""
    from spacr.timelapse import _load_masks_from_merged
    _write_merged(tmp_path, "f0.npy", _planes_first(n_channels=2, mask_planes=3))

    cell, nuc, path = _load_masks_from_merged(
        str(tmp_path), ["f0.npy"], n_channels=2, height=H, width=W,
        nucleus_chan=None, pathogen_chan=None,
    )

    assert cell.any()
    assert not nuc.any()
    assert not path.any()


def test_load_masks_no_mask_planes_returns_zeros(tmp_path):
    from spacr.timelapse import _load_masks_from_merged
    _write_merged(tmp_path, "f0.npy", _planes_first(n_channels=2, mask_planes=0))

    cell, nuc, path = _load_masks_from_merged(
        str(tmp_path), ["f0.npy"], n_channels=2, height=H, width=W
    )

    assert not cell.any() and not nuc.any() and not path.any()


def test_load_masks_no_merged_dir_returns_zero_stacks(tmp_path):
    from spacr.timelapse import _load_masks_from_merged
    cell, nuc, path = _load_masks_from_merged(
        str(tmp_path), ["a.npy", "b.npy"], n_channels=2, height=H, width=W
    )
    for m in (cell, nuc, path):
        assert m.shape == (2, H, W)
        assert m.dtype == np.int32
        assert not m.any()


def test_load_masks_missing_and_non3d_files_leave_zeros(tmp_path):
    from spacr.timelapse import _load_masks_from_merged
    _write_merged(tmp_path, "f0.npy", _planes_first(2, 1))
    _write_merged(tmp_path, "f2.npy", np.ones((H, W), dtype=np.float32))

    cell, _, _ = _load_masks_from_merged(
        str(tmp_path), ["f0.npy", "gone.npy", "f2.npy"],
        n_channels=2, height=H, width=W,
    )

    assert cell[0].any()
    assert not cell[1].any()
    assert not cell[2].any()


def test_load_masks_planes_last_layout(tmp_path):
    from spacr.timelapse import _load_masks_from_merged
    arr = np.moveaxis(_planes_first(n_channels=2, mask_planes=2), 0, -1)
    _write_merged(tmp_path, "f0.npy", arr)

    cell, nuc, _ = _load_masks_from_merged(
        str(tmp_path), ["f0.npy"], n_channels=2, height=H, width=W,
        nucleus_chan=0, pathogen_chan=None,
    )

    assert set(np.unique(cell)) == {0, 1}
    assert set(np.unique(nuc)) == {0, 2}


def test_load_masks_size_mismatch_is_skipped(tmp_path, capsys):
    from spacr.timelapse import _load_masks_from_merged
    _write_merged(tmp_path, "f0.npy", _planes_first(2, 1, height=6, width=6))

    cell, _, _ = _load_masks_from_merged(
        str(tmp_path), ["f0.npy"], n_channels=2, height=H, width=W
    )

    assert not cell.any()
    out = capsys.readouterr().out
    assert "_load_masks_from_merged] Skipping f0.npy" in out


def test_load_masks_reorient_valueerror_is_swallowed(tmp_path, monkeypatch):
    from spacr import timelapse as TL

    def _boom(arr, n_channels, **kwargs):
        raise ValueError("injected reorientation failure")

    monkeypatch.setattr(TL, "_reorient_merged_array", _boom)
    _write_merged(tmp_path, "f0.npy", _planes_first(2, 1))

    cell, nuc, path = TL._load_masks_from_merged(
        str(tmp_path), ["f0.npy"], n_channels=2, height=H, width=W
    )
    assert not cell.any() and not nuc.any() and not path.any()


def test_load_masks_alternate_filename_candidate(tmp_path):
    from spacr.timelapse import _load_masks_from_merged
    _write_merged(tmp_path, "f0.tif", _planes_first(2, 1))
    assert not (tmp_path / "merged" / "f0.npy").exists()

    cell, _, _ = _load_masks_from_merged(
        str(tmp_path), ["f0.tif"], n_channels=2, height=H, width=W
    )
    assert set(np.unique(cell)) == {0, 1}


# ===========================================================================
# _compute_regionprops_stack
# ===========================================================================

def test_regionprops_stack_with_intensity_columns_and_values():
    from spacr.timelapse import _compute_regionprops_stack
    masks, inten = _mask_and_intensity(T=2, n_labels=2, C=2)

    df = _compute_regionprops_stack(
        masks, inten, channel_index=0, object_prefix="cell"
    )

    assert len(df) == 4  # 2 frames x 2 objects
    assert "cell_label" in df.columns and "frame" in df.columns
    for col in ("cell_area", "cell_perimeter", "cell_solidity",
                "cell_centroid-0", "cell_centroid-1",
                "cell_mean_intensity", "cell_max_intensity",
                "cell_min_intensity", "cell_bbox_area",
                "cell_equivalent_diameter", "cell_perimeter_crofton"):
        assert col in df.columns, col

    row = df[(df["frame"] == 1) & (df["cell_label"] == 1)].iloc[0]
    assert row["cell_area"] == 9            # 3x3 block
    assert row["cell_mean_intensity"] == pytest.approx(20.0)  # 10*(t+1), c=0
    assert row["cell_max_intensity"] == pytest.approx(20.0)


def test_regionprops_stack_label_as_track_id_keeps_label_unprefixed():
    from spacr.timelapse import _compute_regionprops_stack
    masks, inten = _mask_and_intensity(T=1, n_labels=2, C=1)

    df = _compute_regionprops_stack(
        masks, inten, channel_index=0, object_prefix="cell",
        label_as_track_id=True,
    )

    assert "track_id" in df.columns
    assert "cell_label" not in df.columns
    assert "cell_track_id" not in df.columns
    assert sorted(df["track_id"].tolist()) == [1, 2]


@pytest.mark.parametrize("stack_kind,channel", [
    ("none", 0),      # intensity_stack is None
    ("real", None),   # channel_index is None
    ("real", 7),      # channel index past the end
    ("real", -1),     # negative channel index
])
def test_regionprops_stack_without_intensity_props(stack_kind, channel):
    from spacr.timelapse import _compute_regionprops_stack
    masks, inten = _mask_and_intensity(T=2, n_labels=1, C=2)
    stack = None if stack_kind == "none" else inten

    df = _compute_regionprops_stack(
        masks, stack, channel_index=channel, object_prefix="nucleus"
    )

    assert len(df) == 2
    assert "nucleus_area" in df.columns
    for col in df.columns:
        assert "intensity" not in col


def test_regionprops_stack_skips_blank_frames():
    from spacr.timelapse import _compute_regionprops_stack
    masks, inten = _mask_and_intensity(T=3, n_labels=1, C=1)
    masks[0] = 0

    df = _compute_regionprops_stack(
        masks, inten, channel_index=0, object_prefix="pathogen"
    )

    assert sorted(df["frame"].unique().tolist()) == [1, 2]
    assert len(df) == 2


def test_regionprops_stack_all_blank_returns_empty_frame(capsys):
    from spacr.timelapse import _compute_regionprops_stack
    masks = np.zeros((3, H, W), dtype=np.int32)

    df = _compute_regionprops_stack(
        masks, None, channel_index=None, object_prefix="cytoplasm"
    )

    assert df.empty
    assert list(df.columns) == ["frame", "cytoplasm_label"]
    assert "cytoplasm: no objects found in any of 3 frames" in capsys.readouterr().out

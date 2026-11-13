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

import functools
import inspect
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


def _case_cell_mean_intensity_per_channel_values_and_shape():
    observed = {}
    from spacr.timelapse import _compute_cell_mean_intensity_per_channel
    masks, inten = _mask_and_intensity(T=3, n_labels=2, C=2)

    df = _compute_cell_mean_intensity_per_channel(masks, inten, channel_index=1)

    observed['list(df.columns)'] = list(df.columns)
    # 3 frames x 2 objects
    observed['len(df)'] = len(df)
    observed["sorted(df['frame'].unique().tolist())"] = sorted(df['frame'].unique().tolist())
    observed["sorted(df['track_id'].unique().tolist())"] = sorted(df['track_id'].unique().tolist())

    row = df[(df["frame"] == 2) & (df["track_id"] == 2)].iloc[0]
    observed["row['cell_mean_intensity_ch1']"] = row['cell_mean_intensity_ch1']
    row0 = df[(df["frame"] == 0) & (df["track_id"] == 1)].iloc[0]
    observed["row0['cell_mean_intensity_ch1']"] = row0['cell_mean_intensity_ch1']
    return observed


_EXPECT_cell_mean_intensity_per_channel_values_and_shape = {
    'list(df.columns)': ['track_id', 'cell_mean_intensity_ch1', 'frame'],
    'len(df)': 6,
    "sorted(df['frame'].unique().tolist())": [0, 1, 2],
    "sorted(df['track_id'].unique().tolist())": [1, 2],
    "row['cell_mean_intensity_ch1']": pytest.approx(50.0 * 3 + 1),
    "row0['cell_mean_intensity_ch1']": pytest.approx(10.0 + 1),
}


def _case_cell_mean_intensity_skips_empty_frames_only():
    """Frames with no labels are dropped, the rest still contribute rows."""
    observed = {}
    from spacr.timelapse import _compute_cell_mean_intensity_per_channel
    masks, inten = _mask_and_intensity(T=3, n_labels=1, C=1)
    masks[1] = 0  # blank the middle frame

    df = _compute_cell_mean_intensity_per_channel(masks, inten, channel_index=0)

    observed["sorted(df['frame'].unique().tolist())"] = sorted(df['frame'].unique().tolist())
    observed['len(df)'] = len(df)
    return observed


_EXPECT_cell_mean_intensity_skips_empty_frames_only = {
    "sorted(df['frame'].unique().tolist())": [0, 2],
    'len(df)': 2,
}


def _case_cell_mean_intensity_returns_an_empty_frame_when_nothing_can_be_measured(
        scenario, channel, column, message, capsys):
    """No stack, an out-of-range channel or no labelled cell at all: empty frame, columns kept, reason printed."""
    observed = {}
    from spacr.timelapse import _compute_cell_mean_intensity_per_channel
    if scenario == "no_stack":
        masks, _ = _mask_and_intensity(T=2, C=1)
        inten = None
    elif scenario == "bad_channel":
        masks, inten = _mask_and_intensity(T=2, C=2)
    else:
        masks = np.zeros((4, H, W), dtype=np.int32)
        inten = np.ones((4, H, W, 1), dtype=np.float32)

    df = _compute_cell_mean_intensity_per_channel(masks, inten, channel)

    observed['df.empty'] = bool(df.empty)
    observed["list(df.columns) == ['frame', 'track_id', column]"] = bool(list(df.columns) == ['frame', 'track_id', column])
    observed['message in capsys.readouterr().out'] = bool(message in capsys.readouterr().out)
    return observed


_EXPECT_cell_mean_intensity_returns_an_empty_frame_when_nothing_can_be_measured = {
    'df.empty': True,
    "list(df.columns) == ['frame', 'track_id', column]": True,
    'message in capsys.readouterr().out': True,
}


# ===========================================================================
# _summarise_child_features_per_parent — remaining branches
# ===========================================================================

def _case_summarise_child_features_default_agg_is_mean_for_other_features():
    """A numeric feature matching none of area/intensity/dist is averaged."""
    observed = {}
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
    observed["row['child_count']"] = row['child_count']
    observed["row['solidity']"] = row['solidity']
    observed["row['perimeter']"] = row['perimeter']
    return observed


_EXPECT_summarise_child_features_default_agg_is_mean_for_other_features = {
    "row['child_count']": 2,
    "row['solidity']": pytest.approx(0.7),
    "row['perimeter']": pytest.approx(20.0),
}


def _case_summarise_child_features_empty_merge_result_returns_stub_frame():
    """Defensive branch: if the join yields nothing, a stub frame comes back.

    A real left-join can never empty a non-empty left frame, so the branch is
    exercised with a duck-typed overlaps object whose merge() returns empty.
    """
    observed = {}
    from spacr.timelapse import _summarise_child_features_per_parent

    class _EmptyMergeOverlaps:
        empty = False

        def __init__(self):
            self.calls = []

        # `validate` is part of the call the function makes: child_props_df is
        # regionprops output and must hold one row per label per frame. The
        # stub records it, so this test pins that contract as well.
        def merge(self, other, on=None, how=None, validate=None):
            self.calls.append((tuple(on), how, validate))
            return pd.DataFrame(columns=["frame", "obj_id"])

    overlaps = _EmptyMergeOverlaps()
    props = pd.DataFrame({"frame": [0], "obj_id": [1], "area": [10]})

    out = _summarise_child_features_per_parent(
        overlaps, props, "cell_id", "obj_id", "child_count"
    )

    observed['overlaps.calls'] = overlaps.calls
    observed['out.empty'] = bool(out.empty)
    observed['list(out.columns)'] = list(out.columns)
    return observed


_EXPECT_summarise_child_features_empty_merge_result_returns_stub_frame = {
    'overlaps.calls': [(('frame', 'obj_id'), 'left', 'many_to_one')],
    'out.empty': True,
    'list(out.columns)': ['frame', 'cell_id', 'child_count'],
}


# ===========================================================================
# _load_intensity_stack_from_merged
# ===========================================================================

def _case_load_intensity_stack_planes_first(tmp_path):
    observed = {}
    from spacr.timelapse import _load_intensity_stack_from_merged
    names = ["p1_A01_1_t000.npy", "p1_A01_1_t001.npy"]
    for i, n in enumerate(names):
        arr = _planes_first(n_channels=2, mask_planes=1)
        arr[0] += i  # make the frames distinguishable
        _write_merged(tmp_path, n, arr)

    stack = _load_intensity_stack_from_merged(
        str(tmp_path), names, n_channels=2, height=H, width=W
    )

    observed['stack.shape'] = stack.shape
    observed['stack.dtype'] = stack.dtype
    observed['np.all(stack[0, :, :, 0] == 100.0)'] = bool(np.all(stack[0, :, :, 0] == 100.0))
    observed['np.all(stack[1, :, :, 0] == 101.0)'] = bool(np.all(stack[1, :, :, 0] == 101.0))
    observed['np.all(stack[:, :, :, 1] == 200.0)'] = bool(np.all(stack[:, :, :, 1] == 200.0))
    return observed


_EXPECT_load_intensity_stack_planes_first = {
    'stack.shape': (2, H, W, 2),
    'stack.dtype': np.float32,
    'np.all(stack[0, :, :, 0] == 100.0)': True,
    'np.all(stack[1, :, :, 0] == 101.0)': True,
    'np.all(stack[:, :, :, 1] == 200.0)': True,
}


def _case_load_intensity_stack_planes_last_layout(tmp_path):
    """(H, W, planes) arrays are reoriented before slicing channels."""
    observed = {}
    from spacr.timelapse import _load_intensity_stack_from_merged
    arr = np.moveaxis(_planes_first(n_channels=2, mask_planes=1), 0, -1)
    observed['arr.shape'] = arr.shape
    _write_merged(tmp_path, "p1_A01_1_t000.npy", arr)

    stack = _load_intensity_stack_from_merged(
        str(tmp_path), ["p1_A01_1_t000.npy"], n_channels=2, height=H, width=W
    )

    observed['stack.shape'] = stack.shape
    observed['np.all(stack[0, :, :, 0] == 100.0)'] = bool(np.all(stack[0, :, :, 0] == 100.0))
    observed['np.all(stack[0, :, :, 1] == 200.0)'] = bool(np.all(stack[0, :, :, 1] == 200.0))
    return observed


_EXPECT_load_intensity_stack_planes_last_layout = {
    'arr.shape': (H, W, 3),
    'stack.shape': (1, H, W, 2),
    'np.all(stack[0, :, :, 0] == 100.0)': True,
    'np.all(stack[0, :, :, 1] == 200.0)': True,
}


def _case_load_intensity_stack_custom_dtype_and_fewer_planes_than_channels(tmp_path):
    """planes < n_channels -> only the available channels are filled."""
    observed = {}
    from spacr.timelapse import _load_intensity_stack_from_merged
    arr = _planes_first(n_channels=2, mask_planes=0)  # 2 planes only
    _write_merged(tmp_path, "p1_A01_1_t000.npy", arr)

    stack = _load_intensity_stack_from_merged(
        str(tmp_path), ["p1_A01_1_t000.npy"], n_channels=3,
        height=H, width=W, dtype=np.uint16,
    )

    observed['stack.dtype'] = stack.dtype
    observed['stack.shape'] = stack.shape
    observed['np.all(stack[0, :, :, 0] == 100)'] = bool(np.all(stack[0, :, :, 0] == 100))
    observed['np.all(stack[0, :, :, 1] == 200)'] = bool(np.all(stack[0, :, :, 1] == 200))
    observed['np.all(stack[0, :, :, 2] == 0)'] = bool(np.all(stack[0, :, :, 2] == 0))
    return observed


_EXPECT_load_intensity_stack_custom_dtype_and_fewer_planes_than_channels = {
    'stack.dtype': np.uint16,
    'stack.shape': (1, H, W, 3),
    'np.all(stack[0, :, :, 0] == 100)': True,
    'np.all(stack[0, :, :, 1] == 200)': True,
    'np.all(stack[0, :, :, 2] == 0)': True,
}


def _case_load_intensity_stack_no_merged_dir_returns_zero_channels(tmp_path):
    observed = {}
    from spacr.timelapse import _load_intensity_stack_from_merged
    stack = _load_intensity_stack_from_merged(
        str(tmp_path), ["a.npy", "b.npy"], n_channels=2, height=H, width=W
    )
    observed['stack.shape'] = stack.shape
    observed['stack.size'] = stack.size
    return observed


_EXPECT_load_intensity_stack_no_merged_dir_returns_zero_channels = {
    'stack.shape': (2, H, W, 0),
    'stack.size': 0,
}


def _case_load_intensity_stack_non_positive_channels(tmp_path, n_channels):
    observed = {}
    from spacr.timelapse import _load_intensity_stack_from_merged
    _write_merged(tmp_path, "p1_A01_1_t000.npy", _planes_first(2, 1))

    stack = _load_intensity_stack_from_merged(
        str(tmp_path), ["p1_A01_1_t000.npy"], n_channels=n_channels,
        height=H, width=W,
    )
    observed['stack.shape'] = stack.shape
    return observed


_EXPECT_load_intensity_stack_non_positive_channels = {
    'stack.shape': (1, H, W, 0),
}


def _case_load_intensity_stack_missing_and_non3d_files_leave_zeros(tmp_path):
    observed = {}
    from spacr.timelapse import _load_intensity_stack_from_merged
    _write_merged(tmp_path, "f0.npy", _planes_first(2, 1))
    _write_merged(tmp_path, "f2.npy", np.ones((H, W), dtype=np.float32))  # 2-D

    stack = _load_intensity_stack_from_merged(
        str(tmp_path), ["f0.npy", "missing.npy", "f2.npy"],
        n_channels=2, height=H, width=W,
    )

    observed['stack.shape'] = stack.shape
    observed['np.all(stack[0, :, :, 0] == 100.0)'] = bool(np.all(stack[0, :, :, 0] == 100.0))
    observed['np.all(stack[1] == 0)'] = bool(np.all(stack[1] == 0))
    observed['np.all(stack[2] == 0)'] = bool(np.all(stack[2] == 0))
    return observed


_EXPECT_load_intensity_stack_missing_and_non3d_files_leave_zeros = {
    'stack.shape': (3, H, W, 2),
    'np.all(stack[0, :, :, 0] == 100.0)': True,
    'np.all(stack[1] == 0)': True,
    'np.all(stack[2] == 0)': True,
}


def _case_load_intensity_stack_alternate_filename_candidate(tmp_path):
    """Falls back to the literal filename when <base>.npy does not exist."""
    observed = {}
    from spacr.timelapse import _load_intensity_stack_from_merged
    _write_merged(tmp_path, "f0.tif", _planes_first(2, 1))
    observed["not (tmp_path / 'merged' / 'f0.npy').exists()"] = bool(not (tmp_path / 'merged' / 'f0.npy').exists())

    stack = _load_intensity_stack_from_merged(
        str(tmp_path), ["f0.tif"], n_channels=2, height=H, width=W
    )
    observed['np.all(stack[0, :, :, 1] == 200.0)'] = bool(np.all(stack[0, :, :, 1] == 200.0))
    return observed


_EXPECT_load_intensity_stack_alternate_filename_candidate = {
    "not (tmp_path / 'merged' / 'f0.npy').exists()": True,
    'np.all(stack[0, :, :, 1] == 200.0)': True,
}


def _case_load_intensity_stack_leaves_a_frame_zero_when_its_file_cannot_be_used(
        tmp_path, monkeypatch, capsys, fault):
    """A frame file of the wrong size, with no planes, or that cannot be reoriented is skipped, not fatal."""
    observed = {}
    from spacr import timelapse as TL
    if fault == "wrong_size":
        arr = _planes_first(2, 1, height=6, width=6)
    elif fault == "zero_planes":
        arr = np.zeros((0, H, W), dtype=np.float32)
    else:
        arr = _planes_first(2, 1)

        def _boom(arr, n_channels, **kwargs):
            raise ValueError("injected reorientation failure")

        monkeypatch.setattr(TL, "_reorient_merged_array", _boom)
    _write_merged(tmp_path, "f0.npy", arr)

    stack = TL._load_intensity_stack_from_merged(
        str(tmp_path), ["f0.npy"], n_channels=2, height=H, width=W
    )

    observed['stack.shape'] = stack.shape
    observed['np.all(stack == 0)'] = bool(np.all(stack == 0))
    out = capsys.readouterr().out
    if fault == "wrong_size":
        assert "_load_intensity_stack_from_merged] Skipping f0.npy" in out
        assert "expected H=8, W=8" in out
    return observed


_EXPECT_load_intensity_stack_leaves_a_frame_zero_when_its_file_cannot_be_used = {
    'stack.shape': (1, H, W, 2),
    'np.all(stack == 0)': True,
}


# ===========================================================================
# _load_masks_from_merged
# ===========================================================================

def _case_load_masks_cell_only_plane(tmp_path):
    observed = {}
    from spacr.timelapse import _load_masks_from_merged
    _write_merged(tmp_path, "f0.npy", _planes_first(n_channels=2, mask_planes=1))

    cell, nuc, path = _load_masks_from_merged(
        str(tmp_path), ["f0.npy"], n_channels=2, height=H, width=W
    )

    observed['cell.dtype'] = cell.dtype
    observed['cell.shape'] = cell.shape
    observed['set(np.unique(cell))'] = set(np.unique(cell))
    observed['cell[0, 0, 0] == 1 and cell[0, 7, 7] == 0'] = bool(cell[0, 0, 0] == 1 and cell[0, 7, 7] == 0)
    observed['not nuc.any()'] = bool(not nuc.any())
    observed['not path.any()'] = bool(not path.any())
    return observed


_EXPECT_load_masks_cell_only_plane = {
    'cell.dtype': np.int32,
    'cell.shape': (1, H, W),
    'set(np.unique(cell))': {0, 1},
    'cell[0, 0, 0] == 1 and cell[0, 7, 7] == 0': True,
    'not nuc.any()': True,
    'not path.any()': True,
}


def _case_load_masks_second_plane_is_nucleus_when_only_nucleus_requested(tmp_path):
    observed = {}
    from spacr.timelapse import _load_masks_from_merged
    _write_merged(tmp_path, "f0.npy", _planes_first(n_channels=2, mask_planes=2))

    cell, nuc, path = _load_masks_from_merged(
        str(tmp_path), ["f0.npy"], n_channels=2, height=H, width=W,
        nucleus_chan=0, pathogen_chan=None,
    )

    observed['set(np.unique(cell))'] = set(np.unique(cell))
    observed['set(np.unique(nuc))'] = set(np.unique(nuc))
    observed['nuc[0, 5, 5]'] = nuc[0, 5, 5]
    observed['not path.any()'] = bool(not path.any())
    return observed


_EXPECT_load_masks_second_plane_is_nucleus_when_only_nucleus_requested = {
    'set(np.unique(cell))': {0, 1},
    'set(np.unique(nuc))': {0, 2},
    'nuc[0, 5, 5]': 2,
    'not path.any()': True,
}


def _case_load_masks_second_plane_is_pathogen_when_only_pathogen_requested(tmp_path):
    observed = {}
    from spacr.timelapse import _load_masks_from_merged
    _write_merged(tmp_path, "f0.npy", _planes_first(n_channels=2, mask_planes=2))

    cell, nuc, path = _load_masks_from_merged(
        str(tmp_path), ["f0.npy"], n_channels=2, height=H, width=W,
        nucleus_chan=None, pathogen_chan=1,
    )

    observed['set(np.unique(cell))'] = set(np.unique(cell))
    observed['not nuc.any()'] = bool(not nuc.any())
    observed['set(np.unique(path))'] = set(np.unique(path))
    observed['path[0, 5, 5]'] = path[0, 5, 5]
    return observed


_EXPECT_load_masks_second_plane_is_pathogen_when_only_pathogen_requested = {
    'set(np.unique(cell))': {0, 1},
    'not nuc.any()': True,
    'set(np.unique(path))': {0, 2},
    'path[0, 5, 5]': 2,
}


def _case_load_masks_three_planes_nucleus_then_pathogen(tmp_path):
    observed = {}
    from spacr.timelapse import _load_masks_from_merged
    _write_merged(tmp_path, "f0.npy", _planes_first(n_channels=2, mask_planes=3))

    cell, nuc, path = _load_masks_from_merged(
        str(tmp_path), ["f0.npy"], n_channels=2, height=H, width=W,
        nucleus_chan=0, pathogen_chan=1, dtype=np.uint16,
    )

    observed['cell.dtype == nuc.dtype == path.dtype == np.uint16'] = bool(cell.dtype == nuc.dtype == path.dtype == np.uint16)
    observed['set(np.unique(cell))'] = set(np.unique(cell))
    observed['set(np.unique(nuc))'] = set(np.unique(nuc))
    observed['set(np.unique(path))'] = set(np.unique(path))
    observed['path[0, 0, 5]'] = path[0, 0, 5]
    return observed


_EXPECT_load_masks_three_planes_nucleus_then_pathogen = {
    'cell.dtype == nuc.dtype == path.dtype == np.uint16': True,
    'set(np.unique(cell))': {0, 1},
    'set(np.unique(nuc))': {0, 2},
    'set(np.unique(path))': {0, 3},
    'path[0, 0, 5]': 3,
}


def _case_load_masks_second_plane_ignored_when_no_child_channels(tmp_path):
    """nucleus_chan and pathogen_chan both None -> extra planes are dropped."""
    observed = {}
    from spacr.timelapse import _load_masks_from_merged
    _write_merged(tmp_path, "f0.npy", _planes_first(n_channels=2, mask_planes=3))

    cell, nuc, path = _load_masks_from_merged(
        str(tmp_path), ["f0.npy"], n_channels=2, height=H, width=W,
        nucleus_chan=None, pathogen_chan=None,
    )

    observed['cell.any()'] = bool(cell.any())
    observed['not nuc.any()'] = bool(not nuc.any())
    observed['not path.any()'] = bool(not path.any())
    return observed


_EXPECT_load_masks_second_plane_ignored_when_no_child_channels = {
    'cell.any()': True,
    'not nuc.any()': True,
    'not path.any()': True,
}


def _case_load_masks_no_mask_planes_returns_zeros(tmp_path):
    observed = {}
    from spacr.timelapse import _load_masks_from_merged
    _write_merged(tmp_path, "f0.npy", _planes_first(n_channels=2, mask_planes=0))

    cell, nuc, path = _load_masks_from_merged(
        str(tmp_path), ["f0.npy"], n_channels=2, height=H, width=W
    )

    observed['not cell.any() and (not nuc.any()) and (not path.any())'] = bool(not cell.any() and (not nuc.any()) and (not path.any()))
    return observed


_EXPECT_load_masks_no_mask_planes_returns_zeros = {
    'not cell.any() and (not nuc.any()) and (not path.any())': True,
}


def _case_load_masks_no_merged_dir_returns_zero_stacks(tmp_path):
    from spacr.timelapse import _load_masks_from_merged
    cell, nuc, path = _load_masks_from_merged(
        str(tmp_path), ["a.npy", "b.npy"], n_channels=2, height=H, width=W
    )
    return {
        "shapes": [m.shape for m in (cell, nuc, path)],
        "dtypes": [m.dtype for m in (cell, nuc, path)],
        "any_nonzero": [bool(m.any()) for m in (cell, nuc, path)],
    }


_EXPECT_load_masks_no_merged_dir_returns_zero_stacks = {
    "shapes": [(2, H, W)] * 3,
    "dtypes": [np.dtype(np.int32)] * 3,
    "any_nonzero": [False] * 3,
}


def _case_load_masks_missing_and_non3d_files_leave_zeros(tmp_path):
    observed = {}
    from spacr.timelapse import _load_masks_from_merged
    _write_merged(tmp_path, "f0.npy", _planes_first(2, 1))
    _write_merged(tmp_path, "f2.npy", np.ones((H, W), dtype=np.float32))

    cell, _, _ = _load_masks_from_merged(
        str(tmp_path), ["f0.npy", "gone.npy", "f2.npy"],
        n_channels=2, height=H, width=W,
    )

    observed['cell[0].any()'] = bool(cell[0].any())
    observed['not cell[1].any()'] = bool(not cell[1].any())
    observed['not cell[2].any()'] = bool(not cell[2].any())
    return observed


_EXPECT_load_masks_missing_and_non3d_files_leave_zeros = {
    'cell[0].any()': True,
    'not cell[1].any()': True,
    'not cell[2].any()': True,
}


def _case_load_masks_planes_last_layout(tmp_path):
    observed = {}
    from spacr.timelapse import _load_masks_from_merged
    arr = np.moveaxis(_planes_first(n_channels=2, mask_planes=2), 0, -1)
    _write_merged(tmp_path, "f0.npy", arr)

    cell, nuc, _ = _load_masks_from_merged(
        str(tmp_path), ["f0.npy"], n_channels=2, height=H, width=W,
        nucleus_chan=0, pathogen_chan=None,
    )

    observed['set(np.unique(cell))'] = set(np.unique(cell))
    observed['set(np.unique(nuc))'] = set(np.unique(nuc))
    return observed


_EXPECT_load_masks_planes_last_layout = {
    'set(np.unique(cell))': {0, 1},
    'set(np.unique(nuc))': {0, 2},
}


def _case_load_masks_leaves_a_frame_empty_when_its_file_cannot_be_used(
        tmp_path, monkeypatch, capsys, fault):
    """A frame file of the wrong size, or one that cannot be reoriented, is skipped for every mask kind."""
    observed = {}
    from spacr import timelapse as TL
    if fault == "wrong_size":
        arr = _planes_first(2, 1, height=6, width=6)
    else:
        arr = _planes_first(2, 1)

        def _boom(arr, n_channels, **kwargs):
            raise ValueError("injected reorientation failure")

        monkeypatch.setattr(TL, "_reorient_merged_array", _boom)
    _write_merged(tmp_path, "f0.npy", arr)

    cell, nuc, path = TL._load_masks_from_merged(
        str(tmp_path), ["f0.npy"], n_channels=2, height=H, width=W
    )

    observed['not cell.any() and (not nuc.any()) and (not path.any())'] = bool(not cell.any() and (not nuc.any()) and (not path.any()))
    if fault == "wrong_size":
        assert "_load_masks_from_merged] Skipping f0.npy" in capsys.readouterr().out
    return observed


_EXPECT_load_masks_leaves_a_frame_empty_when_its_file_cannot_be_used = {
    'not cell.any() and (not nuc.any()) and (not path.any())': True,
}


def _case_load_masks_alternate_filename_candidate(tmp_path):
    observed = {}
    from spacr.timelapse import _load_masks_from_merged
    _write_merged(tmp_path, "f0.tif", _planes_first(2, 1))
    observed["not (tmp_path / 'merged' / 'f0.npy').exists()"] = bool(not (tmp_path / 'merged' / 'f0.npy').exists())

    cell, _, _ = _load_masks_from_merged(
        str(tmp_path), ["f0.tif"], n_channels=2, height=H, width=W
    )
    observed['set(np.unique(cell))'] = set(np.unique(cell))
    return observed


_EXPECT_load_masks_alternate_filename_candidate = {
    "not (tmp_path / 'merged' / 'f0.npy').exists()": True,
    'set(np.unique(cell))': {0, 1},
}


# ===========================================================================
# _compute_regionprops_stack
# ===========================================================================

def _case_regionprops_stack_with_intensity_columns_and_values():
    observed = {}
    from spacr.timelapse import _compute_regionprops_stack
    masks, inten = _mask_and_intensity(T=2, n_labels=2, C=2)

    df = _compute_regionprops_stack(
        masks, inten, channel_index=0, object_prefix="cell"
    )

    observed['len(df)'] = len(df)
    observed["'cell_label' in df.columns and 'frame' in df.columns"] = bool('cell_label' in df.columns and 'frame' in df.columns)
    for col in ("cell_area", "cell_perimeter", "cell_solidity",
                "cell_centroid-0", "cell_centroid-1",
                "cell_mean_intensity", "cell_max_intensity",
                "cell_min_intensity", "cell_bbox_area",
                "cell_equivalent_diameter", "cell_perimeter_crofton"):
        assert col in df.columns, col

    row = df[(df["frame"] == 1) & (df["cell_label"] == 1)].iloc[0]
    observed["row['cell_area']"] = row['cell_area']
    observed["row['cell_mean_intensity']"] = row['cell_mean_intensity']
    observed["row['cell_max_intensity']"] = row['cell_max_intensity']
    return observed


_EXPECT_regionprops_stack_with_intensity_columns_and_values = {
    'len(df)': 4,
    "'cell_label' in df.columns and 'frame' in df.columns": True,
    "row['cell_area']": 9,
    "row['cell_mean_intensity']": pytest.approx(20.0),
    "row['cell_max_intensity']": pytest.approx(20.0),
}


def _case_regionprops_stack_label_as_track_id_keeps_label_unprefixed():
    observed = {}
    from spacr.timelapse import _compute_regionprops_stack
    masks, inten = _mask_and_intensity(T=1, n_labels=2, C=1)

    df = _compute_regionprops_stack(
        masks, inten, channel_index=0, object_prefix="cell",
        label_as_track_id=True,
    )

    observed["'track_id' in df.columns"] = bool('track_id' in df.columns)
    observed["'cell_label' not in df.columns"] = bool('cell_label' not in df.columns)
    observed["'cell_track_id' not in df.columns"] = bool('cell_track_id' not in df.columns)
    observed["sorted(df['track_id'].tolist())"] = sorted(df['track_id'].tolist())
    return observed


_EXPECT_regionprops_stack_label_as_track_id_keeps_label_unprefixed = {
    "'track_id' in df.columns": True,
    "'cell_label' not in df.columns": True,
    "'cell_track_id' not in df.columns": True,
    "sorted(df['track_id'].tolist())": [1, 2],
}


def _case_regionprops_stack_without_intensity_props(stack_kind, channel):
    observed = {}
    from spacr.timelapse import _compute_regionprops_stack
    masks, inten = _mask_and_intensity(T=2, n_labels=1, C=2)
    stack = None if stack_kind == "none" else inten

    df = _compute_regionprops_stack(
        masks, stack, channel_index=channel, object_prefix="nucleus"
    )

    observed['len(df)'] = len(df)
    observed["'nucleus_area' in df.columns"] = bool('nucleus_area' in df.columns)
    for col in df.columns:
        assert "intensity" not in col
    return observed


_EXPECT_regionprops_stack_without_intensity_props = {
    'len(df)': 2,
    "'nucleus_area' in df.columns": True,
}


def _case_regionprops_stack_skips_blank_frames():
    observed = {}
    from spacr.timelapse import _compute_regionprops_stack
    masks, inten = _mask_and_intensity(T=3, n_labels=1, C=1)
    masks[0] = 0

    df = _compute_regionprops_stack(
        masks, inten, channel_index=0, object_prefix="pathogen"
    )

    observed["sorted(df['frame'].unique().tolist())"] = sorted(df['frame'].unique().tolist())
    observed['len(df)'] = len(df)
    return observed


_EXPECT_regionprops_stack_skips_blank_frames = {
    "sorted(df['frame'].unique().tolist())": [1, 2],
    'len(df)': 2,
}


def _case_regionprops_stack_all_blank_returns_empty_frame(capsys):
    observed = {}
    from spacr.timelapse import _compute_regionprops_stack
    masks = np.zeros((3, H, W), dtype=np.int32)

    df = _compute_regionprops_stack(
        masks, None, channel_index=None, object_prefix="cytoplasm"
    )

    observed['df.empty'] = bool(df.empty)
    observed['list(df.columns)'] = list(df.columns)
    observed["'cytoplasm: no objects found in any of 3 frames' in capsys.readouterr().out"] = bool('cytoplasm: no objects found in any of 3 frames' in capsys.readouterr().out)
    return observed


_EXPECT_regionprops_stack_all_blank_returns_empty_frame = {
    'df.empty': True,
    'list(df.columns)': ['frame', 'cytoplasm_label'],
    "'cytoplasm: no objects found in any of 3 frames' in capsys.readouterr().out": True,
}


# ===========================================================================
# behaviour tests: one per public-function group, one parametrised row per case
# ===========================================================================

def _run_case(case, tmp_path, monkeypatch, capsys):
    """Call a case helper with only the fixtures its signature asks for."""
    wanted = inspect.signature(case).parameters
    available = {"tmp_path": tmp_path, "monkeypatch": monkeypatch, "capsys": capsys}
    return case(**{k: v for k, v in available.items() if k in wanted})


_P = functools.partial

MEASUREMENT_CASES = [
    pytest.param(_case_cell_mean_intensity_per_channel_values_and_shape, _EXPECT_cell_mean_intensity_per_channel_values_and_shape, id="cell-mean-values-and-shape"),
    pytest.param(_case_cell_mean_intensity_skips_empty_frames_only, _EXPECT_cell_mean_intensity_skips_empty_frames_only, id="cell-mean-skips-empty-frames"),
    pytest.param(_P(_case_cell_mean_intensity_returns_an_empty_frame_when_nothing_can_be_measured, scenario='no_stack', channel=0, column='cell_mean_intensity_ch0', message='intensity_stack is None'), _EXPECT_cell_mean_intensity_returns_an_empty_frame_when_nothing_can_be_measured, id="cell-mean-empty-no-stack"),
    pytest.param(_P(_case_cell_mean_intensity_returns_an_empty_frame_when_nothing_can_be_measured, scenario='bad_channel', channel=None, column='cell_mean_intensity_chNone', message='invalid channel index'), _EXPECT_cell_mean_intensity_returns_an_empty_frame_when_nothing_can_be_measured, id="cell-mean-empty-bad-channel-None"),
    pytest.param(_P(_case_cell_mean_intensity_returns_an_empty_frame_when_nothing_can_be_measured, scenario='bad_channel', channel=-1, column='cell_mean_intensity_ch-1', message='invalid channel index'), _EXPECT_cell_mean_intensity_returns_an_empty_frame_when_nothing_can_be_measured, id="cell-mean-empty-bad-channel-neg1"),
    pytest.param(_P(_case_cell_mean_intensity_returns_an_empty_frame_when_nothing_can_be_measured, scenario='bad_channel', channel=2, column='cell_mean_intensity_ch2', message='invalid channel index'), _EXPECT_cell_mean_intensity_returns_an_empty_frame_when_nothing_can_be_measured, id="cell-mean-empty-bad-channel-2"),
    pytest.param(_P(_case_cell_mean_intensity_returns_an_empty_frame_when_nothing_can_be_measured, scenario='bad_channel', channel=99, column='cell_mean_intensity_ch99', message='invalid channel index'), _EXPECT_cell_mean_intensity_returns_an_empty_frame_when_nothing_can_be_measured, id="cell-mean-empty-bad-channel-99"),
    pytest.param(_P(_case_cell_mean_intensity_returns_an_empty_frame_when_nothing_can_be_measured, scenario='blank_masks', channel=0, column='cell_mean_intensity_ch0', message='no objects found in any of 4 frames'), _EXPECT_cell_mean_intensity_returns_an_empty_frame_when_nothing_can_be_measured, id="cell-mean-empty-blank-masks"),
    pytest.param(_case_regionprops_stack_with_intensity_columns_and_values, _EXPECT_regionprops_stack_with_intensity_columns_and_values, id="regionprops-intensity-columns-and-values"),
    pytest.param(_case_regionprops_stack_label_as_track_id_keeps_label_unprefixed, _EXPECT_regionprops_stack_label_as_track_id_keeps_label_unprefixed, id="regionprops-label-as-track-id"),
    pytest.param(_P(_case_regionprops_stack_without_intensity_props, stack_kind='none', channel=0), _EXPECT_regionprops_stack_without_intensity_props, id="regionprops-geometry-only-no-stack"),
    pytest.param(_P(_case_regionprops_stack_without_intensity_props, stack_kind='real', channel=None), _EXPECT_regionprops_stack_without_intensity_props, id="regionprops-geometry-only-channel-none"),
    pytest.param(_P(_case_regionprops_stack_without_intensity_props, stack_kind='real', channel=7), _EXPECT_regionprops_stack_without_intensity_props, id="regionprops-geometry-only-channel-past-end"),
    pytest.param(_P(_case_regionprops_stack_without_intensity_props, stack_kind='real', channel=-1), _EXPECT_regionprops_stack_without_intensity_props, id="regionprops-geometry-only-channel-negative"),
    pytest.param(_case_regionprops_stack_skips_blank_frames, _EXPECT_regionprops_stack_skips_blank_frames, id="regionprops-skips-blank-frames"),
    pytest.param(_case_regionprops_stack_all_blank_returns_empty_frame, _EXPECT_regionprops_stack_all_blank_returns_empty_frame, id="regionprops-all-blank-empty-frame"),
]


@pytest.mark.parametrize("case, expected", MEASUREMENT_CASES)
def test_per_frame_measurement_tables_hold_exact_values_columns_and_empty_frames(
        case, expected, tmp_path, monkeypatch, capsys):
    """Cell mean intensity and regionprops tables: values, columns, skipped blank frames, empty frame plus printed reason."""
    observed = _run_case(case, tmp_path, monkeypatch, capsys)
    assert observed == expected

SUMMARY_CASES = [
    pytest.param(_case_summarise_child_features_default_agg_is_mean_for_other_features, _EXPECT_summarise_child_features_default_agg_is_mean_for_other_features, id="default-aggregation-is-mean"),
    pytest.param(_case_summarise_child_features_empty_merge_result_returns_stub_frame, _EXPECT_summarise_child_features_empty_merge_result_returns_stub_frame, id="empty-merge-returns-stub-frame"),
]


@pytest.mark.parametrize("case, expected", SUMMARY_CASES)
def test_summarise_child_features_aggregates_per_parent_and_returns_a_stub_when_the_join_is_empty(
        case, expected, tmp_path, monkeypatch, capsys):
    observed = _run_case(case, tmp_path, monkeypatch, capsys)
    assert observed == expected

INTENSITY_LOAD_CASES = [
    pytest.param(_case_load_intensity_stack_planes_first, _EXPECT_load_intensity_stack_planes_first, id="planes-first"),
    pytest.param(_case_load_intensity_stack_planes_last_layout, _EXPECT_load_intensity_stack_planes_last_layout, id="planes-last"),
    pytest.param(_case_load_intensity_stack_custom_dtype_and_fewer_planes_than_channels, _EXPECT_load_intensity_stack_custom_dtype_and_fewer_planes_than_channels, id="custom-dtype-fewer-planes"),
    pytest.param(_case_load_intensity_stack_no_merged_dir_returns_zero_channels, _EXPECT_load_intensity_stack_no_merged_dir_returns_zero_channels, id="no-merged-dir"),
    pytest.param(_case_load_intensity_stack_missing_and_non3d_files_leave_zeros, _EXPECT_load_intensity_stack_missing_and_non3d_files_leave_zeros, id="missing-and-2d-files"),
    pytest.param(_case_load_intensity_stack_alternate_filename_candidate, _EXPECT_load_intensity_stack_alternate_filename_candidate, id="alternate-filename"),
    pytest.param(_P(_case_load_intensity_stack_non_positive_channels, n_channels=None), _EXPECT_load_intensity_stack_non_positive_channels, id="non-positive-channels-none"),
    pytest.param(_P(_case_load_intensity_stack_non_positive_channels, n_channels=0), _EXPECT_load_intensity_stack_non_positive_channels, id="non-positive-channels-zero"),
    pytest.param(_P(_case_load_intensity_stack_non_positive_channels, n_channels=-1), _EXPECT_load_intensity_stack_non_positive_channels, id="non-positive-channels-negative"),
    pytest.param(_P(_case_load_intensity_stack_leaves_a_frame_zero_when_its_file_cannot_be_used, fault='wrong_size'), _EXPECT_load_intensity_stack_leaves_a_frame_zero_when_its_file_cannot_be_used, id="unusable-file-wrong-size"),
    pytest.param(_P(_case_load_intensity_stack_leaves_a_frame_zero_when_its_file_cannot_be_used, fault='zero_planes'), _EXPECT_load_intensity_stack_leaves_a_frame_zero_when_its_file_cannot_be_used, id="unusable-file-zero-planes"),
    pytest.param(_P(_case_load_intensity_stack_leaves_a_frame_zero_when_its_file_cannot_be_used, fault='reorient_fails'), _EXPECT_load_intensity_stack_leaves_a_frame_zero_when_its_file_cannot_be_used, id="unusable-file-reorient-fails"),
]


@pytest.mark.parametrize("case, expected", INTENSITY_LOAD_CASES)
def test_load_intensity_stack_fills_channels_from_each_layout_and_leaves_unusable_frames_zero(
        case, expected, tmp_path, monkeypatch, capsys):
    observed = _run_case(case, tmp_path, monkeypatch, capsys)
    assert observed == expected

MASK_LOAD_CASES = [
    pytest.param(_case_load_masks_cell_only_plane, _EXPECT_load_masks_cell_only_plane, id="cell-only-plane"),
    pytest.param(_case_load_masks_second_plane_is_nucleus_when_only_nucleus_requested, _EXPECT_load_masks_second_plane_is_nucleus_when_only_nucleus_requested, id="second-plane-nucleus"),
    pytest.param(_case_load_masks_second_plane_is_pathogen_when_only_pathogen_requested, _EXPECT_load_masks_second_plane_is_pathogen_when_only_pathogen_requested, id="second-plane-pathogen"),
    pytest.param(_case_load_masks_three_planes_nucleus_then_pathogen, _EXPECT_load_masks_three_planes_nucleus_then_pathogen, id="three-planes"),
    pytest.param(_case_load_masks_second_plane_ignored_when_no_child_channels, _EXPECT_load_masks_second_plane_ignored_when_no_child_channels, id="no-child-channels"),
    pytest.param(_case_load_masks_no_mask_planes_returns_zeros, _EXPECT_load_masks_no_mask_planes_returns_zeros, id="no-mask-planes"),
    pytest.param(_case_load_masks_no_merged_dir_returns_zero_stacks, _EXPECT_load_masks_no_merged_dir_returns_zero_stacks, id="no-merged-dir"),
    pytest.param(_case_load_masks_missing_and_non3d_files_leave_zeros, _EXPECT_load_masks_missing_and_non3d_files_leave_zeros, id="missing-and-2d-files"),
    pytest.param(_case_load_masks_planes_last_layout, _EXPECT_load_masks_planes_last_layout, id="planes-last"),
    pytest.param(_case_load_masks_alternate_filename_candidate, _EXPECT_load_masks_alternate_filename_candidate, id="alternate-filename"),
    pytest.param(_P(_case_load_masks_leaves_a_frame_empty_when_its_file_cannot_be_used, fault='wrong_size'), _EXPECT_load_masks_leaves_a_frame_empty_when_its_file_cannot_be_used, id="unusable-file-wrong-size"),
    pytest.param(_P(_case_load_masks_leaves_a_frame_empty_when_its_file_cannot_be_used, fault='reorient_fails'), _EXPECT_load_masks_leaves_a_frame_empty_when_its_file_cannot_be_used, id="unusable-file-reorient-fails"),
]


@pytest.mark.parametrize("case, expected", MASK_LOAD_CASES)
def test_load_masks_maps_planes_to_cell_nucleus_pathogen_and_leaves_unusable_frames_empty(
        case, expected, tmp_path, monkeypatch, capsys):
    observed = _run_case(case, tmp_path, monkeypatch, capsys)
    assert observed == expected

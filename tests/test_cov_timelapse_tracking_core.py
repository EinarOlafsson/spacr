"""Coverage for the trackpy-based tracking core of ``spacr.timelapse``.

Region under test: ``_find_optimal_search_range``,
``_facilitate_trackin_with_adaptive_removal`` and ``_trackpy_track_cells``
(spacr/timelapse.py lines ~326-514).

These three functions are the only ones in the module that talk to trackpy,
so every branch here is exercised either against real trackpy on tiny
synthetic label stacks, or by injecting a fake ``tp.link`` / ``tp.link_df``
that fails on demand.

Two real defects, both fixed, are regression-tested here:

* ``_facilitate_trackin_with_adaptive_removal`` used to call
  ``tp.link_df(..., predict=True)``; trackpy has no ``predict`` keyword
  (it is ``predictor=<obj>``), so every linking attempt raised TypeError and
  the function always ended in ``RuntimeError``.
* ``_trackpy_track_cells`` did ``tracks_df['particle'] += 1`` unconditionally,
  but with ``track_by_iou=True`` the tracks frame carries ``track_id``, not
  ``particle`` -> KeyError.
"""
from __future__ import annotations

import functools
import inspect
import os

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Synthetic label-stack builders
# ---------------------------------------------------------------------------

def _two_movers(n_frames=4, size=40, side=8):
    """(T, H, W) int32 stack with two 8x8 squares drifting 1 px/frame.

    Object A moves down the left edge, object B moves right along the bottom.
    The *label ids are swapped on odd frames* so that a correct tracker has to
    produce track ids that do NOT simply echo the original labels.
    """
    m = np.zeros((n_frames, size, size), dtype=np.int32)
    for t in range(n_frames):
        a_lbl, b_lbl = (1, 2) if t % 2 == 0 else (2, 1)
        m[t, 4 + t: 4 + t + side, 4: 4 + side] = a_lbl
        m[t, 24: 24 + side, 24 + t: 24 + t + side] = b_lbl
    return m


def _a_center(t, side=8):
    return (4 + t + side // 2, 4 + side // 2)


def _b_center(t, side=8):
    return (24 + side // 2, 24 + t + side // 2)


def _stack_with_transient(n_frames=4, size=48, side=8):
    """Two persistent squares plus a third that only exists in frames 0 and 1."""
    m = np.zeros((n_frames, size, size), dtype=np.int32)
    for t in range(n_frames):
        m[t, 2 + t: 2 + t + side, 2: 2 + side] = 1
        m[t, 30: 30 + side, 30 + t: 30 + t + side] = 2
        if t < 2:
            m[t, 2: 2 + side, 30 + t: 30 + t + side] = 3
    return m


def _link_df_shim(monkeypatch):
    """Pass-through wrapper around ``tp.link_df`` (isolates these tests from it)."""
    from spacr import timelapse as TL

    real = TL.tp.link_df

    def shim(f, search_range, **kwargs):
        return real(f, search_range=search_range, **kwargs)

    monkeypatch.setattr(TL.tp._load(), "link_df", shim)
    return shim


# ===========================================================================
# _find_optimal_search_range
# ===========================================================================

def _case_find_optimal_search_range_returns_initial_on_first_success(capsys):
    """A search range trackpy accepts is returned untouched (no decrement)."""
    observed = {}
    from spacr.timelapse import _find_optimal_search_range, _prepare_for_tracking

    features = _prepare_for_tracking(_two_movers())
    out = _find_optimal_search_range(features, initial_search_range=20,
                                     increment=5, max_attempts=3, memory=1)
    observed['out'] = out
    observed["'Success with search_range=20' in capsys.readouterr().out"] = bool('Success with search_range=20' in capsys.readouterr().out)
    return observed


_EXPECT_find_optimal_search_range_returns_initial_on_first_success = {
    'out': 20,
    "'Success with search_range=20' in capsys.readouterr().out": True,
}


def _case_find_optimal_search_range_forwards_memory_and_search_range(monkeypatch):
    """The current candidate range and the memory argument reach tp.link."""
    observed = {}
    from spacr import timelapse as TL

    seen = {}

    def fake_link(features, search_range, memory):
        seen["search_range"] = search_range
        seen["memory"] = memory
        seen["n_rows"] = len(features)
        return pd.DataFrame({"particle": [0]})

    monkeypatch.setattr(TL.tp._load(), "link", fake_link)
    features = TL._prepare_for_tracking(_two_movers())
    out = TL._find_optimal_search_range(features, initial_search_range=123,
                                        increment=7, max_attempts=4, memory=9)
    observed['out'] = out
    observed["seen == {'search_range': 123, 'memory': 9, 'n_rows': len(features)}"] = bool(seen == {'search_range': 123, 'memory': 9, 'n_rows': len(features)})
    return observed


_EXPECT_find_optimal_search_range_forwards_memory_and_search_range = {
    'out': 123,
    "seen == {'search_range': 123, 'memory': 9, 'n_rows': len(features)}": True,
}


def _case_find_optimal_search_range_decrements_until_link_succeeds(monkeypatch, capsys):
    """Each failure shrinks the range by ``increment``; the winner is returned."""
    observed = {}
    from spacr import timelapse as TL

    calls = []

    def flaky_link(features, search_range, memory):
        calls.append(search_range)
        if len(calls) <= 2:
            raise RuntimeError("SubnetOversizeException")
        return pd.DataFrame({"particle": [0, 1]})

    monkeypatch.setattr(TL.tp._load(), "link", flaky_link)
    features = TL._prepare_for_tracking(_two_movers())
    out = TL._find_optimal_search_range(features, initial_search_range=100,
                                        increment=10, max_attempts=6, memory=3)
    # 100 fails, 90 fails, 80 succeeds.
    observed['calls'] = calls
    observed['out'] = out
    captured = capsys.readouterr().out
    observed["'Success with search_range=80' in captured"] = bool('Success with search_range=80' in captured)
    observed["'too high' not in captured"] = bool('too high' not in captured)
    return observed


_EXPECT_find_optimal_search_range_decrements_until_link_succeeds = {
    'calls': [100, 90, 80],
    'out': 80,
    "'Success with search_range=80' in captured": True,
    "'too high' not in captured": True,
}


def _case_find_optimal_search_range_exhausts_attempts_and_warns(monkeypatch, capsys):
    """When every attempt fails the floor value is returned with a warning."""
    observed = {}
    from spacr import timelapse as TL

    calls = []

    def always_fail(features, search_range, memory):
        calls.append(search_range)
        raise RuntimeError("SubnetOversizeException")

    monkeypatch.setattr(TL.tp._load(), "link", always_fail)
    features = TL._prepare_for_tracking(_two_movers())
    out = TL._find_optimal_search_range(features, initial_search_range=50,
                                        increment=10, max_attempts=3, memory=3)
    observed['calls'] = calls
    # 50 - 3*10 == 20 == the advertised floor.
    observed['out'] = out
    captured = capsys.readouterr().out
    observed["'timelapse_displacement=20 is too high' in captured"] = bool('timelapse_displacement=20 is too high' in captured)
    return observed


_EXPECT_find_optimal_search_range_exhausts_attempts_and_warns = {
    'calls': [50, 40, 30],
    'out': 20,
    "'timelapse_displacement=20 is too high' in captured": True,
}


def _case_find_optimal_search_range_zero_attempts_is_a_noop(monkeypatch, capsys):
    """max_attempts=0 never calls trackpy and returns the initial range."""
    observed = {}
    from spacr import timelapse as TL

    def boom(*a, **k):  # pragma: no branch - must never run
        raise AssertionError("tp.link must not be called with max_attempts=0")

    monkeypatch.setattr(TL.tp._load(), "link", boom)
    features = TL._prepare_for_tracking(_two_movers())
    out = TL._find_optimal_search_range(features, initial_search_range=37,
                                        increment=10, max_attempts=0, memory=3)
    observed['out'] = out
    observed["'timelapse_displacement=37 is too high' in capsys.readouterr().out"] = bool('timelapse_displacement=37 is too high' in capsys.readouterr().out)
    return observed


_EXPECT_find_optimal_search_range_zero_attempts_is_a_noop = {
    'out': 37,
    "'timelapse_displacement=37 is too high' in capsys.readouterr().out": True,
}


# ===========================================================================
# _facilitate_trackin_with_adaptive_removal
# ===========================================================================

def _case_facilitate_drops_small_frame0_objects_only(monkeypatch):
    """min_mass prunes frame 0 in place; later frames keep every object."""
    observed = {}
    from spacr import timelapse as TL

    masks = _two_movers()
    # Add a 3x3 (mass 9) speck present in every frame -> only frame 0 loses it.
    masks[:, 34:37, 2:5] = 7
    original = masks.copy()

    out_masks, features, tracks = TL._facilitate_trackin_with_adaptive_removal(
        masks, search_range=10, max_attempts=2, memory=1, min_mass=50,
        track_by_iou=True)

    observed['out_masks is masks'] = bool(out_masks is masks)
    observed['7 not in np.unique(out_masks[0])'] = bool(7 not in np.unique(out_masks[0]))
    observed['7 in np.unique(out_masks[1])'] = bool(7 in np.unique(out_masks[1]))
    # The two 8x8 squares survived frame 0 unchanged.
    observed['np.array_equal(out_masks[0] == 1, original[0] == 1)'] = bool(np.array_equal(out_masks[0] == 1, original[0] == 1))
    observed['np.array_equal(out_masks[0] == 2, original[0] == 2)'] = bool(np.array_equal(out_masks[0] == 2, original[0] == 2))
    # Features were recomputed on the filtered stack.
    f0 = features[features["frame"] == 0]
    observed["set(f0['original_label'])"] = set(f0['original_label'])
    observed["set(f0['mass'])"] = set(f0['mass'])
    observed['list(tracks.columns)'] = list(tracks.columns)
    return observed


_EXPECT_facilitate_drops_small_frame0_objects_only = {
    'out_masks is masks': True,
    '7 not in np.unique(out_masks[0])': True,
    '7 in np.unique(out_masks[1])': True,
    'np.array_equal(out_masks[0] == 1, original[0] == 1)': True,
    'np.array_equal(out_masks[0] == 2, original[0] == 2)': True,
    "set(f0['original_label'])": {1, 2},
    "set(f0['mass'])": {64.0},
    'list(tracks.columns)': ['frame', 'original_label', 'track_id'],
}


def _case_facilitate_track_by_iou_gives_one_track_per_object():
    """IoU linking survives the label swap and yields exactly two tracks."""
    observed = {}
    from spacr.timelapse import _facilitate_trackin_with_adaptive_removal

    masks = _two_movers(n_frames=4)
    _, _, tracks = _facilitate_trackin_with_adaptive_removal(
        masks, search_range=10, max_attempts=2, memory=1, min_mass=10,
        track_by_iou=True)

    observed['len(tracks)'] = len(tracks)
    observed["tracks['track_id'].nunique()"] = tracks['track_id'].nunique()
    # Object A is label 1 on even frames and label 2 on odd frames; both rows
    # must carry the same track id.
    def tid(frame, label):
        row = tracks[(tracks["frame"] == frame) & (tracks["original_label"] == label)]
        return int(row["track_id"].iloc[0])
    observed['tid(0, 1) == tid(1, 2) == tid(2, 1) == tid(3, 2)'] = bool(tid(0, 1) == tid(1, 2) == tid(2, 1) == tid(3, 2))
    return observed


_EXPECT_facilitate_track_by_iou_gives_one_track_per_object = {
    'len(tracks)': 8,
    "tracks['track_id'].nunique()": 2,
    'tid(0, 1) == tid(1, 2) == tid(2, 1) == tid(3, 2)': True,
}


def _case_facilitate_auto_search_range_from_area_quantile(monkeypatch):
    """search_range=None -> 2*sqrt(99th-percentile frame-0 area)."""
    observed = {}
    from spacr import timelapse as TL

    masks = _two_movers()
    masks[:, 34:37, 2:5] = 7                       # tiny object, mass 9

    expected_f0 = TL._prepare_for_tracking(masks)
    expected_f0 = expected_f0[expected_f0["frame"] == 0]
    expected = max(1, int(2 * np.sqrt(expected_f0["mass"].quantile(0.99))))
    observed['expected > 1'] = bool(expected > 1)

    seen = {}

    def fake_link_df(features, search_range, memory):
        seen["search_range"] = search_range
        seen["memory"] = memory
        out = features.copy()
        out["particle"] = out["original_label"] - 1
        return out

    monkeypatch.setattr(TL.tp._load(), "link_df", fake_link_df)
    _, _, tracks = TL._facilitate_trackin_with_adaptive_removal(
        masks, search_range=None, max_attempts=3, memory=4, min_mass=50,
        track_by_iou=False)

    observed["seen['search_range'] == expected"] = bool(seen['search_range'] == expected)
    observed["seen['memory']"] = seen['memory']
    observed["'particle' in tracks.columns"] = bool('particle' in tracks.columns)
    return observed


_EXPECT_facilitate_auto_search_range_from_area_quantile = {
    'expected > 1': True,
    "seen['search_range'] == expected": True,
    "seen['memory']": 4,
    "'particle' in tracks.columns": True,
}


def _case_facilitate_retries_with_shrunken_search_range(monkeypatch, capsys):
    """A failed attempt shrinks search_range to 80% and retries."""
    observed = {}
    from spacr import timelapse as TL

    calls = []

    def flaky_link_df(features, search_range, memory):
        calls.append(search_range)
        if len(calls) == 1:
            raise ValueError("SubnetOversizeException")
        out = features.copy()
        out["particle"] = out["original_label"] - 1
        return out

    monkeypatch.setattr(TL.tp._load(), "link_df", flaky_link_df)
    masks, features, tracks = TL._facilitate_trackin_with_adaptive_removal(
        _two_movers(), search_range=100, max_attempts=4, memory=1, min_mass=10,
        track_by_iou=False)

    observed['calls'] = calls
    observed['len(tracks) == len(features)'] = bool(len(tracks) == len(features))
    captured = capsys.readouterr().out
    observed["'reducing search_range to 80' in captured"] = bool('reducing search_range to 80' in captured)
    observed["'Linked on attempt 2 with search_range=80' in captured"] = bool('Linked on attempt 2 with search_range=80' in captured)
    return observed


_EXPECT_facilitate_retries_with_shrunken_search_range = {
    'calls': [100, 80],
    'len(tracks) == len(features)': True,
    "'reducing search_range to 80' in captured": True,
    "'Linked on attempt 2 with search_range=80' in captured": True,
}


def _case_facilitate_search_range_never_shrinks_below_one(monkeypatch, capsys):
    """The max(1, ...) floor keeps search_range positive across many retries."""
    observed = {}
    from spacr import timelapse as TL

    calls = []

    def always_fail(features, search_range, memory):
        calls.append(search_range)
        raise ValueError("nope")

    monkeypatch.setattr(TL.tp._load(), "link_df", always_fail)
    with pytest.raises(RuntimeError) as exc:
        TL._facilitate_trackin_with_adaptive_removal(
            _two_movers(), search_range=3, max_attempts=6, memory=1,
            min_mass=10, track_by_iou=False)

    observed['calls'] = calls
    observed['min(calls)'] = min(calls)
    observed["'Failed to track after 6 attempts' in str(exc.value)"] = bool('Failed to track after 6 attempts' in str(exc.value))
    observed["'last search_range=1' in str(exc.value)"] = bool('last search_range=1' in str(exc.value))
    return observed


_EXPECT_facilitate_search_range_never_shrinks_below_one = {
    'calls': [3, 2, 1, 1, 1, 1],
    'min(calls)': 1,
    "'Failed to track after 6 attempts' in str(exc.value)": True,
    "'last search_range=1' in str(exc.value)": True,
}


def _case_facilitate_raises_runtime_error_after_max_attempts(monkeypatch):
    """Exhausting max_attempts raises RuntimeError, not the inner exception."""
    from spacr import timelapse as TL

    def always_fail(features, search_range, memory):
        raise ZeroDivisionError("inner failure")

    monkeypatch.setattr(TL.tp._load(), "link_df", always_fail)
    with pytest.raises(RuntimeError, match=r"Failed to track after 2 attempts") as excinfo:
        TL._facilitate_trackin_with_adaptive_removal(
            _two_movers(), search_range=10, max_attempts=2, memory=1,
            min_mass=10, track_by_iou=False)
    return {"error": str(excinfo.value)}


_EXPECT_facilitate_raises_runtime_error_after_max_attempts = {
    "error": "Failed to track after 2 attempts; last search_range=6",
}


def _case_facilitate_track_by_iou_ignores_link_df_entirely(monkeypatch):
    """With track_by_iou=True trackpy is never consulted."""
    observed = {}
    from spacr import timelapse as TL

    def boom(*a, **k):
        raise AssertionError("tp.link_df must not be called when track_by_iou=True")

    monkeypatch.setattr(TL.tp._load(), "link_df", boom)
    _, _, tracks = TL._facilitate_trackin_with_adaptive_removal(
        _two_movers(), search_range=10, max_attempts=2, memory=1, min_mass=10,
        track_by_iou=True)
    observed["tracks['track_id'].nunique()"] = tracks['track_id'].nunique()
    return observed


_EXPECT_facilitate_track_by_iou_ignores_link_df_entirely = {
    "tracks['track_id'].nunique()": 2,
}


def _case_facilitate_with_real_trackpy_links_two_particles():
    """Real trackpy links both squares through the module-level ``tp.link_df``."""
    observed = {}
    from spacr import timelapse as TL

    real = TL.tp.link_df

    def shim(f, search_range, **kwargs):
        return real(f, search_range=search_range, **kwargs)

    old = TL.tp.link_df
    TL.tp.link_df = shim
    try:
        masks, features, tracks = TL._facilitate_trackin_with_adaptive_removal(
            _two_movers(), search_range=10, max_attempts=3, memory=1,
            min_mass=10, track_by_iou=False)
    finally:
        TL.tp.link_df = old

    observed["'particle' in tracks.columns"] = bool('particle' in tracks.columns)
    observed["tracks['particle'].nunique()"] = tracks['particle'].nunique()
    observed['len(tracks)'] = len(tracks)
    return observed


_EXPECT_facilitate_with_real_trackpy_links_two_particles = {
    "'particle' in tracks.columns": True,
    "tracks['particle'].nunique()": 2,
    'len(tracks)': 8,
}


def _case_facilitate_real_trackpy_predict_kwarg_is_rejected():
    """The unpatched trackpy path links instead of raising RuntimeError."""
    observed = {}
    from spacr.timelapse import _facilitate_trackin_with_adaptive_removal

    _, _, tracks = _facilitate_trackin_with_adaptive_removal(
        _two_movers(), search_range=10, max_attempts=2, memory=1,
        min_mass=10, track_by_iou=False)
    observed["'particle' in tracks.columns"] = bool('particle' in tracks.columns)
    observed["tracks['particle'].nunique()"] = tracks['particle'].nunique()
    return observed


_EXPECT_facilitate_real_trackpy_predict_kwarg_is_rejected = {
    "'particle' in tracks.columns": True,
    "tracks['particle'].nunique()": 2,
}


# ===========================================================================
# _trackpy_track_cells
# ===========================================================================

def _case_trackpy_track_cells_writes_csv_and_relabels_masks(monkeypatch, tmp_path, capsys):
    """End-to-end: tracks CSV on disk, masks relabelled to 1-based track ids."""
    observed = {}
    from spacr import timelapse as TL

    _link_df_shim(monkeypatch)
    masks = _two_movers(n_frames=4)
    src = str(tmp_path / "masks" / "batch_1.npz")
    os.makedirs(os.path.dirname(src), exist_ok=True)

    stack = TL._trackpy_track_cells(
        src=src, name="plate1_A01_1", batch_filenames=[f"f{i}.tif" for i in range(4)],
        object_type="cell", masks=masks, timelapse_displacement=10,
        timelapse_memory=1, timelapse_remove_transient=True,
        plot=False, save=False, mode="trackpy", track_by_iou=False)

    # 1) return value is a per-frame list of 2-D label images
    observed['isinstance(stack, list) and len(stack) == 4'] = bool(isinstance(stack, list) and len(stack) == 4)
    observed['all((f.shape == (40, 40) for f in stack))'] = bool(all((f.shape == (40, 40) for f in stack)))

    # 2) CSV written next to src
    csv_path = tmp_path / "masks" / "tracks" / "trackpy_tracks_cell_plate1_A01_1.csv"
    observed['csv_path.is_file()'] = bool(csv_path.is_file())
    df = pd.read_csv(csv_path)
    observed["'track_id' in df.columns and 'particle' not in df.columns"] = bool('track_id' in df.columns and 'particle' not in df.columns)
    observed['len(df)'] = len(df)
    observed["sorted(df['track_id'].unique().tolist())"] = sorted(df['track_id'].unique().tolist())

    # 3) relabelling is track-consistent even though the raw labels alternate
    a_ids = {int(stack[t][_a_center(t)]) for t in range(4)}
    b_ids = {int(stack[t][_b_center(t)]) for t in range(4)}
    observed['len(a_ids) == 1 and len(b_ids) == 1'] = bool(len(a_ids) == 1 and len(b_ids) == 1)
    observed['a_ids != b_ids'] = bool(a_ids != b_ids)
    observed['a_ids | b_ids'] = a_ids | b_ids

    observed["'Tracking objects with trackpy' in capsys.readouterr().out"] = bool('Tracking objects with trackpy' in capsys.readouterr().out)
    return observed


_EXPECT_trackpy_track_cells_writes_csv_and_relabels_masks = {
    'isinstance(stack, list) and len(stack) == 4': True,
    'all((f.shape == (40, 40) for f in stack))': True,
    'csv_path.is_file()': True,
    "'track_id' in df.columns and 'particle' not in df.columns": True,
    'len(df)': 8,
    "sorted(df['track_id'].unique().tolist())": [1, 2],
    'len(a_ids) == 1 and len(b_ids) == 1': True,
    'a_ids != b_ids': True,
    'a_ids | b_ids': {1, 2},
    "'Tracking objects with trackpy' in capsys.readouterr().out": True,
}


def _case_trackpy_track_cells_remove_transient_drops_partial_tracks(monkeypatch, tmp_path, capsys):
    """timelapse_remove_transient=True filters objects missing from some frames."""
    observed = {}
    from spacr import timelapse as TL

    _link_df_shim(monkeypatch)
    src = str(tmp_path / "masks" / "batch_2.npz")
    os.makedirs(os.path.dirname(src), exist_ok=True)

    stack = TL._trackpy_track_cells(
        src=src, name="n2", batch_filenames=[f"f{i}.tif" for i in range(4)],
        object_type="nucleus", masks=_stack_with_transient(), timelapse_displacement=10,
        timelapse_memory=0, timelapse_remove_transient=True,
        plot=False, save=False, mode="trackpy", track_by_iou=False)

    df = pd.read_csv(tmp_path / "masks" / "tracks" / "trackpy_tracks_nucleus_n2.csv")
    # 10 detections total (2 persistent x 4 frames + transient x 2) -> 8 kept.
    observed['len(df)'] = len(df)
    observed["df['track_id'].nunique()"] = df['track_id'].nunique()
    observed["'Removed 2 objects that were not present in all frames' in capsys.readouterr().out"] = bool('Removed 2 objects that were not present in all frames' in capsys.readouterr().out)
    # The transient object is wiped from the relabelled masks (mapped to 0).
    observed['int(stack[0][6, 34])'] = int(stack[0][6, 34])
    return observed


_EXPECT_trackpy_track_cells_remove_transient_drops_partial_tracks = {
    'len(df)': 8,
    "df['track_id'].nunique()": 2,
    "'Removed 2 objects that were not present in all frames' in capsys.readouterr().out": True,
    'int(stack[0][6, 34])': 0,
}


def _case_trackpy_track_cells_keeps_transients_when_flag_false(monkeypatch, tmp_path, capsys):
    """timelapse_remove_transient=False keeps every detection."""
    observed = {}
    from spacr import timelapse as TL

    _link_df_shim(monkeypatch)
    src = str(tmp_path / "masks" / "batch_3.npz")
    os.makedirs(os.path.dirname(src), exist_ok=True)

    stack = TL._trackpy_track_cells(
        src=src, name="n3", batch_filenames=[f"f{i}.tif" for i in range(4)],
        object_type="pathogen", masks=_stack_with_transient(), timelapse_displacement=10,
        timelapse_memory=0, timelapse_remove_transient=False,
        plot=False, save=False, mode="trackpy", track_by_iou=False)

    df = pd.read_csv(tmp_path / "masks" / "tracks" / "trackpy_tracks_pathogen_n3.csv")
    observed['len(df)'] = len(df)
    observed["df['track_id'].nunique()"] = df['track_id'].nunique()
    observed["'Removed 0 objects' in capsys.readouterr().out"] = bool('Removed 0 objects' in capsys.readouterr().out)
    # The transient object keeps a non-zero track id in the frames it exists in.
    observed['int(stack[0][6, 34]) != 0'] = bool(int(stack[0][6, 34]) != 0)
    observed['int(stack[2][6, 34])'] = int(stack[2][6, 34])
    return observed


_EXPECT_trackpy_track_cells_keeps_transients_when_flag_false = {
    'len(df)': 10,
    "df['track_id'].nunique()": 3,
    "'Removed 0 objects' in capsys.readouterr().out": True,
    'int(stack[0][6, 34]) != 0': True,
    'int(stack[2][6, 34])': 0,
}


def _case_trackpy_track_cells_auto_displacement_uses_find_optimal(monkeypatch, tmp_path, capsys):
    """timelapse_displacement=None -> _find_optimal_search_range picks it."""
    observed = {}
    from spacr import timelapse as TL

    _link_df_shim(monkeypatch)
    seen = {}
    real_find = TL._find_optimal_search_range

    def spy(features, initial_search_range, increment, max_attempts, memory):
        seen["n_rows"] = len(features)
        seen["initial"] = initial_search_range
        seen["increment"] = increment
        seen["max_attempts"] = max_attempts
        seen["memory"] = memory
        return real_find(features, initial_search_range=initial_search_range,
                         increment=increment, max_attempts=max_attempts,
                         memory=memory)

    monkeypatch.setattr(TL, "_find_optimal_search_range", spy)
    src = str(tmp_path / "masks" / "batch_4.npz")
    os.makedirs(os.path.dirname(src), exist_ok=True)

    stack = TL._trackpy_track_cells(
        src=src, name="n4", batch_filenames=[f"f{i}.tif" for i in range(4)],
        object_type="cell", masks=_two_movers(), timelapse_displacement=None,
        timelapse_memory=1, timelapse_remove_transient=False,
        plot=False, save=False, mode="trackpy", track_by_iou=False)

    observed['seen'] = seen
    observed["'Linked on attempt 1 with search_range=500' in capsys.readouterr().out"] = bool('Linked on attempt 1 with search_range=500' in capsys.readouterr().out)
    observed['len(stack)'] = len(stack)
    return observed


_EXPECT_trackpy_track_cells_auto_displacement_uses_find_optimal = {
    'seen': {'n_rows': 8, 'initial': 500, 'increment': 10, 'max_attempts': 49, 'memory': 3},
    "'Linked on attempt 1 with search_range=500' in capsys.readouterr().out": True,
    'len(stack)': 4,
}


def _case_trackpy_track_cells_falls_back_to_50_when_search_range_is_none(monkeypatch, tmp_path):
    """A None from _find_optimal_search_range is replaced by the 50 px default."""
    observed = {}
    from spacr import timelapse as TL

    monkeypatch.setattr(TL, "_find_optimal_search_range", lambda *a, **k: None)

    seen = {}

    def fake_facilitate(masks, search_range, max_attempts, memory, track_by_iou):
        seen["search_range"] = search_range
        seen["max_attempts"] = max_attempts
        seen["memory"] = memory
        seen["track_by_iou"] = track_by_iou
        features = TL._prepare_for_tracking(masks)
        tracks = features.copy()
        tracks["particle"] = tracks["original_label"] - 1
        return masks, features, tracks

    monkeypatch.setattr(TL, "_facilitate_trackin_with_adaptive_removal", fake_facilitate)
    src = str(tmp_path / "masks" / "batch_5.npz")
    os.makedirs(os.path.dirname(src), exist_ok=True)

    stack = TL._trackpy_track_cells(
        src=src, name="n5", batch_filenames=[f"f{i}.tif" for i in range(4)],
        object_type="cell", masks=_two_movers(), timelapse_displacement=None,
        timelapse_memory=7, timelapse_remove_transient=False,
        plot=False, save=False, mode="trackpy", track_by_iou=True)

    observed['seen'] = seen
    observed['len(stack)'] = len(stack)
    return observed


_EXPECT_trackpy_track_cells_falls_back_to_50_when_search_range_is_none = {
    'seen': {'search_range': 50, 'max_attempts': 100, 'memory': 7, 'track_by_iou': True},
    'len(stack)': 4,
}


def _case_trackpy_track_cells_forwards_to_visualizer(monkeypatch, tmp_path, plot, save):
    """plot or save routes the relabelled stack to the plotting helper."""
    observed = {}
    import spacr.plot
    from spacr import timelapse as TL

    _link_df_shim(monkeypatch)
    recorded = {}

    def fake_viz(masks, tracks_df, save_, src_, name_, plot_, filenames, object_type, mode):
        recorded.update(masks=masks, tracks_df=tracks_df, save=save_, src=src_,
                        name=name_, plot=plot_, filenames=filenames,
                        object_type=object_type, mode=mode)

    monkeypatch.setattr(spacr.plot, "_visualize_and_save_timelapse_stack_with_tracks",
                        fake_viz)
    src = str(tmp_path / "masks" / "batch_6.npz")
    os.makedirs(os.path.dirname(src), exist_ok=True)
    filenames = [f"f{i}.tif" for i in range(4)]

    TL._trackpy_track_cells(
        src=src, name="n6", batch_filenames=filenames, object_type="cell",
        masks=_two_movers(), timelapse_displacement=10, timelapse_memory=1,
        timelapse_remove_transient=False, plot=plot, save=save,
        mode="trackpy", track_by_iou=False)

    observed["recorded['plot'] is plot and recorded['save'] is save"] = bool(recorded['plot'] is plot and recorded['save'] is save)
    observed["recorded['src'] == src"] = bool(recorded['src'] == src)
    observed["recorded['name']"] = recorded['name']
    observed["recorded['object_type']"] = recorded['object_type']
    observed["recorded['mode']"] = recorded['mode']
    observed["recorded['filenames'] == filenames"] = bool(recorded['filenames'] == filenames)
    observed["'track_id' in recorded['tracks_df'].columns"] = bool('track_id' in recorded['tracks_df'].columns)
    observed["recorded['masks'].shape"] = recorded['masks'].shape
    return observed


_EXPECT_trackpy_track_cells_forwards_to_visualizer = {
    "recorded['plot'] is plot and recorded['save'] is save": True,
    "recorded['src'] == src": True,
    "recorded['name']": 'n6',
    "recorded['object_type']": 'cell',
    "recorded['mode']": 'trackpy',
    "recorded['filenames'] == filenames": True,
    "'track_id' in recorded['tracks_df'].columns": True,
    "recorded['masks'].shape": (4, 40, 40),
}


def _case_trackpy_track_cells_skips_visualizer_when_neither_plot_nor_save(monkeypatch, tmp_path):
    """plot=False and save=False must not touch the plotting module."""
    observed = {}
    import spacr.plot
    from spacr import timelapse as TL

    _link_df_shim(monkeypatch)

    def boom(*a, **k):
        raise AssertionError("visualizer called with plot=False, save=False")

    monkeypatch.setattr(spacr.plot, "_visualize_and_save_timelapse_stack_with_tracks",
                        boom)
    src = str(tmp_path / "masks" / "batch_7.npz")
    os.makedirs(os.path.dirname(src), exist_ok=True)

    stack = TL._trackpy_track_cells(
        src=src, name="n7", batch_filenames=["a.tif"] * 4, object_type="cell",
        masks=_two_movers(), timelapse_displacement=10, timelapse_memory=1,
        timelapse_remove_transient=False, plot=False, save=False,
        mode="trackpy", track_by_iou=False)
    observed['len(stack)'] = len(stack)
    return observed


_EXPECT_trackpy_track_cells_skips_visualizer_when_neither_plot_nor_save = {
    'len(stack)': 4,
}


def _case_trackpy_track_cells_creates_tracks_dir_when_missing(monkeypatch, tmp_path):
    """The 'tracks' output folder is created on demand (exist_ok is harmless)."""
    observed = {}
    from spacr import timelapse as TL

    _link_df_shim(monkeypatch)
    src = str(tmp_path / "deep" / "nest" / "batch_8.npz")
    os.makedirs(os.path.dirname(src), exist_ok=True)
    tracks_dir = tmp_path / "deep" / "nest" / "tracks"
    observed['not tracks_dir.exists()'] = bool(not tracks_dir.exists())

    for name in ("n8a", "n8b"):
        TL._trackpy_track_cells(
            src=src, name=name, batch_filenames=["a.tif"] * 4, object_type="cell",
            masks=_two_movers(), timelapse_displacement=10, timelapse_memory=1,
            timelapse_remove_transient=False, plot=False, save=False,
            mode="trackpy", track_by_iou=False)

    observed['tracks_dir.is_dir()'] = bool(tracks_dir.is_dir())
    observed['sorted((p.name for p in tracks_dir.iterdir()))'] = sorted((p.name for p in tracks_dir.iterdir()))
    return observed


_EXPECT_trackpy_track_cells_creates_tracks_dir_when_missing = {
    'not tracks_dir.exists()': True,
    'tracks_dir.is_dir()': True,
    'sorted((p.name for p in tracks_dir.iterdir()))': ['trackpy_tracks_cell_n8a.csv', 'trackpy_tracks_cell_n8b.csv'],
}


def _case_trackpy_track_cells_track_by_iou_returns_mask_stack(tmp_path):
    """track_by_iou=True is an advertised mode and should produce a stack."""
    observed = {}
    from spacr.timelapse import _trackpy_track_cells

    src = str(tmp_path / "masks" / "batch_iou.npz")
    os.makedirs(os.path.dirname(src), exist_ok=True)
    stack = _trackpy_track_cells(
        src=src, name="iou", batch_filenames=[f"f{i}.tif" for i in range(4)],
        object_type="cell", masks=_two_movers(), timelapse_displacement=10,
        timelapse_memory=1, timelapse_remove_transient=False,
        plot=False, save=False, mode="trackpy", track_by_iou=True)
    observed['isinstance(stack, list) and len(stack) == 4'] = bool(isinstance(stack, list) and len(stack) == 4)
    return observed


_EXPECT_trackpy_track_cells_track_by_iou_returns_mask_stack = {
    'isinstance(stack, list) and len(stack) == 4': True,
}


def _case_automatic_tracking_fallback_uses_a_bounded_range_and_retains_source_labels(tmp_path, monkeypatch):
    """A failed range estimate falls back to50 without changing either mover's identity."""
    observed = {}
    from spacr import timelapse as TL

    masks = _two_movers(n_frames=4)
    before = np.asarray(masks).copy()
    chosen = []
    original = TL._facilitate_trackin_with_adaptive_removal

    def track(*args, **kwargs):
        chosen.append(kwargs['search_range'])
        return original(*args, **kwargs)

    monkeypatch.setattr(TL, '_find_optimal_search_range', lambda *args, **kwargs: None)
    monkeypatch.setattr(TL, '_facilitate_trackin_with_adaptive_removal', track)
    src = tmp_path / 'masks' / 'batch.npz'
    src.parent.mkdir()
    output = TL._trackpy_track_cells(str(src), 'known', [f'f{i}.tif' for i in range(4)],
        'cell', masks, None, 1, False, False, False, mode='trackpy', track_by_iou=True)
    observed['chosen'] = chosen
    np.testing.assert_array_equal(masks, before)
    observed['len(output)'] = len(output)
    a_ids = {int(output[index][_a_center(index)]) for index in range(4)}
    b_ids = {int(output[index][_b_center(index)]) for index in range(4)}
    observed['len(a_ids) == len(b_ids) == 1 and a_ids != b_ids'] = bool(len(a_ids) == len(b_ids) == 1 and a_ids != b_ids)
    return observed


_EXPECT_automatic_tracking_fallback_uses_a_bounded_range_and_retains_source_labels = {
    'chosen': [50],
    'len(output)': 4,
    'len(a_ids) == len(b_ids) == 1 and a_ids != b_ids': True,
}


def _case_transient_filter_drops_a_one_frame_mover_without_changing_the_persistent_track(tmp_path):
    observed = {}
    from spacr import timelapse as TL

    masks = np.asarray(_two_movers(n_frames=4))
    for index in range(3):
        transient = masks[index][_b_center(index)]
        masks[index][masks[index] == transient] = 0
    before = masks.copy()
    src = tmp_path / 'masks' / 'transient.npz'
    src.parent.mkdir()
    output = TL._trackpy_track_cells(str(src), 'transient', [f'f{i}.tif' for i in range(4)],
        'cell', masks, 10, 1, True, False, False, mode='trackpy', track_by_iou=True)
    observed['len(output)'] = len(output)
    persistent = {int(output[index][_a_center(index)]) for index in range(4)}
    observed['len(persistent) == 1 and 0 not in persistent'] = bool(len(persistent) == 1 and 0 not in persistent)
    observed['output[3][_b_center(3)]'] = output[3][_b_center(3)]
    np.testing.assert_array_equal(masks, before)
    return observed


_EXPECT_transient_filter_drops_a_one_frame_mover_without_changing_the_persistent_track = {
    'len(output)': 4,
    'len(persistent) == 1 and 0 not in persistent': True,
    'output[3][_b_center(3)]': 0,
}


# ---------------------------------------------------------------------------
# behaviour tests: one per public-function contract, one parametrised row per case
# ---------------------------------------------------------------------------

def _run_case(case, tmp_path, monkeypatch, capsys):
    """Call a case helper with only the fixtures its signature asks for."""
    wanted = inspect.signature(case).parameters
    available = {"tmp_path": tmp_path, "monkeypatch": monkeypatch, "capsys": capsys}
    return case(**{k: v for k, v in available.items() if k in wanted})


_P = functools.partial


SEARCH_RANGE_CASES = [
    pytest.param(_case_find_optimal_search_range_returns_initial_on_first_success, _EXPECT_find_optimal_search_range_returns_initial_on_first_success, id="returns-initial-on-first-success"),
    pytest.param(_case_find_optimal_search_range_forwards_memory_and_search_range, _EXPECT_find_optimal_search_range_forwards_memory_and_search_range, id="forwards-memory-and-search-range"),
    pytest.param(_case_find_optimal_search_range_decrements_until_link_succeeds, _EXPECT_find_optimal_search_range_decrements_until_link_succeeds, id="decrements-until-link-succeeds"),
    pytest.param(_case_find_optimal_search_range_exhausts_attempts_and_warns, _EXPECT_find_optimal_search_range_exhausts_attempts_and_warns, id="exhausts-attempts-and-warns"),
    pytest.param(_case_find_optimal_search_range_zero_attempts_is_a_noop, _EXPECT_find_optimal_search_range_zero_attempts_is_a_noop, id="zero-attempts-is-a-noop"),
]


@pytest.mark.parametrize("case, expected", SEARCH_RANGE_CASES)
def test_find_optimal_search_range_returns_the_first_range_trackpy_accepts_and_warns_with_the_floor_when_none_do(case, expected, tmp_path, monkeypatch, capsys):
    """Search range choice: forwarded arguments, decrement per failure, floor warning, zero attempts."""
    observed = _run_case(case, tmp_path, monkeypatch, capsys)
    assert observed == expected


FACILITATE_CASES = [
    pytest.param(_case_facilitate_drops_small_frame0_objects_only, _EXPECT_facilitate_drops_small_frame0_objects_only, id="drops-small-frame0-objects-only"),
    pytest.param(_case_facilitate_track_by_iou_gives_one_track_per_object, _EXPECT_facilitate_track_by_iou_gives_one_track_per_object, id="track-by-iou-gives-one-track-per-object"),
    pytest.param(_case_facilitate_auto_search_range_from_area_quantile, _EXPECT_facilitate_auto_search_range_from_area_quantile, id="auto-search-range-from-area-quantile"),
    pytest.param(_case_facilitate_retries_with_shrunken_search_range, _EXPECT_facilitate_retries_with_shrunken_search_range, id="retries-with-shrunken-search-range"),
    pytest.param(_case_facilitate_search_range_never_shrinks_below_one, _EXPECT_facilitate_search_range_never_shrinks_below_one, id="search-range-never-shrinks-below-one"),
    pytest.param(_case_facilitate_raises_runtime_error_after_max_attempts, _EXPECT_facilitate_raises_runtime_error_after_max_attempts, id="raises-runtime-error-after-max-attempts"),
    pytest.param(_case_facilitate_track_by_iou_ignores_link_df_entirely, _EXPECT_facilitate_track_by_iou_ignores_link_df_entirely, id="track-by-iou-ignores-link-df-entirely"),
    pytest.param(_case_facilitate_with_real_trackpy_links_two_particles, _EXPECT_facilitate_with_real_trackpy_links_two_particles, id="with-real-trackpy-links-two-particles"),
    pytest.param(_case_facilitate_real_trackpy_predict_kwarg_is_rejected, _EXPECT_facilitate_real_trackpy_predict_kwarg_is_rejected, id="real-trackpy-predict-kwarg-is-rejected"),
]


@pytest.mark.parametrize("case, expected", FACILITATE_CASES)
def test_facilitate_tracking_prunes_small_objects_links_with_the_shrinking_range_and_raises_when_attempts_run_out(case, expected, tmp_path, monkeypatch, capsys):
    """Adaptive linking: min_mass pruning, auto range, 80 percent retries, floor of one, RuntimeError, IoU mode, real trackpy."""
    observed = _run_case(case, tmp_path, monkeypatch, capsys)
    assert observed == expected


TRACK_CELLS_CASES = [
    pytest.param(_case_trackpy_track_cells_writes_csv_and_relabels_masks, _EXPECT_trackpy_track_cells_writes_csv_and_relabels_masks, id="trackpy-track-cells-writes-csv-and-relabels-masks"),
    pytest.param(_case_trackpy_track_cells_remove_transient_drops_partial_tracks, _EXPECT_trackpy_track_cells_remove_transient_drops_partial_tracks, id="trackpy-track-cells-remove-transient-drops-partial-tracks"),
    pytest.param(_case_trackpy_track_cells_keeps_transients_when_flag_false, _EXPECT_trackpy_track_cells_keeps_transients_when_flag_false, id="trackpy-track-cells-keeps-transients-when-flag-false"),
    pytest.param(_case_trackpy_track_cells_auto_displacement_uses_find_optimal, _EXPECT_trackpy_track_cells_auto_displacement_uses_find_optimal, id="trackpy-track-cells-auto-displacement-uses-find-optimal"),
    pytest.param(_case_trackpy_track_cells_falls_back_to_50_when_search_range_is_none, _EXPECT_trackpy_track_cells_falls_back_to_50_when_search_range_is_none, id="trackpy-track-cells-falls-back-to-50-when-search-range-is-none"),
    pytest.param(_P(_case_trackpy_track_cells_forwards_to_visualizer, plot=True, save=False), _EXPECT_trackpy_track_cells_forwards_to_visualizer, id="trackpy-track-cells-forwards-to-visualizer-plot"),
    pytest.param(_P(_case_trackpy_track_cells_forwards_to_visualizer, plot=False, save=True), _EXPECT_trackpy_track_cells_forwards_to_visualizer, id="trackpy-track-cells-forwards-to-visualizer-save"),
    pytest.param(_P(_case_trackpy_track_cells_forwards_to_visualizer, plot=True, save=True), _EXPECT_trackpy_track_cells_forwards_to_visualizer, id="trackpy-track-cells-forwards-to-visualizer-plot-and-save"),
    pytest.param(_case_trackpy_track_cells_skips_visualizer_when_neither_plot_nor_save, _EXPECT_trackpy_track_cells_skips_visualizer_when_neither_plot_nor_save, id="trackpy-track-cells-skips-visualizer-when-neither-plot-nor-save"),
    pytest.param(_case_trackpy_track_cells_creates_tracks_dir_when_missing, _EXPECT_trackpy_track_cells_creates_tracks_dir_when_missing, id="trackpy-track-cells-creates-tracks-dir-when-missing"),
    pytest.param(_case_trackpy_track_cells_track_by_iou_returns_mask_stack, _EXPECT_trackpy_track_cells_track_by_iou_returns_mask_stack, id="trackpy-track-cells-track-by-iou-returns-mask-stack"),
    pytest.param(_case_automatic_tracking_fallback_uses_a_bounded_range_and_retains_source_labels, _EXPECT_automatic_tracking_fallback_uses_a_bounded_range_and_retains_source_labels, id="automatic-tracking-fallback-uses-a-bounded-range-and-retains-source-labels"),
    pytest.param(_case_transient_filter_drops_a_one_frame_mover_without_changing_the_persistent_track, _EXPECT_transient_filter_drops_a_one_frame_mover_without_changing_the_persistent_track, id="transient-filter-drops-a-one-frame-mover-without-changing-the-persistent-track"),
]


@pytest.mark.parametrize("case, expected", TRACK_CELLS_CASES)
def test_trackpy_track_cells_writes_the_tracks_csv_and_returns_track_consistent_relabelled_masks(case, expected, tmp_path, monkeypatch, capsys):
    """Tracks CSV, relabelled masks, transient filtering, range fallbacks, visualizer routing and tracks directory creation."""
    observed = _run_case(case, tmp_path, monkeypatch, capsys)
    assert observed == expected

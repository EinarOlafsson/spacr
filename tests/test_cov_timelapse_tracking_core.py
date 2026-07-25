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

import os

import numpy as np
import pandas as pd
import pytest

import matplotlib
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

    monkeypatch.setattr(TL.tp, "link_df", shim)
    return shim


# ===========================================================================
# _find_optimal_search_range
# ===========================================================================

def test_find_optimal_search_range_returns_initial_on_first_success(capsys):
    """A search range trackpy accepts is returned untouched (no decrement)."""
    from spacr.timelapse import _find_optimal_search_range, _prepare_for_tracking

    features = _prepare_for_tracking(_two_movers())
    out = _find_optimal_search_range(features, initial_search_range=20,
                                     increment=5, max_attempts=3, memory=1)
    assert out == 20
    assert "Success with search_range=20" in capsys.readouterr().out


def test_find_optimal_search_range_forwards_memory_and_search_range(monkeypatch):
    """The current candidate range and the memory argument reach tp.link."""
    from spacr import timelapse as TL

    seen = {}

    def fake_link(features, search_range, memory):
        seen["search_range"] = search_range
        seen["memory"] = memory
        seen["n_rows"] = len(features)
        return pd.DataFrame({"particle": [0]})

    monkeypatch.setattr(TL.tp, "link", fake_link)
    features = TL._prepare_for_tracking(_two_movers())
    out = TL._find_optimal_search_range(features, initial_search_range=123,
                                        increment=7, max_attempts=4, memory=9)
    assert out == 123
    assert seen == {"search_range": 123, "memory": 9, "n_rows": len(features)}


def test_find_optimal_search_range_decrements_until_link_succeeds(monkeypatch, capsys):
    """Each failure shrinks the range by ``increment``; the winner is returned."""
    from spacr import timelapse as TL

    calls = []

    def flaky_link(features, search_range, memory):
        calls.append(search_range)
        if len(calls) <= 2:
            raise RuntimeError("SubnetOversizeException")
        return pd.DataFrame({"particle": [0, 1]})

    monkeypatch.setattr(TL.tp, "link", flaky_link)
    features = TL._prepare_for_tracking(_two_movers())
    out = TL._find_optimal_search_range(features, initial_search_range=100,
                                        increment=10, max_attempts=6, memory=3)
    # 100 fails, 90 fails, 80 succeeds.
    assert calls == [100, 90, 80]
    assert out == 80
    captured = capsys.readouterr().out
    assert "Success with search_range=80" in captured
    assert "too high" not in captured


def test_find_optimal_search_range_exhausts_attempts_and_warns(monkeypatch, capsys):
    """When every attempt fails the floor value is returned with a warning."""
    from spacr import timelapse as TL

    calls = []

    def always_fail(features, search_range, memory):
        calls.append(search_range)
        raise RuntimeError("SubnetOversizeException")

    monkeypatch.setattr(TL.tp, "link", always_fail)
    features = TL._prepare_for_tracking(_two_movers())
    out = TL._find_optimal_search_range(features, initial_search_range=50,
                                        increment=10, max_attempts=3, memory=3)
    assert calls == [50, 40, 30]
    # 50 - 3*10 == 20 == the advertised floor.
    assert out == 20
    captured = capsys.readouterr().out
    assert "timelapse_displacement=20 is too high" in captured


def test_find_optimal_search_range_zero_attempts_is_a_noop(monkeypatch, capsys):
    """max_attempts=0 never calls trackpy and returns the initial range."""
    from spacr import timelapse as TL

    def boom(*a, **k):  # pragma: no branch - must never run
        raise AssertionError("tp.link must not be called with max_attempts=0")

    monkeypatch.setattr(TL.tp, "link", boom)
    features = TL._prepare_for_tracking(_two_movers())
    out = TL._find_optimal_search_range(features, initial_search_range=37,
                                        increment=10, max_attempts=0, memory=3)
    assert out == 37
    assert "timelapse_displacement=37 is too high" in capsys.readouterr().out


# ===========================================================================
# _facilitate_trackin_with_adaptive_removal
# ===========================================================================

def test_facilitate_drops_small_frame0_objects_only(monkeypatch):
    """min_mass prunes frame 0 in place; later frames keep every object."""
    from spacr import timelapse as TL

    masks = _two_movers()
    # Add a 3x3 (mass 9) speck present in every frame -> only frame 0 loses it.
    masks[:, 34:37, 2:5] = 7
    original = masks.copy()

    out_masks, features, tracks = TL._facilitate_trackin_with_adaptive_removal(
        masks, search_range=10, max_attempts=2, memory=1, min_mass=50,
        track_by_iou=True)

    assert out_masks is masks                      # mutated in place
    assert 7 not in np.unique(out_masks[0])        # speck gone from frame 0
    assert 7 in np.unique(out_masks[1])            # untouched elsewhere
    # The two 8x8 squares survived frame 0 unchanged.
    assert np.array_equal(out_masks[0] == 1, original[0] == 1)
    assert np.array_equal(out_masks[0] == 2, original[0] == 2)
    # Features were recomputed on the filtered stack.
    f0 = features[features["frame"] == 0]
    assert set(f0["original_label"]) == {1, 2}
    assert set(f0["mass"]) == {64.0}
    assert list(tracks.columns) == ["frame", "original_label", "track_id"]


def test_facilitate_track_by_iou_gives_one_track_per_object():
    """IoU linking survives the label swap and yields exactly two tracks."""
    from spacr.timelapse import _facilitate_trackin_with_adaptive_removal

    masks = _two_movers(n_frames=4)
    _, _, tracks = _facilitate_trackin_with_adaptive_removal(
        masks, search_range=10, max_attempts=2, memory=1, min_mass=10,
        track_by_iou=True)

    assert len(tracks) == 8                       # 4 frames x 2 objects
    assert tracks["track_id"].nunique() == 2
    # Object A is label 1 on even frames and label 2 on odd frames; both rows
    # must carry the same track id.
    def tid(frame, label):
        row = tracks[(tracks["frame"] == frame) & (tracks["original_label"] == label)]
        return int(row["track_id"].iloc[0])
    assert tid(0, 1) == tid(1, 2) == tid(2, 1) == tid(3, 2)


def test_facilitate_auto_search_range_from_area_quantile(monkeypatch):
    """search_range=None -> 2*sqrt(99th-percentile frame-0 area)."""
    from spacr import timelapse as TL

    masks = _two_movers()
    masks[:, 34:37, 2:5] = 7                       # tiny object, mass 9

    expected_f0 = TL._prepare_for_tracking(masks)
    expected_f0 = expected_f0[expected_f0["frame"] == 0]
    expected = max(1, int(2 * np.sqrt(expected_f0["mass"].quantile(0.99))))
    assert expected > 1                             # sanity: a real value

    seen = {}

    def fake_link_df(features, search_range, memory):
        seen["search_range"] = search_range
        seen["memory"] = memory
        out = features.copy()
        out["particle"] = out["original_label"] - 1
        return out

    monkeypatch.setattr(TL.tp, "link_df", fake_link_df)
    _, _, tracks = TL._facilitate_trackin_with_adaptive_removal(
        masks, search_range=None, max_attempts=3, memory=4, min_mass=50,
        track_by_iou=False)

    assert seen["search_range"] == expected
    assert seen["memory"] == 4
    assert "particle" in tracks.columns


def test_facilitate_retries_with_shrunken_search_range(monkeypatch, capsys):
    """A failed attempt shrinks search_range to 80% and retries."""
    from spacr import timelapse as TL

    calls = []

    def flaky_link_df(features, search_range, memory):
        calls.append(search_range)
        if len(calls) == 1:
            raise ValueError("SubnetOversizeException")
        out = features.copy()
        out["particle"] = out["original_label"] - 1
        return out

    monkeypatch.setattr(TL.tp, "link_df", flaky_link_df)
    masks, features, tracks = TL._facilitate_trackin_with_adaptive_removal(
        _two_movers(), search_range=100, max_attempts=4, memory=1, min_mass=10,
        track_by_iou=False)

    assert calls == [100, 80]                       # int(100 * 0.8)
    assert len(tracks) == len(features)
    captured = capsys.readouterr().out
    assert "reducing search_range to 80" in captured
    assert "Linked on attempt 2 with search_range=80" in captured


def test_facilitate_search_range_never_shrinks_below_one(monkeypatch, capsys):
    """The max(1, ...) floor keeps search_range positive across many retries."""
    from spacr import timelapse as TL

    calls = []

    def always_fail(features, search_range, memory):
        calls.append(search_range)
        raise ValueError("nope")

    monkeypatch.setattr(TL.tp, "link_df", always_fail)
    with pytest.raises(RuntimeError) as exc:
        TL._facilitate_trackin_with_adaptive_removal(
            _two_movers(), search_range=3, max_attempts=6, memory=1,
            min_mass=10, track_by_iou=False)

    assert calls == [3, 2, 1, 1, 1, 1]
    assert min(calls) == 1
    assert "Failed to track after 6 attempts" in str(exc.value)
    assert "last search_range=1" in str(exc.value)


def test_facilitate_raises_runtime_error_after_max_attempts(monkeypatch):
    """Exhausting max_attempts raises RuntimeError, not the inner exception."""
    from spacr import timelapse as TL

    def always_fail(features, search_range, memory):
        raise ZeroDivisionError("inner failure")

    monkeypatch.setattr(TL.tp, "link_df", always_fail)
    with pytest.raises(RuntimeError, match=r"Failed to track after 2 attempts"):
        TL._facilitate_trackin_with_adaptive_removal(
            _two_movers(), search_range=10, max_attempts=2, memory=1,
            min_mass=10, track_by_iou=False)


def test_facilitate_track_by_iou_ignores_link_df_entirely(monkeypatch):
    """With track_by_iou=True trackpy is never consulted."""
    from spacr import timelapse as TL

    def boom(*a, **k):
        raise AssertionError("tp.link_df must not be called when track_by_iou=True")

    monkeypatch.setattr(TL.tp, "link_df", boom)
    _, _, tracks = TL._facilitate_trackin_with_adaptive_removal(
        _two_movers(), search_range=10, max_attempts=2, memory=1, min_mass=10,
        track_by_iou=True)
    assert tracks["track_id"].nunique() == 2


def test_facilitate_with_real_trackpy_links_two_particles():
    """Real trackpy links both squares through the module-level ``tp.link_df``."""
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

    assert "particle" in tracks.columns
    assert tracks["particle"].nunique() == 2
    assert len(tracks) == 8


def test_facilitate_real_trackpy_predict_kwarg_is_rejected():
    """The unpatched trackpy path links instead of raising RuntimeError."""
    from spacr.timelapse import _facilitate_trackin_with_adaptive_removal

    _, _, tracks = _facilitate_trackin_with_adaptive_removal(
        _two_movers(), search_range=10, max_attempts=2, memory=1,
        min_mass=10, track_by_iou=False)
    assert "particle" in tracks.columns
    assert tracks["particle"].nunique() == 2


# ===========================================================================
# _trackpy_track_cells
# ===========================================================================

def test_trackpy_track_cells_writes_csv_and_relabels_masks(monkeypatch, tmp_path, capsys):
    """End-to-end: tracks CSV on disk, masks relabelled to 1-based track ids."""
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
    assert isinstance(stack, list) and len(stack) == 4
    assert all(f.shape == (40, 40) for f in stack)

    # 2) CSV written next to src
    csv_path = tmp_path / "masks" / "tracks" / "trackpy_tracks_cell_plate1_A01_1.csv"
    assert csv_path.is_file()
    df = pd.read_csv(csv_path)
    assert "track_id" in df.columns and "particle" not in df.columns
    assert len(df) == 8
    assert sorted(df["track_id"].unique().tolist()) == [1, 2]   # particle += 1

    # 3) relabelling is track-consistent even though the raw labels alternate
    a_ids = {int(stack[t][_a_center(t)]) for t in range(4)}
    b_ids = {int(stack[t][_b_center(t)]) for t in range(4)}
    assert len(a_ids) == 1 and len(b_ids) == 1
    assert a_ids != b_ids
    assert a_ids | b_ids == {1, 2}

    assert "Tracking objects with trackpy" in capsys.readouterr().out


def test_trackpy_track_cells_remove_transient_drops_partial_tracks(monkeypatch, tmp_path, capsys):
    """timelapse_remove_transient=True filters objects missing from some frames."""
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
    assert len(df) == 8
    assert df["track_id"].nunique() == 2
    assert "Removed 2 objects that were not present in all frames" in capsys.readouterr().out
    # The transient object is wiped from the relabelled masks (mapped to 0).
    assert int(stack[0][6, 34]) == 0


def test_trackpy_track_cells_keeps_transients_when_flag_false(monkeypatch, tmp_path, capsys):
    """timelapse_remove_transient=False keeps every detection."""
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
    assert len(df) == 10
    assert df["track_id"].nunique() == 3
    assert "Removed 0 objects" in capsys.readouterr().out
    # The transient object keeps a non-zero track id in the frames it exists in.
    assert int(stack[0][6, 34]) != 0
    assert int(stack[2][6, 34]) == 0


def test_trackpy_track_cells_auto_displacement_uses_find_optimal(monkeypatch, tmp_path, capsys):
    """timelapse_displacement=None -> _find_optimal_search_range picks it."""
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

    assert seen == {"n_rows": 8, "initial": 500, "increment": 10,
                    "max_attempts": 49, "memory": 3}
    assert "Linked on attempt 1 with search_range=500" in capsys.readouterr().out
    assert len(stack) == 4


def test_trackpy_track_cells_falls_back_to_50_when_search_range_is_none(monkeypatch, tmp_path):
    """A None from _find_optimal_search_range is replaced by the 50 px default."""
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

    assert seen == {"search_range": 50, "max_attempts": 100, "memory": 7,
                    "track_by_iou": True}
    assert len(stack) == 4


@pytest.mark.parametrize("plot,save", [(True, False), (False, True), (True, True)])
def test_trackpy_track_cells_forwards_to_visualizer(monkeypatch, tmp_path, plot, save):
    """plot or save routes the relabelled stack to the plotting helper."""
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

    assert recorded["plot"] is plot and recorded["save"] is save
    assert recorded["src"] == src
    assert recorded["name"] == "n6"
    assert recorded["object_type"] == "cell"
    assert recorded["mode"] == "trackpy"
    assert recorded["filenames"] == filenames
    assert "track_id" in recorded["tracks_df"].columns
    assert recorded["masks"].shape == (4, 40, 40)


def test_trackpy_track_cells_skips_visualizer_when_neither_plot_nor_save(monkeypatch, tmp_path):
    """plot=False and save=False must not touch the plotting module."""
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
    assert len(stack) == 4


def test_trackpy_track_cells_creates_tracks_dir_when_missing(monkeypatch, tmp_path):
    """The 'tracks' output folder is created on demand (exist_ok is harmless)."""
    from spacr import timelapse as TL

    _link_df_shim(monkeypatch)
    src = str(tmp_path / "deep" / "nest" / "batch_8.npz")
    os.makedirs(os.path.dirname(src), exist_ok=True)
    tracks_dir = tmp_path / "deep" / "nest" / "tracks"
    assert not tracks_dir.exists()

    for name in ("n8a", "n8b"):
        TL._trackpy_track_cells(
            src=src, name=name, batch_filenames=["a.tif"] * 4, object_type="cell",
            masks=_two_movers(), timelapse_displacement=10, timelapse_memory=1,
            timelapse_remove_transient=False, plot=False, save=False,
            mode="trackpy", track_by_iou=False)

    assert tracks_dir.is_dir()
    assert sorted(p.name for p in tracks_dir.iterdir()) == [
        "trackpy_tracks_cell_n8a.csv", "trackpy_tracks_cell_n8b.csv"]


def test_trackpy_track_cells_track_by_iou_returns_mask_stack(tmp_path):
    """track_by_iou=True is an advertised mode and should produce a stack."""
    from spacr.timelapse import _trackpy_track_cells

    src = str(tmp_path / "masks" / "batch_iou.npz")
    os.makedirs(os.path.dirname(src), exist_ok=True)
    stack = _trackpy_track_cells(
        src=src, name="iou", batch_filenames=[f"f{i}.tif" for i in range(4)],
        object_type="cell", masks=_two_movers(), timelapse_displacement=10,
        timelapse_memory=1, timelapse_remove_transient=False,
        plot=False, save=False, mode="trackpy", track_by_iou=True)
    assert isinstance(stack, list) and len(stack) == 4

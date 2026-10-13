"""CPU-only tests for the Trackastra tracking backend in spacr.timelapse.

Trackastra is an OPTIONAL dependency and its pretrained weights are fetched
over the network, so nothing here installs or downloads it. The package is
injected into ``sys.modules`` as a stub, which lets these tests pin the part
spaCR actually owns: the adapter contract. Specifically that

  * the tracks table matches the layout trackpy/btrack already emit, because
    the track visualiser and the motility assay both read it,
  * ids are consistent across frames,
  * degenerate inputs (single frame, empty masks, shape mismatch) are handled
    rather than reaching Trackastra,
  * and a missing package produces an actionable message instead of a bare
    ImportError from three frames down.

That last one matters: this repo has a documented habit of failures that look
like "no data", and an optional-dependency ImportError is exactly that shape.
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# stub trackastra
# ---------------------------------------------------------------------------

def _disc(frame, cy, cx, label, r=4):
    """Stamp a filled disc of `label` into `frame`."""
    yy, xx = np.ogrid[:frame.shape[0], :frame.shape[1]]
    frame[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = label
    return frame


def _moving_stack(n_frames=4, size=48, n_objects=2):
    """A (T, Y, X) label stack whose objects drift right by 3 px per frame.

    Labels are deliberately SHUFFLED per frame — object 1 is not label 1 in
    every frame — so a test that passes cannot be passing by accident of the
    labels already being consistent.
    """
    masks = np.zeros((n_frames, size, size), dtype=np.uint16)
    for t in range(n_frames):
        for i in range(n_objects):
            # rotate label ids each frame
            label = (i + t) % n_objects + 1
            _disc(masks[t], cy=12 + i * 18, cx=10 + 3 * t, label=label)
    return masks


class _StubTrackGraph:
    """Stand-in for Trackastra's returned graph; the adapter only passes it on."""


def _install_stub_trackastra(monkeypatch, relabelled=None, record=None):
    """Inject a fake `trackastra` package.

    ``graph_to_ctc`` returns ``relabelled`` as the tracked stack, which is what
    the adapter derives its tracks table from.
    """
    model_mod = types.ModuleType("trackastra.model")
    tracking_mod = types.ModuleType("trackastra.tracking")
    pkg = types.ModuleType("trackastra")

    class _Trackastra:
        @classmethod
        def from_pretrained(cls, name, device="automatic"):
            if record is not None:
                record["model_name"] = name
                record["device"] = device
            return cls()

        def track(self, imgs, masks, mode="greedy"):
            if record is not None:
                record["mode"] = mode
                record["imgs_shape"] = np.asarray(imgs).shape
                record["masks_shape"] = np.asarray(masks).shape
            return _StubTrackGraph()

    def _graph_to_ctc(graph, masks, outdir=None):
        out = relabelled if relabelled is not None else np.asarray(masks)
        return pd.DataFrame(), out

    model_mod.Trackastra = _Trackastra
    tracking_mod.graph_to_ctc = _graph_to_ctc
    pkg.model = model_mod
    pkg.tracking = tracking_mod
    monkeypatch.setitem(sys.modules, "trackastra", pkg)
    monkeypatch.setitem(sys.modules, "trackastra.model", model_mod)
    monkeypatch.setitem(sys.modules, "trackastra.tracking", tracking_mod)


@pytest.fixture(autouse=True)
def _no_figures():
    yield
    try:
        import matplotlib.pyplot as plt
        plt.close("all")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# missing dependency
# ---------------------------------------------------------------------------

def test_missing_trackastra_raises_an_actionable_runtimeerror(monkeypatch, tmp_path):
    """A bare ImportError would read like 'no data'. It must name the fix."""
    from spacr.timelapse import _trackastra_track_cells

    # make the import fail even if trackastra is installed in the env
    monkeypatch.setitem(sys.modules, "trackastra", None)
    monkeypatch.setitem(sys.modules, "trackastra.model", None)

    with pytest.raises(RuntimeError) as exc:
        _trackastra_track_cells(
            src=str(tmp_path / "run" / "x"), name="b1", batch_filenames=[],
            object_type="cell", masks=_moving_stack())

    msg = str(exc.value)
    assert "pip install trackastra" in msg
    assert "trackpy" in msg and "btrack" in msg      # names the alternatives


# ---------------------------------------------------------------------------
# adapter contract
# ---------------------------------------------------------------------------

def test_tracks_table_matches_the_trackpy_btrack_layout(monkeypatch, tmp_path):
    """The visualiser and motility assay read frame/track_id/x/y."""
    from spacr.timelapse import _trackastra_track_cells

    masks = _moving_stack(n_frames=3, n_objects=2)
    # consistent ids across frames — what a real tracker would return
    tracked = np.zeros_like(masks)
    for t in range(3):
        _disc(tracked[t], cy=12, cx=10 + 3 * t, label=1)
        _disc(tracked[t], cy=30, cx=10 + 3 * t, label=2)
    _install_stub_trackastra(monkeypatch, relabelled=tracked)

    src = tmp_path / "run" / "batch"
    src.parent.mkdir(parents=True, exist_ok=True)
    out = _trackastra_track_cells(
        src=str(src), name="b1", batch_filenames=["a.tif", "b.tif", "c.tif"],
        object_type="cell", masks=masks, images=masks.astype(np.float32))

    csv = tmp_path / "run" / "tracks" / "trackastra_tracks_cell_b1.csv"
    assert csv.exists(), "tracks CSV was not written"
    df = pd.read_csv(csv)
    assert list(df.columns) == ["frame", "track_id", "original_label", "x", "y"]
    # two objects, three frames, ids stable
    assert sorted(df["track_id"].unique().tolist()) == [1, 2]
    assert df.groupby("track_id")["frame"].nunique().tolist() == [3, 3]
    # centroids move right by ~3 px per frame, proving x/y come from the masks
    xs = df[df["track_id"] == 1].sort_values("frame")["x"].tolist()
    assert xs[1] - xs[0] == pytest.approx(3.0, abs=0.6)
    assert len(out) == masks.shape[0]   # _masks_to_masks_stack returns a list


def test_images_are_forwarded_so_appearance_is_used(monkeypatch, tmp_path):
    """Trackastra uses appearance as well as geometry; the raw stack must reach it."""
    from spacr.timelapse import _trackastra_track_cells

    rec = {}
    masks = _moving_stack(n_frames=2)
    imgs = (masks > 0).astype(np.float32) * 1234.0
    _install_stub_trackastra(monkeypatch, record=rec)

    src = tmp_path / "run" / "batch"; src.parent.mkdir(parents=True, exist_ok=True)
    _trackastra_track_cells(src=str(src), name="b", batch_filenames=[],
                            object_type="cell", masks=masks, images=imgs)
    assert rec["imgs_shape"] == imgs.shape
    assert rec["masks_shape"] == masks.shape


def test_model_name_and_linking_mode_are_passed_through(monkeypatch, tmp_path):
    from spacr.timelapse import _trackastra_track_cells

    rec = {}
    _install_stub_trackastra(monkeypatch, record=rec)
    src = tmp_path / "run" / "batch"; src.parent.mkdir(parents=True, exist_ok=True)
    _trackastra_track_cells(src=str(src), name="b", batch_filenames=[],
                            object_type="cell", masks=_moving_stack(n_frames=2),
                            model_name="general_2d", linking_mode="ilp")
    assert rec["model_name"] == "general_2d"
    assert rec["mode"] == "ilp"


def test_masks_stand_in_for_images_when_none_given(monkeypatch, tmp_path):
    """images=None must not crash; the masks are used as a stand-in."""
    from spacr.timelapse import _trackastra_track_cells

    rec = {}
    masks = _moving_stack(n_frames=2)
    _install_stub_trackastra(monkeypatch, record=rec)
    src = tmp_path / "run" / "batch"; src.parent.mkdir(parents=True, exist_ok=True)
    _trackastra_track_cells(src=str(src), name="b", batch_filenames=[],
                            object_type="cell", masks=masks, images=None)
    assert rec["imgs_shape"] == masks.shape


# ---------------------------------------------------------------------------
# degenerate inputs
# ---------------------------------------------------------------------------

def test_single_frame_returns_without_calling_trackastra(monkeypatch, tmp_path, capsys):
    """One frame has nothing to link — don't pay for a model load."""
    from spacr.timelapse import _trackastra_track_cells

    rec = {}
    _install_stub_trackastra(monkeypatch, record=rec)
    masks = _moving_stack(n_frames=1)
    src = tmp_path / "run" / "batch"; src.parent.mkdir(parents=True, exist_ok=True)
    out = _trackastra_track_cells(src=str(src), name="b", batch_filenames=[],
                                  object_type="cell", masks=masks)
    assert rec == {}, "Trackastra should not have been constructed"
    assert "nothing to link" in capsys.readouterr().out
    assert len(out) == 1


def test_non_3d_masks_are_rejected(monkeypatch, tmp_path):
    from spacr.timelapse import _trackastra_track_cells

    _install_stub_trackastra(monkeypatch)
    with pytest.raises(ValueError, match=r"\(T, Y, X\)"):
        _trackastra_track_cells(src=str(tmp_path / "r" / "b"), name="b",
                                batch_filenames=[], object_type="cell",
                                masks=np.zeros((8, 8), np.uint16))


def test_mismatched_image_shape_is_rejected(monkeypatch, tmp_path):
    """A silent shape mismatch would give Trackastra misaligned appearance."""
    from spacr.timelapse import _trackastra_track_cells

    _install_stub_trackastra(monkeypatch)
    masks = _moving_stack(n_frames=2, size=48)
    with pytest.raises(ValueError, match="does not match"):
        _trackastra_track_cells(src=str(tmp_path / "r" / "b"), name="b",
                                batch_filenames=[], object_type="cell",
                                masks=masks, images=np.zeros((2, 16, 16), np.float32))


def test_empty_masks_produce_an_empty_tracks_table(monkeypatch, tmp_path):
    """No objects must not raise; it should yield an empty, correctly-shaped table."""
    from spacr.timelapse import _trackastra_track_cells

    blank = np.zeros((3, 32, 32), dtype=np.uint16)
    _install_stub_trackastra(monkeypatch, relabelled=blank)
    src = tmp_path / "run" / "batch"; src.parent.mkdir(parents=True, exist_ok=True)
    out = _trackastra_track_cells(src=str(src), name="b", batch_filenames=[],
                                  object_type="cell", masks=blank)
    df = pd.read_csv(tmp_path / "run" / "tracks" / "trackastra_tracks_cell_b.csv")
    assert df.empty
    assert len(out) == 3


def test_remove_transient_drops_tracks_absent_from_some_frame(monkeypatch, tmp_path, capsys):
    """A track seen in 2 of 3 frames goes when timelapse_remove_transient is on."""
    from spacr.timelapse import _trackastra_track_cells

    tracked = np.zeros((3, 48, 48), dtype=np.uint16)
    for t in range(3):
        _disc(tracked[t], cy=12, cx=10 + 3 * t, label=1)     # all three frames
    for t in range(2):
        _disc(tracked[t], cy=32, cx=10 + 3 * t, label=2)     # only two
    _install_stub_trackastra(monkeypatch, relabelled=tracked)

    src = tmp_path / "run" / "batch"; src.parent.mkdir(parents=True, exist_ok=True)
    _trackastra_track_cells(src=str(src), name="b", batch_filenames=[],
                            object_type="cell", masks=tracked,
                            timelapse_remove_transient=True)
    df = pd.read_csv(tmp_path / "run" / "tracks" / "trackastra_tracks_cell_b.csv")
    assert df["track_id"].unique().tolist() == [1]
    assert "Removed" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# wiring
# ---------------------------------------------------------------------------

def test_trackastra_is_the_default_timelapse_mode():
    """It needs no tuning, so it is the sensible default over trackpy."""
    from spacr.settings import set_default_settings_preprocess_generate_masks as defaults
    s = defaults({"src": "x"})
    assert s["timelapse_mode"] == "trackastra"
    assert s["trackastra_model"] == "general_2d"
    assert s["trackastra_linking"] == "greedy"


def test_trackastra_settings_are_typed_and_categorised():
    from spacr.settings import expected_types, categories
    for k in ("trackastra_model", "trackastra_linking"):
        assert expected_types.get(k) is str
        assert k in categories["Timelapse"]


def test_object_dispatch_imports_the_trackastra_backend():
    """object.py must be able to reach it, or the mode is unreachable."""
    import inspect
    import spacr.object as O
    src = inspect.getsource(O)
    assert "_trackastra_track_cells" in src
    assert "timelapse_mode == 'trackastra'" in src

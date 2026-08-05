"""Live previews for the Timelapse and Motility Assay modules.

Both panels follow the Mask live preview's contract — a standalone
``QWidget``, a ``QThread`` worker that emits results over signals, a
propagate callback into the main settings panel, and a ``build_*_card``
factory — so the suite pins the properties that contract exists to
guarantee:

* the expensive half is **cached**: a tracking change must never
  re-segment, and a metric change must never re-read the merged arrays
  (both asserted against real call counters);
* a sequence is read **lazily** — opening a 40-frame stack decodes nothing,
  and previewing 4 frames of it decodes 4;
* the completion handler runs on the **GUI thread**, asserted by comparing
  ``QThread.currentThread()`` against the widget's own thread;
* the worker **emits** its failures instead of raising out of ``run()``;
* an unreadable input, and a missing optional tracking backend, are
  reported **inline** — never as a traceback and never as a modal dialog
  (the autouse fixture below turns any dialog into a red test);
* velocities always carry a **unit**, and an unknown calibration is stated
  rather than silently defaulted.

Nothing here runs Cellpose, trackpy, btrack, trackastra or ultrack:
segmentation is stubbed with a counting fake, the optional backends are
exercised through injected stub modules, and linking uses spaCR's own pure
numpy IoU linker on 32-pixel synthetic frames.
"""
from __future__ import annotations

import os
import sys
import types

import numpy as np
import pytest

from PySide6.QtCore import QThread


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _no_modal_dialogs(monkeypatch):
    """Blow up loudly if any code path under test opens a modal dialog.

    A QMessageBox in a headless run hangs the suite forever; errors in these
    panels must land in the inline status label instead.
    """
    from PySide6.QtWidgets import QDialog, QFileDialog, QMessageBox

    def _boom(*_a, **_k):
        raise AssertionError(
            "a modal dialog was opened — errors must be reported inline")

    for name in ("about", "critical", "information", "question", "warning"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(_boom))
    for name in ("exec", "exec_", "open", "show"):
        monkeypatch.setattr(QMessageBox, name, _boom, raising=False)
    monkeypatch.setattr(QDialog, "exec", _boom, raising=False)
    for name in ("getOpenFileName", "getSaveFileName", "getExistingDirectory"):
        monkeypatch.setattr(QFileDialog, name, staticmethod(_boom))
    yield


H = W = 32
N_FRAMES = 4


def _disc(mask, cy, cx, r, label):
    yy, xx = np.ogrid[:mask.shape[0], :mask.shape[1]]
    mask[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = label
    return mask


def _synthetic_masks(n_frames=N_FRAMES, drift=3):
    """Two discs walking steadily right — a clean, linkable time series."""
    stack = np.zeros((n_frames, H, W), np.int32)
    for t in range(n_frames):
        _disc(stack[t], 8, 6 + drift * t, 3, 1)
        _disc(stack[t], 22, 6 + drift * t, 3, 2)
    return stack


@pytest.fixture
def frame_dir(tmp_path):
    """A folder of per-frame TIFFs — the commonest timelapse input."""
    import tifffile
    d = tmp_path / "frames"
    d.mkdir()
    masks = _synthetic_masks()
    for t in range(N_FRAMES):
        img = (masks[t] > 0).astype(np.uint16) * 900 + 40
        tifffile.imwrite(str(d / f"field_t{t}.tif"), img)
    return str(d)


@pytest.fixture
def mask_dir(tmp_path):
    """A matching folder of ready-made label images."""
    import tifffile
    d = tmp_path / "labels"
    d.mkdir()
    masks = _synthetic_masks()
    for t in range(N_FRAMES):
        tifffile.imwrite(str(d / f"field_t{t}.tif"), masks[t].astype(np.uint16))
    return str(d)


@pytest.fixture
def big_stack(tmp_path):
    """A 40-frame .npy stack, used to prove nothing is read eagerly."""
    p = tmp_path / "movie.npy"
    np.save(str(p), np.zeros((40, H, W), np.uint16))
    return str(p)


@pytest.fixture
def counted_segmentation(monkeypatch):
    """Replace Cellpose with a counting stub that thresholds the frame.

    The real :func:`segment_sequence` is left alone so the cache/threading
    path under test is the production one; only the per-frame Cellpose call
    is swapped, which is exactly the boundary the re-segmentation counter
    needs to sit on.
    """
    from skimage.measure import label as sk_label
    from spacr.qt.widgets import timelapse_preview as TP

    calls = {"n": 0}

    def _fake(image, params):
        calls["n"] += 1
        plane = TP.frame_channel(np.asarray(image), int(params.get("channel", 0)))
        return sk_label(plane > 100).astype(np.int32)

    monkeypatch.setattr(TP, "segment_frame", _fake)
    return calls


@pytest.fixture
def plate_dir(tmp_path):
    """A plate folder of merged arrays: 4 channels + cell mask + pathogen mask.

    Object 1 drifts steadily and carries a pathogen; object 2 drifts too and
    does not. Object 3 exists for two frames only, so the minimum-length
    cutoff has something real to exclude.
    """
    src = tmp_path / "plate1"
    merged = src / "merged"
    merged.mkdir(parents=True)
    for t in range(N_FRAMES):
        arr = np.zeros((6, H, W), np.float32)
        for c in range(4):
            arr[c] = 10 * (c + 1)
        cell = np.zeros((H, W), np.int32)
        _disc(cell, 8, 6 + 3 * t, 3, 1)
        _disc(cell, 22, 6 + 2 * t, 3, 2)
        if t < 2:
            _disc(cell, 28, 28, 2, 3)
        arr[4] = cell
        pathogen = np.zeros((H, W), np.int32)
        _disc(pathogen, 8, 6 + 3 * t, 1, 1)      # inside object 1 only
        arr[5] = pathogen
        np.save(str(merged / f"plate1_A01_1_{t}.npy"), arr)
    return str(src)


# ---------------------------------------------------------------------------
# Timelapse — lazy sequence access
# ---------------------------------------------------------------------------

def test_open_stack_reads_no_frames(big_stack):
    from spacr.qt.widgets.timelapse_preview import FrameSequence
    seq = FrameSequence.open(big_stack, max_frames=4)
    assert seq.n_available == 40
    assert len(seq) == 4
    assert seq.truncated is True
    assert seq.read_count == 0, "opening a sequence must not decode frames"


def test_frames_are_decoded_one_at_a_time(big_stack):
    from spacr.qt.widgets.timelapse_preview import FrameSequence
    seq = FrameSequence.open(big_stack, max_frames=4)
    for i in range(4):
        seq.frame(i)
    assert seq.read_count == 4, "read more frames than were asked for"
    seq.frame(0)      # still cached
    assert seq.read_count == 4


def test_lru_bounds_the_cache(big_stack):
    from spacr.qt.widgets.timelapse_preview import FrameSequence
    seq = FrameSequence.open(big_stack, max_frames=12)
    seq._cache_size = 2
    for i in range(6):
        seq.frame(i)
    assert len(seq._cache) <= 2, "the frame cache is unbounded"


def test_directory_and_tiff_stack_both_open(frame_dir, tmp_path):
    import tifffile
    from spacr.qt.widgets.timelapse_preview import FrameSequence
    seq = FrameSequence.open(frame_dir, max_frames=10)
    assert len(seq) == N_FRAMES
    assert seq.frame(0).shape == (H, W)

    p = tmp_path / "stack.tif"
    tifffile.imwrite(str(p), _synthetic_masks().astype(np.uint16))
    tif = FrameSequence.open(str(p), max_frames=10)
    assert len(tif) == N_FRAMES and tif.frame(2).shape == (H, W)


def test_frames_sort_naturally(tmp_path):
    """f_10 must come after f_2 — a lexical sort scrambles the time axis."""
    import tifffile
    from spacr.qt.widgets.timelapse_preview import FrameSequence
    d = tmp_path / "nat"
    d.mkdir()
    for t in (1, 2, 10):
        tifffile.imwrite(str(d / f"f_{t}.tif"),
                         np.full((H, W), t, np.uint16))
    seq = FrameSequence.open(str(d), max_frames=10)
    assert [int(seq.frame(i)[0, 0]) for i in range(3)] == [1, 2, 10]


@pytest.mark.parametrize("maker,msg", [
    (lambda p: p / "missing", "No such file"),
    (lambda p: p / "empty", "no"),
])
def test_bad_sequence_paths_raise_readable_errors(tmp_path, maker, msg):
    from spacr.qt.widgets.timelapse_preview import FrameSequence
    target = maker(tmp_path)
    if target.name == "empty":
        target.mkdir()
    with pytest.raises((FileNotFoundError, ValueError)) as exc:
        FrameSequence.open(str(target))
    assert msg.lower() in str(exc.value).lower()


def test_single_frame_input_is_refused(tmp_path):
    from spacr.qt.widgets.timelapse_preview import FrameSequence
    p = tmp_path / "one.npy"
    np.save(str(p), np.zeros((1, H, W), np.uint16))
    with pytest.raises(ValueError, match="at least two"):
        FrameSequence.open(str(p))


def test_channel_axis_is_detected_in_both_orientations():
    from spacr.qt.widgets.timelapse_preview import frame_channel
    last = np.zeros((H, W, 3), np.uint16)
    last[..., 1] = 7
    assert frame_channel(last, 1).shape == (H, W)
    assert int(frame_channel(last, 1).max()) == 7
    first = np.moveaxis(last, -1, 0)
    assert frame_channel(first, 1).shape == (H, W)
    assert int(frame_channel(first, 1).max()) == 7


# ---------------------------------------------------------------------------
# Timelapse — linking + indicators
# ---------------------------------------------------------------------------

def test_iou_linking_produces_stable_track_ids():
    from spacr.qt.widgets.timelapse_preview import link_tracks
    tracks = link_tracks(_synthetic_masks(), mode="iou", iou_threshold=0.1)
    assert set(tracks.columns) >= {"frame", "original_label", "track_id",
                                   "x", "y"}
    assert tracks["track_id"].nunique() == 2, "two objects, two tracks"
    assert tracks.groupby("track_id")["frame"].nunique().min() == N_FRAMES


def test_stats_report_length_shortness_and_fragmentation():
    from spacr.qt.widgets.timelapse_preview import link_tracks, track_stats
    # A gap in the middle of object 2 fragments it into two tracks.
    masks = _synthetic_masks()
    masks[2][masks[2] == 2] = 0
    tracks = link_tracks(masks, mode="iou", iou_threshold=0.1)
    st = track_stats(tracks, n_frames=N_FRAMES, min_length=4,
                     displacement_limit=50.0)
    assert st.n_tracks == 3, "the interrupted object should split in two"
    assert st.n_short >= 1
    assert st.starts_after_first >= 1
    assert st.ends_before_last >= 1
    assert st.fragmentation_events == (st.starts_after_first
                                       + st.ends_before_last)
    text = st.summary()
    for word in ("tracks", "mean length", "shorter than", "fragmentation",
                 "swap risk"):
        assert word in text


def test_suspicious_jump_indicator_follows_the_displacement_limit():
    """The swap indicator is the count of steps longer than the linking radius."""
    import pandas as pd
    from spacr.qt.widgets.timelapse_preview import track_stats
    tracks = pd.DataFrame({
        "frame": [0, 1, 2], "track_id": [1, 1, 1],
        "x": [0.0, 2.0, 60.0], "y": [0.0, 0.0, 0.0],
        "original_label": [1, 1, 1],
    })
    assert track_stats(tracks, 3, 2, displacement_limit=50.0).suspicious_jumps == 1
    assert track_stats(tracks, 3, 2, displacement_limit=100.0).suspicious_jumps == 0
    assert track_stats(tracks, 3, 2, displacement_limit=1.0).suspicious_jumps == 2
    assert track_stats(tracks, 3, 2, displacement_limit=50.0).max_step == 58.0


def test_stats_are_empty_safe():
    import pandas as pd
    from spacr.qt.widgets.timelapse_preview import track_stats
    st = track_stats(pd.DataFrame(), n_frames=0)
    assert st.n_tracks == 0 and st.fragmentation_events == 0
    assert track_stats(None, 0).n_tracks == 0


def test_masks_are_recoloured_by_track_id():
    """An object keeps its colour across frames only if it keeps its id."""
    from spacr.qt.widgets.timelapse_preview import (
        link_tracks, relabel_by_track, render_frame, track_colour)
    masks = _synthetic_masks()
    tracks = link_tracks(masks, mode="iou")
    tracked = relabel_by_track(masks, tracks)
    assert tracked.shape == masks.shape
    ids = set(np.unique(tracked)) - {0}
    assert ids == set(tracks["track_id"].unique())

    rgb = render_frame(masks[1] > 0, labels=tracked[1], tracks=tracks,
                       frame=1, normalise=False)
    assert rgb.shape == (H, W, 3) and rgb.dtype == np.uint8
    painted = {tuple(c) for c in rgb.reshape(-1, 3)}
    assert track_colour(sorted(ids)[0]) in painted


def test_relabel_is_empty_safe():
    from spacr.qt.widgets.timelapse_preview import relabel_by_track
    masks = _synthetic_masks()
    assert relabel_by_track(masks, None).max() == 0


def test_render_frame_without_masks_still_returns_an_image():
    from spacr.qt.widgets.timelapse_preview import render_frame
    rgb = render_frame(np.zeros((H, W), np.uint16))
    assert rgb.shape == (H, W, 3)


# ---------------------------------------------------------------------------
# Timelapse — optional backends
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode,pkg", [("trackastra", "trackastra"),
                                      ("ultrack", "ultrack")])
def test_missing_optional_backend_is_named_not_traced(monkeypatch, mode, pkg):
    import importlib.util
    from spacr.qt.widgets import timelapse_preview as TP
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    ok, why = TP.backend_available(mode)
    assert ok is False
    assert pkg in why and "pip install" in why
    assert "Traceback" not in why
    with pytest.raises(TP.TrackerUnavailable) as exc:
        TP.link_tracks(_synthetic_masks(), mode=mode)
    assert pkg in str(exc.value)


@pytest.mark.parametrize("mode,pkg", [("trackastra", "trackastra"),
                                      ("ultrack", "ultrack")])
def test_missing_backend_lands_in_the_status_label(qtbot, monkeypatch, frame_dir,
                                                   mode, pkg):
    import importlib.util
    from spacr.qt.widgets import timelapse_preview as TP
    panel = TP.TimelapsePreviewPanel()
    qtbot.addWidget(panel)
    assert panel.load_sequence(frame_dir) is True
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    panel._mode_box.setCurrentText(mode)
    panel.run_preview()          # must not raise, must not open a dialog
    assert pkg in panel._status.text()
    assert "pip install" in panel._status.text()


def test_present_optional_backend_is_actually_used(monkeypatch):
    """With trackastra importable, the preview links through it.

    The package is injected as a stub so the path is covered offline and in
    milliseconds — the point is that the branch is wired, not that the real
    transformer works.
    """
    from spacr.qt.widgets import timelapse_preview as TP
    masks = _synthetic_masks()
    calls = {"n": 0}

    class _Model:
        @staticmethod
        def from_pretrained(name, device=None):
            return _Model()

        def track(self, imgs, msk, mode="greedy"):
            calls["n"] += 1
            return {"graph": True}

    import importlib.machinery
    root = types.ModuleType("trackastra")
    model_mod = types.ModuleType("trackastra.model")
    model_mod.Trackastra = _Model
    tracking_mod = types.ModuleType("trackastra.tracking")
    tracking_mod.graph_to_ctc = lambda g, m, outdir=None: (None, masks)
    root.model = model_mod
    root.tracking = tracking_mod
    for name, mod in (("trackastra", root),
                      ("trackastra.model", model_mod),
                      ("trackastra.tracking", tracking_mod)):
        # find_spec() insists on a real __spec__ for an already-imported
        # module, so the stub carries one.
        mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
        monkeypatch.setitem(sys.modules, name, mod)

    tracks = TP.link_tracks(masks, mode="trackastra")
    assert calls["n"] == 1
    assert tracks["track_id"].nunique() == 2


def test_unknown_mode_is_rejected_without_import():
    from spacr.qt.widgets.timelapse_preview import backend_available
    ok, why = backend_available("banana")
    assert ok is False and "banana" in why


def test_link_tracks_validates_its_input():
    from spacr.qt.widgets.timelapse_preview import link_tracks
    with pytest.raises(ValueError, match="T, H, W"):
        link_tracks(np.zeros((H, W), np.int32))
    with pytest.raises(ValueError, match="at least two"):
        link_tracks(np.zeros((1, H, W), np.int32))


# ---------------------------------------------------------------------------
# Timelapse — the panel
# ---------------------------------------------------------------------------

def test_panel_builds_offscreen(qtbot):
    from spacr.qt.widgets.timelapse_preview import TimelapsePreviewPanel
    panel = TimelapsePreviewPanel()
    qtbot.addWidget(panel)
    assert panel._frame_slider.maximum() == 0
    assert panel._play_btn.isEnabled() is False
    assert "Load a sequence" in panel._stats_label.text()


def test_card_factory_returns_panel_and_card(qtbot):
    from spacr.qt.widgets.timelapse_preview import (
        TimelapsePreviewPanel, build_timelapse_preview_card)
    panel, card = build_timelapse_preview_card(object())
    qtbot.addWidget(card)
    assert isinstance(panel, TimelapsePreviewPanel)
    assert card.minimumHeight() >= 300


def test_unreadable_input_reports_inline(qtbot, tmp_path):
    from spacr.qt.widgets.timelapse_preview import TimelapsePreviewPanel
    panel = TimelapsePreviewPanel()
    qtbot.addWidget(panel)
    assert panel.load_sequence(str(tmp_path / "nope")) is False
    assert "Load failed" in panel._status.text()
    assert panel._sequence is None
    assert panel.load_masks(str(tmp_path / "nope")) is False
    assert "Mask load failed" in panel._status.text()


def test_run_without_a_sequence_says_so(qtbot):
    from spacr.qt.widgets.timelapse_preview import TimelapsePreviewPanel
    panel = TimelapsePreviewPanel()
    qtbot.addWidget(panel)
    panel.run_preview()
    assert "Load a sequence" in panel._status.text()


def test_run_preview_segments_then_links(qtbot, frame_dir, counted_segmentation):
    from spacr.qt.widgets.timelapse_preview import TimelapsePreviewPanel
    panel = TimelapsePreviewPanel()
    qtbot.addWidget(panel)
    panel._mode_box.setCurrentText("iou")
    assert panel.load_sequence(frame_dir) is True
    with qtbot.waitSignal(panel.preview_ready, timeout=8000) as sig:
        panel.run_preview()
    stats = sig.args[0]
    assert stats is not None and stats.n_tracks == 2
    assert counted_segmentation["n"] == N_FRAMES
    assert panel._tracked is not None
    assert "Masks built" in panel._status.text()
    assert "without ground truth" in panel._stats_label.text()


def test_tracking_change_never_resegments(qtbot, frame_dir, counted_segmentation):
    """The whole point of the cache: a tracker knob must be free."""
    from spacr.qt.widgets.timelapse_preview import TimelapsePreviewPanel
    panel = TimelapsePreviewPanel()
    qtbot.addWidget(panel)
    panel._mode_box.setCurrentText("iou")
    panel.load_sequence(frame_dir)
    with qtbot.waitSignal(panel.preview_ready, timeout=8000):
        panel.run_preview()
    after_first = counted_segmentation["n"]
    assert after_first == N_FRAMES

    with qtbot.waitSignal(panel.preview_ready, timeout=8000):
        panel._displacement.setValue(120.0)
    assert counted_segmentation["n"] == after_first, \
        "changing a tracking setting re-segmented"
    assert "Re-linked (cached masks)" in panel._status.text()

    with qtbot.waitSignal(panel.preview_ready, timeout=8000):
        panel._iou.setValue(0.25)
    assert counted_segmentation["n"] == after_first


def test_segmentation_change_does_resegment(qtbot, frame_dir, counted_segmentation):
    """The cache key must move when a label image could change."""
    from spacr.qt.widgets.timelapse_preview import TimelapsePreviewPanel
    panel = TimelapsePreviewPanel()
    qtbot.addWidget(panel)
    panel.load_sequence(frame_dir)
    with qtbot.waitSignal(panel.preview_ready, timeout=8000):
        panel.run_preview()
    before = counted_segmentation["n"]
    panel._flow.setValue(0.9)                # different signature
    with qtbot.waitSignal(panel.preview_ready, timeout=8000):
        panel.run_preview()
    assert counted_segmentation["n"] == before + N_FRAMES


def test_min_length_rescoring_neither_relinks_nor_resegments(
        qtbot, frame_dir, counted_segmentation, monkeypatch):
    from spacr.qt.widgets import timelapse_preview as TP
    panel = TP.TimelapsePreviewPanel()
    qtbot.addWidget(panel)
    panel._mode_box.setCurrentText("iou")
    panel.load_sequence(frame_dir)
    with qtbot.waitSignal(panel.preview_ready, timeout=8000):
        panel.run_preview()
    seg_before = counted_segmentation["n"]

    links = {"n": 0}
    real = TP.link_tracks

    def _counting(*a, **k):
        links["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(TP, "link_tracks", _counting)
    with qtbot.waitSignal(panel.preview_ready, timeout=8000):
        panel._min_len.setValue(N_FRAMES + 2)
    assert links["n"] == 0, "re-scoring must not re-link"
    assert counted_segmentation["n"] == seg_before
    assert panel._stats.n_short == panel._stats.n_tracks


def test_loaded_masks_skip_segmentation_entirely(qtbot, frame_dir, mask_dir,
                                                 counted_segmentation):
    from spacr.qt.widgets.timelapse_preview import TimelapsePreviewPanel
    panel = TimelapsePreviewPanel()
    qtbot.addWidget(panel)
    panel._mode_box.setCurrentText("iou")
    panel.load_sequence(frame_dir)
    assert panel.load_masks(mask_dir) is True
    with qtbot.waitSignal(panel.preview_ready, timeout=8000) as sig:
        panel.run_preview()
    assert counted_segmentation["n"] == 0
    assert sig.args[0].n_tracks == 2


def test_completion_handler_runs_on_the_gui_thread(qtbot, frame_dir,
                                                   counted_segmentation):
    from spacr.qt.widgets.timelapse_preview import TimelapsePreviewPanel
    panel = TimelapsePreviewPanel()
    qtbot.addWidget(panel)
    seen = {}

    def _record(_stats):
        seen["thread"] = QThread.currentThread()

    panel.preview_ready.connect(_record)
    panel.load_sequence(frame_dir)
    with qtbot.waitSignal(panel.preview_ready, timeout=8000):
        panel.run_preview()
    assert seen["thread"] is panel.thread(), \
        "the completion handler ran off the GUI thread"


def test_worker_emits_errors_instead_of_raising(qtbot):
    """``run()`` must never let an exception escape a Qt thread."""
    from spacr.qt.widgets.timelapse_preview import (
        TimelapseRequest, _TimelapseWorker)
    worker = _TimelapseWorker(TimelapseRequest())   # no sequence at all
    got = []
    worker.finished_result.connect(lambda res, err: got.append((res, err)))
    with qtbot.waitSignal(worker.finished, timeout=5000):
        worker.start()
    qtbot.waitUntil(lambda: bool(got), timeout=5000)
    result, err = got[0]
    assert result is None
    assert "Load a sequence" in err


def test_panel_reports_a_worker_failure_inline(qtbot, frame_dir, monkeypatch):
    from spacr.qt.widgets import timelapse_preview as TP
    panel = TP.TimelapsePreviewPanel()
    qtbot.addWidget(panel)
    panel.load_sequence(frame_dir)
    monkeypatch.setattr(TP, "segment_sequence",
                        lambda *a, **k: (_ for _ in ()).throw(
                            RuntimeError("cellpose exploded")))
    with qtbot.waitSignal(panel.preview_ready, timeout=8000) as sig:
        panel.run_preview()
    assert sig.args[0] is None
    assert "cellpose exploded" in panel._status.text()


def test_relink_without_cached_masks_says_so(qtbot, frame_dir):
    from spacr.qt.widgets.timelapse_preview import TimelapsePreviewPanel
    panel = TimelapsePreviewPanel()
    qtbot.addWidget(panel)
    panel.load_sequence(frame_dir)
    panel.relink()
    assert "hit Run preview" in panel._status.text()


def test_scrubbing_renders_each_frame(qtbot, frame_dir, counted_segmentation):
    from spacr.qt.widgets.timelapse_preview import TimelapsePreviewPanel
    panel = TimelapsePreviewPanel()
    qtbot.addWidget(panel)
    panel.load_sequence(frame_dir)
    with qtbot.waitSignal(panel.preview_ready, timeout=8000):
        panel.run_preview()
    assert panel._frame_slider.maximum() == N_FRAMES - 1
    for t in range(N_FRAMES):
        panel._frame_slider.setValue(t)
        assert panel._frame_label.text() == f"{t + 1}/{N_FRAMES}"


def test_playback_advances_wraps_and_pauses(qtbot, frame_dir):
    from PySide6.QtCore import Qt
    from spacr.qt.widgets.timelapse_preview import TimelapsePreviewPanel

    panel = TimelapsePreviewPanel()
    qtbot.addWidget(panel)
    assert panel.load_sequence(frame_dir) is True
    assert panel._play_btn.isEnabled() is True

    panel._frame_slider.setValue(panel._frame_slider.maximum())
    panel._advance_frame()
    assert panel._frame_slider.value() == 0

    panel._play_fps.setValue(30)
    qtbot.mouseClick(panel._play_btn, Qt.LeftButton)
    assert panel._play_timer.isActive()
    assert panel._play_btn.text() == "Pause"
    qtbot.waitUntil(lambda: panel._frame_slider.value() > 0, timeout=1000)

    qtbot.mouseClick(panel._play_btn, Qt.LeftButton)
    assert panel._play_timer.isActive() is False
    assert panel._play_btn.text() == "Play"


def test_close_stops_timelapse_playback(qtbot, frame_dir):
    from spacr.qt.widgets.timelapse_preview import TimelapsePreviewPanel

    panel = TimelapsePreviewPanel()
    qtbot.addWidget(panel)
    panel.load_sequence(frame_dir)
    panel._toggle_playback()
    assert panel._play_timer.isActive()
    panel.close()
    assert panel._play_timer.isActive() is False


def test_remove_transient_keeps_only_full_length_tracks(
        qtbot, tmp_path, counted_segmentation, monkeypatch):
    import tifffile
    from spacr.qt.widgets import timelapse_preview as TP
    masks = _synthetic_masks()
    masks[3][masks[3] == 2] = 0          # object 2 vanishes at the end
    d = tmp_path / "gappy"
    d.mkdir()
    for t in range(N_FRAMES):
        tifffile.imwrite(str(d / f"f_{t}.tif"), masks[t].astype(np.uint16))

    panel = TP.TimelapsePreviewPanel()
    qtbot.addWidget(panel)
    panel._mode_box.setCurrentText("iou")
    panel.load_sequence(str(d))
    panel.load_masks(str(d))
    with qtbot.waitSignal(panel.preview_ready, timeout=8000) as sig:
        panel.run_preview()
    assert sig.args[0].n_tracks == 2
    with qtbot.waitSignal(panel.preview_ready, timeout=8000) as sig2:
        panel._remove_transient.setChecked(True)
    assert sig2.args[0].n_tracks == 1


def test_propagate_callback_fires_with_tuned_values(qtbot, frame_dir):
    from spacr.qt.widgets.timelapse_preview import TimelapsePreviewPanel
    panel = TimelapsePreviewPanel()
    qtbot.addWidget(panel)
    captured = {}
    panel.set_propagate_callback(captured.update)

    panel._mode_box.setCurrentText("trackpy")
    panel._displacement.setValue(77.0)
    panel._memory.setValue(5)
    panel._object_box.setCurrentText("nucleus")
    panel._channel.setValue(2)
    panel._propagate_btn.setChecked(True)      # pushes immediately

    assert captured["timelapse_mode"] == "trackpy"
    assert captured["timelapse_displacement"] == 77
    assert captured["timelapse_memory"] == 5
    assert captured["timelapse_objects"] == ["nucleus"]
    assert captured["nucleus_channel"] == 2


@pytest.mark.parametrize("obj", ["cell", "nucleus", "pathogen"])
def test_propagate_keys_are_real_timelapse_settings(qtbot, obj):
    """Every propagated key must exist in the module's settings dict."""
    import spacr.settings as S
    from spacr.qt.widgets.timelapse_preview import TimelapsePreviewPanel
    panel = TimelapsePreviewPanel()
    qtbot.addWidget(panel)
    panel._object_box.setCurrentText(obj)
    defaults = S.get_timelapse_settings(settings={})
    unknown = [k for k in panel.settings_for_propagation()
               if k not in defaults]
    assert not unknown, f"propagating keys the module has no setting for: {unknown}"


def test_propagate_failure_is_swallowed(qtbot):
    """A dead settings panel costs that one push and nothing else.

    The point is not that no exception escaped — it is that the preview is
    still wired up afterwards, so re-opening the settings screen and
    pushing again works.
    """
    from spacr.qt.widgets.timelapse_preview import TimelapsePreviewPanel
    panel = TimelapsePreviewPanel()
    qtbot.addWidget(panel)

    panel._mode_box.setCurrentText("trackpy")
    panel._displacement.setValue(41.0)

    seen_by_broken = []

    def _explode(d):
        seen_by_broken.append(dict(d))
        raise RuntimeError("settings panel is gone")

    panel.set_propagate_callback(_explode)
    panel.propagate_settings()          # must not raise

    # It really was called: a propagate that quietly skipped the callback
    # would also "not raise", and would be a different bug.
    assert len(seen_by_broken) == 1
    assert seen_by_broken[0]["timelapse_displacement"] == 41

    # Contrast: a working callback registered after the failure still gets
    # the panel's real, current settings.
    captured = {}
    panel.set_propagate_callback(captured.update)
    panel._memory.setValue(4)
    panel._object_box.setCurrentText("nucleus")
    panel.propagate_settings()

    assert captured["timelapse_mode"] == "trackpy"
    assert captured["timelapse_displacement"] == 41
    assert captured["timelapse_memory"] == 4
    assert captured["timelapse_objects"] == ["nucleus"]
    assert "nucleus_channel" in captured
    # ... and the broken callback is gone, not still on a listener list.
    assert len(seen_by_broken) == 1


def test_apply_settings_seeds_the_panel(qtbot):
    import spacr.settings as S
    from spacr.qt.widgets.timelapse_preview import TimelapsePreviewPanel
    panel = TimelapsePreviewPanel()
    qtbot.addWidget(panel)
    s = S.get_timelapse_settings(settings={})
    s.update({"timelapse_mode": "iou", "timelapse_displacement": 33,
              "timelapse_memory": 7, "timelapse_objects": ["nucleus"],
              "nucleus_channel": 1, "timelapse_remove_transient": True})
    panel.apply_settings(s)
    p = panel.current_params()
    assert p["mode"] == "iou" and p["displacement"] == 33 and p["memory"] == 7
    assert p["object"] == "nucleus" and p["channel"] == 1
    assert panel._remove_transient.isChecked() is True
    panel.apply_settings({})            # none-safe


# ---------------------------------------------------------------------------
# Motility — units
# ---------------------------------------------------------------------------

def test_uncalibrated_velocity_is_px_per_frame_and_says_so():
    from spacr.qt.widgets.motility_preview import Calibration
    cal = Calibration()
    assert cal.known is False
    assert cal.unit == "px/frame"
    assert cal.factor == 1.0
    caveat = cal.caveat()
    assert "px/frame" in caveat and "NOT" in caveat
    assert "pixels_per_um" in caveat and "seconds_per_frame" in caveat


@pytest.mark.parametrize("ppu,spf", [(1.78, None), (None, 60.0), (0, 60.0)])
def test_half_a_calibration_is_still_uncalibrated(ppu, spf):
    from spacr.qt.widgets.motility_preview import Calibration
    cal = Calibration(pixels_per_um=ppu, seconds_per_frame=spf)
    assert cal.known is False and cal.unit == "px/frame"
    assert cal.caveat()


def test_calibrated_factor_matches_the_pipeline_formula():
    """The preview must report the same number a run would."""
    from spacr.qt.widgets.motility_preview import Calibration
    cal = Calibration(pixels_per_um=1.78, seconds_per_frame=60.0)
    assert cal.known is True
    assert cal.unit == "µm/min"
    assert cal.factor == pytest.approx((1.0 / 1.78) * (60.0 / 60.0))
    assert not cal.caveat()


def test_track_metrics_carry_their_unit():
    import pandas as pd
    from spacr.qt.widgets.motility_preview import Calibration, track_metrics
    points = pd.DataFrame({
        "plateID": ["p"] * 3, "wellID": ["A01"] * 3, "fieldID": ["1"] * 3,
        "cellID": [1, 1, 1], "frame": [0, 1, 2],
        "x": [0.0, 3.0, 6.0], "y": [0.0, 4.0, 8.0],
        "area": [9, 9, 9], "infected": [False, False, False],
    })
    raw = track_metrics(points, Calibration(), min_length=2)
    assert raw.loc[0, "velocity_unit"] == "px/frame"
    assert raw.loc[0, "v_px_per_frame"] == pytest.approx(5.0)
    assert raw.loc[0, "velocity"] == pytest.approx(5.0)
    assert raw.loc[0, "straightness"] == pytest.approx(1.0)

    cal = Calibration(pixels_per_um=2.0, seconds_per_frame=30.0)
    conv = track_metrics(points, cal, min_length=2)
    assert conv.loc[0, "velocity_unit"] == "µm/min"
    assert conv.loc[0, "velocity"] == pytest.approx(5.0 * (1 / 2.0) * (60 / 30.0))


def test_two_point_tracks_are_flagged_short_not_trusted():
    """Straightness is exactly 1.0 on a two-point track — hence the cutoff."""
    import pandas as pd
    from spacr.qt.widgets.motility_preview import Calibration, track_metrics
    points = pd.DataFrame({
        "plateID": ["p"] * 2, "wellID": ["A01"] * 2, "fieldID": ["1"] * 2,
        "cellID": [1, 1], "frame": [0, 1], "x": [0.0, 10.0], "y": [0.0, 0.0],
        "area": [9, 9], "infected": [False, False],
    })
    t = track_metrics(points, Calibration(), min_length=3)
    assert t.loc[0, "straightness"] == pytest.approx(1.0)
    assert bool(t.loc[0, "too_short"]) is True
    assert bool(track_metrics(points, Calibration(), min_length=2)
                .loc[0, "too_short"]) is False


def test_track_metrics_is_empty_safe():
    import pandas as pd
    from spacr.qt.widgets.motility_preview import Calibration, track_metrics
    out = track_metrics(pd.DataFrame(), Calibration(), 3)
    assert out.empty and "velocity" in out.columns
    assert track_metrics(None, Calibration(), 3).empty


# ---------------------------------------------------------------------------
# Motility — QC + summary
# ---------------------------------------------------------------------------

def test_teleport_glitch_is_interpolated_and_impossible_track_dropped():
    import pandas as pd
    from spacr.qt.widgets.motility_preview import smooth_and_filter_tracks

    def _pts(cell_id, xs):
        return pd.DataFrame({
            "plateID": ["p"] * len(xs), "wellID": ["A01"] * len(xs),
            "fieldID": ["1"] * len(xs), "cellID": [cell_id] * len(xs),
            "frame": list(range(len(xs))), "x": xs,
            "y": [0.0] * len(xs), "area": [9] * len(xs),
            "infected": [False] * len(xs),
        })

    glitchy = _pts(1, [0.0, 500.0, 2.0, 3.0])       # one-frame teleport
    cleaned, fixed, dropped = smooth_and_filter_tracks(glitchy, 50.0)
    assert fixed == 1 and dropped == 0
    assert cleaned.loc[cleaned["frame"] == 1, "x"].iloc[0] == pytest.approx(1.0)

    runaway = _pts(2, [0.0, 1.0, 900.0, 901.0])     # a real bad link
    cleaned2, fixed2, dropped2 = smooth_and_filter_tracks(runaway, 50.0)
    assert dropped2 == 1 and cleaned2.empty


def test_smoothing_is_empty_safe():
    import pandas as pd
    from spacr.qt.widgets.motility_preview import smooth_and_filter_tracks
    out, fixed, dropped = smooth_and_filter_tracks(pd.DataFrame(), 50.0)
    assert out.empty and fixed == 0 and dropped == 0


def test_summary_states_the_unit_and_the_infection_split():
    import pandas as pd
    from spacr.qt.widgets.motility_preview import (
        Calibration, summarise, track_metrics)
    rows = []
    for cid, infected, step in ((1, True, 2.0), (2, False, 6.0)):
        for f in range(4):
            rows.append({"plateID": "p", "wellID": "A01", "fieldID": "1",
                         "cellID": cid, "frame": f, "x": step * f, "y": 0.0,
                         "area": 9, "infected": infected})
    tracks = track_metrics(pd.DataFrame(rows), Calibration(), 3)
    s = summarise(tracks, Calibration(), 3, 0.95)
    assert s.n_tracks == 2 and s.n_used == 2 and s.n_short == 0
    assert s.n_infected == 1 and s.n_uninfected == 1
    assert s.mean_velocity_infected == pytest.approx(2.0)
    assert s.mean_velocity_uninfected == pytest.approx(6.0)
    assert s.n_high_straightness == 2      # both are perfectly straight
    text = s.summary()
    assert "px/frame" in text and "infected" in text and "straightness" in text


def test_summary_is_empty_safe():
    import pandas as pd
    from spacr.qt.widgets.motility_preview import Calibration, summarise
    s = summarise(pd.DataFrame(), Calibration(), 3, 0.95)
    assert s.n_tracks == 0 and "n/a" in s.summary()


def test_figure_renders_to_an_rgb_array():
    import pandas as pd
    from spacr.qt.widgets.motility_preview import (
        Calibration, render_motility_figure, track_metrics)
    rows = [{"plateID": "p", "wellID": "A01", "fieldID": "1", "cellID": 1,
             "frame": f, "x": 2.0 * f, "y": 0.0, "area": 9, "infected": True}
            for f in range(4)]
    points = pd.DataFrame(rows)
    tracks = track_metrics(points, Calibration(), 3)
    rgb = render_motility_figure(points, tracks, Calibration(), 3, 0.95,
                                 width_px=400, height_px=200)
    assert rgb.ndim == 3 and rgb.shape[2] == 3 and rgb.dtype == np.uint8
    # And with nothing to draw, it still produces a canvas.
    blank = render_motility_figure(None, None, Calibration(), 3, 0.95,
                                   width_px=400, height_px=200)
    assert blank.shape == rgb.shape


# ---------------------------------------------------------------------------
# Motility — merged-array reading
# ---------------------------------------------------------------------------

def test_merged_dir_resolves_from_plate_or_merged_folder(plate_dir):
    from spacr.qt.widgets.motility_preview import resolve_merged_dir
    assert resolve_merged_dir(plate_dir).endswith("merged")
    inner = os.path.join(plate_dir, "merged")
    assert resolve_merged_dir(inner) == inner


def test_a_folder_without_merged_arrays_is_refused(tmp_path):
    from spacr.qt.widgets.motility_preview import (
        MotilityInputError, resolve_merged_dir)
    d = tmp_path / "empty"
    d.mkdir()
    with pytest.raises(MotilityInputError, match="merged"):
        resolve_merged_dir(str(d))
    with pytest.raises(MotilityInputError, match="No such folder"):
        resolve_merged_dir(str(tmp_path / "ghost"))


def test_groups_are_keyed_by_plate_well_field_and_time_sorted(plate_dir):
    from spacr.qt.widgets.motility_preview import group_merged_files
    groups = group_merged_files(os.path.join(plate_dir, "merged"))
    assert list(groups) == [("plate1", "A01", "1")]
    metas = groups[("plate1", "A01", "1")]
    assert [m["timeID"] for m in metas] == [0, 1, 2, 3]


@pytest.mark.parametrize("planes,n_channels,expected", [
    (5, 4, (4, None)),
    (6, 4, (4, 5)),
    (7, 4, (4, 6)),
    (3, 4, (2, None)),
])
def test_plane_layout_follows_the_documented_merged_layout(planes, n_channels,
                                                           expected):
    from spacr.qt.widgets.motility_preview import default_plane_layout
    assert default_plane_layout(planes, n_channels) == expected


def test_point_table_reads_centroids_and_infection(plate_dir):
    from spacr.qt.widgets.motility_preview import (
        build_point_table, group_merged_files)
    merged = os.path.join(plate_dir, "merged")
    metas = group_merged_files(merged)[("plate1", "A01", "1")]
    pts = build_point_table(merged, metas, n_channels=4, tracked_plane=4,
                            pathogen_plane=5, max_frames=10)
    assert set(pts.columns) >= {"cellID", "frame", "x", "y", "infected"}
    assert pts["frame"].nunique() == N_FRAMES
    assert set(pts["cellID"].unique()) == {1, 2, 3}
    assert bool(pts.loc[pts["cellID"] == 1, "infected"].all()) is True
    assert bool(pts.loc[pts["cellID"] == 2, "infected"].any()) is False
    # object 3 only exists for two frames — the short track the cutoff excludes
    assert int((pts["cellID"] == 3).sum()) == 2


def test_point_table_without_a_pathogen_plane_marks_all_uninfected(plate_dir):
    from spacr.qt.widgets.motility_preview import (
        build_point_table, group_merged_files)
    merged = os.path.join(plate_dir, "merged")
    metas = group_merged_files(merged)[("plate1", "A01", "1")]
    pts = build_point_table(merged, metas, 4, 4, None, max_frames=10)
    assert bool(pts["infected"].any()) is False


def test_point_table_reports_an_empty_mask_plane(tmp_path):
    """A wrong plane index is a user error, and must read as one."""
    from spacr.qt.widgets.motility_preview import (
        MotilityInputError, build_point_table, group_merged_files)
    merged = tmp_path / "plate2" / "merged"
    merged.mkdir(parents=True)
    for t in range(N_FRAMES):
        np.save(str(merged / f"plate2_A01_1_{t}.npy"),
                np.zeros((5, H, W), np.float32))
    metas = group_merged_files(str(merged))[("plate2", "A01", "1")]
    with pytest.raises(MotilityInputError, match="mask plane"):
        build_point_table(str(merged), metas, 4, 4, None, max_frames=10)


def test_point_table_honours_the_frame_cap(plate_dir):
    from spacr.qt.widgets.motility_preview import (
        build_point_table, group_merged_files)
    merged = os.path.join(plate_dir, "merged")
    metas = group_merged_files(merged)[("plate1", "A01", "1")]
    pts = build_point_table(merged, metas, 4, 4, 5, max_frames=2)
    assert pts["frame"].nunique() == 2


# ---------------------------------------------------------------------------
# Motility — the panel
# ---------------------------------------------------------------------------

def test_motility_panel_builds_offscreen(qtbot):
    from spacr.qt.widgets.motility_preview import MotilityPreviewPanel
    panel = MotilityPreviewPanel(threaded=False)
    qtbot.addWidget(panel)
    assert "px/frame" in panel._unit_label.text()
    assert panel.calibration().known is False


def test_motility_card_factory(qtbot):
    from spacr.qt.widgets.motility_preview import (
        MotilityPreviewPanel, build_motility_preview_card)
    panel, card = build_motility_preview_card(object())
    qtbot.addWidget(card)
    assert isinstance(panel, MotilityPreviewPanel)


def test_motility_unreadable_input_reports_inline(qtbot, tmp_path):
    from spacr.qt.widgets.motility_preview import MotilityPreviewPanel
    panel = MotilityPreviewPanel(threaded=False)
    qtbot.addWidget(panel)
    assert panel.load_folder(str(tmp_path / "nope")) is False
    assert "Load failed" in panel._status.text()
    panel.run_preview()
    assert "Load a plate folder" in panel._status.text()


def test_motility_load_detects_groups_and_planes(qtbot, plate_dir):
    from spacr.qt.widgets.motility_preview import MotilityPreviewPanel
    panel = MotilityPreviewPanel(threaded=False)
    qtbot.addWidget(panel)
    assert panel.load_folder(plate_dir) is True
    assert panel._group_box.count() == 1
    assert panel._tracked_plane.value() == 4
    assert panel._pathogen_plane.value() == 5


def test_motility_run_produces_a_summary(qtbot, plate_dir):
    from spacr.qt.widgets.motility_preview import MotilityPreviewPanel
    panel = MotilityPreviewPanel(threaded=False)
    qtbot.addWidget(panel)
    panel.load_folder(plate_dir)
    with qtbot.waitSignal(panel.preview_ready, timeout=8000) as sig:
        panel.run_preview()
    s = sig.args[0]
    assert s is not None
    assert s.n_tracks == 3 and s.n_used == 2 and s.n_short == 1
    assert s.n_infected == 1 and s.n_uninfected == 1
    assert s.unit == "px/frame"
    assert "px/frame" in panel._stats_label.text()
    assert "NOT µm/min" in panel._stats_label.text()
    assert panel._plot.pixmap() is not None


def test_metric_change_recomputes_without_rereading(qtbot, plate_dir, monkeypatch):
    """The cached point table is the whole reason metric knobs feel instant."""
    from spacr.qt.widgets import motility_preview as MP
    panel = MP.MotilityPreviewPanel()
    qtbot.addWidget(panel)
    panel.load_folder(plate_dir)
    with qtbot.waitSignal(panel.preview_ready, timeout=8000):
        panel.run_preview()

    reads = {"n": 0}
    real = MP.build_point_table

    def _counting(*a, **k):
        reads["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(MP, "build_point_table", _counting)
    with qtbot.waitSignal(panel.preview_ready, timeout=8000) as sig:
        panel._min_len.setValue(2)
    assert reads["n"] == 0, "a metric change re-read the merged arrays"
    assert sig.args[0].n_used == 3          # the 2-frame track now qualifies


def test_setting_the_calibration_live_converts_the_velocities(qtbot, plate_dir):
    from spacr.qt.widgets.motility_preview import MotilityPreviewPanel
    panel = MotilityPreviewPanel(threaded=False)
    qtbot.addWidget(panel)
    panel.load_folder(plate_dir)
    with qtbot.waitSignal(panel.preview_ready, timeout=8000) as raw:
        panel.run_preview()
    px_mean = raw.args[0].mean_velocity
    assert raw.args[0].unit == "px/frame"

    panel._pixels_per_um.setValue(4.0)
    with qtbot.waitSignal(panel.preview_ready, timeout=8000) as done:
        panel._seconds_per_frame.setValue(30.0)
    s = done.args[0]
    assert s.unit == "µm/min"
    assert s.mean_velocity == pytest.approx(px_mean * (1 / 4.0) * (60 / 30.0))
    assert s.mean_velocity != pytest.approx(px_mean)
    assert "µm/min" in panel._unit_label.text()
    assert "NOT" not in panel._stats_label.text()


def test_straightness_filter_drops_the_flagged_tracks(qtbot, plate_dir):
    from spacr.qt.widgets.motility_preview import MotilityPreviewPanel
    panel = MotilityPreviewPanel(threaded=False)
    qtbot.addWidget(panel)
    panel.load_folder(plate_dir)
    with qtbot.waitSignal(panel.preview_ready, timeout=8000) as before:
        panel.run_preview()
    assert before.args[0].n_used == 2       # both drift perfectly straight
    with qtbot.waitSignal(panel.preview_ready, timeout=8000) as after:
        panel._straightness_filter.setChecked(True)
    assert after.args[0].n_used == 0


def test_max_displacement_drops_impossible_tracks_live(qtbot, plate_dir):
    from spacr.qt.widgets.motility_preview import MotilityPreviewPanel
    panel = MotilityPreviewPanel(threaded=False)
    qtbot.addWidget(panel)
    panel.load_folder(plate_dir)
    with qtbot.waitSignal(panel.preview_ready, timeout=8000):
        panel.run_preview()
    with qtbot.waitSignal(panel.preview_ready, timeout=8000) as sig:
        panel._max_disp.setValue(2.5)       # object 1 steps 3 px per frame
    assert sig.args[0].tracks_dropped >= 1


def test_motility_completion_handler_runs_on_the_gui_thread(qtbot, plate_dir):
    from spacr.qt.widgets.motility_preview import MotilityPreviewPanel
    panel = MotilityPreviewPanel(threaded=False)
    qtbot.addWidget(panel)
    seen = {}
    panel.preview_ready.connect(
        lambda _s: seen.__setitem__("thread", QThread.currentThread()))
    panel.load_folder(plate_dir)
    with qtbot.waitSignal(panel.preview_ready, timeout=8000):
        panel.run_preview()
    assert seen["thread"] is panel.thread()


def test_motility_worker_emits_errors_instead_of_raising(qtbot):
    from spacr.qt.widgets.motility_preview import (
        MotilityRequest, _MotilityWorker)
    worker = _MotilityWorker(MotilityRequest(merged_dir="/definitely/not/here",
                                             metas=[{"filename": "x.npy"}]))
    got = []
    worker.finished_result.connect(lambda res, err: got.append((res, err)))
    with qtbot.waitSignal(worker.finished, timeout=5000):
        worker.start()
    qtbot.waitUntil(lambda: bool(got), timeout=5000)
    assert got[0][0] is None and got[0][1]


def test_motility_panel_reports_a_worker_failure_inline(qtbot, plate_dir,
                                                        monkeypatch):
    from spacr.qt.widgets import motility_preview as MP
    panel = MP.MotilityPreviewPanel()
    qtbot.addWidget(panel)
    panel.load_folder(plate_dir)
    monkeypatch.setattr(MP, "build_point_table",
                        lambda *a, **k: (_ for _ in ()).throw(
                            RuntimeError("disk on fire")))
    with qtbot.waitSignal(panel.preview_ready, timeout=8000) as sig:
        panel.run_preview()
    assert sig.args[0] is None
    assert "disk on fire" in panel._status.text()


def test_motility_propagation_never_invents_a_calibration(qtbot, plate_dir):
    from spacr.qt.widgets.motility_preview import MotilityPreviewPanel
    panel = MotilityPreviewPanel(threaded=False)
    qtbot.addWidget(panel)
    captured = {}
    panel.set_propagate_callback(captured.update)
    panel._max_disp.setValue(42.0)
    panel._propagate_btn.setChecked(True)
    assert captured["max_displacement"] == 42.0
    assert "pixels_per_um" not in captured, \
        "an unknown calibration must not be pushed into the run"
    assert "seconds_per_frame" not in captured

    panel._pixels_per_um.setValue(1.5)
    panel._seconds_per_frame.setValue(20.0)
    panel.propagate_settings()
    assert captured["pixels_per_um"] == 1.5
    assert captured["seconds_per_frame"] == 20.0


def test_motility_propagate_keys_are_real_settings(qtbot):
    import spacr.settings as S
    from spacr.qt.widgets.motility_preview import MotilityPreviewPanel
    panel = MotilityPreviewPanel(threaded=False)
    qtbot.addWidget(panel)
    panel._pixels_per_um.setValue(1.78)
    panel._seconds_per_frame.setValue(60.0)
    panel._pathogen_plane.setValue(5)
    defaults = S.get_automated_motility_assay_default_settings(settings={})
    unknown = [k for k in panel.settings_for_propagation()
               if k not in defaults]
    assert not unknown, f"unknown motility settings: {unknown}"


def test_motility_apply_settings_round_trips(qtbot):
    import spacr.settings as S
    from spacr.qt.widgets.motility_preview import MotilityPreviewPanel
    panel = MotilityPreviewPanel(threaded=False)
    qtbot.addWidget(panel)
    panel.apply_settings(
        S.get_automated_motility_assay_default_settings(settings={}))
    p = panel.current_params()
    assert p["max_displacement"] == 50.0
    assert p["calibrated"] is True and p["unit"] == "µm/min"
    panel.apply_settings({})            # none-safe


def test_tracked_object_moves_the_mask_plane(qtbot):
    from spacr.qt.widgets.motility_preview import MotilityPreviewPanel
    panel = MotilityPreviewPanel(threaded=False)
    qtbot.addWidget(panel)
    panel._n_channels.setValue(4)
    panel._tracked_object.setCurrentText("pathogen")
    assert panel._tracked_plane.value() == 6
    panel._tracked_object.setCurrentText("cell")
    assert panel._tracked_plane.value() == 4


def test_motility_propagate_failure_is_swallowed(qtbot):
    """Same contract as the timelapse panel: one lost push, panel intact."""
    from spacr.qt.widgets.motility_preview import MotilityPreviewPanel
    panel = MotilityPreviewPanel(threaded=False)
    qtbot.addWidget(panel)

    panel._max_disp.setValue(37.0)
    seen_by_broken = []

    def _explode(d):
        seen_by_broken.append(dict(d))
        raise RuntimeError("gone")

    panel.set_propagate_callback(_explode)
    panel.propagate_settings()          # must not raise

    assert len(seen_by_broken) == 1
    assert seen_by_broken[0]["max_displacement"] == 37.0

    # Contrast: a working callback registered afterwards still receives the
    # panel's real settings — including the value tuned before the failure.
    captured = {}
    panel.set_propagate_callback(captured.update)
    panel._tracked_object.setCurrentText("cell")
    panel._straightness.setValue(0.6)
    panel.propagate_settings()

    assert captured["tracked_object"] == "cell"
    assert captured["max_displacement"] == 37.0
    assert captured["straightness_threshold"] == pytest.approx(0.6)
    assert len(seen_by_broken) == 1


# ---------------------------------------------------------------------------
# Backends that are installed but must not actually run in a test
# ---------------------------------------------------------------------------

class _Evt:
    """Minimal stand-in for a QDragEvent — records what the handler did."""

    def __init__(self, mime):
        self._mime = mime
        self.accepted = False
        self.ignored = False

    def mimeData(self):
        return self._mime

    def acceptProposedAction(self):
        self.accepted = True

    def ignore(self):
        self.ignored = True


def _mime_for(*paths):
    from PySide6.QtCore import QMimeData, QUrl
    m = QMimeData()
    m.setUrls([QUrl.fromLocalFile(str(p)) for p in paths])
    return m


def test_trackpy_backend_is_driven_by_displacement_and_memory(monkeypatch):
    """trackpy is stubbed: the point is the two knobs reach it unchanged."""
    from spacr.qt.widgets import timelapse_preview as TP
    seen = {}

    fake = types.ModuleType("trackpy")
    fake.__spec__ = __import__("importlib.machinery",
                               fromlist=["ModuleSpec"]).ModuleSpec(
        "trackpy", None)

    def _link_df(features, search_range=None, memory=None):
        seen["search_range"] = search_range
        seen["memory"] = memory
        out = features.copy()
        out["particle"] = out["original_label"]
        return out

    fake.link_df = _link_df
    fake.quiet = lambda: seen.setdefault("quiet", True)
    monkeypatch.setitem(sys.modules, "trackpy", fake)

    tracks = TP.link_tracks(_synthetic_masks(), mode="trackpy",
                            displacement=77.0, memory=5)
    assert seen == {"quiet": True, "search_range": 77.0, "memory": 5}
    assert "track_id" in tracks.columns
    assert tracks["track_id"].nunique() == 2


def test_trackpy_survives_a_trackpy_without_quiet(monkeypatch):
    from spacr.qt.widgets import timelapse_preview as TP
    fake = types.ModuleType("trackpy")
    fake.__spec__ = __import__("importlib.machinery",
                               fromlist=["ModuleSpec"]).ModuleSpec(
        "trackpy", None)

    def _link_df(features, search_range=None, memory=None):
        out = features.copy()
        out["particle"] = out["original_label"]
        return out

    fake.link_df = _link_df          # no .quiet at all
    monkeypatch.setitem(sys.modules, "trackpy", fake)
    assert TP.link_tracks(_synthetic_masks(), mode="trackpy")["track_id"].nunique() == 2


def test_btrack_offline_config_failure_is_actionable(monkeypatch):
    """No network in a preview: a config download failure must name the fix."""
    from spacr.qt.widgets import timelapse_preview as TP
    spec = __import__("importlib.machinery",
                      fromlist=["ModuleSpec"]).ModuleSpec("btrack", None)
    fake = types.ModuleType("btrack")
    fake.__spec__ = spec
    datasets = types.ModuleType("btrack.datasets")

    def _no_net():
        raise OSError("connection refused")

    datasets.cell_config = _no_net
    fake.datasets = datasets
    monkeypatch.setitem(sys.modules, "btrack", fake)
    monkeypatch.setitem(sys.modules, "btrack.datasets", datasets)

    with pytest.raises(TP.TrackerUnavailable) as exc:
        TP.link_tracks(_synthetic_masks(), mode="btrack")
    msg = str(exc.value)
    assert "btrack" in msg and "cell_config" in msg
    assert "trackpy" in msg          # names a backend that does work offline


def test_btrack_tracks_are_mapped_back_onto_labels(monkeypatch):
    from spacr.qt.widgets import timelapse_preview as TP
    masks = _synthetic_masks()
    spec = __import__("importlib.machinery",
                      fromlist=["ModuleSpec"]).ModuleSpec("btrack", None)
    fake = types.ModuleType("btrack")
    fake.__spec__ = spec
    datasets = types.ModuleType("btrack.datasets")
    datasets.cell_config = lambda: "cfg"
    fake.datasets = datasets
    utils = types.ModuleType("btrack.utils")
    utils.segmentation_to_objects = lambda m, properties=(): ["obj"]
    fake.utils = utils

    from spacr.timelapse import _prepare_for_tracking
    features = _prepare_for_tracking(masks)

    class _Tracker:
        max_search_radius = 0

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def configure(self, cfg):
            self.cfg = cfg

        def append(self, objects):
            self.objects = objects

        def track(self):
            pass

        @property
        def tracks(self):
            out = []
            for label in (1, 2):
                f = features[features["original_label"] == label]
                out.append({"ID": int(label), "t": f["frame"].tolist(),
                            "x": f["x"].tolist(), "y": f["y"].tolist()})
            return out

    fake.BayesianTracker = _Tracker
    for name, mod in (("btrack", fake), ("btrack.datasets", datasets),
                      ("btrack.utils", utils)):
        monkeypatch.setitem(sys.modules, name, mod)

    tracks = TP.link_tracks(masks, mode="btrack", displacement=64.0)
    assert set(tracks["track_id"].unique()) == {1, 2}
    # nearest-centroid recovery must return each track to its own label
    assert (tracks["track_id"] == tracks["original_label"]).all()


def test_label_recovery_is_empty_frame_safe():
    import pandas as pd
    from spacr.qt.widgets.timelapse_preview import _attach_labels_by_position
    from spacr.timelapse import _prepare_for_tracking
    features = _prepare_for_tracking(_synthetic_masks())
    df = pd.DataFrame({"frame": [99], "track_id": [1], "x": [0.0], "y": [0.0]})
    out = _attach_labels_by_position(df, features)
    assert "original_label" in out.columns and out.empty


# ---------------------------------------------------------------------------
# Drag & drop, pickers, teardown
# ---------------------------------------------------------------------------

def test_timelapse_accepts_a_dropped_folder(qtbot, frame_dir):
    from spacr.qt.widgets.timelapse_preview import TimelapsePreviewPanel
    panel = TimelapsePreviewPanel(threaded=False)
    qtbot.addWidget(panel)
    evt = _Evt(_mime_for(frame_dir))
    panel.dragEnterEvent(evt)
    assert evt.accepted
    panel.dragMoveEvent(evt)
    drop = _Evt(_mime_for(frame_dir))
    panel.dropEvent(drop)
    assert drop.accepted and panel._sequence is not None


def test_timelapse_rejects_an_unsupported_drop(qtbot, tmp_path):
    from PySide6.QtCore import QMimeData
    from spacr.qt.widgets.timelapse_preview import TimelapsePreviewPanel
    panel = TimelapsePreviewPanel()
    qtbot.addWidget(panel)
    doc = tmp_path / "notes.txt"
    doc.write_text("x")
    for handler in (panel.dragEnterEvent, panel.dragMoveEvent, panel.dropEvent):
        evt = _Evt(_mime_for(doc))
        handler(evt)
        assert evt.ignored and not evt.accepted
    empty = _Evt(QMimeData())
    panel.dragEnterEvent(empty)
    assert empty.ignored


def test_motility_accepts_a_dropped_plate_folder(qtbot, plate_dir, tmp_path):
    from PySide6.QtCore import QMimeData
    from spacr.qt.widgets.motility_preview import MotilityPreviewPanel
    panel = MotilityPreviewPanel(threaded=False)
    qtbot.addWidget(panel)
    evt = _Evt(_mime_for(plate_dir))
    panel.dragEnterEvent(evt)
    assert evt.accepted
    panel.dragMoveEvent(evt)
    drop = _Evt(_mime_for(plate_dir))
    panel.dropEvent(drop)
    assert drop.accepted and panel._groups

    doc = tmp_path / "notes.txt"
    doc.write_text("x")
    for handler in (panel.dragEnterEvent, panel.dropEvent):
        bad = _Evt(_mime_for(doc))
        handler(bad)
        assert bad.ignored
    blank = _Evt(QMimeData())
    panel.dragMoveEvent(blank)
    assert blank.ignored


def test_pickers_route_the_chosen_path(qtbot, monkeypatch, frame_dir, mask_dir,
                                       plate_dir):
    """The file dialogs are the only place a modal is legitimate — stub them."""
    from PySide6.QtWidgets import QFileDialog
    from spacr.qt.widgets.motility_preview import MotilityPreviewPanel
    from spacr.qt.widgets.timelapse_preview import TimelapsePreviewPanel

    panel = TimelapsePreviewPanel(threaded=False)
    qtbot.addWidget(panel)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: frame_dir))
    panel._pick_sequence()
    assert panel._sequence is not None
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: mask_dir))
    panel._pick_masks()
    assert panel._mask_sequence is not None

    mot = MotilityPreviewPanel(threaded=False)
    qtbot.addWidget(mot)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: plate_dir))
    mot._pick_folder()
    assert mot._groups

    # A cancelled dialog returns "" and must be a no-op, not a load attempt.
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: ""))
    before = panel._sequence
    panel._pick_sequence()
    panel._pick_masks()
    mot._pick_folder()
    assert panel._sequence is before


def test_closing_a_panel_waits_for_its_worker(qtbot, frame_dir,
                                              counted_segmentation, plate_dir):
    """A QThread collected mid-run aborts the process — close must join it."""
    from spacr.qt.widgets.motility_preview import MotilityPreviewPanel
    from spacr.qt.widgets.timelapse_preview import TimelapsePreviewPanel
    panel = TimelapsePreviewPanel()
    qtbot.addWidget(panel)
    panel.load_sequence(frame_dir)
    panel.run_preview()
    panel.close()                    # while the worker may still be running
    assert panel._worker is None or not panel._worker.isRunning()

    mot = MotilityPreviewPanel(threaded=False)
    qtbot.addWidget(mot)
    mot.load_folder(plate_dir)
    mot.run_preview()
    mot.close()
    assert mot._worker is None or not mot._worker.isRunning()


def test_a_second_run_while_one_is_in_flight_is_refused(qtbot, frame_dir,
                                                        counted_segmentation,
                                                        plate_dir):
    from spacr.qt.widgets.motility_preview import MotilityPreviewPanel
    from spacr.qt.widgets.timelapse_preview import TimelapsePreviewPanel
    panel = TimelapsePreviewPanel()
    qtbot.addWidget(panel)
    panel.load_sequence(frame_dir)
    panel.run_preview()
    if panel._worker is not None and panel._worker.isRunning():
        panel.run_preview()
        assert "already running" in panel._status.text()
    qtbot.waitUntil(lambda: panel._worker is None, timeout=8000)

    mot = MotilityPreviewPanel(threaded=False)
    qtbot.addWidget(mot)
    mot.load_folder(plate_dir)
    mot.run_preview()
    if mot._worker is not None and mot._worker.isRunning():
        mot.run_preview()
        assert "already running" in mot._status.text()
    qtbot.waitUntil(lambda: mot._worker is None, timeout=8000)


def test_an_empty_worker_result_is_reported(qtbot, frame_dir):
    from spacr.qt.widgets.timelapse_preview import TimelapsePreviewPanel
    panel = TimelapsePreviewPanel()
    qtbot.addWidget(panel)
    panel.load_sequence(frame_dir)
    panel._on_worker_done(None, "")
    assert "returned nothing" in panel._status.text()


def test_motility_empty_point_table_is_reported(qtbot, plate_dir):
    import pandas as pd
    from spacr.qt.widgets.motility_preview import MotilityPreviewPanel
    panel = MotilityPreviewPanel(threaded=False)
    qtbot.addWidget(panel)
    panel.load_folder(plate_dir)
    panel._on_worker_done(pd.DataFrame(), "")
    assert "No objects found" in panel._status.text()
    panel._on_worker_done(None, "")
    assert "No objects found" in panel._status.text()


def test_switching_group_clears_the_cache(qtbot, tmp_path):
    from spacr.qt.widgets.motility_preview import MotilityPreviewPanel
    src = tmp_path / "twofield"
    merged = src / "merged"
    merged.mkdir(parents=True)
    for field in (1, 2):
        for t in range(N_FRAMES):
            arr = np.zeros((6, H, W), np.float32)
            cell = np.zeros((H, W), np.int32)
            _disc(cell, 8, 6 + 2 * t, 3, 1)
            arr[4] = cell
            np.save(str(merged / f"plate1_A01_{field}_{t}.npy"), arr)
    panel = MotilityPreviewPanel(threaded=False)
    qtbot.addWidget(panel)
    assert panel.load_folder(str(src)) is True
    assert panel._group_box.count() == 2
    with qtbot.waitSignal(panel.preview_ready, timeout=8000):
        panel.run_preview()
    assert panel._points is not None
    panel._group_box.setCurrentIndex(1)
    assert panel._points is None, "stale points survived a group change"


def test_recompute_before_a_run_is_a_no_op(qtbot):
    from spacr.qt.widgets.motility_preview import MotilityPreviewPanel
    panel = MotilityPreviewPanel(threaded=False)
    qtbot.addWidget(panel)
    panel.recompute()                 # nothing cached yet
    panel._min_len.setValue(9)        # metric change with no data
    assert panel._tracks is None


def test_scoring_change_before_a_run_is_a_no_op(qtbot, frame_dir):
    from spacr.qt.widgets.timelapse_preview import TimelapsePreviewPanel
    panel = TimelapsePreviewPanel()
    qtbot.addWidget(panel)
    panel._min_len.setValue(9)
    panel._displacement.setValue(90.0)
    panel.load_sequence(frame_dir)
    panel._displacement.setValue(95.0)     # still nothing segmented
    assert panel._stats is None


def test_display_knobs_never_touch_the_tracker(qtbot, frame_dir,
                                               counted_segmentation, monkeypatch):
    from spacr.qt.widgets import timelapse_preview as TP
    panel = TP.TimelapsePreviewPanel()
    qtbot.addWidget(panel)
    panel.load_sequence(frame_dir)
    with qtbot.waitSignal(panel.preview_ready, timeout=8000):
        panel.run_preview()
    calls = {"n": 0}
    monkeypatch.setattr(TP, "link_tracks",
                        lambda *a, **k: calls.__setitem__("n", calls["n"] + 1))
    panel._tail.setValue(3)
    panel._normalise.setChecked(False)
    assert calls["n"] == 0
    assert counted_segmentation["n"] == N_FRAMES


def test_masks_only_preview_renders_without_a_source_sequence(qtbot, mask_dir):
    """Masks alone are a legitimate input — the canvas falls back to them."""
    from spacr.qt.widgets.timelapse_preview import TimelapsePreviewPanel
    panel = TimelapsePreviewPanel()
    qtbot.addWidget(panel)
    panel._mode_box.setCurrentText("iou")
    assert panel.load_masks(mask_dir) is True
    with qtbot.waitSignal(panel.preview_ready, timeout=8000) as sig:
        panel.run_preview()
    assert sig.args[0].n_tracks == 2
    panel._frame_slider.setValue(2)
    assert panel._frame_label.text() == f"3/{N_FRAMES}"


def test_render_handles_a_multichannel_frame():
    from spacr.qt.widgets.timelapse_preview import render_frame
    frame = np.zeros((H, W, 3), np.uint16)
    frame[..., 1] = 500
    rgb = render_frame(frame, channel=1, normalise=False)
    assert rgb.shape == (H, W, 3)


def test_track_overlay_clips_at_the_image_edge():
    """Drawing must never index outside the canvas."""
    import pandas as pd
    from spacr.qt.widgets.timelapse_preview import render_frame
    tracks = pd.DataFrame({
        "frame": [0, 1], "track_id": [1, 1],
        "x": [-40.0, 500.0], "y": [-40.0, 500.0],
        "original_label": [1, 1],
    })
    rgb = render_frame(np.zeros((H, W), np.uint16), tracks=tracks, frame=1)
    assert rgb.shape == (H, W, 3)


def test_sequence_index_is_bounds_checked(big_stack):
    from spacr.qt.widgets.timelapse_preview import FrameSequence
    seq = FrameSequence.open(big_stack, max_frames=3)
    with pytest.raises(IndexError):
        seq.frame(3)
    assert "showing 3 of 40" in seq.describe()


def test_describe_says_nothing_about_truncation_when_complete(frame_dir):
    from spacr.qt.widgets.timelapse_preview import FrameSequence
    seq = FrameSequence.open(frame_dir, max_frames=99)
    assert seq.truncated is False
    assert f"{N_FRAMES} frames" in seq.describe()


def test_unsupported_sequence_suffix_is_named(tmp_path):
    from spacr.qt.widgets.timelapse_preview import FrameSequence
    p = tmp_path / "movie.avi"
    p.write_bytes(b"x")
    with pytest.raises(ValueError, match="unsupported input"):
        FrameSequence.open(str(p))


def test_npy_with_too_few_dimensions_is_named(tmp_path):
    from spacr.qt.widgets.timelapse_preview import FrameSequence
    p = tmp_path / "flat.npy"
    np.save(str(p), np.zeros((H, W), np.uint16))
    with pytest.raises(ValueError, match="timelapse stack"):
        FrameSequence.open(str(p))


def test_segmentation_shape_mismatch_is_named(monkeypatch, frame_dir):
    from spacr.qt.widgets import timelapse_preview as TP
    shapes = iter([(H, W), (H, W - 1), (H, W), (H, W)])
    monkeypatch.setattr(TP, "segment_frame",
                        lambda img, params: np.zeros(next(shapes), np.int32))
    seq = TP.FrameSequence.open(frame_dir, max_frames=4)
    with pytest.raises(ValueError, match="different shapes"):
        TP.segment_sequence(seq, {})


# ---------------------------------------------------------------------------
# Remaining branches: ultrack, alternate frame formats, defensive paths
# ---------------------------------------------------------------------------

def _spec(name):
    import importlib.machinery
    return importlib.machinery.ModuleSpec(name, None)


def test_present_ultrack_backend_is_actually_used(monkeypatch):
    """Ultrack is stubbed: the point is the branch is wired, offline."""
    from spacr.qt.widgets import timelapse_preview as TP
    masks = _synthetic_masks()
    seen = {}

    root = types.ModuleType("ultrack")
    root.__spec__ = _spec("ultrack")

    class _Section:
        working_dir = None
        max_distance = 0.0

    class _Config:
        def __init__(self):
            self.data_config = _Section()
            self.linking_config = _Section()
            self.tracking_config = _Section()

    def _track(config, foreground=None, contours=None):
        seen["foreground"] = foreground is not None
        seen["contours"] = contours is not None
        seen["max_distance"] = config.linking_config.max_distance

    root.MainConfig = _Config
    root.track = _track
    root.to_tracks_layer = lambda cfg: ("layer", None)
    root.tracks_to_zarr = lambda cfg, layer: masks
    utils = types.ModuleType("ultrack.utils")
    utils.labels_to_contours = lambda m, sigma=0.0: (m > 0, m)
    root.utils = utils
    utils.__spec__ = _spec("ultrack.utils")
    monkeypatch.setitem(sys.modules, "ultrack", root)
    monkeypatch.setitem(sys.modules, "ultrack.utils", utils)

    tracks = TP.link_tracks(masks, mode="ultrack", displacement=25.0)
    assert seen["max_distance"] == 25.0
    assert seen["foreground"] and seen["contours"]
    assert tracks["track_id"].nunique() == 2


def test_btrack_with_no_tracks_returns_an_empty_table(monkeypatch):
    from spacr.qt.widgets import timelapse_preview as TP
    fake = types.ModuleType("btrack")
    fake.__spec__ = _spec("btrack")
    datasets = types.ModuleType("btrack.datasets")
    datasets.cell_config = lambda: "cfg"
    utils = types.ModuleType("btrack.utils")
    utils.segmentation_to_objects = lambda m, properties=(): []
    fake.datasets = datasets
    fake.utils = utils

    class _Tracker:
        max_search_radius = 0
        tracks = []

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def configure(self, cfg):
            pass

        def append(self, objects):
            pass

        def track(self):
            pass

    fake.BayesianTracker = _Tracker
    for name, mod in (("btrack", fake), ("btrack.datasets", datasets),
                      ("btrack.utils", utils)):
        monkeypatch.setitem(sys.modules, name, mod)
    out = TP.link_tracks(_synthetic_masks(), mode="btrack")
    assert out.empty and "original_label" in out.columns


@pytest.mark.parametrize("ext", ["npy", "png"])
def test_frames_can_be_npy_or_png_files(tmp_path, ext):
    from spacr.qt.widgets.timelapse_preview import FrameSequence
    d = tmp_path / f"as_{ext}"
    d.mkdir()
    masks = _synthetic_masks()
    for t in range(N_FRAMES):
        if ext == "npy":
            np.save(str(d / f"f_{t}.npy"), masks[t].astype(np.uint16))
        else:
            from PIL import Image
            Image.fromarray((masks[t] * 60).astype(np.uint8)).save(
                str(d / f"f_{t}.png"))
    seq = FrameSequence.open(str(d), max_frames=9)
    assert len(seq) == N_FRAMES and seq.frame(1).shape == (H, W)


def test_a_folder_with_one_frame_is_refused(tmp_path):
    import tifffile
    from spacr.qt.widgets.timelapse_preview import FrameSequence
    d = tmp_path / "lonely"
    d.mkdir()
    tifffile.imwrite(str(d / "only.tif"), np.zeros((H, W), np.uint16))
    with pytest.raises(ValueError, match="single frame"):
        FrameSequence.open(str(d))


def test_frame_channel_handles_odd_shapes():
    from spacr.qt.widgets.timelapse_preview import frame_channel
    four_d = np.zeros((1, 1, H, W), np.uint16)
    assert frame_channel(four_d, 0).shape == (H, W)
    # both axes small: fall back to the channel-last reading
    tiny = np.zeros((4, 4, 4), np.uint16)
    tiny[..., 2] = 5
    assert int(frame_channel(tiny, 2).max()) == 5


def test_propagate_toggle_pushes_on_every_run(qtbot, frame_dir,
                                              counted_segmentation):
    from spacr.qt.widgets.timelapse_preview import TimelapsePreviewPanel
    panel = TimelapsePreviewPanel()
    qtbot.addWidget(panel)
    pushes = []
    panel.set_propagate_callback(lambda d: pushes.append(dict(d)))
    panel._propagate_btn.setChecked(True)
    assert len(pushes) == 1
    panel.load_sequence(frame_dir)
    with qtbot.waitSignal(panel.preview_ready, timeout=8000):
        panel.run_preview()
    assert len(pushes) == 2, "a finished run must re-push while the toggle is on"


def test_motility_propagate_toggle_pushes_on_every_run(qtbot, plate_dir):
    from spacr.qt.widgets.motility_preview import MotilityPreviewPanel
    panel = MotilityPreviewPanel(threaded=False)
    qtbot.addWidget(panel)
    pushes = []
    panel.set_propagate_callback(lambda d: pushes.append(dict(d)))
    panel._propagate_btn.setChecked(True)
    panel.load_folder(plate_dir)
    with qtbot.waitSignal(panel.preview_ready, timeout=8000):
        panel.run_preview()
    assert len(pushes) >= 2


def test_apply_settings_moves_the_diameter(qtbot):
    from spacr.qt.widgets.timelapse_preview import TimelapsePreviewPanel
    panel = TimelapsePreviewPanel()
    qtbot.addWidget(panel)
    panel.apply_settings({"timelapse_objects": ["cell"], "cell_diameter": 55})
    assert panel._diameter.value() == pytest.approx(55.0)


def test_motility_apply_settings_survives_junk(qtbot):
    from spacr.qt.widgets.motility_preview import MotilityPreviewPanel
    panel = MotilityPreviewPanel(threaded=False)
    qtbot.addWidget(panel)
    panel.apply_settings({"max_displacement": "not a number"})   # no raise
    assert panel._max_disp.value() == pytest.approx(50.0)


def test_timelapse_apply_settings_survives_junk(qtbot):
    from spacr.qt.widgets.timelapse_preview import TimelapsePreviewPanel
    panel = TimelapsePreviewPanel()
    qtbot.addWidget(panel)
    panel.apply_settings({"timelapse_memory": "three"})          # no raise
    assert panel._memory.value() == 3


def test_a_plate_whose_groups_are_all_single_frame_is_refused(qtbot, tmp_path):
    from spacr.qt.widgets.motility_preview import MotilityPreviewPanel
    merged = tmp_path / "oneshot" / "merged"
    merged.mkdir(parents=True)
    np.save(str(merged / "plate1_A01_1_0.npy"), np.zeros((6, H, W), np.float32))
    panel = MotilityPreviewPanel(threaded=False)
    qtbot.addWidget(panel)
    assert panel.load_folder(str(tmp_path / "oneshot")) is False
    assert "time series" in panel._status.text()


def test_non_npy_files_beside_the_merged_arrays_are_ignored(plate_dir):
    from spacr.qt.widgets.motility_preview import group_merged_files
    merged = os.path.join(plate_dir, "merged")
    with open(os.path.join(merged, "README.txt"), "w") as fh:
        fh.write("not an array")
    groups = group_merged_files(merged)
    assert list(groups) == [("plate1", "A01", "1")]


def test_a_non_3d_merged_array_is_skipped(tmp_path):
    from spacr.qt.widgets.motility_preview import (
        build_point_table, group_merged_files)
    merged = tmp_path / "mixed" / "merged"
    merged.mkdir(parents=True)
    for t in range(N_FRAMES):
        if t == 1:
            np.save(str(merged / f"plate1_A01_1_{t}.npy"),
                    np.zeros((H, W), np.float32))     # 2-D: not a merged array
            continue
        arr = np.zeros((6, H, W), np.float32)
        cell = np.zeros((H, W), np.int32)
        _disc(cell, 8, 6 + 2 * t, 3, 1)
        arr[4] = cell
        np.save(str(merged / f"plate1_A01_1_{t}.npy"), arr)
    metas = group_merged_files(str(merged))[("plate1", "A01", "1")]
    pts = build_point_table(str(merged), metas, 4, 4, None, max_frames=9)
    assert sorted(pts["frame"].unique().tolist()) == [0, 2, 3]


def test_resolve_accepts_a_file_inside_the_plate_folder(plate_dir):
    from spacr.qt.widgets.motility_preview import resolve_merged_dir
    a_file = os.path.join(plate_dir, "merged", "plate1_A01_1_0.npy")
    assert resolve_merged_dir(a_file).endswith("merged")


def test_summary_survives_a_cutoff_that_excludes_everything(qtbot, plate_dir):
    from spacr.qt.widgets.motility_preview import MotilityPreviewPanel
    panel = MotilityPreviewPanel(threaded=False)
    qtbot.addWidget(panel)
    panel.load_folder(plate_dir)
    with qtbot.waitSignal(panel.preview_ready, timeout=8000):
        panel.run_preview()
    with qtbot.waitSignal(panel.preview_ready, timeout=8000) as sig:
        panel._min_len.setValue(99)
    s = sig.args[0]
    assert s.n_used == 0 and "n/a" in s.summary()


def test_a_broken_plot_is_reported_not_raised(qtbot, plate_dir, monkeypatch):
    from spacr.qt.widgets import motility_preview as MP
    panel = MP.MotilityPreviewPanel()
    qtbot.addWidget(panel)
    panel.load_folder(plate_dir)
    monkeypatch.setattr(MP, "render_motility_figure",
                        lambda *a, **k: (_ for _ in ()).throw(
                            RuntimeError("no canvas")))
    with qtbot.waitSignal(panel.preview_ready, timeout=8000):
        panel.run_preview()
    assert "Plot failed" in panel._plot.text()


def test_plane_autodetect_survives_an_unreadable_array(qtbot, plate_dir,
                                                       monkeypatch):
    from spacr.qt.widgets.motility_preview import MotilityPreviewPanel
    panel = MotilityPreviewPanel(threaded=False)
    qtbot.addWidget(panel)
    monkeypatch.setattr(np, "load",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("gone")))
    assert panel.load_folder(plate_dir) is True     # autodetect failure is soft


def test_zero_length_steps_are_skipped_by_the_metrics():
    """A track that never moves has no finite step distribution to speak of."""
    import pandas as pd
    from spacr.qt.widgets.motility_preview import Calibration, track_metrics
    pts = pd.DataFrame({
        "plateID": ["p"] * 3, "wellID": ["A01"] * 3, "fieldID": ["1"] * 3,
        "cellID": [1, 1, 1], "frame": [0, 1, 2],
        "x": [np.nan, np.nan, np.nan], "y": [np.nan, np.nan, np.nan],
        "area": [9, 9, 9], "infected": [False] * 3,
    })
    assert track_metrics(pts, Calibration(), 2).empty


def test_single_observation_tracks_are_dropped_by_the_metrics():
    import pandas as pd
    from spacr.qt.widgets.motility_preview import Calibration, track_metrics
    pts = pd.DataFrame({
        "plateID": ["p"], "wellID": ["A01"], "fieldID": ["1"],
        "cellID": [1], "frame": [0], "x": [1.0], "y": [1.0],
        "area": [9], "infected": [False],
    })
    assert track_metrics(pts, Calibration(), 2).empty


def test_figure_skips_single_point_tracks():
    import pandas as pd
    from spacr.qt.widgets.motility_preview import (
        Calibration, render_motility_figure)
    pts = pd.DataFrame({
        "plateID": ["p"], "wellID": ["A01"], "fieldID": ["1"],
        "cellID": [1], "frame": [0], "x": [1.0], "y": [1.0],
        "area": [9], "infected": [False],
    })
    rgb = render_motility_figure(pts, None, Calibration(), 3, 0.95,
                                 width_px=300, height_px=160)
    assert rgb.shape[2] == 3


def test_a_genuinely_multipage_tiff_is_read_one_page_at_a_time(tmp_path):
    """The cheap TIFF path: one page per frame, decoded on demand."""
    import tifffile
    from spacr.qt.widgets.timelapse_preview import FrameSequence
    p = tmp_path / "pages.tif"
    with tifffile.TiffWriter(str(p)) as tw:
        for t in range(N_FRAMES):
            tw.write(np.full((H, W), t + 1, np.uint16), contiguous=False)
    seq = FrameSequence.open(str(p), max_frames=3)
    assert seq.kind == "tiff"
    assert seq.n_available == N_FRAMES and len(seq) == 3
    assert seq.read_count == 0
    assert int(seq.frame(2)[0, 0]) == 3
    assert seq.read_count == 1, "reading one frame decoded more than one page"


# ---------------------------------------------------------------------------
# Key contract on the centroid join
# ---------------------------------------------------------------------------

def test_tracks_from_features_refuses_a_repeated_object_in_a_frame():
    """One centroid per (frame, label), enforced rather than assumed.

    ``_prepare_for_tracking`` runs regionprops, so a label appears once per
    frame. A features table assembled any other way -- two frames' props
    concatenated without re-indexing, say -- would invent extra track rows
    with fabricated centroids, and the displacement statistics the panel
    reports would be computed over objects that do not exist.
    """
    import pandas as pd
    from spacr.qt.widgets.timelapse_preview import _tracks_from_features

    tracks = pd.DataFrame({"track_id": [1, 2], "frame": [0, 0],
                           "original_label": [1, 2]})
    features = pd.DataFrame({
        "frame": [0, 0, 0],
        "original_label": [1, 1, 2],       # label 1 measured twice in frame 0
        "x": [1.0, 9.0, 2.0],
        "y": [1.0, 9.0, 2.0],
    })

    with pytest.raises(pd.errors.MergeError, match="not a many-to-one merge"):
        _tracks_from_features(tracks, features)


def test_tracks_from_features_allows_two_tracks_on_one_label():
    """The left side is deliberately unconstrained: merge/split events repeat a label."""
    import pandas as pd
    from spacr.qt.widgets.timelapse_preview import _tracks_from_features

    # Two tracks claim label 1 in frame 0 -- exactly what a merge event looks
    # like, and not an error.
    tracks = pd.DataFrame({"track_id": [1, 2], "frame": [0, 0],
                           "original_label": [1, 1]})
    features = pd.DataFrame({"frame": [0], "original_label": [1],
                             "x": [1.0], "y": [2.0]})

    out = _tracks_from_features(tracks, features)
    assert len(out) == 2
    assert list(out["x"]) == [1.0, 1.0]

"""The tracking preview: opening a movie, rendering a frame, one pass, and the panel.

The preview reads a movie lazily, segments it once and re-links it on every
tracking change.  Each case hands one stage something it did not expect -- a
one-frame TIFF, a stack no memory map will open, a whole-frame object, a
track of NaNs, a result whose question has changed -- and asserts the exact
array, track table or status the stage produces instead of raising.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets import timelapse_preview as TP  # noqa: E402

pytestmark = pytest.mark.qt

H = W = 32
N_FRAMES = 4


def _disc(mask, cy, cx, r, label):
    yy, xx = np.ogrid[:mask.shape[0], :mask.shape[1]]
    mask[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = label
    return mask


def _synthetic_masks(n_frames=N_FRAMES, drift=3):
    """Two discs drifting right by ``drift`` pixels a frame."""
    stack = np.zeros((n_frames, H, W), np.int32)
    for t in range(n_frames):
        _disc(stack[t], 8, 6 + drift * t, 3, 1)
        _disc(stack[t], 22, 6 + drift * t, 3, 2)
    return stack


#: What linking ``_synthetic_masks()`` gives: two tracks, three pixels a step.
LINKED = pd.DataFrame({
    "frame": [0, 0, 1, 1, 2, 2, 3, 3],
    "original_label": [1, 2] * 4,
    "track_id": [1, 2] * 4,
    "x": [6.0, 6.0, 9.0, 9.0, 12.0, 12.0, 15.0, 15.0],
    "y": [8.0, 22.0] * 4,
})


def _write_folder(folder, stack, *, as_image):
    import tifffile
    folder.mkdir()
    for t, plane in enumerate(stack):
        data = (plane > 0).astype(np.uint16) * 900 + 40 if as_image else plane
        tifffile.imwrite(str(folder / f"field_t{t}.tif"), data.astype(np.uint16))
    return str(folder)


def _threshold_segmenter(monkeypatch):
    from skimage.measure import label as sk_label
    monkeypatch.setattr(
        TP, "segment_frame",
        lambda image, params: sk_label(
            TP.frame_channel(np.asarray(image), 0) > 100).astype(np.int32))


def _assert_tracks(tracks):
    pd.testing.assert_frame_equal(
        tracks[list(LINKED.columns)].reset_index(drop=True), LINKED,
        check_dtype=False)


def test_frame_sequence_open_refuses_one_frame_and_falls_back_on_damaged_files(
        tmp_path, monkeypatch, subtests):
    """One frame is refused by sentence; bad series metadata and unmappable stacks still read."""
    import tifffile

    with subtests.test(msg="single-frame-refused"):
        path = tmp_path / "one.tif"
        tifffile.imwrite(str(path), np.zeros((H, W), np.uint16))
        with pytest.raises(ValueError) as excinfo:
            TP.FrameSequence.open(str(path), max_frames=4)
        assert "holds 1 frame(s)" in str(excinfo.value)
        assert "multi-frame TIFF or a folder of frames" in str(excinfo.value)

    stack_path = tmp_path / "single_page.tif"
    tifffile.imwrite(str(stack_path), np.arange(N_FRAMES * H * W, dtype=np.uint16)
                     .reshape(N_FRAMES, H, W),
                     photometric="rgb", planarconfig="separate")
    with subtests.test(msg="memory-map-opened-once-and-reused"):
        seq = TP.FrameSequence.open(str(stack_path), max_frames=4)
        assert seq.kind == "tiffmm"
        maps, real_memmap = [], tifffile.memmap

        def counting(*args, **kwargs):
            maps.append(args)
            return real_memmap(*args, **kwargs)

        monkeypatch.setattr(tifffile, "memmap", counting)
        first, third = seq.frame(0), seq.frame(2)
        monkeypatch.setattr(tifffile, "memmap", real_memmap)
        assert len(maps) == 1
        assert (int(first[0, 0]), int(third[0, 0])) == (0, 2 * H * W)
    with subtests.test(msg="unmappable-stack-read-whole"):
        seq = TP.FrameSequence.open(str(stack_path), max_frames=4)

        def refuse(*_a, **_k):
            raise ValueError("cannot memory-map a compressed file")

        monkeypatch.setattr(tifffile, "memmap", refuse)
        plane = seq.frame(2)
        monkeypatch.setattr(tifffile, "memmap", real_memmap)
        assert plane.shape == (H, W)
        assert np.array_equal(plane, np.arange(2 * H * W, 3 * H * W,
                                               dtype=np.uint16).reshape(H, W))
    with subtests.test(msg="unreadable-series-falls-back-to-pages"):
        path = tmp_path / "movie.tif"
        tifffile.imwrite(str(path), np.zeros((N_FRAMES, H, W), np.uint16),
                         photometric="minisblack")
        real_open = tifffile.TiffFile

        class _NoSeries:
            def __init__(self, *args, **kwargs):
                self._inner = real_open(*args, **kwargs)
                self.pages = self._inner.pages

            @property
            def series(self):
                raise ValueError("cannot interpret the series")

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                self._inner.close()
                return False

        monkeypatch.setattr(tifffile, "TiffFile", _NoSeries)
        seq = TP.FrameSequence.open(str(path), max_frames=4)
        assert (len(seq), seq.kind) == (N_FRAMES, "tiff")


def test_render_frame_draws_only_what_there_is_to_draw(subtests):
    """A whole-frame object has no outline, a NaN track no head, a colour frame is not restacked."""
    with subtests.test(msg="one-object-edge-to-edge"):
        image = np.arange(48 * 48, dtype=np.uint16).reshape(48, 48)
        whole = np.ones((48, 48), np.int32)
        assert np.array_equal(
            TP.render_frame(image, labels=whole, tracks=None, frame=0),
            TP.render_frame(image, labels=None, tracks=None, frame=0))
    with subtests.test(msg="all-nan-track-draws-no-head"):
        image = np.zeros((8, 8), dtype=np.uint16)
        nan_track = pd.DataFrame({"frame": [0, 1], "track_id": [1, 1],
                                  "x": [np.nan, np.nan], "y": [np.nan, np.nan]})
        blank = TP.render_frame(image, tracks=nan_track, frame=1)
        drawn = TP.render_frame(image, tracks=nan_track.assign(
            x=[2.0, 4.0], y=[2.0, 4.0]), frame=1)
        assert (blank.shape, blank.dtype, int(blank.max())) == (
            (8, 8, 3), np.uint8, 0)
        assert tuple(int(v) for v in drawn[4, 4]) == TP.track_colour(1)
    with subtests.test(msg="colour-frame-not-restacked"):
        colour = np.zeros((1, H, W, 3), np.uint8)
        colour[..., 0] = 200
        rgb = TP.render_frame(colour, normalise=False, channel=0)
        assert rgb.shape == (H, W, 3)
        assert np.array_equal(rgb, colour[0])


def test_run_preview_pass_segments_only_when_it_has_to(tmp_path, monkeypatch,
                                                       subtests):
    """Ready masks are linked, raw frames segmented, cached masks re-linked; nothing loaded is refused."""
    frames = _write_folder(tmp_path / "frames", _synthetic_masks(), as_image=True)
    labels = _write_folder(tmp_path / "labels", _synthetic_masks(), as_image=False)

    with subtests.test(msg="nothing-loaded"):
        with pytest.raises(ValueError, match="^Load a sequence first.$"):
            TP.run_preview_pass(TP.TimelapseRequest())
    with subtests.test(msg="ready-made-masks-are-not-segmented"):
        monkeypatch.setattr(TP, "segment_frame", lambda *a, **k: pytest.fail(
            "ready-made masks were segmented"))
        out = TP.run_preview_pass(TP.TimelapseRequest(
            mask_sequence=TP.FrameSequence.open(labels, max_frames=N_FRAMES),
            seg={"mask_channel": 0}, track={}))
        assert (out["segmented"], out["masks_built"]) == (False, True)
        assert np.array_equal(out["masks"] > 0, _synthetic_masks() > 0)
        _assert_tracks(out["tracks"])
    with subtests.test(msg="raw-frames-are-segmented"):
        _threshold_segmenter(monkeypatch)
        out = TP.run_preview_pass(TP.TimelapseRequest(
            sequence=TP.FrameSequence.open(frames, max_frames=N_FRAMES),
            seg={}, track={}))
        assert (out["segmented"], out["masks_built"]) == (True, True)
        assert out["masks"].shape == (N_FRAMES, H, W)
        _assert_tracks(out["tracks"])
    with subtests.test(msg="cached-masks-relink"):
        out = TP.run_preview_pass(TP.TimelapseRequest(
            cached_masks=_synthetic_masks(), seg={}, track={}))
        assert (out["segmented"], out["masks_built"]) == (False, False)
        _assert_tracks(out["tracks"])
    with subtests.test(msg="include-images-reads-each-frame-once"):
        seq = TP.FrameSequence.open(frames, max_frames=2)
        out = TP.run_preview_pass(TP.TimelapseRequest(
            sequence=seq, seg={}, track={}, include_images=True))
        assert out["images"][:, 0, 0].tolist() == [40, 40]
        assert out["images"][:, 8, 3].tolist() == [940, 40]
        assert out["segmented"] is True
        assert seq.read_count == 2
    with subtests.test(msg="cached-masks-still-read-raw-frames"):
        seq = TP.FrameSequence.open(frames, max_frames=2)
        monkeypatch.setattr(TP, "link_tracks", lambda *_a, **_k: None)
        out = TP.run_preview_pass(TP.TimelapseRequest(
            sequence=seq, cached_masks=_synthetic_masks(2), seg={}, track={},
            include_images=True))
        assert (out["segmented"], out["masks_built"]) == (False, False)
        assert out["images"][:, 8, 3].tolist() == [940, 40]


class _Movie:
    """Stands in for the movie panel ``_push_to_movie`` publishes into."""

    def __init__(self, max_fields=4):
        self.fields = None
        self._max_fields = max_fields

    def set_fields(self, fields):
        self.fields = fields

    def max_fields(self):
        return self._max_fields


class _Frames:
    """A one-frame sequence returning a fixed array."""

    def __init__(self, frame):
        self._frame = frame

    def __len__(self):
        return 1

    def frame(self, index):
        return self._frame


def test_the_panel_counts_channels_and_publishes_tracked_labels_to_the_movie(
        qtbot, tmp_path, monkeypatch, subtests):
    """Channel counts follow the frame axis rule; the movie gets tracked labels, mask fallbacks or nothing."""
    def panel():
        widget = TP.TimelapsePreviewPanel(threaded=False)
        qtbot.addWidget(widget)
        return widget

    with subtests.test(msg="channel-count"):
        widget = panel()
        assert widget._frame_channel_count() == 0
        for shape, expected in (((3, H, W), 3), ((H, W, 4), 4), ((3, 4, 5), 5)):
            widget._sequence = _Frames(np.zeros(shape, np.uint16))
            assert widget._frame_channel_count() == expected, shape
    with subtests.test(msg="channel-count-of-an-unreadable-frame"):
        widget = panel()
        frames = _write_folder(tmp_path / "frames", _synthetic_masks(),
                               as_image=True)
        widget.load_sequence_async(frames)
        assert widget._frame_channel_count() == 1

        def refuse(self, index):
            raise OSError("the frame went away")

        monkeypatch.setattr(TP.FrameSequence, "frame", refuse)
        assert widget._frame_channel_count() == 0
        monkeypatch.undo()
    with subtests.test(msg="movie-gets-the-tracked-labels"):
        widget = panel()
        widget._movie_images = _synthetic_masks()
        widget._masks = _synthetic_masks()
        widget._tracked = _synthetic_masks() * 2
        widget._tracks = pd.DataFrame({"frame": [0], "track_id": [1],
                                      "x": [1.0], "y": [1.0]})
        movie = widget._movie_panel = _Movie()
        widget._push_to_movie()
        assert len(movie.fields) == 1
        field = movie.fields[0]
        assert np.array_equal(field["images"], _synthetic_masks())
        assert np.array_equal(field["labels"], _synthetic_masks() * 2)
        assert field["tracks"] is widget._tracks
        assert field["title"]
    with subtests.test(msg="movie-falls-back-to-the-masks"):
        widget = panel()
        movie = widget._movie_panel = _Movie()
        widget._sequence = None
        widget._masks = _synthetic_masks()
        widget._tracked = _synthetic_masks()
        widget._push_to_movie()
        assert movie.fields[0]["images"] is widget._masks
    with subtests.test(msg="movie-emptied-with-nothing-at-all"):
        widget = panel()
        movie = widget._movie_panel = _Movie()
        widget._sequence = None
        widget._masks = None
        widget._push_to_movie()
        assert movie.fields == []


def test_the_panel_adopts_only_the_current_worker_result(qtbot, subtests):
    """A superseded result is dropped; a current one is reported and cached only with a signature."""
    def panel():
        widget = TP.TimelapsePreviewPanel(threaded=False)
        qtbot.addWidget(widget)
        return widget

    class _Running:
        def isRunning(self):
            return True

        def wait(self, *_args):
            return True

    with subtests.test(msg="superseded-result-dropped"):
        widget = panel()
        widget.set_preview_status("linking…")
        widget._pending_token, widget._run_token = 1, 2
        live = widget._worker = _Running()
        widget._on_worker_done(pd.DataFrame({"x": [1]}), "")
        assert widget._worker is live
        assert widget.preview_status() == "linking…"
        widget._worker = None
    with subtests.test(msg="current-error-reported"):
        widget = panel()
        widget._pending_token = widget._run_token = 0
        widget._worker = None
        widget._on_worker_done(None, "cellpose would not load")
        assert "cellpose would not load" in widget.preview_status()
    with subtests.test(msg="mask-cache-needs-a-signature"):
        widget = panel()
        masks = _synthetic_masks(2)
        widget._pending_signature = None
        widget._on_worker_done({"masks": masks, "tracks": None,
                                "masks_built": True, "images": None}, "")
        assert widget._mask_cache == {}
        assert widget._status.text().startswith("Masks built + linked")
        widget._pending_signature = ("sig", 1)
        widget._on_worker_done({"masks": masks, "tracks": None,
                                "masks_built": False, "images": None}, "")
        assert list(widget._mask_cache) == [("sig", 1)]
        assert widget._status.text().startswith("Re-linked (cached masks)")

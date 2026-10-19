"""Scrubbing the tracking preview converts each frame once, not each step.

Lesson 17 plays the Timelapse preview inside Make Masks. Every step of the
scrub bar stretched the frame by percentile twice, coloured its outlines by
masking the whole field once per track and converted two pictures, and the
window's paint rate fell from about 50 to 25 frames a second. These tests
count the conversions and the scene rebuilds instead of timing them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets import timelapse_movie as TM  # noqa: E402
from spacr.qt.widgets import timelapse_preview as TP  # noqa: E402

pytestmark = pytest.mark.qt

FRAMES = 6
SIZE = 48


def _labels():
    stack = np.zeros((FRAMES, SIZE, SIZE), np.int32)
    yy, xx = np.ogrid[:SIZE, :SIZE]
    for t in range(FRAMES):
        for label, (cy, cx) in enumerate(((12, 10), (34, 30)), start=1):
            stack[t][(yy - cy) ** 2 + (xx - cx - 2 * t) ** 2 <= 16] = label
    return stack


def _tracks(labels):
    rows = []
    for t in range(labels.shape[0]):
        for label in (1, 2):
            ys, xs = np.nonzero(labels[t] == label)
            rows.append(dict(frame=t, track_id=label, label=label,
                             x=float(xs.mean()), y=float(ys.mean())))
    return pd.DataFrame(rows)


@pytest.fixture()
def panel(qtbot, tmp_path):
    import tifffile
    folder = tmp_path / "frames"
    folder.mkdir()
    labels = _labels()
    rng = np.random.default_rng(3)
    for t in range(FRAMES):
        image = (labels[t] > 0) * 700 + rng.integers(0, 90, (SIZE, SIZE))
        tifffile.imwrite(str(folder / f"field_t{t}.tif"), image.astype(np.uint16))
    widget = TP.TimelapsePreviewPanel(threaded=False)
    qtbot.addWidget(widget)
    widget.resize(900, 700)
    widget.show()
    assert widget.load_sequence(str(folder))
    widget._masks = labels
    widget._tracks = _tracks(labels)
    widget._tracked = labels.copy()
    widget._frame_slider.setMaximum(FRAMES - 1)
    widget._refresh_canvases()
    return widget


def _count(monkeypatch, module, name, counts):
    original = getattr(module, name)

    def counted(*args, **kwargs):
        counts[name] = counts.get(name, 0) + 1
        return original(*args, **kwargs)
    monkeypatch.setattr(module, name, counted)


def test_a_second_pass_over_the_frames_converts_nothing(panel, monkeypatch):
    counts = {}
    for name in ("numpy_to_qpixmap", "_stretch_levels", "_overlay_tracks"):
        _count(monkeypatch, TP, name, counts)
    reads = []
    original = panel._sequence.frame
    monkeypatch.setattr(panel._sequence, "frame",
                        lambda i: reads.append(i) or original(i))
    for index in range(FRAMES):
        panel._frame_slider.setValue(index)
    first = dict(counts)
    assert first["numpy_to_qpixmap"] == 2 * (FRAMES - 1)
    assert first["_stretch_levels"] == FRAMES - 1
    counts.clear()
    reads.clear()
    for _ in range(3):
        for index in reversed(range(FRAMES)):
            panel._frame_slider.setValue(index)
        for index in range(FRAMES):
            panel._frame_slider.setValue(index)
    assert counts == {}
    assert reads == []


def test_stepping_keeps_the_scene_item_and_the_zoom(panel):
    items = (panel._src_view._pixmap_item, panel._out_view._pixmap_item)
    scenes = (len(panel._src_view.scene().items()),
              len(panel._out_view.scene().items()))
    panel._src_view._apply_zoom(2.0, broadcast=True)
    transform = panel._src_view.transform()
    for index in range(1, FRAMES):
        panel._frame_slider.setValue(index)
    assert (panel._src_view._pixmap_item, panel._out_view._pixmap_item) == items
    assert (len(panel._src_view.scene().items()),
            len(panel._out_view.scene().items())) == scenes
    assert panel._src_view.transform() == transform
    assert panel._src_view.scale_factor() == pytest.approx(2.0)


def test_new_tracks_or_drawing_settings_draw_the_frame_again(panel, monkeypatch):
    counts = {}
    _count(monkeypatch, TP, "_overlay_tracks", counts)
    panel._refresh_canvases()
    assert counts == {}
    panel._tracks = _tracks(panel._masks).assign(track_id=lambda d: d.track_id + 4)
    panel._refresh_canvases()
    assert counts == {"_overlay_tracks": 1}
    panel._tail.setValue(3)
    assert counts == {"_overlay_tracks": 2}
    panel._tail.setValue(12)
    assert counts == {"_overlay_tracks": 2}


def test_the_kept_pictures_are_bounded_and_offered_to_the_budget(panel, monkeypatch):
    monkeypatch.setattr(TP, "_FRAME_PICTURE_BUDGET_BYTES",
                        3 * 2 * TP._pixmap_bytes(panel._src_view.picture()))
    for index in range(FRAMES):
        panel._frame_slider.setValue(index)
    assert len(panel._frame_pictures) == 3
    entries = [row for row in panel._cache_budget_entries()
               if row[0][:1] == ("frame picture",)]
    assert len(entries) == 3
    assert panel._drop_cache_budget_entry(entries[0][0]) is True
    assert len(panel._frame_pictures) == 2


def test_a_movie_longer_than_the_budget_keeps_its_first_frames(panel, monkeypatch):
    counts = {}
    for name in ("numpy_to_qpixmap", "_stretch_levels"):
        _count(monkeypatch, TP, name, counts)
    monkeypatch.setattr(TP, "_FRAME_PICTURE_BUDGET_BYTES",
                        2 * 2 * TP._pixmap_bytes(panel._src_view.picture()))
    for index in list(range(FRAMES)) + [0]:
        panel._frame_slider.setValue(index)
    counts.clear()
    for index in range(FRAMES):
        panel._frame_slider.setValue(index)
    assert sorted(key[0] for key in panel._frame_pictures) == [0, 1]
    assert counts == {"numpy_to_qpixmap": 2 * (FRAMES - 2)}


def test_the_kept_levels_stretch_as_the_shared_conversion_does():
    plane = np.random.default_rng(5).integers(0, 4000, (40, 30)).astype(np.uint16)
    assert np.array_equal(TP._stretched(plane, TP._stretch_levels(plane)),
                          TP._to_uint8(plane))
    flat = np.full((8, 8), 7, np.uint16)
    assert np.array_equal(TP._stretched(flat, TP._stretch_levels(flat)),
                          TP._to_uint8(flat))


def test_outlines_match_masking_the_field_once_per_track():
    labels = _labels()[2]
    image = np.arange(SIZE * SIZE, dtype=np.uint16).reshape(SIZE, SIZE)
    drawn = TP.render_frame(image, labels=labels, tracks=None, frame=2)
    base = TP._to_uint8(image)
    expected = np.stack([base] * 3, axis=-1)
    edge = TP._boundary_mask(labels) & (labels > 0)
    for tid in np.unique(labels[edge]):
        expected[edge & (labels == tid)] = TP.track_colour(int(tid))
    assert np.array_equal(drawn, expected)


def test_moving_the_film_strip_ring_restyles_two_cells(qtbot):
    from PySide6.QtGui import QPixmap
    strip = TM.FilmStrip()
    qtbot.addWidget(strip)
    pixmap = QPixmap(8, 8)
    strip.set_frames([pixmap] * 10)
    strip.highlight(0)
    calls = []
    for index, cell in enumerate(strip._cells):
        original = cell.setStyleSheet
        cell.setStyleSheet = (lambda sheet, i=index, f=original:
                              calls.append(i) or f(sheet))
    strip.highlight(5)
    assert sorted(calls) == [0, 5]
    assert "transparent" not in strip._cells[5].styleSheet()
    assert "transparent" in strip._cells[0].styleSheet()

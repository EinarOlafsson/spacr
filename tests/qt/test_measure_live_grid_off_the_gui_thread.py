"""Picking a field or ticking a checked image in Measure Live keeps the window moving.

Lesson 08's capture froze right after a field was picked (clip
preview_05_live_all_channels) and after each tick in the "Checked images"
menu (clip preview_07c_checked_two_fields). The crop pass already ran on a
worker, but it scanned the whole mask once per label and converted and
scanned every companion plane once per crop, so a three-field pass took two
seconds; the first pass imported spacr.measure on that worker without the
heavy-import lock; and the grid then drew every thumbnail, 40 ms and more
at 2560 x 1440, in one event-loop turn.

Now the pass counts labels in one sweep, looks inside each cell's box only,
runs under the responsive switch interval, draws the thumbnails as QImages on
the worker, and the grid fills in slices. Every result is the same as before.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest
from PySide6.QtCore import QElapsedTimer, QTimer
from PySide6.QtGui import QImage, QPixmap

import spacr.measure as M
import spacr.qt.widgets.measure_preview as MP
from spacr.qt.hidpi import scaled_for


def _labels(size=240, step=16, seed=3, dtype=np.uint16):
    """A label image of irregular blobs with gaps in the numbering."""
    rng = np.random.default_rng(seed)
    mask = np.zeros((size, size), dtype)
    label = 0
    for y in range(0, size - step, step):
        for x in range(0, size - step, step):
            label += int(rng.integers(1, 3))
            h, w = rng.integers(3, step - 2, 2)
            mask[y:y + h, x:x + w] = label
            if label % 5 == 0:
                mask[y + h - 1, x + w:x + w + 2] = label
    return mask


def _merged(size=240, seed=3):
    """A merged field: three image channels, then cell, nucleus, pathogen."""
    rng = np.random.default_rng(seed)
    cell = _labels(size, 16, seed)
    nucleus = np.where(_labels(size, 16, seed) > 0, _labels(size, 8, seed + 1), 0)
    pathogen = np.where(cell % 3 == 0, _labels(size, 8, seed + 2), 0)
    image = rng.integers(0, 60000, (size, size, 3)).astype(np.uint16)
    return np.dstack([image, cell, nucleus.astype(np.uint16),
                      pathogen.astype(np.uint16)])


def _old_crop_objects(data, mask_dim, buffer=10, limit=None, min_area=0,
                      max_area=0, **kwargs):
    """The per-label scan this commit replaced, kept as the reference."""
    mask = data[:, :, int(mask_dim)]
    labels = np.unique(mask)
    labels = labels[labels > 0]
    scored = []
    for lbl in labels:
        area = int(np.sum(mask == lbl))
        if min_area and area < min_area:
            continue
        if max_area and area > max_area:
            continue
        scored.append((area, int(lbl)))
    scored.sort(reverse=True)
    if limit:
        scored = scored[:int(limit)]
    boxes = []
    for area, lbl in scored:
        ys, xs = np.where(mask == lbl)
        boxes.append((lbl, area, (int(max(0, ys.min() - buffer)),
                                  int(min(mask.shape[0], ys.max() + 1 + buffer)),
                                  int(max(0, xs.min() - buffer)),
                                  int(min(mask.shape[1], xs.max() + 1 + buffer)))))
    return boxes


def _old_annotate(crops, data, params):
    """The whole-plane annotation this commit replaced, as the reference."""
    cell_mask = data[..., params["cell_dim"]].astype(np.int64, copy=False)
    out = []
    for entry in crops:
        region = cell_mask == int(entry["label"])
        out.append(tuple(
            MP._presence_in(data, params["dims"].get(name), region,
                            int(params["minima"].get(name, 0)))
            for name in ("nucleus", "pathogen", "organelle")))
    return out


@pytest.mark.parametrize("dtype", [np.uint16, np.int32, np.uint8, np.int64])
@pytest.mark.parametrize("kwargs", [
    {}, {"limit": 7}, {"min_area": 40, "max_area": 120}, {"buffer": 0},
])
def test_the_one_sweep_crops_the_same_objects(dtype, kwargs):
    data = _merged()
    mask = data[..., 3].astype(dtype)
    data = np.dstack([data[..., :3].astype(np.float64), mask])
    expected = _old_crop_objects(data, 3, **kwargs)
    got = M.crop_objects_from_array(data, mask_dim=3, channels=(0, 1, 2),
                                    **kwargs)
    assert [(e["label"], e["area"], e["bbox"]) for e in got] == expected
    assert all(type(e["label"]) is int and type(e["area"]) is int for e in got)


@pytest.mark.parametrize("mask", [
    np.zeros((20, 20), np.uint16),
    np.array([[0, -1, 2], [2, 0, 3]], np.int32),
    np.array([[0, 1.0, 2.0], [2.0, 0, 1.0]], np.float32),
    np.array([[0, 7, 7], [9, 0, 0]], np.uint64),
])
def test_masks_the_sweep_cannot_count_still_crop_as_before(mask):
    data = np.dstack([np.ones(mask.shape), mask.astype(np.float64)
                      if mask.dtype.kind == "f" else mask])
    expected = _old_crop_objects(data, 1, buffer=1)
    got = M.crop_objects_from_array(data, mask_dim=1, channels=(0,), buffer=1)
    assert [(e["label"], e["area"], e["bbox"]) for e in got] == expected


def test_a_fractional_label_fails_as_it_always_did():
    mask = np.array([[0, 1.5], [1.5, 0]])
    data = np.dstack([np.ones(mask.shape), mask])
    with pytest.raises(ValueError):
        _old_crop_objects(data, 1)
    with pytest.raises(ValueError):
        M.crop_objects_from_array(data, mask_dim=1, channels=(0,))


def test_the_sweep_gives_up_on_a_sparse_huge_label():
    mask = np.zeros((4, 4), np.int64)
    mask[0, 0] = 10 ** 9
    assert M._label_areas_and_boxes(mask) is None
    assert M._label_areas_and_boxes(np.zeros((0, 3), np.uint16)) is None


PARAMS = [
    {"object": "cell", "cell_dim": 3, "dims": {"nucleus": 4, "pathogen": 5,
                                               "organelle": None},
     "minima": {}, "uninfected": False},
    {"object": "cell", "cell_dim": 3, "dims": {"nucleus": 4, "pathogen": 5,
                                               "organelle": 4},
     "minima": {"nucleus": 20, "pathogen": 6, "organelle": 400},
     "uninfected": True},
]


@pytest.mark.parametrize("params", PARAMS)
@pytest.mark.parametrize("boxed", [True, False])
def test_annotating_in_the_cell_box_gives_the_whole_plane_answer(params, boxed):
    data = _merged()
    crops = M.crop_objects_from_array(data, mask_dim=3, channels=(0, 1, 2))
    MP._annotate_crops(crops, data, params, mask_dim=3 if boxed else None)
    got = [tuple(e["phenotype"][k] for k in ("nucleus", "pathogen",
                                             "organelle")) for e in crops]
    assert got == _old_annotate(crops, data, params)
    assert {e["category"] for e in crops}


def test_a_float_or_negative_companion_plane_counts_as_before():
    data = _merged().astype(np.float64)
    data[..., 4] = np.where(data[..., 4] > 0, data[..., 4] + 0.5, 0)
    data[..., 5] = -data[..., 5]
    params = dict(PARAMS[1], dims={"nucleus": 4, "pathogen": 5,
                                   "organelle": 9})
    crops = M.crop_objects_from_array(data, mask_dim=3, channels=(0,))
    MP._annotate_crops(crops, data, params, mask_dim=3)
    got = [tuple(e["phenotype"][k] for k in ("nucleus", "pathogen",
                                             "organelle")) for e in crops]
    assert got == _old_annotate(crops, data, params)
    planes = MP._CompanionPlanes(data)
    assert planes.area(5, -3) == int(np.count_nonzero(data[..., 5] == -3))
    assert planes.area(4, 10 ** 7) == 0 or planes.area(4, 10 ** 7) is not None


def test_a_label_past_the_counts_has_no_area():
    planes = MP._CompanionPlanes(_merged())
    assert planes.area(3, 10 ** 6) == 0


class _Ratio:
    """Something with a device pixel ratio, for :func:`scaled_for`."""

    def __init__(self, ratio):
        self.ratio = ratio

    def devicePixelRatioF(self):
        return self.ratio


@pytest.mark.parametrize("ratio", [1.0, 1.5, 2.0])
@pytest.mark.parametrize("primaries", ["rgb", "cmy"])
def test_a_worker_thumbnail_has_the_pixels_of_the_gui_one(qapp, ratio,
                                                          primaries):
    rng = np.random.default_rng(int(ratio * 10))
    crop = rng.integers(0, 255, (37, 53, 3)).astype(np.uint8)
    opaque = MP._opaque_pixmap_format()
    worker = MP._thumbnail_image(crop, primaries, 132, ratio, opaque)
    array = MP._preview_crop_pixels(crop, primaries)
    image = QImage(array.data, array.shape[1], array.shape[0],
                   3 * array.shape[1], QImage.Format_RGB888)
    gui = MP._rounded_pixmap(scaled_for(QPixmap.fromImage(image.copy()),
                                        _Ratio(ratio), 132), radius=8)
    shown = QPixmap.fromImage(worker)
    assert shown.devicePixelRatio() == gui.devicePixelRatio()
    assert shown.toImage() == gui.toImage()


def test_thumbnails_are_drawn_on_a_worker_thread(qapp):
    seen = []
    real = MP._thumbnail_image

    def spy(*args):
        seen.append(threading.current_thread() is threading.main_thread())
        return real(*args)

    data = _merged()
    crops = M.crop_objects_from_array(data, mask_dim=3, channels=(0, 1, 2))
    spec = ("rgb", 132, 1.0, MP._opaque_pixmap_format())
    thread = threading.Thread(target=lambda: seen.append(
        len(MP._thumbnail_images(crops, spec))))
    MP._thumbnail_image, saved = spy, MP._thumbnail_image
    try:
        thread.start()
        thread.join()
    finally:
        MP._thumbnail_image = saved
    assert seen[-1] == len(crops) and not any(seen[:-1])


def _panel(qtbot, threaded, folder):
    panel = MP.MeasurePreviewPanel(threaded=threaded)
    qtbot.addWidget(panel)
    panel.resize(1200, 900)
    panel.show()
    panel._mask_dims["cell"].setValue(3)
    panel._mask_dims["nucleus"].setValue(4)
    panel._mask_dims["pathogen"].setValue(5)
    panel._max_crops.setValue(400)
    panel._group_cells.setChecked(False)
    return panel


@pytest.fixture
def fields(tmp_path):
    folder = tmp_path / "merged"
    folder.mkdir()
    paths = []
    for index in range(3):
        path = folder / f"plate1_A01_{index + 1}_1.npy"
        np.save(path, _merged(320, seed=index + 3))
        paths.append(str(path.resolve()))
    return paths


def _idle(panel):
    return not panel._loads_in_flight and not panel._crop_running


def _grid_pictures(panel):
    """Every thumbnail in grid order, as (row, column, pixels, tooltip)."""
    out = []
    for i in range(panel._grid.count()):
        widget = panel._grid.itemAt(i).widget()
        if isinstance(widget, MP._CropThumb):
            row, column, _, _ = panel._grid.getItemPosition(i)
            out.append((row, column, widget.pixmap().toImage(),
                        widget.toolTip()))
    return sorted(out, key=lambda item: (item[0], item[1]))


def _load(qtbot, panel, path):
    panel.load_array(path)
    qtbot.waitUntil(lambda: _idle(panel) and bool(panel._crops),
                    timeout=60000)


def test_the_threaded_grid_is_the_grid_drawn_inline(qtbot, fields):
    inline = _panel(qtbot, False, fields)
    threaded = _panel(qtbot, True, fields)
    for panel in (inline, threaded):
        _load(qtbot, panel, fields[0])
        for path in fields[1:]:
            panel._set_source_checked(path, True)
            qtbot.waitUntil(lambda: _idle(panel), timeout=60000)
    assert len(threaded._crops) == len(inline._crops) > 40
    assert threaded._thumb_images, "the worker drew no thumbnails"
    a, b = _grid_pictures(inline), _grid_pictures(threaded)
    assert len(a) == len(b) == len(inline._crops)
    for left, right in zip(a, b):
        assert left[:2] == right[:2] and left[3] == right[3]
        assert left[2] == right[2]
    threaded.shutdown()
    inline.shutdown()


class _Heartbeat:
    """The longest gap between 1 ms timer ticks on the GUI thread."""

    def __init__(self):
        self.timer = QTimer()
        self.timer.setInterval(1)
        self.clock = QElapsedTimer()
        self.worst = 0
        self.ticks = 0
        self.timer.timeout.connect(self._tick)

    def start(self):
        self.clock.start()
        self.timer.start()

    def _tick(self):
        self.worst = max(self.worst, self.clock.restart())
        self.ticks += 1

    def stop(self):
        self.timer.stop()


def test_filling_the_grid_never_holds_the_loop_for_the_whole_grid(qtbot,
                                                                   fields):
    """Self-calibrated: the inline build is timed on this machine, now."""
    panel = _panel(qtbot, True, fields)
    _load(qtbot, panel, fields[0])
    for path in fields[1:]:
        panel._set_source_checked(path, True)
        qtbot.waitUntil(lambda: _idle(panel), timeout=60000)
    assert len(panel._crops) > 40
    panel._threaded = False
    start = time.perf_counter()
    panel._render_grid()
    inline_ms = (time.perf_counter() - start) * 1000
    panel._threaded = True
    qtbot.wait(50)
    beat = _Heartbeat()
    beat.start()
    panel._render_grid()
    qtbot.waitUntil(lambda: not panel._render_plan, timeout=20000)
    qtbot.wait(30)
    beat.stop()
    assert beat.ticks > 5
    if inline_ms < 4 * MP._RENDER_SLICE_S * 1000:
        pytest.skip(f"the inline grid took {inline_ms:.0f} ms here; too "
                    f"quick to tell slices from one block")
    assert beat.worst < inline_ms / 2, (beat.worst, inline_ms)
    panel.shutdown()


def test_ticking_a_checked_image_returns_at_once(qtbot, fields):
    """Self-calibrated against the same pass run on the GUI thread."""
    panel = _panel(qtbot, True, fields)
    _load(qtbot, panel, fields[0])
    start = time.perf_counter()
    MP._compute_checked_crops(tuple(fields[:2]), fields[0], panel._data,
                              dict(mask_dim=3, channels=[0, 1, 2], limit=400),
                              panel._category_params(), threading.Event())
    inline_ms = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    panel._set_source_checked(fields[1], True)
    tick_ms = (time.perf_counter() - start) * 1000
    qtbot.waitUntil(lambda: _idle(panel), timeout=60000)
    assert {c["source_path"] for c in panel._crops} == set(fields[:2])
    assert tick_ms < max(inline_ms / 2, 15), (tick_ms, inline_ms)
    panel.shutdown()


def test_a_newer_render_abandons_the_older_one(qtbot, fields):
    panel = _panel(qtbot, True, fields)
    _load(qtbot, panel, fields[0])
    placed = []
    real = panel._place_thumb
    panel._place_thumb = lambda *a: (placed.append(a[0]), real(*a))[1]
    MP._RENDER_SLICE_S, saved = 0.0, MP._RENDER_SLICE_S
    try:
        panel._render_grid()
        first = list(placed)
        assert len(first) <= 1 and panel._render_plan
        panel._render_grid()
        qtbot.waitUntil(lambda: not panel._render_plan, timeout=20000)
        qtbot.wait(30)
    finally:
        MP._RENDER_SLICE_S = saved
    assert len(placed) == len(first) + len(panel._crops)
    assert len(_grid_pictures(panel)) == len(panel._crops)
    panel.shutdown()


def test_a_recrop_or_shutdown_stops_the_fill(qtbot, fields):
    panel = _panel(qtbot, True, fields)
    _load(qtbot, panel, fields[0])
    MP._RENDER_SLICE_S, saved = 0.0, MP._RENDER_SLICE_S
    try:
        panel._render_grid()
        assert panel._render_plan and panel._loads_in_flight
        panel.shutdown()
        assert not panel._render_plan
    finally:
        MP._RENDER_SLICE_S = saved


def test_a_changed_scale_or_palette_draws_on_the_gui_thread(qtbot, fields,
                                                            monkeypatch):
    panel = _panel(qtbot, True, fields)
    _load(qtbot, panel, fields[0])
    drawn = []
    real = panel._crop_pixmap
    panel._crop_pixmap = lambda crop: (drawn.append(1), real(crop))[1]
    panel._render_grid()
    qtbot.waitUntil(lambda: not panel._render_plan, timeout=20000)
    assert drawn == []
    panel._thumb_px += 10
    panel._render_grid()
    qtbot.waitUntil(lambda: not panel._render_plan, timeout=20000)
    assert len(drawn) == len(panel._crops)
    drawn.clear()
    panel._thumb_px -= 10
    entry = panel._crops[0]
    panel._thumb_images[0] = (entry["crop"].copy(), panel._thumb_images[0][1])
    panel._render_grid()
    qtbot.waitUntil(lambda: not panel._render_plan, timeout=20000)
    assert drawn == [1]
    panel._drawn_spec = None
    assert not panel._thumb_spec_matches()
    panel.shutdown()


def test_a_thumbnail_failure_leaves_the_drawing_to_the_gui(monkeypatch):
    def boom(*_args):
        raise RuntimeError("no painter")

    monkeypatch.setattr(MP, "_thumbnail_images", boom)
    monkeypatch.setattr(MP, "_import_the_cropper", lambda: None)
    monkeypatch.setattr(MP, "_compute_checked_crops",
                        lambda *a: {"crops": [{"crop": None}]})
    result = MP._crop_pass((), None, None, {}, {}, threading.Event(),
                           None, ("rgb", 1, 1.0, None))
    assert "thumbnails" not in result


def test_the_first_pass_imports_the_cropper_like_the_run(monkeypatch):
    import spacr.qt.bridge as bridge

    calls = []
    monkeypatch.setattr(bridge, "_module_is_imported", lambda name: False)
    monkeypatch.setattr(bridge, "_import_the_entry_module", calls.append)
    MP._import_the_cropper()
    assert calls == ["spacr.measure"]
    monkeypatch.setattr(bridge, "_module_is_imported", lambda name: True)
    MP._import_the_cropper()
    assert calls == ["spacr.measure"]


def test_the_pass_holds_the_responsive_switch_interval(monkeypatch):
    import spacr.qt.gil_priority as gil

    seen = []
    monkeypatch.setattr(MP, "_import_the_cropper", lambda: None)
    monkeypatch.setattr(MP, "_compute_checked_crops",
                        lambda *a: (seen.append(gil.active()),
                                    {"crops": []})[1])
    MP._crop_pass((), None, None, {}, {}, threading.Event())
    assert seen == [True] and not gil.active()


@pytest.mark.parametrize('dtype', [np.int32, np.uint64])
def test_raw_label_types_preserve_object_areas_boxes_pixels_and_source(dtype):
    mask = np.array([[0, 1, 1, 0, 0],
                     [0, 1, 1, 0, 0],
                     [0, 0, 0, 3, 0],
                     [0, 0, 0, 3, 0]], dtype=dtype)
    if np.issubdtype(dtype, np.signedinteger):
        mask[0, 0] = -1
    image = np.arange(20, dtype=dtype).reshape(mask.shape) + 1000
    data = np.dstack([image, mask])
    before = data.copy()
    expected = _old_crop_objects(data, 1, buffer=0)
    counted = M._label_areas_and_boxes(data[..., 1])
    if np.issubdtype(dtype, np.signedinteger):
        assert counted is None, 'Negative background labels require the safe per-label fallback'
    else:
        areas, boxes = counted
        assert areas == [(1, 4), (3, 2)]
        assert boxes == [(slice(0, 2), slice(1, 3)), None,
                         (slice(2, 4), slice(3, 4))]
    crops = M.crop_objects_from_array(data, 1, channels=(0,), buffer=0,
                                     normalize=False, to_rgb=False)
    assert [(entry['label'], entry['area'], entry['bbox']) for entry in crops] == expected
    assert [entry['label'] for entry in crops] == [1, 3]
    for entry in crops:
        top, bottom, left, right = entry['bbox']
        expected_pixels = image[top:bottom, left:right].copy()
        expected_pixels[mask[top:bottom, left:right] != entry['label']] = 0
        np.testing.assert_array_equal(entry['crop'], expected_pixels[..., None])
        assert entry['crop'].dtype == dtype
    np.testing.assert_array_equal(data, before)

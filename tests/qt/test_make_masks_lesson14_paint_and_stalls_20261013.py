"""Lesson 14 (Make Masks) on the capture host: one field, light redraws, a live wand.

The capture host measured Make Masks painting 105 full windows a second
against 58 on Home, and froze sixteen editor clips. Offscreen at 2560x1440
the two causes were:

* A SECOND FIELD. ``MainWindow._install_screen_backdrop`` gave Make Masks
  (not an ``AppScreen``) an ambient field of its own on top of the window's,
  each ticking on its own timer, so every field beat reached the window as
  two flushes: 60 a second offscreen against 30 on Home.
* THE CANVAS REDRAW. Every stroke, click, undo, filter change and resize
  stretched and blended the whole 2,000 px field in float on the GUI thread
  (184 ms for the blend alone), and a resize did it again although nothing
  but the size had changed. A capped Wand click ran three watersheds over
  most of a megapixel there too.

Every timing here is measured against a reference run in the same process,
so a slow or loaded machine moves both sides together.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtCore import QEvent, QObject, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QMouseEvent  # noqa: E402

from spacr.qt import mask_engine as engine  # noqa: E402
from spacr.qt import wand_rescue as wr  # noqa: E402
from spacr.qt.screens import make_masks as mm  # noqa: E402

pytestmark = pytest.mark.qt


def _old_stretch(image, lo, hi):
    """``_stretch_between`` as it was: four float passes."""
    out = np.clip(image, lo, hi)
    out = (out - lo) / (hi - lo)
    return (out * float(np.iinfo(image.dtype).max)).astype(image.dtype)


def _old_overlay(image, mask, alpha=0.5):
    """``overlay_mask`` as it was: a float blend of every channel of every pixel."""
    if image.ndim == 2:
        image = np.stack((image,) * 3, axis=-1)
    m = mask.astype(np.int32)
    max_label = int(np.max(m)) if m.size else 0
    rng = np.random.default_rng(0)
    colors = rng.integers(30, 255, size=(max_label + 1, 3), dtype=np.uint8)
    if max_label >= 0:
        colors[0] = [0, 0, 0]
    colored = colors[m]
    image_8bit = (image.astype(np.float32) / 256.0).clip(0, 255).astype(np.uint8)
    return np.where(m[..., None] > 0,
                    np.clip(image_8bit * (1 - alpha) + colored * alpha, 0, 255),
                    image_8bit).astype(np.uint8)


def _field(n=2000, seed=3):
    """A cell-like 16-bit field and dense labels, the size of the lesson's."""
    rng = np.random.default_rng(seed)
    image = rng.gamma(2.0, 900.0, (n, n)).astype(np.uint16)
    labels = np.zeros((n, n), np.uint16)
    yy, xx = np.ogrid[:n, :n]
    step = max(8, n // 28)
    label = 0
    for cy in range(step // 2, n, step):
        for cx in range(step // 2, n, step):
            label += 1
            disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= (step * 0.42) ** 2
            labels[disk] = label
    return image, labels


def _best(fn, repeats=3):
    """The shortest of ``repeats`` runs of ``fn``, in seconds."""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return min(times)


# ---------------------------------------------------------------------------
# The picture: byte for byte what it was
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label_dtype", [np.uint8, np.uint16, np.int32, np.uint32, bool])
@pytest.mark.parametrize("image_dtype", [np.uint8, np.uint16])
def test_the_canvas_picture_is_the_two_float_passes_to_the_byte(image_dtype, label_dtype):
    image, labels = _field(300, seed=5)
    if image_dtype == np.uint8:
        image = (image >> 8).astype(np.uint8)
    labels = (labels > 0) if label_dtype is bool else labels.astype(label_dtype)
    lo, hi = engine._percentile_bounds(image, 1.0, 99.9)
    for view, part in ((image, labels), (image[7:290:3, 11:250:2], labels[7:290:3, 11:250:2]),
                       (image.T, labels.T), (image[::2].T, np.asfortranarray(labels[::2].T))):
        fast = mm._canvas_picture(view, part, lo, hi)
        slow = mm._rgb32(_old_overlay(_old_stretch(view, lo, hi), part))
        assert fast.dtype == np.uint32 and fast.flags.c_contiguous
        np.testing.assert_array_equal(fast, slow)


def test_a_float_field_or_negative_labels_take_the_general_path():
    image = np.linspace(0, 1, 60 * 40, dtype=np.float32).reshape(60, 40)
    labels = np.zeros((60, 40), np.int32)
    labels[10:20, 5:15] = 3
    with pytest.raises(ValueError):
        mm._canvas_picture(image, labels, 0.1, 0.9)
    field = (np.arange(60 * 40, dtype=np.uint16) * 7).reshape(60, 40)
    labels[0, 0] = -1
    np.testing.assert_array_equal(
        mm._canvas_picture(field, labels, 10.0, 9000.0),
        mm._rgb32(engine.overlay_mask(engine._stretch_between(field, 10.0, 9000.0),
                                      labels, alpha=0.5)))


@pytest.mark.parametrize("alpha", [0.5, 0.3, np.float32(0.7), 1.0])
@pytest.mark.parametrize("label_dtype", [np.uint8, np.uint16, np.int16, np.int32, np.uint32])
def test_overlay_mask_is_the_float_blend_to_the_byte(alpha, label_dtype):
    image, labels = _field(240, seed=11)
    labels = labels.astype(label_dtype)
    np.testing.assert_array_equal(engine.overlay_mask(image, labels, alpha),
                                  _old_overlay(image, labels, alpha))
    small = (image >> 8).astype(np.uint8)
    np.testing.assert_array_equal(engine.overlay_mask(small, labels, alpha),
                                  _old_overlay(small, labels, alpha))
    empty = np.zeros_like(labels)
    np.testing.assert_array_equal(engine.overlay_mask(image, empty, alpha),
                                  _old_overlay(image, empty, alpha))


def test_the_stretch_table_is_the_float_stretch_to_the_byte():
    image, _ = _field(700, seed=2)
    for lo, hi in ((0.0, 1.0), (100.5, 30000.25), (np.float64(7), np.float64(8)),
                   engine._percentile_bounds(image, 1.0, 99.9)):
        np.testing.assert_array_equal(engine._stretch_between(image, lo, hi),
                                      _old_stretch(image, lo, hi))
        view = image[3::2, 5::3]
        np.testing.assert_array_equal(engine._stretch_between(view, lo, hi),
                                      _old_stretch(view, lo, hi))
    small = (image >> 8).astype(np.uint8)
    np.testing.assert_array_equal(engine._stretch_between(small, 3.0, 200.0),
                                  _old_stretch(small, 3.0, 200.0))


# ---------------------------------------------------------------------------
# The canvas: a redraw costs a fraction of what it did, a resize only scales
# ---------------------------------------------------------------------------

@pytest.fixture
def field_canvas(qtbot, qt_theme_applied):
    """A canvas at the lesson's size holding a 2,000 px field and dense labels."""
    canvas = mm._MaskCanvas()
    qtbot.addWidget(canvas)
    canvas.resize(1300, 1150)
    canvas.show()
    qtbot.waitExposed(canvas)
    image, labels = _field()
    canvas.set_image_and_mask(image, labels)
    return canvas


def test_a_redraw_costs_a_fraction_of_the_float_passes(field_canvas, monkeypatch):
    """Self-calibrated: the whole refresh against the old compose alone."""
    monkeypatch.setattr(mm, "device_ratio", lambda _w: 1.0)
    canvas = field_canvas
    x0, y0, x1, y1 = canvas._viewport_bounds()
    step = engine._display_step(x1 - x0, y1 - y0, max(200, canvas.width()),
                                max(200, canvas.height()))
    source = canvas.displayed_source()
    lo, hi = canvas._display_bounds(source)
    view = source[y0:y1:step, x0:x1:step]
    part = canvas.mask[y0:y1:step, x0:x1:step]
    reference = _best(lambda: _old_overlay(_old_stretch(view, lo, hi), part))
    redraw = _best(canvas.refresh)
    assert redraw <= 0.5 * reference, (
        f"refresh {redraw * 1000:.0f} ms against {reference * 1000:.0f} ms for "
        "the float stretch and blend alone")


def test_a_resize_scales_the_kept_picture_instead_of_composing_again(
        field_canvas, monkeypatch):
    canvas = field_canvas
    composed = []
    real = mm._canvas_picture

    def counting(*args):
        composed.append(1)
        return real(*args)

    monkeypatch.setattr(mm, "_canvas_picture", counting)
    canvas.refresh()
    assert composed == [1]
    shown = canvas.pixmap().toImage()
    canvas.resize(1290, 1140)
    canvas.resize(1300, 1150)
    assert composed == [1], "a resize composed the whole field again"
    assert canvas.pixmap().toImage() == shown
    canvas.mask = canvas.mask.copy()
    canvas.resize(1280, 1130)
    assert composed == [1, 1], "a resize kept a picture of another mask"
    canvas.norm_hi = 99.0
    canvas.resize(1300, 1150)
    assert composed == [1, 1, 1], "a resize kept a picture at other levels"


def test_a_load_draws_the_field_once(qtbot, qt_theme_applied, monkeypatch):
    canvas = mm._MaskCanvas()
    qtbot.addWidget(canvas)
    canvas.resize(600, 500)
    composed = []
    real = mm._canvas_picture
    monkeypatch.setattr(mm, "_canvas_picture",
                        lambda *a: composed.append(1) or real(*a))
    image, labels = _field(400)
    canvas.set_image_and_mask(image, labels)
    assert composed == [1]


# ---------------------------------------------------------------------------
# The wand: a full field floods on a worker and lands exactly
# ---------------------------------------------------------------------------

def _capped_runaway(n=1100):
    """A flat field a 0.5 % tolerance runs away over, as the lesson's click does."""
    rng = np.random.default_rng(9)
    image = (20000 + rng.integers(0, 40, (n, n))).astype(np.uint16)
    image[:, n // 2:n // 2 + 6] = 60000
    return image


def _press(canvas, x, y, modifiers=Qt.NoModifier):
    point = canvas._image_to_canvas(x, y)
    pos = QPointF(float(point.x()), float(point.y()))
    return QMouseEvent(QEvent.Type.MouseButtonPress, pos, pos,
                       Qt.LeftButton, Qt.LeftButton, modifiers)


@pytest.fixture
def wand_field(qtbot, qt_theme_applied):
    canvas = mm._MaskCanvas()
    qtbot.addWidget(canvas)
    canvas.resize(900, 900)
    image = _capped_runaway()
    canvas.set_image_and_mask(image, np.zeros(image.shape, np.uint16))
    canvas.mode = mm.MODE_WAND_ADD
    canvas.wand_relative = False
    canvas.wand_tolerance = 60.0
    canvas.wand_max_pixels = 256
    yield canvas
    canvas.close_wand()


def test_a_full_field_wand_click_returns_at_once_and_lands_exactly(qtbot, wand_field):
    canvas = wand_field
    press = _press(canvas, 200, 300)
    x, y = canvas._canvas_to_image(press.position().x(), press.position().y())
    expected, report = wr.magic_wand(
        canvas.image, canvas.mask, x, y, 60.0, 256, action="add",
        **canvas.wand_rescue_settings())
    flood = _best(lambda: wr.magic_wand(
        canvas.image, canvas.mask, x, y, 60.0, 256, action="add",
        **canvas.wand_rescue_settings()), repeats=1)
    before = canvas.mask
    start = time.perf_counter()
    canvas.mousePressEvent(press)
    held = time.perf_counter() - start
    assert canvas._wand_busy()
    assert canvas.mask is before, "the press changed the mask before the flood landed"
    assert held <= 0.25 * flood, (
        f"the press held the GUI thread {held * 1000:.0f} ms of a "
        f"{flood * 1000:.0f} ms flood")
    qtbot.waitUntil(lambda: not canvas._wand_busy(), timeout=60000)
    np.testing.assert_array_equal(canvas.mask, expected)
    detail = canvas.last_edit["detail"]
    assert canvas.last_edit["kind"] == "wand"
    assert detail["kept_px"] == report["kept_px"] and detail["capped"] is True
    assert list(detail)[:3] == ["action", "tolerance", "relative"]
    assert detail["input_kind"] == "as_loaded"


def test_an_edit_while_the_wand_fills_retires_its_region(qtbot, wand_field):
    canvas = wand_field
    said = []
    canvas.status.connect(said.append)
    canvas.mousePressEvent(_press(canvas, 200, 300))
    assert canvas._wand_busy()
    canvas.mousePressEvent(_press(canvas, 600, 300))
    assert any("still filling" in text for text in said)
    replaced = canvas.mask.copy()
    canvas.mask = replaced
    qtbot.waitUntil(lambda: not canvas._wand_busy(), timeout=60000)
    assert canvas.mask is replaced and not replaced.any()
    assert any("was not applied" in text for text in said)


def test_a_stroke_painted_in_place_while_the_wand_fills_retires_its_region(qtbot, wand_field):
    canvas = wand_field
    said = []
    canvas.status.connect(said.append)
    canvas.mousePressEvent(_press(canvas, 200, 300))
    assert canvas._wand_busy()
    engine.paint_disk(canvas.mask, 700, 700, 5, 9)
    qtbot.waitUntil(lambda: not canvas._wand_busy(), timeout=60000)
    assert int((canvas.mask > 0).sum()) == 100 and int(canvas.mask.max()) == 9
    assert any("was not applied" in text for text in said)


def test_a_small_field_still_floods_inside_the_click(qtbot, qt_theme_applied):
    canvas = mm._MaskCanvas()
    qtbot.addWidget(canvas)
    canvas.resize(400, 400)
    image = _capped_runaway(200)
    canvas.set_image_and_mask(image, np.zeros(image.shape, np.uint16))
    canvas.mode = mm.MODE_WAND_ADD
    canvas.wand_relative = False
    canvas.wand_tolerance = 60.0
    canvas.wand_max_pixels = 256
    canvas.mousePressEvent(_press(canvas, 50, 60))
    assert not canvas._wand_busy()
    assert canvas.mask.any() and canvas.last_edit["kind"] == "wand"


def test_the_taper_reads_only_the_flood_and_its_reach():
    """The crop to the flood's box gives the whole field's answer, faster."""
    rng = np.random.default_rng(4)
    image = rng.gamma(2.0, 800.0, (1600, 1600)).astype(np.uint16)
    checked = 0
    for y, x in rng.integers(200, 1400, (60, 2)):
        if checked >= 4:
            break
        tolerance = float(rng.uniform(150, 900))
        flooded = wr.flood_region(image, int(x), int(y), tolerance)
        if not 40 <= flooded.sum() <= flooded.size // 4:
            continue
        cap = max(10, int(flooded.sum() * 0.6))
        provisional = wr.cap_region_from_seed(flooded, (int(y), int(x)), cap)
        cropped = wr.taper_region_to_intensity(image, flooded, provisional, (int(y), int(x)))
        window = wr._taper_window
        try:
            wr._taper_window = lambda *_a, **_k: None
            whole = wr.taper_region_to_intensity(image, flooded, provisional, (int(y), int(x)))
        finally:
            wr._taper_window = window
        np.testing.assert_array_equal(cropped, whole)
        checked += 1
    assert checked >= 3
    yy, xx = np.ogrid[:1600, :1600]
    flooded = (yy - 800) ** 2 + (xx - 800) ** 2 <= 60 ** 2
    provisional = flooded & (xx < 820)
    fast = _best(lambda: wr.taper_region_to_intensity(image, flooded, provisional, (800, 790)))
    window = wr._taper_window
    try:
        wr._taper_window = lambda *_a, **_k: None
        slow = _best(lambda: wr.taper_region_to_intensity(image, flooded, provisional, (800, 790)))
    finally:
        wr._taper_window = window
    assert fast <= 0.5 * slow, f"{fast * 1000:.1f} ms cropped against {slow * 1000:.1f} ms"


# ---------------------------------------------------------------------------
# One field for the window: Make Masks flushes no more often than Home
# ---------------------------------------------------------------------------

class _Flushes(QObject):
    """Counts the window's backing-store flushes (UpdateRequest events)."""

    def __init__(self):
        super().__init__()
        self.count = 0

    def eventFilter(self, obj, event):
        if event.type() == QEvent.UpdateRequest:
            self.count += 1
        return False


def _flushes_per_second(qtbot, window, seconds=1.5):
    counter = _Flushes()
    window.installEventFilter(counter)
    try:
        start = time.perf_counter()
        qtbot.wait(int(seconds * 1000))
        return counter.count / (time.perf_counter() - start)
    finally:
        window.removeEventFilter(counter)


def _ticking_fields(window):
    from spacr.qt.widgets.ambient import AmbientWidget

    return [w for w in window.findChildren(AmbientWidget)
            if w.isVisible() and not w.property("spacrRetiringBackdrop")]


def test_make_masks_repaints_the_window_once_per_field_beat(qtbot, qt_theme_applied):
    """Self-calibrated against Home in the same window and the same second."""
    from spacr.qt.app import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(1600, 1000)
    window.show()
    qtbot.waitExposed(window)
    qtbot.waitUntil(lambda: window.window_backdrop() is not None, timeout=20000)
    if not hasattr(window.window_backdrop(), "_timer"):
        pytest.skip("the window backdrop is not the field in this setup")
    qtbot.wait(500)
    home = _flushes_per_second(qtbot, window)
    if home < 5:
        pytest.skip(f"the field is not animating here ({home:.1f} flushes/s)")
    window.open_module("make_masks")
    qtbot.wait(800)
    screen = window._screens["make_masks"]
    assert screen._uses_window_backdrop is True
    assert len(_ticking_fields(window)) == 1, "Make Masks built a second field"
    masks = _flushes_per_second(qtbot, window)
    assert masks <= 1.35 * home, (
        f"Make Masks flushed the window {masks:.1f} times a second against "
        f"{home:.1f} on Home")


def test_a_screen_themed_before_the_window_field_gives_its_own_up(qtbot, qt_theme_applied):
    """A non-AppScreen screen that built a field while the window had none
    hands it over once the window's arrives."""
    from PySide6.QtWidgets import QWidget

    from spacr.qt.app import MainWindow
    from spacr.qt.widgets.ambient import AmbientWidget

    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(1200, 800)
    window.show()
    qtbot.waitExposed(window)
    qtbot.waitUntil(lambda: window.window_backdrop() is not None, timeout=20000)
    screen = QWidget()
    qtbot.addWidget(screen)
    own = AmbientWidget(screen)
    assert MainWindow._a_screen_backdrop_built_beside_the_window_one(screen) is own
    window._drop_a_redundant_screen_backdrop(screen)
    assert screen._uses_window_backdrop is True
    assert own.parent() is None or own.property("spacrRetiringBackdrop")
    assert MainWindow._a_screen_backdrop_built_beside_the_window_one(screen) is None
    fresh = QWidget()
    qtbot.addWidget(fresh)
    window._install_screen_backdrop(fresh, "make_masks")
    assert not fresh.findChildren(AmbientWidget)
    assert fresh._uses_window_backdrop is True


# ---------------------------------------------------------------------------
# The first save does not import torch on the GUI thread
# ---------------------------------------------------------------------------

def test_opening_a_folder_imports_the_save_path_off_the_gui_thread(monkeypatch):
    from spacr.qt import bridge

    asked = []
    monkeypatch.setattr(mm, "_SAVE_PATH_WARM", [])
    monkeypatch.delitem(sys.modules, "spacr.io", raising=False)
    monkeypatch.setattr(bridge, "_import_the_entry_module",
                        lambda name: asked.append((name, __import__("threading").current_thread().name)))
    mm._warm_the_save_path()
    mm._SAVE_PATH_WARM[0].join(5)
    mm._warm_the_save_path()
    assert asked == [("spacr.io", "spacr-save-path-warm")]


# ---------------------------------------------------------------------------
# Divide: each crossed object is cut inside its own box
# ---------------------------------------------------------------------------

def _old_divide_or_cut(mask, p0, p1, width=engine.DIVIDE_CUT_WIDTH):
    """``_divide_or_cut`` as it was: every step over the whole field."""
    from scipy import ndimage

    if tuple(int(v) for v in p0) == tuple(int(v) for v in p1):
        return mask.copy(), [], []
    band = engine._segment_band(mask.shape, p0, p1, width)
    if not band.any():
        return mask.copy(), [], []
    crossed = [int(v) for v in np.unique(mask[band]) if int(v) > 0]
    if not crossed:
        return mask.copy(), [], []
    out = mask.astype(np.int64, copy=True)
    free_id = engine.next_label(mask)
    splits, cut = [], []
    for source in crossed:
        body = out == source
        remainder = body & ~band
        pieces, count = ndimage.label(remainder, structure=np.ones((3, 3), np.uint8))
        if count == 0:
            continue
        out[body & band] = 0
        if count < 2:
            cut.append(source)
            continue
        areas = np.bincount(pieces.ravel())
        keeps = int(np.argmax(areas[1:])) + 1
        out[body] = 0
        out[pieces == keeps] = source
        for piece in range(1, count + 1):
            if piece == keeps:
                continue
            out[pieces == piece] = free_id
            splits.append((source, free_id))
            free_id += 1
    if not splits and not cut:
        return mask.copy(), [], []
    return engine._fit_label_width(out, mask), splits, cut


def test_a_divide_stroke_gives_the_whole_field_answer_from_each_box():
    _, labels = _field(600, seed=8)
    rng = np.random.default_rng(12)
    labels[100:140, 100:300] = labels.max() + 1
    strokes = [((80, 120), (320, 120)), ((200, 90), (200, 160)),
               ((0, 0), (599, 599)), ((10, 300), (590, 310))]
    strokes += [tuple(map(tuple, rng.integers(0, 600, (2, 2)))) for _ in range(12)]
    for p0, p1 in strokes:
        new, splits, cut = engine._divide_or_cut(labels, p0, p1)
        old, old_splits, old_cut = _old_divide_or_cut(labels, p0, p1)
        np.testing.assert_array_equal(new, old)
        assert new.dtype == old.dtype
        assert (splits, cut) == (old_splits, old_cut)
    _, big_labels = _field(2000)
    stroke = ((300, 1000), (1700, 1010))
    fast = _best(lambda: engine._divide_or_cut(big_labels, *stroke))
    slow = _best(lambda: _old_divide_or_cut(big_labels, *stroke))
    assert fast <= 0.5 * slow, f"{fast * 1000:.0f} ms against {slow * 1000:.0f} ms"

"""Behavioural tests for the six editing features ported into Make Masks.

Every one of them is driven through the widget, offscreen: real mouse and
wheel events on the canvas, real spin boxes and buttons on the panel. A
wand tolerance that is only ever read out of an attribute is a number in a
file, and a pan that is only ever called as a method is not a gesture.

The six, and the thing each is really about:

1. **Relative wand tolerance.** An absolute tolerance is not a portable
   setting — the value that grabs one nucleus at 8-bit selects nothing at
   16-bit. The proof is not that the number changes but that the *selection
   does not*: the same percentage takes the same object out of the same
   scene at both depths.
2. **Six-decimal contrast percentiles.** On a megapixel 16-bit field, 99.9
   and 99.9999 differ by three orders of magnitude in how many pixels they
   clip, and a spin box with the default two decimals stores 99.9999 as
   100.0 — the control looks broken rather than imprecise.
3. **Size / intensity auto-filter.** Applied on load and on demand, 0 means
   the bound is off, and the intensity is measured on the raw image so the
   display percentiles cannot move it.
4. **Right-drag sweep-delete.** One gesture: one undo step, one ledger
   entry naming every object it took out.
5. **Shift/Alt-drag pan and wheel zoom.** From any tool, without putting
   the brush down, at an adjustable step.
6. **Otsu detect, replace or merge.** Merge must not be able to undo the
   editing already done.

Two things spaCR has that the tool ported from does not, and which these
also pin: the append-only curation ledger, and a save that keeps every
object's id instead of renumbering the components.
"""
from __future__ import annotations

import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent, QWheelEvent

from spacr.curation import CurationLog, is_curated, log_path_for
from spacr.qt import mask_engine as engine
from spacr.qt.screens import make_masks as mm
from spacr.qt.screens.make_masks import (
    MIN_VIEWPORT,
    MODE_BRUSH,
    MODE_WAND_ADD,
    MakeMasksScreen,
    _MaskCanvas,
)

# ---------------------------------------------------------------------------
# Geometry — the canvas is pinned at 600x400 and fed a 64x64 image, so
# refresh() fits a 400x400 pixmap centred with a 100 px margin either side.
# ---------------------------------------------------------------------------

CANVAS_W, CANVAS_H = 600, 400
IMG_N = 64
PIXMAP_N = 400
MARGIN_X = (CANVAS_W - PIXMAP_N) // 2       # 100


def canvas_xy(canvas, img_x: float, img_y: float) -> tuple:
    """Canvas-local point at the CENTRE of image pixel (img_x, img_y).

    Reads the live zoom viewport rather than assuming the full image, so
    the same helper aims a click after a wheel-zoom as before one.
    """
    pixmap = canvas.pixmap()
    off_x = (canvas.width() - pixmap.width()) // 2
    off_y = (canvas.height() - pixmap.height()) // 2
    x0, y0, x1, y1 = canvas._viewport_bounds()
    return (off_x + (img_x - x0 + 0.5) * pixmap.width() / (x1 - x0),
            off_y + (img_y - y0 + 0.5) * pixmap.height() / (y1 - y0))


def _mouse(kind, x, y, buttons, button, modifiers=Qt.NoModifier):
    pos = QPointF(float(x), float(y))
    return QMouseEvent(kind, pos, pos, button, buttons, modifiers)


def right_press(canvas, ix, iy):
    return _mouse(QEvent.Type.MouseButtonPress, *canvas_xy(canvas, ix, iy),
                  Qt.RightButton, Qt.RightButton)


def right_move(canvas, ix, iy):
    return _mouse(QEvent.Type.MouseMove, *canvas_xy(canvas, ix, iy),
                  Qt.RightButton, Qt.NoButton)


def right_release(canvas, ix, iy):
    return _mouse(QEvent.Type.MouseButtonRelease, *canvas_xy(canvas, ix, iy),
                  Qt.NoButton, Qt.RightButton)


def drag_with(canvas, modifier, x0, y0, x1, y1):
    """A left-button drag in WIDGET coordinates holding ``modifier``."""
    canvas.mousePressEvent(_mouse(QEvent.Type.MouseButtonPress, x0, y0,
                                  Qt.LeftButton, Qt.LeftButton, modifier))
    canvas.mouseMoveEvent(_mouse(QEvent.Type.MouseMove, x1, y1,
                                 Qt.LeftButton, Qt.NoButton, modifier))
    canvas.mouseReleaseEvent(_mouse(QEvent.Type.MouseButtonRelease, x1, y1,
                                    Qt.NoButton, Qt.LeftButton, modifier))


def wheel(canvas, x, y, notches: int):
    canvas.wheelEvent(QWheelEvent(
        QPointF(float(x), float(y)), QPointF(float(x), float(y)),
        QPoint(0, 0), QPoint(0, 120 * notches),
        Qt.NoButton, Qt.NoModifier, Qt.ScrollUpdate, False))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def three_object_image(peak: int, dtype) -> np.ndarray:
    """One scene, three bright squares, rendered at whatever depth is asked.

    Areas 100 / 16 / 225 px and the third one dim, so a size bound and an
    intensity bound each have exactly one object to pick out.
    """
    img = np.zeros((IMG_N, IMG_N), dtype=dtype)
    img[5:15, 5:15] = peak                 # 100 px, bright
    img[30:34, 30:34] = peak               # 16 px, bright
    img[45:60, 45:60] = int(peak * 0.22)   # 225 px, dim
    return img


def three_object_mask() -> np.ndarray:
    """Labels 3 / 7 / 11 over those squares — deliberately not 1, 2, 3.

    Ids with gaps in them are what a real segmentation looks like after a
    filtering stage, and they are what a save that renumbers destroys.
    """
    mask = np.zeros((IMG_N, IMG_N), dtype=np.uint16)
    mask[5:15, 5:15] = 3
    mask[30:34, 30:34] = 7
    mask[45:60, 45:60] = 11
    return mask


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    """Two fields, each with the three-object image and its label mask."""
    root = tmp_path / "fields"
    (root / "masks").mkdir(parents=True)
    for name in ("f_00.tif", "f_01.tif"):
        imageio.imwrite(root / name, three_object_image(40000, np.uint16))
        imageio.imwrite(root / "masks" / name, three_object_mask())
    return root


@pytest.fixture
def screen(qtbot, qt_theme_applied, folder: Path) -> MakeMasksScreen:
    """A screen with the folder open and the canvas at a known size."""
    s = MakeMasksScreen()
    qtbot.addWidget(s)
    s._canvas.resize(CANVAS_W, CANVAS_H)
    s._open_folder(str(folder))
    assert s._canvas.pixmap().width() == PIXMAP_N, "geometry assumption broke"
    return s


@pytest.fixture
def headless(monkeypatch):
    """Pin the no-display branch rather than inheriting it from the platform.

    Same reason as the fixture of this name in ``test_make_masks_canvas``:
    under a real X server ``is_headless()`` answers False and the screen's
    error path opens a genuine modal that nobody is there to dismiss.
    """
    monkeypatch.setattr(mm, "is_headless", lambda: True)
    return True


@pytest.fixture
def canvas(qtbot, qt_theme_applied) -> _MaskCanvas:
    """A bare canvas holding the three-object scene and its labels."""
    c = _MaskCanvas()
    qtbot.addWidget(c)
    c.resize(CANVAS_W, CANVAS_H)
    c.set_image_and_mask(three_object_image(40000, np.uint16),
                         three_object_mask())
    assert c.pixmap().width() == PIXMAP_N, "geometry assumption broke"
    return c


# ===========================================================================
# 1. Relative magic-wand tolerance
# ===========================================================================

def test_relative_tolerance_is_a_percentage_of_the_images_own_range():
    eight_bit = np.array([[0, 255]], dtype=np.uint8)
    sixteen_bit = np.array([[0, 65535]], dtype=np.uint16)
    assert engine.relative_tolerance(eight_bit, 5.0) == pytest.approx(12.75)
    assert engine.relative_tolerance(sixteen_bit, 5.0) == pytest.approx(3276.75)


def test_relative_tolerance_has_a_floor_of_one_on_a_flat_field():
    """A range of zero would give a tolerance of zero, and a tolerance of
    zero floods only pixels exactly equal to the seed."""
    flat = np.full((8, 8), 700, dtype=np.uint16)
    assert engine.relative_tolerance(flat, 5.0) == 1.0
    assert engine.relative_tolerance(np.zeros((0, 0), np.uint16), 5.0) == 1.0


@pytest.mark.parametrize("peak, dtype", [(255, np.uint8), (40000, np.uint16)])
def test_the_wand_takes_the_same_object_at_8_bit_and_16_bit(
        qtbot, qt_theme_applied, peak, dtype):
    """The point of a relative tolerance: one setting, one selection.

    The same scene at two bit depths, the same 5%, the same click — and
    the same 100 pixels come back. That is what an absolute tolerance
    cannot do, which the next test shows.
    """
    c = _MaskCanvas()
    qtbot.addWidget(c)
    c.resize(CANVAS_W, CANVAS_H)
    c.set_image_and_mask(three_object_image(peak, dtype),
                         np.zeros((IMG_N, IMG_N), np.uint8))
    c.mode = MODE_WAND_ADD
    c.wand_relative = True
    c.wand_tol_pct = 5.0
    c.wand_max_pixels = 10_000
    c.mousePressEvent(_mouse(QEvent.Type.MouseButtonPress,
                             *canvas_xy(c, 10, 10),
                             Qt.LeftButton, Qt.LeftButton))
    assert int((c.mask > 0).sum()) == 100, "the 10x10 bright square, exactly"


def test_one_absolute_tolerance_cannot_serve_both_depths(qtbot,
                                                          qt_theme_applied):
    """Why the default changed: 1000 grey levels is most of an 8-bit image.

    At 16-bit, 1000 sits well inside the gap between the dim square (8800)
    and the bright one (40000) and selects the clicked object. At 8-bit the
    whole image spans 255, so the same 1000 spans everything and the flood
    runs off into the background — the object and the field it sits in
    become one selection.
    """
    def flood(peak, dtype):
        c = _MaskCanvas()
        qtbot.addWidget(c)
        c.resize(CANVAS_W, CANVAS_H)
        c.set_image_and_mask(three_object_image(peak, dtype),
                             np.zeros((IMG_N, IMG_N), np.uint8))
        c.mode = MODE_WAND_ADD
        c.wand_relative = False
        c.wand_tolerance = 1000.0
        c.wand_max_pixels = 10_000
        c.mousePressEvent(_mouse(QEvent.Type.MouseButtonPress,
                                 *canvas_xy(c, 10, 10),
                                 Qt.LeftButton, Qt.LeftButton))
        return int((c.mask > 0).sum())

    assert flood(40000, np.uint16) == 100
    assert flood(255, np.uint8) == IMG_N * IMG_N


def test_the_wand_reports_the_tolerance_it_will_actually_use(canvas):
    canvas.wand_relative = True
    canvas.wand_tol_pct = 5.0
    assert canvas.effective_wand_tolerance() == pytest.approx(2000.0)
    canvas.wand_relative = False
    canvas.wand_tolerance = 137.0
    assert canvas.effective_wand_tolerance() == 137.0


def test_the_relative_checkbox_enables_exactly_one_tolerance_box(screen):
    assert screen._wand_relative.isChecked(), "relative is the default"
    assert screen._wand_pct.isEnabled()
    assert not screen._wand_tol.isEnabled()
    screen._wand_relative.setChecked(False)
    assert not screen._wand_pct.isEnabled()
    assert screen._wand_tol.isEnabled()
    assert screen._canvas.wand_relative is False


def test_the_percentage_box_reaches_the_canvas(screen):
    screen._wand_pct.setValue(12.5)
    assert screen._canvas.wand_tol_pct == 12.5


# ===========================================================================
# 2. Contrast percentiles, six decimals
# ===========================================================================

def test_the_percentile_boxes_keep_six_decimals(screen):
    """Two decimals would store 99.9999 as 100.0 and clip nothing at all."""
    assert screen._norm_hi.decimals() == mm.PERCENTILE_DECIMALS == 6
    assert screen._norm_lo.decimals() == 6
    screen._norm_hi.setValue(99.9999)
    assert screen._norm_hi.value() == pytest.approx(99.9999, abs=1e-9)
    assert screen._canvas.norm_hi == pytest.approx(99.9999, abs=1e-9)
    screen._norm_lo.setValue(0.0001)
    assert screen._canvas.norm_lo == pytest.approx(0.0001, abs=1e-9)


def test_the_last_decimal_is_the_difference_between_hot_pixels_and_none():
    """A megapixel field where the top 0.1% is a hot-pixel population.

    At 99.9 the ceiling lands on the hot pixels and a thousand of them
    saturate; at 99.9999 it lands one pixel from the top and the ceiling is
    set by the hottest pixel alone. The three orders of magnitude between
    those two clips are only reachable with six decimals.
    """
    field = np.full(1000 * 1000, 400, dtype=np.uint16)
    field[:1000] = np.linspace(5000, 65535, 1000).astype(np.uint16)
    rng = np.random.default_rng(0)
    rng.shuffle(field)
    field = field.reshape(1000, 1000)

    coarse = engine.normalize_uint16(field, 1.0, 99.9)
    fine = engine.normalize_uint16(field, 1.0, 99.9999)
    saturated_coarse = int((coarse == 65535).sum())
    saturated_fine = int((fine == 65535).sum())
    assert saturated_coarse > 900, saturated_coarse
    assert saturated_fine <= 2, saturated_fine


def test_a_six_decimal_ceiling_changes_what_the_canvas_draws(canvas):
    """The precision has to survive the whole way to the pixmap."""
    hot = three_object_image(40000, np.uint16)
    hot[0, :4] = 65535
    canvas.set_image_and_mask(hot, np.zeros((IMG_N, IMG_N), np.uint8))

    def rendered_mean():
        image = canvas.pixmap().toImage()
        buf = np.frombuffer(image.constBits(), dtype=np.uint8)
        return float(buf.mean())

    canvas.norm_hi = 99.9
    canvas.refresh()
    bright = rendered_mean()
    canvas.norm_hi = 99.9999
    canvas.refresh()
    dim = rendered_mean()
    assert bright > dim, (
        "a ceiling that ignores the hot pixels must render brighter than "
        f"one set by them: {bright} vs {dim}")


# ===========================================================================
# 3. Size / intensity auto-filter
# ===========================================================================

def test_the_filter_runs_on_load_and_drops_the_object_below_min_area(
        qtbot, qt_theme_applied, folder: Path):
    s = MakeMasksScreen()
    qtbot.addWidget(s)
    s._filter_min_area.setValue(50)
    s._open_folder(str(folder))
    assert sorted(int(v) for v in np.unique(s._canvas.mask) if v) == [3, 11], \
        "the 16-pixel object should have been filtered out on load"
    assert "removed 1 object" in s._status_label.text()


def test_every_bound_is_off_at_zero(screen):
    """Zero is the off switch, not a rejection threshold."""
    assert screen._filter_min_area.value() == 0
    assert screen._filter_max_int.value() == 0.0
    before = screen._canvas.mask.copy()
    assert screen.apply_object_filter() == 0
    assert (screen._canvas.mask == before).all()


@pytest.mark.parametrize("field, value, survivors", [
    ("_filter_min_area", 50, [3, 11]),
    ("_filter_max_area", 120, [3, 7]),
    ("_filter_min_int", 20000.0, [3, 7]),
    ("_filter_max_int", 20000.0, [11]),
])
def test_each_bound_picks_out_its_own_object(screen, field, value, survivors):
    getattr(screen, field).setValue(value)
    screen._btn_filter.click()
    assert sorted(int(v) for v in np.unique(screen._canvas.mask) if v) \
        == survivors


def test_the_intensity_bound_reads_the_raw_image_not_the_display(screen):
    """A filter that moved when you changed the contrast would not be
    reproducible, so the mean is taken on the raw data."""
    screen._norm_lo.setValue(0.0)
    screen._norm_hi.setValue(50.0)          # a wildly different stretch
    screen._filter_min_int.setValue(20000.0)
    screen._btn_filter.click()
    assert sorted(int(v) for v in np.unique(screen._canvas.mask) if v) == [3, 7]


def test_a_filter_that_removes_nothing_says_so_and_edits_nothing(screen):
    screen._filter_min_area.setValue(2)
    before = screen._canvas.mask.copy()
    assert screen.apply_object_filter() == 0
    assert (screen._canvas.mask == before).all()
    assert "nothing outside the bounds" in screen._status_label.text()


def test_the_filter_is_one_undo_step_and_one_ledger_entry(screen):
    screen._filter_min_area.setValue(50)
    screen._btn_filter.click()
    assert screen._btn_undo.isEnabled()
    screen._on_undo()
    assert sorted(int(v) for v in np.unique(screen._canvas.mask) if v) \
        == [3, 7, 11], "undo must bring the filtered object back"
    kinds = [e.kind for e in screen._log.edits]
    assert kinds.count("filter") == 1
    entry = next(e for e in screen._log.edits if e.kind == "filter")
    assert entry.target == [7]
    assert entry.n_changed == 16
    assert entry.detail["n_objects"] == 1


# ===========================================================================
# 4. Right-click-drag sweep-delete
# ===========================================================================

def test_a_right_drag_deletes_every_object_it_crosses(canvas):
    canvas.mousePressEvent(right_press(canvas, 10, 10))
    canvas.mouseMoveEvent(right_move(canvas, 32, 32))
    canvas.mouseReleaseEvent(right_release(canvas, 32, 32))
    assert sorted(int(v) for v in np.unique(canvas.mask) if v) == [11]
    assert canvas.last_edit["kind"] == "sweep_delete"
    assert canvas.last_edit["target"] == [3, 7]
    assert canvas.last_edit["detail"]["n_objects"] == 2


def test_a_sweep_emits_one_stroke_however_many_objects_it_takes(canvas):
    started, finished = [], []
    canvas.stroke_started.connect(lambda: started.append(1))
    canvas.stroke_finished.connect(lambda: finished.append(1))
    canvas.mousePressEvent(right_press(canvas, 10, 10))
    canvas.mouseMoveEvent(right_move(canvas, 32, 32))
    canvas.mouseMoveEvent(right_move(canvas, 50, 50))
    canvas.mouseReleaseEvent(right_release(canvas, 50, 50))
    assert not (canvas.mask > 0).any(), "all three objects gone"
    assert started == [1] and finished == [1], \
        "three deletions, one stroke — otherwise undo steps back one object"


def test_a_sweep_is_a_single_undo_step_and_a_single_ledger_entry(screen):
    c = screen._canvas
    c.mousePressEvent(right_press(c, 10, 10))
    c.mouseMoveEvent(right_move(c, 32, 32))
    c.mouseMoveEvent(right_move(c, 50, 50))
    c.mouseReleaseEvent(right_release(c, 50, 50))
    assert not (c.mask > 0).any()
    screen._on_undo()
    assert sorted(int(v) for v in np.unique(c.mask) if v) == [3, 7, 11], \
        "one undo must bring back all three, not the last one"
    sweeps = [e for e in screen._log.edits if e.kind == "sweep_delete"]
    assert len(sweeps) == 1
    assert sweeps[0].target == [3, 7, 11]
    assert sweeps[0].detail["n_objects"] == 3
    assert sweeps[0].n_changed == 100 + 16 + 225


def test_a_right_click_on_background_leaves_no_undo_step(screen):
    c = screen._canvas
    assert not screen._btn_undo.isEnabled()
    c.mousePressEvent(right_press(c, 25, 25))       # between the squares
    c.mouseReleaseEvent(right_release(c, 25, 25))
    assert sorted(int(v) for v in np.unique(c.mask) if v) == [3, 7, 11]
    assert not screen._btn_undo.isEnabled(), \
        "a click that deleted nothing must not become an undo step"
    assert len(screen._log) == 0


def test_the_sweep_works_from_whatever_tool_is_selected(screen):
    """Right-drag is a gesture, not a mode: it must not need the brush put
    down first."""
    screen._set_mode(MODE_BRUSH)
    c = screen._canvas
    c.mousePressEvent(right_press(c, 10, 10))
    c.mouseReleaseEvent(right_release(c, 10, 10))
    assert sorted(int(v) for v in np.unique(c.mask) if v) == [7, 11]
    assert screen._canvas.mode == MODE_BRUSH, "the tool is still the brush"


# ===========================================================================
# 5. Pan from any tool, and wheel zoom
# ===========================================================================

@pytest.mark.parametrize("modifier", [Qt.ShiftModifier, Qt.AltModifier])
def test_a_modified_drag_pans_instead_of_painting(canvas, modifier):
    canvas.mode = MODE_BRUSH
    canvas.zoom_at(32, 32, 4.0)
    before_mask = canvas.mask.copy()
    before_view = canvas._viewport_bounds()
    drag_with(canvas, modifier, 300, 200, 200, 150)
    assert canvas._viewport_bounds() != before_view, "the view did not move"
    assert (canvas.mask == before_mask).all(), \
        "a pan must not lay down paint"


def test_an_unmodified_drag_still_paints(canvas):
    """The guard is the modifier, not the button: without one held, the
    brush must behave exactly as before."""
    canvas.mode = MODE_BRUSH
    canvas.brush_radius = 4
    before = int((canvas.mask > 0).sum())
    drag_with(canvas, Qt.NoModifier,
              *canvas_xy(canvas, 20, 20), *canvas_xy(canvas, 24, 20))
    assert int((canvas.mask > 0).sum()) > before


def test_panning_an_unzoomed_canvas_does_nothing(canvas):
    assert not canvas.is_zoomed()
    assert canvas.pan_by(5, 5) is False, \
        "the whole image is on screen; there is nowhere to pan to"


def test_a_pan_is_clamped_to_the_image(canvas):
    canvas.zoom_at(32, 32, 4.0)
    canvas.pan_by(-10_000, -10_000)
    x0, y0, x1, y1 = canvas._viewport_bounds()
    assert (x0, y0) == (0, 0)
    canvas.pan_by(10_000, 10_000)
    x0, y0, x1, y1 = canvas._viewport_bounds()
    assert (x1, y1) == (IMG_N, IMG_N)


def test_the_wheel_zooms_in_and_the_viewport_shrinks(canvas):
    assert not canvas.is_zoomed()
    wheel(canvas, *canvas_xy(canvas, 32, 32), 1)
    x0, y0, x1, y1 = canvas._viewport_bounds()
    assert canvas.is_zoomed()
    assert (x1 - x0) < IMG_N and (y1 - y0) < IMG_N


def test_wheel_zoom_speed_changes_how_far_one_notch_goes(qtbot,
                                                          qt_theme_applied):
    def viewport_after_one_notch(speed):
        c = _MaskCanvas()
        qtbot.addWidget(c)
        c.resize(CANVAS_W, CANVAS_H)
        c.set_image_and_mask(three_object_image(40000, np.uint16),
                             three_object_mask())
        c.zoom_speed = speed
        wheel(c, *canvas_xy(c, 32, 32), 1)
        x0, _, x1, _ = c._viewport_bounds()
        return x1 - x0

    gentle = viewport_after_one_notch(1.05)
    brisk = viewport_after_one_notch(2.0)
    assert brisk < gentle, (
        f"a faster wheel must magnify further in one notch: {brisk} vs "
        f"{gentle}")
    assert gentle < IMG_N


def test_the_zoom_speed_box_reaches_the_canvas(screen):
    screen._zoom_speed.setValue(1.4)
    assert screen._canvas.zoom_speed == pytest.approx(1.4)


def test_the_wheel_keeps_the_pixel_under_the_cursor_in_view(canvas):
    """Wheel zoom is aimed: what you point at is what you magnify."""
    target = (50, 50)
    wheel(canvas, *canvas_xy(canvas, *target), 1)
    wheel(canvas, *canvas_xy(canvas, *target), 1)
    x0, y0, x1, y1 = canvas._viewport_bounds()
    assert x0 <= target[0] < x1 and y0 <= target[1] < y1, \
        f"pixel {target} fell out of viewport {(x0, y0, x1, y1)}"


def test_wheeling_back_out_returns_to_the_full_image(canvas):
    wheel(canvas, *canvas_xy(canvas, 32, 32), 1)
    assert canvas.is_zoomed()
    for _ in range(20):
        wheel(canvas, 300, 200, -1)
    assert not canvas.is_zoomed(), "zooming out past the image must reset"
    assert canvas._viewport_bounds() == (0, 0, IMG_N, IMG_N)


def test_the_viewport_will_not_shrink_below_the_floor(canvas):
    for _ in range(200):
        wheel(canvas, *canvas_xy(canvas, 32, 32), 1)
    x0, _, x1, _ = canvas._viewport_bounds()
    assert (x1 - x0) == MIN_VIEWPORT


# ===========================================================================
# 6. Otsu detect, replace or merge
# ===========================================================================

def test_otsu_finds_the_bright_objects():
    """Otsu is a two-class split, and the scene has three classes.

    The two bright squares land above the threshold and the dim one — at
    8800, just under the 8828 Otsu picks — lands with the background. That
    is the honest limit of a global threshold and the reason the button
    offers to MERGE rather than only replace: what it finds is a starting
    point, not the answer.
    """
    detected = engine.otsu_instances(three_object_image(40000, np.uint16),
                                     bright=True, min_area=4)
    assert int(detected.max()) == 2
    assert sorted(int(c) for c in np.bincount(detected.ravel())[1:]) \
        == [16, 100]


def test_otsu_can_take_the_dark_side_instead():
    """Brightfield and stain absorb: the objects are BELOW the threshold."""
    inverted = 65535 - three_object_image(40000, np.uint16)
    detected = engine.otsu_instances(inverted, bright=False, min_area=4)
    assert int(detected.max()) == 2
    assert sorted(int(c) for c in np.bincount(detected.ravel())[1:]) \
        == [16, 100]


def test_otsu_takes_the_wrong_side_when_told_to():
    """The switch has to actually switch, or a brightfield user gets the
    background handed back as objects and no way to tell."""
    detected = engine.otsu_instances(three_object_image(40000, np.uint16),
                                     bright=False, min_area=4)
    areas = np.bincount(detected.ravel())[1:]
    assert int(areas.max()) > 3000, \
        "the dark side of a mostly-black field is the background"


def test_otsu_min_area_drops_the_speckle():
    detected = engine.otsu_instances(three_object_image(40000, np.uint16),
                                     bright=True, min_area=50)
    assert int(detected.max()) == 1, "the 16-pixel square is below 50"


def test_otsu_refuses_an_empty_image():
    with pytest.raises(ValueError, match="empty"):
        engine.otsu_instances(np.zeros((0, 0), np.uint16))


def test_otsu_replace_discards_what_was_there(screen):
    screen._combine_mode.setCurrentText("replace")
    screen._min_area.setValue(4)
    screen._btn_otsu.click()
    assert sorted(int(v) for v in np.unique(screen._canvas.mask) if v) \
        == [1, 2], "replace keeps the detection alone, ids and all"
    assert "replaced into the mask" in screen._status_label.text()


def test_otsu_merge_keeps_every_object_already_curated(screen):
    """Merge is the whole reason the choice exists: a detection run halfway
    through an editing session must not be able to undo the first half."""
    screen._combine_mode.setCurrentText("merge")
    screen._min_area.setValue(4)
    before = screen._canvas.mask.copy()
    screen._btn_otsu.click()
    after = screen._canvas.mask
    for label_id in (3, 7, 11):
        assert ((before == label_id) == (after == label_id)).all(), \
            f"merge moved existing object {label_id}"


def test_otsu_is_one_undo_step_and_a_ledger_entry(screen):
    screen._combine_mode.setCurrentText("replace")
    screen._min_area.setValue(4)
    screen._btn_otsu.click()
    screen._on_undo()
    assert sorted(int(v) for v in np.unique(screen._canvas.mask) if v) \
        == [3, 7, 11]
    entry = next(e for e in screen._log.edits if e.kind == "detect")
    assert entry.target == "replace"
    assert entry.detail["method"] == "otsu"
    assert entry.detail["n_objects"] == 2


def test_a_detection_that_finds_nothing_leaves_the_mask_alone(screen):
    """Replace-with-nothing would wipe the mask on a flat field.

    Clearing a mask is what the Clear button is for, and that one asks
    first; a detection that came up empty must not do it silently.
    """
    screen._combine_mode.setCurrentText("replace")
    screen._min_area.setValue(100_000)      # nothing can pass this
    before = screen._canvas.mask.copy()
    screen._btn_otsu.click()
    assert (screen._canvas.mask == before).all()
    assert not screen._btn_undo.isEnabled()
    assert len(screen._log) == 0
    assert "found no objects" in screen._status_label.text()


def test_combine_refuses_a_mode_it_does_not_know():
    """Silently picking one would throw away the user's edits on a typo."""
    old = np.zeros((4, 4), np.uint8)
    with pytest.raises(ValueError, match="replace.*merge"):
        engine.combine_masks(old, old, "overwrite")


def test_merge_widens_the_dtype_rather_than_wrapping_labels():
    old = np.zeros((40, 40), np.uint8)
    old[0, 0] = 200
    new = np.arange(1, 1601, dtype=np.int32).reshape(40, 40)
    out = engine.combine_masks(old, new, "merge")
    assert out.dtype == np.uint16
    # Merge offsets the incoming ids past the existing maximum, so the top
    # id is 200 + 1600 — past uint8, where it would have wrapped to 8 and
    # fused with another object.
    assert int(out.max()) == 1800


def test_a_failed_detection_is_reported_not_raised(screen, monkeypatch,
                                                    headless):
    def boom(*_a, **_k):
        raise RuntimeError("no threshold here")
    monkeypatch.setattr(engine, "otsu_instances", boom)
    before = screen._canvas.mask.copy()
    screen._btn_otsu.click()
    assert (screen._canvas.mask == before).all()
    assert "no threshold here" in screen._status_label.text()


# ===========================================================================
# Object ids survive a save
# ===========================================================================

def test_a_saved_mask_keeps_the_ids_it_had(screen, folder: Path):
    """The reproducibility hole this replaced: ``label(mask > 0)`` renumbers
    the components, so erasing object 7 of 3/7/11 used to slide the survivors
    down to 1/2 and re-key the mask against every table made from it."""
    c = screen._canvas
    c.mousePressEvent(right_press(c, 32, 32))       # object 7
    c.mouseReleaseEvent(right_release(c, 32, 32))
    screen._on_save()
    disk = imageio.imread(folder / "masks" / "f_00.tif")
    assert sorted(int(v) for v in np.unique(disk) if v) == [3, 11]


def test_a_brush_only_mask_is_numbered_from_one():
    """The documented exception: every stroke writes the same value, so
    there are no ids to keep and numbering the components loses nothing."""
    mask = np.zeros((20, 20), np.uint8)
    mask[2:5, 2:5] = 255
    mask[12:15, 12:15] = 255
    assert sorted(int(v) for v in np.unique(engine.canonical_labels(mask))) \
        == [0, 1, 2]


def test_one_id_naming_two_separated_blobs_is_split():
    """One id must mean one object, or a brush stroke laid down beside a real
    segmentation silently extends an object on the other side of the field."""
    mask = np.zeros((20, 20), np.uint8)
    mask[2:6, 2:6] = 7          # 16 px — the larger piece keeps the id
    mask[14:16, 14:16] = 7      # 4 px
    mask[10, 0] = 4
    out = engine.canonical_labels(mask)
    assert int(out[3, 3]) == 7, "the larger piece keeps the id"
    assert int(out[10, 0]) == 4, "an untouched id is untouched"
    minted = int(out[14, 14])
    assert minted not in (0, 4, 7)
    assert sorted(int(v) for v in np.unique(out) if v) == sorted([4, 7, minted])


def test_a_label_too_large_for_uint16_is_refused_not_truncated():
    """Wrapping would fuse the object with the background and lose it."""
    mask = np.zeros((4, 4), dtype=np.int32)
    mask[0, 0] = 70_000
    mask[3, 3] = 5
    with pytest.raises(ValueError, match="uint16"):
        engine.canonical_labels(mask)


def test_canonical_labels_leaves_a_well_formed_label_image_alone():
    mask = three_object_mask()
    assert (engine.canonical_labels(mask) == mask).all()


def test_mask_save_path_is_where_the_mask_and_its_ledger_agree(tmp_path):
    path = engine.mask_save_path(str(tmp_path), "field.png")
    assert path == os.path.join(str(tmp_path), "masks", "field.tif")
    assert log_path_for(path).endswith("field.tif.curation.json")


# ===========================================================================
# The curation ledger
# ===========================================================================

def test_a_mask_nobody_edited_gets_no_ledger(screen, folder: Path):
    """An empty ledger beside every mask ever opened would make
    ``is_curated`` answer True for everything and so answer nothing."""
    screen._on_save()
    artifact = folder / "masks" / "f_00.tif"
    assert not os.path.exists(log_path_for(str(artifact)))
    assert is_curated(str(artifact)) is False


def test_saving_an_edited_mask_writes_the_ledger_beside_it(screen,
                                                            folder: Path):
    c = screen._canvas
    c.mousePressEvent(right_press(c, 10, 10))
    c.mouseMoveEvent(right_move(c, 32, 32))
    c.mouseReleaseEvent(right_release(c, 32, 32))
    screen._on_save()
    artifact = str(folder / "masks" / "f_00.tif")
    assert is_curated(artifact) is True
    log = CurationLog.read_beside(artifact)
    assert log.artifact == artifact
    assert log.source == engine.CURATION_SOURCE
    assert [e.kind for e in log.edits] == ["sweep_delete"]
    assert log.edits[0].n_changed == 116
    assert "(1 edit(s) recorded)" in screen._status_label.text()


def test_a_brush_stroke_is_recorded_as_a_paint(screen):
    screen._set_mode(MODE_BRUSH)
    screen._brush_slider.setValue(6)
    c = screen._canvas
    drag_with(c, Qt.NoModifier,
              *canvas_xy(c, 20, 20), *canvas_xy(c, 25, 20))
    entry = screen._log.edits[-1]
    assert entry.kind == "paint"
    assert entry.n_changed > 0
    assert entry.detail["radius"] == 6


def test_a_wand_click_records_the_tolerance_it_used(screen):
    screen._set_mode(MODE_WAND_ADD)
    screen._wand_pct.setValue(5.0)
    c = screen._canvas
    c.mousePressEvent(_mouse(QEvent.Type.MouseButtonPress,
                             *canvas_xy(c, 47, 47),
                             Qt.LeftButton, Qt.LeftButton))
    entry = screen._log.edits[-1]
    assert entry.kind == "wand"
    assert entry.detail["relative"] is True
    assert entry.detail["tolerance"] == pytest.approx(2000.0)


def test_undo_appends_to_the_ledger_rather_than_erasing_the_edit(screen):
    """The ledger is append-only: that a stroke was made and then taken back
    is itself part of what happened to the data."""
    c = screen._canvas
    c.mousePressEvent(right_press(c, 10, 10))
    c.mouseReleaseEvent(right_release(c, 10, 10))
    screen._on_undo()
    assert [e.kind for e in screen._log.edits] == ["sweep_delete", "undo"]
    screen._on_redo()
    assert [e.kind for e in screen._log.edits] \
        == ["sweep_delete", "undo", "redo"]


def test_an_edit_that_moved_no_pixels_is_not_recorded(screen):
    """Remove-small at a threshold of 0 removes nothing, by definition."""
    before = screen._canvas.mask.copy()
    screen._min_area.setValue(0)
    screen._on_remove_small()
    assert (screen._canvas.mask == before).all()
    assert len(screen._log) == 0, \
        "a ledger padded with no-ops is one nobody reads"


def test_a_second_session_appends_to_the_first_ones_ledger(qtbot,
                                                            qt_theme_applied,
                                                            folder: Path):
    def edit_and_save():
        s = MakeMasksScreen()
        qtbot.addWidget(s)
        s._canvas.resize(CANVAS_W, CANVAS_H)
        s._open_folder(str(folder))
        c = s._canvas
        target = 10 if (c.mask[10, 10] > 0) else 47
        c.mousePressEvent(right_press(c, target, target))
        c.mouseReleaseEvent(right_release(c, target, target))
        s._on_save()

    edit_and_save()
    edit_and_save()
    log = CurationLog.read_beside(str(folder / "masks" / "f_00.tif"))
    assert [e.kind for e in log.edits] == ["sweep_delete", "sweep_delete"], \
        "the second session overwrote the first session's record"


def test_the_ledger_belongs_to_the_field_on_screen(screen, folder: Path):
    c = screen._canvas
    c.mousePressEvent(right_press(c, 10, 10))
    c.mouseReleaseEvent(right_release(c, 10, 10))
    screen._on_save()
    screen._on_next()
    assert len(screen._log) == 0, "a new field starts its own record"
    screen._on_save()
    assert not os.path.exists(
        log_path_for(str(folder / "masks" / "f_01.tif")))
    assert is_curated(str(folder / "masks" / "f_00.tif")) is True


def test_an_unreadable_ledger_does_not_cost_the_next_edit(qtbot,
                                                           qt_theme_applied,
                                                           folder: Path):
    """A damaged sidecar is a reason to start a fresh record, not a reason to
    refuse to open the field."""
    artifact = str(folder / "masks" / "f_00.tif")
    with open(log_path_for(artifact), "w", encoding="utf-8") as handle:
        handle.write("{not json at all")
    s = MakeMasksScreen()
    qtbot.addWidget(s)
    s._canvas.resize(CANVAS_W, CANVAS_H)
    s._open_folder(str(folder))
    assert s._log is not None and len(s._log) == 0
    c = s._canvas
    c.mousePressEvent(right_press(c, 10, 10))
    c.mouseReleaseEvent(right_release(c, 10, 10))
    s._on_save()
    assert [e.kind for e in CurationLog.read_beside(artifact).edits] \
        == ["sweep_delete"]


def test_a_load_failure_drops_the_ledger_with_the_canvas(screen, headless,
                                                          monkeypatch):
    """The ledger names one artefact; keeping a stale one open would file the
    next edit against the field that failed to load."""
    def boom(*_a, **_k):
        raise OSError("unreadable field")
    monkeypatch.setattr(engine, "load_image_and_mask", boom)
    screen._on_next()
    assert screen._canvas.mask is None
    assert screen._log is None

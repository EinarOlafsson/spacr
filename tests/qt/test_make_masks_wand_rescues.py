"""The magic wand's rescues for a flood that runs away.

The failure these exist for is one failure with three answers. A flood
from one click reaches something bright that is not the object -- debris,
a saturated seam, a well rim -- walks out along it and takes the field.
Lowering the tolerance is not a fix, because the tolerance that takes the
object and the tolerance that escapes are frequently the same number.

So each test here starts from a scene where the plain flood demonstrably
runs away -- the number is asserted, not assumed -- and then measures what
each rescue keeps. The geometry half needs no GUI; the widget half drives
real mouse presses on the canvas and real controls on the panel, because a
setting that is only ever assigned in a test is a number in a file.

Ported from the standalone curation tool, whose seventeen wand controls
were the reason this screen's two looked short.
"""
from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from scipy.ndimage import label

from spacr.qt import wand_rescue as wr
from spacr.qt.screens.make_masks import (
    MODE_WAND_ADD,
    _MaskCanvas,
)

CANVAS_W, CANVAS_H = 600, 400
IMG_N = 64
PIXMAP_N = 400

SEED_X, SEED_Y = 20, 32          # inside the object, clear of the seam
OBJECT_PEAK = 40000


# ---------------------------------------------------------------------------
# Scenes
# ---------------------------------------------------------------------------

def runaway_scene(seam_peak: int = OBJECT_PEAK) -> tuple:
    """A round object welded by a thin seam to a large bright field.

    This is the shape of the failure: the seam is only four pixels wide,
    but it is inside any tolerance that takes the object, so the flood
    crosses it and keeps going. ``seam_peak`` below the object's peak makes
    the seam separable by tolerance alone; equal to it makes the seam
    inseparable, which is the case the straight cut and the taper exist for.

    :returns: ``(image, object_mask)``.
    """
    yy, xx = np.ogrid[:IMG_N, :IMG_N]
    image = np.zeros((IMG_N, IMG_N), dtype=np.uint16)
    obj = (xx - SEED_X) ** 2 + (yy - SEED_Y) ** 2 <= 10 ** 2
    seam = np.zeros((IMG_N, IMG_N), dtype=bool)
    seam[30:34, 28:] = True
    field = np.zeros((IMG_N, IMG_N), dtype=bool)
    field[:, 50:] = True
    image[seam | field] = seam_peak
    image[obj] = OBJECT_PEAK
    return image, obj


def clean_scene() -> tuple:
    """The same object with nothing to escape into."""
    yy, xx = np.ogrid[:IMG_N, :IMG_N]
    image = np.zeros((IMG_N, IMG_N), dtype=np.uint16)
    obj = (xx - SEED_X) ** 2 + (yy - SEED_Y) ** 2 <= 10 ** 2
    image[obj] = OBJECT_PEAK
    return image, obj


TOLERANCE = 2000.0


# ---------------------------------------------------------------------------
# 1. The detector
# ---------------------------------------------------------------------------

def test_the_plain_flood_really_does_run_away():
    """Everything below is measured against this: without a rescue the
    flood takes four times the object it was asked for."""
    image, obj = runaway_scene()
    flooded = wr.flood_region(image, SEED_X, SEED_Y, TOLERANCE)
    assert flooded.sum() == 1292
    assert obj.sum() == 317
    assert flooded.sum() > 4 * obj.sum(), "the scene does not leak"


def test_the_detector_cuts_the_leak_and_names_its_direction():
    image, obj = runaway_scene()
    flooded = wr.flood_region(image, SEED_X, SEED_Y, TOLERANCE)
    trimmed, cuts = wr.trim_directional_runaway(flooded, (SEED_Y, SEED_X))
    assert set(cuts) == {"right"}, "the seam runs right; nothing else leaked"
    assert trimmed.sum() < flooded.sum() / 3
    assert (trimmed & obj).sum() == obj.sum(), "the object itself was cut"


def test_the_detector_does_nothing_to_a_flood_that_did_not_leak():
    """The guards earn their keep here: a compact object's own profile
    widens fast enough to look like a leak to a naive step detector."""
    image, obj = clean_scene()
    flooded = wr.flood_region(image, SEED_X, SEED_Y, TOLERANCE)
    trimmed, cuts = wr.trim_directional_runaway(flooded, (SEED_Y, SEED_X))
    assert cuts == {}
    assert np.array_equal(trimmed, flooded)


def _one_wide_row() -> np.ndarray:
    """A straight band whose row 30 alone is three times as wide."""
    region = np.zeros((64, 64), dtype=bool)
    region[0:41, 10:31] = True         # 20 px wide, rows 0-40
    region[30, 0:61] = True            # one row, 60 px wide
    return region


def test_one_noisy_scanline_does_not_cut_the_object_in_half():
    """Confirmation is the difference between a leak and a speckle: the
    same row cuts at confirm=1 and is ignored at the default 2."""
    region = _one_wide_row()
    _, lenient = wr.trim_directional_runaway(
        region, (10, 20), warmup=1, confirm=1)
    _, strict = wr.trim_directional_runaway(
        region, (10, 20), warmup=1, confirm=2)
    assert lenient == {"down": 30}
    assert strict == {}


def test_the_warm_up_ignores_the_scanlines_next_to_the_click():
    """Two pixels after one pixel is a growth ratio of 2.0 and means
    nothing, so the rows out of the seed are not judged."""
    region = np.zeros((64, 64), dtype=bool)
    region[32, 32] = True
    region[33, 30:40] = True
    region[34:60, 20:50] = True
    _, early = wr.trim_directional_runaway(
        region, (32, 32), warmup=1, min_baseline=1, confirm=1)
    _, warmed = wr.trim_directional_runaway(
        region, (32, 32), warmup=12, min_baseline=8, confirm=2)
    assert early, "the unguarded detector fires on the object's own growth"
    assert warmed == {}


def test_the_minimum_baseline_refuses_to_judge_a_thread():
    """Below the baseline the object has no established width to compare
    a candidate against, so no cut is made whatever the ratio."""
    region = np.zeros((64, 64), dtype=bool)
    region[20:40, 32] = True           # a one-pixel thread
    region[40:60, 10:50] = True        # opening into a slab
    _, tiny_base = wr.trim_directional_runaway(
        region, (25, 32), warmup=2, min_baseline=1, confirm=2)
    _, real_base = wr.trim_directional_runaway(
        region, (25, 32), warmup=2, min_baseline=8, confirm=2)
    assert tiny_base, "a 1 px baseline calls every widening a leak"
    assert real_base == {}


# ---------------------------------------------------------------------------
# 2. The intensity border
# ---------------------------------------------------------------------------

def test_a_dimmer_seam_is_separated_by_tolerance_not_by_a_straight_cut():
    image, obj = runaway_scene(seam_peak=36000)
    region, report = wr.wand_region(image, SEED_X, SEED_Y, 6000.0,
                                    max_pixels=1_000_000)
    assert report["cuts"] == ["right"], "the scene must leak to be a test"
    assert report["intensity_border"] is True
    assert report["refined_tolerance"] < 6000.0
    # The re-flood found a tolerance that excludes the 4000-count gap
    # between seam and object, so the whole object survives and none of the
    # bright field does.
    assert (region & obj).sum() >= obj.sum() - 2
    assert (region & ~obj).sum() == 0


def test_an_equally_bright_seam_leaves_the_cut_standing():
    """No tolerance separates two things of the same value, so the search
    finds nothing and the straight cut is what remains to be tapered."""
    image, obj = runaway_scene()
    region, report = wr.wand_region(image, SEED_X, SEED_Y, TOLERANCE,
                                    max_pixels=1_000_000)
    assert report["cuts"] == ["right"]
    assert report["intensity_border"] is False
    assert (region & obj).sum() == obj.sum()
    assert region.sum() < 500, "the field came along anyway"


def test_more_search_steps_do_not_lose_the_object():
    image, obj = runaway_scene(seam_peak=36000)
    coarse, _ = wr.wand_region(image, SEED_X, SEED_Y, 6000.0,
                               max_pixels=1_000_000, intensity_steps=3)
    fine, _ = wr.wand_region(image, SEED_X, SEED_Y, 6000.0,
                             max_pixels=1_000_000, intensity_steps=14)
    for region in (coarse, fine):
        assert (region & obj).sum() >= obj.sum() - 2
        assert (region & ~obj).sum() == 0


# ---------------------------------------------------------------------------
# 3. The gradient taper
# ---------------------------------------------------------------------------

def test_the_taper_keeps_the_object_and_moves_the_edge_off_the_cut():
    """A directional cut is a straight line down one column. The taper is
    allowed to move it, and does, without giving the leak back."""
    image, obj = runaway_scene()
    flooded = wr.flood_region(image, SEED_X, SEED_Y, TOLERANCE)
    straight, cuts = wr.trim_directional_runaway(flooded, (SEED_Y, SEED_X))
    tapered = wr.taper_region_to_intensity(image, flooded, straight,
                                           (SEED_Y, SEED_X))
    assert cuts == {"right": 50}
    assert (tapered & obj).sum() == obj.sum()
    assert tapered.sum() <= straight.sum()
    assert not np.array_equal(tapered, straight), "the edge did not move"
    assert tapered[:, cuts["right"]:].sum() == 0, "the taper crossed the cut"


def test_the_taper_never_leaves_the_flood_and_always_keeps_the_click():
    image, _ = runaway_scene()
    flooded = wr.flood_region(image, SEED_X, SEED_Y, TOLERANCE)
    straight, _ = wr.trim_directional_runaway(flooded, (SEED_Y, SEED_X))
    for sigma, margin, inset in ((0.0, 1, 0), (2.0, 8, 3), (5.0, 40, 12)):
        tapered = wr.taper_region_to_intensity(
            image, flooded, straight, (SEED_Y, SEED_X),
            sigma=sigma, margin=margin, foreground_erode=inset)
        assert tapered[SEED_Y, SEED_X], f"lost the click at sigma={sigma}"
        assert not (tapered & ~flooded).any(), "grew outside the flood"


def test_the_taper_declines_when_nothing_was_thrown_away():
    """With no discarded part there is no band to decide, so the answer is
    the input rather than a watershed of the whole image."""
    image, _ = clean_scene()
    flooded = wr.flood_region(image, SEED_X, SEED_Y, TOLERANCE)
    tapered = wr.taper_region_to_intensity(image, flooded, flooded,
                                           (SEED_Y, SEED_X))
    assert np.array_equal(tapered, flooded)


# ---------------------------------------------------------------------------
# 4. The pixel budget
# ---------------------------------------------------------------------------

def test_the_budget_keeps_the_pixels_nearest_the_click():
    region = np.ones((64, 64), dtype=bool)
    kept = wr.cap_region_from_seed(region, (32, 32), 300)
    assert kept.sum() == 300
    assert kept[32, 32]
    _, pieces = label(kept, np.ones((3, 3)))
    assert pieces == 1, "the kept piece is not one object"
    ys, xs = np.nonzero(kept)
    reach = np.max(np.abs(ys - 32) + np.abs(xs - 32))
    assert reach < 20, "the budget was spent far from the click"


def test_the_budget_cannot_jump_a_gap_to_a_brighter_patch():
    """Nearest is measured through the flood, so an unreachable blob is
    not eligible however close it looks in a straight line."""
    region = np.zeros((64, 64), dtype=bool)
    region[30:35, 30:35] = True            # 25 px, holds the click
    region[30:50, 40:60] = True            # 400 px, not connected
    kept = wr.cap_region_from_seed(region, (32, 32), 300)
    assert kept.sum() == 25
    assert not kept[:, 40:].any()


def test_an_over_budget_flood_is_salvaged_or_refused_and_says_which():
    flat = np.full((64, 64), 1000, dtype=np.uint16)
    salvaged, kept_report = wr.wand_region(flat, 32, 32, 50.0, max_pixels=300)
    refused, refuse_report = wr.wand_region(flat, 32, 32, 50.0, max_pixels=300,
                                            salvage_over_cap=False)
    assert 0 < salvaged.sum() <= 300
    assert kept_report["capped"] is True and kept_report["rejected"] is False
    assert refused.sum() == 0
    assert refuse_report["rejected"] is True


# ---------------------------------------------------------------------------
# 5. The whole flood, and its report
# ---------------------------------------------------------------------------

def test_the_rescues_are_inert_on_a_flood_that_did_not_run_away():
    """The claim the defaults rest on. If this ever fails, the rescues are
    editing every object rather than the ones that escaped."""
    image, _ = clean_scene()
    plain = wr.flood_region(image, SEED_X, SEED_Y, TOLERANCE)
    rescued, report = wr.wand_region(image, SEED_X, SEED_Y, TOLERANCE,
                                     max_pixels=1_000_000)
    assert np.array_equal(plain, rescued)
    assert report["cuts"] == [] and report["tapered"] is False
    assert report["capped"] is False


def test_switching_the_detector_off_gives_the_runaway_back():
    """The switch is real: the same click, one setting apart, is the whole
    field or the object."""
    image, obj = runaway_scene()
    off, _ = wr.wand_region(image, SEED_X, SEED_Y, TOLERANCE,
                            max_pixels=1_000_000, trim_runaway=False)
    on, _ = wr.wand_region(image, SEED_X, SEED_Y, TOLERANCE,
                           max_pixels=1_000_000)
    assert off.sum() == 1292
    assert on.sum() < off.sum() / 3
    assert (on & obj).sum() == obj.sum()


def test_the_report_counts_what_was_flooded_and_what_was_kept():
    image, _ = runaway_scene()
    region, report = wr.wand_region(image, SEED_X, SEED_Y, TOLERANCE,
                                    max_pixels=1_000_000)
    assert report["flooded_px"] == 1292
    assert report["kept_px"] == int(region.sum())
    assert report["kept_px"] < report["flooded_px"]


def test_a_click_outside_the_image_floods_nothing():
    image, _ = clean_scene()
    region, report = wr.wand_region(image, -1, 5, TOLERANCE)
    assert region.sum() == 0 and report["flooded_px"] == 0


def test_unknown_settings_keys_are_ignored_rather_than_crashing():
    """The canvas builds this dict; a stale key from an older panel must
    not take the wand out."""
    image, _ = clean_scene()
    region, _ = wr.wand_region(image, SEED_X, SEED_Y, TOLERANCE,
                               max_pixels=1_000_000, wand_tol_pct=5.0)
    assert region.sum() == wr.flood_region(
        image, SEED_X, SEED_Y, TOLERANCE).sum()


def test_the_flood_matches_the_wands_own_neighbourhood_rule():
    """Four-connected, and inclusive at the tolerance -- the rule
    :func:`spacr.qt.mask_engine.magic_wand` fills by. A diagonal-only
    neighbour is not reachable."""
    image = np.zeros((9, 9), dtype=np.uint16)
    image[4, 4] = 100
    image[5, 5] = 100          # touches the seed only at a corner
    region = wr.flood_region(image, 4, 4, 0.0)
    assert region[4, 4] and not region[5, 5]
    assert wr.flood_region(image, 4, 4, 100.0).sum() == image.size


# ---------------------------------------------------------------------------
# 6. Through the widget
# ---------------------------------------------------------------------------

def canvas_xy(canvas, img_x: float, img_y: float) -> tuple:
    """Canvas-local point at the centre of image pixel ``(img_x, img_y)``."""
    pixmap = canvas.pixmap()
    off_x = (canvas.width() - pixmap.width()) // 2
    off_y = (canvas.height() - pixmap.height()) // 2
    x0, y0, x1, y1 = canvas._viewport_bounds()
    return (off_x + (img_x - x0 + 0.5) * pixmap.width() / (x1 - x0),
            off_y + (img_y - y0 + 0.5) * pixmap.height() / (y1 - y0))


def left_press(canvas, ix, iy):
    x, y = canvas_xy(canvas, ix, iy)
    pos = QPointF(float(x), float(y))
    return QMouseEvent(QEvent.Type.MouseButtonPress, pos, pos,
                       Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)


@pytest.fixture
def wand_canvas(qtbot, qt_theme_applied) -> _MaskCanvas:
    """A canvas holding the runaway scene with an empty mask."""
    image, _ = runaway_scene()
    c = _MaskCanvas()
    qtbot.addWidget(c)
    c.resize(CANVAS_W, CANVAS_H)
    c.set_image_and_mask(image, np.zeros((IMG_N, IMG_N), dtype=np.uint16))
    assert c.pixmap().width() == PIXMAP_N, "geometry assumption broke"
    c.mode = MODE_WAND_ADD
    c.wand_relative = False
    c.wand_tolerance = TOLERANCE
    c.wand_max_pixels = 1_000_000
    return c


def test_a_wand_click_on_the_canvas_takes_the_object_not_the_field(wand_canvas):
    _, obj = runaway_scene()
    wand_canvas.mousePressEvent(left_press(wand_canvas, SEED_X, SEED_Y))
    painted = wand_canvas.mask > 0
    assert (painted & obj).sum() == obj.sum(), "the object was not taken"
    assert painted.sum() < 500, "the bright field came along"


def test_the_same_click_with_the_detector_off_takes_the_field(wand_canvas):
    wand_canvas.wand_trim_runaway = False
    wand_canvas.mousePressEvent(left_press(wand_canvas, SEED_X, SEED_Y))
    assert int((wand_canvas.mask > 0).sum()) == 1292


def test_the_click_records_why_the_wand_took_what_it_took(wand_canvas):
    wand_canvas.mousePressEvent(left_press(wand_canvas, SEED_X, SEED_Y))
    detail = (wand_canvas.last_edit or {}).get("detail") or {}
    assert detail["cuts"] == ["right"]
    assert detail["flooded_px"] == 1292
    assert detail["kept_px"] == int((wand_canvas.mask > 0).sum())
    assert detail["tapered"] is True


def test_a_refused_flood_leaves_the_mask_alone(wand_canvas):
    wand_canvas.wand_salvage_over_cap = False
    wand_canvas.wand_trim_runaway = False
    wand_canvas.wand_max_pixels = 100
    wand_canvas.mousePressEvent(left_press(wand_canvas, SEED_X, SEED_Y))
    assert not (wand_canvas.mask > 0).any()
    assert (wand_canvas.last_edit or {})["detail"]["rejected"] is True


def test_the_canvas_hands_the_flood_exactly_the_keys_it_understands(
        wand_canvas):
    assert set(wand_canvas.wand_rescue_settings()) == set(wr.RESCUE_DEFAULTS)


# ---------------------------------------------------------------------------
# 7. Through the panel
# ---------------------------------------------------------------------------

@pytest.fixture
def panel(qtbot, qt_theme_applied):
    from spacr.qt.screens.make_masks import MakeMasksScreen
    s = MakeMasksScreen()
    qtbot.addWidget(s)
    return s


@pytest.mark.parametrize("widget_name, attribute, value", [
    ("_wand_salvage", "wand_salvage_over_cap", False),
    ("_wand_runaway_ratio", "wand_runaway_ratio", 3.5),
    ("_wand_runaway_warmup", "wand_runaway_warmup", 30),
    ("_wand_runaway_min_base", "wand_runaway_min_base", 25),
    ("_wand_runaway_confirm", "wand_runaway_confirm", 5),
    ("_wand_intensity_border", "wand_intensity_border", False),
    ("_wand_intensity_steps", "wand_intensity_steps", 12),
    ("_wand_gradient_taper", "wand_gradient_taper", False),
    ("_wand_gradient_sigma", "wand_gradient_sigma", 4.5),
    ("_wand_gradient_margin", "wand_gradient_margin", 20),
    ("_wand_gradient_erode", "wand_gradient_erode", 7),
])
def test_every_rescue_control_reaches_the_canvas(panel, widget_name,
                                                 attribute, value):
    widget = getattr(panel, widget_name)
    if isinstance(value, bool):
        widget.setChecked(value)
    else:
        widget.setValue(value)
    assert getattr(panel._canvas, attribute) == value


def test_the_group_checkbox_is_the_detectors_master_switch(panel):
    assert panel._wand_runaway_group.isChecked(), "the detector ships on"
    panel._wand_runaway_group.setChecked(False)
    assert panel._canvas.wand_trim_runaway is False
    panel._wand_runaway_group.setChecked(True)
    assert panel._canvas.wand_trim_runaway is True


def test_a_setting_that_steers_nothing_is_disabled(panel):
    """The search precision and the taper's three numbers only mean
    something while their own switch is on."""
    panel._wand_intensity_border.setChecked(False)
    assert not panel._wand_intensity_steps.isEnabled()
    panel._wand_intensity_border.setChecked(True)
    assert panel._wand_intensity_steps.isEnabled()

    panel._wand_gradient_taper.setChecked(False)
    for w in (panel._wand_gradient_sigma, panel._wand_gradient_margin,
              panel._wand_gradient_erode):
        assert not w.isEnabled()
    panel._wand_gradient_taper.setChecked(True)
    for w in (panel._wand_gradient_sigma, panel._wand_gradient_margin,
              panel._wand_gradient_erode):
        assert w.isEnabled()


def test_every_rescue_control_carries_a_tooltip(panel):
    """A knob nobody can explain is a knob nobody should turn.

    Looked for on the label as well as on the editor: the screen runs
    :func:`spacr.qt.screens.settings_model.retarget_field_tooltips`, which
    moves a spin box's help onto the label that names the setting, so a
    spin box's own tooltip is empty by the time the panel is built.
    """
    from spacr.qt.screens.settings_model import _sibling_label_for

    for name in ("_wand_salvage", "_wand_runaway_ratio",
                 "_wand_runaway_warmup", "_wand_runaway_min_base",
                 "_wand_runaway_confirm", "_wand_intensity_border",
                 "_wand_intensity_steps", "_wand_gradient_taper",
                 "_wand_gradient_sigma", "_wand_gradient_margin",
                 "_wand_gradient_erode"):
        widget = getattr(panel, name)
        label = _sibling_label_for(widget)
        help_text = widget.toolTip() or (label.toolTip() if label else "")
        assert len(help_text) > 40, name
    assert panel._wand_runaway_group.toolTip()
    assert panel._wand_edge_group.toolTip()


def test_the_panel_defaults_match_the_canvas_defaults(panel):
    """The panel opens saying what the wand will actually do."""
    settings = panel._canvas.wand_rescue_settings()
    assert settings["trim_runaway"] is panel._wand_runaway_group.isChecked()
    assert settings["runaway_ratio"] == panel._wand_runaway_ratio.value()
    assert settings["runaway_warmup"] == panel._wand_runaway_warmup.value()
    assert settings["runaway_min_base"] == panel._wand_runaway_min_base.value()
    assert settings["runaway_confirm"] == panel._wand_runaway_confirm.value()
    assert settings["intensity_steps"] == panel._wand_intensity_steps.value()
    assert settings["gradient_sigma"] == panel._wand_gradient_sigma.value()
    assert settings["gradient_margin"] == panel._wand_gradient_margin.value()
    assert settings["gradient_erode"] == panel._wand_gradient_erode.value()
    assert settings["salvage_over_cap"] is panel._wand_salvage.isChecked()


def test_a_wand_with_no_image_reports_the_same_shape_as_a_real_click():
    """The ledger writes the report verbatim, so the nothing-to-do path
    must not hand it a different set of keys."""
    real = wr.magic_wand(*_clean_arguments())[1]
    empty = wr.magic_wand(None, None, 0, 0, 1.0)[1]
    assert set(empty) == set(real)
    assert empty["rejected"] is True


def _clean_arguments():
    image, _ = clean_scene()
    return (image, np.zeros((IMG_N, IMG_N), dtype=np.uint16),
            SEED_X, SEED_Y, TOLERANCE)


# ---------------------------------------------------------------------------
# a seam can run any of four ways
# ---------------------------------------------------------------------------

def _band_leaking(direction: str) -> np.ndarray:
    """A narrow band from the seed that suddenly widens in ``direction``.

    Built symmetrically so the same shape can be rotated into each of the four
    directions: the detector's arithmetic is per-direction and a bug in one of
    them cannot be seen by testing another.
    """
    region = np.zeros((80, 80), dtype=bool)
    region[30:51, 30:51] = True                  # the object at the seed
    if direction == "down":
        region[51:76, 5:76] = True
    elif direction == "up":
        region[5:30, 5:76] = True
    elif direction == "right":
        region[5:76, 51:76] = True
    else:
        region[5:76, 5:30] = True
    return region


@pytest.mark.parametrize("direction", ["up", "down", "left", "right"])
def test_a_seam_is_cut_whichever_way_it_runs(direction):
    """Each direction is separate arithmetic on a separately built profile.

    "up" and "left" reverse their profile so index 0 is at the click, and
    their cut is ``seed - offset`` where the others are ``seed + offset``.
    Two sign errors and two slice directions live here, and a test that only
    leaks rightward -- which is what the suite had -- cannot see any of them.
    """
    region = _band_leaking(direction)
    seed = (40, 40)

    trimmed, cuts = wr.trim_directional_runaway(
        region, seed, warmup=2, min_baseline=4, confirm=2)

    assert direction in cuts, f"the {direction} seam was not detected"
    assert trimmed.sum() < region.sum(), "nothing was actually removed"
    # The object around the click survives the cut.
    assert trimmed[40, 40], "the cut removed the pixel that was clicked"


@pytest.mark.parametrize("direction", ["up", "down", "left", "right"])
def test_the_cut_is_reported_in_image_coordinates(direction):
    """``cuts`` is what the caller draws, so it must be an image coordinate.

    An offset from the seed would be silently wrong for "up" and "left",
    where the profile is reversed -- and drawing the cut in the wrong place
    is worse than not drawing it, because it says the rescue did something
    it did not.
    """
    region = _band_leaking(direction)
    seed_y, seed_x = 40, 40

    _trimmed, cuts = wr.trim_directional_runaway(
        region, (seed_y, seed_x), warmup=2, min_baseline=4, confirm=2)

    cut = cuts[direction]
    if direction in ("up", "down"):
        assert 0 <= cut < region.shape[0]
        assert (cut < seed_y) if direction == "up" else (cut > seed_y)
    else:
        assert 0 <= cut < region.shape[1]
        assert (cut < seed_x) if direction == "left" else (cut > seed_x)


@pytest.mark.parametrize("seed", [(-1, 10), (10, -1), (999, 10), (10, 999)])
def test_a_seed_outside_the_region_is_returned_untouched(seed):
    """A click can arrive after the view scrolled, or on a resized image.

    Returning the region unchanged is the only safe answer: the profiles are
    built by slicing at the seed, and an out-of-range index would either
    raise or silently profile the wrong half of the object.
    """
    region = _band_leaking("down")

    trimmed, cuts = wr.trim_directional_runaway(region, seed)

    assert cuts == {}
    assert np.array_equal(trimmed, region)
    assert trimmed is not region, "the caller's array was handed back"

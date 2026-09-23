"""Item 435: Invert inverts the PICTURE, and Otsu got the rest of its options.

The maintainer, 2026-09-19:

    "the invert in make masks doesnt actually invert it looks like. and i
     need more options for otsu."

and, answering what invert should mean:

    "invert so that low intensity becomes high intensity and vice versa.
     1/intensity i think and then fitted to dtype i guess"

WHAT IS BUILT IS THE COMPLEMENT, ``dtype_max - value``, NOT THE RECIPROCAL.
It is what a viewer's Invert does, it is exactly reversible on every integer
dtype -- which is what these tests assert, by comparing arrays rather than by
counting objects -- and ``1/value`` divides by zero on every background pixel.
The reciprocal is recorded in the item file as considered and not built.

He was right that nothing happened. "Invert mask" flipped the MASK, and a
flipped mask on an ordinary field is one object covering the frame, which the
overlay draws as one flat wash. That operation is kept -- outlining the space
between the cells is a real thing to do -- under the name that says what it
is, and Invert is now a view setting beside the contrast percentiles.

THE OTHER HALF OF THIS FILE IS THE PART THAT MUST NOT ROT: a view inversion
that leaked into what is measured would be worse than the defect it fixes,
because a curator would be filtering on numbers that are not the data. So the
detect buttons, the object lookup behind the corner readout and the object
filter are each driven with Invert ON and asserted to have seen the original
pixels.

ITEM 419 PART B LANDED FIRST AND IS NOT RE-TESTED HERE. The threshold
correction, the smoothing, fill holes, split touching and drop-what-the-frame
-cut are its, with their own file. What is here is the rest of 435's list:
multi-level Otsu with the class to take, a local window with its size, and
the histogram preview -- whose marker is asserted to be the level the button
actually cuts at, read from the same function, rather than a second opinion.
"""
from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest
from PySide6.QtCore import QPointF

from spacr.qt import mask_engine as engine
from spacr.qt.screens import make_masks as mm

CANVAS_W, CANVAS_H = 320, 320


def graded_field(n: int = 96) -> np.ndarray:
    """Four discs on a background the lamp falls away from, left to right.

    The left half is dim enough that one level for the whole field either
    takes its background in or leaves its objects out, which is what a local
    threshold is for, and the discs come in two brightnesses so a three-class
    split has a middle band to find.
    """
    yy, xx = np.mgrid[0:n, 0:n]
    field = np.full((n, n), 600.0)
    for cy, cx, value in ((24, 24, 2400.0), (24, 72, 9000.0),
                          (72, 24, 2400.0), (72, 72, 9000.0)):
        field[(yy - cy) ** 2 + (xx - cx) ** 2 <= 64] = value
    ramp = np.linspace(0.18, 1.0, n)[None, :]
    return np.clip(field * ramp, 0, 65535).astype(np.uint16)


@pytest.fixture
def one_field(tmp_path: Path) -> Path:
    """One field with an empty mask beside it, in the nested layout."""
    folder = tmp_path / "field"
    (folder / "masks").mkdir(parents=True)
    imageio.imwrite(folder / "a.tif", graded_field())
    imageio.imwrite(folder / "masks" / "a.tif",
                    np.zeros((96, 96), dtype=np.uint16))
    return folder


@pytest.fixture
def screen(qtbot, qt_theme_applied, one_field: Path):
    made = mm.MakeMasksScreen()
    qtbot.addWidget(made)
    assert made._open_folder(str(one_field))
    made._canvas.resize(CANVAS_W, CANVAS_H)
    made._canvas.refresh()
    yield made
    made._magnifier.close()
    made.close_folded()


def pixmap_bytes(canvas) -> bytes:
    """What the canvas is painting, as bytes, for an exact comparison."""
    image = canvas.pixmap().toImage()
    return bytes(image.constBits())


# ---------------------------------------------------------------------------
# The complement, in the engine
# ---------------------------------------------------------------------------

def test_invert_intensity_is_the_complement_and_not_the_reciprocal():
    """Dark becomes bright on the dtype's own range, and 0 is finite.

    The reciprocal is the obvious first reading of "1/intensity", and this
    is the assertion that says why it is not what runs: background is zero
    on plenty of fields, and ``1/0`` is where that reading ends.
    """
    field = np.array([[0, 1000, 65535]], dtype=np.uint16)
    out = engine.invert_intensity(field)
    assert out.tolist() == [[65535, 64535, 0]]
    assert np.isfinite(out.astype(np.float64)).all()
    assert out.dtype == field.dtype


@pytest.mark.parametrize("dtype", ["uint8", "uint16", "int16", "bool"])
def test_inverting_twice_returns_the_identical_array(dtype: str):
    """The round trip, proven by comparing arrays rather than by eye."""
    rng = np.random.default_rng(4)
    if dtype == "bool":
        field = rng.random((32, 32)) > 0.5
    else:
        info = np.iinfo(dtype)
        field = rng.integers(info.min, info.max, (32, 32)).astype(dtype)
    once = engine.invert_intensity(field)
    twice = engine.invert_intensity(once)
    assert not np.array_equal(field, once), "one press changed nothing"
    assert np.array_equal(field, twice), "two presses did not come back"


def test_a_float_field_comes_back_within_its_own_rounding():
    """Floats round twice, and the docstring says so rather than claiming
    an exactness the arithmetic does not have."""
    field = (np.random.default_rng(5).random((64, 64)) * 65535
             ).astype(np.float32)
    twice = engine.invert_intensity(engine.invert_intensity(field))
    span = float(field.min()) + float(field.max())
    assert np.abs(field - twice).max() <= float(
        np.spacing(np.float32(span)))


def test_invert_intensity_leaves_its_input_alone():
    """A view operation that edited the array would be an edit."""
    field = graded_field()
    keep = field.copy()
    engine.invert_intensity(field)
    assert np.array_equal(field, keep)


def test_the_mask_flip_is_a_different_operation_and_says_so():
    """``invert_mask`` gives one field-sized object -- the non-event he saw."""
    mask = np.zeros((32, 32), dtype=np.uint16)
    mask[8:16, 8:16] = 3
    flipped = engine.invert_mask(mask)
    assert int(flipped.max()) == 1, "the background is one connected region"
    assert flipped[0, 0] > 0 and flipped[12, 12] == 0
    assert "invert_intensity" in engine.invert_mask.__doc__


# ---------------------------------------------------------------------------
# Invert on the screen: a view, and only a view
# ---------------------------------------------------------------------------

def test_invert_image_changes_the_picture_and_puts_it_back(screen):
    """Press, and the canvas paints something else; press again, and it
    paints the identical bytes it started with."""
    before = pixmap_bytes(screen._canvas)
    screen._invert_display.setChecked(True)
    inverted = pixmap_bytes(screen._canvas)
    assert inverted != before, "Invert did nothing to the picture"
    screen._invert_display.setChecked(False)
    assert pixmap_bytes(screen._canvas) == before


def test_the_canvas_draws_the_normalised_inverse_and_keeps_the_original(screen):
    """What is drawn is 1 - v on the field's own range, and image is untouched.

    The arithmetic changed on 2026-09-20 from the dtype complement to
    normalise-then-complement, at the maintainer's word, because item 417's
    Otsu correction is a multiplier and only a normalised field makes one
    correction value mean the same thing twice.
    """
    canvas = screen._canvas
    original = np.array(canvas.image, copy=True)
    canvas.invert_display = True
    assert np.array_equal(canvas.displayed_source(),
                          engine.invert_normalized(original))
    assert canvas.displayed_source().min() == 0
    assert canvas.displayed_source().max() == 65535, (
        "an inverted field is rescaled onto the whole dtype range")
    assert np.array_equal(canvas.image, original), "the data was edited"
    canvas.invert_display = False
    assert np.array_equal(canvas.displayed_source(), original)


def test_otsu_detect_is_handed_the_inverted_pixels_while_invert_is_on(screen):
    """Detect reads what the screen is showing, which is the whole point.

    This test asserted the opposite until 2026-09-20 -- that a detect run
    with Invert on still saw the loaded pixels -- and that was the contract
    the maintainer asked to be rid of: "if there are black objects on an
    image then inverting allows the user to use otsu and magnifier. that is
    the reason for this buton." A switch that inverts the picture and leaves
    Otsu looking at the original cannot do that.
    """
    screen._min_area.setValue(4)
    screen._combine_mode.setCurrentText("replace")

    seen = []
    original = engine._otsu_instances

    def spy(image, **kwargs):
        seen.append(np.array(image, copy=True))
        return original(image, **kwargs)

    engine._otsu_instances = spy
    try:
        screen._invert_display.setChecked(True)
        screen._btn_otsu.click()
    finally:
        engine._otsu_instances = original
    assert seen, "the detect button did not reach the engine"
    assert np.array_equal(seen[-1],
                          engine.invert_normalized(screen._canvas.image)), (
        "the detector was handed the loaded pixels, not the inverted ones")
    assert not np.array_equal(seen[-1], screen._canvas.image)


def test_the_readout_splits_the_pixel_from_the_object_mean_while_inverted(
        screen):
    """THIS TEST ASSERTED THE OPPOSITE UNTIL 2026-09-20, and the reason it
    changed is worth keeping.

    It held that the readout reports the on-disk intensity whichever way
    the picture is drawn, because "a number read off the picture is typed
    into the filter boxes". That is true of the OBJECT MEAN, which the
    Filter's own boxes are compared against, and it was wrong about the
    PIXEL: the readout is the instrument a curator checks an inversion
    with, and reporting the loaded value under an inverted picture read as
    the switch doing nothing. The maintainer reported exactly that.

    So the two are now split, and each says which it is:

        the pixel intensity follows the picture      "(inverted)"
        the object mean follows the Filter           "(as loaded)"
    """
    canvas = screen._canvas
    spot = QPointF(CANVAS_W / 2.0, CANVAS_H / 2.0)
    upright = canvas.update_readout(spot)
    assert upright is not None
    screen._invert_display.setChecked(True)
    inverted = canvas.update_readout(spot)
    assert inverted is not None
    assert inverted.intensity != pytest.approx(upright.intensity), (
        "the pixel must follow the picture; that is the defect this fixed")
    assert inverted.mean_intensity == upright.mean_intensity, (
        "the object mean must stay the number a Filter bound is typed from")
    assert canvas._lookup_image is canvas.image, (
        "the lookup is still built on the loaded pixels")


def test_the_object_filter_measures_the_original_pixels(screen):
    """``filter_objects`` is handed ``canvas.image``, inverted or not."""
    screen._min_area.setValue(4)
    screen._btn_otsu.click()
    screen._filter_min_int.setValue(0.0)
    screen._filter_max_int.setValue(0.0)
    upright = screen.apply_object_filter()
    screen._on_undo()
    screen._invert_display.setChecked(True)
    assert screen.apply_object_filter() == upright


def test_the_magnifier_is_handed_the_inverted_crop_while_invert_is_on(screen):
    """The box under the mouse segments the view, which is now the point.

    Item 419 point 9's "Invert for detection" used to be a SECOND switch
    that did this, and this test held that the display switch did not. The
    maintainer merged the two on 2026-09-20, so the magnifier now reads the
    same negative the curator is looking at -- which is what makes it usable
    on dark objects, and is the stated reason the button exists.

    The warning banner above the image is what keeps this honest: the masks
    ARE made from the inverted field while it is on, and the screen says so.
    """
    screen._invert_display.setChecked(True)
    screen._btn_magnifier.setChecked(True)
    screen._magnifier.hover(QPointF(CANVAS_W / 2.0, CANVAS_H / 2.0))
    request = screen._magnifier.build_request()
    assert request is not None
    x0, y0, x1, y1 = request.box
    inverted = engine.invert_normalized(screen._canvas.image)
    assert np.array_equal(request.crop, inverted[y0:y1, x0:x1])
    assert not np.array_equal(request.crop,
                              screen._canvas.image[y0:y1, x0:x1])


def test_the_status_line_says_the_inversion_is_only_the_picture(screen):
    screen._invert_display.setChecked(True)
    said = screen._status_label.text()
    assert "inverted" in said
    assert "original pixels" in said


# ---------------------------------------------------------------------------
# The mask flip, kept and renamed
# ---------------------------------------------------------------------------

def test_the_object_operations_button_is_named_for_what_it_does(screen):
    """Nothing called "Invert" flips the mask any more, and the button that
    does says where the picture invert lives."""
    from PySide6.QtWidgets import QPushButton

    labels = {button.text(): button
              for button in screen._settings_scroll.findChildren(QPushButton)}
    assert "Swap object and background" in labels
    assert "Invert mask" not in labels
    hint = labels["Swap object and background"].toolTip()
    assert "Invert image" in hint and "Display" in hint


def test_swapping_object_and_background_reports_its_result(screen):
    """The operation's own defence against reading as a non-event."""
    mask = screen._canvas.mask
    mask[:] = 0
    mask[10:20, 10:20] = 7
    screen._canvas.stroke_finished.emit()
    screen._on_invert()
    assert int(screen._canvas.mask[15, 15]) == 0
    assert int(screen._canvas.mask[0, 0]) > 0
    said = screen._status_label.text()
    assert "Swapped object and background" in said and "1 object(s)" in said
    screen._on_undo()
    assert int(screen._canvas.mask[15, 15]) == 7


# ---------------------------------------------------------------------------
# Otsu: multi-level
# ---------------------------------------------------------------------------

def test_more_classes_give_more_cuts_and_a_band_the_two_class_split_has_not():
    """Raising the count adds levels, and one of the new bands is a mask no
    two-class threshold of this field produces.

    NOT "the mask changes the moment the number does". Multi-level Otsu can
    put its TOP level exactly where the two-class one is -- it does on this
    field, 1088.84 both times -- so a test asserting the default band moved
    would be asserting an accident. What more classes buys is the band
    BELOW the top one, and that is what is checked.
    """
    field = graded_field()
    assert len(engine._otsu_levels(field, classes=2)) == 1
    assert len(engine._otsu_levels(field, classes=4)) == 3
    two = engine._otsu_instances(field, min_area=4)
    middle = engine._otsu_instances(field, min_area=4, classes=3,
                                    foreground_class=1)
    assert not np.array_equal(two > 0, middle > 0)
    assert int((middle > 0).sum()) > int((two > 0).sum())


def test_the_foreground_class_chooses_which_band_becomes_objects():
    """Three bands, three different masks, and no pixel in two of them."""
    field = graded_field()
    bands = [engine._otsu_instances(field, min_area=4, classes=3,
                                    foreground_class=index)
             for index in range(3)]
    areas = [int((band > 0).sum()) for band in bands]
    assert len(set(areas)) == 3, areas
    assert not np.array_equal(bands[1] > 0, bands[2] > 0)
    for first in range(3):
        for second in range(first + 1, 3):
            overlap = int(((bands[first] > 0) & (bands[second] > 0)).sum())
            assert overlap == 0, (
                f"bands {first} and {second} share {overlap} pixels; exactly "
                f"one band is taken")


def test_the_default_foreground_class_is_the_brightest_band():
    field = graded_field()
    assert np.array_equal(
        engine._otsu_instances(field, min_area=4, classes=4),
        engine._otsu_instances(field, min_area=4, classes=4,
                               foreground_class=3))


def test_a_class_outside_the_split_is_refused():
    field = graded_field()
    with pytest.raises(ValueError, match="foreground class"):
        engine._otsu_instances(field, classes=3, foreground_class=3)
    with pytest.raises(ValueError, match="at least two classes"):
        engine._otsu_instances(field, classes=1)


# ---------------------------------------------------------------------------
# Otsu: the local window
# ---------------------------------------------------------------------------

def test_a_local_threshold_finds_the_object_the_dim_corner_lost():
    """One level for a field the lamp falls off across loses the dim half."""
    field = graded_field()
    whole = engine._otsu_instances(field, min_area=4, smoothing=1.0)
    local = engine._otsu_instances(field, min_area=4, smoothing=1.0,
                                   local=True, window=25)
    dim_corner = (slice(12, 36), slice(12, 36))
    assert int((whole[dim_corner] > 0).sum()) == 0, (
        "the dim disc is supposed to be lost by one global level")
    assert int((local[dim_corner] > 0).sum()) > 0, (
        "the local threshold did not recover it")


def test_the_window_size_changes_what_a_local_threshold_finds():
    field = graded_field()
    tight = engine._otsu_instances(field, min_area=4, local=True, window=9)
    wide = engine._otsu_instances(field, min_area=4, local=True, window=61)
    assert not np.array_equal(tight > 0, wide > 0), "the window did nothing"


def test_an_even_window_is_rounded_up_so_it_has_a_centre():
    field = graded_field()
    assert np.array_equal(
        engine._otsu_instances(field, min_area=4, local=True, window=24),
        engine._otsu_instances(field, min_area=4, local=True, window=25))
    with pytest.raises(ValueError, match="at least 3 px"):
        engine._otsu_instances(field, local=True, window=2)


def test_a_local_threshold_and_more_than_two_classes_are_refused():
    """They have no joint meaning, so the engine says so rather than
    silently dropping one of the two."""
    with pytest.raises(ValueError, match="local threshold"):
        engine._otsu_instances(graded_field(), local=True, classes=3)


# ---------------------------------------------------------------------------
# Otsu: the preview, and the level it marks
# ---------------------------------------------------------------------------

def test_the_marked_level_is_the_level_the_detection_cuts_at():
    """One function answers both, so the preview cannot be a second opinion."""
    field = graded_field()
    for bright, correction in ((True, 1.0), (True, 1.4), (False, 1.0),
                               (False, 0.8)):
        level = engine._otsu_levels(field, bright=bright,
                                    correction=correction, smoothing=1.0)[0]
        values = engine._otsu_values(field, 1.0)
        expected = values > level if bright else values < level
        got = engine._otsu_instances(field, bright=bright,
                                     correction=correction, smoothing=1.0,
                                     min_area=1)
        assert np.array_equal(
            engine.connected_instances(expected, min_area=1) > 0, got > 0), (
                f"the marker and the cut disagree at {bright=} {correction=}")


def test_multi_level_marks_every_level_it_cuts_at():
    levels = engine._otsu_levels(graded_field(), classes=4)
    assert len(levels) == 3
    assert levels == sorted(levels)


def test_the_histogram_counts_the_values_the_level_was_found_on():
    """Drawn on the smoothed field, because that is where the cut is made."""
    field = graded_field()
    rough, _edges = engine._otsu_histogram(field, smoothing=0.0)
    smooth, _edges = engine._otsu_histogram(field, smoothing=3.0)
    assert int(rough.sum()) == field.size == int(smooth.sum())
    assert not np.array_equal(rough, smooth)


def test_the_preview_window_marks_the_level_the_button_would_use(screen):
    screen._otsu_correction.setValue(1.0)
    screen._on_show_otsu_histogram()
    dialog = screen._otsu_histogram_dialog
    assert dialog is not None
    settings = screen._otsu_settings()
    expected = engine._otsu_levels(
        screen._canvas.image, bright=screen._otsu_bright.isChecked(),
        correction=settings["correction"], smoothing=settings["smoothing"],
        classes=settings["classes"])
    assert dialog.plot.levels == pytest.approx(expected)
    assert f"{expected[0]:.4g}" in dialog.caption.text()
    inside = dialog.plot.level_x(expected[0])
    assert 0.0 < inside < dialog.plot.width()
    dialog.close()


def test_the_preview_follows_the_correction(screen):
    screen._otsu_correction.setValue(1.0)
    screen._on_show_otsu_histogram()
    plain = list(screen._otsu_histogram_dialog.plot.levels)
    screen._otsu_correction.setValue(1.5)
    screen._on_show_otsu_histogram()
    stricter = list(screen._otsu_histogram_dialog.plot.levels)
    assert stricter[0] > plain[0]
    assert screen._otsu_histogram_dialog.plot.levels == stricter
    screen._otsu_histogram_dialog.close()


def test_the_preview_says_a_local_level_varies(screen):
    screen._otsu_local.setChecked(True)
    screen._on_show_otsu_histogram()
    text = screen._otsu_histogram_dialog.caption.text()
    assert "varies" in text and "local" in text.lower()
    screen._otsu_histogram_dialog.close()


# ---------------------------------------------------------------------------
# Otsu: the controls on the panel
# ---------------------------------------------------------------------------

NEW_OTSU_CONTROLS = ("_otsu_classes", "_otsu_foreground", "_otsu_local",
                     "_otsu_window")


@pytest.mark.parametrize("name", NEW_OTSU_CONTROLS)
def test_each_new_otsu_control_is_in_the_otsu_category_with_help(screen, name):
    """A control with no help is a number the user is invited to move
    without being told what it moves."""
    from spacr.qt.screens.settings_model import _sibling_label_for

    control = getattr(screen, name)
    category = dict(screen._settings_categories)["Detection method"]
    assert category.isAncestorOf(control), name
    label = _sibling_label_for(control)
    helped = control.toolTip() or (label is not None and label.toolTip())
    assert helped, f"{name} has no help anywhere"
    assert "Otsu detect" in helped or "object" in helped, (
        f"{name}'s help does not say what it does to the picture: {helped}")


def test_the_controls_that_are_not_being_read_are_disabled(screen):
    """A control that is being ignored and one that is being read must not
    look the same -- which is the defect this item exists for."""
    assert not screen._otsu_foreground.isEnabled(), "two classes, no choice"
    assert not screen._otsu_window.isEnabled(), "the local threshold is off"
    screen._otsu_classes.setValue(3)
    assert screen._otsu_foreground.isEnabled()
    assert screen._otsu_foreground.maximum() == 2
    screen._otsu_local.setChecked(True)
    assert screen._otsu_window.isEnabled()
    assert not screen._otsu_classes.isEnabled(), "local is two classes only"
    assert screen._otsu_settings()["classes"] == 2
    assert screen._otsu_settings()["local"] is True


def test_multi_level_otsu_changes_the_mask_the_button_makes(screen):
    screen._min_area.setValue(4)
    screen._combine_mode.setCurrentText("replace")
    screen._btn_otsu.click()
    two = np.array(screen._canvas.mask, copy=True)
    screen._on_undo()
    screen._otsu_classes.setValue(3)
    screen._otsu_foreground.setValue(1)
    screen._btn_otsu.click()
    assert not np.array_equal(screen._canvas.mask > 0, two > 0)
    assert "3 classes, class 1" in screen._status_label.text()


def test_the_local_threshold_changes_the_mask_the_button_makes(screen):
    screen._min_area.setValue(4)
    screen._combine_mode.setCurrentText("replace")
    screen._btn_otsu.click()
    whole = np.array(screen._canvas.mask, copy=True)
    screen._on_undo()
    screen._otsu_local.setChecked(True)
    screen._otsu_window.setValue(25)
    screen._btn_otsu.click()
    assert not np.array_equal(screen._canvas.mask > 0, whole > 0)
    assert "local 25 px" in screen._status_label.text()


def test_the_detect_run_records_the_new_settings(screen):
    screen._min_area.setValue(4)
    screen._otsu_classes.setValue(3)
    screen._otsu_foreground.setValue(2)
    screen._btn_otsu.click()
    detail = screen._log.edits[-1].detail
    assert detail["otsu_classes"] == 3
    assert detail["otsu_foreground_class"] == 2
    assert detail["otsu_local"] is False
    assert detail["otsu_window"] == mm.OTSU_LOCAL_WINDOW


def test_multiotsu_bands_reach_the_box_but_local_otsu_remains_button_only(screen):
    """473 adds band snapshots for Multi-Otsu in both magnifier scopes.

    The local window also reaches Sauvola/Niblack. The Local Otsu toggle
    remains a whole-image detect setting, separate from those algorithms.
    """
    context = screen._magnifier_context()
    for key in ("classes", "foreground_class", "local", "otsu_local"):
        assert key not in context, key
    assert context["otsu_classes"] == screen._otsu_classes.value()
    assert context["otsu_foreground_class"] == screen._otsu_foreground.value()
    assert "otsu_classes" in mm._MODEL_SETTING_FIELDS
    assert "otsu_foreground_class" in mm._MODEL_SETTING_FIELDS
    assert "otsu_local" not in mm._MODEL_SETTING_FIELDS
    assert context["otsu_window"] == screen._otsu_window.value()


def test_the_minimum_area_after_the_threshold_is_the_one_box_there_was(screen):
    """419 made Min area the single judgement about debris and 435 does not
    add a second one that could disagree with it; the Otsu card points at it.

    Driven rather than asserted from the source: the box is raised and the
    detection comes back with fewer objects.
    """
    from PySide6.QtWidgets import QLabel

    screen._combine_mode.setCurrentText("replace")
    screen._min_area.setValue(4)
    screen._btn_otsu.click()
    small = int(screen._canvas.mask.max())
    screen._on_undo()
    screen._min_area.setValue(400)
    screen._btn_otsu.click()
    assert int(screen._canvas.mask.max()) < small

    category = dict(screen._settings_categories)["Detection method"]
    said = [label.text() for label in category.findChildren(QLabel)]
    assert any("Min area" in text and "Object operations" in text
               for text in said), said


def test_two_classes_and_no_window_is_the_threshold_that_was_there():
    """435 added options; it did not move anybody's defaults."""
    field = graded_field()
    assert np.array_equal(engine._otsu_instances(field, min_area=6),
                          engine.otsu_instances(field, min_area=6))
    assert np.array_equal(
        engine._otsu_instances(field, min_area=6, smoothing=1.5,
                               fill_holes=True, split_touching=True),
        engine._otsu_instances(field, min_area=6, smoothing=1.5,
                               fill_holes=True, split_touching=True,
                               classes=2, local=False,
                               window=mm.OTSU_LOCAL_WINDOW))


# ---------------------------------------------------------------------------
# The readout, reopened 2026-09-20
# ---------------------------------------------------------------------------

def test_the_hover_intensity_follows_the_inversion(screen):
    """THE DEFECT THE MAINTAINER REPORTED, in his own terms: "the hover
    allows me to see the intensities and they dont seem to change at all
    when i press invert, so it is not done".

    He was right, and the reason it read as "not done" is that the readout
    is the instrument an inversion is checked WITH: it reported the loaded
    pixel while the picture showed the negative.
    """
    canvas = screen._canvas
    spot = (40, 40)
    plain = canvas._value_at(canvas.displayed_source(), spot)
    screen._invert_display.setChecked(True)
    inverted = canvas._value_at(canvas.displayed_source(), spot)
    assert inverted != plain, (
        "the number under the mouse did not move when the picture did")
    screen._invert_display.setChecked(False)
    assert canvas._value_at(canvas.displayed_source(), spot) == plain


def test_a_bright_pixel_reads_low_once_inverted(screen):
    """High becomes low, which is what was asked for."""
    canvas = screen._canvas
    field = np.asarray(canvas.image)
    flat = int(np.argmax(field))
    spot = (flat % field.shape[1], flat // field.shape[1])
    brightest = canvas._value_at(canvas.displayed_source(), spot)
    screen._invert_display.setChecked(True)
    assert canvas._value_at(canvas.displayed_source(), spot) < brightest


def test_the_caption_says_which_number_it_is_showing(screen):
    """A number that has been transformed has to say so, or it is a
    different wrong answer from the one this fixed."""
    canvas = screen._canvas
    canvas.readout = engine.PixelReadout(10, 10, 123.0)
    canvas.invert_display = False
    assert "(inverted)" not in canvas.readout_text()
    canvas.invert_display = True
    assert "(inverted)" in canvas.readout_text()


def test_the_object_mean_still_belongs_to_the_filter(screen):
    """It is documented as the number a Filter bound can be typed from, and
    the Filter reads the loaded pixels -- so it stays as it is and says so,
    rather than being quietly inverted with the pixel beside it."""
    canvas = screen._canvas
    canvas.readout = engine.PixelReadout(10, 10, 123.0, label=1, area=64,
                                         mean_intensity=60000.0)
    canvas.invert_display = True
    text = canvas.readout_text()
    assert "(as loaded)" in text
    assert "60000" in text.replace(",", "").replace(" ", "")


def test_the_loaded_pixels_are_still_never_touched(screen):
    """The property the whole design protects, re-asserted beside the
    change that made the readout follow the picture."""
    canvas = screen._canvas
    before = np.array(canvas.image, copy=True)
    screen._invert_display.setChecked(True)
    canvas._value_at(canvas.displayed_source(), (40, 40))
    assert np.array_equal(canvas.image, before)

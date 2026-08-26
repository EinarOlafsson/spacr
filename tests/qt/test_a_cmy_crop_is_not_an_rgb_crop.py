"""The annotator's ``Channel colours`` control, measured in pixels.

CMY looked exactly like RGB, and reading the code said it should not: the
mapping in :func:`spacr.crops.apply_display_primaries` is a real subtractive
substitution, and the loader passes a ``display_primaries`` through to it.

The break was between the two. ``_SettingsDialog.collect`` wrote back every
other editor and not that one, so the combo moved, the dialog was accepted,
and the settings object the page loader reads still said ``"rgb"``. A
decorative control is worse than a missing one: it is a setting that lies.

So nothing here reads the mode off the settings object. Every assertion is
made on the PIXELS of one crop taken through the same route the grid takes
it -- ``_load_thumb_image_worker``, the function the page loader hands each
row to -- after the mode has been chosen through the dialog and collected.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def three_stripe_crop(tmp_path):
    """One crop with a red, a green and a blue band, and nothing else.

    Deliberately not noise: a field whose three planes are separated in
    space is one where a channel substitution is visible band by band, so
    the test can say WHICH colour each plane came out as rather than only
    that something changed.
    """
    src = tmp_path / "expt"
    (src / "data").mkdir(parents=True)
    array = np.zeros((24, 24, 3), dtype=np.uint8)
    array[:, 0:8, 0] = 200
    array[:, 8:16, 1] = 200
    array[:, 16:24, 2] = 200
    path = src / "data" / "object.png"
    Image.fromarray(array).save(path)
    return str(path)


def _settings(src_png, size):
    from spacr.qt.annotate_engine import AnnotateSettings

    settings = AnnotateSettings()
    settings.src = os.path.dirname(os.path.dirname(src_png))
    settings.db_path = os.path.join(settings.src, "measurements",
                                    "measurements.db")
    settings.image_size = size
    # Percentile normalisation is on by default and would stretch each plane
    # after the substitution. Off, so the numbers below are the mapping and
    # nothing else.
    settings.normalize_channels = []
    return settings


def _drawn(settings, png):
    from spacr.qt.screens import annotate as screen

    image, _ = screen._load_thumb_image_worker(
        {"png_path": png, "annotation": None}, None, settings)
    return np.asarray(image).astype(int)


def _band_colours(pixels):
    """The colour each of the three bands came out as, sampled mid-band."""
    row = pixels.shape[0] // 2
    sixth = pixels.shape[1] // 6
    return [tuple(pixels[row, sixth]),
            tuple(pixels[row, sixth * 3]),
            tuple(pixels[row, sixth * 5])]


def test_choosing_cmy_in_the_dialog_changes_the_pixels(qtbot,
                                                       three_stripe_crop):
    """The mode has to reach the draw, and only pixels can say that it did."""
    from spacr.qt.screens import annotate as screen

    settings = _settings(three_stripe_crop, (48, 48))
    dialog = screen._SettingsDialog(settings)
    qtbot.addWidget(dialog)

    combo = dialog._display_primaries
    index = combo.findData("cmy")
    assert index >= 0, "the CMY mode is not offered in the dialog at all"
    combo.setCurrentIndex(index)
    chosen = dialog.collect()

    plain = _settings(three_stripe_crop, chosen.image_size)
    plain.display_primaries = "rgb"

    as_cmy = _drawn(chosen, three_stripe_crop)
    as_rgb = _drawn(plain, three_stripe_crop)
    assert as_cmy.shape == as_rgb.shape
    difference = int(np.abs(as_cmy - as_rgb).max())
    assert difference > 32, (
        "a crop drawn in CMY is pixel-for-pixel the RGB one "
        f"(largest difference {difference}); the mode is not reaching the "
        "draw")


def test_cmy_removes_a_plane_s_complement_rather_than_adding_itself(
        qtbot, three_stripe_crop):
    """Subtractive, not RGB relabelled.

    A plane at full strength in CMY paints the two slots that are NOT its
    own: the red band comes out cyan, the green magenta, the blue yellow.
    Relabelling the modes would leave each band lit in a single slot, which
    is what an RGB draw looks like and what the complaint was.
    """
    from spacr.qt.screens import annotate as screen

    settings = _settings(three_stripe_crop, (48, 48))
    dialog = screen._SettingsDialog(settings)
    qtbot.addWidget(dialog)
    dialog._display_primaries.setCurrentIndex(
        dialog._display_primaries.findData("cmy"))
    chosen = dialog.collect()

    red_band, green_band, blue_band = _band_colours(
        _drawn(chosen, three_stripe_crop))

    assert red_band[0] == 0 and red_band[1] > 0 and red_band[2] > 0, (
        f"the red plane drew as {red_band}, not as cyan")
    assert green_band[1] == 0 and green_band[0] > 0 and green_band[2] > 0, (
        f"the green plane drew as {green_band}, not as magenta")
    assert blue_band[2] == 0 and blue_band[0] > 0 and blue_band[1] > 0, (
        f"the blue plane drew as {blue_band}, not as yellow")


def test_the_display_order_control_is_collected_too(qtbot, three_stripe_crop):
    """The row above CMY had the same break and is fixed by the same line.

    ``Display order`` says which source plane fills each colour slot. It was
    never read back either, so ``B G R`` -- the setting that exists to open a
    project in the colours it was authored for -- did nothing at all.
    """
    from spacr.qt.screens import annotate as screen

    settings = _settings(three_stripe_crop, (48, 48))
    dialog = screen._SettingsDialog(settings)
    qtbot.addWidget(dialog)
    index = dialog._display_order.findData("bgr")
    assert index >= 0
    dialog._display_order.setCurrentIndex(index)
    chosen = dialog.collect()

    plain = _settings(three_stripe_crop, chosen.image_size)
    swapped = _band_colours(_drawn(chosen, three_stripe_crop))
    straight = _band_colours(_drawn(plain, three_stripe_crop))
    assert swapped != straight, (
        "B G R drew the same pixels as R G B; the order never reached the "
        "loader")
    # The first band is the red plane. Read in B G R order it fills the blue
    # slot instead.
    assert straight[0][0] == 200 and straight[0][2] == 0
    assert swapped[0][2] == 200 and swapped[0][0] == 0

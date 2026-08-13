"""Colourblind mode. Instruction 89.

    "we should have a colorblind mode that applies to images being shown,
     where RGB images are visualized as CMYK instead"

CMY WAS IMPLEMENTED FIRST AND MEASURED WORSE THAN DOING NOTHING, so this
ships a different mapping and the tests carry the measurement. Cyan and
magenta separate along the red-green axis, which is the axis the deficiency
removes -- simulated, a red and a green stain drawn as cyan and magenta were
10.6 apart against 21.2 for plain RGB.

What ships instead changes ONE channel: red is drawn as yellow, green and
blue stay where a user expects them. Simulated separation of the worst
channel pair, 0-255 scale:

    red / green / blue        283 normal,  21 deuteranope,  60 protanope
    green / blue / yellow     200 normal, 146 deuteranope, 159 protanope

21 is not a small number, it is invisible.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr.crops import (
    DISPLAY_PRIMARIES,
    CropError,
    apply_display_primaries,
)

#: Brettel-style simulations, the same ones used to choose the mapping.
DEUTERANOPE = np.array([[0.625, 0.700, 0.000],
                        [0.375, 0.300, 0.000],
                        [0.000, 0.000, 1.000]], np.float32)
PROTANOPE = np.array([[0.567, 0.433, 0.000],
                      [0.558, 0.442, 0.000],
                      [0.000, 0.242, 0.758]], np.float32)


def _stain(channel, value=200):
    a = np.zeros((2, 2, 3), np.uint8)
    a[..., channel] = value
    return a


def _separation(a, b, simulation, primaries):
    left = simulation @ apply_display_primaries(a, primaries)[0, 0].astype(np.float32)
    right = simulation @ apply_display_primaries(b, primaries)[0, 0].astype(np.float32)
    return float(np.linalg.norm(left - right))


# ---------------------------------------------------------------------------
# The claim, measured
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("simulation,name", [(DEUTERANOPE, "deuteranope"),
                                             (PROTANOPE, "protanope")])
def test_red_and_green_become_separable(simulation, name):
    """The whole point. Without this they are one colour."""
    red, green = _stain(0), _stain(1)
    before = _separation(red, green, simulation, "rgb")
    after = _separation(red, green, simulation, "deuteranope")
    assert after > before * 2, (
        f"{name}: {before:.1f} -> {after:.1f}; the mode must make the two "
        f"stains MORE separable, not less")


def test_the_red_green_pair_is_effectively_invisible_by_default():
    """Documents why the mode is needed at all, so a future change that
    'improves' the default can see what it has to beat."""
    assert _separation(_stain(0), _stain(1), DEUTERANOPE, "rgb") < 30


def test_every_channel_pair_stays_separable_under_both_deficiencies():
    pairs = [(0, 1), (0, 2), (1, 2)]
    for simulation in (DEUTERANOPE, PROTANOPE):
        for a, b in pairs:
            assert _separation(_stain(a), _stain(b), simulation,
                               "deuteranope") > 60


# ---------------------------------------------------------------------------
# It is a display transform and nothing else
# ---------------------------------------------------------------------------

def test_rgb_is_the_default_and_returns_the_same_object():
    image = _stain(0)
    assert apply_display_primaries(image) is image


def test_red_is_drawn_as_yellow():
    assert apply_display_primaries(_stain(0), "deuteranope")[0, 0].tolist() \
        == [200, 200, 0]


def test_green_and_blue_are_left_where_the_user_expects_them():
    assert apply_display_primaries(_stain(1), "deuteranope")[0, 0].tolist() \
        == [0, 200, 0]
    assert apply_display_primaries(_stain(2), "deuteranope")[0, 0].tolist() \
        == [0, 0, 200]


def test_the_dtype_survives():
    for dtype in (np.uint8, np.uint16, np.float32):
        image = (np.zeros((2, 2, 3), dtype))
        image[..., 0] = 100
        assert apply_display_primaries(image, "deuteranope").dtype == dtype


def test_an_overlap_saturates_rather_than_dimming_the_image():
    """Only plane 0 lands in two slots, so a genuine red+green coincidence is
    what saturates -- and that is worth seeing as bright."""
    both = np.zeros((2, 2, 3), np.uint8)
    both[..., 0] = 200
    both[..., 1] = 200
    out = apply_display_primaries(both, "deuteranope")
    assert out[0, 0, 1] == 255


def test_an_unknown_primary_set_is_refused_by_name():
    with pytest.raises(CropError, match="deuteranope"):
        apply_display_primaries(_stain(0), "cmyk")


def test_a_greyscale_image_is_left_alone():
    grey = np.zeros((2, 2), np.uint8)
    assert apply_display_primaries(grey, "deuteranope") is grey


def test_the_modes_separate_the_convention_from_the_accessibility_ones():
    """CMY is how biologists publish; the others are per-deficiency. One
    "colourblind" switch cannot be right for all three deficiencies, because
    which PAIR collapses is what decides the substitution."""
    assert DISPLAY_PRIMARIES == (
        "rgb", "cmy", "deuteranope", "protanope", "tritanope")


# ---------------------------------------------------------------------------
# Through the loader
# ---------------------------------------------------------------------------

def test_the_loader_defaults_to_normal_colour(tmp_path):
    from PIL import Image

    from spacr.qt.annotate_engine import load_crop_image

    path = str(tmp_path / "c.png")
    Image.fromarray(_stain(0)).save(path)
    assert np.asarray(load_crop_image(path))[0, 0].tolist() == [200, 0, 0]


def test_the_loader_applies_the_mode(tmp_path):
    from PIL import Image

    from spacr.qt.annotate_engine import load_crop_image

    path = str(tmp_path / "c.png")
    Image.fromarray(_stain(0)).save(path)
    out = np.asarray(load_crop_image(path, display_primaries="deuteranope"))
    assert out[0, 0].tolist() == [200, 200, 0]


def test_order_is_applied_before_primaries():
    """The order says which plane fills a slot; the primaries say what colour
    the slot is drawn in. Recolouring first would paint planes about to move."""
    import inspect

    from spacr.qt import annotate_engine

    source = inspect.getsource(annotate_engine.load_crop_image)
    assert source.index("apply_display_order(") < source.index(
        "apply_display_primaries(")


def test_it_changes_nothing_on_disk():
    """A crop saved in colourblind colours would silently become the training
    set, so the transform must exist only on the display path."""
    import inspect

    from spacr import crops

    source = inspect.getsource(crops.apply_display_primaries)
    for forbidden in ("imwrite", "save(", "to_sql", "open("):
        assert forbidden not in source


# ---------------------------------------------------------------------------
# CMY is the publishing convention, and is NOT sold as accessibility
# ---------------------------------------------------------------------------

def test_cmy_is_offered_because_it_is_the_convention():
    """Cyan/magenta/yellow is how biologists show multichannel micrographs
    now. It is a figure style, and this asserts it exists as its own mode."""
    out = apply_display_primaries(_stain(0), "cmy")[0, 0].tolist()
    assert out == [0, 100, 100], "plane 0 should read as cyan"
    assert apply_display_primaries(_stain(1), "cmy")[0, 0].tolist() == [100, 0, 100]
    assert apply_display_primaries(_stain(2), "cmy")[0, 0].tolist() == [100, 100, 0]


def test_cmy_does_not_help_a_deuteranope_and_the_test_says_so():
    """Recorded as a MEASUREMENT so nobody re-labels cmy an accessibility
    mode: cyan and magenta separate along the axis the deficiency removes."""
    red, green = _stain(0), _stain(1)
    as_rgb = _separation(red, green, DEUTERANOPE, "rgb")
    as_cmy = _separation(red, green, DEUTERANOPE, "cmy")
    assert as_cmy < as_rgb, (
        f"cmy {as_cmy:.1f} vs rgb {as_rgb:.1f} -- if this ever reverses, the "
        f"comment calling cmy a convention rather than an accessibility mode "
        f"needs revisiting")


TRITANOPE = np.array([[0.950, 0.050, 0.000],
                      [0.000, 0.433, 0.567],
                      [0.000, 0.475, 0.525]], np.float32)


def test_the_tritanope_mode_separates_the_pair_that_collapses_for_them():
    """Blue/yellow collapses for a tritanope, not red/green -- which is why
    one mode cannot serve every deficiency."""
    before = _separation(_stain(2), _stain(1), TRITANOPE, "rgb")
    after = _separation(_stain(2), _stain(1), TRITANOPE, "tritanope")
    assert after > before * 2, f"{before:.1f} -> {after:.1f}"


@pytest.mark.parametrize("mode", ["cmy", "deuteranope", "protanope",
                                  "tritanope"])
def test_no_mode_touches_the_data(mode):
    """Every mode is a display transform; none may write anything."""
    import inspect

    from spacr import crops

    source = inspect.getsource(crops.apply_display_primaries)
    for forbidden in ("imwrite", "save(", "to_sql"):
        assert forbidden not in source
    # and it must not mutate its input
    image = _stain(0)
    before = image.copy()
    apply_display_primaries(image, mode)
    assert np.array_equal(image, before)


# ---------------------------------------------------------------------------
# Reachable where it matters: Annotate
# ---------------------------------------------------------------------------

def _dialog(qtbot):
    from spacr.qt.annotate_engine import AnnotateSettings
    from spacr.qt.screens.annotate import _SettingsDialog

    dialog = _SettingsDialog(AnnotateSettings())
    qtbot.addWidget(dialog)
    return dialog


def test_annotate_offers_every_mode(qtbot):
    """"especially in annotate" -- the maintainer's own emphasis."""
    combo = _dialog(qtbot)._display_primaries
    assert [combo.itemData(i) for i in range(combo.count())] == list(
        DISPLAY_PRIMARIES)


def test_annotate_defaults_to_as_acquired(qtbot):
    assert _dialog(qtbot)._display_primaries.currentData() == "rgb"


def test_the_labels_say_which_deficiency_each_mode_is_for(qtbot):
    """A user picking one needs to know which is theirs."""
    combo = _dialog(qtbot)._display_primaries
    labels = [combo.itemText(i).lower() for i in range(combo.count())]
    assert any("red-green" in l for l in labels)
    assert any("blue-yellow" in l for l in labels)
    assert any("publication" in l for l in labels)


def test_the_tooltip_separates_the_convention_from_accessibility(qtbot):
    tip = _dialog(qtbot)._display_primaries.toolTip().lower()
    assert "disk" in tip, "it must say it changes nothing on disk"
    assert "cmy is not one of them" in tip or "not one of them" in tip

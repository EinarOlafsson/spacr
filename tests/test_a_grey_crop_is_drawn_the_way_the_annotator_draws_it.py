"""A single-plane crop drawn through the annotator's own display settings.

A crop cut from one channel arrives two-dimensional, and every function the
annotation application uses to draw one -- normalise, filter channels,
outline -- works on an RGB image. Handing the grey plane straight to PIL
would either raise or produce a mode nothing downstream expects, so the plane
is repeated across three channels first. The result is that the same object
looks the same in the Cells tab and in the annotator, which is the whole
reason these settings are shared rather than reimplemented.

Also here: the tab lookup for a key this screen does not offer. A retired or
foreign setting has no tab, and saying so with an empty string is what lets
a caller decide where to put it instead of guessing at a category.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr.picture_settings import ALL_KEYS, categories, category_of, draw_crop


def _grey(size=8):
    """A single-plane crop with a full-range gradient."""
    return np.linspace(0, 255, size * size, dtype=np.uint8).reshape(size, size)


def test_a_two_dimensional_crop_comes_back_with_three_channels():
    grey = _grey()

    drawn = draw_crop(grey, {"channels": "r"})

    assert drawn.shape == (8, 8, 3)
    assert drawn.dtype == np.uint8


def test_the_grey_plane_is_repeated_rather_than_placed_in_one_channel():
    """Repeating is what makes 'show me green' show the crop, not a blank.

    Placing the plane in channel 0 and leaving the rest at zero would make
    every channel choice but red produce an empty picture.
    """
    grey = _grey()

    red_only = draw_crop(grey, {"channels": "r"})
    green_only = draw_crop(grey, {"channels": "g"})

    assert np.array_equal(red_only[:, :, 0], grey)
    assert red_only[:, :, 1].max() == 0 and red_only[:, :, 2].max() == 0
    assert np.array_equal(green_only[:, :, 1], grey)
    assert green_only[:, :, 0].max() == 0 and green_only[:, :, 2].max() == 0


def test_a_grey_crop_normalises_like_a_colour_one():
    """Normalisation must reach the repeated plane, not skip it."""
    dim = (_grey() // 4).astype(np.uint8)          # nothing above 63

    drawn = draw_crop(dim, {"normalize_channels": "r,g,b",
                            "percentiles": "2,98"})

    assert drawn.shape == (8, 8, 3)
    assert drawn[:, :, 0].max() > dim.max(), "the crop was never stretched"


def test_a_crop_with_nothing_asked_for_is_handed_back_untouched():
    """The control: no draw setting means no conversion and no copy."""
    grey = _grey()

    assert draw_crop(grey, {}) is grey
    assert draw_crop(grey, {"crop_source": "png"}) is grey


# ---------------------------------------------------------------------------
# which tab a setting is shown on
# ---------------------------------------------------------------------------

def test_a_setting_this_screen_does_not_offer_has_no_tab():
    assert category_of("timelapse_mode") == ""
    assert category_of("") == ""
    assert category_of(None) == ""


def test_every_offered_setting_does_have_a_tab():
    """The control: an empty answer must mean 'not offered', not 'unplaced'."""
    for key in ALL_KEYS:
        assert category_of(key) != "", f"{key} is offered but sits on no tab"


def test_a_tab_never_claims_a_key_it_was_not_given():
    titles = {title for title, _keys in categories()}
    assert category_of("outline") in titles

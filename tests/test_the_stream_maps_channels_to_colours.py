"""Which array plane is drawn in which colour, as a setting.

"currently with stream i get the nucleus red so its loading 0,1,2 as
R,B,G. i need to be able to controll this. i also need to be able to
loade 1,2,4 if i have 3 intensity channels and so on."

Two things were wrong with the old arrangement. The mapping was FIXED, so
a merged array whose nucleus sits on plane 0 came out with the nucleus in
whatever colour the default gave plane 0. And the only control was a list
of colour letters, which can say whether red is drawn but not which plane
red is -- so a five-plane array could not be asked for planes 1, 2 and 4.
"""
from __future__ import annotations

import pytest

from spacr.crops import DEFAULT_PNG_CHANNEL_MAPPING, LOAD_IMAGES, STREAM_IMAGES
from spacr.picture_settings import (ALL_KEYS, OWN_DEFAULTS, applies_to,
                                    to_crop_settings, why_not)

COLOURS = ("red_channel", "green_channel", "blue_channel")


def _streaming(**overrides):
    return dict(OWN_DEFAULTS, crop_source=STREAM_IMAGES, **overrides)


def test_each_colour_has_its_own_control():
    for key in COLOURS:
        assert key in ALL_KEYS, f"{key} is not offered"
        assert key in OWN_DEFAULTS


def test_the_default_is_the_mapping_spacr_already_shipped():
    """A panel nobody touches must cut what it always cut."""
    mapping = to_crop_settings(_streaming())["png_channel_mapping"]

    assert mapping == {"r": DEFAULT_PNG_CHANNEL_MAPPING["r"],
                       "g": DEFAULT_PNG_CHANNEL_MAPPING["g"],
                       "b": DEFAULT_PNG_CHANNEL_MAPPING["b"]}


def test_any_plane_can_feed_any_colour():
    """The case named: planes 1, 2 and 4 of a five-plane array."""
    mapping = to_crop_settings(
        _streaming(red_channel=1, green_channel=2, blue_channel=4)
    )["png_channel_mapping"]

    assert mapping == {"r": 1, "g": 2, "b": 4}


def test_the_order_can_be_reversed():
    """The complaint itself: the nucleus was red because the mapping was
    fixed. Putting plane 0 in blue is a one-field change now."""
    mapping = to_crop_settings(
        _streaming(red_channel=2, green_channel=1, blue_channel=0)
    )["png_channel_mapping"]
    flipped = to_crop_settings(
        _streaming(red_channel=0, green_channel=1, blue_channel=2)
    )["png_channel_mapping"]

    assert mapping["r"] != flipped["r"]
    assert flipped == {"r": 0, "g": 1, "b": 2}


def test_a_blank_colour_is_not_drawn():
    """Two-channel pictures are asked for by leaving one empty, and a
    blank must not fall back to the default and put a plane back."""
    mapping = to_crop_settings(
        _streaming(red_channel=3, green_channel=0, blue_channel="")
    )["png_channel_mapping"]

    assert mapping == {"r": 3, "g": 0, "b": None}


def test_nonsense_in_a_field_leaves_that_colour_out():
    """Rather than raising inside the montage worker, where what the user
    would see is "the montage load failed" with no mention of a setting."""
    mapping = to_crop_settings(
        _streaming(red_channel="two", green_channel=1, blue_channel=0)
    )["png_channel_mapping"]

    assert mapping["r"] is None
    assert mapping["g"] == 1


def test_the_controls_mean_nothing_for_crops_already_on_disk():
    """Greyed with a reason, never hidden."""
    for key in COLOURS:
        assert not applies_to(key, LOAD_IMAGES)
        assert why_not(key, LOAD_IMAGES).strip()
        assert applies_to(key, STREAM_IMAGES)

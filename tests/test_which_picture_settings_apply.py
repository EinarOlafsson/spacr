"""What a mode does not use is greyed, with the reason, and never hidden.

Instruction 170: "settings that do not apply for the chosen method are grayed
out". The keys are the annotator's own -- a Cells tab with its own vocabulary
for the same picture would be two panels that disagree about what "normalize"
means, which is what 145 exists to stop.
"""
import pytest

from spacr.crops import LOAD_IMAGES, STREAM_IMAGES
from spacr.picture_settings import (ALL_KEYS, BOTH_MODES, applies_to,
                                    bounding_box_only, greyed_in, modes,
                                    why_not)


def test_load_images_is_offered_first():
    assert modes()[0][0] == LOAD_IMAGES
    assert modes()[0][1] == "load images"
    assert modes()[1][1] == "stream images"


@pytest.mark.parametrize("key", BOTH_MODES)
@pytest.mark.parametrize("mode", [LOAD_IMAGES, STREAM_IMAGES])
def test_the_shared_settings_apply_to_both(key, mode):
    """They shape the picture AFTER it is obtained, so the route is
    irrelevant."""
    assert applies_to(key, mode)
    assert why_not(key, mode) == ""


def test_the_disk_only_setting_is_greyed_when_streaming():
    assert not applies_to("image_type", STREAM_IMAGES)
    assert applies_to("image_type", LOAD_IMAGES)
    assert "stream images" in why_not("image_type", STREAM_IMAGES)


@pytest.mark.parametrize("key", ["object_array", "coordinate_columns",
                                 "crop_shape"])
def test_the_cut_settings_are_greyed_when_loading(key):
    assert not applies_to(key, LOAD_IMAGES)
    assert applies_to(key, STREAM_IMAGES)
    assert "load images" in why_not(key, LOAD_IMAGES)


def test_every_reason_says_WHY_not_merely_that_it_does_not_apply():
    """A greyed control that says only 'not used' teaches nothing."""
    for mode in (LOAD_IMAGES, STREAM_IMAGES):
        for key in greyed_in(mode):
            reason = why_not(key, mode)
            assert ": it " in reason, f"{key} in {mode} gives no reason"


def test_the_two_modes_grey_different_things():
    assert greyed_in(LOAD_IMAGES) != greyed_in(STREAM_IMAGES)
    assert not set(greyed_in(LOAD_IMAGES)) & set(greyed_in(STREAM_IMAGES))


def test_an_unknown_setting_is_left_alone():
    """Not this module's job to grey a control it has never heard of, and a
    panel that hid the unknown would hide new settings by default."""
    assert applies_to("some_new_knob", LOAD_IMAGES)
    assert why_not("some_new_knob", STREAM_IMAGES) == ""


def test_a_coordinate_cut_declares_itself_a_bounding_box():
    """"this could only do bounding box" -- said BEFORE the cut is made."""
    assert bounding_box_only({"crop_source": STREAM_IMAGES,
                              "coordinate_columns": ["x", "y"]})
    assert not bounding_box_only({"crop_source": STREAM_IMAGES,
                                  "coordinate_columns": []})
    assert not bounding_box_only({"crop_source": LOAD_IMAGES,
                                  "coordinate_columns": ["x", "y"]})


def test_every_key_the_panel_shows_is_covered():
    for key in ALL_KEYS:
        assert applies_to(key, LOAD_IMAGES) or applies_to(key, STREAM_IMAGES)

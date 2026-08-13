"""What a crop holds, and what the model sees. Instruction 91.

    "i would like the user to be able to choose to keep the origional dtpe or
     scale the images to either 1-255 (uint8) or between 0 and 1 ... i tried
     to keep uint16 throughout whenever possible to not loose any information.
     was this a good call?"

It was, and the code already did it. What was missing is the setting AFTER
the dtype: ToTensor divides by 255 and hands the model a float in [0,1]
whatever the file held, so the dtype is a storage choice -- while the mean and
std applied next were a literal in three places, and spaCR's 0.5/0.5 is not
what any ImageNet-pretrained model was fitted on.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr.normalization import (
    CROP_DTYPES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NORMALIZATIONS,
    apply_crop_dtype,
    describe_normalization,
    normalization_stats,
)


# ---------------------------------------------------------------------------
# The dtype on disk
# ---------------------------------------------------------------------------

def test_original_is_the_default_and_changes_nothing():
    array = np.array([[0, 300, 65535]], np.uint16)
    out = apply_crop_dtype(array)
    assert out.dtype == np.uint16
    assert np.array_equal(out, array)


def test_narrowing_goes_through_the_one_rule():
    """A second rescale here would be a second answer to what a 16-bit value
    means as an 8-bit one, and the two would disagree on the same crop."""
    from spacr.crops import narrow_to_uint8

    array = np.array([[0, 256, 65535]], np.uint16)
    assert np.array_equal(apply_crop_dtype(array, "uint8"),
                          narrow_to_uint8(array))


def test_widening_is_a_cast_and_not_a_stretch():
    """Multiplying by 257 to 'use the range' would change every measured
    intensity for no information gained."""
    array = np.array([[0, 128, 255]], np.uint8)
    out = apply_crop_dtype(array, "uint16")
    assert out.dtype == np.uint16
    assert out.tolist() == [[0, 128, 255]]


def test_a_dtype_already_correct_is_returned_untouched():
    array = np.array([[1, 2]], np.uint8)
    assert apply_crop_dtype(array, "uint8") is array


def test_a_float_crop_widens_through_the_same_rule():
    """A float has no declared range, so anything else invents a scale."""
    array = np.array([[0.0, 0.5, 1.0]], np.float32)
    out = apply_crop_dtype(array, "uint16")
    assert out.dtype == np.uint16
    assert out.max() <= 255


def test_an_unknown_dtype_keeps_the_original_rather_than_guessing():
    array = np.array([[1, 2]], np.uint16)
    assert apply_crop_dtype(array, "float128").dtype == np.uint16


@pytest.mark.parametrize("name", CROP_DTYPES)
def test_every_offered_dtype_works(name):
    assert apply_crop_dtype(np.array([[1, 2]], np.uint16), name) is not None


# ---------------------------------------------------------------------------
# The statistics, which is the setting that moves a number
# ---------------------------------------------------------------------------

def test_symmetric_is_what_spacr_has_always_done():
    assert normalization_stats("symmetric") == ((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))


def test_imagenet_is_what_a_pretrained_model_was_fitted_on():
    mean, std = normalization_stats("imagenet")
    assert mean == IMAGENET_MEAN and std == IMAGENET_STD


def test_the_two_are_genuinely_different_and_that_is_the_point():
    """A finetune under symmetric hands pretrained weights inputs
    distributed differently from the ones they learned."""
    symmetric, imagenet = (normalization_stats("symmetric"),
                           normalization_stats("imagenet"))
    assert symmetric != imagenet
    # Mid-grey lands in a different place under each.
    grey = 0.5
    assert (grey - symmetric[0][0]) / symmetric[1][0] == pytest.approx(0.0)
    assert abs((grey - imagenet[0][0]) / imagenet[1][0]) > 0.05


def test_none_applies_nothing():
    assert normalization_stats("none") is None


def test_an_unknown_mode_falls_back_to_what_spacr_did_before():
    """A typo must not train a model on statistics nobody chose, and must
    not stop the run either."""
    assert normalization_stats("imagnet") == normalization_stats("symmetric")


@pytest.mark.parametrize("mode", NORMALIZATIONS)
def test_every_offered_mode_describes_itself(mode):
    assert describe_normalization(mode)


def test_the_description_names_imagenet_so_a_model_card_records_it():
    assert "ImageNet" in describe_normalization("imagenet")


# ---------------------------------------------------------------------------
# Nothing changes for an existing run
# ---------------------------------------------------------------------------

def test_the_defaults_reproduce_todays_behaviour_exactly():
    assert normalization_stats(None) == ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    array = np.array([[7, 9]], np.uint16)
    assert np.array_equal(apply_crop_dtype(array, None), array)


def test_both_settings_are_registered_with_a_type_and_a_tooltip():
    from spacr.settings import expected_types, tooltips

    for key in ("crop_dtype", "input_statistics"):
        assert expected_types[key] is str
        assert len(tooltips[key]) > 100


def test_the_tooltip_says_the_dtype_is_not_a_precision_choice():
    """The thing a user would otherwise get wrong: picking uint16 believing
    it makes the model better."""
    from spacr.settings import tooltips

    assert "does NOT change training precision" in tooltips["crop_dtype"]


def test_normalize_input_is_still_the_on_off_it_always_was():
    """The new setting is WHICH statistics; an existing settings file keeps
    its meaning exactly."""
    from spacr.settings import expected_types

    assert expected_types["normalize_input"] is bool

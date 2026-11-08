"""The per-image diameter switch reads a diameter that is not a number.

A diameter typed as text that is neither a number nor ``auto`` cannot be the
size Cellpose is told, so with the per-image switch on it is treated as
unset and the size is estimated in each image.
"""
from __future__ import annotations

from spacr.object_settings_table import _estimates_diameter_per_image


def test_a_diameter_that_is_not_a_number_still_estimates_per_image():
    settings = {"estimate_diameters_per_image": True, "cell_diameter": "abc"}
    assert _estimates_diameter_per_image(settings, "cell") is True


def test_a_fixed_diameter_is_not_estimated_and_the_switch_can_be_text():
    assert _estimates_diameter_per_image(
        {"estimate_diameters_per_image": "yes", "cell_diameter": 30}, "cell") is False
    assert _estimates_diameter_per_image(
        {"estimate_diameters_per_image": "off", "cell_diameter": "abc"}, "cell") is False

"""The wand taper crops only a plane flood, and an empty flood tapers to nothing.

The crop is a speed-up: it must never turn a volume into a plane-shaped
crop, and a flood with no pixels has no window to read, so the taper keeps
the whole field and answers with an empty region rather than failing.
"""
from __future__ import annotations

import numpy as np

from spacr.qt import wand_rescue as W


def test_a_volume_flood_is_never_cropped_as_a_plane():
    flooded = np.zeros((3, 40, 40), dtype=bool)
    flooded[1, 10:15, 10:15] = True
    assert W._taper_window(flooded, sigma=2.0) is None


def test_an_empty_flood_has_no_window():
    assert W._taper_window(np.zeros((40, 40), dtype=bool), sigma=2.0) is None


def test_a_small_flood_is_cropped_to_its_box_plus_the_gradient_reach():
    flooded = np.zeros((60, 60), dtype=bool)
    flooded[20:25, 30:33] = True
    reach = int(4.0 * 1.5 + 0.5) + 2
    assert W._taper_window(flooded, sigma=1.5) == (
        20 - reach, 25 + reach, 30 - reach, 33 + reach)


def test_an_empty_flood_tapers_to_an_empty_region_of_the_field():
    image = np.random.default_rng(3).uniform(0, 100, (32, 32)).astype(np.float32)
    flooded = np.zeros(image.shape, dtype=bool)
    provisional = np.ones(image.shape, dtype=bool)
    result = W.taper_region_to_intensity(image, flooded, provisional, (16, 16))
    assert result.shape == image.shape
    assert result.dtype == bool
    assert not result.any()

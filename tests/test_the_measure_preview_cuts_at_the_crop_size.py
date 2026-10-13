"""The Measure preview cuts at the size a run will write, and asks for it once.

Item 443. The maintainer reported that "changing the crop width or height
dosnt seem to do anything to the live crops", and the reason was not a missing
signal: ``crop_objects_from_array`` took no size at all, so the preview drew
each object's bounding box while a real run resized to ``png_size``. The panel
also asked for width and height separately, with a "lock aspect" box whose job
was to keep them equal -- and ``crop_size`` already maps onto ``png_size``
(picture_settings) where a scalar already means a square crop (crops).
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr.measure import crop_objects_from_array


def _one_object(side: int = 64) -> np.ndarray:
    """A merged array with a single labelled object in its mask slice."""
    data = np.zeros((side, side, 5), dtype=np.uint16)
    data[10:30, 10:34, :4] = 900
    data[10:30, 10:34, 4] = 1
    return data


def test_a_crop_comes_back_at_the_size_that_was_asked_for():
    """The report: the size must reach the crops themselves."""
    data = _one_object()

    small = crop_objects_from_array(data, mask_dim=4, channels=(0, 1, 2),
                                    size=(64, 64))
    large = crop_objects_from_array(data, mask_dim=4, channels=(0, 1, 2),
                                    size=(256, 256))

    assert small[0]["crop"].shape[:2] == (64, 64)
    assert large[0]["crop"].shape[:2] == (256, 256)


def test_without_a_size_it_is_the_bounding_box_as_before():
    """The old behaviour is what ``size=None`` still means."""
    data = _one_object()

    out = crop_objects_from_array(data, mask_dim=4, channels=(0, 1, 2))

    height, width = out[0]["crop"].shape[:2]
    assert (height, width) != (64, 64)
    assert height < 64 and width < 64


def test_it_resizes_the_way_the_run_resizes():
    """The same PIL call a run makes at save time, not a second implementation.

    measure.py saves with ``Image.fromarray(preview).resize(tuple(png_size))``
    and nothing else. If the preview used a different resampler, the picture a
    user tunes against would not be the picture the run writes.
    """
    from PIL import Image

    data = _one_object()
    box = crop_objects_from_array(data, mask_dim=4, channels=(0, 1, 2))[0]["crop"]
    sized = crop_objects_from_array(data, mask_dim=4, channels=(0, 1, 2),
                                    size=(96, 96))[0]["crop"]

    expected = np.asarray(Image.fromarray(box).resize((96, 96)))
    assert np.array_equal(sized, expected)


def test_a_non_rgb_16_bit_crop_survives_the_resize():
    """``to_rgb=False`` keeps the working dtype, and PIL refuses most of those.

    The preview must not raise on a 16-bit multi-channel crop: a preview that
    raises shows nothing at all, which is worse than the bounding boxes it
    used to show.
    """
    data = _one_object()

    out = crop_objects_from_array(data, mask_dim=4, channels=(0, 1),
                                  to_rgb=False, size=(48, 48))

    assert out[0]["crop"].shape == (48, 48, 2)
    assert out[0]["crop"].dtype == np.uint16


def test_a_size_of_zero_is_ignored_rather_than_raising():
    """A spinner passes through values on its way to the one the user means."""
    data = _one_object()

    out = crop_objects_from_array(data, mask_dim=4, channels=(0, 1, 2),
                                  size=(0, 0))

    assert out[0]["crop"].shape[:2] != (0, 0)

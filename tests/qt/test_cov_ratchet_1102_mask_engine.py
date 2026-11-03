"""Mask engine channel fallbacks, native RGB TIFFs and containers without pixel sizes."""
from __future__ import annotations

import pytest
import numpy as np
import tifffile

from spacr.qt import mask_engine as engine

pytestmark = pytest.mark.qt


def test_a_chosen_channel_outside_the_field_falls_back_to_the_default_channel(tmp_path):
    pixels = np.arange(3 * 9 * 11, dtype=np.uint16).reshape(3, 9, 11)
    tifffile.imwrite(tmp_path / 'field.tif', pixels, photometric='minisblack',
                     metadata={'axes': 'CYX'})
    image, mask, planes, colour, channel = engine._load_channel_field(
        str(tmp_path), 'field.tif', channel=7)
    assert channel == 0 and colour is False
    assert len(planes) == 3
    np.testing.assert_array_equal(planes[2], pixels[2])
    assert image.shape == (9, 11) and mask.shape == (9, 11)
    # a valid channel is kept as asked
    assert engine._load_channel_field(str(tmp_path), 'field.tif', channel=2)[4] == 2
    # a negative channel is out of range as well
    assert engine._load_channel_field(str(tmp_path), 'field.tif', channel=-1)[4] == 0


def test_only_a_single_series_tiff_can_be_a_native_rgb_field():
    series = {'axes': 'YXS', 'shape': (9, 11, 3), 'photometric': 'RGB'}
    assert engine._native_rgb_field({'ext': '.tif', 'series': [series]})
    assert not engine._native_rgb_field({'ext': '.png', 'series': [series]})
    assert not engine._native_rgb_field({'ext': '.tif', 'series': [series, series]})
    assert not engine._native_rgb_field({'ext': '.tif'})


def test_a_container_of_another_extension_has_no_declared_pixel_sizes():
    series = [{'name': 'a'}, {'name': 'b'}]
    assert engine._container_pixel_sizes('/nowhere/field.png', '.png', series) == [None, None]

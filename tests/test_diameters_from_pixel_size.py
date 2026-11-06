"""Blank object diameters follow the pixel size when it is known.

The magnification formula puts A549 cells at 120 px and nuclei at 60 px at
20x, about twice the 55 and 30 px the masks measure. With a pixel size the
default is the typical object size in micrometres over the pixel size. A
diameter the user typed is never replaced, and the per-image estimate is
checked with a stub model, so no GPU or network weights are needed.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr import object_settings_table as table


@pytest.mark.parametrize('pixel_size, cell, nucleus, pathogen', [
    (0.325, 55, 31, 20),
    (0.65, 28, 15, 10),
    (0.1625, 111, 62, 40),
    (0.108, 167, 93, 60),
    ('0.325', 55, 31, 20),
])
def test_the_default_is_typical_size_over_pixel_size(pixel_size, cell, nucleus,
                                                      pathogen):
    """18, 10 and 6.5 um over the pixel size, rounded to whole pixels."""
    assert table._diameter_from_pixel_size(pixel_size, 'cell') == cell
    assert table._diameter_from_pixel_size(pixel_size, 'nucleus') == nucleus
    assert table._diameter_from_pixel_size(pixel_size, 'pathogen') == pathogen


@pytest.mark.parametrize('pixel_size', [None, '', 0, -0.3, float('nan'),
                                        float('inf'), 'abc', True])
def test_an_unusable_pixel_size_keeps_the_magnification_default(pixel_size):
    assert table._diameter_from_pixel_size(pixel_size, 'cell') is None
    assert table._magnification_default(20, 'cell', pixel_size) == 120.0
    assert table._magnification_default(20, 'nucleus', pixel_size) == 60.0


def test_the_pixel_size_wins_over_the_magnification_for_a_blank_diameter():
    settings = {'magnification': 20, 'pixel_size_um': 0.325,
                'cell_diameter': None, 'nucleus_diameter': ''}
    assert table._default_diameter_of(settings, 'cell') == 55.0
    assert table._resolved_diameter(None, 20, 'nucleus', 0.325) == 31.0
    assert table._diameter_used_by(settings, 'cell') == (
        '55 px (pixel size 0.325 um default)')
    assert table._diameter_used_by({'magnification': 20}, 'cell') == (
        '120 px (magnification 20x default)')


@pytest.mark.parametrize('raw, expected', [(80, 80.0), ('44', 44.0),
                                           (0, 'auto'), ('auto', 'auto')])
def test_a_typed_diameter_is_left_alone(raw, expected):
    """A number, 0 or auto means the same with or without a pixel size."""
    assert table._resolved_diameter(raw, 20, 'cell', 0.325) == expected
    assert table._resolved_diameter(raw, 20, 'cell') == expected


def test_object_settings_use_the_pixel_size_and_keep_a_typed_diameter():
    from spacr.settings import _get_object_settings

    base = {'magnification': 20, 'pixel_size_um': 0.325, 'verbose': False,
            'merge_pathogens': False, 'cell_model_name': 'cpsam',
            'nucleus_model_name': 'cpsam', 'pathogen_model_name': 'cpsam'}
    cell = _get_object_settings('cell', {**base, 'cell_diameter': None})
    assert cell['diameter'] == 55
    assert cell['minimum_size'] == 55 ** 2 / 4
    assert cell['maximum_size'] == 55 ** 2 * 10

    typed = _get_object_settings('cell', {**base, 'cell_diameter': 90})
    assert typed['diameter'] == 90.0

    without = _get_object_settings('nucleus', {**base, 'pixel_size_um': None})
    assert without['diameter'] == 60


def test_get_diam_takes_the_pixel_size():
    from spacr.utils import _get_diam

    assert _get_diam(20, 'cell') == 120
    assert _get_diam(20, 'cell', pixel_size_um=0.325) == 55
    assert _get_diam(20, 'cell_large', pixel_size_um=0.325) == 160


def test_held_diameters_freeze_at_the_pixel_size_default():
    held = table._blank_diameters_held_still(
        {'calculate_diameters': False, 'magnification': 20,
         'pixel_size_um': 0.65, 'cell_diameter': None,
         'nucleus_diameter': 40})
    assert held == {'cell_diameter': 28}


class _StubModel:
    """Answers ``eval`` with one disc per image, sized by the diameter.

    At native scale (``diameter=None``) the disc has the image's own true
    diameter; asked for a diameter, it records it.
    """

    def __init__(self, true_diameters):
        self.true = list(true_diameters)
        self.calls = []

    def eval(self, x, diameter=None, **kwargs):
        image = x[0]
        self.calls.append(diameter)
        index = sum(1 for call in self.calls if call is None) - 1
        size = self.true[index] if diameter is None else 10
        mask = np.zeros(image.shape[:2], dtype=np.int32)
        if size:
            yy, xx = np.ogrid[:mask.shape[0], :mask.shape[1]]
            centre = mask.shape[0] // 2
            mask[(yy - centre) ** 2 + (xx - centre) ** 2 <= (size / 2) ** 2] = 1
        flows = [np.zeros_like(mask), np.zeros((2,) + mask.shape),
                 np.zeros(mask.shape)]
        return [mask], [flows], None


def test_each_image_is_segmented_at_its_own_measured_diameter():
    from spacr.object import _per_image_diameter_masks

    images = [np.zeros((128, 128, 1)) for _ in range(3)]
    model = _StubModel([40, 60, 0])
    masks, flows, used = _per_image_diameter_masks(
        model, images, {'normalize': False, 'batch_size': 8}, fallback=55.0)

    assert len(masks) == len(flows) == 3
    assert used[0] == pytest.approx(40, rel=0.05)
    assert used[1] == pytest.approx(60, rel=0.05)
    assert used[2] == 55.0
    assert model.calls[0::2] == [None, None, None]
    assert model.calls[1::2] == used


@pytest.mark.parametrize('diameter, switch, expected', [
    (None, True, True), (0, True, True), ('auto', True, True),
    (55, True, False), (None, False, False), (None, 'false', False),
    ('', 'true', True),
])
def test_only_blank_or_auto_diameters_are_estimated(diameter, switch, expected):
    from spacr.object import _estimates_per_image

    settings = {'cell_diameter': diameter,
                'estimate_diameters_per_image': switch}
    assert _estimates_per_image(settings, 'cell') is expected


def test_cellpose3_uses_its_size_model_when_estimating():
    from spacr.object import _cellpose3_eval_settings

    blank = {'cell_diameter': None, 'estimate_diameters_per_image': True}
    keywords, _ = _cellpose3_eval_settings(blank, 'cell', 55)
    assert keywords['diameter'] == 0.0

    typed = {'cell_diameter': 70, 'estimate_diameters_per_image': True}
    keywords, _ = _cellpose3_eval_settings(typed, 'cell', 55)
    assert keywords['diameter'] == 70.0

    off = {'cell_diameter': None}
    keywords, _ = _cellpose3_eval_settings(off, 'cell', 55)
    assert keywords['diameter'] == 55.0


def test_the_v2_backend_default_follows_the_pixel_size():
    from spacr.pipeline_v2 import _backend_mask_settings

    merged = _backend_mask_settings(
        {'magnification': 20, 'pixel_size_um': 0.325}, 'nucleus',
        diameter=None, flow_threshold=0.4, cellprob_threshold=0.0)
    assert merged['_default_diameter'] == 31.0
    merged = _backend_mask_settings(
        {'magnification': 20}, 'nucleus',
        diameter=None, flow_threshold=0.4, cellprob_threshold=0.0)
    assert merged['_default_diameter'] == 60

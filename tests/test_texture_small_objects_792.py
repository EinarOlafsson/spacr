"""Item 792: GLCM homogeneity is finite for objects smaller than the offset.

``homogeneity_distance_32`` was NaN for every nucleus narrower than 32
pixels, which left BBBC021 well means non-finite. The offset is now capped at
the bounding-box width minus one; objects wide enough are unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest
from skimage.exposure import rescale_intensity
from skimage.feature import graycomatrix, graycoprops

from spacr import measure as M


def _field():
    """A large disc, a small disc, a thin bar and a one-pixel-wide line."""
    labels = np.zeros((80, 120), np.int32)
    yy, xx = np.mgrid[:80, :120]
    labels[(yy - 40) ** 2 + (xx - 30) ** 2 <= 22 ** 2] = 1      # 45 px wide
    labels[(yy - 20) ** 2 + (xx - 90) ** 2 <= 5 ** 2] = 2       # 11 px wide
    labels[55:70, 80:83] = 3                                    # 3 px wide
    labels[60:75, 110] = 4                                      # 1 px wide
    image = np.random.default_rng(0).gamma(2.0, 300.0, labels.shape)
    return labels, image.astype(np.uint16)


def _reference(labels, image, label, d):
    """The pre-fix computation at offset ``d``, for one object."""
    region = labels == label
    rows, cols = np.nonzero(region)
    box = (slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1))
    patch = (region[box] * image[box]).astype(int)
    patch = rescale_intensity(patch, out_range=(0, 255)).astype('uint8')
    glcm = graycomatrix(patch, [d], [0], symmetric=True, normed=True)
    return float(graycoprops(glcm, 'homogeneity')[0, 0])


def test_every_object_gets_a_finite_value_at_every_offset():
    labels, image = _field()
    table = M._calculate_homogeneity(labels, image, [8, 16, 32, 64])
    assert np.isfinite(table.values).all()


def test_an_object_wide_enough_for_every_offset_is_unchanged():
    labels, image = _field()
    table = M._calculate_homogeneity(labels, image, [8, 16, 32])
    for column, d in zip(table.columns, (8, 16, 32)):
        assert table.loc[0, column] == _reference(labels, image, 1, d)


def test_a_small_object_is_read_at_its_widest_horizontal_offset():
    labels, image = _field()
    table = M._calculate_homogeneity(labels, image, [8, 32])
    # The 11-px disc keeps d=8 and reads d=32 at offset 10.
    assert table.loc[1, 'homogeneity_distance_8'] == _reference(labels, image, 2, 8)
    assert table.loc[1, 'homogeneity_distance_32'] == _reference(labels, image, 2, 10)
    assert table.loc[2, 'homogeneity_distance_32'] == _reference(labels, image, 3, 2)
    # A one-pixel-wide object has no horizontal pair: homogeneity 1 by rule.
    assert table.loc[3, 'homogeneity_distance_8'] == 1.0


def test_the_offset_rule():
    assert M._homogeneity_offset(32, 100) == 32
    assert M._homogeneity_offset(32, 33) == 32
    assert M._homogeneity_offset(32, 32) == 31
    assert M._homogeneity_offset(32, 1) == 0


def test_the_vectorised_kernel_caps_the_same_way():
    torch = pytest.importorskip("torch")
    labels, image = _field()
    distances = [2, 8, 32, 200]
    cpu = M._calculate_homogeneity(labels, image, distances)
    gpu = M._torch_homogeneity(labels, image, distances, torch.device("cpu"))
    assert np.isfinite(gpu.values).all()
    np.testing.assert_allclose(gpu.values, cpu.values, rtol=0, atol=1e-12)

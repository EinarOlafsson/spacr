"""Native contribution bounds preserve opaque independent RGB additions."""

import math

import numpy as np
import pytest
from PySide6.QtGui import QColor, QPainterPath

from spacr.qt.widgets import ambient


@pytest.mark.parametrize('rgb', [(1., 1., 1.), (1., .25, .75), (.1, .9, .5)])
@pytest.mark.parametrize('alpha', [.01, .5, .95])
def test_rgb_headroom_edges_keep_channels_independent_and_reject_overflow(rgb, alpha, qapp):
    engine = ambient.make_engine('data_art_fungal_growth', 'spacr', '#101418', seed=42)
    color = QColor.fromRgbF(*rgb, alpha)
    highs = np.array([math.ceil(c * color.alphaF() * 255.) + 1
                      for c in (color.blueF(), color.greenF(), color.redF())], dtype=np.uint32)
    generator = np.random.default_rng(42)
    source_channels = generator.integers(0, highs + 1, size=(4096, 3), dtype=np.uint32)
    destination_channels = generator.integers(0, 256 - highs, size=(4096, 3), dtype=np.uint32)
    source_channels[0] = highs
    destination_channels[0] = 255 - highs
    source = np.sum(source_channels << np.array([0, 8, 16], np.uint32), axis=1,
                    dtype=np.uint32)
    original_source = source.copy()
    target = np.sum(destination_channels << np.array([0, 8, 16], np.uint32), axis=1,
                    dtype=np.uint32) | np.uint32(0xff000000)
    expected_channels = destination_channels + source_channels
    assert np.all(expected_channels <= 255)
    path = QPainterPath()
    path.moveTo(0, 0)
    path.lineTo(10, 10)
    key = ('contribution',)
    indices = np.arange(4096, dtype=np.int32)
    engine._fungal_rasters[key] = (path, indices, source)
    assert engine._reuse_fungal_raster(key, path, color, target)
    assert np.array_equal((target[:, None] >> np.array([0, 8, 16], np.uint32)) & 255,
                          expected_channels)
    assert np.all(target >> 24 == 255)
    assert np.array_equal(source, original_source)
    assert source.base is None and indices.base is None
    target[0] = np.uint32(0xff000000 | int(256 - highs[0]))
    before = target.copy()
    assert not engine._reuse_fungal_raster(key, path, color, target)
    assert np.array_equal(target, before)


@pytest.mark.parametrize('shape', ['crossing', 'loop', 'repeated'])
def test_real_qt_sparse_contributions_obey_color_headroom(shape, qapp):
    engine = ambient.make_engine('data_art_fungal_growth', 'spacr', '#101418', seed=42)
    path = QPainterPath()
    path.moveTo(12.25, 12.5)
    path.lineTo(120.75, 85.5)
    if shape == 'crossing':
        path.moveTo(12.25, 85.5)
        path.lineTo(120.75, 12.5)
    elif shape == 'loop':
        path.cubicTo(8., 8., 145., 8., 12.25, 12.5)
    else:
        path.moveTo(12.25, 12.5)
        path.lineTo(120.75, 85.5)
    color = QColor.fromRgbF(.9, .4, .8, .35)
    assert engine._warm_fungal_raster(('native',), path, color, 2.75, 160, 100)
    source = engine._fungal_rasters[('native',)][2]
    assert source.size > 0 and np.all(source >> 24 == 0)
    for shift, channel in zip((0, 8, 16), (color.blueF(), color.greenF(), color.redF())):
        assert np.all(((source >> shift) & 255)
                      <= math.ceil(channel * color.alphaF() * 255.) + 1)

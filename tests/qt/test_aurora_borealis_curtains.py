"""Owned, moving auroral rays respect the palette and display controls."""

import numpy as np
import pytest

from spacr.qt.widgets import ambient


def _pixels(image):
    return np.asarray(image.constBits()).reshape(image.height(), image.bytesPerLine())[:, :image.width() * 4].reshape(image.height(), image.width(), 4).copy()


@pytest.mark.parametrize("palette", ["spacr", "borealis", "random", "mono"])
def test_curtains_move_without_mutating_previous_frames(qapp, palette):
    engine = ambient.make_engine("aurora", palette, "#101010", seed=17)
    first = engine.shade(800, 600)
    original = _pixels(first)
    engine.set_time(4.5)
    later = _pixels(engine.shade(800, 600))
    assert not np.array_equal(original, later)
    assert np.array_equal(_pixels(first), original)
    assert np.mean(later[:, :, :3]) < 65
    if palette == "mono":
        assert np.max(np.abs(later[:, :, 0].astype(int) - later[:, :, 1])) < 2


def test_spacr_aurora_has_green_rays_and_pink_upper_light(qapp):
    engine = ambient.make_engine("aurora", "spacr", "#101010", seed=17, density=1)
    pixels = _pixels(engine.shade(1000, 750)).astype(int)
    blue, green, red = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2]
    assert np.count_nonzero((green > red + 8) & (green > blue + 5)) > 5000
    assert np.count_nonzero((red > green + 2) & (blue > green + 2)) > 100


def test_density_and_detail_keep_native_sampling_and_owned_frames(qapp):
    engine = ambient.make_engine("aurora", "borealis", "#101010", seed=17, density=0.01)
    assert engine.buffer_size(1280, 720) == (1280, 720)
    sparse = _pixels(engine.shade(1280, 720))
    engine.set_density(1)
    dense = _pixels(engine.shade(1280, 720))
    assert np.mean(dense[:, :, :3]) > np.mean(sparse[:, :, :3])
    engine.set_resolution(0.5)
    assert engine.buffer_size(1280, 720) == (640, 360)
    for second in range(60):
        engine.set_time(second * 3)
        engine.shade(300 + second * 2, 240)
    assert len(engine._ray_material) <= 24

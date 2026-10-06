"""Gravity wakes retain native grains while their stationary fields are reused."""

import hashlib
import math

import numpy as np

import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets import ambient


def _hash(image):
    return hashlib.sha256(image.bits().tobytes()).hexdigest()


def test_stationary_wake_fields_survive_ticks_and_expire_without_cache_growth():
    engine = ambient.make_engine("data_art_impulse_lens", "spacr", "#101418",
                                 seed=42, resolution=2, blur=0)
    engine.set_max_pixels(640 * 360)
    for index in range(37):
        engine.set_time(index / 10)
        engine._add_impulse((index / 40, 0.5))
        engine.shade(640, 360)
    assert len(engine._gravity_impulses) == 24
    lattice = next(iter(engine._material_cache.values()))
    fields = lattice[2]
    assert len(fields) == 24
    origin = engine._gravity_impulses[-1][1]
    retained = fields[origin]
    baseline = _hash(engine.shade(640, 360))
    engine.set_time(3.7)
    assert _hash(engine.shade(640, 360)) != baseline
    assert fields[origin] is retained
    assert len(fields) == 24
    engine.set_time(8.7)
    engine.shade(640, 360)
    assert fields == {}
    engine.set_time(3.6)
    assert _hash(engine.shade(640, 360)) == baseline
    assert len(fields) == 24
    engine.set_size(1.5)
    assert engine._material_cache == {}


def test_duplicate_origin_wakes_share_geometry_without_merging_bursts():
    engine = ambient.make_engine("data_art_impulse_lens", "spacr", "#101418",
                                 seed=42, resolution=2, blur=0)
    engine.set_time(1)
    engine._add_impulse((0.5, 0.5))
    one_burst = _hash(engine.shade(640, 360))
    engine._add_impulse((0.5, 0.5))
    assert len(engine._gravity_impulses) == 2
    assert _hash(engine.shade(640, 360)) != one_burst
    assert len(next(iter(engine._material_cache.values()))[2]) == 1
    engine.set_time(0)
    engine.shade(640, 360)
    assert next(iter(engine._material_cache.values()))[2] == {}


@pytest.mark.parametrize("background", ("#101418", "#f6f7f9"))
@pytest.mark.parametrize("size,density", ((1.0, 1.0), (0.5, 3.0)))
def test_normal_wave_packets_preserve_native_coordinates_intensity_and_raster(
        background, size, density, monkeypatch):
    captures = []
    frames = []
    for unbounded in (False, True):
        engine = ambient.make_engine("data_art_impulse_lens", "spacr", background,
                                     seed=42, resolution=2, blur=0,
                                     size=size, density=density)
        engine.set_max_pixels(3840 * 2160)
        engine.set_time(8.5)
        engine.set_pointer((0.0, 1.0))
        engine._gravity_impulses = [
            (8.5 - index * 0.2,
             (0.5 + 0.48 * math.sin(index), 0.5 + 0.48 * math.cos(index)), 2.0)
            for index in range(24)]
        if unbounded:
            monkeypatch.setattr(engine, "_lens_wave_packet",
                                lambda front: np.exp(-(front / 0.075) ** 2))
        stamp = engine._point_material

        def capture(width, height, x, y, light, spread=False):
            captures.append(tuple(np.array(value, copy=True) for value in (x, y, light)))
            return stamp(width, height, x, y, light, spread=spread)

        monkeypatch.setattr(engine, "_point_material", capture)
        frames.append(_hash(engine.shade(3840, 2160)))
    assert frames[0] == frames[1]
    for bounded, original in zip(captures[0], captures[1]):
        assert np.max(np.abs(bounded - original)) < 1e-7
    displacement_bound = 24 * 0.045 * 2 * math.exp(-60) / 0.055
    assert 3840 * 2 * displacement_bound < 1e-20
    assert 24 * 0.4 * 2 * math.exp(-60) < 2e-25

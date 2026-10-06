"""Gravity wakes retain native grains while their stationary fields are reused."""

import hashlib

import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets import ambient


def _hash(image):
    return hashlib.sha256(image.bits().tobytes()).hexdigest()


def test_stationary_wake_fields_survive_ticks_and_expire_without_cache_growth():
    engine = ambient.make_engine("data_art_impulse_lens", "spacr", "#101418",
                                 seed=42, resolution=2, blur=0)
    engine.set_max_pixels(640 * 360)
    engine.set_gravity_radius(0.5)
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
    engine.set_gravity_radius(0.5)
    engine._add_impulse((0.5, 0.5))
    one_burst = _hash(engine.shade(640, 360))
    engine._add_impulse((0.5, 0.5))
    assert len(engine._gravity_impulses) == 2
    assert _hash(engine.shade(640, 360)) != one_burst
    assert len(next(iter(engine._material_cache.values()))[2]) == 1
    engine.set_time(0)
    engine.shade(640, 360)
    assert next(iter(engine._material_cache.values()))[2] == {}

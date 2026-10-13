"""A disabled pointer is exact; enabled gravity has bounded physical reach."""

import numpy as np
import pytest
from PySide6.QtGui import QImage, QPainter

from spacr.qt.widgets import ambient


def _engine(family):
    if family == 'chromatin_ribbon':
        return ambient._DataArtEngine(
            ambient.PALETTE_SETS['spacr'].colors, '#101418',
            family=family, seed=42, resolution=2, blur=0)
    return ambient.make_engine('data_art_' + family, 'spacr', '#101418',
                               seed=42, resolution=2, blur=0)


def _frame(engine, width=480, height=270):
    engine.set_max_pixels(width * height)
    image = engine.shade(width, height)
    return image.bits().tobytes()


def _coordinates(engine, monkeypatch, width, height):
    frames = []

    def capture(w, h, x, y, light, spread=False):
        frames.append(tuple(np.asarray(value).copy() for value in (x, y, light)))
        image = QImage(w, h, QImage.Format_RGB32)
        image.fill(engine.identity)
        return image

    monkeypatch.setattr(engine, '_point_material', capture)
    image = QImage(width, height, QImage.Format_RGB32)
    image.fill(engine.identity)
    painter = QPainter(image)
    engine._paint_field(painter, width, height)
    painter.end()
    return frames[-1]


@pytest.mark.parametrize('family', ambient._DataArtEngine._families)
def test_zero_radius_ignores_mouse_and_clicks_without_stopping_autonomous_motion(family):
    engine = _engine(family)
    assert engine.gravity_radius == 0
    engine.set_time(8)
    baseline = _frame(engine)
    for pointer in ((0, 0), (1, 1), (.2, .7), None):
        engine.set_pointer(pointer)
        engine._add_impulse((.4, .6))
        assert _frame(engine) == baseline
    assert engine._gravity_impulses == []
    engine.set_time(10)
    if family == 'tissue_facets':
        assert _frame(engine) == baseline
    else:
        assert _frame(engine) != baseline


@pytest.mark.parametrize('value,expected', [(-1, 0), (2, 1), (.3, .3),
                                           (float('nan'), 0), (float('inf'), 0),
                                           (-float('inf'), 0)])
def test_engine_radius_is_finite_and_normalized(value, expected):
    engine = _engine('impulse_lens')
    engine.set_gravity_radius(value)
    assert engine.gravity_radius == expected


@pytest.mark.parametrize('width,height', [(640, 360), (360, 640)])
@pytest.mark.parametrize('family', ['point_atlas', 'genetic_advection'])
def test_pointer_only_changes_coordinates_inside_shorter_edge_radius(
        family, width, height, monkeypatch):
    engine = _engine(family)
    engine.set_time(8)
    engine.set_gravity_radius(.23)
    idle = _coordinates(engine, monkeypatch, width, height)
    engine.set_pointer((.5, .5))
    bent = _coordinates(engine, monkeypatch, width, height)
    distance = np.hypot(idle[0] - width / 2, idle[1] - height / 2)
    outside = distance >= .23 * min(width, height) + 2
    assert np.array_equal(idle[0][outside], bent[0][outside])
    assert np.array_equal(idle[1][outside], bent[1][outside])
    assert np.count_nonzero((idle[0] != bent[0]) | (idle[1] != bent[1])) > 20


@pytest.mark.parametrize('width,height', [(640, 360), (360, 640)])
def test_click_wake_has_finite_reach_and_zero_clears_cached_input(
        width, height, monkeypatch):
    engine = _engine('impulse_lens')
    engine.set_time(8)
    engine.set_gravity_radius(.23)
    idle = _coordinates(engine, monkeypatch, width, height)
    lattice = next(iter(engine._material_cache.values()))
    engine._add_impulse((.5, .5))
    burst = _coordinates(engine, monkeypatch, width, height)
    radius = np.hypot((lattice[0] - .5) * width, (lattice[1] - .5) * height)
    outside = radius >= .23 * min(width, height)
    for original, changed in zip(idle, burst):
        assert np.array_equal(original[outside], changed[outside])
    assert not np.array_equal(idle[0], burst[0])
    assert lattice[2]
    engine.set_gravity_radius(.23)
    assert engine._gravity_impulses
    engine.set_gravity_radius(0)
    assert not engine._gravity_impulses
    assert not lattice[2]
    assert all(np.array_equal(first, second) for first, second in zip(
        idle, _coordinates(engine, monkeypatch, width, height)))


def test_enabled_paper_spins_locally_and_returns_to_exact_still_geometry():
    engine = _engine('tissue_facets')
    baseline = _frame(engine)
    material = next(iter(engine._material_cache.values()))
    engine.set_gravity_radius(.22)
    engine.set_pointer((.5, .5))
    lifted = _frame(engine)
    assert lifted != baseline
    engine.set_time(123)
    assert _frame(engine) != lifted
    assert next(iter(engine._material_cache.values())) is material
    engine.set_gravity_radius(0)
    assert _frame(engine) == baseline


def test_radius_rebuilds_atlas_visible_pool_and_restores_exact_default(monkeypatch):
    engine = _engine('point_atlas')
    engine.set_time(8)
    engine.set_pointer((0, 0))
    baseline = _coordinates(engine, monkeypatch, 960, 540)
    engine.set_gravity_radius(1)
    assert not engine._material_cache
    expanded = _coordinates(engine, monkeypatch, 960, 540)
    assert len(expanded[0]) > len(baseline[0])
    engine.set_gravity_radius(0)
    assert not engine._material_cache
    restored = _coordinates(engine, monkeypatch, 960, 540)
    assert all(np.array_equal(first, second) for first, second in zip(baseline, restored))


@pytest.mark.parametrize('family', ['point_atlas', 'genetic_advection', 'impulse_lens'])
def test_density_changes_actual_sample_population(family, monkeypatch):
    engine = _engine(family)
    engine.set_resolution(1)
    populations = []
    for density in (.25, 1, 3):
        engine.set_density(density)
        points = _coordinates(engine, monkeypatch, 960, 540)
        assert points[0].shape == points[1].shape == points[2].shape
        populations.append(points[0].size)
    assert populations[0] < populations[1] < populations[2]

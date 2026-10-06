"""Real moving geometry, continuous terrain and bounded gravity wakes."""

import hashlib

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QImage, QPainter

from spacr.qt.widgets import ambient


def _engine(family, **kwargs):
    return ambient.make_engine(f"data_art_{family}", "spacr", "#101418",
                               seed=42, **kwargs)


def _render(engine, width=512, height=288):
    image = QImage(width, height, QImage.Format_RGB32)
    image.fill(engine.identity)
    painter = QPainter(image)
    painter.setCompositionMode(engine.mode)
    engine._paint_field(painter, width, height)
    painter.end()
    return image


def _digest(image):
    return hashlib.sha256(image.bits().tobytes()).hexdigest()


def _points(engine, monkeypatch, width=512, height=288):
    received = []

    def capture(w, h, x, y, light, spread=False):
        received.append((np.asarray(x).copy(), np.asarray(y).copy(),
                         np.asarray(light).copy()))
        image = QImage(w, h, QImage.Format_RGB32)
        image.fill(engine.identity)
        return image

    monkeypatch.setattr(engine, "_point_material", capture)
    _render(engine, width, height)
    return received[-1]


@pytest.mark.parametrize("pointer", [None, (0, 0), (1, 1), (0, 1), (1, 0)])
def test_atlas_terrain_extends_beyond_every_viewport_edge(monkeypatch, pointer):
    engine = _engine("point_atlas")
    engine.set_pointer(pointer)
    for stamp in (0.0, 19.0, 241.0):
        engine.set_time(stamp)
        x, y, light = _points(engine, monkeypatch)
        assert x.min() < -50 and x.max() > 562
        assert y.min() < -50 and y.max() > 338
        assert np.isfinite(light).all()
        inside = (x >= 0) & (x < 512) & (y >= 0) & (y < 288)
        assert np.count_nonzero(inside) > 3000


def test_facets_move_cached_polygon_geometry_without_rerolling():
    engine = _engine("tissue_facets")
    engine.set_time(2.0)
    first = _digest(_render(engine))
    geometry = next(iter(engine._material_cache.values()))
    assert isinstance(geometry, tuple)
    assert len(geometry) > 400
    engine.set_time(2.15)
    assert _digest(_render(engine)) != first
    assert next(iter(engine._material_cache.values())) is geometry
    engine.set_time(2.0)
    assert _digest(_render(engine)) == first
    other = _engine("tissue_facets")
    other.set_time(2.0)
    assert _digest(_render(other)) == first


def test_lens_click_strengthens_gravity_then_propagates_and_expires(monkeypatch):
    engine = _engine("impulse_lens")
    engine.set_time(8.0)
    engine.set_pointer((0.5, 0.5))
    engine._gravity_impulses.clear()
    idle = _points(engine, monkeypatch)
    engine._add_impulse((0.5, 0.5))
    burst = _points(engine, monkeypatch)
    idle_radius = np.hypot((idle[0] - 256) / 288, (idle[1] - 144) / 288)
    burst_radius = np.hypot((burst[0] - 256) / 288, (burst[1] - 144) / 288)
    close = (idle_radius > 0.05) & (idle_radius < 0.22)
    assert np.mean(burst_radius[close]) < np.mean(idle_radius[close])
    engine.set_time(8.9)
    ripple = _points(engine, monkeypatch)
    moved = np.hypot(ripple[0] - idle[0], ripple[1] - idle[1])
    assert np.count_nonzero(moved > 0.03) > 100
    engine.set_time(13.1)
    expired = _points(engine, monkeypatch)
    assert np.array_equal(expired[0], idle[0])
    assert np.array_equal(expired[1], idle[1])


def test_mouse_movement_wakes_are_rate_limited_and_remain_after_leaving():
    engine = _engine("impulse_lens")
    engine.set_time(1.0)
    engine.set_pointer((0.2, 0.3))
    assert len(engine._gravity_impulses) == 1
    engine.set_pointer((0.2, 0.3))
    engine.set_pointer((0.9, 0.3))
    assert len(engine._gravity_impulses) == 1
    engine.set_time(1.1)
    engine.set_pointer((0.9, 0.3))
    assert len(engine._gravity_impulses) == 2
    engine.set_pointer(None)
    assert len(engine._gravity_impulses) == 2
    engine.set_pointer((float("nan"), 0.2))
    assert engine.pointer is None
    for index in range(40):
        engine.set_time(1.2 + index * 0.07)
        engine._add_impulse((-1, 2), 10)
    assert len(engine._gravity_impulses) == 24
    assert engine._gravity_impulses[-1][1:] == ((0.0, 1.0), 2.0)
    engine.set_time(12.0)
    engine._add_impulse((0.3, 0.2))
    assert len(engine._gravity_impulses) == 1


@pytest.mark.parametrize("point,strength", [(None, 1), ((0.3, float("inf")), 1),
                                          ((0.3, 0.2), float("nan")),
                                          ((0.3, 0.2), 0), ((0.3, 0.2), -1)])
def test_invalid_gravity_events_do_not_change_pixels(point, strength):
    engine = _engine("impulse_lens")
    before = _digest(_render(engine))
    engine._add_impulse(point, strength)
    assert engine._gravity_impulses == []
    assert _digest(_render(engine)) == before


def test_non_gravity_material_ignores_clicks_and_future_events():
    atlas = _engine("point_atlas")
    atlas._add_impulse((0.2, 0.3))
    assert atlas._gravity_impulses == []
    lens = _engine("impulse_lens")
    lens.set_time(5.0)
    lens._add_impulse((0.3, 0.4))
    lens.set_time(2.0)
    fresh = _engine("impulse_lens")
    fresh.set_time(2.0)
    assert _digest(_render(lens)) == _digest(_render(fresh))


@pytest.mark.parametrize("family", ["point_atlas", "tissue_facets", "impulse_lens"])
def test_geometry_controls_remain_reproducible_and_cache_is_bounded(family):
    engine = _engine(family)
    engine.set_time(6.0)
    before = _digest(_render(engine))
    engine.set_size(1.5)
    assert _digest(_render(engine)) != before
    engine.set_size(1.0)
    assert _digest(_render(engine)) == before
    engine.set_density(0.25)
    assert _digest(_render(engine)) != before
    engine.set_density(3.0)
    assert _digest(_render(engine)) != before
    assert len(engine._material_cache) == 1

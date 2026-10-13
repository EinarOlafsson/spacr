"""A stationary cursor attracts moving advection trails within its radius."""

import numpy as np
import pytest
from PySide6.QtGui import QImage

from spacr.qt.widgets import ambient


def _engine(monkeypatch, width, height):
    engine = ambient.make_engine(
        'data_art_genetic_advection', 'spacr', '#101418', seed=42, density=1)
    engine.set_max_pixels(width * height)
    snapshots = []

    def capture(w, h, x, y, intensity, spread=False):
        snapshots.append((np.asarray(x).copy(), np.asarray(y).copy()))
        image = QImage(w, h, QImage.Format_RGB32)
        image.fill(engine.identity)
        return image

    monkeypatch.setattr(engine, '_point_material', capture)
    return engine, snapshots


@pytest.mark.parametrize('radius', [.01, .10, .50])
def test_stationary_cursor_pull_fades_outside_reach_and_follows_motion(
        radius, monkeypatch):
    width, height = 1920, 1080
    engine, frames = _engine(monkeypatch, width, height)
    engine.set_gravity_radius(radius)
    for instant in (2.0, 2.2):
        engine.set_time(instant)
        engine.set_pointer(None)
        engine.shade(width, height)
        baseline_x, baseline_y = frames[-1]
        engine.set_pointer((.5, .5))
        engine.shade(width, height)
        pulled_x, pulled_y = frames[-1]
        distance = np.hypot(baseline_x - width / 2,
                            baseline_y - height / 2)
        outside = distance >= radius * height + 2
        assert np.array_equal(baseline_x[outside], pulled_x[outside])
        assert np.array_equal(baseline_y[outside], pulled_y[outside])
        inside = distance < radius * height * .8
        old_radius = distance[inside]
        new_radius = np.hypot(pulled_x[inside] - width / 2,
                              pulled_y[inside] - height / 2)
        assert inside.sum() > 10
        assert np.count_nonzero(new_radius < old_radius) > inside.sum() * .60
        assert np.mean(old_radius - new_radius) > 0
    assert not np.array_equal(frames[1][0], frames[3][0])
    engine.set_time(2.0)
    engine.shade(width, height)
    assert np.array_equal(frames[1][0], frames[-1][0])
    assert np.array_equal(frames[1][1], frames[-1][1])


def test_pointer_center_and_zero_radius_are_finite_and_exact(monkeypatch):
    width, height = 960, 540
    engine, frames = _engine(monkeypatch, width, height)
    engine.set_time(4.0)
    engine.shade(width, height)
    baseline = frames[-1]
    engine.set_pointer((.5, .5))
    engine.shade(width, height)
    assert all(np.array_equal(left, right) for left, right in zip(
        baseline, frames[-1]))
    engine.set_gravity_radius(.5)
    engine.shade(width, height)
    assert all(np.isfinite(value).all() for value in frames[-1])
    engine.set_gravity_radius(0)
    engine.shade(width, height)
    assert all(np.array_equal(left, right) for left, right in zip(
        baseline, frames[-1]))


def test_moving_cursor_relocates_the_local_attractor(monkeypatch):
    width, height = 960, 540
    engine, frames = _engine(monkeypatch, width, height)
    engine.set_time(8.0)
    engine.set_gravity_radius(.10)
    engine.set_pointer((.25, .5))
    engine.shade(width, height)
    left_x, left_y = frames[-1]
    engine.set_pointer((.75, .5))
    engine.shade(width, height)
    right_x, right_y = frames[-1]
    assert np.count_nonzero((left_x != right_x) | (left_y != right_y)) > 100
    engine.set_pointer((.25, .5))
    engine.shade(width, height)
    assert np.array_equal(left_x, frames[-1][0])
    assert np.array_equal(left_y, frames[-1][1])


def test_attraction_depends_on_the_recent_path_not_only_the_current_point(
        monkeypatch):
    width, height = 960, 540
    engine, frames = _engine(monkeypatch, width, height)
    engine.set_time(8.0)
    engine.set_gravity_radius(.25)
    for size in (.5, 2.0):
        engine.set_size(size)
        engine.set_pointer(None)
        engine.shade(width, height)
        engine.set_pointer((.5, .5))
        engine.shade(width, height)
    for first, second in zip(frames[0], frames[2]):
        assert np.array_equal(first[0], second[0])
    assert np.count_nonzero(frames[1][0][0] != frames[3][0][0]) > 100

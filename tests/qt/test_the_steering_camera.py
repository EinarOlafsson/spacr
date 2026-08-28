"""The camera that steers the Mandelbrot dive, driven frame by frame.

THIS IS THE REAL CODE, not a simulation of it. The logic used to live inside
the GPU canvas's constructor, which needs a GL context to exist -- so every
claim about how smooth the motion was came from a model written beside it,
and three "fixed" reports in a row were wrong that way.
"""
from __future__ import annotations

import math

import pytest

from spacr.qt.widgets.fractal_mandelbrot import (SteeringCamera,
                                                 steering_from_one_number)

#: Thirty frames a second, which is what the pattern targets.
FRAME = 1.0 / 30.0


def _fly(camera, seconds, targets=(), start=0.0):
    """Run the camera for ``seconds``, aiming at ``targets`` as they fall due.

    :param targets: ``(at_second, (dx, dy))`` pairs.
    :returns: the centre at every frame.
    """
    path = []
    pending = list(targets)
    now = start
    end = start + seconds
    while now < end:
        while pending and pending[0][0] <= now - start:
            _when, offset = pending.pop(0)
            camera.aim_at(offset, depth=0.0, scale=1.0)
        path.append(camera.advance(now))
        now += FRAME
    return path


def _accelerations(path):
    velocity = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])]
    return [math.hypot(b[0] - a[0], b[1] - a[1])
            for a, b in zip(velocity, velocity[1:])]


def test_a_new_target_never_makes_the_camera_jump():
    """The target moves; the camera does not."""
    camera = SteeringCamera(strength=0.09, interval=0.4, duration=3.8)
    path = _fly(camera, 30.0, targets=[(0.0, (0.3, 0.0)),
                                       (10.0, (-0.4, 0.2)),
                                       (20.0, (0.1, -0.5))])
    accelerations = _accelerations(path)
    # A jump would be a single frame carrying most of the distance.
    assert max(accelerations) < 0.01, max(accelerations)


def test_the_motion_never_stops_and_restarts():
    """Slide, stop, slide, stop IS the jerk. The speed may fall but must
    not reach zero and then rise again while a target is set."""
    camera = SteeringCamera(strength=0.09, interval=0.4, duration=3.8)
    path = _fly(camera, 20.0, targets=[(0.0, (0.4, 0.3))])
    speeds = [math.hypot(b[0] - a[0], b[1] - a[1])
              for a, b in zip(path, path[1:])]
    # Monotonically slowing as it arrives -- an exponential approach never
    # speeds back up, which is what "no restart" means here.
    for earlier, later in zip(speeds, speeds[1:]):
        assert later <= earlier + 1e-12, (earlier, later)


def test_it_arrives_rather_than_lagging_for_ever():
    """An asymptotic follow that never gets there would read as drifting."""
    camera = SteeringCamera(strength=0.09, interval=0.4, duration=3.8)
    _fly(camera, 30.0, targets=[(0.0, (0.5, 0.0))])
    assert camera.target is not None
    remaining = math.hypot(camera.target[0] - camera.centre[0],
                           camera.target[1] - camera.centre[1])
    assert remaining < 0.005, remaining


@pytest.mark.parametrize("amount", [i / 10.0 for i in range(11)])
def test_no_steering_setting_produces_a_visible_jerk(amount):
    """Across the whole range of the one control the user sees."""
    derived = steering_from_one_number(amount, 24.0)
    camera = SteeringCamera(
        strength=derived["steering_strength"],
        interval=derived["steering_interval_decades"],
        duration=derived["steering_duration"],
        seconds_per_decade=24.0)
    gap = derived["steering_interval_decades"] * 24.0
    targets = [(i * gap, (0.3 if i % 2 else -0.3, 0.2)) for i in range(4)]
    path = _fly(camera, 4 * gap, targets=targets)
    accelerations = _accelerations(path)
    assert max(accelerations) < 0.01, (amount, max(accelerations))


def test_a_late_frame_is_not_a_cut():
    """The machine slept, or a run took the CPU: one huge step would put
    the camera at the target instantly and look like an edit."""
    camera = SteeringCamera(strength=0.09, interval=0.4, duration=3.8)
    camera.advance(0.0)
    camera.aim_at((0.5, 0.5), depth=0.0, scale=1.0)
    camera.advance(FRAME)
    before = camera.centre
    camera.advance(FRAME + 600.0)          # ten minutes later
    moved = math.hypot(camera.centre[0] - before[0],
                       camera.centre[1] - before[1])
    assert moved < 0.05, moved


def test_zero_strength_does_not_steer_at_all():
    """With no reach there is no direction to look in, so every choice is
    arbitrary -- which is what "a random direction every second" was."""
    camera = SteeringCamera(strength=0.0)
    assert camera.steering is False
    assert camera.wants_a_target(depth=99.0) is False


def test_a_contradictory_setting_is_reconciled():
    """Deriving the numbers in the panel does not stop an older install or
    a settings file carrying a combination that cannot work."""
    camera = SteeringCamera(strength=0.5, interval=0.01, duration=100.0,
                            seconds_per_decade=24.0)
    gap = camera.interval * camera.seconds_per_decade
    assert camera.duration <= 0.45 * gap or camera.duration == 0.5


def test_a_drag_stops_the_camera_chasing_its_target():
    """Otherwise it pulls back and fights the hand."""
    camera = SteeringCamera(strength=0.09)
    camera.aim_at((0.4, 0.0), depth=0.0, scale=1.0)
    assert camera.target is not None
    camera.drag(0.2, -0.1, span=1.0, depth=3.0)
    assert camera.target is None
    assert camera.centre == pytest.approx((-0.2, 0.1))


def test_a_restart_returns_the_anchor_as_well_as_the_depth():
    """Or it begins at the surface already pointed a descent away."""
    camera = SteeringCamera(strength=0.09)
    camera.aim_at((0.4, 0.3), depth=5.0, scale=1.0)
    _fly(camera, 5.0)
    camera.restart()
    assert camera.centre == (0.0, 0.0)
    assert camera.target is None
    assert camera.step == 0
    assert camera.next_steer == 0.0


def test_nothing_found_asks_again_sooner_rather_than_giving_up():
    camera = SteeringCamera(strength=0.09, interval=1.0)
    camera.aim_at(None, depth=10.0, scale=1.0)
    assert camera.target is None
    assert 10.0 < camera.next_steer < 11.0

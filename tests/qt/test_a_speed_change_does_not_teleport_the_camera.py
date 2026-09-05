"""Instruction 327 (2): scrolling changes the rate, not the position.

Reported 2026-08-31: "when the user scrolls and changes the speed the
manderbrot should not jump arround and just stay on the sure defined
trajectory... it ruins the immersion when it jumps".

MEASURED BEFORE THE FIX, because the instruction asks for numbers rather
than an argument. Depth was ``t * speed``, so at t=60s a scroll from
speed 1 to speed 2 moved the camera 5.0 units in one instant -- 3,600
frames of ordinary travel, arriving between two frames.

Integrating instead makes it continuous by construction: the camera is
exactly where it was, and only how fast it leaves changes.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets.fractal_travel import DepthPhase, state_at_seconds

FRAME = 1.0 / 60.0


def _run_to(seconds, speed=1.0):
    """A phase advanced frame by frame, the way the canvas does it."""
    phase = DepthPhase()
    t = 0.0
    while t < seconds:
        t += FRAME
        phase.advance(t, speed)
    return phase, t


# ---------------------------------------------------------------------------
# The jump itself
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("before,after", [(1.0, 2.0), (2.0, 1.0),
                                          (1.0, 4.0), (4.0, 0.25)])
def test_changing_speed_leaves_the_camera_where_it_was(before, after):
    """THE FIX. Any speed change, in either direction, moves nothing."""
    phase, t = _run_to(60.0, before)

    was = state_at_seconds(t, before, 1.0, depth_phase=phase.value).depth
    now = state_at_seconds(t, after, 1.0,
                           depth_phase=phase.advance(t, after)).depth

    assert now == pytest.approx(was, abs=1e-12), (
        f"speed {before} -> {after} moved the camera {now - was:+.6f}")


def test_the_rate_actually_changes_though():
    """Otherwise "speed" would be a control that does nothing, which is a
    worse bug than the jump.

    The step is captured BEFORE and AFTER around each advance. Reading
    `phase.value` after calling `advance` and subtracting gives zero --
    advance has already moved it -- which is how the first draft of this
    test failed.
    """
    phase, t = _run_to(60.0, 1.0)

    before_slow = phase.value
    phase.advance(t + FRAME, 1.0)
    slow_step = phase.value - before_slow

    before_fast = phase.value
    phase.advance(t + 2 * FRAME, 2.0)
    fast_step = phase.value - before_fast

    assert slow_step > 0, "the camera stopped moving"
    assert fast_step == pytest.approx(2.0 * slow_step, rel=1e-9), (
        f"a doubled speed moved {fast_step:.6f} against {slow_step:.6f}")


def test_double_the_speed_travels_twice_as_far_per_frame():
    """The rate is proportional, which is what makes the control legible."""
    one = DepthPhase()
    two = DepthPhase()
    one.advance(0.0, 1.0)
    two.advance(0.0, 2.0)

    one.advance(FRAME, 1.0)
    two.advance(FRAME, 2.0)

    assert two.value == pytest.approx(2.0 * one.value, rel=1e-12)


# ---------------------------------------------------------------------------
# The old behaviour, kept so the size of the defect stays on the record
# ---------------------------------------------------------------------------

def test_the_old_formula_is_what_jumped():
    """WITHOUT a phase, `state_at_seconds` still reproduces `t * speed`.

    Kept deliberately: callers with no phase to keep still work, and the
    number below is the defect this instruction was filed about.
    """
    was = state_at_seconds(60.0, 1.0, 1.0).depth
    now = state_at_seconds(60.0, 2.0, 1.0).depth

    assert now - was == pytest.approx(5.0), (
        "the documented jump changed size; the instruction's numbers "
        "need re-measuring")

    one_frame = state_at_seconds(60.0 + FRAME, 1.0, 1.0).depth - was
    assert abs(now - was) / abs(one_frame) == pytest.approx(3600, rel=0.01)


# ---------------------------------------------------------------------------
# The phase's own contract
# ---------------------------------------------------------------------------

def test_the_phase_only_ever_grows():
    phase = DepthPhase()
    seen = []
    for i in range(1, 200):
        seen.append(phase.advance(i * FRAME, 1.0 + (i % 5)))
    assert seen == sorted(seen)


def test_a_clock_that_goes_backwards_rebases_rather_than_rewinding():
    """A restart resets the wall clock. The distance travelled does not
    un-happen, so the phase holds rather than running backwards."""
    phase = DepthPhase()
    phase.advance(1.0, 1.0)
    phase.advance(2.0, 1.0)
    travelled = phase.value

    assert phase.advance(0.5, 1.0) == travelled
    assert phase.value == travelled

    # And it carries on from there once the clock moves forward again.
    assert phase.advance(1.5, 1.0) > travelled


def test_a_speed_of_zero_holds_position_rather_than_reversing():
    phase = DepthPhase()
    phase.advance(1.0, 2.0)
    held = phase.value

    phase.advance(2.0, 0.0)
    assert phase.value == held

    phase.advance(3.0, -5.0)          # a negative speed is not a reverse
    assert phase.value == held


def test_the_first_advance_establishes_the_clock_rather_than_leaping():
    """A phase whose first sample is at t=1000 must not travel 1000
    seconds' worth on that one call."""
    phase = DepthPhase()
    assert phase.advance(1000.0, 4.0) == 0.0


# ---------------------------------------------------------------------------
# The renderers, which is where the jump the user saw actually was
#
# `DepthPhase` was added to both widgets and asked for by neither: the CPU
# widget built one in `__init__` and then sent the worker `t = sim clock,
# speed = the control`, and the GPU canvas fed the shaders `u_time =
# elapsed, u_speed = the control`. Every pattern but the Mandelbrot then
# positions itself with `t * speed` -- the cascade's and the flight's
# depth, the orbit fold's radial phase -- so the picture still moved the
# whole elapsed time in one frame when the speed changed.
#
# Measured with the CPU kernels at t=57.3s, as mean absolute change per
# pixel over a whole frame, speed 1 -> 2 against ordinary travel:
#
#     orbit    1.0s of travel arrived at once   (26.8 against 5.0 a frame)
#     cascade  more than 1.0s                   (25.2 against 3.5)
#     space    more than 30s                    ( 8.4 against 0.1)
#
# And with `variable_speed` on -- which changes the speed EVERY FRAME --
# 1,066 of 3,000 frames ran BACKWARDS, the worst single step being 29
# frames' worth of travel. The sweep was not a slow change of pace at all.
# ---------------------------------------------------------------------------

import numpy as np                                             # noqa: E402

from spacr.qt.widgets import fractal_travel as F               # noqa: E402

#: The real engine, kept before any fixture replaces it with a cheap one.
_REAL_ORBIT_ENGINE = F.OrbitEngine


class _RecordingEngine:
    """A frame costs a memset, so the real loop can be driven frame by frame."""

    def __init__(self, thread_count):
        self.thread_count = thread_count

    def render(self, width, height, t, speed, dream, iterations,
               pointer_x=0.0, pointer_y=0.0, pull=0.0, push=0.0):
        return np.zeros((height, width, 3), dtype=np.uint8)


def _requests_across_a_speed_change(widget, controls, frames=90,
                                    before=1.0, after=2.0):
    """Drive `_request_frame` by hand and collect what the worker was sent.

    Directly rather than through the timer: the stream then has one frame
    per call and the moment the speed changes is exactly known, which is
    what the assertions are about.
    """
    seen = []
    widget.render_requested.connect(seen.append)
    widget.show()
    controls.speed = before
    for index in range(frames):
        if index == frames // 2:
            controls.speed = after
        widget._busy = False
        widget._request_frame()
    widget.hide()
    return seen, frames // 2


@pytest.fixture
def cpu_widget(qapp, monkeypatch):
    """A live CPU backdrop whose frames cost nothing, and its controls."""
    pytest.importorskip("numba")
    monkeypatch.setattr(F, "OrbitEngine", _RecordingEngine)
    controls = F.RuntimeControls()
    widget = F._make_cpu_widget(F.Settings(pattern="orbit", backend="cpu"),
                                controls, F.HardwareProfile(logical_cpus=4))
    yield widget, controls
    widget.shutdown()
    widget.deleteLater()


def test_the_cpu_worker_is_sent_a_phase_and_never_a_speed(cpu_widget):
    """The kernels multiply `t` by `speed`, so the speed must not be one.

    Handing the phase AND the speed would apply it twice, which is the
    same teleport in a different place.
    """
    widget, controls = cpu_widget
    seen, _ = _requests_across_a_speed_change(widget, controls)

    assert seen, "the widget asked for no frames at all"
    assert {request["speed"] for request in seen} == {1.0}


def test_a_scroll_moves_the_cpu_backdrop_by_one_ordinary_frame(cpu_widget):
    """THE FIX. The frame the scroll lands on is the next frame, not a cut."""
    widget, controls = cpu_widget
    seen, changed = _requests_across_a_speed_change(widget, controls)

    steps = [b["t"] - a["t"] for a, b in zip(seen, seen[1:])]
    across = steps[changed - 1]
    before = steps[changed - 2]

    assert across == pytest.approx(2.0 * before, rel=1e-9), (
        f"the scroll moved the picture {across / before:.1f} frames' worth "
        "in one frame")
    # And every step is forward, at one of the two rates and nothing else.
    assert min(steps) > 0.0
    assert max(steps) == pytest.approx(2.0 * min(steps), rel=1e-9)


def test_the_picture_itself_does_not_jump(cpu_widget):
    """The frames, not the numbers: what the user reported was visible.

    Rendered small and at the default iteration count, because the claim
    is about how far the picture moves between two frames rather than
    about how sharp either one is.
    """
    pytest.importorskip("numba")
    widget, controls = cpu_widget
    seen, changed = _requests_across_a_speed_change(widget, controls,
                                                    frames=60)

    def frame(request):
        """One frame of the real kernel at the instant the widget asked for.

        A FRESH ENGINE, filled four times: the orbit blends the last four
        frames, so an engine carried between two instants would compare two
        blends rather than two pictures.
        """
        engine = _REAL_ORBIT_ENGINE(2)
        output = None
        for _ in range(4):
            output = engine.render(96, 64, request["t"], request["speed"],
                                   1.5, 5)
        return output.astype(np.float64)

    landed = np.abs(frame(seen[changed]) - frame(seen[changed - 1])).mean()
    ordinary = np.abs(frame(seen[changed + 2])
                      - frame(seen[changed + 1])).mean()

    assert landed == pytest.approx(ordinary, rel=0.5), (
        f"the scroll moved the picture {landed:.2f} levels per pixel where "
        f"an ordinary frame at the new speed moves {ordinary:.2f}")


def test_variable_speed_never_runs_the_travel_backwards():
    """A sweeping speed changes the pace, not the position.

    Measured on the old formula over 100 seconds at 30 fps: 1,066 frames
    of 3,000 went backwards and the worst step was 29 frames' worth of
    travel. The sweep read as constant jitter rather than as a slow change
    of pace, and it is on by choice in Preferences.
    """
    controls = F.RuntimeControls(variable_speed=True)
    phase = F.DepthPhase()
    period = 1.0 / 30.0

    old = []
    new = []
    t = 0.0
    for _ in range(3000):
        t += period
        speed = controls.speed_at(t)
        old.append(t * speed)
        new.append(phase.advance(t, speed))

    assert sum(1 for a, b in zip(old, old[1:]) if b < a) > 500, (
        "the old formula stopped going backwards; re-measure the numbers "
        "in this file before deleting it")
    assert all(b >= a for a, b in zip(new, new[1:])), "the travel reversed"
    # No step larger than the fastest the sweep is allowed to travel.
    fastest = period * max(controls.speed_min, controls.speed_max)
    assert max(b - a for a, b in zip(new, new[1:])) <= fastest + 1e-9


def test_the_gpu_canvas_hands_the_shaders_the_phase(qapp):
    """SOURCE, because the canvas needs a GL context to exist.

    Every GPU pattern but the Mandelbrot computes its position as
    `u_time * u_speed`, so pinning the speed at one and moving the clock
    is what makes a scroll continuous there. The Mandelbrot reads neither
    uniform; its dive integrates its own depth.
    """
    import inspect

    body = inspect.getsource(F._make_gpu_widget)
    assert '("u_time", np.float32(phase))' in body
    assert '("u_speed", np.float32(1.0))' in body
    assert "state_at_seconds(phase, 1.0" in body


@pytest.mark.parametrize("module", ["fractal_orbit_gpu", "fractal_cascade",
                                    "fractal_space"])
def test_the_gpu_shaders_still_scale_the_clock_by_the_speed(module):
    """Which is why the canvas must pin it.

    A pattern that stopped doing this would not need the pin -- and a new
    one that does it is covered by the pin already. Either way the pairing
    is the thing worth catching in a test.
    """
    import importlib

    source = importlib.import_module(
        f"spacr.qt.widgets.{module}").FRAGMENT_SHADER
    assert "u_speed" in source, "no speed term left; the pin can go"

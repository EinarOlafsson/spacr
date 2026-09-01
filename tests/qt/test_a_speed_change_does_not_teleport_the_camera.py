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

"""The graphics lane judges each GPU frame like the direct path does.

On the shading thread ``_GraphicsBackend._graphics_frame`` goes through
``_graphics_frame_on_lane``, which has its own copy of the frame
judgement: a renderer's first frame compiles and is not judged, a later
frame that held the interpreter lock past ``_GRAPHICS_HOLD_BUDGET_S``
sends the engine to the CPU for ``_GRAPHICS_SLOW_BACKOFF_S``, and a lane
that found no renderer hands the frame to the CPU without judging
anything.

The lane is a synchronous stand-in and the frame's hold comes from a
test clock, so the outcome depends on neither a thread answering inside
the frame wait nor the host's speed.
"""
from __future__ import annotations

import time
import types

import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets import ambient                             # noqa: E402


class _Backend(ambient._GraphicsBackend):
    """A bare graphics backend: the mixin needs no engine to be driven."""

    name = "lane-probe"

    def __init__(self, mode="gpu"):
        self._graphics_backend = mode


class _InlineLane:
    """A graphics lane that runs each call at once and never times out."""

    def __init__(self):
        self.calls = 0

    def alive(self):
        return True

    def busy(self):
        return False

    def call(self, fn, timeout_s):
        self.calls += 1
        return True, fn(), None


class _Clock:
    """``perf_counter`` for the module under test, advanced by the renderer."""

    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class _TimedRenderer:
    """A renderer whose frames take exactly the hold the test sets."""

    _owner = None

    def __init__(self, clock):
        self.clock = clock
        self.hold_s = 0.0
        self.calls = 0

    def _draw(self, width, height, *args):
        self.calls += 1
        self.clock.now += self.hold_s
        return ("frame", self.calls, width, height, args)


@pytest.fixture
def backend(monkeypatch):
    """A GPU backend whose lane runs inline, with no analysis policy."""
    monkeypatch.setattr(ambient, "_RESOURCE_POLICY", None)
    probe = _Backend()
    probe.__dict__["_graphics_lane"] = _InlineLane()
    return probe


def test_a_frame_over_the_hold_budget_on_the_lane_backs_off_to_the_cpu(
        backend, monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(ambient, "time", types.SimpleNamespace(
        perf_counter=clock, monotonic=time.monotonic))
    renderer = _TimedRenderer(clock)
    monkeypatch.setattr(ambient._GraphicsBackend, "_graphics_renderer",
                        lambda self, width, height, work=0: renderer)
    lane = backend.__dict__["_graphics_lane"]
    over = ambient._GRAPHICS_HOLD_BUDGET_S * 4

    renderer.hold_s = over
    first = backend._graphics_frame_on_lane("_draw", 64, 48, (1,), 0, None)
    assert first == ("frame", 1, 64, 48, (1,))
    assert backend.__dict__["_graphics_judged"] is renderer
    assert "_graphics_slow_until" not in backend.__dict__, (
        "a renderer's first frame compiles and is not judged")

    renderer.hold_s = ambient._GRAPHICS_HOLD_BUDGET_S / 4
    quick = backend._graphics_frame_on_lane("_draw", 64, 48, (2,), 0, None)
    assert quick == ("frame", 2, 64, 48, (2,))
    assert "_graphics_slow_until" not in backend.__dict__

    renderer.hold_s = over
    before = time.monotonic()
    slow = backend._graphics_frame_on_lane("_draw", 64, 48, (3,), 0, None)
    after = time.monotonic()
    assert slow == ("frame", 3, 64, 48, (3,)), "the late frame is still shown"
    until = backend.__dict__["_graphics_slow_until"]
    assert (before + ambient._GRAPHICS_SLOW_BACKOFF_S <= until
            <= after + ambient._GRAPHICS_SLOW_BACKOFF_S)
    assert lane.calls == 3

    assert backend._graphics_frame("_draw", 64, 48, 4) is None
    assert backend._graphics_refusal == (
        "the GPU held the interpreter lock too long")
    assert renderer.calls == 3 and lane.calls == 3, "backed off to the CPU"


def test_no_renderer_on_the_lane_means_a_cpu_frame_and_no_backoff(
        backend, monkeypatch):
    asked = []

    def _no_renderer(self, width, height, work=0):
        asked.append((width, height, work))
        return None

    monkeypatch.setattr(ambient._GraphicsBackend, "_graphics_renderer",
                        _no_renderer)
    lane = backend.__dict__["_graphics_lane"]

    assert backend._graphics_frame_on_lane("_draw", 64, 48, (), 3, None) is None
    assert lane.calls == 1
    assert asked == [(64, 48, 3)]
    assert "_graphics_judged" not in backend.__dict__
    assert "_graphics_slow_until" not in backend.__dict__
    assert backend._graphics_failed is False

    assert backend._graphics_frame_on_lane("_draw", 64, 48, (), 3, None) is None
    assert lane.calls == 2, "no backoff: the lane is asked again"
    assert asked == [(64, 48, 3), (64, 48, 3)]

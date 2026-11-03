"""A GPU frame stuck in the driver never stops the backdrop field.

Lesson 19's capture (item 747): with the GPU busy elsewhere, one impulse-lens
frame took 8.9 s. The shading thread waited for it from submission to
readback, so the field stood still for that long although the GUI thread
kept painting. The renderer now lives on a graphics lane; the shading thread
waits at most ``_GRAPHICS_FRAME_WAIT_S`` (a first frame
``_GRAPHICS_FIRST_FRAME_WAIT_S``), shades that frame on the CPU, stays there
for the backoff and drops the late GPU frame.

The fake renderer sleeps, releasing the interpreter lock as a driver wait
does, so the CPU path can run beside it.
"""
from __future__ import annotations

import threading
import time

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtGui import QImage                                 # noqa: E402
from PySide6.QtWidgets import QWidget                            # noqa: E402

from spacr.qt.widgets import ambient                             # noqa: E402
import types
import logging

pytestmark = pytest.mark.qt

THEME = "data_art_impulse_lens"


@pytest.fixture(autouse=True)
def _no_hub_left_behind(qapp):
    """Take down an application event hub these backdrops installed."""
    from spacr.qt import gil_priority

    before = gil_priority._application_event_hub(qapp, create=False)
    yield
    after = gil_priority._application_event_hub(qapp, create=False)
    if after is not None and after is not before:
        qapp.removeEventFilter(after)
        setattr(qapp, gil_priority._HUB_ATTRIBUTE, before)
        after.deleteLater()


class _StuckRenderer:
    """A GPU renderer whose second frame waits ``stuck_s`` in the driver."""

    def __init__(self, stuck_s):
        self.stuck_s = stuck_s
        self.calls = 0
        self.threads = set()
        self.closed = threading.Event()
        self._owner = None

    def _frame(self, width, height):
        self.calls += 1
        self.threads.add(threading.current_thread().name)
        if self.calls == 2:
            time.sleep(self.stuck_s)
        image = QImage(width, height, QImage.Format_RGB32)
        image.fill(0)
        return image

    def _draw(self, width, height, *args):
        return self._frame(width, height)

    def _draw_subpixel(self, width, height, *args):
        return self._frame(width, height)


def test_a_stuck_gpu_frame_leaves_the_field_on_the_cpu(qtbot, monkeypatch):
    """Measured here, 800x500, a 4 s driver wait on the second GPU frame:
    before the lane the field published nothing for 4 s; with it the longest
    pause between published frames stays under 0.4 s and the GPU is not
    asked again while the frame is stuck."""
    renderer = _StuckRenderer(4.0)
    monkeypatch.setattr(ambient, "_RESOURCE_POLICY", None)
    monkeypatch.setattr(
        ambient._GraphicsBackend, "_graphics_renderer",
        lambda self, width, height, work=0: (
            renderer if threading.current_thread().name.startswith("spacr-ambient-")
            else None))
    monkeypatch.setattr(ambient._GraphicsBackend, "_graphics_accepts",
                        lambda self, width, height, work=0: True, raising=False)
    host = QWidget()
    host.resize(800, 500)
    backdrop = ambient.AmbientWidget(host, theme=THEME, fps=30, seed=7)
    backdrop.follow_parent()
    backdrop._engine._graphics_backend = "gpu"
    qtbot.addWidget(host)
    host.show()
    qtbot.waitExposed(host)
    stamps = []
    try:
        deadline = time.monotonic() + 3.5
        last = -1
        while time.monotonic() < deadline:
            producer = backdrop._producer_box[0]
            shaded = producer.frames_shaded if producer is not None else 0
            if shaded != last:
                stamps.append(time.monotonic())
                last = shaded
            qtbot.wait(10)
        stamps.append(time.monotonic())
    finally:
        backdrop.stop()
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    assert max(gaps) < 0.4, round(max(gaps), 3)
    assert len(stamps) > 20, "the field kept moving on the CPU"
    assert renderer.calls == 2, "the stuck frame kept the GPU from being asked again"
    assert renderer.threads == {"spacr-ambient-gpu"}


def test_a_lane_call_is_bounded_and_a_busy_lane_answers_at_once():
    lane = ambient._GraphicsLane()
    release = threading.Event()
    try:
        started = time.perf_counter()
        done, result, error = lane.call(lambda: release.wait(5) and 1, 0.05)
        assert not done and time.perf_counter() - started < 0.5
        assert lane.busy()
        started = time.perf_counter()
        assert lane.call(lambda: 2, 1.0) == (False, None, None)
        assert time.perf_counter() - started < 0.05
        release.set()
        until = time.monotonic() + 2
        while lane.busy() and time.monotonic() < until:
            time.sleep(0.01)
        assert lane.call(lambda: 3, 1.0) == (True, 3, None)
        done, result, error = lane.call(lambda: 1 / 0, 1.0)
        assert done and isinstance(error, ZeroDivisionError)
    finally:
        released = []
        lane.close(lambda: released.append(threading.current_thread().name), 2.0)
    assert released == ["spacr-ambient-gpu"]
    assert not lane.thread.is_alive()


class _Backend(ambient._GraphicsBackend):
    """A bare graphics backend: the mixin needs no engine to be driven."""

    name = "lane-probe"

    def __init__(self, mode="gpu"):
        self._graphics_backend = mode


class _Boom:
    _owner = None

    def _draw(self, width, height, *args):
        raise RuntimeError("driver lost")


def test_a_draw_error_on_the_shading_thread_pins_the_engine_to_the_cpu(monkeypatch):
    backend = _Backend()
    built = []
    released = []
    monkeypatch.setattr(ambient._GraphicsBackend, "_graphics_renderer",
                        lambda self, width, height, work=0: _Boom())
    monkeypatch.setattr(backend, "_release_point_graphics",
                        lambda: released.append(True))
    frame = backend._graphics_frame("_draw", 64, 48,
                                    build=lambda: built.append(1) or (7,))
    assert frame is None
    assert built == [1]
    assert released == [True]
    assert backend._graphics_failed is True
    assert backend._graphics_refusal.startswith("_draw failed in _draw line ")
    assert "RuntimeError: driver lost" in backend._graphics_refusal


def test_a_frame_while_the_lane_is_busy_stays_on_the_cpu():
    backend = _Backend()
    lane = types.SimpleNamespace(busy=lambda: True)
    backend.__dict__["_graphics_lane"] = lane
    assert backend._graphics_frame_on_lane("_draw", 64, 48, (), 0, None) is None
    assert backend._graphics_refusal == "a GPU frame is still in the driver"


def test_a_canvas_above_4k_is_refused_before_a_lane_starts():
    backend = _Backend()
    assert backend._graphics_accepts(3841, 2161) is False
    assert backend._graphics_refusal == (
        "canvas larger than 3840x2160 or big-endian host")
    assert "_graphics_lane" not in backend.__dict__


def test_closing_without_a_lane_does_nothing_and_with_one_releases_on_it():
    backend = _Backend()
    backend._close_graphics_lane()
    assert "_graphics_lane" not in backend.__dict__
    closed = []
    backend.__dict__["_graphics_lane"] = types.SimpleNamespace(
        close=lambda release, timeout: closed.append((release, timeout)))
    backend._close_graphics_lane()
    assert "_graphics_lane" not in backend.__dict__
    assert closed == [(backend._release_point_graphics,
                       ambient._GRAPHICS_LANE_CLOSE_S)]


def test_a_lane_whose_release_fails_still_ends_and_says_so(caplog):
    lane = ambient._GraphicsLane()

    def failing_release():
        raise OSError("context already gone")

    with caplog.at_level(logging.DEBUG, logger=ambient.LOG.name):
        lane.close(failing_release, 2.0)
    assert not lane.thread.is_alive()
    assert any("could not release the graphics lane's renderer" in r.getMessage()
               for r in caplog.records)


def test_a_lane_closed_from_its_own_thread_does_not_wait_for_itself():
    lane = ambient._GraphicsLane()
    released = []
    started = time.perf_counter()
    done, result, error = lane.call(
        lambda: lane.close(lambda: released.append(threading.current_thread().name),
                           30.0), 5.0)
    assert (done, error) == (True, None)
    lane.thread.join(2.0)
    assert not lane.thread.is_alive()
    assert time.perf_counter() - started < 5.0
    assert released == ["spacr-ambient-gpu"]


def test_retiring_lanes_joins_a_live_one_and_counts_it_but_skips_ended_ones(
        monkeypatch):
    import weakref

    monkeypatch.setattr(ambient, "_LIVE_GRAPHICS_LANES", weakref.WeakSet())
    ended = ambient._GraphicsLane()
    ended.close(None, 2.0)
    busy = ambient._GraphicsLane()
    release = threading.Event()
    try:
        assert busy.call(lambda: release.wait(5), 0.05)[0] is False
        started = time.perf_counter()
        assert ambient._retire_graphics_lanes(0.05) == 1
        assert time.perf_counter() - started < 1.0
        release.set()
        busy.close(None, 2.0)
        assert ambient._retire_graphics_lanes(0.05) == 0
    finally:
        release.set()


def test_exit_retirement_warns_about_a_lane_still_in_the_driver(monkeypatch, caplog):
    import weakref

    monkeypatch.setattr(ambient, "_LIVE_GRAPHICS_LANES", weakref.WeakSet())
    lane = ambient._GraphicsLane()
    release = threading.Event()
    try:
        assert lane.call(lambda: release.wait(5), 0.05)[0] is False
        with caplog.at_level(logging.WARNING, logger=ambient.LOG.name):
            ambient._retire_shading_threads(timeout_s=0.05)
        assert any("graphics lane(s) were still in the driver at exit"
                   in r.getMessage() for r in caplog.records)
    finally:
        release.set()
        lane.close(None, 2.0)

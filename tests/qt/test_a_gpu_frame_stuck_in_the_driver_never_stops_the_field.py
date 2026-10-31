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

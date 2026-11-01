"""A field delivers the rate it targets, in a run and under a slow frame.

Lessons 19 and 22 (workstation, real GPU): a GPU frame a little over 33 ms
sent the field to the CPU for 30 s, a run halved its rate to 30 although the
CPU frame costs about a millisecond, and a frame that finished just after its
tick was shown a whole period late, so 30 became 24-27 and the video held
21-28 distinct frames against the gate's 28. The lane now waits 0.1 s for a
frame, a run keeps the full rate while repaints are cheap, and a tick that
found no new frame asks again in a few milliseconds on the same beat.
Delivered paints are counted against the target, within 10 %.
"""
from __future__ import annotations

import threading
import time

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

from spacr.qt import gil_priority  # noqa: E402
from spacr.qt.widgets import ambient  # noqa: E402

pytestmark = pytest.mark.qt

TOLERANCE = 0.10


@pytest.fixture(autouse=True)
def _a_field_that_runs(monkeypatch):
    from spacr.qt import preferences

    monkeypatch.setattr(preferences, "get_reduce_motion", lambda: False)
    monkeypatch.setattr(ambient, "_system_asks_for_stillness", lambda: False)
    yield
    ambient._retire_shading_threads(timeout_s=3.0)


def _pump(seconds):
    app = QApplication.instance()
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        time.sleep(0.0005)


def _delivered(qtbot, monkeypatch, fps, *, load=False, shade_ms=None,
               seconds=3.0):
    stamps = []
    original = ambient.AmbientWidget.paintEvent

    def counting(self, event):
        stamps.append(time.perf_counter())
        return original(self, event)

    monkeypatch.setattr(ambient.AmbientWidget, "paintEvent", counting)
    if shade_ms is not None:
        shade = ambient._DataArtEngine.shade

        def slow(self, *args, **kwargs):
            time.sleep(shade_ms / 1000.0)
            return shade(self, *args, **kwargs)

        monkeypatch.setattr(ambient._DataArtEngine, "shade", slow)
    host = QWidget()
    host.setProperty("spacrIndependentBackdrop", True)
    host.resize(640, 360)
    field = ambient.AmbientWidget(host, theme="data_art_impulse_lens", fps=fps)
    field.setGeometry(host.rect())
    qtbot.addWidget(host)
    host.show()
    stop = threading.Event()
    threads = []
    if load:
        import numpy as np

        def busy():
            while not stop.is_set():
                a = np.random.rand(700, 700)
                a @ a
                sum(range(20000))

        gil_priority.claim()
        threads = [threading.Thread(target=busy, daemon=True) for _ in range(2)]
        for thread in threads:
            thread.start()
    try:
        _pump(1.0)
        stamps.clear()
        _pump(seconds)
        return len(stamps) / seconds, field
    finally:
        stop.set()
        for thread in threads:
            thread.join(2.0)
        if load:
            gil_priority.release()


@pytest.mark.parametrize("target", [60, 30])
def test_a_field_in_a_run_delivers_its_target_within_ten_percent(
        qtbot, monkeypatch, target):
    rate, field = _delivered(qtbot, monkeypatch, target, load=True)
    assert field._run_paced
    assert rate >= target * (1.0 - TOLERANCE), (target, rate)
    assert rate <= target * (1.0 + TOLERANCE), (target, rate)


def test_a_run_keeps_the_full_rate_while_repaints_are_cheap_and_paces_when_not(
        qtbot, monkeypatch):
    host = QWidget()
    host.setProperty("spacrIndependentBackdrop", True)
    host.resize(320, 200)
    field = ambient.AmbientWidget(host, theme="blobs", fps=60)
    field.setGeometry(host.rect())
    qtbot.addWidget(host)
    field._run_paced = True
    field._apply_rate()
    assert field._rate() == 60
    field._frame_cost_ms = ambient._RUN_PACED_ABOVE_MS * 2
    field._follow_the_frame_cost()
    assert field._rate() == min(60, ambient._RUN_FPS)
    field._frame_cost_ms = (ambient._RUN_FULL_RATE_BELOW_MS
                            + ambient._RUN_PACED_ABOVE_MS) / 2
    field._follow_the_frame_cost()
    assert field._rate() == min(60, ambient._RUN_FPS)
    field._frame_cost_ms = ambient._RUN_FULL_RATE_BELOW_MS / 2
    field._follow_the_frame_cost()
    assert field._rate() == 60


def test_a_35_ms_shade_is_delivered_at_nearly_its_own_pace_not_half_the_rate(
        qtbot, monkeypatch):
    rate, _field = _delivered(qtbot, monkeypatch, 60, shade_ms=35.0)
    assert rate >= 0.85 / 0.035, rate


@pytest.mark.parametrize("frame_ms", [35, 56, 90])
def test_a_gpu_frame_up_to_the_lane_budget_is_waited_for_not_dropped(frame_ms):
    lane = ambient._GraphicsLane()
    try:
        done, result, error = lane.call(
            lambda: (time.sleep(frame_ms / 1000.0), "frame")[1],
            ambient._GRAPHICS_FRAME_WAIT_S)
        assert done and result == "frame" and error is None
    finally:
        lane.close(None, 1.0)


def test_a_gpu_frame_far_past_the_budget_is_still_not_waited_for():
    lane = ambient._GraphicsLane()
    release = threading.Event()
    try:
        done, result, _error = lane.call(
            lambda: release.wait(5.0),
            ambient._GRAPHICS_FRAME_WAIT_S)
        assert not done and result is None
    finally:
        release.set()
        lane.close(None, 2.0)

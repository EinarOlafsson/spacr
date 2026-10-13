"""A backdrop paused for a run blits its still frame instead of reshading.

Lesson 08's Measure run (clip batch_20_batch_settings) froze the window for
250-350 ms several times. Measured offscreen on 2026-10-13 at 2560x1440 with
a real run, a 2 ms heartbeat and a sampler of the GUI thread's stack: a
CPU-heavy run pauses the field animation, which stops its timer and retires
its shading thread, and with no shading thread every paint event shaded a
whole field frame on the GUI thread. Every console line and figure the
window paints over the backdrop asks it for one, and against the run's
worker one impulse-lens frame took 250-350 ms.

A backdrop that stops while on screen now keeps the frame it was showing
and blits it again; any change to the engine drops it, and a hidden one
keeps nothing.
"""
from __future__ import annotations

import threading
import time

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtGui import QImage, QPainter                          # noqa: E402
from PySide6.QtWidgets import QWidget                               # noqa: E402

from spacr.qt.widgets import ambient                                # noqa: E402

pytestmark = pytest.mark.qt

THEME = "data_art_impulse_lens"


class _Busy:
    """The workload policy while a CPU-heavy run goes."""

    enabled = True
    cpu_busy = True
    gpu_busy = False

    def pauses_animation(self):
        return True

    def blocks_gpu(self):
        return True


@pytest.fixture(autouse=True)
def _no_hub_left_behind(qapp):
    """Take down an application event hub these backdrops installed.

    A field backdrop watches the application's mouse events through the
    hub; one left on the application would answer a later test's events.
    """
    from spacr.qt import gil_priority

    before = gil_priority._application_event_hub(qapp, create=False)
    yield
    after = gil_priority._application_event_hub(qapp, create=False)
    if after is not None and after is not before:
        qapp.removeEventFilter(after)
        setattr(qapp, gil_priority._HUB_ATTRIBUTE, before)
        after.deleteLater()


@pytest.fixture
def backdrop(qtbot, monkeypatch):
    """A field backdrop on screen, shading on its own thread."""
    monkeypatch.setattr(ambient, "_RESOURCE_POLICY", None)
    host = QWidget()
    host.resize(640, 400)
    widget = ambient.AmbientWidget(host, theme=THEME, fps=30)
    widget.setGeometry(host.rect())
    qtbot.addWidget(host)
    host.show()
    qtbot.waitExposed(host)
    qtbot.waitUntil(lambda: widget._producer_box[0] is not None,
                    timeout=10000)
    widget.repaint()
    qtbot.waitUntil(lambda: widget._last_frame is not None, timeout=10000)
    yield host, widget
    widget.stop()


def _count_shading(monkeypatch, widget):
    """Count the engine's shading passes from here on."""
    calls = []
    engine = widget._engine
    original = engine._shade

    def counted(width, height):
        calls.append((width, height))
        return original(width, height)

    monkeypatch.setattr(engine, "_shade", counted)
    return calls


def _pause_for_a_run(monkeypatch, widget):
    monkeypatch.setattr(ambient, "_RESOURCE_POLICY", _Busy())
    widget._sync_run_state()


def _grab(widget):
    return widget.grab().toImage().convertToFormat(QImage.Format_RGB32)


def test_a_paused_backdrop_on_screen_repaints_without_shading(
        backdrop, monkeypatch):
    _, widget = backdrop
    shown = widget._last_frame
    _pause_for_a_run(monkeypatch, widget)
    assert not widget.is_running()
    assert widget._producer_box[0] is None
    assert widget._still is not None and widget._still[1] is shown

    calls = _count_shading(monkeypatch, widget)
    for _ in range(12):
        widget.repaint()
    assert calls == []


def test_the_still_is_the_frame_on_screen_when_the_run_started(
        backdrop, monkeypatch):
    _, widget = backdrop
    widget._timer.stop()
    widget.repaint()
    before = _grab(widget)
    _pause_for_a_run(monkeypatch, widget)
    assert _grab(widget) == before


def test_a_still_shaded_after_a_change_is_the_picture_the_old_paint_drew(
        backdrop, monkeypatch):
    _, widget = backdrop
    _pause_for_a_run(monkeypatch, widget)
    widget.set_time(widget.time() + 1.25)
    assert widget._still is None
    calls = _count_shading(monkeypatch, widget)
    now = _grab(widget)
    assert len(calls) == 1
    widget.repaint()
    assert len(calls) == 1

    monkeypatch.setattr(widget, "_paint_still", lambda *args: False)
    assert _grab(widget) == now


def test_a_resized_paused_backdrop_shades_once_for_its_new_size(
        qtbot, backdrop, monkeypatch):
    host, widget = backdrop
    _pause_for_a_run(monkeypatch, widget)
    calls = _count_shading(monkeypatch, widget)
    host.resize(700, 420)
    widget.setGeometry(host.rect())
    qtbot.waitUntil(lambda: widget.width() == 700, timeout=5000)
    for _ in range(5):
        widget.repaint()
    assert len(calls) == 1
    assert widget._still[0][0] == widget._art_render_size(700, 420)


def test_a_hidden_backdrop_keeps_no_still(qtbot, backdrop, monkeypatch):
    host, widget = backdrop
    _pause_for_a_run(monkeypatch, widget)
    assert widget._still is not None
    host.hide()
    qtbot.waitUntil(lambda: not widget.isVisible(), timeout=5000)
    widget.stop()
    assert widget._still is None


def test_an_engine_without_a_buffer_keeps_no_still(qtbot, monkeypatch):
    monkeypatch.setattr(ambient, "_RESOURCE_POLICY", None)
    widget = ambient.AmbientWidget(None, theme="drift", fps=30)
    widget.resize(200, 120)
    qtbot.addWidget(widget)
    widget.show()
    qtbot.waitExposed(widget)
    assert not isinstance(widget._engine, ambient._BufferedEngine)
    _pause_for_a_run(monkeypatch, widget)
    canvas = QImage(200, 120, QImage.Format_RGB32)
    painter = QPainter(canvas)
    try:
        assert widget._paint_still(painter, 200, 120) is False
    finally:
        painter.end()
    assert widget._still is None
    widget.stop()


def _busy_python(stop):
    total = 0
    while not stop.is_set():
        for value in range(2000):
            total += value * value


def test_a_paused_repaint_against_a_python_worker_blocks_far_less(
        backdrop, monkeypatch):
    """Self-calibrated: the old shading paint against the still, same load."""
    _, widget = backdrop
    _pause_for_a_run(monkeypatch, widget)
    stop = threading.Event()
    worker = threading.Thread(target=_busy_python, args=(stop,), daemon=True)
    worker.start()
    try:
        def timed(rounds):
            spent = []
            for _ in range(rounds):
                started = time.perf_counter()
                widget.repaint()
                spent.append(time.perf_counter() - started)
            return sorted(spent)[rounds // 2]

        now = timed(9)
        monkeypatch.setattr(widget, "_paint_still", lambda *args: False)
        before = timed(9)
    finally:
        stop.set()
        worker.join()
    assert now < before / 3, (now, before)

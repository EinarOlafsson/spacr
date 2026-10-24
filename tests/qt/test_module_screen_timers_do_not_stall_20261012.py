"""No periodic timer on a module screen holds the GUI thread.

The tutorial capture on the workstation (real application, GNOME on the GPU
host, nightly d2582450e) found every module screen, and not Home, painting
nothing for about 100 ms once every 2.0 s, which made 50-90 % of the
repeated video frames. The only periodic timer a module screen runs and
Home does not is the System strip's usage poll (``AppScreen._usage_timer``,
2000 ms). Two things in it are fixed here.

* Painting the reading. ``QProgressBar.setValue`` repaints synchronously,
  and each bar is translucent over the window's field, so every tick
  painted the field under four bars and flushed the window four times
  outside the field's own frames; on a composited GPU desktop a flush can
  wait for the display. Offscreen, profiled over ten ticks: 28 synchronous
  field paints and 37 ms in ``setValue``; now none, and 6 ms in all of
  ``set_value``. The bars are also re-polished only when a reading crosses
  into another colour band.
* Taking the reading. The worker asked ``GPUtil``, which starts
  ``nvidia-smi`` every tick (35-100 ms, a driver open and close each time,
  on the card that draws the desktop). It now reads NVML in process and
  starts no process.

Heartbeat (2 ms) on the Mask screen offscreen, field stopped, load 28-31,
gaps a usage tick fell in: worst 8.5 ms in both runs before, 5.1-5.7 ms in
three runs after; GUI CPU in the delivery handler 1.5-3.8 ms before,
0.2-0.7 ms after.

The Mask live preview showed its two 2048-pixel pictures through
``QGraphicsView`` items drawn scaled on every field frame; the items are now
cached at the size they are shown.
"""
from __future__ import annotations

import statistics
import subprocess
import sys
import time
import types

import numpy as np
import pytest

pytest.importorskip("PySide6")

#: The GUI-thread budget for anything a periodic timer does, in ms. The
#: bound actually asserted is this or, on a machine so loaded that ordinary
#: event-loop gaps exceed it, a multiple of those.
STALL_BUDGET_MS = 8.0

#: How many times the 99th-percentile unrelated gap a timer-related gap may
#: reach on a loaded machine.
LOAD_FACTOR = 2.0

SECONDS = 6.5

#: Most heartbeat windows measured before the worst tick must have fit.
WINDOWS = 3

#: How long the Mask screen is left to finish opening first: the categories
#: still waiting to be built are built in idle slices after the first show.
SETTLE_MS = 3000


class _Card:
    """One card as NVML describes it."""


def _fake_nvml(load=37, used=3, total=12, cards=1, broken=False):
    """An initialised NVML module answering for one card."""
    nvml = types.ModuleType("pynvml")
    asked = []

    def count():
        asked.append("count")
        return cards

    def handle(index):
        asked.append(("handle", index))
        return _Card()

    def utilisation(_handle):
        if broken:
            raise RuntimeError("NVML_ERROR_GPU_IS_LOST")
        return types.SimpleNamespace(gpu=load)

    def memory(_handle):
        return types.SimpleNamespace(used=used, total=total)

    nvml.nvmlDeviceGetCount = count
    nvml.nvmlDeviceGetHandleByIndex = handle
    nvml.nvmlDeviceGetUtilizationRates = utilisation
    nvml.nvmlDeviceGetMemoryInfo = memory
    nvml.asked = asked
    return nvml


@pytest.fixture
def no_processes(monkeypatch):
    """Record every process the code under test starts, and start none."""
    started = []

    class _Refused(subprocess.Popen):
        def __init__(self, args, *rest, **kwargs):
            started.append(args)
            raise OSError("no process may be started here")

    monkeypatch.setattr(subprocess, "Popen", _Refused)
    return started


@pytest.fixture
def started_processes(monkeypatch):
    """Record every process the application starts, and let it start."""
    started = []
    real = subprocess.Popen

    class _Recorded(real):
        def __init__(self, args, *rest, **kwargs):
            started.append(args)
            super().__init__(args, *rest, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", _Recorded)
    return started


@pytest.fixture
def gputil(monkeypatch):
    """A GPUtil that records being asked and reports one card."""
    asked = []
    card = types.SimpleNamespace(load=0.5, memoryUtil=0.25)
    fake = types.ModuleType("GPUtil")
    fake.getGPUs = lambda: asked.append(1) or [card]
    monkeypatch.setitem(sys.modules, "GPUtil", fake)
    return asked


@pytest.fixture
def nvidia_host(monkeypatch):
    """A host whose driver tools are installed."""
    from spacr.qt.screens import app_screen

    monkeypatch.setattr(app_screen, "_nvidia_smi_available", lambda: True)
    return app_screen


def _use_nvml(monkeypatch, nvml):
    from spacr.qt.widgets import home

    monkeypatch.setattr(home, "_NVML", nvml)


def test_the_usage_poll_reads_the_card_without_starting_a_process(
        monkeypatch, nvidia_host, gputil, no_processes):
    nvml = _fake_nvml(load=37, used=3, total=12)
    _use_nvml(monkeypatch, nvml)

    sample = nvidia_host._sample_usage(True)

    assert no_processes == []
    assert gputil == [], "GPUtil starts nvidia-smi; NVML answered"
    assert sample["gpu"] == pytest.approx(37.0)
    assert sample["vram"] == pytest.approx(25.0)
    assert 0 <= sample["ram"] <= 100
    assert ("handle", 0) in nvml.asked


def test_a_driver_listing_no_card_reads_as_idle(monkeypatch, nvidia_host,
                                                gputil):
    nvml = _fake_nvml(cards=0)
    _use_nvml(monkeypatch, nvml)

    sample = nvidia_host._sample_usage(False)

    assert (sample["gpu"], sample["vram"]) == (0, 0)
    assert gputil == []
    assert not any(isinstance(step, tuple) for step in nvml.asked)


def test_a_card_reporting_no_memory_reads_zero_rather_than_dividing(
        monkeypatch, nvidia_host):
    _use_nvml(monkeypatch, _fake_nvml(load=5, used=0, total=0))

    assert nvidia_host._nvml_gpu_usage() == (5.0, 0.0)


def test_a_failing_nvml_query_falls_back_to_gputil(monkeypatch, nvidia_host,
                                                   gputil):
    _use_nvml(monkeypatch, _fake_nvml(broken=True))

    sample = nvidia_host._sample_usage(False)

    assert gputil == [1]
    assert sample["gpu"] == pytest.approx(50.0)
    assert sample["vram"] == pytest.approx(25.0)


def test_a_host_without_nvml_still_has_gputil(monkeypatch, nvidia_host,
                                              gputil):
    _use_nvml(monkeypatch, None)

    sample = nvidia_host._sample_usage(False)

    assert gputil == [1]
    assert sample["gpu"] == pytest.approx(50.0)


def test_a_cpu_only_host_never_opens_nvml(monkeypatch, gputil):
    from spacr.qt.screens import app_screen

    monkeypatch.setattr(app_screen, "_nvidia_smi_available", lambda: False)
    monkeypatch.setattr(app_screen, "_nvml_gpu_usage",
                        lambda: pytest.fail("NVML asked on a CPU-only host"))

    sample = app_screen._sample_usage(False)

    assert (sample["gpu"], sample["vram"]) == (0, 0)
    assert gputil == []


# --------------------------------------------------------------------------
# the bars
# --------------------------------------------------------------------------

class _CountingStyle:
    """Stands in for the bar's style and counts polishes."""

    def __init__(self):
        self.polished = 0

    def unpolish(self, _widget):
        pass

    def polish(self, _widget):
        self.polished += 1


def test_a_reading_in_the_same_band_does_not_repolish_the_bar(qtbot):
    from spacr.qt.widgets.usage_bar import UsageBar

    bar = UsageBar("RAM")
    qtbot.addWidget(bar)
    style = _CountingStyle()
    bar._bar.style = lambda: style

    for value in (10, 37, 52, 74):
        bar.set_value(value)
    assert style.polished == 0
    assert bar._pct.text() == "74%"
    assert bar._bar.value() == 74

    bar.set_value(80)
    bar.set_value(85)
    assert style.polished == 1
    assert bar._bar.objectName() == "UsageBarWarn"

    bar.set_value(95)
    bar.set_value(20)
    assert style.polished == 3
    assert bar._bar.objectName() == "UsageBar"


def test_a_new_reading_is_painted_with_the_next_frame_not_now(qtbot):
    """``QProgressBar.setValue`` repaints synchronously; the bar must not.

    A synchronous repaint of a translucent bar paints the field under it
    and flushes the window at once, outside the field's own frames.
    """
    from PySide6.QtCore import QEvent, QObject

    from spacr.qt.widgets.usage_bar import UsageBar

    class _Paints(QObject):
        count = 0

        def eventFilter(self, _watched, event):
            if event.type() == QEvent.Paint:
                self.count += 1
            return False

    bar = UsageBar("CPU")
    qtbot.addWidget(bar)
    bar.resize(300, 24)
    bar.show()
    qtbot.waitExposed(bar)
    qtbot.wait(20)
    paints = _Paints()
    bar._bar.installEventFilter(paints)

    for value in (12, 48, 70, 91, 3):
        bar.set_value(value)

    assert paints.count == 0, "set_value painted the bar synchronously"
    assert bar._bar.updatesEnabled()
    qtbot.waitUntil(lambda: paints.count > 0, timeout=2000)
    assert bar._bar.value() == 3


# --------------------------------------------------------------------------
# the real window
# --------------------------------------------------------------------------

def _hide_tours(window) -> None:
    from PySide6.QtWidgets import QWidget

    for child in window.findChildren(QWidget):
        if type(child).__name__ == "_TourOverlay":
            child.hide()


@pytest.fixture
def applied(qapp):
    """The preferences applied the way the application applies them."""
    from spacr.qt.preferences import apply_preferences_to_app
    from spacr.qt.widgets.field_fade import (invalidate_field_fade,
                                             uninstall_field_fade)

    sheet, palette, font = qapp.styleSheet(), qapp.palette(), qapp.font()
    apply_preferences_to_app(qapp)
    yield qapp
    uninstall_field_fade(qapp)
    invalidate_field_fade()
    qapp.setFont(font)
    qapp.setPalette(palette)
    qapp.setStyleSheet(sheet)


def _window_on_mask(qtbot, size):
    from spacr.qt.app import MainWindow
    from spacr.qt.screens.app_screen import AppScreen

    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(*size)
    window.show()
    qtbot.waitExposed(window)
    window._on_nav_selected("mask")
    qtbot.wait(300)
    _hide_tours(window)
    screen = next(s for s in window.findChildren(AppScreen)
                  if s.isVisible() and s.app_key == "mask")
    return window, screen


def _stop_the_field(window):
    backdrop = window.window_backdrop()
    if backdrop is not None and hasattr(backdrop, "_timer"):
        backdrop._timer.stop()
    return backdrop


def _measure_a_window(qtbot, screen, events, spent):
    """Run the 2 ms heartbeat for :data:`SECONDS` and judge the ticks in it.

    :returns: ``(tick gaps, bound, ordinary p99)`` in ms. A gap is the
        timer's when a usage tick or its delivery happened inside it; the
        bound is :data:`STALL_BUDGET_MS`, or :data:`LOAD_FACTOR` times the
        99th percentile of the gaps no tick was in, whichever is larger.
    """
    from PySide6.QtCore import QTimer

    del events[:]
    del spent[:]
    gaps = []
    last = [time.perf_counter()]

    def beat():
        now = time.perf_counter()
        gaps.append((last[0], now))
        last[0] = now

    heartbeat = QTimer()
    heartbeat.setInterval(2)
    heartbeat.timeout.connect(beat)
    heartbeat.start()
    try:
        qtbot.wait(int(SECONDS * 1000))
    finally:
        heartbeat.stop()
    assert len(events) >= 4, "the usage poll did not tick on Mask"
    timer_gaps, other_gaps = [], []
    for start, end in gaps[1:]:
        ms = (end - start) * 1000.0
        hit = any(start <= at <= end for at in events)
        (timer_gaps if hit else other_gaps).append(ms)
    assert timer_gaps and len(other_gaps) > 100
    ordinary = statistics.quantiles(other_gaps, n=100)[98]
    return timer_gaps, max(STALL_BUDGET_MS, LOAD_FACTOR * ordinary), ordinary


def test_no_periodic_timer_holds_the_gui_thread_on_the_mask_screen(
        qtbot, applied, monkeypatch, started_processes):
    """A 2 ms heartbeat over 6.5 s on Mask; the gaps ticks fall in are short.

    The field's own frames are stopped, so what the heartbeat sees is the
    event loop and the timers. The bound is calibrated against the same
    window's gaps that no tick fell in (see :func:`_measure_a_window`).

    Load-robust without proving less: a tick that holds the GUI thread
    holds it on every tick, so the MEDIAN tick gap and the median handler
    CPU must stay inside the bound in every window, and the WORST tick gap
    must stay inside it in at least one of up to :data:`WINDOWS` windows.
    Under ``-n 4`` at load 29 a single gap was once caught by unrelated
    work landing on a tick; the synchronous repaints this guards against
    put 8.5 ms on most ticks of every window.
    """
    _use_nvml(monkeypatch, _fake_nvml())
    window, screen = _window_on_mask(qtbot, (1600, 1000))
    _stop_the_field(window)
    qtbot.wait(SETTLE_MS)
    qtbot.waitUntil(lambda: screen.active_jobs() == 0, timeout=20000)
    assert screen._usage_timer.isActive()
    assert screen._usage_timer.interval() == 2000

    events = []
    spent = []
    refresh, apply_usage = screen._refresh_usage, screen._apply_usage

    def timed_refresh():
        events.append(time.perf_counter())
        started = time.thread_time()
        refresh()
        spent.append((time.thread_time() - started) * 1000.0)

    def timed_apply(*args, **kwargs):
        events.append(time.perf_counter())
        started = time.thread_time()
        apply_usage(*args, **kwargs)
        spent.append((time.thread_time() - started) * 1000.0)

    screen._usage_timer.timeout.disconnect()
    screen._usage_timer.timeout.connect(timed_refresh)
    screen._apply_usage = timed_apply

    windows = []
    try:
        for _window in range(WINDOWS):
            tick_gaps, bound, ordinary = _measure_a_window(
                qtbot, screen, events, spent)
            typical = statistics.median(tick_gaps)
            assert typical <= bound, (
                f"usage ticks typically held the GUI thread {typical:.1f} ms "
                f"(bound {bound:.1f} ms; ordinary gaps p99 {ordinary:.1f} ms)")
            cpu = statistics.median(spent)
            assert cpu <= bound, (
                f"a usage handler typically spent {cpu:.1f} ms of GUI CPU")
            windows.append((max(tick_gaps), bound, ordinary))
            if max(tick_gaps) <= bound:
                break
    finally:
        screen._usage_timer.stop()
        qtbot.waitUntil(lambda: screen.active_jobs() == 0, timeout=20000)

    smi = [args for args in started_processes if "nvidia-smi" in str(args)]
    assert smi == [], "the usage poll started nvidia-smi"
    assert any(worst <= bound for worst, bound, _ in windows), (
        "in every window a usage tick held the GUI thread past the bound: "
        + "; ".join(f"{worst:.1f} ms against {bound:.1f} "
                    f"(ordinary p99 {ordinary:.1f})"
                    for worst, bound, ordinary in windows))


def _region_ms(backdrop, region, frames=12) -> float:
    """Cheapest GUI-thread CPU of a synchronous field repaint of ``region``."""
    spent = []
    for _frame in range(frames):
        started = time.thread_time()
        backdrop.repaint(region)
        spent.append((time.thread_time() - started) * 1000.0)
    return min(spent)


def test_the_mask_preview_pictures_are_cached_at_the_size_shown(
        qtbot, applied):
    """A field frame over two 2048-pixel previews blits them, not rescales them.

    The field repaints the region of the two preview canvases, which is
    what each field frame does there. The reference is the same repaint
    with the pictures drawn uncached, measured interleaved in the same
    test. Offscreen at 2560x1440, load 24-28, median of nine rounds: 3.1 and
    4.3 ms uncached, 1.4 and 2.0 cached, 0.9 and 1.3 with no picture.
    """
    from PySide6.QtCore import QPoint, QRect
    from PySide6.QtGui import QRegion
    from PySide6.QtWidgets import QGraphicsItem

    from spacr.qt.widgets.live_preview import LivePreviewPanel, numpy_to_qpixmap

    window, screen = _window_on_mask(qtbot, (2560, 1440))
    qtbot.waitUntil(lambda: window.window_backdrop() is not None,
                    timeout=20000)
    backdrop = window.window_backdrop()
    if not hasattr(backdrop, "_timer"):
        pytest.skip("the window backdrop is not the field in this setup")
    screen._on_preview_switch(True)
    qtbot.wait(300)
    _hide_tours(window)
    _stop_the_field(window)
    panel = window.findChildren(LivePreviewPanel)[0]
    picture = (np.random.default_rng(0).random((2048, 2048, 3))
               * 255).astype(np.uint8)
    views = (panel._src_view, panel._mask_view)
    for view, image in zip(views, (picture, picture[::-1])):
        view.set_pixmap(numpy_to_qpixmap(image))
        assert (view._pixmap_item.cacheMode()
                == QGraphicsItem.DeviceCoordinateCache)
    qtbot.wait(50)
    region = QRegion()
    for view in views:
        corner = backdrop.mapFrom(
            window, view.viewport().mapTo(window, QPoint(0, 0)))
        region = region.united(QRect(corner, view.viewport().size()))
    assert not region.isEmpty()

    cached, uncached = [], []
    for _round in range(9):
        for mode, out in ((QGraphicsItem.DeviceCoordinateCache, cached),
                          (QGraphicsItem.NoCache, uncached)):
            for view in views:
                view._pixmap_item.setCacheMode(mode)
            backdrop.repaint(region)
            out.append(_region_ms(backdrop, region))
    ratio = statistics.median(cached) / statistics.median(uncached)
    from PySide6.QtGui import QPixmapCache
    from spacr.qt.gui_scale import current_scale

    diagnostics = {
        "native_window": window.geometry().getRect(),
        "work_area": window.screen().availableGeometry().getRect(),
        "gui_scale": current_scale(),
        "region": region.boundingRect().getRect(),
        "region_rectangles": region.rectCount(),
        "pixmap_cache_kib": QPixmapCache.cacheLimit(),
        "views": [],
        "cached_ms": cached,
        "uncached_ms": uncached,
    }
    for view in views:
        viewport = view.viewport()
        transform = view.viewportTransform()
        pixmap = view._pixmap_item.pixmap()
        diagnostics["views"].append({
            "visible": view.isVisible(),
            "native_viewport": viewport.geometry().getRect(),
            "visible_region": viewport.visibleRegion().boundingRect().getRect(),
            "transform": (transform.m11(), transform.m12(), transform.m21(),
                          transform.m22(), transform.dx(), transform.dy()),
            "viewport_dpr": viewport.devicePixelRatioF(),
            "pixmap_size": (pixmap.width(), pixmap.height()),
            "pixmap_dpr": pixmap.devicePixelRatio(),
        })
    print("preview cache diagnostics:", diagnostics, flush=True)
    assert ratio <= 0.75, (
        f"the cached previews cost {ratio:.2f} of drawing them scaled "
        f"({statistics.median(cached):.2f} against "
        f"{statistics.median(uncached):.2f} ms a repaint); {diagnostics!r}")

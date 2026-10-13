"""A figure too slow to draw stays a raster instead of a GUI-thread canvas.

The tutorial capture of lesson 08 (clip batch_20_batch_settings, Measure)
froze for about a second twice while the run streamed its figures. Measured
offscreen on 2026-10-13 at 2560x1440 with a real Measure run, a 2 ms
heartbeat and a sampler of the GUI thread's stack: each figure that arrived
was put on the live Matplotlib canvas, and the canvas drew it on the GUI
thread on its first paint -- 691-937 ms for the three crop figures (7 and 8
panels of 2000 x 2000 crops on a 70 x 70 inch page, then a 72-panel
montage). The run's worker had already drawn the same figures as display
rasters, in 1.4-3.2 s.

:func:`~spacr.qt.widgets.figure_queue.render_figure_to_png` now records how
long that raster took, and :meth:`FigureQueue.show_live_canvas` leaves a
figure whose raster took longer than ``_LIVE_CANVAS_MAX_RASTER_S`` on the
raster already on disk. An ordinary plot keeps the live canvas.

The block test draws a heavy figure the old way, on the canvas, and compares
that with what showing it costs now, in the same process.
"""
from __future__ import annotations

import threading
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
import numpy as np                                                  # noqa: E402
import pytest                                                       # noqa: E402

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtWidgets import QApplication                          # noqa: E402

from spacr.qt.widgets import figure_queue as fq                     # noqa: E402
from spacr.qt.widgets.figure_queue import FigureQueue               # noqa: E402

pytestmark = pytest.mark.qt


def _heavy_figure():
    """Eight large image panels on a big page, like Measure's crop figures."""
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 8, figsize=(48, 7))
    for ax in axes:
        ax.imshow(rng.random((1500, 1500)), cmap="gray")
        ax.set_title("crop")
    return fig


def _light_figure():
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([0, 1, 2], [1, 3, 2])
    return fig


def _render_on_a_worker(fig, path):
    """Render the display raster the way the run's bridge does: off-thread."""
    done = []
    thread = threading.Thread(
        target=lambda: done.append(fq.render_figure_to_png(fig, str(path), write_pdf=False)))
    thread.start()
    thread.join()
    assert done == [True]
    return str(path)


@pytest.fixture
def queue(qtbot):
    widget = FigureQueue()
    qtbot.addWidget(widget)
    widget.resize(2200, 1000)
    widget.show()
    qtbot.waitExposed(widget)
    return widget


def _settle(seconds=0.3):
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        QApplication.processEvents()


def test_the_raster_time_is_recorded(tmp_path):
    fig = _light_figure()
    try:
        _render_on_a_worker(fig, tmp_path / "light.png")
        seconds = getattr(fig, fq._RASTER_SECONDS_ATTR)
        assert 0 < seconds < 60
    finally:
        plt.close(fig)


def test_a_light_figure_keeps_the_live_canvas(queue, tmp_path):
    fig = _light_figure()
    try:
        png = _render_on_a_worker(fig, tmp_path / "light.png")
        assert not fq._too_slow_for_the_live_canvas(fig)
        queue.add_figure(fig, prerendered_png=png)
        _settle()
        assert queue._canvas is not None and queue._canvas.figure is fig
        assert queue._stack.currentIndex() == 1
    finally:
        plt.close(fig)


def test_a_heavy_figure_is_shown_as_its_raster_without_a_gui_draw(queue, tmp_path):
    """Self-calibrated: showing it now against drawing it on the canvas."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig = _heavy_figure()
    try:
        png = _render_on_a_worker(fig, tmp_path / "heavy.png")
        if not fq._too_slow_for_the_live_canvas(fig):
            pytest.skip("this machine rasters the heavy figure under the limit")
        draws = []
        original = FigureCanvasAgg.draw

        def counted(canvas):
            draws.append(threading.current_thread() is threading.main_thread())
            return original(canvas)

        FigureCanvasAgg.draw = counted
        try:
            started = time.perf_counter()
            index = queue.add_figure(fig, prerendered_png=png)
            QApplication.processEvents()
            shown = time.perf_counter() - started
            _settle()
        finally:
            FigureCanvasAgg.draw = original
        assert True not in draws, "the GUI thread drew the heavy figure"
        assert queue._canvas is None and queue._stack.currentIndex() == 0
        assert queue.has_live_figure(index)
        assert not queue._view._pixmap_item.pixmap().isNull()

        width, height = 2216, 975
        dpi = fig.get_dpi()
        size = fig.get_size_inches().copy()
        fig.set_size_inches(width / dpi, height / dpi)
        try:
            started = time.perf_counter()
            FigureCanvasAgg(fig).draw()
            canvas_draw = time.perf_counter() - started
        finally:
            fig.set_size_inches(*size)
        assert shown * 3 < canvas_draw, (shown, canvas_draw)
    finally:
        plt.close(fig)


def test_the_raster_shown_is_the_one_rendered(queue, tmp_path):
    """Same picture, same slot, same gallery as before the gate."""
    from PySide6.QtGui import QImage

    fig = _light_figure()
    try:
        png = _render_on_a_worker(fig, tmp_path / "light.png")
        expected = QImage(png)
        setattr(fig, fq._RASTER_SECONDS_ATTR, fq._LIVE_CANVAS_MAX_RASTER_S + 1)
        index = queue.add_figure(fig, prerendered_png=png)
        assert index == 0 and queue._count == 1
        assert queue._canvas is None and queue._stack.currentIndex() == 0
        assert QImage(queue._png_paths[index]) == expected
        shown = queue._view._pixmap_item.pixmap().toImage()
        premultiplied = QImage.Format.Format_ARGB32_Premultiplied
        assert (shown.convertToFormat(premultiplied)
                == expected.convertToFormat(premultiplied))
        assert queue.figure_for(index) is fig
        _settle()
        assert queue._canvas is None and queue.figure_for(index) is fig
    finally:
        plt.close(fig)


def test_a_canvas_whose_figure_became_slow_falls_back_to_the_raster(queue, tmp_path):
    fig = _light_figure()
    try:
        png = _render_on_a_worker(fig, tmp_path / "light.png")
        queue.add_figure(fig, prerendered_png=png)
        _settle()
        assert queue._canvas is not None
        setattr(fig, fq._RASTER_SECONDS_ATTR, fq._LIVE_CANVAS_MAX_RASTER_S + 1)
        assert queue.show_live_canvas(fig) is False
        assert queue._canvas is None and queue._stack.currentIndex() == 0
    finally:
        plt.close(fig)


def test_a_figure_never_rastered_or_badly_marked_is_not_judged_slow():
    class Plain:
        pass

    plain = Plain()
    assert not fq._too_slow_for_the_live_canvas(plain)
    plain._spacr_raster_seconds = "not a number"
    assert not fq._too_slow_for_the_live_canvas(plain)


def test_noting_the_time_on_something_that_takes_no_attributes_is_harmless():
    class Sealed:
        __slots__ = ()

    fq._note_raster_seconds(Sealed(), 1.0)

"""A figure may be restyled while a run is drawing one. Instruction 166.

REPORTED 2026-08-19: "i was changing the background graph color of the first
graph while a nonparametric statmodels mixed model was being trained and spacr
crashed."

matplotlib is NOT thread-safe. `bridge._capture_show` renders each open figure
to a PNG ON THE WORKER THREAD while a run streams them, and the style dialog
recolours a figure ON THE GUI THREAD -- so a colour changed mid-fit had two
threads mutating one Figure, and the crash was a segfault in the C layer with
no Python exception and nothing logged.
"""

import threading
import types

import spacr


import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spacr.qt.widgets.figure_queue import (FIGURE_LOCK, _style_figure_colors,
                                           render_figure_to_png)


def test_both_sides_hold_the_lock():
    """A lock one side forgets is not a lock."""
    import inspect

    for fn in (render_figure_to_png, _style_figure_colors):
        assert "with FIGURE_LOCK" in inspect.getsource(fn), fn.__name__


def test_the_lock_is_reentrant():
    """The render path styles before it renders, so a plain Lock would
    deadlock the first time one call did both."""
    with FIGURE_LOCK:
        with FIGURE_LOCK:
            pass
    assert FIGURE_LOCK.acquire(blocking=False)
    FIGURE_LOCK.release()


def test_rendering_and_restyling_the_same_figure_concurrently(tmp_path):
    """THE REPORTED CRASH, driven.

    Without the lock this is a data race on a matplotlib Figure. It will not
    segfault on every run -- a race never does -- so the value here is that it
    exercises the exact pair of calls, repeatedly, on one Figure.
    """
    fig = plt.figure()
    fig.add_subplot(111).plot([0, 1], [0, 1])
    try:
        errors = []
        stop = threading.Event()

        def render():
            i = 0
            while not stop.is_set() and i < 12:
                try:
                    render_figure_to_png(fig, str(tmp_path / f"r{i}.png"))
                except Exception as error:            # noqa: BLE001
                    errors.append(error)
                i += 1

        worker = threading.Thread(target=render)
        worker.start()
        try:
            for shade in ("#ffffff", "#000000") * 6:
                try:
                    _style_figure_colors(fig, shade, "#333333")
                except Exception as error:            # noqa: BLE001
                    errors.append(error)
        finally:
            stop.set()
            worker.join(30)
        assert not errors, errors
    finally:
        plt.close(fig)


def test_a_restyle_still_takes_effect(tmp_path):
    """Serialised, not dropped. A colour the user asked for must land."""
    fig = plt.figure()
    axis = fig.add_subplot(111)
    axis.plot([0, 1], [0, 1])
    try:
        _style_figure_colors(fig, "#123456", "#abcdef")
        assert fig.get_facecolor()[:3] == pytest.approx(
            (0x12 / 255, 0x34 / 255, 0x56 / 255), abs=0.01)
    finally:
        plt.close(fig)


def _hold_the_lock():
    """Hold FIGURE_LOCK on another thread, the way a worker render does.

    :returns: (release, released_at_start) — call ``release`` to let go.
    """
    holding = threading.Event()
    let_go = threading.Event()

    def hold():
        with FIGURE_LOCK:
            holding.set()
            let_go.wait(10)

    thread = threading.Thread(target=hold)
    thread.start()
    assert holding.wait(10), "the holder never took the lock"

    def release():
        let_go.set()
        thread.join(10)
        assert not thread.is_alive()

    return release


class _FigureThatRecordsItsDraws:
    """The parts of a Figure the preview render touches, and nothing else.

    ``savefig`` raises so the render bails out before it reaches ``QPixmap``:
    what is under test is WHEN the draw is allowed to start, and building a
    pixmap would drag a QApplication into a test that does not need one.
    """

    def __init__(self):
        self.drawn = threading.Event()

    def get_size_inches(self):
        return (6.0, 4.0)

    def get_facecolor(self):
        return "#ffffff"

    def savefig(self, *args, **kwargs):
        self.drawn.set()
        raise RuntimeError("far enough: the draw was allowed to begin")


@pytest.fixture
def native_preview(qtbot):
    """A real raster-backed queue with one editable, named Figure."""
    from matplotlib.figure import Figure
    from spacr.qt.widgets.figure_queue import FigureQueue

    queue = FigureQueue()
    qtbot.addWidget(queue)
    queue.set_live_canvas_enabled(False)
    figure = Figure(figsize=(2, 2), label="locked-preview")
    queue.add_figure(figure)
    live = queue.figure_for(0)
    return queue, live


@pytest.mark.qt
def test_the_preview_draw_holds_the_lock_too(qtbot, native_preview, monkeypatch):
    """A real snapshot excludes competing workers throughout serialization."""
    import pickle

    queue, live = native_preview
    live.set_facecolor("#345678")
    competing_acquisitions = []
    original_dumps = pickle.dumps

    def observe_copy(value, *args, **kwargs):
        if value is live:
            def compete():
                acquired = FIGURE_LOCK.acquire(blocking=False)
                competing_acquisitions.append(acquired)
                if acquired:
                    FIGURE_LOCK.release()
            contender = threading.Thread(target=compete)
            contender.start()
            contender.join(2)
        return original_dumps(value, *args, **kwargs)

    monkeypatch.setattr(pickle, "dumps", observe_copy)
    assert queue.refresh_current_figure(preview=True)
    qtbot.waitUntil(lambda: not queue._preview_busy, timeout=10000)
    assert competing_acquisitions == [False]
    assert queue._preview_seq == 1
    picture = queue._view.picture()
    assert picture is not None and not picture.isNull()
    assert picture.toImage().pixelColor(0, 0).name() == "#345678"


def test_a_preview_render_waits_for_a_worker_render(tmp_path):
    """Honoured at run time, not merely written in the source."""
    from spacr.qt.widgets.figure_queue import FigureQueue

    fig = _FigureThatRecordsItsDraws()
    render = FigureQueue._render_preview.__get__(
        types.SimpleNamespace(PREVIEW_MAX_PX=FigureQueue.PREVIEW_MAX_PX),
        FigureQueue)

    release = _hold_the_lock()
    caller = threading.Thread(target=render, args=(fig, tmp_path / "p.png"))
    caller.start()
    try:
        assert not fig.drawn.wait(0.4), (
            "the preview drew while another thread held the figure")
    finally:
        release()
    caller.join(10)
    assert fig.drawn.is_set(), "the preview never drew after the lock was free"


@pytest.mark.qt
def test_copying_a_figure_for_the_preview_worker_waits_too(
        qtbot, native_preview, monkeypatch):
    """Snapshotting waits for ownership while Qt events and later requests proceed."""
    import pickle
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from PySide6.QtCore import QTimer

    queue, live = native_preview
    gui_thread = threading.get_ident()
    copied, drawn, pulses = [], [], []
    original_dumps, original_draw = pickle.dumps, FigureCanvasAgg.draw

    def observe_copy(value, *args, **kwargs):
        if value is live:
            copied.append(threading.get_ident())
        return original_dumps(value, *args, **kwargs)

    def observe_draw(canvas, *args, **kwargs):
        if canvas.figure.get_label() == "locked-preview":
            drawn.append(threading.get_ident())
        return original_draw(canvas, *args, **kwargs)

    monkeypatch.setattr(pickle, "dumps", observe_copy)
    monkeypatch.setattr(FigureCanvasAgg, "draw", observe_draw)
    release = _hold_the_lock()
    try:
        live.set_facecolor("#123456")
        assert queue.refresh_current_figure(preview=True)
        live.set_facecolor("#2468ac")
        assert queue.refresh_current_figure(preview=True)
        QTimer.singleShot(0, lambda: pulses.append(threading.get_ident()))
        qtbot.waitUntil(lambda: bool(pulses), timeout=1000)
        assert pulses == [gui_thread]
        assert copied == [] and drawn == []
        assert queue._preview_seq == 0
    finally:
        release()
    qtbot.waitUntil(lambda: not queue._preview_busy, timeout=10000)
    assert copied == [gui_thread]
    assert drawn and all(thread != gui_thread for thread in drawn)
    assert queue._preview_seq == 1
    picture = queue._view.picture()
    assert picture is not None and not picture.isNull()
    assert picture.toImage().pixelColor(0, 0).name() == "#2468ac"
    queue.close()
    assert not queue._preview_busy and not queue._preview_pending
    assert not queue._jobs.is_busy()

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


def test_the_preview_draw_holds_the_lock_too():
    """The style dialog DRAWS while a control moves, not only recolours.

    `_render_preview` rasterises the figure and `_render_preview_async`
    pickles it, both on the GUI thread and both over the same Figure a run's
    worker may be rendering — `bridge._capture_show` re-renders any figure
    marked `_spacr_live_update` for as long as the fit lasts. A draw racing a
    draw is the same C-layer race as a draw racing a recolour.
    """
    import inspect

    from spacr.qt.widgets.figure_queue import FigureQueue

    for fn in (FigureQueue._render_preview, FigureQueue._render_preview_async):
        assert "with FIGURE_LOCK" in inspect.getsource(fn), fn.__name__


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


def test_copying_a_figure_for_the_preview_worker_waits_too(tmp_path):
    """`pickle.dumps` walks the whole Figure, so it is a draw-shaped read."""
    from spacr.qt.widgets.figure_queue import FigureQueue

    submitted = threading.Event()
    queue = types.SimpleNamespace(
        PREVIEW_MAX_PX=FigureQueue.PREVIEW_MAX_PX,
        _preview_busy=False,
        _preview_pending=False,
        _preview_seq=0,
        _current=0,
        _preview_target_px=lambda: 800.0,
        _on_preview_rendered=lambda payload: None,
        _jobs=types.SimpleNamespace(
            submit=lambda work, done: submitted.set()),
    )
    start = FigureQueue._render_preview_async.__get__(queue, FigureQueue)

    fig = plt.figure()
    fig.add_subplot(111).plot([0, 1], [0, 1])
    try:
        release = _hold_the_lock()
        caller = threading.Thread(target=start, args=(fig,))
        caller.start()
        try:
            assert not submitted.wait(0.4), (
                "the figure was copied while another thread held it")
        finally:
            release()
        caller.join(10)
        assert submitted.is_set(), "the copy never happened once the lock was free"
    finally:
        plt.close(fig)

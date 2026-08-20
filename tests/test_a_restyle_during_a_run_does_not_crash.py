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

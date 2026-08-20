"""Instruction 126 — "the theme starts lagging" while a run is going.

MEASURED BEFORE ANYTHING WAS WRITTEN, because the instruction says to and
because its three candidate causes want three different fixes. Offscreen,
1280x800, `blobs` with its shading already on `ambient._FrameProducer`, a
16 ms timer, two worker threads:

    idle                                 median 16.00 ms   p95  16.04 ms
    numpy worker (what a run mostly is)  median 16.00 ms   p95  20.05 ms
    pure-Python worker                   median 42.42 ms   p95 118.63 ms
    pure-Python, switchinterval 0.001    median 17.74 ms   p95  48.46 ms

So the producer thread had already fixed the common case; what was left was
cause 1, the interpreter lock, against which -- exactly as the instruction
predicted -- moving the shading to another Python thread does nothing.

These tests hold the CONTRACT (claimed for the run, restored after, nesting)
rather than the milliseconds. A timing assertion on a shared CI machine is a
test that fails for reasons that have nothing to do with the code; the
numbers above are the evidence and they are written down where they were
taken.
"""
from __future__ import annotations

import os
import sys
import threading

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from spacr.qt.gil_priority import (BUSY_INTERVAL, active,  # noqa: E402
                                   claim, release, responsive_gui)


@pytest.fixture(autouse=True)
def _restore():
    before = sys.getswitchinterval()
    while active():
        release()
    yield
    while active():
        release()
    sys.setswitchinterval(before)


def test_a_running_worker_asks_for_the_lock_more_often():
    before = sys.getswitchinterval()
    with responsive_gui():
        assert sys.getswitchinterval() == BUSY_INTERVAL
    assert sys.getswitchinterval() == before


def test_it_is_given_back_when_the_run_raises():
    """Otherwise the process pays 1 ms for as long as it lives."""
    before = sys.getswitchinterval()
    with pytest.raises(RuntimeError):
        with responsive_gui():
            raise RuntimeError("the design is not identifiable")
    assert sys.getswitchinterval() == before


def test_two_modules_running_at_once_do_not_hand_it_back_early():
    """Mask and Measure together: the first to finish must not undo it."""
    before = sys.getswitchinterval()
    claim()
    claim()
    release()
    assert sys.getswitchinterval() == BUSY_INTERVAL, "the second run lost it"
    release()
    assert sys.getswitchinterval() == before


def test_releasing_more_than_was_claimed_does_not_go_negative():
    before = sys.getswitchinterval()
    release()
    release()
    assert active() is False
    with responsive_gui():
        assert sys.getswitchinterval() == BUSY_INTERVAL
    assert sys.getswitchinterval() == before


def test_it_is_claimed_from_several_threads_without_losing_count():
    started = threading.Barrier(5)

    def worker():
        started.wait()
        with responsive_gui():
            pass

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert active() is False


def test_the_pipeline_worker_holds_it_for_the_length_of_the_run():
    """The claim is on the RUN, not on the application."""
    pytest.importorskip("PySide6")
    import inspect

    from spacr.qt import bridge

    source = inspect.getsource(bridge.PipelineWorker.run)
    assert "responsive_gui()" in source


def test_a_headless_run_pays_nothing_for_a_window_that_is_not_there():
    """A process-wide 1 ms would tax every `spacr-run` in the interpreter."""
    before = sys.getswitchinterval()
    import spacr.qt.gil_priority          # noqa: F401  (importing is the test)
    assert sys.getswitchinterval() == before


def test_the_backdrop_shades_on_its_own_thread_when_it_has_a_frame_to_hand():
    """The half that was already built, and the half that fixes numpy runs."""
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from spacr.qt.widgets.ambient import AmbientWidget, _BufferedEngine

    app = QApplication.instance() or QApplication([])
    widget = AmbientWidget()
    widget.resize(320, 240)
    widget.show()
    widget.start()
    try:
        if isinstance(widget._engine, _BufferedEngine):
            assert widget._producer_box[0] is not None
        widget.stop()
        # And it costs nothing while stopped: no thread, no frame held.
        assert widget._producer_box[0] is None
    finally:
        widget.stop()
        widget.deleteLater()
        app.processEvents()

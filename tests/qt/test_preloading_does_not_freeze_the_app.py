"""The module preload runs off the GUI thread.

Measured before any change: importing the seven pipeline modules on the GUI
thread, yielding between each, left the interface answering THREE timer
ticks in 4.24 seconds, with one 1408 ms freeze. 94% of that is the two
modules that pull torch, and no amount of yielding breaks up a single
2.3-second import.
"""

import threading
import time

import pytest

pytest.importorskip("PySide6")

from spacr.qt.app import HEAVY_IMPORT_LOCK, _PipelinePreloader  # noqa: E402


def test_it_imports_on_a_worker_thread():
    import inspect

    body = inspect.getsource(_PipelinePreloader.start)
    assert "threading.Thread" in body


@pytest.mark.timing
def test_the_gui_keeps_answering_while_it_preloads(qtbot):
    """The measurement that justifies the change, as a test.

    MARKED `timing` AND EXCLUDED FROM THE PARALLEL SWEEP. It measures how
    often the GUI thread answers, and under `-n 4` it is competing with three
    other pytest workers for the same cores -- so it fails on a machine where
    the application would be perfectly responsive. A timing assertion that
    fails for the wrong reason teaches nobody anything.

    Run it on its own: `pytest -m timing`. The bar is loose even then -- 3
    ticks and a 1.4-second freeze, which is what the GUI-thread version gave,
    fail it by a mile.
    """
    import os

    if os.environ.get("PYTEST_XDIST_WORKER"):
        pytest.skip("a timing measurement cannot share the machine")
    from PySide6.QtCore import QEventLoop, QTimer

    ticks = []
    beat = QTimer()
    beat.setInterval(16)
    beat.timeout.connect(lambda: ticks.append(time.perf_counter()))
    beat.start()

    loop = QEventLoop()
    preloader = _PipelinePreloader(on_done=loop.quit)
    preloader.start()
    QTimer.singleShot(60_000, loop.quit)
    loop.exec()
    beat.stop()

    assert preloader.wait(30.0), "the preload never finished"
    assert len(ticks) > 40, f"the GUI answered only {len(ticks)} times"
    worst = max((b - a) for a, b in zip(ticks, ticks[1:]))
    assert worst < 1.0, f"froze for {worst*1000:.0f} ms"


def test_progress_is_reported_on_the_gui_thread(qtbot):
    """A worker that called back directly would be touching a loading
    screen from off the GUI thread."""
    from PySide6.QtCore import QEventLoop, QTimer

    gui_thread = threading.current_thread()
    seen = []

    loop = QEventLoop()
    preloader = _PipelinePreloader(
        on_step=lambda done, total: seen.append(threading.current_thread()),
        on_done=loop.quit)
    preloader.start()
    QTimer.singleShot(60_000, loop.quit)
    loop.exec()

    assert seen, "no progress was reported at all"
    assert all(thread is gui_thread for thread in seen)


def test_every_module_is_reported_once(qtbot):
    from PySide6.QtCore import QEventLoop, QTimer

    steps = []
    loop = QEventLoop()
    preloader = _PipelinePreloader(
        on_step=lambda done, total: steps.append((done, total)),
        on_done=loop.quit)
    preloader.start()
    QTimer.singleShot(60_000, loop.quit)
    loop.exec()

    assert [d for d, _t in steps] == list(range(1, preloader.total() + 1))
    assert all(total == preloader.total() for _d, total in steps)


def test_done_fires_once(qtbot):
    from PySide6.QtCore import QEventLoop, QTimer

    calls = []
    loop = QEventLoop()
    preloader = _PipelinePreloader(on_done=lambda: (calls.append(1),
                                                    loop.quit()))
    preloader.start()
    QTimer.singleShot(60_000, loop.quit)
    loop.exec()
    # Give the poll several more chances to fire a second time.
    for _ in range(5):
        preloader._drain()
    assert calls == [1]


def test_starting_twice_does_nothing(qtbot):
    preloader = _PipelinePreloader()
    preloader.start()
    first = preloader._thread
    preloader.start()
    assert preloader._thread is first


def test_a_module_that_will_not_import_does_not_stop_the_rest(qtbot,
                                                              monkeypatch):
    """Preloading is optional; one bad module must not strand the loader."""
    from PySide6.QtCore import QEventLoop, QTimer

    monkeypatch.setattr(_PipelinePreloader, "_MODULES",
                        ("spacr.measure", "spacr.not_a_real_module",
                         "spacr.ml"))
    loop = QEventLoop()
    preloader = _PipelinePreloader(on_done=loop.quit)
    preloader.start()
    QTimer.singleShot(30_000, loop.quit)
    loop.exec()
    assert preloader.wait(10.0)


# --- the hazard the old design existed to avoid ---------------------------


def test_the_heavy_lock_exists():
    """The old docstring stayed on the GUI thread to avoid "concurrent Qt,
    CUDA, and OpenGL initialization". That reason is answered, not ignored."""
    assert isinstance(HEAVY_IMPORT_LOCK, type(threading.Lock()))


def test_the_preloader_imports_under_it():
    import inspect

    assert "HEAVY_IMPORT_LOCK" in inspect.getsource(_PipelinePreloader._work)


def test_the_gl_canvas_is_built_under_it():
    """torch brings CUDA up and the fractal brings a GL context up; the two
    must take turns."""
    import inspect

    from spacr.qt.widgets import fractal_travel

    body = inspect.getsource(fractal_travel._make_gpu_widget)
    assert "_heavy_import_lock()" in body
    assert "with lock:" in body


def test_the_widget_still_builds_with_no_application_around_it():
    """The backdrop is usable on its own, so a missing lock is not fatal."""
    from spacr.qt.widgets import fractal_travel

    assert fractal_travel._heavy_import_lock() is not None

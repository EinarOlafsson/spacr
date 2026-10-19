"""QObject lifetimes: stale thread.finished, preloader poll timer, batch teardown."""
from __future__ import annotations

import gc
import threading
import time

import pytest
import shiboken6
from PySide6.QtCore import QCoreApplication, QThread, QTimer
from PySide6.QtWidgets import QApplication

pytestmark = pytest.mark.qt


def _pump(ms=300):
    end = time.monotonic() + ms / 1000.0
    while time.monotonic() < end:
        QCoreApplication.processEvents()
        QCoreApplication.sendPostedEvents(None, 0)
        time.sleep(0.005)


def _hold(release):
    def fn(_settings):
        while not release.is_set() and not QThread.currentThread().isInterruptionRequested():
            time.sleep(0.01)
    return fn


def test_a_stale_thread_finished_does_not_clear_the_next_run(qtbot, qt_theme_applied):
    from spacr.qt.bridge import make_thread, thread_has_stopped
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("mask")
    qtbot.addWidget(screen)
    t1, w1 = make_thread(lambda _s: None, {}, journal=False, capture_figures=False)
    t1.finished.connect(screen._clear_thread_refs)
    screen._thread, screen._worker = t1, w1
    t1.start()
    assert t1.wait(5000)

    release = threading.Event()
    t2, w2 = make_thread(_hold(release), {}, journal=False, capture_figures=False)
    t2.finished.connect(screen._clear_thread_refs)
    screen._thread, screen._worker = t2, w2
    t2.start()

    _pump(300)
    assert screen._thread is t2 and screen._worker is w2
    assert not thread_has_stopped(t2)

    screen.close()
    assert thread_has_stopped(t2)
    release.set()
    _pump(200)
    assert screen._thread is None


def test_the_preloader_poll_timer_is_released_from_the_event_loop(qtbot):
    from PySide6.QtCore import QEventLoop

    from spacr.qt.app import _PipelinePreloader

    loop = QEventLoop()
    preloader = _PipelinePreloader(on_done=loop.quit)
    gc.disable()
    try:
        preloader.start()
        poll = preloader._poll
        assert poll.parent() is not None or QApplication.instance() is None
        QTimer.singleShot(240_000, loop.quit)
        loop.exec()
        qtbot.waitUntil(lambda: preloader._poll is None, timeout=240000)
        assert preloader._poll is None
        assert not poll.isActive()
        _pump(200)
        import weakref
        ref = weakref.ref(poll)
        del poll
        qtbot.waitUntil(lambda: ref() is None or not shiboken6.isValid(ref()),
                        timeout=30000)
    finally:
        gc.enable()


def test_closing_the_batch_screen_drains_its_run(qtbot, qt_theme_applied, tmp_path):
    from spacr.qt.screens.batch import BatchScreen

    src = tmp_path / "plate1"
    src.mkdir()
    (src / "plate1_A01_T0001F001L01A01Z01C01.tif").write_bytes(b"")
    settings = tmp_path / "mask.csv"
    settings.write_text(f"Key,Value\nsrc,{src}\ncell_channel,0\n", encoding="utf-8")
    started = threading.Event()
    release = threading.Event()

    def runner(job, settings_path, log_path):
        started.set()
        release.wait(0.5)
        return 0

    widget = BatchScreen(threaded=True)
    widget.set_runner(runner)
    widget.add_job("mask", str(settings), label="slow")
    assert widget.run() is True
    assert started.wait(180)
    thread, worker = widget._jobs[0]
    import weakref
    tref, wref = weakref.ref(thread), weakref.ref(worker)
    gc.disable()
    try:
        widget.close()
        assert widget._jobs == []
        assert widget._stop_requested is True
        del thread, worker
        release.set()

        def stopped():
            t = tref()
            return t is None or not shiboken6.isValid(t) or not t.isRunning()

        qtbot.waitUntil(stopped, timeout=30000)
        from spacr.qt.bridge import _PARKED_THREADS
        assert widget._thread is None and widget._worker is None
        _ = wref, _PARKED_THREADS
    finally:
        gc.enable()
    widget.deleteLater()

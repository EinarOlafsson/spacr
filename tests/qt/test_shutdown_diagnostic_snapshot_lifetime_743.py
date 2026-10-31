"""Real parked QThreads may finish while their anonymous snapshot is collected."""
import io
import json
import threading

from PySide6.QtCore import QCoreApplication, QEvent, QThread
from shiboken6 import isValid

from spacr import crashreport
from spacr.qt import bridge


def _contexts(capture):
    return [json.loads(line.split('=== spaCR context ', 1)[1].rsplit(' ===', 1)[0])
            for line in capture.getvalue().splitlines()]


def test_real_parked_thread_finishing_during_snapshot_keeps_diagnostics_anonymous_and_deduplicated(qapp, monkeypatch):
    entered, release = threading.Event(), threading.Event()

    class HeldThread(QThread):
        def run(self):
            entered.set()
            release.wait(10)

    thread = HeldThread()
    capture = io.StringIO()
    monkeypatch.setattr(crashreport, '_NATIVE_CAPTURE_HANDLE', capture)
    thread.start()
    try:
        assert entered.wait(5)
        assert bridge.drain_thread(thread, timeout_ms=0) is False
        assert bridge.parked_thread_count() >= 1
        for _ in range(2):
            detail = bridge._record_shutdown_workers('wait expired')
            assert 'shutdown_worker=' in detail and 'shutdown_cancelled=' not in detail
        note, _ = crashreport._native_crash_note({'known': False}, _contexts(capture), [])
        assert note.count('worker 1') == 1
        assert 'Remaining workers (public source only)' in note
        assert str(__file__) not in capture.getvalue() and 'HeldThread' not in note
        release.set()
        assert thread.wait(2000)
        thread.deleteLater()
        frames = bridge.sys._current_frames
        delivered = []

        def frames_then_deliver_real_deletion():
            actual = frames()
            QCoreApplication.sendPostedEvents(thread, QEvent.DeferredDelete)
            delivered.append(not isValid(thread))
            return actual

        with monkeypatch.context() as lifetime:
            lifetime.setattr(bridge.sys, '_current_frames', frames_then_deliver_real_deletion)
            detail = bridge._record_shutdown_workers('wait expired')
        assert delivered == [True]
        assert 'shutdown_worker=' in detail and 'shutdown_interruption=' not in detail
        assert bridge.prune_parked_threads() == 0
        capture.seek(0)
        capture.truncate(0)
        bridge._record_shutdown_workers('workers finished')
        note, _ = crashreport._native_crash_note({'known': False}, _contexts(capture), [])
        assert 'Parked-worker shutdown: workers finished; workers 0' in note
        assert 'Remaining workers' not in note and str(__file__) not in capture.getvalue()
    finally:
        release.set()
        if isValid(thread):
            thread.wait(2000)
            thread.deleteLater()
            QCoreApplication.sendPostedEvents(thread, QEvent.DeferredDelete)
        bridge.prune_parked_threads()

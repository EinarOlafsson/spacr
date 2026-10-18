"""Plate View applies only the newest recompute, whatever order jobs settle in."""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QObject, Signal  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402
from shiboken6 import isValid  # noqa: E402

from spacr.qt.screens import plate_view as pv  # noqa: E402


class _Worker(QObject):
    error = Signal(str)
    finished = Signal(bool)


class _Thread(QObject):
    finished = Signal()

    def start(self):
        pass

    def isRunning(self):
        return True


@pytest.fixture
def held(monkeypatch):
    """make_thread whose jobs only settle when the test says so."""
    made = []

    def fake_make_thread(fn, settings, *_a, **_k):
        worker, thread = _Worker(), _Thread()
        made.append((fn, settings, worker))
        return thread, worker

    monkeypatch.setattr(pv, "make_thread", fake_make_thread)
    return made


def _settle(held, index, ok=True, error=None):
    fn, box, worker = held[index]
    if error:
        worker.error.emit(error)
    else:
        fn(box)
    worker.finished.emit(ok)


def _submit(screen, applied, n):
    for i in range(n):
        screen._run_job(lambda i=i: i, applied.append, supersedes="recompute")


def test_a_stale_recompute_settling_last_is_dropped(qtbot, held):
    screen = pv.PlateViewScreen(threaded=True)
    qtbot.addWidget(screen)
    applied = []
    _submit(screen, applied, 3)

    _settle(held, 2)
    qtbot.waitUntil(lambda: applied == [2], timeout=3000)
    assert screen.is_busy() is False
    _settle(held, 0)
    _settle(held, 1)
    qtbot.wait(100)
    assert applied == [2]
    assert screen.is_busy() is False


def test_a_stale_error_does_not_override_the_latest_result(qtbot, held):
    screen = pv.PlateViewScreen(threaded=True)
    qtbot.addWidget(screen)
    applied = []
    _submit(screen, applied, 2)

    _settle(held, 1)
    qtbot.waitUntil(lambda: applied == [1], timeout=3000)
    before = screen.status_text()
    _settle(held, 0, ok=False, error="Traceback\nValueError: stale")
    qtbot.wait(100)
    assert applied == [1]
    assert screen.status_text() == before
    assert "stale" not in screen.status_text()
    assert screen.is_busy() is False


def test_busy_holds_until_the_latest_job_settles(qtbot, held):
    screen = pv.PlateViewScreen(threaded=True)
    qtbot.addWidget(screen)
    applied = []
    _submit(screen, applied, 2)

    _settle(held, 0)
    qtbot.wait(100)
    assert applied == []
    assert screen.is_busy() is True
    _settle(held, 1)
    qtbot.waitUntil(lambda: applied == [1], timeout=3000)
    assert screen.is_busy() is False


@pytest.mark.parametrize("failed", (False, True))
def test_a_worker_outliving_its_screen_drops_late_signals(qtbot, held, failed):
    """A real Qt worker signal is harmless after its target QObject is gone."""
    screen = pv.PlateViewScreen(threaded=True)
    qtbot.addWidget(screen)
    applied = []
    _submit(screen, applied, 1)
    worker = held[0][2]
    settled = []
    worker.finished.connect(settled.append)

    screen.deleteLater()
    QApplication.sendPostedEvents(screen, QEvent.DeferredDelete)
    assert not isValid(screen)
    assert isValid(worker)

    _settle(held, 0, ok=not failed,
            error="ValueError: late failure" if failed else None)
    qtbot.wait(20)

    assert settled == [not failed]
    assert applied == []

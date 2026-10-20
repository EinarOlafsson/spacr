"""The Convert screen's sweep retires only the jobs whose thread stopped.

Two jobs can be alive at once: a preview still running while an earlier
one has finished. The sweep must leave the running job, and the screen's
current job, in place, and retire the finished one; once the running job
stops, the next sweep releases it too. The threads here are real QThreads:
one running its own event loop, one never started.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QThread

from spacr.qt.screens.convert import ConvertScreen


@pytest.fixture
def screen(qtbot, qt_theme_applied):
    widget = ConvertScreen(threaded=False)
    qtbot.addWidget(widget)
    return widget


def test_the_sweep_keeps_a_running_job_and_retires_a_finished_one(
        qtbot, screen):
    running, finished = QThread(), QThread()
    running.start()
    try:
        qtbot.waitUntil(running.isRunning, timeout=3000)
        current_worker, old_worker = object(), object()
        screen._jobs = [(running, current_worker), (finished, old_worker)]
        screen._thread, screen._worker = running, current_worker

        screen._retire_finished_jobs()

        assert screen._jobs == [(running, current_worker)]
        assert screen._thread is running
        assert screen._worker is current_worker

        screen._retire_job(finished)
        assert screen._thread is running
    finally:
        running.quit()
        running.wait(5000)

    screen._retire_finished_jobs()
    assert screen._jobs == []
    assert screen._thread is None and screen._worker is None

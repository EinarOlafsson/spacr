"""Closing the Data Manager while a worker thread has already gone.

The screen keeps its jobs so it can stop them on the way out -- Qt aborts
the process if a running QThread is destroyed. But a thread that finished
and was deleted is still in the list, and asking a deleted C++ object to
quit raises. Letting that escape ``closeEvent`` would leave the remaining
threads unjoined, which is the exact failure the loop exists to prevent.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

import shiboken6
from PySide6.QtCore import QThread
from PySide6.QtGui import QCloseEvent

from spacr.qt.screens.data_manager import make_data_manager_screen

pytestmark = pytest.mark.qt


@pytest.fixture
def screen(qtbot):
    widget = make_data_manager_screen()
    qtbot.addWidget(widget)
    return widget


class TestClosingWithJobsInTheList:

    def test_a_live_thread_is_asked_to_stop_and_waited_for(self, screen):
        thread = QThread()
        screen._jobs.append((thread, object()))

        screen.closeEvent(QCloseEvent())

        assert screen._jobs == [], "the job list was not cleared"
        assert not thread.isRunning()

    def test_a_thread_whose_c_plus_plus_side_is_gone_does_not_stop_the_close(
            self, screen):
        """THE UNCOVERED EXCEPT.

        A worker that finished has had ``deleteLater`` run on it, so the
        Python wrapper outlives the C++ object and every call through it
        raises. Escaping here would abandon the threads later in the
        list -- and one of those still running when the widget is
        destroyed is what aborts the process.
        """
        gone = QThread()
        alive = QThread()
        screen._jobs.extend([(gone, object()), (alive, object())])
        shiboken6.delete(gone)

        screen.closeEvent(QCloseEvent())     # must not raise

        assert screen._jobs == []
        assert not alive.isRunning(), (
            "the thread after the deleted one was never asked to stop")

    def test_closing_with_no_jobs_at_all_is_ordinary(self, screen):
        assert screen._jobs == []

        screen.closeEvent(QCloseEvent())

        assert screen._jobs == []

    def test_the_list_is_cleared_even_when_every_entry_was_gone(self,
                                                                screen):
        """The clear is outside the loop for this reason: a list of dead
        wrappers is still a list the screen must not keep."""
        threads = [QThread() for _ in range(3)]
        screen._jobs.extend((thread, object()) for thread in threads)
        for thread in threads:
            shiboken6.delete(thread)

        screen.closeEvent(QCloseEvent())

        assert screen._jobs == []

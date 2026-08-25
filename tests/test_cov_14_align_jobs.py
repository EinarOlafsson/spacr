"""A job whose QThread wrapper is already gone still gets retired.

Qt deletes a finished QThread's C++ object on a deferred event, and that
deletion can be processed before the retirement slot runs. Asking a deleted
wrapper whether it finished raises ``RuntimeError`` from PySide6.

A wrapper that has been deleted is finished by definition, so it has to leave
the job list. Treating the exception as "still running" would leave a job
permanently active, and the screen keeps its controls disabled while any job
is active -- the user would be left with a screen that never comes back.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


class _DeletedThread:
    """Stands in for a QThread whose C++ object PySide6 has already freed."""

    def isFinished(self):  # noqa: N802 - Qt name
        raise RuntimeError("Internal C++ object already deleted.")


def test_a_deleted_thread_wrapper_is_retired_not_left_active(qtbot):
    """A wrapper that raises on ``isFinished`` leaves the job list."""
    from spacr.qt.screens.align import AlignScreen

    screen = AlignScreen(threaded=False)
    qtbot.addWidget(screen)
    ghost = _DeletedThread()
    screen._jobs.append((ghost, object()))

    screen._retire_finished_jobs()

    assert screen._jobs == []


def test_a_running_thread_wrapper_is_kept(qtbot):
    """The same pass leaves a job that is genuinely still running alone."""
    from spacr.qt.screens.align import AlignScreen

    class _Running:
        def isFinished(self):  # noqa: N802 - Qt name
            return False

    screen = AlignScreen(threaded=False)
    qtbot.addWidget(screen)
    running = _Running()
    screen._jobs.append((running, object()))

    screen._retire_finished_jobs()

    assert [thread for thread, _worker in screen._jobs] == [running]

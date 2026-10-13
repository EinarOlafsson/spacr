"""A completed agreement job whose completion handler is the thing that fails.

The screen's job plumbing has two failure sources and they are not the same
one. The worker can fail -- that arrives as a traceback on ``worker.error``
and is already covered by the threaded compute tests. The *completion
handler* can also fail, on the GUI thread, after the worker has succeeded:
the report came back fine and painting it raised. That second case has to
end the same way as the first, with ``job_finished(False)``, an inline
message and no half-painted results, or the screen sits busy forever with a
report it never showed.

The worker body itself is driven here through a stand-in for
``make_thread`` that runs the job inline. The real thing runs it on a
QThread, where the job body executes correctly but is invisible to a
tracer that only follows Python-created threads.
"""

from __future__ import annotations

import traceback

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QObject, Signal            # noqa: E402

from spacr.qt.screens.agreement import AgreementScreen   # noqa: E402

pytestmark = pytest.mark.qt


class _InlineWorker(QObject):
    """The two signals ``_run_job`` connects, with no thread behind them."""

    error = Signal(str)
    finished = Signal(bool)


class _InlineThread(QObject):
    """A ``start()`` that runs the job where the test can see it."""

    finished = Signal()

    def __init__(self, run):
        super().__init__()
        self._run = run
        self.started = False

    def start(self):
        self.started = True
        self._run()

    def isRunning(self):
        return False


@pytest.fixture
def inline_jobs(monkeypatch):
    """Run every ``_run_job`` job body inline, signalling as the real one does."""
    import spacr.qt.screens.agreement as mod

    def fake_make_thread(fn, settings, *_args, **_kwargs):
        worker = _InlineWorker()

        def run():
            ok = True
            try:
                fn(settings)
            except Exception:                       # noqa: BLE001
                ok = False
                worker.error.emit(traceback.format_exc())
            worker.finished.emit(ok)
            thread.finished.emit()

        thread = _InlineThread(run)
        return thread, worker

    monkeypatch.setattr(mod, "make_thread", fake_make_thread)


def test_the_job_body_carries_its_result_back_to_the_handler(
        qtbot, inline_jobs):
    """What the worker computed is what the completion handler is given."""
    screen = AgreementScreen(threaded=True)
    qtbot.addWidget(screen)
    delivered = []

    with qtbot.waitSignal(screen.job_finished, timeout=5000) as blocker:
        started = screen._run_job(lambda: {"kappa": 0.5}, delivered.append)

    assert started is True
    assert blocker.args[0] is True
    assert delivered == [{"kappa": 0.5}]
    assert screen.is_busy() is False
    assert screen.active_jobs() == 0


def test_a_completion_handler_that_fails_is_reported_inline(
        qtbot, inline_jobs):
    """A successful worker plus a failing handler is still a failed job."""
    screen = AgreementScreen(threaded=True)
    qtbot.addWidget(screen)

    def explodes(_result):
        raise RuntimeError("could not paint the table")

    with qtbot.waitSignal(screen.job_finished, timeout=5000) as blocker:
        screen._run_job(lambda: {"kappa": 0.5}, explodes)

    assert blocker.args[0] is False
    assert "could not paint the table" in screen.status_text()
    assert screen.report() is None
    assert screen.is_busy() is False


def test_a_worker_that_raises_is_reported_without_a_dialog(
        qtbot, inline_jobs):
    """The other failure source still ends the job and names the line."""
    screen = AgreementScreen(threaded=True)
    qtbot.addWidget(screen)
    delivered = []

    def boom():
        raise ValueError("the database went away")

    with qtbot.waitSignal(screen.job_finished, timeout=5000) as blocker:
        screen._run_job(boom, delivered.append)

    assert blocker.args[0] is False
    assert delivered == []
    assert "the database went away" in screen.status_text()

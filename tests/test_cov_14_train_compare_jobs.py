"""A scan that finishes still has to survive what is done with its result.

The worker returns a payload and the GUI thread applies it. Two things happen
on the GUI thread that the worker cannot vouch for:

* the payload is produced inside the worker's job body, so the value the
  screen later reads is whatever ``fn`` returned there;
* applying that payload can fail -- a run folder whose settings will not
  parse, a curve with no epochs -- and that failure belongs in the screen's
  status line with ``job_finished(False)``, not as an exception escaping a Qt
  slot, where it would be printed to stderr and leave the screen busy for
  ever.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


def test_a_threaded_job_carries_its_result_back_to_the_gui_thread(qtbot):
    """The worker's return value reaches ``on_done`` on the GUI thread.

    The job body itself is driven a second time, directly: it is the callable
    the worker thread invokes, and its contract -- put the return value under
    ``result`` in the shared payload -- is what ``_on_job_settled`` then reads.
    A body that stored it under another key, or not at all, would leave the
    screen applying ``None`` after a scan that worked.
    """
    from spacr.qt.screens.train_compare import TrainCompareScreen

    screen = TrainCompareScreen(threaded=True)
    qtbot.addWidget(screen)
    applied = []

    with qtbot.waitSignal(screen.job_finished, timeout=10000) as caught:
        assert screen._run_job(lambda: "scanned", applied.append) is True
        _thread, worker = screen._jobs[-1]

    assert caught.args == [True]
    assert applied == ["scanned"]
    assert screen._busy is False

    payload = {}
    worker._fn(payload)
    assert payload["result"] == "scanned"


def test_a_result_that_cannot_be_applied_is_reported_not_raised(qtbot):
    """A failure inside ``on_done`` becomes a status line and a False signal.

    Letting it out of the slot would leave ``_busy`` set, and the screen keeps
    Scan and Overlay disabled while it is busy.
    """
    from spacr.qt.screens.train_compare import TrainCompareScreen

    screen = TrainCompareScreen(threaded=False)
    qtbot.addWidget(screen)

    def _explode(_result):
        raise ValueError("the run left no epochs")

    screen._busy = True
    screen._pending = ({"result": "scanned"}, _explode)

    with qtbot.waitSignal(screen.job_finished, timeout=2000) as caught:
        screen._on_job_settled(True)

    assert caught.args == [False]
    assert screen._busy is False
    assert screen._pending is None
    assert "the run left no epochs" in screen.last_error


def test_a_settled_job_with_no_pending_work_still_settles(qtbot):
    """A retirement event with nothing pending clears busy and reports ok."""
    from spacr.qt.screens.train_compare import TrainCompareScreen

    screen = TrainCompareScreen(threaded=False)
    qtbot.addWidget(screen)
    screen._busy = True

    with qtbot.waitSignal(screen.job_finished, timeout=2000) as caught:
        screen._on_job_settled(True)

    assert caught.args == [True]
    assert screen._busy is False

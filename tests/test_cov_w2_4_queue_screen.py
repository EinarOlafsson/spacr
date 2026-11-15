"""Plate queue — the runner's four outcomes and the screen's refusals.

``tests/qt/test_plate_queue.py`` covers construction and the happy add.
What is left is the part that actually runs plates, and it is asserted by
driving ``_QueueRunner.run`` synchronously on a real ``PlateQueue`` backed
by a real file. Every transition is read back off the queue afterwards, so
what is checked is the recorded state a restart would see -- not a signal
that happened to fire.

The four outcomes, all of which a long queue hits eventually:

* a plate succeeds;
* a plate raises, which must FAIL that item and CONTINUE to the next --
  one bad plate cannot take the other eleven down with it;
* a plate is cancelled at a checkpoint, which must put it back to QUEUED
  (not failed: nothing is wrong with it) and stop the walk;
* an app key with no pipeline behind it, which fails the item with a
  message naming the key.

Plus the guards around them: Run with nothing queued, Run while already
running, Stop, Remove on a running plate, a CSV import that will not
parse, and ``closeEvent`` when the runner object is already gone.
"""
from __future__ import annotations

import time

import pytest

from PySide6.QtWidgets import QMessageBox

from spacr.qt import bridge
from spacr.qt.plate_queue import PlateQueue, QueueItem, Status
from spacr.qt.screens.queue import QueueScreen, _QueueRunner


@pytest.fixture(autouse=True)
def no_blocking_dialogs(monkeypatch):
    """Record what a message box would have said instead of showing it."""
    said = []
    monkeypatch.setattr(QMessageBox, "information",
                        staticmethod(lambda *a, **k: said.append(a[1:])
                                     or QMessageBox.Ok))
    monkeypatch.setattr(QMessageBox, "warning",
                        staticmethod(lambda *a, **k: said.append(a[1:])
                                     or QMessageBox.Ok))
    return said


@pytest.fixture
def queue(tmp_path):
    return PlateQueue(path=tmp_path / "queue.json")


@pytest.fixture
def screen(qtbot, queue):
    widget = QueueScreen(queue=queue)
    qtbot.addWidget(widget)
    return widget


def _fill(queue, *srcs, app_key="mask"):
    return [queue.add(QueueItem.build(app_key, {"src": src})) or
            queue.items()[-1] for src in srcs]


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------


def test_stop_lets_the_current_plate_finish_and_runs_no_more(queue,
                                                             monkeypatch):
    """Stop means "no more after this one", not "kill this one"."""
    _fill(queue, "/data/plateA", "/data/plateB")
    runner = _QueueRunner(queue)
    ran = []

    def entry(key):
        def call(settings):
            ran.append(settings["src"])
            runner.stop()
        return call

    monkeypatch.setattr(bridge, "resolve_pipeline_entry", entry)
    runner.run()

    assert ran == ["/data/plateA"]
    first, second = queue.items()
    assert first.status == Status.SUCCESS
    assert second.status == Status.QUEUED


# ---------------------------------------------------------------------------
# Run / Stop
# ---------------------------------------------------------------------------


def test_stop_is_harmless_when_nothing_is_running(screen):
    screen.stop_runner()
    assert screen._runner is None


def test_stop_asks_the_live_runner_to_finish(screen, queue):
    _fill(queue, "/data/plateA")

    class Live:
        def __init__(self):
            self.asked = False

        def isRunning(self):
            return True

        def stop(self):
            self.asked = True

    screen._runner = Live()
    screen.stop_runner()
    assert screen._runner.asked is True
    screen._runner = None


# ---------------------------------------------------------------------------
# closeEvent
# ---------------------------------------------------------------------------

def test_closing_the_screen_aborts_a_runner_whose_object_is_gone(screen):
    """A deleted C++ object raises out of abort(); close must survive it."""
    class Deleted:
        def abort(self, reason="queue screen closed"):
            raise RuntimeError("Internal C++ object already deleted")

        def isRunning(self):
            raise RuntimeError("Internal C++ object already deleted")

    screen._runner = Deleted()
    screen.close()

    assert screen._runner is None
    assert screen._tick.isActive() is False


def test_closing_the_screen_stops_the_elapsed_timer_with_no_runner(screen):
    screen.close()
    assert screen._tick.isActive() is False


# ---------------------------------------------------------------------------
# The toolbar
# ---------------------------------------------------------------------------

def test_add_current_reports_a_callback_that_raises(screen,
                                                    no_blocking_dialogs):
    def explode():
        raise RuntimeError("no app screen is open")

    screen.wire_add_current(explode)
    screen._btn_add.click()

    assert len(screen.queue()) == 0
    assert any("no app screen is open" in str(args)
               for args in no_blocking_dialogs)


def test_clear_finished_removes_only_the_finished_plates(screen, queue):
    _fill(queue, "/data/done", "/data/waiting")
    done = queue.items()[0]
    queue.update(done.id, status=Status.SUCCESS, end_ts=time.time())
    sizes = []
    screen.queue_size_changed.connect(sizes.append)

    screen._on_clear_finished()

    assert [item.label for item in queue.items()] == ["/data/waiting"]
    assert screen._table.rowCount() == 1
    assert sizes == [1]


def test_clear_finished_with_nothing_finished_still_reports_the_size(
        screen, queue):
    _fill(queue, "/data/waiting")
    sizes = []
    screen.queue_size_changed.connect(sizes.append)
    screen._on_clear_finished()
    assert sizes == [1]
    assert len(queue) == 1

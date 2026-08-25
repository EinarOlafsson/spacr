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

from spacr.cancellation import PipelineCancelled
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

def test_a_queue_of_plates_is_walked_in_order_and_recorded_as_finished(
        qtbot, queue, monkeypatch):
    _fill(queue, "/data/plateA", "/data/plateB")
    seen = []
    monkeypatch.setattr(
        bridge, "resolve_pipeline_entry",
        lambda key: (lambda settings: seen.append(settings["src"])))

    runner = _QueueRunner(queue)
    changed = []
    runner.item_state_changed.connect(changed.append)
    runner.run()

    assert seen == ["/data/plateA", "/data/plateB"]
    assert [item.status for item in queue.items()] == [
        Status.SUCCESS, Status.SUCCESS]
    assert all(item.end_ts is not None for item in queue.items())
    # RUNNING then SUCCESS for each plate.
    assert len(changed) == 4


def test_one_plate_that_raises_fails_alone_and_the_rest_still_run(
        qtbot, queue, monkeypatch):
    _fill(queue, "/data/bad", "/data/good")
    ran = []

    def entry(key):
        def call(settings):
            if settings["src"] == "/data/bad":
                raise ValueError("no images under /data/bad")
            ran.append(settings["src"])
        return call

    monkeypatch.setattr(bridge, "resolve_pipeline_entry", entry)
    _QueueRunner(queue).run()

    bad, good = queue.items()
    assert bad.status == Status.FAILED
    assert "no images under /data/bad" in bad.error
    assert good.status == Status.SUCCESS
    assert ran == ["/data/good"]


def test_an_app_key_with_no_pipeline_fails_that_item_naming_the_key(
        queue, monkeypatch):
    _fill(queue, "/data/plateA", app_key="not_an_app")
    monkeypatch.setattr(bridge, "resolve_pipeline_entry", lambda key: None)

    _QueueRunner(queue).run()

    item = queue.items()[0]
    assert item.status == Status.FAILED
    assert "not_an_app" in item.error


def test_a_cancelled_plate_goes_back_to_queued_and_stops_the_walk(
        queue, monkeypatch):
    """Cancelled is not failed: nothing is wrong with the plate."""
    _fill(queue, "/data/plateA", "/data/plateB")
    monkeypatch.setattr(
        bridge, "resolve_pipeline_entry",
        lambda key: (lambda settings: (_ for _ in ()).throw(
            PipelineCancelled("stopped by the user"))))

    _QueueRunner(queue).run()

    first, second = queue.items()
    assert first.status == Status.QUEUED
    assert first.error == ""
    assert first.end_ts is None
    assert second.status == Status.QUEUED


def test_abort_cancels_the_plate_that_is_running_right_now(queue,
                                                           monkeypatch):
    """The token reaches the pipeline through the installed checkpoint."""
    from spacr import cancellation

    runner = _QueueRunner(queue)
    _fill(queue, "/data/plateA", "/data/plateB")

    def entry(key):
        def call(settings):
            runner.abort("teardown")
            cancellation.checkpoint()
        return call

    monkeypatch.setattr(bridge, "resolve_pipeline_entry", entry)
    runner.run()

    assert [item.status for item in queue.items()] == [
        Status.QUEUED, Status.QUEUED]


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


def test_an_empty_queue_finishes_immediately(queue):
    runner = _QueueRunner(queue)
    finished = []
    runner.queue_finished.connect(lambda: finished.append(True))
    runner.run()
    assert finished == [True]


# ---------------------------------------------------------------------------
# Run / Stop
# ---------------------------------------------------------------------------

def test_run_with_nothing_queued_says_so_instead_of_starting_a_thread(
        screen, no_blocking_dialogs):
    screen.start_runner()
    assert screen._runner is None
    assert any("already finished" in str(args) for args in
               no_blocking_dialogs)
    assert screen._btn_stop.isEnabled() is False


def test_run_while_a_run_is_going_does_not_start_a_second_one(
        screen, queue):
    """A second Run must not put two threads on one queue."""
    _fill(queue, "/data/plateA")

    class Busy:
        def isRunning(self):
            return True

    screen._runner = Busy()
    screen.start_runner()
    assert isinstance(screen._runner, Busy)
    screen._runner = None


def test_a_real_run_flips_the_buttons_and_records_the_result(
        screen, queue, monkeypatch, qtbot):
    """Through the real QThread, because that is what the buttons track."""
    _fill(queue, "/data/plateA")
    monkeypatch.setattr(bridge, "resolve_pipeline_entry",
                        lambda key: (lambda settings: None))

    screen.start_runner()
    assert screen._btn_run.isEnabled() is False
    assert screen._btn_stop.isEnabled() is True

    qtbot.waitUntil(lambda: screen._btn_run.isEnabled(), timeout=10000)
    assert screen._btn_stop.isEnabled() is False
    assert queue.items()[0].status == Status.SUCCESS
    assert screen._table.item(0, 3).text() == "success"


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


def test_importing_a_csv_adds_every_plate_in_it(screen, tmp_path,
                                                monkeypatch):
    csv_path = tmp_path / "plates.csv"
    csv_path.write_text("src,magnification\n/data/p1,20\n/data/p2,20\n")
    monkeypatch.setattr(
        "PySide6.QtWidgets.QFileDialog.getOpenFileName",
        staticmethod(lambda *a, **k: (str(csv_path), "CSV files (*.csv)")))
    sizes = []
    screen.queue_size_changed.connect(sizes.append)

    screen._on_import()

    assert len(screen.queue()) == 2
    assert screen._table.rowCount() == 2
    assert sizes == [2]


def test_a_cancelled_import_adds_nothing(screen, monkeypatch):
    monkeypatch.setattr(
        "PySide6.QtWidgets.QFileDialog.getOpenFileName",
        staticmethod(lambda *a, **k: ("", "")))
    screen._on_import()
    assert len(screen.queue()) == 0


def test_an_import_that_cannot_be_read_names_the_file(screen, tmp_path,
                                                      monkeypatch,
                                                      no_blocking_dialogs):
    missing = tmp_path / "not_here.csv"
    monkeypatch.setattr(
        "PySide6.QtWidgets.QFileDialog.getOpenFileName",
        staticmethod(lambda *a, **k: (str(missing), "CSV files (*.csv)")))

    screen._on_import()

    assert len(screen.queue()) == 0
    assert any(str(missing) in str(args) for args in no_blocking_dialogs)


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


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------

def test_the_elapsed_column_ticks_only_for_the_running_plate(screen, queue):
    _fill(queue, "/data/running", "/data/waiting")
    running = queue.items()[0]
    queue.update(running.id, status=Status.RUNNING, start_ts=time.time() - 5)
    screen._refresh_table()

    screen._refresh_elapsed_only()

    assert screen._table.item(0, 4).text().endswith(" s")
    assert float(screen._table.item(0, 4).text().split()[0]) >= 5
    assert screen._table.item(1, 4).text() == ""


def test_a_running_plate_cannot_be_removed(screen, queue,
                                           no_blocking_dialogs):
    _fill(queue, "/data/running")
    running = queue.items()[0]
    queue.update(running.id, status=Status.RUNNING, start_ts=time.time())
    screen._refresh_table()

    screen._on_remove(running.id)

    assert len(queue) == 1
    assert any("running" in str(args) for args in no_blocking_dialogs)
    # The row's own Remove button is disabled too, so the message is a
    # backstop rather than the only guard.
    assert screen._table.cellWidget(0, 5).isEnabled() is False


def test_a_queued_plate_is_removed_by_its_own_button(screen, queue):
    _fill(queue, "/data/waiting")
    screen._refresh_table()
    screen._table.cellWidget(0, 5).click()
    assert len(queue) == 0
    assert screen._table.rowCount() == 0

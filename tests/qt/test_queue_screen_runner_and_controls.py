"""Plate queue: the runner's outcomes and the screen's controls.

``_QueueRunner.run`` is driven synchronously on a real ``PlateQueue`` backed
by a real file, and every transition is read back off the queue -- the state
a restart would see. A plate that raises fails alone; a cancelled one goes
back to QUEUED and stops the walk; an app key with no pipeline fails naming
the key. The screen refuses a second run, an empty run, a running plate's
removal and an unreadable import, each with its exact dialog text.
"""
from __future__ import annotations


import pytest
from PySide6.QtWidgets import QMessageBox

from spacr import cancellation
from spacr.cancellation import PipelineCancelled
from spacr.qt import bridge
from spacr.qt.plate_queue import PlateQueue, QueueItem, Status
from spacr.qt.screens.queue import QueueScreen, _QueueRunner

pytestmark = pytest.mark.qt


@pytest.fixture(autouse=True)
def said(monkeypatch):
    """Record what a message box would have said instead of showing it."""
    record = []

    def _box(kind):
        return staticmethod(lambda *a, **k: record.append((kind,) + a[1:])
                            or QMessageBox.Ok)

    monkeypatch.setattr(QMessageBox, "information", _box("information"))
    monkeypatch.setattr(QMessageBox, "warning", _box("warning"))
    return record


@pytest.fixture
def queue(tmp_path):
    return PlateQueue(path=tmp_path / "queue.json")


@pytest.fixture
def screen(qtbot, queue):
    widget = QueueScreen(queue=queue)
    qtbot.addWidget(widget)
    return widget


def _fill(queue, *srcs, app_key="mask"):
    for src in srcs:
        queue.add(QueueItem.build(app_key, {"src": src}))
    return queue.items()


def _state(queue):
    return [(item.label, item.status, item.error) for item in queue.items()]


def test_the_runner_records_each_outcome_on_the_queue(qtbot, tmp_path,
                                                    monkeypatch, subtests):
    """Success, failure, a missing pipeline, cancel and abort, item by item."""
    def raising(settings):
        if settings["src"] == "/data/bad":
            raise ValueError("no images under /data/bad")

    def cancelling(settings):
        raise PipelineCancelled("stopped by the user")

    cases = [
        ("every plate succeeds in order", ("/data/plateA", "/data/plateB"),
         "mask", lambda runner: (lambda s: None),
         [("/data/plateA", Status.SUCCESS, None),
          ("/data/plateB", Status.SUCCESS, None)],
         ["/data/plateA", "/data/plateB"], 4),
        ("one plate raises, the rest still run", ("/data/bad", "/data/good"),
         "mask", lambda runner: raising,
         [("/data/bad", Status.FAILED, "no images under /data/bad"),
          ("/data/good", Status.SUCCESS, None)],
         ["/data/bad", "/data/good"], 4),
        ("an app key with no pipeline fails naming it", ("/data/plateA",),
         "not_an_app", None,
         [("/data/plateA", Status.FAILED,
           "no pipeline for app_key='not_an_app'")], [], 2),
        ("a cancelled plate goes back to queued and stops the walk",
         ("/data/plateA", "/data/plateB"), "mask",
         lambda runner: cancelling,
         [("/data/plateA", Status.QUEUED, ""),
          ("/data/plateB", Status.QUEUED, None)], ["/data/plateA"], 2),
        ("abort cancels the plate running now",
         ("/data/plateA", "/data/plateB"), "mask",
         lambda runner: (lambda s: (runner.abort("teardown"),
                                    cancellation.checkpoint())),
         [("/data/plateA", Status.QUEUED, ""),
          ("/data/plateB", Status.QUEUED, None)], ["/data/plateA"], 2),
        ("stop lets the current plate finish and runs no more",
         ("/data/plateA", "/data/plateB"), "mask",
         lambda runner: (lambda s: runner.stop()),
         [("/data/plateA", Status.SUCCESS, None),
          ("/data/plateB", Status.QUEUED, None)], ["/data/plateA"], 2),
        ("an empty queue finishes at once", (), "mask",
         lambda runner: (lambda s: None), [], [], 0),
    ]
    for index, (case_id, srcs, app_key, make, state, ran, changes) in \
            enumerate(cases):
        with subtests.test(msg=case_id):
            queue = PlateQueue(path=tmp_path / f"queue{index}.json")
            _fill(queue, *srcs, app_key=app_key)
            runner = _QueueRunner(queue)
            seen = []

            def entry(key, runner=runner, make=make):
                if make is None:
                    return None
                body = make(runner)

                def call(settings):
                    seen.append(settings["src"])
                    body(settings)
                return call

            monkeypatch.setattr(bridge, "resolve_pipeline_entry", entry)
            changed, finished = [], []
            runner.item_state_changed.connect(changed.append)
            runner.queue_finished.connect(lambda: finished.append(True))
            runner.run()
            assert _state(queue) == state
            assert seen == ran
            assert len(changed) == changes
            assert finished == [True]
            reread = PlateQueue(path=tmp_path / f"queue{index}.json")
            assert _state(reread) == state
            for item in queue.items():
                assert (item.end_ts is None) is (item.status == Status.QUEUED)


def test_run_refuses_an_empty_queue_and_a_second_run(screen, queue, said,
                                                    monkeypatch, qtbot,
                                                    subtests):
    """Run says why it will not start, and a real run flips the buttons."""
    with subtests.test(msg="nothing queued: no thread, a sentence instead"):
        screen.start_runner()
        assert screen._runner is None
        assert said == [("information", "Queue",
                         "Nothing to run — every item is already finished.")]
        assert screen._btn_stop.isEnabled() is False
        assert screen._btn_run.isEnabled() is True

    _fill(queue, "/data/plateA")
    with subtests.test(msg="a run already going is not doubled"):
        class Busy:
            def isRunning(self):
                return True

        busy = Busy()
        screen._runner = busy
        screen.start_runner()
        assert screen._runner is busy
        assert screen._btn_run.isEnabled() is True
        screen._runner = None

    with subtests.test(msg="a real run flips the buttons and records it"):
        monkeypatch.setattr(bridge, "resolve_pipeline_entry",
                            lambda key: (lambda settings: None))
        screen.start_runner()
        assert screen._btn_run.isEnabled() is False
        assert screen._btn_stop.isEnabled() is True
        qtbot.waitUntil(lambda: screen._btn_run.isEnabled(), timeout=10000)
        screen._runner.wait(5000)
        assert screen._btn_stop.isEnabled() is False
        assert queue.items()[0].status == Status.SUCCESS
        assert screen._table.item(0, 3).text() == "success"


def test_import_remove_and_the_elapsed_tick(screen, queue, tmp_path, said,
                                           monkeypatch, subtests):
    """Import adds every plate or names the file; Remove spares a running one."""
    sizes = []
    screen.queue_size_changed.connect(sizes.append)
    asked = []

    def pick(path):
        monkeypatch.setattr(
            "PySide6.QtWidgets.QFileDialog.getOpenFileName",
            staticmethod(lambda *a, **k: asked.append(a[1:]) or
                         (path, "CSV files (*.csv)" if path else "")))

    with subtests.test(msg="a cancelled import adds nothing"):
        pick("")
        screen._on_import()
        assert asked == [("Import plates from CSV", "", "CSV files (*.csv)")]
        assert (len(queue), sizes, said) == (0, [], [])

    with subtests.test(msg="an unreadable import names the file"):
        missing = str(tmp_path / "not_here.csv")
        pick(missing)
        screen._on_import()
        assert len(queue) == 0
        assert sizes == []
        assert [s[:2] for s in said] == [("warning", "Queue import")]
        assert said[0][2].startswith(f"Couldn't import {missing}:\n")
        said.clear()

    with subtests.test(msg="a csv adds every plate in it"):
        csv_path = tmp_path / "plates.csv"
        csv_path.write_text("src,magnification\n/data/p1,20\n/data/p2,20\n",
                            encoding="utf-8")
        pick(str(csv_path))
        screen._on_import()
        assert [(i.app_key, i.settings["src"]) for i in queue.items()] == [
            ("mask", "/data/p1"), ("mask", "/data/p2")]
        assert screen._table.rowCount() == 2
        assert sizes == [2]
        assert said == [("information", "Queue import",
                         f"Added 2 plate(s) from {csv_path}.")]
        said.clear()

    first, second = queue.items()
    queue.update(first.id, status=Status.RUNNING, start_ts=100.0,
                 end_ts=105.5)
    queue.update(second.id, status=Status.QUEUED, start_ts=None)
    screen._refresh_table()
    with subtests.test(msg="the elapsed column ticks only the running plate"):
        screen._table.item(0, 4).setText("")
        screen._refresh_elapsed_only()
        assert screen._table.item(0, 4).text() == "5.5 s"
        assert screen._table.item(1, 4).text() == ""

    with subtests.test(msg="a running plate cannot be removed"):
        assert screen._table.cellWidget(0, 5).isEnabled() is False
        screen._on_remove(first.id)
        assert [i.id for i in queue.items()] == [first.id, second.id]
        assert said == [("warning", "Queue", "Can't remove a plate while "
                         "it's running — stop the queue first.")]
        assert sizes == [2]

    with subtests.test(msg="a queued plate goes by its own button"):
        assert screen._table.cellWidget(1, 5).isEnabled() is True
        screen._table.cellWidget(1, 5).click()
        assert [i.id for i in queue.items()] == [first.id]
        assert screen._table.rowCount() == 1
        assert sizes == [2, 1]

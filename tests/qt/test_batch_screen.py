"""Batch Runner screen — :mod:`spacr.qt.screens.batch`.

Everything here runs offscreen with a **mocked runner**: not one job ever
starts a real ``spacr-run`` process, let alone segments an image. What the
screen has to get right is the part that is not the pipeline:

* jobs can be added, duplicated, reordered and removed, and the table shows it;
* a queue saves and loads and comes back the same;
* validation problems land **inline** — never in a modal dialog, which would
  hang a headless run forever (the autouse fixture below makes that a red test
  rather than a hang);
* a job that cannot run is refused when it is *added*, not at 3 a.m.;
* a run updates per-job status live and finishes with a summary;
* and the completion handler runs on the **GUI thread**, because
  ``PipelineWorker.finished`` is emitted on the worker thread and PySide6
  hands a plain closure a direct call there. That is asserted against the
  widget's own thread.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from PySide6.QtCore import QThread

from spacr import batch as bt
from spacr.qt.screens.batch import COLUMNS, BatchScreen


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_modal_dialogs(monkeypatch):
    """Blow up loudly if any code path under test opens a modal dialog.

    ``MakeMasksScreen._load_current`` once hung the whole headless suite on a
    QMessageBox; this fixture makes that failure mode impossible to
    reintroduce here without a red test. The file pickers are only ever
    reached from a button, and the tests call ``save_queue_to`` /
    ``load_queue_from`` directly instead.
    """
    from PySide6.QtWidgets import QDialog, QFileDialog, QMessageBox

    def _boom(*_a, **_k):
        raise AssertionError(
            "a modal dialog was opened — errors must be reported inline")

    for name in ("about", "critical", "information", "question", "warning"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(_boom))
    for name in ("exec", "exec_", "open", "show"):
        monkeypatch.setattr(QMessageBox, name, _boom, raising=False)
    monkeypatch.setattr(QDialog, "exec", _boom, raising=False)
    for name in ("getOpenFileName", "getSaveFileName", "getExistingDirectory"):
        monkeypatch.setattr(QFileDialog, name, staticmethod(_boom))
    yield


class FakeRunner:
    """Stands in for ``spacr-run``. Nothing real is ever executed."""

    def __init__(self, codes=None, default=0):
        self.codes = dict(codes or {})
        self.default = default
        self.calls = []

    def __call__(self, job, settings_path, log_path):
        self.calls.append(job.id)
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text(f"log for {job.id}\n", encoding="utf-8")
        return int(self.codes.get(job.id, self.default))


def _plate(tmp_path, name="plate1"):
    src = tmp_path / name
    src.mkdir(parents=True, exist_ok=True)
    (src / f"{name}_A01_T0001F001L01A01Z01C01.tif").write_bytes(b"")
    return str(src)


def _settings_csv(tmp_path, name, **values):
    path = tmp_path / name
    lines = ["Key,Value"] + [f"{k},{v}" for k, v in values.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


@pytest.fixture
def plate(tmp_path):
    return _plate(tmp_path)


@pytest.fixture
def mask_settings(tmp_path, plate):
    return _settings_csv(tmp_path, "mask.csv", src=plate, cell_channel=0)


@pytest.fixture
def screen(qtbot, qt_theme_applied):
    """A synchronous screen — the queue runs inline so assertions are exact."""
    widget = BatchScreen(threaded=False, runner=FakeRunner())
    qtbot.addWidget(widget)
    return widget


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------


def test_it_builds_offscreen(screen):
    assert screen.queue() is not None
    assert screen.queue().jobs == []
    assert screen._table.columnCount() == len(COLUMNS)
    assert screen.status_text()
    assert screen.last_error == ""
    assert screen.is_busy() is False


def test_the_module_list_comes_from_the_cli_registry(screen):
    """The screen offers exactly what can run headless — nothing else."""
    from spacr.cli import MODULES

    offered = {screen._module_combo.itemData(i)
               for i in range(screen._module_combo.count())}
    assert offered == set(MODULES)
    assert "annotate" not in offered      # GUI-only, cannot be queued


# ---------------------------------------------------------------------------
# editing the queue
# ---------------------------------------------------------------------------


def test_add_a_job_and_see_it_in_the_table(screen, mask_settings):
    assert screen.add_job("mask", mask_settings, label="plate 1 mask") is True

    assert screen._table.rowCount() == 1
    row = screen.row_values(0)
    assert row[COLUMNS.index("#")] == "1"
    assert row[COLUMNS.index("Module")] == "mask"
    assert row[COLUMNS.index("Label")] == "plate 1 mask"
    assert row[COLUMNS.index("Status")] == bt.STATUS_PENDING
    assert screen.queue().ids == ["mask-1"]


def test_adding_emits_queue_changed(qtbot, screen, mask_settings):
    with qtbot.waitSignal(screen.queue_changed, timeout=1000) as blocker:
        screen.add_job("mask", mask_settings)
    assert blocker.args == [1]


def test_a_job_that_cannot_run_is_refused_inline_when_added(screen, tmp_path):
    """No dialog, no silent no-op: the reason is in the problems pane."""
    assert screen.add_job("mask", str(tmp_path / "gone.csv")) is False
    assert screen.queue().jobs == []
    assert "not found" in screen.problems_text()
    assert screen.last_error, "the failure must be reported inline"


def test_reordering_moves_the_job(screen, tmp_path, mask_settings):
    other = _settings_csv(tmp_path, "mask2.csv", src=_plate(tmp_path, "plate2"),
                          cell_channel=0)
    screen.add_job("mask", mask_settings, label="first")
    screen.add_job("mask", other, label="second")
    assert [job.label for job in screen.queue()] == ["first", "second"]

    screen.select_job(screen.queue().ids[1])
    assert screen.move_selected(-1) is True

    assert [job.label for job in screen.queue()] == ["second", "first"]
    assert screen.row_values(0)[COLUMNS.index("Label")] == "second"


def test_duplicate_gives_an_editable_never_run_copy(screen, mask_settings):
    screen.add_job("mask", mask_settings, label="plate 1 mask")
    screen.select_job("mask-1")
    screen.queue().jobs[0].status = bt.STATUS_SUCCESS

    assert screen.duplicate_selected() is True

    assert screen.queue().ids == ["mask-1", "mask-2"]
    clone = screen.queue().find("mask-2")
    assert clone.status == bt.STATUS_PENDING
    assert clone.label.endswith("(copy)")
    assert clone.settings == screen.queue().find("mask-1").settings


def test_remove_drops_the_job_and_the_dependency_on_it(screen, tmp_path, plate,
                                                       mask_settings):
    (Path(plate) / "merged").mkdir(parents=True, exist_ok=True)
    (Path(plate) / "merged" / "f1.npy").write_bytes(b"")
    measure = _settings_csv(tmp_path, "measure.csv", src=plate, cell_mask_dim=4)
    screen.add_job("mask", mask_settings)
    screen.add_job("measure", measure, depends_on=["mask-1"])
    assert screen.queue().find("measure-1").depends_on == ["mask-1"]

    screen.select_job("mask-1")
    assert screen.remove_selected() is True

    assert screen.queue().ids == ["measure-1"]
    assert screen.queue().find("measure-1").depends_on == []
    assert screen._table.rowCount() == 1


def test_editing_buttons_need_a_selection(screen):
    assert screen.duplicate_selected() is False
    assert screen.remove_selected() is False
    assert screen.move_selected(1) is False
    assert screen.last_error


# ---------------------------------------------------------------------------
# validation, inline
# ---------------------------------------------------------------------------


def test_validation_shows_every_problem_inline(screen, tmp_path, mask_settings):
    screen.add_job("mask", mask_settings)
    screen.queue().add(bt.Job(module="maks", settings={"src": "/nope"},
                              id="typo-module"), validate=False)
    screen.queue().add(bt.Job(module="mask", settings={"src": str(tmp_path / "plaet9")},
                              id="typo-src"), validate=False)

    problems = screen.validate_now()

    text = screen.problems_text()
    assert "typo-module" in text and "typo-src" in text
    errors = [p for p in problems if p.is_error]
    assert {p.job_id for p in errors} == {"typo-module", "typo-src"}
    assert screen.has_errors() is True
    assert screen.last_error
    assert screen._btn_run.isEnabled() is False


def test_a_clean_queue_validates_clean(screen, mask_settings):
    screen.add_job("mask", mask_settings)
    problems = screen.validate_now()
    assert not [p for p in problems if p.is_error]
    assert screen.has_errors() is False
    assert screen._btn_run.isEnabled() is True


def test_running_an_invalid_queue_is_refused_before_anything_starts(screen, tmp_path):
    runner = FakeRunner()
    screen.set_runner(runner)
    screen.queue().add(bt.Job(module="mask", settings={"src": str(tmp_path / "plaet9")},
                              id="typo-src"), validate=False)

    assert screen.run() is False

    assert runner.calls == []
    assert screen.last_error
    assert "typo-src" in screen.problems_text()


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------


def test_save_and_load_round_trip(screen, tmp_path, mask_settings):
    screen.add_job("mask", mask_settings, label="plate 1 mask")
    screen.add_job("mask", mask_settings, label="plate 1 mask again")
    path = tmp_path / "night.queue.json"

    assert screen.save_queue_to(str(path)) is True
    assert path.is_file()
    assert json.loads(path.read_text(encoding="utf-8"))["jobs"]

    fresh = BatchScreen(threaded=False, runner=FakeRunner())
    try:
        assert fresh.load_queue_from(str(path)) is True
        assert fresh.queue().ids == screen.queue().ids
        assert [j.label for j in fresh.queue()] == ["plate 1 mask",
                                                    "plate 1 mask again"]
        assert fresh._table.rowCount() == 2
        assert fresh.queue_path() == str(path)
    finally:
        fresh.deleteLater()


def test_loading_a_broken_file_reports_inline(screen, tmp_path):
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")

    assert screen.load_queue_from(str(broken)) is False
    assert "not valid JSON" in screen.problems_text()
    assert screen.last_error


# ---------------------------------------------------------------------------
# running (mocked), synchronous
# ---------------------------------------------------------------------------


def test_a_mocked_run_updates_per_job_status(screen, tmp_path, mask_settings):
    other = _settings_csv(tmp_path, "mask2.csv", src=_plate(tmp_path, "plate2"),
                          cell_channel=0)
    screen.add_job("mask", mask_settings, label="ok job")
    screen.add_job("mask", other, label="doomed job")
    runner = FakeRunner(codes={"mask-2": 1})
    screen.set_runner(runner)
    screen.save_queue_to(str(tmp_path / "q.json"))
    screen._threshold_spin.setValue(0)

    assert screen.run() is True

    assert runner.calls == ["mask-1", "mask-2"]
    assert screen.row_status(0) == bt.STATUS_SUCCESS
    assert screen.row_status(1) == bt.STATUS_FAILED
    assert screen.is_busy() is False
    assert screen.result() is not None
    assert "Failures, grouped" in screen.problems_text()


def test_a_skipped_job_says_so_in_the_table(screen, tmp_path, plate, mask_settings):
    (Path(plate) / "merged").mkdir(parents=True, exist_ok=True)
    (Path(plate) / "merged" / "f1.npy").write_bytes(b"")
    measure = _settings_csv(tmp_path, "measure.csv", src=plate, cell_mask_dim=4)
    screen.add_job("mask", mask_settings, label="mask")
    screen.add_job("measure", measure, label="measure", depends_on=["mask-1"])
    runner = FakeRunner(codes={"mask-1": 1})
    screen.set_runner(runner)
    screen._threshold_spin.setValue(0)

    screen.run()

    assert runner.calls == ["mask-1"], "the dependent job was run anyway"
    assert screen.row_status(1) == bt.STATUS_SKIPPED
    assert "mask-1" in screen.queue().find("measure-1").error


def test_the_run_selects_and_shows_the_running_jobs_log(screen, mask_settings):
    screen.add_job("mask", mask_settings)
    screen.run()
    assert "log for mask-1" in screen.log_text()


def test_progress_bar_tracks_the_queue(screen, tmp_path, mask_settings):
    screen.add_job("mask", mask_settings)
    screen.add_job("mask", mask_settings)
    screen.run()
    assert screen._progress.maximum() == 2
    assert screen._progress.value() == 2


def test_run_is_refused_when_the_queue_is_empty(screen):
    assert screen.run() is False
    assert screen.last_error


def test_job_status_changed_is_emitted(qtbot, screen, mask_settings):
    screen.add_job("mask", mask_settings)
    seen = []
    screen.job_status_changed.connect(lambda jid, st: seen.append((jid, st)))
    screen.run()
    assert ("mask-1", bt.STATUS_RUNNING) in seen
    assert ("mask-1", bt.STATUS_SUCCESS) in seen


# ---------------------------------------------------------------------------
# threading
# ---------------------------------------------------------------------------


def test_the_run_happens_off_the_gui_thread_and_settles_on_it(qtbot, qt_theme_applied,
                                                              tmp_path, mask_settings):
    """The load-bearing threading test.

    ``PipelineWorker.finished`` is emitted in the worker thread, and PySide6
    invokes a plain closure connected to it *there*. Touching a QTableWidget
    from that thread is undefined behaviour. So the run's completion is
    relayed through a signal into a bound method, and this asserts the
    completion really did land back on the widget's own thread — while the
    job itself really did run on another one.
    """
    widget = BatchScreen(threaded=True)
    qtbot.addWidget(widget)
    threads = []

    def _runner(job, settings_path, log_path):
        threads.append(QThread.currentThread())
        Path(log_path).write_text("ran\n", encoding="utf-8")
        return 0

    widget.set_runner(_runner)
    widget.add_job("mask", mask_settings, label="threaded job")

    with qtbot.waitSignal(widget.queue_finished, timeout=15000) as blocker:
        assert widget.run() is True

    assert blocker.args == [True]
    assert widget.settled_thread is widget.thread(), \
        "the completion handler did not run on the GUI thread"
    assert threads and threads[0] is not widget.thread(), \
        "the job ran on the GUI thread — the window would have frozen"
    assert widget.row_status(0) == bt.STATUS_SUCCESS
    assert widget.is_busy() is False

    # The QThread must not be garbage-collected while running — that segfaults
    # the process. It is released only after its own event loop exits.
    qtbot.waitUntil(lambda: widget.active_jobs() == 0, timeout=15000)


def test_a_second_run_while_busy_is_ignored(screen, mask_settings, monkeypatch):
    screen.add_job("mask", mask_settings)
    screen._busy = True
    assert screen.run() is False
    screen._busy = False


def test_stop_asks_the_queue_to_halt_between_jobs(screen, mask_settings):
    screen.add_job("mask", mask_settings)
    assert screen.stop() is False           # not running
    screen._busy = True
    assert screen.stop() is True
    assert screen._stop_requested is True
    screen._busy = False


# ---------------------------------------------------------------------------
# persistence during a run
# ---------------------------------------------------------------------------


def test_the_run_keeps_the_queue_file_up_to_date(screen, tmp_path, mask_settings):
    """A machine that reboots mid-queue is resumed from this file."""
    path = tmp_path / "night.queue.json"
    screen.add_job("mask", mask_settings)
    screen.save_queue_to(str(path))

    screen.run()

    saved = bt.load_queue(path)
    assert saved.find("mask-1").status == bt.STATUS_SUCCESS
    assert saved.find("mask-1").log_path
    assert os.path.isfile(saved.find("mask-1").log_path)


def test_a_partial_job_is_shown_as_partial_not_success(screen, tmp_path,
                                                       plate, mask_settings):
    """Exit 0 with 40 failed fields is not a success, and the table says so."""
    from spacr.errors import RunLedger

    measurements = Path(plate) / "measurements"
    measurements.mkdir(parents=True, exist_ok=True)

    def _partial(job, settings_path, log_path):
        Path(log_path).write_text("done\n", encoding="utf-8")
        ledger = RunLedger("mask")
        ledger.record_success("field1")
        ledger.record_failure("field2", exc=ValueError("unreadable"))
        ledger.stamp(measurements / "measurements.db")
        return 0

    screen.set_runner(_partial)
    screen.add_job("mask", mask_settings)

    screen.run()

    assert screen.row_status(0) == "success (partial)"
    assert screen.queue().find("mask-1").is_partial is True
    assert "PARTIAL" in screen.problems_text()
    assert screen.last_error

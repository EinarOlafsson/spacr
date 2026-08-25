"""Batch Runner refusals, file pickers and the reports nobody watches.

This screen is meant to be left alone overnight, so the paths that matter are
the ones where something has already gone wrong: a settings file deleted after
its job was added, a queue that cannot be written, a runner that throws before
the first job, a log file that will not open. Every one of them has to land in
the inline status line -- a modal dialog here hangs an unattended run until
morning.
"""

from __future__ import annotations

import builtins
import os

import pytest

from PySide6.QtCore import QThread
from PySide6.QtWidgets import QFileDialog

from spacr import batch as bt
from spacr.qt.screens.batch import COLUMNS, BatchScreen, _split_list

pytestmark = pytest.mark.qt


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


class _FakeRunner:
    """Stands in for ``spacr-run``; nothing real is ever executed."""

    def __call__(self, job, settings_path, log_path):
        from pathlib import Path

        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text(f"log for {job.id}\n", encoding="utf-8")
        return 0


@pytest.fixture()
def screen(qtbot):
    """A synchronous screen, so every assertion is exact."""
    widget = BatchScreen(threaded=False, runner=_FakeRunner())
    qtbot.addWidget(widget)
    return widget


@pytest.fixture()
def settings(tmp_path):
    return _settings_csv(tmp_path, "mask.csv", src=_plate(tmp_path),
                         cell_channel=0)


# ---------------------------------------------------------------------------
# editing refusals
# ---------------------------------------------------------------------------

def test_duplicating_a_job_whose_settings_vanished_reports_why(screen,
                                                               settings):
    """The copy is validated like any other job, and a failure stays inline.

    A settings file can be moved between building the queue and duplicating a
    job in it. Adding an unrunnable copy would put the failure off until the
    night run; refusing it here says which file is gone.
    """
    screen.add_job("mask", settings, label="first")
    screen.select_job(screen.queue().ids[0])
    os.remove(settings)

    assert screen.duplicate_selected() is False
    assert screen.queue().ids == ["mask-1"]
    assert "not found" in screen.problems_text()
    assert "Could not duplicate" in screen.last_error


def test_the_running_job_cannot_be_removed_out_from_under_the_queue(screen,
                                                                    settings):
    """Removing the job that is executing would orphan its process."""
    screen.add_job("mask", settings)
    screen.select_job("mask-1")
    screen.queue().jobs[0].status = bt.STATUS_RUNNING
    screen._busy = True

    assert screen.remove_selected() is False
    assert screen.queue().ids == ["mask-1"]
    assert "stop the queue first" in screen.last_error


def test_a_running_queue_cannot_be_reordered(screen, settings, tmp_path):
    """Reordering mid-run would change what runs next while it is being read."""
    other = _settings_csv(tmp_path, "mask2.csv",
                          src=_plate(tmp_path, "plate2"), cell_channel=0)
    screen.add_job("mask", settings, label="first")
    screen.add_job("mask", other, label="second")
    screen.select_job("mask-2")
    screen._busy = True

    assert screen.move_selected(-1) is False
    assert screen.queue().ids == ["mask-1", "mask-2"]
    assert "stop it before reordering" in screen.last_error


def test_selecting_a_job_that_is_not_in_the_queue_is_refused(screen,
                                                             settings):
    """A stale id from a reloaded queue must not select the wrong row."""
    screen.add_job("mask", settings)

    assert screen.select_job("mask-99") is False


# ---------------------------------------------------------------------------
# the queue file
# ---------------------------------------------------------------------------

def test_a_queue_that_cannot_be_written_says_so_inline(screen, settings,
                                                       tmp_path):
    """An unwritable destination is a status line, never a lost queue."""
    screen.add_job("mask", settings)
    # The parent is an existing FILE, so the folder cannot be created for it.
    target = tmp_path / "mask.csv" / "queue.json"

    assert screen.save_queue_to(str(target)) is False
    assert "Could not save the queue" in screen.last_error
    assert not target.exists()


# ---------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------

def test_a_runner_that_throws_before_the_first_job_releases_the_screen(
        screen, settings, monkeypatch):
    """A crash in the queue runner must not leave the screen stuck busy.

    Every control is disabled while ``_busy`` is set, so a failure that left
    it set would freeze the window with no way to start again.
    """
    screen.add_job("mask", settings)

    def explode(*args, **kwargs):
        raise RuntimeError("no CUDA device")

    monkeypatch.setattr(bt, "run_queue", explode)

    assert screen.run() is False
    assert screen.is_busy() is False
    assert "The queue runner failed: no CUDA device" in screen.last_error


def test_a_worker_error_line_is_reported_without_its_traceback(screen):
    """The last line of a worker traceback is the part worth showing."""
    screen._busy = True
    screen._on_worker_error_text(
        "Traceback (most recent call last):\n  ...\nValueError: bad src")

    assert screen.is_busy() is False
    assert screen.last_error.endswith("ValueError: bad src")


def test_a_worker_that_died_with_no_text_still_reports_a_failure(screen):
    """An empty error string must not read as a successful run."""
    screen._busy = True
    screen._on_worker_error_text("")

    assert screen.is_busy() is False
    assert "unknown error" in screen.last_error


def test_progress_from_a_worker_thread_reaches_the_gui_thread(screen,
                                                              settings, qtbot):
    """Progress emitted off the GUI thread is relayed, not applied in place.

    ``run_queue`` calls back on its own thread; touching the table there is a
    crash rather than a wrong number, so the report hops through a queued
    signal and only then updates the widget.
    """
    import threading

    screen.add_job("mask", settings)
    seen = []
    screen._progress_relayed.connect(lambda p: seen.append(p))
    report = bt.Progress(event="job_started", job_id="mask-1", index=1,
                         total=1, status=bt.STATUS_RUNNING,
                         message="starting mask-1")

    thread = threading.Thread(
        target=lambda: screen._relay_progress(report))
    thread.start()
    thread.join()
    qtbot.waitUntil(lambda: bool(seen), timeout=2000)

    assert seen == [report]
    assert screen.status_text()


# ---------------------------------------------------------------------------
# the table and the log
# ---------------------------------------------------------------------------

def test_a_job_that_was_never_run_says_so_in_words(screen, settings):
    """``not_run`` is a machine word; the table shows the human one."""
    screen.add_job("mask", settings)
    screen.queue().jobs[0].status = bt.STATUS_NOT_RUN
    screen._refresh_table()

    assert screen.row_status(0) == "not run"


def test_the_elapsed_cell_ticks_only_for_the_job_that_is_running(
        screen, settings, tmp_path):
    """One cell per tick: refreshing the table would churn the selection.

    The tick also refreshes the log of the running job when that row is the
    selected one, which is how a night run shows output as it arrives.
    """
    other = _settings_csv(tmp_path, "mask2.csv",
                          src=_plate(tmp_path, "plate2"), cell_channel=0)
    screen.add_job("mask", settings, label="first")
    screen.add_job("mask", other, label="second")
    screen._refresh_table()

    log = tmp_path / "mask-2.log"
    log.write_text("segmenting field 3\n", encoding="utf-8")
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    running = screen.queue().jobs[1]
    running.status = bt.STATUS_RUNNING
    running.started = (now - timedelta(seconds=65)).isoformat()
    running.finished = now.isoformat()
    running.log_path = str(log)
    screen.select_job("mask-2")

    screen._refresh_running_row()

    ticked = bt.fmt_duration(running.elapsed_s)
    assert ticked
    assert screen.row_values(1)[COLUMNS.index("Time")] == ticked
    assert screen.row_values(0)[COLUMNS.index("Time")] != ticked
    assert "segmenting field 3" in screen.log_text()


def test_a_selection_change_with_nothing_selected_does_nothing(screen):
    """An empty table fires this too, and it must not read row -1."""
    screen._on_selection_changed()

    assert screen.selected_job() is None
    assert screen.log_text() == ""


def test_a_log_file_that_cannot_be_read_says_so_in_the_log_pane(
        screen, settings, tmp_path, monkeypatch):
    """An unreadable log is reported in place of the log, not as a crash.

    The log pane is refreshed on a timer, so an exception here would repeat
    once a second for the rest of the night.
    """
    screen.add_job("mask", settings)
    log = tmp_path / "locked.log"
    log.write_text("something", encoding="utf-8")
    job = screen.queue().jobs[0]
    job.log_path = str(log)

    real_open = builtins.open

    def refuse(path, *args, **kwargs):
        if str(path) == str(log):
            raise OSError("permission denied")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", refuse)
    try:
        screen._load_log(job)
    finally:
        monkeypatch.undo()

    assert "could not read" in screen.log_text()
    assert "permission denied" in screen.log_text()


# ---------------------------------------------------------------------------
# the buttons that open a file picker
# ---------------------------------------------------------------------------

def test_the_add_button_reads_the_boxes_beside_it(screen, settings):
    """Add takes what is typed, including the comma-separated lists."""
    screen._module_combo.setCurrentIndex(screen._module_combo.findData("mask"))
    screen._settings_edit.setText(settings)
    screen._label_edit.setText("typed in")
    screen._overrides_edit.setText("cell_channel=0, nucleus_channel=0")

    screen._on_add_clicked()

    assert screen.queue().jobs, screen.problems_text()
    job = screen.queue().jobs[0]
    assert job.module == "mask"
    assert job.label == "typed in"
    assert job.override_args == ["cell_channel=0", "nucleus_channel=0"]


def test_choosing_a_settings_file_fills_the_box(screen, settings,
                                                monkeypatch):
    """The picker only fills the field; nothing is added until Add."""
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (settings, "")))

    screen._pick_settings_file()

    assert screen._settings_edit.text() == settings
    assert screen.queue().jobs == []


def test_a_cancelled_picker_leaves_the_box_alone(screen, monkeypatch):
    """Cancel returns an empty path, which must not blank the field."""
    screen._settings_edit.setText("keep me")
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: ("", "")))

    screen._pick_settings_file()

    assert screen._settings_edit.text() == "keep me"


def test_choosing_a_queue_file_loads_it(screen, settings, tmp_path,
                                        monkeypatch):
    """The Open button goes through the same loader the tests call directly."""
    screen.add_job("mask", settings, label="saved job")
    path = str(tmp_path / "queue.json")
    assert screen.save_queue_to(path) is True
    screen._queue = bt.Queue(name="queue")
    screen._refresh_table()

    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (path, "")))
    screen._pick_queue_to_load()

    assert [job.label for job in screen.queue().jobs] == ["saved job"]


def test_a_cancelled_queue_open_changes_nothing(screen, settings,
                                                monkeypatch):
    """Cancel must not empty a queue the user has been building."""
    screen.add_job("mask", settings)
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: ("", "")))

    screen._pick_queue_to_load()

    assert screen.queue().ids == ["mask-1"]


def test_choosing_where_to_save_writes_the_queue_there(screen, settings,
                                                       tmp_path, monkeypatch):
    """Save-as writes through the same path the run keeps up to date."""
    screen.add_job("mask", settings)
    path = str(tmp_path / "saved.json")
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (path, "")))

    screen._pick_queue_to_save()

    assert os.path.isfile(path)


def test_a_cancelled_save_writes_nothing(screen, settings, tmp_path,
                                         monkeypatch):
    """Cancel is not an empty filename to write to."""
    screen.add_job("mask", settings)
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: ("", "")))

    screen._pick_queue_to_save()

    assert not any(name.endswith(".json") for name in os.listdir(tmp_path))


# ---------------------------------------------------------------------------
# the comma-separated input boxes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ("a, b ,c", ["a", "b", "c"]),
    ("  ", []),
    ("", []),
    (None, []),
    ("a,,b", ["a", "b"]),
])
def test_a_comma_separated_box_drops_the_blanks(text, expected):
    """An empty entry between two commas is a typo, not a dependency."""
    assert _split_list(text) == expected

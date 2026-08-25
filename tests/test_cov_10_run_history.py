"""The run-history screen when the journal, or a run folder, will not answer.

Run history is the only record of what a run actually did, so a screen that
comes up blank after a failed enumeration is worse than one that says why.
The branches here cover a refresh that overlaps another, an enumeration that
raises on the worker thread and inline, a close that happens while a worker
is still winding down, a performance field that is not a number, and the
selection actions that reach outside the application.
"""
from __future__ import annotations

import os
import types

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

from PySide6.QtCore import QObject, Signal                 # noqa: E402
from PySide6.QtGui import QDesktopServices                # noqa: E402

from spacr.qt.screens import run_history as RH            # noqa: E402
from spacr.qt.screens.run_history import RunHistoryScreen  # noqa: E402


def _record(run_id, app_key="mask", status="success", directory="/tmp/runs"):
    return {
        "run_id": run_id,
        "dir": f"{directory}/{run_id}",
        "app_key": app_key,
        "status": status,
        "start_utc": "2026-08-01T10:00:00+00:00",
        "performance": {"wall_s": 12.0, "process_cpu_s": 8.0,
                        "input_files": 2, "input_bytes": 2048,
                        "output_files": 1, "output_bytes": 4096},
        "warnings": [],
        "settings": {"src": "/data"},
        "inputs": {},
        "outputs": {},
        "environment": {},
    }


@pytest.fixture
def screen(qtbot):
    made = RunHistoryScreen(threaded=False)
    qtbot.addWidget(made)
    return made


def test_a_byte_count_that_is_not_a_number_reads_as_unknown():
    """A journal written by an older version can carry a missing or textual
    size. Formatting it as ``0 B`` would claim the run produced nothing."""
    assert RH._bytes(None) == "—"
    assert RH._bytes("not a size") == "—"
    assert RH._bytes(2048) == "2.0 KiB"


def test_a_second_refresh_while_one_is_running_is_ignored(screen,
                                                          monkeypatch):
    """Two enumerations racing would leave the table showing whichever
    finished last rather than the newest journal."""
    calls = []
    monkeypatch.setattr(RH, "search_runs",
                        lambda: calls.append(1) or [_record("r1")])
    screen._busy = True
    screen.refresh()
    assert calls == []
    screen._busy = False
    screen.refresh()
    assert len(calls) == 1


def test_an_enumeration_that_fails_says_so_instead_of_going_blank(
        screen, monkeypatch):
    """An unreadable runs folder must leave a sentence naming the error, not
    an empty table that reads as "you have never run anything"."""
    def _explode():
        raise OSError("runs directory is not readable")

    monkeypatch.setattr(RH, "search_runs", _explode)
    screen.refresh()
    assert screen.records == []
    assert "OSError" in screen.last_error
    assert "Could not load run history" in screen._status.text()


class _InlineWorker(QObject):
    """A worker that emits ``finished`` the way the real one does."""

    finished = Signal(bool)

    def request_cancel(self, reason=""):
        return None


class _InlineThread(QObject):
    """A QThread stand-in that runs the job body on the calling thread.

    The body under test is the closure ``refresh`` hands to ``make_thread``.
    Started on a real QThread it runs outside the tracer, so this drives the
    same closure directly rather than asserting on a mock of it.
    """

    finished = Signal()

    def __init__(self, fn, settings, worker):
        super().__init__()
        self._fn = fn
        self._settings = settings
        self._worker = worker

    def isRunning(self):
        return False

    def start(self):
        # The real job body catches its own failures and reports them through
        # the screen, so the worker's ``ok`` is True whenever the body
        # returned at all. Anything escaping here is a genuine failure and
        # must reach the test rather than be turned into ok=False.
        self._fn(self._settings)
        self._worker.finished.emit(True)
        self.finished.emit()


@pytest.fixture
def inline_threads(monkeypatch):
    """Make ``make_thread`` run the job body inline, signals and all."""
    def _make(fn, settings, app_key="", **kwargs):
        worker = _InlineWorker()
        return _InlineThread(fn, settings, worker), worker

    monkeypatch.setattr(RH, "make_thread", _make)


def test_a_threaded_enumeration_loads_the_records(qtbot, monkeypatch,
                                                  inline_threads):
    """The job body reads the journal and parks the result for the GUI
    thread to apply; the records have to arrive in the table."""
    monkeypatch.setattr(RH, "search_runs", lambda: [_record("r1")])
    made = RunHistoryScreen(threaded=True)
    qtbot.addWidget(made)
    with qtbot.waitSignal(made.history_refreshed, timeout=5000):
        made.refresh()
    assert [r["run_id"] for r in made.records] == ["r1"]
    assert made._table.rowCount() == 1


def test_a_threaded_enumeration_that_raises_reports_the_error(
        qtbot, monkeypatch, inline_threads):
    """An exception inside the job body must be carried back as a message
    rather than escaping the worker, and the table must not keep stale
    rows."""
    def _explode():
        raise RuntimeError("journal is corrupt")

    monkeypatch.setattr(RH, "search_runs", _explode)
    made = RunHistoryScreen(threaded=True)
    qtbot.addWidget(made)
    made.refresh()
    assert made.records == []
    assert "journal is corrupt" in made.last_error
    assert "Could not load run history" in made._status.text()


def test_closing_while_a_worker_refuses_to_cancel_still_releases_it(
        screen, monkeypatch):
    """A worker that will not take a cancel must not stop the screen from
    closing; the ownership pair is released either way."""
    from spacr.qt import bridge

    monkeypatch.setattr(bridge, "drain_thread",
                        lambda *args, **kwargs: None)

    class _Stubborn:
        def request_cancel(self, reason):
            raise RuntimeError("this worker will not cancel")

    screen._jobs.append((object(), _Stubborn()))
    screen.close()
    assert screen._jobs == []


def test_the_module_filter_hides_the_other_modules(screen, monkeypatch):
    """The filter is how a user finds one module's runs among hundreds. A
    row from another module surviving it makes the count meaningless."""
    monkeypatch.setattr(RH, "search_runs",
                        lambda: [_record("r1", app_key="mask"),
                                 _record("r2", app_key="measure")])
    screen.refresh()
    assert screen._table.rowCount() == 2
    screen._module.setCurrentIndex(screen._module.findData("measure"))
    assert screen._table.rowCount() == 1
    assert "Showing 1 of 2" in screen._status.text()


def test_opening_the_selected_run_folder_asks_the_desktop(screen, monkeypatch,
                                                          tmp_path):
    """"Open folder" is the one action that leaves the application. It has to
    hand over the selected run's own directory."""
    opened = []
    monkeypatch.setattr(QDesktopServices, "openUrl",
                        staticmethod(lambda url: opened.append(url) or True))
    monkeypatch.setattr(RH, "search_runs",
                        lambda: [_record("r1", directory=str(tmp_path))])
    screen.refresh()
    screen._open_selected_folder()
    assert len(opened) == 1
    assert opened[0].toLocalFile().endswith("r1")


def test_selecting_a_run_that_is_not_in_the_table_reports_that(screen,
                                                               monkeypatch):
    """A caller that jumps to a run filtered out, or deleted since, needs to
    know nothing was selected rather than acting on whichever row was
    already current."""
    monkeypatch.setattr(RH, "search_runs", lambda: [_record("r1")])
    screen.refresh()
    assert screen.select_run("/tmp/runs/r1") is True
    assert screen.select_run("/tmp/runs/never_ran") is False

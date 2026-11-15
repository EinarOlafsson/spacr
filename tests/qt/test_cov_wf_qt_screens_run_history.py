"""Run History: the paths a real journal takes that the happy path never does.

The dashboard is the only place a user can look at a job that has already
finished, so every one of these branches is the screen answering for a run
that did *not* go well -- a cancelled job, a folder whose manifest never got
written, a screen reopened from the sidebar, a window closed while a scan is
still in flight. Each of them is exercised here against a controlled record
list rather than a real ``runs/`` tree, because the point is what the widget
renders for a given record, not what the journal writer produces.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import QThread

from spacr.qt.screens.run_history import RunHistoryScreen


def _record(run_id, *, app_key="classify", status="success",
            start_utc="2026-08-29T10:11:12+00:00", settings=None,
            dir_path=None, warnings=None):
    """One ``search_runs`` record, shaped exactly as run_journal emits it."""
    return {
        "dir": Path(dir_path) if dir_path is not None
               else Path("/nonexistent/spacr-runs") / run_id,
        "run_id": run_id,
        "app_key": app_key,
        "status": status,
        "start_utc": start_utc,
        "end_utc": "",
        "elapsed_s": 1.5,
        "performance": {
            "wall_s": 1.5, "process_cpu_s": 0.9,
            "input_files": 1, "input_bytes": 2048,
            "output_files": 1, "output_bytes": 4096,
        },
        "settings": {} if settings is None else dict(settings),
        "inputs": {}, "outputs": {}, "models": {},
        "warnings": list(warnings or []),
        "failure": "",
        "environment": {},
        "manifest": {},
    }


@pytest.fixture
def journal(monkeypatch):
    """Stand in for the on-disk journal and count how often it is read."""
    state = {"records": [], "calls": 0}

    def _search():
        state["calls"] += 1
        return list(state["records"])

    monkeypatch.setattr("spacr.qt.screens.run_history.search_runs", _search)
    return state


@pytest.fixture
def make_screen(qtbot, qt_theme_applied):
    """Build a synchronous RunHistoryScreen registered with qtbot."""
    def _make(threaded=False):
        widget = RunHistoryScreen(threaded=threaded)
        qtbot.addWidget(widget)
        return widget
    return _make


def test_a_run_whose_settings_were_lost_is_not_handed_to_a_module(
        make_screen, journal, tmp_path):
    """An empty settings map must not be loaded into a module as if real.

    A run folder whose ``settings.json`` is missing or unparseable still
    lists in the table -- that visibility is the point of the ``corrupt``
    status -- but there is nothing to reload. Emitting the hand-off anyway
    would blank out whatever the user currently has typed into that module's
    form and replace it with nothing. The run beside it, which does have
    settings, still has to hand them over.
    """
    good_dir = tmp_path / "good"
    good_dir.mkdir()
    lost_dir = tmp_path / "lost"
    lost_dir.mkdir()
    journal["records"] = [
        _record("good", settings={"src": "/data/plate1"}, dir_path=good_dir),
        _record("lost", status="corrupt", settings={}, dir_path=lost_dir,
                warnings=["settings.json unreadable"]),
    ]
    screen = make_screen()
    emitted = []
    screen.settings_requested.connect(
        lambda app_key, settings: emitted.append((app_key, settings)))
    screen.refresh()

    assert screen.select_run(lost_dir)
    assert screen._load_settings.isEnabled() is False
    screen._load_selected_settings()
    assert emitted == []

    assert screen.select_run(good_dir)
    assert screen._load_settings.isEnabled() is True
    screen._load_selected_settings()
    assert emitted == [("classify", {"src": "/data/plate1"})]


def test_closing_the_screen_drains_a_job_whose_worker_is_already_gone(
        make_screen, journal, monkeypatch):
    """A pair with no worker must still get its thread drained.

    ``closeEvent`` walks the screen's ownership pairs and cancels each
    worker before draining. A pair can reach it with no worker at all -- the
    worker half is dropped as soon as its thread retires -- and a walk that
    skipped the drain for those pairs would leave the QThread running and
    ownerless, which is precisely the state ``MainWindow.closeEvent`` reads
    as "analysis still running" and refuses to quit on. A worker whose C++
    half is already deleted raises out of ``request_cancel`` and must not
    stop the remaining pairs being drained either.
    """
    cancelled = []
    drained = []

    class _Worker:
        def request_cancel(self, reason):
            cancelled.append(reason)

    class _DeadWorker:
        def request_cancel(self, reason):
            raise RuntimeError("Internal C++ object already deleted")

    def _drain(thread, worker=None, timeout_ms=0):
        drained.append((thread, worker))
        return True

    monkeypatch.setattr("spacr.qt.bridge.drain_thread", _drain)

    journal["records"] = []
    screen = make_screen()
    live, ownerless, dead = QThread(), QThread(), QThread()
    screen._jobs = [(live, _Worker()), (ownerless, None), (dead, _DeadWorker())]

    screen.close()

    assert cancelled == ["run-history screen closed"]
    assert [pair[0] for pair in drained] == [live, ownerless, dead]
    assert drained[1][1] is None
    assert screen._jobs == []

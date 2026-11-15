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

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


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


def test_selecting_a_run_that_is_not_in_the_table_reports_that(screen,
                                                               monkeypatch):
    """A caller that jumps to a run filtered out, or deleted since, needs to
    know nothing was selected rather than acting on whichever row was
    already current."""
    monkeypatch.setattr(RH, "search_runs", lambda: [_record("r1")])
    screen.refresh()
    assert screen.select_run("/tmp/runs/r1") is True
    assert screen.select_run("/tmp/runs/never_ran") is False

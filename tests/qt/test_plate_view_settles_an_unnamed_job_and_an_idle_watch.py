"""Plate View settles a job reported without its id, and an idle watch reads nothing.

A worker that predates job ids reports ``finished`` with id 0; the oldest
job in flight is then the one that settled. A job id no longer in flight is
never stale. The live watch plate, refreshed before any run began, has no
source folder and so reads no ledger.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.qt.screens import plate_view as pv  # noqa: E402
from tests.qt.test_plate_view_job_order import held  # noqa: E402,F401

pytestmark = pytest.mark.qt


def test_a_job_settled_without_its_id_finishes_the_oldest_one(qtbot, held):
    screen = pv.PlateViewScreen(threaded=True)
    qtbot.addWidget(screen)
    applied = []
    screen._run_job(lambda: "first", applied.append)
    screen._run_job(lambda: "second", applied.append)
    fn, box, _worker = held[0]
    fn(box)
    finished = []
    screen.job_finished.connect(finished.append)
    screen._on_job_settled(True)
    assert applied == ["first"] and finished == [True]
    assert len(screen._pending) == 1 and screen.is_busy() is True


def test_a_job_no_longer_in_flight_is_never_stale(qtbot, held):
    screen = pv.PlateViewScreen(threaded=True)
    qtbot.addWidget(screen)
    screen._run_job(lambda: 1, lambda _r: None, supersedes="recompute")
    assert screen._is_stale(999) is False


def test_an_idle_watch_plate_reads_no_ledger(qtbot, monkeypatch):
    plate = pv._WatchLivePlate()
    qtbot.addWidget(plate)
    opened = []
    monkeypatch.setattr("builtins.open", lambda *a, **k: opened.append(a) or pytest.fail("read a ledger"))
    plate.refresh()
    assert opened == [] and not plate.is_active()

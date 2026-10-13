"""Item 641: performance regression guards that cost almost nothing to run.

* ``import spacr`` and ``import spacr.qt`` stay light: no numpy, pandas,
  torch or Qt widgets at import, and a generous wall-clock ceiling.
* Home is built without reading the run journal on the GUI thread, and the
  Totals panel's watermark count opens only the run folders in its window
  (an 11,000-run journal cost 2.8 s and 530 MB at every launch).
* The database browser's next page never waits behind a full-table
  ``COUNT(*)``, and browsing writes no run-journal record.

Counted wherever a count can say it; the two wall-clock ceilings are an
order of magnitude above the measured value so a loaded runner cannot trip
them.
"""
from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import time

import pytest

pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

HEAVY = ("numpy", "pandas", "torch", "scipy", "PySide6.QtWidgets")


@pytest.mark.parametrize("module", ["spacr", "spacr.qt"])
def test_the_package_imports_light(module):
    code = (
        "import sys, time\n"
        "t = time.perf_counter()\n"
        f"import {module}\n"
        "elapsed = time.perf_counter() - t\n"
        f"heavy = [m for m in {HEAVY!r} if m in sys.modules]\n"
        "print(repr((elapsed, heavy)))\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, timeout=120, check=True).stdout
    elapsed, heavy = eval(out.strip().splitlines()[-1])
    assert heavy == [], f"import {module} pulled in {heavy}"
    assert elapsed < 1.5, f"import {module} took {elapsed:.2f} s"


def _journal(root, names, start):
    for i, name in enumerate(names):
        folder = root / name
        folder.mkdir()
        (folder / "manifest.json").write_text(json.dumps(
            {"app_key": "mask", "start_utc": start(i)}))


def test_the_totals_window_opens_only_the_runs_inside_it(tmp_path, monkeypatch):
    from spacr import run_journal
    from spacr.qt.widgets import home

    root = tmp_path / "runs"
    root.mkdir()
    old = [f"2026-01-{d:02d}_120000_{d:08x}__mask" for d in range(1, 29)]
    new = [f"2026-10-04_1{h}0000_{h:08x}__mask" for h in range(3)]
    _journal(root, old, lambda i: f"2026-01-{i + 1:02d}T12:00:00+00:00")
    _journal(root, new, lambda i: f"2026-10-04T1{i}:00:00+00:00")
    monkeypatch.setattr(run_journal, "runs_root", lambda: root)

    opened = []
    real = json.loads

    def counted(text, *a, **k):
        opened.append(1)
        return real(text, *a, **k)

    monkeypatch.setattr(json, "loads", counted)
    counts = home.TotalsPanel._totals_since("2026-10-03T00:00:00+00:00")

    assert counts["total_runs"] == 3
    assert counts["mask_runs"] == 3
    assert len(opened) == 3, f"opened {len(opened)} manifests for 3 runs"


def test_home_panels_can_be_built_without_reading_the_journal(qtbot,
                                                              monkeypatch):
    from spacr.qt.widgets import home

    def refuse(self):
        raise AssertionError("read the run journal on the GUI thread")

    monkeypatch.setattr(home.RecentRunsPanel, "read", refuse)
    monkeypatch.setattr(home.TotalsPanel, "read", refuse)
    recent = home.RecentRunsPanel(read_now=False)
    totals = home.TotalsPanel(read_now=False)
    qtbot.addWidget(recent)
    qtbot.addWidget(totals)
    assert not totals._reset.isEnabled()


@pytest.fixture
def wide_db(tmp_path):
    path = tmp_path / "measurements.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE cell (a INTEGER, b REAL, c TEXT)")
    con.executemany("INSERT INTO cell VALUES (?, ?, ?)",
                    [(i, i * 0.5, f"r{i}") for i in range(1000)])
    con.commit()
    con.close()
    return path


def test_the_next_page_does_not_wait_for_the_count(qtbot, qt_theme_applied,
                                                   wide_db, tmp_path,
                                                   monkeypatch):
    from spacr import run_journal
    from spacr.qt.screens.db_browser import DbBrowserScreen, ReadOnlyDb

    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(run_journal, "runs_root", lambda: runs)
    real_count = ReadOnlyDb.count

    def slow_count(self, *a, **k):
        time.sleep(6.0)
        return real_count(self, *a, **k)

    monkeypatch.setattr(ReadOnlyDb, "count", slow_count)
    w = DbBrowserScreen(threaded=True)
    qtbot.addWidget(w)
    w.set_database(str(wide_db))
    w.select_table("cell")
    qtbot.waitUntil(lambda: w.loaded_rows() > 0, timeout=20000)
    first = w.loaded_rows()

    asked = time.perf_counter()
    qtbot.waitUntil(lambda: w.fetch_more() or w.loaded_rows() > first,
                    timeout=20000)
    qtbot.waitUntil(lambda: w.loaded_rows() > first, timeout=20000)
    waited = time.perf_counter() - asked

    assert waited < 4.0, f"the next page waited {waited:.1f} s for COUNT(*)"
    qtbot.waitUntil(lambda: w.active_jobs() == 0, timeout=30000)
    assert list(runs.iterdir()) == [], "browsing wrote run-journal records"
    w.close()


def test_a_quickened_log_file_skips_the_stat_and_still_rolls_over(
        tmp_path, monkeypatch):
    import logging
    import logging.handlers
    import os

    from spacr.logging_util import _quicken

    handler = _quicken(logging.handlers.RotatingFileHandler(
        tmp_path / "run.log", maxBytes=200_000, backupCount=1,
        encoding="utf-8"))
    log = logging.getLogger("spacr.test641")
    log.propagate = False
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    stats = []
    real = os.path.isfile
    monkeypatch.setattr(os.path, "isfile",
                        lambda p: stats.append(p) or real(p))
    try:
        for i in range(200):
            log.info("field %d measured", i)
        assert stats == [], "the log file was statted on every record"
        for i in range(8000):
            log.info("field %d measured %s", i, "x" * 40)
    finally:
        log.removeHandler(handler)
        handler.close()
    assert (tmp_path / "run.log.1").exists()
    assert (tmp_path / "run.log").stat().st_size <= 200_000


def test_a_burst_of_output_reaches_the_gui_in_few_chunks_and_in_order(qtbot):
    from spacr.qt.bridge import PipelineWorker

    def chatty(_settings):
        for i in range(3000):
            print(f"Progress: {i}/3000, operation_type: test")

    worker = PipelineWorker(chatty, {}, journal=False, capture_figures=False)
    chunks = []
    worker.line_ready.connect(chunks.append)
    worker.run()
    text = "".join(chunks)
    assert text.splitlines() == [
        f"Progress: {i}/3000, operation_type: test" for i in range(3000)]
    assert len(chunks) < 300, f"{len(chunks)} emissions for 3000 lines"

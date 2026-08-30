"""Annotator Agreement — the bookkeeping corners a normal compute never turns.

``tests/qt/test_agreement_screen.py`` drives the screen the way a user does:
open a database, tick two columns, read κ and the disagreement list. The
paths below are the ones that only happen when one of that story's
assumptions is false, and each of them is load-bearing:

* a **report that came back with no pairs at all**. Nothing on screen may
  claim a pairing that does not exist — a stale "alice vs bob" entry left in
  the confusion selector would offer the user a matrix for a comparison this
  report never made;
* **resolving a crop before any database is open**. ``_resolve_crop``'s
  fallback rebases a relative crop path on the run folder, which it derives
  from the database path. With no database there is nothing to rebase
  against, and reaching for it anyway would raise inside a preview whose
  whole contract is "a missing picture is not an error";
* the **thread bookkeeping when more than one compute is outstanding**. The
  sweep must retire only the threads that have stopped, and retiring a job
  that is not the current one must not blank the current one's handles.
  Getting either wrong is a QThread collected while running (a hard process
  crash) or a screen that reports itself busy forever;
* a **worker traceback with no text in it**. The screen quotes the last
  non-empty line of a failed worker's traceback inline; when there is no
  line to quote the failure still has to be reported, or the previous
  success message stays up and the user reads a stale κ as a fresh one.
"""
from __future__ import annotations

import dataclasses
import os
import sqlite3

import pytest
from PySide6.QtGui import QImage

from spacr.qt.screens.agreement import AgreementScreen

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_META = ("png_path", "file_name", "plateID", "rowID", "columnID", "fieldID",
         "prcfo", "cell_id")

#: 8 rows both annotators reached, 6 of them agreeing.
_ROWS = [(1, 1), (1, 1), (1, 2), (2, 2), (2, 1), (1, 1), (2, 2), (2, 2)]


@pytest.fixture
def run_folder(tmp_path):
    """A real ``<src>/measurements/measurements.db`` with two annotators.

    Built the way ``spacr.qt.annotate_engine`` writes annotations — one
    INTEGER column per pass on ``png_list`` — so the screen computes a real
    report rather than one the test made up.
    """
    src = tmp_path / "run"
    meas = src / "measurements"
    pngs = src / "cell_png"
    meas.mkdir(parents=True)
    pngs.mkdir(parents=True)
    db_path = meas / "measurements.db"

    crops = []
    for i in range(len(_ROWS)):
        crop = pngs / f"plate1_A01_1_{i}.png"
        image = QImage(8, 8, QImage.Format_RGB32)
        image.fill(0xFF3366AA)
        assert image.save(str(crop), "PNG")
        crops.append(str(crop))

    con = sqlite3.connect(db_path)
    try:
        con.execute("CREATE TABLE png_list (%s)"
                    % ", ".join(f'"{c}" TEXT' for c in _META))
        for col in ("alice", "bob"):
            con.execute(f'ALTER TABLE png_list ADD COLUMN "{col}" INTEGER')
        placeholders = ", ".join("?" * (len(_META) + 2))
        con.executemany(
            f"INSERT INTO png_list VALUES ({placeholders})",
            [(crops[i], os.path.basename(crops[i]), "plate1", "r1", "c1",
              "f1", f"plate1_A01_1_o{i}", f"o{i}", *labels)
             for i, labels in enumerate(_ROWS)])
        con.commit()
    finally:
        con.close()
    return {"src": str(src), "db": str(db_path), "crops": crops}


@pytest.fixture
def screen(qtbot, qt_theme_applied):
    """A synchronous screen — the compute runs inline, so assertions are exact."""
    w = AgreementScreen(threaded=False)
    qtbot.addWidget(w)
    return w


class _FakeThread:
    """A stand-in for a QThread whose running state the test decides.

    Real threads race, and these paths are about *which* of several jobs the
    bookkeeping touches — that has to be pinned exactly. Same injection idiom
    as ``tests/qt/test_cov_wf_qt_screens_db_browser.py``.
    """

    def __init__(self, running: bool):
        self._running = running
        self.calls = []

    def isRunning(self):  # noqa: N802 - QThread's spelling
        return self._running

    def quit(self):
        self.calls.append("quit")

    def wait(self, msecs):
        self.calls.append(("wait", msecs))
        return True


# ---------------------------------------------------------------------------
# A report with no pairs in it
# ---------------------------------------------------------------------------

def test_a_report_with_no_pairs_leaves_no_pair_to_choose(screen, run_folder):
    """The confusion selector may not offer a comparison the report never made.

    The pair combo is how the user asks for one pair's confusion matrix, and
    every entry in it is a promise that ``report.pairs`` has that pair. A
    report can arrive with an empty ``pairs`` list — a column that turned out
    to be entirely NULL collapses the pairing — and if the combo were filled
    unconditionally, or simply left holding the previous run's entries, the
    user would pick "alice vs bob" and be shown numbers from a report that
    has been replaced. Both halves are driven here: the same screen first
    renders a real two-annotator report, then the degenerate one.
    """
    assert screen.set_database(run_folder["db"]) is True
    assert screen.selected_columns() == ["alice", "bob"]
    assert screen.compute() is True

    # A report that does have a pair fills the combo and the matrix.
    assert screen._pair_combo.count() == 1
    assert screen._pair_combo.itemText(0) == "alice vs bob"
    assert len(screen.kappa_rows()) == 1
    assert screen.kappa_rows()[0][:2] == ["alice", "bob"]
    assert screen.confusion_rows(), "a real pair must render a matrix"

    real = screen.report()
    assert real is not None and len(real.pairs) == 1
    pairless = dataclasses.replace(real, pairs=[])

    screen._apply_result({"report": pairless,
                          "disagreements": screen._disagreements})

    assert screen._pair_combo.count() == 0
    assert screen.kappa_rows() == []
    assert screen.report() is pairless
    # The overall summary is still reported: no pairs is not a failure.
    assert "Overall" in screen.summary_text()
    assert screen.last_error == ""


# ---------------------------------------------------------------------------
# Resolving a crop path with no database open
# ---------------------------------------------------------------------------

def test_a_crop_path_cannot_be_rebased_without_a_database(screen, run_folder):
    """A crop lookup before a database is open must answer, not raise.

    ``png_list`` stores absolute paths from the machine that ran Measure, so
    a copied dataset resolves them a second way: strip the leading separator
    and rebase on the run folder, which is the database's grandparent. That
    fallback needs a database path to derive the folder from. Asked to
    resolve a path while no database is open — the state the screen starts in
    — it has nothing to rebase against and must simply report "not found";
    ``os.path.dirname`` of an empty string is ``''`` and the join would
    silently look in the process's current directory instead.
    """
    assert screen.database_path() == ""

    existing = run_folder["crops"][0]
    assert screen._resolve_crop(existing) == existing        # absolute, on disk
    assert screen._resolve_crop("cell_png/gone.png") is None  # nothing to rebase
    assert screen._resolve_crop(None) is None

    # With the database open the very same relative path does resolve, which
    # is what proves the None above came from the missing database and not
    # from the path being unusable.
    assert screen.set_database(run_folder["db"]) is True
    relative = os.path.relpath(existing, run_folder["src"])
    assert screen._resolve_crop(relative) == os.path.join(
        run_folder["src"], relative)
    assert screen._resolve_crop("cell_png/gone.png") is None


# ---------------------------------------------------------------------------
# Thread bookkeeping with more than one job outstanding
# ---------------------------------------------------------------------------

def test_the_sweep_retires_the_stopped_job_and_keeps_the_running_one(
        qtbot, qt_theme_applied):
    """A QThread still running must survive the sweep, or the process dies.

    ``_retire_finished_jobs`` runs on every ``thread.finished`` and walks the
    whole job list, because the emitter may already be gone by the time it
    runs. Two computes can be outstanding at once (the user hits Compute
    again while the first is winding down), and dropping the screen's last
    reference to a QThread that has not stopped lets Python collect it under
    Qt's feet — an immediate abort, not an exception. Equally, failing to
    drop the stopped one leaves ``active_jobs()`` above zero forever and
    every ``waitUntil(active_jobs() == 0)`` sits there until it times out.
    """
    w = AgreementScreen(threaded=True)
    qtbot.addWidget(w)
    alive, stopped = _FakeThread(True), _FakeThread(False)
    alive_worker, stopped_worker = object(), object()
    w._jobs = [(alive, alive_worker), (stopped, stopped_worker)]
    w._thread, w._worker = alive, alive_worker
    try:
        w._retire_finished_jobs()

        assert w._jobs == [(alive, alive_worker)]
        assert w.active_jobs() == 1
        assert w._thread is alive
        assert w._worker is alive_worker
    finally:
        w._jobs = []
        w._thread = w._worker = None


def test_retiring_an_older_job_does_not_blank_the_current_one(
        qtbot, qt_theme_applied):
    """The newest compute must keep its handles when an older job retires.

    ``_thread``/``_worker`` name the *most recent* job; ``_jobs`` owns every
    one still winding down. When an older thread finishes, its entry has to
    leave ``_jobs`` while the current job's handles stay exactly where they
    are — clearing them unconditionally would drop the only Python reference
    to a QThread that is still running, which takes the process down. Both
    directions are driven here: the older job first, then the current one,
    whose retirement *does* have to clear the handles so a finished thread is
    not kept alive forever.
    """
    w = AgreementScreen(threaded=True)
    qtbot.addWidget(w)
    older, current = _FakeThread(False), _FakeThread(False)
    older_worker, current_worker = object(), object()
    w._jobs = [(older, older_worker), (current, current_worker)]
    w._thread, w._worker = current, current_worker
    try:
        w._retire_job(older)

        assert w._jobs == [(current, current_worker)]
        assert w.active_jobs() == 1
        assert w._thread is current
        assert w._worker is current_worker

        w._retire_job(current)

        assert w._jobs == []
        assert w.active_jobs() == 0
        assert w._thread is None
        assert w._worker is None
    finally:
        w._jobs = []
        w._thread = w._worker = None


# ---------------------------------------------------------------------------
# A worker traceback with nothing quotable in it
# ---------------------------------------------------------------------------

def test_a_blank_worker_traceback_still_reports_the_failure(
        screen, run_folder):
    """A failure with no text must not leave the previous success on screen.

    ``_on_worker_error_text`` shows the last non-empty line of the worker's
    traceback inline — never a dialog, which would hang a headless run. A
    worker that dies without producing any text (killed mid-write, or an
    exception whose own formatting failed) leaves no line to quote at all.
    The status must still say the agreement failed and the κ table must be
    torn down: leaving the previous run's "Cohen's κ +0.500" up would tell
    the user this database was scored when it never was.
    """
    assert screen.set_database(run_folder["db"]) is True
    assert screen.compute() is True
    assert len(screen.kappa_rows()) == 1
    assert "Cohen" in screen.status_text()

    # A traceback with text names its last line.
    screen._on_worker_error_text(
        "Traceback (most recent call last):\n"
        "  File \"x.py\", line 1\n"
        "sqlite3.DatabaseError: file is not a database\n")
    assert screen.status_text() == (
        "Agreement failed: sqlite3.DatabaseError: file is not a database")
    assert screen.report() is None
    assert screen.kappa_rows() == []

    # And a traceback that is nothing but whitespace still fails loudly.
    assert screen.compute() is True
    assert len(screen.kappa_rows()) == 1

    screen._on_worker_error_text("   \n  \n ")
    assert screen.status_text() == "Agreement failed: "
    assert screen.last_error == "Agreement failed: "
    assert screen.report() is None
    assert screen.kappa_rows() == []
    assert screen.summary_text() == ""

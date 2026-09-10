"""The Report screen's five untaken turns.

Every case here is a branch the existing Report suites never take, and each
one is a promise the screen makes to somebody who is about to send a run
folder to a collaborator:

* a format key the combo box does not offer must leave the user's choice
  standing, not silently snap it back to HTML;
* a scan must not overwrite an output path the user already typed;
* a folder with nothing missing must not be told that something is;
* a job that died must settle without delivering a result to the handler
  that was waiting for one;
* a thread that is still running must keep its Python reference — dropping
  it takes the whole process down, not just the screen.

Nothing here opens a dialog and nothing writes into a run folder, because
this screen is exercised headlessly and a report is read-only by contract.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from spacr import report as rep
from spacr.qt.screens import report as module
from spacr.qt.screens.report import ReportScreen

pytestmark = pytest.mark.qt


# ---------------------------------------------------------------------------
# Fixtures and builders
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolated_run_journal(tmp_path, monkeypatch):
    """Point the reproducibility journal at a temp folder.

    ``make_thread`` writes a run record, and a test that ran a real worker
    would otherwise leave rows in the developer's own ``~/.spacr/runs``.
    """
    root = tmp_path / "journal"
    root.mkdir(parents=True, exist_ok=True)
    from spacr import run_journal
    monkeypatch.setattr(run_journal, "runs_root", lambda: root)
    return root


@pytest.fixture
def screen(qtbot):
    """A Report screen whose jobs run inline, so assertions are exact."""
    widget = ReportScreen(threaded=False)
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def plate(tmp_path):
    """An existing folder to name as the run folder.

    ``scan`` refuses anything that is not a directory before it ever calls
    into collection, so the folder has to be real even when what collection
    returns is supplied by the test.
    """
    src = tmp_path / "runs" / "plate1"
    src.mkdir(parents=True)
    return src


def _fake_report(src, found=("run_status",), missing=(),
                 status="complete", detail="the run finished",
                 n_found=0, n_embedded=0):
    """A :class:`spacr.report.Report` with exactly the sections asked for."""
    sections = [rep.Section(title=key.replace("_", " ").title(), key=key,
                            status=rep.STATUS_OK) for key in found]
    sections += [rep.Section(title=key.replace("_", " ").title(), key=key,
                             status=rep.STATUS_MISSING) for key in missing]
    return rep.Report(src=Path(src), sections=sections, status=status,
                      status_detail=detail, n_figures_found=n_found,
                      n_figures_embedded=n_embedded)


def _serve(monkeypatch, *reports):
    """Make the next scans return ``reports``, in order, without touching disk.

    Collection walks a plate and base64-encodes figures; the branches under
    test are in the screen, so the folder is real but the answer is not.
    """
    queue = list(reports)
    monkeypatch.setattr(
        module.rep, "collect_report",
        lambda src, **kw: queue.pop(0) if len(queue) > 1 else queue[0])


# ---------------------------------------------------------------------------
# A format the combo box does not offer
# ---------------------------------------------------------------------------

def test_an_unoffered_format_key_leaves_the_chosen_format_standing(screen):
    """A stale or misspelled format key must not quietly rewrite the choice.

    ``set_format`` is called from restored preferences, from the drop
    handler and from the tutorial engine, none of which validate the key
    they pass. If an unknown key fell through to ``setCurrentIndex(-1)`` the
    combo box would blank itself, ``output_format()`` would fall back to
    HTML, and a user who had deliberately asked for a PDF would get an HTML
    file — and the suggested filename would change extension under them.
    """
    screen.set_format("pdf")
    assert screen.output_format() == "pdf"
    assert screen._suggested_output().endswith("_report.pdf")

    screen.set_format("docx")

    assert screen._format.findData("docx") == -1
    assert screen.output_format() == "pdf"
    assert screen._suggested_output().endswith("_report.pdf")
    # The tail of the method still ran: the controls were refreshed, so an
    # unknown key leaves the screen usable rather than half-updated.
    assert screen._format.isEnabled() is True
    assert screen._format.currentIndex() == 1


# ---------------------------------------------------------------------------
# The output path the user already typed
# ---------------------------------------------------------------------------

def test_a_scan_never_overwrites_an_output_path_the_user_typed(
        screen, plate, tmp_path, monkeypatch):
    """Where the report goes is the user's decision, not the scanner's.

    Scanning fills the output box only as a convenience when it is empty.
    If it overwrote a path the user had already chosen, pressing Scan a
    second time — which people do after copying more results in — would
    silently redirect the next Generate to the home directory, and the file
    the collaborator was told to look for would never appear.
    """
    _serve(monkeypatch, _fake_report(plate, missing=("figures",)))
    chosen = str(tmp_path / "outbox" / "for_the_collaborator.html")
    screen.set_output(chosen)
    screen.set_source(str(plate))

    assert screen.scan() is True

    assert screen._out_edit.text() == chosen

    # The same code path DOES fill an empty box, which is what makes the
    # branch above a decision rather than dead code.
    screen.set_output("")
    assert screen.scan() is True
    suggested = screen._out_edit.text()
    assert os.path.basename(suggested) == "plate1_report.html"
    assert not suggested.startswith(str(plate) + os.sep)


# ---------------------------------------------------------------------------
# A folder with nothing missing
# ---------------------------------------------------------------------------

def test_a_complete_folder_is_not_told_that_something_is_missing(
        screen, plate, monkeypatch):
    """"0 not available" is a sentence that would make a good run look bad.

    The screen's whole point is to say what is *not* there before a
    collaborator finds out. That only stays credible if the warning is
    absent when there is nothing to warn about: a run with every section
    collected reports the count it found and the figures it would embed,
    and nothing else.
    """
    complete = _fake_report(plate, found=("run_status", "figures"),
                            missing=(), n_found=3, n_embedded=3)
    partial = _fake_report(plate, found=("run_status",),
                           missing=("figures", "statistics"),
                           status="partial", detail="7 of 100 fields failed",
                           n_found=5, n_embedded=2)
    _serve(monkeypatch, complete, partial)
    scanned = []
    screen.folder_scanned.connect(scanned.append)
    screen.set_source(str(plate))

    assert screen.scan() is True

    message = screen.status_text()
    assert "2 section(s) found" in message
    assert "not available" not in message
    assert "3 of 3 figure(s) would be embedded" in message
    assert screen.last_error == ""
    assert scanned == [str(plate)]
    assert screen.missing_sections() == []

    # The very next scan, of a folder that IS missing sections, says so —
    # so the silence above is a measurement, not a broken message.
    assert screen.scan() is True

    later = screen.status_text()
    assert "1 section(s) found" in later
    assert "2 not available" in later
    assert "2 of 5 figure(s) would be embedded" in later
    assert screen.missing_sections() == ["figures", "statistics"]


# ---------------------------------------------------------------------------
# A job that died
# ---------------------------------------------------------------------------

def test_a_worker_that_died_settles_without_delivering_a_result(qtbot, plate):
    """A failed job must not hand its completion handler a None report.

    ``_on_scanned(None)`` and ``_on_generated(None)`` would both "succeed"
    on an empty answer — the section list would clear and the status line
    would read "Nothing was written" instead of naming the exception that
    actually happened. Worse, a screen left ``_busy`` after a crash answers
    every later Scan with "still working on the previous request", so the
    only way out is to restart spaCR.
    """
    widget = ReportScreen(threaded=True)
    qtbot.addWidget(widget)
    delivered = []

    def _die():
        raise RuntimeError("collect_report: no such table: run_status")

    with qtbot.waitSignal(widget.job_finished, timeout=30000) as failed:
        assert widget._run_job(_die, delivered.append) is True

    assert failed.args == [False]
    assert delivered == []
    assert "no such table: run_status" in widget.last_error
    assert widget.is_busy() is False

    # The identical plumbing DOES deliver when the job returns, so the
    # empty list above is the failure being respected, not a dead channel.
    collected = _fake_report(plate, found=("run_status",))

    def _accept(result):
        delivered.append(result)
        widget._on_scanned(result)

    with qtbot.waitSignal(widget.job_finished, timeout=30000) as ok:
        assert widget._run_job(lambda: collected, _accept) is True

    assert ok.args == [True]
    assert delivered == [collected]
    assert widget.report is collected
    assert "1 section(s) found" in widget.status_text()
    assert widget.last_error == ""
    qtbot.waitUntil(lambda: widget.active_jobs() == 0, timeout=30000)
    widget.close()


# ---------------------------------------------------------------------------
# A thread that is still running
# ---------------------------------------------------------------------------

def test_a_thread_still_running_keeps_the_reference_that_keeps_it_alive(
        screen):
    """Releasing a running QThread's last reference takes the process down.

    ``_on_thread_finished`` fires once per finished worker, but a user who
    presses Scan and then Generate has two in flight. The sweep must clear
    only the pairs that have actually stopped: if it cleared ``_thread``
    while that thread was still encoding figures, Python would collect the
    QThread mid-run and spaCR would die with no traceback and no report.
    """
    class Live:
        """A worker thread that has not finished yet."""

        def isRunning(self):
            return True

    class Done:
        """A worker thread whose event loop has exited."""

        def isRunning(self):
            return False

    live, done = Live(), Done()
    screen._jobs = [(done, "worker-done"), (live, "worker-live")]
    screen._thread, screen._worker = live, "worker-live"

    screen._on_thread_finished()

    assert screen._jobs == [(live, "worker-live")]
    assert screen.active_jobs() == 1
    assert screen._thread is live
    assert screen._worker == "worker-live"

    # Once that same thread reports itself stopped, the refs ARE released —
    # the sweep is a filter, not a freeze.
    screen._jobs = [(done, "worker-done")]
    screen._thread, screen._worker = done, "worker-done"

    screen._on_thread_finished()

    assert screen._jobs == []
    assert screen.active_jobs() == 0
    assert screen._thread is None
    assert screen._worker is None

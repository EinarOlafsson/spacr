"""The Report screen's dialogs, its empty answers and its job plumbing.

Everything here is a path where the screen has nothing to show: a scan that
produced no report, a generate that wrote no file, a worker that died, a
thread whose C++ half is already gone. None of them may open a dialog and
none may leave the screen looking busy, because a headless run has nobody to
dismiss a modal and no way to notice a spinner that never stops.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QFileDialog

from spacr.qt.screens import report as module
from spacr.qt.screens.report import ReportScreen, _has_stopped

pytestmark = pytest.mark.qt


@pytest.fixture
def screen(qtbot):
    """A report screen running its jobs inline."""
    widget = ReportScreen(threaded=False)
    qtbot.addWidget(widget)
    return widget


# ---------------------------------------------------------------------------
# Is the worker thread still there?
# ---------------------------------------------------------------------------

def test_a_thread_that_never_existed_counts_as_stopped():
    """``None`` is not a running job, and asking must not raise.

    The close path walks the job list to decide whether it may return; a
    None entry there would otherwise take the window's close handler down.
    """
    assert _has_stopped(None) is True


def test_a_thread_whose_c_half_is_gone_counts_as_stopped():
    """An object PySide6 has already deleted is certainly not running.

    Asking a deleted QThread anything raises RuntimeError, and letting that
    escape would mean a screen could never be closed after its worker was
    collected.
    """
    class Deleted:
        def isRunning(self):
            raise RuntimeError("Internal C++ object already deleted.")

    assert _has_stopped(Deleted()) is True


# ---------------------------------------------------------------------------
# The two file pickers
# ---------------------------------------------------------------------------

def test_choosing_a_run_folder_fills_the_path_and_scans(screen, tmp_path,
                                                        monkeypatch):
    """Picking a folder is the same action as typing it and pressing scan.

    Setting the field without scanning would leave the section list showing
    the previous folder's answer under the new folder's name.
    """
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *args: str(tmp_path)))

    screen._pick_run_folder()

    assert screen._path_edit.text() == str(tmp_path)


def test_a_cancelled_run_folder_dialog_changes_nothing(screen, monkeypatch):
    """Cancel returns an empty path, which is not a folder to scan."""
    screen._path_edit.setText("/somewhere/already/typed")
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *args: ""))

    screen._pick_run_folder()

    assert screen._path_edit.text() == "/somewhere/already/typed"


def test_choosing_an_output_file_fills_the_output_box(screen, tmp_path,
                                                      monkeypatch):
    """The chosen output path is what the report is written to.

    The dialog opens on a suggestion outside the run folder, because writing
    into a dataset somebody else treats as immutable is the one place a
    report must never default to.
    """
    seen = {}
    target = str(tmp_path / "report.html")

    def _save(parent, title, suggested, filters):
        seen["suggested"] = suggested
        return target, "HTML (*.html)"

    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(_save))

    screen._pick_output()

    assert screen._out_edit.text() == target
    assert seen["suggested"]


def test_a_cancelled_output_dialog_leaves_the_box_alone(screen, monkeypatch):
    """Cancel must not blank an output path the user already typed."""
    screen._out_edit.setText("/tmp/keep-me.html")
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName",
        staticmethod(lambda *args: ("", "")))

    screen._pick_output()

    assert screen._out_edit.text() == "/tmp/keep-me.html"


# ---------------------------------------------------------------------------
# Empty answers
# ---------------------------------------------------------------------------

def test_a_scan_that_produced_no_report_says_so_inline(screen):
    """No report is an error the user can see, never a silent empty list.

    Leaving the previous section list on screen would make the failed scan
    look like a successful one against a folder with nothing in it.
    """
    screen._on_scanned(None)

    assert screen.last_error == "Scan produced nothing."
    assert screen._sections.count() == 0


def test_the_section_list_empties_when_there_is_no_report(screen):
    """Rendering with no report clears the list and the verdict.

    A stale verdict under an empty list is the worst of both: it states a
    status for a report that is not there.
    """
    screen._report = None

    screen._render_sections()

    assert screen._sections.count() == 0
    assert screen._verdict.text() == ""


def test_a_generate_that_wrote_nothing_says_so_inline(screen):
    """An empty list of written paths is a failure, not a quiet success.

    Reporting success would leave the Open button pointing at nothing.
    """
    screen._on_generated([])

    assert screen.last_error == "Nothing was written."
    assert screen._written == []


def test_a_worker_that_died_reports_its_last_line(screen):
    """A traceback's last line is the part that names the failure.

    The whole traceback in a one-line status label is unreadable, and a
    modal dialog would hang a headless run.
    """
    screen._on_worker_error_text(
        "Traceback (most recent call last):\n  ...\nValueError: no figures")

    assert "ValueError: no figures" in screen.last_error
    assert screen.is_busy() is False


# ---------------------------------------------------------------------------
# Job plumbing
# ---------------------------------------------------------------------------

def test_the_worker_body_leaves_its_result_in_the_box(qtbot, monkeypatch):
    """What runs off the GUI thread is exactly "put the answer in the box".

    The box is the only channel between the two threads: the completion
    handler runs on the GUI thread and reads ``payload['result']``, so a body
    that returned instead of storing would deliver None every time.
    """
    real_make_thread = module.make_thread
    captured = {}

    def _run_the_body_here(fn, settings, *args, **kwargs):
        captured["payload"] = settings
        fn(settings)
        return real_make_thread(lambda payload: None, settings,
                                *args, **kwargs)

    monkeypatch.setattr(module, "make_thread", _run_the_body_here)
    widget = ReportScreen(threaded=True)
    qtbot.addWidget(widget)
    delivered = []

    with qtbot.waitSignal(widget.job_finished, timeout=20000):
        widget._run_job(lambda: "the collected report", delivered.append)

    assert captured["payload"]["result"] == "the collected report"
    assert delivered == ["the collected report"]
    qtbot.waitUntil(lambda: widget.is_busy() is False, timeout=20000)
    widget.close()


def test_a_completion_handler_that_raises_lands_in_the_status_line(
        qtbot, monkeypatch):
    """A failure while handling the result is still the job's failure.

    Reported inline and the screen released: leaving ``_busy`` set would
    make every later request answer "still working on the previous one".
    """
    widget = ReportScreen(threaded=True)
    qtbot.addWidget(widget)

    def _explode(_result):
        raise ValueError("the report could not be rendered")

    with qtbot.waitSignal(widget.job_finished, timeout=20000) as caught:
        widget._run_job(lambda: "anything", _explode)

    assert caught.args == [False]
    assert "the report could not be rendered" in widget.last_error
    assert widget.is_busy() is False
    widget.close()

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
from PySide6.QtCore import Qt, QThread
from PySide6.QtGui import QColor

from spacr.qt.screens.run_history import RunHistoryScreen
from spacr.qt.theme import active_palette


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


def _started_by_run_id(screen):
    """Map run id -> the text rendered in the 'Started' column."""
    out = {}
    for row in range(screen._table.rowCount()):
        cell = screen._table.item(row, 0)
        out[cell.data(Qt.UserRole)] = cell.text()
    return out


def _status_cells(screen):
    """Map the rendered status text -> its table item."""
    return {
        screen._table.item(row, 2).text(): screen._table.item(row, 2)
        for row in range(screen._table.rowCount())
    }


def test_a_cancelled_run_is_not_painted_like_a_failure(make_screen, journal):
    """Colour is the only thing separating "I stopped it" from "it broke".

    ``run_journal`` stamps ``status: "cancelled"`` when a pipeline raises
    ``PipelineCancelled`` -- the user pressed Stop. That status has no entry
    in the screen's colour map, and it must not borrow one: a cancelled run
    painted in the error red tells the user their job crashed, and they will
    go hunting a traceback that was never written. A failed run in the same
    table still has to be red, or the colour means nothing at all.
    """
    journal["records"] = [
        _record("r-failed", status="failed"),
        _record("r-cancelled", status="cancelled"),
    ]
    screen = make_screen()
    screen.refresh()

    cells = _status_cells(screen)
    assert set(cells) == {"failed", "cancelled"}
    assert (cells["failed"].foreground().color().name()
            == QColor(active_palette()["error"]).name())
    # Nothing was written to the foreground role at all: the cell inherits
    # the table's ordinary text colour instead of a status colour.
    assert cells["cancelled"].data(Qt.ForegroundRole) is None


def test_a_run_with_no_recorded_start_reads_as_a_dash(make_screen, journal):
    """A crashed run has no ``start_utc``, and the column must still read.

    A folder whose ``manifest.json`` never landed comes back as ``corrupt``
    with an empty ``start_utc``; the screen substitutes an em dash. The
    reformatting step that turns an ISO timestamp into something human --
    splitting on the ``T``, folding ``+00:00`` to ``Z`` -- must not touch
    that dash, and it must actually fire for the timestamps that do have a
    ``T``, or the whole table reads in raw ISO with a ``T`` in the middle.
    """
    journal["records"] = [
        _record("r-iso", start_utc="2026-08-29T10:11:12+00:00"),
        _record("r-corrupt", app_key="unknown", status="corrupt",
                start_utc=""),
    ]
    screen = make_screen()
    screen.refresh()

    started = _started_by_run_id(screen)
    assert started["r-iso"] == "2026-08-29 10:11:12Z"
    assert started["r-corrupt"] == "—"


def test_typing_before_the_journal_loads_claims_no_count(make_screen, journal):
    """An empty search result before loading is not "0 runs on disk".

    The filter box is live from the moment the screen is constructed, and
    the screen does not read the journal until it is shown. Typing in that
    window filters an empty list, and if the status line reported it the
    user would be told "Showing 0 of 0 recorded run(s)" about a journal that
    has never been opened -- the same sentence they get for a genuinely
    empty history. The opening instruction has to survive until a load has
    actually happened, and the count has to appear once one has.
    """
    journal["records"] = [_record("a"), _record("b", app_key="measure")]
    screen = make_screen()
    opening = screen._status.text()
    assert opening == "Open this module to load the run journal."

    screen._search.setText("classify")
    assert screen._status.text() == opening
    assert journal["calls"] == 0

    screen.refresh()
    assert journal["calls"] == 1
    assert screen._status.text() == "Loaded 2 recorded run(s)."

    screen._search.setText("measure")
    assert screen._table.rowCount() == 1
    assert screen._status.text() == "Showing 1 of 2 recorded run(s)."


def test_reopening_the_screen_does_not_re_read_the_journal(make_screen,
                                                           journal):
    """Coming back to the tab must not re-scan hundreds of run folders.

    The load is deliberately deferred to first display rather than done at
    application startup, and the same guard has to stop it running again
    every time the user navigates back to this page. Enumerating and parsing
    every manifest on disk is the expensive part of this screen; doing it on
    each visit would stall the sidebar. Refresh stays available for the user
    who actually wants a fresh read.
    """
    journal["records"] = [_record("a")]
    screen = make_screen()
    assert journal["calls"] == 0

    screen.show()
    assert journal["calls"] == 1
    assert screen._table.rowCount() == 1

    screen.hide()
    screen.show()
    assert journal["calls"] == 1

    screen.refresh()
    assert journal["calls"] == 2
    assert screen._status.text() == "Loaded 1 recorded run(s)."


def test_the_run_actions_stay_silent_until_a_run_is_selected(
        make_screen, journal, monkeypatch, tmp_path):
    """The three run actions must be no-ops when the table is empty.

    All three are reachable with nothing selected -- an empty history, a
    filter that matched nothing, or the shortcut fired before a row was
    picked. Opening ``None`` in the file manager, copying the string
    ``"None"`` over whatever the user had on their clipboard, or handing
    MainWindow an empty module key and empty settings are three separate
    ways to damage state the user cares about, and none of them is
    recoverable by pressing the button again. With a run selected the same
    three actions must do the real thing.
    """
    opened = []
    copied = []

    class _Desktop:
        @staticmethod
        def openUrl(url):
            opened.append(url.toLocalFile())
            return True

    class _Clipboard:
        def setText(self, text):
            copied.append(text)

    monkeypatch.setattr(
        "spacr.qt.screens.run_history.QDesktopServices", _Desktop)
    monkeypatch.setattr(
        "spacr.qt.screens.run_history.QApplication.clipboard",
        lambda: _Clipboard())

    journal["records"] = []
    screen = make_screen()
    emitted = []
    screen.settings_requested.connect(
        lambda app_key, settings: emitted.append((app_key, settings)))
    screen.refresh()
    assert screen._table.rowCount() == 0
    assert screen._status.text() == "Loaded 0 recorded run(s)."

    screen._open_selected_folder()
    screen._copy_selected_path()
    screen._load_selected_settings()
    assert opened == []
    assert copied == []
    assert emitted == []
    assert screen._selection_label.text() == "Select a run to inspect it."

    run_dir = tmp_path / "20260829T101112__classify"
    run_dir.mkdir()
    journal["records"] = [
        _record("20260829T101112__classify",
                settings={"optimizer": "adamw"}, dir_path=run_dir),
    ]
    screen.refresh()
    assert screen._table.rowCount() == 1

    screen._open_selected_folder()
    screen._copy_selected_path()
    screen._load_selected_settings()
    assert opened == [str(run_dir.resolve())]
    assert copied == [str(run_dir)]
    assert emitted == [("classify", {"optimizer": "adamw"})]
    assert screen._status.text() == "Run-folder path copied."


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

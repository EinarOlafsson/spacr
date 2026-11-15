"""Run History: loading the journal, filtering it and acting on a run.

The journal is read once on first display and again only on Refresh; a
refresh that overlaps another is dropped; an enumeration that raises -- inline
or on the worker -- becomes a sentence naming the error rather than an empty
table; the filters keep the count honest; and the three run actions do
nothing until a run is selected.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from PySide6.QtCore import QObject, Qt, Signal

from spacr.qt.screens import run_history as RH
from spacr.qt.screens.run_history import RunHistoryScreen

pytestmark = pytest.mark.qt


def _record(run_id, *, app_key="classify", status="success",
            start_utc="2026-08-29T10:11:12+00:00", settings=None,
            dir_path=None):
    """One ``search_runs`` record, shaped as run_journal emits it."""
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
        "warnings": [],
        "failure": "",
        "environment": {},
        "manifest": {},
    }


@pytest.fixture
def journal(monkeypatch):
    """Stand in for the on-disk journal and count how often it is read.

    ``state["raise"]`` makes the next reads raise that exception instead.
    """
    state = {"records": [], "calls": 0, "raise": None}

    def _search():
        state["calls"] += 1
        if state["raise"] is not None:
            raise state["raise"]
        return list(state["records"])

    monkeypatch.setattr(RH, "search_runs", _search)
    return state


@pytest.fixture
def make_screen(qtbot, qt_theme_applied):
    def _make(threaded=False):
        widget = RunHistoryScreen(threaded=threaded)
        qtbot.addWidget(widget)
        return widget
    return _make


class _InlineWorker(QObject):
    finished = Signal(bool)


class _InlineThread(QObject):
    """A QThread stand-in that runs the job body on the calling thread."""

    finished = Signal()

    def __init__(self, fn, settings, worker):
        super().__init__()
        self._fn = fn
        self._settings = settings
        self._worker = worker

    def isRunning(self):
        return False

    def start(self):
        self._fn(self._settings)
        self._worker.finished.emit(True)
        self.finished.emit()


@pytest.fixture
def inline_threads(monkeypatch):
    made = []

    def _make(fn, settings, app_key="", **kwargs):
        made.append((app_key, kwargs))
        worker = _InlineWorker()
        return _InlineThread(fn, settings, worker), worker

    monkeypatch.setattr(RH, "make_thread", _make)
    return made


def _column(screen, column):
    return {screen._table.item(r, 0).data(Qt.UserRole):
            screen._table.item(r, column).text()
            for r in range(screen._table.rowCount())}


def test_the_journal_is_read_once_and_a_failed_read_says_why(
        make_screen, journal, inline_threads, subtests):
    """Inline and threaded loads, overlaps, reopening and failures."""
    with subtests.test(msg="typing before the first load claims no count"):
        journal["records"] = [_record("a"), _record("b", app_key="measure")]
        screen = make_screen()
        opening = "Open this module to load the run journal."
        assert screen._status.text() == opening
        screen._search.setText("classify")
        assert screen._status.text() == opening
        assert journal["calls"] == 0

    with subtests.test(msg="a refresh while one is running is ignored"):
        screen._busy = True
        screen.refresh()
        assert journal["calls"] == 0
        screen._busy = False

    with subtests.test(msg="showing the screen loads it, once"):
        screen._search.setText("")
        screen.show()
        assert journal["calls"] == 1
        assert screen._table.rowCount() == 2
        assert screen._status.text() == "Loaded 2 recorded run(s)."
        screen.hide()
        screen.show()
        assert journal["calls"] == 1
        screen.refresh()
        assert journal["calls"] == 2

    with subtests.test(msg="an inline failure names the error"):
        journal["raise"] = OSError("runs directory is not readable")
        screen.refresh()
        assert screen.records == []
        assert screen._table.rowCount() == 0
        assert screen.last_error == "OSError: runs directory is not readable"
        assert screen._status.text() == (
            "Could not load run history: OSError: runs directory is not "
            "readable")
        journal["raise"] = None

    with subtests.test(msg="a threaded load lands the records"):
        threaded = make_screen(threaded=True)
        journal["records"] = [_record("r1")]
        refreshed = []
        threaded.history_refreshed.connect(refreshed.append)
        threaded.refresh()
        assert refreshed == [1]
        assert [r["run_id"] for r in threaded.records] == ["r1"]
        assert threaded._table.rowCount() == 1
        assert inline_threads[-1] == ("run_history_refresh",
                                      {"journal": False})

    with subtests.test(msg="a threaded load that raises reports the error"):
        journal["raise"] = RuntimeError("journal is corrupt")
        threaded.refresh()
        assert threaded.records == []
        assert threaded._table.rowCount() == 0
        assert threaded.last_error == "RuntimeError: journal is corrupt"
        assert threaded._status.text() == (
            "Could not load run history: RuntimeError: journal is corrupt")
        assert refreshed == [1]


def test_rows_and_filters_keep_the_count_honest(make_screen, journal,
                                               subtests):
    """Start times read human; module and search filters say what they show."""
    journal["records"] = [
        _record("r-iso", app_key="mask"),
        _record("r-corrupt", app_key="unknown", status="corrupt",
                start_utc=""),
        _record("r-measure", app_key="measure",
                start_utc="2026-08-30 08:00:00"),
    ]
    screen = make_screen()
    screen.refresh()
    with subtests.test(msg="a missing start reads as a dash"):
        assert _column(screen, 0) == {"r-iso": "2026-08-29 10:11:12Z",
                                      "r-corrupt": "—",
                                      "r-measure": "2026-08-30 08:00:00"}
        assert _column(screen, 5)["r-iso"] == "1 / 2.0 KiB"

    with subtests.test(msg="the module filter hides the other modules"):
        assert [screen._module.itemData(i)
                for i in range(screen._module.count())] == [
            "", "mask", "measure", "unknown"]
        screen._module.setCurrentIndex(screen._module.findData("measure"))
        assert len(screen._visible_rows()) == 1
        assert screen._table.rowCount() == 3
        assert screen._status.text() == "Showing 1 of 3 recorded run(s)."

    with subtests.test(msg="the search narrows within all modules"):
        screen._module.setCurrentIndex(0)
        assert screen._status.text() == "Showing 3 of 3 recorded run(s)."
        screen._search.setText("r-iso")
        assert len(screen._visible_rows()) == 1
        assert screen._status.text() == "Showing 1 of 3 recorded run(s)."


def test_the_run_actions_stay_silent_until_a_run_is_selected(
        make_screen, journal, monkeypatch, tmp_path, subtests):
    """Open, copy and load act on the selected run only."""
    opened, copied = [], []

    class _Desktop:
        @staticmethod
        def openUrl(url):
            opened.append(os.path.normpath(url.toLocalFile()))
            return True

    class _Clipboard:
        def setText(self, text):
            copied.append(text)

    monkeypatch.setattr(RH, "QDesktopServices", _Desktop)
    monkeypatch.setattr(RH.QApplication, "clipboard",
                        lambda: _Clipboard())
    screen = make_screen()
    emitted = []
    screen.settings_requested.connect(
        lambda app_key, settings: emitted.append((app_key, settings)))

    with subtests.test(msg="an empty history does nothing"):
        screen.refresh()
        assert screen._status.text() == "Loaded 0 recorded run(s)."
        screen._open_selected_folder()
        screen._copy_selected_path()
        screen._load_selected_settings()
        assert (opened, copied, emitted) == ([], [], [])
        assert screen._selection_label.text() == "Select a run to inspect it."
        assert screen._status.text() == "Loaded 0 recorded run(s)."

    with subtests.test(msg="a run with no settings loads nothing"):
        journal["records"] = [_record("bare", dir_path=tmp_path)]
        screen.refresh()
        screen._load_selected_settings()
        assert emitted == []

    with subtests.test(msg="a selected run is opened, copied and loaded"):
        run_dir = tmp_path / "20260829T101112__classify"
        run_dir.mkdir()
        journal["records"] = [
            _record("20260829T101112__classify",
                    settings={"optimizer": "adamw"}, dir_path=run_dir)]
        screen.refresh()
        screen._open_selected_folder()
        screen._copy_selected_path()
        screen._load_selected_settings()
        assert opened == [str(run_dir.resolve())]
        assert copied == [str(run_dir)]
        assert emitted == [("classify", {"optimizer": "adamw"})]
        assert screen._status.text() == "Run-folder path copied."

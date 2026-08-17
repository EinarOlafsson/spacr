"""The Runs tab answers "what have I run", not "what did the sweep try".

Asked for on 2026-08-17, instruction 125 C:

    "the runs tab should capture all the runs in a sweep and all the runs run
     in the normal module."

The tab was fed by `sweep_results.csv` and nothing else. An ordinary press of
Run did not appear in it, and neither did the re-fit from 124 E -- which is a
new run by construction, and the one case that makes the omission obvious:
the user asks for a second model FROM the plot, gets a second folder on disk,
and has nowhere to see that a second run happened.

They are the same kind of thing -- a fit, its settings, its figures and a
folder to read them back out of -- so they go in the same table, described by
the SAME COLUMNS. That last part is the substance rather than the decoration:
the tab is where a run is compared with a trial, and two rows that do not
share a `regression_type` column cannot be compared at a glance.
"""
from __future__ import annotations

import os

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

pytestmark = pytest.mark.qt


class _Signal:
    def connect(self, *_args, **_kwargs):
        pass


class _FakeThread:
    def __init__(self):
        self.started = _Signal()
        self.finished = _Signal()

    def start(self):
        pass

    def isRunning(self):
        return False


class _FakeWorker:
    def __init__(self):
        for name in ("line_ready", "error", "figure_ready", "result_ready",
                     "finished"):
            setattr(self, name, _Signal())
        self.was_cancelled = False


def _settings(**over):
    settings = {
        "count_data": ["/data/screen/counts.csv"],
        "score_data": ["/data/screen/scores.csv"],
        "regression_type": "ols",
        "multiple_testing_method": "fdr_bh",
        "fdr_alpha": 0.05,
        "min_cell_count": 25,
        "plot": True,
    }
    settings.update(over)
    return settings


@pytest.fixture()
def trials():
    return pd.DataFrame([
        {"trial_id": 1, "status": "ok", "regression_type": "ols",
         "fdr_alpha": 0.05, "n_below_alpha": 12, "seconds": 4.1, "folder": ""},
        {"trial_id": 2, "status": "ok", "regression_type": "ridge",
         "fdr_alpha": 0.05, "n_below_alpha": 40, "seconds": 5.0, "folder": ""},
    ])


@pytest.fixture()
def screen(qtbot, monkeypatch):
    """A regression screen whose Run button starts nothing.

    The worker is faked rather than skipped: `_on_run` is the method that has
    to record the run, and a test that called the recorder directly would
    pass with the wiring cut.
    """
    from spacr.qt.screens.app_screen import AppScreen

    widget = AppScreen("regression")
    qtbot.addWidget(widget)
    monkeypatch.setattr("spacr.qt.screens.app_screen.make_thread",
                        lambda entry, settings: (_FakeThread(), _FakeWorker()))
    monkeypatch.setattr(widget._settings_model, "collect", _settings)
    return widget


def _rows(screen):
    frame = screen._sweep_runs._frame
    return [] if frame is None else frame.to_dict("records")


def _console_text(screen) -> str:
    for name in ("copy_all", "toPlainText"):
        getter = getattr(screen._console, name, None)
        if callable(getter):
            try:
                return str(getter())
            except Exception:
                continue
    return ""


# --------------------------------------------------------------------------- #
#  A run of the module is a run
# --------------------------------------------------------------------------- #

def test_pressing_run_puts_that_run_in_the_runs_tab(screen):
    """The whole of the request in one line: an ordinary run of the module
    used to leave no trace in the tab that is meant to list runs."""
    assert _rows(screen) == []

    screen._on_run(False)

    rows = _rows(screen)
    assert len(rows) == 1, "pressing Run left the Runs tab empty"
    assert rows[0]["source"] == "run"
    assert screen._sweep_runs.table.table.rowCount() == 1


def test_a_run_is_recorded_when_it_starts_not_when_it_ends(screen):
    """The same rule the figure grid marks its sections by. A run that fails
    or is still going is a fact worth seeing rather than a gap -- and a row
    that appeared only on success would omit exactly the runs a user is
    trying to account for."""
    screen._on_run(False)

    rows = _rows(screen)
    assert rows[0]["status"] == "running"
    assert "still going" in screen._sweep_runs._status.text()


def test_a_finished_run_stops_claiming_to_be_running(screen):
    """Left at "running" the row is not merely stale: picking it is refused,
    for a run whose results are sitting on disk."""
    screen._on_run(False)
    screen._on_finished(True)

    rows = _rows(screen)
    assert rows[0]["status"] == "ok"
    assert rows[0]["seconds"] is not None


def test_a_run_that_failed_says_so_on_its_row(screen):
    screen._on_run(False)
    screen._on_finished(False)

    assert _rows(screen)[0]["status"] == "failed"


def test_a_stopped_run_is_not_recorded_as_a_failure(screen):
    """The user stopped it. "Failed" would put a red mark against a decision
    they made deliberately, and the console already distinguishes the two."""
    screen._on_run(False)
    screen._worker.was_cancelled = True
    screen._on_finished(False)

    assert _rows(screen)[0]["status"] == "stopped"


def test_the_run_carries_the_folder_it_wrote_so_it_can_be_opened(screen):
    """A row with no folder can be looked at and not opened, which is half a
    Runs tab: `_show_trial` navigates by folder."""
    screen._on_run(False)
    screen._on_pipeline_result({
        "results": pd.DataFrame({"feature": ["a", "b"],
                                 "coefficient": [1.0, 2.0],
                                 "p_value": [0.01, 0.4]}),
        "res_folder": "/tmp/results_1", "settings": _settings()})

    row = _rows(screen)[0]
    assert row["folder"] == "/tmp/results_1"
    assert row["n_results"] == 2


def test_the_runs_settings_are_on_the_row_beside_the_trials(screen):
    """Not decoration. Comparing a run with a trial is what the tab is for,
    and two rows that do not share a `regression_type` column cannot be."""
    screen._on_run(False)

    row = _rows(screen)[0]
    assert row["regression_type"] == "ols"
    assert row["multiple_testing_method"] == "fdr_bh"
    assert row["fdr_alpha"] == 0.05


# --------------------------------------------------------------------------- #
#  A re-fit is a run too (124 E)
# --------------------------------------------------------------------------- #

def test_a_refit_appears_as_a_run_of_its_own(screen):
    """"a re-fit is a new run and must show up as one". It lands in its own
    folder beside the run it came from, and the tab is where the two are
    compared."""
    screen._on_run(False)
    screen._on_refit(_settings(regression_type="ridge"))

    rows = _rows(screen)
    assert len(rows) == 2, "the re-fit did not reach the Runs tab"
    assert [row["source"] for row in rows] == ["run", "re-fit"]
    assert rows[1]["regression_type"] == "ridge"


def test_a_refit_is_told_apart_from_an_ordinary_run(screen):
    """By the time the worker starts they are the same call, so the one place
    that can tell them apart is where the override arrives."""
    screen._on_run(override=_settings())

    row = _rows(screen)[0]
    assert row["source"] == "re-fit"
    assert row["run"].startswith("re-fit"), row["run"]


def test_a_refit_refused_mid_run_does_not_leave_a_phantom_row(screen):
    """Two regressions writing at once is refused. A row for a run that was
    never started is worse than no row: it can be clicked."""
    screen._on_run(False)

    class _Running:
        def isRunning(self):
            return True

    screen._thread = _Running()
    assert screen._on_refit(_settings(regression_type="ridge")) is False
    assert len(_rows(screen)) == 1


# --------------------------------------------------------------------------- #
#  One table, both kinds
# --------------------------------------------------------------------------- #

def test_the_sessions_runs_and_the_sweeps_trials_share_one_table(
        screen, trials):
    screen._on_run(False)
    screen._sweep_runs.set_frame(trials, source="/tmp/sweep_results.csv")

    rows = _rows(screen)
    assert [row["source"] for row in rows] == [
        "run", "sweep trial", "sweep trial"]
    assert screen._sweep_runs.table.table.rowCount() == 3


def test_reading_the_sweeps_table_does_not_drop_the_sessions_runs(
        qtbot, tmp_path, trials):
    """Opening the tab re-reads `sweep_results.csv`. A reload that replaced
    the whole table would delete the run the user pressed Run for."""
    from spacr.qt.widgets.sweep_runs import SweepRunsPanel

    panel = SweepRunsPanel()
    qtbot.addWidget(panel)
    panel.record_run("run  10:00:00", "run", _settings())
    trials.to_csv(tmp_path / "sweep_results.csv", index=False)

    assert panel.load(tmp_path) is True
    assert panel.table.table.rowCount() == 3
    assert panel.reload() is True
    assert panel.table.table.rowCount() == 3


def test_a_missing_sweep_table_does_not_empty_the_tab(qtbot, tmp_path):
    """The sweep's table being absent says nothing about what this session
    has run, and the tab is opened long before any sweep exists."""
    from spacr.qt.widgets.sweep_runs import SweepRunsPanel

    panel = SweepRunsPanel()
    qtbot.addWidget(panel)
    panel.record_run("run  10:00:00", "run", _settings())

    assert panel.load(tmp_path) is False
    assert panel.table.table.rowCount() == 1
    assert "No results table" in panel._status.text(), panel._status.text()


def test_the_trial_numbers_do_not_turn_into_decimals(qtbot, trials):
    """A run has no trial number, so concatenating it in fills the column
    with NaN and pandas promotes it to float: trial 1 reads "1.0" and its 12
    hits read "12.0" in a table whose whole job is being scanned."""
    from spacr.qt.widgets.sweep_runs import SweepRunsPanel

    panel = SweepRunsPanel()
    qtbot.addWidget(panel)
    panel.record_run("run  10:00:00", "run", _settings())
    panel.set_frame(trials)

    columns = list(panel._frame.columns)
    trial_column = columns.index("trial_id")
    texts = [panel.table.table.item(row, trial_column).text()
             for row in range(panel.table.table.rowCount())]
    assert texts[1:] == ["1", "2"], texts


def test_every_column_a_run_records_is_a_column_the_ordering_knows(qtbot):
    """A setting recorded under a name `ordered_columns` does not list lands
    past the last sweep column -- the far right of a twenty-column table.
    Recorded, and never seen."""
    from spacr.qt.widgets.sweep_runs import (PREFERRED_COLUMNS,
                                             RUN_SETTING_COLUMNS)

    missing = [name for name in RUN_SETTING_COLUMNS
               if name not in PREFERRED_COLUMNS]
    assert missing == [], missing


def test_which_run_a_row_is_comes_first(qtbot, trials):
    """Over a mixed table the first question is which run this is, and a
    trial number names nothing when half the rows are not trials."""
    from spacr.qt.widgets.sweep_runs import SweepRunsPanel

    panel = SweepRunsPanel()
    qtbot.addWidget(panel)
    panel.record_run("run  10:00:00", "run", _settings())
    panel.set_frame(trials)

    assert list(panel._frame.columns)[:2] == ["run", "source"]
    assert list(panel._frame["run"]) == ["run  10:00:00", "trial 1", "trial 2"]


def test_the_summary_counts_both_kinds(qtbot, trials):
    """A number with no breakdown reads as "the sweep ran three trials"."""
    from spacr.qt.widgets.sweep_runs import SweepRunsPanel

    panel = SweepRunsPanel()
    qtbot.addWidget(panel)
    panel.record_run("run  10:00:00", "run", _settings())
    panel.update_run(1, status="ok")
    panel.set_frame(trials)

    text = panel._status.text()
    assert "3 runs" in text, text
    assert "1 from this session" in text and "2 from the sweep" in text, text


# --------------------------------------------------------------------------- #
#  Picking one
# --------------------------------------------------------------------------- #

def test_picking_a_run_that_is_still_going_says_so(screen):
    """Not "did not produce a regression" -- it has not finished trying, and
    a click that reports a failure that has not happened is worse than one
    that reports nothing."""
    screen._on_run(False)
    screen._show_trial(_rows(screen)[0])

    text = _console_text(screen)
    assert "still going" in text, text


def test_picking_a_run_names_it_the_way_its_row_does(screen):
    """"Trial nan" is how a mixed table stops being readable."""
    screen._on_run(False)
    screen._on_finished(True)
    row = dict(_rows(screen)[0])
    row["folder"] = "/nope/not/a/folder"
    screen._show_trial(row)

    text = _console_text(screen)
    assert "no saved results" in text.lower(), text
    assert "nan" not in text.lower(), text
    assert row["run"] in text, text


def test_a_finished_run_can_be_opened_from_its_row(screen, tmp_path):
    """The end of the request: a run made in this session is reachable from
    the tab exactly as a sweep trial is."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    folder = tmp_path / "results_1"
    folder.mkdir()
    pd.DataFrame({
        "feature": [f"fraction:grna[{i}_1]" for i in range(20)],
        "coefficient": np.linspace(-2, 2, 20),
        "p_value": np.linspace(0.001, 0.9, 20),
    }).to_csv(folder / "results.csv", index=False)
    figure = plt.figure(figsize=(4, 3))
    figure.add_subplot(111).plot([0, 1], [1, 0])
    figure.savefig(folder / "volcano.png")
    plt.close(figure)

    screen._on_run(False)
    screen._update_run_in_runs_tab(folder=str(folder))
    screen._on_finished(True)
    screen._show_trial(_rows(screen)[0])

    assert screen._results_panel.table.table.rowCount() == 20
    assert len(screen._figure_grid._cells) == 1


# --------------------------------------------------------------------------- #
#  The grid and the table agree about which run this is
# --------------------------------------------------------------------------- #

def test_the_grid_heading_and_the_runs_row_name_the_same_run(screen):
    """Two labels generated separately are two clocks. A user looking at
    "run 14:32:05" on the grid has to be able to find it in the table."""
    screen._on_run(False)

    marks = [mark["label"] for mark in screen._figure_queue._runs]
    assert marks, "the run was not marked on the figure grid"
    assert marks[-1] == _rows(screen)[0]["run"]


def test_a_screen_without_a_runs_tab_still_runs(qtbot, monkeypatch):
    """Every module screen goes through the same `_on_run`; only the
    regression screens have the tab. Recording must not be a crash for the
    other twenty-seven."""
    from spacr.qt.screens.app_screen import AppScreen

    widget = AppScreen("mask")
    qtbot.addWidget(widget)
    monkeypatch.setattr("spacr.qt.screens.app_screen.make_thread",
                        lambda entry, settings: (_FakeThread(), _FakeWorker()))
    monkeypatch.setattr(widget._settings_model, "collect",
                        lambda: {"src": "/data/plate"})

    assert getattr(widget, "_sweep_runs", None) is None
    widget._on_run(False)
    assert widget._thread is not None, "recording swallowed the run itself"
    widget._on_finished(True)
    assert widget._btn_run.isEnabled(), "the screen did not come back"

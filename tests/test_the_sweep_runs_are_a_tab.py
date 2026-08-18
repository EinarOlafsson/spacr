"""The parameter search is the module setup plus one tab for the runs.

Instruction 116, corrected by the maintainer on 2026-08-16:

    "actually the paramiter search should be like the main module setup just
     an extra tab for the runs."

This SUPERSEDES the earlier "run table above the figure grid", which was
written hours before from the maintainer's first description and must not be
built.

What it was: a bespoke screen carrying its own copies of the trials table,
the figure queue and the results panel. Every fix to the shared module screen
-- the grid, the console, the settings panel, the drop handling -- had to be
made again there, and this repository already pays for that kind of drift.

What it is now: this screen, with one more tab. Picking a run swaps the
figures on the right, which is the substance of the request and the only part
of it that did not change.

DELIBERATELY NOT THE SHAPE THE RESULTS GET (119 A). A results table is
hundreds of findings read ALONGSIDE the figure describing them, so both are
on screen at once. A runs table is a short list scanned top to bottom, and
picking one replaces everything to its right -- so it is a tab. One is
reading within a run; the other is navigating between runs. Anyone who
unifies them has removed the reason for both.
"""
from __future__ import annotations

import os

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

pytestmark = pytest.mark.qt


@pytest.fixture()
def trials():
    return pd.DataFrame([
        {"trial_id": 1, "status": "ok", "regression_type": "ols",
         "fdr_alpha": 0.05, "n_below_alpha": 12, "positive_rank": 3,
         "seconds": 4.1, "folder": ""},
        {"trial_id": 2, "status": "ok", "regression_type": "ridge",
         "fdr_alpha": 0.05, "n_below_alpha": 40, "positive_rank": 91,
         "seconds": 5.0, "folder": ""},
        {"trial_id": 3, "status": "error", "regression_type": "poisson",
         "fdr_alpha": 0.05, "n_below_alpha": 0, "positive_rank": None,
         "seconds": 0.2, "folder": "", "error_type": "LinAlgError",
         "error": "singular design"},
    ])


def _console_text(screen) -> str:
    """Whatever the console has on it, however it stores it."""
    console = screen._console
    for name in ("copy_all", "toPlainText"):
        getter = getattr(console, name, None)
        if callable(getter):
            try:
                return str(getter())
            except Exception:
                continue
    return ""


@pytest.fixture()
def screen(qtbot):
    from spacr.qt.screens.app_screen import AppScreen

    widget = AppScreen("regression")
    qtbot.addWidget(widget)
    return widget


# --------------------------------------------------------------------------- #
#  The tab
# --------------------------------------------------------------------------- #

def test_the_runs_are_a_tab_beside_the_results(screen):
    """RUNS FIRST. This asserted ["Results", "Runs", "Measurements"] until
    2026-08-17, when the maintainer asked for the other order by name:
    "the run tab should be before the results tab and results should be shown
    for the chosen run" (instruction 128 J).

    The order is the reading order of the screen. Picking a row in Runs now
    re-points Results at that run, so Results is the DETAIL of what Runs has
    selected, and a detail tab ahead of the thing it details reads backwards.
    """
    tabs = screen._results_tabs
    assert [tabs.tabText(i) for i in range(tabs.count())] == [
        "Runs", "Results", "Measurements", "Cells"]


def test_results_is_what_opens_first(screen):
    """A finished regression opens into its results, not into a run list.

    UNCHANGED IN SUBSTANCE, only in index: 128 J moved Runs in front of
    Results, so "opens into its results" is index 1 rather than 0. J asks for
    a different tab ORDER and says nothing about which tab a finished
    regression lands on, so this property survives the reorder -- and it has
    to be set explicitly now, because QTabWidget's own default is index 0,
    which is the run list this test exists to say we do not open into.
    """
    tabs = screen._results_tabs
    assert tabs.currentWidget() is screen._results_panel
    assert tabs.currentIndex() == 1


def test_the_runs_tab_is_not_a_second_bespoke_screen(screen):
    """It reuses the shared table widget, which already sorts numerically,
    filters and copies. A second implementation is a second set of bugs."""
    from spacr.qt.widgets.fast_plots import ResultsTable

    assert isinstance(screen._sweep_runs.table, ResultsTable)


# --------------------------------------------------------------------------- #
#  Reading the table
# --------------------------------------------------------------------------- #

def test_it_reads_a_sweep_results_csv(qtbot, tmp_path, trials):
    from spacr.qt.widgets.sweep_runs import SweepRunsPanel

    trials.to_csv(tmp_path / "sweep_results.csv", index=False)
    panel = SweepRunsPanel()
    qtbot.addWidget(panel)

    assert panel.load(tmp_path) is True
    assert panel.table.table.rowCount() == 3


def test_a_missing_table_says_so_rather_than_looking_empty(qtbot, tmp_path):
    from spacr.qt.widgets.sweep_runs import SweepRunsPanel

    panel = SweepRunsPanel()
    qtbot.addWidget(panel)

    assert panel.load(tmp_path) is False
    assert "No results table" in panel._status.text()


def test_failed_trials_are_counted_out_loud(qtbot, tmp_path, trials):
    """A sweep whose trials mostly failed still writes a full-looking table."""
    from spacr.qt.widgets.sweep_runs import SweepRunsPanel

    trials.to_csv(tmp_path / "sweep_results.csv", index=False)
    panel = SweepRunsPanel()
    qtbot.addWidget(panel)
    panel.load(tmp_path)

    assert "1 of which did not produce a regression" in panel._status.text()


def test_the_useful_columns_come_first_and_nothing_is_hidden(trials):
    """Ordering, not filtering. A column nobody listed is still the user's
    own result, and hiding it means leaving the application to read it."""
    from spacr.qt.widgets.sweep_runs import ordered_columns

    order = ordered_columns(trials)
    assert order[0] == "trial_id"
    assert order.index("regression_type") < order.index("seconds")
    assert set(order) == set(trials.columns), "a column was dropped"


# --------------------------------------------------------------------------- #
#  Picking a run
# --------------------------------------------------------------------------- #

def test_picking_a_run_reports_its_real_values_not_display_strings(
        qtbot, tmp_path, trials):
    """A trial re-run from "0.05" instead of 0.05 is not the trial that was
    recorded."""
    from spacr.qt.widgets.sweep_runs import SweepRunsPanel

    trials.to_csv(tmp_path / "sweep_results.csv", index=False)
    panel = SweepRunsPanel()
    qtbot.addWidget(panel)
    panel.load(tmp_path)

    seen = []
    panel.trial_activated.connect(seen.append)
    panel.table.table.selectRow(0)

    assert seen, "selecting a run emitted nothing"
    assert isinstance(seen[-1]["fdr_alpha"], float)
    assert seen[-1]["trial_id"] == 1


def test_a_failed_trial_says_why_instead_of_doing_nothing(screen, trials):
    """A click that produces no visible change reads as a broken table."""
    record = trials.iloc[2].to_dict()
    screen._show_trial(record)

    assert _console_text(screen), "picking a failed trial said nothing"
    text = _console_text(screen)
    assert "singular design" in text or "LinAlgError" in text


def test_a_trial_with_no_saved_output_says_so(screen, trials):
    record = trials.iloc[0].to_dict()
    record["folder"] = "/nope/not/a/folder"
    screen._show_trial(record)

    assert "no saved results" in _console_text(screen).lower()


def test_a_saved_trial_is_shown_without_refitting(screen, tmp_path, trials):
    """"A saved run is instant and a re-fit is a minute." Re-fitting to see
    something already on disk is a minute of waiting for the same answer."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    folder = tmp_path / "trial_0001"
    folder.mkdir()
    pd.DataFrame({
        "feature": [f"fraction:grna[{i}_1]" for i in range(20)],
        "coefficient": np.linspace(-2, 2, 20),
        "p_value": np.linspace(0.001, 0.9, 20),
    }).to_csv(folder / "results.csv", index=False)
    for name in ("volcano", "residuals"):
        figure = plt.figure(figsize=(4, 3))
        figure.add_subplot(111).plot([0, 1], [1, 0])
        figure.savefig(folder / f"{name}.png")
        plt.close(figure)

    record = dict(trials.iloc[0])
    record["folder"] = str(folder)
    screen._show_trial(record)

    assert screen._results_panel.table.table.rowCount() == 20
    assert len(screen._figure_grid._cells) == 2, (
        "the trial's own figures did not reach the grid")


def test_the_grid_shows_the_trials_figures_not_the_last_runs(screen, tmp_path):
    """Otherwise picking a run changes the numbers and leaves the pictures,
    which is worse than changing neither."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for _ in range(4):                      # figures from an earlier run
        figure = plt.figure(figsize=(3, 2))
        figure.add_subplot(111).plot([0, 1], [0, 1])
        screen._on_figure_ready(figure)
    screen._refresh_figure_grid()
    assert len(screen._figure_grid._cells) == 4

    folder = tmp_path / "trial_0002"
    folder.mkdir()
    figure = plt.figure(figsize=(4, 3))
    figure.add_subplot(111).plot([0, 1], [1, 0])
    figure.savefig(folder / "only_one.png")
    plt.close(figure)

    assert screen._load_trial_figures(str(folder)) == 1
    assert len(screen._figure_grid._cells) == 1


def test_the_runs_table_speaks_about_runs(qtbot, tmp_path, trials):
    """It reuses the coefficient table's widget, and inherited its words: a
    box telling the user to type a gene, beside a "significant only" filter
    that cannot do anything to a list of trials."""
    from spacr.qt.widgets.sweep_runs import SweepRunsPanel

    trials.to_csv(tmp_path / "sweep_results.csv", index=False)
    panel = SweepRunsPanel()
    qtbot.addWidget(panel)
    panel.load(tmp_path)

    placeholder = panel.table._filter.placeholderText()
    assert "gene" not in placeholder.lower(), placeholder
    assert "run" in placeholder.lower(), placeholder
    assert panel.table._only_hits.isHidden() or \
        not panel.table._only_hits.isVisibleTo(panel), (
            "a 'significant only' filter over a table of trials")


def test_the_coefficient_table_keeps_its_own_words(qtbot):
    """configure() is opt-in; the results table must be unaffected."""
    from spacr.qt.widgets.fast_plots import ResultsTable

    table = ResultsTable()
    qtbot.addWidget(table)

    assert "gene" in table._filter.placeholderText().lower()


# --------------------------------------------------------------------------- #
#  The measurement scan is the same surface with a different thing varying
# --------------------------------------------------------------------------- #

def test_the_measurement_scan_is_a_tab_too(screen):
    """"instead of a paramiter search a serch for which measurement has genes
    with clear effect sizes". Structurally the parameter search with the
    DEPENDENT VARIABLE varying, so it sits beside the runs rather than in a
    screen of its own."""
    from spacr.qt.widgets.measurement_scan_panel import MeasurementScanPanel

    assert isinstance(screen._scan_panel, MeasurementScanPanel)
    assert screen._results_tabs.widget(2) is screen._scan_panel


def test_a_scan_with_nothing_loaded_says_so(screen):
    """An empty table reads as "no measurement has an effect", which is the
    opposite of "no data was scanned"."""
    assert screen._scan_panel.run_scan() is False
    assert "Nothing to scan" in screen._scan_panel._status.text()

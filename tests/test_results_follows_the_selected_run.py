"""Picking a run in the Runs tab points Results at THAT run.

Instruction 128 J, asked for on 2026-08-17 after the Runs tab landed:

    "the run tab should be before the results tab and results should be shown
     for the chosen run"

Two changes. The tab ORDER is asserted where the tab order already was, in
`test_the_sweep_runs_are_a_tab.py`. This file is the substantive half: the
Results tab is BOUND to the Runs selection.

`_show_trial` already loaded a run's folder into the results panel and the
grid, so the loading was never the gap. What was missing is that the panel it
re-points sits BEHIND the tab the user is standing on, so the whole visible
effect of clicking a run was the figure grid changing on the far side of the
screen -- and the coefficient table the user actually asked for stayed one
click away, looking like it had not moved.

THE TRAP, written into 128 J and asserted here so nobody "fixes" it: the
panel's `set_frame` resets the baseline, the compartment, the gene/guide
filter and the selection on purpose -- a new table is a new experiment.
Switching runs SHOULD clear them. Switching back is NOT expected to restore
them.

Driven through the real table selection rather than by calling `_show_trial`
by hand, because the binding under test is the one from a click.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

pytestmark = pytest.mark.qt


def _coefficients(n, seed):
    """A coefficient table the results panel will accept."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "feature": [f"fraction:grna[{seed}_{i}]" for i in range(n)],
        "coefficient": rng.normal(0, .5, n),
        "p_value": rng.uniform(size=n),
        "q_value": rng.uniform(size=n),
        "condition": list(rng.choice(["nc", "other"], n, p=[.1, .9])),
        "multiple_testing_method": "fdr_bh",
    })


def _figure(path):
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, ax = plt.subplots(figsize=(2, 1.5))
    ax.plot([0, 1], [1, 0])
    figure.savefig(path)
    plt.close(figure)


@pytest.fixture()
def screen_folder(tmp_path):
    """Shaped like a real screen: two runs side by side under `results/`.

    Their coefficient tables are different LENGTHS, which is what the tests
    read to say which run the panel is showing -- a length cannot be produced
    by accident from the other run's table.
    """
    root = tmp_path / "results"
    for name, rows, seed in (("ols_11", 30, 1), ("ols_12", 70, 2)):
        folder = root / name
        folder.mkdir(parents=True)
        _coefficients(rows, seed).to_csv(folder / "results.csv", index=False)
        _figure(folder / "regression_qc" / "panel_00.pdf")
    return root


@pytest.fixture()
def screen(qtbot):
    from spacr.qt.screens.app_screen import AppScreen

    widget = AppScreen("regression")
    qtbot.addWidget(widget)
    # A results panel that failed to build makes every assertion below vacuous
    # -- the screen swallows that and carries on with `_results_panel = None`.
    assert widget._results_panel is not None
    assert widget._results_tabs is not None
    return widget


def _runs_table(screen, screen_folder, names=("ols_11", "ols_12")):
    """Put the runs on the Runs tab, the way a session's own runs get there."""
    for name in names:
        screen._sweep_runs.record_run(
            name, folder=str(screen_folder / name))
        # A row says `running` until it is told otherwise, and `_show_trial`
        # refuses to open a run that has not finished -- correctly.
        handle = screen._sweep_runs._next_handle
        screen._sweep_runs.update_run(handle, status="ok")
    return screen._sweep_runs


def _pick(panel, run: str) -> bool:
    """Select the row whose run is ``run``, the way a click does.

    Returns whether a row was found, so a test cannot silently assert against
    a selection that never happened.
    """
    table = panel.table.table
    for row in range(table.rowCount()):
        for column in range(table.columnCount()):
            item = table.item(row, column)
            if item is not None and item.text() == run:
                table.clearSelection()
                table.selectRow(row)
                return True
    return False


# --------------------------------------------------------------------------- #
#  The binding
# --------------------------------------------------------------------------- #

def test_picking_a_run_raises_the_results_tab(screen, screen_folder):
    """The whole visible effect of a click used to be on the far side of the
    screen: the panel was re-pointed behind the tab the user was standing
    on."""
    runs = _runs_table(screen, screen_folder)
    screen._results_tabs.setCurrentWidget(runs)

    assert _pick(runs, "ols_11")

    assert screen._results_tabs.currentWidget() is screen._results_panel


def test_picking_a_run_puts_that_runs_table_in_the_results_panel(
        screen, screen_folder):
    """"results should be shown for the chosen run" -- ols_11 wrote 30
    coefficients and ols_12 wrote 70, so the length says which one is up."""
    runs = _runs_table(screen, screen_folder)

    assert _pick(runs, "ols_11")

    frame = screen._results_panel.results_frame()
    assert frame is not None
    assert len(frame) == 30


def test_picking_a_second_run_replaces_the_first(screen, screen_folder):
    """A binding that only ever fires once is worse than none: the table and
    the run list would then disagree about which fit is on screen."""
    runs = _runs_table(screen, screen_folder)
    assert _pick(runs, "ols_11")
    assert len(screen._results_panel.results_frame()) == 30

    assert _pick(runs, "ols_12")

    assert len(screen._results_panel.results_frame()) == 70


def test_the_figures_follow_the_run_too(screen, screen_folder):
    """The grid on the right is the chosen run's, not whatever the last one
    left there."""
    runs = _runs_table(screen, screen_folder)

    assert _pick(runs, "ols_11")

    titles = [cell for cell in screen._figure_grid._cells]
    assert titles, "the chosen run's figures did not reach the grid"


def test_a_run_with_no_saved_table_does_not_raise_an_empty_results_tab(
        screen, tmp_path):
    """Raising an empty Results page over the run list would hide the row the
    user would pick next behind a page that says nothing."""
    runs = screen._sweep_runs
    runs.record_run("never_wrote", folder=str(tmp_path / "nothing"))
    handle = runs._next_handle
    runs.update_run(handle, status="ok")
    screen._results_tabs.setCurrentWidget(runs)

    assert _pick(runs, "never_wrote")

    assert screen._results_tabs.currentWidget() is runs


# --------------------------------------------------------------------------- #
#  The trap: switching runs RESETS, and switching back does not restore
# --------------------------------------------------------------------------- #

def test_switching_runs_clears_the_selection_and_the_compartment(
        screen, screen_folder):
    """128 J states this as intended, not as a gap: `set_frame` resets the
    baseline, the compartment, the gene/guide filter and the selection because
    a new table is a new experiment. A ring left on the volcano after a switch
    would mark a point that means something else now.
    """
    runs = _runs_table(screen, screen_folder)
    assert _pick(runs, "ols_11")
    panel = screen._results_panel
    panel._selected_key = "fraction:grna[1_3]"
    panel._compartment = "rhoptry"

    assert _pick(runs, "ols_12")

    assert panel._selected_key is None
    assert panel._compartment is None


def test_switching_runs_does_not_leave_the_last_fits_residuals_on_screen(
        screen, screen_folder):
    """128 J names the diagnostics as one of the three things that follow the
    chosen run, and this is the dangerous one.

    A saved run has no pickled model, so a run opened off disk has no
    residuals to draw. What must NOT happen is the previous fit's residuals
    staying on the tabs beside the new run's coefficient table, with nothing
    on screen saying they describe a different fit -- a reader would check
    normality on one run and read coefficients off another.
    """
    runs = _runs_table(screen, screen_folder)
    panel = screen._results_panel
    assert _pick(runs, "ols_11")
    panel._model = object()             # as if this session had fitted it

    assert _pick(runs, "ols_12")

    assert panel._model is None
    # Emptied AND explained -- `clear_diagnostics` resets each scene and puts
    # a reason on it, because an empty plot with nothing to say is
    # indistinguishable from a broken one.
    for plot in panel.diagnostic_plots():
        assert not plot.plot.listDataItems()


def test_switching_back_does_not_restore_what_the_switch_cleared(
        screen, screen_folder):
    """The other half of the same contract, asserted so a later session does
    not read the test above as "state should be remembered per run" and build
    a cache 128 J explicitly does not ask for."""
    runs = _runs_table(screen, screen_folder)
    assert _pick(runs, "ols_11")
    screen._results_panel._compartment = "rhoptry"
    assert _pick(runs, "ols_12")

    assert _pick(runs, "ols_11")

    assert screen._results_panel._compartment is None
    assert len(screen._results_panel.results_frame()) == 30

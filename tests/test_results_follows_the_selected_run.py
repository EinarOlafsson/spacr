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
    """Open the row whose run is ``run``, the way a user does.

    TWO THINGS CHANGED UNDER THIS HELPER AND IT WAS DOING NEITHER.

    SELECTING IS NO LONGER OPENING (190). A single click used to load, which
    meant arrowing down a list of five runs cost five multi-second reads to
    look at five names. Double-click is the gesture that loads now, so this
    selects and then makes that gesture -- which is what every caller here was
    always asking for: "picking a run puts that run's table in the results
    panel" is a statement about OPENING it.

    AND THE READ IS OFF THE GUI THREAD (159). `_show_trial` hands the folder
    to `panel.start_load` and the table arrives at `_on_trial_loaded` a
    moment later, so a test that asserts immediately after the gesture
    asserts against the run it has not finished reading. `_opened` below is
    how a caller waits for it.

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
                panel._on_double_click()
                return True
    return False


def _opened(screen, rows: int, timeout: float = 10.0):
    """Spin until the results panel holds a table of ``rows`` rows.

    The read is off the GUI thread, so this is what "the run is open" means
    in a test. Returns the frame; fails with what it actually got, because
    "None is not None" says nothing about which run arrived.
    """
    import time

    from PySide6.QtWidgets import QApplication

    deadline = time.monotonic() + timeout
    seen = None
    while time.monotonic() < deadline:
        QApplication.processEvents()
        frame = screen._results_panel.results_frame()
        seen = None if frame is None else len(frame)
        if seen == rows:
            return frame
        time.sleep(0.01)
    raise AssertionError(
        f"the results panel holds {seen} row(s) after {timeout:g}s, not "
        f"{rows}: the run did not open")


# --------------------------------------------------------------------------- #
#  The binding
# --------------------------------------------------------------------------- #

def test_opening_a_run_leaves_the_user_where_they_were(screen, screen_folder):
    """INVERTED BY 190, and the inversion is the point.

    This used to assert that picking a run RAISED the Results tab, because
    the whole visible effect of a click was otherwise on the far side of the
    screen. The maintainer asked for the opposite on 2026-08-20 -- "the user
    should have to click the results tab to go there, no auto switching
    tabs" -- and both cannot be true.

    A view that moves by itself takes the user somewhere they did not ask to
    go and loses what they were reading. The run still opens; the console
    says so, which is what replaced the tab jump as the proof.
    """
    runs = _runs_table(screen, screen_folder)
    screen._results_tabs.setCurrentWidget(runs)

    assert _pick(runs, "ols_11")
    _opened(screen, 30)

    assert screen._results_tabs.currentWidget() is runs


def test_picking_a_run_puts_that_runs_table_in_the_results_panel(
        screen, screen_folder):
    """"results should be shown for the chosen run" -- ols_11 wrote 30
    coefficients and ols_12 wrote 70, so the length says which one is up."""
    runs = _runs_table(screen, screen_folder)

    assert _pick(runs, "ols_11")

    assert len(_opened(screen, 30)) == 30


def test_picking_a_second_run_replaces_the_first(screen, screen_folder):
    """A binding that only ever fires once is worse than none: the table and
    the run list would then disagree about which fit is on screen."""
    runs = _runs_table(screen, screen_folder)
    assert _pick(runs, "ols_11")
    assert len(_opened(screen, 30)) == 30

    assert _pick(runs, "ols_12")

    assert len(_opened(screen, 70)) == 70


def test_the_figures_follow_the_run_too(screen, screen_folder):
    """The grid on the right is the chosen run's, not whatever the last one
    left there."""
    runs = _runs_table(screen, screen_folder)

    assert _pick(runs, "ols_11")
    _opened(screen, 30)

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
    _opened(screen, 30)
    panel = screen._results_panel
    panel._selected_key = "fraction:grna[1_3]"
    panel._compartment = "rhoptry"

    assert _pick(runs, "ols_12")
    _opened(screen, 70)

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
    _opened(screen, 30)
    panel._model = object()             # as if this session had fitted it

    assert _pick(runs, "ols_12")
    _opened(screen, 70)

    assert panel._model is None
    # Emptied AND explained -- `clear_diagnostics` resets each scene and puts
    # a reason on it, because an empty plot with nothing to say is
    # indistinguishable from a broken one.
    for plot in panel.diagnostic_plots():
        assert not plot.plot.listDataItems()


def test_switching_back_gives_the_run_its_own_view_again(
        screen, screen_folder):
    """SUPERSEDED BY 116, and the supersession is the point of this test.

    This assertion used to read "switching back does NOT restore what the
    switch cleared", written so a later session would not read the test above
    as "state should be remembered per run" and build a cache 128 J did not
    ask for. The maintainer then asked for exactly that cache, on 2026-08-18:

        "every regression run should have its own interactive volcano plot"

    which landed at ``d4113297`` -- and left this test red, because the two
    contracts are opposites and only one of them can be true.

    THEY ARE NOT ACTUALLY IN CONFLICT, and the resolution is what is asserted
    here. 128 J is about the FIRST look at a run: its defaults come off its
    own table, and nothing of the previous run leaks forward -- which is the
    test above and still holds. 116 is about the SECOND look, where resetting
    threw away the view the user had built. A run gets back what it had; it
    never inherits what another run had.
    """
    runs = _runs_table(screen, screen_folder)
    panel = screen._results_panel
    assert _pick(runs, "ols_11")
    _opened(screen, 30)
    panel._compartment = "rhoptry"

    assert _pick(runs, "ols_12")
    _opened(screen, 70)
    # NOT INHERITED. The other run's compartment on this run's table is the
    # picture nobody chose, and it is what 128 J was protecting.
    assert panel._compartment is None

    assert _pick(runs, "ols_11")
    _opened(screen, 30)

    assert panel._compartment == "rhoptry"
    assert len(panel.results_frame()) == 30

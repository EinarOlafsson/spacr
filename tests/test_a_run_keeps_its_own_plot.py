"""Instruction 116's new section: a run owns its plot state.

    Requested 2026-08-18 -- "every regression run should have its own
    interactive volcano plot".

`regression_results.py` holds ONE `VolcanoPlot()` for the whole screen, so a
run does not have a volcano: the screen has one and a run borrows it. Three
things follow, and they are why this is worth doing:

  * A RUN IS NOT EVALUABLE FROM ITS ROW if evaluating run B destroys run A.
    154 F now produces a run per column, and comparing them is the point.
  * THE STATE A USER BUILDS IS LOST ON EVERY SWITCH -- a chosen level, a
    colouring, a typed axis limit, a selected gene all belong to the run
    being looked at and all belonged to the widget.
  * AND THE MIRROR OF IT, which is worse and was not in the instruction:
    state LEAKED FORWARD. Measured on the real panel -- after typing an
    x-range and a 2-spread cut on run A, opening run B drew run B inside run
    A's window with run A's cut, a picture nobody chose with nothing on
    screen saying where it came from.

THE STATE HALF ONLY, deliberately. 129 measured live pyqtgraph tiles at
74.99 ms per window-drag frame against 5.19 ms for photographs on a 16.7 ms
budget, so N live volcanoes for N runs is not the answer. The state is small;
the widget is not.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

pytestmark = pytest.mark.qt


def _frame(seed, n=90):
    rng = np.random.default_rng(seed)
    features = [f"fraction:grna[{200000 + i // 3}_{i % 3}]" for i in range(n)]
    features[:12] = [f"gene_fraction:gene[{200000 + i}]" for i in range(12)]
    return pd.DataFrame({
        "feature": features,
        "coefficient": rng.normal(0, 0.5, n),
        "p_value": rng.uniform(1e-9, 0.99, n),
        "condition": ["pos", "neg", "other"] * (n // 3)})


@pytest.fixture()
def panel(qtbot):
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    widget = RegressionResultsPanel()
    qtbot.addWidget(widget)
    return widget


def _build_a_view(panel):
    """What a user does to a run they are evaluating."""
    panel.set_level("gene")
    panel._colour_by.setCurrentIndex(panel._colour_by.findData("condition"))
    panel.set_threshold_multiplier(2.0)
    panel.volcano.set_axis_limits(x=(-1.5, 1.5))
    panel._select_key(str(panel.filtered_frame()["feature"].iloc[2]))


def test_a_run_returned_to_is_exactly_as_it_was_left(panel):
    """The acceptance test of the ask, in one function.

    "Open run A's volcano, colour it by localisation, type an axis limit,
    select a gene. Open run B. Return to A: the colouring, the limit and the
    selection are exactly as left."
    """
    panel.set_frame(_frame(0), source="/runs/A/results.csv")
    _build_a_view(panel)
    built = panel.plot_state()

    panel.set_frame(_frame(7), source="/runs/B/results.csv")
    panel.set_frame(_frame(0), source="/runs/A/results.csv")

    assert panel.plot_state() == built


def test_each_piece_of_it_comes_back_named(panel):
    """Asserted one by one, so a regression says WHICH one was lost."""
    panel.set_frame(_frame(0), source="/runs/A/results.csv")
    _build_a_view(panel)
    chosen = panel.plot_state()["selected_key"]

    panel.set_frame(_frame(7), source="/runs/B/results.csv")
    panel.set_frame(_frame(0), source="/runs/A/results.csv")

    state = panel.plot_state()
    assert state["level"] == "gene"
    assert state["colour_by"] == "condition"
    assert state["threshold_multiplier"] == 2.0
    assert state["x_limits"] == (-1.5, 1.5)
    assert state["selected_key"] == chosen
    # And it reached the widgets, not only the dict.
    assert panel.level() == "gene"
    assert panel._colour_by.currentData() == "condition"


def test_a_run_never_opened_before_is_not_given_the_last_ones_view(panel):
    """The leak. It is the half the instruction did not name and the worse one.

    A volcano showing run B inside run A's typed x-range is a figure nobody
    chose, and there is nothing on it saying so.
    """
    from spacr.qt.widgets.regression_results import (
        DEFAULT_THRESHOLD_MULTIPLIER)

    panel.set_frame(_frame(0), source="/runs/A/results.csv")
    _build_a_view(panel)

    panel.set_frame(_frame(7), source="/runs/B/results.csv")

    fresh = panel.plot_state()
    assert fresh["x_limits"] is None
    assert fresh["y_limits"] is None
    assert fresh["threshold_multiplier"] == DEFAULT_THRESHOLD_MULTIPLIER
    assert fresh["selected_key"] is None
    assert fresh["level"] == "grna", "the table's own default, not A's"


def test_the_axis_window_is_only_remembered_when_it_was_typed(panel):
    """Storing the view range would freeze an auto-ranged plot at whatever it
    happened to show, so a run returned to would stop following its data."""
    panel.set_frame(_frame(0), source="/runs/A/results.csv")

    state = panel.plot_state()

    assert state["x_limits"] is None and state["y_limits"] is None


def test_the_colour_column_is_remembered_by_name_not_by_index(panel):
    """The combo is rebuilt from each table's own columns, so index 3 is a
    different column in the next run."""
    panel.set_frame(_frame(0), source="/runs/A/results.csv")
    panel._colour_by.setCurrentIndex(panel._colour_by.findData("condition"))

    # A run whose table has an extra category column ahead of `condition`.
    other = _frame(7)
    other.insert(1, "batch", ["b1", "b2", "b3"] * (len(other) // 3))
    panel.set_frame(other, source="/runs/B/results.csv")
    panel.set_frame(_frame(0), source="/runs/A/results.csv")

    assert panel._colour_by.currentData() == "condition"


def test_a_selection_the_restored_level_hides_is_not_an_error(panel):
    """A saved selection is a feature NAME and the level may filter it out.

    Missing is not an error; it is a row the user cannot currently see.
    """
    panel.set_frame(_frame(0), source="/runs/A/results.csv")
    panel.set_level(None)
    panel._select_key(str(panel.filtered_frame()["feature"].iloc[0]))

    panel.set_frame(_frame(7), source="/runs/B/results.csv")
    # The stored state is edited AFTER leaving A -- leaving is when it is
    # written, so planting one before would simply be overwritten. A gene
    # picked at level=None is not in the guide table.
    # FILED UNDER THE RUN'S FOLDER, not the path it happened to arrive as:
    # a live run hands over `<results>/ols_3` and the same run picked in the
    # Runs tab hands over `<results>/ols_3/results.csv`, and while those were
    # two keys one run had two entries and got neither back.
    panel._plot_states["/runs/A"]["level"] = "grna"

    panel.set_frame(_frame(0), source="/runs/A/results.csv")  # must not raise

    assert panel.level() == "grna"
    # THE KEY IS KEPT, and that is the existing contract rather than a gap:
    # `set_level` does the same, so switching the filter back re-rings the
    # row. What must not happen is a raise, which is what an unguarded
    # `_select_key` over a filtered table would do.
    assert panel.plot_state()["selected_key"].startswith("gene_fraction")
    assert panel.filtered_frame() is not None, "the panel is still usable"


def test_a_run_with_no_path_is_not_remembered(panel):
    """There is no key to come back through, so there is nothing to store."""
    panel.set_frame(_frame(0), source="")
    panel.set_level("gene")
    panel.set_frame(_frame(7), source="/runs/B/results.csv")

    assert panel.remembered_runs() == ()


def test_the_run_on_screen_joins_the_store_when_it_is_left(panel):
    """`remembered_runs` is what has been STORED, not what is being shown.

    The two are different questions and a caller asking the first usually
    means it -- so the run on screen, which holds its state live in the
    widgets, is deliberately not in the list until it is left.
    """
    panel.set_frame(_frame(0), source="/runs/A/results.csv")
    assert panel.remembered_runs() == ()

    panel.set_frame(_frame(7), source="/runs/B/results.csv")

    assert panel.remembered_runs() == ("/runs/A",)


def test_a_deleted_run_takes_its_plot_state_with_it(panel):
    """Instruction 146. Otherwise a later run written into the same folder
    inherits the deleted one's level and colouring."""
    panel.set_frame(_frame(0), source="/runs/A/results.csv")
    panel.set_level("gene")
    panel.set_frame(_frame(7), source="/runs/B/results.csv")
    assert "/runs/A" in panel.remembered_runs()

    assert panel.forget_plot_state("/runs/A/results.csv") is True
    assert panel.forget_plot_state("/runs/A/results.csv") is False
    assert panel.remembered_runs() == ()

    panel.set_frame(_frame(0), source="/runs/A/results.csv")
    assert panel.level() == "grna", "the forgotten view came back"


def test_applying_nothing_is_refused_rather_than_half_applied(panel):
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    fresh = RegressionResultsPanel()
    assert fresh.apply_plot_state({"level": "gene"}) is False, (
        "a panel with no table has nothing to apply a level to")
    panel.set_frame(_frame(0), source="/runs/A/results.csv")
    assert panel.apply_plot_state(None) is False
    assert panel.apply_plot_state({}) is True


def test_the_state_is_data_and_nothing_else(panel):
    """It has to be small: the whole design is N states and one widget."""
    panel.set_frame(_frame(0), source="/runs/A/results.csv")

    state = panel.plot_state()

    assert isinstance(state, dict)
    for name, value in state.items():
        assert value is None or isinstance(
            value, (str, int, float, tuple)), (name, type(value))


def test_a_live_run_and_the_same_run_off_disk_are_one_run(panel):
    """Instruction 116 through the shape the application actually produces.

    `perform_regression` hands its coefficients over with `res_folder` -- a
    DIRECTORY -- so a run watched live is filed under `<results>/ols_3`. Pick
    that same run in the Runs tab afterwards and it is loaded from
    `<results>/ols_3/results.csv`. Two paths, one run: the view the user built
    while the run was live must be there when they come back to it, which is
    the exact reset 116 was asked to stop.
    """
    panel.set_frame(_frame(0), source="/runs/A")      # as a live run arrives
    panel.set_level("gene")

    panel.set_frame(_frame(7), source="/runs/B/results.csv")
    panel.set_frame(_frame(0), source="/runs/A/results.csv")   # off disk

    assert panel.level() == "gene"
    # ONE RUN, ONE KEY. Two entries for A -- one per path it arrived as --
    # is how it got neither back, and it is invisible from the level alone.
    assert set(panel.remembered_runs()) == {"/runs/A", "/runs/B"}

""""Graph type" on the right-click menu of any plot, not just seven of them.

Asked for repeatedly, most recently as: "i have asked many times at this
point for an option when i right click on a graph, called graph type that
would allow the user to switch between graph types ... i want this for all
graphs in the core modules, really for all matplotlib graphs in the
software."

The menu existed and was reachable from almost nowhere. It is built only
when the figure carries ``_spacr_replot``, and the sole place that attaches
one is ``plot.create_grouped_plot`` -- so every QC panel, diagnostic and
one-off plot in spaCR was skipped, which is nearly all of them.

Two things were wrong and both are fixed here:

* a figure without a recipe now gets one DERIVED from its own artists --
  bar heights, scatter offsets, line vertices -- so the menu can redraw it;
* ``create_grouped_plot`` had no branch for ``line`` or ``jitter_bar``.
  Those two fell through its if/elif chain, drew nothing, and the function
  returned ``plt.gcf()`` -- an EMPTY figure. Two of the seven menu entries
  blanked the plot and reported no error. ``_create_line_graph`` was broken
  in a second way as well: it indexes ``data_column[1]``, so with a single
  data column it raised IndexError, which was swallowed.

The real-data checks in this file read plate1's measurements.db, because a
synthetic frame would not have caught either defect: both need a plot the
software actually draws.
"""
from __future__ import annotations

import os
import sqlite3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from PySide6.QtWidgets import QWidget

from spacr.plot import create_grouped_plot
from spacr.qt.widgets.figure_settings import (GROUPED_PLOT_TYPES, _replot,
                                              build_figure_context_menu,
                                              derive_replot_recipe)


#: A real measurements.db to switch graph types on, named by the
#: environment rather than hard-coded: a path under one user's home is not a
#: precondition any other machine has, and a test that silently does nothing
#: everywhere else is worse than one that says it was skipped.
#:
#: Point SPACR_PLATE1_DB at any measurements.db with a `cell` table. On the
#: machine this was written on that is plate1's.
PLATE1_DB = os.environ.get("SPACR_PLATE1_DB", "")


def _artists(figure):
    if figure is None:
        return 0
    return sum(len(a.patches) + len(a.collections) + len(a.lines)
               for a in figure.axes)


def _grouped():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "group": np.repeat(["ctrl", "treat", "ko"], 40),
        "value": np.concatenate([rng.normal(3, .6, 40),
                                 rng.normal(5.5, .6, 40),
                                 rng.normal(2.1, .6, 40)]),
    })


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# every type draws something
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind,_label", GROUPED_PLOT_TYPES)
def test_every_offered_graph_type_actually_draws(kind, _label):
    """A menu entry that returns an empty figure is worse than no entry."""
    figure, _results = create_grouped_plot(
        df=_grouped(), grouping_column="group", data_column="value",
        graph_type=kind, save=False)

    assert figure is not None, f"{kind} produced no figure"
    assert _artists(figure) > 0, (
        f"{kind} produced a figure with nothing drawn in it")


def test_an_unknown_graph_type_is_refused_by_name():
    """Falling through the chain in silence is what blanked the plot."""
    with pytest.raises(ValueError, match="graph_type"):
        create_grouped_plot(df=_grouped(), grouping_column="group",
                            data_column="value", graph_type="spiral",
                            save=False)


def test_a_line_over_one_data_column_does_not_raise():
    """`data_column[1]` on a one-column frame is an IndexError, swallowed."""
    figure, _results = create_grouped_plot(
        df=_grouped(), grouping_column="group", data_column="value",
        graph_type="line", save=False)

    assert _artists(figure) > 0
    # One point per group, joined: the groups are the x axis.
    labels = [t.get_text() for t in figure.axes[0].get_xticklabels()]
    assert set(labels) >= {"ctrl", "treat", "ko"}


# ---------------------------------------------------------------------------
# the recipe is derived for figures that never carried one
# ---------------------------------------------------------------------------

def test_a_plain_bar_chart_yields_its_own_numbers():
    figure, axes = plt.subplots()
    axes.bar(["ctrl", "treat", "ko"], [3.0, 5.5, 2.1])

    recipe = derive_replot_recipe(figure)
    assert recipe is not None
    got = dict(zip(recipe["df"]["group"], recipe["df"]["value"]))
    assert got == {"ctrl": 3.0, "treat": 5.5, "ko": 2.1}


def test_a_scatter_yields_one_row_per_point():
    figure, axes = plt.subplots()
    rng = np.random.default_rng(1)
    axes.scatter(rng.integers(0, 3, 25), rng.normal(5, 1, 25))

    recipe = derive_replot_recipe(figure)
    assert recipe is not None
    assert len(recipe["df"]) == 25


def test_a_grid_of_panels_is_left_alone():
    """Redrawing four panels as one violin would throw three of them away."""
    figure, axes = plt.subplots(2, 2)
    for axis in axes.ravel():
        axis.bar(["a", "b"], [1.0, 2.0])

    assert derive_replot_recipe(figure) is None


def test_an_empty_figure_yields_no_recipe():
    figure, _axes = plt.subplots()
    assert derive_replot_recipe(figure) is None


def test_the_menu_offers_graph_type_on_a_figure_with_no_recipe(qapp):
    figure, axes = plt.subplots()
    axes.bar(["ctrl", "treat"], [3.0, 5.0])
    assert getattr(figure, "_spacr_replot", None) is None

    host = QWidget()
    try:
        menu = build_figure_context_menu(host, figure)
        titles = [a.menu().title() for a in menu.actions()
                  if a.menu() is not None]
        assert "Graph type" in titles
    finally:
        host.deleteLater()
        qapp.processEvents()


# ---------------------------------------------------------------------------
# real data
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not PLATE1_DB or not os.path.exists(PLATE1_DB),
                    reason="set SPACR_PLATE1_DB to a real measurements.db")
def test_every_type_switches_on_a_figure_built_from_real_measurements():
    """186 real cells over four fields, switched through all seven types."""
    with sqlite3.connect(PLATE1_DB) as db:
        frame = pd.read_sql(
            "SELECT fieldID, cell_area FROM cell "
            "WHERE fieldID IN ('f1','f9','f10','f11')", db)
    assert len(frame) > 100, "the fixture query returned almost nothing"

    figure, _results = create_grouped_plot(
        df=frame, grouping_column="fieldID", data_column="cell_area",
        graph_type="bar", save=False)

    for kind, _label in GROUPED_PLOT_TYPES:
        redrawn = _replot(figure, kind)
        assert redrawn is not None, f"{kind} did not redraw"
        assert _artists(redrawn) > 0, f"{kind} redrew an empty figure"


@pytest.mark.skipif(not PLATE1_DB or not os.path.exists(PLATE1_DB),
                    reason="set SPACR_PLATE1_DB to a real measurements.db")
def test_a_scatter_of_real_measurements_recovers_every_cell(qapp):
    """The derived frame is the plotted data, not a summary of it."""
    with sqlite3.connect(PLATE1_DB) as db:
        frame = pd.read_sql(
            "SELECT cell_area, cell_perimeter FROM cell", db).dropna()

    figure, axes = plt.subplots()
    axes.scatter(frame["cell_area"], frame["cell_perimeter"], s=4)

    recipe = derive_replot_recipe(figure)
    assert recipe is not None
    assert len(recipe["df"]) == len(frame)

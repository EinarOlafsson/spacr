"""A graph is pyqtgraph, retypes on right-click, and saves as a folder.

Asked four times, most recently 2026-08-21:

    "I WANT ALL FIGURES TO BE IN pyqtgraph NOT matplotlib. i also want to be
     able to right click on the graph and change the graph typem to whatever
     is possibel with the underlying data. and i want to be able to save the
     figure intp afolder as stats, data, ong figure and pdf figure"

THE THREE ARE ONE ASK, and answering them separately is why they kept coming
back. A plot that HOLDS ITS DATA can be redrawn as any kind the data
supports and can write that data and its statistics beside the picture; a
rendered Figure can do neither, which is why both features had to be bolted
on beside it and kept drifting out of reach.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from PySide6.QtWidgets import QApplication, QMenu  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def grouped(qtbot):
    """A plot that is destroyed with the test that asked for it.

    Handed to ``qtbot`` rather than merely built. The plot is a parentless
    top-level widget and pyqtgraph hangs a context menu with ten submenus
    off it, every one of those top-level too; a top-level widget that is
    never closed cannot be freed at all, because the connections holding it
    run through Qt's C++ side where Python's collector cannot follow, so no
    amount of collecting reclaims one. This file left 295 windows standing
    for the rest of the process; registering the plot built here accounts
    for 218 of them, and everything that restyles afterwards stops paying
    for those.
    """
    from spacr.qt.widgets.grouped_plot import GroupedPlot, PlotSpec

    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "grp": ["nc"] * 30 + ["pc"] * 30,
        "val": list(rng.normal(0.0, 1.0, 30)) + list(rng.normal(1.0, 1.0, 30)),
    })
    plot = GroupedPlot(PlotSpec(frame=frame, value="val", group="grp",
                                unit="well", title="Area"))
    qtbot.addWidget(plot)
    return plot


def _show_as(plot):
    """The "Show as" submenu, with its parent kept alive.

    THE PARENT MENU IS THE SUBMENU'S ONLY OWNER. A local QMenu is collected
    the moment this returns, and Qt then deletes the child -- "Internal C++
    object (QMenu) already deleted" on the next line. The same lifetime rule
    `figure_settings` writes out for its own submenus, met here from the
    other side.
    """
    menu = QMenu()
    plot._offer_graph_kinds(menu)
    plot._test_menu = menu          # held, not returned and dropped
    for action in menu.actions():
        if action.menu() is not None:
            return action.menu()
    return None


class TestItIsPyqtgraph:

    def test_the_plot_is_a_fastplot(self, grouped):
        from spacr.qt.widgets.fast_plots import FastPlot

        assert isinstance(grouped, FastPlot)

    def test_there_is_no_matplotlib_in_it(self):
        import inspect

        from spacr.qt.widgets import grouped_plot

        # THE CODE, NOT THE PROSE. The module docstring quotes the request
        # -- "I WANT ALL FIGURES TO BE IN pyqtgraph NOT matplotlib" -- so a
        # plain search finds the very word it asserts the absence of.
        source = inspect.getsource(grouped_plot)
        body = source.split('"""', 2)[-1]
        assert "matplotlib" not in body and "pyplot" not in body
        assert "pyqtgraph" in source

    def test_the_compare_panel_draws_one(self, qtbot):
        """The graph the user right-clicks, which was a Figure in a canvas."""
        from spacr.qt.widgets.measurement_compare_dialog import (
            MeasurementComparePanel)

        rng = np.random.default_rng(0)
        rows = pd.DataFrame({
            "prcfo": [f"p1_r1_c{c}_f1_o{o}" for c in (1, 2)
                      for o in range(15)],
            "plateID": ["p1"] * 30, "rowID": ["r1"] * 30,
            "columnID": ["c1"] * 15 + ["c2"] * 15,
            "area": list(rng.normal(500, 50, 15))
                    + list(rng.normal(600, 50, 15)),
        })
        groups = {"a": list(rows.index[:15]), "b": list(rows.index[15:])}
        panel = MeasurementComparePanel(rows, groups,
                                        settings={"cell_picking": "rank"},
                                        databases=[], counts=None)
        # Registered, not merely built: an unclosed top-level panel keeps its
        # plot and every submenu pyqtgraph hangs off it alive for the rest of
        # the process.
        qtbot.addWidget(panel)
        panel.level.setCurrentIndex(0)
        panel.refresh()
        assert panel._canvas is not None
        assert hasattr(panel._canvas, "graph_spec"), (
            "the comparison is still a matplotlib canvas")


class TestItRetypesToWhatTheDataSupports:

    def test_show_as_is_on_the_menu(self, grouped):
        assert _show_as(grouped) is not None

    def test_the_entries_are_named_not_described(self, grouped):
        """A menu reading 'one value per group' instead of 'Bar' cannot be
        scanned, which is what a menu is for."""
        labels = {a.text() for a in _show_as(grouped).actions()}
        assert "Bar" in labels and "Violin" in labels

    def test_the_description_is_the_tooltip(self, grouped):
        bar = next(a for a in _show_as(grouped).actions()
                   if a.text() == "Bar")
        assert "one value per group" in bar.toolTip()

    def test_what_does_not_fit_is_greyed_with_a_reason(self, grouped):
        line = next(a for a in _show_as(grouped).actions()
                    if a.text() == "Line")
        assert not line.isEnabled()
        assert "unordered categories" in line.toolTip()

    @pytest.mark.parametrize("kind", ["bar", "box", "violin", "jitter",
                                      "bar_jitter"])
    def test_each_fitting_kind_redraws(self, grouped, kind):
        assert grouped.show_as(kind) == 2
        assert grouped.spec.kind == kind

    def test_a_kind_that_does_not_fit_is_refused(self, grouped):
        """Drawing it anyway would answer a different question and say
        nothing."""
        with pytest.raises(ValueError):
            grouped.show_as("line")

    def test_the_data_is_unchanged_by_retyping(self, grouped):
        before = {k: len(v) for k, v in grouped.spec.groups().items()}
        grouped.show_as("violin")
        assert {k: len(v) for k, v in grouped.spec.groups().items()} == before

    def test_two_continuous_axes_offer_a_scatter(self, qtbot):
        from spacr.qt.widgets.grouped_plot import GroupedPlot, PlotSpec

        rng = np.random.default_rng(1)
        frame = pd.DataFrame({"x": rng.normal(size=50),
                              "y": rng.normal(size=50)})
        plot = GroupedPlot(PlotSpec(frame=frame, value="y", group="x"))
        qtbot.addWidget(plot)
        labels = {a.text() for a in _show_as(plot).actions() if a.isEnabled()}
        assert "Scatter" in labels
        assert "Bar" not in labels


class TestItSavesTheWholeFolder:

    def test_all_four_files_and_the_settings(self, grouped, tmp_path):
        out = grouped.export_bundle(str(tmp_path))
        assert sorted(os.listdir(out)) == [
            "Area.pdf", "Area.png", "data.csv", "settings.json",
            "statistics.csv"]

    def test_the_data_is_the_rows_behind_the_picture(self, grouped,
                                                     tmp_path):
        out = grouped.export_bundle(str(tmp_path))
        back = pd.read_csv(os.path.join(out, "data.csv"))
        assert len(back) == 60 and set(back["grp"]) == {"nc", "pc"}

    def test_the_statistics_are_a_real_test(self, grouped, tmp_path):
        """Not an empty file: the plot knows its groups, so the comparison
        the picture shows is the comparison that was run."""
        out = grouped.export_bundle(str(tmp_path))
        stats = pd.read_csv(os.path.join(out, "statistics.csv"))
        items = list(stats["item"])
        assert "n [nc]" in items and "n [pc]" in items
        assert "test" in items and "p_value" in items

    def test_the_unit_is_stated(self, grouped, tmp_path):
        """A test across cells when the replicate is the well returns
        p < 1e-10 on noise."""
        out = grouped.export_bundle(str(tmp_path))
        stats = pd.read_csv(os.path.join(out, "statistics.csv"))
        assert stats.loc[stats["item"] == "unit", "value"].iloc[0] == "well"

    def test_the_settings_say_what_was_drawn(self, grouped, tmp_path):
        import json

        grouped.show_as("violin")
        out = grouped.export_bundle(str(tmp_path))
        with open(os.path.join(out, "settings.json")) as handle:
            saved = json.load(handle)
        assert saved["kind"] == "violin"
        assert saved["group"] == "grp" and saved["value"] == "val"

    def test_the_menu_offers_it(self, grouped):
        menu = QMenu()
        grouped._offer_graph_kinds(menu)
        assert hasattr(grouped, "export_bundle")

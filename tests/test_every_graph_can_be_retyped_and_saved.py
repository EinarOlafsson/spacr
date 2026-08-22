"""Every graph can be retyped and saved whole (200 A, 223, group colours).

Reported 2026-08-21: "one of my instructions was that i could right click on
any graph and change it to a different graph type depending on the data and
that when downloading a graph i would downloade a folder ... i see none of
this".

THEY WERE BUILT ON THE WRONG PATH. Both landed on `FastPlot`, the pyqtgraph
plots -- and a run's figures are matplotlib, drawn into the figure queue. A
feature that exists and cannot be reached from where the user is looking is
a feature that does not exist.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def grouped(app):
    """A real jitter-over-bar, through spaCR's own drawer."""
    import matplotlib
    matplotlib.use("Agg", force=True)

    from spacr.plot import create_grouped_plot

    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "grp": ["nc"] * 30 + ["pc"] * 30,
        "val": list(rng.normal(0.0, 1.0, 30)) + list(rng.normal(1.0, 1.0, 30)),
    })
    figure, _results = create_grouped_plot(
        df=frame, grouping_column="grp", data_column="val",
        graph_type="jitter_bar", save=False)
    return figure


@pytest.fixture
def menu(app, grouped):
    from spacr.qt.widgets.figure_settings import build_figure_context_menu

    return build_figure_context_menu(QWidget(), grouped,
                                     on_change=lambda **_k: None)


def _submenu(menu, title):
    for action in menu.actions():
        if action.text() == title and action.menu() is not None:
            return action.menu()
    return None


class TestTheGraphTypeCanBeChanged:

    def test_show_as_is_on_the_menu(self, menu):
        assert _submenu(menu, "Show as") is not None

    def test_it_offers_the_grouped_types(self, menu):
        from spacr.qt.widgets.figure_settings import GROUPED_PLOT_TYPES

        labels = {a.text() for a in _submenu(menu, "Show as").actions()}
        assert labels == {label for _k, label in GROUPED_PLOT_TYPES}

    def test_a_type_that_does_not_fit_is_greyed(self, menu):
        """"depending on the data" -- a line through unordered categories is
        a row of markers joined for no reason."""
        line = next(a for a in _submenu(menu, "Show as").actions()
                    if a.text() == "Line")
        assert not line.isEnabled()

    def test_and_it_says_why(self, menu):
        """Greyed without a reason is a control the user keeps trying."""
        line = next(a for a in _submenu(menu, "Show as").actions()
                    if a.text() == "Line")
        assert line.toolTip()

    def test_the_ones_that_fit_are_offered(self, menu):
        offered = [a.text() for a in _submenu(menu, "Show as").actions()
                   if a.isEnabled()]
        assert "Box" in offered and "Violin" in offered

    def test_the_fitness_comes_from_the_shared_table(self, grouped):
        """A second opinion here would let the menu offer a type the drawer
        cannot draw."""
        from spacr.qt.widgets.figure_settings import _which_types_fit

        fits, why = _which_types_fit(grouped._spacr_replot)
        assert "box" in fits and "line" not in fits
        assert why.get("line")

    def test_changing_it_produces_a_new_figure(self, grouped):
        from spacr.qt.widgets.figure_settings import _replot

        drawn = _replot(grouped, "box", None)
        assert drawn is not None and drawn is not grouped
        assert drawn._spacr_replot["graph_type"] == "box"


class TestTheGroupIsTheUnit:
    """"i also want to modify thing on the group level not individual points
    and barts"."""

    def test_there_is_a_group_colour_menu(self, menu):
        assert _submenu(menu, "Group colours") is not None

    def test_one_entry_per_group(self, menu):
        entries = {a.text().rstrip("…")
                   for a in _submenu(menu, "Group colours").actions()}
        assert entries == {"nc", "pc"}

    def test_it_is_not_per_element(self, menu):
        """A jitter-over-bar draws dozens of marks per group, and the
        question is 'make THIS CONDITION blue'."""
        assert len(_submenu(menu, "Group colours").actions()) == 2

    def test_the_colour_goes_on_the_recipe_not_the_artists(self, grouped):
        """Setting an artist's colour lasts until the next redraw and then
        silently reverts -- which is what 'changing the colors changes
        nothing' looks like from the other side."""
        from spacr.qt.widgets.figure_settings import _replot

        recipe = dict(grouped._spacr_replot)
        recipe["colors"] = {"nc": "#ff0000", "pc": "#00ff00"}
        grouped._spacr_replot = recipe
        drawn = _replot(grouped, "bar", None)
        assert drawn._spacr_replot["colors"]["nc"] == "#ff0000"

    def test_many_groups_are_capped_and_the_rest_named(self, app):
        """A menu showing the first twenty-four of ninety and saying nothing
        looks like a menu that has them all."""
        from PySide6.QtWidgets import QMenu

        from spacr.qt.widgets.figure_settings import _add_group_colours

        frame = pd.DataFrame({"grp": [f"g{i}" for i in range(40)],
                              "val": range(40)})
        holder = QMenu()
        _add_group_colours(holder, object(),
                           {"df": frame, "grouping_column": "grp",
                            "data_column": "val"},
                           None, QWidget())
        actions = _submenu(holder, "Group colours").actions()
        assert len(actions) == 25
        assert "more groups not listed" in actions[-1].text()
        assert not actions[-1].isEnabled()


class TestSavingWritesTheWholeFolder:

    def test_the_action_is_on_the_menu(self, menu):
        titles = [a.text() for a in menu.actions()]
        assert "Save figure, data and statistics…" in titles

    def test_it_writes_all_five_files(self, grouped, tmp_path):
        from spacr.qt.widgets.figure_settings import save_figure_bundle

        out = save_figure_bundle(grouped, str(tmp_path), name="g")
        assert sorted(os.listdir(out)) == [
            "data.csv", "g.pdf", "g.png", "settings.json", "statistics.csv"]

    def test_both_images_use_the_shared_figure_writer(
            self, grouped, tmp_path, monkeypatch):
        from spacr import plot
        from spacr.qt.widgets.figure_settings import save_figure_bundle

        calls = []

        def _save(_figure, path, **kwargs):
            calls.append((os.path.basename(path), kwargs))
            with open(path, "wb") as handle:
                handle.write(b"rendered")
            return path

        monkeypatch.setattr(plot, "save_figure", _save)
        out = save_figure_bundle(grouped, str(tmp_path), name="g")

        assert [name for name, _kwargs in calls] == ["g.pdf", "g.png"]
        assert [kwargs["fmt"] for _name, kwargs in calls] == ["pdf", "png"]
        assert all(kwargs["close"] is False for _name, kwargs in calls)
        assert sorted(os.listdir(out)) == [
            "data.csv", "g.pdf", "g.png", "settings.json", "statistics.csv"]

    def test_the_data_is_the_rows_it_was_drawn_from(self, grouped, tmp_path):
        from spacr.qt.widgets.figure_settings import save_figure_bundle

        out = save_figure_bundle(grouped, str(tmp_path), name="g")
        back = pd.read_csv(os.path.join(out, "data.csv"))
        assert len(back) == 60
        assert set(back["grp"]) == {"nc", "pc"}

    def test_the_statistics_compare_the_groups_the_picture_shows(
            self, grouped, tmp_path):
        """The groups come from the recipe, which is what makes the test the
        same comparison as the plot."""
        from spacr.qt.widgets.figure_settings import save_figure_bundle

        out = save_figure_bundle(grouped, str(tmp_path), name="g")
        stats = pd.read_csv(os.path.join(out, "statistics.csv"))
        items = list(stats["item"])
        assert "n [nc]" in items and "n [pc]" in items
        assert "test" in items and "p_value" in items

    def test_a_figure_with_no_recipe_still_gets_every_file(self, tmp_path):
        """An absent statistics file reads as a bug."""
        from matplotlib.figure import Figure

        from spacr.qt.widgets.figure_settings import save_figure_bundle

        bare = Figure()
        bare.add_subplot(111).plot([1, 2, 3], [1, 4, 9])
        out = save_figure_bundle(bare, str(tmp_path), name="plain")
        assert "statistics.csv" in os.listdir(out)
        stats = pd.read_csv(os.path.join(out, "statistics.csv"))
        assert stats["value"].iloc[0] == "none"

    def test_the_settings_travel_with_it(self, grouped, tmp_path):
        """Without the filters recorded the numbers cannot be reproduced."""
        import json

        from spacr.qt.widgets.figure_settings import save_figure_bundle

        out = save_figure_bundle(grouped, str(tmp_path), name="g")
        with open(os.path.join(out, "settings.json")) as handle:
            saved = json.load(handle)
        assert saved["grouping_column"] == "grp"
        assert "df" not in saved, "the frame is data.csv, not a setting"

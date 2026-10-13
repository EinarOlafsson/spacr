"""The figure context menu, its redraw, and what each does when it fails.

A right-click menu is built over whatever figure happens to be on screen --
one that carries its source rows, one that carries nothing but artists, one
whose drawer refuses the type it is asked for. None of those may raise out
of a right-click, and each of them has a different right answer.
"""
from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("PySide6")

from PySide6.QtGui import QColor  # noqa: E402
from PySide6.QtWidgets import QMenu, QWidget  # noqa: E402

from spacr.qt.widgets import figure_settings as fs  # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture()
def parent(qapp):
    widget = QWidget()
    yield widget
    widget.deleteLater()


def _recipe():
    return {
        "df": pd.DataFrame({fs.DERIVED_GROUP: ["a", "a", "b", "b"],
                            fs.DERIVED_VALUE: [1.0, 2.0, 3.0, 4.0]}),
        "grouping_column": fs.DERIVED_GROUP,
        "data_column": fs.DERIVED_VALUE,
        "graph_type": "bar",
    }


# --------------------------------------------------------------------------
# reading the pairs off the artists
# --------------------------------------------------------------------------

class _AxesThatRefuses:
    """An axes whose artist lists cannot be walked."""

    def __init__(self, which):
        self._which = which

    def get_xticks(self):
        return []

    def get_xticklabels(self):
        return []

    def get_xlim(self):
        return (0.0, 1.0)

    def _refuse(self):
        raise RuntimeError("this artist list has gone")

    @property
    def patches(self):
        if self._which == "patches":
            self._refuse()
        return []

    @property
    def collections(self):
        if self._which == "collections":
            self._refuse()
        return []

    @property
    def lines(self):
        if self._which == "lines":
            self._refuse()
        return []


@pytest.mark.parametrize("family", ["patches", "collections", "lines"])
def test_an_artist_family_that_cannot_be_walked_costs_only_itself(family):
    assert fs._pairs_from_axes(_AxesThatRefuses(family)) == [], (
        "a family that cannot be read is skipped, not raised out of a menu")


def test_the_pairs_come_off_the_artists_that_were_drawn():
    figure = plt.figure()
    axis = figure.add_subplot(111)
    axis.bar(["a", "b"], [3.0, 5.0])
    try:
        pairs = fs._pairs_from_axes(axis)
    finally:
        plt.close(figure)

    assert sorted(pairs) == [("a", 3.0), ("b", 5.0)], (
        "the bar heights are the values and the tick text is the group")


# --------------------------------------------------------------------------
# redrawing as another type
# --------------------------------------------------------------------------

def test_a_drawer_that_refuses_the_type_returns_nothing_and_says_why(
        monkeypatch, caplog):
    figure = plt.figure()
    figure._spacr_replot = _recipe()

    def refuse(**_kwargs):
        raise ValueError("cannot draw a violin from two points")

    monkeypatch.setattr("spacr.plot.create_grouped_plot", refuse)
    try:
        with caplog.at_level(logging.DEBUG, logger=fs.LOG.name):
            assert fs._replot(figure, "violin") is None
    finally:
        plt.close(figure)

    assert any("could not redraw" in record.getMessage()
               for record in caplog.records)


def test_a_figure_with_no_rows_is_not_redrawn_at_all():
    figure = plt.figure()
    try:
        assert fs._replot(figure, "violin") is None, (
            "nothing to redraw from is not the same as a failed redraw")
    finally:
        plt.close(figure)


def test_a_listener_that_takes_no_figure_is_still_told(monkeypatch):
    figure = plt.figure()
    figure._spacr_replot = _recipe()
    drawn = plt.figure()
    calls = []

    monkeypatch.setattr("spacr.plot.create_grouped_plot",
                        lambda **_kwargs: (drawn, None))
    try:
        result = fs._replot(figure, "box", on_change=lambda: calls.append(1))
    finally:
        plt.close(figure)
        plt.close(drawn)

    assert result is drawn
    assert calls == [1], "the no-argument toggles are told something moved"


def test_a_listener_that_raises_does_not_lose_the_redrawn_figure(monkeypatch,
                                                                  caplog):
    figure = plt.figure()
    figure._spacr_replot = _recipe()
    drawn = plt.figure()

    def explode(_new_figure):
        raise RuntimeError("the canvas has gone")

    monkeypatch.setattr("spacr.plot.create_grouped_plot",
                        lambda **_kwargs: (drawn, None))
    try:
        with caplog.at_level(logging.DEBUG, logger=fs.LOG.name):
            result = fs._replot(figure, "box", on_change=explode)
    finally:
        plt.close(figure)
        plt.close(drawn)

    assert result is drawn, "the figure was drawn; only the telling failed"
    assert any("redraw notification failed" in record.getMessage()
               for record in caplog.records)


# --------------------------------------------------------------------------
# the group colours
# --------------------------------------------------------------------------

def test_no_grouping_column_means_no_group_colour_menu(parent):
    menu = QMenu(parent)

    fs._add_group_colours(menu, None, {"df": None}, None, parent)
    fs._add_group_colours(menu, None, {"df": pd.DataFrame({"a": [1]}),
                                       "grouping_column": "missing"},
                          None, parent)

    assert menu.actions() == [], (
        "a colour menu with nothing to colour is a menu that does nothing")


def test_every_group_gets_an_entry_and_the_rest_are_counted(parent):
    recipe = _recipe()
    recipe["df"] = pd.DataFrame({
        fs.DERIVED_GROUP: [f"g{i}" for i in range(30)],
        fs.DERIVED_VALUE: list(range(30)),
    })
    menu = QMenu(parent)

    fs._add_group_colours(menu, None, recipe, None, parent)

    submenu = menu.actions()[0].menu()
    texts = [action.text() for action in submenu.actions()]
    assert len(texts) == 25, "24 groups plus the line that names the rest"
    assert "6 more groups not listed" in texts[-1]
    assert submenu.actions()[-1].isEnabled() is False


def test_a_chosen_group_colour_is_stored_on_the_recipe_and_redrawn(
        parent, monkeypatch):
    figure = plt.figure()
    recipe = _recipe()
    drawn = plt.figure()
    monkeypatch.setattr("spacr.plot.create_grouped_plot",
                        lambda **kwargs: (drawn, None))
    monkeypatch.setattr(fs, "pick_colour",
                        lambda *args, **kwargs: QColor("#aabbcc"))
    menu = QMenu(parent)
    fs._add_group_colours(menu, figure, recipe, None, parent)

    try:
        menu.actions()[0].menu().actions()[0].trigger()
    finally:
        plt.close(figure)
        plt.close(drawn)

    assert recipe["colors"] == {"a": "#aabbcc"}, (
        "painting the artist instead reverts at the next redraw")
    assert figure._spacr_replot is recipe


def test_a_cancelled_colour_leaves_the_recipe_alone(parent, monkeypatch):
    figure = plt.figure()
    recipe = _recipe()
    monkeypatch.setattr(fs, "pick_colour",
                        lambda *args, **kwargs: QColor())
    menu = QMenu(parent)
    fs._add_group_colours(menu, figure, recipe, None, parent)

    try:
        menu.actions()[0].menu().actions()[0].trigger()
    finally:
        plt.close(figure)

    assert "colors" not in recipe


# --------------------------------------------------------------------------
# the evidence bundle
# --------------------------------------------------------------------------

def test_a_cancelled_folder_chooser_writes_nothing(parent, monkeypatch):
    written = []
    monkeypatch.setattr(
        "PySide6.QtWidgets.QFileDialog.getExistingDirectory",
        staticmethod(lambda *args, **kwargs: ""))
    monkeypatch.setattr(fs, "save_figure_bundle",
                        lambda *args, **kwargs: written.append(1))
    menu = QMenu(parent)
    figure = plt.figure()

    fs._add_bundle_save(menu, figure, parent)
    try:
        menu.actions()[0].trigger()
    finally:
        plt.close(figure)

    assert written == []


def test_a_bundle_that_cannot_be_written_does_not_raise_out_of_the_menu(
        parent, monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(
        "PySide6.QtWidgets.QFileDialog.getExistingDirectory",
        staticmethod(lambda *args, **kwargs: str(tmp_path)))

    def refuse(*_args, **_kwargs):
        raise OSError("read-only folder")

    monkeypatch.setattr(fs, "save_figure_bundle", refuse)
    menu = QMenu(parent)
    figure = plt.figure()

    fs._add_bundle_save(menu, figure, parent)
    try:
        with caplog.at_level(logging.DEBUG, logger=fs.LOG.name):
            menu.actions()[0].trigger()
    finally:
        plt.close(figure)

    assert any("figure bundle" in record.getMessage()
               for record in caplog.records)


# --------------------------------------------------------------------------
# the menu's own legend switch
# --------------------------------------------------------------------------

def test_the_menu_can_add_a_legend_an_axes_never_had(parent):
    figure = plt.figure()
    axis = figure.add_subplot(111)
    axis.plot([0, 1], [1, 2], label="named")
    assert axis.get_legend() is None

    menu = fs.build_figure_context_menu(parent, figure)
    try:
        legend = [a for a in menu.actions() if a.text() == "Legend"][0]
        legend.setChecked(True)

        assert axis.get_legend() is not None, (
            "the labelled series is what a legend would be made of")
    finally:
        menu.deleteLater()
        plt.close(figure)


def test_the_menu_leaves_an_unlabelled_axes_without_a_legend(parent):
    figure = plt.figure()
    axis = figure.add_subplot(111)
    axis.plot([0, 1], [1, 2])

    menu = fs.build_figure_context_menu(parent, figure)
    try:
        legend = [a for a in menu.actions() if a.text() == "Legend"][0]
        legend.setChecked(True)

        assert axis.get_legend() is None, (
            "legend() with nothing labelled warns and makes nothing")
    finally:
        menu.deleteLater()
        plt.close(figure)


# --------------------------------------------------------------------------
# the house-style file entries
# --------------------------------------------------------------------------

def test_a_style_file_reaches_a_listener_that_takes_no_preview_keyword(
        parent, monkeypatch, tmp_path):
    """The preview keyword is an addition; older listeners take no arguments."""
    import json

    style = tmp_path / "house.json"
    style.write_text(json.dumps({
        "spacr_style_kind": fs.GRAPH_STYLE_FILE_KIND,
        "general": {"font_size": 14},
        "per_graph": {"bar": {"bar_width": 0.6}},
    }), encoding="utf-8")

    applied = []
    calls = []
    monkeypatch.setattr(
        "PySide6.QtWidgets.QFileDialog.getOpenFileName",
        staticmethod(lambda *args, **kwargs: (str(style), "")))
    monkeypatch.setattr(fs, "apply_graph_style",
                        lambda general, per_graph: applied.append(
                            (general, per_graph)))

    menu = QMenu(parent)
    fs.add_graph_style_file_entries(menu, parent,
                                    on_change=lambda: calls.append(1))
    [action for action in menu.actions()
     if action.text().startswith("Load")][0].trigger()

    assert applied == [({"font_size": 14}, {"bar": {"bar_width": 0.6}})]
    assert calls == [1]


def test_a_file_that_is_not_a_style_is_reported_and_applies_nothing(
        parent, monkeypatch, tmp_path):
    not_a_style = tmp_path / "something.json"
    not_a_style.write_text('{"hello": 1}', encoding="utf-8")

    applied = []
    warned = []
    monkeypatch.setattr(
        "PySide6.QtWidgets.QFileDialog.getOpenFileName",
        staticmethod(lambda *args, **kwargs: (str(not_a_style), "")))
    monkeypatch.setattr(fs, "apply_graph_style",
                        lambda *args: applied.append(args))
    monkeypatch.setattr(
        "PySide6.QtWidgets.QMessageBox.warning",
        staticmethod(lambda _owner, _title, text: warned.append(text)))

    menu = QMenu(parent)
    fs.add_graph_style_file_entries(menu, parent)
    [action for action in menu.actions()
     if action.text().startswith("Load")][0].trigger()

    assert applied == [], "a file that is not a style changes nothing"
    assert warned and "not a spaCR graph style" in warned[0]


# --------------------------------------------------------------------------
# the rest of the redraw's failure modes
# --------------------------------------------------------------------------

def test_a_drawer_that_returns_no_figure_is_not_reported_as_a_redraw(
        monkeypatch):
    figure = plt.figure()
    figure._spacr_replot = _recipe()
    told = []

    monkeypatch.setattr("spacr.plot.create_grouped_plot",
                        lambda **_kwargs: (None, None))
    try:
        assert fs._replot(figure, "box",
                          on_change=lambda *_a: told.append(1)) is None
    finally:
        plt.close(figure)

    assert told == [], "there is no new figure to tell anyone about"


def test_a_no_argument_listener_that_raises_is_logged_not_propagated(
        monkeypatch, caplog):
    figure = plt.figure()
    figure._spacr_replot = _recipe()
    drawn = plt.figure()

    def explode():
        raise RuntimeError("the toggle's window has gone")

    monkeypatch.setattr("spacr.plot.create_grouped_plot",
                        lambda **_kwargs: (drawn, None))
    try:
        with caplog.at_level(logging.DEBUG, logger=fs.LOG.name):
            assert fs._replot(figure, "box", on_change=explode) is drawn
    finally:
        plt.close(figure)
        plt.close(drawn)

    assert any("redraw notification failed" in record.getMessage()
               for record in caplog.records)


# --------------------------------------------------------------------------
# which types fit
# --------------------------------------------------------------------------

def test_a_type_the_data_cannot_carry_takes_its_aliases_down_with_it(
        monkeypatch):
    """The two modules spell one plot differently; the reason must follow."""
    monkeypatch.setattr(
        "spacr.graph_types.offer",
        lambda *_args, **_kwargs: [
            ("bar_jitter", "Jitter over bar", "only one point per group"),
            ("bar", "Bar", ""),
        ])

    fits, why = fs._which_types_fit(_recipe())

    assert "bar" in fits
    assert why["jitter_bar"] == "only one point per group"
    assert why["jitter_box"] == "only one point per group", (
        "the drawer's name for the same plot gets the same reason")


def test_a_type_the_data_does_carry_brings_its_aliases_with_it(monkeypatch):
    monkeypatch.setattr(
        "spacr.graph_types.offer",
        lambda *_args, **_kwargs: [("bar_jitter", "Jitter over bar", "")])

    fits, _why = fs._which_types_fit(_recipe())

    assert set(fits) >= {"jitter_bar", "jitter_box"}


# --------------------------------------------------------------------------
# group colours: the frames that cannot be read
# --------------------------------------------------------------------------

class _FrameWithNoReadableColumn:
    columns = (fs.DERIVED_GROUP,)

    def __getitem__(self, _key):
        raise RuntimeError("the frame behind this figure has gone")


def test_a_frame_whose_column_cannot_be_read_adds_no_colour_menu(parent):
    menu = QMenu(parent)
    recipe = {"df": _FrameWithNoReadableColumn(),
              "grouping_column": fs.DERIVED_GROUP}

    fs._add_group_colours(menu, None, recipe, None, parent)

    assert menu.actions() == []


def test_a_frame_with_no_rows_adds_no_colour_menu(parent):
    menu = QMenu(parent)
    recipe = {"df": pd.DataFrame({fs.DERIVED_GROUP: [],
                                  fs.DERIVED_VALUE: []}),
              "grouping_column": fs.DERIVED_GROUP}

    fs._add_group_colours(menu, None, recipe, None, parent)

    assert menu.actions() == [], "no groups is not one empty group"

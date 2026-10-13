"""The figure dialog and menu meeting artists and hosts that answer oddly.

Every path here is taken when the thing being styled is not the
well-behaved matplotlib line the module was written against: a colour Qt
cannot parse, a series that can be sized but not outlined, an axes with no
spines, a figure that refuses to keep the recipe read back off its own
artists, a widget that will not hold the save dialog it opened.

The promises are that the dialog opens on any figure spaCR can show, that it
never offers a control which cannot reach the mark it names, and that the
right-click menu still redraws. Each assertion is what a user would see: the
text on a swatch, the rows the form has, the width an artist ended at.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("PySide6")

from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.spines import Spines  # noqa: E402
from PySide6.QtWidgets import QFormLayout, QLabel, QWidget  # noqa: E402

import spacr.graph_types as graph_types  # noqa: E402
from spacr.qt.widgets import figure_settings as fs  # noqa: E402

pytestmark = pytest.mark.qt


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _form_rows(form) -> dict:
    """``{label text: field widget}`` for one form layout."""
    found = {}
    for index in range(form.rowCount()):
        label = form.itemAt(index, QFormLayout.LabelRole)
        field = form.itemAt(index, QFormLayout.FieldRole)
        if label is None or field is None:
            continue
        widget = label.widget()
        if isinstance(widget, QLabel) and field.widget() is not None:
            found.setdefault(widget.text(), field.widget())
    return found


def _rows(root) -> dict:
    """``{label text: field widget}`` for every form row under a widget."""
    found = {}
    for form in root.findChildren(QFormLayout):
        for text, widget in _form_rows(form).items():
            found.setdefault(text, widget)
    return found


class _SizeOnlySeries:
    """A series artist that can be coloured, sized and faded -- no more.

    The dialog decides which controls to build by asking the artist what it
    can do, so an artist narrower than ``Line2D`` is what those ``hasattr``
    guards are for.
    """

    def __init__(self, colour="#00ff00"):
        self.colour = colour
        self.sizes = None
        self.alpha_set = None

    def get_color(self):
        return self.colour

    def set_color(self, colour):
        self.colour = colour

    def get_alpha(self):
        return 0.5

    def set_alpha(self, value):
        self.alpha_set = value

    def set_sizes(self, values):
        self.sizes = list(values)


class _BareSpines(Spines):
    """A spine set that reports itself empty while the axes still draws.

    Named spines stay reachable so matplotlib's transforms keep working;
    only iteration is empty, which is what an axes that draws its own frame
    looks like to this module.
    """

    def __iter__(self):
        return iter(())

    def __len__(self):
        return 0


@pytest.fixture()
def figure():
    """One axes, one labelled line -- the ordinary case to compare against."""
    fig = plt.figure(figsize=(4.0, 3.0))
    axis = fig.add_subplot(111)
    axis.plot([0, 1, 2], [1.0, 2.0, 3.0], label="one")
    axis.set_title("only")
    yield fig
    plt.close(fig)


# ---------------------------------------------------------------------------
# the colour swatch
# ---------------------------------------------------------------------------

def test_a_swatch_shows_a_colour_qt_cannot_parse_without_painting_itself(qapp):
    """A colour Qt rejects must not leave the button painted with another.

    The swatch is both label and sample: it writes the colour into its own
    stylesheet. Painting an unparseable value anyway would leave the button
    showing a colour the figure does not have, so it shows the text and
    stays unpainted instead.
    """
    good = fs._colour_button("red", lambda _colour: None)

    assert good.text() == "#ff0000"
    assert "background-color: #ff0000" in good.styleSheet()

    original = fs._as_hex
    try:
        fs._as_hex = lambda colour, fallback="#1f77b4": "chartreuse-ish"
        bad = fs._colour_button("red", lambda _colour: None)
    finally:
        fs._as_hex = original

    assert bad.text() == "chartreuse-ish", "the value is still readable"
    assert bad.styleSheet() == "", (
        "nothing was painted, so no colour is claimed")


# ---------------------------------------------------------------------------
# the shared rules used past the per-series detail limit
# ---------------------------------------------------------------------------

def test_the_shared_size_rule_reaches_only_the_series_that_can_be_sized(
        qapp, figure):
    """One size control for a volcano's 27 collections must not raise on one.

    Past the detail limit the dialog offers rules instead of a block per
    series. A rule assuming every series takes a point size would raise on
    the first line collection it met, and a raise inside a spin box's signal
    is a dialog that stops responding mid-drag with the sizes half applied.
    """
    axis = figure.axes[0]
    edges = LineCollection([[(0.0, 0.0), (1.0, 1.0)]], linewidths=1.0)
    axis.add_collection(edges)
    points = axis.scatter([0, 1], [1.0, 2.0])
    dialog = fs.FigureSettingsDialog(figure)
    host = QWidget()
    try:
        form = QFormLayout(host)
        dialog._add_series_rules(
            form, axis, [("edges", edges), ("points", points)])
        rows = _form_rows(form)

        assert "Point size (all)" in rows
        rows["Point size (all)"].setValue(120.0)

        assert points.get_sizes().tolist() == [120.0], "the sizeable one grew"
        assert not hasattr(edges, "set_sizes"), (
            "the line collection has no size to set, and was stepped over")
        assert edges.get_linewidth() == [1.0], "so nothing else moved either"
        assert dialog._redraw.isActive(), "and a redraw was still asked for"
    finally:
        host.deleteLater()
        dialog.deleteLater()


def test_the_shared_outline_rule_reaches_only_the_series_that_have_one(
        qapp, figure):
    """The outline rule is offered to every series, including the unoutlined.

    "Outline width (all)" is one control over a mixed set of artists. The
    ones that draw an outline take it; a points-only artist has no line to
    widen and must be stepped over rather than break the control for the
    series that were about to take it.
    """
    axis = figure.axes[0]
    edges = LineCollection([[(0.0, 0.0), (1.0, 1.0)]], linewidths=1.0)
    axis.add_collection(edges)
    dots = _SizeOnlySeries()
    dialog = fs.FigureSettingsDialog(figure)
    host = QWidget()
    try:
        form = QFormLayout(host)
        dialog._add_series_rules(
            form, axis, [("edges", edges), ("dots", dots)])
        rows = _form_rows(form)
        rows["Outline width (all)"].setValue(2.5)

        assert edges.get_linewidth() == [2.5], "the outlined one widened"
        assert not hasattr(dots, "set_linewidth"), (
            "the points-only artist has no outline, and was stepped over")

        # The rules that DO apply to it still reach it, so being stepped
        # over was about this control and not about this artist.
        rows["Point size (all)"].setValue(64.0)
        rows["Opacity (all)"].setValue(0.4)
        assert dots.sizes == [64.0]
        assert dots.alpha_set == pytest.approx(0.4)
    finally:
        host.deleteLater()
        dialog.deleteLater()


# ---------------------------------------------------------------------------
# the figure tab reading its opening colours off the axes
# ---------------------------------------------------------------------------

def test_an_axes_with_no_spines_opens_the_line_colour_at_the_default(qapp):
    """The Line colour swatch must not inherit the font colour by accident.

    It opens showing the first spine's colour, which is the ink it changes.
    An axes with no spines has no such colour, so the swatch opens at the
    plain default: showing the font colour there would name a colour the
    axis lines do not have, and one click would repaint every line with it.
    """
    fig, axis = plt.subplots()
    coloured, coloured_axis = plt.subplots()
    try:
        axis.plot([0, 1], [1.0, 2.0])
        axis.xaxis.label.set_color("#ff0000")
        for spine in axis.spines.values():
            spine.set_edgecolor("#00ff00")
        axis.spines = _BareSpines(**dict(axis.spines._dict))
        fig.canvas.draw()

        bare_dialog = fs.FigureSettingsDialog(fig)
        try:
            rows = _rows(bare_dialog)
            assert rows["Font colour"].text() == "#ff0000", (
                "the label's colour is still read")
            assert rows["Line colour"].text() == "#000000", (
                "no spine to read, so the default rather than the font ink")
            assert rows["Spine width"].value() == pytest.approx(1.0)
        finally:
            bare_dialog.deleteLater()

        # The same figure WITH its spines reads the spine colour, which is
        # what makes the fallback above a fallback.
        coloured_axis.plot([0, 1], [1.0, 2.0])
        coloured_axis.xaxis.label.set_color("#ff0000")
        for spine in coloured_axis.spines.values():
            spine.set_edgecolor("#00ff00")
        full_dialog = fs.FigureSettingsDialog(coloured)
        try:
            assert _rows(full_dialog)["Line colour"].text() == "#00ff00"
        finally:
            full_dialog.deleteLater()
    finally:
        plt.close(fig)
        plt.close(coloured)


# ---------------------------------------------------------------------------
# per-artist controls are built from what the artist can do
# ---------------------------------------------------------------------------

def test_an_artist_is_only_offered_the_controls_it_can_take(qapp, figure,
                                                            monkeypatch):
    """A control the artist cannot answer is a control that does nothing.

    The per-series block is assembled by asking the artist what it supports.
    An artist that can only be coloured, sized and faded must not get a line
    width or line style row: they would take a value and change nothing on
    screen -- the failure reported as "the settings do not work".
    """
    line_dialog = fs.FigureSettingsDialog(figure)
    try:
        line_rows = _rows(line_dialog)
        assert "  Line width" in line_rows and "  Line style" in line_rows, (
            "a Line2D takes both, so both are offered")
    finally:
        line_dialog.deleteLater()

    dots = _SizeOnlySeries()
    monkeypatch.setattr(fs, "_series_of", lambda axis: [("dots", dots)])
    dots_dialog = fs.FigureSettingsDialog(figure)
    try:
        rows = _rows(dots_dialog)

        assert "  Line width" not in rows, "it has no width to set"
        assert "  Line style" not in rows, "and no style either"
        assert "  Marker size" not in rows, "and no marker"
        assert rows["  Colour"].text() == "#00ff00", "its colour is read back"
        assert rows["  Opacity"].value() == pytest.approx(0.5)

        rows["  Point size"].setValue(90.0)
        assert dots.sizes == [90.0], "and the control it did get reaches it"
    finally:
        dots_dialog.deleteLater()


# ---------------------------------------------------------------------------
# reading a recipe back off a figure's artists
# ---------------------------------------------------------------------------

def test_a_reconstruction_names_every_group_it_finds():
    """A redraw needs groups to draw, and it always has them.

    This used to assert the opposite half too: that a recipe whose groups
    were all NaN is refused. That guard was deleted on 2026-08-31 --
    instruction 310 A15 -- because no figure can produce it. The group
    names come only from `_named`, which returns a tick label or
    `f"{float(x):g}"`, so a NaN x arrives as the STRING "nan" and the
    column is never empty.

    Reaching it needed `_pairs_from_axes` monkeypatched to return real
    NaNs, which is not a figure -- and A15's own note says that was the
    only way. A test that manufactures an impossible state to cover a
    guard is testing the monkeypatch.

    What survives is the half with a subject: a figure with named bars
    reconstructs into named groups. The premise underneath is pinned in
    tests/qt/test_a_replot_recipe_always_has_named_groups.py.
    """
    fig, axis = plt.subplots()
    try:
        axis.bar(["ctrl", "treat"], [3.0, 5.0])

        recipe = fs.derive_replot_recipe(fig)
        assert recipe is not None
        assert sorted(recipe["df"]["group"]) == ["ctrl", "treat"], (
            "named bars reconstruct into named groups")
    finally:
        plt.close(fig)


def test_the_jitter_alias_is_only_offered_when_the_table_names_it(qapp):
    """The menu may only offer what the drawer can actually draw.

    ``graph_types`` says ``bar_jitter`` where the drawer says ``jitter_bar``
    and ``jitter_box``, so those two inherit whatever the table says about
    ``bar_jitter``. Were the table to stop listing it, inventing the aliases
    anyway would put entries on the menu the table never approved.
    """
    frame = pandas.DataFrame({"group": ["a", "a", "b", "b"],
                              "value": [1.0, 2.0, 3.0, 4.0]})
    recipe = {"df": frame, "grouping_column": "group",
              "data_column": "value"}
    original = graph_types.offer
    try:
        graph_types.offer = lambda f, x="", y="": [
            ("bar", "Bar", ""), ("bar_jitter", "Bar with jitter", "")]
        fits, why = fs._which_types_fit(recipe)
        assert set(fits) == {"bar", "bar_jitter", "jitter_bar", "jitter_box"}
        assert why == {}

        graph_types.offer = lambda f, x="", y="": [
            ("bar", "Bar", ""),
            ("bar_jitter", "Bar with jitter", "needs the observations")]
        fits, why = fs._which_types_fit(recipe)
        assert fits == ("bar",)
        assert why["jitter_bar"] == "needs the observations", (
            "the reason is carried over to the drawer's own names")

        graph_types.offer = lambda f, x="", y="": [
            ("bar", "Bar", ""), ("violin", "Violin", "too few points")]
        fits, why = fs._which_types_fit(recipe)
        assert fits == ("bar",), "no alias was invented for an absent type"
        assert why == {"violin": "too few points"}
    finally:
        graph_types.offer = original


# ---------------------------------------------------------------------------
# the context menu and the styled save on hosts that will not hold anything
# ---------------------------------------------------------------------------

def test_a_figure_that_will_not_keep_the_derived_recipe_still_gets_the_menu(
        qapp):
    """Caching the recipe is an optimisation; the menu is the feature.

    A recipe read off the artists is stored on the figure so the next
    right-click does not repeat the work. A figure that refuses the
    attribute must still get its "Graph type" submenu: losing the cache
    costs milliseconds, losing the menu loses the feature.
    """
    class Locked(Figure):
        """A figure that will not take spaCR's cached recipe."""

        def __setattr__(self, name, value):
            if name == "_spacr_replot":
                raise RuntimeError("this figure will not take a recipe")
            super().__setattr__(name, value)

    locked = Locked()
    locked.add_subplot(111).bar(["ctrl", "treat", "ko"], [3.0, 5.0, 2.0])
    ordinary, axis = plt.subplots()
    host = QWidget()
    try:
        axis.bar(["ctrl", "treat", "ko"], [3.0, 5.0, 2.0])

        kept_menu = fs.build_figure_context_menu(host, ordinary)
        assert "Graph type" in [a.menu().title() for a in kept_menu.actions()
                                if a.menu() is not None]
        assert ordinary._spacr_replot["grouping_column"] == "group", (
            "an ordinary figure keeps what was derived")
        assert ordinary._spacr_replot_derived is True

        menu = fs.build_figure_context_menu(host, locked)
        titles = [a.menu().title() for a in menu.actions()
                  if a.menu() is not None]

        assert "Graph type" in titles, "the menu is built from the recipe"
        assert getattr(locked, "_spacr_replot", None) is None, (
            "and the refusal of the cache was swallowed")
        kept_menu.deleteLater()
        menu.deleteLater()
    finally:
        host.deleteLater()
        plt.close(ordinary)


def test_a_parent_that_cannot_hold_the_save_dialog_still_gets_it_back(
        qapp, figure):
    """The caller can still keep the window even when the parent cannot.

    The styled save dialog is parked on the widget that opened it so Python
    does not collect it mid-use. A host that will not take the attribute
    must still be handed the dialog back: that return value is the caller's
    last chance to keep the window from vanishing on the next collection.
    """
    class Deaf(QWidget):
        """A host that refuses to store anything for anyone."""

        def __setattr__(self, name, value):
            if name == "_spacr_save_dialogs":
                raise RuntimeError("nothing may be kept here")
            super().__setattr__(name, value)

    ordinary = QWidget()
    deaf = Deaf()
    kept = fs._open_styled_save(ordinary, figure)
    refused = fs._open_styled_save(deaf, figure)
    try:
        assert ordinary._spacr_save_dialogs == [kept], (
            "an ordinary parent holds the dialog it opened")
        assert refused is not None and refused.parent() is deaf
        assert getattr(deaf, "_spacr_save_dialogs", None) is None, (
            "this one could not hold it, so the caller has the only handle")
    finally:
        for dialog in (kept, refused):
            if dialog is not None:
                dialog.close()
                dialog.deleteLater()
        ordinary.deleteLater()
        deaf.deleteLater()

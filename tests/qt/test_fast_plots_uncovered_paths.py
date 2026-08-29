"""The defensive paths of the interactive plots, driven rather than assumed.

Every guard in ``fast_plots`` stands between the user and a traceback: a
settings store that is not there, an axis a subclass removed, an exporter
from an older pyqtgraph, a brush that is not a brush. Each one exists so the
plot keeps drawing when the thing behind it is missing, and a guard nothing
has ever taken is a guard nobody knows the shape of.

So each test here BREAKS the collaborator the guard names and then asks the
widget a question a user could ask -- what does the menu say, what did the
axis end up painted, how many rings are on the plot, what did the exporter
receive -- rather than watching the guard swallow something.
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

import pyqtgraph as pg                                  # noqa: E402
from PySide6.QtCore import QRectF                       # noqa: E402
from PySide6.QtGui import QColor                        # noqa: E402
from PySide6.QtWidgets import QMenu                     # noqa: E402

from spacr.qt.widgets import fast_plots as fp           # noqa: E402

pytestmark = pytest.mark.qt


# --------------------------------------------------------------------------- #
#  helpers
# --------------------------------------------------------------------------- #

def _blank_module(monkeypatch, dotted: str) -> None:
    """Make ``from <dotted> import anything`` raise ImportError.

    A module object with none of the names in it is what a partial upgrade
    or a stripped wheel looks like from the import statement's point of
    view -- the module resolves and the attribute does not -- which is the
    shape these guards were written against.
    """
    monkeypatch.setitem(sys.modules, dotted, types.ModuleType(dotted))


def _raiser(message: str):
    """A callable that always raises ``RuntimeError(message)``."""
    def raise_it(*_args, **_kwargs):
        raise RuntimeError(message)
    return raise_it


@pytest.fixture
def style():
    from spacr.volcano_style import VolcanoStyle

    return VolcanoStyle()


@pytest.fixture
def plot(qtbot):
    """A bare FastPlot with two labelled axes and nothing drawn on it."""
    widget = fp.FastPlot(title="fast", x_label="x", y_label="y")
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def scatter_plot(qtbot):
    """A FastPlot carrying six keyed points, the shape a volcano has."""
    widget = fp.FastPlot(title="fast", x_label="x", y_label="y")
    qtbot.addWidget(widget)
    widget.set_keys([f"g{i}" for i in range(6)])
    widget.add_scatter(np.arange(6, dtype="float64"),
                       np.arange(6, dtype="float64") + 1.0, size=6)
    return widget


class _RaisingAxes:
    """A plot item whose ``getAxis`` always fails; everything else is real."""

    def __init__(self, wrapped=None):
        self._wrapped = wrapped

    def getAxis(self, _edge):
        raise RuntimeError("this plot has no such axis")

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


def _lose_the_axes(monkeypatch, widget) -> None:
    """Make every ``getAxis`` on ``widget``'s plot fail, restorably."""
    monkeypatch.setattr(widget.plot, "plotItem",
                        _RaisingAxes(widget.plot.plotItem))


# --------------------------------------------------------------------------- #
#  no settings store
# --------------------------------------------------------------------------- #

def test_saving_a_house_style_without_a_settings_store_says_so(monkeypatch,
                                                               style):
    """"Use as the default" reports the missing store instead of lying."""
    said: list = []
    menu = QMenu()
    fp.add_style_file_entries(menu, style, note=said.append,
                              ask_path=lambda mode, name: "")
    make_default = [a for a in menu.actions()
                    if a.text().startswith("Use as the default")][0]

    _blank_module(monkeypatch, "spacr.qt.preferences")
    make_default.trigger()

    assert said == ["There is no settings store to save a default into."]


def test_clearing_a_house_style_says_nothing_once_the_store_has_gone(
        monkeypatch, style, tmp_path):
    """"Clear the default" is silent when the store goes between the two.

    The entry is only ENABLED because a default was read back a moment
    earlier, so a store that has become unreadable since is a race rather
    than something to put on the status line -- and the enabled state is
    itself the readout for "is a house style in force here?".
    """
    from PySide6.QtCore import QSettings
    from spacr.qt import preferences

    store = tmp_path / "prefs.ini"
    monkeypatch.setattr(
        preferences, "_settings",
        lambda: QSettings(str(store), QSettings.Format.IniFormat))
    preferences.set_figure_style_default("volcano", fp.style_as_dict(style))

    said: list = []
    menu = QMenu()
    fp.add_style_file_entries(menu, style, note=said.append,
                              ask_path=lambda mode, name: "")
    clear = [a for a in menu.actions() if a.text() == "Clear the default"][0]
    assert clear.isEnabled() is True

    _blank_module(monkeypatch, "spacr.qt.preferences")
    clear.trigger()

    assert said == []


def test_the_clear_entry_is_greyed_when_the_store_cannot_be_read(monkeypatch,
                                                                 style):
    """No store means no default to clear, so the entry says so and greys."""
    _blank_module(monkeypatch, "spacr.qt.preferences")
    menu = QMenu()
    fp.add_style_file_entries(menu, style, ask_path=lambda mode, name: "")

    clear = [a for a in menu.actions() if a.text() == "Clear the default"][0]
    assert clear.isEnabled() is False
    assert clear.toolTip() == "No volcano default is saved."


def test_a_default_style_cannot_be_applied_without_a_settings_store(
        monkeypatch, style):
    """``apply_default_style`` changes no field when the store is missing."""
    _blank_module(monkeypatch, "spacr.qt.preferences")
    before = fp.style_as_dict(style)

    assert fp.apply_default_style(style) == []
    assert fp.style_as_dict(style) == before


def test_figure_colours_fall_back_to_white_ink_without_a_settings_store(
        monkeypatch):
    """A plot built with no preferences still gets visible axes.

    Transparent ground and white ink, because every spaCR theme but one is
    dark: black on transparent would be an invisible axis.
    """
    _blank_module(monkeypatch, "spacr.qt.preferences")

    assert fp._figure_colors() == ("none", fp._FALLBACK_FOREGROUND)


def test_a_plot_still_builds_when_the_theme_module_is_missing(monkeypatch,
                                                              qtbot):
    """No ``make_transparent`` is not a reason to have no plot at all."""
    _blank_module(monkeypatch, "spacr.qt.theme")

    widget = fp.FastPlot(title="themeless", x_label="x", y_label="y")
    qtbot.addWidget(widget)

    assert widget.plots_available is True
    assert widget.plot.getAxis("bottom").labelText == "x"


# --------------------------------------------------------------------------- #
#  a column that cannot be counted
# --------------------------------------------------------------------------- #

class _Unprintable:
    """A cell that refuses to become a string, as a badly-typed column does."""

    def __str__(self):
        raise TypeError("this value has no text form")

    def __repr__(self):
        return "<unprintable>"


def test_a_column_that_cannot_be_stringified_is_not_offered_as_a_shape(plot):
    """One unreadable column costs that column, not the whole shape menu."""
    # The table a plot was drawn from; subclasses set it the same way.
    plot._frame = pd.DataFrame({
        "n_guides": [1, 2, 1, 2],
        "junk": [_Unprintable() for _ in range(4)],
    })

    assert plot.shape_columns() == ["n_guides"]


# --------------------------------------------------------------------------- #
#  items the scale cannot read
# --------------------------------------------------------------------------- #

def test_an_oblique_reference_line_is_drawn_but_never_rescaled(plot):
    """A 45-degree line stands for neither an x nor a y, so it is left be.

    It is still ON the plot -- refusing to record it is not refusing to draw
    it -- and a log switch simply has nothing to rescale it by.
    """
    line = pg.InfiniteLine(pos=(0.0, 0.0), angle=45.0)
    plot.plot.addItem(line)

    assert line in plot.plot.plotItem.items
    assert all(record.item is not line for record in plot._drawn)


def test_bars_that_cannot_be_re_measured_block_the_log_scale_by_name(plot):
    """The refusal names the bar rather than silently leaving it behind."""
    bars = pg.BarGraphItem(x=[0.0, 1.0], height=[1.0, 2.0], width=0.5)
    bars.boundingRect()     # cache the picture so drawing stays possible
    bars._getNormalizedCoords = _raiser("this bar cannot be re-measured")
    plot.plot.addItem(bars)

    assert plot.log_reason("y") == \
        "log y: one of the bars cannot be re-measured"
    assert plot.log_reason("x") == \
        "log x: one of the bars cannot be re-measured"
    assert plot.set_log_axes(y=True) == (False, False)


def test_a_curve_that_cannot_report_its_data_is_ignored_by_the_scale(plot):
    """An unreadable curve is skipped; the readable one still counts."""
    mute = pg.PlotDataItem()
    mute.getData = _raiser("nothing has been set on this curve")
    plot.plot.addItem(mute)
    good = pg.PlotDataItem([1.0, 2.0], [3.0, 4.0])
    plot.plot.addItem(good)

    assert [record.item for record in plot._drawn] == [good]


def test_a_view_box_with_no_manual_range_signal_still_gives_a_plot(monkeypatch,
                                                                   qtbot):
    """An older ViewBox without ``sigRangeChangedManually`` costs the pins.

    The plot must still build and still label its axes: the signal only
    drives "a hand-drag forgets a typed limit", which is a refinement rather
    than the plot.
    """
    class _NoSignal:
        def __getattr__(self, name):
            raise AttributeError(name)

    built = pg.PlotWidget.__init__

    def without_the_signal(self, *args, **kwargs):
        built(self, *args, **kwargs)
        self.plotItem.vb.sigRangeChangedManually = _NoSignal()

    monkeypatch.setattr(pg.PlotWidget, "__init__", without_the_signal)
    widget = fp.FastPlot(title="old", x_label="x", y_label="y")
    qtbot.addWidget(widget)

    assert widget.plot.getAxis("left").labelText == "y"
    assert widget.log_axes() == (False, False)


# --------------------------------------------------------------------------- #
#  an axis that is not there
# --------------------------------------------------------------------------- #

def test_a_plot_whose_axis_cannot_be_read_still_judges_the_log_scale(
        plot, monkeypatch):
    """``log_reason`` answers from the DATA when the axis cannot be asked."""
    plot.add_scatter(np.array([1.0, 2.0]), np.array([-1.0, 3.0]))
    _lose_the_axes(monkeypatch, plot)

    reason = plot.log_reason("y")

    assert reason == ("log y: 1 of 2 points are at or below zero and have "
                      "no logarithm")


def test_axis_items_is_empty_when_no_axis_can_be_fetched(plot, monkeypatch):
    """The line control finds nothing to paint rather than falling over."""
    _lose_the_axes(monkeypatch, plot)

    assert plot.axis_items() == []
    assert plot.set_line_colour("#123456") == 0


def test_the_font_control_still_repaints_the_title_without_axes(plot,
                                                                monkeypatch):
    """Title colour is applied even when neither axis can be reached."""
    plot.plot.setTitle("a title")
    _lose_the_axes(monkeypatch, plot)

    plot.set_font_colour("#00ff00")

    assert plot.plot.plotItem.titleLabel.opts["color"] == "#00ff00"


def test_a_restyle_without_axes_still_repaints_the_title(plot, monkeypatch):
    """A theme switch on a plot with unreachable axes still moves the title."""
    plot.plot.setTitle("a title")
    _lose_the_axes(monkeypatch, plot)

    plot.restyle(background="none", foreground="#ff00ff")

    assert plot._foreground == "#ff00ff"
    assert plot.plot.plotItem.titleLabel.opts["color"] == "#ff00ff"


def test_a_split_without_a_left_axis_is_recorded_but_writes_no_ticks(
        plot, monkeypatch):
    """The label says the band is gone; the ruler is left as pyqtgraph's.

    Rewriting the tick strings is what makes a split honest, so an axis that
    cannot be fetched must leave them exactly as they were rather than
    half-installing a ruler that reads one thing and measures another.
    """
    plot.add_scatter(np.array([0.0, 1.0]), np.array([0.0, 100.0]))
    left = plot.plot.getAxis("left")
    _lose_the_axes(monkeypatch, plot)

    assert plot.set_y_split(10.0, 90.0) == ""

    assert left.labelText == "y (split, 10-90 not drawn)"
    assert "tickStrings" not in left.__dict__
    assert "tickValues" not in left.__dict__


def test_the_print_look_skips_an_item_whose_axes_cannot_be_fetched(monkeypatch):
    """A chrome repaint that cannot read an axis has nothing to undo."""
    stub = types.ModuleType("spacr.figure_style")
    stub.saved_figure_appearance = lambda: types.SimpleNamespace(
        mode="print", ground="#ffffff")
    stub.export_colour = lambda current, kind, look: "#000000"
    monkeypatch.setitem(sys.modules, "spacr.figure_style", stub)

    assert fp.FastPlot._wear_the_print_look(_RaisingAxes()) == []


class _NoTickPen:
    """An axis from a pyqtgraph too old to separate the tick pen."""

    def __init__(self, wrapped):
        self._wrapped = wrapped

    def setTickPen(self, _pen):
        raise TypeError("this pyqtgraph has no tick pen")

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


def test_an_axis_without_a_tick_pen_still_gets_its_spine_painted(monkeypatch,
                                                                 plot):
    """The spine is the part that must not be lost on an old pyqtgraph."""
    axes = [_NoTickPen(plot.plot.getAxis(edge))
            for edge in ("bottom", "left", "top", "right")]
    monkeypatch.setattr(plot, "axis_items", lambda: axes)

    plot.set_line_colour("#010203")

    assert [axis._wrapped.pen().color().name() for axis in axes] == \
        ["#010203"] * 4


# --------------------------------------------------------------------------- #
#  text on things that will not take it
# --------------------------------------------------------------------------- #

class _AwkwardLabel:
    """A line label that refuses a colour."""

    def setColor(self, _colour):
        raise RuntimeError("this label takes no colour")


def test_a_line_label_that_refuses_a_colour_does_not_stop_the_others(plot):
    """Every other caption still gets the ink; the awkward one is skipped."""
    plot.add_line(y=0.5, label="p=0.05")
    stubborn = pg.InfiniteLine(pos=2.0, angle=0.0)
    stubborn.label = _AwkwardLabel()
    plot.plot.addItem(stubborn)

    plot.set_font_colour("#ff8800")

    captions = [item.label for item in plot.line_items()
                if isinstance(getattr(item, "label", None), pg.InfLineLabel)]
    assert captions, "the readable threshold caption should still be there"
    for caption in captions:
        assert QColor(caption.color).name() == "#ff8800"


class _AwkwardLegendText:
    """A legend entry whose text cannot be re-set."""

    text = "a level"

    def setText(self, *_args, **_kwargs):
        raise RuntimeError("this legend item takes no text")


def test_a_legend_entry_that_refuses_a_recolour_leaves_the_axes_painted(plot):
    """One odd legend entry costs that entry, not the whole restyle."""
    plot.plot.addLegend()
    plot.plot.plotItem.legend.items = [(None, _AwkwardLegendText())]

    plot.set_font_colour("#22aa44")

    assert plot.plot.getAxis("left").textPen().color().name() == "#22aa44"


# --------------------------------------------------------------------------- #
#  rings and legends that have already gone
# --------------------------------------------------------------------------- #

def _make_removal_fail(monkeypatch, widget):
    """Make ``removeItem`` on this plot raise, as a torn-down scene does."""
    monkeypatch.setattr(widget.plot, "removeItem",
                        _raiser("that item is no longer in this scene"))


def test_a_legend_is_forgotten_even_when_the_scene_will_not_release_it(
        plot, monkeypatch):
    """Turning the legend off clears the reference whatever the scene says."""
    plot._legend_colours = {"hit": "#C44E52"}
    plot._toggle_legend(True)
    assert plot.plot.plotItem.legend is not None

    monkeypatch.setattr(plot.plot.plotItem.scene(), "removeItem",
                        _raiser("already detached"))
    plot._toggle_legend(False)

    assert plot.plot.plotItem.legend is None


def test_a_single_selection_still_clears_when_the_ring_cannot_be_removed(
        scatter_plot, monkeypatch):
    """A ring that will not come off does not keep the key selected."""
    assert scatter_plot.highlight_key("g2") is True
    _make_removal_fail(monkeypatch, scatter_plot)

    assert scatter_plot.highlight_key(None) is False
    assert scatter_plot.selected_keys() == []
    assert scatter_plot._highlight is None


def test_a_multi_selection_still_replaces_when_a_ring_cannot_be_removed(
        scatter_plot, monkeypatch):
    """The new set of keys wins even if the old rings stay in the scene."""
    assert scatter_plot.highlight_keys(["g0", "g1", "g2"]) == 3
    _make_removal_fail(monkeypatch, scatter_plot)

    assert scatter_plot.highlight_keys(["g4"]) == 1

    assert scatter_plot.selected_keys() == ["g4"]
    assert scatter_plot._extra_highlights == []


def test_a_histogram_bar_can_be_re_outlined_when_the_old_ring_is_gone(
        qtbot, monkeypatch):
    """A second click on a bar outlines that bar rather than raising."""
    histogram = fp.PValueHistogram()
    qtbot.addWidget(histogram)
    histogram.set_p_values(np.linspace(0.01, 0.99, 40))
    assert histogram.highlight_bin(1) is True

    _make_removal_fail(monkeypatch, histogram)

    assert histogram.highlight_bin(2) is True
    assert histogram._highlight.opts["x0"] == [histogram._edges[2]]


# --------------------------------------------------------------------------- #
#  a click that lands nowhere
# --------------------------------------------------------------------------- #

def test_a_click_that_cannot_be_mapped_selects_no_bar(qtbot):
    """A press with no view to map into leaves the selection as it was."""
    histogram = fp.PValueHistogram()
    qtbot.addWidget(histogram)
    histogram.set_p_values(np.linspace(0.01, 0.99, 40),
                           keys=[f"g{i}" for i in range(40)])
    histogram.select_bin(0)
    before = histogram.selected_keys()
    assert before, "the first bar should hold at least one coefficient"

    class _UnmappableEvent:
        def button(self):
            raise RuntimeError("this event has no button")

    histogram._on_scene_clicked(_UnmappableEvent())

    assert histogram.selected_keys() == before


# --------------------------------------------------------------------------- #
#  exporters from another pyqtgraph
# --------------------------------------------------------------------------- #

class _StubParameters(dict):
    """``exporter.parameters()`` that refuses the keys it does not know."""

    def __init__(self, known=(), **values):
        super().__init__(**values)
        self._known = set(known)

    def __setitem__(self, key, value):
        if key not in self._known:
            raise KeyError(key)
        super().__setitem__(key, value)

    def param(self, name):
        raise KeyError(name)


class _StubExporter:
    """The smallest thing shaped like a pyqtgraph ImageExporter."""

    def __init__(self, parameters):
        self._parameters = parameters
        self.exported_to = None
        self.widthChanged = object()
        self.heightChanged = object()

    def parameters(self):
        return self._parameters

    def getSourceRect(self):
        return QRectF(0.0, 0.0, 400.0, 300.0)

    def export(self, path=None, **_kwargs):
        self.exported_to = path
        return None


def test_a_raster_export_still_writes_when_the_page_colour_is_refused(
        plot, tmp_path):
    """An exporter with no ``background`` parameter still produces the file.

    The page colour is a preference; the file is the request. Losing the
    first must not lose the second.
    """
    parameters = _StubParameters()
    exporter = _StubExporter(parameters)
    exporters = types.SimpleNamespace(ImageExporter=lambda _item: exporter)
    path = tmp_path / "plot.png"

    plot._write_export(plot.plot.plotItem, str(path), 120.0, 90.0, exporters)

    assert exporter.exported_to == str(path)
    assert "background" not in parameters


def test_an_exporter_with_another_api_keeps_the_size_it_had(plot):
    """``_shape_the_image`` gives up on the size rather than on the export."""
    parameters = _StubParameters(known=("width", "height"))
    dict.__setitem__(parameters, "width", 42)
    dict.__setitem__(parameters, "height", 42)

    plot._shape_the_image(_StubExporter(parameters), 120.0, 90.0)

    assert (parameters["width"], parameters["height"]) == (42, 42)


def test_a_snapshot_survives_an_exporter_that_takes_neither_shape_nor_ground(
        plot, monkeypatch):
    """Neither the aspect nor the transparent page is worth losing the tile."""
    parameters = _StubParameters(known=("width",))
    exporter = _StubExporter(parameters)
    exporters = types.SimpleNamespace(ImageExporter=lambda _item: exporter)
    monkeypatch.setattr(plot, "canvas_ratio", lambda: 0.75)

    assert plot._render_snapshot(exporters, 320, None) is None
    assert parameters["width"] == 320
    assert "background" not in parameters


# --------------------------------------------------------------------------- #
#  brushes that are not brushes
# --------------------------------------------------------------------------- #

class _NotABrush:
    """Something in the brush list that cannot say what colour it is."""

    def color(self):
        raise AttributeError("this is not a brush")


def test_an_unreadable_brush_falls_back_to_the_first_palette_colour(qtbot):
    """The point keeps its q-fade even when its base colour cannot be read."""
    volcano = fp.VolcanoPlot()
    qtbot.addWidget(volcano)
    q = np.array([1e-6, 0.5, 1.0])
    brushes = [pg.mkBrush(QColor("#4C72B0")), _NotABrush(),
               pg.mkBrush(QColor("#DD8452"))]

    faded = volcano._q_opacity(brushes, q, 3)

    assert faded[1].color().name() == QColor(fp.colour_for(0)).name()
    assert faded[0].color().alpha() > faded[2].color().alpha()


# --------------------------------------------------------------------------- #
#  the shared label rule
# --------------------------------------------------------------------------- #

def test_coefficient_labels_fall_back_to_the_named_column(monkeypatch):
    """Without ``figures.panels`` the label column itself is used."""
    _blank_module(monkeypatch, "spacr.figures.panels")
    frame = pd.DataFrame({"feature": ["a", "b"], "gene": ["G1", "G2"]})

    labels = fp.EffectRankPlot._label_series(frame, "gene")

    assert list(labels) == ["G1", "G2"]


def test_coefficient_labels_fall_back_to_row_numbers_without_the_column(
        monkeypatch):
    """No panels module and no such column leaves the row index as the name."""
    _blank_module(monkeypatch, "spacr.figures.panels")
    frame = pd.DataFrame({"coefficient": [0.1, 0.2, 0.3]})

    labels = fp.EffectRankPlot._label_series(frame, "gene")

    assert list(labels) == ["0", "1", "2"]

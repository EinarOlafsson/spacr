"""The refusals, fallbacks and re-placements ``fast_plots`` answers with.

Every path here is one a user meets when something is not ideal: a dialog
they cancelled, a smoother that ran out of memory, a build with no
pyqtgraph, a typed limit that cannot survive a log scale, an axis nobody
named, a broken y-axis being put back together. Each exists so the figure
keeps being a figure, so every test below breaks the collaborator and then
asks what a user would ask: what does the axis read, where did the bar end
up, what is in the status line, what came out restyled.
"""
from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

import pyqtgraph as pg  # noqa: E402
from PySide6.QtCore import QSize  # noqa: E402
from PySide6.QtGui import QColor, QResizeEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QInputDialog  # noqa: E402

from spacr.qt.widgets import fast_plots as fp  # noqa: E402

pytestmark = pytest.mark.qt


@dataclasses.dataclass
class _HouseStyle:
    """A figure style shaped like the ones this package ships."""

    point_colour: str = "#112233"
    point_size: int = 6
    limits: tuple = (0.0, 1.0)
    caption: str = "before"
    compartments: tuple = ("cell", "nucleus")

    CHOICES = {"marker": ("dot", "cross")}


@pytest.fixture
def plot(qtbot):
    """A live FastPlot with both axes named and nothing drawn yet."""
    widget = fp.FastPlot(title="fast", x_label="ex", y_label="why")
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def points(qtbot):
    """A FastPlot carrying three strictly positive points."""
    widget = fp.FastPlot(title="fast", x_label="ex", y_label="why")
    qtbot.addWidget(widget)
    widget.add_scatter(np.array([1.0, 2.0, 3.0]),
                       np.array([1.0, 2.0, 300.0]), size=6)
    return widget


# --------------------------------------------------------------- style fields

def test_a_style_menu_finds_a_closed_set_and_a_stable_kind_for_the_default():
    """Two ways a settings menu misreads a style it was not written for.

    A field whose legal values are declared on the CLASS is found only by
    walking past every dataclass field without matching one; stop early and
    the entry becomes a free-text box a user can type "corss" into. The saved
    default is keyed on the kind, so a class not named ``...Style`` still
    needs its own key or one lab's colours open on another figure.
    """
    style = _HouseStyle()
    assert fp.style_field_choices(style, "marker") == ("dot", "cross")
    assert fp.style_field_choices(style, "caption") == ()
    assert fp.style_field_choices(style, "caption",
                                  {"caption": ("a", "b")}) == ("a", "b")

    class RankedBars:
        pass

    class RankedBarsStyle:
        pass

    assert fp.style_kind(RankedBars()) == "ranked_bars"
    assert fp.style_kind(RankedBarsStyle()) == "ranked_bars"


def test_unticking_a_compartment_removes_it_and_unticking_it_again_is_calm():
    """Un-ticking is the way back out of a multi-select, and it must work twice.

    A user who ticks "nucleus" and changes their mind has to reach the value
    they started from, and a menu rebuilt from a style file can hand this an
    option that is already absent -- if that removed something else the
    compartment list would drift every time the menu was reopened.
    """
    style, seen = _HouseStyle(), []
    order = ("cell", "nucleus", "pathogen")

    def _note(name, value):
        seen.append((name, value))

    fp._toggle_style_member(style, "compartments", "pathogen", True, order,
                            _note)
    assert style.compartments == ("cell", "nucleus", "pathogen")
    fp._toggle_style_member(style, "compartments", "nucleus", False, order,
                            _note)
    assert style.compartments == ("cell", "pathogen")
    fp._toggle_style_member(style, "compartments", "nucleus", False, order,
                            _note)
    assert style.compartments == ("cell", "pathogen")
    assert [name for name, _ in seen] == ["compartments"] * 3
    assert seen[-1][1] == ("cell", "pathogen")


def test_a_cancelled_style_dialog_leaves_every_kind_of_field_alone(monkeypatch):
    """Cancel has to mean cancel, or opening the menu edits the figure.

    Each dialog hands back a real-looking value beside its outcome flag -- an
    invalid QColor reads as black, a cancelled spin box reads as whatever it
    showed -- so writing it back would repaint the figure of anybody who
    opened the entry to look and pressed Escape. A pair is two prompts making
    one value, so abandoning either has to abandon both.
    """
    style, answers = _HouseStyle(), []
    monkeypatch.setattr(fp, "pick_colour",
                        lambda *_a, **_k: QColor("#ff8800"))
    fp._ask_style_value(None, style, "point_colour", "#112233", "colour", None)
    assert style.point_colour == "#ff8800"
    monkeypatch.setattr(fp, "pick_colour", lambda *_a, **_k: QColor())
    fp._ask_style_value(None, style, "point_colour", "#ff8800", "colour", None)
    assert style.point_colour == "#ff8800"

    monkeypatch.setattr(QInputDialog, "getDouble",
                        staticmethod(lambda *_a, **_k: answers.pop(0)))
    answers[:] = [(11.4, True)]
    fp._ask_style_value(None, style, "point_size", 6, "number", None)
    assert style.point_size == 11
    answers[:] = [(99.0, False)]
    fp._ask_style_value(None, style, "point_size", 11, "number", None)
    assert style.point_size == 11

    answers[:] = [(2.0, True), (8.0, True)]
    fp._ask_style_value(None, style, "limits", (0.0, 1.0), "pair", None)
    assert style.limits == (2.0, 8.0)
    answers[:] = [(5.0, False)]
    fp._ask_style_value(None, style, "limits", (2.0, 8.0), "pair", None)
    assert style.limits == (2.0, 8.0) and answers == []

    monkeypatch.setattr(QInputDialog, "getText",
                        staticmethod(lambda *_a, **_k: ("after", True)))
    fp._ask_style_value(None, style, "caption", "before", "text", None)
    assert style.caption == "after"
    monkeypatch.setattr(QInputDialog, "getText",
                        staticmethod(lambda *_a, **_k: ("wiped", False)))
    fp._ask_style_value(None, style, "caption", "after", "text", None)
    assert style.caption == "after"


def test_the_hard_pyqtgraph_guard_is_silent_while_the_library_is_there(
        monkeypatch):
    """The guard must be free when it passes and explicit when it does not.

    It stands in front of the few operations with no graceful fallback: if it
    raised on a healthy install nothing would draw, and if it stayed quiet on
    a stripped one the caller would fail three frames later with a traceback
    naming a symptom instead of the missing wheel.
    """
    assert fp._require_pyqtgraph() is None

    monkeypatch.setattr(fp, "HAVE_PYQTGRAPH", False)
    with pytest.raises(RuntimeError) as caught:
        fp._require_pyqtgraph()
    assert str(caught.value) == fp.PYQTGRAPH_MISSING_MESSAGE


def test_a_violin_of_one_repeated_value_is_refused_rather_than_drawn_flat():
    """A violin from a column with no spread would claim a spread it lacks.

    Every replicate reading the same number is a real case -- a saturated
    channel, a control column of zeros -- and the honest picture is no violin
    at all. A profile returned anyway would have a width taken from the bin
    count rather than from the data.
    """
    assert fp._violin_profile([4.0, 4.0, 4.0], 0.5) == (None, None)

    centres, widths = fp._violin_profile([1.0, 2.0, 3.0], 0.5)
    assert centres[0] == 1.0 and centres[-1] == 3.0
    assert widths[0] == 0.0 and widths[-1] == 0.0
    assert float(widths.max()) == pytest.approx(0.5)
    assert centres.size == widths.size == 8


# ------------------------------------------------------- the build without pg

def test_a_build_without_pyqtgraph_answers_every_question_it_is_asked(
        qtbot, monkeypatch):
    """A missing optional wheel must cost the user the picture, not the panel.

    Everything a host asks a plot -- can this axis be logged, can the lines be
    recoloured, how many did you restyle, what shape is the canvas -- is asked
    before anybody knows whether pyqtgraph is installed, so a widget that
    raised on the third question would take the whole results panel with it.
    """
    drawn = fp.FastPlot(title="fast", x_label="ex", y_label="why")
    qtbot.addWidget(drawn)
    assert len(drawn.axis_items()) == 4

    monkeypatch.setattr(fp, "HAVE_PYQTGRAPH", False)
    bare = fp.FastPlot(title="fast", x_label="ex", y_label="why")
    qtbot.addWidget(bare)
    missing = "this build has no pyqtgraph, so nothing is drawn"
    assert bare.plots_available is False
    assert (bare.line_colour_reason(), bare.log_reason("y"),
            bare.y_split_reason(1.0, 2.0)) == (missing, missing, missing)
    assert bare.axis_items() == []
    assert bare.set_line_style(colour="#ff0000") == 0
    assert bare.set_log_axes(y=True) == (False, False)

    bare.offer_levels([("gene", lambda: None, True)], note="genes only")
    assert bare.level_note() == "genes only" and bare._level_box is None
    bare.set_canvas_shape("square")
    assert (bare.canvas_shape(), bare.canvas_ratio()) == ("square", 1.0)
    bare.auto_range_axes()
    assert bare.pinned_limits() == {"x": None, "y": None}


def test_a_level_the_control_does_not_offer_is_never_run(plot):
    """A stale index must not fire somebody else's callback.

    The level box is refilled whenever the run changes, so a click arriving
    against the old list would re-fit at a level this run does not have and
    redraw the figure from rows that are not there.
    """
    picked = []
    plot.offer_levels([("gene", lambda: picked.append("gene"), True),
                       ("guide", lambda: picked.append("guide"), False)],
                      note="gene-level drawn; guide-level not")
    plot._on_level_chosen(1)
    assert picked == ["guide"]
    plot._on_level_chosen(7)
    assert picked == ["guide"]
    assert plot._level_box.count() == 2
    assert plot._level_box.currentIndex() == 0
    assert "gene-level drawn" in plot._status.text()


# -------------------------------------------------------------- the smoothers

def test_a_smoother_answers_in_a_sentence_whether_it_fits_or_fails(
        plot, monkeypatch):
    """A diagnostic curve is an extra; neither outcome may cost the figure.

    The points are already on screen when the smoother is asked for, so an
    exception out of the fitter would take away a plot the user is reading.
    And a curve that fits looks like a finding, so the sentence has to carry
    "it decides no hit" whether or not the method has a note to add.
    """
    import spacr.nonparametric_fits as fits
    from spacr.nonparametric_fits import Curve

    def _refuse(x, y, method="lowess", **_k):
        raise ValueError("a gaussian process needs at most 2,000 rows")

    monkeypatch.setattr(fits, "smooth", _refuse)
    assert plot.add_smoother([1.0, 2.0], [1.0, 2.0], method="gp") == (
        "a gaussian process needs at most 2,000 rows")

    def _explode(x, y, method="lowess", **_k):
        raise MemoryError("no room for the kernel")

    monkeypatch.setattr(fits, "smooth", _explode)
    assert plot.add_smoother([1.0, 2.0], [1.0, 2.0], method="knn") == (
        "knn could not be drawn: no room for the kernel")

    made = Curve(method="lowess", x=np.array([0.0, 1.0]),
                 y=np.array([0.0, 1.0]), note="")
    monkeypatch.setattr(fits, "smooth", lambda *_a, **_k: made)
    assert plot.add_smoother([0.0, 1.0], [0.0, 1.0]) == (
        "lowess curve laid over the points. It is a diagnostic: it decides "
        "no hit.")
    made.note = "x standardised"
    assert plot.add_smoother([0.0, 1.0], [0.0, 1.0]) == (
        "lowess curve laid over the points (x standardised). It is a "
        "diagnostic: it decides no hit.")


# ------------------------------------------------- the registry of what moves

def test_the_log_scale_moves_the_curves_it_can_see_and_drops_the_rest(points):
    """A stale record makes the scale reach through a dangling reference.

    An empty curve has no coordinates to transform and an item the plot has
    since removed is not on screen any more, so neither belongs in the
    registry -- while a curve that IS there must be handed new arrays, or the
    trend line would stay linear beside dots that moved to logarithms.
    """
    gone = points.plot.plot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    curve = points.plot.plot(np.array([1.0, 2.0, 3.0]),
                             np.array([10.0, 100.0, 1000.0]))
    points.plot.addItem(pg.PlotDataItem())
    assert len(points._drawn) == 3

    points.plot.removeItem(gone)
    assert points.set_log_axes(y=True) == (False, True)
    assert [entry.kind for entry in points._drawn] == ["points", "line"]
    assert np.allclose(curve.getData()[1], [1.0, 2.0, 3.0])
    assert np.allclose(points._drawn[-1].y, [10.0, 100.0, 1000.0])


def test_bars_measured_from_a_positive_baseline_move_onto_the_log_scale(plot):
    """A bar the scale leaves behind is a bar drawn at the wrong height.

    Bars are placed by their four edges rather than by a centre, so the bottom
    and the top both have to be transformed. If only the top moved, a
    ranked-bar figure on a log axis would show bars whose lengths mean
    nothing, which is worse than refusing the scale outright.
    """
    bar = pg.BarGraphItem(x=[1.0, 2.0], height=[3.0, 4.0], width=0.5,
                          y0=[1.0, 1.0])
    plot.plot.addItem(bar)
    assert plot.set_log_axes(y=True) == (False, True)
    assert np.allclose(bar.opts["y0"], np.log10([1.0, 1.0]))
    assert np.allclose(bar.opts["y1"], np.log10([4.0, 5.0]))
    assert np.allclose(bar.opts["x0"], [0.75, 1.75])
    assert np.allclose(bar.opts["x1"], [1.25, 2.25])


def test_a_record_with_no_coordinates_leaves_its_item_exactly_as_it_was(plot):
    """The placer must be a no-op for anything it cannot honestly move.

    It runs over every registered item on every scale change, and the registry
    can hold a record with nothing to place -- a bar whose edges could not be
    re-measured, an item whose setter is an attribute rather than a method.
    Guessing a coordinate would put a mark at a number nobody computed.
    """
    line = pg.InfiniteLine(pos=2.0, angle=90)
    plot.plot.addItem(line)
    plot._place(plot._drawn[-1]._replace(x=np.array([7.0])))
    assert line.value() == 7.0
    plot._place(plot._drawn[-1]._replace(x=None, y=None))
    assert line.value() == 7.0

    dots = pg.ScatterPlotItem(x=[1.0, 2.0], y=[3.0, 4.0])
    plot.plot.addItem(dots)
    plot._place(plot._drawn[-1]._replace(x=None, y=None))
    assert list(dots.data["x"]) == [1.0, 2.0]
    assert list(dots.data["y"]) == [3.0, 4.0]
    assert dots.bounds == [None, None]

    class _Recorder:
        def __init__(self):
            self.calls = []

        def setData(self, **kwargs):
            self.calls.append(kwargs)

    class _Frozen:
        setData = "an attribute, not a method"

    counts = plot._counts_of(None, None)
    recorder, frozen = _Recorder(), _Frozen()
    plot._place(fp._Drawn(recorder, np.array([1.0]), np.array([2.0]), {},
                          "line", counts))
    assert [sorted(call) for call in recorder.calls] == [["x", "y"]]
    plot._place(fp._Drawn(frozen, np.array([1.0]), np.array([2.0]), {},
                          "line", counts))
    assert frozen.setData == "an attribute, not a method"


# ----------------------------------------------------------- the y-axis split

def test_a_split_on_a_logged_axis_hides_decades_and_says_so_on_the_axis(qtbot):
    """The break is defined against what is DRAWN, so a logged split hides decades.

    The hidden band is logged before it is compared with the marks, and the
    gap is sized from the data that is KEPT -- sizing it from a band forty
    decades tall would draw a break thirteen times taller than the figure. A
    column of all-missing values contributes nothing to that measurement.
    """
    widget = fp.FastPlot(title="fast", x_label="ex", y_label="why")
    qtbot.addWidget(widget)
    widget.add_scatter(np.array([1.0, 2.0, 3.0, 4.0]),
                       np.array([1.0, 2.0, 1e5, 2e5]), size=6)
    widget.add_scatter(np.array([5.0, 6.0]), np.array([np.nan, np.nan]),
                       size=6)
    assert widget.set_log_axes(y=True) == (False, True)
    assert widget.set_y_split(10.0, 1e4) == ""
    assert widget.y_split() == (10.0, 10000.0)
    assert widget._split_drawn() == (1.0, 4.0)
    kept = np.log10(2e5) - np.log10(1e5) + np.log10(2.0) - np.log10(1.0)
    assert widget._split_gap == pytest.approx(kept * 0.06)
    assert widget.plot.getAxis("left").labelText == (
        "why (log scale, split, 10-10000 not drawn)")


def test_the_break_is_sized_from_what_is_kept_below_it(plot):
    """A split above everything on the plot must still leave the marks alone.

    The gap is a fraction of the span that survives, and here every survivor
    is on one side of the band. A point below the band keeps its own
    coordinate exactly; if the compression reached it, every mark would move
    when the user hid an empty strip above them.
    """
    plot.add_scatter(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]),
                     size=6)
    assert plot.set_y_split(10.0, 100.0) == ""
    assert plot._split_gap == pytest.approx((3.0 - 1.0) * 0.06)
    assert float(plot._compress(np.array([3.0]))[0]) == 3.0
    assert "10 to 100 is not drawn" in plot._status.text()


def test_each_refused_split_names_its_own_reason(points):
    """A refusal the user cannot act on is the same as a control that is dead.

    Three different mistakes reach this: a bound that is not a number, a band
    that is upside down, and a non-positive bound on a logged axis. One
    generic "cannot split there" would leave the user changing the wrong
    number, so each answer names the thing to fix.
    """
    assert points.y_split_reason(float("nan"), 1.0) == (
        "a split needs two finite numbers")
    assert points.y_split_reason(5.0, 5.0) == (
        "the top of the hidden band has to be above its bottom")
    assert points.set_log_axes(y=True) == (False, True)
    assert points.y_split_reason(-1.0, 5.0) == (
        "a logged axis has no coordinate at or below zero")
    assert points.set_y_split(float("inf"), 1.0) == (
        "a split needs two finite numbers")
    assert points.y_split() is None
    assert "y split: a split needs two finite numbers" in points._status.text()


def test_the_tick_functions_are_installed_once_and_read_data_units_after(
        points):
    """The axis has to number the data whatever the ruler underneath it is.

    A split makes the left axis piecewise linear, so its numbers are computed
    rather than read off. A second layer of that would compress the coordinate
    twice and print numbers that are not on the figure; and a tick function
    still held after the split is removed must fall back to the axis' own.
    """
    assert points.set_y_split(10.0, 100.0) == ""
    axis = points.plot.getAxis("left")
    installed = points._axis_ticks
    wrapped_values, wrapped_strings = axis.tickValues, axis.tickStrings

    assert points.set_y_split(20.0, 90.0) == ""
    assert points._axis_ticks is installed
    assert axis.tickValues is wrapped_values
    hidden = float(points._compress(np.array([200.0]))[0])
    assert axis.tickStrings([hidden], 1.0, 1.0) == ["200"]

    points.clear_y_split()
    assert axis.tickValues is not wrapped_values
    assert wrapped_values(0.0, 300.0, 400) == axis.tickValues(0.0, 300.0, 400)
    assert (wrapped_strings([0.0, 100.0], 1.0, 1.0)
            == axis.tickStrings([0.0, 100.0], 1.0, 1.0))


def test_a_split_with_no_left_axis_is_still_undone(points, monkeypatch):
    """The way out of a split cannot depend on the axis being reachable.

    A subclass that removed the left axis -- or an older pyqtgraph that raises
    for it -- leaves the split applied with no tick functions to put back.
    Refusing to clear it there would strand the figure on a broken ruler with
    no control able to straighten it.
    """
    def _absent(*_args, **_kwargs):
        raise RuntimeError("no such axis")

    monkeypatch.setattr(points.plot, "getAxis", _absent, raising=False)
    assert points.set_y_split(10.0, 100.0) == ""
    assert points._axis_ticks is None

    points.clear_y_split()
    assert points.y_split() is None and points._split_gap == 0.0
    assert "y axis split removed." in points._status.text()
    points.set_status("still here")
    points.clear_y_split()
    assert points._status.text() == "still here   y axis split removed."


# ---------------------------------------------------------- axes, text, shape

def test_an_axis_nobody_named_grows_no_label_when_the_scale_changes(qtbot):
    """Writing an empty label onto a bare axis grows a blank strip under it.

    The control panel's x-axis is deliberately unlabelled because its ticks
    already name the groups, and every scale change relabels both axes while
    ``setLabel`` shows the label as a side effect. The axis that asked for
    nothing is skipped; a named one picks up the note saying how it is drawn.
    """
    bare = fp.FastPlot()
    named = fp.FastPlot(x_label="effect size")
    qtbot.addWidget(bare)
    qtbot.addWidget(named)
    assert bare.set_log_axes(x=False, y=False) == (False, False)
    assert bare.plot.getAxis("bottom").labelText == ""
    assert named.set_log_axes(x=True) == (True, False)
    assert named.plot.getAxis("bottom").labelText == "effect size (log scale)"


def test_the_text_controls_reach_everything_and_survive_a_scale_change(points):
    """A font control that half applies reads as a font control that is broken.

    Relabelling passes the style along with the text, so a size chosen from
    the menu is dropped unless it is written back -- the report reads "font
    size keeps resetting itself". And "all font" includes the legend, so one
    entry the sweep cannot read must not cost the rest of it their colour.
    """
    class _LegendText:
        def __init__(self):
            self.text, self.colour = "sgRNA", None

        def setText(self, text, color=None):
            self.text, self.colour = text, color

    label = _LegendText()
    points.plot.plotItem.legend = SimpleNamespace(
        items=[object(), ("sample", label)])
    points.set_font_size(17)
    points.set_font_colour("#ff0000")
    assert points.set_log_axes(y=True) == (False, True)

    left = points.plot.getAxis("left")
    assert left.labelStyle["font-size"] == "17pt"
    assert left.labelText == "why (log scale)"
    assert (points.font_size(), points.font_colour()) == (17, "#ff0000")
    assert label.colour == "#ff0000"
    assert left.textPen().color().name() == "#ff0000"


def test_a_typed_limit_below_zero_is_forgotten_when_the_axis_is_logged(points):
    """A bound with no logarithm can be neither kept nor quietly moved.

    Limits are remembered in data units and re-imposed whenever the ruler
    changes. One that cannot survive the new ruler is dropped and the axis
    goes back to its data; substituting a bound the user did not type would
    show a range nobody chose, while the legal axis keeps its window.
    """
    points.set_axis_limits(x=(0.0, 5.0), y=(-5.0, 400.0))
    assert points.pinned_limits() == {"x": (0.0, 5.0), "y": (-5.0, 400.0)}
    assert points.set_log_axes(y=True) == (False, True)
    assert points.pinned_limits() == {"x": (0.0, 5.0), "y": None}


def test_a_shaped_canvas_is_re_imposed_when_the_window_changes(plot):
    """A square figure that stops being square when a panel is dragged is wrong.

    The shape is what the exported page will be, so the canvas has to hold it
    while the user resizes the window around it. "Free" is the other half of
    the same control: once released, the plot takes the whole box again
    rather than staying frozen at the last square.
    """
    plot.resize(400, 320)
    plot.set_canvas_shape("square")
    assert plot.plot.height() == plot.plot.maximumWidth()

    QApplication.sendEvent(plot, QResizeEvent(QSize(520, 460),
                                              QSize(400, 320)))
    assert plot.canvas_shape() == "square"
    assert plot.plot.height() == plot.plot.maximumWidth()
    assert plot.plot.height() <= 460

    plot.set_canvas_shape("free")
    assert plot.canvas_ratio() is None
    assert plot.plot.maximumWidth() == fp.QWIDGET_SIZE_MAX
    assert plot.plot.maximumHeight() == fp.QWIDGET_SIZE_MAX

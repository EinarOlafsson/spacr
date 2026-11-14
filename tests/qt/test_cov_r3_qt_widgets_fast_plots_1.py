"""What the fast plots do when the data, the build or the box lets them down.

Every path here is a refusal a user can walk into without doing anything
wrong: a panel dragged shut to nothing, a feature column that is all blanks,
a radar whose spokes drop below three, a build whose figure-style module was
stripped out of the wheel, a colour scale pyqtgraph will not hand over. The
plot's contract in all of them is the same -- keep drawing what CAN be drawn
and say honestly how much that was -- so each test below breaks the one thing
the guard names and then asks the widget a question a user could ask: how
many marks went on, what is in the ring, what colour is the page, what does
the menu offer.
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

import pyqtgraph as pg                                  # noqa: E402
from PySide6.QtCore import QPoint, QRectF               # noqa: E402
from PySide6.QtGui import QColor                        # noqa: E402

from spacr.qt.widgets import fast_plots as fp           # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture
def plot(qtbot):
    """A live FastPlot with both axes named, nothing drawn on it yet."""
    widget = fp.FastPlot(title="fast", x_label="ex", y_label="why")
    qtbot.addWidget(widget)
    return widget


def _style_stub(monkeypatch, **names):
    """Put a partial ``spacr.figure_style`` in front of the real one.

    A module object carrying only the names given is what a stripped wheel or
    a half-finished upgrade looks like from an import statement's point of
    view, which is the shape both guards below were written against.
    """
    stub = types.ModuleType("spacr.figure_style")
    for name, value in names.items():
        setattr(stub, name, value)
    monkeypatch.setitem(sys.modules, "spacr.figure_style", stub)
    return stub


# ------------------------------------------------------------------ the page

def test_a_canvas_squeezed_to_nothing_is_not_stretched_for_the_page(plot):
    """A pane dragged shut must not be resized into a division by zero.

    Export holds the plot item at the page's proportions for one render. If
    the panel has been collapsed the item is zero pixels wide, and scaling
    that by the shape's ratio would either keep it invisible or leave the
    exporter measuring a degenerate rectangle. The hold declines instead, and
    -- this is the part a user sees -- puts the geometry back either way, so a
    cancelled export does not leave the figure a different size than it found.
    """
    plot.set_canvas_shape("square")
    item = plot.plot.plotItem
    item.setMinimumSize(0, 0)
    item.setGeometry(QRectF(0.0, 0.0, 0.0, 0.0))

    with plot._held_at_the_page_shape() as held:
        assert held is False
        assert item.geometry().width() == 0.0
    assert item.geometry().width() == 0.0

    item.setGeometry(QRectF(0.0, 0.0, 400.0, 100.0))
    with plot._held_at_the_page_shape() as held:
        assert held is True
        assert item.geometry().height() == pytest.approx(400.0)
    assert item.geometry().height() == pytest.approx(100.0)


def test_the_right_click_menu_opens_where_the_click_landed(plot, monkeypatch):
    """A context menu that ignores the cursor appears somewhere else entirely.

    The style menu is the only way to reach most of a plot's controls, and it
    is opened by right-clicking the plot. The click arrives in the plot's own
    coordinates and a menu is placed in screen coordinates, so the handler has
    to map between them; skip that and the menu opens at the top-left of the
    desktop while the user is looking at the bottom of a panel.
    """
    shown = []

    class _Menu:
        def exec(self, where):
            shown.append(where)

    monkeypatch.setattr(plot, "build_style_menu", _Menu)
    point = QPoint(37, 11)
    plot._style_menu(point)

    assert shown == [plot.plot.mapToGlobal(point)]


# -------------------------------------------------------------- the menu body

def test_the_diagnostic_curves_get_their_own_heading_only_when_offered(plot):
    """A smoother listed under the fit reads as a choice of fit, and is not.

    A lowess line drawn over a regression decides no hit; the fit underneath
    it does. The menu therefore puts the diagnostics behind a heading that
    says so, and a plot whose host cannot redraw them must not show the
    heading at all -- an entry that calls nothing is worse than no entry.
    """
    def _headings(widget):
        return [action.text()
                for action in widget.build_style_menu().actions()]

    assert not any("Diagnostic curve" in text for text in _headings(plot))

    picked = []
    plot.offer_smoothers(picked.append, chosen="lowess")
    heading = [text for text in _headings(plot) if "Diagnostic curve" in text]
    assert heading == ["Diagnostic curve (decides no hit)"]

    entries = plot._smoother_options()
    assert entries[0][0] == "None"
    assert any(chosen for _label, _call, chosen in entries[1:]), (
        "the smoother already drawn should be ticked")
    entries[0][1]()
    assert picked == [""]


def test_a_build_without_the_localisation_list_offers_no_all_entry(monkeypatch):
    """The "colour every compartment" entry needs the list it names.

    The sentinel that means "all of them" lives in the localisation module. A
    build shipped without it -- a trimmed wheel, a partial upgrade -- must
    fall back to naming compartments one at a time rather than raising while
    a right-click menu is being assembled, which would take the whole menu.
    """
    from spacr.localisation import ALL

    assert fp.FastPlot._all_compartments() == ALL

    monkeypatch.setitem(sys.modules, "spacr.localisation",
                        types.ModuleType("spacr.localisation"))
    assert fp.FastPlot._all_compartments() is None


def test_a_fade_channel_with_no_levels_leaves_every_point_as_it_was(plot):
    """An empty third encoding must not make the points transparent.

    Opacity is the third categorical channel, and a fully transparent point is
    a deleted point. When the column driving it turns out to have no levels --
    an empty frame, a filter that matched nothing -- the ramp is empty and the
    brushes have to come back untouched rather than being re-alphaed against
    an empty scale.
    """
    assert fp.FastPlot._opacity_alphas(0) == []
    assert fp.FastPlot._opacity_alphas(1) == [70]
    ramp = fp.FastPlot._opacity_alphas(3)
    assert ramp[0] == 70 and ramp[-1] == 255 and len(ramp) == 3

    brushes = [pg.mkBrush(QColor("#112233")), pg.mkBrush(QColor("#445566"))]
    assert fp.FastPlot._categorical_opacity(brushes, [], 2) is brushes

    faded = fp.FastPlot._categorical_opacity(brushes, ["a", "b"], 2)
    assert [b.color().alpha() for b in faded] == [70, 255]


# ------------------------------------------------------------- the saved look

def test_an_unreadable_saved_look_leaves_the_page_transparent(plot,
                                                              monkeypatch):
    """A broken preference must not stamp a colour onto everybody's export.

    The page colour comes from the saved figure appearance. If reading it
    raises -- a corrupt settings file, a module that is not there -- the only
    safe answer is a transparent page: guessing white would put a white
    rectangle behind a figure destined for a dark slide, and the user has no
    way to tell it was guessed.
    """
    _style_stub(monkeypatch,
                saved_figure_appearance=lambda: types.SimpleNamespace(
                    ground="#ff0000"))
    assert plot._export_ground().name().lower() == "#ff0000"

    def _boom():
        raise RuntimeError("the preference store is unreadable")

    _style_stub(monkeypatch, saved_figure_appearance=_boom)
    ground = plot._export_ground()
    assert ground.alpha() == 0


# ------------------------------------------------------------ rings and picks

def test_a_ring_is_refused_for_a_row_that_was_never_plotted(plot):
    """Selecting an unplottable gene must say so, not ring its neighbour.

    A row with an unusable p-value is dropped before drawing but keeps its
    identifier, so a table selection can name it. Ringing the nearest drawn
    point instead would silently tell the user the wrong gene is the one they
    clicked, and rings are how the linked views agree with each other.
    """
    plot.add_scatter(np.array([1.0, 2.0, 3.0]),
                     np.array([1.0, np.nan, 3.0]))
    plot.set_keys(["a", "b", "c"])

    assert plot.highlight_key("a") is True
    assert plot.highlight_key("b") is False

    assert plot.highlight_keys(["b", "c"]) == 1
    assert plot.selected_keys() == ["b", "c"], (
        "an unplottable key stays selected even without a ring")
    assert plot.highlight_keys(["a", "c"]) == 2


def test_a_rectangle_selection_steps_over_a_position_it_cannot_read(plot):
    """One unreadable position must not cost the user the whole drag.

    Rubber-band selection walks every drawn position. A plot type that filled
    the map with something that is not a pair of numbers would, unguarded,
    raise inside the drag and select nothing -- so the row is skipped and the
    points that ARE readable are still picked.
    """
    plot.add_scatter(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    plot.set_keys(["a", "b"])
    plot._row_xy[7] = ("not", "a number")

    assert plot.select_in_rect(0.0, 0.0, 5.0, 5.0) == ["a", "b"]
    assert plot.select_in_rect(0.0, 0.0, 1.5, 1.5) == ["a"]


# ------------------------------------------------------------- the chart types

def test_a_ranked_bar_chart_of_blanks_draws_no_bars(plot):
    """A chart of nothing must report nothing, not an empty axis with names.

    Feature importances arrive as a column that can be entirely missing when
    a fit did not converge. Drawing twenty labelled rows with no bars looks
    like twenty features of importance zero, which is a claim about the model
    rather than an admission that nothing was measured.
    """
    assert plot.add_ranked_bars(["a", "b"], [np.nan, np.nan]) == 0
    assert plot.add_ranked_bars(["a", "b"], [1.0, 2.0]) == 2


def test_a_beeswarm_survives_a_blank_feature_and_a_scale_that_will_not_load(
        plot, monkeypatch):
    """A SHAP summary must still draw when the colour scale is unavailable.

    Colour carries the feature's own value, but it is the second reading of
    the chart; the spread of the contributions is the first. So a pyqtgraph
    that cannot hand over the lookup table falls back to one flat colour
    rather than dropping the chart, and a feature whose contributions are all
    missing takes no row instead of an empty one that reads as "no effect".
    """
    def _no_colormap(_name):
        raise RuntimeError("this build has no colour maps")

    monkeypatch.setattr(pg.colormap, "get", _no_colormap)

    matrix = np.array([[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]])
    features = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

    assert plot.add_beeswarm(["kept", "blank"], matrix, features) == 3
    scatters = [item for item in plot.plot.plotItem.items
                if isinstance(item, pg.ScatterPlotItem)]
    assert len(scatters) == 1, "only the feature with numbers takes a row"


def test_a_shade_ramp_marks_what_it_has_no_number_for(plot):
    """A missing feature value must look missing, not like the scale's floor.

    Every point in a beeswarm is coloured by its own feature value. A blank
    coloured with the bottom of the ramp is indistinguishable from a genuine
    smallest value, so blanks take the muted colour instead -- and an entire
    column of blanks takes it for every point rather than dividing by a span
    of nothing.
    """
    lookup = pg.colormap.get("viridis").getLookupTable(
        nPts=fp.COLORMAP_STEPS, alpha=True)
    missing = QColor(fp.MISSING_COLOUR).name()

    blank = fp.FastPlot._shade([np.nan, np.nan], lookup)
    assert [colour.name() for colour in blank] == [missing, missing]

    mixed = fp.FastPlot._shade([0.0, np.nan, 1.0], lookup)
    assert mixed[1].name() == missing
    assert mixed[0].name() != missing and mixed[2].name() != missing
    assert mixed[0].name() != mixed[2].name()


def test_a_radar_drops_blank_spokes_and_refuses_what_is_left(plot):
    """Fewer than three spokes is a line, not a radar, and must be refused.

    A radar is read as a shape, and two spokes make no shape at all. Blanks
    are dropped rather than plotted at zero -- a zero radius is a measured
    value and would pull the polygon into the middle -- so a chart can fall
    below three while drawing, and at that point it has to decline rather than
    draw a sliver the reader would interpret.
    """
    assert plot.radar_frame() is None
    assert plot.add_radar(["a", "b", "c"], [1.0, np.nan, 3.0]) == 0
    assert plot.radar_frame() is None

    assert plot.add_radar(["a", "b", "c", "d"],
                          [1.0, np.nan, 3.0, 4.0]) == 3
    frame = plot.radar_frame()
    assert list(frame["name"]) == ["a", "c", "d"]
    assert list(frame["value"]) == [1.0, 3.0, 4.0]



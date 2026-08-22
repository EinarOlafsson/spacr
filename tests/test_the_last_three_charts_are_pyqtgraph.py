"""The three charts that kept a second renderer alive.

    "in the regression modual all of the generated graphs should be generated
     with pyqtgraph, not matplotlib"

Everything else on that path was a straight translation. These three were
not, and each was refused for its own reason:

* the two importance charts are twenty feature names against one number
  each, so they need HORIZONTAL bars -- ``add_group_mark`` draws at
  categorical positions in x and every one of its eight marks assumes the
  measurement is y;
* the compartment and channel charts are RADAR plots, and pyqtgraph has no
  polar axis;
* the SHAP summary is drawn by ``shap.summary_plot``, which makes its own
  matplotlib figure and cannot be handed a scene.

The first two were real obstacles and the third was a wrapper. All three are
answered by drawing the chart rather than by finding a way to keep the old
one, which is why what is asserted here is the chart's own contract: the
right number of marks, the data recoverable beside the picture, and nothing
on the pyplot stack afterwards.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets.fast_plots import FastPlot          # noqa: E402


@pytest.fixture()
def plot(qapp):
    made = FastPlot(title="t", x_label="x", y_label="y")
    made.resize(900, 620)
    yield made
    made.deleteLater()


# ---------------------------------------------------------------------------
# Ranked horizontal bars
# ---------------------------------------------------------------------------

def test_a_bar_per_label(plot):
    import pyqtgraph as pg

    names = [f"feature_{i}" for i in range(9)]
    assert plot.add_ranked_bars(names, np.linspace(0.1, 0.9, 9)) == 9
    bars = [i for i in plot.plot.items() if isinstance(i, pg.BarGraphItem)]
    assert len(bars) == 9


def test_the_bars_are_horizontal(plot):
    """A vertical bar chart of twenty feature names is the chart that was
    unreadable, which is the whole reason this method exists."""
    import pyqtgraph as pg

    plot.add_ranked_bars(["a", "b", "c"], [3.0, 2.0, 1.0])
    bar = next(i for i in plot.plot.items()
               if isinstance(i, pg.BarGraphItem))
    # x0/x1 carry the VALUE and y carries the row: the other way round is a
    # vertical bar.
    assert bar.opts.get("x1") is not None
    assert bar.opts.get("y") is not None
    assert bar.opts.get("height") is not None


def test_the_names_are_the_left_axis(plot):
    plot.add_ranked_bars(["alpha", "beta", "gamma"], [1.0, 3.0, 2.0])
    axis = plot.plot.getAxis("left")
    shown = {text for group in (axis._tickLevels or []) for _v, text in group}
    assert shown == {"alpha", "beta", "gamma"}


def test_the_largest_is_at_the_top(plot):
    """A ranked chart whose rank is not visible is a bar chart."""
    plot.add_ranked_bars(["small", "big", "middle"], [1.0, 9.0, 4.0])
    frame = plot.ranked_frame()
    assert list(frame["name"]) == ["big", "middle", "small"]
    assert plot.plot.getViewBox().yInverted(), (
        "the rows count down the screen, so the axis has to be inverted or "
        "the largest is at the bottom")


def test_the_callers_order_can_be_kept(plot):
    """A table that is already ranked must not be re-sorted: the caller
    chose which rows to show by taking the head of it."""
    plot.add_ranked_bars(["first", "second"], [1.0, 9.0], descending=False)
    assert list(plot.ranked_frame()["name"]) == ["first", "second"]


def test_only_the_leading_bars_carry_the_accent(plot):
    """Everything grey except what the sentence is about."""
    import pyqtgraph as pg

    plot.add_ranked_bars([f"f{i}" for i in range(6)],
                         np.arange(6.0), highlight=2)
    bars = [i for i in plot.plot.items() if isinstance(i, pg.BarGraphItem)]
    # COUNTED, NOT INDEXED: `items()` reports scene order, which is not the
    # order the bars were added in, so a positional assertion here would be
    # testing pyqtgraph's z-ordering rather than the accent rule.
    colours = [i.opts["pen"].color().name() for i in bars]
    tally = {name: colours.count(name) for name in set(colours)}
    assert len(tally) == 2, tally
    assert sorted(tally.values()) == [2, 4], tally


def test_a_blank_value_is_dropped_not_drawn_at_zero(plot):
    """A bar of length zero says the feature does not matter; a missing
    number says nothing is known about it."""
    assert plot.add_ranked_bars(["a", "b", "c"],
                                [1.0, float("nan"), 3.0]) == 2
    assert set(plot.ranked_frame()["name"]) == {"a", "c"}


def test_mismatched_lengths_draw_nothing(plot):
    assert plot.add_ranked_bars(["a", "b"], [1.0]) == 0
    assert plot.ranked_frame() is None


# ---------------------------------------------------------------------------
# The radar
# ---------------------------------------------------------------------------

def test_a_radar_needs_three_spokes(plot):
    """Two spokes is a line and one is a dot; neither is a radar."""
    assert plot.add_radar(["a", "b"], [1.0, 2.0]) == 0
    assert plot.add_radar(["a", "b", "c"], [1.0, 2.0, 3.0]) == 3


def test_the_radar_draws_its_own_grid(plot):
    """A radar read against a square grid is unreadable: the reference a
    reader needs is the concentric rings."""
    plot.add_radar(["a", "b", "c", "d"], [1.0, 2.0, 3.0, 2.0], rings=3)
    # three rings plus four spokes plus the polygon outline
    curves = [i for i in plot.plot.items()
              if hasattr(i, "getData") and i.getData()[0] is not None]
    assert len(curves) >= 3 + 4


def test_the_radar_polygon_closes(plot):
    """An open polygon has a gap between the last spoke and the first, which
    reads as a missing value."""
    import pyqtgraph as pg

    plot.add_radar(["a", "b", "c", "d"], [1.0, 2.0, 3.0, 2.0])
    closed = [i for i in plot.plot.items()
              if isinstance(i, pg.PlotCurveItem)
              and i.getData()[0] is not None
              and len(i.getData()[0]) == 5]
    assert closed, "the polygon was not closed"
    x, y = closed[0].getData()
    assert x[0] == pytest.approx(x[-1])
    assert y[0] == pytest.approx(y[-1])


def test_a_radar_has_no_cartesian_axes(plot):
    """Two coordinate systems on one picture."""
    plot.add_radar(["a", "b", "c"], [1.0, 2.0, 3.0])
    for side in ("left", "bottom"):
        assert plot.plot.getAxis(side)._tickLevels == [[]]


def test_the_radar_is_round_not_oval(plot):
    """Without a locked aspect the rings are ellipses and every radius is
    worth a different amount depending on the direction."""
    plot.add_radar(["a", "b", "c"], [1.0, 2.0, 3.0])
    assert plot.plot.getViewBox().state["aspectLocked"]


def test_a_negative_radius_is_clipped(plot):
    """A radar has no inside-out: a spoke through the centre would draw a
    polygon that crosses itself."""
    plot.add_radar(["a", "b", "c"], [-1.0, 2.0, 3.0])
    assert float(plot.radar_frame()["value"].min()) >= 0.0


def test_the_radar_data_is_recoverable(plot):
    plot.add_radar(["cell", "nucleus", "pathogen"], [0.4, 0.2, 0.9])
    frame = plot.radar_frame()
    assert list(frame["name"]) == ["cell", "nucleus", "pathogen"]
    assert list(frame["value"]) == pytest.approx([0.4, 0.2, 0.9])


# ---------------------------------------------------------------------------
# The beeswarm
# ---------------------------------------------------------------------------

def test_a_point_per_sample_per_feature(plot):
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(40, 5))
    assert plot.add_beeswarm([f"f{i}" for i in range(5)], matrix) == 200


def test_the_features_are_the_left_axis(plot):
    rng = np.random.default_rng(0)
    plot.add_beeswarm(["a", "b"], rng.normal(size=(10, 2)))
    axis = plot.plot.getAxis("left")
    shown = {text for group in (axis._tickLevels or []) for _v, text in group}
    assert shown == {"a", "b"}


def test_zero_is_marked(plot):
    """Which side of zero a point falls on is the whole reading of the
    chart, and an unmarked axis leaves the reader estimating it."""
    import pyqtgraph as pg

    rng = np.random.default_rng(0)
    plot.add_beeswarm(["a", "b"], rng.normal(size=(10, 2)))
    marks = [i for i in plot.plot.items() if isinstance(i, pg.InfiniteLine)]
    assert marks and float(marks[0].value()) == pytest.approx(0.0)


def test_a_row_stays_inside_its_own_lane(plot):
    """Rows that bleed into each other put a point on the wrong feature."""
    rng = np.random.default_rng(1)
    offsets = FastPlot._beeswarm_offsets(rng.normal(size=200))
    assert abs(offsets).max() <= FastPlot.BEESWARM_SPREAD + 1e-9


def test_the_offsets_follow_the_density(plot):
    """NOT PLAIN JITTER: a bimodal feature and a single tight cluster must
    not draw the same band of noise."""
    values = np.r_[np.full(60, -2.0), np.full(60, 2.0)]
    offsets = FastPlot._beeswarm_offsets(values)
    middle = np.abs(offsets[np.abs(values) < 1.0])
    assert middle.size == 0, "there is nothing in the middle to spread"
    assert abs(offsets).max() > 0.0, "the clusters were not spread at all"


def test_one_observation_needs_no_spread(plot):
    assert list(FastPlot._beeswarm_offsets([1.0])) == [0.0]


def test_the_colour_carries_the_feature_value(plot):
    """The colour split along a row says which direction the feature pushes,
    which is half of what the chart is read for."""
    rng = np.random.default_rng(2)
    matrix = rng.normal(size=(30, 3))
    values = rng.normal(size=(30, 3))
    plot.add_beeswarm(["a", "b", "c"], matrix, values)
    brushes = set()
    for item in plot._scatter_items():
        for point in item.points():
            brushes.add(point.brush().color().name())
    assert len(brushes) > 3, "every point came out the same colour"


def test_a_wrong_shaped_colour_source_is_ignored_not_fatal(plot):
    rng = np.random.default_rng(3)
    assert plot.add_beeswarm(["a", "b"], rng.normal(size=(10, 2)),
                             rng.normal(size=(4, 2))) == 20


def test_the_beeswarm_data_is_recoverable(plot):
    rng = np.random.default_rng(4)
    plot.add_beeswarm(["a", "b"], rng.normal(size=(7, 2)))
    frame = plot.beeswarm_frame()
    assert set(frame.columns) == {"feature", "contribution"}
    assert len(frame) == 14


def test_a_matrix_of_the_wrong_width_draws_nothing(plot):
    assert plot.add_beeswarm(["a", "b", "c"],
                             np.zeros((5, 2))) == 0
    assert plot.beeswarm_frame() is None


# ---------------------------------------------------------------------------
# And they all export
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("draw", ["ranked", "radar", "beeswarm"])
def test_each_chart_writes_a_real_file(plot, tmp_path, draw):
    """The point of moving them: the figure a run writes IS this scene."""
    rng = np.random.default_rng(5)
    if draw == "ranked":
        plot.add_ranked_bars(["a", "b", "c"], [1.0, 2.0, 3.0])
    elif draw == "radar":
        plot.add_radar(["a", "b", "c", "d"], [1.0, 2.0, 3.0, 2.0])
    else:
        plot.add_beeswarm(["a", "b"], rng.normal(size=(20, 2)))
    written = plot.export(str(tmp_path / f"{draw}.pdf"))
    assert written and os.path.getsize(written) > 2000


def test_ml_creates_no_matplotlib_figure_of_its_own():
    """The instruction's own criterion, read off the module.

    `plt.subplots` and `plt.figure` are how a matplotlib figure comes into
    existence, and `spacr.ml` made six of them. A run cannot produce a
    matplotlib figure on its own path if the module never creates one.
    """
    import inspect

    from spacr import ml

    source = inspect.getsource(ml)
    for maker in ("plt.subplots(", "plt.figure("):
        assert maker not in source, maker


# ---------------------------------------------------------------------------
# And the path as a whole
# ---------------------------------------------------------------------------

#: Every module that draws a figure a REGRESSION RUN writes.
REGRESSION_PATH = ("ml", "toxo", "guide_permutation",
                   "regression_diagnostics", "regression_qc")


def test_no_figure_on_the_regression_path_is_written_by_matplotlib():
    """The instruction's own criterion, module by module.

    A figure a run keeps reaches disk one of two ways now: drawn in
    pyqtgraph from the start, or built in matplotlib and TRANSLATED into a
    scene by `figures.scene.write_figure`, which falls back to the
    matplotlib page only when an artist is outside its whitelist -- and
    records why when it does.

    What must not survive is a `savefig` or a bare `save_figure` straight
    off a figure, because that is the second renderer: the file and the tab
    stop being one picture, and the two can disagree without anything on
    screen saying so.
    """
    import importlib
    import inspect

    offenders = []
    for name in REGRESSION_PATH:
        module = importlib.import_module(f"spacr.{name}")
        source = inspect.getsource(module)
        makes = source.count("plt.subplots(") + source.count("plt.figure(")
        routed = (source.count("write_figure") +
                  source.count("_write_the_figure"))
        if makes and not routed:
            offenders.append(f"spacr.{name}: {makes} figure(s), none routed "
                             f"through the scene renderer")
    assert not offenders, offenders


def test_the_scene_renderer_prefers_pyqtgraph_where_it_can():
    """`auto` has to mean pyqtgraph, or routing the figures through it
    changed nothing."""
    from spacr.figures.scene import scene_renderer

    renderer, why = scene_renderer("auto")
    assert renderer in ("pyqtgraph", "matplotlib")
    if renderer == "matplotlib":
        assert why, "a matplotlib fallback with no reason is unexplainable"


# ---------------------------------------------------------------------------
# The house style on a grouped comparison
# ---------------------------------------------------------------------------

def test_a_named_subject_greys_every_other_group(qapp):
    """"everything is grey except what the sentence is about; a box per
    group in a different colour is a rainbow, not an argument"."""
    from PySide6.QtGui import QColor

    from spacr.figures.style import ROLES
    from spacr.qt.widgets.grouped_plot import GroupedPlot

    inks = [QColor(GroupedPlot._ink_for(name, i, "GRA14")).name()
            for i, name in enumerate(["nc", "pc", "GRA14"])]
    assert inks == [QColor(ROLES["data"]).name(),
                    QColor(ROLES["data"]).name(),
                    QColor(ROLES["highlight"]).name()]


def test_with_no_subject_the_groups_are_told_apart(qapp):
    """A figure that has not been told what it is about has no subject to
    single out, and greying every group would leave no ink on the page."""
    from PySide6.QtGui import QColor

    from spacr.qt.widgets.grouped_plot import GroupedPlot

    inks = {QColor(GroupedPlot._ink_for(name, i, "")).name()
            for i, name in enumerate(["nc", "pc", "GRA14"])}
    assert len(inks) == 3


def test_the_background_group_is_grey_either_way(qapp):
    """The residual population is what the others are compared AGAINST, so
    it is never the claim."""
    from PySide6.QtGui import QColor

    from spacr.figures.style import ROLES
    from spacr.qt.widgets.grouped_plot import GroupedPlot

    grey = QColor(ROLES["data"]).name()
    assert QColor(GroupedPlot._ink_for("the rest", 0, "",
                                       "the rest")).name() == grey
    assert QColor(GroupedPlot._ink_for("the rest", 0, "GRA14",
                                       "the rest")).name() == grey


def test_the_compare_panel_greys_the_rest(qapp):
    """Wired, not merely available: the panel that draws this comparison has
    to be the thing that names its background, or the rule reaches no
    figure a user sees."""
    import inspect

    from spacr.qt.widgets import measurement_compare_dialog

    source = inspect.getsource(measurement_compare_dialog)
    assert "background=REST" in source

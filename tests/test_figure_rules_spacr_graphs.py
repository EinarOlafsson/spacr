"""spaCR's comparison figures follow the spacr-graphs rules (maintainer, 2026-11-19).

One test per rule that applies to spaCR's own figure layer
(:mod:`spacr.figures.style`) and to the comparison writers built on it
(``spacrGraph``, ``create_grouped_plot``): the plot area is n condition
slots wide, every bar or box is the same width, the control is grey, every
data element is at 60 % opacity, a group with more than 20 points is a box,
error bars and medians are black, no text is under 8 pt and the axes are a
full box with outward ticks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("matplotlib")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
from matplotlib.colors import to_hex, to_rgb  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from spacr.figures import style  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    """Close the pyplot figures the spaCR writers open, so none leak into
    later tests in the same worker (which count open figures)."""
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


def _frame(groups, per, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"g": np.repeat([f"c{i}" for i in range(groups)], per),
                         "v": rng.normal(size=groups * per)})


def _graph(groups, per, **kw):
    from spacr.plot import spacrGraph

    graph = spacrGraph(_frame(groups, per), "g", "v", **kw)
    graph.create_plot()
    return graph.fig, graph.fig.axes[0]


def _plot_width(figure, ax):
    return ax.get_position().width * figure.get_size_inches()[0]


@pytest.mark.parametrize("groups", [2, 3, 6])
def test_the_plot_area_is_n_slots_and_every_bar_one_width(groups):
    figure, ax = _graph(groups, 6)
    assert _plot_width(figure, ax) == pytest.approx(groups * style.SLOT_IN)
    widths = {round(p.get_width(), 9) for p in ax.patches}
    assert widths == {style.BAR_FRAC}


def test_graphs_with_the_same_number_of_groups_have_the_same_width():
    first, a = _graph(3, 6, compare_group="c0")
    second, b = _graph(3, 60)
    assert _plot_width(first, a) == pytest.approx(_plot_width(second, b))


def test_the_control_is_grey_and_the_rest_one_accent():
    _figure, ax = _graph(3, 6, compare_group="c1")
    colours = [to_hex(p.get_facecolor()[:3]).upper() for p in ax.patches]
    assert colours[1] == style.Palette.CONTROL
    assert colours[0] == colours[2] == style.Palette.ACCENT


def test_every_data_element_is_at_sixty_percent():
    _figure, ax = _graph(3, 6)
    artists = list(ax.patches) + list(ax.collections)
    assert artists and {round(a.get_alpha(), 6) for a in artists} == {0.6}
    assert all(t.get_alpha() in (None, 1.0) for t in ax.get_xticklabels())


def test_more_than_twenty_points_is_a_box_and_twenty_or_fewer_a_bar():
    figure = Figure()
    ax = figure.add_subplot(111)
    kinds = style.condition_plot(ax, [np.arange(20.0), np.arange(21.0)])
    assert kinds == ["bar", "box"]
    _fig, many = _graph(2, 40)
    assert not [p for p in many.patches if isinstance(p, matplotlib.patches.Rectangle)]


def test_medians_whiskers_and_error_bars_are_black():
    figure = Figure()
    ax = figure.add_subplot(111)
    style.condition_plot(ax, [np.arange(40.0)])
    line_colours = {to_hex(line.get_color()) for line in ax.lines}
    assert line_colours == {"#000000"}
    from spacr.plot import create_grouped_plot

    bars, _ = create_grouped_plot(_frame(3, 6), "g", "v", graph_type="bar")
    assert style.figure_problems(bars) == []


def test_no_text_is_under_eight_points_and_axes_are_boxed():
    rc = style.rc("print")
    assert min(style.TYPE_SCALE.values()) >= 8.0
    assert rc["axes.spines.top"] and rc["axes.spines.right"]
    assert rc["xtick.direction"] == rc["ytick.direction"] == "out"
    figure, _ax = _graph(4, 6)
    assert [p for p in style.figure_problems(figure) if "pt" in p] == []


def test_the_palette_has_no_brown_and_a_grey_control():
    for colour in style.SERIES:
        red, green, blue = to_rgb(colour)
        assert not (red > green > blue and red - blue > 0.3 and red < 0.8)
    assert style.condition_colours(3, control=0)[0] == style.Palette.CONTROL


def test_square_axes_and_the_centred_diverging_map():
    figure = Figure(figsize=(4, 3))
    ax = style.square_axes(figure.add_subplot(111))
    assert ax.get_box_aspect() == 1
    assert style.figure_problems(figure) == []
    norm = style.centred_norm(0.0, 0.33, 1.0)
    assert norm(0.33) == pytest.approx(0.5)
    assert all(channel > 0.99 for channel in style.diverging_cmap()(0.5)[:3])

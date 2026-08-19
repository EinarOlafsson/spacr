"""A grouped comparison draws a box with jitter, not a bar with jitter.

Instruction 139 B, asked for on 2026-08-18: "the bargraphs with jutter plot
backgrounds should be boxplots with jutter in the same style as the graph
skill in the repo that is based on sebastian louridos papers".

IT IS A STATISTICAL CORRECTION, NOT A PREFERENCE, which is why the default
moves rather than the option merely existing. A bar drawn at a mean with
points behind it shows ONE number and hides the shape: two groups with the
same mean and completely different spreads draw the same bar. A box shows the
median, the quartiles and the whiskers, so the reader sees the distribution
the points already imply.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from spacr.figures.style import ROLES  # noqa: E402
from spacr.plot import create_grouped_plot  # noqa: E402


@pytest.fixture
def same_mean_different_spread():
    """The figure that makes the case. Both groups have mean ~0; one has
    three times the spread. A bar chart draws them identically."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "grp": np.repeat(["tight", "wide"], 60),
        "val": np.concatenate([rng.normal(0, 1, 60), rng.normal(0, 3, 60)]),
    })


@pytest.fixture(autouse=True)
def _clean():
    plt.close("all")
    yield
    plt.close("all")


def _axis(df, **kwargs):
    create_grouped_plot(df, "grp", "val", **kwargs)
    return plt.gcf().axes[0]


def test_the_default_is_a_box_with_jitter():
    """Three modules defaulted to 'jitter', 'jitter_bar' and 'violin'. None
    of them showed the quartiles."""
    from spacr.settings import (get_analyze_recruitment_default_settings,
                                set_default_analyze_screen)

    for factory in (get_analyze_recruitment_default_settings,
                    set_default_analyze_screen):
        try:
            resolved = factory({})
        except Exception:
            continue
        if "graph_type" in resolved:
            assert resolved["graph_type"] == "jitter_box", factory.__name__


def test_every_observation_is_drawn(same_mean_different_spread):
    """The box summarises; the points are the evidence. Both are on the
    figure, so a reader can see the summary and check it."""
    ax = _axis(same_mean_different_spread, graph_type="jitter_box")
    drawn = sum(len(c.get_offsets()) for c in ax.collections)
    assert drawn == len(same_mean_different_spread)


def test_the_box_is_unfilled_so_the_points_carry_the_ink(
        same_mean_different_spread):
    """A filled box per group is a rainbow behind a dot strip, and the eye
    goes to the fill rather than to the observations."""
    ax = _axis(same_mean_different_spread, graph_type="jitter_box")
    assert ax.patches, "no box was drawn"
    for box in ax.patches:
        assert box.get_facecolor()[3] == 0.0, "the box is filled"


def test_the_points_are_the_house_grey(same_mean_different_spread):
    from matplotlib.colors import to_rgb

    ax = _axis(same_mean_different_spread, graph_type="jitter_box")
    expected = to_rgb(ROLES["data"])
    for collection in ax.collections:
        for rgba in collection.get_facecolors():
            assert tuple(round(v, 3) for v in rgba[:3]) == tuple(
                round(v, 3) for v in expected)


def test_the_extremes_are_not_drawn_twice(same_mean_different_spread):
    """`showfliers` is off, and that is not hiding them.

    The strip draws EVERY observation. seaborn's own flier markers would
    double-plot the extreme points and only those, which reads as the tails
    being twice as dense as they are.
    """
    ax = _axis(same_mean_different_spread, graph_type="jitter_box")
    markers = [line for line in ax.lines
               if line.get_marker() not in ("", "None", None)]
    assert not markers, "seaborn drew fliers on top of the strip"


def test_the_box_shows_a_spread_a_bar_cannot(same_mean_different_spread):
    """The whole argument, measured.

    Two groups with the same mean and different spreads: a bar chart draws
    two bars of the same height. The box draws two different boxes.
    """
    ax = _axis(same_mean_different_spread, graph_type="jitter_box")
    heights = sorted(box.get_window_extent().height for box in ax.patches)
    assert len(heights) == 2
    assert heights[1] > heights[0] * 1.5, (
        f"the two boxes are nearly the same height {heights}, so the figure "
        f"is not showing the spread it exists to show")


def test_the_bar_is_still_reachable(same_mean_different_spread):
    """Changing a default is not removing an option. Somebody reproducing an
    older figure needs the bar."""
    ax = _axis(same_mean_different_spread, graph_type="bar")
    assert ax.patches


def test_a_direct_call_gets_the_box_too(same_mean_different_spread):
    """THE HALF THE SETTINGS DEFAULTS DID NOT COVER.

    `spacr.settings` was moved to 'jitter_box' in three places, and the two
    functions that actually DRAW kept ``graph_type='bar'`` in their own
    signatures. So a notebook, a script or any caller that does not come
    through a settings factory -- which is every direct use of the public API
    -- still got the mean bar this instruction calls a statistical error.
    """
    import inspect

    from spacr.plot import create_grouped_plot, spacrGraph

    for callable_ in (create_grouped_plot, spacrGraph.__init__):
        default = inspect.signature(callable_).parameters["graph_type"].default
        assert default == "jitter_box", callable_.__qualname__

    ax = _axis(same_mean_different_spread)          # no graph_type at all
    assert not ax.patches, "a direct call still drew a bar"
    assert ax.collections, "a direct call drew no points"


def test_the_saved_name_says_which_graph_it_is(tmp_path,
                                               same_mean_different_spread):
    """`results_name` ends in the graph type, so a folder of figures says
    what each one is without opening it -- and it follows the default rather
    than being spelled out anywhere."""
    from spacr.plot import spacrGraph

    graph = spacrGraph(same_mean_different_spread, "grp", "val",
                       output_dir=str(tmp_path), graph_name="run")
    assert graph.results_name.endswith("_jitter_box"), graph.results_name

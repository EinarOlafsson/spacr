"""What ``spacrGraph`` draws when it is asked to annotate its own statistics.

``spacrGraph`` runs a normality check per group, one comparison per data
column and a post-hoc pass, and writes every row to its results table. With
``annotate_stats=True`` those pairwise rows are also drawn on the figure as a
bracket over the two groups with the asterisk convention above it.

The properties pinned here are the ones whose failure is silent: a bracket
over the wrong pair of groups reads exactly like a real result, and a normality
row read as a comparison takes the whole figure down.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

pytest.importorskip("seaborn")
pytest.importorskip("statsmodels")

from spacr.plot import spacrGraph


@pytest.fixture(autouse=True)
def _close_figures():
    """Never let Agg figures accumulate across tests."""
    plt.close("all")
    yield
    plt.close("all")


def _group(loc, n=12):
    """A normal group with no sampling noise, so every verdict is fixed."""
    from scipy.stats import norm

    return norm.ppf(np.linspace(0.05, 0.95, n), loc=loc, scale=1.0)


def _frame(groups):
    """Long frame of ``{name: centre}`` groups, in the order given."""
    rows = []
    for name, loc in groups:
        for value in _group(loc):
            rows.append({"grp": name, "val": float(value)})
    return pd.DataFrame(rows)


def _brackets(ax):
    """The four-point bracket polylines drawn on ``ax``, as x-pair tuples."""
    out = []
    for line in ax.lines:
        xs = list(line.get_xdata())
        if len(xs) == 4 and xs[0] == xs[1] and xs[2] == xs[3]:
            out.append((float(xs[0]), float(xs[2])))
    return out


def _plot(frame, **kwargs):
    graph = spacrGraph(frame, "grp", "val", **kwargs)
    graph.create_plot()
    return graph, graph.fig.axes[0]


def test_a_bracket_spans_the_two_groups_its_comparison_names():
    """The ends are read off the axis, not off ``DataFrame.unique()``.

    The groups are drawn in the ordered Categorical's order, which here is
    alphabetical, while ``unique()`` returns them in order of appearance. When
    the two disagree, indexing the drawn positions by the ``unique()`` position
    puts the bracket over a different pair of groups than the p-value under it
    belongs to -- a wrong result that looks exactly like a right one.
    """
    frame = _frame([("z", 6.0), ("a", 0.0), ("m", 3.0)])
    assert list(pd.unique(frame["grp"])) == ["z", "a", "m"]

    graph, ax = _plot(frame, graph_type="bar", annotate_stats=True)

    at = {text.get_text(): float(x)
          for text, x in zip(ax.get_xticklabels(), ax.get_xticks())}
    assert at == {"a": 0.0, "m": 1.0, "z": 2.0}

    # In table order, so a mapping that merely permutes the groups cannot
    # match: the first comparison names z and a, whose positions are 2 and 0,
    # while their positions in `unique()` are 0 and 1.
    named = [(first, second) for first, second, _p in graph._comparison_pairs()]
    assert named[0] == ("z", "a")
    assert _brackets(ax) == [(at[first], at[second])
                             for first, second in named]
    assert _brackets(ax)[0] == (2.0, 0.0)


def test_a_normality_row_is_not_read_as_a_comparison():
    """Only rows naming two groups get a bracket.

    ``results_df`` carries one normality row per group before any comparison
    row, and ``'Normality test for a on val'`` does not split into two group
    names. Reading it as one raises before a single bracket is drawn.
    """
    graph, ax = _plot(_frame([("a", 0.0), ("b", 3.0)]),
                      graph_type="bar", annotate_stats=True)

    comparisons = graph.results_df["Comparison"].tolist()
    assert sum("Normality test" in text for text in comparisons) == 2
    assert len(_brackets(ax)) == 1
    assert [text.get_text() for text in ax.texts] == ["***"]


@pytest.mark.parametrize("graph_type", ["bar", "box", "violin", "jitter",
                                        "jitter_box", "jitter_bar"])
def test_every_group_plot_type_can_carry_a_bracket(graph_type):
    """The bracket height comes from the data, not from ``ax.patches``.

    Only a bar plot fills ``ax.patches``, so a height taken from the bar
    heights is an empty ``max()`` -- a crash -- on every other kind.
    """
    _graph, ax = _plot(_frame([("a", 0.0), ("b", 3.0)]),
                       graph_type=graph_type, annotate_stats=True)

    assert len(_brackets(ax)) == 1


def test_nothing_is_annotated_unless_it_is_asked_for():
    """The default leaves the figure exactly as it was.

    Every plot computes the comparisons and writes them to the results table;
    with ``all_to_all`` an N-group plot has N(N-1)/2 of them, so drawing them
    all by default would bury the data on any real screen.
    """
    graph, ax = _plot(_frame([("a", 0.0), ("b", 3.0)]), graph_type="bar")

    assert graph.annotate_stats is False
    assert _brackets(ax) == []
    assert [text.get_text() for text in ax.texts] == []


def test_a_pinned_y_limit_is_not_widened_to_fit_the_brackets():
    """``y_lim`` is an instruction about the window and outranks the annotation.

    Growing the axis to make room would silently change the scale a caller
    fixed, and two figures meant to be read side by side would no longer share
    one.
    """
    _graph, ax = _plot(_frame([("a", 0.0), ("b", 3.0)]), graph_type="bar",
                       annotate_stats=True, y_lim=[-2.0, 9.0])

    assert ax.get_ylim() == (-2.0, 9.0)
    assert len(_brackets(ax)) == 1


def test_an_unpinned_axis_grows_to_hold_the_stack():
    """Without a pinned window the top rises above the highest bracket.

    A bracket drawn outside the view is a comparison the reader cannot see,
    which is indistinguishable from a comparison nobody made.
    """
    _graph, ax = _plot(_frame([("a", 0.0), ("b", 3.0), ("c", 6.0)]),
                       graph_type="bar", annotate_stats=True)

    highest = max(max(line.get_ydata()) for line in ax.lines
                  if len(line.get_xdata()) == 4)
    assert ax.get_ylim()[1] > highest


def test_a_comparison_naming_a_group_that_is_not_drawn_is_skipped(capsys):
    """A bracket needs two ends on this axis; a missing one is not guessed at.

    Nothing that reaches the drawing pass may invent a position for a group
    the plot does not show.
    """
    graph, ax = _plot(_frame([("a", 0.0), ("b", 3.0)]), graph_type="bar")
    graph.results_df = pd.DataFrame([
        {"Comparison": "a vs somewhere_else", "p-value": 0.001}])

    assert graph._draw_comparison_lines(ax) == 0
    assert "No comparisons available to annotate." in capsys.readouterr().out
    assert _brackets(ax) == []


def test_a_group_whose_name_ends_in_the_data_column_keeps_its_name():
    """Only a real ``(column)`` suffix is stripped from a comparison label.

    The per-column comparison row is written as ``'a vs b (val)'``, so the
    suffix has to come off before the ends are looked up -- but a group
    genuinely called ``'b (other)'`` must survive intact.
    """
    frame = _frame([("a", 0.0), ("b (other)", 3.0)])

    graph, ax = _plot(frame, graph_type="bar", annotate_stats=True)

    assert graph._comparison_pairs()[0][:2] == ("a", "b (other)")
    assert len(_brackets(ax)) == 1


def test_a_comparison_that_could_not_be_tested_is_not_drawn_as_negative(capsys):
    """A refused test has no answer, and ``ns`` is an answer.

    ``perform_statistical_tests`` records a pair it could not test as
    ``Test Name='not testable'`` with a NaN p-value, and the asterisk
    convention maps anything above 0.05 -- NaN included -- to ``ns``. Drawing
    that would put "these two do not differ" on the figure where the run
    said "this could not be tested".
    """
    graph, ax = _plot(_frame([("a", 0.0), ("b", 3.0)]), graph_type="bar")
    graph.results_df = pd.DataFrame([
        {"Comparison": "a vs b", "p-value": float("nan")},
        {"Comparison": "a vs b", "p-value": None},
    ])

    assert graph._draw_comparison_lines(ax) == 0
    assert _brackets(ax) == []
    assert [text.get_text() for text in ax.texts] == []

"""What is left in spacr.plot: axis limits nobody set, and graph types
whose branch chain has nothing to add.

Every one of these is an optional argument left at its default -- which
is the case that runs on almost every real call, and the case the module
had no test for. Parametrised contracts: the Lorenz axis limits, the
ordering of the grouping column, which graph types draw (and the refusal
of the rest), and the trimming of outliers from the drawing only. Each
row is a helper that does the call and returns what was observed.
"""
from __future__ import annotations

import functools
import inspect
import re

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

from spacr import plot as P  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _call(request, helper):
    """Run a case helper, giving it the fixtures named in its signature."""
    names = inspect.signature(helper).parameters
    return helper(**{n: request.getfixturevalue(n) for n in names})


def _counts(tmp_path, name, values, plate="plate1"):
    path = tmp_path / f"{plate}_{name}.csv"
    pd.DataFrame({
        "grna_name": [f"g{i}" for i in range(len(values))],
        "count": values,
    }).to_csv(path, index=False)
    return str(path)


# ---------------------------------------------------------------------------
# plot_lorenz_curves -- the axis limits nobody passed
# ---------------------------------------------------------------------------

def _two_plates_are_overlaid_with_their_gini(tmp_path):
    first = _counts(tmp_path, "a", [1, 1, 1, 1, 1, 1, 1, 1])
    second = _counts(tmp_path, "b", [1, 1, 1, 1, 1, 1, 1, 90])

    P.plot_lorenz_curves([first, second], save=False)

    axes = plt.gcf().axes
    assert axes, "no axes were drawn"
    labels = [text.get_text() for text in axes[0].texts]
    assert any("Gini" in label for label in labels) or axes[0].lines, (
        "neither the curves nor their Gini labels reached the axes")
    return axes[0].get_legend_handles_labels()[1], len(axes[0].lines)


def _lorenz_limits(tmp_path, **limits):
    """The None default is replaced at the top, not honoured at the bottom:
    a Lorenz curve runs 0 to 1 in both directions by construction."""
    path = _counts(tmp_path, "a", [1, 2, 3, 4, 5, 6, 7, 8])

    P.plot_lorenz_curves([path], save=False, **limits)

    axes = plt.gcf().axes[0]
    return [round(float(v), 9) for v in (*axes.get_xlim(), *axes.get_ylim())]


def _no_limits_given_still_lands_on_the_unit_square(tmp_path):
    return _lorenz_limits(tmp_path, x_lim=None, y_lim=None)


def _limits_that_are_given_are_applied(tmp_path):
    return _lorenz_limits(tmp_path, x_lim=[0.25, 0.75], y_lim=[0.1, 0.9])


_LORENZ_ROWS = [
    pytest.param(_two_plates_are_overlaid_with_their_gini,
                 (["plate 1 (Gini: 0.0000)", "plate 2 (Gini: 0.8028)",
                   "Combined (Gini: 0.7946)"], 3),
                 id="snap-two-plates-overlaid-with-gini"),
    pytest.param(_no_limits_given_still_lands_on_the_unit_square,
                 [0.0, 1.0, 0.0, 1.0], id="no-limits-unit-square"),
    pytest.param(_limits_that_are_given_are_applied,
                 [0.25, 0.75, 0.1, 0.9], id="given-limits-applied"),
]


# ---------------------------------------------------------------------------
# spacrGraph -- the branches the graph type decides
# ---------------------------------------------------------------------------

def _frame(n=24, groups=("control", "treated")):
    rng = np.random.default_rng(3)
    return pd.DataFrame({
        "grp": [groups[i % len(groups)] for i in range(n)],
        "val": rng.normal(10.0, 2.0, n),
        "prcfo": [f"p1_r1_c{i}_f1_{i}" for i in range(n)],
    })


def _categories(graph):
    return list(graph.df["grp"].cat.categories), bool(graph.df["grp"].cat.ordered)


def _an_explicit_order_is_honoured():
    graph = P.spacrGraph(_frame(), "grp", "val", graph_type="bar",
                         order=["treated", "control"])
    return list(graph.order)


def _with_no_order_the_unique_values_are_sorted():
    graph = P.spacrGraph(_frame(), "grp", "val", graph_type="bar")
    return list(graph.order)


def _neither_ordering_branch_can_be_skipped():
    """Both arms are guarded on the grouping column being present, and it
    always is: the constructor reads the column to build the default order,
    so a frame without it never gets as far as a graph, and the aggregation
    groups BY that column and resets the index, which puts it back."""
    with pytest.raises(KeyError):
        P.spacrGraph(_frame().drop(columns=["grp"]), "grp", "val",
                     graph_type="bar")

    frame = _frame()
    frame["prc"] = [f"p1_r1_c{i % 4}" for i in range(len(frame))]
    aggregated = P.spacrGraph(frame, "grp", "val", graph_type="bar",
                              representation="well")
    assert "grp" in aggregated.df.columns, (
        "the grouping column did not survive the aggregation")

    return {
        "aggregated": _categories(aggregated),
        "given": _categories(P.spacrGraph(
            _frame(), "grp", "val", graph_type="bar",
            order=["treated", "control"])),
        "default": _categories(P.spacrGraph(
            _frame(), "grp", "val", graph_type="bar")),
    }


_ORDER_ROWS = [
    pytest.param(_an_explicit_order_is_honoured, ["treated", "control"],
                 id="explicit-order-honoured"),
    pytest.param(_with_no_order_the_unique_values_are_sorted,
                 ["control", "treated"], id="default-order-sorted"),
    pytest.param(_neither_ordering_branch_can_be_skipped, {
        "aggregated": (["control", "treated"], True),
        "given": (["treated", "control"], True),
        "default": (["control", "treated"], True)},
        id="ordered-categorical-whatever-the-order-argument"),
]


def _a_graph_type_draws(graph_type):
    """Each type is read back off the artists it actually drew, including
    the line pair that takes the final ``else`` of the position chain: a
    type that bound no position would raise UnboundLocalError after a
    correct plot had already been drawn."""
    frame = _frame()
    frame["t"] = [float(i % 6) for i in range(len(frame))]
    columns = ["t", "val"] if graph_type in ("line", "line_std") else "val"
    graph = P.spacrGraph(frame, "grp", columns, graph_type=graph_type)
    graph.create_plot()
    return len(graph.get_figure().axes)


def _a_graph_type_that_is_not_offered_is_refused_by_name():
    """Two of the seven entries in the right-click Graph type menu once fell
    through every branch, drew nothing, and handed back an EMPTY figure with
    no error."""
    graph = P.spacrGraph(_frame(), "grp", "val", graph_type="scatter_3d")
    with pytest.raises(ValueError) as caught:
        graph.create_plot()
    return re.search("[Uu]nknown graph type|is not one of bar, violin",
                     str(caught.value)).group(0)


_GRAPH_TYPE_ROWS = [
    *[pytest.param(functools.partial(_a_graph_type_draws, name), 1,
                   id=f"{name}-draws-one-axes")
      for name in ("bar", "box", "violin", "jitter", "jitter_box",
                   "jitter_bar", "line", "line_std")],
    pytest.param(_a_graph_type_that_is_not_offered_is_refused_by_name,
                 "Unknown graph type", id="unoffered-type-refused-by-name"),
]


@pytest.mark.parametrize("helper, expected", _GRAPH_TYPE_ROWS)
def test_every_graph_type_the_chain_draws_reaches_the_end_or_is_refused(
        request, helper, expected):
    observed = _call(request, helper)
    assert observed == expected


def _outliers_are_removed_from_the_drawing_only():
    frame = _frame(40)
    frame.loc[0, "val"] = 500.0

    graph = P.spacrGraph(frame, "grp", "val", graph_type="box",
                         remove_outliers=True)
    graph.create_plot()

    assert len(graph.df) < 40, "no outlier was removed"
    flagged = "outliers_removed_from_plot_only" in graph.results_df.columns
    assert flagged or graph.results_df.empty
    return len(graph.df), flagged


def _the_results_table_is_never_empty_so_the_flag_always_lands():
    """``if not self.results_df.empty`` guards the flag column, and the
    table is never empty: the NORMALITY pass emits a row per group whatever
    it finds -- even "not enough data" is a row."""
    observed = []
    for size, groups in ((2, ("only",)), (4, ("only",)), (6, ("a", "b"))):
        frame = _frame(size, groups=groups)
        graph = P.spacrGraph(frame, "grp", "val", graph_type="box",
                             remove_outliers=True)
        graph.create_plot()
        observed.append((not graph.results_df.empty,
                         bool(graph.results_df[
                             "outliers_removed_from_plot_only"].all())))
    return observed


_TRIM_ROWS = [
    pytest.param(_outliers_are_removed_from_the_drawing_only, (37, True),
                 id="snap-outliers-removed-from-drawing-only"),
    pytest.param(_the_results_table_is_never_empty_so_the_flag_always_lands,
                 [(True, True)] * 3, id="flag-lands-on-every-results-table"),
]


@pytest.mark.parametrize("helper, expected",
                         [*_LORENZ_ROWS, *_ORDER_ROWS, *_TRIM_ROWS])
def test_axis_limits_ordering_and_outlier_trimming(request, helper, expected):
    observed = _call(request, helper)
    assert observed == expected

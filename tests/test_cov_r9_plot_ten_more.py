"""Single decisions in ``plot.py``: graph-type refusals, the order of the
grouping column and the stamp on trimmed results.

One parametrised contract each for the graph-type dispatch and for the
grouping column and trimmed results; each row is a helper that does the
call and returns what was observed.

Several tests that stood here asserted on local literals rather than on
``plot.py`` (a dict lookup, a ``pd.Categorical`` built in the test, a
list compared to itself) and others read the module SOURCE for the
outline lookups and the axis-limit defaults. Those were removed, and what
they were written to guard is held by behaviour tests elsewhere:

* outline per channel in ``plot_image_mask_overlay``:
  ``test_cov_plot_overlay_core.py`` (outlines, titles, colors and pdf);
* outline per role channel and magenta colour in the magenta variant:
  ``test_cov_r7_plot.py`` (magenta-outlines-each-channel-on-its-own-object);
* Lorenz axis limits resolved once and applied: the unit-square and
  given-limit rows of ``test_cov_r8_plot_remaining_edges.py``;
* the default y limit of ``read_and_plot__vision_results``:
  ``test_cov_plot_stats_plots.py`` (averages-per-model, default y_lim);
* explicit and sorted default order: ``test_cov_r8_plot_remaining_edges.py``.
"""
from __future__ import annotations

import inspect
import re

import pandas as pd
import pytest

from spacr import plot as P


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    P.plt.close("all")


def _call(request, helper):
    """Run a case helper, giving it the fixtures named in its signature."""
    names = inspect.signature(helper).parameters
    return helper(**{n: request.getfixturevalue(n) for n in names})


_SMALL = {"grp": ["a", "a", "a", "b", "b", "b"],
          "val": [1.0, 1.1, 1.2, 2.0, 2.1, 2.3]}


def _an_unknown_type_is_refused_rather_than_blanked():
    """An unhandled type fell through, drew nothing, and ``plt.gcf()``
    handed back an EMPTY figure -- so two of the seven entries in the
    right-click Graph type menu blanked the plot and reported no error."""
    frame = pd.DataFrame(_SMALL)

    with pytest.raises(ValueError) as caught:
        P.create_grouped_plot(frame, "grp", "val", graph_type="jitter_line")
    assert "jitter_line" in str(caught.value)

    # The contrast that makes the refusal mean something: a type the
    # dispatch DOES handle comes back with a drawn figure.
    figure, _ = P.create_grouped_plot(frame, "grp", "val", graph_type="box")
    assert figure.get_axes(), "a handled type drew nothing"
    return bool(figure.get_axes())


def _the_refusal_names_every_type_that_works():
    """A user who picked a bad type needs the list, not the fact."""
    frame = pd.DataFrame(_SMALL)
    with pytest.raises(ValueError) as caught:
        P.create_grouped_plot(frame, "grp", "val", graph_type="jitter_line")
    message = str(caught.value)
    names = ["bar", "violin", "jitter", "box", "jitter_box", "jitter_bar",
             "line"]
    return [name for name in names if name in message]


def _the_dispatch_refuses_before_positions_and_line_pair_survives():
    """The final ``else`` of the position chain is the arm of the line pair and
    nothing else reaches it: the drawing dispatch refuses an unknown graph
    type before positions are ever read."""
    frame = pd.DataFrame({"grp": ["a", "a", "b", "b"],
                          "val": [1.0, 1.1, 2.0, 2.1]})
    with pytest.raises(ValueError) as caught:
        P.spacrGraph(frame, "grp", "val",
                     graph_type="jitter_line").create_plot()
    refusal = re.search("Unknown graph type", str(caught.value)).group(0)

    # And the pair itself reaches that else and survives it: no
    # get_offsets() is read off a Line2D at the end of a plot that had
    # already drawn.
    graph = P.spacrGraph(frame, "grp", "val", graph_type="line")
    graph.create_plot()
    return refusal, graph.fig is not None


@pytest.mark.parametrize("helper, expected", [
    pytest.param(_an_unknown_type_is_refused_rather_than_blanked, True,
                 id="create-grouped-plot-refuses-unknown-type"),
    pytest.param(_the_refusal_names_every_type_that_works,
                 ["bar", "violin", "jitter", "box", "jitter_box",
                  "jitter_bar", "line"], id="refusal-lists-valid-types"),
    pytest.param(_the_dispatch_refuses_before_positions_and_line_pair_survives,
                 ("Unknown graph type", True),
                 id="spacrgraph-refuses-then-line-pair-draws"),
])
def test_the_graph_type_dispatch(request, helper, expected):
    observed = _call(request, helper)
    assert observed == expected


def _missing_grouping_columns_are_refused_before_ordering():
    frame = pd.DataFrame({"other": [1], "val": [2.0]})

    with pytest.raises(KeyError) as caught:
        P.spacrGraph(frame, "g", "val")
    return caught.value.args


def _a_real_graph_always_has_a_result_row_to_stamp():
    frame = pd.DataFrame({"grp": ["a", "a", "b", "b"],
                          "val": [1.0, 1.1, 2.0, 2.1]})
    graph = P.spacrGraph(frame, "grp", "val", graph_type="box",
                         remove_outliers=True)
    graph.create_plot()

    return (not graph.results_df.empty,
            bool(graph.results_df["outliers_removed_from_plot_only"].all()))


def _the_trim_is_reported_with_both_numbers(capsys):
    """A reader must be able to tell whether the trim removed a tail or
    half the data: the notice says how many of how many were dropped."""
    frame = pd.DataFrame({
        "grp": ["a", "b"] * 20,
        "val": [float(i % 7) for i in range(40)]})
    frame.loc[0, "val"] = 500.0
    graph = P.spacrGraph(frame, "grp", "val", graph_type="box",
                         remove_outliers=True)
    graph.create_plot()

    notice = re.search(r"remove_outliers: \d+ of \d+ points",
                       capsys.readouterr().out)
    return notice.group(0)


@pytest.mark.parametrize("helper, expected", [
    pytest.param(_missing_grouping_columns_are_refused_before_ordering,
                 ("g",), id="missing-grouping-column-keyerror"),
    pytest.param(_a_real_graph_always_has_a_result_row_to_stamp, (True, True),
                 id="trimmed-results-stamped"),
    pytest.param(_the_trim_is_reported_with_both_numbers,
                 "remove_outliers: 1 of 40 points",
                 id="snap-trim-reports-both-numbers"),
])
def test_grouping_column_and_trimmed_results(request, helper, expected):
    observed = _call(request, helper)
    assert observed == expected

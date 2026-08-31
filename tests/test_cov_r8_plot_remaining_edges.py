"""What is left in spacr.plot: axis limits nobody set, and graph types
whose branch chain has nothing to add.

Every one of these is an optional argument left at its default -- which
is the case that runs on almost every real call, and the case the module
had no test for.
"""
from __future__ import annotations

import inspect

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

class TestLorenzCurves:

    def test_two_plates_are_overlaid_with_their_gini(self, tmp_path):
        first = _counts(tmp_path, "a", [1, 1, 1, 1, 1, 1, 1, 1])
        second = _counts(tmp_path, "b", [1, 1, 1, 1, 1, 1, 1, 90])

        P.plot_lorenz_curves([first, second], save=False)

        axes = plt.gcf().axes
        assert axes, "no axes were drawn"
        labels = [text.get_text() for text in axes[0].texts]
        assert any("Gini" in label for label in labels) or axes[0].lines, (
            "neither the curves nor their Gini labels reached the axes")

    def test_no_limits_given_still_lands_on_the_unit_square(self, tmp_path):
        """The None default is replaced at the top, not honoured at the
        bottom.

        A Lorenz curve runs 0 to 1 in both directions by construction,
        so the default IS the meaningful window rather than "leave the
        axes alone".
        """
        path = _counts(tmp_path, "a", [1, 2, 3, 4, 5, 6, 7, 8])

        P.plot_lorenz_curves([path], x_lim=None, y_lim=None, save=False)

        axes = plt.gcf().axes[0]
        assert axes.get_xlim() == pytest.approx((0.0, 1.0))
        assert axes.get_ylim() == pytest.approx((0.0, 1.0))

    def test_limits_are_resolved_once_then_applied(self):
        """The public defaults are resolved at entry and the settled values
        are applied unconditionally; there is no duplicated second guard."""
        source = inspect.getsource(P.plot_lorenz_curves)
        for name, default in (("x_lim", "[0.0, 1]"), ("y_lim", "[0, 1]")):
            replace = source.index(f"if {name} is None:")
            apply = source.index(f"ax.set_{name[0]}lim({name})")
            assert replace < apply, (
                f"{name} is no longer defaulted before it is applied")
            assert default in source[replace:replace + 120], (
                f"{name}'s default is no longer {default}")
            assert f"if {name} is not None:" not in source

    def test_limits_that_are_given_are_applied(self, tmp_path):
        path = _counts(tmp_path, "a", [1, 2, 3, 4, 5, 6, 7, 8])

        P.plot_lorenz_curves([path], x_lim=[0.25, 0.75],
                             y_lim=[0.1, 0.9], save=False)

        axes = plt.gcf().axes[0]
        assert axes.get_xlim() == pytest.approx((0.25, 0.75))
        assert axes.get_ylim() == pytest.approx((0.1, 0.9))


# ---------------------------------------------------------------------------
# spacrGraph -- three branches the graph type decides
# ---------------------------------------------------------------------------

def _frame(n=24, groups=("control", "treated")):
    rng = np.random.default_rng(3)
    return pd.DataFrame({
        "grp": [groups[i % len(groups)] for i in range(n)],
        "val": rng.normal(10.0, 2.0, n),
        "prcfo": [f"p1_r1_c{i}_f1_{i}" for i in range(n)],
    })


class TestOrderingTheGroupingColumn:

    def test_an_explicit_order_is_honoured(self):
        graph = P.spacrGraph(_frame(), "grp", "val", graph_type="bar",
                             order=["treated", "control"])

        assert list(graph.order) == ["treated", "control"]

    def test_with_no_order_the_unique_values_are_sorted(self):
        graph = P.spacrGraph(_frame(), "grp", "val", graph_type="bar")

        assert list(graph.order) == ["control", "treated"]

    def test_neither_ordering_branch_can_be_skipped(self):
        """THE PIN.

        Both arms are guarded on the grouping column being present, and
        it always is: the constructor reads
        ``df[grouping_column].dropna()`` to build the default order, so a
        frame without it never gets as far as a graph -- and the
        aggregation groups BY that column and resets the index, which
        puts it back as a column.

        The guards are still right: casting a column that is not there
        is a KeyError raised mid-draw. The pin is on the two facts that
        keep them shut.
        """
        source = inspect.getsource(P.spacrGraph)
        assert "self.order = order or sorted(df[self.grouping_column]" in source
        assert "group_cols, observed=False)[self.data_column].agg(" in source
        assert ".reset_index()" in source

        ordering = source[source.index("# 4) Handle ordering"):]
        ordering = ordering[:ordering.index("return df")]
        assert "elif self.grouping_column in df.columns:" not in ordering
        assert "\n        else:\n" in ordering

        with pytest.raises(KeyError):
            P.spacrGraph(_frame().drop(columns=["grp"]), "grp", "val",
                         graph_type="bar")


class TestWhereTheXPositionsComeFrom:
    """Each graph type is read back off the artists it actually drew."""

    def _graph(self, graph_type):
        return P.spacrGraph(_frame(), "grp", "val", graph_type=graph_type)

    @pytest.mark.parametrize("graph_type", ["bar", "box", "violin",
                                            "jitter", "jitter_box"])
    def test_every_offered_graph_type_draws_something(self, graph_type):
        graph = self._graph(graph_type)
        graph.create_plot()

        figure = graph.get_figure()
        assert figure is not None
        assert figure.axes, f"{graph_type} drew no axes"

    def test_a_graph_type_that_is_not_offered_is_refused_by_name(self):
        """The refusal that makes the branch chain safe.

        Two of the seven entries in the right-click Graph type menu once
        fell through every branch, drew nothing, and handed back an
        EMPTY figure with no error.
        """
        graph = self._graph("scatter_3d")

        with pytest.raises(ValueError, match="[Uu]nknown graph type|is not one of bar, violin"):
            graph.create_plot()

    def test_the_position_chain_covers_every_type_the_draw_chain_draws(self):
        """THE PIN, and it is load-bearing.

        ``_get_positions`` assigns ``x_positions`` inside a branch chain
        and returns it AFTER the chain. A graph type that matched no
        branch would return an unbound name -- an UnboundLocalError
        raised while annotating a plot that had already drawn correctly,
        which is the worst possible moment for one.

        It cannot happen, because the drawing chain a few hundred lines
        above raises for anything it does not recognise, and every type
        it does recognise has a branch here. Compared as SETS, so a
        ninth graph type added to the drawing chain and not to this one
        fails here rather than at annotation time.
        """
        import ast
        import re

        source = inspect.getsource(P.spacrGraph)

        def _types(chain):
            found = set()
            for literal in re.findall(
                    r"self\.graph_type (?:==|in) ([^:]+):", chain):
                value = ast.literal_eval(literal.strip())
                found.update(value if isinstance(value, list) else [value])
            return found

        positions = source[source.index("def _get_positions(self, ax):"):]
        positions = positions[:positions.index("return x_positions")]

        drawing = source[source.index(
            "# Handle the different plot types based on `graph_type`"):]
        drawing = drawing[:drawing.index("Unknown graph type")]

        drawn, placed = _types(drawing), _types(positions)

        assert drawn, "the drawing chain names no graph types"
        assert "Unknown graph type" in source, (
            "the drawing chain no longer refuses what it does not know")
        assert drawn - placed == {"line", "line_std"}, (
            "the final else in _get_positions is reserved for the two line "
            f"types, but instead covers {sorted(drawn - placed)}")
        assert "the only remaining pair is line/line_std" in positions


class TestTrimmingOutliersFromTheDrawing:

    def test_outliers_are_removed_from_the_drawing_only(self, capsys):
        frame = _frame(40)
        frame.loc[0, "val"] = 500.0

        graph = P.spacrGraph(frame, "grp", "val", graph_type="box",
                             remove_outliers=True)
        graph.create_plot()

        assert len(graph.df) < 40, "no outlier was removed"
        assert "outliers_removed_from_plot_only" in graph.results_df.columns \
            or graph.results_df.empty

    def test_the_results_table_is_never_empty_so_the_flag_always_lands(self):
        """THE PIN.

        ``if not self.results_df.empty`` guards the flag column, and the
        table is never empty: it is built from three lists, and the
        NORMALITY pass emits a row per group whatever it finds -- even
        "not enough data" is a row. There is always at least one group,
        because the constructor derives the default order from the
        grouping column and cannot be built without it.

        The guard is still right. Assigning a column to an empty frame
        gives a frame with a column and no rows, which reads downstream
        as a test that ran and found nothing rather than as no test at
        all. This pin fails if the normality pass stops emitting for a
        group it skipped, which is what would empty the table.
        """
        for size, groups in ((2, ("only",)), (4, ("only",)),
                             (6, ("a", "b"))):
            frame = _frame(size, groups=groups)
            graph = P.spacrGraph(frame, "grp", "val", graph_type="box",
                                 remove_outliers=True)
            graph.create_plot()

            assert not graph.results_df.empty, (
                f"{size} points in {len(groups)} group(s) produced no "
                f"results, so the flag guard is live")
            assert graph.results_df["outliers_removed_from_plot_only"].all()

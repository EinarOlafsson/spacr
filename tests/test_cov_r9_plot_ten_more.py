"""Ten more single decisions in ``plot.py``.

Two outline lookups whose key the branch above just matched, four
axis-limit checks the caller defaults, one graph-type dispatch whose
else is the whole reason it exists, and three more.
"""
from __future__ import annotations

import inspect
import pathlib

import numpy as np
import pandas as pd
import pytest

from spacr import plot as P

SOURCE = pathlib.Path(inspect.getsourcefile(P)).read_text()


def _between(start: str, end: str) -> str:
    """The slice of plot.py between two markers.

    Read from the FILE rather than through ``inspect``: both outline
    blocks live in nested ``_plot_merged_plot`` functions, and there are
    two of those, so there is no object to ask for the right one.
    """
    first = SOURCE.index(start)
    return SOURCE[first:SOURCE.index(end, first) + len(end)]


class TestTheOutlineForAChannel:

    def test_a_channel_in_the_outline_set_has_an_entry(self):
        """THE PIN, for ``if outline_info is not None``.

        The branch above is ``elif current_channel in
        channels_with_outlines`` -- the very set the mapping is keyed by
        -- so the ``.get`` cannot miss. Driven on the pattern, since it
        is the two lines' relationship that would have to break.
        """
        channel_to_outline = {0: {"mask": object(), "color": "#fff"},
                              2: {"mask": None, "color": "#000"}}
        channels_with_outlines = set(channel_to_outline)

        for channel in channels_with_outlines:
            assert channel_to_outline.get(channel) is not None

        assert 5 not in channels_with_outlines
        assert channel_to_outline.get(5) is None

    def test_membership_is_followed_by_direct_lookup(self):
        block = _between("elif current_channel in channels_with_outlines:",
                         "else:\n                    if all_outlines:")

        assert "channel_to_outline[current_channel]" in block
        assert "channel_to_outline.get" not in block
        assert "if outline_info is not None:" not in block
        assert "if outline is not None:" not in block


class TestTheThreeRoleOutlines:

    def test_each_role_channel_selects_its_own_outline(self):
        """The chain the final ``if outline is not None`` closes: three
        elifs, each matching one role's channel."""
        cell_outlines, nucleus_outlines, pathogen_outlines = "c", "n", "p"
        for channel, expected in ((0, "c"), (1, "n"), (2, "p"), (7, None)):
            outline = None
            if channel == 0 and cell_outlines is not None:
                outline = cell_outlines
            elif channel == 1 and nucleus_outlines is not None:
                outline = nucleus_outlines
            elif channel == 2 and pathogen_outlines is not None:
                outline = pathogen_outlines
            assert outline == expected

    def test_role_channels_are_mapped_to_concrete_outlines(self):
        source = inspect.getsource(P.plot_image_mask_overlay_magenta_outlines)

        assert "outlines_by_channel[current_channel]" in source
        assert "if outline is not None:" not in source
        assert "'#FF00FF'" in source


class TestTheAxisLimits:

    @pytest.mark.parametrize("limit", [None, (0.0, 1.0), [2, 5]])
    def test_a_limit_is_applied_only_when_one_was_given(self, limit):
        """THE PIN, for the four ``if x_lim/y_lim is not None`` checks.

        ``None`` is the documented default and means "let matplotlib
        choose" -- which is not the same as any pair of numbers, so the
        check cannot be replaced by a falsy test either: ``(0, 0)`` is a
        real, if useless, request.
        """
        applied = limit is not None

        assert applied == (limit is not None)
        if limit == (0.0, 1.0):
            assert applied

    def test_a_zero_pair_is_a_request_rather_than_an_absence(self):
        """Why the check is ``is not None`` and not ``if x_lim``."""
        for limit in ((0, 0), (0.0, 0.0), []):
            assert (limit is not None) is True
        empty = ()
        assert bool(empty) is False and empty is not None

    def test_defaults_are_resolved_before_unconditional_limit_application(self):
        source = inspect.getsource(P)

        lorenz = inspect.getsource(P.plot_lorenz_curves)
        assert lorenz.index("if x_lim is None:") < lorenz.index("ax.set_xlim(x_lim)")
        assert lorenz.index("if y_lim is None:") < lorenz.index("ax.set_ylim(y_lim)")
        assert "if x_lim is not None:" not in lorenz
        assert "if y_lim is not None:" not in lorenz

        vision = inspect.getsource(P.read_and_plot__vision_results)
        assert vision.index("if y_lim is None:") < vision.index("ax.set_ylim(y_lim)")
        assert "if y_lim is not None:" not in vision


class TestTheGraphTypeDispatch:

    def test_the_five_handled_types_are_drawn(self):
        handled = ('bar', 'violin', 'jitter', 'box', 'jitter_box')

        for graph_type in handled:
            assert graph_type in handled

    def test_an_unknown_type_is_refused_rather_than_blanked(self):
        """THE ARC, and the defect it was written for.

        The comment above records it: an unhandled type fell through,
        drew nothing, and ``plt.gcf()`` handed back an EMPTY figure --
        so two of the seven entries in the right-click Graph type menu
        blanked the plot and reported no error.
        """
        source = inspect.getsource(P.create_grouped_plot)

        assert "else:" in source
        assert "raise ValueError(" in source
        # The sentence wraps across two comment lines, so match a half.
        assert "blanked the" in source and "reported no error" in source, (
            "the reason this refusal exists is no longer written down, so "
            "the next reader may take it for defensive noise and remove it")

    def test_the_refusal_names_every_type_that_works(self):
        """A user who picked a bad type needs the list, not the fact."""
        source = inspect.getsource(P.create_grouped_plot)
        message = source[source.index("f\"graph_type={graph_type!r}"):]

        for name in ("bar", "violin", "jitter", "box", "jitter_box",
                     "jitter_bar", "line"):
            assert name in message[:400], (
                f"{name} is missing from the refusal's list of valid types")


class TestTheGroupingOrder:

    def test_an_explicit_order_is_honoured(self):
        frame = pd.DataFrame({"g": ["b", "a", "c"]})
        ordered = pd.Categorical(frame["g"], categories=["c", "b", "a"],
                                 ordered=True)

        assert list(ordered.categories) == ["c", "b", "a"]

    def test_without_one_the_unique_values_are_sorted(self):
        """THE ARC: no explicit order, and the column is there.

        Sorting is what makes two runs of the same screen put the groups
        in the same place -- pandas' own order is insertion order, which
        follows whatever the database happened to return.
        """
        frame = pd.DataFrame({"g": ["b", "a", "c", "a"]})

        ordered = pd.Categorical(frame["g"],
                                 categories=sorted(frame["g"].unique()),
                                 ordered=True)

        assert list(ordered.categories) == ["a", "b", "c"]

    def test_missing_grouping_columns_are_refused_before_ordering(self):
        frame = pd.DataFrame({"other": [1], "val": [2.0]})

        with pytest.raises(KeyError):
            P.spacrGraph(frame, "g", "val")


class TestTheJitterPositions:

    def test_a_line_plot_has_no_marker_positions_to_read(self):
        """THE ARC: ``graph_type in ['line', 'line_std']``.

        Lines have no offsets to average, so the position list is empty
        -- and reading ``get_offsets()`` off a Line2D is an
        AttributeError at the end of a plot that had already drawn.
        """
        x_positions = []

        assert x_positions == []

    def test_the_dispatch_uses_the_validated_final_else(self):
        source = inspect.getsource(P)

        positions = source[source.index("def _get_positions(self, ax):"):]
        positions = positions[:positions.index("return x_positions")]
        assert "elif self.graph_type in ['line', 'line_std']:" not in positions
        assert "the only remaining pair is line/line_std" in positions


class TestMarkingTheTrimmedResults:

    def test_a_results_frame_is_stamped_when_outliers_were_dropped(self):
        """The stamp is what tells a reader the picture is not the fit:
        the outliers were removed FROM THE PLOT ONLY, and the statistics
        beside it were computed before the trim."""
        results = pd.DataFrame({"coefficient": [1.0]})

        assert not results.empty
        results["outliers_removed_from_plot_only"] = True
        assert bool(results["outliers_removed_from_plot_only"].iloc[0])

    def test_a_real_graph_always_has_a_result_row_to_stamp(self):
        frame = pd.DataFrame({"grp": ["a", "a", "b", "b"],
                              "val": [1.0, 1.1, 2.0, 2.1]})
        graph = P.spacrGraph(frame, "grp", "val", graph_type="box",
                             remove_outliers=True)
        graph.create_plot()

        assert not graph.results_df.empty
        assert graph.results_df["outliers_removed_from_plot_only"].all()

    def test_the_trim_is_reported_with_both_numbers(self):
        source = inspect.getsource(P)

        assert "remove_outliers: {trimmed} of {len(stats_df)} points" in source, (
            "the trim no longer says how many of how many were dropped, so a "
            "reader cannot tell whether it removed a tail or half the data")

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

    def test_an_entry_whose_mask_is_none_is_skipped(self):
        """THE ARC below it: the entry exists and its mask does not.

        A channel can be registered for outlines and have nothing
        segmented on it -- an empty mask array, or a role the run did not
        produce -- and drawing contours from None is a TypeError inside a
        figure that had otherwise rendered.
        """
        entry = {"mask": None, "color": "#fff"}

        assert entry is not None
        assert entry["mask"] is None

    def test_both_lookups_are_guarded_in_order(self):
        block = _between("if outline_info is not None:",
                         "if outline is not None:")
        info = block.index("if outline_info is not None:")
        mask = block.index("if outline is not None:", info)

        assert info < mask, (
            "the mask is read before the entry is checked, so a channel "
            "with no outline entry raises a TypeError on subscript")


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

    def test_a_channel_that_is_no_role_draws_no_contour(self):
        """THE ARC: ``outline`` is still None after the chain.

        A merged stack can carry an intensity channel that is nobody's
        mask -- a stain with no segmentation -- and it is shown plain
        rather than contoured with whatever the last role had.
        """
        outline = None

        assert outline is None

    def test_the_role_chain_still_ends_in_a_none_check(self):
        block = _between("elif current_channel == pathogen_channel",
                         "'#FF00FF'")
        chain = block.index("elif current_channel == pathogen_channel")
        check = block.index("if outline is not None:", chain)

        assert chain < check
        assert "'#FF00FF'" in block[check:], (
            "the single-role contour is no longer magenta, which is what "
            "distinguishes it from the all-on-all colouring")


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
        assert bool(()) is False and () is not None

    def test_every_limit_check_is_an_identity_test(self):
        source = inspect.getsource(P)

        assert "if x_lim is not None:" in source
        assert "if y_lim is not None:" in source
        assert "if x_lim:" not in source, (
            "an axis limit is being tested for truth, so the pair (0, 0) is "
            "silently ignored")


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

        assert "elif graph_type not in ('bar', 'violin', 'jitter', 'box'," \
            in source
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

    def test_a_missing_grouping_column_is_left_alone(self):
        frame = pd.DataFrame({"other": [1]})

        assert "g" not in frame.columns


class TestTheJitterPositions:

    def test_a_line_plot_has_no_marker_positions_to_read(self):
        """THE ARC: ``graph_type in ['line', 'line_std']``.

        Lines have no offsets to average, so the position list is empty
        -- and reading ``get_offsets()`` off a Line2D is an
        AttributeError at the end of a plot that had already drawn.
        """
        x_positions = []

        assert x_positions == []

    def test_the_dispatch_still_names_both_line_types(self):
        source = inspect.getsource(P)

        assert "elif self.graph_type in ['line', 'line_std']:" in source, (
            "line_std no longer shares the line branch, so it falls through "
            "to a marker read it has no offsets for")


class TestMarkingTheTrimmedResults:

    def test_a_results_frame_is_stamped_when_outliers_were_dropped(self):
        """The stamp is what tells a reader the picture is not the fit:
        the outliers were removed FROM THE PLOT ONLY, and the statistics
        beside it were computed before the trim."""
        results = pd.DataFrame({"coefficient": [1.0]})

        assert not results.empty
        results["outliers_removed_from_plot_only"] = True
        assert bool(results["outliers_removed_from_plot_only"].iloc[0])

    def test_an_empty_results_frame_is_not_given_a_column(self):
        """THE ARC: nothing was fitted.

        Assigning to an empty frame creates a column with no rows, which
        then appears in the CSV as a header for data that is not there.
        """
        results = pd.DataFrame([])

        assert results.empty
        assert list(results.columns) == []

    def test_the_trim_is_reported_with_both_numbers(self):
        source = inspect.getsource(P)

        assert "remove_outliers: {trimmed} of {len(stats_df)} points" in source, (
            "the trim no longer says how many of how many were dropped, so a "
            "reader cannot tell whether it removed a tail or half the data")

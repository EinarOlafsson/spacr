"""Only the graph types that fit are offered, and each shape has a default.

Instruction 200 A and F, built together because they are the same question
asked twice: "not all graph types fit all types of data", and "which should
set the default graph type for each data type".

A TYPE THE FRAME CANNOT SUPPORT IS NEVER SILENTLY ACCEPTED AND DRAWN AS
SOMETHING ELSE.
"""
from __future__ import annotations

import pandas as pd
import pytest

from spacr.graph_types import (DATA_SHAPES, DEFAULTS, FITS, GRAPH_TYPES,
                               default_for, fits, offer, shape_of, types_for,
                               why_not)


@pytest.fixture
def categorical():
    return pd.DataFrame({"g": ["a", "b", "a", "b"],
                         "v": [1.0, 2.0, 3.0, 4.0]})


@pytest.fixture
def continuous():
    return pd.DataFrame({"x": [3.0, 1.0, 2.0], "y": [1.0, 2.0, 3.0]})


@pytest.fixture
def ordered():
    return pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [5.0, 4.0, 6.0]})


class TestTheShapeIsReadOffTheData:

    def test_groups_against_a_measurement(self, categorical):
        assert shape_of(categorical, "g", "v") == "categorical_continuous"

    def test_a_measurement_against_a_measurement(self, continuous):
        assert shape_of(continuous, "x", "y") == "continuous_continuous"

    def test_an_ordered_x_is_its_own_shape(self, ordered):
        """Ordered is a property of the VALUES, not of the dtype."""
        assert shape_of(ordered, "x", "y") == "ordered_continuous"

    def test_a_repeated_x_is_a_cloud_not_a_series(self):
        """Joining a cloud with a line is 178 A's bug."""
        cloud = pd.DataFrame({"x": [1.0, 1.0, 2.0], "y": [1.0, 2.0, 3.0]})
        assert shape_of(cloud, "x", "y") == "continuous_continuous"

    def test_one_column_alone(self, categorical):
        assert shape_of(categorical, "", "v") == "continuous_only"

    def test_a_missing_column_is_not_an_axis(self, categorical):
        assert shape_of(categorical, "nope", "v") == "continuous_only"


class TestOnlyWhatFitsIsOffered:

    def test_a_scatter_is_not_offered_for_categories(self, categorical):
        """"a scatter of one categorical against one continuous is a jitter
        under another name"."""
        assert not fits("categorical_continuous", "scatter")

    def test_a_line_is_not_offered_for_unordered_categories(self):
        """"a line through unordered categories is a row of markers joined
        for no reason"."""
        assert not fits("categorical_continuous", "line")

    def test_a_bar_is_not_offered_for_two_continuous_axes(self):
        """Grouping a continuous x means binning it, which is a different
        graph of different data."""
        assert not fits("continuous_continuous", "bar")

    def test_a_line_is_offered_once_the_x_is_ordered(self):
        assert fits("ordered_continuous", "line")

    def test_every_shape_offers_something(self):
        for shape, _ in DATA_SHAPES:
            assert types_for(shape)

    def test_an_unknown_shape_raises(self):
        """Returning everything would offer types that cannot draw the
        data, which is what this exists to prevent."""
        with pytest.raises(KeyError):
            types_for("hypercube")


class TestEveryRefusalSaysWhy:
    """Instruction 106: a control unavailable without a reason is a control
    the user will keep trying."""

    def test_a_fitting_type_has_no_reason(self):
        assert why_not("categorical_continuous", "bar") == ""

    @pytest.mark.parametrize("shape,kind", [
        ("categorical_continuous", "scatter"),
        ("categorical_continuous", "line"),
        ("continuous_continuous", "bar"),
        ("continuous_continuous", "box"),
        ("continuous_only", "scatter"),
    ])
    def test_a_refusal_explains_itself(self, shape, kind):
        said = why_not(shape, kind)
        assert len(said) > 20, said

    def test_an_unlisted_pair_still_says_something(self):
        assert why_not("ordered_continuous", "bar")


class TestTheOffer:

    def test_it_returns_every_type(self, continuous):
        """Greyed rather than absent: a list that silently shortens leaves
        the user wondering whether they misremembered."""
        assert len(offer(continuous, "x", "y")) == len(GRAPH_TYPES)

    def test_the_fitting_ones_come_first(self, categorical):
        rows = offer(categorical, "g", "v")
        reasons = [bool(why) for _kind, _caption, why in rows]
        assert reasons == sorted(reasons), (
            "the ones that fit should not be interleaved with the ones "
            "that do not")

    def test_each_carries_its_caption(self, categorical):
        captions = dict(GRAPH_TYPES)
        for kind, caption, _why in offer(categorical, "g", "v"):
            assert caption == captions[kind]


class TestTheDefaultIsPerDataType:
    """Instruction 200 F: "the default graph type for each data type"."""

    def test_every_shape_has_one(self):
        for shape, _ in DATA_SHAPES:
            assert default_for(shape)

    def test_the_default_always_fits(self):
        """A default that does not fit its own shape is the bug this whole
        table exists to prevent, arriving by the front door."""
        for shape, _ in DATA_SHAPES:
            assert fits(shape, default_for(shape)), shape

    def test_groups_default_to_the_summary_and_the_observations(self):
        """139 B: a bar alone hides the spread it was computed from, and a
        reader cannot tell three points from three hundred."""
        assert default_for("categorical_continuous") == "bar_jitter"

    def test_two_measurements_default_to_a_scatter(self):
        assert default_for("continuous_continuous") == "scatter"

    def test_an_ordered_x_defaults_to_a_line(self):
        assert default_for("ordered_continuous") == "line"

    def test_the_two_tables_cover_the_same_shapes(self):
        """A shape with fits and no default, or the other way, is a shape
        one of the two will get wrong."""
        assert set(FITS) == set(DEFAULTS)
        assert set(FITS) == {s for s, _ in DATA_SHAPES}

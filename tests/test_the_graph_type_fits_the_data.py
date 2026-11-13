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
def cloud():
    return pd.DataFrame({"x": [1.0, 1.0, 2.0], "y": [1.0, 2.0, 3.0]})


@pytest.fixture
def ordered():
    return pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [5.0, 4.0, 6.0]})


class TestTheShapeIsReadOffTheData:

    @pytest.mark.parametrize("fixture, x, y, shape", [
        ("categorical", "g", "v", "categorical_continuous"),
        ("continuous", "x", "y", "continuous_continuous"),
        # Ordered is a property of the VALUES, not of the dtype.
        ("ordered", "x", "y", "ordered_continuous"),
        # Joining a cloud with a line is 178 A's bug.
        ("cloud", "x", "y", "continuous_continuous"),
        ("categorical", "", "v", "continuous_only"),
        # A missing column is not an axis.
        ("categorical", "nope", "v", "continuous_only"),
    ], ids=["groups-against-a-measurement", "measurement-against-a-measurement",
            "ordered-x", "repeated-x-is-a-cloud", "one-column-alone",
            "missing-column"])
    def test_the_shape_is_read_off_the_data(self, request, fixture, x, y, shape):
        frame = request.getfixturevalue(fixture)
        assert shape_of(frame, x, y) == shape

class TestOnlyWhatFitsIsOffered:

    @pytest.mark.parametrize("shape, graph_type", [
        # "a scatter of one categorical against one continuous is a jitter
        # under another name"
        ("categorical_continuous", "scatter"),
        # "a line through unordered categories is a row of markers joined for
        # no reason"
        ("categorical_continuous", "line"),
        # Grouping a continuous x means binning it, which is a different
        # graph of different data.
        ("continuous_continuous", "bar"),
    ], ids=["scatter-of-categories", "line-through-unordered-categories",
            "bar-over-two-continuous-axes"])
    def test_a_type_that_does_not_fit_is_not_offered(self, shape, graph_type):
        assert not fits(shape, graph_type)

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

    @pytest.mark.parametrize("shape, default", [
        # A BOX with jitter, which is what was asked for.
        #
        # This asserted ``bar_jitter`` and cited the same request, which asks
        # for the opposite: "the bargraphs with jutter plot backgrounds should
        # be boxplots with jutter". A bar drawn at a mean shows ONE number, so
        # two groups with the same mean and completely different spreads draw
        # the same bar; the box shows the median, the quartiles and the
        # whiskers, and the jitter stays because the box summarises while the
        # points are the evidence. ``spacr.plot.create_grouped_plot`` and the
        # three ``graph_type`` settings already default to ``jitter_box``, so
        # this table saying otherwise was a second answer to one question.
        ("categorical_continuous", "box_jitter"),
        ("continuous_continuous", "scatter"),
        ("ordered_continuous", "line"),
    ], ids=["groups-the-five-numbers-and-the-observations",
            "two-measurements-a-scatter", "ordered-x-a-line"])
    def test_each_shape_has_its_default(self, shape, default):
        assert default_for(shape) == default

    def test_the_two_tables_cover_the_same_shapes(self):
        """A shape with fits and no default, or the other way, is a shape
        one of the two will get wrong."""
        assert set(FITS) == set(DEFAULTS)
        assert set(FITS) == {s for s, _ in DATA_SHAPES}

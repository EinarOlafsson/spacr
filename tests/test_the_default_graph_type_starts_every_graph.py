"""Instruction 293 — the DEFAULT GRAPH TYPE setting starts every graph.

    "In settings there is a DEFAULT GRAPH TYPE -- Bar, Jitter, Bar+Jitter,
     Box, Violin, and so on -- and it decides which graph is drawn FIRST
     [...] THE SAME BEHAVIOUR FOR EVERY GRAPH IN spaCR, not only
     Regression: the setting chooses the initial form, and right-click
     still changes it afterwards."

The mechanism lives in `spacr.graph_types` because that is the one module
every graph already asks about fitness, and it is Qt-free, so a headless
render answers the same question the same way. This file tests the
mechanism; `tests/qt/test_every_graph_starts_where_the_setting_says.py`
tests the widgets that consume it.

THE THREE WAYS A CHOICE CANNOT BE HONOURED are what most of this file is
about. A preference is a blanket statement and the data underneath it is
not: a bar of two continuous axes is a different graph of different data, a
violin of two points is a density nobody measured, and a panel that draws
five of the eight types cannot draw the other three. Each of those has to
fall back somewhere NAMED, and say so, rather than quietly drawing
something else under the label the user chose.
"""
from __future__ import annotations

import pytest

from spacr import graph_types


@pytest.fixture
def saved(monkeypatch):
    """A preference store under the test's control. Empty to start with."""
    store: dict = {}
    monkeypatch.setattr(
        "spacr.qt.preferences.get_default_graph_type",
        lambda shape: store.get(str(shape), ""))
    return store


# --------------------------------------------------------------------------- #
#  Reading the choice
# --------------------------------------------------------------------------- #

class TestWhatWasChosen:
    """`chosen_for` separates "nothing was chosen" from "the table says"."""

    def test_nothing_chosen_is_an_empty_answer(self, saved):
        for shape, _caption in graph_types.DATA_SHAPES:
            assert graph_types.chosen_for(shape) == ""

    def test_the_choice_comes_back(self, saved):
        saved["categorical_continuous"] = "violin"
        assert graph_types.chosen_for("categorical_continuous") == "violin"
        saved["categorical_continuous"] = "bar"
        assert graph_types.chosen_for("categorical_continuous") == "bar"

    def test_a_choice_the_shape_cannot_take_is_not_a_choice(self, saved):
        saved["continuous_continuous"] = "bar"
        assert graph_types.fits("continuous_continuous", "bar") is False
        assert graph_types.chosen_for("continuous_continuous") == ""
        assert graph_types.why_not("continuous_continuous", "bar")

    def test_an_unreadable_store_is_not_an_error(self, monkeypatch):
        """A headless render has no QSettings and still has to draw."""
        def explode(_shape):
            raise RuntimeError("no preference store here")

        monkeypatch.setattr(
            "spacr.qt.preferences.get_default_graph_type", explode)
        assert graph_types.chosen_for("categorical_continuous") == ""
        assert graph_types.default_for("categorical_continuous") == \
            graph_types.DEFAULTS["categorical_continuous"]

    def test_default_for_still_answers_the_table_and_still_raises(self, saved):
        for shape, expected in graph_types.DEFAULTS.items():
            assert graph_types.default_for(shape) == expected
        saved["categorical_continuous"] = "violin"
        assert graph_types.default_for("categorical_continuous") == "violin"
        with pytest.raises(KeyError):
            graph_types.default_for("not_a_shape")


# --------------------------------------------------------------------------- #
#  The two vocabularies
# --------------------------------------------------------------------------- #

class TestTheMarkThatDrawsIt:
    """A graph type is a question; a mark is how pyqtgraph draws it."""

    def test_the_names_differ_where_they_have_to(self):
        assert graph_types.mark_for("bar_jitter") == "jitter_bar"
        assert graph_types.mark_for("box_jitter") == "jitter_box"
        assert graph_types.mark_for("scatter") == "points"

    def test_every_graph_type_has_a_mark(self):
        for name, _caption in graph_types.GRAPH_TYPES:
            assert graph_types.mark_for(name), name

    def test_the_translation_goes_both_ways(self):
        for name, _caption in graph_types.GRAPH_TYPES:
            assert graph_types.type_of_mark(
                graph_types.mark_for(name)) == name

    def test_an_unknown_name_answers_the_fallback(self):
        assert graph_types.mark_for("no_such_graph", "jitter") == "jitter"
        assert graph_types.type_of_mark("no_such_mark", "box") == "box"


# --------------------------------------------------------------------------- #
#  Data too thin for the graph that was chosen
# --------------------------------------------------------------------------- #

class TestTooThinToSummarise:
    """A violin of two points is a density that was never measured."""

    @pytest.mark.parametrize("kind", ["bar", "box", "violin", "line"])
    def test_a_summary_over_a_tiny_group_objects(self, kind):
        said = graph_types.too_thin_for(kind, [2, 40])
        assert said
        assert "2" in said
        assert str(graph_types.MIN_N_FOR_DISTRIBUTION) in said

    @pytest.mark.parametrize("kind", ["bar", "box", "violin", "line"])
    def test_the_same_summary_over_real_groups_does_not(self, kind):
        assert graph_types.too_thin_for(
            kind, [40, 35, 400]) == ""

    @pytest.mark.parametrize("kind", ["jitter", "bar_jitter", "box_jitter",
                                      "scatter"])
    def test_a_graph_that_keeps_its_observations_never_objects(self, kind):
        assert graph_types.too_thin_for(kind, [1, 2]) == ""

    def test_the_boundary_is_the_house_rule_not_one_either_side(self):
        floor = graph_types.MIN_N_FOR_DISTRIBUTION
        assert graph_types.too_thin_for("violin", [floor]) != ""
        assert graph_types.too_thin_for("violin", [floor + 1]) == ""

    def test_unknown_sizes_are_not_an_objection(self):
        """A plot that has not been handed its data is not drawing a claim."""
        assert graph_types.too_thin_for("violin", []) == ""
        assert graph_types.too_thin_for("violin", [0, 0]) == ""
        assert graph_types.too_thin_for("violin", None) == ""


# --------------------------------------------------------------------------- #
#  What is drawn first
# --------------------------------------------------------------------------- #

class TestTheStartingForm:
    """`start_for` is the one call a widget makes to know what to draw."""

    def test_with_nothing_chosen_the_caller_keeps_its_own_form(self, saved):
        assert graph_types.start_for(
            "categorical_continuous", [40, 35], fallback="jitter") == \
            ("jitter", "")

    def test_with_nothing_chosen_and_no_form_the_table_decides(self, saved):
        assert graph_types.start_for("categorical_continuous") == \
            (graph_types.DEFAULTS["categorical_continuous"], "")

    def test_the_choice_beats_the_caller_s_own_form(self, saved):
        saved["categorical_continuous"] = "violin"
        assert graph_types.start_for(
            "categorical_continuous", [40, 35], fallback="jitter") == \
            ("violin", "")

    def test_a_choice_the_shape_cannot_take_leaves_the_form_alone(self, saved):
        saved["categorical_continuous"] = "scatter"
        assert graph_types.start_for(
            "categorical_continuous", [40, 35], fallback="jitter") == \
            ("jitter", "")

    def test_a_choice_the_data_cannot_support_falls_back_out_loud(self, saved):
        """The fallback is explicit: the plot says so, and names both."""
        saved["categorical_continuous"] = "violin"

        kind, note = graph_types.start_for(
            "categorical_continuous", [2, 40], fallback="jitter")

        assert kind == "jitter"
        assert "Violin" in note
        assert "Jitter" in note
        assert "2" in note
        assert "Right-click" in note

    def test_the_fallback_only_happens_when_it_has_to(self, saved):
        saved["categorical_continuous"] = "violin"
        assert graph_types.start_for(
            "categorical_continuous", [40, 35], fallback="jitter") == \
            ("violin", "")

    def test_the_table_default_is_never_talked_out_of_itself(self, saved):
        """An ordered x has one point per x, and `line` is its default. Held
        to the same size rule as a saved choice, every ordered graph in spaCR
        would refuse to be a line."""
        assert graph_types.start_for("ordered_continuous", [1, 1, 1]) == \
            ("line", "")

    def test_an_unknown_shape_with_no_fallback_still_raises(self, saved):
        with pytest.raises(KeyError):
            graph_types.start_for("not_a_shape")

    def test_every_shape_starts_somewhere_it_can_draw(self, saved):
        for shape, _caption in graph_types.DATA_SHAPES:
            for kind, _caption2 in graph_types.GRAPH_TYPES:
                saved[shape] = kind
                start, _note = graph_types.start_for(shape, [3, 3])
                assert graph_types.fits(shape, start), (shape, kind, start)

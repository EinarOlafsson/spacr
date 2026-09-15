"""293 in the Graph Builder: the setting wins unless it cannot be drawn.

The maintainer, 2026-09-15: "The setting wins, unless the data can't be drawn
that way. Fall back to what the data supports only when the chosen type is
impossible (e.g. a box plot of 2 points), and say so in the figure note."

TWO QUESTIONS, KEPT SEPARATE. `infer_kind` answers "what kind of chart do
these columns imply" -- a question about the DATA. The Default Graph Type
preference answers "which of the forms that fit do I want first". They can
disagree honestly, which is why the preference is layered on top rather than
folded into the inference.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from spacr.qt.widgets import graph_spec as gs                  # noqa: E402


@pytest.fixture
def choose(monkeypatch):
    """Store a Default Graph Type preference the way the app stores one.

    Patched at the PREFERENCE, never at `chosen_for` -- that function holds
    the "a saved choice that does not FIT the data is not a choice for it"
    rule, and replacing it would test a state the application cannot reach.
    """
    def _set(value):
        from spacr.qt import preferences
        monkeypatch.setattr(preferences, "get_default_graph_type",
                            lambda shape: value, raising=False)
    return _set


def _grouped():
    """One categorical against one continuous: the BOX case."""
    spec = gs.GraphSpec(x="gene", y="area")
    return spec, {"gene": gs.CATEGORICAL, "area": gs.CONTINUOUS}


def test_with_no_preference_the_inference_still_decides(choose):
    """THE HALF THAT MUST NOT REGRESS.

    A preference nobody expressed must not move an existing chart.
    """
    choose("")
    spec, kinds = _grouped()
    assert spec.resolved_kind(kinds) == gs.BOX
    assert spec._kind_note(kinds) == ""


def test_the_chosen_type_wins_when_the_builder_can_draw_it(choose):
    """The whole ask: the setting decides what is drawn first."""
    choose("violin")
    spec, kinds = _grouped()
    assert spec.resolved_kind(kinds) == gs.VIOLIN, (
        "the user chose violin for grouped data and got something else")
    assert spec._kind_note(kinds) == "", (
        "the choice was honoured, so there is nothing to explain")


def test_a_choice_this_builder_cannot_draw_falls_back_and_says_so(choose):
    """"Fall back ... and say so in the figure note", exactly.

    `box_jitter` FITS categorical_continuous -- it is a legitimate answer for
    this data -- but the Graph Builder has no such kind. That is the case the
    maintainer's "unless the data can't be drawn that way" covers, and the
    note is what stops it being a silently ignored setting.
    """
    choose("box_jitter")
    spec, kinds = _grouped()
    assert spec.resolved_kind(kinds) == gs.BOX
    note = spec._kind_note(kinds)
    assert note, "the preference was dropped without a word"
    assert "not a chart this builder draws" in note
    assert note in spec.describe(kinds), (
        "the reason never reached the caption, which is the one line the "
        "user actually reads")


def test_a_choice_that_does_not_fit_the_shape_is_not_a_choice_for_it(choose):
    """`chosen_for` drops it before this layer sees it.

    A scatter of grouped data is a different graph of different data. It is
    refused upstream, so there is nothing to explain and no note.
    """
    choose("scatter")
    spec, kinds = _grouped()
    assert spec.resolved_kind(kinds) == gs.BOX
    assert spec._kind_note(kinds) == ""


def test_an_explicit_pin_still_beats_the_setting(choose):
    """A pin is the user asking for THIS chart, now.

    The preference is a standing answer; a pin is a present one. If the
    setting could override a pin, right-clicking to change a chart would stop
    working the moment somebody set a default.
    """
    choose("violin")
    _spec, kinds = _grouped()
    # GraphSpec is frozen: a pin is constructed, never assigned.
    spec = gs.GraphSpec(x="gene", y="area", kind=gs.BAR)
    assert spec.resolved_kind(kinds) == gs.BAR
    assert spec._kind_note(kinds) == ""
    assert "(pinned)" in spec.describe(kinds)


def test_the_shapes_without_a_preference_are_left_alone(choose):
    """A bar of counts and a contingency heatmap are not a choice of form.

    The setting names which of several pictures OF THE SAME DATA to draw.
    Two categorical columns give a contingency count; there is no second
    picture to prefer, so the preference must not reach it.
    """
    choose("violin")
    spec = gs.GraphSpec(x="gene", y="plate")
    kinds = {"gene": gs.CATEGORICAL, "plate": gs.CATEGORICAL}
    assert spec.resolved_kind(kinds) == gs.HEATMAP
    assert spec._kind_note(kinds) == ""

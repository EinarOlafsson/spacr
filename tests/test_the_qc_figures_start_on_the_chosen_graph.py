"""Instruction 293, for the three regression QC figures.

"In settings there is a DEFAULT GRAPH TYPE ... and it decides which graph is
drawn FIRST" -- and, said twice in the same file, "THE SAME BEHAVIOUR FOR
EVERY GRAPH IN SPACR, not only Regression".

`spacr/ml.py` drew its three QC figures as a hardcoded `'jitter_bar'`. 293's
own WHAT STILL DOES NOT list named those three lines, together with the
`settings.py` pipeline defaults and the Graph Builder's `infer_kind`, and said
each is "a one-line change once its owner points it at
`graph_types.start_for`".

THE TRAP IN THAT ONE LINE is that the two halves of spaCR do not spell the
graph types the same way: `graph_types` stores `bar_jitter` and
`spacr.plot.spacrGraph` draws `jitter_bar`. A substitution that skips
`mark_for` hands matplotlib a name its own error message lists as unknown.
"""
from __future__ import annotations

import pytest

from spacr import graph_types
from spacr.ml import qc_graph_type_and_note


def _choose(monkeypatch, value):
    """Store ``value`` as the user's preference, the way the app stores it.

    PATCHED AT THE PREFERENCE, NOT AT `chosen_for`. An earlier version of
    this file replaced `chosen_for` itself -- which is where the
    "a saved choice that does not fit the data is not a choice for it"
    rule lives, so patching it out made the last test below assert that an
    impossible state produced a wrong answer. It reported a product defect
    that was entirely the test's own doing.
    """
    from spacr.qt import preferences

    monkeypatch.setattr(preferences, "get_default_graph_type",
                        lambda shape: value, raising=False)

#: What `spacrGraph` will actually draw, from its own error message.
DRAWABLE = {"bar", "violin", "jitter", "box", "jitter_box", "jitter_bar",
            "line"}


def test_a_user_who_chose_nothing_sees_what_they_saw_before(monkeypatch):
    """THE HALF THAT MUST NOT REGRESS.

    A preference nobody expressed must not move an existing view. These
    figures were `jitter_bar` before the setting was wired in, so with
    nothing chosen they must still be `jitter_bar` -- not the table's
    `box_jitter` default, which is a different figure.
    """
    _choose(monkeypatch, "")
    graph_type, note = qc_graph_type_and_note()
    assert graph_type == "jitter_bar", (
        f"with no preference stored the QC figures became {graph_type!r}; "
        f"they were 'jitter_bar' before 293 was wired in and a preference "
        f"nobody expressed must not move an existing view")
    assert note == "", "nothing was overridden, so there is nothing to say"


@pytest.mark.parametrize("chosen,drawn", [
    ("violin", "violin"),
    ("box", "box"),
    ("bar", "bar"),
    ("jitter", "jitter"),
    ("box_jitter", "jitter_box"),
])
def test_the_chosen_type_reaches_the_figures(monkeypatch, chosen, drawn):
    """The setting decides what is drawn FIRST, which is the whole ask.

    `box_jitter -> jitter_box` is the case that matters: it is the pair whose
    spelling differs between the two halves, so it fails loudly if `mark_for`
    is ever dropped from the path.
    """
    _choose(monkeypatch, chosen)
    graph_type, _note = qc_graph_type_and_note()
    assert graph_type == drawn, (
        f"the user chose {chosen!r} and the QC figures drew {graph_type!r}")


def test_every_type_that_fits_this_shape_is_one_spacrgraph_can_draw():
    """PROOF THE TRANSLATION IS COMPLETE, not merely present.

    A `mark_for` that returned its fallback for half the table would satisfy
    every test above -- the fallback IS a drawable name -- and would quietly
    ignore five of the six choices a user can make for this shape.
    """
    for graph_type in graph_types.FITS["categorical_continuous"]:
        mark = graph_types.mark_for(graph_type, fallback="jitter_bar")
        assert mark in DRAWABLE, (
            f"{graph_type!r} fits this data shape but translates to "
            f"{mark!r}, which spacrGraph does not draw")
        assert mark != "jitter_bar" or graph_type == "bar_jitter", (
            f"{graph_type!r} fell through to the fallback instead of being "
            f"translated; mark_for has no entry for it")


def test_a_choice_this_shape_cannot_take_is_not_a_choice_for_it(monkeypatch):
    """A scatter of grouped data is a different graph of different data.

    `chosen_for` drops a saved choice that does not FIT the shape, and says
    so in its own docstring, so the QC figures keep their own starting form
    rather than being handed `points` -- which `spacrGraph` cannot draw at
    all.

    THIS TEST PREVIOUSLY "FOUND" THAT BUG. It did not: it had replaced
    `chosen_for`, which is where the fit rule lives, so it asserted about a
    state the application cannot reach. Patching the PREFERENCE instead
    exercises the real rule, and the real rule is correct.
    """
    _choose(monkeypatch, "scatter")
    graph_type, note = qc_graph_type_and_note()
    assert graph_type == "jitter_bar", (
        f"a saved 'scatter' does not fit categorical_continuous, so the "
        f"figures should keep their own form; got {graph_type!r}")
    assert graph_type in DRAWABLE
    assert note == "", (
        "nothing was overridden -- the choice was never applicable, which "
        "chosen_for handles before start_for sees it")

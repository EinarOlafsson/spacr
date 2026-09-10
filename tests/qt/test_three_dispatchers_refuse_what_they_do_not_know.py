"""Three if-chains that end in a fallthrough, driven with something
outside the set they handle.

Instruction 288. Each was marked ``# pragma: no cover - every kind /
node type / token is handled above``, and each reason is true for the
values the application supplies. None of the three functions is private
to its caller, though: a kind, a node and a key token all arrive from
somewhere, and the fallthrough is what happens when that somewhere grows
a value the chain has not been taught.

Driving them costs one call each and turns "we believe nothing else can
arrive" into "and if it does, here is what happens".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")


# ---------------------------------------------------------------------------
# formula.evaluate: a node type the walker does not know
# ---------------------------------------------------------------------------

def test_a_node_type_the_walker_cannot_evaluate_is_named():
    """THE ARM, and the message matters: a formula error the user sees
    has to say what it could not do, not merely that it failed."""
    from spacr.qt.widgets.formula import FormulaError, evaluate

    class _Unknown:
        """A node the parser does not produce."""

    frame = pd.DataFrame({"a": [1.0, 2.0]})

    with pytest.raises(FormulaError) as caught:
        evaluate(_Unknown(), frame)

    assert "_Unknown" in str(caught.value), (
        f"the refusal does not name the node type: {caught.value}")


def test_the_node_types_it_does_know_still_evaluate():
    """So the refusal above is about the unknown type, not about a
    walker that refuses everything."""
    from spacr.qt.widgets.formula import evaluate, parse

    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    result = evaluate(parse("a * 2"), frame)

    assert list(np.asarray(result)) == [2.0, 4.0, 6.0]


# ---------------------------------------------------------------------------
# annotate.handle_key: a token outside the keymap
# ---------------------------------------------------------------------------

def test_an_unhandled_key_token_is_declined_not_swallowed(qtbot,
                                                          monkeypatch):
    """THE ARM. `handle_key` returns whether it CONSUMED the key, so an
    unknown token must return False -- returning True would swallow a
    key the window still wants."""
    from spacr.qt.screens import annotate as A

    # `key_token` is a MODULE-LEVEL function, not a method -- patching an
    # attribute on the screen leaves handle_key calling the real one, and
    # the fallthrough is never reached. The coverage JSON is what said so.
    monkeypatch.setattr(A, "key_token", lambda key, text: "not_a_token")

    screen = A.AnnotateScreen.__new__(A.AnnotateScreen)

    assert screen.handle_key(0, "") is False


def test_a_key_with_no_token_at_all_is_also_declined(monkeypatch):
    """The earlier guard, so the two refusals are not confused: an
    unrecognised KEY returns None from key_token and never reaches the
    chain."""
    from spacr.qt.screens import annotate as A

    monkeypatch.setattr(A, "key_token", lambda key, text: None)
    screen = A.AnnotateScreen.__new__(A.AnnotateScreen)

    assert screen.handle_key(0, "") is False


def test_a_token_it_does_know_is_consumed(monkeypatch):
    """So "returns False" is about the unknown token, not about a
    handler that declines everything."""
    from spacr.qt.screens import annotate as A

    monkeypatch.setattr(A, "key_token", lambda key, text: "space")
    screen = A.AnnotateScreen.__new__(A.AnnotateScreen)
    monkeypatch.setattr(type(screen), "_kbd_step",
                        lambda self, delta: True, raising=False)

    assert screen.handle_key(0, "") is True


# ---------------------------------------------------------------------------
# graph_builder._draw_panel_marks: a kind outside the chain
# ---------------------------------------------------------------------------

def test_an_unknown_graph_kind_draws_nothing_and_says_so_by_returning(qtbot):
    """THE ARM. The return value is an updater for a cheap highlight
    repaint, so ``None`` means "nothing to repaint" -- which is the
    honest answer for a kind that drew nothing."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from spacr.qt.widgets.graph_builder import GraphCanvas

    canvas = GraphCanvas()
    qtbot.addWidget(canvas)

    figure, axis = plt.subplots()
    try:
        rows = pd.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0]})

        class _Data:
            strategy = None

        updater = canvas._draw_panel_marks(
            axis, rows, None, "a_kind_that_does_not_exist", _Data(), {})

        assert updater is None
        assert not axis.collections and not axis.lines, (
            "an unknown kind drew something onto the axis")
    finally:
        plt.close(figure)

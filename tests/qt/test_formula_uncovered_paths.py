"""What the formula language says about itself, and about what it refuses.

A token prints as ``kind:text@position`` — the three things every parse
error quotes back at the user, which is why the token knows how to say them
in that order rather than as a dataclass repr.

:func:`~spacr.qt.widgets.formula.evaluate` is typed on ``Node``, the exported
base of the five node types, so a sixth one — a node added to the grammar and
not yet given an evaluation rule — reaches the walker. It has to come back as
a :class:`~spacr.qt.widgets.formula.FormulaError` like every other thing a
formula cannot do; falling off the end of the walk would hand the caller
``None`` and compute a column of nothing.

Two more sentences the user reads: the arity message for a function typed
with empty parentheses, and the notice under a computed column, whose two
halves are different accusations — "the arithmetic produced an infinity" is a
formula to fix, "the input was missing" is a table to look at.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from spacr.qt.widgets import formula
from spacr.qt.widgets.formula import (
    Binary, Column, FormulaError, Node, Number, evaluate, parse, tokenize,
)


@dataclass(frozen=True)
class _Placeholder(Node):
    """A node type the grammar knows and the walker does not."""

    label: str = "unimplemented"


def test_a_token_prints_its_kind_its_text_and_where_it_was_found():
    """The position is the one an error message quotes, counted from zero."""
    printed = [str(token) for token in tokenize("area / 2")]

    assert printed == ["name:area@0", "op:/@5", "number:2@7", "end:@8"]


def test_a_backticked_column_prints_as_the_name_without_its_backticks():
    """The token carries the name the frame is indexed by, not the source."""
    printed = [str(token) for token in tokenize("`cell area`")]

    assert printed == ["name:cell area@0", "end:@11"]


def test_evaluating_a_node_type_with_no_rule_raises_and_names_the_type():
    frame = pd.DataFrame({"area": [1.0, 2.0, 3.0]})

    with pytest.raises(FormulaError) as raised:
        evaluate(_Placeholder(), frame)

    assert "_Placeholder" in str(raised.value)


def test_an_unknown_node_buried_in_an_expression_raises_rather_than_computing():
    """The guard fires inside the recursion, not only at the top of it.

    A silent ``None`` from one operand would be broadcast by the arithmetic
    above it and land in the frame as a column of nothing.
    """
    frame = pd.DataFrame({"area": [1.0, 2.0, 3.0]})
    node = Binary("+", Column("area"), _Placeholder())

    with pytest.raises(FormulaError) as raised:
        evaluate(node, frame)

    assert "_Placeholder" in str(raised.value)


def test_the_five_node_types_the_walker_does_know_still_evaluate():
    """The guard is a floor under the grammar, not a change to it."""
    frame = pd.DataFrame({"area": [1.0, 2.0, 3.0]})

    values = evaluate(parse("-area + mean(area) * 2"), frame)

    assert list(values) == [3.0, 2.0, 1.0]
    assert isinstance(parse("2"), Number)
    assert formula.unparse(parse("area / 2")) == "(area / 2.0)"


def test_a_function_called_with_no_arguments_at_all_is_parsed_then_refused():
    """``mean()`` reaches the arity check rather than the argument loop.

    The empty parentheses are a real thing to type — a user who knows the
    function's name and not its shape — and the message has to be the one
    that says what the function takes, not a syntax error about ``)``.
    """
    with pytest.raises(FormulaError) as raised:
        parse("mean()")

    message = str(raised.value)
    assert "takes 1 argument(s), not 0" in message
    assert "mean(x)" in message


def test_a_three_argument_function_called_empty_reports_its_own_arity():
    with pytest.raises(FormulaError) as raised:
        parse("clip()")

    assert "takes 3 argument(s), not 0" in str(raised.value)


def test_a_column_that_is_only_short_of_inputs_says_so_and_blames_nothing_else():
    """Missing inputs are reported alone when nothing else went wrong.

    The two halves of the notice are different accusations: "the arithmetic
    produced an infinity" is a formula to fix, "the input was missing" is a
    table to look at. A row counted twice, or under the wrong heading, sends
    the reader after the wrong one.
    """
    frame = pd.DataFrame({"area": [1.0, float("nan"), 3.0, float("nan")]})

    _computed, results = formula.compute(
        frame, [formula.ColumnFormula("half", "area / 2")])
    result = results[0]

    assert (result.n_rows, result.n_nonfinite, result.n_input_missing) == (4, 2, 2)
    assert result.notice == (
        "half: 2 of 4 rows have a finite value · 2 had a missing input")


def test_a_column_whose_arithmetic_blew_up_blames_the_calculation_instead():
    """The contrast: no missing inputs, so only the calculation is named."""
    frame = pd.DataFrame({"area": [1.0, 0.0, 3.0], "weight": [1.0, 0.0, 1.0]})

    _computed, results = formula.compute(
        frame, [formula.ColumnFormula("ratio", "area / weight")])
    result = results[0]

    assert result.n_input_missing == 0
    assert "became NaN or infinite in the calculation" in result.notice
    assert "had a missing input" not in result.notice

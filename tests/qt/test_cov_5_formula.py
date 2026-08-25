"""A computed column either means something or says why it does not.

The formula language turns a sentence a user typed into a column the rest of
the analysis treats as data. Two things must never happen: an expression that
is not a formula producing a column anyway, and a column whose values depend
on which rows are in the table failing to say so. These drive the refusals,
the not-enough-data answers, and the "this uses the whole table" flag.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets.formula import (FUNCTIONS, ColumnFormula, FormulaError,
                                      FormulaSet, Node, evaluate, parse,
                                      referenced_columns, tokenize, unparse)


@pytest.fixture()
def frame() -> pd.DataFrame:
    return pd.DataFrame({"area": [10.0, 20.0, 30.0, 40.0],
                         "flag": [True, False, True, False],
                         "gene": ["a", "b", "a", "b"]})


# --------------------------------------------------------------------------- #
#  Reading the text
# --------------------------------------------------------------------------- #

def test_a_backtick_that_never_closes_says_where_it_opened():
    """An unterminated backtick names the position and the right spelling.

    Backticks are how a column with a space in its name is written. The half
    of the formula after the opening tick would otherwise be swallowed into a
    column name that does not exist, and the error would be about the name.
    """
    with pytest.raises(FormulaError, match="unterminated"):
        tokenize("`cell area / 2")


def test_backticks_around_nothing_are_not_a_column():
    """Empty backticks are refused rather than read as a blank name.

    A blank column name matches nothing, so the formula would fail later with
    "there is no column called ''", which does not point at the typo.
    """
    with pytest.raises(FormulaError, match="empty column name"):
        tokenize("`  ` / 2")


# --------------------------------------------------------------------------- #
#  Reading the structure
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("expression, message", [
    ("", "the formula is empty"),
    ("area +", "ends early"),
    ("and area", "needs something before it"),
    ("area not", "already ended before it"),
    (")", r"unexpected '\)'"),
    (",", "unexpected ','"),
    ("mean(area", r"missing '\)'"),
])
def test_an_expression_that_is_not_a_formula_says_what_is_wrong(
        expression, message):
    """Each malformed expression raises with the sentence that fits it.

    The box is edited character by character, so most of what it holds is
    incomplete most of the time. One generic "invalid formula" for all of
    these leaves the user guessing which half of the line is the problem.
    """
    with pytest.raises(FormulaError, match=message):
        parse(expression)


def test_the_error_for_a_truncated_expression_names_what_came_last():
    """"ends early" quotes the token before the end.

    Without it the message says only that something is missing, and in a long
    expression the user has to find the end themselves.
    """
    with pytest.raises(FormulaError, match=r"missing after '\+'"):
        parse("area + ")


def test_a_node_the_printer_does_not_know_is_refused():
    """``unparse`` raises rather than printing an empty string.

    The printed form is what a saved analysis records as the formula. An
    empty string there would round-trip into an unparseable saved figure with
    no sign of what was lost.
    """
    with pytest.raises(FormulaError, match="cannot print"):
        unparse(Node())


def test_a_negated_column_is_still_a_reference_to_that_column(frame):
    """``-area`` reports ``area`` among its referenced columns.

    The reference list decides which columns a computed column needs. Missing
    the operand of a sign would let a formula be evaluated against a table
    that does not have the column at all.
    """
    assert referenced_columns(parse("-area")) == ("area",)
    assert referenced_columns(parse("-(area / mean(area))")) == ("area",)


# --------------------------------------------------------------------------- #
#  Evaluating
# --------------------------------------------------------------------------- #

def test_a_leading_plus_leaves_the_values_alone(frame):
    """Unary ``+`` is the identity, not an error and not a negation.

    People write ``+0.5`` beside ``-0.5`` for symmetry. Rejecting it makes a
    readable formula unusable; getting the sign wrong is worse.
    """
    assert list(evaluate(parse("+area"), frame)) == [10.0, 20.0, 30.0, 40.0]


def test_a_true_false_column_computes_as_one_and_zero(frame):
    """A boolean column is usable arithmetic, valued 1 and 0.

    Annotation and QC columns are booleans. Refusing them would mean a
    fraction of flagged objects could not be written as a formula at all.
    """
    assert list(evaluate(parse("flag + 0"), frame)) == [1.0, 0.0, 1.0, 0.0]


def test_an_aggregate_of_a_bare_number_is_that_number(frame):
    """``mean(3)`` is 3, broadcast over the table rather than an error.

    Aggregates receive a column-length array; a constant argument has to be
    widened to that length or the aggregate reads a zero-dimensional value
    and returns nothing.
    """
    assert evaluate(parse("mean(3)"), frame) == 3.0


def test_a_function_that_cannot_compute_says_which_function_it_was(frame):
    """An error inside a function is reported with the function's name.

    Raw numpy messages ("cannot convert float NaN to integer") name nothing
    in the formula the user typed, and a long expression can hold several
    calls.
    """
    with pytest.raises(FormulaError, match=r"round\(\) could not be computed"):
        evaluate(parse("round(area, log(-1))"), frame)


def test_one_object_has_no_spread_and_says_nan_not_zero():
    """A single row gives NaN from std, var and zscore.

    A spread of 0 reads as "perfectly reproducible", which is the strongest
    possible claim, and it would be made from one measurement.
    """
    one = pd.DataFrame({"area": [5.0]})

    assert np.isnan(evaluate(parse("std(area)"), one))
    assert np.isnan(evaluate(parse("var(area)"), one))
    assert np.isnan(np.asarray(evaluate(parse("zscore(area)"), one))).all()


def test_a_column_of_nothing_measured_aggregates_to_nan():
    """Aggregates over an all-missing column are NaN, never 0.

    ``sum`` of nothing is 0 in numpy, and 0 in a results table is a
    measurement. NaN is the only value that says the column was empty.
    """
    missing = pd.DataFrame({"area": [np.nan, np.nan]})

    assert np.isnan(evaluate(parse("mean(area)"), missing))
    assert np.isnan(evaluate(parse("sum(area)"), missing))
    assert np.isnan(evaluate(parse("quantile(area, 0.5)"), missing))


def test_rank_is_one_based_and_leaves_missing_values_missing(frame):
    """Ranks start at 1 and a missing value gets no rank.

    ``rank(area) / count(area)`` is the percentile, which only works if the
    largest rank equals the count -- a 0-based rank would put the top object
    one place below where it belongs.
    """
    with_gap = pd.DataFrame({"area": [30.0, np.nan, 10.0, 20.0]})

    ranks = evaluate(parse("rank(area)"), with_gap)

    assert list(ranks[[0, 2, 3]]) == [3.0, 1.0, 2.0]
    assert np.isnan(ranks[1])
    assert np.nanmax(ranks) == evaluate(parse("count(area)"), with_gap)


# --------------------------------------------------------------------------- #
#  What a formula says about itself
# --------------------------------------------------------------------------- #

def test_a_formula_says_when_its_values_depend_on_the_other_rows():
    """A table-dependent call anywhere in the expression sets the flag.

    A z-score is not a property of the object: the same formula over two
    plates gives two different columns. The flag is what stops a reader
    treating such a column as a measurement.
    """
    plain = ColumnFormula(name="ratio", expression="abs(area / 2)")
    nested = ColumnFormula(name="scaled", expression="abs(zscore(area))")
    negated = ColumnFormula(name="flip", expression="-mean(area)")

    assert plain.uses_whole_table() is False
    assert nested.uses_whole_table() is True
    assert negated.uses_whole_table() is True
    assert "uses the whole table" in nested.describe()
    assert "uses the whole table" not in plain.describe()


def test_a_set_of_formulas_lists_them_and_can_be_emptied():
    """The description names every formula, and clearing leaves none.

    The description is the record of what a saved analysis added to the
    table; a cleared set has to say "no computed columns" rather than an
    empty string that reads as a missing label.
    """
    formulas = FormulaSet().add(ColumnFormula("ratio", "area / 2"))
    formulas.add(ColumnFormula("scaled", "zscore(area)"))

    described = formulas.describe()
    assert "ratio = area / 2" in described
    assert "scaled = zscore(area)" in described
    assert " · " in described

    assert formulas.clear() is formulas
    assert formulas.names == ()
    assert formulas.describe() == "no computed columns"

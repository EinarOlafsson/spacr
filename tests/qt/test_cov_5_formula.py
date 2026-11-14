"""A computed column either means something or says why it does not.

The formula language turns a sentence a user typed into a column the rest of
the analysis treats as data. Two things must never happen: an expression that
is not a formula producing a column anyway, and a column whose values depend
on which rows are in the table failing to say so. These drive the refusals,
the not-enough-data answers, and the "this uses the whole table" flag.
"""
from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets.formula import (ColumnFormula, FormulaError,
                                      FormulaSet, Node, parse,
                                      referenced_columns, tokenize, unparse)


@pytest.fixture()
def frame() -> pd.DataFrame:
    return pd.DataFrame({"area": [10.0, 20.0, 30.0, 40.0],
                         "flag": [True, False, True, False],
                         "gene": ["a", "b", "a", "b"]})


# --------------------------------------------------------------------------- #
#  Reading the text
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("text, match", [
    # An unterminated backtick names the position and the right spelling.
    # Backticks are how a column with a space in its name is written; the
    # half of the formula after the opening tick would otherwise be swallowed
    # into a column name that does not exist.
    ("`cell area / 2", "unterminated"),
    # Empty backticks are refused rather than read as a blank name, which
    # would fail later with "there is no column called ''".
    ("`  ` / 2", "empty column name"),
], ids=["a_backtick_that_never_closes", "backticks_around_nothing"])
def test_a_backtick_that_never_closes_says_where_it_opened(text, match):
    with pytest.raises(FormulaError, match=match):
        tokenize(text)


# --------------------------------------------------------------------------- #
#  Reading the structure
# --------------------------------------------------------------------------- #


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

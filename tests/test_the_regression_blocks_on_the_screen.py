"""Two screens in one fit means the screen is in the model.

Instruction 122's own acceptance criterion, and the one it was still failing:

    "screenID is available as a blocking factor / fixed effect in the
     regression, not merely as a label."

`screenID` became a real column on every merged frame, and `measurement_scan`
could block on it — but `perform_regression` could not. `prepare_formula`
built `... + rowID + columnID` and never mentioned the screen, so two stacked
screens were fitted as though they were one experiment.

WHY THAT IS NOT COSMETIC. A systematic difference between two experiments
that is not a term in the model does not vanish; it is charged to whichever
guides happen to be over-represented in one of them. The result is a hit that
looks exactly like a real one, in the direction nobody questions, because the
whole point of stacking the screens was to find effects the single screen was
underpowered for.

AND THE OTHER DIRECTION IS AS BAD. A single-screen project's `screenID` has
ONE value. Blocking on a constant makes the design rank-deficient, and
statsmodels answers a rank-deficient design with a pseudo-inverse rather than
refusing — so the run appears to succeed and returns standard errors that mean
nothing, with no error anywhere. That is why the decision is made from the
DATA and not from a setting somebody might tick.
"""
from __future__ import annotations

import pandas as pd
import pytest

from spacr.ml import prepare_formula, screen_is_blockable
from spacr.schema import SCREEN_KEY


def _frame(screens):
    return pd.DataFrame({
        SCREEN_KEY: screens,
        "plateID": ["plate1"] * len(screens),
        "predictions": [0.5] * len(screens),
    })


# --------------------------------------------------------------------------- #
#  When the screen is a term
# --------------------------------------------------------------------------- #

def test_two_screens_put_the_screen_in_the_formula():
    formula = prepare_formula("predictions", block_screen=True)
    assert f"+ {SCREEN_KEY}" in formula, formula


def test_the_mixed_branch_gets_it_too():
    """The random row/column structure absorbs plate, row and column. It does
    not absorb the screen, and a mixed fit over two experiments needs the term
    exactly as much as a fixed one does."""
    formula = prepare_formula("predictions", random_row_column_effects=True,
                              block_screen=True)
    assert f"+ {SCREEN_KEY}" in formula, formula
    assert "rowID" not in formula, "the mixed branch grew a fixed row term"


def test_the_rest_of_the_formula_is_unchanged():
    with_screen = prepare_formula("predictions", block_screen=True)
    without = prepare_formula("predictions", block_screen=False)

    assert with_screen == f"{without} + {SCREEN_KEY}"


# --------------------------------------------------------------------------- #
#  When it must not be
# --------------------------------------------------------------------------- #

def test_it_is_off_by_default():
    """A single-screen project is the normal case and must be untouched."""
    assert SCREEN_KEY not in prepare_formula("predictions")


def test_one_screen_is_not_blockable():
    assert screen_is_blockable(_frame(["screen1"] * 6)) is False


def test_no_screen_column_at_all_is_not_blockable():
    """Every project that predates instruction 122."""
    frame = _frame(["screen1"] * 4).drop(columns=[SCREEN_KEY])
    assert screen_is_blockable(frame) is False


def test_two_screens_are_blockable():
    assert screen_is_blockable(_frame(["A", "A", "B", "B"])) is True


def test_none_is_not_blockable():
    assert screen_is_blockable(None) is False


# --------------------------------------------------------------------------- #
#  The rule is stated once
# --------------------------------------------------------------------------- #

def test_it_agrees_with_the_scan_on_the_same_column():
    """`measurement_scan._dummy_block` drops a one-level block for the same
    reason. If the two disagreed, a frame could be blocked on by the scan and
    not by the regression, and the two would report different effects from
    the same data."""
    from spacr.measurement_scan import _dummy_block

    for screens in (["A"] * 4, ["A", "A", "B", "B"], ["A", "B", "C", "C"]):
        columns, _levels = _dummy_block(screens)
        assert (columns.shape[1] > 0) is screen_is_blockable(_frame(screens)), (
            f"the scan and the regression disagree about {screens}")


# --------------------------------------------------------------------------- #
#  Through patsy, which is what actually builds the design
# --------------------------------------------------------------------------- #

def test_the_screen_term_becomes_real_design_columns():
    """A formula string is not a design. This is the check that the term
    survives into the matrix the model is fitted on."""
    patsy = pytest.importorskip("patsy")
    import numpy as np

    rng = np.random.default_rng(0)
    n = 40
    frame = pd.DataFrame({
        SCREEN_KEY: ["A"] * (n // 2) + ["B"] * (n // 2),
        "rowID": [f"r{i % 4}" for i in range(n)],
        "columnID": [f"c{i % 5}" for i in range(n)],
        "predictions": rng.normal(size=n),
        "fraction": rng.uniform(size=n),
        "gene_fraction": rng.uniform(size=n),
        "grna": [f"g{i % 6}" for i in range(n)],
        "gene": [f"G{i % 3}" for i in range(n)],
    })

    formula = prepare_formula("predictions", block_screen=True)
    _y, X = patsy.dmatrices(formula, data=frame, return_type="dataframe")

    screen_terms = [c for c in X.columns if SCREEN_KEY in c]
    assert len(screen_terms) == 1, (
        f"expected one screen contrast for two screens, got {screen_terms}")
    assert np.linalg.matrix_rank(X.to_numpy()) == X.shape[1], (
        "the design with a screen term is rank-deficient")


def test_a_constant_screen_would_have_been_rank_deficient():
    """The failure the data-driven check exists to prevent, demonstrated --
    so nobody 'simplifies' screen_is_blockable into `SCREEN_KEY in columns`."""
    patsy = pytest.importorskip("patsy")
    import numpy as np

    rng = np.random.default_rng(1)
    n = 30
    frame = pd.DataFrame({
        SCREEN_KEY: ["only_one"] * n,          # a single-screen project
        "rowID": [f"r{i % 3}" for i in range(n)],
        "columnID": [f"c{i % 5}" for i in range(n)],
        "predictions": rng.normal(size=n),
        "fraction": rng.uniform(size=n),
        "gene_fraction": rng.uniform(size=n),
        "grna": [f"g{i % 4}" for i in range(n)],
        "gene": [f"G{i % 2}" for i in range(n)],
    })

    assert screen_is_blockable(frame) is False, (
        "a one-value screen column was judged blockable")

    # And this is why: forced on, the term contributes nothing estimable.
    forced = prepare_formula("predictions", block_screen=True)
    _y, X = patsy.dmatrices(forced, data=frame, return_type="dataframe")
    assert not [c for c in X.columns if SCREEN_KEY in c], (
        "patsy kept a constant screen contrast")

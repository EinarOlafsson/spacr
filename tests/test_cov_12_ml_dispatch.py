"""regression_model refuses the settings each family cannot honour.

Every branch of the dispatcher can be reached from a settings file, so each one
has to fail with a sentence naming the setting rather than with an error from
three frames inside statsmodels or scikit-learn. The families that carry their
own result objects are checked here too: they are what the coefficient table
and the saved summary are read off.
"""
from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest

from spacr import ml
from spacr.regression_spec import UNSUPPORTED_REGRESSION_TYPES


def simple_design(n=30, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    return pd.DataFrame({'Intercept': np.ones(n), 'fraction': x}), x


# ---------------------------------------------------------------------------
# the dispatcher's refusals
# ---------------------------------------------------------------------------

def test_a_regression_type_spacr_removed_says_why_it_is_gone():
    """A retired family is refused with the recorded reason, not "unsupported".

    The name is still valid in an old settings CSV, so the message has to say
    what replaced it rather than list every type spaCR does have.
    """
    name = next(iter(UNSUPPORTED_REGRESSION_TYPES))
    design, x = simple_design()

    with pytest.raises(ValueError) as caught:
        ml.regression_model(design, 0.5 * x, regression_type=name)
    message = str(caught.value)
    assert name in message
    assert UNSUPPORTED_REGRESSION_TYPES[name] in message


def test_weighted_least_squares_refuses_weights_that_are_not_cell_counts():
    """Zero, negative or non-finite weights cannot be per-well cell counts.

    A zero weight drops a well from the fit while leaving it in the row count,
    so the reported n and the fitted n disagree with nothing saying so.
    """
    design, x = simple_design()
    weights = np.ones(len(design))
    weights[2] = -1.0

    with pytest.raises(ValueError, match='finite and positive'):
        ml.regression_model(design, 0.5 * x, regression_type='wls',
                            weights=weights)


def test_the_hinge_refuses_a_penalty_that_is_not_positive():
    """``alpha`` is the inverse margin, so zero or negative has no meaning.

    scikit-learn's own error for C <= 0 names an argument spaCR never exposes;
    this one names the setting the user actually set.
    """
    design, x = simple_design()
    y = (x > 0).astype(float)

    with pytest.raises(ValueError, match='alpha must be positive for hinge'):
        ml.regression_model(design, y, regression_type='hinge', alpha=0.0,
                            hinge_threshold=0.5)


def test_an_automatic_hinge_penalty_needs_two_wells_in_each_class(capsys):
    """With a minority class too small to split, the default penalty is used.

    A two-fold cross-validation on one positive well chooses a penalty from
    noise, which is worse than not choosing one -- and it is said out loud.
    """
    rng = np.random.default_rng(4)
    n = 20
    design = pd.DataFrame({'Intercept': np.ones(n),
                           'fraction': rng.normal(size=n)})
    y = np.zeros(n)
    y[0] = 1.0                       # exactly one well in the minority class

    model = ml.regression_model(design, y, regression_type='hinge',
                                alpha='auto', hinge_threshold=0.5)

    printed = capsys.readouterr().out
    assert "alpha='auto' needs at least two wells in each class" in printed
    assert 'falling back to alpha=1' in printed
    assert model.coef_.shape[-1] == design.shape[1]


def test_rra_refuses_a_design_whose_guide_columns_never_vary():
    """With every guide column constant there is no marginal effect to rank.

    Ranking constants would give each guide a rank it did not earn and shift
    every real guide's, so the fit refuses and names the fraction threshold.
    """
    n = 20
    design = pd.DataFrame({
        'Intercept': np.ones(n),
        'fraction:grna[g1]': np.ones(n),
        'fraction:grna[g2]': np.full(n, 2.0),
    })
    rng = np.random.default_rng(6)

    with pytest.raises(ValueError, match='ranked no guide'):
        ml.regression_model(design, rng.normal(size=n), regression_type='rra',
                            rra_permutations=50)


# ---------------------------------------------------------------------------
# the horseshoe fit's inputs and its result object
# ---------------------------------------------------------------------------

def horseshoe_inputs(n=20, seed=8):
    rng = np.random.default_rng(seed)
    design = pd.DataFrame({'Intercept': np.ones(n),
                           'fraction': rng.uniform(0.1, 0.9, n)})
    counts = rng.integers(1, 10, n).astype(float)
    exposure = np.full(n, 100.0)
    return design, counts, exposure


def test_the_horseshoe_exposure_must_line_up_with_the_response():
    """One cell count per well, and the same wells, or the offset is wrong.

    log(Ntotal) is the offset that makes the coefficients rates rather than
    headcounts; a recycled exposure puts one well's offset on another.
    """
    design, counts, _ = horseshoe_inputs()
    with pytest.raises(ValueError, match='horseshoe exposure has 3 entries'):
        ml._fit_horseshoe_poisson(design, counts, np.array([1.0, 2.0, 3.0]))


def test_the_horseshoe_exposure_must_be_a_positive_cell_count():
    """A zero or negative exposure has no logarithm and is refused.

    Without this the offset becomes -inf and the well silently leaves the fit.
    """
    design, counts, exposure = horseshoe_inputs()
    exposure = exposure.copy()
    exposure[1] = 0.0
    with pytest.raises(ValueError, match='must be finite and.*positive'):
        ml._fit_horseshoe_poisson(design, counts, exposure)


def test_a_well_cannot_hold_more_positives_than_it_holds_cells():
    """Npositive above Ntotal is refused, naming how many wells break it.

    The response counts positive objects out of the imaged ones; more positives
    than cells means the two columns are not what the model thinks they are.
    """
    design, counts, exposure = horseshoe_inputs()
    exposure = exposure.copy()
    exposure[0] = 1.0
    counts = counts.copy()
    counts[0] = 9.0
    with pytest.raises(ValueError, match='Npositive <= Ntotal'):
        ml._fit_horseshoe_poisson(design, counts, exposure)


def test_a_posterior_summary_missing_a_column_stops_the_run_by_name():
    """A renamed power_model column is named here, not surfaced as a KeyError.

    power_model has its own release cadence; the coefficient table needs those
    five columns and cannot guess a replacement for one that moved.
    """
    estimates = pd.DataFrame({'gene': ['g1'], 'mean': [0.5], 'sd': [0.1]})
    with pytest.raises(ValueError) as caught:
        ml._HorseshoeResults(fit=None, estimates=estimates)
    message = str(caught.value)
    assert "'prob_positive'" in message and "'identified'" in message


def test_a_horseshoe_fit_that_did_not_converge_says_so(capsys):
    """A fit below its own convergence criterion is reported as provisional.

    The coefficients still exist and still get written; a silent unconverged
    posterior is one nobody knows to re-run.
    """
    estimates = pd.DataFrame({
        'gene': ['g1', 'g2'],
        'mean': [0.5, -0.2],
        'sd': [0.1, 0.2],
        'prob_positive': [0.99, 0.10],
        'identified': [True, True],
    })
    results = ml._HorseshoeResults(
        fit=types.SimpleNamespace(converged=False), estimates=estimates)

    assert 'did not meet its own convergence criterion' in \
        capsys.readouterr().out
    assert results.converged is False
    assert list(results.params.index) == ['g1', 'g2']
    assert results.pvalues['g1'] == pytest.approx(0.02)


def test_the_rra_summary_is_the_per_gene_table():
    """``summary()`` hands back the gene table so the run can save it.

    RRA's answer IS the per-gene aggregation; a coefficient dump would lose the
    rho and the direction split that the method exists to produce.
    """
    genes = pd.DataFrame({'gene': ['g1'], 'rho_neg': [0.01], 'rho_pos': [0.9]})
    results = ml._RRAResults(scores=[0.4, -0.1],
                             p_values=[0.02, float('nan')],
                             index=['fraction:grna[g1_1]', 'Intercept'],
                             genes=genes)
    assert results.summary() is genes
    assert list(results.params.index) == ['fraction:grna[g1_1]', 'Intercept']

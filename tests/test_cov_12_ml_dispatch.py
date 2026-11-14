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

def case_a_regression_type_spacr_removed_says_why_it_is_gone():
    """A retired family is refused with the recorded reason, not "unsupported".

    The name is still valid in an old settings CSV, so the message has to say
    what replaced it rather than list every type spaCR does have.
    """
    _seen = []
    name = next(iter(UNSUPPORTED_REGRESSION_TYPES))
    design, x = simple_design()

    with pytest.raises(ValueError) as caught:
        ml.regression_model(design, 0.5 * x, regression_type=name)
    _seen.append(('ValueError' if isinstance(caught.value, ValueError) else type(caught.value).__name__))
    message = str(caught.value)
    _seen.append(bool(
        name in message))
    _seen.append(bool(
        UNSUPPORTED_REGRESSION_TYPES[name] in message))
    return _seen


def case_weighted_least_squares_refuses_weights_that_are_not_cell_counts():
    """Zero, negative or non-finite weights cannot be per-well cell counts.

    A zero weight drops a well from the fit while leaving it in the row count,
    so the reported n and the fitted n disagree with nothing saying so.
    """
    _seen = []
    design, x = simple_design()
    weights = np.ones(len(design))
    weights[2] = -1.0

    with pytest.raises(ValueError, match='finite and positive') as _exc1:
        ml.regression_model(design, 0.5 * x, regression_type='wls',
                            weights=weights)
    _seen.append(('ValueError' if isinstance(_exc1.value, ValueError) else type(_exc1.value).__name__))
    return _seen


def case_the_hinge_refuses_a_penalty_that_is_not_positive():
    """``alpha`` is the inverse margin, so zero or negative has no meaning.

    scikit-learn's own error for C <= 0 names an argument spaCR never exposes;
    this one names the setting the user actually set.
    """
    _seen = []
    design, x = simple_design()
    y = (x > 0).astype(float)

    with pytest.raises(ValueError, match='alpha must be positive for hinge') as _exc1:
        ml.regression_model(design, y, regression_type='hinge', alpha=0.0,
                            hinge_threshold=0.5)
    _seen.append(('ValueError' if isinstance(_exc1.value, ValueError) else type(_exc1.value).__name__))
    return _seen


def case_an_automatic_hinge_penalty_needs_two_wells_in_each_class(capsys):
    """With a minority class too small to split, the default penalty is used.

    A two-fold cross-validation on one positive well chooses a penalty from
    noise, which is worse than not choosing one -- and it is said out loud.
    """
    _seen = []
    rng = np.random.default_rng(4)
    n = 20
    design = pd.DataFrame({'Intercept': np.ones(n),
                           'fraction': rng.normal(size=n)})
    y = np.zeros(n)
    y[0] = 1.0                       # exactly one well in the minority class

    model = ml.regression_model(design, y, regression_type='hinge',
                                alpha='auto', hinge_threshold=0.5)

    printed = capsys.readouterr().out
    _seen.append(("alpha='auto' needs at least two wells in each class" if "alpha='auto' needs at least two wells in each class" in (printed) else ("missing", "alpha='auto' needs at least two wells in each class")))
    _seen.append(('falling back to alpha=1' if 'falling back to alpha=1' in (printed) else ("missing", 'falling back to alpha=1')))
    _seen.append(bool(
        model.coef_.shape[-1] == design.shape[1]))
    return _seen


def case_rra_refuses_a_design_whose_guide_columns_never_vary():
    """With every guide column constant there is no marginal effect to rank.

    Ranking constants would give each guide a rank it did not earn and shift
    every real guide's, so the fit refuses and names the fraction threshold.
    """
    _seen = []
    n = 20
    design = pd.DataFrame({
        'Intercept': np.ones(n),
        'fraction:grna[g1]': np.ones(n),
        'fraction:grna[g2]': np.full(n, 2.0),
    })
    rng = np.random.default_rng(6)

    with pytest.raises(ValueError, match='ranked no guide') as _exc1:
        ml.regression_model(design, rng.normal(size=n), regression_type='rra',
                            rra_permutations=50)
    _seen.append(('ValueError' if isinstance(_exc1.value, ValueError) else type(_exc1.value).__name__))
    return _seen


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


def case_the_horseshoe_exposure_must_line_up_with_the_response():
    """One cell count per well, and the same wells, or the offset is wrong.

    log(Ntotal) is the offset that makes the coefficients rates rather than
    headcounts; a recycled exposure puts one well's offset on another.
    """
    _seen = []
    design, counts, _ = horseshoe_inputs()
    with pytest.raises(ValueError, match='horseshoe exposure has 3 entries') as _exc1:
        ml._fit_horseshoe_poisson(design, counts, np.array([1.0, 2.0, 3.0]))
    _seen.append(('ValueError' if isinstance(_exc1.value, ValueError) else type(_exc1.value).__name__))
    return _seen


def case_the_horseshoe_exposure_must_be_a_positive_cell_count():
    """A zero or negative exposure has no logarithm and is refused.

    Without this the offset becomes -inf and the well silently leaves the fit.
    """
    _seen = []
    design, counts, exposure = horseshoe_inputs()
    exposure = exposure.copy()
    exposure[1] = 0.0
    with pytest.raises(ValueError, match='must be finite and.*positive') as _exc1:
        ml._fit_horseshoe_poisson(design, counts, exposure)
    _seen.append(('ValueError' if isinstance(_exc1.value, ValueError) else type(_exc1.value).__name__))
    return _seen


def case_a_well_cannot_hold_more_positives_than_it_holds_cells():
    """Npositive above Ntotal is refused, naming how many wells break it.

    The response counts positive objects out of the imaged ones; more positives
    than cells means the two columns are not what the model thinks they are.
    """
    _seen = []
    design, counts, exposure = horseshoe_inputs()
    exposure = exposure.copy()
    exposure[0] = 1.0
    counts = counts.copy()
    counts[0] = 9.0
    with pytest.raises(ValueError, match='Npositive <= Ntotal') as _exc1:
        ml._fit_horseshoe_poisson(design, counts, exposure)
    _seen.append(('ValueError' if isinstance(_exc1.value, ValueError) else type(_exc1.value).__name__))
    return _seen


def case_a_posterior_summary_missing_a_column_stops_the_run_by_name():
    """A renamed power_model column is named here, not surfaced as a KeyError.

    power_model has its own release cadence; the coefficient table needs those
    five columns and cannot guess a replacement for one that moved.
    """
    _seen = []
    estimates = pd.DataFrame({'gene': ['g1'], 'mean': [0.5], 'sd': [0.1]})
    with pytest.raises(ValueError) as caught:
        ml._HorseshoeResults(fit=None, estimates=estimates)
    _seen.append(('ValueError' if isinstance(caught.value, ValueError) else type(caught.value).__name__))
    message = str(caught.value)
    _seen.append(bool(
        "'prob_positive'" in message and "'identified'" in message))
    return _seen


def case_a_horseshoe_fit_that_did_not_converge_says_so(capsys):
    """A fit below its own convergence criterion is reported as provisional.

    The coefficients still exist and still get written; a silent unconverged
    posterior is one nobody knows to re-run.
    """
    _seen = []
    estimates = pd.DataFrame({
        'gene': ['g1', 'g2'],
        'mean': [0.5, -0.2],
        'sd': [0.1, 0.2],
        'prob_positive': [0.99, 0.10],
        'identified': [True, True],
    })
    results = ml._HorseshoeResults(
        fit=types.SimpleNamespace(converged=False), estimates=estimates)

    _seen.append(('did not meet its own convergence criterion' if 'did not meet its own convergence criterion' in (capsys.readouterr().out) else ("missing", 'did not meet its own convergence criterion')))
    _seen.append(bool(
        results.converged is False))
    _seen.append(keep(list(results.params.index)))
    _seen.append(bool(
        results.pvalues['g1'] == pytest.approx(0.02)))
    _seen.append(bool(
        results.summary() is estimates))
    return _seen



def case_a_converged_horseshoe_result_retains_its_posterior_without_warning(capsys):
    """Propagate the supplied fit flag; this fixture does not certify a sampler."""
    _seen = []
    from spacr.power_model import PowerFit, gather_model_estimate

    draws = np.array([[.2, -.5, np.nan], [.4, -.2, np.nan],
                      [-.1, .1, np.nan], [.3, -.4, np.nan]])
    fit = PowerFit(
        backend="numpyro", requested_backend="numpyro", method="nuts",
        draws=draws, intercept_draws=np.zeros(4),
        genes=np.array(["positive", "negative", "Intercept"]), converged=True,
    )
    estimates = gather_model_estimate(fit)
    results = ml._HorseshoeResults(fit, estimates)

    _seen.append(bool(
        results.converged is True))
    _seen.append(bool(
        results.fit is fit))
    _seen.append(keep(list(results.params.index)))
    np.testing.assert_allclose(results.params.to_numpy(), draws[:, :2].mean(axis=0))
    np.testing.assert_allclose(results.bse.to_numpy(), draws[:, :2].std(axis=0, ddof=1))
    _seen.append(keep(results.pvalues.to_dict()))
    _seen.append(bool(
        results.summary() is estimates))
    _seen.append(keep(estimates["identified"].tolist()))
    _seen.append(keep(capsys.readouterr().out))
    return _seen


def case_a_horseshoe_result_carries_posterior_mean_counts_for_qc():
    """The QC report reads fitted values; the adapter gives the posterior mean.

    ``fittedvalues`` is E[Ntotal * exp(intercept + standardised X @ beta)]
    over the draws, on the count scale the response is on, and
    ``resid_pearson`` divides by sqrt(mu) because the likelihood is Poisson.
    Without these the regression QC report refused the horseshoe fit.
    """
    _seen = []
    from spacr import regression_qc as rq
    from spacr.power_model import ModelData, PowerFit, gather_model_estimate

    design = np.array([[1.0, 0.0], [1.0, 0.5], [1.0, 1.0], [1.0, 0.25]])
    counts = np.array([2.0, 4.0, 9.0, 3.0])
    exposure = np.array([10.0, 12.0, 20.0, 8.0])
    data = ModelData(wells=np.arange(4), genes=np.array(["Intercept", "g1"], dtype=object),
                     Npositive=counts, Ntotal=exposure, log10expression=design,
                     unidentified_genes=("Intercept",))
    means = design.mean(axis=0)
    scales = np.where(design.std(axis=0) > 0, design.std(axis=0), 1.0)
    draws = np.array([[np.nan, 0.3], [np.nan, 0.5], [np.nan, 0.4]])
    intercepts = np.array([-1.2, -1.0, -1.1])
    fit = PowerFit(backend="numpyro", requested_backend="numpyro", method="nuts",
                   draws=draws, intercept_draws=intercepts, genes=data.genes,
                   converged=True, data=data,
                   diagnostics={"column_means": means, "column_scales": scales})
    results = ml._HorseshoeResults(fit, gather_model_estimate(fit))

    standardised = (design[:, 1] - means[1]) / scales[1]
    expected = np.mean([exposure * np.exp(b0 + b * standardised)
                        for b0, b in zip(intercepts, draws[:, 1])], axis=0)
    np.testing.assert_allclose(results.fittedvalues, expected)
    np.testing.assert_allclose(results.resid_pearson,
                               (counts - expected) / np.sqrt(expected))

    X = pd.DataFrame(design, columns=["Intercept", "g1"])
    ctx = rq.build_context(results, X, counts, regression_type="horseshoe")
    np.testing.assert_allclose(ctx.fitted, expected)
    _seen.append(keep((ctx.family, ctx.link)))
    _seen.append(bool(
        ctx.standardisation_available))
    return _seen


def case_the_rra_summary_is_the_per_gene_table():
    """``summary()`` hands back the gene table so the run can save it.

    RRA's answer IS the per-gene aggregation; a coefficient dump would lose the
    rho and the direction split that the method exists to produce.
    """
    _seen = []
    genes = pd.DataFrame({'gene': ['g1'], 'rho_neg': [0.01], 'rho_pos': [0.9]})
    results = ml._RRAResults(scores=[0.4, -0.1],
                             p_values=[0.02, float('nan')],
                             index=['fraction:grna[g1_1]', 'Intercept'],
                             genes=genes)
    _seen.append(bool(
        results.summary() is genes))
    _seen.append(keep(list(results.params.index)))
    return _seen


from tests.observed_case import keep, observe  # noqa: E402

@pytest.mark.parametrize(("case", "expected"), [
    pytest.param(case_a_regression_type_spacr_removed_says_why_it_is_gone, ['ValueError', True, True], id="a_regression_type_spacr_removed_says_why_it_is_gone"),
    pytest.param(case_weighted_least_squares_refuses_weights_that_are_not_cell_counts, ['ValueError'], id="weighted_least_squares_refuses_weights_that_are_not_cell_counts"),
    pytest.param(case_the_hinge_refuses_a_penalty_that_is_not_positive, ['ValueError'], id="the_hinge_refuses_a_penalty_that_is_not_positive"),
    pytest.param(case_an_automatic_hinge_penalty_needs_two_wells_in_each_class, ["alpha='auto' needs at least two wells in each class", 'falling back to alpha=1', True], id="an_automatic_hinge_penalty_needs_two_wells_in_each_class"),
    pytest.param(case_rra_refuses_a_design_whose_guide_columns_never_vary, ['ValueError'], id="rra_refuses_a_design_whose_guide_columns_never_vary"),
    pytest.param(case_the_horseshoe_exposure_must_line_up_with_the_response, ['ValueError'], id="the_horseshoe_exposure_must_line_up_with_the_response"),
    pytest.param(case_the_horseshoe_exposure_must_be_a_positive_cell_count, ['ValueError'], id="the_horseshoe_exposure_must_be_a_positive_cell_count"),
    pytest.param(case_a_well_cannot_hold_more_positives_than_it_holds_cells, ['ValueError'], id="a_well_cannot_hold_more_positives_than_it_holds_cells"),
    pytest.param(case_a_posterior_summary_missing_a_column_stops_the_run_by_name, ['ValueError', True], id="a_posterior_summary_missing_a_column_stops_the_run_by_name"),
    pytest.param(case_a_horseshoe_fit_that_did_not_converge_says_so, ['did not meet its own convergence criterion', True, ['g1', 'g2'], True, True], id="a_horseshoe_fit_that_did_not_converge_says_so"),
    pytest.param(case_a_converged_horseshoe_result_retains_its_posterior_without_warning, [True, True, ['positive', 'negative'], {'positive': 0.5, 'negative': 0.5}, True, [True, True, False], ''], id="a_converged_horseshoe_result_retains_its_posterior_without_warning"),
    pytest.param(case_a_horseshoe_result_carries_posterior_mean_counts_for_qc, [('Poisson', 'Log'), True], id="a_horseshoe_result_carries_posterior_mean_counts_for_qc"),
    pytest.param(case_the_rra_summary_is_the_per_gene_table, [True, ['fraction:grna[g1_1]', 'Intercept']], id="the_rra_summary_is_the_per_gene_table"),
])
def test_12_ml_dispatch_contracts(request, case, expected):
    """One contract per row: the case records what it checked, the row states it."""
    observed = observe(case, request)
    assert observed == expected

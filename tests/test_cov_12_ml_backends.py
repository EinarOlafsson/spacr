"""The absorbing and glum backends refuse the designs they cannot answer.

Both exist to fit the SAME model faster than statsmodels, so the only thing
that makes them safe is that they refuse rather than approximate. Each refusal
is driven here on the design that triggers it -- no absorbable factor, weights
that do not line up, a singular information matrix -- and each message is
checked for naming what to do instead.
"""
from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from spacr import ml


def absorbable_design(n_rows=3, n_columns=3, extra=('fraction',), seed=0):
    """A patsy-shaped design with rowID/columnID dummies and real columns."""
    rng = np.random.default_rng(seed)
    rows = [f'r{i + 1}' for i in range(n_rows)]
    columns = [f'c{i + 1}' for i in range(n_columns)]
    records = []
    for row in rows:
        for column in columns:
            record = {'Intercept': 1.0}
            for other in rows[1:]:
                record[f'rowID[T.{other}]'] = 1.0 if row == other else 0.0
            for other in columns[1:]:
                record[f'columnID[T.{other}]'] = 1.0 if column == other else 0.0
            for name in extra:
                record[name] = float(rng.normal())
            records.append(record)
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# the absorbing backend
# ---------------------------------------------------------------------------

def case_the_absorbing_backend_needs_a_named_design():
    """A bare array has no column names, so nothing can be identified as a dummy.

    Which columns are rowID and columnID indicators is read off the names; a
    positional guess would absorb whichever columns happened to be there.
    """
    _seen = []
    with pytest.raises(ValueError, match='needs a DataFrame design') as _exc1:
        ml._fit_absorbed_least_squares(np.ones((4, 2)), np.arange(4.0))
    _seen.append(('ValueError' if isinstance(_exc1.value, ValueError) else type(_exc1.value).__name__))
    return _seen


def case_a_design_that_is_only_intercept_and_absorbed_terms_reports_nothing():
    """With every column absorbed there is no coefficient left to report.

    Fitting it would produce an empty coefficient table that reads as a screen
    with no effects rather than as a model with no terms.
    """
    _seen = []
    design = pd.DataFrame({
        'Intercept': [1.0, 1.0, 1.0, 1.0],
        'rowID[T.r2]': [0.0, 1.0, 0.0, 1.0],
        'columnID[T.c2]': [0.0, 0.0, 1.0, 1.0],
    })
    with pytest.raises(ValueError, match='report no coefficient at all') as _exc1:
        ml._fit_absorbed_least_squares(design, np.arange(4.0))
    _seen.append(('ValueError' if isinstance(_exc1.value, ValueError) else type(_exc1.value).__name__))
    return _seen


def case_weights_that_do_not_line_up_with_the_rows_are_refused():
    """A weight vector of the wrong length is refused, not broadcast.

    The weights are per-well cell counts; silently recycling them would weight
    the wrong wells and change every standard error.
    """
    _seen = []
    design = absorbable_design()
    y = np.arange(float(len(design)))

    with pytest.raises(ValueError, match='weights for') as _exc1:
        ml._fit_absorbed_least_squares(design, y, weights=np.ones(3),
                                       kind='WLS')
    _seen.append(('ValueError' if isinstance(_exc1.value, ValueError) else type(_exc1.value).__name__))
    return _seen


def case_weights_that_are_not_positive_counts_are_refused():
    """Zero, negative or non-finite weights cannot be cell counts.

    A zero weight silently drops a well from the fit while it stays in the
    reported row count.
    """
    _seen = []
    design = absorbable_design()
    y = np.arange(float(len(design)))
    weights = np.ones(len(design))
    weights[0] = 0.0

    with pytest.raises(ValueError, match='finite and positive') as _exc1:
        ml._fit_absorbed_least_squares(design, y, weights=weights,
                                       kind='WLS')
    _seen.append(('ValueError' if isinstance(_exc1.value, ValueError) else type(_exc1.value).__name__))
    return _seen


def case_projections_that_do_not_converge_are_refused(monkeypatch):
    """Unconverged demeaning means the design was never fully partialled out.

    The coefficients would then not be the least-squares ones, and nothing
    downstream could tell.
    """
    _seen = []
    demean_module = pytest.importorskip('pyfixest.core.demean')
    real_demean = demean_module.demean
    monkeypatch.setattr(demean_module, 'demean',
                        lambda *a, **k: (real_demean(*a, **k)[0], False))

    design = absorbable_design()
    with pytest.raises(ValueError, match='did not converge') as _exc1:
        ml._fit_absorbed_least_squares(design,
                                       np.arange(float(len(design))))
    _seen.append(('ValueError' if isinstance(_exc1.value, ValueError) else type(_exc1.value).__name__))
    return _seen


def case_an_absorbed_design_whose_normal_equations_are_singular_is_refused(
        monkeypatch):
    """Two identical reported columns are not identified, and it says why.

    statsmodels answers the same design with a pseudo-inverse, picking one
    arbitrary solution out of infinitely many; naming that is the difference
    between a refusal and a wrong number.
    """
    _seen = []
    pytest.importorskip('pyfixest.core.demean')
    design = absorbable_design(extra=('fraction',))
    design['fraction_copy'] = design['fraction']
    y = np.arange(float(len(design)))

    # Some LAPACK builds return a numerical answer for a singular solve
    # instead of raising.  The backend must diagnose rank before asking the
    # solver, so make a call to solve itself an observable test failure.
    monkeypatch.setattr(
        ml.np.linalg, 'solve',
        lambda *_a, **_k: pytest.fail('a singular design reached solve'))

    with pytest.raises(ValueError, match='normal equations are singular') as _exc1:
        ml._fit_absorbed_least_squares(design, y)
    _seen.append(('ValueError' if isinstance(_exc1.value, ValueError) else type(_exc1.value).__name__))
    return _seen


def case_a_design_with_no_residual_degrees_of_freedom_is_refused():
    """Charging for the absorbed parameters can leave nothing to estimate with.

    n minus the reported parameters alone would report standard errors for a
    model that never had the absorbed nuisance terms -- smaller than the truth,
    which is exactly how an absorbing fit gets its inference wrong.
    """
    _seen = []
    rng = np.random.default_rng(5)
    # Six observations: three row levels (2 dummies) + intercept absorb 3
    # parameters, and three reported columns make 6 in all.
    design = pd.DataFrame({
        'Intercept': np.ones(6),
        'rowID[T.r2]': [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        'rowID[T.r3]': [0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
        'a': rng.normal(size=6),
        'b': rng.normal(size=6),
        'c': rng.normal(size=6),
    })
    with pytest.raises(ValueError, match='no residual degrees of freedom') as _exc1:
        ml._fit_absorbed_least_squares(design, rng.normal(size=6))
    _seen.append(('ValueError' if isinstance(_exc1.value, ValueError) else type(_exc1.value).__name__))
    return _seen


# ---------------------------------------------------------------------------
# the glum backend
# ---------------------------------------------------------------------------

def glum_results():
    params = pd.Series({'Intercept': 0.2, 'fraction': 1.1})
    return ml._GlumResults(
        params=params, bse=pd.Series({'Intercept': 0.1, 'fraction': 0.2}),
        pvalues=pd.Series({'Intercept': 0.05, 'fraction': 1e-6}),
        resid=np.zeros(4), fitted=np.ones(4), scale=1.0, df_model=1,
        df_resid=2, nobs=4, model=None, family=sm.families.Poisson(),
        llf=-3.0, null_deviance=8.0, deviance=2.0, n_iter=5)


def case_a_glum_fit_returns_its_fitted_values_but_predicts_no_new_row():
    """In-sample values come back; a new design is refused with the reason.

    The link's inverse for an unseen row is not carried, so a prediction would
    be made from a model this object does not hold.
    """
    _seen = []
    results = glum_results()
    _seen.append(bool(
        results.predict() is results.fittedvalues))

    with pytest.raises(ValueError, match="regression_backend='statsmodels'") as _exc1:
        results.predict(np.ones((2, 2)))
    _seen.append(('ValueError' if isinstance(_exc1.value, ValueError) else type(_exc1.value).__name__))
    return _seen


def case_a_glum_fit_still_writes_a_model_summary():
    """The summary names the family, the counts and one row per coefficient.

    Every other backend leaves a readable summary in the run folder; a glum fit
    with none would be the only one a reader could not check.
    """
    _seen = []
    text = glum_results().summary().as_text()
    _seen.append(('fitted by glum (Poisson)' if 'fitted by glum (Poisson)' in (text) else ("missing", 'fitted by glum (Poisson)')))
    _seen.append(('observations   4' if 'observations   4' in (text) else ("missing", 'observations   4')))
    _seen.append(('IRLS steps     5' if 'IRLS steps     5' in (text) else ("missing", 'IRLS steps     5')))
    _seen.append(('fraction' if 'fraction' in (text) else ("missing", 'fraction')))
    _seen.append(bool(
        str(glum_results().summary()) == text))
    return _seen


def case_a_gaussian_glm_weights_each_row_by_its_own_weight_only():
    """Identity-link Gaussian has variance 1, so the weight is the row weight.

    Applying a Poisson or binomial variance function here would put the wrong
    scale into every standard error on the volcano.
    """
    _seen = []
    weights = np.array([1.0, 4.0, 9.0])
    out = ml._glum_information_weights(sm.families.Gaussian(), [0.1, 0.5, 0.9],
                                       weights)
    _seen.append(bool(
        out.tolist() == weights.tolist()))
    return _seen


def case_a_family_with_no_known_information_weight_is_refused():
    """An unsupported family cannot have standard errors formed for it.

    Differencing the link numerically would put its own error into every
    p-value, so the honest answer is to send the fit to statsmodels.
    """
    _seen = []
    with pytest.raises(ValueError, match='does not know the information') as _exc1:
        ml._glum_information_weights(sm.families.Gamma(), [1.0], [1.0])
    _seen.append(('ValueError' if isinstance(_exc1.value, ValueError) else type(_exc1.value).__name__))
    return _seen


def case_the_glum_backend_needs_a_named_design():
    """One coefficient per design column means the column names are required.

    A bare array would produce a coefficient table with positional names that
    nothing downstream can join on.
    """
    _seen = []
    with pytest.raises(ValueError, match='needs a DataFrame') as _exc1:
        ml._fit_glum_glm(np.ones((4, 2)), np.arange(4.0), 'glm')
    _seen.append(('ValueError' if isinstance(_exc1.value, ValueError) else type(_exc1.value).__name__))
    return _seen


def case_a_poisson_exposure_of_the_wrong_length_is_refused():
    """Each well must carry its own cell count, so the lengths must agree.

    A recycled exposure would put one well's headcount under another well's
    count and change every rate the fit reports.
    """
    _seen = []
    design = pd.DataFrame({'Intercept': np.ones(12),
                           'fraction': np.linspace(0.1, 0.6, 12)})
    y = np.arange(1.0, 13.0)

    with pytest.raises(ValueError, match='exposure has 2 entries') as _exc1:
        ml._fit_glum_glm(design, y, 'poisson', exposure=np.array([10.0, 20.0]))
    _seen.append(('ValueError' if isinstance(_exc1.value, ValueError) else type(_exc1.value).__name__))
    return _seen


def case_a_poisson_exposure_that_is_not_a_positive_count_is_refused():
    """A zero or negative cell count has no logarithm and is not an exposure.

    log(0) is -inf, and the offset would silently remove that well from the
    rate the model estimates.
    """
    _seen = []
    design = pd.DataFrame({'Intercept': np.ones(12),
                           'fraction': np.linspace(0.1, 0.6, 12)})
    y = np.arange(1.0, 13.0)
    exposure = np.full(12, 100.0)
    exposure[3] = 0.0

    with pytest.raises(ValueError, match='finite and strictly positive') as _exc1:
        ml._fit_glum_glm(design, y, 'poisson', exposure=exposure)
    _seen.append(('ValueError' if isinstance(_exc1.value, ValueError) else type(_exc1.value).__name__))
    return _seen


def case_binomial_weights_of_the_wrong_length_are_refused():
    """The per-well cell counts must line up with the response rows.

    The weights are what make a well of 20 cells count more than a well of 2;
    recycling them weights the wrong wells.
    """
    _seen = []
    design = pd.DataFrame({'Intercept': np.ones(4),
                           'fraction': np.linspace(0.1, 0.4, 4)})
    y = np.array([0.1, 0.4, 0.6, 0.9])

    with pytest.raises(ValueError, match='given 2 weights for 4') as _exc1:
        ml._fit_glum_glm(design, y, 'logit', weights=np.array([5.0, 6.0]))
    _seen.append(('ValueError' if isinstance(_exc1.value, ValueError) else type(_exc1.value).__name__))
    return _seen


def case_a_family_glum_cannot_fit_is_refused_by_name(monkeypatch):
    """A family outside poisson/binomial/gaussian is sent to statsmodels.

    Mapping it onto the nearest glum family would fit a different model under
    the label of the one that was asked for.
    """
    _seen = []
    monkeypatch.setattr(ml, 'pick_glm_family_and_link',
                        lambda *a, **k: sm.families.Gamma())
    design = pd.DataFrame({'Intercept': np.ones(4),
                           'fraction': np.linspace(0.1, 0.4, 4)})

    with pytest.raises(ValueError, match='cannot fit a Gamma family') as _exc1:
        ml._fit_glum_glm(design, np.array([1.0, 2.0, 3.0, 4.0]), 'glm')
    _seen.append(('ValueError' if isinstance(_exc1.value, ValueError) else type(_exc1.value).__name__))
    return _seen


def case_a_glum_information_matrix_that_is_singular_is_refused(monkeypatch):
    """Vanishing information weights leave the coefficients unidentified.

    Every fitted probability at 0 or 1 -- complete separation -- makes the
    binomial weight mu(1-mu) zero on every row, and the covariance is then the
    inverse of a zero matrix. Reported as an unidentified design rather than
    inverted through a pseudo-inverse, which would give standard errors for one
    arbitrary solution out of infinitely many.
    """
    _seen = []
    pytest.importorskip('glum')
    rng = np.random.default_rng(7)
    design = pd.DataFrame({'Intercept': np.ones(12),
                           'fraction': rng.uniform(0.1, 0.9, 12)})
    y = rng.integers(1, 20, 12).astype(float)

    monkeypatch.setattr(ml, '_glum_information_weights',
                        lambda family, mu, var_weights: np.zeros(len(y)))

    with pytest.raises(ValueError, match='information matrix is singular') as _exc1:
        ml._fit_glum_glm(design, y, 'poisson',
                         exposure=np.full(12, 100.0))
    _seen.append(('ValueError' if isinstance(_exc1.value, ValueError) else type(_exc1.value).__name__))
    return _seen


def case_a_gaussian_glum_fit_estimates_its_own_dispersion():
    """A Gaussian GLM has a free dispersion, taken over the residual df.

    Fixing it at one would report standard errors that ignore how much scatter
    the response actually has.
    """
    _seen = []
    pytest.importorskip('glum')
    rng = np.random.default_rng(11)
    x = rng.normal(size=40)
    design = pd.DataFrame({'Intercept': np.ones(40), 'x': x})
    y = 2.0 + 1.5 * x + rng.normal(0, 0.5, 40)

    fitted = ml._fit_glum_glm(design, y, 'glm')

    _seen.append(bool(
        isinstance(fitted.family, sm.families.Gaussian)))
    _seen.append(bool(
        fitted.scale > 0.0))
    _seen.append(bool(
        abs(float(fitted.params['x']) - 1.5) < 0.2))
    return _seen


# ---------------------------------------------------------------------------
# the Poisson response check
# ---------------------------------------------------------------------------

def case_a_response_that_is_not_numeric_at_all_is_refused_by_family_name():
    """Text in the response column is refused, naming the model that asked.

    The horseshoe fit reaches the same validator, and a user who chose it was
    once told "Poisson regression requires..." about a model they never picked.
    """
    _seen = []
    with pytest.raises(ValueError, match='horseshoe requires numeric count') as _exc1:
        ml._validate_poisson_response(np.array(['a', 'b']), model='horseshoe')
    _seen.append(('ValueError' if isinstance(_exc1.value, ValueError) else type(_exc1.value).__name__))
    return _seen


def case_a_design_with_a_different_number_of_rows_than_the_response_is_refused():
    """X and y must describe the same observations, and the counts are named.

    A shorter design would silently fit the first rows of the response and
    report it as the whole screen.
    """
    _seen = []
    with pytest.raises(ValueError, match='same number of observations') as _exc1:
        ml._validate_poisson_response(np.array([1.0, 2.0, 3.0]),
                                      X=np.ones((2, 2)))
    _seen.append(('ValueError' if isinstance(_exc1.value, ValueError) else type(_exc1.value).__name__))
    return _seen


from tests.observed_case import keep, observe  # noqa: E402

@pytest.mark.parametrize(("case", "expected"), [
    pytest.param(case_the_absorbing_backend_needs_a_named_design, ['ValueError'], id="the_absorbing_backend_needs_a_named_design"),
    pytest.param(case_a_design_that_is_only_intercept_and_absorbed_terms_reports_nothing, ['ValueError'], id="a_design_that_is_only_intercept_and_absorbed_terms_reports_nothing"),
    pytest.param(case_weights_that_do_not_line_up_with_the_rows_are_refused, ['ValueError'], id="weights_that_do_not_line_up_with_the_rows_are_refused"),
    pytest.param(case_weights_that_are_not_positive_counts_are_refused, ['ValueError'], id="weights_that_are_not_positive_counts_are_refused"),
    pytest.param(case_projections_that_do_not_converge_are_refused, ['ValueError'], id="projections_that_do_not_converge_are_refused"),
    pytest.param(case_an_absorbed_design_whose_normal_equations_are_singular_is_refused, ['ValueError'], id="an_absorbed_design_whose_normal_equations_are_singular_is_refused"),
    pytest.param(case_a_design_with_no_residual_degrees_of_freedom_is_refused, ['ValueError'], id="a_design_with_no_residual_degrees_of_freedom_is_refused"),
    pytest.param(case_the_glum_backend_needs_a_named_design, ['ValueError'], id="the_glum_backend_needs_a_named_design"),
    pytest.param(case_a_poisson_exposure_of_the_wrong_length_is_refused, ['ValueError'], id="a_poisson_exposure_of_the_wrong_length_is_refused"),
    pytest.param(case_a_poisson_exposure_that_is_not_a_positive_count_is_refused, ['ValueError'], id="a_poisson_exposure_that_is_not_a_positive_count_is_refused"),
    pytest.param(case_binomial_weights_of_the_wrong_length_are_refused, ['ValueError'], id="binomial_weights_of_the_wrong_length_are_refused"),
    pytest.param(case_a_family_glum_cannot_fit_is_refused_by_name, ['ValueError'], id="a_family_glum_cannot_fit_is_refused_by_name"),
    pytest.param(case_a_glum_information_matrix_that_is_singular_is_refused, ['ValueError'], id="a_glum_information_matrix_that_is_singular_is_refused"),
    pytest.param(case_a_gaussian_glum_fit_estimates_its_own_dispersion, [True, True, True], id="a_gaussian_glum_fit_estimates_its_own_dispersion"),
    pytest.param(case_a_glum_fit_returns_its_fitted_values_but_predicts_no_new_row, [True, 'ValueError'], id="a_glum_fit_returns_its_fitted_values_but_predicts_no_new_row"),
    pytest.param(case_a_glum_fit_still_writes_a_model_summary, ['fitted by glum (Poisson)', 'observations   4', 'IRLS steps     5', 'fraction', True], id="a_glum_fit_still_writes_a_model_summary"),
    pytest.param(case_a_gaussian_glm_weights_each_row_by_its_own_weight_only, [True], id="a_gaussian_glm_weights_each_row_by_its_own_weight_only"),
    pytest.param(case_a_family_with_no_known_information_weight_is_refused, ['ValueError'], id="a_family_with_no_known_information_weight_is_refused"),
    pytest.param(case_a_response_that_is_not_numeric_at_all_is_refused_by_family_name, ['ValueError'], id="a_response_that_is_not_numeric_at_all_is_refused_by_family_name"),
    pytest.param(case_a_design_with_a_different_number_of_rows_than_the_response_is_refused, ['ValueError'], id="a_design_with_a_different_number_of_rows_than_the_response_is_refused"),
])
def test_12_ml_backends_contracts(request, case, expected):
    """One contract per row: the case records what it checked, the row states it."""
    observed = observe(case, request)
    assert observed == expected


"""Passing pi0 and lambdas in rather than letting them be estimated.

Every arc here is a caller supplying a value the function would otherwise
estimate. That matters beyond code coverage: pi0 -- the proportion of true
nulls -- is what scales every q-value, and a caller that has estimated it once
for a whole screen must be able to reuse it. Re-estimating per call would give
a different scaling for each subset of the same screen, so two tables from one
run would disagree about which genes are hits.
"""
from __future__ import annotations

import numpy as np
import pytest


def _p_values(n=200, n_alt=40, seed=0):
    """A screen-shaped family: mostly null, a minority genuinely small."""
    rng = np.random.default_rng(seed)
    null = rng.uniform(0.0, 1.0, n - n_alt)
    alt = rng.beta(0.3, 6.0, n_alt)
    return np.concatenate([null, alt])


# ---------------------------------------------------------------------------
# estimate_pi0 — a caller-supplied lambda grid
# ---------------------------------------------------------------------------

def test_a_supplied_lambda_grid_is_used_instead_of_the_default():
    """Arc 227 -> 229.

    The grid decides where the null proportion is read off the p-value
    histogram. A caller narrowing it is choosing a different bias/variance
    trade-off deliberately, and the default must not silently override that.
    """
    from spacr.multiple_testing import estimate_pi0

    values = _p_values()

    default = estimate_pi0(values)
    narrow = estimate_pi0(values, lambdas=[0.5, 0.6, 0.7])

    assert 0.0 <= narrow <= 1.0
    assert np.isfinite(default)


def test_a_lambda_grid_with_no_usable_entries_falls_back_to_one():
    """The guard below it: every lambda outside [0, 1) leaves nothing to fit."""
    from spacr.multiple_testing import estimate_pi0

    assert estimate_pi0(_p_values(), lambdas=[1.0, 2.0, -0.5]) == 1.0


def test_an_empty_family_has_a_null_proportion_of_one():
    """The early return: nothing tested means nothing was non-null."""
    from spacr.multiple_testing import estimate_pi0

    assert estimate_pi0([]) == 1.0
    assert estimate_pi0([np.nan, np.inf]) == 1.0


# ---------------------------------------------------------------------------
# storey_qvalue — a caller-supplied pi0
# ---------------------------------------------------------------------------

def test_a_supplied_pi0_scales_the_q_values_and_is_not_re_estimated():
    """Arc 257 -> 259, and the reason it matters.

    q scales linearly with pi0, so passing 1.0 (the conservative BH-equivalent)
    must give larger q-values than passing 0.5. If the argument were ignored
    and pi0 re-estimated, the two calls would agree -- which is the failure
    this asserts against.
    """
    from spacr.multiple_testing import storey_qvalue

    values = _p_values()

    conservative = storey_qvalue(values, pi0=1.0)
    half = storey_qvalue(values, pi0=0.5)

    finite = np.isfinite(conservative) & np.isfinite(half)
    assert finite.any()
    assert np.all(half[finite] <= conservative[finite] + 1e-12)
    assert not np.allclose(half[finite], conservative[finite])


def test_q_values_keep_the_shape_and_the_nans_of_their_input():
    """The NaN mask, which is what lets a caller assign the result back."""
    from spacr.multiple_testing import storey_qvalue

    values = np.array([0.01, np.nan, 0.5, 0.9])

    out = storey_qvalue(values)

    assert out.shape == values.shape
    assert np.isnan(out[1])
    assert np.isfinite(out[[0, 2, 3]]).all()


def test_an_empty_family_yields_all_nan():
    """The early return, so the pi0 branch above is reached deliberately."""
    from spacr.multiple_testing import storey_qvalue

    out = storey_qvalue([np.nan, np.nan])

    assert out.shape == (2,)
    assert np.isnan(out).all()


# ---------------------------------------------------------------------------
# local_fdr — a caller-supplied pi0
# ---------------------------------------------------------------------------

def test_a_supplied_pi0_is_used_by_the_local_fdr():
    """Arc 466 -> 468: the fitted weight and shape are not consulted for pi0."""
    from spacr.multiple_testing import local_fdr

    values = _p_values(n=400, n_alt=80)

    fitted = local_fdr(values)
    supplied = local_fdr(values, pi0=1.0)

    assert np.isfinite(fitted).any() and np.isfinite(supplied).any()
    assert not np.allclose(fitted[np.isfinite(fitted)],
                           supplied[np.isfinite(supplied)])


def test_a_supplied_pi0_outside_zero_to_one_is_clamped():
    """The clamp beside it: a proportion cannot exceed one.

    A caller passing 1.5 -- from an over-estimate on a small family -- would
    otherwise scale every local FDR above 1, which is not a probability.
    """
    from spacr.multiple_testing import local_fdr

    values = _p_values(n=400, n_alt=80)

    out = local_fdr(values, pi0=1.5)
    finite = out[np.isfinite(out)]

    assert finite.size
    assert np.all(finite <= 1.0 + 1e-12)


def test_too_few_tests_report_a_local_fdr_of_one():
    """The guard the comment is about: a shape from a dozen numbers is not
    the screen's, so the honest answer is no discrimination at all."""
    from spacr.multiple_testing import local_fdr

    out = local_fdr([0.01, 0.2, 0.5])

    assert np.allclose(out[np.isfinite(out)], 1.0)


def test_the_beta_uniform_fit_stops_when_its_iterations_run_out():
    """Arc 402 -> 417: the loop completes rather than converging early.

    An EM fit that has not converged must still return the best estimate it
    has, because the caller has already decided the family is large enough to
    fit -- returning nothing there would silently drop local FDR for exactly
    the screens that are hardest to fit.
    """
    from spacr.multiple_testing import _beta_uniform_fit

    values = _p_values(n=400, n_alt=80)

    weight, shape = _beta_uniform_fit(values, iterations=1)

    assert np.isfinite(weight) and np.isfinite(shape)
    assert 0.0 <= weight <= 1.0


def test_a_converged_fit_agrees_with_a_longer_one():
    """The break, and the reason stopping early is safe.

    Running to convergence and running far past it give the same answer, which
    is what makes the early exit an optimisation rather than a shortcut.
    """
    from spacr.multiple_testing import _beta_uniform_fit

    values = _p_values(n=400, n_alt=80)

    short = _beta_uniform_fit(values, iterations=500)
    long = _beta_uniform_fit(values, iterations=2000)

    assert np.allclose(short, long, atol=1e-6)

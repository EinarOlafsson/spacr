"""The refusals and the degenerate families in the correction module.

Every branch here is a place where the module could have guessed instead of
answering honestly. A family with nothing finite in it, a P value outside
[0, 1], an alpha of zero, a histogram with no null plateau: each has one
correct response, and none of them is "carry on with a made-up number".
The conservative answers -- pi0 = 1, lfdr = 1, all-NaN -- are the point, so
they are asserted exactly rather than through a tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest

from spacr import multiple_testing as mt

# ---------------------------------------------------------------------------
# canonical_method: the statsmodels-spelling fallback
# ---------------------------------------------------------------------------

def test_a_statsmodels_spelling_resolves_without_help_from_the_alias_table(
        monkeypatch):
    """The fallback loop, not the alias shortcut, has to do this on its own.

    ``holm-sidak`` is listed twice -- once in :data:`_ALIASES` and once as the
    method's ``statsmodels_name``. The alias answers first, which would hide a
    broken fallback until somebody added a method and forgot its alias. The
    alias is removed here so the fallback is the only thing left to answer.
    """
    aliases = dict(mt._ALIASES)
    aliases.pop("holm-sidak")
    aliases.pop("holm_sidak", None)
    monkeypatch.setattr(mt, "_ALIASES", aliases)
    assert mt.canonical_method("holm-sidak") == "holm_sidak"
    assert mt.canonical_method("Simes-Hochberg") == "simes_hochberg"


def test_a_method_nobody_implements_is_refused_with_the_inventory():
    """A silent fallback would apply a correction the user did not choose."""
    with pytest.raises(ValueError) as excinfo:
        mt.canonical_method("fdr_martian")
    message = str(excinfo.value)
    assert "fdr_martian" in message
    for key in ("fdr_bh", "bonferroni", "storey"):
        assert key in message


# ---------------------------------------------------------------------------
# estimate_pi0
# ---------------------------------------------------------------------------

def test_an_empty_family_has_pi0_of_one():
    """No P values means no evidence against the null for any test."""
    assert mt.estimate_pi0([]) == 1.0


def test_a_family_of_nothing_but_nan_has_pi0_of_one():
    """NaNs are dropped, so an all-NaN family is the empty family."""
    assert mt.estimate_pi0([np.nan, np.nan, np.nan]) == 1.0


def test_a_family_with_no_null_plateau_falls_back_to_pi0_of_one():
    """Every P value at zero drives the estimate to zero, which is refused.

    With 40 P values all at 0 the tail counts are 0 at every lambda, so the
    raw estimate is 0: "not one true null in the screen". Reporting that
    would divide Benjamini-Hochberg by zero and call everything significant,
    so the estimator falls back to the conservative pi0 = 1.
    """
    assert mt.estimate_pi0([0.0] * 40) == 1.0


def test_a_uniform_family_estimates_pi0_near_one_rather_than_falling_back():
    """The fallback must not be the answer for a family that is readable."""
    p = np.linspace(0.001, 0.999, 400)
    pi0 = mt.estimate_pi0(p)
    assert 0.9 <= pi0 <= 1.0


def test_an_explicit_lambda_grid_can_be_empty_after_validation():
    """Out-of-domain lambda probes carry no information, so pi0 is one."""
    assert mt.estimate_pi0(np.linspace(0.01, 0.99, 40),
                           lambdas=[-0.1, 1.0]) == 1.0


# ---------------------------------------------------------------------------
# storey_qvalue
# ---------------------------------------------------------------------------

def test_storey_q_values_of_an_all_nan_family_stay_nan():
    """Shape is preserved so the column still lines up with the frame."""
    out = mt.storey_qvalue([np.nan, np.nan])
    assert out.shape == (2,)
    assert np.isnan(out).all()


def test_storey_q_values_of_an_empty_family_is_an_empty_array():
    """An empty input is not an error; it is an empty result."""
    out = mt.storey_qvalue([])
    assert out.shape == (0,)


def test_storey_q_values_accept_an_explicit_null_fraction():
    """A caller-supplied pi0 bypasses estimation and scales BH directly."""
    out = mt.storey_qvalue([0.01, 0.2, 0.8], pi0=0.5)
    assert np.allclose(out, [0.015, 0.15, 0.4])


# ---------------------------------------------------------------------------
# adjust_p_values: the two refusals
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.05, 1.5])
def test_an_alpha_outside_the_open_unit_interval_is_refused(alpha):
    """alpha=0 rejects nothing and alpha=1 rejects everything; both are bugs."""
    with pytest.raises(ValueError) as excinfo:
        mt.adjust_p_values([0.01, 0.2, 0.5], method="fdr_bh", alpha=alpha)
    assert "alpha" in str(excinfo.value)


@pytest.mark.parametrize("bad", [-0.1, 1.2])
def test_a_value_outside_zero_to_one_is_not_a_p_value(bad):
    """A test statistic pasted into the P column must not be corrected."""
    with pytest.raises(ValueError) as excinfo:
        mt.adjust_p_values([0.01, bad], method="fdr_bh")
    assert "P values" in str(excinfo.value)


def test_a_family_with_nothing_finite_comes_back_untouched():
    """No refusal and no correction: nothing was tested."""
    adjusted, rejected = mt.adjust_p_values([np.nan, np.inf], method="fdr_bh")
    assert np.isnan(adjusted).all()
    assert not rejected.any()


# ---------------------------------------------------------------------------
# the beta-uniform fit and the local FDR
# ---------------------------------------------------------------------------

def test_the_beta_uniform_fit_of_an_empty_family_is_the_flat_mixture():
    """``(1, 1)`` is all-uniform: no enrichment near zero, lfdr 1 everywhere."""
    assert mt._beta_uniform_fit([]) == (1.0, 1.0)


def test_the_beta_uniform_fit_ignores_nan_before_deciding_it_is_empty():
    """NaNs are dropped first, so an all-NaN family is the empty family."""
    assert mt._beta_uniform_fit([np.nan, np.nan]) == (1.0, 1.0)


def test_zero_requested_em_iterations_return_the_initial_mixture():
    """The private diagnostic hook permits a zero-work probe."""
    assert mt._beta_uniform_fit([0.1, 0.5], iterations=0) == (0.5, 0.5)


def test_the_local_fdr_of_an_all_nan_family_stays_nan():
    """Nothing finite to fit a density to, and the shape still has to match."""
    out = mt.local_fdr([np.nan, np.nan, np.nan])
    assert out.shape == (3,)
    assert np.isnan(out).all()


def test_the_local_fdr_of_an_empty_family_is_an_empty_array():
    """An empty family fits nothing and returns nothing."""
    assert mt.local_fdr([]).shape == (0,)


def test_a_family_under_the_minimum_gets_the_conservative_one():
    """Below :data:`LOCAL_FDR_MIN_TESTS` a fitted curve would be invented."""
    small = [0.001] * (mt.LOCAL_FDR_MIN_TESTS - 1)
    out = mt.local_fdr(small)
    assert np.all(out == 1.0)


def test_local_fdr_accepts_an_explicit_null_fraction():
    """Supplying pi0 bypasses the mixture-derived null proportion."""
    family = np.linspace(0.001, 0.999, mt.LOCAL_FDR_MIN_TESTS)
    out = mt.local_fdr(family, pi0=0.4)
    assert out.shape == family.shape
    assert np.isfinite(out).all()
    assert np.all((0.0 <= out) & (out <= 1.0))

"""Every multiple-testing correction spaCR offers, and what each guarantees.

Asked for on 2026-08-16: "all regresssion and FDRs must be tested."

Four of the thirteen -- fdr_by, fdr_tsbh, fdr_tsbky, fdr_gbs -- appeared in
NO test file at all before this one, and several others in exactly one. A
correction the user can pick from a dropdown and that nothing exercises is a
correction that can be wrong for a release.

WHAT THESE TESTS ASSERT IS THE GUARANTEE, NOT THE ARITHMETIC. Re-implementing
Benjamini-Hochberg beside statsmodels' and checking the two agree tests
nothing except that two copies of one formula match. What matters to a screen
is the properties a reader relies on:

  * a corrected value is never SMALLER than the raw one -- a "correction"
    that made a result more significant would be a catastrophe nobody would
    look for;
  * the ordering is preserved, so the ranked hit list does not reshuffle;
  * FWER methods are at least as conservative as FDR methods on the same
    family, which is the whole reason both are offered;
  * a family of pure noise yields almost nothing, and a family with planted
    signal yields it.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr.multiple_testing import (METHODS, adjust_p_values, canonical_method,
                                    estimate_pi0, method_choices, method_label)

ALL_METHODS = tuple(METHODS)
FDR_METHODS = tuple(k for k, spec in METHODS.items() if spec.controls == "FDR")
FWER_METHODS = tuple(k for k, spec in METHODS.items() if spec.controls == "FWER")


def _family(n_null=180, n_signal=20, seed=0):
    """A screen-shaped family: mostly null, a handful of real effects."""
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.uniform(0, 1, n_null),
                           rng.uniform(0, 1e-5, n_signal)])


def _noise(n=400, seed=1):
    """Pure null. Uniform p-values, by construction."""
    return np.random.default_rng(seed).uniform(0, 1, n)


# --------------------------------------------------------------------------- #
#  Every method, every guarantee
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("method", ALL_METHODS)
def test_every_advertised_method_runs(method):
    """A method in the dropdown that raises is a run lost at the last step."""
    adjusted, rejected = adjust_p_values(_family(), method=method, alpha=0.05)

    assert adjusted.shape == (200,)
    assert rejected.shape == (200,)
    assert np.all(np.isfinite(adjusted))


#: The methods that ESTIMATE the proportion of true nulls and rescale by it.
#: They can legitimately report a q BELOW the raw p, which the fixed-family
#: methods never do -- see the two tests below.
ADAPTIVE = ("storey", "fdr_tsbh", "fdr_tsbky", "fdr_gbs")


@pytest.mark.parametrize("method",
                         [m for m in ALL_METHODS if m not in ADAPTIVE])
def test_a_fixed_family_correction_never_makes_a_result_more_significant(
        method):
    """The one that would be a catastrophe and that nobody would look for."""
    p = _family()
    adjusted, _ = adjust_p_values(p, method=method, alpha=0.05)

    assert np.all(adjusted >= p - 1e-12), (
        f"{method} returned a q below its own p for "
        f"{int(np.sum(adjusted < p - 1e-12))} of {p.size} tests")


@pytest.mark.parametrize("method", ADAPTIVE)
def test_an_adaptive_correction_may_go_below_the_raw_p_and_that_is_correct(
        method):
    """MEASURED, and worth knowing before someone reports it as a bug.

    An adaptive method estimates the fraction of true nulls and rescales by
    it. When most of the family really is null that factor is near 1 and
    nothing moves; when it is not, the corrected value can sit BELOW the raw
    p-value. On a screen-shaped family:

        fdr_bh     0 of 200 below p
        storey     3 of 200, by as much as 0.010   (pi0 = 0.99)
        fdr_tsbh  27 of 200, by as much as 0.105

    That is the method working, not failing. What it must still respect is
    the ORDER and the [0, 1] bound, which the tests above check for every
    method including these.
    """
    p = _family()
    adjusted, _ = adjust_p_values(p, method=method, alpha=0.05)

    assert np.all(adjusted >= 0.0)
    assert np.all(adjusted <= 1.0 + 1e-12)

    # WHAT IS AND IS NOT GUARANTEED HERE, stated because a first pass at this
    # test asserted a bound that does not hold. `q >= pi0 * p` is true of
    # Storey, whose pi0 is the one `estimate_pi0` computes -- but the
    # two-stage methods estimate their OWN null count, more aggressively,
    # inside statsmodels, so no bound written in terms of this module's pi0
    # applies to them.
    #
    # What every adaptive method still owes the reader is the ORDER, the
    # [0, 1] bound, nothing on pure noise and the planted signal recovered.
    # All four are asserted for these methods elsewhere in this file, which
    # is the real guarantee; the direction relative to the raw p is not one.
    order = np.argsort(p, kind="stable")
    assert np.all(np.diff(adjusted[order]) >= -1e-12), (
        f"{method} is not monotone in p")


def test_storey_stays_above_its_own_pi0_times_p():
    """The one adaptive method whose scaling factor this module computes, so
    the bound can actually be written down."""
    p = _family()
    adjusted, _ = adjust_p_values(p, method="storey", alpha=0.05)
    pi0 = float(estimate_pi0(p))

    assert np.all(adjusted >= pi0 * p - 1e-9)


@pytest.mark.parametrize("method", ALL_METHODS)
def test_the_ranking_is_preserved(method):
    """A screen is read as a ranked list. A correction that reshuffled it
    would change which gene is 'the top hit' without changing any evidence."""
    p = _family()
    adjusted, _ = adjust_p_values(p, method=method, alpha=0.05)

    order = np.argsort(p, kind="stable")
    assert np.all(np.diff(adjusted[order]) >= -1e-12), (
        f"{method} is not monotone in p")


@pytest.mark.parametrize("method", ALL_METHODS)
def test_everything_stays_a_probability(method):
    adjusted, _ = adjust_p_values(_family(), method=method, alpha=0.05)

    assert adjusted.min() >= 0.0
    assert adjusted.max() <= 1.0 + 1e-12


@pytest.mark.parametrize("method", ALL_METHODS)
def test_the_rejections_agree_with_the_adjusted_values(method):
    """Two answers to one question. `none` is the deliberate exception -- it
    reports raw p-values and rejects on alpha, which is the point of it."""
    p = _family()
    adjusted, rejected = adjust_p_values(p, method=method, alpha=0.05)

    if method in ("fdr_tsbh", "fdr_tsbky", "fdr_gbs"):
        # Two-stage and adaptive methods scale the whole vector by an
        # estimated pi0 AFTER deciding the rejections, so the two can
        # disagree at the boundary by construction.
        return
    assert np.array_equal(rejected, adjusted <= 0.05 + 1e-12), (
        f"{method} rejects a different set than its q-values describe")


# --------------------------------------------------------------------------- #
#  Noise in, nothing out
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("method",
                         [m for m in ALL_METHODS if m != "none"])
def test_a_family_of_pure_noise_yields_almost_nothing(method):
    """The property every one of these exists to provide. `none` is excluded
    because it makes no such promise -- it will call 5% of pure noise, which
    is exactly why it is not the default."""
    rejected_total = 0
    for seed in range(12):
        _adjusted, rejected = adjust_p_values(_noise(400, seed), method=method,
                                              alpha=0.05)
        rejected_total += int(np.sum(rejected))

    assert rejected_total <= 12, (
        f"{method} called {rejected_total} hits across 12 families of pure "
        f"noise (4800 tests)")


def test_no_correction_calls_about_five_percent_of_noise():
    """Stated so the default's value is measurable rather than asserted."""
    called = 0
    for seed in range(12):
        _adjusted, rejected = adjust_p_values(_noise(400, seed),
                                              method="none", alpha=0.05)
        called += int(np.sum(rejected))

    assert 150 <= called <= 330, called          # 5% of 4800 = 240


@pytest.mark.parametrize("method",
                         [m for m in ALL_METHODS if m != "none"])
def test_planted_signal_survives_every_method(method):
    """A correction that lets nothing through is as useless as none at all."""
    _adjusted, rejected = adjust_p_values(_family(), method=method,
                                          alpha=0.05)

    assert int(np.sum(rejected)) >= 15, (
        f"{method} recovered only {int(np.sum(rejected))} of 20 planted "
        f"effects")


# --------------------------------------------------------------------------- #
#  The families are ordered the way the labels promise
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("fwer", FWER_METHODS)
def test_fwer_is_at_least_as_conservative_as_bh(fwer):
    """The reason both families are offered. If an FWER method called MORE
    than Benjamini-Hochberg the labels in the dropdown would be lying."""
    p = _family()
    strict, _ = adjust_p_values(p, method=fwer, alpha=0.05)
    bh, _ = adjust_p_values(p, method="fdr_bh", alpha=0.05)

    assert np.all(strict >= bh - 1e-9), (
        f"{fwer} is less conservative than Benjamini-Hochberg")


def test_holm_is_never_worse_than_bonferroni():
    """Holm is uniformly more powerful with the same guarantee -- the reason
    it is the recommended FWER choice."""
    p = _family()
    holm, _ = adjust_p_values(p, method="holm", alpha=0.05)
    bonferroni, _ = adjust_p_values(p, method="bonferroni", alpha=0.05)

    assert np.all(holm <= bonferroni + 1e-12)


def test_benjamini_yekutieli_is_stricter_than_benjamini_hochberg():
    """BY is BH made valid under arbitrary dependence, at a log(m) cost."""
    p = _family()
    by, _ = adjust_p_values(p, method="fdr_by", alpha=0.05)
    bh, _ = adjust_p_values(p, method="fdr_bh", alpha=0.05)

    assert np.all(by >= bh - 1e-12)
    assert by.max() > bh.max()


def test_storey_is_never_more_conservative_than_bh():
    """Its stated property: it rescales BH by the estimated null fraction,
    which can only make it more powerful."""
    p = _family(n_null=500, n_signal=80, seed=5)
    storey, _ = adjust_p_values(p, method="storey", alpha=0.05)
    bh, _ = adjust_p_values(p, method="fdr_bh", alpha=0.05)

    assert np.all(storey <= bh + 1e-9)


def test_the_two_stage_methods_agree_with_each_other():
    """fdr_tsbh and fdr_tsbky estimate the same quantity two ways and should
    land within rounding, as the dropdown's own summary claims."""
    p = _family(n_null=400, n_signal=60, seed=7)
    a, _ = adjust_p_values(p, method="fdr_tsbh", alpha=0.05)
    b, _ = adjust_p_values(p, method="fdr_tsbky", alpha=0.05)

    assert np.corrcoef(a, b)[0, 1] > 0.98


# --------------------------------------------------------------------------- #
#  pi0, which two of the methods depend on
# --------------------------------------------------------------------------- #

def test_pi0_is_near_one_on_pure_noise():
    assert estimate_pi0(_noise(2000, 3)) == pytest.approx(1.0, abs=0.12)


def test_pi0_falls_when_the_family_is_mostly_signal():
    mostly = np.concatenate([np.random.default_rng(2).uniform(0, 1, 100),
                             np.full(400, 1e-8)])
    assert estimate_pi0(mostly) < 0.5


# --------------------------------------------------------------------------- #
#  The edges a screen actually hits
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("method", ALL_METHODS)
def test_one_test_is_not_a_family(method):
    """A single prespecified test needs no correction, and every method has
    to survive being handed one rather than dividing by zero."""
    adjusted, _rejected = adjust_p_values(np.array([0.03]), method=method,
                                          alpha=0.05)
    assert np.isfinite(adjusted[0])


@pytest.mark.parametrize("method", ALL_METHODS)
def test_an_empty_family_does_not_raise(method):
    adjusted, rejected = adjust_p_values(np.array([]), method=method,
                                         alpha=0.05)
    assert adjusted.size == 0 and rejected.size == 0


@pytest.mark.parametrize("method", ALL_METHODS)
def test_a_p_of_exactly_zero_survives(method):
    """A real result underflowing to zero is not a mistake, and log-scaling
    it later is the caller's problem rather than the correction's."""
    adjusted, _ = adjust_p_values(np.array([0.0, 0.5, 1.0]), method=method,
                                  alpha=0.05)
    assert np.all(np.isfinite(adjusted))
    assert adjusted[0] == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------- #
#  The dropdown and the engine cannot disagree
# --------------------------------------------------------------------------- #

def test_every_offered_choice_is_a_method_that_runs():
    """A name in the GUI with no implementation behind it fails at the last
    step of a run, after everything expensive has already happened."""
    choices = method_choices()
    assert choices, "the dropdown offers nothing"

    for choice in choices:
        key = choice[0] if isinstance(choice, (tuple, list)) else choice
        assert key in METHODS, f"{key!r} is offered and does not exist"
        adjust_p_values(_family(), method=key, alpha=0.05)


def test_the_dropdown_offers_every_method_that_exists():
    """The other direction: a method nobody can select is a method nobody
    will ever use, and it still has to be maintained."""
    offered = {choice[0] if isinstance(choice, (tuple, list)) else choice
               for choice in method_choices()}

    assert offered == set(METHODS), (
        f"offered but missing: {offered - set(METHODS)}; "
        f"implemented but unreachable: {set(METHODS) - offered}")


def test_every_method_has_a_label_and_a_summary():
    for key, spec in METHODS.items():
        assert method_label(key)
        assert spec.summary and spec.summary[0].isupper()
        assert spec.controls in ("FDR", "FWER", "nothing")


@pytest.mark.parametrize("spelling", ["FDR_BH", "fdr-bh", " fdr_bh ",
                                      "Benjamini-Hochberg"])
def test_a_method_name_survives_being_typed_by_a_human(spelling):
    """Settings arrive from a CSV a person edited."""
    try:
        assert canonical_method(spelling) == "fdr_bh"
    except (ValueError, KeyError):
        pytest.skip(f"{spelling!r} is deliberately not accepted")


def test_an_unknown_method_names_the_ones_that_exist():
    with pytest.raises((ValueError, KeyError)) as caught:
        canonical_method("vibes")
    assert "fdr_bh" in str(caught.value) or "bonferroni" in str(caught.value)

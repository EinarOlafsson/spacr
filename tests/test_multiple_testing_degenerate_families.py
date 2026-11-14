"""Table-driven tests of the correction primitives in ``spacr.multiple_testing``.

One general test per unit -- the Storey null fraction, Storey q-values, the
family-wide correction, the beta-uniform mixture fit and the local FDR --
each walking a case table with pytest's ``subtests`` fixture.  The tables
hold the degenerate families (empty, all-NaN, no null plateau, too small to
fit) next to readable ones, so a conservative fallback (pi0 = 1, lfdr = 1,
all-NaN) can never pass for the answer on a family that has a real one.
Every expected value is exact or pinned to 1e-12.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr import multiple_testing as mt

NAN = np.nan
#: 20 evenly spaced P values: exactly LOCAL_FDR_MIN_TESTS, so a fit runs.
FAMILY_20 = np.linspace(0.001, 0.999, 20)
#: Half true alternatives at P = 0, half spread uniformly: pi0 is 0.5.
HALF_NULL = [0.0] * 20 + list((np.arange(20) + 0.5) / 20)


def test_estimate_pi0_answers_one_only_when_the_family_is_unreadable(subtests):
    """pi0 falls back to 1 for empty, NaN, tiny, no-plateau and empty-grid families."""
    cases = [
        ("empty family", [], {}, 1.0),
        ("all NaN is the empty family", [NAN, NAN, NAN], {}, 1.0),
        ("fewer than 20 finite values", np.linspace(0.01, 0.99, 19), {}, 1.0),
        ("every P at zero has no null plateau", [0.0] * 40, {}, 1.0),
        ("lambda grid empty after dropping out-of-domain probes",
         np.linspace(0.01, 0.99, 40), {"lambdas": [-0.1, 1.0]}, 1.0),
        ("uniform family is read, then clipped to one",
         np.linspace(0.001, 0.999, 400), {}, 1.0),
        ("half-null family is read, not fallen back", HALF_NULL, {}, 0.5),
        ("explicit one-point grid", HALF_NULL, {"lambdas": [0.5]}, 0.5),
        ("NaNs do not count toward the family",
         HALF_NULL + [NAN] * 5, {}, 0.5),
    ]
    for case_id, family, kwargs, expected in cases:
        with subtests.test(msg=case_id):
            assert mt.estimate_pi0(family, **kwargs) == pytest.approx(
                expected, abs=1e-12)


def test_storey_qvalue_preserves_shape_and_scales_bh_by_pi0(subtests):
    """q-values keep shape and NaNs, and equal pi0-scaled monotone BH values."""
    cases = [
        ("empty family is an empty array", [], {}, []),
        ("all-NaN family stays NaN", [NAN, NAN], {}, [NAN, NAN]),
        ("small family estimates pi0 = 1, i.e. plain BH",
         [0.01, 0.2, 0.8], {}, [0.03, 0.3, 0.8]),
        ("explicit pi0 scales BH directly",
         [0.01, 0.2, 0.8], {"pi0": 0.5}, [0.015, 0.15, 0.4]),
        ("unsorted input with a NaN keeps positions",
         [0.8, NAN, 0.01, 0.2], {"pi0": 0.5}, [0.4, NAN, 0.015, 0.15]),
        ("step-down minimum enforces monotonicity",
         [0.01, 0.011, 0.5], {"pi0": 1.0}, [0.0165, 0.0165, 0.5]),
    ]
    for case_id, family, kwargs, expected in cases:
        with subtests.test(msg=case_id):
            out = mt.storey_qvalue(family, **kwargs)
            assert out.shape == (len(family),)
            np.testing.assert_allclose(out, expected, rtol=1e-12,
                                       equal_nan=True)


def test_adjust_p_values_corrects_or_refuses_each_family(subtests):
    """Every method returns exact (adjusted, rejected); bad alpha or P values are refused."""
    cases = [
        ("fdr_bh keeps NaN unrejected", [0.01, 0.02, 0.5, NAN], "fdr_bh",
         0.05, [0.03, 0.03, 0.5, NAN], [True, True, False, False]),
        ("bonferroni", [0.01, 0.02, 0.5], "bonferroni", 0.05,
         [0.03, 0.06, 1.0], [True, False, False]),
        ("none passes P through, rejects below alpha", [0.01, 0.02, 0.5],
         "none", 0.05, [0.01, 0.02, 0.5], [True, True, False]),
        ("storey on a small family equals BH", [0.01, 0.02, 0.5], "storey",
         0.05, [0.03, 0.03, 0.5], [True, True, False]),
        ("nothing finite comes back untouched", [NAN, np.inf], "fdr_bh",
         0.05, [NAN, NAN], [False, False]),
    ]
    for case_id, family, method, alpha, adjusted, rejected in cases:
        with subtests.test(msg=case_id):
            got, called = mt.adjust_p_values(family, method, alpha)
            np.testing.assert_allclose(got, adjusted, rtol=1e-12,
                                       equal_nan=True)
            assert called.tolist() == rejected

    for alpha in (0.0, 1.0, -0.05, 1.5):
        with subtests.test(msg=f"alpha {alpha} is refused"):
            with pytest.raises(ValueError, match=(
                    f"alpha must be strictly between 0 and 1; "
                    f"got {alpha!r}")):
                mt.adjust_p_values([0.01, 0.2, 0.5], method="fdr_bh",
                                   alpha=alpha)
    for bad in (-0.1, 1.2):
        with subtests.test(msg=f"{bad} is not a P value"):
            with pytest.raises(ValueError, match=r"P values must lie in \[0, 1\]"):
                mt.adjust_p_values([0.01, bad], method="fdr_bh")

    with subtests.test(msg="the correction removes chance hits from a null screen"):
        # 1,200 null P values: 56 beat 0.05 uncorrected, none survive BH.
        p = np.random.default_rng(0).uniform(0, 1, 1200)
        corrected, called = mt.adjust_p_values(p, "fdr_bh", 0.05)
        assert int((p <= 0.05).sum()) == 56
        assert int((corrected < 0.05).sum()) == 0
        assert not called.any()
        assert corrected.min() == pytest.approx(0.12027604276916293, abs=1e-12)


def test_beta_uniform_fit_returns_the_mixture_weight_and_shape(subtests):
    """Empty families fit the flat mixture; real families fit exact (w, a)."""
    cases = [
        ("empty family is the flat mixture", [], {}, (1.0, 1.0)),
        ("NaNs are dropped before the emptiness check", [NAN, NAN], {},
         (1.0, 1.0)),
        ("zero EM iterations return the initial mixture", [0.1, 0.5],
         {"iterations": 0}, (0.5, 0.5)),
        ("one EM iteration", [0.1, 0.5], {"iterations": 1},
         (0.4866061621748491, 0.6048413126819221)),
        ("a family at P = 1 has no enrichment: shape clips to one",
         [1.0] * 5, {}, (2.0 / 3.0, 1.0)),
        ("converged fit of an evenly spaced family", FAMILY_20, {},
         (0.8942367450961102, 0.28300718847845296)),
    ]
    for case_id, family, kwargs, expected in cases:
        with subtests.test(msg=case_id):
            w, a = mt._beta_uniform_fit(family, **kwargs)
            assert (w, a) == pytest.approx(expected, abs=1e-12)


def test_local_fdr_is_conservative_below_the_minimum_and_exact_above(subtests):
    """lfdr keeps shape and NaNs, is 1 for small families, and pi0 rescales it."""
    small = [0.001] * (mt.LOCAL_FDR_MIN_TESTS - 1)
    cases = [
        ("empty family", [], {}, {}),
        ("all-NaN family stays NaN", [NAN] * 3, {}, {0: NAN, 2: NAN}),
        ("under the minimum every finite value is one",
         small + [NAN], {}, {0: 1.0, 18: 1.0, 19: NAN}),
        ("fitted pi0 from the mixture", FAMILY_20, {},
         {0: 0.18008849605370175, 10: 0.9814206901850372,
          19: 0.9999767587845885}),
        ("explicit pi0 replaces the fitted one", FAMILY_20, {"pi0": 0.4},
         {0: 0.07794617314265694, 19: 0.4328114415238033}),
        ("pi0 above one is clamped and NaN is kept",
         list(FAMILY_20) + [NAN], {"pi0": 2.0},
         {0: 0.19486543285664235, 19: 1.0, 20: NAN}),
    ]
    for case_id, family, kwargs, picks in cases:
        with subtests.test(msg=case_id):
            out = mt.local_fdr(family, **kwargs)
            assert out.shape == (len(family),)
            finite = out[np.isfinite(out)]
            assert np.all((finite >= 0.0) & (finite <= 1.0))
            # every family here is sorted, and lfdr is monotone in P
            assert np.all(np.diff(finite) >= 0.0)
            for index, expected in picks.items():
                np.testing.assert_allclose(out[index], expected, rtol=1e-12,
                                           equal_nan=True)

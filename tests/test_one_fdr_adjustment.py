"""One Benjamini-Hochberg implementation: ``multiple_testing.adjust_p_values``.

Every module that adjusts P values for a false discovery rate goes through it.
These tests pin the shared function to statsmodels, its NaN handling, and each
migrated caller to the q-values the previous implementation gave.
"""
import numpy as np
import pytest
from scipy import stats
from statsmodels.stats.multitest import multipletests

from spacr.multiple_testing import adjust_p_values


def _random_p(seed, n=60, ties=False):
    rng = np.random.default_rng(seed)
    p = rng.beta(0.6, 2.0, n)
    if ties:
        p = np.round(p, 2)
    return p


def _reference_bh(p):
    return multipletests(p, method="fdr_bh")[1]


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("ties", [False, True])
def test_adjust_matches_statsmodels_bh(seed, ties):
    p = _random_p(seed, ties=ties)
    q, _ = adjust_p_values(p, "fdr_bh")
    assert np.allclose(q, _reference_bh(p), rtol=0, atol=1e-12)
    assert np.allclose(q, stats.false_discovery_control(p), rtol=0, atol=1e-12)


def test_nan_p_values_stay_nan_and_do_not_poison_the_rest():
    p = _random_p(1, 30)
    with_nan = p.copy()
    with_nan[[3, 17]] = np.nan
    q, rejected = adjust_p_values(with_nan, "fdr_bh")
    kept = ~np.isnan(with_nan)
    assert np.isnan(q[~kept]).all()
    assert not rejected[~kept].any()
    assert np.allclose(q[kept], _reference_bh(with_nan[kept]), rtol=0, atol=1e-12)


def test_all_nan_and_empty_inputs():
    assert np.isnan(adjust_p_values([np.nan, np.nan], "fdr_bh")[0]).all()
    assert adjust_p_values([], "fdr_bh")[0].size == 0


def test_n_tests_counts_untested_as_one():
    p = _random_p(2, 25)
    q, _ = adjust_p_values(p, "fdr_bh", n_tests=100)
    assert np.allclose(q, _reference_bh(np.r_[p, np.ones(75)])[:25], atol=1e-12)
    assert np.allclose(adjust_p_values(p, "fdr_bh", n_tests=10)[0], _reference_bh(p))


@pytest.mark.parametrize("seed", range(4))
def test_holm_matches_statsmodels(seed):
    p = _random_p(seed, 20)
    assert np.allclose(adjust_p_values(p, "holm")[0],
                       multipletests(p, method="holm")[1], rtol=0, atol=1e-12)


def _legacy_gmc_bh(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    q[order] = np.minimum.accumulate((p[order] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.clip(q, 0, 1)


def _legacy_voyager_bh(pvalues):
    n = len(pvalues)
    order = sorted(range(n), key=lambda i: pvalues[i])
    adjusted = [1.0] * n
    running = 1.0
    for rank in range(n, 0, -1):
        i = order[rank - 1]
        running = min(running, pvalues[i] * n / rank)
        adjusted[i] = running
    return adjusted


@pytest.mark.parametrize("seed", range(5))
def test_migrated_callers_give_the_previous_q_values(seed):
    from spacr import gene_classes, gene_measurement_compare as gmc, hits
    from spacr import ml, ops_image_umap, ops_mitotic, voyager3

    p = _random_p(seed, 40, ties=bool(seed % 2))
    expected = _reference_bh(p)
    assert np.allclose(gmc._bh(p), _legacy_gmc_bh(p), rtol=0, atol=1e-12)
    assert np.allclose(ops_mitotic._bh(p), _legacy_gmc_bh(p), rtol=0, atol=1e-12)
    assert np.allclose(voyager3._bh(list(p)), _legacy_voyager_bh(list(p)),
                       rtol=0, atol=1e-12)
    assert isinstance(voyager3._bh(list(p)), list)
    for got in (gene_classes.bh(p), hits.benjamini_hochberg(p),
                ops_image_umap._bh(p), ml._drug_screen_bh(p)):
        assert np.allclose(got, expected, rtol=0, atol=1e-12)
    assert np.allclose(ops_image_umap._bh(p, 90),
                       _reference_bh(np.r_[p, np.ones(50)])[:40], atol=1e-12)


def test_nan_handling_of_the_previously_poisoned_callers():
    from spacr import gene_measurement_compare as gmc, ops_mitotic, voyager3

    p = np.array([0.01, np.nan, 0.04, 0.03])
    expected = [0.03, np.nan, 0.04, 0.04]
    for got in (gmc._bh(p), ops_mitotic._bh(p), np.array(voyager3._bh(list(p)))):
        assert np.allclose(got, expected, equal_nan=True)

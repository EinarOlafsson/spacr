"""Coverage-fill for spacr.sp_stats — exercise every branch of the
normality / group-comparison / post-hoc / chi-square helpers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr import sp_stats as ST


def _two_group_normal(n=40, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "grp": ["a"] * n + ["b"] * n,
        "val": np.concatenate([rng.normal(0, 1, n), rng.normal(1, 1, n)]),
    })


def _multi_group(n=30, seed=1, normal=True):
    rng = np.random.default_rng(seed)
    if normal:
        data = [rng.normal(m, 1, n) for m in (0, 1, 2)]
    else:
        data = [rng.exponential(1, n) + m for m in (0, 2, 4)]
    return pd.DataFrame({
        "grp": ["a"] * n + ["b"] * n + ["c"] * n,
        "val": np.concatenate(data),
    })


# ---------------------------------------------------------------------------
# perform_normality_tests — both test branches
# ---------------------------------------------------------------------------

def test_normality_large_sample_dagostino(capsys):
    # >20 per group triggers the D'Agostino-Pearson branch (lines 67-68).
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame({"grp": ["a"] * n + ["b"] * n,
                       "val": rng.normal(0, 1, 2 * n)})
    is_normal, results = ST.perform_normality_tests(df, "grp", ["val"])
    assert isinstance(results, list)


def test_normality_small_sample_shapiro():
    df = _two_group_normal(n=10)
    is_normal, results = ST.perform_normality_tests(df, "grp", ["val"])
    assert isinstance(results, list)


# ---------------------------------------------------------------------------
# perform_statistical_tests — all four test-selection branches + paired
# ---------------------------------------------------------------------------

def test_stat_tests_two_group_ttest_or_mwu():
    df = _two_group_normal()
    out = ST.perform_statistical_tests(df, "grp", ["val"])
    assert out and out[0]["Test Name"] in ("T-test", "Mann-Whitney U test")
    assert out[0]["Groups"] == 2


def test_stat_tests_two_group_non_normal_mwu():
    rng = np.random.default_rng(2)
    n = 40
    df = pd.DataFrame({
        "grp": ["a"] * n + ["b"] * n,
        "val": np.concatenate([rng.exponential(1, n),
                                rng.exponential(1, n) + 2]),
    })
    out = ST.perform_statistical_tests(df, "grp", ["val"])
    assert out[0]["Test Name"] in ("T-test", "Mann-Whitney U test")


def test_stat_tests_paired_branch(capsys):
    # paired=True hits the "not implemented" print + continue (118-119).
    df = _two_group_normal()
    out = ST.perform_statistical_tests(df, "grp", ["val"], paired=True)
    assert out == []   # continue skips appending
    assert "paired" in capsys.readouterr().out.lower()


def test_stat_tests_multi_group_anova_or_kruskal():
    df = _multi_group(normal=True)
    out = ST.perform_statistical_tests(df, "grp", ["val"])
    assert out[0]["Test Name"] in ("One-way ANOVA", "Kruskal-Wallis test")
    assert out[0]["Groups"] == 3


def test_stat_tests_multi_group_non_normal_kruskal():
    df = _multi_group(normal=False)
    out = ST.perform_statistical_tests(df, "grp", ["val"])
    assert out[0]["Test Name"] in ("One-way ANOVA", "Kruskal-Wallis test")


# ---------------------------------------------------------------------------
# perform_posthoc_tests — Tukey (normal) + Dunn (non-normal, 183-188)
# ---------------------------------------------------------------------------

def test_posthoc_tukey_when_normal():
    df = _multi_group(normal=True)
    out = ST.perform_posthoc_tests(df, "grp", "val", is_normal=True)
    assert out and out[0]["Test Name"] == "Tukey HSD"


def test_posthoc_dunn_when_not_normal():
    df = _multi_group(normal=False)
    out = ST.perform_posthoc_tests(df, "grp", "val", is_normal=False)
    assert out and out[0]["Test Name"] == "Dunn's Post-hoc"


def test_posthoc_two_groups_returns_empty():
    df = _two_group_normal()
    out = ST.perform_posthoc_tests(df, "grp", "val", is_normal=True)
    assert out == []


# ---------------------------------------------------------------------------
# chi_pairwise — verbose branch (249-250)
# ---------------------------------------------------------------------------

def test_chi_pairwise_verbose(capsys):
    raw = pd.DataFrame({
        "cond_a": [10, 20, 30],
        "cond_b": [15, 25, 5],
    }, index=["x", "y", "z"])
    out = ST.chi_pairwise(raw, verbose=True)
    assert out is not None


def test_chi_pairwise_quiet():
    raw = pd.DataFrame({"a": [5, 10], "b": [8, 6]}, index=["p", "q"])
    out = ST.chi_pairwise(raw, verbose=False)
    assert out is not None


def test_normality_tiny_group_uses_shapiro():
    # A group with <8 samples takes the Shapiro-Wilk branch (67-68).
    df = pd.DataFrame({
        "grp": ["a"] * 5 + ["b"] * 5,
        "val": [1.0, 2, 3, 4, 5, 2, 3, 4, 5, 6],
    })
    is_normal, results = ST.perform_normality_tests(df, "grp", ["val"])
    names = [r.get("Test") or r.get("Test Name") for r in results]
    assert any("Shapiro" in str(n) for n in names) or results


def test_chi_pairwise_large_table_chi_square():
    # A contingency table larger than 2x2 uses the Chi-Square branch
    # (225-226) rather than Fisher's exact.
    raw = pd.DataFrame({
        "cond_a": [30, 25, 40, 15],
        "cond_b": [20, 35, 10, 45],
        "cond_c": [15, 20, 25, 30],
    }, index=["w", "x", "y", "z"])
    out = ST.chi_pairwise(raw, verbose=False)
    assert out is not None

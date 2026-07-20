"""
Tests for spacr.sp_stats — statistical helpers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import spacr.sp_stats as ST


# ---------------------------------------------------------------------------
# choose_p_adjust_method: decision tree over (num_groups, num_data_points)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("groups,n,expected", [
    # <=10 comparisons and n>5 → holm
    (4, 10, "holm"),   # C(4,2)=6 comparisons, n=10 → holm
    (3, 30, "holm"),   # 3 comparisons, plenty of data → holm
    # >10 comparisons and n<=5 → fdr_bh
    (6, 5, "fdr_bh"),  # C(6,2)=15 comparisons, small n
    # <=10 comparisons but n<=5 → sidak (both n<=5 AND comparisons<=10)
    (4, 5, "sidak"),   # C(4,2)=6 comparisons, n=5 → sidak
    # >10 comparisons and n>5 → bonferroni (very conservative)
    (6, 30, "bonferroni"),  # 15 comparisons, larger n
])
def test_choose_p_adjust_method(groups, n, expected):
    assert ST.choose_p_adjust_method(groups, n) == expected


# ---------------------------------------------------------------------------
# perform_levene_test
# ---------------------------------------------------------------------------

def test_levene_returns_two_scalars_on_equal_variance(rng):
    df = pd.DataFrame({
        "group": ["a"] * 20 + ["b"] * 20,
        "x": np.concatenate([rng.normal(0, 1, 20), rng.normal(0, 1, 20)]),
    })
    stat, p = ST.perform_levene_test(df, "group", "x")
    assert isinstance(stat, float)
    assert 0.0 <= p <= 1.0


def test_levene_detects_unequal_variance(rng):
    df = pd.DataFrame({
        "group": ["a"] * 40 + ["b"] * 40,
        "x": np.concatenate([rng.normal(0, 1, 40), rng.normal(0, 5, 40)]),
    })
    stat, p = ST.perform_levene_test(df, "group", "x")
    assert p < 0.05, f"expected sig. Levene p-value, got {p}"


# ---------------------------------------------------------------------------
# perform_normality_tests — smoke: returns a list; handles tiny groups.
# ---------------------------------------------------------------------------

def test_perform_normality_tests_returns_expected_shape(rng, capsys):
    df = pd.DataFrame({
        "grp": ["a"] * 30 + ["b"] * 30,
        "val": np.concatenate([rng.normal(0, 1, 30), rng.normal(0, 1, 30)]),
    })
    out = ST.perform_normality_tests(df, "grp", ["val"])
    # The function returns either a (is_normal, results) tuple or a list;
    # accept whichever the current signature yields, but confirm results exist.
    if isinstance(out, tuple) and len(out) == 2:
        _, results = out
    else:
        results = out
    assert results and any(r.get("Column") == "val" for r in results)


def test_normality_tests_gracefully_handle_tiny_groups(rng, capsys):
    df = pd.DataFrame({
        "grp": ["a"] * 2 + ["b"] * 30,
        "val": np.concatenate([rng.normal(0, 1, 2), rng.normal(0, 1, 30)]),
    })
    out = ST.perform_normality_tests(df, "grp", ["val"])
    if isinstance(out, tuple) and len(out) == 2:
        _, results = out
    else:
        results = out
    # Tiny group 'a' (n=2) should be skipped, not raised on.
    skipped = [r for r in results if r.get("Test Name") == "Skipped"]
    assert skipped, "expected the n=2 group to be marked Skipped"


# ---------------------------------------------------------------------------
# perform_statistical_tests — end-to-end smoke on a small DataFrame
# ---------------------------------------------------------------------------

def test_perform_statistical_tests_two_group_smoke(rng, capsys):
    df = pd.DataFrame({
        "grp": ["a"] * 25 + ["b"] * 25,
        "val": np.concatenate([rng.normal(0, 1, 25), rng.normal(2, 1, 25)]),  # clearly different
    })
    results = ST.perform_statistical_tests(df, "grp", ["val"])
    assert len(results) == 1
    r = results[0]
    for key in ("Column", "Test Name", "Test Statistic", "p-value", "Groups"):
        assert key in r
    assert r["Groups"] == 2
    assert r["p-value"] < 0.05


def test_perform_statistical_tests_multi_group_smoke(rng, capsys):
    df = pd.DataFrame({
        "grp": (["a"] * 20) + (["b"] * 20) + (["c"] * 20),
        "val": np.concatenate([
            rng.normal(0, 1, 20),
            rng.normal(2, 1, 20),
            rng.normal(4, 1, 20),
        ]),
    })
    results = ST.perform_statistical_tests(df, "grp", ["val"])
    assert results[0]["Groups"] == 3


# ---------------------------------------------------------------------------
# chi_pairwise — contingency-table smoke
# ---------------------------------------------------------------------------

def test_chi_pairwise_two_by_two_uses_fisher(capsys):
    counts = pd.DataFrame(
        {"pos": [30, 5, 10], "neg": [10, 30, 20]},
        index=["a", "b", "c"],
    )
    out = ST.chi_pairwise(counts, verbose=False)
    assert isinstance(out, pd.DataFrame)
    # 3 groups → C(3,2) = 3 pairwise rows
    assert len(out) == 3
    # For 2-column contingency tables the test used is Fisher's exact
    assert (out["Test Name"] == "Fisher's Exact Test").all()
    for col in ("Group 1", "Group 2", "Test Name"):
        assert col in out.columns

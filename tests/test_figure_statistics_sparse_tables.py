"""Sparse categorical figures keep their test choice and reported p values."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr.figures.stats import _adjust, _auto_statistics, _shapiro_rows


def test_sparse_two_by_three_table_reports_reproducible_monte_carlo_test():
    """Fisher's 2x2 formula cannot be used for a sparse three-column table."""
    frame = pd.DataFrame({
        "arm": ["A", "A", "A", "B", "B", "B"],
        "outcome": ["red", "blue", "green"] * 2,
        "count": [3, 1, 0, 0, 2, 4],
    })
    first = _auto_statistics(frame, "arm", "outcome", count="count")
    second = _auto_statistics(frame, "arm", "outcome", count="count")
    row = first.loc[first.test_stage == "contingency"].iloc[0]
    assert row.test_name == "Fisher-Freeman-Halton (Monte Carlo)"
    assert row.chosen_by == "auto"
    assert "larger than 2x2" in row.reason
    assert 0 < row.p_value <= 1
    assert second.loc[second.test_stage == "contingency", "p_value"].iloc[0] == row.p_value


def test_user_chosen_chi_square_reaches_every_pair_of_three_proportions():
    """Pairwise results keep the user's test and one corrected p-value family."""
    frame = pd.DataFrame({
        "arm": ["A"] * 10 + ["B"] * 10 + ["C"] * 10,
        "hit": [1] * 5 + [0] * 5 + [1] * 7 + [0] * 3 + [1] * 3 + [0] * 7,
    })
    table = _auto_statistics(frame, "arm", "hit", test="chi-square")
    pairs = table.loc[table.test_stage == "pairwise"]
    assert set(pairs.test_name) == {"chi-square"}
    assert set(pairs.chosen_by) == {"user"}
    assert set(pairs.groups) == {"A vs B", "A vs C", "B vs C"}
    assert (pairs.p_adjusted >= pairs.p_value).all()
    assert len(table.loc[table.test_stage == "omnibus"]) == 1


def test_identical_certain_outcomes_have_no_spurious_proportion_difference():
    """A zero pooled standard error reports p=1 for identical all-hit groups."""
    frame = pd.DataFrame({"arm": ["A"] * 6 + ["B"] * 6,
                          "hit": [1] * 12})
    table = _auto_statistics(frame, "arm", "hit", test="two-proportion z")
    row = table.loc[table.test_stage == "pairwise"].iloc[0]
    assert row.test_name == "two-proportion z"
    assert row.statistic == 0
    assert row.p_value == pytest.approx(1.0)
    assert row.effect_size == 0


def test_missing_comparators_do_not_invent_an_inferential_test():
    """One-group and two-pair plots report why no comparison was possible."""
    one_group = pd.DataFrame({"arm": ["A"] * 5,
                              "value": [1.0, 2.0, 3.0, 4.0, 5.0]})
    one_proportion = pd.DataFrame({"arm": ["A"] * 6,
                                   "hit": [0, 1, 0, 1, 0, 1]})
    short_correlation = pd.DataFrame({"x": [1.0, 2.0],
                                      "y": [2.0, 4.0]})
    for frame, x, y, reason in (
            (one_group, "arm", "value", "fewer than two groups"),
            (one_proportion, "arm", "hit", "fewer than two groups"),
            (short_correlation, "x", "y", "fewer than three pairs")):
        table = _auto_statistics(frame, x, y)
        assert len(table) == 1
        assert table.iloc[0].test_name == "none"
        assert reason in table.iloc[0].reason


def test_unbalanced_groups_are_not_claimed_as_paired_without_subject_ids():
    """An uneven run falls back to independent groups and records why."""
    frame = pd.DataFrame({
        "arm": ["A"] * 5 + ["B"] * 6,
        "value": [1.0, 2.2, 2.9, 4.1, 5.2,
                  2.0, 2.8, 4.0, 5.1, 6.2, 7.0],
    })
    table = _auto_statistics(frame, "arm", "value", paired=True)
    pairwise = table.loc[table.test_stage == "pairwise"]
    assert len(pairwise) == 1
    assert pairwise.iloc[0].test_name not in ("paired t", "Wilcoxon signed-rank")
    assert "treated as independent" in pairwise.iloc[0].reason


def test_pairwise_contingency_skips_a_pair_with_only_one_outcome():
    """Two arms with identical outcome categories lack a 2-column table."""
    frame = pd.DataFrame({
        "arm": ["A", "A", "B", "B", "C", "C"],
        "outcome": ["red", "blue"] * 3,
        "count": [5, 0, 5, 0, 0, 5],
    })
    table = _auto_statistics(frame, "arm", "outcome", count="count")
    pairs = table.loc[table.test_stage == "pairwise"]
    assert set(pairs.groups) == {"A vs C", "B vs C"}
    assert "A vs B" not in set(pairs.groups)


def test_sparse_two_group_proportions_use_fishers_exact_test():
    """A rare hit is tested exactly, with no normal approximation."""
    frame = pd.DataFrame({"arm": ["A"] * 6 + ["B"] * 6,
                          "hit": [1] * 3 + [0] * 3 + [1] + [0] * 5})
    table = _auto_statistics(frame, "arm", "hit")
    row = table.loc[table.test_stage == "pairwise"].iloc[0]
    assert row.test_name == "Fisher's exact"
    assert row.chosen_by == "auto"
    assert row.p_value == pytest.approx(0.5454545454545454)


def test_degenerate_or_invalid_categorical_tables_explain_the_refusal():
    """A one-row table gets no test; an impossible count is not reported as p."""
    one_row = pd.DataFrame({"arm": ["A", "A"],
                            "outcome": ["red", "blue"]})
    no_test = _auto_statistics(one_row, "arm", "outcome")
    assert no_test.iloc[0].test_name == "none"
    assert "at least two rows" in no_test.iloc[0].reason

    negative = pd.DataFrame({
        "arm": ["A", "A", "B", "B"],
        "outcome": ["red", "blue"] * 2,
        "count": [-1, 3, 2, 4],
    })
    refused = _auto_statistics(negative, "arm", "outcome", count="count")
    assert refused.iloc[0].test_name == "refused"
    assert "nonnegative" in refused.iloc[0].reason.lower()


def test_numeric_positions_with_binary_outcome_have_no_supported_test():
    """A binary outcome against a numeric x-axis is not a group comparison."""
    frame = pd.DataFrame({"position": [1.0, 2.0, 3.0, 4.0],
                          "hit": [0, 1, 0, 1]})
    table = _auto_statistics(frame, "position", "hit")
    assert table.iloc[0].test_name == "none"
    assert "no pair of variables" in table.iloc[0].reason


def test_constant_samples_and_one_p_value_do_not_gain_false_certainty():
    """A constant sample has no Shapiro p; one hypothesis needs no correction."""
    rows = _shapiro_rows([("constant", np.ones(5)),
                          ("too small", np.array([1.0, 2.0]))])
    assert [row["groups"] for row in rows] == ["constant", "too small"]
    assert all("no spread" in row["reason"] for row in rows)
    assert all(np.isnan(row["p_value"]) for row in rows)
    adjusted, method = _adjust([0.03], "fdr_bh")
    assert adjusted.tolist() == [0.03] and method == ""


def test_forced_friedman_rejects_unequal_groups_without_subject_ids():
    """A repeated-measures test cannot pair observations by row order."""
    frame = pd.DataFrame({"arm": ["A"] * 5 + ["B"] * 6 + ["C"] * 5,
                          "value": list(range(5)) + list(range(6)) +
                                   list(range(2, 7))})
    table = _auto_statistics(frame, "arm", "value", paired=True,
                             test="Friedman")
    row = table.loc[table.test_stage == "omnibus"].iloc[0]
    assert row.test_name == "Friedman" and row.chosen_by == "user"
    assert "sizes differ" in row.reason
    assert np.isnan(row.p_value)


def test_user_can_choose_pairwise_test_after_three_group_omnibus():
    """A chosen post-hoc test is recorded instead of silently changed."""
    frame = pd.DataFrame({"arm": ["A"] * 6 + ["B"] * 6 + ["C"] * 6,
                          "value": [1, 2, 3, 4, 5, 6,
                                    2, 3, 4, 5, 6, 7,
                                    3, 4, 5, 6, 7, 8]})
    table = _auto_statistics(frame, "arm", "value", test="Mann-Whitney U")
    pairs = table.loc[table.test_stage == "pairwise"]
    assert len(pairs) == 3
    assert set(pairs.test_name) == {"Mann-Whitney U"}
    assert set(pairs.chosen_by) == {"user"}

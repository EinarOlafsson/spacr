"""``chi_pairwise`` on the degenerate contingency tables a real screen produces.

Both of these crashed, and both are routine for a sparse per-well table — a
plate where one condition simply never produced a 16-parasite vacuole, or a
run with a single condition:

  * fewer than two groups -> the p-value correction got an empty list and
    raised ``ZeroDivisionError``
  * a category no group observed, or a group with no observations ->
    ``chi2_contingency`` computes a zero expected frequency and raises
    ``ValueError``

Found by the replication assay, which had to guard the call site because it
could not touch this file.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr.sp_stats import chi_pairwise

COLUMNS = ['Group 1', 'Group 2', 'Test Name', 'p-value', 'p-value_adj',
           'adj', 'note']


def _counts(rows, index, columns=(1, 2, 4)):
    return pd.DataFrame(rows, index=index, columns=list(columns))


# ---------------------------------------------------------------------------
# The two crashes
# ---------------------------------------------------------------------------

def test_a_single_group_returns_an_empty_frame_not_zero_division():
    out = chi_pairwise(_counts([[5, 3, 2]], ["only"]))
    assert list(out.columns) == COLUMNS
    assert len(out) == 0


def test_no_groups_at_all_returns_an_empty_frame():
    out = chi_pairwise(pd.DataFrame(columns=[1, 2, 4]))
    assert len(out) == 0


def test_a_category_neither_group_observed_is_dropped_not_fatal():
    """Zero on both sides carries no information about this pair, and it is
    exactly what makes the expected frequency zero."""
    out = chi_pairwise(_counts([[5, 0, 2], [4, 0, 3]], ["a", "b"]))
    assert len(out) == 1
    assert np.isfinite(out.loc[0, "p-value"])
    assert "1 empty categor" in out.loc[0, "note"]


def test_dropping_an_empty_category_gives_the_same_p_as_omitting_it():
    with_empty = chi_pairwise(_counts([[5, 0, 2, 9], [4, 0, 3, 7]],
                                      ["a", "b"], columns=(1, 2, 4, 8)))
    without = chi_pairwise(_counts([[5, 2, 9], [4, 3, 7]],
                                   ["a", "b"], columns=(1, 4, 8)))
    assert with_empty.loc[0, "p-value"] == pytest.approx(without.loc[0, "p-value"])


def test_a_group_with_no_observations_is_reported_not_crashed():
    out = chi_pairwise(_counts([[5, 1, 2], [0, 0, 0]], ["a", "b"]))
    assert len(out) == 1
    assert out.loc[0, "Test Name"] == "not testable"
    assert np.isnan(out.loc[0, "p-value"])
    assert "no observations for b" in out.loc[0, "note"]


def test_two_groups_that_only_share_one_category_are_not_testable():
    """One surviving column means there is nothing to compare."""
    out = chi_pairwise(_counts([[5, 0, 0], [3, 0, 0]], ["a", "b"]))
    assert out.loc[0, "Test Name"] == "not testable"
    assert "fewer than two categories" in out.loc[0, "note"]


# ---------------------------------------------------------------------------
# The ordinary path is unchanged
# ---------------------------------------------------------------------------

def test_two_by_two_uses_fishers_exact():
    out = chi_pairwise(_counts([[10, 2], [3, 12]], ["a", "b"], columns=(1, 2)))
    assert out.loc[0, "Test Name"] == "Fisher's Exact Test"
    assert out.loc[0, "p-value"] < 0.01


def test_larger_tables_use_chi_square():
    out = chi_pairwise(_counts([[30, 10, 2], [3, 12, 28]], ["a", "b"]))
    assert out.loc[0, "Test Name"] == "Pairwise Chi-Square Test"
    assert out.loc[0, "p-value"] < 0.01


def test_identical_distributions_do_not_reject():
    out = chi_pairwise(_counts([[20, 20, 20], [20, 20, 20]], ["a", "b"]))
    assert out.loc[0, "p-value"] == pytest.approx(1.0)


def test_three_groups_give_three_pairs_and_a_correction():
    out = chi_pairwise(_counts([[30, 10, 2], [3, 12, 28], [15, 15, 15]],
                               ["a", "b", "c"]))
    assert len(out) == 3
    assert out["adj"].nunique() == 1
    # a correction can only raise a p-value
    assert (out["p-value_adj"] >= out["p-value"] - 1e-12).all()


def test_untestable_pairs_do_not_inflate_the_correction_family():
    """Correcting across tests that never ran would penalise the real
    comparisons for nothing."""
    three = chi_pairwise(_counts([[30, 10, 2], [3, 12, 28], [15, 15, 15]],
                                 ["a", "b", "c"]))
    # same three groups plus a fourth that has no observations at all
    four = chi_pairwise(_counts([[30, 10, 2], [3, 12, 28], [15, 15, 15],
                                 [0, 0, 0]], ["a", "b", "c", "d"]))
    real = four[four["Test Name"] != "not testable"].reset_index(drop=True)
    assert len(real) == 3
    assert len(four) == 6                     # the 3 d-pairs are reported
    if three.loc[0, "adj"] == real.loc[0, "adj"]:
        assert real["p-value_adj"].tolist() == pytest.approx(
            three["p-value_adj"].tolist())


def test_untestable_pairs_carry_a_nan_adjusted_p():
    out = chi_pairwise(_counts([[30, 10, 2], [0, 0, 0]], ["a", "b"]))
    assert np.isnan(out.loc[0, "p-value_adj"])


def test_the_note_column_is_empty_for_an_ordinary_pair():
    out = chi_pairwise(_counts([[30, 10, 2], [3, 12, 28]], ["a", "b"]))
    assert out.loc[0, "note"] == ""


def test_verbose_prints_something_for_a_single_group(capsys):
    chi_pairwise(_counts([[5, 3, 2]], ["only"]), verbose=True)
    assert "no pair to compare" in capsys.readouterr().out


def test_column_order_is_stable():
    out = chi_pairwise(_counts([[30, 10, 2], [3, 12, 28]], ["a", "b"]))
    assert list(out.columns) == COLUMNS

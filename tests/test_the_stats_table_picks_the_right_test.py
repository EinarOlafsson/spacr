"""The test is chosen from the data, and the choice is defensible.

Asked for on 2026-08-16: "a stats table is generated with the correct stats
( automatic detection of equal variance nr of groups ...".

THE CHOICES ARE MECHANICAL:

    groups  variance   distribution   test
    2       equal      ~normal        Student's t
    2       unequal    ~normal        Welch's t
    2       any        not normal     Mann-Whitney U
    >2      equal      ~normal        one-way ANOVA
    >2      unequal    ~normal        Welch's ANOVA
    >2      any        not normal     Kruskal-Wallis

TWO WAYS TO GET THIS QUIETLY WRONG, and neither raises -- both hand back a
confident number. Both are pinned here.

1. THE ASSUMPTION TESTS ARE THEMSELVES TESTS. On n = 3, Levene has almost no
   power: "p = 0.7, variances equal" means "we could not tell". Reading that
   as licence to use Student's t is how a screen publishes a difference that
   is not there.

2. THE MINIMUM OF k TESTS IS NOT A p-VALUE. Checking normality per group and
   comparing the worst to 0.05 rejects on 18% of four-group comparisons over
   perfectly normal data. Measured, on this code, before it was corrected.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from spacr.figures.stats import (CONVENTION, MIN_N_FOR_ASSUMPTIONS,
                                 check_equal_variance, check_normality,
                                 compare, stars, table)


def _normal(rng, n=40, sd=1.0, mean=0.0):
    return rng.normal(mean, sd, n)


# --------------------------------------------------------------------------- #
#  The table of choices
# --------------------------------------------------------------------------- #

def test_two_normal_equal_variance_groups_get_a_students_t():
    rng = np.random.default_rng(3)
    result = compare({"a": _normal(rng, 80), "b": _normal(rng, 80, mean=.6)})
    assert result.test == "Student's t"


def test_two_normal_unequal_variance_groups_get_welch():
    rng = np.random.default_rng(3)
    result = compare({"a": _normal(rng, 80), "b": _normal(rng, 80, sd=6)})
    assert result.test == "Welch's t"


def test_a_skewed_pair_gets_mann_whitney():
    rng = np.random.default_rng(3)
    result = compare({"a": rng.exponential(1, 80),
                      "b": rng.exponential(2, 80)})
    assert result.test == "Mann-Whitney U"


def test_several_normal_unequal_variance_groups_get_welch_anova():
    rng = np.random.default_rng(11)
    groups = {name: _normal(rng, 60, sd=sd)
              for name, sd in zip("abcd", (1, 1, 5, 5))}
    assert compare(groups).test == "Welch's ANOVA"


def test_several_skewed_groups_get_kruskal():
    rng = np.random.default_rng(3)
    groups = {name: rng.exponential(scale, 60)
              for name, scale in zip("abcd", (1, 1.5, 2, 2.5))}
    assert compare(groups).test == "Kruskal-Wallis"


def test_paired_data_gets_a_paired_test():
    rng = np.random.default_rng(5)
    before = _normal(rng, 60)
    result = compare({"before": before, "after": before + rng.normal(.4, .2, 60)},
                     paired=True)
    assert "paired" in result.test or "Wilcoxon" in result.test


# --------------------------------------------------------------------------- #
#  Trap 1: an assumption test with no power
# --------------------------------------------------------------------------- #

def test_three_replicates_do_not_license_a_students_t():
    """The field standard is n = 3. Levene cannot see anything there, and
    "did not reject" is not "equal"."""
    result = compare({"wt": [1.0, 1.1, 0.9], "ko": [2.0, 2.1, 1.9]},
                     unit="well")

    assert result.test != "Student's t", (
        "an assumption test with no power was read as the assumption holding")
    assert result.test == "Mann-Whitney U"


def test_the_uninformative_check_says_so_rather_than_passing():
    small = [np.array([1.0, 1.1, 0.9]), np.array([2.0, 2.1, 1.9])]

    variance = check_equal_variance(small)
    normality = check_normality(small)

    for check in (variance, normality):
        assert check.informative is False
        assert check.passed is False
        assert "too few" in check.verdict


def test_a_check_that_could_not_see_is_not_a_check_that_passed():
    """`informative` and `passed` are different questions and the second must
    never be inferred from a NaN p-value."""
    small = [np.array([1.0, 2.0, 3.0]), np.array([1.5, 2.5, 3.5])]
    check = check_equal_variance(small)

    assert np.isnan(check.p_value)
    assert check.passed is False


# --------------------------------------------------------------------------- #
#  Trap 2: the minimum of k tests is not a p-value
# --------------------------------------------------------------------------- #

def test_normal_groups_are_not_sent_to_a_rank_test_by_luck():
    """MEASURED, because this was wrong and looked fine.

    Checking normality per group and comparing the WORST to 0.05 rejected on
    ~18% of four-group comparisons over perfectly normal data -- the caller
    re-derived the decision from the p-value and threw away the Bonferroni
    correction the check had applied. The parametric branch was nearly dead
    code and nothing said so.
    """
    rng = np.random.default_rng(1)
    chosen = Counter()
    for _ in range(200):
        chosen[compare({k: _normal(rng, 40) for k in "abcd"}).test] += 1

    rank = chosen["Kruskal-Wallis"]
    assert rank <= 200 * 0.12, (
        f"{rank}/200 normal-data comparisons went to a rank test; the "
        f"correction promises about 5%")
    assert chosen["one-way ANOVA"] > 150, chosen


def test_the_caller_reads_the_verdict_rather_than_the_p_value():
    """Named so that re-deriving `p_value >= 0.05` in compare() fails here."""
    import ast
    import inspect

    from spacr.figures import stats as module

    source = inspect.getsource(module.compare)
    tree = ast.parse(source.lstrip())
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            text = ast.dump(node)
            assert not ("p_value" in text and "0.05" in text), (
                "compare() is re-deriving an assumption from its p-value; "
                "read Assumption.passed instead")


# --------------------------------------------------------------------------- #
#  What a reportable result carries
# --------------------------------------------------------------------------- #

def test_a_result_is_never_a_bare_p_value():
    rng = np.random.default_rng(2)
    result = compare({"a": _normal(rng, 60), "b": _normal(rng, 60, mean=.7)},
                     unit="well")

    sentence = result.sentence()
    assert result.test in sentence
    assert "n=60" in sentence
    assert CONVENTION in sentence
    assert np.isfinite(result.effect_size)
    assert result.ci is not None


def test_small_samples_get_hedges_not_cohen():
    """Cohen's d is biased upward on the replicate counts this field uses."""
    rng = np.random.default_rng(2)
    result = compare({"a": _normal(rng, 12), "b": _normal(rng, 12, mean=1)})
    assert result.effect_name == "Hedges' g"


def test_the_unit_of_replication_is_stated():
    """Testing across CELLS when the replicate is the WELL is
    pseudoreplication and returns p < 1e-10 on noise."""
    rng = np.random.default_rng(2)
    result = compare({"a": _normal(rng, 30), "b": _normal(rng, 30)},
                     unit="well")

    assert result.unit == "well"
    assert "wells" in result.sentence()


def test_the_table_corrects_across_the_comparisons():
    """Six pairwise tests at 0.05 is a 26% chance of one false positive, and
    the individual p-values give no hint of it."""
    rng = np.random.default_rng(4)
    comparisons = [compare({"a": _normal(rng, 40), "b": _normal(rng, 40)})
                   for _ in range(6)]

    frame = table(comparisons)

    assert (frame["p_adjusted"] >= frame["p_value"] - 1e-12).all()
    assert set(frame["correction"]) == {"fdr_bh"}


def test_the_table_shows_its_working():
    rng = np.random.default_rng(4)
    frame = table([compare({"a": _normal(rng, 40), "b": _normal(rng, 40)})])

    for column in ("test", "n", "unit", "effect_size", "why_this_test",
                   "levene_p", "levene_verdict", "shapiro_verdict"):
        assert column in frame.columns, column


# --------------------------------------------------------------------------- #
#  Refusals and edges
# --------------------------------------------------------------------------- #

def test_one_group_is_refused_not_returned_as_nan():
    """A comparison that could not be made is not a comparison with an
    unknown answer."""
    with pytest.raises(ValueError, match="at least two groups"):
        compare({"only": [1.0, 2.0, 3.0]})


def test_a_group_of_one_is_refused():
    with pytest.raises(ValueError, match="fewer than"):
        compare({"a": [1.0], "b": [1.0, 2.0, 3.0]})


def test_forcing_a_test_overrides_the_choice():
    rng = np.random.default_rng(6)
    result = compare({"a": _normal(rng, 60), "b": _normal(rng, 60)},
                     force="Mann-Whitney U")

    assert result.test == "Mann-Whitney U"
    assert "forced" in result.reason


def test_non_significant_is_written_out_not_omitted():
    """A missing bracket reads as a comparison nobody made."""
    assert stars(0.4) == "n.s."
    assert stars(float("nan")) == "n.s."
    assert stars(0.03) == "*"
    assert stars(1e-5) == "****"

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
from scipy import stats as scipy_stats

from spacr.figures.stats import (CONVENTION, MIN_N_FOR_ASSUMPTIONS,
                                 _epsilon_squared, _eta_squared,
                                 _hedges_g, check_equal_variance,
                                 check_normality, compare, stars, table)


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


# --------------------------------------------------------------------------- #
#  A group with no spread. Found 2026-08-17 while making sp_stats delegate to
#  this module (instruction 127, finding 2), because both branches were
#  `# pragma: no cover` and neither had ever been run.
# --------------------------------------------------------------------------- #

def test_a_constant_group_is_not_declared_normal():
    """scipy returns p = 1.0 for constant input rather than raising, and 1.0
    reads as "as consistent with normal as data gets".

    An arm where the classifier called every object class 0 is a constant
    column, and this package produces those routinely. It used to PASS the
    normality check on the strength of that 1.0 and license a parametric test.
    """
    check = check_normality([np.zeros(12), np.zeros(12)])

    assert check.passed is False
    assert check.informative is False
    assert "no spread at all" in check.verdict


def test_a_constant_group_does_not_produce_a_verdict_from_a_nan():
    """Levene returns NaN for constant groups, and `nan >= 0.05` is False, so
    the check wrote "variances differ (p < 0.05)" into a results table from a
    number that does not exist.

    The branch it takes is the safe one either way. The sentence a reviewer
    reads was simply untrue, which is the part that mattered.
    """
    check = check_equal_variance([np.zeros(12), np.zeros(12)])

    assert check.passed is False
    assert check.informative is False
    assert "p < 0.05" not in check.verdict
    assert "no spread to compare" in check.verdict


def test_two_constant_groups_get_an_honest_p_rather_than_nan():
    """End to end: the pair used to come back as Welch's t with p = nan and a
    reason that claimed both assumptions had been checked. It is a rank test
    on tied data now, and p = 1.0 is the true answer for two identical arms.
    """
    result = compare({"a": np.zeros(12), "b": np.zeros(12)})

    assert result.test == "Mann-Whitney U"
    assert result.p_value == 1.0
    assert "no spread" in result.reason


def test_levene_on_a_single_group_says_so_instead_of_raising():
    """`check_equal_variance` is public and `sp_stats.perform_levene_test`
    hands it whatever groups a frame contains -- one condition included, where
    scipy raises "Must enter at least two input sample vectors".
    """
    check = check_equal_variance([np.arange(12.0)])

    assert check.passed is False
    assert check.informative is False
    assert check.verdict == "could not be computed"
    assert np.isnan(check.statistic) and np.isnan(check.p_value)


def test_the_worst_group_reports_its_own_statistic():
    """`worst_p` started at 1.0, so a group whose Shapiro p came back exactly
    1.0 never recorded its statistic and the result carried a NaN statistic
    beside a real p-value. Perfectly regular data is where p reaches 1.0.
    """
    quantiles = (np.arange(1, 41) - 0.5) / 40
    perfect = scipy_stats.norm.ppf(quantiles)
    check = check_normality([perfect])

    assert check.passed is True
    assert np.isfinite(check.statistic)
    assert check.p_value == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------- #
#  Edges the engine's own suite had never driven, closed 2026-08-17 when
#  sp_stats was folded onto this module and the file's coverage was measured.
# --------------------------------------------------------------------------- #

def test_a_p_value_that_is_not_a_number_reads_as_not_significant():
    """A p column arriving as text or None must not put stars on a figure.

    Screen result frames reach the panels with missing p-values in them --
    a comparison that was never run, a column read back from a CSV as an
    empty string -- and the alternative to "n.s." is a TypeError inside the
    renderer.
    """
    assert stars("not a number") == "n.s."
    assert stars(None) == "n.s."


def test_a_paired_skewed_pair_gets_the_signed_rank_test():
    """Paired and NOT normal is the Wilcoxon signed-rank cell of the table;
    a paired t on skewed differences is the same mistake as an unpaired one.
    """
    rng = np.random.default_rng(11)
    before = rng.exponential(1, 40)
    result = compare({"before": before, "after": before + rng.exponential(2, 40)},
                     paired=True)

    assert result.test == "Wilcoxon signed-rank"


def test_forcing_a_test_the_engine_cannot_run_is_refused():
    """The override is for a reader with a reason the data cannot express, not
    for a free-text field: a typo has to fail loudly rather than silently draw
    whatever the last branch happened to compute.
    """
    rng = np.random.default_rng(11)
    with pytest.raises(ValueError, match="unknown test"):
        compare({"a": _normal(rng, 40), "b": _normal(rng, 40)},
                force="Studentz t")


def test_the_legend_line_reports_the_adjusted_p_once_it_exists():
    """After `table` corrects across the comparisons, the sentence must quote
    the corrected number and name the correction -- reporting the raw p beside
    a corrected figure is how a family-wise claim gets made by accident.
    """
    rng = np.random.default_rng(12)
    comparisons = [compare({"a": _normal(rng, 40), "b": _normal(rng, 40, mean=m)})
                   for m in (0.0, 0.8)]
    table(comparisons)

    sentence = comparisons[0].sentence()
    assert "adjusted p" in sentence
    assert "fdr_bh" in sentence


def test_an_empty_table_still_has_the_columns_a_reader_expects():
    """A screen with nothing to compare writes a header, not a zero-column
    frame that breaks whatever reads the CSV back.
    """
    frame = table([])

    assert len(frame) == 0
    assert {"test", "groups", "n", "p_value", "effect_size"} <= set(frame.columns)


def test_a_group_that_is_all_missing_could_not_be_computed():
    """Shapiro returns NaN rather than raising for an all-missing group, and
    NaN is not a p-value: without the finite guard the comparison would report
    a verdict built from it.
    """
    check = check_normality([np.full(12, np.nan)])

    assert check.passed is False
    assert check.informative is False
    assert check.verdict == "could not be computed"


@pytest.mark.parametrize("helper,arguments,expected_name", [
    (_hedges_g, (np.array([1.0]), np.array([1.0, 2.0])), "Cohen's d"),
    (_epsilon_squared, ([np.array([1.0]), np.array([2.0])], 1.0),
     "epsilon squared"),
    (_eta_squared, ([np.zeros(3), np.zeros(3)],), "eta squared"),
], ids=["hedges g", "epsilon squared", "eta squared"])
def test_an_effect_size_with_no_denominator_is_nan_and_still_named(
        helper, arguments, expected_name):
    """One observation, one group per observation, or no variance at all: the
    effect size is undefined and must come back as NaN rather than a division
    result.

    Driven directly because `compare` refuses these inputs before the effect
    size is reached -- which is the right order, and is also why the guards
    had never been run.
    """
    value, name = helper(*arguments)

    assert np.isnan(value)
    assert name == expected_name


def test_a_group_scipy_refuses_is_skipped_and_the_rest_still_decide(monkeypatch):
    """One group scipy cannot test must not take the whole check down with it.

    The Bonferroni divisor is the tell: it counts the groups that were actually
    tested, so a skipped group makes the surviving check STRICTER rather than
    silently correcting for a test that never ran. Driven with a stand-in
    because no cleaned, above-floor float array makes scipy's shapiro raise --
    which is why this branch had never been run.
    """
    from scipy import stats as real_scipy_stats

    real_shapiro = real_scipy_stats.shapiro

    def refuses_the_negative_group(values):
        if values[0] < 0:
            raise RuntimeError("scipy refused this group")
        return real_shapiro(values)

    monkeypatch.setattr(real_scipy_stats, "shapiro", refuses_the_negative_group)

    quantiles = (np.arange(1, 41) - 0.5) / 40
    normal = scipy_stats.norm.ppf(quantiles)          # starts negative
    skewed = -np.log(1.0 - quantiles)                 # strictly positive
    check = check_normality([normal, skewed])

    assert check.informative is True
    assert "worst of 1 group(s)" in check.verdict
    assert check.passed is False

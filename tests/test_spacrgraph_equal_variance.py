"""Levene's test was computed, discarded, and its assumption used anyway.

`spacrGraph.__call__` bound ``levene_stat, levene_p`` and never read either,
then `perform_statistical_tests` called `ttest_ind` -- whose ``equal_var``
defaults to True -- and `f_oneway`, which is the equal-variance ANOVA. So the
software tested the equal-variance assumption, threw the answer away, and ran
the test that depends on it regardless.

That is not a cosmetic gap. Student's t-test on groups with unequal spread
reports a p-value that is too small when the smaller group is the more
variable one, which in a screen is the treated arm more often than not. The
figure then carries a star it has not earned.

The fix routed Levene's verdict into the test it was computed for, per
column, and named the test that actually ran so the figure legend cannot
claim Student's when Welch's was used.

WHAT CHANGED ON 2026-08-17, instruction 127 finding 2, the half left open:
`spacrGraph` no longer runs its own checks at all. It is a translation layer
onto :func:`spacr.figures.stats.compare`, the one engine, and every test here
now drives `perform_statistical_tests` -- the thing that actually decides --
rather than the private `_equal_variance` helper, which is gone. Two
consequences show up below:

* An assumption check with no power counts as FAILED. Levene on three
  replicates cannot reject, and reading "did not reject" as "variances are
  equal" is how a screen reports a difference that is not there.
* The n at which the two t-tests are compared had to move from 8 up to 12,
  because below ten observations the engine will not run a t-test at all.
  Student's p on that data is 0.0498 and Welch's is 0.4510 -- the unearned
  star, on the correct side of 0.05, in one number.
"""

import numpy as np
import pandas as pd
import pytest

from spacr.plot import _welch_anova, spacrGraph


def frame(groups, column="value", grouping="condition"):
    rows = []
    for name, values in groups.items():
        rows.extend({grouping: name, column: float(v)} for v in values)
    return pd.DataFrame(rows)


def graph_for(df, column="value", grouping="condition"):
    """A spacrGraph that will not try to draw anything."""
    return spacrGraph(df, grouping_column=grouping, data_column=[column])


# ---------------------------------------------------------------------------
# the verdict reaches the test
# ---------------------------------------------------------------------------

def test_wildly_unequal_spread_selects_welch():
    rng = np.random.default_rng(0)
    df = frame({"ctrl": rng.normal(0.0, 0.05, 40),
                "treated": rng.normal(0.1, 3.0, 40)})
    graph = graph_for(df)

    results = graph.perform_statistical_tests(["ctrl", "treated"], True)
    assert results[0]["Test Name"] == "Welch's T-test"
    assert "variances differ" in results[0]["Why This Test"], (
        "the row has to say which assumption sent it to Welch's")


def test_matched_spread_keeps_student():
    """The historical test stays the answer when the assumption holds."""
    rng = np.random.default_rng(1)
    df = frame({"ctrl": rng.normal(0.0, 1.0, 60),
                "treated": rng.normal(0.4, 1.0, 60)})
    graph = graph_for(df)

    results = graph.perform_statistical_tests(["ctrl", "treated"], True)
    assert results[0]["Test Name"] == "T-test"


def test_the_two_tests_actually_disagree_on_this_data():
    """Otherwise the fix would be untested however the name came out.

    Unequal spread AND unequal n is the case Student's gets wrong, so it is
    the case worth pinning: the p-values must differ, and Student's must be
    the smaller -- the unearned star this change removes.

    The small arm carries 12, not the 8 this test used before 2026-08-17.
    Eight is below :data:`spacr.figures.stats.MIN_N_FOR_ASSUMPTIONS`, so the
    engine now refuses to run any t-test on it and the comparison this test
    exists to make cannot be made there. Twelve is the smallest round number
    above the floor, and on it Student's lands at 0.0498 -- a starred bar --
    where Welch's lands at 0.4510.
    """
    from scipy.stats import ttest_ind

    rng = np.random.default_rng(2)
    small_loud = rng.normal(0.6, 3.0, 12)
    big_quiet = rng.normal(0.0, 0.4, 80)
    df = frame({"treated": small_loud, "ctrl": big_quiet})

    student = ttest_ind(small_loud, big_quiet, equal_var=True)[1]
    welch = ttest_ind(small_loud, big_quiet, equal_var=False)[1]
    assert student < 0.05 < welch, (
        "pick data where the correction crosses the significance line")

    graph = graph_for(df)
    results = graph.perform_statistical_tests(["treated", "ctrl"], True)
    assert results[0]["Test Name"] == "Welch's T-test"
    assert results[0]["p-value"] == pytest.approx(welch)


def test_the_verdict_is_taken_per_column_not_from_the_first():
    """perform_levene_test only ever looked at data_column[0].

    The test loop runs over every column, so one verdict reused across all of
    them trades one wrong assumption for another.
    """
    rng = np.random.default_rng(3)
    rows = []
    for name, (sd_a, sd_b) in {"ctrl": (1.0, 1.0), "treated": (1.0, 6.0)}.items():
        equal = rng.normal(0.0, sd_a, 50)
        unequal = rng.normal(0.0, sd_b, 50)
        rows.extend({"condition": name, "same_spread": float(x),
                     "different_spread": float(y)}
                    for x, y in zip(equal, unequal))
    df = pd.DataFrame(rows)

    graph = spacrGraph(df, grouping_column="condition",
                       data_column=["same_spread", "different_spread"])

    names = {r["Column"]: r["Test Name"]
             for r in graph.perform_statistical_tests(["ctrl", "treated"], True)}
    assert names["same_spread"] == "T-test"
    assert names["different_spread"] == "Welch's T-test"


# ---------------------------------------------------------------------------
# more than two groups
# ---------------------------------------------------------------------------

def test_three_groups_with_unequal_spread_use_welchs_anova():
    rng = np.random.default_rng(4)
    df = frame({"a": rng.normal(0.0, 0.2, 30),
                "b": rng.normal(0.3, 4.0, 30),
                "c": rng.normal(0.6, 0.9, 30)})
    graph = graph_for(df)
    results = graph.perform_statistical_tests(["a", "b", "c"], True)
    assert results[0]["Test Name"] == "Welch's ANOVA"
    assert np.isfinite(results[0]["p-value"])


def test_three_groups_with_matched_spread_keep_one_way_anova():
    rng = np.random.default_rng(5)
    df = frame({"a": rng.normal(0.0, 1.0, 40),
                "b": rng.normal(0.3, 1.0, 40),
                "c": rng.normal(0.6, 1.0, 40)})
    graph = graph_for(df)
    results = graph.perform_statistical_tests(["a", "b", "c"], True)
    assert results[0]["Test Name"] == "One-way ANOVA"


def test_welch_anova_matches_pingouin():
    """An independent implementation, because a wrong statistic that runs is
    worse than the equal-variance one it replaced."""
    pg = pytest.importorskip("pingouin")

    rng = np.random.default_rng(6)
    groups = [rng.normal(i * 0.6, sd, n)
              for i, (n, sd) in enumerate([(20, 1.0), (40, 3.0), (15, 0.5)])]
    stat, p = _welch_anova(groups)

    long = pd.DataFrame({
        "v": np.concatenate(groups),
        "g": np.concatenate([[i] * len(x) for i, x in enumerate(groups)])})
    ref = pg.welch_anova(data=long, dv="v", between="g")
    assert stat == pytest.approx(float(ref["F"][0]))
    assert p == pytest.approx(float(ref["p-unc"][0]))


# ---------------------------------------------------------------------------
# degenerate input no longer buys a parametric test
# ---------------------------------------------------------------------------
# THIS ONE REVERSED on 2026-08-17. It used to assert that a case Levene
# cannot judge KEEPS the equal-variance test, on the reasoning that "the
# equal-variance test is what these plates were measured with" and moving
# would change old numbers. Einar's instruction 127 finding 2 says the
# opposite and says why: an assumption check that could not see is not an
# assumption check that passed, and "Levene did not reject on n = 3" means
# "we could not tell". Taking the parametric branch there is how a screen
# reports a difference that is not there. Old numbers move, and that is the
# point of the change rather than an objection to it.

def test_a_group_levene_cannot_judge_does_not_buy_the_parametric_test():
    """Three constant replicates per arm cannot license Student's t."""
    groups = {"a": [1.0, 1.0, 1.0], "b": [2.0, 2.0, 2.0]}
    graph = graph_for(frame(groups))

    row = graph.perform_statistical_tests(list(groups), True)[0]
    assert row["Test Name"] == "Mann-Whitney U test", (
        "a check with no power was read as a check that passed")
    assert "too few for a variance test to have power" in row["Why This Test"]


def test_a_group_too_small_to_test_is_refused_rather_than_named():
    """One observation per arm is not a comparison with an unknown answer.

    Reporting 'T-test, p = nan' names a test that never ran, which is the
    same defect as claiming Student's when Welch's was used.
    """
    groups = {"a": [1.0], "b": [2.0]}
    graph = graph_for(frame(groups))

    row = graph.perform_statistical_tests(list(groups), True)[0]
    assert row["Test Name"] == "not testable"
    assert np.isnan(row["p-value"])
    assert "fewer than 2 usable observations" in row["Why This Test"]


def test_welch_anova_refuses_rather_than_returning_a_number():
    assert all(np.isnan(v) for v in _welch_anova([[1.0, 2.0, 3.0]]))
    assert all(np.isnan(v) for v in _welch_anova([[1.0, 1.0], [2.0, 2.0]]))


# ---------------------------------------------------------------------------
# n_object and n_well were the same number
# ---------------------------------------------------------------------------

def test_n_object_and_n_well_are_different_numbers():
    """Both used to be read off the AGGREGATED frame.

    `preprocess_data` collapses self.df to one row per well when
    representation='well'; raw_df is the copy taken before that. n_object was
    computed from grouped_data, which is built from self.df, so a plate of
    4,380 cells in 12 wells reported n_object = 12 -- the well count, under
    the object column's name.

    The post-hoc rows in the same results CSV already did it correctly, so
    the two row types disagreed about the same comparison in the same file.
    """
    rng = np.random.default_rng(0)
    rows = []
    for cond in ("nc", "pc"):
        for well in range(1, 7):
            for _ in range(365):
                rows.append({"condition": cond,
                             "prc": f"p1_r1_c{well}_{cond}",
                             "value": float(rng.normal(
                                 0.4 if cond == "pc" else 0.0, 1.0))})
    df = pd.DataFrame(rows)

    graph = spacrGraph(df, grouping_column="condition",
                       data_column=["value"], representation="well")
    row = graph.perform_statistical_tests(["nc", "pc"], True)[0]

    assert row["n_object"] == len(df) == 4380
    assert row["n_well"] == df["prc"].nunique() == 12
    assert row["n_object"] != row["n_well"], (
        "n_object is being read off the aggregated frame again")


def test_n_object_counts_only_rows_that_carry_the_measurement():
    """A NaN measurement is not an object the test used."""
    rng = np.random.default_rng(1)
    rows = []
    for cond in ("nc", "pc"):
        for well in range(1, 4):
            for i in range(40):
                value = np.nan if i < 5 else float(rng.normal())
                rows.append({"condition": cond, "prc": f"w{well}_{cond}",
                             "value": value})
    df = pd.DataFrame(rows)

    graph = spacrGraph(df, grouping_column="condition",
                       data_column=["value"], representation="well")
    row = graph.perform_statistical_tests(["nc", "pc"], True)[0]

    assert row["n_object"] == int(df["value"].notna().sum())
    assert row["n_object"] < len(df)

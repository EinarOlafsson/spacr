"""One engine decides which statistical test applies, and both doors reach it.

Instruction 127, finding 2, filed 2026-08-17: `spacr.sp_stats` and
`spacr.figures.stats` both chose a test from the data, and run on the same five
inputs they disagreed on three of them -- always in the same direction, with
`sp_stats` taking the parametric test wherever the assumption checks had no
power to refuse it.

    case                        sp_stats   was      figures.stats
    normal, equal var, n=30     T-test              Student's t
    normal, UNEQUAL var, n=30   T-test              Welch's t
    skewed (exponential) n=30   T-test              Mann-Whitney U
    n=3 vs 24  (THE REAL CASE)  T-test              Mann-Whitney U
    n=5 vs 5                    T-test              Mann-Whitney U

That is not a tidiness complaint. Two engines that disagree about which test
applies will eventually put two different p-values for one comparison into one
paper, and the screen this package was written for has three positive controls
at three replicates each -- the row where the disagreement is largest.

`sp_stats` is now a translation layer over the engine, so this file pins
something stronger than "same family": the same inputs through both entry
points must return the same test AND a bit-identical p-value. A family-level
assertion would pass again on the day someone reintroduces a second
implementation that happens to agree on the cases anyone thought to check.

THE DATA HERE IS CONSTRUCTED, NOT SAMPLED. Each group is the exact quantiles of
the distribution it is named after, so "this pair is normal" and "this pair is
skewed" are properties of the fixture rather than of a seed, and the n=3 row
says what it means to say: a perfectly normal three-replicate group still does
not license a t-test, because nothing at n=3 could have told us it was normal.
"""
from __future__ import annotations

import inspect
import re

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

from spacr import sp_stats as ST
from spacr.figures import stats as FS


def _normal_sample(n, mean=0.0, sd=1.0):
    """The exact quantiles of a normal distribution: as normal as n points get."""
    quantiles = (np.arange(1, n + 1) - 0.5) / n
    return mean + sd * scipy_stats.norm.ppf(quantiles)


def _exponential_sample(n, scale=1.0):
    """The exact quantiles of an exponential: skewed by construction."""
    quantiles = (np.arange(1, n + 1) - 0.5) / n
    return -scale * np.log(1.0 - quantiles)


#: The instruction's five rows, plus the three-group cases, with the test the
#: one engine chooses for each. Named in the ENGINE's vocabulary; `sp_stats`
#: maps them onto the spelling its CSVs have always used.
_ROWS = [
    ("normal, equal var, n=30",
     _normal_sample(30), _normal_sample(30, mean=0.8), "Student's t"),
    ("normal, UNEQUAL var, n=30",
     _normal_sample(30), _normal_sample(30, mean=0.8, sd=5), "Welch's t"),
    ("skewed (exponential) n=30",
     _exponential_sample(30), _exponential_sample(30, scale=2.5),
     "Mann-Whitney U"),
    ("n=3 vs 24",
     _normal_sample(3), _normal_sample(24, mean=0.8), "Mann-Whitney U"),
    ("n=5 vs 5",
     _normal_sample(5), _normal_sample(5, mean=0.8), "Mann-Whitney U"),
]


def _frame(groups):
    labels, values = [], []
    for name, array in groups.items():
        labels.extend([name] * len(array))
        values.append(array)
    return pd.DataFrame({"grp": labels, "val": np.concatenate(values)})


# --------------------------------------------------------------------------- #
#  The table from instruction 127, pinned down every row
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("case,a,b,expected",
                         _ROWS, ids=[row[0] for row in _ROWS])
def test_both_entry_points_run_the_same_test_on_the_same_data(case, a, b,
                                                              expected):
    """sp_stats and figures.stats cannot disagree, because there is one engine.

    Asserted on the test NAME, the p-value and the statistic together. Two of
    the three used to differ on this data; the p-values differed by enough to
    move a borderline hit across 0.05.
    """
    groups = {"A": a, "B": b}
    adapter = ST.perform_statistical_tests(_frame(groups), "grp", ["val"])[0]
    engine = FS.compare(groups)

    assert engine.test == expected, f"{case}: engine chose {engine.test}"
    assert adapter["Test Name"] == ST._ENGINE_TEST_NAMES[expected]
    assert adapter["p-value"] == engine.p_value
    assert adapter["Test Statistic"] == engine.statistic


@pytest.mark.parametrize("case,a,b,expected",
                         _ROWS, ids=[row[0] for row in _ROWS])
def test_the_adapter_reports_the_group_sizes_it_actually_tested(case, a, b,
                                                                expected):
    """n per group travels with the result, so a reader can see it was three.

    A p-value with no n beside it is the reason the n=3 rows went unnoticed:
    the saved CSV looked exactly like the n=30 one.
    """
    groups = {"A": a, "B": b}
    row = ST.perform_statistical_tests(_frame(groups), "grp", ["val"])[0]
    assert row["n"] == f"{len(a)} / {len(b)}"
    assert row["Why This Test"] == FS.compare(groups).reason


def test_three_replicates_do_not_license_a_t_test_through_sp_stats():
    """THE REAL CASE. Perfectly normal data, n=3, and still a rank test.

    This is the row that matters: the screen has three positive controls, and
    the old code ran Student's t on them because Shapiro-Wilk on three points
    cannot reject anything. "Not rejected" was read as "normal".
    """
    frame = _frame({"nc": _normal_sample(3), "pc": _normal_sample(24, mean=0.8)})
    row = ST.perform_statistical_tests(frame, "grp", ["val"])[0]

    assert row["Test Name"] == "Mann-Whitney U test"
    assert "too few for a normality test to have power" in row["Why This Test"]


def test_an_unequal_variance_pair_says_welch_and_really_is_welch():
    """The label used to read "T-test" whichever t-test ran -- and it was
    always Student's, because the old call passed no ``equal_var`` and scipy
    defaults it to True. Pinned against scipy both ways so the name and the
    arithmetic have to agree.
    """
    a, b = _normal_sample(30), _normal_sample(30, mean=0.8, sd=5)
    row = ST.perform_statistical_tests(_frame({"A": a, "B": b}),
                                       "grp", ["val"])[0]
    welch = scipy_stats.ttest_ind(a, b, equal_var=False)
    student = scipy_stats.ttest_ind(a, b, equal_var=True)

    assert row["Test Name"] == "Welch's T-test"
    assert row["p-value"] == pytest.approx(welch.pvalue, rel=1e-12)
    assert row["p-value"] != pytest.approx(student.pvalue, rel=1e-12)
    # The df is the tell that the two are genuinely different tests and not
    # two names for one call: Student's pools to n1 + n2 - 2, Welch's does not.
    assert student.df == 58.0
    assert welch.df < 58.0


@pytest.mark.parametrize("groups,expected", [
    ({"a": _normal_sample(30), "b": _normal_sample(30, mean=0.5),
      "c": _normal_sample(30, mean=1.0)}, "one-way ANOVA"),
    ({"a": _normal_sample(30), "b": _normal_sample(30, mean=0.5, sd=4),
      "c": _normal_sample(30, mean=1.0)}, "Welch's ANOVA"),
    ({"a": _exponential_sample(30), "b": _exponential_sample(30, scale=2),
      "c": _exponential_sample(30, scale=3)}, "Kruskal-Wallis"),
], ids=["equal variance", "unequal variance", "skewed"])
def test_three_groups_reach_the_same_omnibus_test_through_both_doors(groups,
                                                                     expected):
    """Welch's ANOVA exists in the engine and never existed in sp_stats, so a
    heteroscedastic three-arm screen used to get a plain one-way ANOVA.
    """
    adapter = ST.perform_statistical_tests(_frame(groups), "grp", ["val"])[0]
    engine = FS.compare(groups)

    assert engine.test == expected
    assert adapter["Test Name"] == ST._ENGINE_TEST_NAMES[expected]
    assert adapter["p-value"] == engine.p_value
    assert adapter["Groups"] == 3


# --------------------------------------------------------------------------- #
#  The vocabulary cannot drift away from the engine
# --------------------------------------------------------------------------- #

def test_every_test_the_engine_can_run_has_a_name_for_the_csv():
    """The engine names tests for a figure legend, sp_stats for a CSV column.
    Two vocabularies are only safe while one is derived from the other: add a
    test to the engine without naming it here and a screen result reaches a
    spreadsheet under a name nobody chose.
    """
    engine_names = set(re.findall(r'name == "([^"]+)"',
                                  inspect.getsource(FS._run)))
    assert engine_names, "could not read the engine's test names from _run"
    missing = engine_names - set(ST._ENGINE_TEST_NAMES)
    assert not missing, f"engine tests with no CSV name: {sorted(missing)}"
    stale = set(ST._ENGINE_TEST_NAMES) - engine_names
    assert not stale, f"CSV names for tests the engine cannot run: {sorted(stale)}"


# --------------------------------------------------------------------------- #
#  The normality verdict has one source
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("values,expected", [
    (_normal_sample(60), True),
    (_exponential_sample(60), False),
], ids=["normal", "skewed"])
def test_is_normal_is_the_engines_verdict_and_not_a_second_opinion(values,
                                                                   expected):
    """`perform_normality_tests` must return the engine's own `passed`.

    Re-deriving it from the reported p-values is what discarded the Bonferroni
    correction the check applies across groups -- the bug figures/stats records
    at :func:`check_normality`, where it sent 18% of four-group comparisons on
    perfectly normal data to a rank test.
    """
    frame = _frame({"a": values[:30], "b": values[30:]})
    is_normal, _rows = ST.perform_normality_tests(frame, "grp", ["val"])

    assert is_normal is expected
    assert is_normal is FS.check_normality([values[:30], values[30:]]).passed


def test_a_normality_row_says_when_the_check_could_not_see():
    """Three replicates get a row that says so rather than a reassuring p.

    The old code printed a Shapiro p-value for n=3 with nothing marking it as
    uninformative, which is the number a reader takes as evidence of normality.
    """
    frame = _frame({"a": _normal_sample(3), "b": _normal_sample(24)})
    _is_normal, rows = ST.perform_normality_tests(frame, "grp", ["val"])
    small = next(row for row in rows if row["n"] == 3)

    assert small["Informative"] is False
    assert "too few for a normality test to have power" in small["Verdict"]
    assert np.isnan(small["Test Statistic"])


def test_a_group_of_two_is_still_reported_as_skipped():
    """Shapiro-Wilk needs three points to have a statistic at all, and the
    caller writing normality_results.csv expects one row per group.
    """
    frame = _frame({"a": _normal_sample(2), "b": _normal_sample(30)})
    _is_normal, rows = ST.perform_normality_tests(frame, "grp", ["val"])

    assert [row["Test Name"] for row in rows] == ["Skipped", "Shapiro-Wilk"]
    assert rows[0]["n"] == 2


def test_the_verdict_covers_every_column_not_just_the_last_one():
    """`is_normal` used to be recomputed each loop and returned for the LAST
    column, so a normal column examined after a skewed one reported True and
    the caller ran a t-test on the skewed one.
    """
    frame = pd.DataFrame({
        "grp": ["a"] * 30 + ["b"] * 30,
        "skewed": np.concatenate([_exponential_sample(30),
                                  _exponential_sample(30, scale=2)]),
        "normal": np.concatenate([_normal_sample(30),
                                  _normal_sample(30, mean=0.5)]),
    })
    assert ST.perform_normality_tests(frame, "grp", ["normal"])[0] is True
    assert ST.perform_normality_tests(frame, "grp", ["skewed"])[0] is False
    assert ST.perform_normality_tests(
        frame, "grp", ["skewed", "normal"])[0] is False


def test_no_column_examined_is_not_evidence_of_normality():
    """An empty column list used to raise UnboundLocalError; answering True
    would be worse, because True is the answer that licenses a t-test.
    """
    frame = _frame({"a": _normal_sample(30), "b": _normal_sample(30)})
    is_normal, rows = ST.perform_normality_tests(frame, "grp", [])
    assert is_normal is False
    assert rows == []


# --------------------------------------------------------------------------- #
#  Levene reports what it can see
# --------------------------------------------------------------------------- #

def test_levene_is_median_centred_now():
    """Brown-Forsythe, not scipy's mean-centred default. The robust form is the
    right one precisely when normality is in question, which is every time this
    is called -- it is called before anyone knows.
    """
    a, b = _exponential_sample(40), _exponential_sample(40, scale=3)
    frame = _frame({"a": a, "b": b})
    stat, p = ST.perform_levene_test(frame, "grp", "val")

    median = scipy_stats.levene(a, b, center="median")
    mean = scipy_stats.levene(a, b, center="mean")
    assert stat == pytest.approx(median.statistic, rel=1e-12)
    assert p == pytest.approx(median.pvalue, rel=1e-12)
    assert stat != pytest.approx(mean.statistic, rel=1e-6)


def test_levene_returns_nan_rather_than_a_confident_number_on_three_wells():
    """On n=3 Levene has almost no power, so "p = 0.7, variances are equal"
    means "we could not tell". Writing 0.7 into variance_results.csv invites
    exactly the reading instruction 124 K names.
    """
    frame = _frame({"a": _normal_sample(3), "b": _normal_sample(24, sd=6)})
    stat, p = ST.perform_levene_test(frame, "grp", "val")
    assert np.isnan(stat) and np.isnan(p)


# --------------------------------------------------------------------------- #
#  A comparison that cannot be made is reported, not invented
# --------------------------------------------------------------------------- #

def test_a_single_group_is_reported_as_not_testable():
    """One group used to reach scipy's kruskal and raise ValueError out of a
    loop that was writing one CSV row per column.
    """
    frame = _frame({"only": _normal_sample(10)})
    row = ST.perform_statistical_tests(frame, "grp", ["val"])[0]

    assert row["Test Name"] == "not testable"
    assert np.isnan(row["p-value"])
    assert "at least two groups" in row["Why This Test"]


def test_a_group_of_one_is_not_silently_compared():
    """n=1 against n=24 used to come back as a Mann-Whitney "comparison" with a
    p-value. One observation cannot be compared with anything.
    """
    frame = _frame({"a": _normal_sample(1), "b": _normal_sample(24)})
    row = ST.perform_statistical_tests(frame, "grp", ["val"])[0]

    assert row["Test Name"] == "not testable"
    assert "fewer than 2 usable observations" in row["Why This Test"]
    assert row["n"] == "1 / 24"


def test_asking_for_a_paired_test_across_three_groups_no_longer_runs_an_unpaired_one(capsys):
    """`paired=True` is still unimplemented, but it used to be honoured only on
    the two-group path: three groups ran an ordinary Kruskal-Wallis and handed
    back a p-value for a test nobody asked for.
    """
    frame = _frame({"a": _normal_sample(30), "b": _normal_sample(30, mean=.5),
                    "c": _normal_sample(30, mean=1.0)})
    assert ST.perform_statistical_tests(frame, "grp", ["val"],
                                        paired=True) == []
    assert "paired" in capsys.readouterr().out.lower()

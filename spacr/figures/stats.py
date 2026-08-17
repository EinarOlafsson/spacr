"""Pick the right test from the data, and show the working.

Asked for on 2026-08-16: "a stats table is generated with the correct stats
( automatic detection of equal variance nr of groups ...".

The choice is mechanical, so the software makes it:

    groups  variance   distribution   test
    2       equal      ~normal        Student's t (two-sided)
    2       unequal    ~normal        Welch's t
    2       any        not normal     Mann-Whitney U
    >2      equal      ~normal        one-way ANOVA
    >2      unequal    ~normal        Welch's ANOVA
    >2      any        not normal     Kruskal-Wallis

TWO THINGS THAT MAKE THIS EASY TO GET QUIETLY WRONG, and neither raises an
error when you get it wrong -- they hand back a confident number instead.

**THE ASSUMPTION TESTS ARE THEMSELVES TESTS.** On n = 3 Levene has almost no
power, so "p = 0.7, variances are equal" actually means "we could not tell".
Reading that as licence to use Student's t is how a screen reports a
difference that is not there. Below :data:`MIN_N_FOR_ASSUMPTIONS` this module
records the check as UNINFORMATIVE and takes the robust branch -- Welch,
Mann-Whitney -- which costs a little power when the assumption did hold and
protects the result when it did not. That asymmetry is the whole argument:
one direction loses a bit of sensitivity, the other publishes a false
positive.

**THE UNIT OF REPLICATION.** spaCR measures thousands of cells across a
handful of wells. A test across CELLS when the replicate is the WELL is
pseudoreplication and will return p < 1e-10 on pure noise, because n is
inflated by a factor of a thousand. Every result here states what n counted,
and :func:`compare` takes a ``unit`` so a caller can aggregate first.

A p-value alone is not reportable. Every result carries the test by name, n
per group, an effect size, and the assumption checks with their own numbers.

THIS IS THE ONE ENGINE THAT CHOOSES A TEST. :mod:`spacr.sp_stats` used to
choose its own and disagreed with this one on three of five inputs, always by
taking the parametric branch where the checks had no power to refuse it
(instruction 127, finding 2). It is now a translation layer onto :func:`compare`,
:func:`check_normality` and :func:`check_equal_variance` that keeps its older
signatures and result keys. Change the choices here and both entry points move
together; ``tests/test_one_engine_decides_which_test_applies.py`` fails if they
ever come apart again.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

#: Below this many observations in a group, an assumption test has so little
#: power that failing to reject says nothing. Ten is where Shapiro-Wilk starts
#: to be able to see a clear departure; three, which is a common replicate
#: count in this field, is nowhere near it.
MIN_N_FOR_ASSUMPTIONS = 10

#: Below this many observations a group cannot be tested at all.
MIN_N_FOR_TEST = 2

#: The asterisk convention, stated once and reported with every result --
#: the skill is explicit that a bare p is not acceptable.
CONVENTION = "*p<0.05, **p<0.01, ***p<0.001, ****p<0.0001"


def stars(p) -> str:
    """The asterisks for a p-value, or ``n.s.`` written out.

    Non-significant comparisons are SHOWN rather than omitted, which is what
    the published figures do -- a missing bracket reads as a comparison
    nobody made.
    """
    try:
        p = float(p)
    except (TypeError, ValueError):
        return "n.s."
    if not np.isfinite(p):
        return "n.s."
    for threshold, mark in ((1e-4, "****"), (1e-3, "***"),
                            (1e-2, "**"), (5e-2, "*")):
        if p < threshold:
            return mark
    return "n.s."


@dataclass
class Assumption:
    """One assumption check, and whether it could see anything."""

    name: str
    statistic: float
    p_value: float
    #: False when the groups were too small for the check to have power. A
    #: check that could not see is not a check that passed.
    informative: bool
    #: What the check concluded, in words, including "could not tell".
    verdict: str
    #: WHETHER THE ASSUMPTION HOLDS. The check decides this itself and the
    #: caller reads it; it must never be re-derived from `p_value`.
    #:
    #: That is not a style preference. The normality check compares the worst
    #: of k groups against a BONFERRONI threshold, and a caller re-deriving
    #: `p_value >= 0.05` silently discards the correction: on four normal
    #: groups that sent 18% of comparisons to a rank test instead of 5%, and
    #: the parametric branch was nearly dead code. The bug was invisible
    #: because both numbers looked reasonable on their own.
    passed: bool = False


@dataclass
class Comparison:
    """One test, everything needed to report it, and how it was chosen."""

    test: str
    statistic: float
    p_value: float
    #: Group labels in the order they were tested.
    groups: Sequence[str]
    #: n per group. The unit of replication, not the row count of a frame.
    n: Sequence[int]
    #: What one observation IS -- 'well', 'cell', 'guide'. Stated because
    #: testing across the wrong one is the commonest way to get p < 1e-10 on
    #: noise.
    unit: str = "observation"
    effect_size: float = float("nan")
    effect_name: str = ""
    ci: Optional[Sequence[float]] = None
    assumptions: List[Assumption] = field(default_factory=list)
    #: Why this test and not another.
    reason: str = ""
    #: Correction applied across several comparisons, if any.
    correction: str = ""
    p_adjusted: float = float("nan")

    @property
    def marks(self) -> str:
        return stars(self.p_adjusted if np.isfinite(self.p_adjusted)
                     else self.p_value)

    def sentence(self) -> str:
        """The legend line: test, n, convention. Never a bare p."""
        counts = ", ".join(f"n={value}" for value in self.n)
        text = (f"{self.test}, {counts} {self.unit}s; "
                f"p = {self.p_value:.3g}")
        if np.isfinite(self.p_adjusted):
            text += f", adjusted p = {self.p_adjusted:.3g} ({self.correction})"
        if np.isfinite(self.effect_size):
            text += f"; {self.effect_name} = {self.effect_size:.3g}"
        if self.ci is not None:
            text += f" [{self.ci[0]:.3g}, {self.ci[1]:.3g}]"
        return text + f". {CONVENTION}."


def _clean(values) -> np.ndarray:
    array = np.asarray(values, dtype="float64")
    return array[np.isfinite(array)]


def check_normality(groups: Sequence[np.ndarray]) -> Assumption:
    """Shapiro-Wilk per group, and whether it could see anything."""
    from scipy import stats

    smallest = min((group.size for group in groups), default=0)
    if smallest < MIN_N_FOR_ASSUMPTIONS:
        return Assumption(
            "Shapiro-Wilk", float("nan"), float("nan"), False,
            f"the smallest group has {smallest} observations, too few for a "
            f"normality test to have power — treated as NOT normal, which is "
            f"the safe direction",
            passed=False)
    # A GROUP WITH NO SPREAD IS NOT A NORMAL GROUP. scipy hands back p = 1.0
    # and a NaN statistic for constant input rather than raising, and 1.0 read
    # as a p-value says "as consistent with normal as data gets" -- so a column
    # where every object in an arm was called class 0 used to PASS the
    # normality check and license a parametric test. Real input in this
    # package, not a contrived one.
    flat = [group for group in groups if float(np.ptp(group)) == 0.0]
    if flat:
        return Assumption(
            "Shapiro-Wilk", float("nan"), float("nan"), False,
            f"{len(flat)} group(s) have no spread at all, so a normality test "
            f"has nothing to describe — treated as NOT normal, which is the "
            f"safe direction",
            passed=False)
    # Start above 1.0 so the first group always records its statistic. Starting
    # AT 1.0 meant a group whose p came back exactly 1.0 never updated
    # `worst_stat`, and the result reported a NaN statistic beside a real p.
    worst_p, worst_stat, tested = float("inf"), float("nan"), 0
    for group in groups:
        try:
            statistic, p = stats.shapiro(group[:5000])
        except Exception:
            continue
        tested += 1
        if p < worst_p:
            worst_p, worst_stat = float(p), float(statistic)
    if not tested or not np.isfinite(worst_p):
        return Assumption("Shapiro-Wilk", float("nan"), float("nan"), False,
                          "could not be computed", passed=False)

    # THE MINIMUM OF k TESTS IS NOT A p-VALUE.
    #
    # Taking the worst group and comparing it to 0.05 tests normality k
    # times and reports the most extreme, which is a multiple-comparison
    # problem in the assumption check itself: with four normal groups of 40
    # there is a ~19% chance the worst one falls below 0.05 by luck, and the
    # whole comparison then flips to a rank test on data that was fine.
    #
    # Bonferroni across the groups. Conservative in the direction that
    # matters -- it makes "not normal" harder to claim, and the cost of
    # wrongly claiming it is only a little power, while the cost of the
    # opposite is a parametric test on data that does not support one.
    threshold = 0.05 / max(tested, 1)
    normal = worst_p >= threshold
    return Assumption(
        "Shapiro-Wilk", worst_stat, worst_p, True,
        f"consistent with normal across {tested} group(s)" if normal
        else (f"departs from normal (worst of {tested} group(s) "
              f"p = {worst_p:.3g} < {threshold:.3g}, Bonferroni)"),
        passed=normal)


def check_equal_variance(groups: Sequence[np.ndarray]) -> Assumption:
    """Levene, MEDIAN-centred.

    The median-centred form (Brown-Forsythe) is the robust one and is the
    right choice precisely when normality is itself in question -- which is
    every time this function is called, since it is called before we know.
    """
    from scipy import stats

    smallest = min((group.size for group in groups), default=0)
    if smallest < MIN_N_FOR_ASSUMPTIONS:
        return Assumption(
            "Levene (median-centred)", float("nan"), float("nan"), False,
            f"the smallest group has {smallest} observations, too few for a "
            f"variance test to have power — treated as UNEQUAL, so the test "
            f"below does not assume what it could not check",
            passed=False)
    try:
        # Levene's denominator is zero when every group is constant. The NaN it
        # produces is handled below, so the numpy warning on the way there is
        # noise in a caller's console, not information.
        with np.errstate(invalid="ignore", divide="ignore"):
            statistic, p = stats.levene(*groups, center="median")
    except Exception:
        return Assumption("Levene (median-centred)", float("nan"),
                          float("nan"), False, "could not be computed",
                          passed=False)
    # NaN IS NOT A SMALL p. Levene returns NaN rather than raising when every
    # group is constant (its denominator is zero), and `nan >= 0.05` is False,
    # so the check used to write "variances differ (p < 0.05)" into a results
    # table on the strength of a number that does not exist. The branch it
    # picks is the safe one either way; the sentence a reviewer reads was a
    # false statement.
    if not np.isfinite(p):
        return Assumption("Levene (median-centred)", float("nan"),
                          float("nan"), False,
                          "the groups have no spread to compare, so the "
                          "variance test has no value — treated as UNEQUAL, "
                          "so the test below does not assume what it could "
                          "not check",
                          passed=False)
    equal = float(p) >= 0.05
    return Assumption(
        "Levene (median-centred)", float(statistic), float(p), True,
        "variances consistent with equal" if equal
        else "variances differ (p < 0.05)",
        passed=equal)


def _hedges_g(a: np.ndarray, b: np.ndarray) -> tuple:
    """Standardised difference, with the small-sample correction."""
    na, nb = a.size, b.size
    if na < 2 or nb < 2:
        return float("nan"), "Cohen's d"
    pooled = np.sqrt(((na - 1) * np.var(a, ddof=1)
                      + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    if not pooled:
        return float("nan"), "Cohen's d"
    d = float((np.mean(a) - np.mean(b)) / pooled)
    total = na + nb
    if total < 50:
        # Hedges' correction. On the replicate counts this field actually
        # uses, Cohen's d is biased upward by several percent.
        return d * (1 - 3 / (4 * total - 9)), "Hedges' g"
    return d, "Cohen's d"


def _epsilon_squared(groups: Sequence[np.ndarray], statistic: float) -> tuple:
    """Effect size for a rank test across more than two groups."""
    n = sum(group.size for group in groups)
    k = len(groups)
    if n <= k:
        return float("nan"), "epsilon squared"
    return float((statistic - k + 1) / (n - k)), "epsilon squared"


def _eta_squared(groups: Sequence[np.ndarray]) -> tuple:
    """Proportion of variance explained, for a parametric >2-group test."""
    everything = np.concatenate(groups)
    grand = float(np.mean(everything))
    between = sum(group.size * (float(np.mean(group)) - grand) ** 2
                  for group in groups)
    total = float(np.sum((everything - grand) ** 2))
    if not total:
        return float("nan"), "eta squared"
    return float(between / total), "eta squared"


def compare(groups: Mapping[str, Sequence], *, unit: str = "observation",
            paired: bool = False, force: Optional[str] = None) -> Comparison:
    """Choose and run the right test for these groups.

    :param groups: ``{label: values}``. Two or more.
    :param unit: what ONE observation is -- 'well', 'cell', 'guide'. Stated
        in the result, because a test across cells when the replicate is the
        well is pseudoreplication and returns p < 1e-10 on noise.
    :param paired: the groups are matched (the same wells before and after).
    :param force: a test name to use instead of the chosen one.
    :returns: a :class:`Comparison`.
    :raises ValueError: with fewer than two groups, or a group too small to
        test. Refused rather than returned as NaN: a comparison that could not
        be made is not a comparison with an unknown answer.
    """
    labels = list(groups)
    if len(labels) < 2:
        raise ValueError(
            f"a comparison needs at least two groups, got {len(labels)}")
    arrays = [_clean(groups[label]) for label in labels]
    too_small = [label for label, values in zip(labels, arrays)
                 if values.size < MIN_N_FOR_TEST]
    if too_small:
        raise ValueError(
            f"these groups have fewer than {MIN_N_FOR_TEST} usable "
            f"observations and cannot be tested: {too_small}")

    normality = check_normality(arrays)
    variance = check_equal_variance(arrays)
    # READ THE CHECK'S OWN VERDICT. Re-deriving it here from `p_value >= 0.05`
    # is what discarded the Bonferroni correction the normality check applies
    # across groups, and sent 18% of four-group comparisons on perfectly
    # normal data to a rank test instead of 5%.
    normal = normality.passed
    equal = variance.passed

    counts = [int(values.size) for values in arrays]
    reason_bits = [normality.verdict, variance.verdict]

    if force:
        chosen = force
        reason_bits.insert(0, "forced by the caller")
    elif len(arrays) == 2:
        if paired:
            chosen = "paired t" if normal else "Wilcoxon signed-rank"
        elif not normal:
            chosen = "Mann-Whitney U"
        else:
            chosen = "Student's t" if equal else "Welch's t"
    else:
        if not normal:
            chosen = "Kruskal-Wallis"
        else:
            chosen = "one-way ANOVA" if equal else "Welch's ANOVA"

    statistic, p = _run(chosen, arrays, paired=paired)

    if len(arrays) == 2:
        effect, effect_name = _hedges_g(arrays[0], arrays[1])
        ci = _difference_ci(arrays[0], arrays[1], equal=equal)
    elif chosen == "Kruskal-Wallis":
        effect, effect_name = _epsilon_squared(arrays, statistic)
        ci = None
    else:
        effect, effect_name = _eta_squared(arrays)
        ci = None

    return Comparison(
        test=chosen, statistic=float(statistic), p_value=float(p),
        groups=labels, n=counts, unit=unit,
        effect_size=effect, effect_name=effect_name, ci=ci,
        assumptions=[normality, variance],
        reason="; ".join(reason_bits))


def _run(name: str, arrays: Sequence[np.ndarray], *, paired: bool) -> tuple:
    from scipy import stats

    if name == "Student's t":
        return stats.ttest_ind(arrays[0], arrays[1], equal_var=True)
    if name == "Welch's t":
        return stats.ttest_ind(arrays[0], arrays[1], equal_var=False)
    if name == "paired t":
        return stats.ttest_rel(arrays[0], arrays[1])
    if name == "Wilcoxon signed-rank":
        return stats.wilcoxon(arrays[0], arrays[1])
    if name == "Mann-Whitney U":
        return stats.mannwhitneyu(arrays[0], arrays[1],
                                  alternative="two-sided")
    if name == "Kruskal-Wallis":
        return stats.kruskal(*arrays)
    if name == "one-way ANOVA":
        return stats.f_oneway(*arrays)
    if name == "Welch's ANOVA":
        return _welch_anova(arrays)
    raise ValueError(f"unknown test {name!r}")


def _welch_anova(arrays: Sequence[np.ndarray]) -> tuple:
    """Welch's one-way ANOVA. scipy has no direct implementation.

    The heteroscedastic analogue of f_oneway: each group weighted by its own
    precision rather than pooled, which is what makes it valid when the
    variances differ.
    """
    from scipy import stats

    k = len(arrays)
    n = np.array([group.size for group in arrays], dtype=float)
    means = np.array([group.mean() for group in arrays])
    variances = np.array([group.var(ddof=1) for group in arrays])
    weights = n / variances
    total_weight = weights.sum()
    grand = float((weights * means).sum() / total_weight)
    numerator = float((weights * (means - grand) ** 2).sum() / (k - 1))
    lam = float((((1 - weights / total_weight) ** 2) / (n - 1)).sum())
    denominator = 1 + (2 * (k - 2) / (k ** 2 - 1)) * lam
    statistic = numerator / denominator
    df2 = (k ** 2 - 1) / (3 * lam)
    return statistic, float(stats.f.sf(statistic, k - 1, df2))


def _difference_ci(a: np.ndarray, b: np.ndarray, *, equal: bool,
                   level: float = 0.95):
    """95% interval for the difference in means."""
    from scipy import stats

    na, nb = a.size, b.size
    diff = float(np.mean(a) - np.mean(b))
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    if equal:
        pooled = ((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)
        se = np.sqrt(pooled * (1 / na + 1 / nb))
    else:
        se = np.sqrt(va / na + vb / nb)
    # Bail on a degenerate spread BEFORE the degrees of freedom are computed.
    # With zero variance the Welch df is 0/0, which prints a RuntimeWarning on
    # the way to a number this function is about to discard -- and two constant
    # arms is a real case, not a contrived one.
    if not np.isfinite(se) or se == 0:
        return None
    df = (na + nb - 2) if equal else (
        (va / na + vb / nb) ** 2
        / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)))
    margin = float(stats.t.ppf(0.5 + level / 2, df) * se)
    return (diff - margin, diff + margin)


def table(comparisons: Sequence[Comparison], *, correction: str = "fdr_bh"):
    """Every comparison as one frame, corrected across them.

    Correcting ACROSS the comparisons is the part a hand-written stats table
    always forgets: six pairwise tests at 0.05 is a 26% chance of at least one
    false positive, and the individual p-values give no hint of it.
    """
    import pandas as pd

    if not comparisons:
        return pd.DataFrame(columns=["test", "groups", "n", "unit",
                                     "statistic", "p_value", "p_adjusted",
                                     "effect_size", "effect", "reason"])
    if correction and len(comparisons) > 1:
        from ..multiple_testing import adjust_p_values, canonical_method

        method = canonical_method(correction)
        adjusted, _ = adjust_p_values(
            np.array([c.p_value for c in comparisons], dtype=float),
            method=method, alpha=0.05)
        for comparison, value in zip(comparisons, adjusted):
            comparison.p_adjusted = float(value)
            comparison.correction = method

    rows = []
    for comparison in comparisons:
        row = {
            "test": comparison.test,
            "groups": " vs ".join(str(label) for label in comparison.groups),
            "n": " / ".join(str(value) for value in comparison.n),
            "unit": comparison.unit,
            "statistic": comparison.statistic,
            "p_value": comparison.p_value,
            "p_adjusted": comparison.p_adjusted,
            "correction": comparison.correction,
            "significance": comparison.marks,
            "effect_size": comparison.effect_size,
            "effect": comparison.effect_name,
            "ci_low": comparison.ci[0] if comparison.ci else float("nan"),
            "ci_high": comparison.ci[1] if comparison.ci else float("nan"),
            "why_this_test": comparison.reason,
        }
        for assumption in comparison.assumptions:
            # A column name that goes into a CSV a reviewer will open:
            # "shapiro", not "Shapiro-Wilk" or "shapiro-wilk".
            key = "".join(ch for ch in assumption.name.split()[0].lower()
                          if ch.isalnum() or ch == "_").split("wilk")[0]
            key = key.rstrip("_-") or "check"
            row[f"{key}_p"] = assumption.p_value
            row[f"{key}_verdict"] = assumption.verdict
            row[f"{key}_informative"] = assumption.informative
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = ["Assumption", "CONVENTION", "Comparison", "MIN_N_FOR_ASSUMPTIONS",
           "MIN_N_FOR_TEST", "check_equal_variance", "check_normality",
           "compare", "stars", "table"]

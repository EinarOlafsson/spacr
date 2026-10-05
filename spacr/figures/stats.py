"""Pick the right test from the data, and show the working.

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
records the check as UNINFORMATIVE and selects Welch's t-test or
Mann-Whitney, which costs a little power when the assumption did hold and
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
taking the parametric branch where the checks had no power to refuse it. It is
now a translation layer onto :func:`compare`,
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

#: Asterisk convention reported with every result so readers can interpret
#: significance marks without inferring thresholds.
CONVENTION = "*p<0.05, **p<0.01, ***p<0.001, ****p<0.0001"


def stars(p) -> str:
    """The asterisks for a p-value, or ``n.s.`` written out.

    Non-significant comparisons are SHOWN rather than omitted, which is what
    the published figures do -- a missing bracket reads as a comparison
    nobody made.

    :param p: p-value to translate into the reporting convention.
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
    """One assumption check, and whether it could see anything.

    :param name: name of the assumption test.
    :param statistic: test statistic, or ``nan`` when it could not be computed.
    :param p_value: test p-value, or ``nan`` when it could not be computed.
    :param informative: whether the check had enough usable data to interpret.
    :param verdict: plain-language conclusion, including inconclusive cases.
    :param passed: decision made by the check's own rule; callers must not
        re-derive it from ``p_value``.
    """

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
    """One test, everything needed to report it, and how it was chosen.

    :ivar test: name of the selected statistical test.
    :ivar statistic: statistic returned by that test.
    :ivar p_value: unadjusted p-value returned by that test.
    :ivar groups: group labels in the order tested.
    :ivar n: usable observation counts for those groups, in matching order.
    :ivar unit: independent unit represented by one observation, such as a
        well, cell, or guide; this prevents reporting row count as replication.
    :ivar effect_size: estimated magnitude of the group difference on the
        scale named by ``effect_name``.
    :ivar effect_name: statistic used for ``effect_size``, such as Cohen's d.
    :ivar ci: lower and upper confidence bounds for the reported effect, or
        ``None`` when the selected method cannot provide them.
    :ivar assumptions: diagnostic checks that selected or qualified this test.
    :ivar reason: plain-language explanation of why this test was selected.
    :ivar correction: multiple-testing method applied to obtain ``p_adjusted``.
    :ivar p_adjusted: corrected p-value, or ``nan`` when no correction applies.
    """

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
        """The significance stars for this comparison.

        FROM THE ADJUSTED P WHEN THERE IS ONE. Starring an unadjusted p in a
        figure that made many comparisons is how a multiple-testing problem
        becomes a claim; the raw value is used only when no adjustment was
        made.

        :returns: the stars, or an empty string.
        """
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
    """Drop the non-finite values from an array.

    :param values: the values.
    :returns: the finite ones, as float64 -- a test run over ``nan`` returns
        ``nan`` rather than failing, which is worse than dropping them.
    """
    array = np.asarray(values, dtype="float64")
    return array[np.isfinite(array)]


def check_normality(groups: Sequence[np.ndarray]) -> Assumption:
    """Shapiro-Wilk per group, and whether it could see anything.

    :param groups: numeric sample array for each group being compared.
    """
    from scipy import stats

    smallest = min((group.size for group in groups), default=0)
    if smallest < MIN_N_FOR_ASSUMPTIONS:
        return Assumption(
            "Shapiro-Wilk", float("nan"), float("nan"), False,
            f"the smallest group has {smallest} observations, too few for a "
            f"normality test to have power — treated as NOT normal, which is "
            f"the safe direction",
            passed=False)
    flat = [group for group in groups if float(np.ptp(group)) == 0.0]
    if flat:
        return Assumption(
            "Shapiro-Wilk", float("nan"), float("nan"), False,
            f"{len(flat)} group(s) have no spread at all, so a normality test "
            f"has nothing to describe — treated as NOT normal, which is the "
            f"safe direction",
            passed=False)
    worst_p, worst_stat, tested = float("inf"), float("nan"), 0
    for group in groups:
        try:
            statistic, p = stats.shapiro(group[:5000])
        except Exception:
            continue
        if not np.isfinite(statistic) or not np.isfinite(p):
            continue
        tested += 1
        if p < worst_p:
            worst_p, worst_stat = float(p), float(statistic)
    if not tested or not np.isfinite(worst_p):
        return Assumption("Shapiro-Wilk", float("nan"), float("nan"), False,
                          "could not be computed", passed=False)

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

    The median-centred Brown-Forsythe form is less sensitive to non-normal
    data than the mean-centred form. This function is called before the
    normality verdict is known.

    :param groups: numeric sample array for each group being compared.
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
        with np.errstate(invalid="ignore", divide="ignore"):
            statistic, p = stats.levene(*groups, center="median")
    except Exception:
        return Assumption("Levene (median-centred)", float("nan"),
                          float("nan"), False, "could not be computed",
                          passed=False)
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
    """Run one named statistical test.

    :param name: the test.
    :param arrays: the samples.
    :param paired: whether the samples are paired.
    :returns: whatever scipy returns for that test -- the statistic and its
        P value.
    """
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
        pooled = np.concatenate((arrays[0], arrays[1]))
        if pooled.size and float(np.ptp(pooled)) == 0.0:
            return arrays[0].size * arrays[1].size / 2.0, 1.0
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

    :param comparisons: completed comparison results to tabulate and correct
        as one family.
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
            key = "".join(ch for ch in assumption.name.split()[0].lower()
                          if ch.isalnum() or ch == "_").split("wilk")[0]
            key = key.rstrip("_-") or "check"
            row[f"{key}_p"] = assumption.p_value
            row[f"{key}_verdict"] = assumption.verdict
            row[f"{key}_informative"] = assumption.informative
        rows.append(row)
    return pd.DataFrame(rows)


#: Columns of the single statistics table written beside a saved figure.
_STAT_COLUMNS = ("test_stage", "test_name", "groups", "statistic", "df",
                 "p_value", "p_adjusted", "correction", "effect_size",
                 "effect", "n", "chosen_by", "reason")

#: Tests a user may choose in place of the automatic one, by data kind.
_OVERRIDES = {
    "groups": ("Student's t", "Welch's t", "Mann-Whitney U", "paired t",
               "Wilcoxon signed-rank", "one-way ANOVA", "Welch's ANOVA",
               "Kruskal-Wallis", "Friedman"),
    "correlation": ("Pearson", "Spearman"),
    "contingency": ("chi-square", "Fisher's exact"),
    "proportion": ("two-proportion z", "chi-square", "Fisher's exact"),
}

_TWO_GROUP_TESTS = ("Student's t", "Welch's t", "Mann-Whitney U", "paired t",
                    "Wilcoxon signed-rank")
_OMNIBUS_TESTS = ("one-way ANOVA", "Welch's ANOVA", "Kruskal-Wallis",
                  "Friedman")


def _row(stage, name, **values) -> dict:
    """One row of the statistics table, every column present."""
    row = {column: "" for column in _STAT_COLUMNS}
    row.update(test_stage=stage, test_name=name, statistic=float("nan"),
               p_value=float("nan"), p_adjusted=float("nan"),
               effect_size=float("nan"), chosen_by="auto")
    row.update(values)
    return row


def _data_kind(frame, x: str, y: str) -> str:
    """What kind of comparison the two plotted columns support.

    :returns: ``groups`` (categories against a measurement), ``proportion``
        (categories against a 0/1 outcome), ``contingency`` (categories
        against categories), ``correlation`` (two measurements) or ``none``.
    """
    import pandas as pd

    columns = getattr(frame, "columns", ())
    if not x or not y or x not in columns or y not in columns:
        return "none"
    x_numeric = pd.api.types.is_numeric_dtype(frame[x]) and not \
        pd.api.types.is_bool_dtype(frame[x])
    y_values = frame[y].dropna()
    y_binary = (pd.api.types.is_bool_dtype(frame[y]) or (
        pd.api.types.is_numeric_dtype(frame[y]) and len(y_values)
        and set(np.unique(y_values.astype(float))) <= {0.0, 1.0}))
    y_numeric = pd.api.types.is_numeric_dtype(frame[y]) and not \
        pd.api.types.is_bool_dtype(frame[y])
    if x_numeric and y_numeric and not y_binary:
        return "correlation"
    if not x_numeric and y_binary:
        return "proportion"
    if not x_numeric and y_numeric:
        return "groups"
    if not x_numeric and not y_numeric:
        return "contingency"
    return "none"


def _adjust(p_values, correction: str):
    """Corrected p values for one family, and the method's canonical name."""
    from ..multiple_testing import adjust_p_values, canonical_method

    values = np.asarray(p_values, dtype=float)
    if values.size < 2:
        return values.copy(), ""
    method = canonical_method(correction or "fdr_bh")
    adjusted, _ = adjust_p_values(values, method=method, alpha=0.05)
    return adjusted, method


def _shapiro_rows(named) -> list:
    """One Shapiro-Wilk row per named sample."""
    from scipy import stats

    rows = []
    for label, values in named:
        if values.size < 3 or float(np.ptp(values)) == 0.0:
            rows.append(_row("normality", "Shapiro-Wilk", groups=label,
                             n=int(values.size),
                             reason="too few values, or no spread, to test"))
            continue
        statistic, p = stats.shapiro(values[:5000])
        rows.append(_row("normality", "Shapiro-Wilk", groups=label,
                         statistic=float(statistic), p_value=float(p),
                         n=int(values.size),
                         reason="normal" if p >= 0.05 else "not normal"))
    return rows


def _group_statistics(frame, x, y, *, order, force, paired, pair,
                      correction) -> list:
    """Normality, equal variance, omnibus and pairwise rows for groups."""
    import pandas as pd
    from scipy import stats

    data = frame.copy()
    data[x] = data[x].astype(str)
    labels = [str(v) for v in (order or pd.unique(data[x].dropna()))]
    labels = [label for label in labels if (data[x] == label).any()]
    note = ""
    if paired and pair and pair in data.columns:
        wide = data.pivot_table(index=pair, columns=x, values=y,
                                aggfunc="mean")
        wide = wide[[label for label in labels if label in wide.columns]]
        wide = wide.dropna()
        arrays = [wide[label].to_numpy(dtype=float) for label in labels]
    else:
        arrays = [_clean(data.loc[data[x] == label, y]) for label in labels]
        if paired and len({a.size for a in arrays}) != 1:
            paired = False
            note = ("paired was asked for, but the groups differ in size and "
                    "no pairing column was given, so the groups are treated "
                    "as independent; ")
    small = [label for label, a in zip(labels, arrays)
             if a.size < MIN_N_FOR_TEST]
    if len(labels) < 2 or small:
        return [_row("pairwise", "none", groups=" vs ".join(labels),
                     reason=("fewer than two groups to compare"
                             if len(labels) < 2 else
                             f"too few values to test in {small}"))]

    rows = []
    if paired and len(arrays) == 2:
        difference = arrays[0] - arrays[1]
        rows += _shapiro_rows([(f"{labels[0]} - {labels[1]}", difference)])
        normality = check_normality([difference])
    else:
        rows += _shapiro_rows(zip(labels, arrays))
        normality = check_normality(arrays)
    normal = normality.passed
    for row in rows:
        row["reason"] = f"{row['reason']}; overall: {normality.verdict}"

    smallest = min(a.size for a in arrays)
    if normal:
        statistic, p = stats.bartlett(*arrays)
        variance_name = "Bartlett"
    else:
        with np.errstate(invalid="ignore", divide="ignore"):
            statistic, p = stats.levene(*arrays, center="median")
        variance_name = "Levene (median-centred)"
    equal = bool(np.isfinite(p) and p >= 0.05
                 and smallest >= MIN_N_FOR_ASSUMPTIONS)
    rows.append(_row(
        "equal_variance", variance_name, groups=" / ".join(labels),
        statistic=float(statistic), p_value=float(p),
        df=str(len(arrays) - 1),
        n=" / ".join(str(a.size) for a in arrays),
        reason=("Bartlett because every group looked normal; "
                if normal else "Levene because normality failed; ")
        + ("variances equal" if equal else
           ("too few values to trust, treated as unequal"
            if smallest < MIN_N_FOR_ASSUMPTIONS else "variances differ"))))

    counts = " / ".join(str(a.size) for a in arrays)
    if len(arrays) == 2:
        if paired:
            auto = "paired t" if normal else "Wilcoxon signed-rank"
        elif not normal:
            auto = "Mann-Whitney U"
        else:
            auto = "Student's t" if equal else "Welch's t"
        chosen = force if force in _TWO_GROUP_TESTS else auto
        statistic, p = _run(chosen, arrays, paired=chosen in (
            "paired t", "Wilcoxon signed-rank"))
        a, b = arrays
        if chosen == "Student's t":
            df = str(a.size + b.size - 2)
        elif chosen == "Welch's t":
            va, vb = np.var(a, ddof=1) / a.size, np.var(b, ddof=1) / b.size
            df = f"{(va + vb) ** 2 / (va ** 2 / (a.size - 1) + vb ** 2 / (b.size - 1)):.4g}"
        elif chosen == "paired t":
            df = str(a.size - 1)
        else:
            df = ""
        effect, effect_name = _hedges_g(a, b)
        rows.append(_row(
            "pairwise", chosen, groups=f"{labels[0]} vs {labels[1]}",
            statistic=float(statistic), p_value=float(p), df=df,
            effect_size=effect, effect=effect_name, n=counts,
            chosen_by="user" if chosen != auto else "auto",
            reason=note + ("chosen by the user" if chosen != auto else
                           ("paired; " if paired else "")
                           + ("normal" if normal else "not normal")
                           + (", equal variances" if equal else "")
                           + f" -> {auto}")))
        return rows

    if paired:
        auto = "Friedman"
    elif not normal:
        auto = "Kruskal-Wallis"
    else:
        auto = "one-way ANOVA" if equal else "Welch's ANOVA"
    omnibus = force if force in _OMNIBUS_TESTS else auto
    total = sum(a.size for a in arrays)
    k = len(arrays)
    if omnibus == "Friedman":
        if len({a.size for a in arrays}) != 1:
            rows.append(_row("omnibus", "Friedman", groups=" / ".join(labels),
                             chosen_by="user" if omnibus != auto else "auto",
                             reason="Friedman needs every group measured on "
                                    "the same subjects; the sizes differ"))
            return rows
        statistic, p = stats.friedmanchisquare(*arrays)
        df = str(k - 1)
        effect = float(statistic) / (arrays[0].size * (k - 1))
        effect_name = "Kendall's W"
    else:
        statistic, p = _run(omnibus, arrays, paired=False)
        if omnibus == "Kruskal-Wallis":
            df = str(k - 1)
            effect, effect_name = _epsilon_squared(arrays, statistic)
        else:
            if omnibus == "Welch's ANOVA":
                n = np.array([a.size for a in arrays], dtype=float)
                w = n / np.array([a.var(ddof=1) for a in arrays])
                lam = float((((1 - w / w.sum()) ** 2) / (n - 1)).sum())
                df = f"{k - 1}, {(k ** 2 - 1) / (3 * lam):.4g}"
            else:
                df = f"{k - 1}, {total - k}"
            effect, effect_name = _eta_squared(arrays)
    rows.append(_row(
        "omnibus", omnibus, groups=" / ".join(labels),
        statistic=float(statistic), p_value=float(p), df=df,
        effect_size=float(effect), effect=effect_name, n=counts,
        chosen_by="user" if omnibus != auto else "auto",
        reason=note + ("chosen by the user" if omnibus != auto else
                       ("repeated measures" if paired else
                        ("normal" if normal else "not normal")
                        + (", equal variances" if equal else
                           ", unequal variances" if normal else ""))
                       + f" -> {auto}")))

    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    if force in _TWO_GROUP_TESTS:
        posthoc, by = force, "user"
    else:
        posthoc = {"one-way ANOVA": "Tukey HSD", "Welch's ANOVA":
                   "Games-Howell", "Kruskal-Wallis": "Dunn",
                   "Friedman": "Wilcoxon signed-rank"}[omnibus]
        by = "user" if omnibus != auto else "auto"
    raw, family_adjusted = [], None
    statistics_ = []
    if posthoc == "Tukey HSD":
        result = stats.tukey_hsd(*arrays)
        family_adjusted = [float(result.pvalue[i, j]) for i, j in pairs]
        statistics_ = [float(result.statistic[i, j]) for i, j in pairs]
        raw = family_adjusted
    elif posthoc == "Games-Howell":
        for i, j in pairs:
            a, b = arrays[i], arrays[j]
            va, vb = a.var(ddof=1) / a.size, b.var(ddof=1) / b.size
            t = (a.mean() - b.mean()) / np.sqrt(va + vb)
            dof = (va + vb) ** 2 / (va ** 2 / (a.size - 1)
                                     + vb ** 2 / (b.size - 1))
            statistics_.append(float(t))
            raw.append(float(stats.studentized_range.sf(
                abs(t) * np.sqrt(2), k, dof)))
        family_adjusted = raw
    elif posthoc == "Dunn":
        pooled = np.concatenate(arrays)
        ranks = stats.rankdata(pooled)
        _values, ties = np.unique(pooled, return_counts=True)
        tie = float((ties ** 3 - ties).sum()) / (12.0 * (total - 1))
        edges = np.cumsum([0] + [a.size for a in arrays])
        means = [ranks[edges[g]:edges[g + 1]].mean() for g in range(k)]
        for i, j in pairs:
            sigma = np.sqrt((total * (total + 1) / 12.0 - tie)
                            * (1.0 / arrays[i].size + 1.0 / arrays[j].size))
            z = (means[i] - means[j]) / sigma
            statistics_.append(float(z))
            raw.append(float(2 * stats.norm.sf(abs(z))))
    else:
        for i, j in pairs:
            statistic, p = _run(posthoc, [arrays[i], arrays[j]],
                                paired=posthoc in ("paired t",
                                                   "Wilcoxon signed-rank"))
            statistics_.append(float(statistic))
            raw.append(float(p))
    if family_adjusted is None:
        adjusted, method = _adjust(raw, correction)
    else:
        adjusted, method = np.asarray(family_adjusted), \
            f"{posthoc} (family-wise)"
    for (i, j), statistic, p, p_adj in zip(pairs, statistics_, raw,
                                           adjusted):
        effect, effect_name = _hedges_g(arrays[i], arrays[j])
        rows.append(_row(
            "pairwise", posthoc, groups=f"{labels[i]} vs {labels[j]}",
            statistic=statistic, p_value=p, p_adjusted=float(p_adj),
            correction=method, effect_size=effect, effect=effect_name,
            n=f"{arrays[i].size} / {arrays[j].size}", chosen_by=by,
            reason=f"post-hoc after {omnibus}" if by == "auto"
            else "chosen by the user"))
    return rows


def _correlation_statistics(frame, x, y, *, force) -> list:
    """Normality of both measurements, then Pearson or Spearman."""
    from scipy import stats

    pairs = frame[[x, y]].apply(lambda s: s.astype(float)).dropna()
    a, b = pairs[x].to_numpy(), pairs[y].to_numpy()
    if a.size < 3:
        return [_row("correlation", "none", groups=f"{x} vs {y}",
                     n=int(a.size), reason="fewer than three pairs")]
    rows = _shapiro_rows([(x, a), (y, b)])
    normal = check_normality([a, b]).passed
    auto = "Pearson" if normal else "Spearman"
    chosen = force if force in ("Pearson", "Spearman") else auto
    if chosen == "Pearson":
        statistic, p = stats.pearsonr(a, b)
    else:
        statistic, p = stats.spearmanr(a, b)
    rows.append(_row(
        "correlation", chosen, groups=f"{x} vs {y}",
        statistic=float(statistic), p_value=float(p), df=str(a.size - 2),
        effect_size=float(statistic), effect="r" if chosen == "Pearson"
        else "rho", n=int(a.size),
        chosen_by="user" if chosen != auto else "auto",
        reason="chosen by the user" if chosen != auto else
        ("both normal" if normal else "not both normal") + f" -> {auto}"))
    return rows


def _table_test(table, force: str = ""):
    """Chi-square or Fisher's exact on one contingency table.

    :returns: ``(name, statistic, df, p, chosen_by, reason)``.
    """
    from scipy import stats

    observed = np.asarray(table, dtype=float)
    _chi, _p, dof, expected = stats.chi2_contingency(observed,
                                                     correction=False)
    small = bool((expected < 5).any())
    two_by_two = observed.shape == (2, 2)
    auto = "Fisher's exact" if small else "chi-square"
    chosen = force if force in ("chi-square", "Fisher's exact") else auto
    by = "user" if chosen != auto else "auto"
    why = ("an expected count is under 5" if small
           else "every expected count is 5 or more")
    if chosen == "Fisher's exact" and two_by_two:
        statistic, p = stats.fisher_exact(observed)
        return chosen, float(statistic), "", float(p), by, f"{why}; 2x2"
    if chosen == "Fisher's exact":
        rng = np.random.default_rng(0)
        rows_of = np.repeat(np.arange(observed.shape[0]),
                            observed.sum(axis=1).astype(int))
        cols_of = np.repeat(np.arange(observed.shape[1]),
                            observed.sum(axis=0).astype(int))
        reference = _chi
        hits = 0
        for _ in range(2000):
            shuffled = np.zeros_like(observed)
            np.add.at(shuffled, (rows_of, rng.permutation(cols_of)), 1)
            with np.errstate(invalid="ignore", divide="ignore"):
                value = np.nansum((shuffled - expected) ** 2 / expected)
            hits += value >= reference - 1e-12
        return ("Fisher-Freeman-Halton (Monte Carlo)", float(reference),
                str(dof), float((hits + 1) / 2001), by,
                f"{why}; larger than 2x2, so the exact p is estimated from "
                "2000 seeded permutations")
    return chosen, float(_chi), str(dof), float(_p), by, why


def _cramers_v(table) -> float:
    """Cramér's V for a contingency table."""
    from scipy import stats

    observed = np.asarray(table, dtype=float)
    chi = stats.chi2_contingency(observed, correction=False)[0]
    total = observed.sum()
    smaller = min(observed.shape) - 1
    return float(np.sqrt(chi / (total * smaller))) if total and smaller \
        else float("nan")


def _contingency_statistics(frame, x, y, *, force, correction,
                            count="") -> list:
    """Chi-square or Fisher's exact, then corrected pairwise tables."""
    import pandas as pd

    if count and count in frame.columns:
        table = frame.pivot_table(index=x, columns=y, values=count,
                                  aggfunc="sum", fill_value=0)
    else:
        table = pd.crosstab(frame[x].astype(str), frame[y].astype(str))
    if table.shape[0] < 2 or table.shape[1] < 2:
        return [_row("contingency", "none", groups=f"{x} x {y}",
                     reason="the table needs at least two rows and two "
                            "columns")]
    name, statistic, df, p, by, why = _table_test(table.to_numpy(), force)
    rows = [_row("contingency", name, groups=f"{x} x {y}",
                 statistic=statistic, df=df, p_value=p,
                 effect_size=_cramers_v(table.to_numpy()),
                 effect="Cramér's V", n=int(table.to_numpy().sum()),
                 chosen_by=by, reason=why)]
    labels = list(table.index)
    if len(labels) > 2:
        found = []
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                part = table.loc[[labels[i], labels[j]]]
                part = part.loc[:, part.sum(axis=0) > 0]
                if part.shape[1] < 2:
                    continue
                found.append((labels[i], labels[j], part,
                              _table_test(part.to_numpy(), force)))
        adjusted, method = _adjust([item[3][3] for item in found],
                                   correction)
        for (left, right, part, result), p_adj in zip(found, adjusted):
            name, statistic, df, p, by, why = result
            rows.append(_row(
                "pairwise", name, groups=f"{left} vs {right}",
                statistic=statistic, df=df, p_value=p,
                p_adjusted=float(p_adj), correction=method,
                effect_size=_cramers_v(part.to_numpy()), effect="Cramér's V",
                n=int(part.to_numpy().sum()), chosen_by=by, reason=why))
    return rows


def _proportion_statistics(frame, x, y, *, order, force,
                           correction) -> list:
    """Two-proportion z-test or chi-square, with corrected pairs."""
    import pandas as pd
    from scipy import stats

    data = frame[[x, y]].dropna().copy()
    data[x] = data[x].astype(str)
    data[y] = data[y].astype(float)
    labels = [str(v) for v in (order or pd.unique(data[x]))]
    labels = [label for label in labels if (data[x] == label).any()]
    hits = np.array([data.loc[data[x] == g, y].sum() for g in labels])
    totals = np.array([(data[x] == g).sum() for g in labels], dtype=float)
    if len(labels) < 2:
        return [_row("proportion", "none", reason="fewer than two groups")]

    def _z(i, j):
        """Pooled two-proportion z statistic and two-sided p."""
        pooled = (hits[i] + hits[j]) / (totals[i] + totals[j])
        se = np.sqrt(pooled * (1 - pooled)
                     * (1 / totals[i] + 1 / totals[j]))
        if not se:
            return 0.0, 1.0
        z = (hits[i] / totals[i] - hits[j] / totals[j]) / se
        return float(z), float(2 * stats.norm.sf(abs(z)))

    def _one(i, j, stage):
        """One comparison of two groups' proportions."""
        table = np.array([[hits[i], totals[i] - hits[i]],
                          [hits[j], totals[j] - hits[j]]])
        expected = stats.contingency.expected_freq(table)
        small = bool((expected < 5).any())
        auto = "Fisher's exact" if small else "two-proportion z"
        chosen = force if force in _OVERRIDES["proportion"] else auto
        if chosen == "two-proportion z":
            statistic, p = _z(i, j)
            df = ""
        elif chosen == "Fisher's exact":
            statistic, p = stats.fisher_exact(table)
            df = ""
        else:
            statistic, p, dof, _e = stats.chi2_contingency(
                table, correction=False)
            df = str(dof)
        return _row(
            stage, chosen, groups=f"{labels[i]} vs {labels[j]}",
            statistic=float(statistic), p_value=float(p), df=df,
            effect_size=float(hits[i] / totals[i] - hits[j] / totals[j]),
            effect="difference in proportions",
            n=f"{int(totals[i])} / {int(totals[j])}",
            chosen_by="user" if chosen != auto else "auto",
            reason=("chosen by the user" if chosen != auto else
                    "an expected count is under 5" if small else
                    "every expected count is 5 or more") + f" -> {auto}")

    if len(labels) == 2:
        return [_one(0, 1, "pairwise")]
    table = np.column_stack([hits, totals - hits])
    statistic, p, dof, _e = stats.chi2_contingency(table, correction=False)
    rows = [_row("omnibus", "chi-square", groups=" / ".join(labels),
                 statistic=float(statistic), p_value=float(p), df=str(dof),
                 effect_size=_cramers_v(table), effect="Cramér's V",
                 n=" / ".join(str(int(t)) for t in totals),
                 reason="three or more proportions -> chi-square")]
    pairwise = [_one(i, j, "pairwise") for i in range(len(labels))
                for j in range(i + 1, len(labels))]
    adjusted, method = _adjust([row["p_value"] for row in pairwise],
                               correction)
    for row, p_adj in zip(pairwise, adjusted):
        row.update(p_adjusted=float(p_adj), correction=method)
    return rows + pairwise


def _auto_statistics(frame, x: str = "", y: str = "", *,
                     test: Optional[str] = None,
                     paired: Optional[bool] = None,
                     pair: str = "", correction: str = "fdr_bh",
                     order=None, count: str = ""):
    """Every applicable test for the plotted columns, as ONE table.

    The data decide which family applies: categories against categories are
    a contingency table (chi-square, or Fisher's exact when an expected count
    is under 5); categories against a 0/1 outcome are proportions
    (two-proportion z-test or chi-square); two measurements are a correlation
    (Pearson when both are normal, Spearman otherwise); categories against a
    measurement are groups. For groups the rows run in order: Shapiro-Wilk
    per group, an equal-variance test (Bartlett when every group is normal,
    Levene otherwise), then for two groups Student's t, Welch's t,
    Mann-Whitney U, the paired t-test or Wilcoxon signed-rank; for three or
    more an omnibus test (one-way ANOVA, Welch's ANOVA, Kruskal-Wallis, or
    Friedman for repeated measures) followed by its post-hoc (Tukey HSD,
    Games-Howell, Dunn, or pairwise Wilcoxon) with the multiple-comparison
    correction.

    :param frame: the tidy data the figure was drawn from.
    :param x: the grouping or first variable.
    :param y: the measured or second variable.
    :param test: a test name to use instead of the automatic choice; rows it
        changes say ``user`` in ``chosen_by``.
    :param paired: the groups are repeated measures of the same subjects;
        ``None`` decides from the data, which is paired when ``pair`` names
        a column.
    :param pair: the column naming the subject, used to align paired values.
    :param correction: multiple-comparison method for pairwise p values.
    :param order: group order, or ``None`` for order of appearance.
    :param count: for a contingency table, a column holding counts.
    :returns: a frame with the columns of :data:`_STAT_COLUMNS`.
    """
    import pandas as pd

    kind = _data_kind(frame, x, y) if frame is not None else "none"
    if paired is None:
        paired = bool(kind == "groups" and pair and pair in frame.columns)
    try:
        if kind == "groups":
            rows = _group_statistics(frame, x, y, order=order, force=test,
                                     paired=bool(paired), pair=pair,
                                     correction=correction)
        elif kind == "correlation":
            rows = _correlation_statistics(frame, x, y, force=test)
        elif kind == "contingency":
            rows = _contingency_statistics(frame, x, y, force=test,
                                           correction=correction,
                                           count=count)
        elif kind == "proportion":
            rows = _proportion_statistics(frame, x, y, order=order,
                                          force=test, correction=correction)
        else:
            rows = [_row("none", "none", reason=(
                "the figure has no pair of variables to compare, so no "
                "test was run"))]
    except Exception as error:
        rows = [_row(kind, "refused",
                     reason=f"{type(error).__name__}: {error}")]
    return pd.DataFrame(rows, columns=list(_STAT_COLUMNS))


def _statistics_text(table) -> str:
    """A readable summary of :func:`_auto_statistics` output."""
    lines = []
    for _index, row in table.iterrows():
        parts = [f"[{row['test_stage']}] {row['test_name']}"]
        if row["groups"]:
            parts.append(str(row["groups"]))
        for key, label in (("statistic", "stat"), ("p_value", "p"),
                           ("p_adjusted", "p adj")):
            value = row[key]
            if isinstance(value, float) and np.isfinite(value):
                parts.append(f"{label} = {value:.4g}")
        if row["df"]:
            parts.append(f"df = {row['df']}")
        if row["correction"]:
            parts.append(str(row["correction"]))
        if row["n"] != "":
            parts.append(f"n = {row['n']}")
        parts.append(f"({row['chosen_by']}: {row['reason']})")
        lines.append("; ".join(parts))
    return "\n".join(lines) + f"\n{CONVENTION}\n"


__all__ = ["Assumption", "CONVENTION", "Comparison", "MIN_N_FOR_ASSUMPTIONS",
           "MIN_N_FOR_TEST", "check_equal_variance", "check_normality",
           "compare", "stars", "table"]

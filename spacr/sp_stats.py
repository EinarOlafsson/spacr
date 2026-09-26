"""Provide statistical tests and multiple-comparison helpers.

Group-comparison functions delegate test selection to
:mod:`spacr.figures.stats` while preserving this module's established call
signatures and result keys. Two groups use Student's t, Welch's t, or
Mann–Whitney U as supported by the data; larger designs use one-way ANOVA,
Welch's ANOVA, or Kruskal–Wallis. Results identify the selected test and
include the assumption checks used to select it.

:func:`perform_normality_tests` reports underpowered checks as uninformative,
and :func:`perform_levene_test` uses the median-centred Brown–Forsythe
statistic. Imports of the plotting-backed statistical engine remain local so
callers that only need adjustment or contingency-table helpers avoid loading
the plotting stack.

Arrayed-screen hit statistics
-----------------------------

:func:`score_arrayed_screen` scores every well of an arrayed screen against
its negative control: robust z (median/MAD), SSMD with the method-of-moments,
UMVUE and robust estimators for designs with and without replicates (Zhang
2011, J Biomol Screen 16:775), and the B-score (Brideau et al. 2003), whose
row and column effects come from :func:`median_polish`, a port of R's
``stats::medpolish``. Scores are computed per plate or against the pooled
negative control, hits are called at chosen thresholds, and
:func:`write_hit_report` writes the ranked hit table as CSV and one plate
heatmap per statistic through :func:`spacr.plot.save_figure`. Wells are
located by :mod:`spacr.plate_qc` and controls named in the
:mod:`spacr.well_spec` notation, and the per-plate Z' comes from the
control-chart screen's own :func:`~spacr.qt.widgets.control_chart.zprime_frame`.

The constants: ``MAD_SCALE`` is ``1 / Phi^-1(0.75)`` = 1.4826, which makes
the MAD of a normal sample estimate its standard deviation (the factor R's
``mad()`` and so cellHTS2 apply). ``SSMD_ESTIMATORS`` are ``mm`` (method of
moments), ``umvue`` (uniformly minimal variance unbiased) and ``robust``
(median/MAD, SSMD*). ``DEFAULT_HIT_THRESHOLDS`` are 3 for every statistic:
SSMD 3 is Zhang's "strong" effect, and 3 robust sigma is the usual cut-off
for robust z and the B-score. ``MEDIAN_POLISH_MAX_ITER`` and
``MEDIAN_POLISH_EPS`` are R's ``medpolish`` defaults (10 and 0.01), kept so
the residuals agree with the B-score cellHTS2 computes.
"""

from dataclasses import dataclass, field as _field
from math import lgamma as _lgamma
from statistics import NormalDist as _NormalDist
from typing import Any, Dict, List, Optional, Sequence, Tuple
import warnings as _warnings

from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact
import itertools
from statsmodels.stats.multitest import multipletests

_ENGINE_TEST_NAMES = {
    "Student's t": 'T-test',
    "Welch's t": "Welch's T-test",
    'paired t': 'Paired T-test',
    'Wilcoxon signed-rank': 'Paired Wilcoxon test',
    'Mann-Whitney U': 'Mann-Whitney U test',
    'one-way ANOVA': 'One-way ANOVA',
    "Welch's ANOVA": "Welch's ANOVA",
    'Kruskal-Wallis': 'Kruskal-Wallis test',
}


def _grouped_values(df, grouping_column, data_column):
    """``{group: finite values}`` in the frame's own group order.

    Order matters: the sign of a t statistic is the sign of group 0 minus
    group 1, so the order the caller's frame presents the groups in is the
    order the result is reported in.

    Cleaning is delegated to the engine's own ``_clean`` rather than repeated
    here. Two spellings of "which values count" is the same class of defect as
    two spellings of "which test applies".
    """
    from .figures.stats import _clean

    return {group: _clean(df.loc[df[grouping_column] == group, data_column])
            for group in df[grouping_column].unique()}


def choose_p_adjust_method(num_groups, num_data_points):
    """Recommend a multiple-comparison correction method for the given design.

    :param num_groups: Number of unique groups being compared.
    :param num_data_points: Number of data points per group (balanced groups assumed).
    :returns: One of ``'holm'``, ``'fdr_bh'``, ``'sidak'``, or ``'bonferroni'``.
    """
    num_comparisons = (num_groups * (num_groups - 1)) // 2

    if num_comparisons <= 10 and num_data_points > 5:
        return 'holm'
    elif num_comparisons > 10 and num_data_points <= 5:
        return 'fdr_bh'
    elif num_comparisons <= 10:
        return 'sidak'
    else:
        return 'bonferroni'

def perform_normality_tests(df, grouping_column, data_columns):
    """Report per-group normality, and say when the check had no power.

    The VERDICT and the reported ROWS both come from
    :func:`spacr.figures.stats.check_normality`, so the summary and the detail
    cannot drift apart. That check is Shapiro-Wilk against a Bonferroni
    threshold across the groups, and it refuses -- reporting NaN and
    ``Informative=False`` -- when the smallest group is below
    :data:`spacr.figures.stats.MIN_N_FOR_ASSUMPTIONS`. A row whose statistic is
    NaN is not a failed computation; it is the check saying it could not see.

    This module used to run D'Agostino-Pearson or Shapiro per group and read
    "not rejected" as "normal", which on three replicates is a decision the
    data cannot support. The p-values it printed for such groups looked
    perfectly reasonable, which is why the defect survived.

    Groups with fewer than three observations are still reported as
    ``'Skipped'``: Shapiro-Wilk genuinely cannot run on two points.

    :param df: Input DataFrame containing the grouping and value columns.
    :param grouping_column: Column name identifying the group of each row.
    :param data_columns: Iterable of numeric column names to test.
    :returns: Tuple ``(is_normal, results)``. ``is_normal`` is True only when
        every requested column passes -- it used to be the verdict for the LAST
        column examined, so a two-column call answered about the wrong one.
        ``results`` is a list of per-group dicts carrying ``Comparison``,
        ``Test Statistic``, ``p-value``, ``Test Name``, ``Column``, ``n``,
        ``Informative`` and ``Verdict``.
    """
    from .figures.stats import check_normality

    normality_results = []
    column_verdicts = []

    for column in data_columns:
        groups = _grouped_values(df, grouping_column, column)
        for group, data in groups.items():
            n_samples = int(data.size)

            if n_samples < 3:
                print(f"Skipping normality test for group '{group}' on column '{column}' - Not enough data.")
                normality_results.append({
                    'Comparison': f'Normality test for {group} on {column}',
                    'Test Statistic': None,
                    'p-value': None,
                    'Test Name': 'Skipped',
                    'Column': column,
                    'n': n_samples,
                    'Informative': False,
                    'Verdict': (f'{n_samples} observations, too few to run a '
                                f'normality test at all'),
                })
                continue

            check = check_normality([data])
            normality_results.append({
                'Comparison': f'Normality test for {group} on {column}',
                'Test Statistic': check.statistic,
                'p-value': check.p_value,
                'Test Name': check.name,
                'Column': column,
                'n': n_samples,
                'Informative': check.informative,
                'Verdict': check.verdict,
            })

        column_verdicts.append(
            check_normality(list(groups.values())).passed)

    is_normal = bool(column_verdicts) and all(column_verdicts)
    return is_normal, normality_results


def perform_levene_test(df, grouping_column, data_column):
    """Levene's test for equal variance, MEDIAN-centred.

    Delegates to :func:`spacr.figures.stats.check_equal_variance`. Two things
    moved when it did, and both change the number a caller writes into a CSV:

    * The centring is the median (Brown-Forsythe), not SciPy's default mean.
      Median centring is less sensitive to non-normal data, and this function
      is called before the normality verdict is known.
    * Below :data:`spacr.figures.stats.MIN_N_FOR_ASSUMPTIONS` observations in
      the smallest group the result is ``(nan, nan)``. On three replicates
      Levene has almost no power, so "p = 0.7, variances are equal" means "we
      could not tell", and printing 0.7 into a results table invites exactly
      the reading that publishes a difference that is not there.

    :param df: Input DataFrame containing the grouping and value columns.
    :param grouping_column: Column name identifying the group of each row.
    :param data_column: Numeric column to test.
    :returns: Tuple ``(statistic, p_value)``, both NaN when the check had no
        power.
    """
    from .figures.stats import check_equal_variance

    groups = _grouped_values(df, grouping_column, data_column)
    check = check_equal_variance(list(groups.values()))
    return check.statistic, check.p_value


def perform_statistical_tests(df, grouping_column, data_columns, paired=False):
    """Run a supported group comparison for each data column.

    Parameters
    ----------
    df : pandas.DataFrame
        Data containing the grouping and numeric value columns.
    grouping_column : str
        Column identifying each observation's group.
    data_columns : iterable of str
        Numeric columns to test.
    paired : bool, default=False
        Request paired analysis. Paired analysis is not implemented; when
        enabled, no result rows are returned.

    Returns
    -------
    list of dict
        Per-column test name, statistic, p-value, sample counts, effect size,
        and selection rationale. Refused comparisons use
        ``Test Name='not testable'`` and include the reason.

    Notes
    -----
    :func:`spacr.figures.stats.compare` selects Student's t, Welch's t,
    Mann-Whitney U, one-way ANOVA, Welch's ANOVA, or Kruskal-Wallis from the
    available groups and informative assumption checks.
    """
    from .figures.stats import compare

    unique_groups = df[grouping_column].unique()
    test_results = []

    for column in data_columns:
        if paired:
            print("Performing paired tests (not implemented in this template).")
            continue

        groups = _grouped_values(df, grouping_column, column)
        counts = ' / '.join(str(int(values.size)) for values in groups.values())
        try:
            result = compare(groups)
        except ValueError as refusal:
            test_results.append({
                'Column': column,
                'Test Name': 'not testable',
                'Test Statistic': float('nan'),
                'p-value': float('nan'),
                'Groups': len(unique_groups),
                'n': counts,
                'Effect Size': float('nan'),
                'Effect': '',
                'Why This Test': str(refusal),
            })
            continue

        test_results.append({
            'Column': column,
            'Test Name': _ENGINE_TEST_NAMES.get(result.test, result.test),
            'Test Statistic': result.statistic,
            'p-value': result.p_value,
            'Groups': len(unique_groups),
            'n': ' / '.join(str(value) for value in result.n),
            'Effect Size': result.effect_size,
            'Effect': result.effect_name,
            'Why This Test': result.reason,
        })

    return test_results


def perform_posthoc_tests(df, grouping_column, data_column, is_normal):
    """Run pairwise post-hoc tests across groups with p-value adjustment.

    Uses Tukey HSD when data is normal, Dunn's test otherwise with a correction
    method chosen by :func:`choose_p_adjust_method`.

    ``is_normal`` should come from :func:`perform_normality_tests`, which is
    the one engine's verdict. Passing a hand-computed one puts the omnibus test
    and the pairwise tests on different footing -- Kruskal-Wallis across the
    groups followed by Tukey between them is two different assumptions about
    one dataset.

    :param df: Input DataFrame containing the grouping and value columns.
    :param grouping_column: Column name identifying the group of each row.
    :param data_column: Numeric column to compare across groups.
    :param is_normal: Whether the data satisfy the normality assumption.
    :returns: List of dicts with pairwise comparison metadata and p-values.
    """
    unique_groups = df[grouping_column].unique()
    posthoc_results = []

    if len(unique_groups) > 2:
        num_groups = len(unique_groups)
        num_data_points = len(df[data_column].dropna()) // num_groups
        p_adjust_method = choose_p_adjust_method(num_groups, num_data_points)

        if is_normal:
            tukey_result = pairwise_tukeyhsd(df[data_column], df[grouping_column], alpha=0.05)
            for comparison, p_value in zip(tukey_result._results_table.data[1:], tukey_result.pvalues):
                posthoc_results.append({
                    'Comparison': f"{comparison[0]} vs {comparison[1]}",
                    'Original p-value': None,
                    'Adjusted p-value': p_value,
                    'Adjusted Method': 'Tukey HSD',
                    'Test Name': 'Tukey HSD'
                })
        else:
            raw_dunn_result = sp.posthoc_dunn(df, val_col=data_column, group_col=grouping_column, p_adjust=None)
            adjusted_dunn_result = sp.posthoc_dunn(df, val_col=data_column, group_col=grouping_column, p_adjust=p_adjust_method)
            for i, group_a in enumerate(adjusted_dunn_result.index):
                for j, group_b in enumerate(adjusted_dunn_result.columns):
                    if i < j:
                        posthoc_results.append({
                            'Comparison': f"{group_a} vs {group_b}",
                            'Original p-value': raw_dunn_result.iloc[i, j],
                            'Adjusted p-value': adjusted_dunn_result.iloc[i, j],
                            'Adjusted Method': p_adjust_method,
                            'Test Name': "Dunn's Post-hoc"
                        })

    return posthoc_results

def chi_pairwise(raw_counts, verbose=False):
    """Run pairwise chi-square (or Fisher's exact) tests across group pairs.

    Uses Fisher's exact for 2x2 contingency tables and chi-square otherwise,
    then applies a multiple-comparison correction selected via
    :func:`choose_p_adjust_method`.

    Two degenerate inputs used to crash rather than report, and both are
    routine for a sparse per-well contingency table:

    * **Fewer than two groups.** There is no pair to compare, so the p-value
      correction was handed an empty list and raised ``ZeroDivisionError``.
      An empty result frame is the correct answer, not an exception.
    * **A category no group observed**, or a group with no observations at
      all. ``chi2_contingency`` computes an expected frequency of zero and
      raises ``ValueError``. A category with zero counts on both sides of a
      pair carries no information about that pair, so it is dropped before
      testing -- which is the standard handling, not a fudge. If dropping
      leaves fewer than two categories, or either group is empty, the test is
      genuinely undefined and the pair is reported with a NaN p-value and a
      reason instead of being silently omitted.

    :param raw_counts: Contingency-table DataFrame indexed by group.
    :param verbose: When True, print the resulting DataFrame.
    :returns: DataFrame with Group 1, Group 2, Test Name, p-value,
        p-value_adj, adj and note. Empty (with those columns) when there is
        no pair to compare.
    """
    columns = ['Group 1', 'Group 2', 'Test Name', 'p-value', 'p-value_adj',
               'adj', 'note']
    pairwise_results = []
    groups = raw_counts.index.unique()
    raw_p_values = []

    num_groups = len(groups)
    num_data_points = raw_counts.sum(axis=1).mean()

    if num_groups < 2:
        if verbose:
            print(f"\nPairwise Frequency Analysis: {num_groups} group(s), "
                  f"so there is no pair to compare.")
        return pd.DataFrame(columns=columns)

    p_adjust_method = choose_p_adjust_method(num_groups, num_data_points)

    for group1, group2 in itertools.combinations(groups, 2):
        pair = raw_counts.loc[[group1, group2]]
        kept = pair.loc[:, (pair != 0).any(axis=0)]
        contingency_table = kept.values
        note = ''
        n_dropped = pair.shape[1] - kept.shape[1]
        if n_dropped:
            note = f"{n_dropped} empty categor{'y' if n_dropped == 1 else 'ies'} dropped"

        if contingency_table.shape[1] < 2 or (contingency_table.sum(axis=1) == 0).any():
            empty_groups = [g for g, total in zip((group1, group2),
                                                  contingency_table.sum(axis=1))
                            if total == 0]
            reason = (f"no observations for {', '.join(map(str, empty_groups))}"
                      if empty_groups else
                      "fewer than two categories with any counts")
            pairwise_results.append({
                'Group 1': group1, 'Group 2': group2,
                'Test Name': 'not testable', 'p-value': float('nan'),
                'note': reason,
            })
            raw_p_values.append(float('nan'))
            continue

        if contingency_table.shape[1] == 2:
            oddsratio, p_value = fisher_exact(contingency_table)
            test_name = "Fisher's Exact Test"
        else:
            chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)
            test_name = 'Pairwise Chi-Square Test'

        pairwise_results.append({
            'Group 1': group1,
            'Group 2': group2,
            'Test Name': test_name,
            'p-value': p_value,
            'note': note,
        })
        raw_p_values.append(p_value)

    raw = np.asarray(raw_p_values, dtype=float)
    testable = ~np.isnan(raw)
    corrected_p_values = np.full(raw.shape, np.nan)
    if testable.any():
        corrected_p_values[testable] = multipletests(
            raw[testable], method=p_adjust_method)[1]

    for i, result in enumerate(pairwise_results):
        result['p-value_adj'] = corrected_p_values[i]

    pairwise_df = pd.DataFrame(pairwise_results)

    pairwise_df['adj'] = p_adjust_method
    pairwise_df = pairwise_df.reindex(columns=columns)

    if verbose:
        print("\nPairwise Frequency Analysis Results:")
        print(pairwise_df.to_string(index=False))

    return pairwise_df


MAD_SCALE = 1.0 / _NormalDist().inv_cdf(0.75)
HIT_METHODS = ("ssmd", "robust_z", "b_score")
HIT_METHOD_LABELS = {
    "ssmd": "SSMD",
    "robust_z": "robust z",
    "b_score": "B-score",
}

SSMD_ESTIMATORS = ("mm", "umvue", "robust")
HIT_DIRECTIONS = ("both", "up", "down")
HIT_SCOPES = ("plate", "pooled")

ROLE_NEGATIVE = "negative"
ROLE_POSITIVE = "positive"
ROLE_SAMPLE = "sample"
WELL_ROLES = (ROLE_NEGATIVE, ROLE_POSITIVE, ROLE_SAMPLE)

DEFAULT_HIT_THRESHOLDS = {"ssmd": 3.0, "robust_z": 3.0, "b_score": 3.0}
MEDIAN_POLISH_MAX_ITER = 10
MEDIAN_POLISH_EPS = 0.01


class HitScoringError(ValueError):
    """Raised when a screen cannot be scored, with the reason and the way out."""


def _finite(values) -> np.ndarray:
    """Return the finite entries of ``values`` as a flat float array."""
    array = np.asarray(values, dtype=float).ravel()
    return array[np.isfinite(array)]


def mad(values, *, scale: bool = True) -> float:
    """Median absolute deviation of the finite entries of ``values``.

    :param values: numbers; NaN and infinities are ignored.
    :param scale: multiply by :data:`MAD_SCALE` so the result estimates a
        normal standard deviation, as R's ``mad()`` does.
    :returns: the MAD, or NaN with no finite value.
    """
    data = _finite(values)
    if not data.size:
        return float("nan")
    raw = float(np.median(np.abs(data - np.median(data))))
    return raw * MAD_SCALE if scale else raw


def robust_z_scores(values, reference) -> np.ndarray:
    """Robust z against a reference: ``(x - median) / (1.4826 MAD)``.

    With the negative-control wells as ``reference`` this is Zhang's z*
    score: how far a well is from the negative control, in robust standard
    deviations of the negative control.

    :param values: the wells to score.
    :param reference: the reference wells, normally the negative controls.
    :returns: one score per value; NaN everywhere when the reference has
        fewer than two finite wells or no spread.
    """
    values = np.asarray(values, dtype=float)
    ref = _finite(reference)
    if ref.size < 2:
        return np.full(values.shape, np.nan)
    spread = mad(ref)
    if not np.isfinite(spread) or spread <= 0:
        return np.full(values.shape, np.nan)
    return (values - float(np.median(ref))) / spread


def _gamma_ratio(n: int) -> float:
    """``Gamma((n-1)/2) / Gamma((n-2)/2)``, through log-gamma for large n."""
    return float(np.exp(_lgamma((n - 1) / 2.0) - _lgamma((n - 2) / 2.0)))


def _check_estimator(estimator: str) -> None:
    """Refuse an SSMD estimator that is not one of :data:`SSMD_ESTIMATORS`."""
    if estimator not in SSMD_ESTIMATORS:
        raise HitScoringError(
            f"unknown SSMD estimator {estimator!r}; choose one of "
            f"{', '.join(SSMD_ESTIMATORS)}")


def ssmd_unreplicated(values, reference, estimator: str = "mm") -> np.ndarray:
    """SSMD of single wells against a negative reference, without replicates.

    Zhang (2011), assuming a tested well has the variability of the negative
    reference ``N``:

    * ``mm``: ``(x - mean_N) / (sqrt(2) s_N)``;
    * ``umvue``: ``(x - mean_N) / (sqrt(2 (n_N - 1) / K) s_N)`` with
      ``K = 2 (Gamma((n_N-1)/2) / Gamma((n_N-2)/2))^2``, needing ``n_N >= 3``;
    * ``robust`` (SSMD*): ``(x - median_N) / (1.4826 sqrt(2) MAD_N)``.

    :param values: the wells to score.
    :param reference: the negative-reference wells.
    :param estimator: one of :data:`SSMD_ESTIMATORS`.
    :returns: one SSMD per value; NaN where the reference cannot support the
        estimator (too few wells, or no spread).
    :raises HitScoringError: for an unknown estimator.
    """
    _check_estimator(estimator)
    values = np.asarray(values, dtype=float)
    if estimator == "robust":
        return robust_z_scores(values, reference) / np.sqrt(2.0)
    ref = _finite(reference)
    n = ref.size
    empty = np.full(values.shape, np.nan)
    if n < (3 if estimator == "umvue" else 2):
        return empty
    sd = float(ref.std(ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        return empty
    ssmd = (values - float(ref.mean())) / (np.sqrt(2.0) * sd)
    if estimator == "umvue":
        k = 2.0 * _gamma_ratio(n) ** 2
        ssmd = ssmd * np.sqrt(k / (n - 1.0))
    return ssmd


def ssmd_replicated(differences, estimator: str = "umvue") -> float:
    """SSMD of one treatment from its replicate paired differences.

    ``differences`` are ``D_j = x_j - median(negative reference)`` for each
    replicate ``j``, the reference taken on the replicate's own plate (Zhang
    2011, screens with replicates):

    * ``mm``: ``mean(D) / sd(D)``;
    * ``umvue``: ``sqrt(2/(n-1)) Gamma((n-1)/2) / Gamma((n-2)/2) mean(D) /
      sd(D)``, needing ``n >= 3``;
    * ``robust``: ``median(D) / (1.4826 MAD(D))``.

    :param differences: the replicate differences of one treatment.
    :param estimator: one of :data:`SSMD_ESTIMATORS`.
    :returns: the SSMD, or NaN with too few replicates or no spread.
    :raises HitScoringError: for an unknown estimator.
    """
    _check_estimator(estimator)
    d = _finite(differences)
    n = d.size
    if estimator == "robust":
        spread = mad(d) if n >= 2 else float("nan")
        if not np.isfinite(spread) or spread <= 0:
            return float("nan")
        return float(np.median(d)) / spread
    if n < (3 if estimator == "umvue" else 2):
        return float("nan")
    sd = float(d.std(ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        return float("nan")
    ssmd = float(d.mean()) / sd
    if estimator == "umvue":
        ssmd *= np.sqrt(2.0 / (n - 1.0)) * _gamma_ratio(n)
    return float(ssmd)


@dataclass
class MedianPolish:
    """Tukey's median polish of one plate: ``x = overall + row + column + residual``.

    :param overall: the fitted grand effect.
    :param row: one effect per plate row; NaN for a row with no fitted well.
    :param column: one effect per plate column; NaN likewise.
    :param residuals: the residual matrix; NaN where the input was NaN.
    :param iterations: sweeps run.
    :param converged: whether the relative change fell under the tolerance.
    """

    overall: float
    row: np.ndarray
    column: np.ndarray
    residuals: np.ndarray
    iterations: int
    converged: bool

    def fitted(self) -> np.ndarray:
        """The additive fit ``overall + row + column`` on the full grid."""
        return self.overall + self.row[:, None] + self.column[None, :]


def _nanmedian(array, axis=None):
    """``np.nanmedian`` returning NaN for an all-NaN slice, without a warning."""
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(array, axis=axis)


def median_polish(matrix, *, max_iter: int = MEDIAN_POLISH_MAX_ITER,
                  eps: float = MEDIAN_POLISH_EPS) -> MedianPolish:
    """Tukey's two-way median polish, NaN-aware, following R's ``medpolish``.

    The sweep order, the centring of the effects and the stopping rule are
    R's, so the fit matches ``stats::medpolish(x, na.rm = TRUE)`` and the
    cellHTS2 B-score built on it. A row or column with no finite well gets a
    NaN effect instead of a guessed one.

    :param matrix: 2-D array, NaN for a well left out of the fit.
    :param max_iter: most sweeps.
    :param eps: stop when ``|sum|r| - previous| < eps * sum|r|``.
    :returns: the :class:`MedianPolish`.
    :raises HitScoringError: for an input that is not two-dimensional.
    """
    z = np.array(matrix, dtype=float, copy=True)
    if z.ndim != 2:
        raise HitScoringError("median polish needs a two-dimensional plate")
    n_rows, n_cols = z.shape
    present = np.isfinite(z)
    z[~present] = np.nan
    row = np.zeros(n_rows)
    col = np.zeros(n_cols)
    overall = 0.0
    old = 0.0
    converged = False
    iterations = 0
    for iterations in range(1, int(max_iter) + 1):
        delta_r = np.nan_to_num(_nanmedian(z, axis=1))
        z = z - delta_r[:, None]
        row = row + delta_r
        delta = float(np.nan_to_num(_nanmedian(col)))
        col = col - delta
        overall += delta
        delta_c = np.nan_to_num(_nanmedian(z, axis=0))
        z = z - delta_c[None, :]
        col = col + delta_c
        delta = float(np.nan_to_num(_nanmedian(row)))
        row = row - delta
        overall += delta
        new = float(np.nansum(np.abs(z)))
        converged = new == 0 or abs(new - old) < eps * new
        if converged:
            break
        old = new
    row = np.where(present.any(axis=1), row, np.nan)
    col = np.where(present.any(axis=0), col, np.nan)
    return MedianPolish(overall=float(overall), row=row, column=col,
                        residuals=z, iterations=iterations,
                        converged=bool(converged))


def b_scores(matrix, *, fit_mask=None, scale: Optional[float] = None
             ) -> Tuple[np.ndarray, MedianPolish, float]:
    """B-score of every well on one plate (Brideau et al. 2003).

    Row and column effects are fitted by :func:`median_polish` on the wells
    in ``fit_mask`` (normally the sample wells, so a control column does not
    become a column effect), then removed from every well on the plate. The
    residuals are divided by the scaled MAD of the fitted wells' residuals,
    as cellHTS2 does; Brideau's unscaled MAD differs by the constant 1.4826
    only and ranks the wells identically. A row or column with no fitted
    well, such as a column holding only controls, has no effect to remove,
    so its wells are scored against the overall and the other effect alone.

    :param matrix: the plate, NaN for an absent well.
    :param fit_mask: boolean matrix of the wells the effects are fitted on;
        ``None`` fits on every finite well.
    :param scale: divide by this instead of the plate's own residual MAD,
        which is how the pooled scope applies the screen-wide MAD.
    :returns: ``(scores, polish, scale_used)``; a score is NaN for an absent
        well, and everywhere when the fitted residuals have no spread.
    """
    values = np.asarray(matrix, dtype=float)
    fit = np.isfinite(values)
    if fit_mask is not None:
        fit &= np.asarray(fit_mask, dtype=bool)
    polish = median_polish(np.where(fit, values, np.nan))
    residuals = values - (polish.overall
                          + np.nan_to_num(polish.row)[:, None]
                          + np.nan_to_num(polish.column)[None, :])
    if scale is None:
        scale = mad(residuals[fit])
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0:
        return np.full(values.shape, np.nan), polish, scale
    return residuals / scale, polish, scale


def call_hits(scores, threshold: float, direction: str = "both") -> np.ndarray:
    """Which scores pass ``threshold`` in ``direction``. NaN is never a hit.

    :param scores: the statistic per well.
    :param threshold: the cut-off; its sign is ignored.
    :param direction: ``both`` (``|s| >= t``), ``up`` (``s >= t``) or ``down``
        (``s <= -t``).
    :returns: a boolean array.
    :raises HitScoringError: for an unknown direction.
    """
    if direction not in HIT_DIRECTIONS:
        raise HitScoringError(
            f"unknown direction {direction!r}; choose one of "
            f"{', '.join(HIT_DIRECTIONS)}")
    s = np.asarray(scores, dtype=float)
    t = abs(float(threshold))
    finite = np.isfinite(s)
    filled = np.where(finite, s, 0.0)
    if direction == "up":
        hit = filled >= t
    elif direction == "down":
        hit = filled <= -t
    else:
        hit = np.abs(filled) >= t
    return hit & finite


def _levels(values) -> Tuple[str, ...]:
    """Normalise a level or a list of levels to a tuple of non-empty strings."""
    if values is None:
        return ()
    if isinstance(values, str):
        values = [values]
    return tuple(str(v) for v in values if str(v).strip())


def _well_mode(series: pd.Series) -> Any:
    """The most common non-null value among a well's objects, or None."""
    clean = series.dropna()
    if not len(clean):
        return None
    return clean.astype(str).mode().iloc[0]


def _layout_for(n_rows: int, n_cols: int) -> int:
    """The smallest :mod:`spacr.well_spec` layout holding an observed extent."""
    from . import well_spec

    for wells in sorted(well_spec.LAYOUTS):
        rows, cols = well_spec.LAYOUTS[wells]
        if n_rows <= rows and n_cols <= cols:
            return wells
    return max(well_spec.LAYOUTS)


def screen_wells(frame: pd.DataFrame, value_col: str, *,
                 plate_column: Optional[str] = None,
                 control_column: Optional[str] = None,
                 negative_levels=(), positive_levels=(),
                 negative_wells=None, positive_wells=None,
                 treatment_column: Optional[str] = None,
                 grouping: str = "mean", min_count: int = 0) -> pd.DataFrame:
    """Collapse a measurement table to one row per well, with each well's role.

    Wells are located by the plate-QC reader (:mod:`spacr.plate_qc`: ``prc``,
    a rowID/columnID pair or a ``well`` column), so this and the Plate Viewer
    agree about where every object sits. Controls are named by a column and
    its levels, as the control-chart screen names them, or by plate position
    in the ``negative_control_wells`` / ``positive_control_wells`` notation of
    :mod:`spacr.well_spec` (``c1``, ``r1``, ``A01``), or both.

    :param frame: per-object or per-well table.
    :param value_col: the measurement to score.
    :param plate_column: column naming the plate; default the reader's choice.
    :param control_column: column holding the control labels.
    :param negative_levels: level(s) of ``control_column`` that are the
        negative control.
    :param positive_levels: level(s) that are the positive control.
    :param negative_wells: well spec of the negative-control wells.
    :param positive_wells: well spec of the positive-control wells.
    :param treatment_column: column naming what is in each well, for the
        replicate SSMD; each well keeps its most common value.
    :param grouping: ``mean`` or ``median`` of the objects in a well.
    :param min_count: drop wells with fewer objects than this.
    :returns: one row per well: ``plateID``, ``well``, ``row_index``,
        ``column_index``, ``prc``, ``n``, ``value``, ``role`` and, when asked,
        ``treatment``.
    :raises HitScoringError: when the table cannot be scored.
    """
    from . import plate_qc, schema, well_spec

    if grouping not in ("mean", "median"):
        raise HitScoringError(
            f"grouping must be 'mean' or 'median', not {grouping!r}")
    if frame is None or not len(frame):
        raise HitScoringError("the table is empty")
    if value_col not in frame.columns:
        raise HitScoringError(f"the table has no column {value_col!r}")
    for label, column in (("control", control_column),
                          ("treatment", treatment_column),
                          ("plate", plate_column)):
        if column and column not in frame.columns:
            raise HitScoringError(
                f"the {label} column {column!r} is not in the table")
    negative_levels = _levels(negative_levels)
    positive_levels = _levels(positive_levels)
    overlap = set(negative_levels) & set(positive_levels)
    if overlap:
        raise HitScoringError(
            f"{', '.join(sorted(overlap))} is named both negative and "
            f"positive control")
    if (negative_levels or positive_levels) and not control_column:
        raise HitScoringError(
            "control levels are named but no control column to find them in")

    try:
        located, _notes = plate_qc._identify_wells(frame)
    except ValueError as exc:
        raise HitScoringError(str(exc)) from exc
    if plate_column:
        located["plateID"] = frame[plate_column].astype(str).to_numpy()
    located["row_index"] = located["rowID"].map(plate_qc.parse_row_label)
    located["column_index"] = located["columnID"].map(
        plate_qc.parse_column_label)
    located = located.dropna(subset=["row_index", "column_index"])
    if not len(located):
        raise HitScoringError("no row of the table sits in a readable well")
    located["row_index"] = located["row_index"].astype(int)
    located["column_index"] = located["column_index"].astype(int)
    located["plateID"] = located["plateID"].astype(str)
    located["__value__"] = pd.to_numeric(located[value_col], errors="coerce")

    keys = ["plateID", "row_index", "column_index"]
    grouped = located.groupby(keys, sort=True, observed=True)
    wells = grouped["__value__"].agg(grouping).rename("value").to_frame()
    wells["n"] = grouped.size()
    if control_column:
        wells["__label__"] = grouped[control_column].agg(_well_mode)
    if treatment_column:
        wells["treatment"] = grouped[treatment_column].agg(_well_mode)
    wells = wells.reset_index()
    if min_count and int(min_count) > 0:
        wells = wells[wells["n"] >= int(min_count)].reset_index(drop=True)
    if not len(wells):
        raise HitScoringError(
            f"no well has {int(min_count)} or more objects; lower min_count")

    negative = np.zeros(len(wells), dtype=bool)
    positive = np.zeros(len(wells), dtype=bool)
    if control_column:
        labels = wells["__label__"].astype(str).to_numpy()
        negative |= np.isin(labels, list(negative_levels))
        positive |= np.isin(labels, list(positive_levels))
        wells = wells.drop(columns="__label__")
    if negative_wells or positive_wells:
        layout = _layout_for(int(wells["row_index"].max()),
                             int(wells["column_index"].max()))
        cells = list(zip(wells["row_index"].astype(int),
                         wells["column_index"].astype(int)))
        try:
            neg_cells = well_spec.parse(negative_wells, layout)
            pos_cells = well_spec.parse(positive_wells, layout)
        except well_spec.WellSpecError as exc:
            raise HitScoringError(str(exc)) from exc
        negative |= np.asarray([c in neg_cells for c in cells], dtype=bool)
        positive |= np.asarray([c in pos_cells for c in cells], dtype=bool)
    clash = negative & positive
    if clash.any():
        raise HitScoringError(
            f"{int(clash.sum())} well(s) are named both negative and "
            f"positive control")
    role = np.full(len(wells), ROLE_SAMPLE, dtype=object)
    role[negative] = ROLE_NEGATIVE
    role[positive] = ROLE_POSITIVE
    wells["role"] = role
    wells["well"] = [plate_qc.well_id(r, c) for r, c in
                     zip(wells["row_index"], wells["column_index"])]
    wells["prc"] = [schema.compose_prc(p, int(r), int(c)) for p, r, c in
                    zip(wells["plateID"], wells["row_index"],
                        wells["column_index"])]
    order = ["plateID", "well", "row_index", "column_index", "prc", "n",
             "value", "role"] + (["treatment"] if treatment_column else [])
    wells = wells[order]
    wells.attrs = {"value_col": value_col, "grouping": grouping,
                   "min_count": int(min_count or 0)}
    return wells


def _plate_grid(rows: np.ndarray, cols: np.ndarray, values) -> np.ndarray:
    """Place per-well values onto the plate's nominal grid, NaN elsewhere."""
    from . import plate_qc

    _fmt, n_rows, n_cols = plate_qc.infer_plate_format(int(rows.max()),
                                                        int(cols.max()))
    grid = np.full((n_rows, n_cols), np.nan)
    grid[rows - 1, cols - 1] = np.asarray(values, dtype=float)
    return grid


@dataclass
class ArrayedHitResult:
    """Per-well scores, per-treatment SSMD and per-plate QC of one screen.

    :param wells: one row per well with ``robust_z``, ``ssmd``, ``b_score``,
        ``difference``, the ``hit_<method>`` flags, ``hit`` (by
        ``options['rank_by']``) and ``rank``.
    :param treatments: replicate SSMD per treatment; empty without a
        treatment column.
    :param plates: per-plate summary including Z'.
    :param options: the settings the scores were computed with.
    :param notes: sentences about what could not be scored and why.
    """

    wells: pd.DataFrame
    treatments: pd.DataFrame
    plates: pd.DataFrame
    options: Dict[str, Any]
    notes: List[str] = _field(default_factory=list)

    def hits(self) -> pd.DataFrame:
        """The called sample wells, strongest first."""
        return hit_table(self.wells)

    def report(self) -> str:
        """The result in sentences, for a text panel or a log."""
        o = self.options
        method = HIT_METHOD_LABELS.get(o["rank_by"], o["rank_by"])
        samples = int((self.wells["role"] == ROLE_SAMPLE).sum())
        lines = [
            f"Hit scoring of {o['value_col']} over {len(self.plates)} "
            f"plate(s) and {samples} sample well(s), against the negative "
            f"control ({o['scope']} scope, SSMD estimator "
            f"{o['ssmd_estimator']}).",
            f"Hits called by {method} at {o['thresholds'][o['rank_by']]:g} "
            f"({o['direction']}): {int(self.wells['hit'].sum())}.",
        ]
        for name in HIT_METHODS:
            count = int(self.wells[f"hit_{name}"].sum())
            lines.append(
                f"  {HIT_METHOD_LABELS[name]} at {o['thresholds'][name]:g}: "
                f"{count} well(s)")
        if len(self.treatments):
            lines.append(
                f"Replicate SSMD ({o['replicate_estimator']}) over "
                f"{len(self.treatments)} treatment(s): "
                f"{int(self.treatments['hit'].sum())} called.")
        for _, row in self.plates.iterrows():
            z = row.get("zprime")
            if z is not None and np.isfinite(z):
                lines.append(f"  plate {row['plateID']}: Z' = {z:.2f}")
        lines.extend(self.notes)
        return "\n".join(lines)


def score_screen(wells: pd.DataFrame, *, scope: str = "plate",
                 ssmd_estimator: str = "mm",
                 replicate_estimator: str = "umvue",
                 thresholds: Optional[Dict[str, float]] = None,
                 direction: str = "both", rank_by: str = "ssmd"
                 ) -> ArrayedHitResult:
    """Score every well of an arrayed screen against its negative control.

    Robust z and SSMD are taken against the negative-control wells of the
    well's own plate (``scope='plate'``) or of every plate together
    (``scope='pooled'``). The B-score is always fitted per plate, because row
    and column effects belong to a plate; the pooled scope divides by the
    screen-wide residual MAD instead of each plate's own. Controls are scored
    too, which is how a positive control shows the assay window, but only
    sample wells are called as hits.

    :param wells: the table :func:`screen_wells` returns.
    :param scope: one of :data:`HIT_SCOPES`.
    :param ssmd_estimator: per-well SSMD estimator, one of
        :data:`SSMD_ESTIMATORS`.
    :param replicate_estimator: SSMD estimator for treatments with
        replicates.
    :param thresholds: cut-off per method; missing ones use
        :data:`DEFAULT_HIT_THRESHOLDS`.
    :param direction: one of :data:`HIT_DIRECTIONS`.
    :param rank_by: the method that decides ``hit`` and ``rank``.
    :returns: an :class:`ArrayedHitResult`.
    :raises HitScoringError: for an unknown option or a screen without
        negative-control wells.
    """
    if scope not in HIT_SCOPES:
        raise HitScoringError(
            f"unknown scope {scope!r}; choose one of {', '.join(HIT_SCOPES)}")
    if rank_by not in HIT_METHODS:
        raise HitScoringError(
            f"unknown method {rank_by!r}; choose one of "
            f"{', '.join(HIT_METHODS)}")
    _check_estimator(ssmd_estimator)
    _check_estimator(replicate_estimator)
    if direction not in HIT_DIRECTIONS:
        raise HitScoringError(
            f"unknown direction {direction!r}; choose one of "
            f"{', '.join(HIT_DIRECTIONS)}")
    cuts = dict(DEFAULT_HIT_THRESHOLDS)
    cuts.update({k: abs(float(v)) for k, v in (thresholds or {}).items()
                 if k in HIT_METHODS and v is not None})

    out = wells.copy().reset_index(drop=True)
    notes: List[str] = []
    values = out["value"].to_numpy(dtype=float)
    plates_col = out["plateID"].astype(str).to_numpy()
    role = out["role"].to_numpy()
    negative = role == ROLE_NEGATIVE
    sample = role == ROLE_SAMPLE
    if not negative.any():
        raise HitScoringError(
            "no negative-control well: robust z and SSMD are distances from "
            "the negative control. Name its level or its wells.")

    plate_names = list(dict.fromkeys(plates_col))
    robust = np.full(len(out), np.nan)
    ssmd = np.full(len(out), np.nan)
    diff = np.full(len(out), np.nan)
    pooled_ref = values[negative]
    for plate in plate_names:
        on = plates_col == plate
        ref = pooled_ref if scope == "pooled" else values[on & negative]
        if _finite(ref).size < 2:
            notes.append(
                f"plate {plate}: fewer than two negative-control wells with a "
                f"value, so robust z and SSMD are not defined there")
            continue
        robust[on] = robust_z_scores(values[on], ref)
        ssmd[on] = ssmd_unreplicated(values[on], ref, ssmd_estimator)
        diff[on] = values[on] - float(np.median(_finite(ref)))
        if not np.isfinite(ssmd[on]).any():
            notes.append(
                f"plate {plate}: the negative control cannot support the "
                f"{ssmd_estimator} SSMD (too few wells or no spread)")

    grids = {}
    residual_pool: List[np.ndarray] = []
    row_index = out["row_index"].to_numpy(dtype=int)
    column_index = out["column_index"].to_numpy(dtype=int)
    for plate in plate_names:
        on = np.flatnonzero(plates_col == plate)
        rows, cols = row_index[on], column_index[on]
        grid = _plate_grid(rows, cols, values[on])
        fit = _plate_grid(rows, cols, sample[on].astype(float)) == 1.0
        _scores, polish, _scale = b_scores(grid, fit_mask=fit)
        grids[plate] = (on, rows, cols, grid, fit)
        residuals = grid - polish.fitted()
        residual_pool.append(residuals[fit & np.isfinite(residuals)])
    pooled_scale = (mad(np.concatenate(residual_pool)) if residual_pool
                    else float("nan"))
    b = np.full(len(out), np.nan)
    polishes: Dict[str, Tuple[MedianPolish, float]] = {}
    for plate in plate_names:
        on, rows, cols, grid, fit = grids[plate]
        scores, polish, scale = b_scores(
            grid, fit_mask=fit,
            scale=pooled_scale if scope == "pooled" else None)
        polishes[plate] = (polish, scale)
        b[on] = scores[rows - 1, cols - 1]
        if not np.isfinite(scores).any():
            notes.append(f"plate {plate}: too few sample wells for a B-score")

    out["robust_z"] = robust
    out["ssmd"] = ssmd
    out["b_score"] = b
    out["difference"] = diff
    for name in HIT_METHODS:
        out[f"hit_{name}"] = call_hits(out[name], cuts[name], direction) & sample
    out["n_methods_called"] = out[[f"hit_{m}" for m in HIT_METHODS]].sum(
        axis=1).astype(int)
    out["hit"] = out[f"hit_{rank_by}"]
    out["rank"] = _rank(out[rank_by].to_numpy(dtype=float), sample, direction)

    options = {"value_col": wells.attrs.get("value_col", "value"),
               "scope": scope, "ssmd_estimator": ssmd_estimator,
               "replicate_estimator": replicate_estimator,
               "thresholds": cuts, "direction": direction,
               "rank_by": rank_by}
    treatments = treatment_ssmd(out, estimator=replicate_estimator,
                                threshold=cuts["ssmd"], direction=direction)
    plates = _plate_summary(out, plate_names, polishes)
    return ArrayedHitResult(wells=out, treatments=treatments, plates=plates,
                            options=options, notes=notes)


def _rank(scores: np.ndarray, eligible: np.ndarray, direction: str
          ) -> np.ndarray:
    """1-based rank of each eligible finite score, strongest first; NaN else."""
    if direction == "both":
        key = np.abs(scores)
    elif direction == "up":
        key = scores
    else:
        key = -scores
    ok = np.asarray(eligible, dtype=bool) & np.isfinite(key)
    rank = np.full(scores.shape, np.nan)
    order = np.flatnonzero(ok)[np.argsort(-key[ok], kind="stable")]
    rank[order] = np.arange(1, order.size + 1)
    return rank


def treatment_ssmd(scored: pd.DataFrame, *, estimator: str = "umvue",
                   threshold: float = DEFAULT_HIT_THRESHOLDS["ssmd"],
                   direction: str = "both") -> pd.DataFrame:
    """Replicate SSMD per treatment from the sample wells' paired differences.

    :param scored: the ``wells`` of an :class:`ArrayedHitResult`; without a
        ``treatment`` column the result is empty.
    :param estimator: one of :data:`SSMD_ESTIMATORS`.
    :param threshold: SSMD cut-off for the ``hit`` call.
    :param direction: one of :data:`HIT_DIRECTIONS`.
    :returns: one row per treatment, strongest first: ``treatment``,
        ``n_replicates``, ``plates``, ``mean_difference``, ``ssmd_mm``,
        ``ssmd_umvue``, ``ssmd_robust``, ``ssmd`` (the chosen estimator),
        ``median_robust_z``, ``median_b_score``, ``hit`` and ``rank``.
    """
    _check_estimator(estimator)
    columns = ["treatment", "n_replicates", "plates", "mean_difference",
               "ssmd_mm", "ssmd_umvue", "ssmd_robust", "ssmd",
               "median_robust_z", "median_b_score", "hit", "rank"]
    if "treatment" not in scored.columns:
        return pd.DataFrame(columns=columns)
    rows = scored[(scored["role"] == ROLE_SAMPLE)
                  & scored["treatment"].notna()]
    records = []
    for name, group in rows.groupby("treatment", sort=True):
        d = group["difference"].to_numpy(dtype=float)
        finite = np.isfinite(d)
        record = {
            "treatment": name,
            "n_replicates": int(finite.sum()),
            "plates": ",".join(sorted(set(group["plateID"].astype(str)))),
            "mean_difference": float(d[finite].mean()) if finite.any()
            else float("nan"),
        }
        for est in SSMD_ESTIMATORS:
            record[f"ssmd_{est}"] = ssmd_replicated(d, est)
        record["ssmd"] = record[f"ssmd_{estimator}"]
        record["median_robust_z"] = float(_nanmedian(
            group["robust_z"].to_numpy(dtype=float)))
        record["median_b_score"] = float(_nanmedian(
            group["b_score"].to_numpy(dtype=float)))
        records.append(record)
    if not records:
        return pd.DataFrame(columns=columns)
    table = pd.DataFrame(records)
    table["hit"] = call_hits(table["ssmd"], threshold, direction)
    table["rank"] = _rank(table["ssmd"].to_numpy(dtype=float),
                          np.ones(len(table), dtype=bool), direction)
    return table.sort_values("rank", na_position="last",
                             ignore_index=True)[columns]


def _plate_summary(scored: pd.DataFrame, plate_names: Sequence[str],
                   polishes: Dict[str, Tuple[MedianPolish, float]]
                   ) -> pd.DataFrame:
    """One row per plate: control counts and spread, Z', hits, polish state.

    Z' comes from :func:`spacr.qt.widgets.control_chart.zprime_frame` on the
    well values, the function the control-chart screen charts, so the two
    never disagree about a plate's assay window.
    """
    records = []
    for plate in plate_names:
        on = scored[scored["plateID"].astype(str) == plate]
        neg = _finite(on.loc[on["role"] == ROLE_NEGATIVE, "value"])
        pos = on[on["role"] == ROLE_POSITIVE]
        polish, scale = polishes[plate]
        record = {
            "plateID": plate,
            "n_wells": int(len(on)),
            "n_sample": int((on["role"] == ROLE_SAMPLE).sum()),
            "n_negative": int(neg.size),
            "n_positive": int(len(pos)),
            "negative_median": float(np.median(neg)) if neg.size
            else float("nan"),
            "negative_mad": mad(neg),
            "negative_mean": float(neg.mean()) if neg.size else float("nan"),
            "negative_sd": float(neg.std(ddof=1)) if neg.size > 1
            else float("nan"),
            "positive_median_ssmd": float(_nanmedian(
                pos["ssmd"].to_numpy(dtype=float))) if len(pos)
            else float("nan"),
            "b_score_scale": scale,
            "polish_iterations": polish.iterations,
            "polish_converged": polish.converged,
        }
        for method in HIT_METHODS:
            record[f"hits_{method}"] = int(on[f"hit_{method}"].sum())
        records.append(record)
    table = pd.DataFrame(records)
    table["zprime"] = np.nan
    try:
        from .qt.widgets.control_chart import (ControlChartSpec,
                                               ZPRIME_PLATE, ZPRIME_VALUE,
                                               zprime_frame)
        spec = ControlChartSpec(
            value="value", plate="plateID", control_column="role",
            control_levels=(ROLE_POSITIVE, ROLE_NEGATIVE),
            positive_levels=(ROLE_POSITIVE,),
            negative_levels=(ROLE_NEGATIVE,))
        zp = zprime_frame(scored, spec)
        mapping = dict(zip(zp[ZPRIME_PLATE].astype(str), zp[ZPRIME_VALUE]))
        table["zprime"] = table["plateID"].map(mapping).astype(float)
    except (ImportError, ValueError):
        pass
    return table


def hit_table(scored: pd.DataFrame, *, hits_only: bool = True
              ) -> pd.DataFrame:
    """The ranked hit table: sample wells, strongest first.

    :param scored: the ``wells`` of an :class:`ArrayedHitResult`.
    :param hits_only: keep only the called wells.
    :returns: the ranked rows with the identifying columns first.
    """
    rows = scored[scored["role"] == ROLE_SAMPLE]
    if hits_only:
        rows = rows[rows["hit"].astype(bool)]
    lead = ["rank", "plateID", "well", "prc"] + (
        ["treatment"] if "treatment" in rows.columns else []) + [
        "n", "value", "ssmd", "robust_z", "b_score", "n_methods_called"]
    rest = [c for c in rows.columns if c not in lead]
    return rows.sort_values("rank", na_position="last",
                            ignore_index=True)[lead + rest]


def score_arrayed_screen(frame: pd.DataFrame, value_col: str, *,
                         plate_column: Optional[str] = None,
                         control_column: Optional[str] = None,
                         negative_levels=(), positive_levels=(),
                         negative_wells=None, positive_wells=None,
                         treatment_column: Optional[str] = None,
                         grouping: str = "mean", min_count: int = 0,
                         **options) -> ArrayedHitResult:
    """:func:`screen_wells` then :func:`score_screen`, in one call.

    Example:
        .. code-block:: python

            from spacr.sp_stats import score_arrayed_screen, write_hit_report
            result = score_arrayed_screen(
                df, 'cell_area', negative_wells='c1', positive_wells='c24',
                treatment_column='gene', rank_by='ssmd', scope='plate')
            write_hit_report(result, 'results/hits')

    :param frame: the measurement table.
    :param value_col: the measurement to score.
    :param options: passed to :func:`score_screen` (``scope``,
        ``ssmd_estimator``, ``replicate_estimator``, ``thresholds``,
        ``direction``, ``rank_by``).
    :returns: an :class:`ArrayedHitResult`.
    """
    wells = screen_wells(
        frame, value_col, plate_column=plate_column,
        control_column=control_column, negative_levels=negative_levels,
        positive_levels=positive_levels, negative_wells=negative_wells,
        positive_wells=positive_wells, treatment_column=treatment_column,
        grouping=grouping, min_count=min_count)
    return score_screen(wells, **options)


def hit_heatmap(result: ArrayedHitResult, method: str, *,
                target: Optional[str] = None):
    """Every plate's ``method`` scores on one diverging scale, hits outlined.

    Drawn by :func:`spacr.figures.plates.build_plates`, the house plate small
    multiple, with :func:`spacr.figures.plates.score_ramp` and a scale
    symmetric about zero, so the same colour is the same score on every plate
    and on either side of the negative control.

    :param result: the scored screen.
    :param method: one of :data:`HIT_METHODS`.
    :param target: ``'screen'`` or ``'print'``; default the preference.
    :returns: ``(figure, panel)`` as ``build_plates`` returns them.
    :raises HitScoringError: for an unknown method.
    """
    from .figures.plates import build_plates, score_ramp
    from .figures.style import theme_target

    if method not in HIT_METHODS:
        raise HitScoringError(
            f"unknown method {method!r}; choose one of "
            f"{', '.join(HIT_METHODS)}")
    target = target or theme_target()
    frame = result.wells[["prc", method]].copy()
    frame[f"hit_{method}"] = result.wells[f"hit_{method}"].astype(float)
    scores = _finite(frame[method])
    cut = float(result.options["thresholds"][method])
    span = cut
    if scores.size:
        span = max(cut, float(np.quantile(np.abs(scores), 0.98)))
    return build_plates(frame, method, grouping="mean",
                        cmap=score_ramp(target), limits=(-span, span),
                        target=target, outline=f"hit_{method}")


def write_hit_report(result: ArrayedHitResult, out_dir, *,
                     methods: Sequence[str] = HIT_METHODS,
                     target: Optional[str] = None) -> Dict[str, str]:
    """Write the scored screen: CSV tables and one plate heatmap per method.

    Figures go through :func:`spacr.plot.save_figure`, so their format,
    resolution and print repaint follow the user's figure preferences. The
    low-contrast colour warning is off for these: the centre of a diverging
    score map is near the page colour on purpose, because a score of zero
    is the negative control and is meant to recede.

    :param result: the scored screen.
    :param out_dir: folder to write into; created if absent.
    :param methods: which heatmaps to draw.
    :param target: figure target passed to :func:`hit_heatmap`.
    :returns: ``{name: path}`` of everything written: ``hit_table`` (the
        ranked hits), ``hit_scores_wells``, ``hit_plates``,
        ``hit_treatments`` with a treatment column, and
        ``hit_heatmap_<method>`` per figure.
    """
    import os

    from .figures.plates import plate_figure_name
    from .plot import save_figure
    from .tabular import write_table

    os.makedirs(out_dir, exist_ok=True)
    written: Dict[str, str] = {}
    tables = {"hit_table": hit_table(result.wells),
              "hit_scores_wells": result.wells,
              "hit_plates": result.plates}
    if len(result.treatments):
        tables["hit_treatments"] = result.treatments
    for name, table in tables.items():
        path = os.path.join(str(out_dir), f"{name}.csv")
        write_table(table, path, canonicalise=False)
        written[name] = path
    for method in methods:
        figure, panel = hit_heatmap(result, method, target=target)
        if not panel.drawn:
            import matplotlib.pyplot as plt

            plt.close(figure)
            continue
        path = os.path.join(str(out_dir),
                            plate_figure_name(method, prefix="hit_heatmap"))
        written[f"hit_heatmap_{method}"] = save_figure(
            figure, path, close=True, announce_colours=False)
    return written

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

Image-based profiling
---------------------

The profiling helpers turn Measure's per-object tables into well and
treatment profiles and score them, following pycytominer and copairs so the
numbers agree with those tools on the same features. Each object table is
aggregated to a per-well median, a plate map adds the treatment annotations,
features are normalised plate by plate against the negative-control wells
(robust MAD by default), uninformative and redundant features are removed
(variance, frequency, correlation, missing values and outliers), and
replicate wells are collapsed into consensus profiles. Replicate
reproducibility is scored as mean average precision (mAP) with copairs'
permutation null and Benjamini-Hochberg correction: phenotypic activity
(replicates against controls) and, given a phenotype label, phenotypic
consistency (treatments that share the label). Percent replicating is
reported beside it. Profiles are written as CSV and Parquet with
``Metadata_`` columns, the consensus also as GCT, with an mAP plot and a
consensus-similarity heatmap. Measure runs the whole recipe at the end of a
run when ``profiling`` is on.
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
import re
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


_PROFILE_KEYS = ("plateID", "rowID", "columnID")
_PROFILE_TABLES = ("cell", "cytoplasm", "nucleus", "pathogen")
_PROFILE_AGGREGATIONS = ("median", "mean")
_PROFILE_NORMALIZATIONS = ("mad_robustize", "standardize", "robustize", "none")
_PROFILE_SELECTIONS = ("variance_threshold", "frequency_threshold",
                       "correlation_threshold", "drop_na_columns",
                       "drop_outliers")
_PROFILE_DEFAULT_SELECTIONS = ("variance_threshold", "frequency_threshold",
                               "correlation_threshold", "drop_na_columns",
                               "drop_outliers")
_PROFILE_CONSENSUS = ("median", "mean", "modz")
_PROFILE_SIMILARITIES = ("cosine", "correlation", "abs_cosine", "euclidean")
_PROFILE_NULL_SIZE = 10000
_PROFILE_DB_TABLES = ("profile_wells", "profile_consensus", "profile_map")
_SQLITE_COLUMN_LIMIT = 1999
_WELL_NAME_COLUMNS = ("wellID", "well", "Metadata_Well", "well_position")
_PLATE_NAME_COLUMNS = ("plateID", "Metadata_Plate")
_POSITION_FEATURE = re.compile(r"(?:centroid(?:_weighted)?|bbox)-\d+$")


class _ProfilingError(ValueError):
    """Raised when profiles cannot be built, with the reason and the way out."""


@dataclass
class _ProfilingResult:
    """Everything one profiling run produced.

    :param wells: one row per well: the aggregated per-object measurements
        with their annotations, before normalisation.
    :param normalized: the same wells after per-plate normalisation.
    :param selected: the normalised wells restricted to the kept features.
    :param consensus: one row per treatment, the consensus of its replicate
        wells over the kept features, with ``n_replicates``.
    :param features: every feature column the wells carry.
    :param kept: the feature columns feature selection kept.
    :param excluded: ``{operation: [feature, ...]}``, what each selection
        step removed, in the order the steps ran.
    :param group_columns: the annotation columns that name a treatment.
    :param metadata: every non-feature column of the well tables.
    :param activity: one row per non-control well: its average precision at
        retrieving its own replicates against the negative controls.
    :param activity_map: one row per treatment: mean average precision,
        permutation p value, Benjamini-Hochberg corrected p value and calls.
    :param consistency: per-treatment average precision at retrieving the
        treatments that share its phenotype label, or an empty frame.
    :param consistency_map: one row per phenotype label, or an empty frame.
    :param replicating: one row per treatment with two or more replicates:
        the median correlation between its replicates and whether it clears
        the null.
    :param percent_replicating: the share of those treatments, in percent,
        whose replicate correlation exceeds the null quantile.
    :param options: the settings the run used.
    :param notes: what the run decided on its own, for the log.
    """

    wells: pd.DataFrame
    normalized: pd.DataFrame
    selected: pd.DataFrame
    consensus: pd.DataFrame
    features: List[str]
    kept: List[str]
    excluded: Dict[str, List[str]]
    group_columns: List[str]
    metadata: List[str]
    activity: pd.DataFrame
    activity_map: pd.DataFrame
    consistency: pd.DataFrame = _field(default_factory=pd.DataFrame)
    consistency_map: pd.DataFrame = _field(default_factory=pd.DataFrame)
    replicating: pd.DataFrame = _field(default_factory=pd.DataFrame)
    percent_replicating: float = float("nan")
    options: Dict[str, Any] = _field(default_factory=dict)
    notes: List[str] = _field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        """A JSON-ready digest: counts, the headline metrics and the options.

        :returns: a plain dict.
        """
        significant = 0
        if len(self.activity_map) and "below_corrected_p" in self.activity_map:
            significant = int(self.activity_map["below_corrected_p"].sum())
        plates = sorted({str(v) for v in self.wells.get("plateID", [])})
        consistent = None
        if len(self.consistency_map):
            consistent = int(self.consistency_map["below_corrected_p"].sum())
        return {
            "plates": plates,
            "wells": int(len(self.wells)),
            "treatments": int(len(self.consensus)),
            "features": int(len(self.features)),
            "kept_features": int(len(self.kept)),
            "excluded": {op: len(names) for op, names in self.excluded.items()},
            "group_columns": list(self.group_columns),
            "mean_average_precision": (
                float(self.activity_map["mean_average_precision"].mean())
                if len(self.activity_map) else None),
            "phenotypically_active": significant,
            "treatments_scored": int(len(self.activity_map)),
            "phenotypically_consistent": consistent,
            "percent_replicating": (
                None if not np.isfinite(self.percent_replicating)
                else float(self.percent_replicating)),
            "options": dict(self.options),
            "notes": list(self.notes),
        }


def _profile_features(frame: pd.DataFrame, exclude: Sequence[str] = ()
                      ) -> List[str]:
    """The numeric measurement columns of ``frame``, in column order.

    A column is a feature when it is numeric, is not identity or provenance
    (:func:`spacr.schema.is_provenance_column`, the boundary every model path
    uses), is not a position in the image (an absolute centroid or bounding
    box coordinate, which says where an object was rather than what it
    looked like), does not start with ``Metadata_`` and is not in
    ``exclude``.

    :param frame: a per-object or per-well table.
    :param exclude: further columns to leave out.
    :returns: the feature column names.
    """
    from .schema import is_provenance_column

    skip = set(exclude)
    names = []
    for name in frame.columns:
        text = str(name)
        if text in skip or text.startswith("Metadata_"):
            continue
        if _POSITION_FEATURE.search(text):
            continue
        if not pd.api.types.is_numeric_dtype(frame[name]):
            continue
        if pd.api.types.is_bool_dtype(frame[name]):
            continue
        if is_provenance_column(text):
            continue
        names.append(text)
    return names


def _aggregate_profiles(frame: pd.DataFrame, features: Sequence[str], *,
                        strata: Sequence[str] = _PROFILE_KEYS,
                        operation: str = "median",
                        count_column: Optional[str] = None) -> pd.DataFrame:
    """Collapse per-object rows to one row per stratum, usually per well.

    Each feature is summarised separately over the objects of a stratum,
    skipping missing values, as pycytominer's ``aggregate`` does. A stratum
    with a missing key is kept as its own group rather than dropped.

    :param frame: per-object rows.
    :param features: the feature columns to summarise.
    :param strata: the columns that name a group.
    :param operation: ``'median'`` or ``'mean'``.
    :param count_column: when given, a column of this name holds the number
        of objects in each group.
    :returns: the strata, the optional count, then the features.
    :raises _ProfilingError: for an unknown operation or a missing stratum.
    """
    if operation not in _PROFILE_AGGREGATIONS:
        raise _ProfilingError(
            f"unknown aggregation {operation!r}; choose "
            f"{' or '.join(_PROFILE_AGGREGATIONS)}")
    missing = [name for name in strata if name not in frame.columns]
    if missing:
        raise _ProfilingError(
            f"the table has no {', '.join(missing)} column, so its objects "
            "cannot be grouped into wells")
    values = frame.loc[:, list(features)].astype(float)
    values = pd.concat([frame.loc[:, list(strata)], values], axis=1)
    grouped = values.groupby(list(strata), dropna=False, sort=True)
    out = (grouped.median() if operation == "median"
           else grouped.mean()).reset_index()
    if count_column:
        counts = grouped.size().reset_index(name=count_column)
        out = counts.merge(out, on=list(strata), how="right")
    return out


def _read_well_profiles(databases: Sequence[Any], *,
                        tables: Optional[Sequence[str]] = None,
                        operation: str = "median",
                        report=print) -> Tuple[pd.DataFrame, List[str]]:
    """Aggregate every object table of one or more measurement databases.

    Each object table (cell, cytoplasm, nucleus, pathogen, and any table
    named in ``tables``) is read through :func:`spacr.tabular.read_database`,
    reduced to one row per well, and the compartments are joined on
    plate, row and column. A feature that does not already start with its
    table's name is prefixed with it, so compartments never collide. The
    object count of each compartment is kept as ``n_<table>``, and wells are
    ordered by plate, row and column number (A2 before A10).

    :param databases: ``measurements.db`` paths, one per plate or more.
    :param tables: object tables to use; ``None`` uses the default four that
        each database has.
    :param operation: ``'median'`` or ``'mean'`` over the objects of a well.
    :param report: called with one progress line per table; ``None`` is quiet.
    :returns: ``(wells, features)``.
    :raises _ProfilingError: when no database holds an object table, or when
        two databases carry the same plate.
    """
    from .tabular import database_tables, read_database

    frames = []
    seen_plates: Dict[str, str] = {}
    for db in databases:
        present = set(database_tables(db))
        wanted = [t for t in (tables or _PROFILE_TABLES) if t in present]
        if tables and len(wanted) < len(tables):
            absent = sorted(set(tables) - present)
            raise _ProfilingError(
                f"{db} has no {', '.join(absent)} table; profiling needs the "
                "object tables Measure writes")
        per_db = None
        for table in wanted:
            (objects,) = read_database(db, [table], report=None,
                                       migrate=False, read_only=True)
            if objects.empty:
                continue
            features = _profile_features(objects)
            renamed = {name: (name if name.startswith(f"{table}_")
                              else f"{table}_{name}") for name in features}
            objects = objects.rename(columns=renamed)
            wells = _aggregate_profiles(
                objects, [renamed[name] for name in features],
                operation=operation, count_column=f"n_{table}")
            if report:
                report(f"Profiles: {db} {table}: {len(objects)} objects in "
                       f"{len(wells)} wells, {len(features)} features")
            per_db = wells if per_db is None else per_db.merge(
                wells, on=list(_PROFILE_KEYS), how="outer")
            del objects
        if per_db is None:
            continue
        for plate in per_db["plateID"].dropna().astype(str).unique():
            if plate in seen_plates and seen_plates[plate] != str(db):
                raise _ProfilingError(
                    f"plate {plate!r} is in both {seen_plates[plate]} and "
                    f"{db}; give the plates distinct plateID values before "
                    "profiling them together")
            seen_plates[plate] = str(db)
        frames.append(per_db)
    if not frames:
        raise _ProfilingError(
            "none of the databases holds a cell, cytoplasm, nucleus or "
            "pathogen table with rows; run Measure first")
    wells = pd.concat(frames, ignore_index=True, sort=False)
    where = _well_coordinates(wells)
    order = pd.DataFrame({"plate": wells["plateID"].astype(str),
                          "row": where["_row"], "col": where["_col"]})
    wells = wells.loc[order.sort_values(["plate", "row", "col"],
                                        kind="stable").index]
    wells = wells.reset_index(drop=True)
    counts = [c for c in wells.columns if c.startswith("n_")
              and c[2:] in set(tables or _PROFILE_TABLES)]
    features = [c for c in wells.columns
                if c not in _PROFILE_KEYS and c not in counts]
    wells = wells.loc[:, list(_PROFILE_KEYS) + counts + features]
    return wells, features


def _well_coordinates(frame: pd.DataFrame) -> pd.DataFrame:
    """Plate, 1-based row and 1-based column of every row of ``frame``.

    Rows and columns may be spelled any way :mod:`spacr.plate_qc` reads
    (``r3``/``C``/``3`` and ``c7``/``7``), or given as one well column
    (``wellID``, ``well``, ``Metadata_Well`` or ``well_position``) such as
    ``C07``. The plate comes from ``plateID`` or ``Metadata_Plate`` and is
    ``None`` when the frame has neither.

    :param frame: a well table or a plate map.
    :returns: a frame with ``_plate``, ``_row`` and ``_col``, aligned to
        ``frame``.
    :raises _ProfilingError: when nothing in the frame names a well.
    """
    from .plate_qc import _parse_well_label, parse_column_label, parse_row_label

    out = pd.DataFrame(index=frame.index)
    well = next((c for c in _WELL_NAME_COLUMNS if c in frame.columns), None)
    plate = next((c for c in _PLATE_NAME_COLUMNS if c in frame.columns), None)
    if "rowID" in frame.columns and "columnID" in frame.columns:
        out["_row"] = frame["rowID"].map(parse_row_label)
        out["_col"] = frame["columnID"].map(parse_column_label)
    elif well is not None:
        parsed = frame[well].map(_parse_well_label)
        out["_row"] = [p[0] if p else None for p in parsed]
        out["_col"] = [p[1] if p else None for p in parsed]
    else:
        raise _ProfilingError(
            "the plate map names no well: give it rowID and columnID columns, "
            "or a well column such as A01")
    out["_plate"] = frame[plate].astype(str) if plate else None
    return out


def _annotate_profiles(wells: pd.DataFrame, plate_map: pd.DataFrame, *,
                       report=print) -> Tuple[pd.DataFrame, List[str]]:
    """Join a plate map's annotations onto the well profiles.

    A plate map with a plate column is matched plate by plate; one without
    applies to every plate, the usual case when every plate has the same
    layout. Wells the map does not cover keep empty annotations.

    :param wells: one row per well, with plate, row and column.
    :param plate_map: one row per well position, with any annotation
        columns (treatment, dose, gene, phenotype label ...).
    :param report: called with a line naming unmatched wells; ``None`` is
        quiet.
    :returns: ``(annotated wells, annotation column names)``.
    :raises _ProfilingError: when the map names no well, or one well twice.
    """
    keys = ({"rowID", "columnID", "prc", "prcf", "fieldID"}
            | set(_WELL_NAME_COLUMNS) | set(_PLATE_NAME_COLUMNS))
    annotations = [c for c in plate_map.columns if c not in keys]
    where = _well_coordinates(plate_map)
    by_plate = bool(where["_plate"].notna().all()) and any(
        c in plate_map.columns for c in _PLATE_NAME_COLUMNS)
    join = ["_plate", "_row", "_col"] if by_plate else ["_row", "_col"]
    mapped = pd.concat([where[join], plate_map[annotations]], axis=1)
    duplicated = mapped.duplicated(subset=join, keep=False)
    if duplicated.any():
        raise _ProfilingError(
            f"the plate map lists {int(duplicated.sum())} well position(s) "
            "more than once; give each well one row")
    located = pd.concat([wells.drop(columns=[c for c in annotations
                                             if c in wells.columns]),
                         _well_coordinates(wells)], axis=1)
    out = located.merge(mapped, on=join, how="left")
    unmatched = int(out[annotations].isna().all(axis=1).sum()) if annotations else 0
    if unmatched and report:
        report(f"Profiles: the plate map does not cover {unmatched} of "
               f"{len(out)} wells; their annotations are empty")
    return out.drop(columns=["_plate", "_row", "_col"]), annotations


def _add_well_names(frame: pd.DataFrame) -> pd.DataFrame:
    """Add a ``wellID`` column (``C07``) from rowID and columnID.

    :param frame: a well table.
    :returns: a copy with ``wellID`` set; unparseable positions stay empty.
    """
    from .plate_qc import well_id

    where = _well_coordinates(frame)
    names = [well_id(int(r), int(c)) if pd.notna(r) and pd.notna(c) else None
             for r, c in zip(where["_row"], where["_col"])]
    out = frame.copy()
    out["wellID"] = names
    return out


def _normalize_profiles(frame: pd.DataFrame, features: Sequence[str], *,
                        method: str = "mad_robustize",
                        by: Optional[str] = "plateID",
                        reference: Optional[pd.Series] = None,
                        epsilon: float = 1e-18) -> pd.DataFrame:
    """Normalise features within each plate against its reference wells.

    The centre and scale are fitted on the reference wells of a plate
    (normally its negative controls; every well when ``reference`` is
    ``None``) and applied to all wells of that plate:

    * ``mad_robustize``: ``(x - median) / (1.4826 MAD + epsilon)``, as
      pycytominer's RobustMAD;
    * ``standardize``: ``(x - mean) / SD``, population SD, a zero SD left
      at 1, as scikit-learn's StandardScaler;
    * ``robustize``: ``(x - median) / IQR``, a zero IQR left at 1, as
      scikit-learn's RobustScaler;
    * ``none``: unchanged.

    Missing values are skipped when fitting and stay missing.

    :param frame: one row per well.
    :param features: the columns to normalise.
    :param method: one of the four above.
    :param by: the column that names a plate; ``None`` fits once over all
        rows.
    :param reference: boolean mask aligned to ``frame``, true for reference
        wells.
    :param epsilon: added to the MAD, so a constant feature does not divide
        by zero.
    :returns: a copy of ``frame`` with the features normalised.
    :raises _ProfilingError: for an unknown method, or a plate with no
        reference well.
    """
    if method not in _PROFILE_NORMALIZATIONS:
        raise _ProfilingError(
            f"unknown normalisation {method!r}; choose one of "
            f"{', '.join(_PROFILE_NORMALIZATIONS)}")
    out = frame.copy()
    features = list(features)
    if method == "none" or not features:
        return out
    values = frame.loc[:, features].to_numpy(dtype=float)
    result = values.copy()
    if reference is None:
        reference_mask = np.ones(len(frame), dtype=bool)
    else:
        reference_mask = np.asarray(
            pd.Series(reference).reindex(frame.index).fillna(False),
            dtype=bool)
    if by is None or by not in frame.columns:
        labels = np.zeros(len(frame), dtype=int)
        names = ["all"]
    else:
        codes, names = pd.factorize(frame[by].astype(str), sort=True)
        labels = codes
    tiny = 10 * np.finfo(float).eps
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", RuntimeWarning)
        for code, name in enumerate(names):
            rows = labels == code
            fit = values[rows & reference_mask]
            if not len(fit):
                raise _ProfilingError(
                    f"plate {name} has no reference well to normalise "
                    "against; check the negative control, or normalise "
                    "against every well")
            if method == "mad_robustize":
                centre = np.nanmedian(fit, axis=0)
                spread = np.nanmedian(np.abs(fit - centre), axis=0) * MAD_SCALE
                result[rows] = (values[rows] - centre) / (spread + epsilon)
            elif method == "standardize":
                centre = np.nanmean(fit, axis=0)
                spread = np.nanstd(fit, axis=0)
                spread = np.where(spread < tiny, 1.0, spread)
                result[rows] = (values[rows] - centre) / spread
            else:
                centre = np.nanmedian(fit, axis=0)
                q25, q75 = np.nanpercentile(fit, [25, 75], axis=0)
                spread = q75 - q25
                spread = np.where(spread < tiny, 1.0, spread)
                result[rows] = (values[rows] - centre) / spread
    out[features] = result
    return out


def _low_variance(values: pd.DataFrame, min_variance: float) -> List[str]:
    """Features whose variance is not above ``min_variance``.

    :param values: the rows the selection is fitted on, features only.
    :param min_variance: the variance a feature must exceed to be kept.
    :returns: the excluded names.
    """
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", RuntimeWarning)
        variance = np.nanvar(values.to_numpy(dtype=float), axis=0)
    keep = variance > float(min_variance)
    return [name for name, ok in zip(values.columns, keep) if not ok]


def _low_frequency(values: pd.DataFrame, freq_cut: float,
                   unique_cut: float) -> List[str]:
    """Near-constant features, by caret's ``nearZeroVar`` rule.

    A feature is excluded when its second most common value is rarer than
    ``freq_cut`` times its most common, or when its distinct values are
    fewer than ``unique_cut`` times the number of rows.

    :param values: the rows the selection is fitted on, features only.
    :param freq_cut: the second-to-first frequency ratio a feature must
        reach.
    :param unique_cut: the distinct-value fraction a feature must reach.
    :returns: the excluded names.
    """
    excluded = []
    rows = len(values)
    for name in values.columns:
        counts = values[name].value_counts()
        ratio = 0.0 if len(counts) < 2 else counts.iloc[1] / counts.iloc[0]
        unique = values[name].nunique() / rows if rows else 0.0
        if ratio < freq_cut or unique < unique_cut:
            excluded.append(name)
    return excluded


def _too_many_missing(values: pd.DataFrame, cutoff: float) -> List[str]:
    """Features missing in more than ``cutoff`` of the rows.

    :param values: the rows the selection is fitted on, features only.
    :param cutoff: the largest missing fraction a kept feature may have.
    :returns: the excluded names.
    """
    fraction = values.isna().mean()
    return fraction[fraction > cutoff].index.tolist()


def _feature_correlation(values: pd.DataFrame, method: str) -> pd.DataFrame:
    """Feature-by-feature correlation matrix.

    Pearson on complete data uses :func:`numpy.corrcoef`; missing values or
    another method use pandas' pairwise-complete correlation.

    :param values: rows by features.
    :param method: ``'pearson'``, ``'spearman'`` or ``'kendall'``.
    :returns: a square frame labelled by feature.
    """
    array = values.to_numpy(dtype=float)
    if method == "pearson" and np.isfinite(array).all():
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", RuntimeWarning)
            return pd.DataFrame(np.corrcoef(array.T), index=values.columns,
                                columns=values.columns)
    return values.corr(method=method)


def _redundant(values: pd.DataFrame, threshold: float,
               method: str = "pearson") -> List[str]:
    """Features correlated above ``threshold`` with another feature.

    Every pair whose correlation exceeds the threshold (signed, as
    pycytominer tests it) loses the member with the larger summed absolute
    correlation to all features, so the more redundant one goes.

    :param values: the rows the selection is fitted on, features only.
    :param threshold: the correlation above which a pair is redundant.
    :param method: the correlation, ``'pearson'`` by default.
    :returns: the excluded names.
    """
    if values.shape[1] < 2:
        return []
    corr = _feature_correlation(values, method)
    order = corr.abs().sum().sort_values(kind="stable").index
    rank = {name: position for position, name in enumerate(order)}
    matrix = corr.to_numpy()
    lower = np.tril(np.ones(matrix.shape, dtype=bool), k=-1)
    with np.errstate(invalid="ignore"):
        hits = np.argwhere(lower & (matrix > threshold))
    names = list(corr.index)
    excluded = set()
    for i, j in hits:
        a, b = names[i], names[j]
        excluded.add(a if rank[a] > rank[b] else b)
    return sorted(excluded, key=names.index)


def _extreme(values: pd.DataFrame, cutoff: float) -> List[str]:
    """Features with any absolute value above ``cutoff``.

    After normalisation to control units a feature this far out is almost
    always a division by a near-zero spread rather than biology.

    :param values: the rows the selection is fitted on, features only.
    :param cutoff: the largest absolute value a kept feature may reach.
    :returns: the excluded names.
    """
    high = values.max().abs()
    low = values.min().abs()
    return high[(high > cutoff) | (low > cutoff)].index.tolist()


def _select_profile_features(
        frame: pd.DataFrame, features: Sequence[str],
        operations: Sequence[str] = _PROFILE_DEFAULT_SELECTIONS, *,
        reference: Optional[pd.Series] = None,
        na_cutoff: float = 0.05, corr_threshold: float = 0.9,
        corr_method: str = "pearson", freq_cut: float = 0.05,
        unique_cut: float = 0.01, min_variance: float = 1e-6,
        outlier_cutoff: float = 500.0
) -> Tuple[List[str], Dict[str, List[str]]]:
    """Drop uninformative and redundant features, one step after another.

    The steps run in the order given, each on the features the earlier
    steps kept, and are fitted on the ``reference`` rows (every row when
    ``None``):

    * ``variance_threshold``: variance not above ``min_variance``;
    * ``frequency_threshold``: near-constant, by ``freq_cut`` and
      ``unique_cut``;
    * ``correlation_threshold``: the more redundant of every pair
      correlated above ``corr_threshold``;
    * ``drop_na_columns``: missing in more than ``na_cutoff`` of rows;
    * ``drop_outliers``: an absolute value above ``outlier_cutoff``.

    Each step and default matches pycytominer's ``feature_select``.

    :param frame: one row per well.
    :param features: the candidate feature columns.
    :param operations: the steps, from those above.
    :param reference: boolean mask of the rows to fit on.
    :param na_cutoff: see ``drop_na_columns``.
    :param corr_threshold: see ``correlation_threshold``.
    :param corr_method: ``'pearson'``, ``'spearman'`` or ``'kendall'``.
    :param freq_cut: see ``frequency_threshold``.
    :param unique_cut: see ``frequency_threshold``.
    :param min_variance: see ``variance_threshold``.
    :param outlier_cutoff: see ``drop_outliers``.
    :returns: ``(kept features, {step: excluded features})``.
    :raises _ProfilingError: for an unknown step.
    """
    unknown = [op for op in operations if op not in _PROFILE_SELECTIONS]
    if unknown:
        raise _ProfilingError(
            f"unknown feature selection {', '.join(map(repr, unknown))}; "
            f"choose from {', '.join(_PROFILE_SELECTIONS)}")
    rows = frame if reference is None else frame.loc[
        pd.Series(reference).reindex(frame.index).fillna(False).astype(bool)]
    kept = list(features)
    excluded: Dict[str, List[str]] = {}
    for op in operations:
        values = rows.loc[:, kept].astype(float)
        if op == "variance_threshold":
            dropped = _low_variance(values, min_variance)
        elif op == "frequency_threshold":
            dropped = _low_frequency(values, freq_cut, unique_cut)
        elif op == "correlation_threshold":
            dropped = _redundant(values, corr_threshold, corr_method)
        elif op == "drop_na_columns":
            dropped = _too_many_missing(values, na_cutoff)
        else:
            dropped = _extreme(values, outlier_cutoff)
        excluded[op] = list(dropped)
        gone = set(dropped)
        kept = [name for name in kept if name not in gone]
    return kept, excluded


def _modz(values: pd.DataFrame, *, min_weight: float = 0.01,
          precision: int = 4) -> pd.Series:
    """The moderated z-score consensus of replicate profiles.

    Each replicate is weighted by its mean Spearman correlation with the
    other replicates (negative correlations count as zero, and no weight
    falls below ``min_weight``), the weights are made to sum to one and
    rounded, and the weighted profiles are summed, as pycytominer's MODZ.

    :param values: replicates by features.
    :param min_weight: the smallest weight a replicate can have.
    :param precision: decimals the weights are rounded to.
    :returns: one consensus value per feature.
    """
    if len(values) == 1:
        return values.iloc[0].astype(float)
    transposed = values.transpose()
    corr = transposed.corr(method="spearman").to_numpy(copy=True)
    np.fill_diagonal(corr, np.nan)
    corr = np.clip(corr, 0, None)
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", RuntimeWarning)
        raw = np.nanmean(corr, axis=1)
    raw = np.where(np.isnan(raw), raw, np.maximum(raw, min_weight))
    total = np.nansum(raw)
    weights = (np.full(len(raw), 1.0 / len(raw)) if total == 0
               else raw / total)
    weights = np.round(weights, precision)
    return transposed.mul(weights, axis="columns").sum(axis="columns")


def _consensus_profiles(frame: pd.DataFrame, features: Sequence[str],
                        by: Sequence[str], *,
                        operation: str = "median") -> pd.DataFrame:
    """One profile per treatment from its replicate wells.

    :param frame: one row per well.
    :param features: the feature columns.
    :param by: the annotation columns that name a treatment.
    :param operation: ``'median'``, ``'mean'`` or ``'modz'``.
    :returns: the ``by`` columns, ``n_replicates`` and the features.
    :raises _ProfilingError: for an unknown operation.
    """
    if operation not in _PROFILE_CONSENSUS:
        raise _ProfilingError(
            f"unknown consensus {operation!r}; choose one of "
            f"{', '.join(_PROFILE_CONSENSUS)}")
    by = list(by)
    if operation in ("median", "mean"):
        return _aggregate_profiles(frame, features, strata=by,
                                   operation=operation,
                                   count_column="n_replicates")
    rows = []
    for key, group in frame.groupby(by, dropna=False, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        profile = _modz(group.loc[:, list(features)].astype(float))
        rows.append(dict(zip(by, key), n_replicates=len(group),
                         **profile.to_dict()))
    return pd.DataFrame(rows, columns=by + ["n_replicates"] + list(features))


def _similarity_block(block: np.ndarray, all_rows: np.ndarray,
                      similarity: str) -> np.ndarray:
    """Similarity of each row of ``block`` to each row of ``all_rows``.

    :param block: query rows, already prepared by :func:`_prepare_rows`.
    :param all_rows: every row, prepared the same way.
    :param similarity: one of ``cosine``, ``correlation``, ``abs_cosine``
        or ``euclidean`` (``1 / (1 + distance)``).
    :returns: ``len(block)`` by ``len(all_rows)``.
    """
    if similarity == "euclidean":
        sq = ((block ** 2).sum(axis=1)[:, None]
              + (all_rows ** 2).sum(axis=1)[None, :]
              - 2.0 * block @ all_rows.T)
        return 1.0 / (1.0 + np.sqrt(np.clip(sq, 0.0, None)))
    sims = block @ all_rows.T
    return np.abs(sims) if similarity == "abs_cosine" else sims


def _prepare_rows(values: np.ndarray, similarity: str) -> np.ndarray:
    """Centre and scale rows so a dot product is the chosen similarity.

    :param values: rows by features.
    :param similarity: see :func:`_similarity_block`.
    :returns: the prepared rows.
    """
    rows = np.asarray(values, dtype=float)
    if similarity == "euclidean":
        return rows
    if similarity == "correlation":
        rows = rows - rows.mean(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def _metadata_codes(meta: pd.DataFrame, columns: Sequence[str]
                    ) -> Dict[str, np.ndarray]:
    """Integer codes per column, ``-1`` where the value is missing.

    :param meta: the metadata rows.
    :param columns: the columns to encode.
    :returns: ``{column: codes}``.
    """
    return {c: pd.factorize(meta[c])[0] for c in columns}


def _pair_mask(codes: Dict[str, np.ndarray], i: int, same: Sequence[str],
               diff: Sequence[str]) -> np.ndarray:
    """Rows that share every ``same`` column with row ``i`` and differ in
    every ``diff`` column.

    A missing value neither matches nor differs, as in SQL.

    :param codes: from :func:`_metadata_codes`.
    :param i: the query row.
    :param same: columns that must be equal.
    :param diff: columns that must differ.
    :returns: a boolean mask; row ``i`` itself is always false.
    """
    n = len(next(iter(codes.values()))) if codes else 0
    mask = np.ones(n, dtype=bool)
    for column in same:
        c = codes[column]
        mask &= (c == c[i]) & (c >= 0) & (c[i] >= 0)
    for column in diff:
        c = codes[column]
        mask &= (c != c[i]) & (c >= 0) & (c[i] >= 0)
    mask[i] = False
    return mask


def _expected_ap(positives: int, negatives: int) -> float:
    """Expected average precision of a random ranking.

    :param positives: the number of positives in the ranked list.
    :param negatives: the number of negatives.
    :returns: the expectation, used to normalise an average precision.
    """
    total = positives + negatives
    if total <= 1:
        return 1.0 if positives == 1 else 0.0
    if positives == 0:
        return 0.0
    if positives == total:
        return 1.0
    harmonic = float(np.sum(1.0 / np.arange(1, total + 1)))
    return (1.0 / total) * ((positives - 1.0) / (total - 1.0)
                            * (total - harmonic) + harmonic)


def _profile_average_precision(
        meta: pd.DataFrame, feats: Any, pos_sameby: Sequence[str],
        pos_diffby: Sequence[str] = (), neg_sameby: Sequence[str] = (),
        neg_diffby: Sequence[str] = (), *, similarity: str = "cosine",
        block_size: int = 512) -> pd.DataFrame:
    """Average precision of every profile at retrieving its positives.

    Each profile is a query. Its positives are the profiles that share all
    ``pos_sameby`` columns and differ in all ``pos_diffby`` columns; its
    negatives share all ``neg_sameby`` and differ in all ``neg_diffby``.
    Positives and negatives are ranked together by similarity to the query
    and the average precision of the positives in that ranking is its
    score, as in copairs. Similarities are ranked at single precision, as
    copairs ranks them, and a tie ranks the positive first.

    :param meta: one metadata row per profile.
    :param feats: profiles by features, aligned to ``meta``.
    :param pos_sameby: columns a positive shares with the query.
    :param pos_diffby: columns a positive differs in.
    :param neg_sameby: columns a negative shares with the query.
    :param neg_diffby: columns a negative differs in.
    :param similarity: see :func:`_similarity_block`.
    :param block_size: queries per matrix product, bounding memory.
    :returns: ``meta`` with ``average_precision``, ``n_pos_pairs``,
        ``n_total_pairs`` and ``normalized_average_precision``; a query
        without positives gets no score.
    :raises _ProfilingError: for an unknown similarity, or when no query
        has a positive or none has a negative.
    """
    if similarity not in _PROFILE_SIMILARITIES:
        raise _ProfilingError(
            f"unknown similarity {similarity!r}; choose one of "
            f"{', '.join(_PROFILE_SIMILARITIES)}")
    meta = meta.reset_index(drop=True).copy()
    values = np.asarray(feats, dtype=float)
    columns = list(dict.fromkeys(list(pos_sameby) + list(pos_diffby)
                                 + list(neg_sameby) + list(neg_diffby)))
    codes = _metadata_codes(meta, columns)
    prepared = _prepare_rows(values, similarity)
    n = len(meta)
    ap = np.full(n, np.nan)
    n_pos = np.zeros(n, dtype=int)
    n_total = np.zeros(n, dtype=int)
    any_pos = any_neg = False
    for start in range(0, n, max(1, int(block_size))):
        stop = min(n, start + max(1, int(block_size)))
        sims = _similarity_block(prepared[start:stop], prepared,
                                 similarity).astype(np.float32)
        for offset, i in enumerate(range(start, stop)):
            pos = np.flatnonzero(_pair_mask(codes, i, pos_sameby, pos_diffby))
            neg = np.flatnonzero(_pair_mask(codes, i, neg_sameby, neg_diffby))
            any_pos |= bool(pos.size)
            any_neg |= bool(neg.size)
            n_pos[i] = pos.size
            n_total[i] = pos.size + neg.size
            if not pos.size:
                continue
            ranked = np.concatenate([sims[offset, pos], sims[offset, neg]])
            labels = np.concatenate([np.ones(pos.size), np.zeros(neg.size)])
            order = np.argsort(-ranked, kind="stable")
            hits = labels[order]
            precision = np.cumsum(hits) / np.arange(1, hits.size + 1)
            ap[i] = float((precision * hits).sum() / pos.size)
    if not any_pos:
        raise _ProfilingError(
            "no profile has a replicate to retrieve; average precision needs "
            "at least two wells of some treatment")
    if not any_neg:
        raise _ProfilingError(
            "no profile has a negative to rank against; check the negative "
            "control")
    meta["average_precision"] = ap
    meta["n_pos_pairs"] = n_pos
    meta["n_total_pairs"] = n_total
    normalised = np.full(n, np.nan)
    for i in np.flatnonzero(np.isfinite(ap)):
        mu = _expected_ap(int(n_pos[i]), int(n_total[i] - n_pos[i]))
        normalised[i] = np.clip((ap[i] - mu) / max(1.0 - mu, 1e-10), -1, 1)
    meta["normalized_average_precision"] = normalised
    return meta


def _null_average_precision(null_size: int, positives: int, total: int,
                            seed: int, *, chunk_cells: int = 4_000_000
                            ) -> np.ndarray:
    """Average precisions of random rankings, the null for one list shape.

    Each draw scatters ``positives`` positives at random among ``total``
    ranks and scores the result. The draws use the same generator, seed and
    order as copairs, so the null, and every p value computed from it, is
    the one copairs computes. Draws are made in chunks to bound memory
    without changing them.

    :param null_size: the number of random rankings.
    :param positives: positives in the list.
    :param total: the list length.
    :param seed: the generator seed for this list shape.
    :param chunk_cells: largest ``rows x total`` array made at once.
    :returns: ``null_size`` average precisions, single precision.
    """
    rng = np.random.default_rng(int(seed))
    dtype = np.uint16 if total < 2 ** 16 else np.uint32
    step = max(1, int(chunk_cells) // max(1, int(total)))
    ranks = np.arange(1, positives + 1, dtype=np.float32)
    out = np.empty(int(null_size), dtype=np.float32)
    for start in range(0, int(null_size), step):
        rows = min(step, int(null_size) - start)
        order = np.tile(np.arange(total, dtype=dtype), (rows, 1))
        rng.permuted(order, axis=1, out=order)
        placed = np.sort(order[:, :positives], axis=1)
        precision = ranks / (placed + 1)
        out[start:start + rows] = (precision.sum(axis=1)
                                   / positives).astype(np.float32)
    return out


def _profile_map(ap: pd.DataFrame, sameby: Sequence[str], *,
                 null_size: int = _PROFILE_NULL_SIZE,
                 threshold: float = 0.05, seed: int = 0) -> pd.DataFrame:
    """Mean average precision per group, with a permutation p value.

    The queries of a group are averaged into its mAP. Its null is the
    element-wise mean of its queries' random-ranking nulls
    (:func:`_null_average_precision`, one per list shape), the p value is
    ``(1 + draws above the mAP) / (1 + null_size)``, and p values are
    corrected across groups by Benjamini-Hochberg, as copairs'
    ``mean_average_precision`` does.

    :param ap: the output of :func:`_profile_average_precision`.
    :param sameby: the columns that name a group.
    :param null_size: random rankings per list shape; the smallest possible
        p value is ``1 / (1 + null_size)``.
    :param threshold: the level the calls are made at.
    :param seed: the seed the per-shape seeds are drawn from.
    :returns: one row per group: the ``sameby`` columns,
        ``mean_average_precision``, ``mean_normalized_average_precision``,
        ``n_queries``, ``p_value``, ``corrected_p_value``, ``below_p`` and
        ``below_corrected_p``.
    """
    from .multiple_testing import adjust_p_values

    sameby = list(sameby)
    scored = ap[ap["average_precision"].notna()
                & (ap["n_pos_pairs"] > 0)].reset_index(drop=True)
    columns = sameby + ["mean_average_precision",
                        "mean_normalized_average_precision", "n_queries",
                        "p_value", "corrected_p_value", "below_p",
                        "below_corrected_p"]
    if scored.empty:
        return pd.DataFrame(columns=columns)
    shapes = scored[["n_pos_pairs", "n_total_pairs"]].to_numpy(dtype=np.int64)
    unique, inverse = np.unique(shapes, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).ravel()
    seeds = np.random.default_rng(seed).integers(8096, size=len(unique))
    nulls = np.empty((len(unique), int(null_size)), dtype=np.float32)
    for index, (positives, total) in enumerate(unique):
        nulls[index] = _null_average_precision(
            null_size, int(positives), int(total), int(seeds[index]))
    rows = []
    for key, group in scored.groupby(sameby, observed=True, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        score = float(group["average_precision"].mean())
        null = nulls[inverse[group.index.to_numpy()]].mean(axis=0)
        p = (float((null > score).sum()) + 1.0) / (int(null_size) + 1.0)
        rows.append(dict(
            zip(sameby, key), mean_average_precision=score,
            mean_normalized_average_precision=float(
                group["normalized_average_precision"].mean()),
            n_queries=int(len(group)), p_value=p))
    out = pd.DataFrame(rows)
    corrected, _ = adjust_p_values(out["p_value"].to_numpy(), "fdr_bh",
                                   alpha=threshold)
    out["corrected_p_value"] = corrected
    out["below_p"] = out["p_value"] < threshold
    out["below_corrected_p"] = out["corrected_p_value"] < threshold
    return out.loc[:, columns]


def _phenotypic_activity(frame: pd.DataFrame, features: Sequence[str],
                         group_columns: Sequence[str], negative: pd.Series, *,
                         similarity: str = "cosine",
                         null_size: int = _PROFILE_NULL_SIZE,
                         threshold: float = 0.05, seed: int = 0
                         ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """How well each treatment's replicates find each other among controls.

    Every non-control well ranks its replicates (same treatment) against
    the negative-control wells; controls are not queries and are never
    positives of each other. A treatment whose mAP beats the null is
    phenotypically active: its replicates agree and differ from control.
    This is copairs' phenotypic-activity recipe.

    :param frame: one row per well, normalised and feature-selected.
    :param features: the feature columns.
    :param group_columns: the annotation columns that name a treatment.
    :param negative: boolean mask, true for negative-control wells.
    :param similarity: see :func:`_similarity_block`.
    :param null_size: see :func:`_profile_map`.
    :param threshold: see :func:`_profile_map`.
    :param seed: see :func:`_profile_map`.
    :returns: ``(per-well average precision, per-treatment mAP)``.
    """
    group_columns = list(group_columns)
    meta = frame.loc[:, [c for c in dict.fromkeys(
        group_columns + ["plateID", "wellID"]) if c in frame.columns]].copy()
    is_negative = np.asarray(
        pd.Series(negative).reindex(frame.index).fillna(False), dtype=bool)
    reference = np.where(is_negative, np.arange(len(frame)), -1)
    meta["_reference"] = reference
    ap = _profile_average_precision(
        meta, frame.loc[:, list(features)], group_columns + ["_reference"],
        (), (), group_columns + ["_reference"], similarity=similarity)
    ap = ap.loc[~is_negative].drop(columns=["_reference"])
    mapped = _profile_map(ap, group_columns, null_size=null_size,
                          threshold=threshold, seed=seed)
    return ap.reset_index(drop=True), mapped


def _phenotypic_consistency(consensus: pd.DataFrame, features: Sequence[str],
                            label: str, *, similarity: str = "cosine",
                            null_size: int = _PROFILE_NULL_SIZE,
                            threshold: float = 0.05, seed: int = 0
                            ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """How well treatments that share a phenotype label find each other.

    Each consensus profile ranks the other treatments with its label
    against the treatments with other labels, and a label whose mAP beats
    the null is phenotypically consistent: copairs' consistency recipe, as
    used to retrieve compounds of one mechanism or genes of one pathway.

    :param consensus: one row per treatment.
    :param features: the feature columns.
    :param label: the annotation column holding the phenotype label.
    :param similarity: see :func:`_similarity_block`.
    :param null_size: see :func:`_profile_map`.
    :param threshold: see :func:`_profile_map`.
    :param seed: see :func:`_profile_map`.
    :returns: ``(per-treatment average precision, per-label mAP)``.
    """
    labelled = consensus.loc[consensus[label].notna()].reset_index(drop=True)
    ap = _profile_average_precision(
        labelled.drop(columns=list(features)), labelled.loc[:, list(features)],
        [label], (), (), [label], similarity=similarity)
    return ap, _profile_map(ap, [label], null_size=null_size,
                            threshold=threshold, seed=seed)


def _percent_replicating(frame: pd.DataFrame, features: Sequence[str],
                         group_columns: Sequence[str], *,
                         exclude: Optional[pd.Series] = None,
                         n_null: int = 1000, quantile: float = 0.95,
                         seed: int = 0) -> Tuple[pd.DataFrame, float]:
    """Percent replicating: treatments whose replicates correlate above chance.

    For each treatment with two or more wells, the median Pearson
    correlation between its replicate wells is compared with the
    ``quantile`` of a null built as copairs' ``correlation_test`` builds
    it: each of ``n_null`` draws is the median correlation of ``k`` random
    pairs of wells from different treatments, where ``k`` is one less than
    the median replicate count (at most 50).

    :param frame: one row per well.
    :param features: the feature columns.
    :param group_columns: the annotation columns that name a treatment.
    :param exclude: boolean mask of wells to leave out, normally the
        negative controls.
    :param n_null: the number of null draws.
    :param quantile: the null quantile a treatment must exceed.
    :param seed: the random seed.
    :returns: ``(per-treatment table, percent replicating)``; the percent
        is NaN when no treatment has replicates.
    """
    keep = np.ones(len(frame), dtype=bool)
    if exclude is not None:
        keep &= ~np.asarray(pd.Series(exclude).reindex(frame.index)
                            .fillna(False), dtype=bool)
    wells = frame.loc[keep].reset_index(drop=True)
    columns = list(group_columns) + ["n_replicates",
                                     "median_replicate_correlation",
                                     "above_null"]
    if wells.empty:
        return pd.DataFrame(columns=columns), float("nan")
    prepared = _prepare_rows(wells.loc[:, list(features)].to_numpy(float),
                             "correlation")
    corr = prepared @ prepared.T
    group_code = pd.factorize(pd.MultiIndex.from_frame(
        wells.loc[:, list(group_columns)].astype(str)))[0]
    rows = []
    sizes = []
    for key, members in wells.groupby(list(group_columns), sort=True).groups.items():
        members = np.asarray(list(members))
        if members.size < 2:
            continue
        upper = np.triu_indices(members.size, k=1)
        rows.append((key if isinstance(key, tuple) else (key,), members.size,
                     float(np.nanmedian(corr[np.ix_(members, members)][upper]))))
        sizes.append(members.size)
    if not rows:
        return pd.DataFrame(columns=columns), float("nan")
    k = int(min(max(int(np.median(sizes)) - 1, 1), 50))
    rng = np.random.default_rng(seed)
    null = np.array([])
    if len(np.unique(group_code)) > 1:
        first = rng.integers(len(wells), size=int(n_null) * k)
        second = rng.integers(len(wells), size=first.size)
        clash = group_code[first] == group_code[second]
        while clash.any():
            second[clash] = rng.integers(len(wells), size=int(clash.sum()))
            clash = group_code[first] == group_code[second]
        null = np.nanmedian(corr[first, second].reshape(int(n_null), k),
                            axis=1)
    cut = (float(np.nanquantile(null, quantile)) if null.size
           else float("nan"))
    table = pd.DataFrame(
        [dict(zip(group_columns, key), n_replicates=n,
              median_replicate_correlation=value,
              above_null=bool(np.isfinite(cut) and value > cut))
         for key, n, value in rows], columns=columns)
    percent = (100.0 * float(table["above_null"].mean())
               if np.isfinite(cut) else float("nan"))
    return table, percent


def _external_profiles(frame: pd.DataFrame, metadata: Sequence[str]
                       ) -> pd.DataFrame:
    """Rename metadata the way pycytominer and copairs expect it.

    Every non-feature column gets the ``Metadata_`` prefix; plateID and
    wellID become ``Metadata_Plate`` and ``Metadata_Well``. Feature names are left
    as they are, so ``features=`` lists written for spaCR still apply.

    :param frame: a well or consensus table.
    :param metadata: its non-feature columns.
    :returns: a renamed copy.
    """
    special = {"plateID": "Metadata_Plate", "wellID": "Metadata_Well"}
    mapping = {}
    for name in metadata:
        if name not in frame.columns or str(name).startswith("Metadata_"):
            continue
        mapping[name] = special.get(name, f"Metadata_{name}")
    return frame.rename(columns=mapping)


def _write_gct(frame: pd.DataFrame, features: Sequence[str],
               metadata: Sequence[str], path: str) -> str:
    """Write profiles as a GCT 1.3 file, the format Morpheus opens.

    Features are rows and profiles are columns, with the metadata as column
    annotations.

    :param frame: one profile per row.
    :param features: the feature columns.
    :param metadata: the annotation columns.
    :param path: the destination.
    :returns: the path written.
    """
    import csv
    import os

    metadata = [m for m in metadata if m in frame.columns]
    ids = [f"profile_{i + 1}" for i in range(len(frame))]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["#1.3"])
        writer.writerow([len(features), len(frame), 0, len(metadata)])
        writer.writerow(["id"] + ids)
        for name in metadata:
            writer.writerow([name] + ["" if pd.isna(v) else str(v)
                                      for v in frame[name]])
        values = frame.loc[:, list(features)].to_numpy(dtype=float)
        for column, name in enumerate(features):
            writer.writerow([name] + ["" if not np.isfinite(v) else repr(float(v))
                                      for v in values[:, column]])
    return path


def _map_figure(mapped: pd.DataFrame, label: str, threshold: float):
    """mAP against significance, one point per group, calls coloured.

    :param mapped: the output of :func:`_profile_map`.
    :param label: what a point is, for the title.
    :param threshold: the corrected p value line.
    :returns: the matplotlib figure.
    """
    import matplotlib.pyplot as plt

    from .figures.style import figure_style, theme_target

    with figure_style(theme_target()):
        figure, axis = plt.subplots(figsize=(4.2, 3.6))
        p = mapped["corrected_p_value"].astype(float).clip(lower=1e-300)
        y = -np.log10(p)
        called = mapped["below_corrected_p"].astype(bool).to_numpy()
        axis.scatter(mapped.loc[~called, "mean_average_precision"],
                     y[~called], s=14, color="0.6", label="not significant")
        axis.scatter(mapped.loc[called, "mean_average_precision"],
                     y[called], s=14, color="#b5471b", label="significant")
        axis.axhline(-np.log10(threshold), color="0.4", lw=0.8, ls="--")
        axis.set_xlim(-0.02, 1.02)
        axis.set_xlabel("mean average precision")
        axis.set_ylabel("-log10 corrected p")
        axis.set_title(f"{label}\n{int(called.sum())} of {len(called)} "
                       "significant", fontsize=9)
        axis.legend(frameon=False, fontsize=7, loc="upper left")
        figure.tight_layout()
    return figure


def _similarity_figure(consensus: pd.DataFrame, features: Sequence[str],
                       labels: Sequence[str]):
    """Cosine similarity between consensus profiles, as a heatmap.

    Treatments are ordered by average-linkage clustering, so similar
    profiles sit together; names are shown when there are 60 or fewer.

    :param consensus: one row per treatment.
    :param features: the feature columns.
    :param labels: one name per row.
    :returns: the matplotlib figure.
    """
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import leaves_list, linkage

    from .figures.style import figure_style, theme_target

    prepared = _prepare_rows(consensus.loc[:, list(features)].to_numpy(float),
                             "cosine")
    prepared = np.nan_to_num(prepared)
    sims = np.clip(prepared @ prepared.T, -1, 1)
    order = np.arange(len(sims))
    if len(sims) > 2 and np.linalg.norm(prepared, axis=1).min() > 0:
        order = leaves_list(linkage(prepared, method="average",
                                    metric="cosine"))
    sims = sims[np.ix_(order, order)]
    with figure_style(theme_target()):
        side = min(9.0, 3.0 + 0.09 * len(sims))
        figure, axis = plt.subplots(figsize=(side + 1.0, side))
        image = axis.imshow(sims, cmap="RdBu_r", vmin=-1, vmax=1,
                            interpolation="nearest")
        if len(sims) <= 60:
            names = [str(labels[i]) for i in order]
            axis.set_xticks(range(len(sims)), names, rotation=90, fontsize=6)
            axis.set_yticks(range(len(sims)), names, fontsize=6)
        else:
            axis.set_xticks([])
            axis.set_yticks([])
        axis.set_title("consensus profile similarity (cosine)")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        figure.tight_layout()
    return figure


def _group_labels(frame: pd.DataFrame, group_columns: Sequence[str]
                  ) -> List[str]:
    """One printable name per row from its treatment columns.

    :param frame: a consensus table.
    :param group_columns: the treatment columns.
    :returns: the names, the column values joined by ``' | '``.
    """
    return [" | ".join(str(v) for v in row)
            for row in frame.loc[:, list(group_columns)].itertuples(
                index=False, name=None)]


def _write_profiles(result: _ProfilingResult, out_dir: str, *,
                    db_path: Optional[str] = None,
                    figures: bool = True) -> Dict[str, str]:
    """Write a profiling run: tables other tools read, figures and a summary.

    Well, normalised, feature-selected and consensus profiles are written
    as CSV and Parquet with pycytominer-style ``Metadata_`` columns (so
    pycytominer, copairs and pandas read them as they are), the consensus
    also as GCT for Morpheus, the average-precision and mAP tables and
    percent replicating as CSV, and ``profiling_summary.json``. Tables are
    written through :func:`spacr.tabular.write_table` and figures through
    :func:`spacr.plot.save_figure`; the similarity map's pale middle is
    zero similarity and is meant to recede, so its low-contrast colour
    warning is off. With ``db_path``, the selected well
    profiles, the consensus and the mAP table also go into that database as
    ``profile_wells``, ``profile_consensus`` and ``profile_map``, replacing
    earlier ones; a table wider than SQLite allows is left out of it.

    :param result: the run.
    :param out_dir: the folder to write into; created if absent.
    :param db_path: a ``measurements.db`` to add the tables to, or None.
    :param figures: draw the mAP and similarity figures.
    :returns: ``{name: path}`` of everything written.
    """
    import json
    import os

    from .plot import save_figure
    from .tabular import write_database, write_table

    os.makedirs(out_dir, exist_ok=True)
    written: Dict[str, str] = {}
    meta = [c for c in result.metadata if c in result.wells.columns]
    profiles = {
        "well_profiles": (result.wells, result.features, meta),
        "well_profiles_normalized": (result.normalized, result.features, meta),
        "well_profiles_feature_select": (result.selected, result.kept, meta),
        "consensus_profiles": (
            result.consensus, result.kept,
            list(result.group_columns) + ["n_replicates"]),
    }
    for name, (frame, features, columns) in profiles.items():
        external = _external_profiles(frame, columns)
        for suffix in (".csv", ".parquet"):
            path = os.path.join(out_dir, name + suffix)
            try:
                written[name + suffix] = write_table(
                    external, path, canonicalise=False)
            except ImportError:
                continue
    written["consensus_profiles.gct"] = _write_gct(
        result.consensus, result.kept,
        list(result.group_columns) + ["n_replicates"],
        os.path.join(out_dir, "consensus_profiles.gct"))
    tables = {"average_precision_activity": result.activity,
              "map_activity": result.activity_map,
              "average_precision_consistency": result.consistency,
              "map_consistency": result.consistency_map,
              "percent_replicating": result.replicating}
    for name, table in tables.items():
        if table is None or not len(table.columns):
            continue
        written[name] = write_table(table, os.path.join(out_dir, name + ".csv"),
                                    canonicalise=False)
    path = os.path.join(out_dir, "profiling_summary.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(result.summary(), handle, indent=2, default=str)
    written["summary"] = path
    if figures and len(result.activity_map):
        figure = _map_figure(result.activity_map, "phenotypic activity",
                             float(result.options.get("threshold", 0.05)))
        written["map_activity_figure"] = save_figure(
            figure, os.path.join(out_dir, "map_activity.pdf"), close=True)
    if figures and len(result.consistency_map):
        figure = _map_figure(result.consistency_map, "phenotypic consistency",
                             float(result.options.get("threshold", 0.05)))
        written["map_consistency_figure"] = save_figure(
            figure, os.path.join(out_dir, "map_consistency.pdf"), close=True)
    if figures and len(result.consensus) >= 2 and result.kept:
        figure = _similarity_figure(
            result.consensus, result.kept,
            _group_labels(result.consensus, result.group_columns))
        written["similarity_figure"] = save_figure(
            figure, os.path.join(out_dir, "consensus_similarity.pdf"),
            close=True, announce_colours=False)
    if db_path:
        for table, frame in zip(_PROFILE_DB_TABLES, (
                result.selected.loc[:, meta + result.kept],
                result.consensus, result.activity_map)):
            if frame.shape[1] > _SQLITE_COLUMN_LIMIT or not len(frame.columns):
                continue
            write_database(frame, db_path, table, if_exists="replace")
            written[f"db:{table}"] = f"{db_path}:{table}"
    return written


def _as_list(value: Any) -> List[str]:
    """A setting that may be blank, one value or a list, as a list of text.

    :param value: the setting.
    :returns: the non-empty entries as strings.
    """
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            import ast

            try:
                return _as_list(ast.literal_eval(text))
            except (ValueError, SyntaxError):
                pass
        return [part.strip() for part in text.split(",") if part.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value)]


def _profile_wells(wells: pd.DataFrame, features: Sequence[str], *,
                   group_columns: Sequence[str],
                   negative_controls: Sequence[str] = (),
                   normalization: str = "mad_robustize",
                   feature_selection: Sequence[str] = _PROFILE_DEFAULT_SELECTIONS,
                   correlation_threshold: float = 0.9,
                   consensus: str = "median",
                   phenotype_column: Optional[str] = None,
                   similarity: str = "cosine",
                   null_size: int = _PROFILE_NULL_SIZE,
                   threshold: float = 0.05, seed: int = 0,
                   report=print) -> _ProfilingResult:
    """Normalise, select, build consensus profiles and score them.

    The whole profiling recipe on well-level profiles: per-plate
    normalisation against the negative-control wells (every well when no
    control is named), feature selection fitted on all wells, consensus
    profiles per treatment, phenotypic activity mAP of every treatment
    against the controls, phenotypic consistency mAP when a phenotype label
    is given, and percent replicating.

    :param wells: one row per well, with plateID and the annotation columns.
    :param features: its feature columns.
    :param group_columns: the annotation columns that name a treatment;
        wells that share them are replicates.
    :param negative_controls: values of the first group column that mark a
        negative-control well.
    :param normalization: see :func:`_normalize_profiles`.
    :param feature_selection: see :func:`_select_profile_features`.
    :param correlation_threshold: see :func:`_select_profile_features`.
    :param consensus: see :func:`_consensus_profiles`.
    :param phenotype_column: annotation column with a phenotype label, or
        None to skip the consistency score.
    :param similarity: see :func:`_similarity_block`.
    :param null_size: see :func:`_profile_map`.
    :param threshold: the level mAP calls are made at.
    :param seed: the random seed for the nulls.
    :param report: called with progress lines; ``None`` is quiet.
    :returns: the :class:`_ProfilingResult`.
    :raises _ProfilingError: when a group column is missing, a named
        control matches no well, or too few features survive.
    """
    group_columns = list(group_columns)
    missing = [c for c in group_columns if c not in wells.columns]
    if missing or not group_columns:
        raise _ProfilingError(
            f"no {', '.join(missing) or 'treatment'} column among the well "
            "annotations; name the column that says which treatment each "
            "well received")
    notes: List[str] = []
    wells = _add_well_names(wells.reset_index(drop=True))
    controls = [str(v) for v in negative_controls]
    negative = wells[group_columns[0]].astype(str).isin(controls)
    if controls and not negative.any():
        present = sorted(wells[group_columns[0]].dropna().astype(str).unique())
        raise _ProfilingError(
            f"no well has {group_columns[0]} {' or '.join(controls)}; the "
            f"column holds {', '.join(present[:12])}"
            + (" and more" if len(present) > 12 else ""))
    reference = negative if controls else None
    if not controls:
        notes.append("No negative control named: each plate is normalised "
                     "against all of its wells and activity is not scored.")
    normalized = _normalize_profiles(wells, features, method=normalization,
                                     reference=reference)
    kept, excluded = _select_profile_features(
        normalized, features, feature_selection,
        corr_threshold=correlation_threshold)
    if len(kept) < 2:
        raise _ProfilingError(
            f"feature selection kept {len(kept)} of {len(features)} "
            "features; loosen it or measure more features")
    if report:
        report(f"Profiles: {len(wells)} wells, {len(kept)} of "
               f"{len(features)} features kept after "
               f"{', '.join(feature_selection) or 'no selection'}")
    metadata = [c for c in wells.columns if c not in set(features)]
    selected = normalized.loc[:, metadata + kept]
    finite = np.isfinite(selected.loc[:, kept].to_numpy(float)).all(axis=1)
    if not finite.all():
        notes.append(f"{int((~finite).sum())} well(s) with a missing kept "
                     "feature were left out of the scores.")
    scoring = selected.loc[finite].reset_index(drop=True)
    grouped = scoring.dropna(subset=group_columns)
    consensus_frame = _consensus_profiles(grouped, kept, group_columns,
                                          operation=consensus)
    if phenotype_column:
        labels = (grouped.groupby(group_columns, sort=True)[phenotype_column]
                  .first().reset_index())
        consensus_frame = consensus_frame.merge(labels, on=group_columns,
                                                how="left")
    activity = pd.DataFrame()
    activity_map = pd.DataFrame()
    if controls:
        is_negative = grouped[group_columns[0]].astype(str).isin(controls)
        try:
            activity, activity_map = _phenotypic_activity(
                grouped.reset_index(drop=True), kept, group_columns,
                is_negative.reset_index(drop=True), similarity=similarity,
                null_size=null_size, threshold=threshold, seed=seed)
        except _ProfilingError as exc:
            notes.append(f"Phenotypic activity was not scored: {exc}.")
    consistency = pd.DataFrame()
    consistency_map = pd.DataFrame()
    if phenotype_column:
        pool = consensus_frame
        if controls:
            pool = pool.loc[~pool[group_columns[0]].astype(str).isin(controls)]
        try:
            consistency, consistency_map = _phenotypic_consistency(
                pool.reset_index(drop=True), kept, phenotype_column,
                similarity=similarity, null_size=null_size,
                threshold=threshold, seed=seed)
        except _ProfilingError as exc:
            notes.append(f"Phenotypic consistency was not scored: {exc}.")
    exclude = (grouped[group_columns[0]].astype(str).isin(controls)
               if controls else None)
    replicating, percent = _percent_replicating(
        grouped.reset_index(drop=True), kept, group_columns,
        exclude=None if exclude is None else exclude.reset_index(drop=True),
        seed=seed)
    return _ProfilingResult(
        wells=wells, normalized=normalized, selected=selected,
        consensus=consensus_frame, features=list(features), kept=kept,
        excluded=excluded, group_columns=group_columns, metadata=metadata,
        activity=activity, activity_map=activity_map,
        consistency=consistency, consistency_map=consistency_map,
        replicating=replicating, percent_replicating=percent,
        options=dict(group_columns=group_columns,
                     negative_controls=controls,
                     normalization=normalization,
                     feature_selection=list(feature_selection),
                     correlation_threshold=correlation_threshold,
                     consensus=consensus, phenotype_column=phenotype_column,
                     similarity=similarity, null_size=null_size,
                     threshold=threshold, seed=seed),
        notes=notes)


def _profile_measurements(settings: Dict[str, Any], db_path: str, *,
                          report=print) -> Tuple[_ProfilingResult,
                                                 Dict[str, str]]:
    """Profile a finished Measure run and write the results beside it.

    Reads the run's ``measurements.db`` and any further databases named in
    ``profiling_databases``, aggregates each well, joins the plate map in
    ``profiling_metadata``, runs :func:`_profile_wells` with the
    ``profiling_*`` settings and writes everything with
    :func:`_write_profiles` into a ``profiles`` folder next to the database.

    :param settings: the run's settings.
    :param db_path: the run's ``measurements.db``.
    :param report: called with progress lines; ``None`` is quiet.
    :returns: ``(result, {name: path})``.
    :raises _ProfilingError: when the profiles cannot be built.
    """
    import os

    from .tabular import read_table

    databases = [db_path] + [p for p in _as_list(
        settings.get("profiling_databases")) if os.path.abspath(p)
        != os.path.abspath(db_path)]
    wells, features = _read_well_profiles(databases, report=report)
    plate_map = str(settings.get("profiling_metadata") or "").strip()
    if plate_map:
        wells, _ = _annotate_profiles(
            wells, read_table(plate_map, report=None), report=report)
    group_columns = _as_list(settings.get("profiling_treatment_column")
                             or "columnID")
    phenotype = str(settings.get("profiling_phenotype_column") or "").strip()
    result = _profile_wells(
        wells, features, group_columns=group_columns,
        negative_controls=_as_list(settings.get("profiling_negative_control")),
        normalization=str(settings.get("profiling_normalization")
                          or "mad_robustize"),
        feature_selection=(
            _PROFILE_DEFAULT_SELECTIONS
            if settings.get("profiling_feature_selection") is None
            else _as_list(settings.get("profiling_feature_selection"))),
        correlation_threshold=float(
            settings.get("profiling_correlation_threshold", 0.9)),
        phenotype_column=phenotype or None, report=report)
    out_dir = os.path.join(os.path.dirname(os.path.abspath(db_path)),
                           "profiles")
    written = _write_profiles(result, out_dir, db_path=db_path,
                              figures=True)
    return result, written

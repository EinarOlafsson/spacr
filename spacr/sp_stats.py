"""Statistical tests and multiple-comparison helpers for screen results.

WHICH TEST APPLIES IS NOT DECIDED IN THIS MODULE. :mod:`spacr.figures.stats`
is the one engine that makes that choice, and the three functions below are a
translation layer onto it that keeps this module's older call signatures and
result keys working.

Why the module stopped choosing for itself, measured on 2026-08-17 and filed
as finding 2 of instruction 127: the two implementations were run on the same
five inputs and disagreed on three of them, always in the same direction --
this one took the parametric test wherever the assumption checks had no power
to refuse it.

    case                        was              is now
    normal, equal var, n=30     T-test           T-test
    normal, UNEQUAL var, n=30   T-test           Welch's T-test
    skewed (exponential) n=30   T-test           Mann-Whitney U test
    n=3 vs 24                   T-test           Mann-Whitney U test
    n=5 vs 5                    T-test           Mann-Whitney U test

The mechanism was a normality check with no power floor: on n = 3 Shapiro-Wilk
cannot reject, so "not rejected" was read as "normal" and the t-test ran. The
engine refuses below :data:`spacr.figures.stats.MIN_N_FOR_ASSUMPTIONS` and
records the check as uninformative, which is the safe direction -- one way
loses a little power, the other publishes a difference that is not there.

Two consequences worth knowing before comparing an old CSV with a new one:

* ``Test Name`` now says WHICH t-test ran. "T-test" used to be printed for
  unequal-variance data too, and it was Student's every time -- the old call
  passed no ``equal_var``, and scipy defaults it to True.
* :func:`perform_levene_test` returns the median-centred (Brown-Forsythe)
  statistic, and NaN rather than a confident number when the smallest group is
  too small for the check to have power.

The engine is imported INSIDE each function on purpose. ``spacr.figures``
eagerly imports the panel catalog, which costs 82 ms of matplotlib at import
time (measured with ``python -X importtime``); this module is imported for
:func:`choose_p_adjust_method` and :func:`chi_pairwise` by callers that never
run a group comparison. Do not lift these into module scope.
"""

from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact
import itertools
from statsmodels.stats.multitest import multipletests

# The engine names the tests for a figure legend; this module's callers write
# them into screen CSVs and have done since before the engine existed. Mapped
# rather than renamed, so every existing reader of ``Test Name`` keeps working
# -- and mapped onto the spelling ``spacrGraph`` already prints, so the two
# report vocabularies in this package agree. Pinned by
# tests/test_one_engine_decides_which_test_applies.py, which asserts that
# every test the engine can run is named here: a test the engine learns must
# not arrive in a CSV under a name nobody chose.
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
    num_comparisons = (num_groups * (num_groups - 1)) // 2  # Number of pairwise comparisons

    # Decision logic for choosing the adjustment method
    if num_comparisons <= 10 and num_data_points > 5:
        return 'holm'  # Balanced between power and Type I error control
    elif num_comparisons > 10 and num_data_points <= 5:
        return 'fdr_bh'  # FDR control for large number of comparisons and small sample size
    elif num_comparisons <= 10:
        return 'sidak'  # Less conservative than Bonferroni, good for independent comparisons
    else:
        return 'bonferroni'  # Very conservative, use for strict control of Type I errors

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
                # Shapiro-Wilk needs three points to have a statistic at all.
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

        # The verdict is the engine's own, taken across the groups together --
        # never re-derived from the per-group p-values above, because that
        # would throw away the Bonferroni correction the check applies.
        column_verdicts.append(
            check_normality(list(groups.values())).passed)

    # No column examined is not evidence of normality. `all([])` is True, and
    # returning True there would license a parametric test off an empty call.
    is_normal = bool(column_verdicts) and all(column_verdicts)
    return is_normal, normality_results


def perform_levene_test(df, grouping_column, data_column):
    """Levene's test for equal variance, MEDIAN-centred.

    Delegates to :func:`spacr.figures.stats.check_equal_variance`. Two things
    moved when it did, and both change the number a caller writes into a CSV:

    * The centring is the median (Brown-Forsythe), not scipy's default mean.
      The median-centred form is the robust one and is the right choice
      precisely when normality is itself in question -- which is every time
      this function is called, since it is called before anyone knows.
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
    """Run the group-comparison test the data supports, per data column.

    The choice is made by :func:`spacr.figures.stats.compare` and nothing in
    this module second-guesses it: two groups get Student's t, Welch's t or
    Mann-Whitney U, and three or more get one-way ANOVA, Welch's ANOVA or
    Kruskal-Wallis, from the normality and equal-variance checks. An assumption
    check that had no power counts as FAILED, so three replicates buy a rank
    test rather than a t-test.

    :param df: Input DataFrame containing the grouping and value columns.
    :param grouping_column: Column name identifying the group of each row.
    :param data_columns: Iterable of numeric column names to test.
    :param paired: When True, paired-sample analysis is requested. Still not
        implemented: the call prints and returns no rows. It used to be
        honoured only on the two-group path, so asking for a paired test across
        three groups silently ran an unpaired one; now every group count
        refuses the same way.
    :returns: List of per-column result dicts with ``Column``, ``Test Name``,
        ``Test Statistic``, ``p-value``, ``Groups``, and -- so a saved table is
        reportable rather than a bare p -- ``n`` per group, ``Effect Size``,
        ``Effect`` and ``Why This Test``. A comparison the engine refuses to
        make is reported as ``Test Name='not testable'`` with the reason,
        the same convention :func:`chi_pairwise` uses for an undefined pair.
    """
    from .figures.stats import compare

    unique_groups = df[grouping_column].unique()
    test_results = []

    for column in data_columns:
        if paired:
            print("Performing paired tests (not implemented in this template).")
            continue  # Extend as needed

        groups = _grouped_values(df, grouping_column, column)
        counts = ' / '.join(str(int(values.size)) for values in groups.values())
        try:
            result = compare(groups)
        except ValueError as refusal:
            # Fewer than two groups, or a group too small to test. Refusing is
            # the engine's design: a comparison that could not be made is not a
            # comparison with an unknown answer. Reported as a row rather than
            # raised, because the caller is usually writing a CSV per column.
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
        num_data_points = len(df[data_column].dropna()) // num_groups  # Assuming roughly equal data points per group
        p_adjust_method = choose_p_adjust_method(num_groups, num_data_points)

        if is_normal:
            # Tukey's HSD automatically adjusts p-values
            tukey_result = pairwise_tukeyhsd(df[data_column], df[grouping_column], alpha=0.05)
            for comparison, p_value in zip(tukey_result._results_table.data[1:], tukey_result.pvalues):
                posthoc_results.append({
                    'Comparison': f"{comparison[0]} vs {comparison[1]}",
                    'Original p-value': None,  # Tukey HSD does not provide raw p-values
                    'Adjusted p-value': p_value,
                    'Adjusted Method': 'Tukey HSD',
                    'Test Name': 'Tukey HSD'
                })
        else:
            # Dunn's test with p-value adjustment
            raw_dunn_result = sp.posthoc_dunn(df, val_col=data_column, group_col=grouping_column, p_adjust=None)
            adjusted_dunn_result = sp.posthoc_dunn(df, val_col=data_column, group_col=grouping_column, p_adjust=p_adjust_method)
            for i, group_a in enumerate(adjusted_dunn_result.index):
                for j, group_b in enumerate(adjusted_dunn_result.columns):
                    if i < j:  # Only consider unique pairs
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
    groups = raw_counts.index.unique()  # Use index from raw_counts for group pairs
    raw_p_values = []  # Store raw p-values for correction later

    # Calculate the number of groups and average number of data points per group
    num_groups = len(groups)
    num_data_points = raw_counts.sum(axis=1).mean()  # Average total data points per group

    if num_groups < 2:
        if verbose:
            print(f"\nPairwise Frequency Analysis: {num_groups} group(s), "
                  f"so there is no pair to compare.")
        return pd.DataFrame(columns=columns)

    p_adjust_method = choose_p_adjust_method(num_groups, num_data_points)

    for group1, group2 in itertools.combinations(groups, 2):
        pair = raw_counts.loc[[group1, group2]]
        # A category neither group observed contributes nothing to this pair
        # and is exactly what makes the expected frequency zero.
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

        if contingency_table.shape[1] == 2:  # Fisher's Exact Test for 2x2 tables
            oddsratio, p_value = fisher_exact(contingency_table)
            test_name = "Fisher's Exact Test"
        else:  # Chi-Square Test for larger tables
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

    # Apply p-value correction over the pairs that were actually testable.
    # Correcting across untestable pairs would inflate the family size and
    # penalise the real comparisons for tests that never ran.
    raw = np.asarray(raw_p_values, dtype=float)
    testable = ~np.isnan(raw)
    corrected_p_values = np.full(raw.shape, np.nan)
    if testable.any():
        corrected_p_values[testable] = multipletests(
            raw[testable], method=p_adjust_method)[1]

    # Add corrected p-values to results
    for i, result in enumerate(pairwise_results):
        result['p-value_adj'] = corrected_p_values[i]

    pairwise_df = pd.DataFrame(pairwise_results)

    pairwise_df['adj'] = p_adjust_method
    pairwise_df = pairwise_df.reindex(columns=columns)

    if verbose:
        # Print pairwise results
        print("\nPairwise Frequency Analysis Results:")
        print(pairwise_df.to_string(index=False))

    return pairwise_df

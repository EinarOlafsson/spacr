from scipy.stats import shapiro, normaltest, levene, ttest_ind, mannwhitneyu, kruskal, f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact
import itertools
from statsmodels.stats.multitest import multipletests


def choose_p_adjust_method(num_groups, num_data_points):
    """
    Selects the most appropriate p-value adjustment method based on data characteristics.
    
    Parameters:
    - num_groups: Number of unique groups being compared
    - num_data_points: Number of data points per group (assuming balanced groups)
    
    Returns:
    - A string representing the recommended p-adjustment method
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
    """Perform normality tests for each group and data column."""
    unique_groups = df[grouping_column].unique()
    normality_results = []

    for column in data_columns:
        for group in unique_groups:
            data = df.loc[df[grouping_column] == group, column].dropna()
            n_samples = len(data)

            if n_samples < 3:
                # Skip test if there aren't enough data points
                print(f"Skipping normality test for group '{group}' on column '{column}' - Not enough data.")
                normality_results.append({
                    'Comparison': f'Normality test for {group} on {column}',
                    'Test Statistic': None,
                    'p-value': None,
                    'Test Name': 'Skipped',
                    'Column': column,
                    'n': n_samples
                })
                continue

            # Choose the appropriate normality test based on the sample size
            if n_samples >= 8:
                stat, p_value = normaltest(data)
                test_name = "D'Agostino-Pearson test"
            else:
                stat, p_value = shapiro(data)
                test_name = "Shapiro-Wilk test"

            normality_results.append({
                'Comparison': f'Normality test for {group} on {column}',
                'Test Statistic': stat,
                'p-value': p_value,
                'Test Name': test_name,
                'Column': column,
                'n': n_samples
            })

        # Check if all groups are normally distributed (p > 0.05)
        normal_p_values = [result['p-value'] for result in normality_results if result['Column'] == column and result['p-value'] is not None]
        is_normal = all(p > 0.05 for p in normal_p_values)

    return is_normal, normality_results


def perform_levene_test(df, grouping_column, data_column):
    """Perform Levene's test for equal variance."""
    unique_groups = df[grouping_column].unique()
    grouped_data = [df.loc[df[grouping_column] == group, data_column].dropna() for group in unique_groups]
    stat, p_value = levene(*grouped_data)
    return stat, p_value

def perform_statistical_tests(df, grouping_column, data_columns, paired=False):
    """Perform statistical tests for each data column."""
    unique_groups = df[grouping_column].unique()
    test_results = []

    for column in data_columns:
        grouped_data = [df.loc[df[grouping_column] == group, column].dropna() for group in unique_groups]
        if len(unique_groups) == 2:  # For two groups
            if paired:
                print("Performing paired tests (not implemented in this template).")
                continue  # Extend as needed
            else:
                # Check normality for two groups
                is_normal, _ = perform_normality_tests(df, grouping_column, [column])
                if is_normal:
                    stat, p = ttest_ind(grouped_data[0], grouped_data[1])
                    test_name = 'T-test'
                else:
                    stat, p = mannwhitneyu(grouped_data[0], grouped_data[1])
                    test_name = 'Mann-Whitney U test'
        else:
            # Check normality for multiple groups
            is_normal, _ = perform_normality_tests(df, grouping_column, [column])
            if is_normal:
                stat, p = f_oneway(*grouped_data)
                test_name = 'One-way ANOVA'
            else:
                stat, p = kruskal(*grouped_data)
                test_name = 'Kruskal-Wallis test'

        test_results.append({
            'Column': column,
            'Test Name': test_name,
            'Test Statistic': stat,
            'p-value': p,
            'Groups': len(unique_groups)
        })

    return test_results


def perform_posthoc_tests(df, grouping_column, data_column, is_normal):
    """Perform post-hoc tests for multiple groups with both original and adjusted p-values."""
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
    """
    Perform pairwise chi-square or Fisher's exact tests between all unique group pairs
    and apply p-value correction.

    Parameters:
    - raw_counts (DataFrame): Contingency table with group-wise counts.
    - verbose (bool): Whether to print results for each pair.

    Returns:
    - pairwise_df (DataFrame): DataFrame with pairwise test results, including corrected p-values.
    """
    pairwise_results = []
    groups = raw_counts.index.unique()  # Use index from raw_counts for group pairs
    raw_p_values = []  # Store raw p-values for correction later
    
    # Calculate the number of groups and average number of data points per group
    num_groups = len(groups)
    num_data_points = raw_counts.sum(axis=1).mean()  # Average total data points per group
    p_adjust_method = choose_p_adjust_method(num_groups, num_data_points)

    for group1, group2 in itertools.combinations(groups, 2):
        contingency_table = raw_counts.loc[[group1, group2]].values
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
            'p-value': p_value
        })
        raw_p_values.append(p_value)

    # Apply p-value correction
    corrected_p_values = multipletests(raw_p_values, method=p_adjust_method)[1]

    # Add corrected p-values to results
    for i, result in enumerate(pairwise_results):
        result['p-value_adj'] = corrected_p_values[i]

    pairwise_df = pd.DataFrame(pairwise_results)
    
    pairwise_df['adj'] = p_adjust_method

    if verbose:
        # Print pairwise results
        print("\nPairwise Frequency Analysis Results:")
        print(pairwise_df.to_string(index=False))

    return pairwise_df

spacr.sp_stats
==============

.. py:module:: spacr.sp_stats




Module Contents
---------------

.. py:function:: choose_p_adjust_method(num_groups, num_data_points)

   Selects the most appropriate p-value adjustment method based on data characteristics.

   Args:
   - num_groups: Number of unique groups being compared
   - num_data_points: Number of data points per group (assuming balanced groups)

   Returns:
   - A string representing the recommended p-adjustment method


.. py:function:: perform_normality_tests(df, grouping_column, data_columns)

   Perform normality tests (Shapiro-Wilk or D'Agostino-Pearson) on grouped data.

   :param df: Input DataFrame containing the data to test.
   :type df: pd.DataFrame
   :param grouping_column: Column name to group data by (e.g., condition or treatment).
   :type grouping_column: str
   :param data_columns: List of column names containing numeric data to test for normality.
   :type data_columns: list of str

   :returns:     - is_normal (bool): True if all group-column combinations pass the normality test (p > 0.05), else False.
                 - normality_results (list of dict): List of results for each group-column combination including:
                     - 'Comparison': Description of the test
                     - 'Test Statistic': Computed test statistic (None if skipped)
                     - 'p-value': P-value of the test (None if skipped)
                     - 'Test Name': Name of the test used or 'Skipped'
                     - 'Column': The data column tested
                     - 'n': Sample size
   :rtype: tuple


.. py:function:: perform_levene_test(df, grouping_column, data_column)

   Perform Levene’s test for equal variances across groups.

   :param df: The DataFrame containing the data.
   :type df: pd.DataFrame
   :param grouping_column: The column indicating group membership.
   :type grouping_column: str
   :param data_column: The column containing the numeric data.
   :type data_column: str

   :returns: (statistic, p-value) from Levene’s test.
   :rtype: tuple


.. py:function:: perform_statistical_tests(df, grouping_column, data_columns, paired=False)

   Perform statistical tests to compare groups for each specified data column.

   :param df: The DataFrame containing the data.
   :type df: pd.DataFrame
   :param grouping_column: The column indicating group membership.
   :type grouping_column: str
   :param data_columns: List of column names to perform tests on.
   :type data_columns: list
   :param paired: Whether to use paired tests (only for two-group comparisons).
   :type paired: bool

   :returns:

             Each dict contains:
                 - 'Column': Name of the column tested.
                 - 'Test Name': Statistical test used.
                 - 'Test Statistic': Test statistic value.
                 - 'p-value': P-value of the test.
                 - 'Groups': Number of groups compared.
   :rtype: list of dict


.. py:function:: perform_posthoc_tests(df, grouping_column, data_column, is_normal)

   Perform post-hoc pairwise comparisons between groups after a significant overall test (e.g., ANOVA or Kruskal-Wallis).

   :param df: Input DataFrame containing the data to analyze.
   :type df: pd.DataFrame
   :param grouping_column: Column name representing group membership.
   :type grouping_column: str
   :param data_column: Column name with the continuous variable to compare.
   :type data_column: str
   :param is_normal: Indicator of whether the data meets normality assumptions
                     (determines test type: Tukey HSD if True, Dunn's test if False).
   :type is_normal: bool

   :returns:

             List of dictionaries summarizing pairwise comparisons, each including:
                 - 'Comparison': Description of the group pair.
                 - 'Original p-value': Raw p-value (None for Tukey HSD).
                 - 'Adjusted p-value': Corrected p-value for multiple testing.
                 - 'Adjusted Method': Method used for p-value adjustment.
                 - 'Test Name': The post-hoc test performed ("Tukey HSD" or "Dunn's Post-hoc").
   :rtype: list of dict


.. py:function:: chi_pairwise(raw_counts, verbose=False)

   Perform pairwise statistical tests (Chi-Square or Fisher's Exact) on contingency tables
   derived from count data, and apply multiple testing correction.

   :param raw_counts: A DataFrame where rows represent groups and columns represent categories.
                      The values are raw counts.
   :type raw_counts: pd.DataFrame
   :param verbose: If True, prints the resulting pairwise test summary.
   :type verbose: bool

   :returns:

             A DataFrame with pairwise comparisons including:
                 - 'Group 1': First group in the comparison
                 - 'Group 2': Second group in the comparison
                 - 'Test Name': Type of statistical test used ('Chi-Square' or 'Fisher's Exact')
                 - 'p-value': Raw p-value for the test
                 - 'p-value_adj': Adjusted p-value after multiple testing correction
                 - 'adj': Name of the correction method used
   :rtype: pd.DataFrame



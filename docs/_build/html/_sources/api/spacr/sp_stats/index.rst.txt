spacr.sp_stats
==============

.. py:module:: spacr.sp_stats




Module Contents
---------------

.. py:function:: choose_p_adjust_method(num_groups, num_data_points)

   Selects the most appropriate p-value adjustment method based on data characteristics.

   Parameters:
   - num_groups: Number of unique groups being compared
   - num_data_points: Number of data points per group (assuming balanced groups)

   Returns:
   - A string representing the recommended p-adjustment method


.. py:function:: perform_normality_tests(df, grouping_column, data_columns)

   Perform normality tests for each group and data column.


.. py:function:: perform_levene_test(df, grouping_column, data_column)

   Perform Levene's test for equal variance.


.. py:function:: perform_statistical_tests(df, grouping_column, data_columns, paired=False)

   Perform statistical tests for each data column.


.. py:function:: perform_posthoc_tests(df, grouping_column, data_column, is_normal)

   Perform post-hoc tests for multiple groups with both original and adjusted p-values.


.. py:function:: chi_pairwise(raw_counts, verbose=False)

   Perform pairwise chi-square or Fisher's exact tests between all unique group pairs
   and apply p-value correction.

   Parameters:
   - raw_counts (DataFrame): Contingency table with group-wise counts.
   - verbose (bool): Whether to print results for each pair.

   Returns:
   - pairwise_df (DataFrame): DataFrame with pairwise test results, including corrected p-values.



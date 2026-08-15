Plate-aware guide permutation analysis
======================================

Use ``analysis_mode='guide_permutation'`` when the independent experimental
unit is a well and a simultaneous guide model is not identifiable because
there are more guide columns than wells or because guide fractions are highly
correlated.  The mode uses the same per-cell score CSV and sequencing-count
CSV inputs as :func:`spacr.ml.perform_regression`.

Each guide is tested separately after its well-level fraction and the
well-level phenotype have both been adjusted for plate (and any optional
measured nuisance columns).  Two-sided empirical P values come from
Freedman--Lane residual permutations restricted within plate.  Multiple-test
correction is then applied separately within each requested minimum-support
family.  These are marginal associations: co-occurring guides may share a
signal, so the output is not a set of mutually adjusted causal coefficients.

Example
-------

.. code-block:: python

   from spacr.ml import perform_regression

   output = perform_regression({
       "analysis_mode": "guide_permutation",
       "score_data": ["plate1_scores.csv", "plate2_scores.csv"],
       "count_data": ["plate1_counts.csv", "plate2_counts.csv"],
       "plates_score": [1, 2],
       "plates_count": [1, 2],
       "dependent_variable": "prediction_probability_class_1",
       "score_column": "prediction_probability_class_1",
       "agg_type": "median",
       "transform": "log",
       "min_cell_count": 100,
       "fraction_threshold": 0.02,
       "guide_min_wells": [1, 2, 3, 4],
       "guide_primary_min_wells": 1,
       "guide_permutations": 200_000,
       "guide_permutation_seed": 20260814,
       "guide_permutation_block": "plateID",
       "guide_nuisance_columns": [],
       "multiple_testing_method": "fdr_bh",
       "fdr_alpha": 0.05,
   })

   results = output["results"]
   primary_hits = output["significant"]  # selected support family

The important output columns are ``wells_with_guide``,
``standardized_marginal_effect``, ``permutation_p_value``,
``adjusted_p_value`` and ``significant``.  ``guide_min_wells`` accepts either
one positive integer or a list.  The empirical P value for a guide is computed
once at the smallest requested threshold; only the multiple-testing family
and adjusted value change across support thresholds.
``guide_primary_min_wells`` selects which requested family is returned in
``output['significant']``; when it is ``None``, spaCR uses the smallest family
so the primary correction retains every tested guide.

Multiple-testing choices
------------------------

``multiple_testing_method`` accepts ``fdr_bh`` (Benjamini--Hochberg),
``fdr_by``, ``bonferroni``, ``holm`` or ``none``.  ``fdr_bh`` is the default.
For reproducibility, record the requested support families, permutation count,
seed, blocking column, nuisance columns, presence threshold, correction method
and alpha together with the input-file checksums.

Direct analysis of an existing long table
-----------------------------------------

If a previous run already produced ``regression_data.csv``, it can be analyzed
without re-reading the per-cell and sequencing inputs:

.. code-block:: python

   import pandas as pd
   from spacr.guide_permutation import analyse_long_guide_table

   table = pd.read_csv("regression_data.csv")
   results = analyse_long_guide_table(
       table,
       "log_prediction_probability_class_1",
       min_wells=[1, 2, 3, 4],
       block_column="plateID",
       n_permutations=200_000,
       random_state=20260814,
       multiple_testing="fdr_bh",
       alpha=0.05,
   )

# Interpret a vision-model score

**Purpose.** Model vision-classifier outputs using measured morphology and intensity features to identify associated quantitative attributes.

**Recommended use.** Use to relate image-model predictions to interpretable cellular measurements while retaining grouped validation.

**Primary outputs.** Surrogate-model fidelity metrics and ranked feature explanations.

**Desktop route.** Classify → Explain CV Model

---

> Paths in this notebook are placeholders. Set `src` and any other required path to the experimental data before execution.
> spaCR writes outputs within, or immediately adjacent to, the configured source directory unless an explicit output path is set.

## 2. API entry point

This workflow calls the following public function:

- [`spacr.surrogate.run_explain_cv`](https://einarolafsson.github.io/spacr/api/spacr/surrogate/index.html#spacr.surrogate.run_explain_cv)

```python
run_explain_cv(settings)
```

Parameter definitions and defaults are generated from the same public API and are listed in the settings reference below.

## 3. Settings and API reference

This linked reference contains the full generated setting descriptions. Return to the notebook to edit essential inputs or open Advanced settings. Required inputs must be supplied; conditional inputs apply only to the indicated branch. Unchanged optional values use the installed API defaults. Descriptions and displayed defaults reflect the spaCR source used to generate this reference.

### [`spacr.surrogate.run_explain_cv`](https://einarolafsson.github.io/spacr/api/spacr/surrogate/index.html#spacr.surrogate.run_explain_cv)


#### Essential inputs

- **`db_path`** *(required)* — (str) - Exact measurements.db whose objects an analysis reads. Choosing another database is refused when object or crop identities do not match, preventing cross-experiment joins. Default '' requires selection.
- **`predictions_file`** *(required)* — (str) - Existing per-object prediction CSV joined to measured objects. Analysis modules read this exact output and do not silently rerun a model or substitute another run. Default '' requires selection.

<details>
<summary>Advanced settings reference — click to expand</summary>


<details>
<summary>Source &amp; provenance (2 settings)</summary>

#### Source & provenance

- **`path_column`** *(optional)* — (str) - Column in a prediction CSV containing crop paths for a one-to-one object join. Change it only when the exporter used another name. Default path. Default: `'path'`.
- **`prediction_column`** *(optional)* — (str) - Column containing the CV class the surrogate must reproduce. Selecting a score instead changes the learning target and meaning of fidelity. Default pred. Default: `'pred'`.

</details>

<details>
<summary>Surrogate &amp; validation (6 settings)</summary>

#### Surrogate & validation

- **`surrogate_model`** *(optional)* — (str) - Estimator family used to reproduce CV decisions: Random Forest, histogram gradient boosting, or optional XGBoost. Missing XGBoost is refused rather than silently substituted. Default random_forest. Default: `'random_forest'`.
- **`surrogate_split_by`** *(optional)* — (str) - Independent acquisition unit kept intact between fitting and fidelity evaluation. Use well by default, or plate for a harder batch-generalization test. Default: `'well'`.
- **`surrogate_test_size`** *(optional)* — (float) - Fraction of independent well or plate groups held out for fidelity and permutation importance. Larger values strengthen evaluation but leave fewer groups for fitting. Default 0.3. Default: `0.3`.
- **`surrogate_n_estimators`** *(optional)* — (int) - Number of trees or boosting iterations in the selected surrogate. Raising it can improve fidelity but increases runtime and may overfit small training sets. Default 300. Default: `300`.
- **`surrogate_random_seed`** *(optional)* — (int) - Seed shared by grouped splitting, estimator fitting and importance calculations. Keep it fixed for exact reproduction and change it only for sensitivity checks. Default 0. Default: `0`.
- **`surrogate_min_fidelity_improvement`** *(optional)* — (float) - Accuracy improvement over the majority-class baseline required before feature importances are presented as explanations. Raising it withholds more weak surrogates. Default 0.05. Default: `0.05`.

</details>

<details>
<summary>Importance &amp; diagnostics (6 settings)</summary>

#### Importance & diagnostics

- **`surrogate_importance_methods`** *(optional)* — (list) - Feature-importance measures to report: 'gain' (the model's native split gain; not available for histogram gradient boosting), 'permutation' (held-out score drop when one feature is shuffled) and 'shap' (mean absolute SHAP; needs the optional shap package). Each is written as a column of feature_importance.csv and a panel of the ranked bar plot. Default ['gain', 'permutation', 'shap']. Default: `['gain', 'permutation', 'shap']`.
- **`surrogate_n_repeats`** *(optional)* — (int) - Repeated held-out shuffles per feature for permutation importance. More repeats stabilize rankings but multiply scoring time without changing the fitted model. Default 5. Default: `5`.
- **`surrogate_shap_explainer`** *(optional)* — (str) - How SHAP is computed: 'tree' (exact TreeSHAP for tree ensembles), 'kernel' (model-agnostic KernelSHAP against a k-means background; slow, capped at 100 explained objects) or 'auto' (TreeSHAP, falling back to KernelSHAP for a model it cannot read). Default 'auto'. Default: `'auto'`.
- **`surrogate_shap_max_samples`** *(optional)* — (int) - Maximum held-out objects receiving signed SHAP values and dependence plots. Raising it improves coverage at substantial computational cost; sampled IDs are recorded. Default 500. Default: `500`.
- **`surrogate_exclude`** *(optional)* — (list) - Additional measured features barred from the explanatory matrix. Use it for known artifacts or post-treatment annotations; model outputs, classes and identifiers remain excluded automatically. Default []. Default: `[]`.
- **`surrogate_correlation_threshold`** *(optional)* — (float) - Absolute Spearman correlation above which a held-out feature pair is disclosed. Lower values reveal more redundancy and produce a larger audit table. Default 0.9. Default: `0.9`.

</details>

<details>
<summary>Output &amp; runtime (2 settings)</summary>

#### Output & runtime

- **`dst`** *(optional)* — (str) - Folder receiving versioned tables, manifests and figures. Default '' uses a module-specific folder beside the primary input, keeping different analyses separated. Default: `''`.
- **`verbose`** *(optional)* — (bool) - Print the resolved settings table, channel and model choices per object type, row counts per table, and object counts after each filter. It only adds console output; enable it to identify which stage produced an unexpected object count. The default is True for mask, UMAP, screen analysis, barcode mapping and Cellpose training, and False for measure, plotting helpers and regression. Invasion and Replication also start with console detail disabled. Default: `True`.

</details>

</details>

## 4. Run

Return to the notebook, configure inputs, and run its analysis cell only when ready. Long operations report progress through spaCR's logging system; set `SPACR_LOG_LEVEL=DEBUG` before starting Jupyter for more detail.

## Outputs and next steps

Surrogate-model fidelity metrics and ranked feature explanations.

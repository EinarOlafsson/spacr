# Classify objects from measured features

**Purpose.** Fit a supervised tabular classifier to per-object measurements and apply it to the full dataset.

**Recommended use.** Use when morphology, intensity, or spatial measurements provide an interpretable representation of the phenotype.

**Primary outputs.** Per-object predictions, validation metrics, feature importance, and SHAP summaries when supported.

**Desktop route.** Classify → Machine Learning

---

> Paths in this notebook are placeholders. Set `src` and any other required path to the experimental data before execution.
> spaCR writes outputs within, or immediately adjacent to, the configured source directory unless an explicit output path is set.

## 2. API entry point

This workflow calls the following public function:

- [`spacr.ml.generate_ml_scores`](https://einarolafsson.github.io/spacr/api/spacr/ml/index.html#spacr.ml.generate_ml_scores)

```python
generate_ml_scores(settings)
```

Parameter definitions and defaults are generated from the same public API and are listed in the settings reference below.

## 3. Settings and API reference

This linked reference contains the full generated setting descriptions. Return to the notebook to edit essential inputs or open Advanced settings. Required inputs must be supplied; conditional inputs apply only to the indicated branch. Unchanged optional values use the installed API defaults. Descriptions and displayed defaults reflect the spaCR source used to generate this reference.

### [`spacr.ml.generate_ml_scores`](https://einarolafsson.github.io/spacr/api/spacr/ml/index.html#spacr.ml.generate_ml_scores)


#### Essential inputs

- **`src`** *(required)* — (str, path) - Folder the current step reads from and writes into: raw images for mask generation, the merged/ folder of .npy stacks for measure, the plate root for dataset/regression steps, or the folder of .fastq.gz reads for sequencing. Outputs (stack/, masks/, measurements/measurements.db, datasets/, results/) are created inside it. A list of paths, or a "['a','b']" string, processes several plates in one run. No usable default: the settings factories fill a placeholder ('path' or '/path/to/src'), so this must be supplied.
- **`annotation_column`** *(conditionally required)* — (str) - Integer column in the png_list table that stores manual class labels. The Annotate app adds it with ALTER TABLE if it is absent and writes labels to it. This column provides the reference labels when dataset_mode is 'annotation' and is the fallback when annotation_columns is unset. Supplying it while dataset_mode is unset also selects annotation mode for compatibility with older settings files. Default None.
- **`positive_control_id`** *(conditionally required)* — (str) - Identifier of the positive-control class. In ML screening it is the value in location_column (e.g. 'c2') whose objects are labelled class 1 for training; in gRNA regression it is a gene/gRNA ID substring (e.g. '239740') matched against coefficient names to tag them 'pc' in the results and volcano plot. Defaults 'c2' and '239740' respectively.
- **`negative_control_id`** *(conditionally required)* — (str) - Identifier of the negative-control class. In ML screening it is the value in location_column (e.g. 'c1') whose objects are labelled class 0 for training; in gRNA regression it is a gene/gRNA ID substring (e.g. '233460') matched against coefficient names to tag them 'nc' in the results and volcano plot. Defaults 'c1' and '233460' respectively.

<details>
<summary>Advanced settings reference — click to expand</summary>


<details>
<summary>Labels &amp; Classes (2 settings)</summary>

#### Labels & Classes

- **`dataset_mode`** *(optional)* — (str) - How training classes are defined: 'metadata' splits crops by well metadata, 'annotation' by the values in one or more annotation columns of png_list. Either way the classes themselves are set in the Classes editor, which names a column and a value per class. A settings file written before the 'measurement' basis was removed still loads: it is read as 'annotation', which is what its threshold rules resolved to after writing their label column. Any other value aborts and returns no dataset. Default 'metadata'. Default: `'metadata'`.
- **`location_column`** *(optional)* — (str) - Metadata column searched for positive_control_id and negative_control_id values when labelling rows for machine-learning training, normally 'columnID' or 'rowID'. Set 'rowID' when controls are arranged along plate rows instead of columns. annotation_column overrides this setting when specified. Default 'columnID'. Default: `'columnID'`.

</details>

<details>
<summary>Feature Preparation (7 settings)</summary>

#### Feature Preparation

- **`channel_of_interest`** *(optional)* — (int, list, or str) - Measurements available to the model. Specify one channel to train on that channel alone, multiple channels to use their combination of measurements, or 'shape' to use outline-derived features. An empty value includes every measurement. A colocalisation feature is associated with both measured channels, so selecting either channel also includes their shared colocalisation features. This setting also selects the channel used for recruitment measurements. Default 3 in machine-learning steps and 1 or 2 elsewhere. Default: `3`.
- **`exclude`** *(optional)* — (str or list) - Exclude measurement features before UMAP or ML. Exact columns take priority, then whole-token channel_N groups (channel_0 differs from channel_01), then full-match regexes (.*channel_0.* matches contained text). Unmatched selectors warn; invalid regexes raise a pattern-naming error. Logs give hit counts; verbose adds matched names. Applied with channel_of_interest. Features, not rows: use exclude_rows for rows. Default None keeps all features. Default: `None`.
- **`nuclei_limit`** *(optional)* — (int, bool, or None) - Cap on nuclei per cell, applied when the per-object tables are merged. None disables the filter, True keeps only single-nucleus cells, and an integer N keeps cells with N or fewer. Cells over the cap are dropped from the merged table entirely. Do not pass False: it is interpreted as 0 and removes every cell, leaving an empty analysis rather than raising an error. Default None. Merged Classifier starts at True and Recruitment starts at 1, so both initially retain only single-nucleus cells. Replication starts at 10. Default: `True`.
- **`pathogen_limit`** *(optional)* — (int, bool, or None) - Maximum pathogens per cell. True or 1 = single pathogen only; None or False = no limit; int = custom limit. Default varies by module (1, 3, 10 or 1000 depending on the factory that fills it), so check the module's own settings rather than assuming one value. Default: `3`.
- **`remove_highly_correlated_features`** *(optional)* — (bool) - In the machine-learning feature table, drop any feature whose absolute Pearson correlation with an already-kept feature exceeds 0.95, applied after the channel_of_interest filter. Leave it on so redundant measurements do not split importance scores and slow fitting; turn it off only when you need every original column. Default True. Note the UMAP path uses remove_highly_correlated instead. Default: `True`.
- **`remove_low_variance_features`** *(optional)* — (bool) - Drop numeric features whose variance across objects falls below 0.01 before model fitting -- near-constant columns that carry no discriminative signal but still cost time and dilute importance rankings. Turn it off only when your features live on a very small numeric scale, where genuine signal can fall under that fixed cut-off. Default True. Default: `True`.
- **`min_cells_per_well`** *(optional)* — (int) - Wells with fewer than this many cells are dropped. In a regression it is scored objects and the well is left out of the fit; in the machine-learning screen it is measured cells and the well is left out of the plate heatmap, whose pivot is then filled with 0, so an excluded well renders at the bottom of the colour scale rather than blank. Raising it removes noisy, sparsely imaged wells at the cost of power. Set 0 to switch it off. Default 100 for a regression, 25 for the screen. Default: `25`.

</details>

<details>
<summary>Plate &amp; Batch Correction (8 settings)</summary>

#### Plate & Batch Correction

- **`batch_correction`** *(optional)* — (str) - Plate/batch correction applied before Image UMAP, ML screen classification or phenotype regression. 'none' leaves measurements alone; 'center' removes each plate's mean shift; 'zscore' aligns plate means and variances; 'robust_zscore' uses median/MAD and tolerates outliers; 'combat' models the batch effect while protecting the terms named in batch_covariate_column. Correct when plates were stained or imaged separately; leave off when they were not, since every method removes real signal that happens to align with plate. See spacr.batch_correction.correct_batch_effects. Default 'none'. Default: `'none'`.
- **`batch_column`** *(optional)* — (str) - Metadata column that identifies independent acquisition batches, normally 'plateID'. Every analyzed row must have a value and at least batch_min_samples rows must occur in each batch. Use an acquisition date or instrument ID only if that is the nuisance source you intend to remove. Default 'plateID'. API: spacr.batch_correction.correct_batch_effects. Default: `'plateID'`.
- **`batch_control_column`** *(optional)* — (str or None) - Metadata column containing reference-control labels for control_center, normally 'columnID' for plate controls. It is ignored by center, zscore, robust_zscore, and none. Blank follows col_to_compare in Image UMAP or location_column in Classify (ML); regression defaults to 'columnID'. API: spacr.batch_correction.correct_batch_effects. Default: `None`.
- **`batch_control_values`** *(optional)* — (str, number, list or None) - Reference/negative-control value(s) in batch_control_column used by control_center. Each plate needs at least batch_min_samples matching rows. Image UMAP falls back to neg and Classify (ML) to negative_control_id when this field is blank; regression requires an explicit value. Default varies by module. API: spacr.batch_correction.correct_batch_effects. Default: `None`.
- **`batch_covariate_column`** *(optional)* — (str, list or None) - Metadata column(s) naming the biological effects ComBat must preserve, for example 'condition' or 'condition,timepoint'. ComBat estimates the batch effect from residuals after fitting these terms, so unlisted effects may be removed with the plate effect. Include every treatment effect that must remain in the corrected data. See spacr.batch_correction.correct_batch_effects. Default None. Default: `None`.
- **`batch_combat_mean_only`** *(optional)* — (bool) - True corrects only the additive batch shift and leaves each batch's scale alone. Use it when the plates differ in level but not in spread, or when a batch has too few rows for a stable variance estimate. False (the default) corrects both location and scale, which is standard ComBat. Ignored by every method other than combat. API: spacr.batch_correction.correct_batch_effects. Default: `False`.
- **`batch_min_samples`** *(optional)* — (int) - Minimum number of rows required in every batch, and minimum matching reference controls per batch for control_center. Correction stops with an actionable error below this threshold because a one- or two-object plate estimate is unstable. Default 3. API: spacr.batch_correction.correct_batch_effects. Default: `3`.
- **`batch_missing_control`** *(optional)* — (str) - Policy when control_center cannot find enough reference controls on a plate: 'error' stops rather than silently mixing corrected and raw plates; 'skip' leaves that plate unchanged and records a warning. Default 'error'. API: spacr.batch_correction.correct_batch_effects. Default: `'error'`.

</details>

<details>
<summary>Classifier &amp; Validation (7 settings)</summary>

#### Classifier & Validation

- **`model_type_ml`** *(optional)* — (str) - Classifier fitted by ml_analysis to separate positive- from negative-control wells and rank per-object features by permutation importance. Options are xgboost (default), lightgbm, catboost, random_forest, extra_trees, gradient_boosting, logistic_regression, svm and mlp; lightgbm and catboost require their optional packages. reg_alpha, reg_lambda and learning_rate affect only boosted models; logistic_regression provides a linear reference model. Default: `'xgboost'`.
- **`n_estimators`** *(optional)* — (int) - Number of trees or boosting rounds in the tabular ML classifier - n_estimators for RandomForest/ExtraTrees/XGBoost/LightGBM, iterations for CatBoost, max_iter for HistGradientBoosting. More rounds keep improving fit up to a plateau while training time grows linearly; boosted models can overfit past it. Default 1000. Default: `1000`.
- **`learning_rate`** *(optional)* — (float) - Initial optimizer step size. Values that are too high may prevent convergence; values that are too low may slow convergence or converge to a suboptimal solution. A value near 1e-3 is commonly used for training from random initialization, while 1e-4 to 1e-5 is appropriate for fine-tuning ImageNet weights (init_weights=True). The selected schedule modifies this initial value during training. Default 0.001. Default: `0.001`.
- **`test_size`** *(optional)* — (float) - Fraction of labelled single-object rows reserved as the test split for the tabular machine-learning classifier; the remainder trains the model. Increasing the value reduces uncertainty in the accuracy estimate but leaves fewer labelled rows for training. Valid range 0-1; default 0.2 (20% test). Default: `0.2`.
- **`cross_validation`** *(optional)* — (bool) - Score the feature-based classifier (Classify, 'ml' family) with class-stratified cross-validation instead of a single train/test split: up to 5 folds, whole cv_group_by groups per fold, so every control object receives an out-of-fold prediction. Each fold's optimal probability threshold is computed and printed when verbose is on; predictions are not re-thresholded. Gives a less variable accuracy estimate on small control sets, at roughly 5x the training time. Default True. Default: `True`.
- **`reg_alpha`** *(optional)* — (float) - L1 penalty on leaf weights for the gradient-boosted classifier (XGBoost and LightGBM; ignored by the other model_type_ml choices). Raising it drives more leaf weights to exactly zero, shrinking the model and its effective feature set - raise it when training accuracy far exceeds test accuracy. Any value &gt;= 0. Default 0.1. Default: `0.1`.
- **`reg_lambda`** *(optional)* — (float) - L2 penalty on leaf weights for the gradient-boosted classifier (XGBoost, LightGBM, and CatBoost's l2_leaf_reg). Raising it shrinks all weights smoothly rather than zeroing them, damping the influence of any single feature and curbing overfitting, at the risk of underfitting if pushed too far. Any value &gt;= 0. Default 1.0. Default: `1.0`.

</details>

<details>
<summary>Feature Selection &amp; Importance (3 settings)</summary>

#### Feature Selection & Importance

- **`prune_features`** *(optional)* — (bool) - Before training, keep only the top_features columns with the highest ANOVA F-score against the control labels (sklearn SelectKBest with f_classif). Speeds up fitting and can curb overfitting on small control sets, but discards features the model might have used and scores each feature in isolation, ignoring interactions. Default False. Default: `False`.
- **`top_features`** *(optional)* — (int) - Feature cap in the ML screen analysis: how many rows the feature-importance and permutation-importance bar plots show, and how many top-ranked features the SHAP refit and its summary plot use. It is also the k of the SelectKBest pruning applied before the model is fitted, but only when prune_features is True - with prune_features at its default False the classifier trains on every feature and this is reporting/SHAP scope only. Raise for a fuller picture, lower for readable plots. Default 30. Default: `30`.
- **`n_repeats`** *(optional)* — (int) - Number of random shuffles per feature when computing permutation importance for the ML classifier. More repeats reduce uncertainty in the importance ranking but require an additional prediction pass per feature and repeat. Default 10; use 3-5 for a faster preliminary assessment of wide feature tables. Default: `10`.

</details>

<details>
<summary>Plots &amp; Heatmaps (4 settings)</summary>

#### Plots & Heatmaps

- **`cmap`** *(optional)* — (str) - Matplotlib colormap applied to single-channel image previews and plate heatmaps. Perceptually uniform maps ('viridis', 'inferno', 'magma') preserve the relative visibility of intensity differences; 'gray' resembles the raw single-channel microscope image. Any registered matplotlib name is accepted, with an '_r' suffix to reverse it. Default 'inferno' for image plots and 'viridis' for plate heatmaps. Default: `'viridis'`.
- **`heatmap_feature`** *(optional)* — (str) - Numeric column that is aggregated per well and color-mapped in the plate heatmap after ML scoring, e.g. 'predictions' for the classifier score or 'recruitment' for the pathogen/cytoplasm intensity ratio. Must be a numeric column of the scored dataframe or the run raises ValueError listing the valid names. Default 'predictions'. Default: `'predictions'`.
- **`grouping`** *(optional)* — (str) - How per-object values collapse to one number per well in the plate heatmap: 'mean' averages heatmap_feature over the objects in a well, 'sum' totals them, 'count' ignores the feature and colors wells by object count. Use 'count' to spot uneven seeding or dropout, 'mean' for phenotype strength. Default 'mean'; any other value raises ValueError. Default: `'mean'`.
- **`min_max`** *(optional)* — (str) - Color limits for the plate heatmap: 'allq' scales to the 2nd-98th percentile of well values so a handful of extreme wells cannot flatten the rest, 'all' scales to the true min and max. A two-element list is also accepted, where floats are read as quantiles and integers as absolute vmin/vmax. Default 'allq'. Default: `'allq'`.

</details>

<details>
<summary>Runtime &amp; Reliability (3 settings)</summary>

#### Runtime & Reliability

- **`verbose`** *(optional)* — (bool) - Print the resolved settings table, channel and model choices per object type, row counts per table, and object counts after each filter. It only adds console output; enable it to identify which stage produced an unexpected object count. The default is True for mask, UMAP, screen analysis, barcode mapping and Cellpose training, and False for measure, plotting helpers and regression. Invasion and Replication also start with console detail disabled. Default: `True`.
- **`n_jobs`** *(optional)* — (int) - CPU workers for parallel stages: measurement, mask adjustment, DataLoader loading, and the sklearn/UMAP calls where -1 means every core. Raise it to shorten CPU-bound steps until RAM or disk I/O saturates. Note the measure-and-crop pipeline overrides your value with cpu_count()-4. Defaults vary by pipeline: cpu_count()-4, -1, or None. Default: `-1`.
- **`ram_guard`** *(optional)* — (bool) - Keep worker pools from filling RAM. Before any module starts its workers, spaCR estimates each worker's memory from one input (a field, mask, crop batch or table) and lowers n_jobs to the count that leaves 12.5% of RAM free, printing a warning; the GUI asks first. Off keeps your n_jobs, and Measure fields still wait while free RAM is below that reserve. Default True. Default: `True`.

</details>

</details>

## 4. Run

Return to the notebook, configure inputs, and run its analysis cell only when ready. Long operations report progress through spaCR's logging system; set `SPACR_LOG_LEVEL=DEBUG` before starting Jupyter for more detail.

## Outputs and next steps

Per-object predictions, validation metrics, feature importance, and SHAP summaries when supported.

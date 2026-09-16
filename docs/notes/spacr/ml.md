# Notes from `spacr/ml.py`

Prose lifted out of `spacr/ml.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [display](#display) (1 entry)
- [Module level](#module-level) (11 entries)
- [_calibrated_fraction_threshold](#_calibrated_fraction_threshold) (2 entries)
- [_graph_sequencing_stats](#_graph_sequencing_stats) (1 entry)
- [_keep_figures_with_the_run](#_keep_figures_with_the_run) (1 entry)
- [QuasiBinomial.__init__](#quasibinomial__init__) (1 entry)
- [calculate_p_values](#calculate_p_values) (3 entries)
- [perform_mixed_model](#perform_mixed_model) (4 entries)
- [centre_on_controls](#centre_on_controls) (1 entry)
- [prepare_formula](#prepare_formula) (4 entries)
- [fit_mixed_model](#fit_mixed_model) (9 entries)
- [check_and_clean_data.check_collinearity](#check_and_clean_datacheck_collinearity) (5 entries)
- [check_and_clean_data](#check_and_clean_data) (7 entries)
- [minimum_cell_simulation](#minimum_cell_simulation) (22 entries)
- [_bootstrap_wald_p_values](#_bootstrap_wald_p_values) (2 entries)
- [_gene_of_design_column](#_gene_of_design_column) (1 entry)
- [label_control_condition](#label_control_condition) (4 entries)
- [process_model_coefficients](#process_model_coefficients) (4 entries)
- [_show_response_distribution](#_show_response_distribution) (3 entries)
- [check_distribution](#check_distribution) (8 entries)
- [double_transform_warning](#double_transform_warning) (1 entry)
- [resolve_glm_transform_conflict](#resolve_glm_transform_conflict) (2 entries)
- [pick_glm_family_and_link](#pick_glm_family_and_link) (1 entry)
- [_choose_glm_family](#_choose_glm_family) (2 entries)
- [binarise_response](#binarise_response) (2 entries)
- [_left_blank](#_left_blank) (1 entry)
- [_absorbed_factor_codes](#_absorbed_factor_codes) (1 entry)
- [_fit_absorbed_least_squares](#_fit_absorbed_least_squares) (7 entries)
- [_GlumResults.__init__](#_glumresults__init__) (1 entry)
- [_fit_glum_glm](#_fit_glum_glm) (5 entries)
- [regression_model](#regression_model) (12 entries)
- [regression_model._glm_auto](#regression_model_glm_auto) (4 entries)
- [regression_model._glm_poisson](#regression_model_glm_poisson) (1 entry)
- [regression_model._wls](#regression_model_wls) (1 entry)
- [regression_model._rlm](#regression_model_rlm) (1 entry)
- [regression_model._hinge](#regression_model_hinge) (2 entries)
- [regression_model._find_best_hinge_alpha](#regression_model_find_best_hinge_alpha) (1 entry)
- [regression_model._group_lasso](#regression_model_group_lasso) (4 entries)
- [regression_model._rra](#regression_model_rra) (4 entries)
- [_fit_horseshoe_poisson](#_fit_horseshoe_poisson) (2 entries)
- [_HorseshoeResults.__init__](#_horseshoeresults__init__) (1 entry)
- [_reconcile_random_row_column_effects](#_reconcile_random_row_column_effects) (2 entries)
- [_write_regression_qc](#_write_regression_qc) (4 entries)
- [_wide_fixed_effect_design](#_wide_fixed_effect_design) (1 entry)
- [regression](#regression) (22 entries)
- [regression_levels](#regression_levels) (2 entries)
- [_show_well_distributions](#_show_well_distributions) (3 entries)
- [_show_plates](#_show_plates) (1 entry)
- [_show_house_style_panels](#_show_house_style_panels) (2 entries)
- [_write_regression_sheet](#_write_regression_sheet) (1 entry)
- [fit_quality_note](#fit_quality_note) (1 entry)
- [summary_for_console](#summary_for_console) (1 entry)
- [_split_prc](#_split_prc) (1 entry)
- [_is_row_column_pair](#_is_row_column_pair) (1 entry)
- [resolve_auto_inference](#resolve_auto_inference) (3 entries)
- [normalize_regression_input_pairs](#normalize_regression_input_pairs) (2 entries)
- [load_regression_input_pairs](#load_regression_input_pairs) (4 entries)
- [load_regression_input_pairs.read](#load_regression_input_pairsread) (1 entry)
- [_check_score_count_pairing](#_check_score_count_pairing) (2 entries)
- [_usable_nuisance_columns](#_usable_nuisance_columns) (2 entries)
- [_report_exchangeability](#_report_exchangeability) (1 entry)
- [resolve_regression_src](#resolve_regression_src) (1 entry)
- [_run_guide_permutation_analysis](#_run_guide_permutation_analysis) (22 entries)
- [_perform_regression_set_paths](#_perform_regression_set_paths) (5 entries)
- [_next_results_folder](#_next_results_folder) (1 entry)
- [_annotate_level_coefficients](#_annotate_level_coefficients) (1 entry)
- [_level_control_rows](#_level_control_rows) (2 entries)
- [_call_level_hits](#_call_level_hits) (17 entries)
- [_diagnostic_screen_design](#_diagnostic_screen_design) (1 entry)
- [_write_regression_diagnostics](#_write_regression_diagnostics) (1 entry)
- [perform_regression](#perform_regression) (3 entries)
- [_perform_regression._perform_regression_read_data](#_perform_regression_perform_regression_read_data) (3 entries)
- [_perform_regression._count_variable_instances](#_perform_regression_count_variable_instances) (1 entry)
- [_perform_regression._qc_plot](#_perform_regression_qc_plot) (1 entry)
- [_perform_regression.grna_metricks](#_perform_regressiongrna_metricks) (5 entries)
- [_perform_regression.get_outlier_reference_values](#_perform_regressionget_outlier_reference_values) (2 entries)
- [_perform_regression.bootstrap_selection_frequencies](#_perform_regressionbootstrap_selection_frequencies) (5 entries)
- [_perform_regression](#_perform_regression) (72 entries)
- [_perform_regression._stack](#_perform_regression_stack) (2 entries)
- [_assign_prcfo_parts](#_assign_prcfo_parts) (1 entry)
- [process_reads](#process_reads) (9 entries)
- [beta_logit](#beta_logit) (1 entry)
- [clean_controls](#clean_controls) (1 entry)
- [process_scores](#process_scores) (9 entries)
- [generate_ml_scores](#generate_ml_scores) (11 entries)
- [_resolve_controls](#_resolve_controls) (2 entries)
- [ml_analysis._match_control_values](#ml_analysis_match_control_values) (3 entries)
- [ml_analysis](#ml_analysis) (43 entries)
- [_shap_explainers](#_shap_explainers) (3 entries)
- [shap_analysis](#shap_analysis) (2 entries)
- [write_plot](#write_plot) (1 entry)
- [find_optimal_threshold](#find_optimal_threshold) (1 entry)
- [_calculate_similarity](#_calculate_similarity) (5 entries)
- [_calculate_similarity.safe_similarity](#_calculate_similaritysafe_similarity) (1 entry)
- [_announce_the_bundle](#_announce_the_bundle) (1 entry)
- [_draw_the_cell_count_sweep](#_draw_the_cell_count_sweep) (1 entry)
- [_draw_importance_in_pyqtgraph](#_draw_importance_in_pyqtgraph) (1 entry)
- [interpret_vision_model](#interpret_vision_model) (16 entries)
- [interpret_vision_model.create_extended_radar_plot](#interpret_vision_modelcreate_extended_radar_plot) (1 entry)
- [interpret_vision_model.extract_compartment_channel](#interpret_vision_modelextract_compartment_channel) (4 entries)
- [interpret_vision_model.read_and_preprocess_data](#interpret_vision_modelread_and_preprocess_data) (9 entries)

## display

### lines 80-83

```python
def display(*args, **kwargs):
```

IPython may be mid-init (partially imported by another thread) — use a no-op fallback so importing this module never blocks. spaCR only calls display() from notebook contexts anyway; the Qt GUI ignores it.

## Module level

### line 132, trailing  _(unsure)_

```python
from .openmp_guard import single_threaded_openmp, guarded_n_jobs
```

see spacr/openmp_guard.py — duplicate libomp is fatal

### line 133, trailing  _(unsure)_

```python
from .plot import save_figure
```

every kept figure goes through the format/DPI preference

### lines 203-205

```python
from .figures.style import ROLES, figure_style, theme_target
```

THE HOUSE STYLE (136). `figures.style` imports matplotlib only inside its own functions, so naming it here costs nothing at import time.

### lines 208-211

```python
if not (sys.platform.startswith(('win', 'darwin')) or os.environ.get('DISPLAY')):
```

Only demote to Agg when there is genuinely nowhere to draw. Doing it unconditionally at import time silently killed inline plotting for anyone who imported spacr.ml in a notebook, because it overrode a backend the user had already selected. spacr.cli and both GUIs set their own backend.

### line 282, trailing  _(unsure)_

```python
'alpha': None,
```

'alpha' is the PENALTY for ridge/lasso, not a

### line 283

```python
}
```

probability, so it is deliberately not checked.

### lines 1957-1959

```python
'spline',
```

`spline` IS an OLS fit -- on a design with a spline basis over the covariates -- so its results object is the same one and its coefficients come out the same way.

### lines 2902-2904

```python
_ABSORBED_FIXED_EFFECTS = ('rowID', 'columnID')
```

THE ABSORBING BACKEND (instruction 141 G.1) -- pyfixest

### lines 3291-3293

```python
_GLUM_FAMILIES = {
```

THE FAST-GLM BACKEND (instruction 141 G.3) -- glum

### line 3309, trailing  _(unsure)_

```python
'glm': None,
```

chosen from the response, like the statsmodels path

### line 12501  _(unsure)_

```python
interperate_vision_model = interpret_vision_model
```

Backward compatibility for the misspelling published in earlier releases.

## _calibrated_fraction_threshold

### lines 501-502

```python
print(f"fraction-threshold calibration did not run: {exc}")
```

NAMED, NOT SWALLOWED. A user who ticked the box is owed the reason it did nothing, or they will believe it worked.

### lines 505-510

```python
chosen = result.get("chosen") if isinstance(result, dict) else None
```

`chosen` IS THE KEY THE SWEEP WRITES. It reported `threshold` here, which `sweep_fraction_threshold` has never returned -- so every screen that ticked the box was told the sweep preferred nothing, whatever it had actually measured, and went on using the number the settings gave. `threshold` IS a key, on each row of `candidates`; reading it off the result was reading a per-candidate name at the top level.

## _graph_sequencing_stats

### lines 526-530

```python
from .sequencing import graph_sequencing_stats
```

Keep this lazy to avoid expanding ml.py's already-heavy import graph, while giving callers and tests a stable dependency boundary. Importing the helper directly inside perform_regression made it impossible to substitute reliably after package lazy-loader tests replaced a module object in sys.modules.

## _keep_figures_with_the_run

### lines 609-610

```python
print(f"Could not keep {os.path.basename(path)} with the run: "
```

Advisory. A figure that could not be copied must not cost the run the threshold it just computed.

## QuasiBinomial.__init__

### lines 689-691

```python
self.variance = _DispersedVariance(self.__dict__['variance'], dispersion)
```

See _DispersedVariance: without this the method below is shadowed by the instance attribute statsmodels just installed, so the dispersion was silently ignored by every fit using this family.

## calculate_p_values

### lines 724-726

```python
y_true = np.asarray(y).ravel()
```

Coerce y and y_pred to 1D arrays before doing arithmetic so the subtraction does not try to broadcast a length-N array against a single-column DataFrame.

### lines 734-735  _(unsure)_

```python
return np.full(X.shape[1], np.nan)
```

More features than observations; this happens easily with screen-scale one-hot designs. Standard OLS-style p-values are undefined here.

### line 740  _(unsure)_

```python
XtX = X.T @ X
```

OLS-style standard errors of the coefficients.

## perform_mixed_model

### line 805  _(unsure)_

```python
if groups is None:
```

Ensure groups are defined correctly and check for multicollinearity

### lines 818-819

```python
raise ValueError(
```

Silent misalignment here would assign each row to the wrong cluster, which changes every standard error and nothing would look wrong.

### lines 824-826

```python
X_np = np.asarray(X, dtype=float)
```

Check for multicollinearity by calculating the VIF for each feature. variance_inflation_factor divides by (1 - R^2) and returns inf for a perfectly aliased column, so this doubles as the rank check below.

### lines 854-873

```python
try:
```

No variance components: this is the plain random-intercept model `MixedLM(y, X, groups=groups)` fits, which is `re_formula='1'`.

THE GPU IS SHARED, AND RUNNING OUT ON IT IS NOT A CRASH. Reported 2026-08-21: `CUDACachingAllocator ... memory allocation failed with OOM on device 0 while trying to allocate 2587885568 bytes (free: 2000093184, total: 25295519744)` -- a 25 GB card with 2 GB free, because something else on the machine had the rest.

`mixed_gpu._refuse_if_too_large` already checks free memory before building the DESIGN, and it cannot be enough on a shared device: it covers one allocation, the optimiser makes others, and the free figure it read can be stale by the time any of them run. A co-tenant that allocates between the check and the fit turns a correct check into a wrong one.

So the fit falls back to the CPU rather than failing. The same model, the same numbers, slower -- which is the trade the user would have made if asked, and asking is not possible from inside a worker thread twenty minutes into a run.

## centre_on_controls

### lines 1092-1095

```python
mask = None
```

THE GUIDE COLUMN OR THE GENE COLUMN, whichever this frame carries and whichever the control names. `nc` is read as a GENE when it is bare and as a GUIDE when it holds an underscore, which is the rule the rest of the module already applies to it.

## prepare_formula

### lines 1183-1191

```python
origin = ' - 1' if mode in ('zero', 'value') else ''
```

PATSY'S OWN SUPPRESSION. `- 1` removes the intercept column from the design; there is no other way to say it in a formula, and taking the column out of the design matrix afterwards would leave the formula describing a model that was not the one fitted.

'value' SUPPRESSES IT TOO, and that is what pins it. Fitting

`y - c ~ terms - 1` is fitting `y = c + terms`, so the intercept is exactly c; leaving the term in would estimate one NEAR c instead, which is not what asking for a number means.

### lines 1206-1209

```python
return f'{dependent_variable} ~ {term}{screen}{origin}'
```

OUT. The screen either has no plate-position effect to model -- a randomised layout -- or the caller is spending its 35 parameters somewhere else; see the measurement in this function's docstring for what that costs on a plate that does have one.

### lines 1212-1213

```python
return f'{dependent_variable} ~ {term}{screen}{origin}'
```

Row and column become variance components in fit_mixed_model, so they must not also be fixed terms here.

### lines 1215-1219

```python
return (f'{dependent_variable} ~ {term} + plateID + rowID + '
```

FIXED LAYOUT ADJUSTMENT. Plate is deliberately explicit rather than assumed to be represented by row/column: the same row and column labels recur on every plate, so rowID + columnID alone cannot absorb a shift in the overall mean between plates. Patsy emits no contrast column when there is only one plate and k-1 columns when there are k plates.

## fit_mixed_model

### lines 1359-1362

```python
guides_per_gene = df.groupby(gene_column, observed=True)[
```

THE NESTING HAS TO HAVE SOMETHING TO ESTIMATE. With one guide per gene everywhere, the guide variance component is confounded with the residual and MixedLM returns a boundary variance of zero for it - a number that looks like an answer and is not one.

### lines 1385-1388

```python
mixed_model = mixedlm_torch(formula, df, groups,
```

THE SAME CALL, one line apart. `mixedlm_torch` takes statsmodels' argument shape on purpose so the choice of who fits it cannot become a second code path with its own bugs; everything after this point reads the result the same way.

### lines 1397-1400

```python
raise
```

THE BACKEND'S OWN REFUSAL SURVIVES. Wrapped in the "MixedLM could not fit this frame" message below it would read as a problem with the screen, and the user would go looking at their data for a missing CUDA device.

### lines 1403-1406

```python
raise ValueError(
```

SAY WHAT COULD NOT BE EXPRESSED, rather than falling back to a model nobody asked for. The old plate-grouped model is not a substitute: it answers a different question, and substituting it silently is the class of failure this module is most careful about.

### lines 1420-1423

```python
fixed_names = set(map(str, mixed_model.fe_params.index))
```

FIXED EFFECTS AND VARIANCE COMPONENTS, kept apart by name. MixedLMResults.params is the fixed effects followed by the variance parameters; fe_params is the fixed half alone, so the difference is what says which is which without parsing ' Var' out of a string.

### lines 1430-1437

```python
parameter_p = np.where(
```

A VARIANCE COMPONENT'S WALD P VALUE IS NOT A TEST EITHER, and statsmodels reports one anyway (0.331 and 0.975 for the two components on the synthetic nesting; NaN for others, which is why the NaN alone cannot be relied on to mark them). The null it would test is sigma^2 = 0, which sits on the BOUNDARY of the parameter space, so the normal reference distribution the Wald statistic assumes does not hold and the number is not a probability of anything. Reported as NaN, with the variance itself in `coefficient` where it belongs.

### lines 1447-1449

```python
blups = {}
```

THE BLUPS, ONE PER GUIDE, WITH NO P VALUE. random_effects is {gene: Series}; each Series carries the group's own intercept under 'Group' and one entry per variance-component column.

### lines 1468-1471

```python
print(f"Mixed model fitted by regression_backend={backend_label(backend)}")
```

WHICH BACKEND PRODUCED IT, on every run and not only the fast one (instruction 141: "the run says which backend produced it"). Two runs of the same screen whose numbers differ in the 4th significant figure are explicable only if the log says which fitted them.

### lines 1478-1489

```python
if not bool(getattr(mixed_model, 'converged', True)):
```

A NON-CONVERGED MLE STILL RETURNS A COEFFICIENT AND A P VALUE, and nothing in statsmodels' return value says it should not be believed. Measured on the maintainer's TSG101 screen (389 genes, 823 guides, 610 wells): this fit does not converge inside twenty minutes, and a 50-gene subset converges to a gene-intercept variance on the boundary in 16 seconds. Both would have written results.csv in silence.

It WARNS rather than raising, because the maintainer chose 'mixed' as the default and a boundary variance component is a normal, informative outcome -- "the genes do not differ in intercept beyond what the gene fixed effect already explains" -- not a broken run. What is not acceptable is not being told.

## check_and_clean_data.check_collinearity

### line 1550  _(unsure)_

```python
df_encoded = df[columns]
```

Only include fraction and the dependent variable for collinearity check

### line 1553  _(unsure)_

```python
df_encoded = df_encoded.apply(pd.to_numeric, errors='coerce')
```

Ensure all data in df_encoded is numeric

### line 1556  _(unsure)_

```python
if np.linalg.matrix_rank(df_encoded.values) < df_encoded.shape[1]:
```

Check for perfect multicollinearity (i.e., rank deficiency)

### line 1561  _(unsure)_

```python
vif_data = pd.DataFrame()
```

Calculate VIF for each feature

### lines 1573-1580

```python
high_vif_columns = vif_data[vif_data["VIF"] > 10]["Feature"].tolist()
```

Report high VIF (> 10) but do NOT drop. The only columns checked here are 'fraction' and the dependent variable, and both are required downstream: 'gene_fraction' is derived from 'fraction' and the regression formula regresses the dependent variable on it. The previous revision dropped every column above the threshold, so any dependent variable even approximately proportional to 'fraction' (VIF -> inf) dropped both and made the caller die on KeyError: 'Column not found: fraction'.

## check_and_clean_data

### line 1589  _(unsure)_

```python
df = handle_missing_values(df, ['fraction', dependent_variable])
```

Step 1: Handle missing values in relevant fields

### line 1595  _(unsure)_

```python
df_cleaned = check_collinearity(df, ['fraction', dependent_variable])
```

Step 3: Check for multicollinearity in fraction and the dependent variable

### line 1598  _(unsure)_

```python
df_cleaned['gene'] = df['gene']
```

Ensure that the prc, plate, row, and column columns are still included for random effects

### lines 1606-1609

```python
if 'cell_count' in df.columns:
```

check_collinearity only returns 'fraction' and the dependent variable, so 'cell_count' used to be stripped unconditionally. regression() then found no 'cell_count' column and passed weights=None, which made the documented GLM-binomial var_weights=cell_count path dead code.

### lines 1613-1619

```python
from .schema import SCREEN_KEY
```

'screenID' is a DESIGN COLUMN when the frame holds more than one screen: regression() asks screen_is_blockable() of the CLEANED frame and patsy then builds the '+ screenID' term from that same frame. Stripping it here made both impossible at once -- the answer was always False, so two screens were pooled with nothing printed and no term in the model, which charges the difference between the experiments to whichever guides are over-represented in one of them.

### lines 1625-1637

```python
grna_key = ['prc', 'gene', 'grna']
```

'gene_fraction' is the share of the well's library that belongs to the gene: the sum of its gRNAs' fractions IN THAT WELL, counted once each.

The obvious spelling - groupby(['prc', 'gene'])['fraction'].sum() over the frame - is right only while the frame has exactly one row per (well, gRNA). With agg_type=None (which quantile regression forces, see get_perform_regression_default_settings) perform_regression deliberately joins the well's gRNAs against the well's CELLS, so every (well, gRNA) row appears once per cell and the sum came out multiplied by the well's cell count. Two consequences, both silent: every gene coefficient was divided by roughly that factor, and - because wells do not all hold the same number of cells - the inflation differed per well, so gene_fraction was no longer comparable across the plate.

### lines 1642-1645

```python
offenders = per_grna.loc[clash, grna_key].drop_duplicates()
```

One gRNA cannot hold two different shares of the same well's library. Deduplicating past this would pick whichever row sorted first and every gene coefficient downstream would rest on that coin flip.

## minimum_cell_simulation

### line 1696  _(unsure)_

```python
if isinstance(settings['score_data'], str):
```

Load and process data

### lines 1702-1703

```python
df = tabular.read_table(score_data)
```

ONE READER: canonical metadata names, one column per key and the `pplate1` repair, all decided in spacr.tabular rather than here.

### line 1715

```python
cell_counts = df.groupby('prc').size().reset_index(name='cell_count')
```

Compute the number of cells per well and select the top 100 wells by cell count

### line 1719

```python
df = df[df['prc'].isin(top_wells)]
```

Filter the data to include only the top 100 wells

### line 1722  _(unsure)_

```python
diff_data = []
```

Initialize storage for absolute difference data

### line 1725  _(unsure)_

```python
for i, (prc, group) in enumerate(df.groupby('prc')):
```

Group by wells and iterate over them

### lines 1727-1738

```python
original_mean = group[settings['dependent_variable']].mean()
```

`dependent_variable`, NOT `score_column`. The two named the same measurement -- settings.py defaulted one to the other and the tooltip said they must agree -- so instruction 135 A retired the duplicate. This function was its only regression-path reader and kept the old name, which killed every run here with KeyError: 'score_column', AFTER the settings had been canonicalised and before a single well was fitted.

`score_column` still exists and still means something ELSE: in interpret_vision_model below it names the CNN score column, default 'cv_predictions'. That is why this is three targeted edits and not a rename.

### line 1743  _(unsure)_

```python
for sample_size in sample_sizes:
```

Iterate over sample sizes and compute absolute difference

### line 1747  _(unsure)_

```python
for _ in range(num_repeats):
```

Perform multiple random samples to reduce noise

### line 1754  _(unsure)_

```python
avg_abs_diff = np.mean(abs_diffs)
```

Compute the average absolute difference across all repeats

### line 1757  _(unsure)_

```python
diff_data.append((sample_size, avg_abs_diff))
```

Store the result for plotting

### line 1760  _(unsure)_

```python
diff_df = pd.DataFrame(diff_data, columns=['sample_size', 'avg_abs_diff'])
```

Convert absolute difference data to DataFrame for plotting

### line 1763  _(unsure)_

```python
summary_df = diff_df.groupby('sample_size').agg(
```

Group by sample size to calculate mean and standard deviation

### line 1772  _(unsure)_

```python
if isinstance(settings['tolerance'], int):
```

Convert percentage to fraction

### line 1774, trailing

```python
tolerance_fraction = settings['tolerance'] / 100
```

Convert 2% to 0.02

### line 1780  _(unsure)_

```python
relative_thresholds = {
```

Compute the relative threshold for each well

### line 1786  _(unsure)_

```python
summary_df['relative_threshold'] = summary_df['sample_size'].map(
```

Detect the elbow point when mean absolute difference is below the relative threshold

### line 1788, trailing  _(unsure)_

```python
lambda size: np.mean([relative_thresholds[prc] for prc in top_wells])
```

Average across selected wells

### line 1793

```python
if not elbow_df.empty:
```

Select the first occurrence if it exists; otherwise, use the last point

### line 1795, trailing  _(unsure)_

```python
elbow_point = elbow_df.iloc[0]
```

First point where condition is met

### line 1797, trailing  _(unsure)_

```python
elbow_point = summary_df.iloc[-1]
```

Fallback to last point

### lines 1799-1806

```python
if dst is None:
```

THE SWEEP, IN PYQTGRAPH. A line through an ordered x with the spread it was summarised from behind it, and the chosen threshold marked `FastPlot.add_curve` draws exactly that, so the file and the tab are one scene.

WHERE THE FIGURE GOES. A `dst` the caller named is used as given; the fallback is the historical screen folder, and it is derived here rather than in the signature because it depends on `settings`.

## _bootstrap_wald_p_values

### lines 1897-1898  _(unsure)_

```python
one_class += 1
```

One-class resample: the estimator has no boundary to fit. Common on a screen with few positive wells, and not an error.

### lines 1916-1922

```python
dropped = int(n_boot) - len(draws)
```

Only "none of them" used to be reported, and that is the case where the numbers are least dangerous, because it raises. 199 of 200 failing gave a standard deviation taken over one draw — zero by construction — which makes every p-value exactly 1.0: a hit list that reads like a clean screen with no significant gRNAs in it, with nothing anywhere saying the inference did not happen. So say how many draws the p-values rest on whenever it is not all of them.

## _gene_of_design_column

### lines 2014-2016

```python
return _GUIDE_SUFFIX.sub('', identifier) or identifier
```

A gene whose guides carry no numeric suffix would collapse to the empty string, which is one group for every such guide in the screen; keep the id itself instead, which makes it its own single-guide gene.

## label_control_condition

### lines 2105-2109

```python
nc_name = '' if nc is None else str(nc)
```

Control names as TEXT. A gene id like 233460 is a perfectly good negative_control, and a settings file round-trips it back as the INT 233460 -- at which point `nc in row['feature']` raises "'in <string>' requires string as left operand, not int" and the whole regression dies on a value that was legal the moment it was typed into the GUI.

### lines 2114-2119

```python
from .control_names import rows_for
```

ONE MATCHER (184 C). `nc` and `pc` were matched as SUBSTRINGS of the model term, which is how `nc='23346'` claims `233460` AND `2334600` and the rows it steals are then reported as controls, which is worse than missing them. `spacr.control_names` reads a typed control as a gene or a guide by the same rule `process_reads` already applies to the data, and matches WHOLE values at that level.

### lines 2123-2128

```python
genes = guides.fillna('').str.split('_').str[0]
```

pandas 3 preserves missing values through ``astype(str)``.  When every coefficient is a continuous term (for example Intercept + fraction), every extracted guide is missing and ``str.split`` then produces an all-float intermediate on which a second ``.str`` access raises.  A missing guide means "no guide", so normalize it to empty text before splitting; it cannot match any nonblank control.

### lines 2137-2138

```python
for name, tag in ((pc_name, 'pc'), (nc_name, 'nc')):
```

PRECEDENCE UNCHANGED: nc over pc over the list, so a guide named twice is reported once and always the same way.

## process_model_coefficients

### lines 2201-2204

```python
coef_df = coef_df[coef_df['feature'].isin(
```

MixedLMResults.params appends the random-effect variance components ('Group Var', 'Group x ... Cov'). They are variances, not effects on the response, and their p-value is NaN - leaving them in put a row on the volcano plot that no gene owns.

### lines 2219-2225

```python
coefs = np.asarray(model.coef_).ravel()
```

LinearSVC has no likelihood, so there is no Wald test to run and calculate_p_values is meaningless here (its residual is the 0/1 misclassification, not a Gaussian error). The p-value reported is a BOOTSTRAP Wald: refit the same estimator on hinge_n_boot resamples of the wells, take the empirical standard deviation of each coefficient and compare the point estimate to it. It is a stability statistic, not a likelihood-ratio test, and the tooltip for hinge_n_boot says so.

### lines 2247-2262

```python
coef_df['condition'] = label_control_condition(
```

ONE LABELLER, shared with the guide-permutation path. It grew a second copy the moment the permutation table needed a `condition` column too, and two copies of "what counts as a control" is how the run and the panel come to disagree about which coefficients the cut is measured on. LOUD, NOT FATAL (184 D). A control that matches nothing is not "no controls" -- it is every normalisation, every volcano baseline and the whole effect-size cut computed against an empty set, while the run finishes and the figures draw. So it has to be said.

IT MUST NOT RAISE, AND THE INSTRUCTION SAID IT SHOULD. spaCR SHIPS nc='233460' and pc='220950' -- Toxoplasma gene ids -- so raising would make every screen that is not this one fail on a value the user never typed. "Error, not a silent zero" is right about the silence and wrong about the exception: the fix for silence is a sentence nobody can miss. `strict=True` remains available for a caller that knows the control was chosen rather than defaulted.

### lines 2268-2270

```python
nuisance = coef_df['feature'].astype(str).str.match(
```

Layout terms are nuisance adjustments, not screen hits. Match only the Patsy term prefix so a legitimate guide whose name happens to contain "plate", "row", or "column" is not discarded.

## _show_response_distribution

### lines 2332-2334

```python
wanted = [str(dependent_variable),
```

``process_scores`` can rename the response, while ``before_df`` retains its original name. Prefer the requested names and otherwise use the final numeric response column in the aggregated table.

### lines 2346-2355

```python
_draw_response_panel_in_pyqtgraph(
```

PYQTGRAPH, so the panel in the tab and the panel in the run folder are one scene. `fast_panel` overlays the two distributions as outlines on one pair of axes -- two shapes on separate axes with separate scales is the one layout that cannot answer whether the transform moved the shape.

The matplotlib version this replaced was wrapped in

`figure_style(theme_target())`, which is how a matplotlib artist takes the theme. A pyqtgraph scene takes it from the palette when it is built, so there is nothing left for that context to do.

### lines 2361-2362

```python
print(f"the response distribution panel could not be drawn "
```

A diagnostic figure must not invalidate the regression run, but a rendering failure remains visible in the run log.

## check_distribution

### line 2378  _(unsure)_

```python
if np.all((y == 0) | (y == 1)):
```

Check if the dependent variable is binary (only 0 and 1)

### line 2383  _(unsure)_

```python
elif (y > 0).all() and (y < 1).all():
```

Continuous data between 0 and 1 (excluding exact 0 and 1)

### line 2385  _(unsure)_

```python
if np.any((y < epsilon) | (y > 1 - epsilon)):
```

Check if the data is close to 0 or 1 (boundary issues)

### line 2393  _(unsure)_

```python
elif (y >= 0).all() and (y <= 1).all():
```

Continuous data between 0 and 1 (including exact 0 or 1)

### line 2398  _(unsure)_

```python
stat, p_value = stats.normaltest(y)  # D’Agostino and Pearson’s test for normality
```

Check if the data is normally distributed for OLS suitability

### line 2399, trailing  _(unsure)_

```python
stat, p_value = stats.normaltest(y)
```

D’Agostino and Pearson’s test for normality

### line 2406  _(unsure)_

```python
if stats.kstest(y, 'beta', args=(2, 2)).pvalue > 0.05:
```

Check if the data fits a Beta distribution

### line 2408  _(unsure)_

```python
if np.any((y < epsilon) | (y > 1 - epsilon)):
```

Check if the data is close to 0 or 1 (boundary issues)

## double_transform_warning

### lines 2516-2528

```python
return (
```

WHAT THIS SENTENCE HAS TO CARRY, and it lost three of them once:

that the response is transformed TWICE -- the fault, in the word a reader will remember it by; the composed function, so it can be checked; THE SYMPTOM. A user does not notice a double transform; they notice a pseudo-R-squared of -20.3 and have no way to connect the two. Naming McFadden here is what connects them, and it is the whole point of the warning existing at all; what to do about it. spaCR now resolves this itself -- the response is fitted as measured and the family's link does the transforming once -- so the warning explains what was avoided rather than offering a choice that no longer exists.

## resolve_glm_transform_conflict

### lines 2541-2560

```python
def resolve_glm_transform_conflict(dependent_variable, transform='',
```

A link-like transform combined with a non-identity family link transforms the response twice.  The two valid resolutions answer different questions, so the caller selects the response scale explicitly.

A log-transformed response handed to a family with a logit link fits logit(log(y)), which nothing measures. There are two defensible fixes and they are different science, so spaCR offers both rather than choosing:

'untransformed'  choose the family on the measured response and let the link do the transforming. `pred` is a proportion, so Binomial/Logit is right for it and the log is redundant.

'transformed'    keep the transform and fit an identity link. log(p) is an ordinary continuous response and a Gaussian model of it is a standard thing to fit.

'warn'           what spaCR did before this setting existed: fit the transformed response, choose the family from it, and print the warning. Retained for reproducibility of earlier runs; new analyses should choose one of the two explicit scales.

### line 2599

```python
prefix = f"{kind}_"
```

Fit the response as measured and let the family's link do the work.

## pick_glm_family_and_link

### lines 2630-2632

```python
warning = double_transform_warning(name, transform, family)
```

AFTER the choice, because the warning depends on which link was picked and BEFORE the fit, because by the time this reaches the summary the fit has already run on a doubly transformed response.

## _choose_glm_family

### lines 2651-2659

```python
print(f"{scale} is strictly between 0 and 1. Using Binomial family "
```

A proportion strictly inside (0, 1) is a binomial mean, and a binomial GLM with a logit link is the standard model for it. This branch used to raise "Use BetaModel for this data; GLM is not applicable", which was not a principled refusal: the very next branch fits exactly this family as soon as a single well sits at 0.0 or 1.0, so one boundary well flipped the same screen from "not applicable" to "fine". Beta regression IS usually the better model here - hence the recommendation - but it is a recommendation, and regression_type='beta' is how you take it.

### lines 2672-2673  _(unsure)_

```python
_validate_poisson_response(values, minimum_samples=1)
```

Family selection may be used for a short preview without fitting. The actual GLM boundary below enforces the sample/design minimum.

## binarise_response

### lines 2700-2709

```python
def binarise_response(y, threshold=None, name='response'):
```

THE BACKEND TABLES LIVE IN A MODULE THAT IMPORTS NOTHING.

`get_setting_dependencies` reads REGRESSION_SETTINGS_USED to decide which widgets on a settings panel apply to each other, and importing it from here dragged this module's `from .plot import save_figure` -- and so torch, cv2 and IPython -- onto the GUI thread every time a panel was built: 2.2s and 900 MB to look up a dict of strings.

Re-exported rather than moved-and-forgotten, so every existing `from spacr.ml import REGRESSION_TYPES` keeps working.

### lines 2751-2754

```python
if _left_blank(threshold):
```

A BLANK BOX IS NO CUT, not a cut spelled ''. Both callers can be reached from the panel, and `float('')` is not an error anybody can act on. Cut here as well as in `regression_model` because the QC path (`_write_regression_qc`) comes in by the other door.

## _left_blank

### line 2812, trailing  _(unsure)_

```python
return bool(value != value)
```

NaN is the only value unequal to itself

## _absorbed_factor_codes

### lines 2951-2953

```python
continue
```

A factor with ONE level emits no columns at all -- patsy folds it into the intercept. A screen on a single plate row is the documented case (see prepare_formula), and it is not an error.

## _fit_absorbed_least_squares

### lines 3199-3202

```python
p_full = len(keep) + n_absorbed_params
```

Degrees of freedom are a property of the design, not of pyfixest's projection.  Check them before importing the optional backend so an invalid request gets the same useful diagnosis on Python 3.9, where pyfixest itself is unavailable.

### lines 3212-3213

```python
from pyfixest.core.demean import demean
```

Import only after backend-independent validation. pyfixest requires Python >=3.10, while spaCR's supported floor is Python 3.9.

### lines 3218-3221

```python
demeaned, converged = demean(stacked, codes, w, tol=1e-10)
```

tol is on the alternating projections, not on the answer: 1e-10 is tighter than the 1e-6 pyfixest defaults to, because the agreement this backend is held to (instruction 141 D) is against statsmodels' exact solve rather than against another approximation.

### lines 3233-3235

```python
Xw = X_d * w[:, None]
```

WEIGHTED normal equations. `demean` already takes weighted group means, which is the weighted Frisch-Waugh-Lovell projection, so the only thing left is to weight the cross-products.

### lines 3239-3243

```python
_rank = int(np.linalg.matrix_rank(xtx))
```

RANK BEFORE SOLVE, not the solver's exception. LAPACK builds disagree about a singular system: some raise, and some return one arbitrary member of an infinite solution set. Diagnosing rank first makes the refusal the same everywhere, which matters because the alternative is a coefficient table that looks fine and is not identified.

### lines 3256-3259

```python
rss = float(resid @ (resid * w))
```

DEGREES OF FREEDOM ARE CHARGED FOR WHAT WAS ABSORBED. n - p_kept alone would report the standard errors of a model that never had the 36 nuisance parameters, which is smaller than the truth and is exactly the way an absorbing fit gets its inference wrong.

### lines 3270-3273

```python
full_resid = resid
```

The residual of the DEMEANED regression is the residual of the full one -- that is what Frisch-Waugh-Lovell says -- so the fitted values follow from it and the diagnostics see the same numbers statsmodels would have shown them.

## _GlumResults.__init__

### lines 3376-3381

```python
self.llnull = None if llnull is None else float(llnull)
```

THE NULL LOG-LIKELIHOOD, because that is what the goodness-of-fit line divides by. `fit_quality_note` takes `llnull` and falls back to `null_deviance / -2` -- so a backend that carried only the deviance would print a DIFFERENT McFadden from statsmodels for the identical fit, which is the one thing this class exists not to do. The null model is fitted here anyway; this only carries its answer.

## _fit_glum_glm

### lines 3502-3506

```python
n_total = None
```

THE SAME OFFSET THE STATSMODELS BRANCH USES. Without it the coefficients are effects on the well's headcount rather than on the per-cell rate -- see `_poisson_offset` for the simulation that measured what that costs -- and a backend that dropped it would be answering a different question, not answering the same one faster.

### lines 3541-3543

```python
from glum import GeneralizedLinearRegressor
```

Keep validation independent of the optional solver. glum requires Python >=3.10, so Python 3.9 callers must still receive spaCR's precise input error instead of an unrelated ModuleNotFoundError.

### lines 3546-3549

```python
estimator = GeneralizedLinearRegressor(
```

alpha=0 is the UNPENALISED fit, which is the only one that can agree with statsmodels. fit_intercept=False because patsy already put an 'Intercept' column in the design and a second one would be collinear with it.

### lines 3567-3568  _(unsure)_

```python
scale = float(np.sum(info_w * (y_flat - mu) ** 2)) / (n - len(beta))
```

A Gaussian GLM has a FREE dispersion, and statsmodels estimates it as the Pearson chi-square over the residual degrees of freedom.

### lines 3585-3589

```python
null_kwargs = {'family': family}
```

THE NULL MODEL IS FITTED, not approximated, because `regression_model` prints McFadden's R2 for 'glm' and 'poisson' off `null_deviance` and a backend that changed that number would have changed a number a reader compares between runs. It is an intercept-only GLM: one column, so it costs nothing next to the fit above.

## regression_model

### lines 3743-3748

```python
if _left_blank(cov_type):
```

AN EMPTY COVARIANCE BOX IS NO COVARIANCE ESTIMATOR, not an estimator named ''. The panel's line edit and a saved settings CSV both write `''` for a box nobody typed in, and three of the branches below pass it on when it "is not None" -- so a logit, probit or quasi_binomial fit from the screen's own saved settings died inside statsmodels with "cov_type not recognized", naming a value the user never chose.

### lines 3751-3755

```python
if _left_blank(hinge_threshold):
```

AND AN EMPTY THRESHOLD BOX IS NO THRESHOLD. `hinge` READS hinge_threshold, so nothing refused the blank: it reached `binarise_response`, which asked float('') for a number and died with "could not convert string to float: ''" -- a message with neither the setting's name nor the model's in it.

### lines 3760-3762

```python
'alpha': 1.0 if use_auto_alpha else alpha,
```

'auto' and None mean "no penalty chosen, cross-validate it", which is not a value an unpenalised model is being asked to honour, so they count as the default here rather than as a request.

### lines 3775-3777

```python
backend = _require_backend(regression_type, regression_backend)
```

WHO fits it, checked before WHAT is fitted is dispatched. A backend that cannot fit this family, is not installed, or wants a GPU that is not here fails now rather than after the design has been built.

### lines 4260-4264

```python
'quasi_binomial': lambda: _glm_binomial(link=sm.families.links.Logit(),
```

Quasi-binomial is a binomial mean with a free dispersion. statsmodels spells that as scale='X2' (dispersion from the Pearson chi-square) on a Binomial family, which is what widens the standard errors; the QuasiBinomial family above takes a dispersion the caller already knows and is not what an overdispersed screen needs.

### lines 4268-4269  _(unsure)_

```python
'logit':  lambda: _glm_binomial(link=sm.families.links.Logit()),
```

logit and probit on a CONTINUOUS fraction y are routed through GLM-Binomial with var_weights = cell_count. sm.Logit / sm.Probit require binary y.

### lines 4288-4292

```python
if backend == 'pyfixest':
```

THE ALTERNATIVE FITTERS COME BEFORE THE DEFAULT MAP, and each one is held to instruction 141 D: it fits the SAME model and reports the same numbers, or it is not offered. What each one may be chosen for is policed by `backend_status` above, so an unroutable pairing has already been refused by name and this is only the dispatch.

### lines 4295-4301

```python
raise ValueError(
```

A SANDWICH IS NOT ABSORBED FOR FREE. HC0's meat is

X~' diag(e^2) X~ and does survive Frisch-Waugh-Lovell, but HC1/HC2/HC3 correct by the FULL model's leverage, which an absorbed fit never forms -- so three of the four spaCR offers would come out different from the statsmodels number under the same label. Instruction 141 D calls that a bug, so it is refused instead of approximated.

### lines 4312-4314

```python
raise ValueError(
```

The same refusal `_wls` makes, made here too: WLS with unit weights IS OLS, and a run labelled 'wls' that fitted OLS is the silent mislabelling that branch exists to prevent.

### lines 4326-4339

```python
if 'nothing to absorb' not in str(nothing_to_absorb):
```

NOTHING TO ABSORB IS NOT A FAILURE, and this used to end the run 20 seconds in. Reported from a live fit on 2026-08-20: `model_plate_position=False` takes rowID and columnID out of the design, and the absorbing backend then refuses because it has no fixed effects to project out.

ITS OWN REFUSAL SAYS WHY THAT IS THE WRONG ANSWER: "the fit would be the statsmodels fit with an extra projection in front of it". With nothing to absorb the two backends compute the SAME numbers, so falling back is not substituting a different method -- it is the identical fit by the only route left. That is what makes this fallback safe where the montage's multivariate one was not: there, the alternative answered a different question.

### lines 4375-4379

```python
if use_auto_alpha:
```

Every coefficient shrunk to exactly zero is not a finding, it is a penalty set too high for the scale of this design - and it reaches the user as "0 significant gRNAs", which is indistinguishable from a screen with no hits. The default alpha=1 does this to a fraction-scale design every time.

### lines 4381-4386

```python
raise ValueError(
```

The penalty was not mis-set, it was CHOSEN: cross-validation preferred the empty model to every non-empty one it tried, so no gRNA predicted the held-out wells better than the mean did. Telling this user to "set alpha to 'auto'" - which the old message did, unconditionally - is telling them to do the thing they just did.

## regression_model._glm_auto

### lines 3856-3861

```python
fit_y = y
```

WHICH SCALE THE FAMILY IS CHOSEN ON (instruction 182). The response itself is swapped by the CALLER -- see `regression` -- because everything downstream of the fit (the coefficient table, McFadden, the residual panels) reads the same `y` and a model fitted on a different one would silently disagree with all of it. All that is left here is the link.

### line 3864

```python
family = sm.families.Gaussian(link=sm.families.links.Identity())
```

The transform IS the link, so the family must not add another.

### lines 3874-3876

```python
return sm.GLM(fit_y, X, family=family, offset=_poisson_offset()).fit(
```

Same exposure the explicit 'poisson' branch uses. A family chosen BY the data must be fitted the same way as one chosen by name, or 'glm' and 'poisson' silently disagree on the same response.

### lines 3881-3887

```python
kwargs['var_weights'] = np.asarray(weights).ravel()
```

A per-well fraction estimated from 30 cells and one estimated from 400 carry very different amounts of information, and the binomial variance function only knows that if it is told. Weight them exactly as the explicit 'logit'/'probit' branches do, and ONLY for the binomial families: var_weights on the Poisson or Gaussian branch would be re-weighting a response that already has the right variance.

## regression_model._glm_poisson

### lines 3896-3899

```python
return sm.GLM(y, X, family=family, offset=_poisson_offset()).fit(
```

offset(log(cell_count)) turns the fit from "how many positive objects are in this well" into "what fraction of this well's cells are positive", which is the quantity the screen is about. See _poisson_offset for the measurement that made this non-optional.

## regression_model._wls

### lines 3905-3908

```python
if weights is None:
```

WLS with unit weights IS OLS. Saying so is the point: a user who picks 'wls' on a table with no cell_count column would otherwise get an OLS fit labelled 'wls' in the results folder name, the volcano filename and the settings CSV, and nothing anywhere would disagree.

## regression_model._rlm

### lines 3926-3927

```python
return sm.RLM(y, X, M=sm.robust.norms.HuberT(t=huber_t)).fit()
```

HuberT's t is in units of the ESTIMATED residual scale (MAD), not of y, so the same t means the same thing whatever the response units.

## regression_model._hinge

### lines 3942-3946

```python
if use_auto_alpha:
```

LinearSVC minimises C * sum(hinge) + 0.5 * ||w||^2, so its C is the INVERSE of a regularisation strength. Mapping alpha -> 1/alpha keeps "larger alpha shrinks harder" true across every penalised backend; without it, alpha would mean the opposite here than it does for lasso and ridge, on the same settings key.

### lines 3948-3953

```python
return _find_best_hinge_alpha(y_binary)
```

alpha='auto' means "choose the penalty by cross-validation" for lasso, ridge and elasticnet, and it has to mean the same thing here. It used to mean C = 1: 'auto' and alpha=1.0 produced byte-identical coefficients, so a user who asked for a cross-validated margin got an arbitrary fixed one under a label that says otherwise.

## regression_model._find_best_hinge_alpha

### lines 4015-4018

```python
best = float(strengths[len(strengths) - 1
```

Ties go to the STRONGER penalty (the larger alpha): among margins that separate the held-out wells equally well, the one that shrinks hardest is the one that generalises, and argmax on a raw list would instead take the weakest.

## regression_model._group_lasso

### lines 4053-4056

```python
gene_terms = _level_term_mask(columns)
```

WHICH COLUMNS THE ANSWER IS ABOUT, computed before the fit because the cross-validation below needs it too: a penalty that leaves two row dummies standing and not one gene has selected nothing this module can report.

### lines 4058-4066

```python
if _left_blank(group_lasso_lambda) or (
```

'auto' CROSS-VALIDATES THE PENALTY, and it is what the panel posts for this backend. A penalty is only large or small relative to the design it is applied to: the shipped 0.05 is nearly half of the tsg101 screen's own ceiling of 0.1285, so every one of its 297 gene blocks came back exactly zero and the run was refused -- from settings in which nobody had touched the penalty (236 C7).

ANNOUNCED, never quiet. A penalty chosen for the user and not named is one they cannot put in a methods section.

### lines 4083-4092

```python
if not gene_terms.any():
```

THE REFUSAL IS ABOUT THE GENE BLOCKS, not about the design as a whole. `np.any(beta)` is not the test: the row and column dummies are singleton groups with far larger correlations than any guide block, so they survive a penalty that has already emptied every gene -- measured on the 384-well synthetic screen, lambda=0.02 leaves two row terms standing and not one gene. The user is then handed a fit whose every gRNA coefficient is zero, which reads downstream as "0 significant gRNAs" and is indistinguishable from a screen with no hits. That is the same failure the lasso branch below refuses, and it has to be refused on the same grounds.

### lines 4104-4108

```python
ceiling = group_lasso_module.max_lambda(design, y_flat, blocks)
```

max_lambda is the smallest penalty that zeroes EVERY group, so it is an upper bound rather than the working value; on the same fixture it is 0.384 and the planted gene is recovered at 0.001. The message names it because a scale is what the user is missing, and "lower it" alone gives them none.

## regression_model._rra

### lines 4158-4165

```python
centred = design - design.mean(axis=0)
```

THE PER-GUIDE SCORE IS THE MARGINAL SLOPE -- the least-squares slope of the response on that guide's column ALONE, one parameter at a time. It is not the joint fit's coefficient, and that is the whole point of offering RRA: with 823 guides and 610 wells the joint fit is undefined, and every backend that forms one is answering a question the data cannot support (instruction 133). A marginal slope exists at any width and is the direct analogue of MAGeCK's per-guide log fold change, which is what alpha-RRA ranks.

### lines 4173-4176

```python
ranked = np.where(moving, slopes, np.nan)
```

A CONSTANT COLUMN IS NOT RANKED. The intercept explains no variation in the response, so it has no slope to rank; NaN is what rank_aggregate drops, and dropping it is right because ranking it would give it a rank it did not earn and shift every real guide's.

### lines 4188-4194

```python
two_sided = np.minimum(1.0, 2.0 * np.minimum(
```

BOTH TAILS, COMBINED THE STANDARD WAY. rank_aggregate reports depletion and enrichment separately because they are two questions; the coefficient table has one p_value column, so the two one-sided permutation P values become the two-sided min(1, 2 * min(p_neg, p_pos)). Taking the smaller tail WITHOUT doubling would be a one-sided test chosen after seeing which way the gene went, which is the classic way to halve a P value for free.

### lines 4199-4205

```python
p_values = np.array(
```

ONE ROW PER DESIGN COLUMN, exactly as every other backend produces, and the mapping is: the column keeps its OWN marginal slope as the coefficient and carries its GENE's aggregated P value. RRA tests genes, not guides, so a gene's guides share its P value; at level='grna' that makes the BH family the guide count rather than the gene count, which is conservative, and the level='gene' fit is the one whose family matches what was tested.

## _fit_horseshoe_poisson

### lines 4477-4480

```python
constant = [name for name, column in zip(columns, design.T)
```

A constant column - patsy's Intercept - is confounded with the model's own intercept term, which power_model fits separately. Naming it here is what makes power_model return NaN for it rather than a shrunk-to-zero coefficient that reads as "this term was tested and found null".

### lines 4492-4499

```python
fit = fit_model(model_data, seed=0, standardize=True)
```

standardize=True, unlike power_model's own default. The horseshoe's global scale is calibrated for spaCRPower's log10 read fraction, which has a spread of about 1; spaCR's design is gRNA FRACTIONS, whose columns have standard deviations around 0.05, and on that scale the prior shrinks every coefficient to ~1e-4 and separates nothing. Scaling each column to unit SD is what makes the shrinkage comparable across terms which is the entire point of the model - at the cost that beta is then "per standard deviation of that gRNA's fraction", not per unit.

## _HorseshoeResults.__init__

### lines 4537-4539

```python
raise ValueError(
```

power_model is a separate module with its own release cadence; a renamed column must stop the run here, where it can be named, rather than surface as a KeyError from inside pandas.

## _reconcile_random_row_column_effects

### lines 4725-4734

```python
if not settings.get('model_plate_position', True):
```

OUT PLUS RANDOM IS NOT A STATE (instruction 143 A). Plate position has three: out of the model (model_plate_position=False), in as fixed effects (True), in as variance components (True plus this flag). Asking for variance components on terms that are not in the model is a fourth, and it is refused here -- before a folder is named or a file is written rather than resolved to whichever of the two the reader guesses, because both guesses fit a DIFFERENT model from the one asked for and neither would say so. Same seam, same voice, as the model conflict below; prepare_formula refuses the same pair for a caller that never goes through a settings dict.

### lines 4755-4757

```python
_reject_unused_settings('mixed', {
```

The mixed branch reads none of the per-model knobs, so any of them set away from its default is a request nothing will honour. Same seam, same message shape, as the fixed-effects path.

## _write_regression_qc

### lines 4851-4852

```python
metadata = None
```

Per-well labels: what turns "well 41 is an outlier" into a plate, a row and a column somebody can go back to the microscope with.

### lines 4861-4866

```python
if len(metadata) != len(X):
```

`.loc` with a duplicated index does not raise: it returns the cross product, so a frame whose labels repeat comes back with n*n rows. regression_qc_report would then refuse the whole report -- losing every panel, including the variance-homogeneity one -- over metadata that is only ever used for LABELS. Losing the labels is the proportionate answer.

### lines 4877-4882

```python
try:
```

NO `fmt`. The panels are figures the user keeps, so they follow the format preference -- and `regression_qc_report` reads it itself now, so resolving it here as well would be two places deciding one thing. An explicit `fmt` is a caller FORCING a format, which this caller is not doing; passing the preference under that name made a preference indistinguishable from an override.

### lines 4889-4892

```python
print(f"Regression QC report could not be written: "
```

A diagnostic that fails must never destroy a fit that already succeeded and cost an hour. The report itself already downgrades a failing panel to FAILED; this catches the rarer case where the report as a whole cannot be built.

## _wide_fixed_effect_design

### lines 5007-5009

```python
return y, X, wide
```

Predictor terms are namespaced and categorical terms are column-scoped. Ten thousand generated designs confirmed that those sets cannot collide; the former unreachable duplicate-column guard therefore added no safety.

## regression

### lines 5100-5107

```python
level_dst = dst if level_dst is None else level_dst
```

create_volcano_filename names a quantile run by the quantile it fitted rather than by the model name, because two quantiles of the same screen are two different results that must not overwrite each other. That used to be alpha, which is no longer the quantile. Per-level figures go where the caller says. A single-level run keeps writing straight into dst, which is every existing path; a two-fit run gets one subfolder per level so the second fit's regression_figure.pdf does not land on top of the first fit's.

### lines 5117-5119

```python
wanted = resolve_levels(regression_type, level)
```

ONE LEVEL PER FIT, and `mixed` decides its own. 'both' is refused rather than quietly resolved to one of them: a caller that asked for two fits and silently got one would read the guide table as if it were both.

### lines 5136-5140

```python
dependent_variable, transform, glm_force_identity, conflict_note = (
```

WHICH SCALE THE GLM FITS (instruction 182), decided BEFORE the design matrices are built so the fit, the coefficient table, McFadden and the residual panels all read the same response. Both ways out of the double transform are offered because both are defensible and they are different science; spaCR does not choose between them.

### lines 5151-5154

```python
intercept_offset = 0.0
```

WHAT THE INTERCEPT IS, decided before the design is built so the fit, the coefficient table and every panel read one response. 'control' shifts the response; 'zero' takes the term out of the formula below; 'fitted' does neither and is what every run did before this existed.

### lines 5168-5171

```python
intercept_offset = float(intercept_value or 0.0)
```

PINNED, NOT NUDGED. Shifting the response by the number and suppressing the term fits `y = c + terms`, so the intercept is exactly what was asked for -- an estimated one would land near it and read as though the number had been a suggestion.

### lines 5181-5184

```python
qc_design = None
```

The QC report needs the design that was fitted. The mixed branch below never builds one -- fit_mixed_model takes the formula and the frame and keeps its design to itself -- so X and y simply do not exist there, and this stays None to say so rather than letting a NameError find out.

### lines 5187-5199

```python
block_screen = screen_is_blockable(df)
```

INSTRUCTION 122: BLOCK ON THE SCREEN WHEN THERE IS MORE THAN ONE.

Two screens sharing a guide library are stacked into one frame and fitted together -- twice the wells -- but only if the screen is in the model. A systematic difference between two experiments that is not a term gets charged to whichever guides are over-represented in one of them, which is a false hit that looks exactly like a real one.

Decided from the DATA, not from a setting, because the wrong answer is silent in both directions: a constant screenID term makes the design rank-deficient and statsmodels answers with a pseudo-inverse instead of refusing, so a single-screen run would come back with standard errors that mean nothing and no error anywhere.

### lines 5205-5210

```python
if regression_type == 'mixed' or random_row_column_effects:
```

THE MIXED BRANCH IS THE NESTED MODEL, and it is reached by NAME as well as by the random row/column flag. `regression_type='mixed'` used to fall through to the fixed-effects branch and be fitted by regression_model with groups=plateID, which is a different model from the one fit_mixed_model built -- two things called 'mixed' in one function. There is one now: gene fixed, guide random nested inside it.

### lines 5246-5248

```python
model_index = y.index
```

Rows patsy actually kept. Every per-row vector handed to the model below - weights, groups, exposure - is taken through this index, so a row patsy dropped (a NaN predictor) cannot shift the rest by one.

### lines 5251-5261

```python
if draw_shared_panels and not _show_well_distributions(
```

THE HOUSE-STYLE DISTRIBUTIONS, not the old ones. spacr.figures. distributions draws the same two panels -- the guide fractions and the response -- in the one visual system, and writes the same file names, so the grid, the queue and the tests still find them.

Falls back to the old plot_histogram if the new module cannot draw them, because a figure is not worth losing a fit over.

DRAWN ONCE PER RUN, NOT ONCE PER FIT. They describe the data, which is the same data both levels are fitted to, so a two-fit run that drew them twice would put two identical panels on the figure grid.

### lines 5267-5281

```python
print('Data will not be scaled: the design is fractions and dummies '
```

No scaling, for any type. The design this pipeline builds is one level's fraction terms plus row and column dummies: dummies and fractions, already on one common [0, 1] scale, so there is nothing for a scaler to put on a common footing. What MinMax scaling DID do was divide each gRNA's column by that gRNA's own maximum fraction, which rescales its coefficient by a different constant per feature and the volcano plot then ranks gRNAs against each other on those coefficients. It also zeroed the intercept column outright (see scale_variables), fitting every unscaled-exempt model through the origin. The exemption list this replaces named lasso and ridge as "already 0/1 from one-hot categorical predictors", which is the right reason - it is just as true of every other type here.

scale_variables stays public and correct for callers that scale their own designs; this pipeline no longer needs it.

### lines 5286-5287  _(unsure)_

```python
weights = (fit_df['cell_count'].loc[model_index]
```

Per-well cell counts: var_weights for the binomial links, the WLS weights, and the Poisson exposure for the horseshoe model.

### lines 5290-5292

```python
groups = None
```

`mixed` never reaches here any more -- it is caught by name at the top and fitted by fit_mixed_model with the gene/guide nesting -- so there is no grouping vector to build on this branch.

### lines 5315-5319

```python
response_name=str(y.name) if hasattr(y, 'name') else '',
```

WHAT THE RESPONSE IS CALLED AND WHAT WAS DONE TO IT (182 A/C). The family sniffer sees values, not a column, so without these it said "Data strictly between 0 and 1" about a logged proportion and a reader could not tell which scale it had looked at.

### lines 5331-5336

```python
if plot and legacy_volcano:
```

THE OLD VOLCANO IS OFF UNLESS ASKED FOR. "your new volcano plot is much much faster than my old one so hide my old version behid a boolean that defaults to off". It is not deleted -- it is what published figures were made with -- but a run does not draw it now, and a run that does draw it produces two volcanoes in two visual idioms, which is the thing that made the grid look wrong.

### lines 5338-5344

```python
volcano_plot(
```

plot.volcano_plot is keyword-only past its first argument and has no defaults for the two column names, so the old positional volcano_plot(coef_df, volcano_path) raised TypeError on every plot=True call. coef_df is the frame built by process_model_coefficients / fit_mixed_model, whose columns are feature / coefficient / p_value; the coefficients are already on a signed log-odds-style scale, so no x transform is applied.

### lines 5355-5357

```python
qc_manifest = None
```

After the volcano, so the report can name a file that is already on disk. Skipped without a destination: regression_qc_report raises on a falsy dst on purpose, and a fit run with dst=None has nowhere to put diagnostics.

### lines 5360-5364

```python
qc_manifest = _write_regression_qc(
```

KEPT, not just written. Instruction 115: the manifest holds the per-panel VERDICT and the renderer that drew each panel, which is the thing a caller most wants out of a run -- and until now it went to disk and nowhere else, so `perform_regression`'s own return value could not say whether the fit it just handed back was diagnosable.

### lines 5370-5380

```python
if level_dst:
```

THE HOUSE-STYLE PANELS. Asked for on 2026-08-16: "the all figures section should look like a publication ready figure ... with each panel having an uppercase letter ... and be on a grid", and "there are no additional plots that i asked for and all the old plotts look exactly the same".

SHOWN, not merely written. A PDF on disk changed nothing about what the application displays, which is exactly the complaint -- the grid still held the same old pictures. These go through plt.show(), which the Qt bridge intercepts, so each panel arrives in the figure queue and lands on the grid as its own lettered cell.

### lines 5385-5389

```python
try:
```

THE PARAGRAPH. "id also like a little written summary at the end in the console saying what is significant and so on". Printed last so it is the thing left on screen when a run finishes, and built from the same numbers the panels are -- a summary that recomputed them could disagree with the pictures beside it and a reader could not tell which was wrong.

### lines 5402-5404

```python
coef_df = coef_df.copy()
```

WHICH MODEL PRODUCED THIS ROW, carried on the table itself. Two fits write two tables and the volcano chooses between them; a row that cannot say which family it belongs to is a row whose q value cannot be read.

### lines 5407-5411

```python
if qc_manifest is not None and coef_df is not None:
```

THE MANIFEST RIDES ON THE FRAME (115). `regression` returns a 3-tuple that `regression_levels` and every caller unpack positionally, so growing it would be a change to all of them for one optional fact. `.attrs` is pandas' own place for exactly this and survives the frame being passed around; a caller that does not know about it is unaffected.

## regression_levels

### lines 5459-5462

```python
print(f"regression_type='mixed' fits the gene fixed with guides "
```

SAID, not silently overridden. The GUI greys `level` out for mixed (instruction 106), but a script can still set it, and a run that quietly ignored it would hand back a gene table to a caller who asked for a guide table.

### lines 5479-5481

```python
regression_type = fits[one][2]
```

check_distribution may have chosen the type on the first fit; the second must be the SAME model, not a second auto-selection that could pick differently and give two tables from two backends.

## _show_well_distributions

### lines 5502-5504

```python
per_panel = {"response": {"column": response_name}, "guide_fraction": {}}
```

The response panel takes the COLUMN NAME; the fraction panel takes nothing. Passing the response series would be handing a panel the values when it wants to know which column to read and label.

### lines 5524-5527

```python
save_figure(figure, os.path.join(str(dst), f"{name}.pdf"),
```

THROUGH THE PREFERENCE, not a literal .pdf. A user who set

"PNG" in Preferences and got PDFs anyway is the exact complaint `save_figure` was written to end, and the new panels quietly re-introduced it -- three times.

### lines 5534-5539

```python
plt.close(figure)
```

RELEASE THE MANAGER, KEEP THE FIGURE. `plt.show` is the bridge hand-off and the figure has to still be registered during it; after that, pyplot's registry is the only thing holding it, and a fit that draws two panels per run and never releases them is how a long session ends up with hundreds of live canvases. The Figure object survives `close` -- whatever the bridge kept still draws.

## _show_plates

### line 5576, trailing  _(unsure)_

```python
plt.close(figure)
```

same hand-off, same release

## _show_house_style_panels

### lines 5610-5612

```python
figure.set_label(panel.title)
```

The figure carries its own name, so the grid captions it "volcano" rather than "fig_00003" -- a temp file's stem is an implementation detail of how the picture reached the screen, not a caption.

### line 5617, trailing  _(unsure)_

```python
plt.close(figure)
```

the hand-off is done; release the manager

## _write_regression_sheet

### lines 5644-5652

```python
path = publish(sheet.figure, path, bbox_inches='tight') or path
```

PUBLISHED, not merely saved. Instruction 139 C: saving a figure and showing it are the SAME event. This is THE publication figure of a regression run and it was the one figure of the run nobody could look at -- written and then closed in the next breath, so no `plt.show()` ever walked past it and the gallery never held it.

`publish` still writes through `spacr.plot.save_figure`, so the sheet remains the last figure that should ignore the format and resolution the user chose.

## fit_quality_note

### lines 5712-5717

```python
try:
```

THE NULL LOG-LIKELIHOOD, NOT THE NULL DEVIANCE. This read

`model.null_deviance / -2`, which equals the null log-likelihood only when the saturated log-likelihood is zero -- true for 0/1 binomial data and false for the per-well PROPORTIONS this pipeline actually fits, so the ratio mixed two conventions. statsmodels fits the null model itself and exposes it as `llnull`.

## summary_for_console

### lines 5781-5789

```python
lines = text.splitlines()
```

THE COEFFICIENT ROWS ONLY. Located from the column header -- the line carrying "coef" and "std err" -- and ended at the next '=' rule, rather than by taking everything after the last separator. That simpler cut swallowed the notes table with the rows and MISCOUNTED the coefficients by however many notes the family happens to print.

The notes are KEPT. Durbin-Watson and, above all, the condition number are how a reader sees the collinearity that a screen's design has, and they are six lines.

## _split_prc

### lines 5886-5887

```python
raise schema.KeyParseError(
```

An empty row or column is not a missing token, it is a key every well of the plate shares: group on it and the wells merge.

## _is_row_column_pair

### lines 5925-5927

```python
return True
```

parse_well puts an unrecognisable well into both slots verbatim, so an equal unprefixed pair is that passthrough and not a prcf tail (a field never equals the column it sits in).

## resolve_auto_inference

### lines 6016-6032

```python
per_object = str(settings.get('analysis_unit') or 'well').lower() != 'well'
```

AUTO MUST NOT CHOOSE A MODE THAT CANNOT RUN ON THIS DATA.

The permutation test needs one row per WELL. With per-object rows it refuses -- correctly, and with a clear message -- but "auto" means "choose for me", and choosing something that raises the moment it is used is not a choice. `agg_type is None` is the reliable signal, not `analysis_unit`: some regression types force it to None themselves (quantile fits objects by construction), so a user who set analysis_unit='well' still ends up with object rows. ABSENT IS THE DEFAULT, NOT THE OPPOSITE OF IT. `settings.get('agg_type')` returns None for a key that was never set as readily as for one set to None, and those are opposite answers: the shipped default is agg_type='mean' with analysis_unit='well', so a dict that has not been through `set_default_analysis_settings` -- a sweep trial, a refit, a caller that assembled its own -- was read as PER OBJECT and auto then chose the simultaneous model for a design it could not identify. That is the one outcome this function exists to prevent.

### lines 6046-6047

```python
return 'guide_permutation', (
```

Cannot measure the design; the safe default is the test that stays valid at any width.

### line 6056  _(unsure)_

```python
parameters = 1 + blocks + n_guides
```

intercept + block fixed effects + one coefficient per guide

## normalize_regression_input_pairs

### lines 6109-6117

```python
'database': raw.get('database') or raw.get('measurements'),
```

THE MEASUREMENTS DATABASE SURVIVES THE ROUND TRIP.

This dict is written back over `settings['paired_data']` below, so any key it does not name is ERASED -- and the settings CSV a run saves is what a user reloads. Without this line a reloaded run comes back with an empty database column and no sign that it ever had one. Nothing in the fit reads it (the regression runs on scores and counts), which is exactly why it would have gone unnoticed.

### lines 6151-6153

```python
settings['score_data'] = unique('score')
```

Existing downstream threshold/path helpers still consume these flat views. They are projections of the explicit pairs, not a second pairing mechanism, and repeated shared files are read only once there.

## load_regression_input_pairs

### lines 6174-6192

```python
_parsed: dict = {}
```

ONE PARSE PER FILE, NOT ONE PER PAIR ROW, AND NO PARSE AT ALL WHEN THE FRAME IS ALREADY IN THIS PROCESS.

The Measurements tab points EVERY pair row's score at the single merged frame, so a four-plate screen handed the same file four times and this parsed it four times. That file is 2.75 GB on a four-plate screen: the process sat at 82% CPU with zero disk I/O -- reading it back out of the page cache -- for minutes, having already written it.

The merge that produced it runs in this same process, so `frame_handoff` lets it offer the frame under the path it wrote; then there is no parse to pay for and no 2.75 GB round trip through the filesystem. A caller that offered nothing reads the file exactly as before.

NO BLANKET COPY. Four copies of a 2.75 GB frame is eleven gigabytes of allocation for a mutation that happens on ONE branch below -- stamping `plateID` onto a file that names no plate. That branch copies; the filtering branches build new frames of their own and cannot reach the cached one.

### lines 6242-6243

```python
score = score[score['plateID'].astype(str).isin(count_plates)]
```

A single consolidated score file may intentionally be reused in several rows, one per plate-specific count file.

### lines 6258-6269

```python
if (len(score_plates) > 1 and count is not None
```

ONE SIDE HOLDS EVERY PLATE AND THE OTHER NAMES NONE. This is the Measurements tab's own shape: `column_run_settings` points every pair row's score at the single merged frame, which carries all four plates, while a real count CSV carries `row_name, column_name, grna_name, count` and no plate column at all. Copying four plates onto a partner that names none is not possible, and refusing was wrong: the pair ROW already says which plate this row is, and that is the third resolution rule this function documents. So use it -- and only when the plate it names is one the partner actually holds, so a screen whose plates are named anything else still refuses rather than inventing a match.

### lines 6299-6301

```python
if score is not None and not score_plates:
```

THE ONLY MUTATION, so the only place a copy is owed: `read` hands back the cached frame itself, and stamping a plate onto it would write the first pair row's plate into every later row's score.

## load_regression_input_pairs.read

### lines 6205-6207

```python
note = frame_handoff.describe(path)
```

SAY SO. Between the merge finishing and the fit starting the run used to print nothing at all for minutes, which is what made a working run look dead.

## _check_score_count_pairing

### lines 6370-6384

```python
comparable = min(score_wells, count_wells)
```

THE DENOMINATOR IS THE SMALLER SIDE, and getting that wrong made this guard reject a correct run.

The two sides are not expected to be the same size. Sequencing covers every well on the plate; imaging keeps only the wells that survive segmentation and the minimum-cell filter. On the TSG101 screen that is 463 score wells against 1,344 count wells -- and all 463 found a partner, which is a perfect join. Measured against the count side it reads as 34%, and the guard refused to run a screen that was completely paired.

What actually matters is whether the wells that CAN be fitted found their partner, so the denominator is the smaller side. An unusually large unused remainder on either side is worth saying out loud, but it is not an error: those wells simply contribute nothing.

### lines 6389-6390  _(unsure)_

```python
if record is not None:
```

Persist the join counts so the run summary can report the data that entered the analysis without relying on the transient console log.

## _usable_nuisance_columns

### lines 6505-6513

```python
if usable:
```

AND THEY MUST NOT MAKE THE DESIGN SINGULAR. `rowID` and `columnID` are a DEFAULT now, and on a layout where the plates align with plate position -- every plate its own block of columns, say -- the position dummies are a linear combination of the block dummies and `_nuisance_design` refuses the whole design.

A DEFAULT MUST NOT BE ABLE TO KILL A RUN. Dropped one at a time, worst last, so a screen where only one of the two is collinear keeps the other.

### lines 6523-6527

```python
if "rank deficient" not in str(exc):
```

ONLY RANK DEFICIENCY DROPS A COLUMN. `_nuisance_design` raises the same exception type when the BLOCK column is absent, and treating that as collinearity threw away a perfectly good nuisance column -- caught by the test that passes a frame with no plate column at all.

## _report_exchangeability

### lines 6573-6575

```python
positions = {c: outcomes[c] for c in present
```

POSITION IS MEASURED EVEN WHEN IT WAS REMOVED, which is the point of measuring it: a column already in `nuisance` should come back explaining nothing, and if it does not, the removal did not work.

## resolve_regression_src

### line 6619  _(unsure)_

```python
wanted = os.path.abspath(os.path.expanduser(requested.strip()))
```

Resolve user-home and relative components before checking the parent.

## _run_guide_permutation_analysis

### lines 6671-6676

```python
primary = settings.get('guide_primary_min_wells')
```

A BLANK BOX MEANS "the first threshold", the same as an absent key. `guide_primary_min_wells` is an optional field, so the panel leaves it empty and the settings CSV writes an empty cell -- which read back as '' and reached int(''), taking the whole nonparametric path down with "invalid literal for int() with base 10: ''". The permutation test was unreachable from the screen's own saved settings (236 C7).

### lines 6686-6690

```python
outcomes = [outcome] if isinstance(outcome, str) else list(outcome)
```

One or several responses. Naming more than one fits each independently and corrects each as its OWN multiple-testing family -- pooling them would make two correlated readouts of the same wells look like twice as many tests. Concordance between independently trained classifiers is evidence precisely because the families are separate.

### lines 6697-6728

```python
per_object = str(settings.get('analysis_unit', 'well')).lower() != 'well'
```

THE PERMUTATION TEST IS A TEST ABOUT WELLS, so it needs one row per well. `analysis_unit='cell'` (agg_type=None) hands it one row per CELL, and the phenotype then varies within a well -- which the permutation code catches, but nine frames deep and phrased as a data-integrity failure:

ValueError: Phenotype/block/nuisance values are not constant within well 'plate1_r1_c12'.

Reported 2026-08-17 after a 20-second run that had already written its regression data, three summary plots and their statistics. The combination is not a corrupt table; it is two settings that cannot both be honoured, and saying so costs nothing and is checkable HERE, before any of that work.

It refuses rather than aggregating silently: rolling cells up to wells changes what was analysed, and a run that quietly analysed something other than what was asked for is the failure this module is most careful about elsewhere. AND `agg_type is None` SAYS THE SAME THING. The check above reads `analysis_unit`, which is what a user SETS; `agg_type` is what decides whether the rows actually got rolled up, and some types force it to None themselves -- quantile fits objects by construction. A run with analysis_unit='well' and agg_type=None therefore reached the permutation test with per-object rows and died on

ValueError: Phenotype/block/nuisance values are not constant within well 'plate1_r1_c4'

which names a well and a pandas invariant rather than the two settings that cannot both be honoured. The message this branch already carries is the right one; it just was not reachable that way.

### lines 6758-6762

```python
_report_exchangeability(data, outcomes, settings, destination)
```

WHETHER THE SHUFFLE WAS ALLOWED (224). The test permutes phenotype residuals within each block, which is valid only if those residuals are exchangeable there -- and nothing said so until now. A parametric fit writes a QC folder; this path returned before it, so the analysis that residualises was the one that showed no residuals.

### lines 6764-6780

```python
results = results.copy()
```

THE SAME ALIASES ON THE FULL TABLE, and on the primary slice taken from it further down -- one block now, rather than two lists that had to stay in step. results.csv on disk holds the primary slice while the returned output['results'] holds every minimum-wells family, and when only one of them was aliased everything that consumes a coefficient table (the results panel, guide concordance, the volcano, the sweep's hit counts) raised KeyError('feature') on the nonparametric path while working fine on the parametric one.

Built HERE, before anything is saved or drawn, because the effect-size cut below is measured on `coefficient` and the volcano has to draw the cut it produces.

ADDED rather than swapped: a caller that wants the permutation quantities themselves still has every one of them, and the names say that the inferential quantities are marginal effects, empirical P values and already-adjusted values.

### lines 6789-6805

```python
results['condition'] = label_control_condition(
```

AN EFFECT-SIZE CUT IS NOT A PARAMETRIC IDEA, and saying it was is the answer the maintainer was given: "why cant i see the coefficient threshold if im running nonparametric regression?"

A P value says an effect is distinguishable from zero. The effect-size cut says it is big enough to be worth an experiment. That is a question about the COEFFICIENT, and this table has a real one for every guide `standardized_marginal_effect`, aliased to `coefficient` two lines up, 1,726 of them on the screen this was reported from. How the P value was obtained does not change how wide a control's effect is.

`condition` is what the cut is measured on, and this table did not carry it. Measured before this was written, on a permutation-shaped frame: `RegressionResultsPanel._threshold_sentence()` answered "No control coefficients, so no effect-size cut." -- and the run itself returned from `perform_regression` before the parametric branch that computes one, so a permutation run drew no cut and reported no cut either.

### lines 6814-6817

```python
control_effects = results.loc[
```

MEASURED ON THE PRIMARY FAMILY. The same guide appears once per minimum-wells threshold with an identical coefficient, so pooling the families would count each control up to four times and shrink the spread the cut is built from.

### lines 6825-6828

```python
centre=None)
```

The MEDIAN of the controls, computed inside, rather than the mean: `000000_22` is a non-targeting control and the strongest effect in the screen at +4.37, and a mean centre moves the cut for every guide because of it.

### lines 6832-6833

```python
results['effect_size_threshold'] = (
```

RECORDED PER ROW, not only printed. A cut a reader cannot recompute from the results CSV is a cut they cannot report.

### lines 6846-6853

```python
have = results.loc[
```

A THRESHOLD NOTHING REACHES IS AN ANSWER, NOT A FAILURE. `guide_min_wells` is a SWEEP -- [1, 2, 3, 4] asks the same question four times at four strictnesses -- and on a one-plate screen no guide appears in four wells. The analysis is finished by the time this loop runs, so raising here threw away the results for 1, 2 and 3 as well, at the drawing stage, with a message about a plot. Reported by driving the tsg101 screen (236 C7).

### lines 6864-6866

```python
stem = (f'guide_permutation_min_{threshold}_wells'
```

One response keeps the historical filenames and keys, so scripts that look for guide_permutation_min_1_wells.pdf still find it.

### lines 6879-6880  _(unsure)_

```python
effect_threshold=effect_threshold,
```

DRAWN, not only computed. The cut is the same number on the plot, in the CSV and in the log line above.

### lines 6885-6888

```python
try:
```

Diagnostics are written for every run, not on request. The failure this analysis mode exists to prevent -- a confident coefficient from a rank-deficient design -- is invisible on the volcano and obvious on the design panel, so the design panel has to be produced by default.

### lines 6913-6917

```python
prefix = f'{response}_' if len(outcomes) > 1 else ''
```

Namespace the KEYS per response as well as the filenames. Both responses write distinct files, but they returned the same keys, so a two-classifier run reported only the second one's paths and the first classifier's diagnostics looked as if they were never produced.

### lines 6925-6934

```python
primary_table = results.loc[
```

ONE SET OF ALIASES, built once above and inherited by this slice.

results.csv on disk holds primary_table while the returned output['results'] holds the full multi-threshold frame, and the two used to be aliased by two separate blocks of code -- one name, two shapes. Everything that consumes a coefficient table (the results panel, guide concordance, the volcano, the sweep's hit counts) raised KeyError('feature') on the nonparametric path while working fine on the parametric one. Slicing the aliased frame is what makes them the same table by construction rather than by two lists staying in step.

### lines 6939-6947

```python
called = primary_table['significant'].astype(bool)
```

A HIT CLEARS BOTH BARS, the way the parametric path's hit list does: corrected P below alpha AND an effect at least as wide as the cut. `passes_effect_size` is all-True when there is no cut to apply -- a control-free screen, or a `threshold_method='none'` -- so a run without controls calls exactly the hits it called before.

Nothing is dropped silently: every guide keeps its row, its

`effect_size_threshold` and its `passes_effect_size` in results.csv, and the line below says how many the cut removed.

### lines 6956-6980

```python
wanted_level = str(settings.get('level') or 'both').strip().lower()
```

THE GENE PASS. Instruction 132 gives the parametric path two fits and two tables; the permutation path answered only the guide question, so choosing inference='nonparametric' silently lost the gene level altogether -- results_gene.csv was never written by this branch at all.

Each gene is tested as a SET: its regressor is the SUM of its guides' fractions, which is the same `gene_fraction` the parametric gene fit uses, residualized against the same block design and permuted with the same Freedman--Lane scheme and the same seed. It is NOT a combination of the guides' P values -- Fisher and Stouffer both assume independence, and guides scored in the same wells share that well's phenotype, plate and cells, so combining them would claim a confidence the design cannot support.

ITS OWN BH FAMILY, never pooled with the guides: same wells, and the gene regressor is literally the sum of the guide regressors. WHICH LEVELS THIS RUN REPORTS -- the same `level` key the fitted path reads, so one control answers the question on both sides. It used to be `guide_permutation_gene_level` alone, which is in no category and so had no control at all: on the permutation side the level was unchoosable, and `level` itself was greyed out because a mixed regression_type does not read it. Between them a reader who asked for genes had no way to ask.

`guide_permutation_gene_level` still WINS when it is set explicitly, so a saved settings file that names it keeps meaning what it said.

### lines 7036-7055

```python
levelled = primary_table.copy()
```

`results.csv` CARRIES EVERY LEVEL THE RUN PRODUCED, which is the convention the fitted path already follows: a level='both' regression writes its guide and gene rows into one table and the results panel filters them apart by the `level` column.

This branch used to write the guide table alone, so a permutation run that HAD tested genes -- and written them to results_gene.csv -- showed a reader nothing when they asked for genes. The rows existed, in a file the panel never opens, because it loads results.csv and stops.

The two tables do not share a schema (a gene has `wells_with_gene` and `guides_in_gene`; a guide has `wells_with_guide`), so the union carries blanks where a column belongs to the other level. That is correct: the question "how many wells hold this guide" has no answer for a gene.

`level` decides which of them results.csv CARRIES. The guide pass runs either way -- a gene's regressor is the sum of its guides' fractions, so there is no gene answer without it -- and results_grna.csv always holds those rows. What level='gene' means is that the reader asked for genes, so genes are what the primary table reports.

### lines 7070-7071

```python
(gene_primary if gene_primary is not None
```

Written every run, empty when the pass could not be made, because a file that is absent is indistinguishable from a run that crashed.

### lines 7079-7103

```python
'results': combined,
```

ONE ROW PER GUIDE, NOT ONE PER GUIDE PER FAMILY.

`guide_min_wells` defaults to [1, 2, 3, 4], so this analysis runs FOUR times at four inclusion thresholds -- four separate analyses of the same guides. `results` is all four stacked: 1,612 rows for 789 guides on the real screen, with `225160_2` appearing four times at the identical effect 0.25406.

Handing that to the results panel drew every guide FOUR TIMES on one volcano. Reported as "GRA14 and 225160 occur in the top right side of the graph 4 times each which is obviously wrong", and it was twice I explained it away as a q-value tie artefact before checking the row counts, which the maintainer had already told me: "my data say 1612 gRNAs".

The panel gets the PRIMARY family, which is exactly what

`results.csv` on disk already holds, so the file and the screen finally agree. Every family stays reachable: `families` carries the full frame and each one is still written to its own `guide_permutation_min_<n>_wells.csv`.

`combined`, NOT `primary_table`: the file gained the gene rows, and a caller reading the dict must not get a different table from a caller reading the file of the same name. `primary` below is the guide rows alone for anyone who wants exactly those.

### lines 7106-7107  _(unsure)_

```python
'gene_results': gene_primary,
```

The gene pass, corrected within itself. None when it was declined with guide_permutation_gene_level=False or could not be made.

### lines 7112-7114

```python
'effect_size_threshold': effect_threshold,
```

The cut, and the sentence that attributes it. A threshold a reader cannot attribute is a threshold they cannot put in a methods section, which is why the rule travels with the number.

## _perform_regression_set_paths

### lines 7135-7137

```python
csv_path = settings['count_data'][0]
```

_perform_regression_read_data has already normalised both keys to lists by the time this runs, so the old scalar fallbacks here were unreachable.

### lines 7140-7141  _(unsure)_

```python
automatic = os.path.dirname(settings['count_data'][0])
```

A configured output root takes precedence. Blank values retain the established behavior of writing beside the first count table.

### line 7145  _(unsure)_

```python
if how != 'automatic':
```

Report any explicit override, including a documented fallback.

### lines 7149-7162

```python
kind = results_folder_kind(settings)
```

WHERE A RUN'S OUTPUT GOES: <count data folder>/results/<type>, and never on top of an earlier run.

Asked for on 2026-08-16: "just store everything in the same location as the first count data ... then the type so for me .../claude/results/ols. if there is already an ols folder then ols_1 then ols_2 and so on".

The old path was <src>/results/<score_source>/<type>/list -- two levels nobody asked for, one of them named after a CSV, and a fixed leaf that meant a second run of the same type silently replaced the first. That is also why the results panel could not find anything: the path it had to guess at was four levels deep and named after a file rather than the run.

### line 7166

```python
settings["_regression_folder"] = res_folder
```

WHERE A FAILURE REPORT GOES, recorded as soon as the folder exists.

## _next_results_folder

### line 7227, trailing  _(unsure)_

```python
except OSError:
```

unreadable: treat as taken and move on

## _annotate_level_coefficients

### lines 7260-7272

```python
carried = dict(getattr(coef_df, "attrs", {}) or {})
```

n_grna / n_gene are value_counts frames, so one row per gRNA and one per gene. coef_df is many rows against either of them — every gene[...] term carries grna=None and vice versa — so many-to-one is the contract, and it is the right side (the counts) that must stay unique: a duplicate there would fan the coefficient table out and every hit would be written to results_significant.csv more than once.

CARRY `.attrs` ACROSS. `DataFrame.merge` does not propagate it `copy` and `concat` do, which is what makes the loss easy to miss and `regression` puts the QC manifest there for `_perform_regression` to read back. Without this the manifest died between the two, so a run's `output` carried no 'qc' key at all and instruction 115's verdict never reached the caller.

## _level_control_rows

### lines 7296-7302

```python
from .control_names import matches, resolve_controls
```

THE SAME MATCHER THE VOLCANO USES (184 C). This took

`name.split('_')[0]` as the gene, which reads `TGGT1` as the gene of `TGGT1_000000_1` -- so a control pasted from a library file selected nothing at gene level and the gene table silently got no effect-size cut, which is the exact failure this function was written to fix, one spelling further along. `spacr.control_names` measures the organism prefix instead of assuming there is not one.

### lines 7313-7315

```python
if level == 'gene':
```

AT THIS FIT'S OWN LEVEL. A gene fit has no guide column -- every `gene_fraction:gene[...]` row carries grna=None -- so a guide-level control is matched by the gene it belongs to there.

## _call_level_hits

### lines 7396-7399

```python
coef_df = coef_df.copy()
```

OWNED, not borrowed. Every caller so far handed in a slice of a bigger frame -- the mixed path slices the BLUP rows off -- and assigning `q_value` onto a slice raises SettingWithCopyWarning, which this suite promotes to an error and pandas 3 will make a hard failure.

### lines 7402-7407

```python
reg_threshold = 0
```

reg_threshold used to be bound only inside the branch below, so a control-free screen (settings['nontargeting_control_grnas'] is None) hit UnboundLocalError as soon as the toxo volcano block read it. 0 is custom_volcano_plot's own default and means "no coefficient cut-off, select on p <= 0.05 alone", which is the only sensible threshold when there are no controls to calibrate against.

### lines 7415-7418

```python
measured_threshold, threshold_rule = coefficient_threshold(
```

SEVEN METHODS, in one place. It was two -- std and var -- and the maintainer asked for "at least 4 more" reachable from the plot, so the arithmetic moved to `spacr.thresholds` where the GUI can reach the same list rather than keeping a second copy of it.

### lines 7423-7426

```python
centre=None)
```

The MEDIAN of the controls, computed inside, rather than the mean this used to add: `000000_22` is a non-targeting control and the strongest effect in this whole screen at +4.37, and a mean centre moves the cut for every guide because of it.

### lines 7431-7452

```python
reg_threshold = (0 if measured_threshold is None
```

`coefficient_threshold` answers None when NO CUT CAN BE MADE

`threshold_method='none'`, fewer than two control coefficients, or a set of controls with no spread at all. It is deliberately not a silent 0, so that the caller has to decide what to do about it, and every one of this function's three readers wanted a number:

the hit list below compared a Series against None. pandas 2.x evaluates `series >= None` as all-False rather than raising, so BOTH masks were empty and the run wrote an EMPTY results_significant.csv -- measured on the synthetic screen, 16 of 16 corrected hits lost, with nothing said. `custom_volcano_plot` does `abs(threshold)` and died with `TypeError: bad operand type for abs(): 'NoneType'` after the whole fit, every results CSV and every QC panel had been written. `plot.volcano_plot` takes it as `fold_change_threshold`.

0 is what "no coefficient cut" already means to all three -- the value a control-free screen has carried for as long as this line has existed -- and `threshold_rule`, printed above, is what says WHY there is none. The reason is on the record; only the sentinel is normalised.

### lines 7456-7461

```python
print(f"Effect-size cut ({level}): no control gRNAs were named, so "
```

SAID, not left silent. A run WITH controls prints its cut and the rule behind it; a run without them printed nothing, so "there is no effect-size cut" looked exactly like "that line scrolled past". It is the more surprising of the two, because a hit list called on the corrected P value alone is a different claim from one that also had to clear a width.

### lines 7473-7476

```python
n_boot = settings.get('lasso_n_boot', 200)
```

Lasso and elastic net have no valid frequentist p-values (the ones process_model_coefficients attaches are OLS-style and ignore the penalty). Use bootstrap selection frequency as the feature-importance ranking. Treat as a selection method, not a hypothesis test.

### lines 7482-7486

```python
model_plate_position=settings.get('model_plate_position', True))
```

The bootstrap must resample the design the fit used, so the selection frequencies are frequencies for THAT model. Reading the setting rather than passing True was worth a comment: with plate position out of the fit and in the bootstrap, a guide would be selected against a different set of competitors.

### lines 7488-7489  _(unsure)_

```python
cleaned_df = check_and_clean_data(merged_df.copy(), dependent_variable)
```

Apply the same preprocessing the OLS path uses, so derived columns referenced by the formula (e.g. gene_fraction) exist in the bootstrap.

### lines 7502-7509

```python
coef_df = coef_df.merge(sel_df, on='feature', how='left',
```

One row per model term on both sides: coef_df['feature'] is the design-matrix column index (X.columns for lasso), sel_df['feature'] is the same index taken off the reference design built once at the top of bootstrap_selection_frequencies. Both are pandas Index objects from patsy, so the join is one-to-one by construction, and a duplicate on either side means the two designs have gone out of step — which is exactly when a selection frequency must not be silently attached to the wrong coefficient.

### lines 7523-7539

```python
from .multiple_testing import adjust_p_values, canonical_method
```

THE CORRECTION IS APPLIED HERE, and until instruction 128 it never was.

`multiple_testing_method` has existed as a setting, been offered in the panel and been named in Methods sections, while this branch called a hit on the RAW OLS p-value. With 1,208 coefficients an uncorrected 0.05 expects about sixty false positives from noise alone, and that is the defect behind a published volcano whose figure showed a P = 0.05 line while its Methods claimed BH q < 0.05.

The family is the guide/gene coefficients OF THIS FIT -- not the intercept and not the row/column nuisance terms, which are covariates rather than hypotheses and would only inflate the family, and not the other fit's coefficients either.

'none' reproduces the historical rule exactly, so a run that wants the old behaviour can still ask for it and is on record as having asked.

### lines 7545-7559

```python
cut_alpha = float(settings.get('p_threshold_alpha', alpha) or alpha)
```

WHERE THE LINE IS DRAWN, AND ON WHICH P. Instruction 135, asked for on 2026-08-17: "add a setting that setts what alpha the p threshold is set at and if adjusted p or raw p is used".

Until these two, the CORRECTION's alpha was also the hit cut, and the cut was always on the adjusted P -- while the volcano's own right-click menu could switch the axis to the raw P. So the exported hit list and the picture printed beside it could mean two different things by "significant", with nothing saying which.

They are separate from `fdr_alpha` on purpose. `fdr_alpha` is the level the CORRECTION targets, an input to the procedure; this is the level a coefficient is CALLED at. Same number by default, and a reader is entitled to move one without the other -- correcting at 0.05 and reporting at 0.01 is an ordinary thing to want.

### lines 7563-7565

```python
from .hits import tested_family
```

ONE STATEMENT OF WHAT IS BEING TESTED, shared with the volcano. A plot drawn from a different family than the one corrected here is a plot of a different experiment; see spacr.hits.tested_family.

### lines 7570-7576

```python
tested &= coef_df['p_value'].notna()
```

A ROW WITHOUT A P VALUE IS NOT A TEST. The mixed fit returns variance components and guide BLUPs alongside its fixed effects, and both carry a NaN p-value BY CONSTRUCTION -- a BLUP is a shrunken prediction of a random effect, not an estimate of a parameter that could be zero. Left in the family they would enlarge it (weakening every real q value) and come back with a q value of their own, which is a p-value manufactured for a quantity that has none.

### lines 7578-7582

```python
if 'term_type' in coef_df.columns:
```

A VARIANCE IS NOT A HYPOTHESIS ABOUT A GENE. The mixed fit's 'grna Var' row carries a real Wald p-value (0.109 on the synthetic nesting), so the NaN guard above does not catch it -- and left in the family it both enlarges the correction and comes back with a q value, which would put a 'hit' on the volcano that no gene owns. Only fixed effects are tests.

### lines 7598-7601

```python
print(f"  Calling hits on the {cut_kind} P at {cut_alpha:g}"
```

SAID OUT LOUD, because it is the one line that decides what the exported table means. A cut on the raw P over hundreds of guides is a defensible choice and an indefensible accident, and the only way to tell them apart is whether the run announced it.

### lines 7607-7634

```python
coef_df['effect_size_threshold'] = (
```

THE EFFECT-SIZE CUT IS A WIDTH, SO IT IS SYMMETRIC. Until instruction 128 this was a pair of one-sided masks whose UNION was every row:

high = coefficient >= reg_threshold low  = coefficient <= reg_threshold

`reg_threshold` is `|median| + k x spread`, so it is never negative and every coefficient satisfies one side or the other. Measured on the synthetic screen with `threshold_method='std'`: the cut was 0.57, and all 16 corrected hits survived it -- the narrowest at |coefficient| = 0.0026, more than two hundred times inside the cut. `custom_volcano_plot`, handed the SAME number, marks hits with `abs(coefficient) >= abs(threshold)` and would have called none of them, so the figure and results_significant.csv described different experiments and only the figure was right.

A cut that admits +0.9 but not -0.9 would also call half a screen: a guide that moves the phenotype DOWN by more than the controls ever move is exactly as much a hit as one that moves it up, and which direction is 'good' is the biology's business, not the filter's.

This is the rule `_run_guide_permutation_analysis` already applies (`passes_effect_size`), so the parametric and nonparametric paths now call a hit the same way.

NOTHING IS DROPPED SILENTLY: every coefficient keeps its row in results.csv with its q value, and the line below says how many the cut removed and how wide it was.

## _diagnostic_screen_design

### lines 7743-7745

```python
return data, None
```

``write_diagnostic_suite`` has always accepted an already-wide DataFrame.  Do not mistake its guide names for missing long-table metadata and try to pivot it.

## _write_regression_diagnostics

### lines 7837-7840

```python
note_path = os.path.join(destination, "residual_panels_not_available.txt")
```

THE NOTE IS A FILE, not a print. A run is read from its folder afterwards, usually by someone who did not watch it run, and a console line is gone by then -- which is exactly how an inapplicable panel becomes indistinguishable from a missing one.

## perform_regression

### lines 7952-7958

```python
try:
```

DUCK-TYPED, NOT `isinstance(settings, dict)`. The contract test in tests/test_regression_entry_points.py scans every call this function hands `settings` to, so that the keys each one reads can be checked for a default -- which is how six missing defaults were once found. A bare `isinstance(settings, ...)` registers as such a call and the scan then asks which spacr module `isinstance` lives in. Asking forgiveness keeps the settings dict out of a call the scan has to reason about.

### line 7961, trailing  _(unsure)_

```python
except AttributeError:
```

not a mapping; nothing to do

### lines 7984-7986

```python
raise
```

RE-RAISED UNCHANGED. The reporter adds to a failure; it must never replace one, or a caller that handles a specific exception type stops seeing it.

## _perform_regression._perform_regression_read_data

### lines 8141-8146

```python
looks_like_counts = sorted(
```

Name the likeliest cause, not only the symptom. A count table has grna/count columns and no score column, so a score slot holding one is a swapped input -- the commonest way to reach this error, and invisible in a bare "not found in the DataFrame" followed by a column dump the user has to interpret themselves.

### lines 8170-8176

```python
_reject_impossible_probabilities(settings)
```

The whitelist is REGRESSION_TYPES itself, not a copy of it. The copy that used to live here disagreed with the dispatcher in both directions: it refused 'beta' and 'quasi_binomial', which regression_model fits and check_distribution auto-selects, and it accepted 'gls', 'wls', 'rlm' and 'quantile', which had no backend - 'quantile' failing only at the last statement, after every CSV and QC plot had been written.

### lines 8195-8197

```python
_reconcile_random_row_column_effects(settings)
```

Order matters: the reconcile can rewrite regression_type to

'mixed', and the run-level knobs have to be policed against the model that will actually be fitted.

## _perform_regression._count_variable_instances

### lines 8207-8210

```python
for col in (column_1, column_2):
```

The single call site always passes both column names, so the variable-arity returns this used to carry (two-tuple / bare df) were unreachable; it now always returns the three-tuple its caller unpacks.

## _perform_regression._qc_plot

### lines 8225-8245

```python
def _qc_plot(plot_settings):
```

WHAT THESE COUNT, because the names invite a wrong reading and the maintainer asked outright whether they work.

`df` is one row per (well, guide). So:

n_grna  for a guide = the number of WELLS that guide appears in. n_gene  for a gene  = the number of (well x guide) ROWS it has, i.e. wells MULTIPLIED BY guides.

n_gene is therefore NOT "how many guides target this gene" and NOT "how many wells this gene is in". On the real screen gene 244480 has ONE guide and n_gene = 5 (that guide in five wells), while 239740 has TWO guides and n_gene = 15. A reader comparing n_gene across genes is comparing a product, not a count of anything.

Left as the product rather than quietly redefined:

`min_observations_per_hit` filters on it and the results CSVs of every past run carry it, so changing what the number MEANS is a separate decision from fixing WHICH ROWS it is taken over. The guide-support table beside it already reports guides per gene, which is the number a reader usually wants.

## _perform_regression.grna_metricks

### lines 8279-8281

```python
unique_triplets = df[['grna', 'gene', 'plateID']].drop_duplicates()
```

4) Merge These Counts into a Single DataFrame

Because each grna is typically associated with one gene, we bring them together. First, create a unique (grna, gene, plate) reference from the original df

### lines 8284-8293

```python
merged_df = pd.merge(unique_triplets, grna_well_counts,
```

Merge the grna_well_count.

Both count frames come straight off a groupby on their own join key, so each holds exactly one row per key: the joins are many-to-one and must not change the row count of unique_triplets. Stating that is not decoration — a gRNA mapped to two genes puts the same (grna, plateID) on the left twice, and if the right side ever gained a duplicate too (two count CSVs for one plate concatenated, say) the result would quietly gain rows and every well count written to grna_well.csv would be counted more than once, with no error anywhere.

### lines 8298-8299  _(unsure)_

```python
merged_df = pd.merge(merged_df, gene_well_counts,
```

Merge the gene_well_count. Many gRNAs share a gene, so the left side is legitimately many; gene_well_counts is one row per (gene, plate).

### line 8304  _(unsure)_

```python
final_grna_df = merged_df[['grna', 'plateID', 'grna_well_count', 'gene_well_count']]
```

Keep only the columns needed (if you want to keep 'gene', remove the drop below)

### lines 8307-8308  _(unsure)_

```python
prc_gene_count_df = (df.groupby('prc')['gene'].nunique().reset_index(name='gene_count'))
```

5) Compute gene_count per prc

For each prc (well), how many distinct genes are there?

## _perform_regression.get_outlier_reference_values

### line 8327  _(unsure)_

```python
lower_bound = Q1 - 1.5 * IQR
```

Determine the outlier cutoffs

### line 8331  _(unsure)_

```python
outlier_mask = (df[outlier_col] < lower_bound) | (df[outlier_col] > upper_bound)
```

Create a mask for outliers

## _perform_regression.bootstrap_selection_frequencies

### lines 8384-8386

```python
y0, X0 = dmatrices(formula, data=X, return_type='dataframe')
```

Build the reference design once so the feature index is stable. The response comes back with it, because choosing a group-lasso penalty below needs the pair rather than the design alone.

### lines 8390-8403

```python
blocks = (_design_column_groups(feature_index)
```

THE GROUP LASSO IS RESAMPLED THROUGH ITS OWN SOLVER, not through sklearn's. Falling through to `_estimator()` would have fitted an ORDINARY lasso on every resample and reported its stability under the group lasso's name -- selecting one guide out of a gene's correlated set, which is the exact behaviour the group penalty exists to remove.

PER FEATURE, not per gene, even though spacr.group_lasso. stability_selection answers per gene: `_call_level_hits` merges this frame onto the coefficient table on `feature`, one to one. Nothing is lost by it -- a block is entirely zero or entirely non-zero, so every column of a gene carries that gene's frequency -- and the resampling scheme stays the one the other two penalties use, so the number in the column means the same thing whichever penalty wrote it.

### lines 8406-8411

```python
block_penalty = None
```

THE SAME PENALTY THE FIT USED, chosen once rather than per resample. 'auto' reaches here too -- it is the panel's default for this backend -- and `float('auto')` is not a number. Cross- validating inside the bootstrap would also mean 200 different penalties, so the frequency would be "how often a gene survives SOME penalty", which is a different and weaker claim.

### line 8449  _(unsure)_

```python
dropped += 1
```

A resample can occasionally drop a factor level entirely.

### lines 8465-8470

```python
if dropped:
```

Same trap as _bootstrap_wald_p_values: only "none of them" raised, and that is the harmless case. `selection_frequency` is divided by `successful`, so 199 of 200 resamples dropping gives a stability frequency computed from a single draw — every selected feature at 1.00 and every other at 0.00 — reported in the same column, with the same name, as a frequency over 200.

## _perform_regression

### lines 8487-8491

```python
from .regression_layout import normalise_count_table_layout
```

ONE CANONICAL BOUNDARY FOR LONG AND WIDE COUNT INPUTS. Downstream filtering has always consumed one row per (well, guide); accepting a one-guide-per-column table anywhere later would make every model invent its own melt rule. The inverse conversion remains available to the fixed-effects fit through model_data_layout='wide'.

### lines 8508-8521

```python
count_data_df['rowID'] = (
```

A count CSV can carry rowID as the composite '<plate>_<row>' that 'plate_row' columns are written with (process_reads splits the same shape). Reduce it to the row by taking the token after the LAST separator, per row: the plate is the component that may itself contain one, the row never is.

This used to read count_data_df['rowID'].iloc[0] alone, count its parts, and apply split[1] to the whole frame, which is wrong in three ways that all end in a silently mis-keyed regression: 'exp1_plate1_r2' has three parts, not two, so it was left untouched and prc became 'plate1_exp1_plate1_r2_c1'; on a frame where only some rows carry the plate prefix, split[1] is NaN for every row that does not, so their rowID was erased; an empty count table raised IndexError on the .iloc[0].

### lines 8527-8528

```python
if {'plateID', 'rowID', 'columnID'}.issubset(score_data_df.columns):
```

Pair resolution is authoritative. Filenames are suggestions in the UI, never a second silent source of plate identity here.

### line 8533  _(unsure)_

```python
if settings.get('verbose'):
```

test 1

### lines 8558-8561

```python
corrected, correction_report = correct_from_metadata(
```

Beside `correction_kwargs`, not inside it: that helper's output is `**`-splatted into several different signatures, and adding a key to it turns every caller that has not grown the parameter into a TypeError. combat's two keys are named here instead.

### lines 8574-8592

```python
shift = float(np.abs(
```

IT SAYS HOW FAR IT MOVED THE DATA, and says so when the answer is "not at all".

Instruction 135 D, reported on 2026-08-17: "plate and batch correction is good but im not sure i see a diference when i use it". MEASURED rather than explained. On three plates with a real offset, `center` and `zscore` collapse the centroid spread from 0.527 to exactly 0.000 and move each value by ~0.46 on average the correction works. On ONE plate they are an exact no-op, mean|delta| = 0.000000, because there is no between-plate variance to remove; on plates that genuinely agree they move values by ~0.01. Both are correct, and both used to print a centroid-spread line that a reader could not tell apart from a correction that had done something.

The centroid spread alone does not answer the question either: it goes to 0.000 in every case that ran, including the ones that changed nothing. The mean absolute shift is the number a user comparing two runs is actually looking for.

### lines 8620-8621

```python
score_data_df.loc[:, dependent_variable] = corrected[
```

ASSIGNED LAST, so the shift above compares the corrected values with the originals rather than with themselves.

### lines 8630-8634

```python
count_source = os.path.dirname(settings['count_data'][0])
```

The volcano goes with the rest of the run's output, not beside the INPUT data. Writing it to the count CSV's folder put the module's headline figure two directories away from every table and plot it belongs with so a run that had produced it perfectly well looked like it had produced no graph at all, which is exactly how it was reported.

### lines 8642-8650

```python
try:
```

THE CONTROL BLOCKS COME OUT TOO, WITHOUT BEING TYPED TWICE (221). A well of pure control is not a screen well -- it holds one guide by construction, so its phenotype says what that guide does and nothing about any gene under test -- and left in it is modelled as a random draw from the library, at high leverage when the control is strong.

ADDED TO `filter_value` RATHER THAN FILTERED SEPARATELY, so there is one removal, one printed line per well and one place a reader has to look to know what left the run.

### lines 8659-8662

```python
filter_column = settings['filter_column']
```

filter_column used to be bound only in the `isinstance(..., str)` branch, so both None (the natural "do not filter" value) and the list form that process_reads documents left it unbound and the process_reads call below raised UnboundLocalError. clean_controls handles str / list / None.

### lines 8667-8676

```python
try:
```

OUTLIERS GO NOW, BEFORE ANYTHING COUNTS THEM (instruction 210). Every normalising step below -- the cell-count threshold, the guide fractions, the aggregation -- has its denominator set by which objects are present, so a segmentation artefact removed AFTER the fractions are formed leaves its reads redistributed across the guides in its well. Removed here, it never contributed.

OFF UNLESS ASKED FOR, and always reported: a filter that silently drops objects is a filter that will be forgotten and then blamed on the annotation.

### lines 8686-8688

```python
print(f"[outliers] the pre-annotation filter did not run "
```

SAID OUT LOUD. A filter the user switched on that did not run is the one thing worse than one that ran silently: the numbers below would be the unfiltered ones and nothing would say so.

### lines 8696-8698

```python
_AUTOMATIC_SETTINGS.clear()
```

Which settings this run DERIVED rather than being given. Reset here, at the top of the run, so a second run in one process cannot inherit the first one's -- a GUI session runs many.

### lines 8701-8721

```python
screen_folders = _screen_figure_folders(settings)
```

WRITTEN INTO THE RUN FOLDER, not copied into it afterwards.

`cell_min_threshold.pdf` used to go to <count folder>/results/ -- the SCREEN folder, one path shared by every run of the screen -- and this call site snapshotted that folder and copied back whatever appeared, the way it still does for the sequencing sweep below.

That worked for one run at a time and NOT for the sweep. `parameter_sweep.run_sweep_parallel` fits n_jobs trials of the same screen in a ProcessPoolExecutor. `_trial_settings` gives each trial its own `src`, so the RUN folders are already separate -- but the default destination here comes from `count_data`, which every trial shares, and this figure is drawn on EVERY trial (the call is unconditional; only whether its ANSWER is used depends on min_cells_per_well). So n_jobs workers wrote one path at once, and "every figure whose stamp changed since I started" cannot tell one worker's curve from another's: a trial could file the neighbouring trial's picture as its own, or copy one mid-write.

`res_folder` is this run's own folder, so naming it removes the shared path rather than working around it.

### lines 8737-8741

```python
_before_transform = None
```

THE RESPONSE BEFORE THE TRANSFORM, kept for the panel below. Taken here because `process_scores` applies the transform and hands back only the result -- and the whole point of instruction 218's panel is the comparison, which is unrecoverable once the untransformed values are gone.

### lines 8750-8751  _(unsure)_

```python
_before_transform = None
```

A panel is not worth losing a run to. The comparison is dropped, the fit is not.

### lines 8768-8780

```python
measured = _calibrated_fraction_threshold(settings)
```

MEASURED FROM THE CONTROL WELLS, when the user asked for that.

`target_unique_count` answers a different question -- how many gRNAs a well should end up with -- and answers it from the counts alone. This one asks which cut-off makes the imaging and the sequencing agree, which is the question a screen is actually asking, and it can only be asked where the plate design names pure control wells.

A sweep that cannot run says so and falls through to whatever the settings already chose. It must not take the run down: the calibration is an improvement on a number that already has a value, not a prerequisite for having one.

### lines 8787-8796

```python
before_sweep = _figure_stamps(screen_folders)
```

THE gRNA THRESHOLD GRAPH BELONGS TO THE RUN.

`graph_sequencing_stats` derives its own destination from

`count_data[0]` inside spacr.sequencing, so the sweep curve and the unique-count plate heatmap land in the SCREEN folder. Both are streamed through `plt.show()` and so still reach the live figure queue -- measured, not assumed -- but neither was ever in the run folder, which is what the all-figures grid walks for a saved run and what a reader opens by hand. Reported as "for some reason now i dont see the grna threshold graph".

### lines 8804-8817

```python
before_sweep = _figure_stamps(screen_folders)
```

AND IT IS DRAWN ANYWAY. Reported twice -- "for some reason now i dont see the grna threshold graph", and again 2026-08-21: "in the figure view i nevers ee the frna threhsold graph".

THE ANSWER LAST TIME WAS A SENTENCE SAYING WHY, which is not what was asked for. The sweep is a fact about the SCREEN -- how many guides survive at each threshold -- and it is worth the same whether spaCR chose the threshold or the user did. It is arguably worth MORE when the user chose one, because then it is the only thing that says where their number sits on the curve.

The default is 0.02, so the old gate meant the graph was never drawn on an ordinary run: the one case it fired in was the one nobody was in.

### lines 8826-8836

```python
if _AUTOMATIC_SETTINGS:
```

WHAT THE RUN ACTUALLY USED, said where the settings are read.

Asked for 2026-08-17: "if no fraction threshold and min cell cound is set these are set automatically, these automatic values should be shown in the runs values rows".

The settings table is printed -- and `save_settings` writes the CSV BEFORE either of these is derived, so both showed `None` there and the numbers only ever appeared in passing prose ("Closest Fraction Threshold: 0.0168"). A settings record that says None for a value the run chose is a record you cannot reproduce the run from.

### lines 8841-8843

```python
try:
```

Re-saved so the CSV carries the resolved values rather than the Nones it was written with. Same path, so it is one file and the complete version wins.

### lines 8849-8853

```python
_exclusions = settings.setdefault("_regression_exclusions", {})
```

WHERE THE EXCLUSION COUNTS ARE COLLECTED (instruction 156). One dict on the settings, filled by whichever step drops rows, read by the run summary at the end. It lives on `settings` rather than on the frame because a count has to survive the joins and re-indexes the frame does not carry `.attrs` through.

### lines 8861-8864

```python
if settings.get('exclude_grnas'):
```

Keep the optional keyword out of legacy calls when it is inactive. Besides preserving compatibility for callers that wrap process_reads, this makes the active setting explicit at the only boundary where it matters: the raw count table, before well totals and fractions exist.

### lines 8877-8879

```python
if settings['verbose']:
```

COUNTED AFTER THE MERGE, NOT HERE. See below -- the counts are taken from `merged_df`, which is what actually reached the fit. Counting `independent_df` here counted rows the inner merge was about to drop.

### lines 8884-8899

```python
merge_validate = (
```

The regression's own join, and the one whose cardinality decides every number this function goes on to report. independent_df is one row per (well, gRNA); what dependent_df is depends on agg_type:

agg_type in {'mean', 'median', 'quantile'} or poisson process_scores groups on prc, so it is exactly one row per well and this is many-to-one. A duplicated prc on that side would multiply every gRNA row of that well, inflating cell_count, the per-well gRNA counts and the regression's effective n, with no error and no visible symptom. agg_type is None (forced for quantile regression, see settings.py) process_scores returns one row per OBJECT, so the join is a deliberate cross product of the well's gRNAs with the well's cells. That is many-to-many and saying so explicitly is what stops a blanket 'many_to_one' here from crashing quantile regression on perfectly good data.

### lines 8908-8921

```python
_merged_for_counts, n_grna, n_gene = _count_variable_instances(
```

n_grna / n_gene DESCRIBE THE ROWS THAT REACHED THE FIT.

They were counted on `independent_df`, BEFORE this merge -- and the merge is an INNER join (no `how=`), so every sequencing well without an imaging partner was counted and then dropped. On the real screen that is 724 of 1,344 wells: "Paired 620 wells. 724 sequencing well(s) ... take no part in the regression." Measured on a synthetic case with half the wells unpaired, every count came out EXACTLY 2x too high.

It matters beyond the display. `min_observations_per_hit` filters the hit list on these significant[significant['n_grna'] > settings['min_observations_per_hit']] so an inflated count lets a guide through a filter it should fail which is a hit reported on evidence that is not there.

### lines 8938-8942

```python
cell_settings = {'src':data_path,
```

plot_data_from_csv reads settings['remove_outliers'] directly and never applies its own defaults, so omitting the key raised KeyError on the very first QC plot; combined with the swallow-everything try/except around this block, grna_well.csv and well_grna.csv were then silently never written.

### lines 8964-8976

```python
merged_df = merged_df[
```

.copy() IS LOAD-BEARING, not tidiness. Without it this is a slice of `merged_df`, and `grna_metricks` calls `_assign_prc_parts`, which ASSIGNS plateID/rowID/columnID onto the frame it is handed. Writing to a slice raises SettingWithCopyWarning -- which this suite promotes to an error (pytest.ini) and which pandas may in any case decline to write through -- and the exception was caught by the blanket `except` at the bottom of this QC block, so `grna_well.csv` and `well_grna.csv` were never written and the only trace was the warning text printed on its own line. Reproduced 2026-08-17: with outlier_detection=True the run completed and produced every regression output, and the two gRNA-coverage tables were simply absent.

### lines 9026-9027

```python
if str(settings.get('inference', 'parametric')).lower() == 'auto':
```

inference='auto' is decided here and not in settings.py, because it is the first point at which the guides and analysed wells can be counted.

### lines 9033-9041

```python
for _one in resolve_levels(settings.get('regression_type'),
```

The user chose the simultaneous fit. It is theirs to choose, and it runs -- but a fit with more parameters than wells returns one arbitrary solution out of infinitely many, and saying nothing is how a published figure came to carry coefficients that could not be reproduced from their own inputs. So: run it, and say so loudly. ONE CHECK PER FIT. `level='both'` runs two models of very different widths and only one of them may be too wide, so a single verdict for the run would either cry wolf about the gene fit or stay silent about the guide fit.

### lines 9051-9059

```python
_chosen = settings.get('regression_type')
```

SAID BEFORE THE FIT, not only in the summary afterwards. With inference='nonparametric' -- the default since 2026-08-18 -- the permutation path fits no model, so regression_type is never read. Verified on the maintainer's four-plate screen: 'ols' and 'mixed' produced byte-identical results, 1612 rows across all 24 columns. That is why "i ran a mixed model and an ols model and even if the ols model is marked as loaded i think i still see the mixed results" was a correct observation: they ARE the same numbers. A user who is told this before the run does not queue the second one.

### lines 9080-9090

```python
try:
```

THE SUMMARY, BEFORE THE EARLY RETURN. Instruction 156 placed its call at the end of the parametric path, which this branch never reaches -- so the ONE mode that has no statsmodels summary to fall back on was also the one mode that wrote no spaCR summary either, which is exactly the run the maintainer reported: "No summary: this run came back without a fitted model, so there is none to summarise", from a nonparametric mixed fit.

There is no `model` here and there never will be: a permutation test has no design matrix and no coefficient covariance. That is what the summary says, rather than being the reason it is absent.

### lines 9097-9102

```python
write_run_summary(
```

No `inference=` argument: `_is_nonparametric` reads it off the settings, which is the more robust answer -- an `inference='auto'` resolved into `analysis_mode`, and a settings CSV predating the `inference` key, both still come out right, where a keyword passed from here would only be right at this one call site.

### lines 9110-9119

```python
output.setdefault('res_folder', res_folder)
```

THE KEYS EVERY CONSUMER OF A RUN READS, on this branch too. `app_screen._on_regression_done` and the Measurements queue both take the run's folder from `res_folder`, and this early return was the one path that did not carry it -- so the DEFAULT inference produced a complete results folder that the GUI then registered with no folder at all, which is the "No summary: this panel was opened from a results table on disk" the maintainer reported. A copy of `settings`, for the same reason the parametric path hands one back: the shared settings/ file is overwritten by the next run of the same screen, so it describes the wrong one.

### lines 9125-9129

```python
if not _show_plates(merged_df, orig_dv, res_folder):
```

EVERY PLATE AS ONE FIGURE, on one colour scale, with square wells. The old call wrote one wide, short PDF per measurement into a fixed name -- so repeat runs overwrote each other, four plates took eight grid slots, and each plate got its OWN colour scale, which makes two plates incomparable at a glance. See spacr.figures.plates.

### lines 9135-9150

```python
_stage(settings, "fitting the model")
```

TWO FITS, NOT ONE DESIGN WITH BOTH LEVELS IN IT.

`gene_fraction` is the SUM of the gene's gRNA fractions

(check_and_clean_data), so the design this pipeline fitted until instruction 132 -- `fraction:grna + gene_fraction:gene + rowID columnID` -- contained a block of columns and its own sums. Measured on the maintainer's TSG101 screen: 1945 rows, 1248 parameters, RANK 862, an exact 386-dimensional null space, condition number 2.3e18. statsmodels pseudo-inverted it and reported a coefficient and a P value for every term; the residual sum of squares is bit-identical at the answer it gave and at that answer plus seven times a null vector. 102 single-guide genes came back as exact duplicates of their one guide -- 244480 and 244480_3 both 3.389291 at 2.873149e-13.

Split in two, each level is full rank: 859 parameters at rank 859 for the guide fit, 425 at 425 for the gene fit.

### lines 9155-9158

```python
regression_backend=settings.get('regression_backend',
```

WHO fits it (instruction 141). `.get`, not indexed, for the same reason as `model_plate_position` below: no settings CSV written before 2026-08-18 carries this key, and what every one of those files meant is the backend that produced them.

### lines 9164-9169

```python
model_plate_position=settings.get('model_plate_position', True),
```

IS PLATE POSITION IN THE MODEL AT ALL (instruction 143 A). `.get`, not indexed, for the same reason as the three below: no settings CSV written before 2026-08-18 carries this key, and the value an absent one meant is True -- layout adjustment is the backwards-compatible API behaviour. Fixed fits now include plateID explicitly as well as rowID and columnID.

### lines 9174-9175

```python
verbose=bool(settings.get('verbose')),
```

183: a quiet run gets the summary HEADER and a pointer at the file; verbose gets every coefficient, which is what verbose is for.

### lines 9177-9179

```python
transform=str(settings.get('transform') or ''),
```

182 A/C: what was already done to the response, so the family sniffer can name the scale it examined and refuse to be quiet about a link stacked on a transform.

### lines 9189-9194

```python
group_lasso_lambda=settings.get('group_lasso_lambda', 'auto'),
```

DEFAULTED HERE, not indexed. `group_lasso_lambda`, `rra_alpha` and `rra_permutations` are declared in spacr.settings, but a settings CSV written before instruction 133 has none of them and must still run -- and every one of these three is only read by the backend that names it, so its default is never the difference between two answers for any other type.

### lines 9198-9204

```python
qc=bool(settings.get('regression_qc', True)),
```

THE QC SUITE HAS TO BE DECLINABLE, and until this line it was not. `regression()` grew a `qc` parameter precisely so a parameter sweep could turn it off, and then nothing passed one -- so every trial of every sweep paid the full diagnostic suite: ~5.8 s and ~19 figures plus a combined PDF, i.e. roughly ten minutes and two thousand files per hundred trials, with no way to say no. On a single analysis it is exactly what you want, which is why it stays on by default.

### lines 9207-9210

```python
intercept=str(settings.get('intercept') or 'fitted'),
```

WHAT THE INTERCEPT IS. `.get`, not indexed, for the reason the three above give: no settings CSV written before this key existed carries it, and the value an absent one meant is a fitted intercept -- which is what every run before it did.

### lines 9216-9221

```python
settings['_regression_diagnostics'] = _write_regression_diagnostics(
```

THE DIAGNOSTICS, WRITTEN HERE BECAUSE THIS IS WHERE THE INPUTS ARE. `spacr.regression_diagnostics` has computed all of these since it was written -- after a fit that returned a confident P value for every one of 824 guides in 587 wells out of a rank-deficient matrix -- and until now nothing called it, so the checks that would have caught that failure were unreachable by a user. Instruction 322.

### lines 9230-9235

```python
if regression_type == 'mixed' and 'gene' in level_tables:
```

THE MIXED FIT IS ALREADY BOTH LEVELS, so it is split by TERM TYPE rather than fitted twice. Its gene rows are fixed effects with standard errors and p-values; its guide rows are BLUPs -- shrunken predictions of a random effect. A BLUP has no null hypothesis to reject, so it gets no q value here and no line in the hit list, and results_grna.csv from a mixed run says so in its `term_type` column.

### lines 9250-9253

```python
corrected = {}
```

EACH FIT CORRECTED WITHIN ITSELF. Two families, never one: same wells, and the gene regressor IS the sum of the guide regressors, so pooling would both break the independence the correction assumes and double the family for no protection.

### line 9259  _(unsure)_

```python
corrected[one] = table
```

BLUPs: no test, so nothing to correct and nothing to call.

### lines 9271-9273

```python
primary = 'grna' if 'grna' in fits else next(iter(fits))
```

THE PRIMARY LEVEL is the guide when there is one: the guide is the unit the screen measures, and it is what results.csv, the volcano's default and the model summary have always been about.

### lines 9278-9284

```python
grna_coef_df = corrected.get('grna')
```

ONE ROW PER GUIDE / PER GENE in the per-level files. The intercept and the mixed fit's variance components are terms of the fit, not units of the screen, and results_gene.csv has never carried them -- the results panel, `hits.load_results` and the volcano all read these files as a list of things that were tested. They stay in results.csv, which is the whole fit. (`n_grna` / `n_gene` are NaN exactly for the rows that name no unit, which is the same rule this used before the split.)

### lines 9291-9294

```python
template = corrected[primary].iloc[0:0]
```

A LEVEL THAT WAS NOT FITTED GETS AN EMPTY TABLE WITH THE RIGHT COLUMNS, not a missing file. `hits.load_results`, the results panel and `run_compare` all read results_gene.csv / results_grna.csv by name, and a file that is absent is indistinguishable from a run that crashed.

### lines 9321-9334

```python
if _annotation_source(settings):
```

EVERY EXPORTED TABLE CARRIES THE ANNOTATION, not just the volcano's colours. Instruction 133, asked for on 2026-08-17: "if it is on all the exported tables should be merged with the relevant Toxoplasma information".

Until this block `toxo=True` reached two places -- the volcano and two heatmaps -- and the CSV a reader actually opens came out as bare gene numbers and coefficients. The annotation was then joined by hand in a spreadsheet, which is where wrong-key mistakes live.

`spacr.annotation` declares every merge many_to_one and collapses each source to one row per gene first, so this cannot change a row count. It is checked anyway: this table's contract is one row per coefficient and it is worth being the kind of code that says so.

### lines 9360-9363

```python
supplementary(
```

The DeepTMHMM topology, as its own supplementary table: 72 columns of segment coordinates beside a coefficient is a table nobody opens twice, and "where does its third helix start" is a different question from "does this protein have a signal peptide".

### lines 9373-9377

```python
if settings['verbose']:
```

WRITTEN WHETHER OR NOT ANYBODY IS WATCHING. The save used to sit inside the `verbose` branch beside the print, so a quiet run -- the normal case -- left no summary on disk at all, and the results panel re-opened from that folder had nothing to read back. Printing is a console preference; the summary is part of the run's output.

### lines 9383-9393

```python
try:
```

THE spaCR SUMMARY, for EVERY mode -- instruction 156. The block above writes the statsmodels summary and only two of the supported regression types reach it; a nonparametric run has no fitted model at all, so it got nothing. This writes what spaCR itself knows about the fit -- the design, the assumptions with their tests, the call, and what was excluded -- so a mode statsmodels cannot summarise still has a summary.

GUARDED, and deliberately so: a run must not die for a summary. The module is optional at this point in its life, and a failure here is reported rather than raised, because losing an hour's fit to a reporting bug is the trade nobody would make.

### lines 9418-9428

```python
merged_df = tabular.read_table(results_path, report=None)
```

THE VOLCANO MUST NOT DEPEND ON HAVING A METADATA FILE.

These three names were bound ONLY inside the loop below. With no metadata file the loop never ran, and the toxo block -- which reads all three unconditionally -- raised NameError before drawing anything. The run otherwise completed, wrote its histograms, heatmaps and every results CSV, and simply produced no volcano: the one figure the module exists to make, missing with no error the user could see.

The results tables are the correct default. Metadata is an annotation join that adds columns; it is not what makes a volcano plottable.

### lines 9436-9443

```python
try:
```

AN UNREADABLE ANNOTATION FILE MUST NOT DESTROY A FINISHED FIT.

The regression is complete and written by this point; this loop only decorates the results with gene metadata. An empty or missing file here raised EmptyDataError straight out of perform_regression, so a perfectly good run was reported as a failure and its coefficients went unused -- which is what it looks like from a sweep, where one bad metadata path fails every trial that touches it.

### lines 9462-9469

```python
draw_legacy_volcano = bool(settings.get('legacy_volcano', False))
```

ONE BOOLEAN FOR EVERY OLD VOLCANO. "hide my old version behind a boolean that defaults to off" -- and the first attempt gated exactly one of the three call sites, the one a Toxoplasma screen never reaches. `toxo` defaults to TRUE, so `custom_volcano_plot` below is the picture the maintainer was still being shown after being told it was hidden.

Resolved ONCE, here, so the three branches cannot disagree, and read through the same key the fit already reads at the `regression()` call.

### lines 9483-9497

```python
gene_list = custom_volcano_plot(
```

THE GENE TABLE, ALWAYS. The `volcano` setting used to choose between the merged, gene and gRNA tables here, and it is GONE "remove the Volcano setting in regression, it is now redundant".

It is redundant because 129 A moved that choice onto the plot: the interactive volcano filters to genes or guides by right-click, on the SAME fit, with no re-run. A setting chosen before the run could only ever answer it once.

THIS CALL IS NOT ONLY A PICTURE, which is why the branch collapses to `gene` rather than disappearing. `custom_volcano_plot` also RETURNS the hit list that the GT1 phenotype plot and the ME49 transcription heatmap are built from, and those are gene-level reports -- so the gene table is the one they need, and it was already this setting's default.

### lines 9501-9508

```python
save_path=volcano_path, x_lim=settings.get('x_lim'),
```

`.get`, NOT `[...]`. Both keys are optional axis limits with a documented None meaning ("auto-scale", and [-0.5, 0.5] for x_lim), and `get_perform_regression_default_settings` does not put either of them in the dict -- so this raised `KeyError: 'x_lim'` from inside the Toxoplasma block, AFTER the fit, every results CSV and every QC panel had been written. A key that is absent and a key that is None mean the same thing to `custom_volcano_plot`, and neither is an error.

### lines 9514-9519

```python
if not draw_legacy_volcano:
```

SAY WHERE IT WENT. Every other artifact this module writes announces itself ("Saved regression data to ...", "Plot -> ..."), and the volcano -- the figure the module exists to produce -- was written silently. With nothing naming it, a run that had drawn one perfectly well was indistinguishable from a run that had drawn none, and was reported as "I can't see the regression plot".

### lines 9521-9523

```python
pass
```

Nothing was drawn, so nothing is claimed. A stale file left by an EARLIER run sits at this exact path, and reporting it would announce a figure this run did not make.

### lines 9535-9541

```python
metadata_files = list(settings.get('metadata_files') or [])
```

These two OPTIONAL reports need two specific curated tables -- a GT1 phenotype table and an ME49 expression table, in that positional order. Indexing [1] and [0] unconditionally meant a run with no metadata files died with `IndexError: list index out of range` AFTER the volcano had been drawn, so the run was reported as failed and the figure it had just produced looked like it was never made. They are extras; missing them is not a failure.

### lines 9549-9552

```python
data_GT1 = (tabular.read_table(metadata_files[1], low_memory=False,
```

canonicalise=False: these are curated third-party annotation tables whose headers are the vendor's ('Gene ID', 'sense - EES1'), not spaCR metadata, and the columns below are selected by those exact names.

### lines 9563-9565

```python
if gene_list and have_curated_tables:
```

The whole block was duplicated verbatim below this point: the same two reports were built twice, the second copy unguarded, so a run that survived the first died in the second. One copy, guarded.

### lines 9583-9591

```python
if not _toxoplasma_is_on(settings) and draw_legacy_volcano:
```

A VOLCANO IS NOT A TOXOPLASMA FEATURE.

Everything above sits under `if _toxoplasma_is_on(settings)`, because the compartment colouring needs the LOPIT table. But the volcano itself is the figure this module exists to produce, and gating it on an organism-specific flag meant a run with toxo=False wrote sixteen diagnostic figures and NOT the one the user came for -- silently, with nothing saying why. Drawn here without the compartment colouring, which is the only part that ever needed the metadata.

### line 9595  _(unsure)_

```python
_source = results_path_gene
```

The gene table, for the same reason as the toxo branch above.

### lines 9620-9628

```python
_warn_if_penalised_no_hits(settings, coef_df)
```

A PENALISED FIT THAT FINDS NOTHING IS NOT THE SAME AS NO SIGNAL.

ridge's p-values come from calculate_p_values, which divides an unpenalised standard error into a shrunken coefficient: conservative by construction, and deliberately so. On this screen every one of them comes back at q=1.0 -- including the two genes that OLS puts at q=2e-05. A user reading that sees "no hits" and cannot tell it apart from "no effect", which is the one conclusion the number does not support.

### lines 9631-9637

```python
try:
```

WHAT THE VOLCANO CANNOT SHOW.

A gene backed by ONE surviving guide and a gene whose guides all agree are the same single dot, and they rank by the same p-value -- but only one of them is independent evidence. On this screen the top of the list is a single-guide gene sitting above two genes with full guide support, so the ordering alone misleads about which hits to follow up.

### lines 9653-9662

```python
output = {'results':coef_df,
```

THE MODEL AND THE DESIGN COME BACK TOO.

Returning only the coefficient table meant every downstream consumer could report WHAT was significant and nothing about whether the fit deserved to be believed: no R-squared, no residuals to test for heteroscedasticity, no way to count the wells and guides that actually reached the design. A sweep row could say '10 hits' and not whether the run that produced them was well specified.

Both are already in scope here; they were simply dropped on the way out.

### lines 9669-9674

```python
'settings': dict(settings)}
```

THE SETTINGS THAT PRODUCED IT, so a caller offering to re-fit the same screen through a different model has the run's own dict rather than a file. The saved copy under settings/ is overwritten by every later run of the same screen, so on a second run it describes the wrong one. Copied, because the caller is a GUI and this dict is still being read here.

### lines 9677-9685

```python
manifest = getattr(coef_df, "attrs", {}).get("qc_manifest")
```

THE QC VERDICT, CARRIED OUT OF THE RUN (instruction 115). It was written to disk and nowhere else, so the dict a caller gets back could not say whether the fit it was holding is diagnosable -- and the manifest is the only thing in the run that knows: it carries the per-panel verdict, the WORST of them, and the renderer that drew each one.

`.attrs` off the coefficient frame, which is where `regression` put it, and absent rather than None when QC did not run: a key holding None is indistinguishable from a suite that ran and concluded nothing.

### lines 9689-9691

```python
worst = manifest.get('verdict')
```

The key the report writes is `verdict`, with `verdict_level` beside it. Both are lifted, because a caller wants the LEVEL to decide what to show and the verdict itself to say why.

## _perform_regression._stack

### lines 9305-9308

```python
def _stack(frames):
```

results.csv is BOTH tables stacked, each row carrying the `level` it was fitted and corrected at. One row per guide and one per gene, never a gene once per guide -- which is what the collinear single design produced and what put every gene on the volcano several times.

### lines 9311-9314

```python
kept = [frame for frame in frames if len(frame)]
```

pd.concat([]) raises "No objects to concatenate", and a run where neither level called a hit is an ordinary outcome, not an error it is what a screen with nothing in it looks like. The empty table keeps its columns so results_significant.csv still has a header.

## _assign_prcfo_parts

### lines 9778-9783

```python
parsed = [schema.parse_prcfo(value) for value in values.astype(str)]
```

The width check above answers "do these rows agree with each other?", which schema deliberately does not; schema.parse_prcfo answers "what is this one key?", which this used to do positionally. Parsing right to left is what makes the six-token form safe: the timepoint is optional and in the middle, so counting from the left puts the object id in 'timeID' and drops it.

## process_reads

### line 9826  _(unsure)_

```python
csv_df = tabular.read_table(csv_path)
```

Read the CSV file into a DataFrame

### lines 9834-9839

```python
if exclude_grnas and 'grna' in csv_df.columns:
```

EXCLUDE KNOWN CONTAMINANTS BEFORE FORMING THE DENOMINATOR.

Applying this after ``groupby('prc')['count'].sum()`` would leave the excluded reads in ``total_counts`` and depress every retained guide's fraction. The setting accepts either exact guide identifiers or a gene identifier, which resolves to every guide assigned to that gene.

### lines 9889-9894

```python
pieces = csv_df['plate_row'].astype(str).str.rsplit(
```

'plate_row' is '<plate>_<row>'. Split on the LAST separator rather than on every one: the plate is the component that may itself contain a separator ('exp1_plate1_r2'), the row is not, so counting from the right is the only reading that survives it. The two-column positional split this replaces raised the opaque "Columns must be same length as key" on such a plate.

### line 9932  _(unsure)_

```python
if not all(col in csv_df.columns for col in ['rowID','columnID','grna','count']):
```

Ensure the necessary columns are present

### line 9936  _(unsure)_

```python
csv_df['prc'] = _compose_prc_column(csv_df)
```

Create the prc column

### lines 9942-9946

```python
merged_df = pd.merge(csv_df, grouped_df, on='prc', validate='many_to_one')
```

grouped_df is one row per well by construction (groupby('prc')), csv_df is one row per (well, gRNA): many-to-one. The contract matters because the very next line divides by total_counts — a duplicated well total would duplicate every gRNA row of that well and the fractions would still sum to 1 per copy, so the corruption would be invisible in every downstream QC.

### lines 9950-9956

```python
if fraction_threshold is not None:
```

Filter rows with fraction under the threshold if fraction_threshold is not None: observations_before = len(merged_df) merged_df = merged_df[merged_df['fraction'] >= fraction_threshold] observations_after = len(merged_df) removed = observations_before - observations_after print(f'Removed {removed} observation below fraction threshold: {fraction_threshold}')

### lines 9973-9978

```python
if record is not None:
```

RECORDED, NOT ONLY PRINTED (instruction 156). The summary used to say "the run printed how many it removed and did not record it, so the count is in the console log and not in any file this summary can read" -- which was honest and is a gap rather than an answer. A console scrolls; a run somebody asks about tomorrow needs the number. Accumulated because this runs once per plate.

### lines 10002-10019

```python
tokens = merged_df['grna'].astype(str).str.split(schema.KEY_SEPARATOR)
```

This split IS positional, legitimately: the pooled-library naming convention is '<org>_<gene>_<guide>' ('TGGT1_GENEA_g1') and there is nothing in the name itself that says which token is which. So the assumption is stated and checked rather than removed.

What is removed is the bare `except Exception`. It made two very different inputs look identical from the outside: every name a single token ('g0', 'g1') — a library that simply has no org/gene structure. Three keys against one split column raised, and skipping is the right answer. names of mixed width ('TGGT1_GENEA_g1' next to 'GENEA_g1'). str.split(expand=True) pads with None instead of raising, so a short name got its GUIDE token as its gene and then grna=None out of the gene + '_' + guide concatenation — its reads were silently deleted from the screen while every long name sailed through. Requiring every name to have the same three components refuses the second case outright instead of half-applying to it.

## beta_logit

### lines 10067-10068

```python
inside = array[finite]
```

Only squeeze when an endpoint is actually present: a response already inside (0, 1) is left exactly as the user measured it.

## clean_controls

### lines 10132-10134

```python
columns = list(column) if isinstance(column, (list, tuple, set)) else [column]
```

A bare `column in df.columns` raised "TypeError: unhashable type: 'list'" for the list form that process_reads accepts and documents. Anything that is not a sequence of names stays a single name, as before.

## process_scores

### lines 10182-10185

```python
n_plates_in_df = df['plateID'].nunique(dropna=True) if 'plateID' in df.columns else 0
```

Only stamp a single plateID on every row when the caller asked for it AND the frame is single-plate (or has no plateID at all). For a multi-plate frame, ignore 'plate' so wells from different plates do not get collapsed to the same prc and silently averaged together by the groupby below.

### lines 10212-10216

```python
if invert_dependent_variable in (True, 1):
```

Optional inversion of the raw dependent variable, applied before aggregation and before any transform. False / 0 : no inversion True  / 1 : x -> 1 - x   (complement; for probability / score in [0, 1]) 1        : x -> 1 / x   (reciprocal; for rate- or time-like quantities)

### lines 10241-10246

```python
count_models = ('poisson', 'horseshoe')
```

Both count models take the well's SUM. 'horseshoe' is spaCRPower's Npositive ~ ... + offset(log(Ntotal)): its response is the number of positive objects in the well, not their mean, and the exposure it is offset by is the cell_count computed just below. Aggregating it like a continuous score would hand a Poisson model a fraction, which _validate_poisson_response refuses - loudly, but at the very end.

### lines 10271-10282

```python
summed = pd.to_numeric(dependent_df.get(dependent_variable),
```

REFUSED HERE, NOT AT THE END OF THE FIT. The comment above already says a continuous score hands a Poisson model a fraction and that `_validate_poisson_response` refuses it "loudly, but at the very end" -- and the end is after both input CSVs are read, the QC tables written and the diagnostic plots drawn. Measured on the maintainer's own run 2026-08-19: 19.2 seconds to be told that two settings could not both be honoured, which was checkable the moment the response was summed.

The well SUM is what these models fit, so that is the number to judge: a per-well sum of counts is an integer, and a per-well sum of a classification score is not.

### lines 10289-10297

```python
example = float(finite.iloc[0])
```

THE MESSAGE NAMES THE CAUSE, not the symptom. The one this replaces said "requires integer count data; use a continuous response model for fractional values" -- true, and it left the reader to work out WHY their counts were fractional. They are fractional because these models take the well's POSITIVE COUNT as the sum of a per-cell 0/1 label, and a classification SCORE is a probability: summing 152 cells at ~0.14 gives 21.68, which is not a count of anything.

### line 10310  _(unsure)_

```python
cell_count = grouped.size().reset_index(name='cell_count')
```

Calculate cell_count for all cases

### lines 10314-10317

```python
dependent_df = pd.merge(dependent_df, cell_count, on='prc',
```

No aggregation, so dependent_df is still one row per object and cell_count is one row per well: many-to-one. Stating it pins the thing that makes the unaggregated path safe — the well's cell count is broadcast onto its objects, never the other way round.

### lines 10333-10345

```python
if transform is not None and regression_type in count_models:
```

A COUNT MODEL'S RESPONSE MUST STAY A COUNT.

The sum above is deliberate -- Poisson and horseshoe model the number of positive objects in a well, not their average -- and then a transform was applied to it anyway. The default transform is 'log', so the integer count left here as a float and _validate_poisson_response refused it at the very END of a run that had already read both CSVs and fitted nothing. Neither count family could be started at all.

settings.py now clears `transform` for these families before the run, so this is the second line of defence -- and it is the one that covers a direct regression() or process_scores() call, which does not pass through the settings layer at all.

### lines 10352-10355

```python
column = pd.to_numeric(dependent_df[dependent_variable],
```

A logit is only defined on a proportion. Saying so BEFORE the fit is the difference between a wrong number and a stopped run: a response in raw intensity units transformed this way produces coefficients that look ordinary and mean nothing.

## generate_ml_scores

### lines 10486-10491

```python
from .training_basis import resolve_basis
```

The basis is now EXPLICIT. This used to read "if annotation_column is not None", which meant filling in an annotation column silently stopped the module training on plate controls, with nothing in the settings panel saying so. `resolve_basis` keeps that old rule as the fallback for a settings CSV with no `dataset_mode`, so an existing project runs exactly as it did -- see spacr.training_basis.

### lines 10507-10519

```python
_label_column = settings['annotation_column']
```

DERIVED, NOT WRITTEN BACK. This used to be settings['location_column'] = settings['annotation_column'] and that assignment mutated the CALLER'S settings dict -- a user-facing value, shown in the panel and saved with the project.

The mutation outlived the run. A user who tried annotation mode once and then switched dataset_mode back to 'metadata' still had `location_column` naming their annotation column, which is not in the measurement frame, so the next run died at `df[[location_ column]]` with a pandas KeyError that pointed nowhere near the cause. They could not get out by changing the mode; they had to know an invisible write had happened and undo it by hand. (Issues #91, #92, #93 -- one defect, walked through in sequence.)

### lines 10522-10526

```python
migrate_prediction_columns(db_loc[0])
```

Repair-on-read, the same contract utils.rename_columns_in_db has: a database written before the prediction columns were namespaced still carries the ML stage's scores under 'predictions', and is migrated here so the caller never has to do anything by hand. Skipped when the current name already exists.

### lines 10532-10539

```python
measurement_rows = len(df)
```

png_list can legitimately hold more than one crop per object — a database measured twice (cell crops, then pathogen crops) appends to the same table — so the annotation side is 'many'. The measurement side must not be: _read_and_merge_data groups on prcfo, so a repeat there means two source directories were concatenated under the same plate id and the same object identity now describes two different objects. That has to stop here rather than double every measurement row and quietly double the training set.

### lines 10555-10560

```python
if len(unique_values) < 2:
```

A BINARY CLASSIFIER NEEDS TWO OBSERVED CLASSES. The former one-class fallback randomly labelled unannotated objects as a made-up second class. That made the split run, but it changed unknown samples into ground truth and made every downstream metric scientifically false. Unannotated rows remain available for scoring after a real two-class model is trained; they are never promoted into training examples.

### lines 10585-10592

```python
from .utils import feature_selection
```

RECRUITMENT NEEDS EXACTLY ONE CHANNEL, and the setting can now name several, or a shape group, or nothing. `feature_selection` returns a bare int only for the one-channel case -- which is the only case in which "the pathogen's intensity over the cytoplasm's" names a number.

It used to read `settings['channel_of_interest'] in [0,1,2,3]`, so the panel's multi-select answer `[3]` -- the same feature space as the old `3` -- would have skipped recruitment silently.

### lines 10597-10600

```python
pathogen_col = f"pathogen_channel_{recruitment_channel}_mean_intensity"
```

`if "a" and "b" in df.columns` only membership-tests "b": the first operand is a non-empty literal and therefore always truthy. A measurements DB whose pathogen table lacks the channel mean intensity died with KeyError instead of skipping recruitment.

### lines 10613-10615

```python
batch_kwargs['batch_covariate_column'] = settings.get(
```

Added here rather than in `correction_kwargs` — see the note at its other call site. `ml_analysis` grew both parameters; the helper's other consumers did not.

### lines 10620-10622

```python
_training_column = _label_column or settings['location_column']
```

`_label_column` is set only on the annotation path and is what that path trains against; metadata runs use the caller's own setting. Either way `settings['location_column']` is left exactly as the user wrote it.

### lines 10677-10695

```python
from .figure_sink import publish
```

PUBLISHED, not merely saved -- instruction 139 C. `plot_permutation`, `plot_feature_importance` and `shap_analysis` all RETURN a figure and none of them shows it, and `shap_analysis` closes its own, so these four were written to the results folder and then never seen again by anybody running the app. `publish` writes through `spacr.plot.save_figure` exactly as before -- same file, same format preference -- and announces the figure as part of the same event.

The plate heatmap is the one that can arrive twice: `plot_plates` shows it itself when `verbose` is on. The bridge de-duplicates by figure, so it is one tile either way.

A FIGURE THAT WAS NEVER DRAWN IS NOT PUBLISHED, and `publish` is where that is decided. `ml_analysis` returns ``feature_importance_fig = None`` for every model without ``feature_importances_`` -- logistic regression and HistGradientBoostingClassifier, two of the offered `model_type_ml` values -- and the old `save_figure(figs[1], ...)` went straight into ``None.savefig`` and took the whole scoring run down AFTER the model had been fitted and every object scored.

### lines 10704-10720

```python
settings['csv_path'] = data_path
```

The model scored every object in every source database, so the scores belong back on every one of those databases -- not only in a CSV, and not only when a flag is set. The Annotate app, the active-learning queue and every GUI table read png_list, so a score that stops at results.csv is a score nothing downstream can see.

This replaces utils.add_column_to_database, which had three problems for this use: it re-read the CSV that was just written, it appended 'predictions_1', 'predictions_2', ... on every re-run instead of updating in place, and it replaced every 0 with a 2 (the Annotate app's class encoding) so the database disagreed with the CSV from the same run. merge_ml_predictions writes 'predictions' (the class, same column name as before) plus the new 'ml_pred' (the positive-class probability, which the ML stage never stored at all). Neither collides with the CV stage's 'cv_predictions' / 'pred', so running Classify (CV) and Classify (ML) over one database leaves four readable columns rather than two overwritten ones.

## _resolve_controls

### lines 10772-10776

```python
any_found = (matches(column, negative_control).any()
```

NEITHER may match before anything is derived. If ONE does, the user has a real partial match -- 'c1' present and 'c2' mistyped, say -- and deriving would silently replace the control they got RIGHT along with the one they got wrong. The refusal downstream names only the missing one, which is the useful message; overriding both would hide it.

### lines 10784-10785  _(unsure)_

```python
return negative_control, positive_control, False
```

Three or more classes, or one: the user has to say which two, and the refusal below will list what is there.

## ml_analysis._match_control_values

### line 10933  _(unsure)_

```python
try:
```

1. exact match

### line 10939  _(unsure)_

```python
try:
```

2. numeric match

### line 10948  _(unsure)_

```python
try:
```

3. stripped string match

## ml_analysis

### lines 10963-10967

```python
random_state = _run_random_state(42)
```

The run's seed, not a literal. Every estimator below takes this as random_state, and an estimator given an explicit random_state ignores the NumPy global stream -- so hard-coding 42 here silently overrode whatever the user set as random_seed. Falls back to 42 outside a run, which is what it always was.

### lines 10974-10979

```python
if location_column not in df.columns:
```

THE POISONED-SETTINGS SIGNATURE, NAMED RATHER THAN RAISED THROUGH. `df[[name]]` on a missing column raises a pandas KeyError from three frames down that says only "None of [Index([...])] are in the [columns]" -- issue #93. It points at the column and not at the reason the column is being asked for, which was an annotation-mode run that overwrote `location_column` and left it overwritten.

### lines 10984-10988

```python
raise ValueError(
```

The hint is unconditional because the cause is: any missing location_column reaching here is either a typo or the overwrite, and naming the overwrite costs a sentence while a user who cannot find it loses an afternoon. Phrased as a possibility, not a diagnosis, because a typo deserves the column list either way.

### lines 11000-11004

```python
if df.empty:
```

Name an empty measurement source before feature filtering turns it into an empty training set and the control guard misleadingly blames the configured control values.  Keep the missing-column diagnosis above: that remains the more actionable error when the requested column does not exist at all.

### lines 11011-11014

```python
location_values = df[location_column]
```

A populated table can still have no usable labels.  This is a data- population problem, not a typo in positive_control/negative_control. Do not handle duplicate columns here: ``df[name]`` is then a DataFrame, and the dedicated duplicate-column diagnosis below remains authoritative.

### lines 11059-11061

```python
df['prcfo'] = df.index.astype(str)
```

The merged measurement index is the canonical object identity. Keep it beside the filtered features now so duplicate indexes in annotation mode remain positionally aligned instead of being multiplied by .loc.

### lines 11070-11072

```python
negative_control, positive_control, _derived_classes = _resolve_controls(
```

Subset the dataframe based on specified column values if isinstance(negative_control, str): df1 = df[df[location_column] == negative_control].copy()

### lines 11074-11075  _(unsure)_

```python
negative_control, positive_control, _derived_classes = _resolve_controls(
```

elif isinstance(negative_control, list):

df1 = df[df[location_column].isin(negative_control)].copy()

### lines 11077-11080

```python
negative_control, positive_control, _derived_classes = _resolve_controls(
```

elif isinstance(negative_control, (int, float)):

df1 = df[df[location_column] == negative_control].copy() if verbose: print(f'Negative control: {negative_control}, samples: {len(df1)}')

### lines 11085-11086  _(unsure)_

```python
negative_control, positive_control, _derived_classes = _resolve_controls(
```

elif isinstance(positive_control, list):

df2 = df[df[location_column].isin(positive_control)].copy()

### lines 11088-11089  _(unsure)_

```python
negative_control, positive_control, _derived_classes = _resolve_controls(
```

elif isinstance(positive_control, (int, float)):

df2 = df[df[location_column] == positive_control].copy()

### lines 11094-11104

```python
negative_control, positive_control, _derived_classes = _resolve_controls(
```

THE CONTROLS MUST BE VALUES OF THE COLUMN BEING MATCHED. In annotation mode `location_column` is the ANNOTATION column, whose values are the class labels -- 1.0 and 2.0, say -- while positive_control and negative_control default to plate column names like 'c1' and 'c2'. Applying one to the other finds nothing, which is issues #91 and #92.

When the named controls appear nowhere in the column but it holds exactly TWO classes, those two ARE the classes: the lower value is the negative and the higher the positive. That is the ordinary annotation case and it should not require the user to restate what the column already says.

### line 11117  _(unsure)_

```python
df1['target'] = 0 # Negative control
```

Create target variable

### line 11118, trailing  _(unsure)_

```python
df1['target'] = 0
```

Negative control

### line 11119, trailing  _(unsure)_

```python
df2['target'] = 1
```

Positive control

### line 11121  _(unsure)_

```python
combined_df = pd.concat([df1, df2])
```

Combine the subsets for analysis

### lines 11128-11143

```python
untrained = []
```

A CLASS NOBODY NAMED IS STILL SCORED, AND THAT HAS TO BE SAID.

This fit is binary by construction: one arm is the negative control, the other the positive, and every remaining row of the input is scored afterwards by the model. That is the point of a screen -- the unknown population is what the scores are for -- but with THREE or more classes in the column it is easy to believe all of them were trained on. They were not, and nothing said so (instruction 236 D13).

Both controls take a LIST, so classes can be pooled into the two arms deliberately: positive_control=['c3', 'c4'] trains one arm on both. A SENTENCE MUST NOT BE ABLE TO BREAK A RUN. This is cosmetic, and it sits ABOVE the guard that refuses a table with two columns of one name -- where `df[location_column]` is a DataFrame and `.unique()` does not exist. It raised AttributeError there and masked the guard's own message, which is the one the user needed.

### lines 11164-11178

```python
if df1.empty or df2.empty:
```

REFUSE HERE, NAMING WHAT IS ACTUALLY IN THE COLUMN.

When neither control matches, df1 and df2 are both empty, combined_df is empty, and the failure surfaces as

ValueError: With n_samples=0, test_size=0.2 and train_size=None, the resulting train set will be empty

from inside sklearn's train_test_split, three frames below anything a user recognises. That traceback was auto-filed to the spaCR tracker TEN TIMES in one day (issues #79-#90) and names neither the setting that is wrong nor the value it should have had.

The verbose branch above would have said "samples: 0", but verbose is False on every shipped path.

### lines 11182-11186

```python
raise ValueError(
```

TWO COLUMNS OF THAT NAME. `df[name]` is then a DataFrame, every matching strategy in `_match_control_values` fails against it, and no control is ever found. Worth its own sentence: the fix is to the TABLE, not to the control values, and no amount of correcting positive_control will help.

### line 11217  _(unsure)_

```python
selected_features = X.columns[selector.get_support()]
```

Get the selected feature names

### lines 11231-11233

```python
from .classifier_evaluation import grouped_split, split_group_values
```

Split on an actual experimental unit. The index is the canonical prcfo in merged measurement frames, even when filtering removed its component metadata columns from X.

### lines 11238-11242

```python
held = holdout_plate
```

A NAMED HOLDOUT BEATS A RANDOM ONE. Cross-validation splits within the data it is given, so a model can learn the PLATE rather than the phenotype and every number it reports still looks fine. Naming a plate trains without it and scores on it, which is the one number that says whether the classifier generalises.

### line 11261  _(unsure)_

```python
combined_df['data_usage'] = 'train'
```

Add data usage labels

### line 11276  _(unsure)_

```python
if model_type == 'random_forest':
```

Initialize the model based on model_type

### line 11285, trailing  _(unsure)_

```python
model = HistGradientBoostingClassifier(max_iter=n_estimators, random_state=random_state)
```

Supports n_jobs internally

### lines 11311-11313

```python
model = CalibratedClassifierCV(
```

scikit-learn 1.9 deprecated SVC(probability=True). A calibrated decision-function SVC provides the same predict_proba contract without relying on the mode removed in 1.11.

### lines 11327-11329

```python
model.spacr_split_report_ = split_report.to_dict()
```

Estimators returned here can be persisted with joblib/pickle. Keeping the report on the object makes grouping provenance travel with such a model rather than existing only in stdout or the scored CSV.

### line 11360  _(unsure)_

```python
predictions_test = model.predict(X_test)
```

Predict for the current test set

### line 11372  _(unsure)_

```python
df.loc[X_test.index, 'predictions'] = predictions_test
```

Assign predictions and probabilities to the test set in the DataFrame

### line 11377  _(unsure)_

```python
fold_report = classification_report(
```

Evaluate performance for the current fold

### line 11387  _(unsure)_

```python
metrics_df = pd.concat(fold_metrics).groupby(level=0).mean()
```

Aggregate metrics across all folds

### line 11390  _(unsure)_

```python
model.fit(X, y)
```

Re-train on full data (X, y) and then apply to entire df

### line 11392, trailing  _(unsure)_

```python
all_predictions = model.predict(df[features])
```

Predict on entire df

### line 11395  _(unsure)_

```python
prediction_probabilities = model.predict_proba(df[features])
```

Get prediction probabilities for all rows in df

### lines 11404-11406

```python
else:
```

Generate metrics DataFrame final_report_dict = classification_report(y, all_predictions, output_dict=True) metrics_df = pd.DataFrame(final_report_dict).transpose()

### line 11410  _(unsure)_

```python
predictions_test = model.predict(X_test)
```

Predicting the target variable for the test set

### line 11422  _(unsure)_

```python
X_all = df[features]
```

Predicting the target variable for all other rows in the dataframe

### line 11427  _(unsure)_

```python
prediction_probabilities = model.predict_proba(X_all)
```

Get prediction probabilities for all rows in the dataframe

### lines 11445-11446  _(unsure)_

```python
metrics_df['split_group_by'] = split_report.group_by
```

``model_metrics.csv`` is the classical model's durable card. Repeat the scalar provenance on its rows so it survives CSV and remains filterable.

### lines 11452-11453

```python
perm_importance = permutation_importance(model, X_train, y_train, n_repeats=n_repeats, random_sta...
```

joblib workers are fresh threads, so they do not inherit the region's single-thread OpenMP clamp and re-enter the model with a full team.

### lines 11467-11472

```python
if hasattr(model, 'feature_importances_'):
```

Feature importance for models that support it. Use hasattr rather than a hardcoded model list: HistGradientBoostingClassifier (model_type= 'gradient_boosting') does NOT expose feature_importances_, so the old list-based check raised AttributeError. Models without the attribute (e.g. logistic_regression) fall through to the else branch, which must also define feature_importance_fig or the return raises UnboundLocalError.

### lines 11485-11498

```python
feature_importance_df = permutation_df.rename(
```

NO NATIVE IMPORTANCES IS NOT NO IMPORTANCES. Four of the nine models this module offers -- gradient_boosting, logistic_ regression, svm and mlp -- do not expose `feature_importances_`, and this branch used to hand back an empty frame and no figure. A user who picks logistic_regression, which the setting's own tooltip recommends as "a good linear sanity check", lost the feature-importance QC panel entirely and was told nothing.

THE PERMUTATION IMPORTANCE IS ALREADY COMPUTED, a few lines up, for every model, because it is model-agnostic by construction. It is a DIFFERENT QUANTITY from a tree's split-gain importance it measures what the fitted model loses when a column is shuffled so the panel says which one it is drawing rather than passing one off as the other.

### lines 11513-11515

```python
df = _assign_prcfo_parts(df, object_column='object')
```

Six tokens on a timelapse, five otherwise; see _assign_prcfo_parts. The five-name split raised ValueError here on every timelapse database, discarding a model that had already been fitted and scored.

## _shap_explainers

### lines 11576-11581

```python
automatic_note = ""
```

Older supported SHAP releases accept XGBoost here and silently choose their automatic tree explainer; newer releases reject the same categorical model and reach the explicit TreeExplainer below. In both cases say which semantics the panel used.  The note cannot be tied only to the fallback or identical runs become silent on the minimum dependency stack.

### lines 11594-11595  _(unsure)_

```python
try:
```

A tree whose splits are categorical. The library's own message names this remedy, and it is what xgboost needs.

### lines 11606-11609

```python
predict = (getattr(model, "predict_proba", None)
```

Not a tree and not linear -- a support vector machine, a neural net. The model-agnostic explainer takes a FUNCTION, not an estimator, and the background is summarised because it costs O(background) per explained row.

## shap_analysis

### lines 11653-11658

```python
if len(shap_values.shape) == 3:
```

TreeExplainer returns one output axis for every classifier class in recent SHAP releases: (samples, features, classes).  A 3-D input is interaction values to anything downstream, which both misrepresents the data and crashes when feature_names is a plain list.  The classifiers used by this pipeline are binary, so explain the positive class.  Keep the only output for estimators with a singleton axis.

### lines 11676-11679

```python
order = np.argsort(np.nanmean(np.abs(matrix), axis=0))[::-1]
```

RANKED BY MEAN ABSOLUTE CONTRIBUTION, which is the order the library uses and the only one that answers "which of these matters": a feature that pushes hard in both directions has a mean near zero and belongs at the top, not the bottom.

## write_plot

### lines 11724-11727

```python
publish_file(written, title=title or None)
```

BY NAME. `publish_file(path, title=None)` names its second parameter, and passing it positionally makes every caller that stands in for the sink -- a GUI bridge, a test double -- have to guess that the second positional is the tile's title.

## find_optimal_threshold

### lines 11739-11743

```python
denominator = precision + recall
```

A precision-recall sweep can contain points where precision and recall are both 0 (every predicted positive is a true negative). The plain 2*(p*r)/(p+r) produced NaN there, and np.argmax returns the index of the first NaN rather than the true F1 maximum, so the returned threshold could be one whose F1 is 0. F1 is 0 by definition when p + r == 0.

## _calculate_similarity

### line 11766  _(unsure)_

```python
if isinstance(val1, str):
```

Separate positive and negative control wells

### line 11776  _(unsure)_

```python
scaler = StandardScaler()
```

Standardize features for Mahalanobis distance

### line 11780

```python
cov_matrix = np.cov(scaled_features, rowvar=False)
```

Regularize the covariance matrix to avoid singularity

### line 11786  _(unsure)_

```python
epsilon = 1e-5
```

Add a small value to the diagonal elements for regularization

### line 11798  _(unsure)_

```python
try:
```

Calculate similarity scores

## _calculate_similarity.safe_similarity

### line 11790  _(unsure)_

```python
def safe_similarity(func, row, control, *args, **kwargs):
```

Calculate similarity scores

## _announce_the_bundle

### lines 11842-11844

```python
chosen = next((f for f in written if f.lower().endswith(".pdf")), None)
```

The preference names a format this bundle does not hold -- a bundle writes pdf and png whatever the preference says. Announce the vector one rather than nothing.

## _draw_the_cell_count_sweep

### lines 12009-12010

```python
plot.add_line(x=float(mark), colour=ROLES["reference"],
```

THE REFERENCE ROLE, NOT BLACK. A black guide line is invisible on spaCR's dark theme.

## _draw_importance_in_pyqtgraph

### lines 12096-12099

```python
if not plot.add_ranked_bars(list(shown['feature']),
```

THE HOUSE RULE: everything grey except what the sentence is about. The sentence here is "these are the features that matter", so the leading three carry the accent and the rest are the context they are being compared against.

## interpret_vision_model

### lines 12180-12184

```python
from .io import (_read_and_merge_data, _report_fan_out, JoinFanOut,
```

io._results_to_csv has the signature (src, df, df_well) and writes cells.csv / wells.csv; it was being called as (df, filename=...), which raised TypeError on every save=True run. The importance tables get their own writer, _save_importance_csv, which follows the same <src>/results convention.

### lines 12364-12371

```python
if settings['feature_importance'] or settings['permutation_importance'] or settings['shap']:
```

Step 1: Feature Importance using Random Forest

The outer guard used to read `feature_importance or feature_importance` — the same key OR'd with itself — so the forest was never fitted unless feature importance was explicitly requested. Permutation importance then hit UnboundLocalError on `model`, and SHAP on `feature_importance_df`, even though the docstring documents the three explainers as independent toggles. The forest and the importance frame are shared by all three; only the reporting and the CSV write belong to feature_importance itself.

### lines 12384-12388

```python
_draw_importance_in_pyqtgraph(
```

DRAWN IN PYQTGRAPH, not matplotlib. The tab and the file are one scene now, so the picture in a paper is the picture on screen. `add_ranked_bars` is what made this possible: twenty feature names need horizontal bars, and until it existed the only thing that could draw them was `plt.barh`.

### line 12399  _(unsure)_

```python
if settings['permutation_importance']:
```

Step 2: Permutation Importance

### line 12407

```python
_draw_importance_in_pyqtgraph(
```

PYQTGRAPH, for the reason given at the feature-importance chart.

### line 12418  _(unsure)_

```python
if settings['shap']:
```

Step 3: SHAP Analysis

### line 12428  _(unsure)_

```python
model = RandomForestClassifier(random_state=_run_random_state(42), n_jobs=settings['n_jobs'])
```

Refit the model on this subset of features

### line 12432  _(unsure)_

```python
if settings['shap_sample']:
```

Sample a smaller subset of rows to speed up SHAP

### lines 12434-12437

```python
sample = max(1, min(int(len(X_top) / 100), len(X_top)))
```

int(len/100) floors to 0 for any experiment with fewer than 100 objects, which handed shap an empty background AND an empty matrix to explain -> IndexError. Clamp to at least one row; for >=100 objects the clamp is a no-op.

### line 12443  _(unsure)_

```python
explainer = shap.Explainer(model.predict, X_sample)
```

Initialize SHAP explainer with the same subset of features

### lines 12447-12453

```python
_draw_shap_summary_in_pyqtgraph(
```

THE SUMMARY, IN PYQTGRAPH. `shap.summary_plot` draws into a matplotlib figure it makes itself, so it cannot be handed a pyqtgraph scene -- it was the last thing on this path keeping the second renderer alive. The chart is a beeswarm: one row per feature, every sample's contribution as a point, coloured by that sample's own value for the feature. `FastPlot.add_beeswarm` draws exactly that, so the saved file and the tab are one picture.

### line 12458  _(unsure)_

```python
shap_df = pd.DataFrame(shap_values.values, columns=X_sample.columns)
```

Convert SHAP values to a DataFrame for easier manipulation

### line 12461  _(unsure)_

```python
shap_df.columns = pd.MultiIndex.from_tuples(
```

Apply the function to create MultiIndex columns with compartment and channel

### line 12467  _(unsure)_

```python
shap_features = shap_df.abs().T
```

Aggregate SHAP values by compartment and channel

### line 12474  _(unsure)_

```python
combined_compartment = {}
```

Calculate combined importance for each pair of compartments and channels

### line 12487  _(unsure)_

```python
all_compartment_importance = list(compartment_mean.values) + list(combined_compartment.values())
```

Prepare values and labels for radar charts

## interpret_vision_model.create_extended_radar_plot

### line 12194  _(unsure)_

```python
def create_extended_radar_plot(values, labels, title):
```

Radar plot for individual and combined values, in pyqtgraph.

## interpret_vision_model.extract_compartment_channel

### line 12211  _(unsure)_

```python
compartment = feature_name.split('_')[0]
```

Identify compartment as the first part before an underscore

### line 12217  _(unsure)_

```python
channels = []
```

Identify channels based on substring presence

### line 12228  _(unsure)_

```python
if channels:
```

If multiple channels are found, join them with a '+'

### line 12232, trailing  _(unsure)_

```python
channel = 'morphology'
```

Use 'morphology' if no channel identifier is found

## interpret_vision_model.read_and_preprocess_data

### line 12248  _(unsure)_

```python
df['object_label'] = df['object_label'].str.replace('o', '')
```

Clean and align columns for merging

### lines 12251-12260

```python
join_cols = ['plateID', 'rowID', 'columnID', 'fieldID', 'object_label']
```

The join key is prcfo, spelled out as the columns it is made of the same key spacr.predictions uses to merge scores onto png_list, because this is the same question: which object is this crop?

The timepoint is part of that key. _read_and_merge_data returns one row per object PER FRAME, so joining a timelapse database without it matches every frame's object to every frame's score and multiplies the frame by the number of frames. (That used to be masked by _split_data dropping the timepoint from prcf on the way in, which collapsed the frames before they got here; it no longer does.)

### lines 12264-12269

```python
name_col = next((c for c in ('path', 'png_path', 'file_name')
```

A scores CSV written by apply_model_to_tar carries the crop file name, and the crop file name carries all of this -- so re-derive it with the writer's own parser rather than trusting the positional guess process_vision_results makes. On a timelapse crop (plate_well_field_time_object) that guess reads the TIMEPOINT as the object id, so its 'object' column is simply wrong there.

### line 12282  _(unsure)_

```python
df['object_label'] = df['object_label'].str.replace('o', '').astype(str)
```

Remove the 'o' prefix from 'object_label' in df, ensuring it is a string type

### line 12301  _(unsure)_

```python
df[join_cols] = df[join_cols].astype(str)
```

Ensure all join columns have the same data type in both DataFrames

### line 12305  _(unsure)_

```python
scores_df = scores_df[join_cols + [settings['score_column']]]
```

Select only the necessary columns from scores_df for merging

### lines 12308-12325

```python
try:
```

Now merge DataFrames.

The key contract is many-to-one — one score per object — and it is spelled out, because _report_fan_out does NOT enforce it here. That was the claim this comment used to make and it is false: the check is `len(merged) <= len(left)`, which is only equivalent to the cardinality contract for a LEFT join. This join is INNER, so scored objects fanning out and unscored objects dropping out cancel in the row count. Four objects, a scores file holding o1 twice and o2 once: the merge returns three rows, three <= four, nothing is raised, and o1's measurements are in the training set twice — the exact silent duplication the check was added to stop.

pandas is the thing that can actually see the duplicate key, so it does the checking; the message is translated back into the one _report_fan_out would have given, which names the cause (a scores file written twice) and the fix (de-duplicate it) instead of saying only "Merge keys are not unique in right dataset".

### lines 12344-12347

```python
_report_fan_out(df, merged_df, join_cols,
```

Belt and braces on the row count as well: many_to_one covers a duplicated scores key, this covers anything that would grow df for some other reason. The scores are per object, so the join can only ever shrink df (an object with no score drops out).

### lines 12351-12353

```python
X = schema.model_feature_frame(
```

Model inputs come from the measurement schema. Numeric identity and provenance columns (object_label, measurement_ndim, voxel sizes, etc.) are not biological features.

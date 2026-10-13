# Notes from `spacr/utils.py`

Prose lifted out of `spacr/utils.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (7 entries)
- [display](#display) (1 entry)
- [_LazyModule._load](#_lazymodule_load) (3 entries)
- [_merge_by_intensity](#_merge_by_intensity) (2 entries)
- [_filter_objects](#_filter_objects) (6 entries)
- [_organelle_diagnostic](#_organelle_diagnostic) (3 entries)
- [debug.decorator.wrapper](#debugdecoratorwrapper) (1 entry)
- [object_label_from_png_id](#object_label_from_png_id) (1 entry)
- [_one_object_label](#_one_object_label) (1 entry)
- [filepaths_to_database](#filepaths_to_database) (3 entries)
- [activation_correlations_to_database](#activation_correlations_to_database) (1 entry)
- [calculate_activation_correlations](#calculate_activation_correlations) (20 entries)
- [load_settings](#load_settings) (3 entries)
- [load_settings.parse_value](#load_settingsparse_value) (11 entries)
- [save_settings](#save_settings) (1 entry)
- [reset_mp](#reset_mp) (1 entry)
- [close_multiprocessing_processes](#close_multiprocessing_processes) (2 entries)
- [smooth_hull_lines](#smooth_hull_lines) (4 entries)
- [_outline_and_overlay.process_dim](#_outline_and_overlayprocess_dim) (4 entries)
- [_outline_and_overlay](#_outline_and_overlay) (4 entries)
- [_get_cellpose_batch_size](#_get_cellpose_batch_size) (3 entries)
- [_extract_filename_metadata](#_extract_filename_metadata) (1 entry)
- [_update_database_with_merged_info](#_update_database_with_merged_info) (4 entries)
- [_generate_representative_images](#_generate_representative_images) (4 entries)
- [_map_values](#_map_values) (2 entries)
- [normalize_to_dtype](#normalize_to_dtype) (1 entry)
- [_generate_names](#_generate_names) (2 entries)
- [_find_bounding_box](#_find_bounding_box) (4 entries)
- [_field_key_predicate](#_field_key_predicate) (1 entry)
- [_release_imported_rows_once](#_release_imported_rows_once) (4 entries)
- [_merge_and_save_to_database](#_merge_and_save_to_database) (5 entries)
- [_widen_table_for](#_widen_table_for) (3 entries)
- [_sqlite_value](#_sqlite_value) (1 entry)
- [_insert_frame](#_insert_frame) (1 entry)
- [_append_frame](#_append_frame) (2 entries)
- [_append_to_measurements_db](#_append_to_measurements_db) (1 entry)
- [_check_integrity](#_check_integrity) (1 entry)
- [_get_percentiles](#_get_percentiles) (6 entries)
- [_crop_center](#_crop_center) (8 entries)
- [_get_diam](#_get_diam) (1 entry)
- [_get_object_settings](#_get_object_settings) (1 entry)
- [_pivot_counts_table._read_table_to_dataframe](#_pivot_counts_table_read_table_to_dataframe) (1 entry)
- [_pivot_counts_table._pivot_dataframe](#_pivot_counts_table_pivot_dataframe) (2 entries)
- [_pivot_counts_table](#_pivot_counts_table) (3 entries)
- [annotate_conditions._map_or_default](#annotate_conditions_map_or_default) (4 entries)
- [annotate_conditions](#annotate_conditions) (3 entries)
- [_split_data](#_split_data) (6 entries)
- [_calculate_recruitment](#_calculate_recruitment) (1 entry)
- [EarlyFusion](#earlyfusion) (1 entry)
- [SpatialAttention](#spatialattention) (1 entry)
- [CustomCellClassifier](#customcellclassifier) (1 entry)
- [TorchModel.__init__](#torchmodel__init__) (10 entries)
- [TorchModel._get_weight_choice](#torchmodel_get_weight_choice) (1 entry)
- [TorchModel._init_base_model](#torchmodel_init_base_model) (2 entries)
- [TorchModel._remove_head_for_features](#torchmodel_remove_head_for_features) (9 entries)
- [TorchModel._infer_feature_dim](#torchmodel_infer_feature_dim) (3 entries)
- [TorchModel._run_backbone_raw](#torchmodel_run_backbone_raw) (3 entries)
- [TorchModel._run_backbone](#torchmodel_run_backbone) (1 entry)
- [TorchModel.forward](#torchmodelforward) (1 entry)
- [TorchModel_v2.__init__](#torchmodel_v2__init__) (7 entries)
- [TorchModel_v2._init_base_model](#torchmodel_v2_init_base_model) (2 entries)
- [TorchModel_v2._get_weight_choice](#torchmodel_v2_get_weight_choice) (1 entry)
- [TorchModel_v2._remove_head_for_features](#torchmodel_v2_remove_head_for_features) (1 entry)
- [TorchModel_v2._infer_feature_dim](#torchmodel_v2_infer_feature_dim) (1 entry)
- [TorchModel_v2._run_backbone](#torchmodel_v2_run_backbone) (1 entry)
- [TorchModel_v2.forward](#torchmodel_v2forward) (2 entries)
- [FocalLossWithLogits.forward](#focallosswithlogitsforward) (10 entries)
- [_list_torchvision_model_names](#_list_torchvision_model_names) (2 entries)
- [choose_model](#choose_model) (8 entries)
- [calculate_loss](#calculate_loss) (3 entries)
- [pick_best_model.sort_key](#pick_best_modelsort_key) (1 entry)
- [save_file_lists](#save_file_lists) (1 entry)
- [augment_single_image](#augment_single_image) (3 entries)
- [suggest_training_changes._normalize_cols](#suggest_training_changes_normalize_cols) (3 entries)
- [suggest_training_changes._poly_slope](#suggest_training_changes_poly_slope) (1 entry)
- [suggest_training_changes](#suggest_training_changes) (15 entries)
- [_infer_indices](#_infer_indices) (1 entry)
- [estimate_class_counts](#estimate_class_counts) (3 entries)
- [build_loss._infer_indices](#build_loss_infer_indices) (1 entry)
- [build_loss](#build_loss) (3 entries)
- [build_loss._auto_choice](#build_loss_auto_choice) (1 entry)
- [build_loss.loss_fn](#build_lossloss_fn) (1 entry)
- [annotate_predictions](#annotate_predictions) (1 entry)
- [add_images_to_tar](#add_images_to_tar) (1 entry)
- [generate_fraction_map](#generate_fraction_map) (2 entries)
- [fishers_odds](#fishers_odds) (9 entries)
- [model_metrics](#model_metrics) (3 entries)
- [lasso_reg](#lasso_reg) (5 entries)
- [MLR](#mlr) (5 entries)
- [get_files_from_dir](#get_files_from_dir) (1 entry)
- [create_circular_mask](#create_circular_mask) (2 entries)
- [apply_mask](#apply_mask) (3 entries)
- [invert_image](#invert_image) (1 entry)
- [match_masks](#match_masks) (1 entry)
- [pad_to_same_shape](#pad_to_same_shape) (1 entry)
- [compute_ap_over_iou_thresholds](#compute_ap_over_iou_thresholds) (2 entries)
- [dice_coefficient](#dice_coefficient) (4 entries)
- [boundary_f1_score](#boundary_f1_score) (4 entries)
- [_remove_noninfected](#_remove_noninfected) (1 entry)
- [_remove_outside_objects](#_remove_outside_objects) (4 entries)
- [_remove_multiobject_cells](#_remove_multiobject_cells) (3 entries)
- [merge_touching_objects](#merge_touching_objects) (8 entries)
- [remove_intensity_objects](#remove_intensity_objects) (3 entries)
- [_find_similar_sized_images](#_find_similar_sized_images) (11 entries)
- [_relabel_parent_with_child_labels](#_relabel_parent_with_child_labels) (11 entries)
- [_exclude_objects](#_exclude_objects) (6 entries)
- [_filter_object](#_filter_object) (1 entry)
- [_filter_cp_masks](#_filter_cp_masks) (1 entry)
- [_get_regex](#_get_regex) (1 entry)
- [_run_test_mode](#_run_test_mode) (5 entries)
- [_installed_cellpose_models](#_installed_cellpose_models) (1 entry)
- [_resolve_cellpose_pretrained](#_resolve_cellpose_pretrained) (2 entries)
- [_choose_model](#_choose_model) (1 entry)
- [SelectChannels.__call__](#selectchannels__call__) (3 entries)
- [SaliencyMapGenerator.compute_saliency_maps](#saliencymapgeneratorcompute_saliency_maps) (2 entries)
- [SaliencyMapGenerator.compute_saliency_and_predictions](#saliencymapgeneratorcompute_saliency_and_predictions) (2 entries)
- [SaliencyMapGenerator.plot_activation_grid](#saliencymapgeneratorplot_activation_grid) (4 entries)
- [SaliencyMapGenerator.percentile_normalize](#saliencymapgeneratorpercentile_normalize) (1 entry)
- [GradCAMGenerator.get_layer](#gradcamgeneratorget_layer) (1 entry)
- [GradCAMGenerator.compute_gradcam_maps](#gradcamgeneratorcompute_gradcam_maps) (4 entries)
- [GradCAMGenerator.compute_gradcam_and_predictions](#gradcamgeneratorcompute_gradcam_and_predictions) (1 entry)
- [GradCAMGenerator.plot_activation_grid](#gradcamgeneratorplot_activation_grid) (4 entries)
- [GradCAMGenerator.percentile_normalize](#gradcamgeneratorpercentile_normalize) (1 entry)
- [class_visualization](#class_visualization) (12 entries)
- [GradCAM.__call__](#gradcam__call__) (1 entry)
- [show_cam_on_image](#show_cam_on_image) (2 entries)
- [recommend_target_layers](#recommend_target_layers) (1 entry)
- [reduction_and_clustering](#reduction_and_clustering) (3 entries)
- [plot_embedding](#plot_embedding) (2 entries)
- [setup_plot](#setup_plot) (1 entry)
- [plot_clusters](#plot_clusters) (1 entry)
- [plot_images_by_cluster](#plot_images_by_cluster) (1 entry)
- [plot_image](#plot_image) (1 entry)
- [plot_clusters_grid](#plot_clusters_grid) (1 entry)
- [plot_grid](#plot_grid) (7 entries)
- [generate_path_list_from_db](#generate_path_list_from_db) (3 entries)
- [measure_test_mode](#measure_test_mode) (1 entry)
- [preprocess_data](#preprocess_data) (6 entries)
- [remove_highly_correlated_columns](#remove_highly_correlated_columns) (1 entry)
- [_resolve_missing_model_features](#_resolve_missing_model_features) (2 entries)
- [filter_dataframe_features](#filter_dataframe_features) (3 entries)
- [check_overlap](#check_overlap) (1 entry)
- [find_non_overlapping_position](#find_non_overlapping_position) (3 entries)
- [extract_features](#extract_features) (1 entry)
- [check_normality](#check_normality) (1 entry)
- [perform_statistical_tests](#perform_statistical_tests) (1 entry)
- [_merge_cells_without_nucleus](#_merge_cells_without_nucleus) (6 entries)
- [_merge_cells_based_on_parasite_overlap](#_merge_cells_based_on_parasite_overlap) (10 entries)
- [process_masks.read_files_in_batches](#process_masksread_files_in_batches) (1 entry)
- [process_masks.remove_objects_not_in_largest_cluster](#process_masksremove_objects_not_in_largest_cluster) (1 entry)
- [process_masks](#process_masks) (5 entries)
- [merge_regression_res_with_metadata](#merge_regression_res_with_metadata) (9 entries)
- [merge_regression_res_with_metadata.extract_and_clean_gene](#merge_regression_res_with_metadataextract_and_clean_gene) (3 entries)
- [process_vision_results](#process_vision_results) (3 entries)
- [get_ml_results_paths](#get_ml_results_paths) (1 entry)
- [augment_image](#augment_image) (6 entries)
- [augment_dataset](#augment_dataset) (3 entries)
- [convert_and_relabel_masks](#convert_and_relabel_masks) (2 entries)
- [get_cuda_version](#get_cuda_version) (1 entry)
- [prepare_batch_for_segmentation](#prepare_batch_for_segmentation) (1 entry)
- [map_condition](#map_condition) (1 entry)
- [download_models](#download_models) (7 entries)
- [generate_cytoplasm_mask](#generate_cytoplasm_mask) (2 entries)
- [add_column_to_database](#add_column_to_database) (9 entries)
- [fill_holes_in_mask](#fill_holes_in_mask) (4 entries)
- [rename_columns_in_db](#rename_columns_in_db) (1 entry)
- [group_feature_class](#group_feature_class) (3 entries)
- [cleanup_pipeline_folders](#cleanup_pipeline_folders) (2 entries)
- [delete_intermedeate_files](#delete_intermedeate_files) (2 entries)
- [filter_and_save_csv](#filter_and_save_csv) (2 entries)
- [extract_tar_bz2_files](#extract_tar_bz2_files) (2 entries)
- [calculate_shortest_distance](#calculate_shortest_distance) (4 entries)
- [format_path_for_system](#format_path_for_system) (4 entries)
- [normalize_src_path](#normalize_src_path) (5 entries)
- [generate_image_path_map](#generate_image_path_map) (4 entries)
- [copy_images_to_consolidated](#copy_images_to_consolidated) (3 entries)
- [remove_outliers_by_group](#remove_outliers_by_group) (2 entries)
- [generate_image_path_map, 2026-09-19](#generate_image_path_map-2026-09-19) (1 entry)
- [measure_test_mode, 2026-09-19](#measure_test_mode-2026-09-19) (1 entry)
- [process_mask_file_adjust_cell, 2026-09-19](#process_mask_file_adjust_cell-2026-09-19) (2 entries)
- [check_mask_folder, 2026-09-19](#check_mask_folder-2026-09-19) (1 entry)

## Module level

### line 7

```python
_trapezoid = getattr(np, 'trapezoid', None) or np.trapz
```

np.trapz was removed in numpy 2.0; np.trapezoid is the replacement.

### lines 63-64  _(unsure)_

```python
cp_models = _DeferredModule('cellpose.models')
```

Only _get_cellpose_model reads this proxy. Database, plotting and embedding callers of utils.py no longer import Cellpose (and its model stack) at all.

### line 74, trailing

```python
except ImportError:
```

scikit-image 0.22-0.24

### lines 112-114

```python
from .figures.style import figure_style, theme_target
```

THE HOUSE STYLE (136). `figures.style` imports matplotlib only inside its own functions, so naming it here costs nothing at import time.

### lines 463-465

```python
from . import schema, tabular
```

The one definition of what a spaCR database key is. Imported at module scope rather than lazily because every key built in this file goes through it and it costs nothing: schema.py is stdlib-only by design.

### lines 1210-1213

```python
**{role: f'{role}_id' for role in schema.ORGANELLE_ROLES},
```

'organelle' was missing, and _map_wells_png always returns an object id, so `columns` came out one short of `parts` and filepaths_to_database raised "Columns must be same length as key" -- AFTER the organelle PNGs were on disk but before any of them was registered in png_list.

### lines 10707-10709

```python
from .database_schema import (
```

These names remain public from ``spacr.utils`` for compatibility, while the authoritative definitions live beside the versioned migration that uses them.

## display

### lines 119-122

```python
def display(*args, **kwargs):
```

IPython may be mid-init (partially imported by another thread) — use a no-op fallback so importing this module never blocks. spaCR only calls display() from notebook contexts anyway; the Qt GUI ignores it.

## _LazyModule._load

### lines 340-343

```python
import sys as _sys
```

``sys.modules[root] = None`` is Python's explicit "this import is unavailable" sentinel. Respect it before inspecting distribution metadata: an explicitly blocked import is absent for this process, even if an old distribution happens to be present on disk.

### lines 358-360

```python
current = None
```

Let the real import below provide Python's normal missing package error; this check is specifically about an installed but unsupported version.

### lines 389-392

```python
self.__dict__['_module'] = None
```

An import can fail after populating several package children. Remove only entries created by this attempt; leaving them behind can turn the next attempt into a different, misleading failure.

## _merge_by_intensity

### lines 634-637

```python
boundaries = {}
```

EVERY BOUNDARY IS MEASURED EVEN WHEN NONE WILL MERGE, because the report is the point: a threshold that matches nothing has to be able to say what the boundaries actually were, or the user has no way to pick a better number except by guessing again.

### lines 664-667

```python
if merged == 0:
```

IT MUST BE ABLE TO REFUSE, AND SAY WHICH REFUSAL IT IS. A threshold above everything merges nothing and one below everything merges the field into one object; both were silent before, and both look like the setting having no effect rather than having far too much.

## _filter_objects

### line 793  _(unsure)_

```python
areas = {}
```

Pre-compute areas

### line 798  _(unsure)_

```python
removed_by_area = 0
```

Area filter

### line 814  _(unsure)_

```python
if remove_border:
```

Border filter

### lines 828-838

```python
total_removed = len(remove)
```

THE INTENSITY-PERCENTILE BAND IS GONE (391), and it was worse than merely relative. It dropped objects outside a quantile band of the FIELD'S OWN distribution, so it always removed roughly its share however bright the field: a 0/99 setting dropped the brightest object in every field whatever its intensity, and with two objects in a field it dropped one of them unconditionally. A filter that cannot decline to fire is not a filter, it is a quota.

Nothing replaces it here. Area and border remain, and intensity is now the merge step's business, where an absolute threshold in raw units can say what it matched and what it did not.

### line 840  _(unsure)_

```python
total_removed = len(remove)
```

Apply removal

### line 848, trailing  _(unsure)_

```python
remaining_count = len(np.unique(result)) - 1
```

exclude 0

## _organelle_diagnostic

### line 1082  _(unsure)_

```python
diag_img = img_norm.copy()
```

Draw blob circles on the normalised image

### line 1107  _(unsure)_

```python
radius = settings.get('organelle_tophat_radius', 5)
```

otsu / adaptive: show top-hat filtered image

### line 1136  _(unsure)_

```python
smooth = gaussian(img, sigma=1)
```

otsu / adaptive: show Gaussian smoothed

## debug.decorator.wrapper

### line 1171, trailing  _(unsure)_

```python
old_level = log.level
```

may be logging.NOTSET

## object_label_from_png_id

### lines 1263-1266

```python
return series.map(_one_object_label).astype(float)
```

map, not a vectorised .str: the column's dtype is whatever SQLite and pandas agreed on for the values that happen to be in it, and the point is to accept all of them. The index is preserved so a caller can line the result back up with the rows it came from.

## _one_object_label

### line 1284, trailing  _(unsure)_

```python
return np.nan
```

True is not object 1

## filepaths_to_database

### lines 1314-1320

```python
columns = columns + ['timeID']
```

'timeID', not 'time_id'. _merge_and_save_to_database writes 'timeID' onto every object table, so the old spelling gave one database two names for one concept: _split_data raised KeyError('timeID') on png_list and silently skipped building prcft, and any join between png_list and the cell table on time matched nothing. Databases already carrying 'time_id' are repaired in place on first read by rename_columns_in_db.

### line 1325  _(unsure)_

```python
if crop_mode in PNG_OBJECT_ID_COLUMNS:
```

Same column set as before, from the single mapping the readers use.

### lines 1331-1334

```python
_append_to_measurements_db(
```

Same per-field write as the measurement tables, so it gets the same treatment: a locked database is retried rather than dropping this field's crop rows, and a differing column set widens the table instead of refusing the whole frame. Both used to be swallowed by a print.

## activation_correlations_to_database

### line 1387  _(unsure)_

```python
png_df.set_index('file_name', inplace=True)
```

Align both DataFrames by file_name

## calculate_activation_correlations

### line 1418  _(unsure)_

```python
if manders_thresholds is None:
```

Ensure tensors are detached and moved to CPU before converting to numpy

### line 1427  _(unsure)_

```python
activation_maps = activation_maps.unsqueeze(1)  # Now shape is (batch_size, 1, height, width)
```

If activation maps have no channels, add a dummy channel dimension

### line 1428, trailing  _(unsure)_

```python
activation_maps = activation_maps.unsqueeze(1)
```

Now shape is (batch_size, 1, height, width)

### line 1432  _(unsure)_

```python
if (height != act_height) or (width != act_width):
```

Ensure that the inputs and activation maps are the same size

### line 1436  _(unsure)_

```python
correlations_dict = {'file_name': []}
```

Dictionary to collect correlation results

### line 1439  _(unsure)_

```python
for in_c in range(in_channels):
```

Initialize correlation columns based on input channels and activation map channels

### line 1447  _(unsure)_

```python
for b in range(batch_size):
```

Loop over the batch

### line 1449, trailing  _(unsure)_

```python
input_img = inputs[b]
```

Input image channels (C, H, W)

### line 1450, trailing  _(unsure)_

```python
activation_map = activation_maps[b]
```

Activation map channels (C, H, W)

### line 1452  _(unsure)_

```python
correlations_dict['file_name'].append(file_names[b])
```

Add the file name to the current row

### line 1455  _(unsure)_

```python
for in_c in range(in_channels):
```

Calculate correlations for each channel pair

### line 1457, trailing  _(unsure)_

```python
input_raw = input_img[in_c].flatten().numpy()
```

Flatten the input image channel

### lines 1462-1465

```python
finite = np.isfinite(input_raw) & np.isfinite(activation_raw)
```

Mask the two vectors JOINTLY. Filtering each independently dropped different positions from each, so the surviving elements no longer described the same pixels — pearsonr was correlating misaligned data (or raising on length mismatch).

### line 1470  _(unsure)_

```python
if input_channel.size > 0 and activation_channel.size > 0:
```

Check if there are valid (non-empty) arrays left to calculate the Pearson correlation

### line 1474, trailing  _(unsure)_

```python
pearson_corr = np.nan
```

Assign NaN if there are no valid data points

### line 1477  _(unsure)_

```python
for threshold in manders_thresholds:
```

Compute Manders correlations for each threshold

### line 1479  _(unsure)_

```python
if input_channel.size > 0 and activation_channel.size > 0:
```

Get the top percentile pixels based on intensity in both channels

### line 1484  _(unsure)_

```python
mask = (input_channel >= input_threshold) & (activation_channel >= activation_threshold)
```

Mask the pixels above the threshold

### line 1487  _(unsure)_

```python
if np.sum(mask) > 0:
```

If we have enough pixels, calculate Manders correlation

### line 1502  _(unsure)_

```python
df_correlations = pd.DataFrame(correlations_dict)
```

Convert the dictionary to a DataFrame

## load_settings

### lines 1548-1551

```python
df = tabular.read_table(csv_file_path, report=None)
```

ONE READER. A settings CSV is key/value so the vocabulary is a no-op on it, but the `~` and `$VAR` expansion is not: a settings file carried between machines routinely holds one, and it used to be a FileNotFoundError naming a path the user can see is right.

### line 1557  _(unsure)_

```python
if setting_key not in df.columns or setting_value not in df.columns:
```

Ensure the columns exist, in either of the two spellings spacr writes.

### line 1609  _(unsure)_

```python
result_dict = {key: parse_value(value) for key, value in zip(df[setting_key], df[setting_value])}
```

Convert the DataFrame to a dictionary, with parsing of each value

## load_settings.parse_value

### line 1570  _(unsure)_

```python
if pd.isna(value) or value == '':
```

Handle empty values

### lines 1574-1577

```python
if not isinstance(value, str):
```

Anything pandas already typed (int/float/bool from a numeric CSV column) is returned as-is. The string-only logic below calls value.startswith(...) unconditionally, which raised AttributeError on every non-str cell.

### line 1581  _(unsure)_

```python
if value == 'True':
```

Handle boolean values

### line 1587  _(unsure)_

```python
if value.startswith(('(', '[', '{')):  # If it starts with (, [ or {, use ast.literal_eval
```

Handle lists, tuples, dictionaries, and other literals

### line 1588, trailing  _(unsure)_

```python
if value.startswith(('(', '[', '{')):
```

If it starts with (, [ or {, use ast.literal_eval

### line 1596, trailing  _(unsure)_

```python
pass
```

If there's an error, return the value as-is

### line 1598  _(unsure)_

```python
try:
```

Handle numeric values (integers and floats)

### line 1601, trailing  _(unsure)_

```python
return float(value)
```

If it contains a dot, convert to float

### line 1602, trailing

```python
return int(value)
```

Otherwise, convert to integer

### line 1604, trailing  _(unsure)_

```python
pass
```

If it's not a valid number, return the value as-is

### line 1606  _(unsure)_

```python
return value
```

Return the original value if no other type matched

## save_settings

### lines 1790-1793

```python
try:
```

Persisting settings is a best-effort side effect — it must never crash the pipeline. A src that is missing / read-only / owned by another user (e.g. a settings CSV carried over from another machine) would otherwise raise PermissionError/OSError from makedirs and abort the whole run.

## reset_mp

### line 1910, trailing  _(unsure)_

```python
elif system in ('Linux', 'Darwin'):
```

Darwin is macOS

## close_multiprocessing_processes

### line 1943  _(unsure)_

```python
if proc.info['pid'] == current_pid:
```

Skip the current process

### line 1950, trailing

```python
proc.wait(timeout=5)
```

Wait up to 5 seconds for the process to terminate

## smooth_hull_lines

### line 2000  _(unsure)_

```python
vertices = hull.points[hull.vertices]
```

Extract vertices of the hull

### line 2002  _(unsure)_

```python
vertices = np.vstack([vertices, vertices[0, :]])
```

Close the loop

### line 2004  _(unsure)_

```python
tck, u = splprep(vertices.T, u=None, s=0.0)
```

Parameterize the vertices

### line 2006  _(unsure)_

```python
new_points = splev(np.linspace(0, 1, 100), tck)
```

Evaluate spline at new parameter values

## _outline_and_overlay.process_dim

### line 2047, trailing  _(unsure)_

```python
outline = np.zeros_like(mask, dtype=np.uint8)
```

Use uint8 for contour detection efficiency

### line 2049  _(unsure)_

```python
for j in np.unique(mask):
```

Find and draw contours

### line 2052, trailing  _(unsure)_

```python
continue
```

Skip background

### line 2054  _(unsure)_

```python
cv_contours = [np.flip(contour.astype(int), axis=1) for contour in contours]
```

Convert contours for OpenCV format and draw directly to optimize

## _outline_and_overlay

### lines 2060-2071

```python
outlines = [process_dim(mask_dim) for mask_dim in mask_dims]
```

Drawn on the CALLING thread, deliberately. This used to run in a ThreadPoolExecutor, which aborted the whole process -- SIGABRT, core dumped, no traceback -- once Qt and Tk had both been initialised earlier in the same session. cv2 and skimage's contour code are not safe to call off the main thread with two GUI toolkits resident, and there is nothing to catch: the process is simply gone, taking every result with it.

Giving the pool up cost nothing. There are at most three mask dimensions (cell, nucleus, pathogen) and find_contours holds the GIL throughout, so the threads were buying 3-5% -- measured on 3x60 objects at 1024px (1257 ms serial vs 1222 ms threaded) and 3x200 at 2048px (17.3 s vs 16.5 s). A 1.03x speedup is not worth a core dump.

### line 2074  _(unsure)_

```python
for i, outline in enumerate(outlines):
```

Overlay outlines onto the RGB image

### line 2079, trailing  _(unsure)_

```python
continue
```

Skip background

### line 2081, trailing  _(unsure)_

```python
overlayed_image[mask] = color
```

Direct assignment with broadcasting

## _get_cellpose_batch_size

### line 2127  _(unsure)_

```python
if torch.cuda.is_available():
```

Check if CUDA is available

### line 2130, trailing  _(unsure)_

```python
vram_gb = device_properties.total_memory / (1024**3)
```

Convert bytes to gigabytes

### lines 2134-2137

```python
if vram_gb < 8:
```

The bounds must form an exhaustive ladder: the previous

`> 8 and < 12` style left 8.0/12.0/24.0 GB unmatched, so batch_size was never assigned and the print below raised UnboundLocalError, which the bare except silently turned into a batch size of 8.

## _extract_filename_metadata

### lines 2186-2188

```python
well = match.group('wellID')
```

Undo zero padding so '001' and '1' are one key. _int_or_token keeps a token it cannot read instead of substituting '0': every unreadable well used to collapse onto well '0'.

## _update_database_with_merged_info

### line 2246  _(unsure)_

```python
if columns is None:
```

Connect to the SQLite database

### lines 2265-2269

```python
print('Merging on cell failed, trying with cell_id')
```

cell_id is the FALLBACK. Previously this second try ran unconditionally at the same indentation, so a successful object_label build was immediately overwritten — and when cell_id was absent the exception was merely printed, leaving prcfo built from the wrong column or missing entirely.

### line 2276  _(unsure)_

```python
try:
```

Merge the existing DataFrame with the new info based on the 'prcfo' column

### line 2289  _(unsure)_

```python
try:
```

Drop the existing table and replace it with the updated DataFrame

## _generate_representative_images

### line 2337  _(unsure)_

```python
df['new_measurement'] = (_compartment_column(compartments[0])
```

Two or more compartments: rank on the ratio between the first two.

### lines 2341-2345

```python
df['new_measurement'] = _compartment_column(compartments[0])
```

A single named compartment has no ratio partner, so rank on its own measurement. This branch used to be missing entirely: a one-element list satisfied the isinstance check, failed the len > 1 check and skipped the else, so 'new_measurement' was never created and _filter_closest_to_stat raised KeyError below.

### lines 2350-2351

```python
df['new_measurement'] = df['cell_area']
```

Unrecognised input (a bare string, None, anything else): fall back to a generic ranking rather than guessing which compartment was meant.

### lines 2364-2367

```python
fig = _plot_images_on_grid(png_paths_by_condition, [channel], um_per_pixel, scale_bar_length_um, ...
```

Pass the single-channel list inline. Rebinding channel_indices here mutated the list being iterated over AND the value used by every later condition, so only the first channel was ever rendered per-channel after the first condition.

## _map_values

### line 2372  _(unsure)_

```python
def _map_values(row, values, locs):
```

Adjusted mapping function to infer type from location identifiers

### line 2377  _(unsure)_

```python
type_ = 'rowID' if locs[0][0][0] == 'r' else 'columnID'
```

Determine if we're dealing with row or column based on first location identifier

## normalize_to_dtype

### line 2434  _(unsure)_

```python
img = rescale_intensity(img, in_range=(img_min, img_max), out_range=out_range)
```

Normalize to the range (0, 1) for visualization

## _generate_names

### lines 2483-2487

```python
img_name = f"{file_name}_{object_id_str}.png"
```

The final token is the CROPPED organelle label, not its parent cell. png_list stores it in ``<role>_id`` and joins it to that role's object table. The legacy implementation wrote the cell label here, so an organelle crop could be keyed to an unrelated organelle that happened to reuse the same integer label.

### lines 2492-2498

```python
raise ValueError(
```

Every caller reaches cv2.imwrite(os.path.join(fldr, img_name), ...). 'organelle' is a declared crop_mode -- settings.py lists it, validate.py allows it and measure.py has a branch for its mask but it had no branch HERE, so img_name stayed "" and OpenCV died with "could not find a writer for the specified extension", taking the whole field down after the measurements were already written. An empty name is never something to hand to a file writer.

## _find_bounding_box

### line 2522  _(unsure)_

```python
y_min, y_max = object_indices[0].min(), object_indices[0].max()
```

Determine the bounding box coordinates

### line 2526  _(unsure)_

```python
y_min = max(y_min - buffer, 0)
```

Add buffer to the bounding box coordinates

### line 2532  _(unsure)_

```python
new_mask = np.zeros_like(crop_mask)
```

Create a new mask with the same dimensions as crop_mask

### line 2535  _(unsure)_

```python
new_mask[y_min:y_max+1, x_min:x_max+1] = _id
```

Fill in the bounding box area with the _id

## _field_key_predicate

### lines 2705-2708

```python
return '0', []
```

An empty frame identifies no field, so it must match no row. The alternative -- ``()``, an empty OR -- is not valid SQL, and a predicate that fails to parse in a delete is a worse answer than one that selects nothing.

## _release_imported_rows_once

### line 2849, trailing  _(unsure)_

```python
return 0
```

no import ever wrote into this table

### line 2854, trailing  _(unsure)_

```python
return 0
```

claimed once, already handed back

### line 2872, trailing  _(unsure)_

```python
return 0
```

their rows are for other fields

### lines 2913-2914  _(unsure)_

```python
writer = connect(db_path, timeout=DB_WRITE_TIMEOUT)
```

Only now, and only for a table that really holds their copy of this field, is a write connection opened at all.

## _merge_and_save_to_database

### lines 2953-2956

```python
print(f"Warning: {table_type} has {len(morph_df)} morphology rows but an "
```

An object table with morphology but no intensity means the two measurement passes disagreed about which objects exist. Silently writing nothing lost a whole field's worth of objects with no trace, so say it out loud.

### line 2989, trailing  _(unsure)_

```python
cols = merged_df.columns.tolist()
```

get the list of all columns

### lines 2993-2996

```python
column_list = ['object_label'] + _META
```

A child table measured without a cell mask genuinely has no parent to link to. Since the fix in measure._intensity_measurements the link no longer depends on radial_dist, so reaching here means cell_mask_dim was None.

### line 3003, trailing  _(unsure)_

```python
merged_df = merged_df[cols]
```

rearrange the columns

### lines 3013-3017

```python
_release_imported_rows_for_field(
```

F34. A foreign import copies its rows into the canonical table when the destination is empty; appending beside them makes every downstream count the sum of two populations. The copy for this field is handed back first, or nothing is written -- both before the insert, never after.

## _widen_table_for

### line 3046, trailing  _(unsure)_

```python
if not have:
```

table does not exist yet; to_sql creates it

### line 3055  _(unsure)_

```python
if 'duplicate column name' not in str(e).lower():
```

Another worker widened the table for the same column between the

### lines 3057-3058

```python
if 'duplicate column name' not in str(e).lower():
```

the one we were about to add, so this is success, not failure and letting it escape would have cost the caller its whole frame.

## _sqlite_value

### line 3095  _(unsure)_

```python
return int(value.value)
```

Match pandas.to_sql: timedelta64 values are stored as nanoseconds.

## _insert_frame

### lines 3132-3134

```python
if isinstance(series.dtype, getattr(pd, 'ArrowDtype', ())):
```

pandas writes numpy-backed timedeltas in their native unit, but normalises Arrow-backed durations to nanoseconds.  Preserve both behaviours, including numpy's iNaT sentinel for missing values.

## _append_frame

### lines 3184-3185  _(unsure)_

```python
frame.iloc[:0].to_sql(
```

Schema only. Rows are written exactly once by the direct

INSERT on the next pass.

### lines 3193-3196

```python
if 'has no column named' not in message:
```

Widen ONLY when the append actually complains about a column. Probing PRAGMA table_info on every write cost a round trip per field per table and is pure waste on the overwhelmingly common path where the schema already matches.

## _append_to_measurements_db

### lines 3259-3262

```python
print(f"SQLite error writing {table}: {e}")
```

Not contention - an unopenable path, a read-only file. That is a setup problem, and the pre-existing contract is to report it and let the run continue; spacr.errors decides whether a run that lost a table is complete.

## _check_integrity

### lines 3448-3452

```python
raise ValueError(
```

object_label is read from label_list[0]; with no label column that list is empty and the old code died on IndexError with no indication of what was wrong. A measurement frame always carries one, and _merge_and_save_to_database merges the two frames on object_label, so arriving here without one means the wrong frame was passed.

## _get_percentiles

### line 3471, trailing  _(unsure)_

```python
if non_zero_img.size > 0:
```

check if there are non-zero values

### line 3472, trailing

```python
img_min = np.percentile(non_zero_img, p1)
```

change percentile from 0.02 to 2

### line 3473, trailing

```python
img_max = np.percentile(non_zero_img, p2)
```

change percentile from 0.98 to 98

### line 3475, trailing  _(unsure)_

```python
else:
```

if there are no non-zero values, just use the image as it is

### line 3476, trailing

```python
img_min = np.percentile(img, p1)
```

change percentile from 0.02 to 2

### line 3477, trailing

```python
img_max = np.percentile(img, p2)
```

change percentile from 0.98 to 98

## _crop_center

### line 3483  _(unsure)_

```python
cell_mask[cell_mask != 0] = 1
```

Convert all non-zero values in mask to 1

### line 3485, trailing  _(unsure)_

```python
mask_3d = np.repeat(cell_mask[:, :, np.newaxis], img.shape[2], axis=2).astype(img.dtype)
```

Create 3D mask

### line 3486, trailing  _(unsure)_

```python
img = np.multiply(img, mask_3d).astype(img.dtype)
```

Multiply image with mask to set pixel values outside of the mask to 0

### line 3487, trailing  _(unsure)_

```python
centroid = np.round(ndi.center_of_mass(cell_mask)).astype(int)
```

Compute centroid of the mask

### line 3489  _(unsure)_

```python
pad_width = max(new_width, new_height)
```

Pad the image and mask to ensure the crop will not go out of bounds

### line 3494  _(unsure)_

```python
centroid += pad_width
```

Update centroid coordinates due to padding

### line 3497  _(unsure)_

```python
start_y = max(0, centroid[0] - new_height // 2)
```

Compute bounding box

### line 3503  _(unsure)_

```python
img = img[start_y:end_y, start_x:end_x, :]
```

Crop to bounding box

## _get_diam

### lines 3535-3537

```python
raise ValueError(
```

Guard against unsupported object types — previously this fell through to ``int(diameter)`` with ``diameter`` unbound, raising a confusing UnboundLocalError instead of a clear message.

## _get_object_settings

### lines 3565-3567

```python
from .settings import normalize_cellpose_model_name
```

'cpsam' unless the user pointed at their own checkpoint; a pre-SAM name left in an old settings file is mapped forward here, once, rather than carried into segmentation as if it still chose different weights.

## _pivot_counts_table._read_table_to_dataframe

### line 3609  _(unsure)_

```python
return tabular.read_database(
```

Connect to the SQLite database

## _pivot_counts_table._pivot_dataframe

### line 3615  _(unsure)_

```python
pivoted_df = df.pivot(index='file_name', columns='count_type', values='object_count').reset_index()
```

Pivot the DataFrame

### lines 3617-3618

```python
pivoted_df = pivoted_df.fillna(0)
```

Because the pivot operation can introduce NaN values for missing data, you might want to fill those NaNs with a default value, like 0

## _pivot_counts_table

### line 3624  _(unsure)_

```python
pivoted_df = _pivot_dataframe(df)
```

Pivot the DataFrame to have one row per filename and a column for each object type

### line 3626  _(unsure)_

```python
conn = sqlite3.connect(db_path, timeout=30)
```

Reconnect to the SQLite database to overwrite the 'object_counts' table with the pivoted DataFrame

### line 3628  _(unsure)_

```python
pivoted_df.to_sql('pivoted_counts', conn, if_exists='replace', index=False)
```

When overwriting, ensure that you drop the existing table or use if_exists='replace' to overwrite it

## annotate_conditions._map_or_default

### line 3742  _(unsure)_

```python
df[column_name] = values
```

If a single string is provided and loc is None, assign the value to all rows

### line 3746  _(unsure)_

```python
df[column_name] = values[0]
```

If a list of values is provided but no loc, assign the first value to all rows

### line 3750  _(unsure)_

```python
value_dict = {val: key for key, loc_list in zip(values, loc) for val in loc_list}
```

Perform location-based mapping

### lines 3752-3755

```python
df[column_name] = pd.Series(np.nan, index=df.index, dtype=object)
```

Start with NaN, but in an object column: the labels written below are strings, and `df[column_name] = np.nan` produced a float64 column, so every .loc assignment was an incompatible-dtype set. pandas 2.x warns and silently upcasts; pandas 3.0 raises.

## annotate_conditions

### line 3762  _(unsure)_

```python
_map_or_default('host_cells', cells, cell_loc, df)
```

Handle cells, pathogens, and treatments using the consolidated logic

### lines 3767-3771

```python
if pathogens is not None:
```

Normalise any None left by the mapping above to np.nan, so the pd.notna() filter that builds 'condition' treats both the same. Plain reassignment, not chained inplace: under pandas copy-on-write (the 3.0 default) df[col].fillna(..., inplace=True) mutates a temporary and is a silent no-op.

### line 3777

```python
df['condition'] = df.apply(
```

Create the 'condition' column by excluding any NaN values, safely checking if 'host_cells', 'pathogen', and 'treatment' exist

## _split_data

### lines 3791-3798

```python
time_col = _time_column(df.columns)
```

Ensure 'prcft' column exists if a timepoint column is present.

This used to hard-code 'timeID' inside a bare try/except, so on the png_list table — which was written with 'time_id' — it printed "Exception 'timeID'" and silently produced no prcft at all. Asking which spelling is present makes the difference between "this is not a timelapse run" (nothing to build, no message) and a real failure (which now propagates instead of being printed and forgotten).

### lines 3810-3824

```python
try:
```

Ensure 'prcf' column exists.

The timepoint belongs in it. `_map_wells(timelapse=True)` — the writer that put prcf into the database in the first place — builds plate_row_column_field_TIME, and this rebuild used to drop that last component, overwriting the database's own key with a coarser one. Since prcfo is derived from prcf immediately below and is what callers group on, every object was then collapsed across all of its timepoints: a 2-field x 3-frame x 2-cell run came out of _read_and_merge_data as 4 rows with the three frames averaged together, and the caller's own time-carrying prcfo (io._read_and_merge_data assigns one from the database's prcf) was silently replaced on the way in. A timepoint column is written only by a timelapse run, so keying on it when it is present is the same condition as prcft above and leaves non-timelapse frames byte for byte as they were.

### line 3838  _(unsure)_

```python
df['prcfo'] = df['prcf'].astype(str) + '_' + df[object_type].astype(str)
```

Create the 'prcfo' column

### line 3842  _(unsure)_

```python
df_numeric = df.select_dtypes(include=np.number)
```

Split the DataFrame into numeric and non-numeric parts

### lines 3846-3865

```python
from .merge_tables import aggregation_for
```

HOW EACH COLUMN COMBINES, from the one place that decides it.

This used to be a second, independent implementation: a `sum_keywords` substring match, everything else averaged. It disagreed with `merge_tables` on three kinds of column, and each disagreement produced a number rather than an error

`object_label`     averaged. Three pathogens labelled 1, 2 and 3 came back as 2.0: a label for an object that need not exist, indistinguishable from a measurement. `count_*`          averaged. Counts add; a cell with 2 and 3 of something has 5 of it, not 2.5. `total_*`,         averaged. Something already integrated over an `integrated_*`     object is a total, and totals add.

AREAS SUM, LENGTHS DO NOT, which is the rule those keywords got right: four pathogens occupy the sum of their areas, but two nuclei each 10 units long are not one nucleus 20 units long. `aggregation_for` holds that rule now, matching on word boundaries rather than substrings, so the two readers cannot drift apart again.

### line 3871  _(unsure)_

```python
if len(agg_dict) > 0 and not df_numeric.empty:
```

Apply custom aggregation

## _calculate_recruitment

### lines 3924-3933

```python
return df
```

for chan in channels:

df[f'nucleus_coordinates_{chan}'] = df[[f'nucleus_channel_{chan}_centroid_weighted_local-0', f'nucleus_channel_{chan}_centroid_weighted_local-1']].values.tolist() df[f'pathogen_coordinates_{chan}'] = df[[f'pathogen_channel_{chan}_centroid_weighted_local-0', f'pathogen_channel_{chan}_centroid_weighted_local-1']].values.tolist() df[f'cell_coordinates_{chan}'] = df[[f'cell_channel_{chan}_centroid_weighted_local-0', f'cell_channel_{chan}_centroid_weighted_local-1']].values.tolist() df[f'cytoplasm_coordinates_{chan}'] = df[[f'cytoplasm_channel_{chan}_centroid_weighted_local-0', f'cytoplasm_channel_{chan}_centroid_weighted_local-1']].values.tolist()

df[f'pathogen_cell_distance_channel_{chan}'] = df.apply(lambda row: np.sqrt((row[f'pathogen_coordinates_{chan}'][0] - row[f'cell_coordinates_{chan}'][0])**2 (row[f'pathogen_coordinates_{chan}'][1] - row[f'cell_coordinates_{chan}'][1])**2), axis=1) df[f'nucleus_cell_distance_channel_{chan}'] = df.apply(lambda row: np.sqrt((row[f'nucleus_coordinates_{chan}'][0] - row[f'cell_coordinates_{chan}'][0])**2 (row[f'nucleus_coordinates_{chan}'][1] - row[f'cell_coordinates_{chan}'][1])**2), axis=1)

## EarlyFusion

### line 4044  _(unsure)_

```python
class EarlyFusion(nn.Module):
```

Early Fusion Block

## SpatialAttention

### line 4063  _(unsure)_

```python
class SpatialAttention(nn.Module):
```

Spatial Attention Mechanism

## CustomCellClassifier

### line 4115  _(unsure)_

```python
class CustomCellClassifier(nn.Module):
```

Final Classifier

## TorchModel.__init__

### line 4177, trailing  _(unsure)_

```python
num_classes: int = 2,
```

>=2 => multiclass head; ==1 => binary head (BCE)

### line 4178, trailing  _(unsure)_

```python
multilabel: bool = False,
```

kept for external loss/metrics decisions

### line 4179, trailing  _(unsure)_

```python
image_size: int = 224,
```

actual training resolution (ViT/inception need it)

### line 4205  _(unsure)_

```python
self.base_model = self._init_base_model(pretrained=bool(pretrained))
```

1) Initialize backbone

### line 4208  _(unsure)_

```python
if self.model_name == "maxvit_t" and hasattr(self.base_model, "classifier"):
```

2) Special-case: keep all but last linear block for MaxViT-T

### line 4210  _(unsure)_

```python
seq = list(self.base_model.classifier.children())
```

remove final Linear only (keep preceding norm/dropout/etc.)

### line 4215  _(unsure)_

```python
if dropout_rate is not None:
```

3) If a custom dropout rate is provided, push it into any existing Dropout modules

### line 4219  _(unsure)_

```python
self._remove_head_for_features()
```

4) Remove the original classification head so we can infer feature dim

### line 4222  _(unsure)_

```python
self.num_ftrs = self._infer_feature_dim()
```

5) Infer flattened feature dimension with a dummy forward

### line 4225  _(unsure)_

```python
if self.use_dropout:
```

6) Build spaCR head (optional dropout + linear classifier)

## TorchModel._get_weight_choice

### lines 4230-4232  _(unsure)_

```python
def _get_weight_choice(self):
```

Backbone init / head removal / feature dim

## TorchModel._init_base_model

### line 4263  _(unsure)_

```python
return fn(weights=weights if pretrained else None)
```

Newer API

### line 4266  _(unsure)_

```python
return fn(pretrained=pretrained)
```

Older API fallback

## TorchModel._remove_head_for_features

### line 4284  _(unsure)_

```python
if hasattr(self.base_model, "aux_logits"):
```

Some models (Inception/GoogLeNet) expose aux heads

### line 4288  _(unsure)_

```python
if hasattr(self.base_model, "fc"):           # ResNet/RegNet/ResNeXt/GoogLeNet/Inception
```

Common conv backbones

### line 4289, trailing  _(unsure)_

```python
if hasattr(self.base_model, "fc"):
```

ResNet/RegNet/ResNeXt/GoogLeNet/Inception

### line 4292, trailing  _(unsure)_

```python
if hasattr(self.base_model, "classifier"):
```

DenseNet/MobileNet/EfficientNet/ConvNeXt/SqueezeNet/MNASNet/MaxViT

### line 4293  _(unsure)_

```python
if self.model_name != "maxvit_t":
```

MaxViT handled earlier; here we blank the whole thing

### line 4297, trailing  _(unsure)_

```python
if hasattr(self.base_model, "_fc"):
```

Older EfficientNet

### line 4300  _(unsure)_

```python
if hasattr(self.base_model, "heads"):        # ViT (torchvision)
```

Vision Transformers

### line 4301, trailing  _(unsure)_

```python
if hasattr(self.base_model, "heads"):
```

ViT (torchvision)

### line 4304, trailing  _(unsure)_

```python
if hasattr(self.base_model, "head"):
```

Swin

## TorchModel._infer_feature_dim

### line 4307  _(unsure)_

```python
def _infer_feature_dim(self) -> int:
```

If none matched, we’ll still try to forward and flatten later.

### line 4318, trailing  _(unsure)_

```python
out = self._run_backbone_raw(x)
```

raw backbone call (unwrapped)

### line 4319  _(unsure)_

```python
if isinstance(out, torch.Tensor) and out.ndim > 2:
```

Flatten if spatial

## TorchModel._run_backbone_raw

### lines 4345-4346  _(unsure)_

```python
if hasattr(out, "logits"):
```

Unwrap common container types

Inception* returns namedtuple with .logits (if aux disabled we still may get a container)

### line 4350  _(unsure)_

```python
out = out[0]
```

e.g., some models return (logits, aux) even when aux disabled; take primary

### line 4353  _(unsure)_

```python
raise RuntimeError(
```

Detection/segmentation heads return dicts — not supported in this wrapper

## TorchModel._run_backbone

### line 4371  _(unsure)_

```python
if isinstance(out, torch.Tensor) and out.ndim > 2:
```

Ensure 2D features (N, F)

## TorchModel.forward

### line 4384, trailing  _(unsure)_

```python
logits = self.spacr_classifier(feats)
```

(N, num_classes)

## TorchModel_v2.__init__

### line 4404, trailing  _(unsure)_

```python
num_classes: int = 2,
```

arbitrary classes (>=2 => multiclass; 1 => binary head)

### line 4405, trailing  _(unsure)_

```python
multilabel: bool = False
```

kept for external loss/metrics decisions (not used internally)

### line 4418  _(unsure)_

```python
self.base_model = self._init_base_model(pretrained)
```

1) init backbone

### line 4421  _(unsure)_

```python
if self.model_name == "maxvit_t" and hasattr(self.base_model, "classifier"):
```

2) special-case: keep all but the last linear block for maxvit_t

### line 4427  _(unsure)_

```python
if dropout_rate is not None:
```

3) apply custom dropout rate to any existing dropout modules in backbone

### line 4431  _(unsure)_

```python
self.num_ftrs = self._infer_feature_dim()
```

4) discover feature dim

### line 4434  _(unsure)_

```python
self._init_spacr_classifier(dropout_rate)
```

5) add spaCR head

## TorchModel_v2._init_base_model

### line 4463  _(unsure)_

```python
return fn(weights=weights if pretrained else None)
```

Newer torchvision API: weights=enum or None

### line 4466  _(unsure)_

```python
return fn(pretrained=bool(pretrained))
```

Older API fallback: pretrained=bool

## TorchModel_v2._get_weight_choice

### line 4470  _(unsure)_

```python
"""The torchvision ``DEFAULT`` weights enum for this model, or ``None``.
```

Return DEFAULT weights enum if available; else None

## TorchModel_v2._remove_head_for_features

### line 4482  _(unsure)_

```python
"""Replace the classifier head with identity so the backbone returns features.
```

Remove final classifier so backbone returns features

## TorchModel_v2._infer_feature_dim

### line 4505  _(unsure)_

```python
if out.ndim > 2:
```

If backbone returns spatial map, flatten to (N, C*)

## TorchModel_v2._run_backbone

### line 4525  _(unsure)_

```python
"""Run the backbone, through gradient checkpointing when enabled.
```

Wrap for checkpoint (expects a function)

## TorchModel_v2.forward

### line 4542  _(unsure)_

```python
if feats.ndim > 2:
```

Ensure 2D features (N, F)

### line 4547, trailing  _(unsure)_

```python
logits = self.spacr_classifier(feats)
```

(N, C) where C==num_classes

## FocalLossWithLogits.forward

### line 4577  _(unsure)_

```python
if logits.ndim == 1 or logits.size(-1) == 1 or (
```

Binary / multilabel (BCE-style)

### line 4584, trailing  _(unsure)_

```python
pt = target * p + (1 - target) * (1 - p)
```

pt = p if y=1 else (1-p)

### line 4587  _(unsure)_

```python
if target.dtype != torch.long:
```

Multiclass CE-style: logits (N,C), target (N,) long

### line 4590, trailing  _(unsure)_

```python
logp = F.log_softmax(logits, dim=1)
```

(N,C)

### line 4591, trailing  _(unsure)_

```python
p = torch.exp(logp)
```

(N,C)

### line 4592  _(unsure)_

```python
pt = p.gather(1, target.unsqueeze(1)).squeeze(1)  # (N,)
```

gather the prob of the true class

### line 4593, trailing  _(unsure)_

```python
pt = p.gather(1, target.unsqueeze(1)).squeeze(1)
```

(N,)

### line 4594, trailing  _(unsure)_

```python
ce = F.nll_loss(logp, target, reduction="none")
```

per-sample CE

### line 4596  _(unsure)_

```python
alpha = self.alpha.to(logits.device)[target]   # (N,)
```

class-wise alpha

### line 4597, trailing  _(unsure)_

```python
alpha = self.alpha.to(logits.device)[target]
```

(N,)

## _list_torchvision_model_names

### lines 4758-4774

```python
def _list_torchvision_model_names() -> set[str]:
```

def print_model_summary(base_model, channels, height, width): """ Prints the summary of a given base model.

Args:

base_model (torch.nn.Module): The base model to print the summary of. channels (int): The number of input channels. height (int): The height of the input. width (int): The width of the input.

Returns:

None """ device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") base_model.to(device) summary(base_model, (channels, height, width)) return

### lines 4797-4798  _(unsure)_

```python
return {
```

Older torchvision: the factories are lower_snake_case and the classes and weights enums are not.

## choose_model

### lines 4858-4861

```python
print(
```

NOT `end="\r"`. A carriage return with no newline leaves the cursor at the start of THIS line, so whatever is printed next overwrites it and in a captured log the two run together as "use_checkpoint: FalsePASS". The banner is one line of its own.

### line 4868

```python
if model_type == "custom":
```

CUSTOM BRANCH

### line 4875

```python
head_dim = max(1, int(num_classes))
```

TORCHVISION CLASSIFICATION (via your TorchModel wrapper)

### lines 4877-4878  _(unsure)_

```python
img_size = int(height) if height else 224
```

Use the real training resolution so ViT/Swin/inception (which are resolution-sensitive) infer the right feature dim + pass the sanity check.

### line 4880, trailing  _(unsure)_

```python
base_model = TorchModel(
```

relies on your wrapper class being available in this module

### line 4889  _(unsure)_

```python
try:
```

Forward sanity-check to ensure classification logits shape

### line 4893  _(unsure)_

```python
dummy = torch.randn(1, 3, img_size, img_size)
```

Keep 3 channels for sanity-check; most pretrained backbones expect 3

### lines 4903-4905

```python
raise ValueError(
```

ALSO RAISES. A backbone that builds and does not produce logits is a broken model, not a missing one, and returning None sent it down the same silent path as a misspelled name.

## calculate_loss

### line 4961, trailing  _(unsure)_

```python
output = output.unsqueeze(1)
```

(N,) -> (N,1)

### line 4973  _(unsure)_

```python
if prefer_focal:
```

Multiclass single-label with class indices (N,)

### line 4978  _(unsure)_

```python
if target.ndim == 1:
```

Multilabel (assume float/one-hot), ensure (N,C)

## pick_best_model.sort_key

### lines 5028-5029

```python
pass
```

A broken candidate ranks last; loading the selected artifact will still report the actual corruption rather than hiding it.

## save_file_lists

### lines 5064-5066

```python
tabular.write_table(df, f'{dst}/{data_set}.csv', canonicalise=False)
```

canonicalise=False: the single column is NAMED by the caller's `data_set`, so the vocabulary would be renaming an identifier the caller chose rather than a metadata column somebody spelled loosely.

## augment_single_image

### line 5085

```python
img_rot_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
```

90 degree rotation

### line 5089

```python
img_rot_180 = cv2.rotate(img, cv2.ROTATE_180)
```

180 degree rotation

### line 5093

```python
img_rot_270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
```

270 degree rotation

## suggest_training_changes._normalize_cols

### line 5168  _(unsure)_

```python
m = {c: c.strip().lower() for c in df.columns}
```

Lowercase and strip; map common variants

### lines 5172-5175

```python
df = df.loc[:, ~df.columns.duplicated(keep='first')]
```

FIX: drop duplicate columns — keeps the first occurrence

This happens when _save_progress appends with headers repeatedly, or when the same metric appears under multiple names that alias to the same canonical name after normalization

### lines 5193-5194  _(unsure)_

```python
df = df.loc[:, ~df.columns.duplicated(keep='first')]
```

FIX: deduplicate again after aliasing — two different original names (e.g. "acc" and "accuracy") can both map to "accuracy"

## suggest_training_changes._poly_slope

### line 5204  _(unsure)_

```python
mask = np.isfinite(y)
```

robust to NaNs: drop them

## suggest_training_changes

### line 5236  _(unsure)_

```python
for col in ("epoch", "loss"):
```

Required columns (soft-fail if absent)

### lines 5243-5247

```python
best_pos = int(va["loss"].argmin())
```

core scalars idxmin returns an index LABEL; .loc on a duplicated or non-RangeIndex then returns a Series rather than a scalar (which is why _scalar exists to paper over it) and best_epoch could come from the wrong row. Use the positional argmin with .iloc so the row is unambiguous.

### line 5272  _(unsure)_

```python
val_mean = float(np.nanmean(va_last)) if len(va_last) else np.nan
```

noise/instability

### line 5282  _(unsure)_

```python
f1_nan_train = "f1_macro" in tr.columns and np.isnan(tr["f1_macro"]).mean() > 0.2
```

macro-F1 NaN detection (common when a split has a single label)

### line 5286  _(unsure)_

```python
since_best = int(tr.shape[0] - (best_pos + 1))
```

improvement since best

### line 5308  _(unsure)_

```python
if E < min_epochs:
```

1) Too early to judge

### line 5312  _(unsure)_

```python
if len(va_last) >= max(5, last_k//2) and abs(slope_va) < plateau_eps:
```

Still continue to surface other obvious issues below.

### line 5314  _(unsure)_

```python
if len(va_last) >= max(5, last_k//2) and abs(slope_va) < plateau_eps:
```

2) Plateau (no meaningful val loss improvement recently)

### line 5323  _(unsure)_

```python
overfit_like = False
```

3) Overfitting (train improving, val degrading, or large accuracy gap)

### line 5338  _(unsure)_

```python
train_acc_low = ("accuracy" in tr.columns and final.get("train_accuracy", 0.0) < 0.70)
```

4) Underfitting (both losses high; train acc low and no decreasing trend)

### line 5349  _(unsure)_

```python
if unstable:
```

5) Unstable training (high variance in recent val loss)

### line 5357  _(unsure)_

```python
if f1_nan_train or f1_nan_val:
```

6) F1 NaNs (often single-class in split/batch or metric bug)

### line 5365

```python
if since_best >= max(5, last_k//2) and val_loss_delta_from_best > plateau_eps:
```

7) Regressed after best

### line 5373  _(unsure)_

```python
if ("accuracy" in va.columns and "f1_macro" in va.columns
```

8) If accuracy present but macro-F1 << accuracy -> imbalance hint

### line 5384  _(unsure)_

```python
out["suggestions"] = list(dict.fromkeys(out["suggestions"]))
```

De-duplicate while preserving order. ``dict`` preserves insertion order.

## _infer_indices

### line 5395  _(unsure)_

```python
return (target.view(-1) > 0.5).long()
```

binary float → {0,1}

## estimate_class_counts

### line 5411

```python
if src is not None and classes is not None:
```

fast path: count files on disk instead of loading images

### line 5417  _(unsure)_

```python
n = sum(1 for f in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir, f)))
```

count only files, skip subdirectories

### line 5423  _(unsure)_

```python
print("Warning: counting classes by iterating DataLoader (slow on NAS). "
```

slow fallback: iterate the DataLoader (original behavior)

## build_loss._infer_indices

### line 5468  _(unsure)_

```python
if target.ndim == 2:
```

Accept indices (N,) or one-hot (N,C); return indices (N,)

## build_loss

### line 5473  _(unsure)_

```python
class_weights = None
```

Priors/weights from counts if provided

### line 5482

```python
if logit_adjust_tau > 0:
```

Menon et al. 2020: logit adjustment

### lines 5484-5487

```python
logit_adjust = (float(logit_adjust_tau) * priors.log()).to(dtype=torch.float)
```

Menon et al. 2020 train-time adjustment is +tau*log(prior), applied as `logits + adjust` below. The negated form is the POST-HOC inference correction; used during training it pushes the model the wrong way and compounds the class imbalance.

## build_loss._auto_choice

### line 5532  _(unsure)_

```python
def _auto_choice() -> str:
```

Auto heuristic

## build_loss.loss_fn

### line 5605  _(unsure)_

```python
if target.ndim == 1:
```

expect one-hot/float (N,C) or indices (N,)

## annotate_predictions

### lines 5726-5728

```python
df['cond'] = pd.Series(
```

Keep the semantic distinction between an explicitly empty condition (``""``) and an unknown one (``None``).  Pandas 3 can otherwise infer a nullable string column and normalise the latter to ``nan``.

## add_images_to_tar

### line 5762, trailing

```python
if counter.value % 10 == 0:
```

Print every 100 updates

## generate_fraction_map

### lines 5781-5783

```python
independent_variables = pd.DataFrame(
```

An explicit float dtype prevents pandas from creating an object frame and then silently downcasting it in fillna(), behaviour that is deprecated and will change in a future pandas release.

### lines 5801-5804

```python
return independent_variables
```

NOTE: previously this unconditionally wrote the result to a hardcoded developer-machine path ('/mnt/data/CellVoyager/.../iv.csv'), which raised for any other environment. Removed — callers persist the returned frame themselves if they need it.

## fishers_odds

### line 5815

```python
df['high_phenotype'] = df[phenotyp_col] < threshold
```

Binning based on phenotype score (e.g., above 0.8 as high)

### line 5823  _(unsure)_

```python
for mutant in mutants:
```

Perform Fisher's exact test for each mutant

### line 5826, trailing  _(unsure)_

```python
if contingency_table.shape == (2, 2):
```

Check for 2x2 shape

### line 5830  _(unsure)_

```python
results.append((mutant, float('nan'), float('nan')))
```

Optionally handle non-2x2 tables (e.g., append NaN or other placeholders)

### line 5833  _(unsure)_

```python
results_df = pd.DataFrame(results, columns=['Mutant', 'OddsRatio', 'PValue'])
```

Convert results to DataFrame for easier handling

### line 5835  _(unsure)_

```python
filtered_results_df = results_df.dropna(
```

Remove rows with undefined odds ratios or p-values

### line 5841  _(unsure)_

```python
if len(pvalues) > 0:
```

Check if pvalues array is empty

### line 5843  _(unsure)_

```python
adjusted_pvalues = multipletests(pvalues, method='fdr_bh')[1]
```

Apply Benjamini-Hochberg correction

### line 5845  _(unsure)_

```python
filtered_results_df.loc[:, 'AdjustedPValue'] = adjusted_pvalues
```

Add adjusted p-values back to the dataframe

## model_metrics

### line 5858  _(unsure)_

```python
rmse = np.sqrt(model.mse_resid)
```

Calculate additional metrics

### line 5863  _(unsure)_

```python
print("\nAdditional Metrics:")
```

Display the additional metrics

### lines 5869-5873

```python
with figure_style(theme_target()):
```

Residual Plots

THE STYLE HAS TO BE ON BEFORE THE FIGURE EXISTS: rcParams reach an artist when it is CREATED, so a context opened after `plt.subplots` would leave the spines, ticks and labels at the caller's globals.

## lasso_reg

### line 5926  _(unsure)_

```python
X = merged_df[['gene', 'grna', 'plateID', 'rowID', 'columnID']]
```

Separate predictors and response

### line 5930  _(unsure)_

```python
encoder = OneHotEncoder(drop='first')  # drop one category to avoid the dummy variable trap
```

One-hot encode the categorical predictors

### line 5931, trailing

```python
encoder = OneHotEncoder(drop='first')
```

drop one category to avoid the dummy variable trap

### line 5937

```python
ridge = Ridge(alpha=alpha_value)
```

Fit ridge regression

### line 5942

```python
lasso = Lasso(alpha=alpha_value)
```

Fit Lasso regression

## MLR

### lines 5962-5965

```python
model = smf.ols("pred ~ gene + grna + gene:grna + plate + row + column", merged_df).fit()
```

Main effects must stay in the formula. With only the interaction term, patsy full-rank-codes the second factor as grna[<level>] (no "T."), so the "[T." filter used to pull out max effects below matched nothing and the returned effects were empty.

### line 5967  _(unsure)_

```python
model_metrics(model)
```

Display model metrics and summary

### line 5971  _(unsure)_

```python
std_resid = model.get_influence().resid_studentized_internal
```

Filter outliers

### line 5981  _(unsure)_

```python
model = smf.ols("pred ~ gene + grna + gene:grna + row + column", merged_df_filtered).fit()
```

Refit the model with filtered data

### line 5989  _(unsure)_

```python
interaction_coeffs = {key: val for key, val in model.params.items() if "gene[T." in key and ":grn...
```

Extract interaction coefficients and determine the maximum effect size

## get_files_from_dir

### lines 6025-6026  _(unsure)_

```python
return glob.glob(os.path.join(dir_path, file_extension))
```

``glob`` is imported as the module here (see glob.glob usage elsewhere), so it must be called as glob.glob — a bare glob(...) raised TypeError.

## create_circular_mask

### line 6038, trailing  _(unsure)_

```python
if center is None:
```

use the middle of the image

### line 6040, trailing  _(unsure)_

```python
if radius is None:
```

use the smallest distance between the center and image walls

## apply_mask

### line 6064, trailing  _(unsure)_

```python
h, w = image.shape[:2]
```

Assuming image is grayscale or RGB

### line 6067  _(unsure)_

```python
if len(image.shape) > 2:
```

If the image has more than one channel, repeat the mask for each channel

### line 6071  _(unsure)_

```python
masked_image = np.where(mask, image, output_value)
```

Apply the mask - set pixels outside of the mask to output_value

## invert_image

### lines 6102-6106

```python
pivot = np.asarray(info.min + info.max, dtype=image.dtype)
```

min + max: the dtype ceiling for an unsigned image (0 + 255), and -1 for a signed one, so the reflection maps [min, max] onto itself either way. The pivot always fits the dtype (it IS max when unsigned, -1 when signed), so subtracting in the image's own dtype is exact -- and avoids the uint64 promotion to float64 that a Python int operand would cause.

## match_masks

### line 6233, trailing  _(unsure)_

```python
break
```

Move on to the next predicted mask

## pad_to_same_shape

### line 6276  _(unsure)_

```python
shape_diff = np.array([max(mask1.shape[0], mask2.shape[0]) - mask1.shape[0],
```

Find the shape differences

## compute_ap_over_iou_thresholds

### line 6311  _(unsure)_

```python
if not 0 <= precision <= 1 or not 0 <= recall <= 1:
```

Check that precision and recall are within the range [0, 1]

### line 6316  _(unsure)_

```python
precision_recall_pairs = sorted(precision_recall_pairs, key=lambda x: x[1])
```

Sort by recall values

## dice_coefficient

### line 6374  _(unsure)_

```python
mask1 = np.where(mask1 > 0, 1, 0)
```

Convert to binary masks

### line 6378  _(unsure)_

```python
intersection = np.sum(mask1 & mask2)
```

Calculate intersection and total

### line 6382  _(unsure)_

```python
if total == 0:
```

Handle the case where both masks are empty

### line 6386  _(unsure)_

```python
return 2.0 * intersection / total
```

Return the Dice coefficient

## boundary_f1_score

### line 6423  _(unsure)_

```python
boundary_true = extract_boundaries(mask_true, dilation_radius)
```

Assume extract_boundaries is defined to extract object boundaries with given dilation_radius

### line 6427  _(unsure)_

```python
intersection = np.logical_and(boundary_true, boundary_pred)
```

Calculate intersection of boundaries

### line 6430  _(unsure)_

```python
precision = np.sum(intersection) / (np.sum(boundary_pred) + 1e-6)
```

Calculate precision and recall for boundary detection

### line 6434  _(unsure)_

```python
f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
```

Calculate F1 score as harmonic mean of precision and recall

## _remove_noninfected

### lines 6460-6463

```python
labels_in_cell = labels_in_cell[labels_in_cell != 0]
```

Count actual pathogens, not uniques. `len(...) <= 1` assumed a background pixel was always present inside the cell, so a cell completely filled by its pathogen yielded [pid] (len 1) and was deleted as "uninfected" — exactly backwards.

## _remove_outside_objects

### lines 6480-6485

```python
if pathogen_dim is None:
```

A DIM OF None IS np.newaxis, NOT A MISSING CHANNEL. `stack[:, :, None]` does not raise: it returns the WHOLE STACK with an axis inserted, so `nucleus_mask` became every channel at once and zeroing a nucleus label zeroed every object that happened to share the number. Measured on an 8x8 stack: a cell labelled 5, nowhere near the pathogen, was erased in full when nucleus_dim was None.

### line 6487, trailing  _(unsure)_

```python
return stack
```

nothing to remove

### line 6494, trailing  _(unsure)_

```python
cell_in_pathogen_region = cell_in_pathogen_region[cell_in_pathogen_region != 0]
```

Exclude background

### lines 6496-6499

```python
pathogen_mask[pathogen_region] = 0
```

Resolve the nucleus through the pathogen's FOOTPRINT. The old `nucleus_mask == pathogen_label` reused a pathogen label id as a nucleus label id — independent label spaces — so it deleted an arbitrary unrelated nucleus that merely shared the number.

## _remove_multiobject_cells

### lines 6515-6516

```python
if mask_dim is None or object_dim is None:
```

See `_remove_outside_objects`: a dim of None is np.newaxis and silently widens the view to the whole stack instead of raising.

### lines 6527-6530

```python
labels_in_cell = labels_in_cell[labels_in_cell != 0]
```

Strip background before counting. `> 2` and the `[1:]` slice both assumed a background pixel inside every cell, so a cell fully covered by two objects read as len 2 and was kept, and the slice then skipped a real object instead of the 0.

### lines 6536-6539

```python
if pathogen_mask is not None:
```

Resolve the pathogens through the cell FOOTPRINT. labels_in_cell are object_dim label ids, and nucleus/pathogen masks are labeled independently from 1 — reusing them as pathogen ids deletes unrelated pathogens whenever object_dim is not the pathogen dim.

## merge_touching_objects

### line 6563  _(unsure)_

```python
for label in labels:
```

Calculating perimeter of each object

### line 6565, trailing  _(unsure)_

```python
if label != 0:
```

Ignore background

### line 6568  _(unsure)_

```python
shared_perimeters = {}
```

Detect touching objects and find the shared boundary

### line 6572, trailing  _(unsure)_

```python
if label != 0:
```

Ignore background

### line 6573  _(unsure)_

```python
dilated_label = morphology.dilation(mask == label)
```

Find the objects that this object is touching

### line 6577, trailing  _(unsure)_

```python
if touching_label != label:
```

Exclude the object itself

### line 6580

```python
for (label1, label2), shared_perimeter in shared_perimeters.items():
```

Merge objects if more than 25% of their boundary is touching

### line 6583, trailing  _(unsure)_

```python
mask[mask == label2] = label1
```

Merge label2 into label1

## remove_intensity_objects

### line 6595  _(unsure)_

```python
props = regionprops_table(mask, image, properties=('label', 'mean_intensity'))
```

Calculate the mean intensity of each object in the original image

### line 6597  _(unsure)_

```python
if mode == 'low':
```

Find the labels of the objects with mean intensity below the threshold

### line 6602  _(unsure)_

```python
mask[np.isin(mask, labels_to_remove)] = 0
```

Remove these objects from the mask

## _find_similar_sized_images

### line 6619  _(unsure)_

```python
size_to_paths = defaultdict(list)
```

Dictionary to hold image sizes and their paths

### line 6621  _(unsure)_

```python
for path in file_list:
```

Iterate over image paths to get their dimensions

### line 6625  _(unsure)_

```python
if img.ndim == 3:  # Color image
```

Find indices where the image is not padded (non-zero)

### line 6626, trailing  _(unsure)_

```python
if img.ndim == 3:
```

Color image

### line 6628, trailing  _(unsure)_

```python
else:
```

Grayscale image

### line 6630  _(unsure)_

```python
coords = np.argwhere(mask)
```

Find the bounding box of non-zero regions

### line 6632, trailing  _(unsure)_

```python
if coords.size == 0:
```

Skip images that are completely padded

### line 6635, trailing

```python
y1, x1 = coords.max(axis=0) + 1
```

Add 1 because slice end index is exclusive

### line 6636  _(unsure)_

```python
cropped_img = img[y0:y1, x0:x1]
```

Crop the image to remove padding

### line 6638  _(unsure)_

```python
height, width = cropped_img.shape[:2]
```

Get dimensions of the cropped image

### line 6643  _(unsure)_

```python
largest_group = max(size_to_paths.values(), key=len)
```

Find the largest group of images with the most similar size and shape

## _relabel_parent_with_child_labels

### line 6649  _(unsure)_

```python
parent_labels = label(parent_mask, background=0)
```

Label parent mask to identify unique objects

### line 6651  _(unsure)_

```python
child_labels = child_mask
```

Use the original child mask labels directly, without relabeling

### line 6654  _(unsure)_

```python
parent_mask_new = np.zeros_like(parent_mask)
```

Create a new parent mask for updated labels

### line 6657  _(unsure)_

```python
unique_child_labels = np.unique(child_labels)[1:]  # Skip background
```

Directly relabel parent cells based on overlapping child labels

### line 6658, trailing  _(unsure)_

```python
unique_child_labels = np.unique(child_labels)[1:]
```

Skip background

### lines 6663-6664  _(unsure)_

```python
for parent_label in overlapping_parent_label:
```

Since each parent is assumed to overlap with exactly one nucleus, directly set the parent label to the child label where overlap occurs

### line 6666, trailing  _(unsure)_

```python
if parent_label != 0:
```

Skip background

### lines 6669-6670  _(unsure)_

```python
for parent_label in np.unique(parent_mask_new)[1:]:  # Skip background
```

For cells containing multiple nucleus, standardize all nucleus to the first label This will be done only if needed, as per your condition

### line 6671, trailing  _(unsure)_

```python
for parent_label in np.unique(parent_mask_new)[1:]:
```

Skip background

### line 6674, trailing  _(unsure)_

```python
child_labels_in_parent = child_labels_in_parent[child_labels_in_parent != 0]
```

Exclude background

### line 6677  _(unsure)_

```python
first_child_label = child_labels_in_parent[0]
```

Standardize to the first child label within this parent

## _exclude_objects

### line 6686  _(unsure)_

```python
filtered_cells = np.zeros_like(cell_mask) # Initialize a new mask to store the filtered cells.
```

Remove cells with no nucleus or cytoplasm (or pathogen)

### line 6687, trailing  _(unsure)_

```python
filtered_cells = np.zeros_like(cell_mask)
```

Initialize a new mask to store the filtered cells.

### line 6689, trailing  _(unsure)_

```python
if cell_label == 0:
```

Skip background

### line 6691, trailing  _(unsure)_

```python
cell_region = cell_mask == cell_label
```

Get a mask for the current cell.

### line 6692  _(unsure)_

```python
has_nucleus = np.any(nucleus_mask[cell_region])
```

Check existence of nucleus, cytoplasm and pathogen in the current cell.

### line 6702  _(unsure)_

```python
nucleus_mask = nucleus_mask * (filtered_cells > 0)
```

Remove objects outside of cells

## _filter_object

### line 6753

```python
remove = np.where(too_small | too_big)[0]
```

Label 0 is the background and is never an object.

## _filter_cp_masks

### line 6800  _(unsure)_

```python
distance_threshold = 0.25
```

Set a threshold for the minimum distance to consider clusters distinct

## _get_regex

### lines 6889-6892

```python
raise ValueError(
```

NAME THE VOCABULARY. Falling through left `regex` unbound, so an unrecognised metadata_type raised "cannot access local variable 'regex'" from inside this function -- an error that names an implementation detail and not the setting the user got wrong.

## _run_test_mode

### line 6917, trailing  _(unsure)_

```python
test_images = 1
```

Use only 1 set for timelapse to ensure full sequence inclusion

### line 6938  _(unsure)_

```python
set_identifiers = list(images_by_set.keys())
```

Prepare for random selection

### line 6942, trailing  _(unsure)_

```python
random.shuffle(set_identifiers)
```

Randomize the order

### line 6944  _(unsure)_

```python
selected_sets = set_identifiers[:test_images]
```

Select a subset based on the test_images count

### line 6947  _(unsure)_

```python
print(f'Using {len(selected_sets)} random image set(s) for test model')
```

Print information about the number of sets used

## _installed_cellpose_models

### lines 6993-6994

```python
return ()
```

A deferred import that fails must not stop a run choosing a model; the caller's fallback is the pre-4.2 behaviour.

## _resolve_cellpose_pretrained

### lines 7071-7083

```python
if name and name in _installed_cellpose_models():
```

A model the INSTALLED Cellpose actually ships is returned as itself.

This used to stop at `cpsam`, because Cellpose 4.0 had exactly one stock model. 4.2 ships four -- cpsam_v2 (its new default), cpdino, cpdino-vitb, cpsam -- and `settings.cellpose_model_choices` reads that list from the API, so a dropdown offers them the moment a user upgrades. Everything below then treated them as UNKNOWN and substituted cpsam: the menu offered a model the pipeline quietly refused to load, and the run SUCCEEDED with weights nobody asked for, which is the worst shape this bug could take.

Asked of the installed library rather than hard-coded, so a 4.3 that adds a fifth needs no release here.

### line 7092  _(unsure)_

```python
if os.path.isfile(name):
```

Anything that is not a known model name is meant to be a checkpoint.

## _choose_model

### lines 7141-7142

```python
kwargs = cellpose_kwargs()
```

device= from the caller still wins; only the flags it cannot know about (gpu, and the dtype the device can hold) come from here.

## SelectChannels.__call__

### line 7161, trailing  _(unsure)_

```python
img[0, :, :] = 0
```

Zero out the red channel

### line 7163, trailing  _(unsure)_

```python
img[1, :, :] = 0
```

Zero out the green channel

### line 7165, trailing  _(unsure)_

```python
img[2, :, :] = 0
```

Zero out the blue channel

## SaliencyMapGenerator.compute_saliency_maps

### line 7207  _(unsure)_

```python
scores = self.model(X).squeeze()
```

Forward pass

### line 7210  _(unsure)_

```python
target_scores = scores * (2 * y - 1)
```

For binary classification, target scores can be the single output

## SaliencyMapGenerator.compute_saliency_and_predictions

### lines 7227-7234

```python
raw = self.model(X)
```

Branch on the UN-squeezed logits. `(scores > 0).long()` is only a class label for a single-logit head; for a (B, C>1) head it is a per-logit boolean MASK, which then indexed as if it were a class and raised "a Tensor with 2 elements cannot be converted to Scalar" for every model train_test_model produces with the default two classes. argmax must be taken on the LOGITS, not on the mask: logits (-0.5, -0.2) mask to (0, 0), whose argmax is 0 while the true class is 1.

### line 7245  _(unsure)_

```python
self.model.zero_grad()
```

Compute saliency maps

## SaliencyMapGenerator.plot_activation_grid

### lines 7285-7291

```python
with figure_style(theme_target()):
```

squeeze=False keeps axs 2-D; without it matplotlib collapses a single-row grid to 1-D and the axs[i // 8, i % 8] index below raised IndexError for every batch of 8 or fewer images. THE STYLE HAS TO BE ON BEFORE THE FIGURE EXISTS: rcParams reach an artist when it is CREATED, so a context opened after `plt.subplots` would leave the spines, ticks and labels at the caller's globals.

### lines 7295-7297

```python
for ax in axs.flat:
```

An incomplete last row must be absent, not seven empty framed panels beside the final sample. Used panels stay off too because an image grid has no meaningful ticks or spines.

### lines 7305-7308

```python
if overlay:
```

The MAP is always drawn. It used to be inside `if overlay`, so overlay=False produced a grid of bare class labels on empty axes -- no input, no map, nothing. overlay now means what its name says: draw the input UNDER the map, or the map alone.

### line 7318  _(unsure)_

```python
ax.text(5, 25, str(predictions[i].item()), fontsize=12, color='white', weight='bold',
```

Add class label in the top-left corner

## SaliencyMapGenerator.percentile_normalize

### line 7333, trailing  _(unsure)_

```python
for c in range(img.shape[2]):
```

Iterate over each channel

## GradCAMGenerator.get_layer

### line 7381  _(unsure)_

```python
modules = target_layer.split('.')
```

Recursively find the layer specified in target_layer

## GradCAMGenerator.compute_gradcam_maps

### line 7396  _(unsure)_

```python
scores = self.model(X).squeeze()
```

Forward pass

### line 7399  _(unsure)_

```python
target_scores = scores * (2 * y - 1)
```

Perform backward pass

### line 7404  _(unsure)_

```python
pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
```

Compute GradCAM

### lines 7409-7413

```python
gradcam = torch.mean(self.activations, dim=1, keepdim=True)
```

keepdim keeps the map 4-D (N, 1, H, W) even when the target layer's spatial dims have collapsed to 1x1 on small inputs; squeeze() plus two unsqueeze(0) calls produced a 2-D tensor that F.interpolate rejects with "Input and output must have the same number of spatial dimensions".

## GradCAMGenerator.compute_gradcam_and_predictions

### lines 7435-7438

```python
raw = self.model(X)
```

See compute_saliency_and_predictions: `(scores > 0).long()` is a class label only for a single-logit head. For a (B, C>1) head it is a per-logit mask, so predictions[i] below was a C-element tensor rather than a scalar class index.

## GradCAMGenerator.plot_activation_grid

### lines 7482-7487

```python
with figure_style(theme_target()):
```

See SaliencyMapGenerator.plot_activation_grid — squeeze=False is required so the 2-D index below works for a single-row grid. THE STYLE HAS TO BE ON BEFORE THE FIGURE EXISTS: rcParams reach an artist when it is CREATED, so a context opened after `plt.subplots` would leave the spines, ticks and labels at the caller's globals.

### lines 7498-7499  _(unsure)_

```python
if overlay:
```

Same contract as the saliency twin: the map always draws, and overlay decides whether the input is drawn beneath it.

### lines 7509-7510

```python
ax.text(5, 25, str(predictions[i].item()), fontsize=12, color='white', weight='bold',
```

ax.imshow(X[i].permute(1, 2, 0).detach().cpu().numpy())  # Original image ax.imshow(gradcam_map, cmap='jet', alpha=0.5)  # Overlay the gradcam map

### line 7512  _(unsure)_

```python
ax.text(5, 25, str(predictions[i].item()), fontsize=12, color='white', weight='bold',
```

Add class label in the top-left corner

## GradCAMGenerator.percentile_normalize

### line 7527, trailing  _(unsure)_

```python
for c in range(img.shape[2]):
```

Iterate over each channel

## class_visualization

### line 7598  _(unsure)_

```python
SQUEEZENET_MEAN = [0.485, 0.456, 0.406]
```

Assuming these are defined somewhere in your codebase

### lines 7602-7603

```python
model = torch.load(model_path, weights_only=False)
```

weights_only=False is the pre-torch-2.6 default this call site was written against; these checkpoints are whole nn.Module pickles.

### lines 7606-7607  _(unsure)_

```python
from .accelerator import is_cuda
```

A CUDA tensor TYPE, which only CUDA has. Every other backend takes a plain float tensor and is moved with .to(device).

### line 7614  _(unsure)_

```python
img = torch.randn(1, len_chans, img_size, img_size).mul_(1.0).type(dtype).requires_grad_()
```

Randomly initialize the image as a PyTorch Tensor, and make it requires gradient.

### line 7618  _(unsure)_

```python
ox, oy = random.randint(0, max_jitter), random.randint(0, max_jitter)
```

Randomly jitter the image a bit; this gives slightly nicer results

### line 7622  _(unsure)_

```python
score = model(img)
```

Forward pass

### line 7630  _(unsure)_

```python
target_score = target_score - l2_reg * torch.norm(img)
```

Add regularization

### line 7633  _(unsure)_

```python
target_score.backward()
```

Backward pass

### line 7636  _(unsure)_

```python
with torch.no_grad():
```

Gradient ascent step

### line 7641  _(unsure)_

```python
img.data.copy_(jitter(img.data, -ox, -oy))
```

Undo the random jitter

### line 7644  _(unsure)_

```python
for c in range(3):
```

As regularizer, clamp and periodically blur the image

### line 7652  _(unsure)_

```python
if t == 0 or (t + 1) % show_every == 0 or t == num_iterations - 1:
```

Periodically show the image

## GradCAM.__call__

### lines 7747-7749

```python
cam = cv2.resize(np.atleast_2d(cam), (x.size(2), x.size(3)))
```

np.atleast_2d guards the case where the target layer's spatial dims have collapsed to 1x1 (small inputs): cam would otherwise be 0-d and cv2.resize rejects it.

## show_cam_on_image

### lines 7805-7806

```python
warnings.warn(
```

Clipping keeps hot pixels hot. The warning is what makes the normalization the caller skipped visible.

### lines 7819-7821

```python
raise ValueError(
```

Unreachable for an image in [0, 1]: jet maps even a zero mask to BGR (128, 0, 0), so the blend always has a positive pixel. Getting here means the image was negative enough to cancel the heatmap out.

## recommend_target_layers

### line 7841  _(unsure)_

```python
if target_layers:
```

Choose the last conv layer as the recommended target layer

## reduction_and_clustering

### lines 8033-8036

```python
perplexity = min(requested_perplexity, float(len(values) - 1))
```

A row limit can leave fewer rows than the saved/default perplexity. Use the largest valid neighbourhood rather than failing after data loading; retain the requested setting in the settings file and report the adjustment in verbose mode.

### lines 8050-8051

```python
if not prefer_gpu:
```

sklearn accepts n_jobs; cuML releases differ, so do not forward that CPU-only tuning argument to a requested GPU constructor.

### lines 8128-8130

```python
raise ValueError(f"Unsupported clustering method: {clustering}. Supported methods are 'dbscan' an...
```

Without this the name stays unbound and the next line dies with a bare UnboundLocalError. search_reduction_and_clustering already raises this; the two are now consistent.

## plot_embedding

### lines 8198-8202

```python
with plt.rc_context():
```

`setup_plot` pushes the theme's colours into matplotlib's process-wide rcParams so the panels it builds match the GUI. Scoped to this figure: a UMAP drawn on the dark theme used to leave every later figure of the session with white-on-white text once the user moved to a screen that draws on paper.

### lines 8219-8221

```python
fig._spacr_umap_payload = interactive_payload
```

The Qt bridge recognises this attribute and keeps the underlying points/image/database identities instead of flattening the result into a PNG-only gallery entry.

## setup_plot

### lines 8327-8332

```python
fig, ax = plt.subplots(1, 1, figsize=(figuresize, figuresize))
```

NOT `figure_style` HERE, DELIBERATELY. `setup_plot` exists to draw in the GUI THEME's colours, and the context above is already applying them. A house-style context nested inside would win the inner rc_context is the one in force -- and hand back a figure painted for print on a dark screen, which is the exact failure `theme_target` exists to prevent.

## plot_clusters

### lines 8373-8375

```python
if smooth_lines:
```

A ConvexHull needs >=3 non-collinear points; with too few or collinear points (common for tiny/degenerate clusters) Qhull raises. Skip the outline in that case rather than crashing the whole plot.

## plot_images_by_cluster

### lines 8498-8502

```python
for cluster_label in np.unique(labels):
```

NOT zip(np.unique(labels), colors). The colour was never read, so the only thing that zip contributed was a LENGTH -- and because np.unique counts the -1 noise label, a palette sized to the real clusters was one short and THE LAST CLUSTER WAS SILENTLY NOT PLOTTED. The palette does not get to decide how many clusters are drawn.

## plot_image

### lines 8534-8536

```python
if remove_image_canvas:
```

remove_canvas() inspects PIL's ``img.mode``, so it must run BEFORE the array conversion — converting first made remove_image_canvas=True raise AttributeError: 'numpy.ndarray' object has no attribute 'mode'.

## plot_clusters_grid

### lines 8600-8601

```python
if len(indices) > image_nr:
```

No -1 guard needed: the comprehension above already excludes the DBSCAN noise label, so this loop never sees it.

## plot_grid

### line 8639, trailing  _(unsure)_

```python
max_figsize = 200
```

Set a maximum figure size

### lines 8644-8647

```python
with figure_style(theme_target()):
```

THE STYLE HAS TO BE ON BEFORE THE FIGURE EXISTS:

rcParams reach an artist when it is CREATED, so a context opened after `plt.subplots` would leave the spines, ticks and labels at the caller's globals.

### line 8652, trailing  _(unsure)_

```python
grid_axes = [grid_axes]
```

Ensure grid_axes is always iterable

### lines 8672-8676

```python
if isinstance(cluster_label, str) and verbose:
```

Both branches WRAP. A string label is positioned, an integer label indexes the palette directly -- and DBSCAN numbers its clusters 0..k-1, so a run with more clusters than colours used to die here with a bare "list index out of range" from colors[cluster_label]. Reusing a colour is a worse figure; crashing is a lost run.

### line 8696  _(unsure)_

```python
spacing_factor = 0.5  # Adjust this value to control the spacing between labels
```

Add cluster labels beside the UMAP plot

### line 8697, trailing  _(unsure)_

```python
spacing_factor = 0.5
```

Adjust this value to control the spacing between labels

### line 8700, trailing  _(unsure)_

```python
label_y = 1 - (i + 1) * (spacing_factor / num_clusters)
```

Adjust y position for each label

## generate_path_list_from_db

### line 8719  _(unsure)_

```python
print(f"Reading DataBase: {db_path}")
```

Connect to the database and retrieve the image paths

### line 8727  _(unsure)_

```python
cursor.execute("SELECT png_path FROM png_list WHERE png_path LIKE ?", (f"%{file_metadata}%",))
```

If file_metadata is a single string

### line 8736  _(unsure)_

```python
cursor.execute("SELECT png_path FROM png_list")
```

If file_metadata is None or empty

## measure_test_mode

### lines 8859-8861

```python
all_files = [f for f in os.listdir(settings['src'])
```

isfile: os.listdir also returns subdirectories, and shutil.copy on one raises IsADirectoryError -- one stray folder under merged/ took the whole run down.

## preprocess_data

### lines 9013-9016

```python
df = schema.coerce_model_feature_types(
```

Measurement values inserted into SQLite as numeric text make pandas use object dtype for the whole column. Normalize only losslessly numeric declared features before the strict schema boundary; malformed text still raises an actionable ModelFeatureSchemaError.

### line 9030  _(unsure)_

```python
if filter_by is not None:
```

Apply filtering based on the `filter_by` parameter

### lines 9045-9047

```python
numeric_data = schema.model_feature_frame(
```

Select declared measurements. Numeric provenance such as object_label, measurement_ndim and voxel sizes must never enter an embedding merely because pandas gave it a numeric dtype.

### lines 9055-9060

```python
numeric_data = _resolve_missing_model_features(numeric_data)
```

The unfiltered UMAP/statistics path does not pass through

``filter_dataframe_features``.  Give it the same missing-measurement contract before transformations or estimators see an all-NaN feature. Resolve every non-finite representation, not only pandas NA. A feature frame containing only +/- infinity has no ``isna`` bit set but is just as unfit for correlation filters and estimators as one containing NaN.

### line 9084  _(unsure)_

```python
if log_data:
```

Apply log transformation

### line 9114  _(unsure)_

```python
numeric_data = numeric_data.fillna(numeric_data.mean())
```

Fill NaN values with the column mean

## remove_highly_correlated_columns

### line 9156  _(unsure)_

```python
to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
```

Find columns with correlation greater than the threshold

## _resolve_missing_model_features

### lines 9245-9248

```python
df = df.replace([np.inf, -np.inf], np.nan)
```

Ratios measured over an empty compartment legitimately reach the model table as +/- infinity. Pandas does not classify those values as missing, and scikit-learn's scalers refuse them, so normalize them into the same explicit missing-value contract before deciding which columns survive.

### lines 9267-9268  _(unsure)_

```python
print(
```

Keep the long-standing diagnostic text stable for callers and logs, while spelling out that only wholly absent columns are removed now.

## filter_dataframe_features

### lines 9305-9313

```python
df = schema.coerce_model_feature_types(df, exclude=excluded_features)
```

Repair before the strict boundary judges. A measurement that is NULL in every row of the database -- mode_intensity in anything measured before the SciPy shim, skew/kurtosis wherever every object is uniform -- reaches here as an OBJECT column of None, because pandas types an all-NULL result set from its rows and never asks SQLite what it declared. model_feature_ columns then refused it, naming one column, and the run stopped on data that has nothing wrong with it. See schema.coerce_model_feature_types: numeric text is recovered loudly, unreadable text still refuses, and it refuses with every offending column named at once.

### lines 9333-9345

```python
selection = feature_selection(channel_of_interest)
```

WHICH FEATURES THE MODEL SEES, from one setting that takes every shape a user can mean by it -- a channel, several channels, 'morphology', a column-name fragment, or a mixture. See `feature_selection`, which is where the four doors into this question (the panel's chip strip, a settings CSV, a script, and the default) are made to agree.

THE UNION, not the intersection: [1, 'morphology'] is channel 1's intensities AND the shapes, which is the combination the request asked to be straightforward. And colocalisation belongs to both channels it measures, so asking for channel 1 keeps `cell_channel_1_channel_2_pearsons` -- which is how "localization" is reachable without a setting of its own.

### lines 9354-9358

```python
df = _resolve_missing_model_features(df)
```

Resolve missingness before variance and correlation filtering.  An all-missing feature then disappears explicitly; a one-value feature becomes constant after imputation and the ordinary variance rule drops it.  Running those filters first lets pandas' pairwise NaN rules make a different accidental decision for each missingness pattern.

## check_overlap

### line 9378  _(unsure)_

```python
def check_overlap(current_position, other_positions, threshold):
```

Create a function to check if images overlap

## find_non_overlapping_position

### line 9404  _(unsure)_

```python
def find_non_overlapping_position(x, y, image_positions, threshold, max_attempts=100):
```

Define a function to try random positions around a given point

### line 9415, trailing  _(unsure)_

```python
offset_range = 10
```

Adjust the range for random offsets

### line 9425, trailing  _(unsure)_

```python
return x, y
```

Return the original position if no suitable position found

## extract_features

### line 9519, trailing  _(unsure)_

```python
model = torch.nn.Sequential(*list(model.children())[:-1])
```

Remove the last classification layer

## check_normality

### line 9550, trailing  _(unsure)_

```python
if p < alpha:
```

null hypothesis: x comes from a normal distribution

## perform_statistical_tests

### lines 9619-9624

```python
if len(groups) < 2:
```

A clustering algorithm is allowed to find one population. Neither ANOVA nor Kruskal-Wallis is defined for fewer than two groups, but that must not turn an otherwise valid embedding into an exception. Keep the feature in its normality-selected result table and mark the unavailable statistic explicitly; ``combine_results`` then retains the feature importance and its stable one-row schema.

## _merge_cells_without_nucleus

### lines 9708-9709  _(unsure)_

```python
nuc_labels = np.unique(nuclei_mask[nuclei_mask > 0])
```

1 — Identify which cell IDs contain a nucleus

### line 9717  _(unsure)_

```python
keep = labels > 0
```

drop background (label 0) from *both* arrays

### line 9722, trailing  _(unsure)_

```python
if labels.size:
```

at least one non-zero overlap

### lines 9725-9727  _(unsure)_

```python
boundaries = find_boundaries(adj_cell_mask, mode="thick")
```

2 — Build an adjacency map between neighbouring cell IDs

### lines 9745-9747  _(unsure)_

```python
cells_no_nuc = set(np.unique(adj_cell_mask)) - {0} - cells_with_nuc
```

3 — Relabel nucleus-free cells that touch nucleus-bearing neighbours

### line 9752  _(unsure)_

```python
target = sorted(neighbours)[0]
```

Choose the first nucleus-bearing neighbour deterministically

## _merge_cells_based_on_parasite_overlap

### line 9766  _(unsure)_

```python
for parasite_id in range(1, num_parasites + 1):
```

Merge cells based on parasite overlap

### line 9773  _(unsure)_

```python
overlap_percentages = [
```

Calculate the overlap percentages

### line 9778  _(unsure)_

```python
for cell_label, overlap_percentage in zip(overlapping_cell_labels, overlap_percentages):
```

Merge cells if overlap percentage is above the threshold

### line 9785  _(unsure)_

```python
for nucleus_id in range(1, num_nuclei + 1):
```

Merge cells based on nucleus overlap

### line 9792  _(unsure)_

```python
overlap_percentages = [
```

Calculate the overlap percentages

### line 9803  _(unsure)_

```python
labeled_cells = label(cell_mask)  # Re-label after merging based on overlap
```

Check for cells without nuclei and merge based on shared perimeter

### line 9804, trailing  _(unsure)_

```python
labeled_cells = label(cell_mask)
```

Re-label after merging based on overlap

### line 9814  _(unsure)_

```python
perimeter = region.perimeter
```

Cell does not overlap with any nucleus

### line 9823  _(unsure)_

```python
shared_borders = [
```

Calculate shared border length with neighboring cells

### line 9829  _(unsure)_

```python
if shared_borders:
```

Merge with the neighbor cell with the largest shared border percentage above the threshold

## process_masks.read_files_in_batches

### line 9952, trailing  _(unsure)_

```python
files.sort()
```

Sort to ensure matching order

## process_masks.remove_objects_not_in_largest_cluster

### lines 9971-9974

```python
for idx, region in enumerate(measure.regionprops(mask)):
```

`labels` is the per-file slice of the KMeans label array, ordered by regionprops enumeration — NOT indexed by label value. Sparse or non-contiguous labels made `labels[region.label - 1]` read the wrong cluster (or run off the end of the slice).

## process_masks

### line 9993  _(unsure)_

```python
for batch in read_files_in_batches(mask_folder, batch_size):
```

Step 1: Accumulate properties over all files

### line 10007  _(unsure)_

```python
kmeans = cluster_objects(all_properties, n_clusters)
```

Step 2: Perform clustering on accumulated properties

### line 10012  _(unsure)_

```python
plot_clusters(all_properties, labels)
```

Step 3: Plot clusters using PCA

### line 10015  _(unsure)_

```python
label_index = 0
```

Step 4: Remove objects not in the largest cluster and overwrite files in batches

### lines 10024-10026

```python
continue
```

Object-free field of view: np.bincount([]).argmax() raises

"attempt to get argmax of an empty sequence". There is nothing to cluster, so leave the mask on disk untouched.

## merge_regression_res_with_metadata

### line 10042  _(unsure)_

```python
df_results = tabular.read_table(results_file, report=None)
```

Read the CSV files into dataframes

### lines 10044-10046

```python
df_metadata = tabular.read_table(metadata_file, canonicalise=False,
```

canonicalise=False: this is a third-party gene annotation file whose header is the vendor's ('Gene ID'), not spaCR metadata, and renaming a column of it would break the merge two lines below.

### lines 10066-10073

```python
identifier_column = next(
```

The identifier column is DETECTED, not assumed.

'Gene ID' is the header of the bundled toxoplasma_metadata.csv, and hard-coding it meant any other annotation table died on `KeyError: 'Gene ID'` -- a message naming a column the user's file does not have and never claimed to, after the whole regression had already run. A gRNA barcode export keyed on 'name' (TGGT1_225160_2) carries exactly the same identifier in exactly the same shape.

### lines 10089-10090

```python
df_metadata = df_metadata.dropna(subset=['gene'])
```

Drop rows where gene extraction failed df_results = df_results.dropna(subset=['gene'])

### lines 10092-10095

```python
df_metadata = df_metadata.dropna(subset=['gene'])
```

Metadata rows whose ID had no parsable gene must not act as a join key: pandas treats NaN keys as equal, so every unparsable result row (e.g. 'Intercept') would otherwise fan out against every unparsable metadata row.

### lines 10098-10106

```python
duplicated_genes = df_metadata['gene'].duplicated(keep=False)
```

One annotation row per gene, enforced rather than assumed. Curated exports list a gene once per transcript/isoform -- the bundled 'toxoplasma_metadata.csv' repeats 30 Gene IDs two to four times, each copy carrying a different protein length and GO term set. Joined as-is those genes came back two to four times in the regression results, and every downstream consumer (volcano plots, the significant-hit tables, toxo.py) counted each copy as an independent hit. The result must stay one row per regression feature, so the metadata is collapsed to the first row per gene and the collapse is reported rather than hidden.

### lines 10119-10120

```python
merged_df = pd.merge(df_results, df_metadata, on='gene', how='left',
```

many_to_one: many regression terms can name one gene (one row per gRNA in the per-gRNA results), but each gene gets one annotation row.

### line 10124  _(unsure)_

```python
base, ext = os.path.splitext(results_file)
```

Generate the new file name

### line 10128  _(unsure)_

```python
tabular.write_table(merged_df, new_file)
```

Save the merged dataframe to the new file

## merge_regression_res_with_metadata.extract_and_clean_gene

### line 10052  _(unsure)_

```python
match = re.search(r'\[(.*?)\]', feature)
```

Extract the part between '[' and ']'

### line 10056  _(unsure)_

```python
gene = re.sub(r'^T\.', '', gene)
```

Remove 'T.' if present

### line 10058  _(unsure)_

```python
gene = gene.split('_')[0]
```

Remove everything after and including '_'

## process_vision_results

### lines 10140-10150

```python
mapped_values = df['path'].apply(lambda x: _map_wells_png(x))
```

`_map_wells_png`, NOT `_map_wells`. These paths are CROPS

`plate1_E01_18_1_250.png`, which is plate_well_field_time_object -- and `_map_wells` parses a FIELD stem, which is three parts or four with a timepoint. Five parts is neither, so it raised for every row and returned its 'error' tuple, and an entire inference run came out with plateID/rowID/columnID/fieldID = 'error' and prc = 'error_error_error'.

Which means the scores could not be joined back to a well: no per-well aggregate, no regression on a CV model's output, and the only sign was a screenful of "Error processing filename" that the run scrolled past while reporting success. The crop parser has always existed beside it.

### lines 10157-10159

```python
df['object'] = (df['path'].str.rsplit('/', n=1).str[-1]
```

The object id is the LAST component of the crop name, not the fourth: a timelapse crop is plate_well_field_time_object, so [3] is the TIMEPOINT. Splitting from the right is correct for both layouts.

### lines 10162-10167

```python
df['prc'] = schema.compose_prc_column(df)
```

ONE COMPOSER. A bare `plateID + '_' + rowID + '_' + columnID` is correct only while no plate id contains the separator or a `%`, and the regression path composes the SAME key through `schema.compose_prc`, which escapes both. A plate called `exp1_plate2` therefore produced two different strings for one well and the join between them matched nothing.

## get_ml_results_paths

### lines 10213-10217

```python
feature_string = feature_folder_name(channel_of_interest)
```

NAMED FROM THE CANONICAL SELECTION, so two spellings of one feature space -- `1` and `[1]`, `'1,2'` and `[1, 2]` -- write to one folder. It used to raise on a free-text filter that `filter_dataframe_features` has always accepted, so `channel_of_interest='mean_intensity'` filtered the features and then died on the way to naming the folder.

## augment_image

### line 10255  _(unsure)_

```python
if isinstance(image, Image.Image):
```

Convert PIL image to numpy array if necessary

### line 10259  _(unsure)_

```python
if len(image.shape) == 2:
```

Handle grayscale images

### line 10263  _(unsure)_

```python
transformations = [
```

Rotations and reflections

### line 10265, trailing  _(unsure)_

```python
None,
```

Original

### line 10278  _(unsure)_

```python
flipped = cv2.flip(rotated, 1)
```

Reflections

### line 10282  _(unsure)_

```python
augmented_images = [Image.fromarray(img) for img in augmented_images]
```

Convert numpy arrays back to PIL images

## augment_dataset

### line 10300  _(unsure)_

```python
if not isinstance(img, torch.Tensor):
```

Ensure the image is a tensor

### line 10304  _(unsure)_

```python
angles = [0, 90, 180, 270]
```

Rotations and reflections

### line 10311  _(unsure)_

```python
flipped = torchvision.transforms.functional.hflip(rotated)
```

Reflections

## convert_and_relabel_masks

### lines 10341-10343

```python
if mask.dtype != np.int64:
```

print(mask.shape) print(mask.dtype) Check the current dtype

### line 10355  _(unsure)_

```python
unique_relabeled = np.unique(relabeled_mask)
```

Check that relabeling worked correctly

## get_cuda_version

### line 10399  _(unsure)_

```python
def get_cuda_version():
```

Function to determine the CUDA version

## prepare_batch_for_segmentation

### line 10432  _(unsure)_

```python
for i in range(batch.shape[0]):
```

Normalize each image in the batch

## map_condition

### line 10459  _(unsure)_

```python
def map_condition(col_value, neg='c1', pos='c2', mix='c3'):
```

Define the mapping function

## download_models

### line 10495  _(unsure)_

```python
if not os.path.exists(local_dir):
```

Create the local directory if it doesn't exist

### line 10507, trailing  _(unsure)_

```python
print(f"Files in repository: {files}")
```

Debugging print to check file list

### line 10514, trailing  _(unsure)_

```python
print(f"Downloading file from: {url}")
```

Debugging

### line 10517, trailing  _(unsure)_

```python
print(f"HTTP response status: {response.status_code}")
```

Debugging

### line 10520  _(unsure)_

```python
local_file_path = os.path.join(local_dir, os.path.basename(file_name))
```

Save the file locally

### line 10526, trailing  _(unsure)_

```python
break
```

Exit the retry loop if successful

### line 10533, trailing  _(unsure)_

```python
return local_dir
```

Return the directory where models are saved

## generate_cytoplasm_mask

### line 10558  _(unsure)_

```python
nucleus_mask = np.array(nucleus_mask)
```

Make sure the nucleus and cell masks are numpy arrays

### lines 10562-10564

```python
cytoplasm_mask = np.where(nucleus_mask != 0, 0, cell_mask)
```

Generate cytoplasm mask: everything inside the cell that is not nucleus. NOTE: this used to read np.logical_or(nucleus_mask != 0) — logical_or needs TWO operands, so the function raised TypeError on every call.

## add_column_to_database

### line 10587  _(unsure)_

```python
df = tabular.read_table(settings['csv_path'], report=None)
```

Read the DataFrame from the provided CSV path

### lines 10593-10595

```python
df[settings['update_column']] = df[settings['update_column']].replace(0, 2)
```

Plain reassignment, not chained inplace: under pandas copy-on-write (the 3.0 default) the inplace form mutates a temporary and is a silent no-op.

### line 10598  _(unsure)_

```python
conn = sqlite3.connect(settings['db_path'], timeout=30)
```

Connect to the SQLite database

### line 10602  _(unsure)_

```python
cursor.execute(f"PRAGMA table_info({settings['table_name']})")
```

Get the existing columns in the database table

### line 10606  _(unsure)_

```python
if settings['update_column'] in columns_in_db:
```

Add a suffix if the update column already exists in the database

### line 10617  _(unsure)_

```python
cursor.execute(f"ALTER TABLE {settings['table_name']} ADD COLUMN {new_column_name} INTEGER")
```

Add the new column with INTEGER type to the database table

### line 10621  _(unsure)_

```python
for index, row in df.iterrows():
```

Iterate over the DataFrame and update the new column in the database

### line 10626  _(unsure)_

```python
if pd.isna(value_to_update):
```

Handle NaN values by converting them to None (SQLite equivalent of NULL)

### line 10630  _(unsure)_

```python
query = f"""
```

Prepare and execute the SQL update query

## fill_holes_in_mask

### line 10654  _(unsure)_

```python
labeled_mask, num_features = ndimage.label(mask)
```

Ensure the mask is integer-labeled

### line 10657  _(unsure)_

```python
filled_mask = np.zeros_like(labeled_mask)
```

Create an empty mask to store the result

### line 10662  _(unsure)_

```python
object_mask = (labeled_mask == i)
```

Create a binary mask for the current object

### line 10668  _(unsure)_

```python
filled_mask[filled_object] = i
```

Assign the original label back to the filled object

## rename_columns_in_db

### lines 10789-10793

```python
by_table = {}
```

A measurements table carries one of these per object type, per channel and per percentile — several hundred on a four-channel run — so a line each would bury everything else the read prints. One line per table with an example says the same thing; the full list is the return value.

## group_feature_class

### line 10833  _(unsure)_

```python
if feature_groups is None:
```

Function to determine compartment based on multiple matches

### lines 10846-10848

```python
df[name] = pd.Series(
```

Preserve unmatched features as real ``None`` values.  Pandas 3 may otherwise infer a nullable string dtype and expose those entries as ``nan``, which changes the public result even though ``isna`` agrees.

### lines 10857-10858  _(unsure)_

```python
df['channel'] = df['channel'].fillna('morphology')
```

See add_column_to_database: chained inplace is a no-op under pandas copy-on-write.

## cleanup_pipeline_folders

### line 10904  _(unsure)_

```python
if stack_files and not stack_files.issubset(merged_files):
```

Only safe to delete stack/+masks/ if every field of view was merged.

### line 10915  _(unsure)_

```python
for d in os.listdir(src):
```

Numeric per-channel folders (1, 2, 3, …) if any survived.

## delete_intermedeate_files

### lines 10958-10960

```python
if 'src' not in settings:
```

Validate the inputs BEFORE the completeness guard. These checks used to be nested inside it, so a missing src or missing orig/ backup reported nothing at all whenever the guard happened to be closed.

### lines 10971-10975

```python
merged_len = len(os.listdir(merged_stack)) if os.path.isdir(merged_stack) else 0
```

Only drop the intermediates once merged/ is at least as populated as stack/, i.e. every field made it through. Count FILES, not characters: the old `len(merged_stack) == len(path_stack)` compared len(src)+7 against len(src)+6 — always off by one, so the guard never opened and this function silently deleted nothing.

## filter_and_save_csv

### line 11016  _(unsure)_

```python
df = tabular.read_table(input_csv, report=None)
```

Read the input CSV file into a DataFrame

### line 11019  _(unsure)_

```python
filtered_df = df[(df[column_name] > upper_threshold) | (df[column_name] < lower_threshold)]
```

Filter rows based on the thresholds

## extract_tar_bz2_files

### line 11038  _(unsure)_

```python
for file_name in os.listdir(folder_path):
```

Iterate over files in the folder

### line 11044  _(unsure)_

```python
os.makedirs(extract_folder, exist_ok=True)
```

Create the subfolder for extraction if it doesn't exist

## calculate_shortest_distance

### line 11074  _(unsure)_

```python
centroid_distance = np.sqrt(
```

Compute centroid-to-centroid Euclidean distance

### line 11080  _(unsure)_

```python
object1_radius = df[f'{object1}_feret_diameter_max'] / 2
```

Estimate object radii using Feret diameters

### line 11084  _(unsure)_

```python
shortest_distance = centroid_distance - (object1_radius + object2_radius)
```

Compute shortest edge-to-edge distance

### line 11087  _(unsure)_

```python
df[f'{object1}_{object2}_shortest_distance'] = np.maximum(shortest_distance, 0)
```

Ensure distances are non-negative (overlapping objects should have distance 0)

## format_path_for_system

### line 11104  _(unsure)_

```python
if system in ["Linux", "Darwin"]:  # Darwin is macOS
```

Convert Windows-style paths to Unix-style (Linux/macOS)

### line 11105, trailing  _(unsure)_

```python
if system in ["Linux", "Darwin"]:
```

Darwin is macOS

### line 11108  _(unsure)_

```python
elif system == "Windows":
```

Convert Unix-style paths to Windows-style

### line 11115  _(unsure)_

```python
new_path = os.path.normpath(formatted_path)
```

Normalize path to ensure consistency

## normalize_src_path

### line 11137, trailing  _(unsure)_

```python
return src
```

Already a list, return as-is

### line 11141  _(unsure)_

```python
evaluated_src = ast.literal_eval(src)
```

Check if it is a string representation of a list

### line 11144, trailing  _(unsure)_

```python
return evaluated_src
```

Convert to real list

### line 11146, trailing  _(unsure)_

```python
pass
```

Not a valid list, treat as a string

### line 11148, trailing  _(unsure)_

```python
return src
```

Return as a string if not a list

## generate_image_path_map

### lines 11167-11172

```python
dirnames[:] = [name for name in dirnames if name != "consolidated"]
```

NEVER RE-CONSOLIDATE OUR OWN OUTPUT. `consolidated` is created INSIDE the folder being walked, so a second run over the same `src` finds the copies from the first one and makes copies of those, prefixed again -- doubling the plate on every run and producing `consolidated_plate1_A01_f1_c1.tif`. Pruning the walk is what makes the operation repeatable.

### lines 11180-11188

```python
if relative_path == os.curdir:
```

Construct new filename: Embed folder hierarchy into the name.

`os.path.relpath(root, root)` is `'.'`, NOT `''`, so an image sitting directly in `src` used to be renamed `._name.tif`. That is a hidden file on Unix and the AppleDouble resource-fork convention on macOS, and `spacr.io` skips anything beginning with a dot -- so consolidating a flat folder made every image in it silently disappear from the run rather than failing.

### line 11195  _(unsure)_

```python
new_filename = f"{folder_info}_{file}" if folder_info else file
```

Generate new filename

### line 11198  _(unsure)_

```python
original_path = os.path.join(dirpath, file)
```

Store in dictionary (original path -> new path)

## copy_images_to_consolidated

### line 11224, trailing  _(unsure)_

```python
new_filename = os.path.basename(new_path)
```

Extract only the new filename

### line 11225, trailing  _(unsure)_

```python
new_file_path = os.path.join(consolidated_folder, new_filename)
```

Place in 'consolidated' folder

### line 11227, trailing  _(unsure)_

```python
shutil.copy2(original_path, new_file_path)
```

Copy file with metadata preserved

## remove_outliers_by_group

### lines 11312-11314

```python
if threshold < 0:
```

A negative threshold inverts the band: under 'iqr' it makes the keep interval empty, so every group with a nonzero IQR loses ALL its rows and the caller gets a near-empty frame with no error. Refuse it.

### lines 11331-11335

```python
keep = (df[value_col] - mean).abs() <= threshold * std
```

A single-row group has std NaN, and NaN comparisons are False, so 'zscore' used to DELETE every singleton while 'iqr' kept it (its quartiles collapse onto the value). One row cannot be an outlier within its own group under either definition; the two methods now agree instead of disagreeing on the smallest groups.

## generate_image_path_map, 2026-09-19

```python
if file.startswith('.'):
```

`consolidate=True` flattens every sub-folder of `src` into `consolidated/`, naming each copy `<sub-folders>_<file>`. On a macOS external volume (GitHub #121 and #117: exFAT, FAT and many SMB shares) every image has an AppleDouble sidecar, `._<name>`, with the same `.tif` ending. The walk took `sub/._img1.tif` for an image and named its copy `sub_._img1.tif`. The prefix moves the dot off the front, so no listing downstream could tell the copy was a sidecar any more; whether it was then read as a tiff depended on the regex. Dot-files are skipped here, before the rename can hide them, and dot-folders are pruned from the walk too: a volume root carries `.Spotlight-V100`, `.Trashes` and `.fseventsd`, none of which holds a plate. Reasons for the dot-file rule as a whole are in `docs/notes/spacr/io.md` under `_listdir_visible`.

## measure_test_mode, 2026-09-19

```python
if f.endswith('.npy') and not f.startswith('.')
```

Test mode sampled every file in `merged/`, so `.spacr_plane_layout.json` and, on a macOS external volume, the `._<field>.npy` sidecar of each field (item 429) could take a field's place in `test/merged`. Test mode then measured fewer fields than `test_nr` asked for, and a sampled sidecar failed its worker. Only visible `.npy` fields are sampled now, and the "fewer than test_nr" message counts fields.

```python
if os.path.isfile(layout):
```

Sampling only `.npy` files meant the layout manifest was never copied, where before it went across on the runs where `random.sample` happened to pick it. `crops.read_merged_plane_layout` treats that manifest as the authority for the folder it sits in and returns `None` without it, which every reader takes as a legacy folder and answers with `DEFAULT_MASK_DIMS`. `measure_crop` itself is safe -- `reconcile_merged_mask_dims` runs against the real `merged/` before `measure_test_mode` -- but anything else pointed at `test/merged` (`measure.generate_object_dataset`, `crops.open_merged_field`, `align`) would read the wrong plane on a plate whose layout is not the default. The manifest is now copied beside the sampled fields whenever the source folder has one.

## process_mask_file_adjust_cell, 2026-09-19

```python
cell_mask = np.load(cell_path, allow_pickle=False)
```

The four loads (pathogen, cell, nucleus, organelle) passed `allow_pickle=True` since July 2025. Every writer of these files saves `mask.astype(np.uint16)` and did then, so the flag loaded nothing a plain load would not, and it made loading a mask run whatever a pickled `.npy` in the folder named: measured, an object array whose pickle calls `os.mkdir` created its directory when `adjust_cells` read it. Item 429 decided against `allow_pickle` for the normalised archives for the same reason. Pinned by `tests/test_spacr_masks_load_without_unpickling.py`.

```python
_save_array_atomic(cell_path, merged_cell_mask)
```

`adjust_cells` rewrites `masks/cell_mask_stack/<field>.npy` over the mask Cellpose wrote. `np.save` onto that name truncated it first and then wrote, so a run killed during the write (SIGKILL, OOM, a full disk) left a short file under the final name; before `check_mask_folder` checked masks with `resume` off, the next run took it for done. The write now goes through `spacr.io._save_array_atomic`, item 430's `_replace_atomically`: the adjusted mask is written into a hidden `.partial` sibling, flushed, and renamed over the old one, so a kill leaves the previous whole mask. Measured with `np.save` made to die half-way through the write: the old code left the first half of the file under the final name (`validate_merged_field`: truncated), the new one leaves the original bytes and no sibling. Pinned by `tests/test_a_mask_is_checked_before_it_is_reused.py`.

## check_mask_folder, 2026-09-19

```python
mask_count = sum(
    1 for path in mask_paths if validate_merged_field(path)[0])
```

With `resume` off, every `.npy` in the mask folder was counted, so a mask a killed run left empty or truncated made the count equal the stack count, the log said "All masks have been generated", segmentation was skipped, and the merge died on the short file (`ValueError: Failed to read all data for array ... (file seems not fully written?)`). Item 430 had left this: only `resume` on checked. The check reads each header and compares the file's length with it (`spacr.resume.validate_merged_field`), which costs one small read per mask, so it now runs whether or not `resume` is set. `resume` is still accepted, because callers pass it. `spacr.io._check_masks` makes the same change per field and names each damaged mask in the log as it queues it again.

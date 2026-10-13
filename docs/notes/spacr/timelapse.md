# Notes from `spacr/timelapse.py`

Prose lifted out of `spacr/timelapse.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (5 entries)
- [_npz_to_movie](#_npz_to_movie) (9 entries)
- [_scmovie](#_scmovie) (7 entries)
- [_sort_key](#_sort_key) (2 entries)
- [_masks_to_gif](#_masks_to_gif) (2 entries)
- [_timelapse_masks_to_gif](#_timelapse_masks_to_gif) (6 entries)
- [_relabel_masks_based_on_tracks](#_relabel_masks_based_on_tracks) (4 entries)
- [_require_2d_frames](#_require_2d_frames) (1 entry)
- [link_by_iou](#link_by_iou) (4 entries)
- [_find_optimal_search_range](#_find_optimal_search_range) (1 entry)
- [_track_by_iou](#_track_by_iou) (6 entries)
- [_facilitate_trackin_with_adaptive_removal](#_facilitate_trackin_with_adaptive_removal) (5 entries)
- [_trackpy_track_cells](#_trackpy_track_cells) (3 entries)
- [_trackastra_track_cells](#_trackastra_track_cells) (3 entries)
- [_ultrack_track_cells](#_ultrack_track_cells) (5 entries)
- [_track_well_ids](#_track_well_ids) (1 entry)
- [_btrack_track_cells](#_btrack_track_cells) (21 entries)
- [preprocess_pathogen_data](#preprocess_pathogen_data) (4 entries)
- [infected_vs_noninfected](#infected_vs_noninfected) (5 entries)
- [_explode_peak_ids](#_explode_peak_ids) (3 entries)
- [summarize_per_well](#summarize_per_well) (9 entries)
- [summarize_per_well_inf_non_inf](#summarize_per_well_inf_non_inf) (6 entries)
- [analyze_calcium_oscillations](#analyze_calcium_oscillations) (19 entries)
- [_generate_mask_random_cmap](#_generate_mask_random_cmap) (4 entries)
- [create_results_figure](#create_results_figure) (1 entry)
- [_make_intensity_motility_panel](#_make_intensity_motility_panel) (23 entries)
- [_make_intensity_motility_panel._plot_hist_qc](#_make_intensity_motility_panel_plot_hist_qc) (8 entries)
- [_make_intensity_motility_panel._plot_pca_qc](#_make_intensity_motility_panel_plot_pca_qc) (3 entries)
- [_make_intensity_motility_panel._plot_xgb_prob_qc](#_make_intensity_motility_panel_plot_xgb_prob_qc) (1 entry)
- [_make_intensity_motility_panel._plot_inf_uninf_bar](#_make_intensity_motility_panel_plot_inf_uninf_bar) (6 entries)
- [_make_intensity_motility_panel._plot_all_tracks](#_make_intensity_motility_panel_plot_all_tracks) (3 entries)
- [_make_intensity_motility_panel._plot_origin](#_make_intensity_motility_panel_plot_origin) (1 entry)
- [_reorient_merged_array](#_reorient_merged_array) (1 entry)
- [_parse_merged_filename](#_parse_merged_filename) (1 entry)
- [_summarise_child_features_per_parent](#_summarise_child_features_per_parent) (2 entries)
- [_load_intensity_stack_from_merged](#_load_intensity_stack_from_merged) (3 entries)
- [_load_masks_from_merged](#_load_masks_from_merged) (6 entries)
- [_compute_regionprops_stack](#_compute_regionprops_stack) (1 entry)
- [_process_merged_group](#_process_merged_group) (14 entries)
- [_smooth_tracks_and_features](#_smooth_tracks_and_features) (9 entries)
- [_debug_plot_merged_planes](#_debug_plot_merged_planes) (16 entries)
- [_infection_qc_pca_clustering](#_infection_qc_pca_clustering) (38 entries)
- [_infection_qc_pca_clustering._evaluate_embedding](#_infection_qc_pca_clustering_evaluate_embedding) (3 entries)
- [_infection_qc_pca_clustering._search_umap](#_infection_qc_pca_clustering_search_umap) (2 entries)
- [_infection_qc_pca_clustering._search_tsne._run_tsne](#_infection_qc_pca_clustering_search_tsne_run_tsne) (1 entry)
- [_infection_qc_pca_clustering._search_tsne](#_infection_qc_pca_clustering_search_tsne) (2 entries)
- [_apply_infection_intensity_qc](#_apply_infection_intensity_qc) (10 entries)
- [_compute_velocities_and_well_summary](#_compute_velocities_and_well_summary) (2 entries)
- [_validate_db_table_name](#_validate_db_table_name) (1 entry)
- [_feature_velocity_correlations](#_feature_velocity_correlations) (1 entry)
- [_make_motility_plots](#_make_motility_plots) (5 entries)
- [_select_infection_feature_columns](#_select_infection_feature_columns) (7 entries)
- [_make_adjusted_qc_panel](#_make_adjusted_qc_panel) (2 entries)
- [_infection_qc_histogram](#_infection_qc_histogram) (16 entries)
- [_infection_qc_xgboost](#_infection_qc_xgboost) (38 entries)
- [automated_motility_assay](#automated_motility_assay) (25 entries)

## Module level

### lines 11-13

```python
from .figures.style import (ROLES, TYPE_SCALE, Palette, figure_style,
```

Aliased: this module also defines its own `save_figure(fig, src, figure_number)` helper below, which would otherwise shadow this import for every call site after it. Every kept figure still goes through the format/DPI preference.

### line 30, trailing  _(unsure)_

```python
from .openmp_guard import single_threaded_openmp
```

duplicate libomp is fatal — see that module

### lines 36-39

```python
try:
```

np.trapz was REMOVED in numpy 2.0 and np.trapezoid is its replacement. The old fallback here was already dead: scipy.integrate.trapz went in SciPy 1.14 and the declared scipy>=1.12,<2.0 resolves 1.18, so under numpy 2 this module failed to import at all -- the single module in spaCR that did.

### line 42, trailing

```python
except ImportError:
```

numpy < 2.0

### lines 49-58

```python
tp = _LazyModule("trackpy")
```

Trackpy imports Numba at module import time, which makes every caller pay for a tracking backend it may never use.  Keep the existing ``tp.link_df`` / ``tp.filter_stubs`` call sites, but load the optional backend only when the Trackpy path is actually used.

It also used to fail outright: a checkout-local ``tools/coverage`` directory became a namespace package whenever coverage.py itself was not importable, and shadowed it.  That directory is ``tools/coverage_scripts`` as of 2026-09-13, so the shadow is gone -- but laziness is still right on the cost argument alone.

## _npz_to_movie

### line 75  _(unsure)_

```python
fourcc = cv2.VideoWriter_fourcc(*'XVID')
```

Define the codec and create VideoWriter object

### line 80  _(unsure)_

```python
height, width = arrays[0].shape[:2]
```

Initialize VideoWriter with the size of the first image

### line 85  _(unsure)_

```python
if frame.dtype == np.float32:
```

Handle float32 images by scaling or normalizing

### line 90

```python
elif frame.dtype == np.uint16:
```

Convert 16-bit image to 8-bit

### line 94  _(unsure)_

```python
if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] in [1, 2]):
```

Handling 1-channel (grayscale) or 2-channel images

### line 99  _(unsure)_

```python
rgb_frame = np.zeros((height, width, 3), dtype=np.uint8)
```

Create an RGB image with the first channel as red, second as green, blue set to zero

### line 101, trailing  _(unsure)_

```python
rgb_frame[..., 0] = frame[..., 0]
```

Red channel

### line 102, trailing  _(unsure)_

```python
rgb_frame[..., 1] = frame[..., 1]
```

Green channel

### line 111  _(unsure)_

```python
out.write(rgb_to_cv2(frame))
```

OpenCV's writer is a BGR boundary; the arrays above remain RGB.

## _scmovie

### line 131  _(unsure)_

```python
filename_regex = re.compile(r'(\w+)_(\w+)_(\w+)_(\d+)_(\d+).png')
```

Regular expression to parse the filename

### line 133  _(unsure)_

```python
grouped_images = defaultdict(list)
```

Dictionary to hold lists of images by plate, well, field, and object number

### line 135  _(unsure)_

```python
for filename in os.listdir(folder_path):
```

Iterate over all PNG files in the folder

### line 144  _(unsure)_

```python
images = sorted(images, key=lambda x: x[0])
```

Sort images by time using sorted and lambda function for custom sort key

### line 147  _(unsure)_

```python
max_height = max_width = 0
```

Determine the size to which all images should be padded

### line 153  _(unsure)_

```python
plate, well, field, object_number = key
```

Initialize VideoWriter

### line 159  _(unsure)_

```python
for image_path in image_paths:
```

Process each image

## _sort_key

### line 186  _(unsure)_

```python
return (plate, well, field, int(time))
```

Assuming plate, well, and field are to be returned as is and time converted to int for sorting

### line 189

```python
return ('', '', '', 0)
```

Return a tuple that sorts this file as "earliest" or "lowest"

## _masks_to_gif

### line 216, trailing  _(unsure)_

```python
random_colors[:, 3] = 1
```

Full opacity

### line 217, trailing  _(unsure)_

```python
random_colors[0] = [0, 0, 0, 1]
```

Background color

## _timelapse_masks_to_gif

### line 255  _(unsure)_

```python
name = f'{key[0]}_{key[1]}_{key[2]}'
```

Generate the name for the GIF based on plate, well, field

### line 260  _(unsure)_

```python
mask_arrays = []
```

Initialize an empty list to store masks for the current object type

### line 264  _(unsure)_

```python
array = np.load(file)
```

Load only the current time series array

### line 269  _(unsure)_

```python
mask_arrays_np = np.array(mask_arrays)
```

Convert mask_arrays list to a numpy array for processing

### line 271  _(unsure)_

```python
filenames = [os.path.basename(f) for f in file_list]
```

Generate filenames for each frame in the time series

### line 273  _(unsure)_

```python
_masks_to_gif(mask_arrays_np, gif_folder, name, filenames, object_type)
```

Create the GIF for the current time series and object type

## _relabel_masks_based_on_tracks

### line 288  _(unsure)_

```python
relabeled_masks = np.zeros(masks.shape, dtype=masks.dtype)
```

Initialize an array to hold the relabeled masks with the same shape and dtype as the input masks

### line 291  _(unsure)_

```python
for frame_number in range(masks.shape[0]):
```

Iterate through each frame

### line 293  _(unsure)_

```python
frame_tracks = tracks[tracks['frame'] == frame_number]
```

Extract the mapping for the current frame from the tracks DataFrame

### line 300  _(unsure)_

```python
relabeled_masks[frame_number][current_mask == original_label] = new_label
```

Where the current mask equals the original label, set it to the new label value

## _require_2d_frames

### lines 337-340

```python
shape = f'{len(frames)} frames of mixed shape'
```

A ragged list -- 2-D frames alongside a volume -- cannot be stacked at all, and numpy's "inhomogeneous shape after 1 dimensions" is precisely the opaque message this guard exists to replace. Describing the list is enough for the diagnostic.

## link_by_iou

### line 396  _(unsure)_

```python
bool_prev = {L: mask_prev==L for L in labels_prev}
```

Precompute masks as boolean

### line 399  _(unsure)_

```python
cost = np.ones((len(labels_prev), len(labels_next)), dtype=float)
```

Cost matrix = 1 - IoU

### lines 407-408

```python
cost[i, j] = 1 - inter/union
```

Both labels came from np.unique on their masks, so each owns at least one pixel and their union cannot be empty.

### line 410  _(unsure)_

```python
row_ind, col_ind = linear_sum_assignment(cost)
```

Solve assignment

## _find_optimal_search_range

### line 435  _(unsure)_

```python
tp.link(features, search_range=optimal_search_range, memory=memory)
```

Attempt to link features with the current search range

## _track_by_iou

### line 479  _(unsure)_

```python
labels0 = np.unique(masks[0])[1:]
```

1) initialize: every label in frame 0 starts its own track

### line 482, trailing  _(unsure)_

```python
track_map = {}
```

(frame,label) -> track_id

### line 487  _(unsure)_

```python
for t in range(1, n_frames):
```

2) iterate through frames

### line 492  _(unsure)_

```python
for L_prev, L_curr in matches:
```

a) assign matched labels to existing tracks

### line 497  _(unsure)_

```python
for L in np.unique(curr)[1:]:
```

b) any label in curr not matched → new track

### line 503  _(unsure)_

```python
records = []
```

3) flatten into DataFrame

## _facilitate_trackin_with_adaptive_removal

### line 527  _(unsure)_

```python
features = _prepare_for_tracking(masks)
```

1) initial features & filter frame 0 by area

### line 533  _(unsure)_

```python
features = _prepare_for_tracking(masks)
```

2) recompute features on filtered masks

### line 536  _(unsure)_

```python
if search_range is None:
```

3) default search_range = 2×sqrt(99th‑pct area)

### line 541

```python
for attempt in range(1, max_attempts + 1):
```

4) attempt linking, shrinking search_range on failure

### lines 547-551

```python
tracks_df = tp.link_df(features, search_range=search_range, memory=memory)
```

NB: trackpy has no 'predict' keyword (it is 'predictor=<obj>'), so tp.link_df(..., predict=True) raised TypeError on every attempt; the broad except below swallowed it and the function always ended in RuntimeError. Link the same way _find_optimal_search_range calibrates the range: plain tp.link.

## _trackpy_track_cells

### lines 594-604

```python
masks = np.asarray(masks)
```

`spacr.object.generate_cellpose_masks_sam` passes a LIST of 2-D frames, and everything below this line indexes it as an array: `_track_by_iou` reads `masks.shape[0]` and `_relabel_masks_based_on_tracks` builds `np.zeros(masks.shape, ...)`, both AttributeError on a list. In the timelapse_mode='iou' path the first of those is raised inside the retry loop of `_facilitate_trackin_with_adaptive_removal`, which swallowed it, shrank the search range 100 times and reported "Failed to track after 100 attempts" — a message about displacement for a bug about a type. One coercion at the door fixes both, and is a no-op when the caller already passes an array.

### lines 616-619

```python
tracks_df = tracks_df.rename(columns={'track_id': 'particle'})
```

_track_by_iou returns ['frame', 'original_label', 'track_id'] and no centroids, so the unconditional tracks_df['particle'] += 1 below used to raise KeyError for the advertised timelapse_mode='iou'. Map it onto the trackpy layout; x/y are needed by the track visualiser downstream.

### lines 621-627

```python
tracks_df = tracks_df.merge(features[['frame', 'original_label', 'x', 'y']], on=['frame', 'origin...
```

many_to_one: `features` is regionprops output, so a label occurs once per frame and this attaches a centroid without changing the row count. `tracks_df` is the left side because an IoU link table may legitimately hold the same (frame, label) twice — a label that both starts a track and continues another — while a duplicated (frame, label) in `features` would silently double every track row and corrupt the relabelling below.

## _trackastra_track_cells

### lines 680-681  _(unsure)_

```python
from .plot import _visualize_and_save_timelapse_stack_with_tracks
```

Function-local, matching the sibling trackers: spacr.utils imports torch, and spacr.plot pulls the whole plotting stack.

### lines 689-690

```python
raise RuntimeError(
```

Fail loud and actionable rather than surfacing a bare ImportError from three frames down. trackastra is an optional dependency.

### lines 712-713  _(unsure)_

```python
ctc_df, masks_tracked = graph_to_ctc(track_graph, masks, outdir=None)
```

graph_to_ctc gives the canonical (label, start_frame, end_frame, parent) table plus a relabelled stack whose ids are consistent across frames.

## _ultrack_track_cells

### lines 921-922  _(unsure)_

```python
from .plot import _visualize_and_save_timelapse_stack_with_tracks
```

Function-local, matching the sibling trackers: spacr.utils imports torch, and spacr.plot pulls the whole plotting stack.

### lines 930-932

```python
raise RuntimeError(
```

Fail loud and actionable rather than surfacing a bare ImportError from three frames down. ultrack is an optional dependency, and a bare ImportError here reads to the user as "no data".

### lines 960-962

```python
work_dir = tempfile.mkdtemp(prefix='spacr_ultrack_')
```

mkdtemp + finally rather than TemporaryDirectory: the sqlite file can still be held open by a worker when the solve ends, and ignore_errors keeps a locked file from turning a finished run into a traceback.

### lines 975-976

```python
sigma = float(contour_sigma)
```

sigma=None means "no smoothing" to Ultrack; 0.0 is the spaCR-side spelling of the same thing because the GUI has no tri-state float.

### lines 986-988

```python
masks_tracked = np.asarray(tracks_to_zarr(config, tracks_table))
```

tracks_to_zarr paints track_id into the segmentation, so the exported stack already has ids consistent across frames. Materialise it before the working directory goes away — the zarr may be backed by it.

## _track_well_ids

### lines 1083-1085

```python
if str(row) == _UNPARSED_KEY and not warned_unparsed:
```

rowID == columnID == the well exactly as it was written (or the 'error' sentinel). Both are the well this row belongs to; neither has a row/column decomposition to render.

## _btrack_track_cells

### lines 1182-1184  _(unsure)_

```python
if isinstance(masks_3D, list):
```

Normalise masks_3D to a 3D ndarray (T, Y, X)

### line 1215  _(unsure)_

```python
if radius is None:
```

Auto radius if requested

### line 1222  _(unsure)_

```python
FEATURES = [
```

Shape-based features only (robust + what your config already expects)

### lines 1245-1249

```python
columns = (
```

Do not construct BayesianTracker for an empty segmentation. Besides doing no useful work, construction loads btrack's native ``libtracker`` and can fail on an otherwise valid machine whose libstdc++ is older than the wheel's build toolchain. Empty input has a complete, deterministic answer without that native dependency.

### line 1275  _(unsure)_

```python
CONFIG_FILE = btrack_datasets.cell_config()
```

Fetch/configure the motion model only when there is something to track.

### line 1284  _(unsure)_

```python
tracker.update_method = BayesianUpdates.APPROXIMATE
```

Use APPROXIMATE updates for large datasets (recommended by btrack docs)

### line 1288  _(unsure)_

```python
tracker.features = FEATURES
```

Features used by the visual model

### line 1306  _(unsure)_

```python
logger.debug(
```

Fallback for older btrack APIs

### lines 1319-1321  _(unsure)_

```python
do_optimize = bool(run_optimization)
```

Global optimisation (GLPK) – conditionally disabled for large problems

### line 1334  _(unsure)_

```python
glpk_options = {}
```

Build GLPK options from user parameters

### line 1337  _(unsure)_

```python
glpk_options["tm_lim"] = int(optimizer_time_limit_s * 1000)
```

GLPK tm_lim is in milliseconds

### line 1360  _(unsure)_

```python
logger.warning(
```

If GLPK misbehaves, fall back to pre-optimisation tracks

### line 1368  _(unsure)_

```python
tracks = tracker.tracks
```

After this point, tracker.tracks always contains the tracks we will use

### line 1390  _(unsure)_

```python
if timelapse_remove_transient and not tracks_df.empty:
```

Optionally remove transient tracks (very short trajectories)

### lines 1397-1400

```python
logger.warning(
```

btrack completes normally on a batch where nothing was segmented, but pd.DataFrame([]) has no columns at all, so the rounding and the merge below used to raise KeyError: 'x'. Give the empty frame its schema so the no-tracks case flows through to an all-zero mask stack.

### lines 1409-1411  _(unsure)_

```python
logger.debug("Preparing objects_df from masks_3D...")
```

Map track positions back to original labels

### line 1416  _(unsure)_

```python
tracks_df["x"] = tracks_df["x"].round(2)
```

Harmonise precision before merge

### lines 1423-1430

```python
merged_df = pd.merge(
```

many_to_many, and deliberately so: this is a POSITIONAL join. btrack reports track positions and regionprops reports object centroids, and the two are matched by rounding both to 2 decimals. Two objects in one frame can round to the same centroid (touching or nested masks do it), and one object can be claimed by two tracks at a merge/split event, so neither side is unique on the key and a stricter contract would crash a run that is merely ambiguous. The ambiguity is real: such rows produce duplicate (track_id, frame) pairs downstream.

### lines 1443-1446

```python
logger.warning(
```

Series.apply on an empty Series returns an empty Series rather than the 5-column frame the assignment expects, so the metadata block below used to raise ValueError: Columns must be same length as key (not caught by the IndexError handler). Emit an empty, correctly-shaped table instead.

### lines 1457-1474

```python
final_df['wellID'] = _track_well_ids(
```

Composed from the row and column _map_wells just parsed, not re-split off the file name: the positional split copied the file's own spelling through while rowID/columnID beside it were canonicalised, so a track table could say wellID 'a1' next to rowID 'r1' / columnID 'c1'. Composing makes the three agree character for character ('a1', 'A-01' and ' A01 ' all become 'A01'); 'A01' either way for a name already written that way.

It does NOT repair a plate id containing an underscore -- an earlier version of this comment claimed it did. schema.parse_ field_stem splits a field stem LEFT to right and takes parts [0:3], so 'exp_plate1_A01_3' puts 'plate1' in the well slot as a positional passthrough, which is exactly what token [1] said too.

Through _track_well_ids, NOT schema.well_id directly: that one raises for a positional well ('plate1_5_3') and for the 'error' sentinel _map_wells returns above, both of which reach here on data this pipeline has always tracked. See _track_well_ids.

### line 1513  _(unsure)_

```python
mask_stack = _masks_to_masks_stack(masks)
```

Return in your standard mask stack format

## preprocess_pathogen_data

### line 1614  _(unsure)_

```python
parasite_counts = pathogen_df.groupby(group_keys).size().reset_index(name='parasite_count')
```

Group by identifiers and count the number of parasites

### line 1617  _(unsure)_

```python
value_columns = [
```

Aggregate numerical columns and take the first of object columns

### lines 1630-1632

```python
pathogen_agg = pathogen_agg.merge(parasite_counts, on=group_keys,
```

Merge the counts back into the aggregated data. one_to_one: both sides are groupby(group_keys) reductions of the same frame, so each holds exactly one row per host cell and the join must not change the row count.

### lines 1641-1642  _(unsure)_

```python
pathogen_agg.rename(columns={'cell_id': 'object_label'}, inplace=True)
```

The host-cell link becomes this frame's object_label, so it merges straight onto the cell table's own object_label.

## infected_vs_noninfected

### line 1669  _(unsure)_

```python
infected_cells_df = result_df[result_df.groupby('plate_row_column_field_object')['parasite_count'...
```

Separate the merged dataframe into two groups based on pathogen_count

### line 1673  _(unsure)_

```python
with figure_style(theme_target()):
```

Plotting

### line 1677  _(unsure)_

```python
for group_id in infected_cells_df['plate_row_column_field_object'].unique():
```

Plot for cells that were infected at some time

### line 1682

```python
for group_id in uninfected_cells_df['plate_row_column_field_object'].unique():
```

Plot for cells that were never infected

### line 1687  _(unsure)_

```python
axs[0].set_title('Cells Infected at Some Time')
```

Set the titles and labels

## _explode_peak_ids

### lines 1780-1790

```python
head, separator, tail = text.rpartition(schema.KEY_SEPARATOR)
```

Legacy spelling: prcf + '_' + a BARE object label, which is what this module wrote before the object index gained its 'o' prefix (see the 'plate_row_column_field_object' composition in analyze_calcium_oscillations). A peak_details.csv saved by an older run still carries it, and it is still read right to left here.

The trailing token must be all digits, not merely something schema.object_id would accept: 't3' would be accepted (as object 3) and would turn a plain timelapse *prcf* into a plausible-looking object key, which is the silent misreading this function exists to stop. An id with no object in it is not an object id.

### lines 1811-1815

```python
for column in schema.FIELD_KEY_COLUMNS:
```

np.array, not a list and not a Series: a list of nothing gives a float64 column, and the well id built from it below then fails with a ufunc type error on an empty frame; a Series would align on the LABEL and write the wrong rows when the caller's index repeats. An object-dtype array assigns positionally and keeps its dtype whatever the length.

### lines 1819-1820  _(unsure)_

```python
peak_details_df['object_number'] = np.array(
```

'object_number' keeps the 'o<N>' spelling the split produced, so it stays a string and does not join the numeric columns that get averaged below.

## summarize_per_well

### lines 1835-1836  _(unsure)_

```python
_explode_peak_ids(peak_details_df, 'summarize_per_well')
```

Step 1: Recover the identity from the 'ID' key (see _explode_peak_ids for why this is a right-to-left parse and not a positional split).

### line 1839  _(unsure)_

```python
peak_details_df['well_ID'] = peak_details_df['rowID'] + '_' + peak_details_df['columnID']
```

Step 2: Create 'well_ID' by combining 'rowID' and 'columnID'

### line 1842  _(unsure)_

```python
filtered_df = peak_details_df[peak_details_df['amplitude'].notna()]
```

Filter entries where 'amplitude' is not null

### line 1845  _(unsure)_

```python
numeric_cols = filtered_df.select_dtypes(include=['number']).columns
```

Preparation for Step 3: Identify numeric columns for averaging from the filtered dataframe

### line 1848  _(unsure)_

```python
summary_df = filtered_df.groupby('well_ID').agg(
```

Step 3: Calculate summary statistics

### line 1851, trailing  _(unsure)_

```python
unique_IDs_with_amplitude=('ID', 'nunique'),
```

Count unique IDs per well with non-null amplitude

### line 1852, trailing  _(unsure)_

```python
**{col: (col, 'mean') for col in numeric_cols}
```

exclude 'amplitude' from averaging if it's numeric

### lines 1855-1873

```python
peak_details_df['_field_object'] = (
```

Step 3: how many CELLS the well holds.

FIELD + OBJECT, and it has to be exactly that pair -- both halves are load-bearing and each one alone is wrong in a different direction.

`object_number` alone UNDERCOUNTS. It is the label the segmenter assigned within a FIELD and restarts at 1 in every one, while `well_ID` is row+column and spans them all, so nunique() returned the size of the well's largest field. Four fields of ~60 cells is ~240, reported as ~60, and peaks_per_cell came out four times too high.

`ID` alone OVERCOUNTS on timelapse data. A timelapse key carries the timepoint -- plate1_r1_c1_f1_t3_o7 -- so the same tracked cell at t3 and t4 counts as two, and peaks_per_cell comes out too LOW. `test_a_timelapse_object_key_keeps_its_timepoint_out_of_the_identity` states the contract: the object key identifies a TRACK.

field + object is right for both: the field disambiguates the restarting labels, and the timepoint is left out of the identity.

### lines 1881-1886

```python
summary_df = summary_df.merge(summary_df_2, on='well_ID', how='left',
```

Join on well_ID rather than assigning the column positionally: summary_df is built from the amplitude-filtered frame, so a well whose peaks all have a null amplitude is missing from it and every later well used to inherit the previous well's cell count (and an empty summary grew ghost NaN rows). one_to_one: both frames are groupby('well_ID') reductions, so a well appears at most once in each and this join adds a column, never a row.

## summarize_per_well_inf_non_inf

### line 1906  _(unsure)_

```python
_explode_peak_ids(peak_details_df, 'summarize_per_well_inf_non_inf')
```

Step 1: Recover the identity from the 'ID' key (see _explode_peak_ids).

### line 1909  _(unsure)_

```python
peak_details_df['well_ID'] = peak_details_df['rowID'] + '_' + peak_details_df['columnID']
```

Step 2: Create 'well_ID' by combining 'rowID' and 'columnID'

### lines 1912-1913  _(unsure)_

```python
peak_details_df['infected_status'] = peak_details_df['infected'].apply(lambda x: 'infected' if x ...
```

Assume 'pathogen_count' indicates infection if > 0

Add an 'infected_status' column to classify cells

### line 1916  _(unsure)_

```python
numeric_cols = peak_details_df.select_dtypes(include=['number']).columns
```

Preparation for Step 3: Identify numeric columns for averaging

### lines 1919-1921

```python
peak_details_df['_field_object'] = (
```

Step 3: Calculate summary statistics field + object, not object_number and not ID -- see summarize_per_well for why each alone is wrong in a different direction.

### line 1931  _(unsure)_

```python
summary_df['peaks_per_cell'] = summary_df['peaks_per_well'] / summary_df['cells_per_well']
```

Calculate peaks per cell

## analyze_calcium_oscillations

### line 1964  _(unsure)_

```python
conn = sqlite3.connect(db_loc, timeout=30)
```

Load data

### line 1966  _(unsure)_

```python
cell_df = pd.read_sql(f"SELECT * FROM {'cell'}", conn)
```

Load cell table

### lines 1969-1973

```python
merge_keys = _object_group_keys(cell_df, 'object_label')
```

The merge keys are the ones the measurement writer emits: columnID (not column_name), timeID (not timeid) and, in the child tables, cell_id (not pathogen_cell_id). Every one of the old names was absent from a real measurements.db, so this function raised KeyError before it ever merged. timeID is only present for a timelapse run, hence resolved per frame.

### lines 1988-1993

```python
cell_df = cell_df.merge(pathogen_df, on=merge_keys, how='left', suffixes=('', '_pathogen'), valid...
```

many_to_one: preprocess_pathogen_data aggregates the parasite table to one row per host cell, so this attaches a parasite count without changing the number of cell rows. The left side is not asserted unique because a database with no timeID column keys several frames of the same cell alike; the right side must be, or every one of those frames would be duplicated and the peak counts inflated with it.

### line 1998  _(unsure)_

```python
if cytoplasm:
```

Optionally load cytoplasm table and merge

### lines 2001-2006

```python
cell_df = cell_df.merge(cytoplasm_df, on=merge_keys, how='left', suffixes=('', '_cytoplasm'), val...
```

Merge on specified columns. many_to_one: the cytoplasm table carries one object per cell per frame, so it must not hold two rows for a merge key -- that would duplicate the cell's whole intensity trace and double every peak detected in it. Raising here is the point: the duplication is invisible in the result, which just looks like a cell with twice as many timepoints.

### lines 2013-2018

```python
parsed_prcf = [schema.parse_prcf(value) for value in cell_df['prcf']]
```

Continue with your existing processing on cell_df now containing merged data... Prepare DataFrame (use cell_df instead of df) schema.parse_prcf reads the key right to left, so a plate id that itself contains an underscore does not shift every column one place along, and the optional timepoint is recognised by being a 't<N>' rather than by being the fifth token.

### lines 2022-2027

```python
time_key = _resolve_time_key(cell_df)
```

The time axis comes from the timeID column when the database has one and from the trailing 't<N>' element of prcf otherwise. A non-timelapse database has neither, and the old positional prcf_components[4] used to raise a bare KeyError(4) on it; there is no oscillation to measure without a time axis, so say so and stop like the other unanalysable cases below.

### lines 2041-2045

```python
cell_df['plate_row_column_field_object'] = [
```

'o' prefixes the object index the way every other spaCR object key spells it (prcfo is plate_row_column_field[_time]_o<N>), so plate1_r1_c1_f1_o2 can no longer be misread as a fifth well coordinate. This key deliberately omits the time element: it identifies one cell's track ACROSS time, which is what the per-track groupby below needs.

### lines 2051-2054

```python
if 'parasite_count' not in cell_df.columns:
```

'parasite_count' only exists when the (optional) pathogen table was merged above. The per-track loop below reads it unconditionally, so the documented default call (pathogen=None) used to die with KeyError. Default to 0, i.e. every cell uninfected, which is the right answer with no pathogen data.

### line 2060  _(unsure)_

```python
try:
```

Fit exponential decay model to all scaled fluorescence data

### line 2070  _(unsure)_

```python
corrected_dfs = []
```

Normalizing corrected fluorescence for each cell

### line 2098  _(unsure)_

```python
peaks, properties = find_peaks(group['delta_' + measurement], height=peak_height)
```

Detect peaks

### line 2101  _(unsure)_

```python
group_filtered = group.copy()
```

Set values < 0 to 0

### line 2121, trailing  _(unsure)_

```python
'time': np.nan,
```

The time of the peak

### line 2130  _(unsure)_

```python
for i, peak in enumerate(peaks):
```

Inside the for loop where peaks are detected

### line 2140  _(unsure)_

```python
peak_segment_y = group['delta_' + measurement].iloc[start_idx:end_idx + 1]
```

Using indices to slice for AUC calculation

### line 2182  _(unsure)_

```python
with figure_style(theme_target()):
```

Plotting

### line 2208  _(unsure)_

```python
infected_cells = result_df[result_df.groupby('plate_row_column_field_object')['parasite_count'].t...
```

Identify cells with and without pathogens

## _generate_mask_random_cmap

### line 2240  _(unsure)_

```python
num_objects = np.sum(unique_labels != 0)
```

Only count non-zero labels as objects

### line 2242  _(unsure)_

```python
random_colors = np.random.rand(num_objects + 1, 4)
```

+1 so index 0 is background

### line 2244, trailing  _(unsure)_

```python
random_colors[:, 3] = 1.0
```

full alpha

### line 2245  _(unsure)_

```python
random_colors[0, :] = [0.0, 0.0, 0.0, 1.0]
```

background = black, fully opaque

## create_results_figure

### lines 2257-2259

```python
with figure_style(theme_target()):
```

The context has to be open when the AXES are created, not only the Figure: rcParams reach an artist at construction, so the spines, the ticks and the label colour are decided by these four lines.

## _make_intensity_motility_panel

### line 2307, trailing  _(unsure)_

```python
import matplotlib.image as mpimg
```

used for the small QC PNG in mask panel

### lines 2316-2318  _(unsure)_

```python
label_lower = str(label_tag).lower()
```

Panel type / strategy / QC payload availability

### line 2331  _(unsure)_

```python
hist_data = settings.get("infection_hist_data", None)
```

Global QC payloads (built in QC helpers)

### line 2339  _(unsure)_

```python
qc_panel_needed_mask = (
```

Mask panel: small embedded QC PNG if available

### line 2348  _(unsure)_

```python
qc_axes_count = 0
```

Adjusted panel: method-specific QC axes

### line 2351  _(unsure)_

```python
if qc_strategy == "histogram":
```

Histogram: we *always* allocate one QC axis and can compute from df_well

### line 2354  _(unsure)_

```python
elif qc_strategy in {"pca", "umap", "tsne"} and has_pca:
```

PCA/UMAP/t-SNE: need pca_data for embedding

### line 2357  _(unsure)_

```python
elif qc_strategy == "xgboost" and has_xgb:
```

XGBoost: probability separation + feature importance

### line 2361  _(unsure)_

```python
origin_xlim = settings.get("motility_xlim", settings.get("motility_origin_xlim"))
```

Motility axis limits: driven by motility_xlim / motility_ylim

### line 2365  _(unsure)_

```python
if pixels_per_um is not None and pixels_per_um > 0:
```

Coordinate scaling

### line 2731  _(unsure)_

```python
df_well = all_df[
```

Subset data for this well

### line 2753  _(unsure)_

```python
available_channels = [
```

Determine which channels are available *for this well*

### line 2766  _(unsure)_

```python
has_p75_path = False
```

Extra intensity plots for pathogen channel

### lines 2781-2783

```python
n_cols = n_int_plots + 3 + (1 if qc_panel_needed_mask else 0) + qc_axes_count
```

+3 for: all-tracks motility, infected origin, uninfected origin +1 for small QC PNG in mask panel, +qc_axes_count for adjusted panel QC subplots

### lines 2788-2791

```python
axes = np.array(axes).ravel()
```

`plt.subplots` returns a bare Axes only for a 1x1 grid, and n_cols cannot be 1: `available_channels` is non-empty by the guard above and the sum adds a fixed 3 on top of it, so the smallest panel is four columns wide and `axes` is always an array.

### line 2796  _(unsure)_

```python
for ch in available_channels:
```

intensity violins per channel (per well)

### line 2809  _(unsure)_

```python
if pathogen_chan is not None and ch == pathogen_chan:
```

If this is the pathogen channel, append p75 and ratio plots if available

### line 2965  _(unsure)_

```python
if qc_panel_needed_mask and axis_idx < len(axes):
```

optional small QC PNG (mask panel only)

### line 2993

```python
src = hist_data if hist_data is not None else df_well
```

Prefer global payload if present; otherwise compute from per-well df

### line 3003  _(unsure)_

```python
ax_prob = axes[axis_idx]
```

qc_axes_count reserves exactly these two slots.

### line 3011  _(unsure)_

```python
meta_tag = f"{plate_id}_{well_id}"
```

Plate/well tag for title & filename

### lines 3020-3022

```python
if is_adjusted_panel:
```

Filenames:

mask/original: plate1_A03.pdf adjusted:      plate1_A03_xgboost_adjusted.pdf

### line 3028  _(unsure)_

```python
out_name = f"{meta_tag}_{label_tag}_{method_label}.pdf"
```

fallback for any unexpected label_tag

## _make_intensity_motility_panel._plot_hist_qc

### lines 2377-2379  _(unsure)_

```python
def _plot_hist_qc(ax, source):
```

Helpers for QC subplots (used in adjusted panel)

### line 2393  _(unsure)_

```python
if isinstance(source, dict):
```

Case 1: payload dict from settings

### line 2401  _(unsure)_

```python
df_vals = source
```

Case 2: compute from per-well DataFrame

### line 2404  _(unsure)_

```python
intensity_col = settings.get("infection_hist_intensity_col", None)
```

Decide which intensity column to use

### line 2429  _(unsure)_

```python
cell_level = (
```

Collapse to one value per cell-track

### lines 2446-2449

```python
all_vals = np.concatenate(
```

No "both sides empty" skip: `mask_inf` and `~mask_inf` partition `cell_level`, which the `cell_level.empty` guard above has already established is non-empty, so the two arrays cannot both have size 0.

### line 2462  _(unsure)_

```python
ax.hist(
```

Now plot

### line 2478  _(unsure)_

```python
reference_line(ax, x=thr_val)
```

A threshold is a reference, not a result: thin, dashed, grey.

## _make_intensity_motility_panel._plot_pca_qc

### line 2505

```python
method_label = str(pdata.get("method_label", "PCA"))
```

Method label stored by _infection_qc_pca_clustering: 'PCA', 'UMAP', or 't-SNE'

### line 2528  _(unsure)_

```python
ax.set_xlabel(f"{method_label} 1")
```

Generic axis labels that respect the embedding method

### line 2532  _(unsure)_

```python
ax.set_title(f"{method_label} of features\n(adjusted labels)")
```

Title also reflects method

## _make_intensity_motility_panel._plot_xgb_prob_qc

### line 2566  _(unsure)_

```python
prob_col_candidates = []
```

Resolve probability column

## _make_intensity_motility_panel._plot_inf_uninf_bar

### lines 2635-2638

```python
cell_level = (
```

No "column missing" skip: all three call sites below name a column they have just found in the frame they pass -- the per-channel one comes from `available_channels`, the p75 one from `has_p75_path`, and `rel_intensity` is computed on the frame one line before the call.

### lines 2676-2678

```python
vp = ax.violinplot(
```

No "nothing to draw" skip: `mask_inf` and `~mask_inf` partition `cell_level`, non-empty by the guard above, so at least one of `vals_inf` / `vals_uninf` has a size and `data` always gets a member.

### line 2680  _(unsure)_

```python
vp = ax.violinplot(
```

Violin plots

### line 2690  _(unsure)_

```python
ink = resolve_ink(theme_target())
```

Infected takes the highlight, uninfected the control grey.

### lines 2697-2698

```python
means = [float(np.nanmean(d)) for d in data]
```

Overlay means, in the ink rather than a hard-coded black that disappears into spaCR's dark ground.

### line 2704  _(unsure)_

```python
flat = np.concatenate(data)
```

If all values are non-negative, anchor at 0

## _make_intensity_motility_panel._plot_all_tracks

### lines 2844-2847

```python
xs_all = []
```

`well_tracks` is non-empty: the per-well guard above skips the whole well -- and never reaches this figure -- when it is not. A track too short to draw is a different case, and the `xs_all` check below still handles it.

### line 2880  _(unsure)_

```python
x_margin = 0.05 * (xs_all.max() - xs_all.min() + 1e-9)
```

auto limits from data

### lines 2910-2912

```python
)
```

NO BOX. A rounded white panel is furniture the style has no other boxes to match, and on the dark theme it is a white rectangle over the data.

## _make_intensity_motility_panel._plot_origin

### line 2919  _(unsure)_

```python
def _plot_origin(ax, want_infected: bool):
```

motility origin plots (infected vs uninfected) for this well

## _reorient_merged_array

### line 3176  _(unsure)_

```python
plane_axis = int(np.argmin(shape))
```

Fallback: choose the smallest axis as planes

## _parse_merged_filename

### line 3225  _(unsure)_

```python
digits = "".join(ch for ch in time_str if ch.isdigit())
```

Extract numeric time index, tolerate formats like "t000"

## _summarise_child_features_per_parent

### lines 3311-3315

```python
df = overlaps_df.merge(child_props_df, on=["frame", child_label_col], how="left", validate="many_...
```

many_to_one: overlaps_df holds one row per (frame, parent, child) pair, so a child shared by two parents appears twice on the left and that is the overlap being summarised. child_props_df is regionprops output, one row per label per frame; a duplicate there would count the same child twice into n_children and skew every aggregate below it.

### lines 3352-3354

```python
summary = agg_df.merge(counts, on=group_cols, how="left",
```

one_to_one: both sides are groupby(group_cols) reductions of the same frame, so each parent appears once in each and the summary must keep exactly one row per parent object per frame.

## _load_intensity_stack_from_merged

### line 3401  _(unsure)_

```python
try:
```

Standardise to (planes, H, W)

### line 3410  _(unsure)_

```python
print(
```

Skip unexpected size

### line 3422, trailing  _(unsure)_

```python
img = arr[:use_planes].transpose(1, 2, 0)
```

(H, W, C)

## _load_masks_from_merged

### line 3486  _(unsure)_

```python
try:
```

Standardise to (planes, H, W)

### line 3503  _(unsure)_

```python
continue
```

Only intensity planes, no masks

### line 3508  _(unsure)_

```python
cell_masks[t] = arr[n_channels].astype(dtype)
```

First mask plane is always cell

### line 3511  _(unsure)_

```python
if n_masks >= 2:
```

Second mask plane (if present) is nucleus OR pathogen depending on settings

### line 3518  _(unsure)_

```python
nucleus_masks[t] = arr[n_channels + 1].astype(dtype)
```

both requested → expect nucleus here

### line 3521  _(unsure)_

```python
if n_masks >= 3 and pathogen_chan is not None:
```

Third mask plane (if present) is pathogen when both nuc+pathogen exist

## _compute_regionprops_stack

### line 3566

```python
geom_props = [
```

Avoid properties that rely on normalized central moments

## _process_merged_group

### line 3660  _(unsure)_

```python
metas = []
```

sort filenames by timeID

### line 3675  _(unsure)_

```python
first_path = os.path.join(merged_dir, sorted_basenames[0])
```

infer size from first file (respecting orientation)

### line 3734  _(unsure)_

```python
has_nucleus = np.any(nucleus_masks)
```

cytoplasm = cell minus (nucleus union pathogen)

### line 3745  _(unsure)_

```python
cell_props_df = _compute_regionprops_stack(
```

regionprops for cell geometry (+ intensities in cell_chan)

### line 3773, trailing  _(unsure)_

```python
channel_index=cell_chan,
```

use same channel as cell by default

### line 3778  _(unsure)_

```python
percentile_dfs_cell = []
```

per-channel intensity percentiles for each compartment

### line 3785  _(unsure)_

```python
df_p = _compute_intensity_percentiles_per_channel(
```

cell: track_id labels

### lines 3832-3839

```python
if percentile_dfs_cell:
```

merge percentile features into base props.

Every frame joined below — the per-channel percentile tables and the props tables they are attached to — is one row per object per frame: regionprops emits a label once per frame, and the percentile helpers reduce each label to one row. So all of these are one_to_one, and a violation means a label was measured twice in a frame, which would duplicate that object's whole row and be invisible in the output.

### line 3886  _(unsure)_

```python
per_channel_intensity_dfs = []
```

per-channel cell mean intensities (one column per channel)

### lines 3901-3902

```python
cell_intensity_df = cell_intensity_df.merge(
```

one_to_one: one mean per track per frame per channel, so widening by channel must not add rows.

### line 3919  _(unsure)_

```python
nucleus_summary = None
```

overlaps and summaries

### lines 3989-3993

```python
if nucleus_summary is not None and not nucleus_summary.empty:
```

enriched_df is the per-(frame, track) cell table and must stay exactly that: every summary attached below has already been reduced to one row per parent by _summarise_child_features_per_parent, so one_to_one holds and a breach would multiply the cell rows the whole downstream QC (track velocities, infection calls, the SQLite snapshot) counts.

### line 4024  _(unsure)_

```python
meta_records = []
```

attach metadata (plate, well, field, timeID, etc.)

### lines 4032-4034

```python
enriched_df = enriched_df.merge(meta_df, on="frame", how="left",
```

many_to_one: meta_df is built by enumerating the frames, so it holds one row per frame index, while enriched_df holds one row per object per frame. A duplicated frame in meta_df would clone every object in it.

## _smooth_tracks_and_features

### line 4081  _(unsure)_

```python
candidate_cols = [
```

Only smooth scalar features with well-defined numeric dtype

### line 4095

```python
for col in [y_col, x_col] + cell_feature_cols:
```

Ensure we are not writing floats into int columns (avoid FutureWarning)

### lines 4113-4116

```python
y = g[y_col].to_numpy(dtype=float, copy=True)
```

copy=True is required: for an already-float64 column to_numpy returns a VIEW onto the group's buffer, so the in-place glitch repair below also mutated `g` and the write-back guard (y[i] != g[y_col].iloc[i]) could never fire - the corrected centroid was silently dropped.

### line 4122  _(unsure)_

```python
if n >= 3:
```

1) detect and interpolate single-frame centroid glitches

### line 4139  _(unsure)_

```python
for i_local in glitch_frames:
```

interpolate centroid + scalar features at glitch frames

### line 4153  _(unsure)_

```python
drop_track = False
```

2) drop tracks with big jumps not explainable as glitches

### line 4170  _(unsure)_

```python
for i_local, global_idx in enumerate(idx):
```

write back smoothed centroid

### line 4177  _(unsure)_

```python
if len(idx) < 3 or not cell_feature_cols:
```

3) z-score based smoothing of scalar features

### line 4202  _(unsure)_

```python
for col, mapping in updates.items():
```

apply all updates in one go

## _debug_plot_merged_planes

### line 4235, trailing  _(unsure)_

```python
import matplotlib as mpl
```

needed by _generate_mask_random_cmap if defined elsewhere

### line 4245  _(unsure)_

```python
if arr.ndim == 3:
```

Re-orient to (planes, y, x)

### line 4253  _(unsure)_

```python
if arr.shape[-1] >= n_channels:
```

Take first timepoint; assume (T, Y, X, planes) or similar

### line 4257  _(unsure)_

```python
planes = arr.reshape(-1, arr.shape[-2], arr.shape[-1])
```

fallback: collapse time into planes

### line 4260  _(unsure)_

```python
planes = arr
```

Fallback, try to interpret leading axis as planes

### line 4284

```python
norm_intensity = []
```

Normalize intensity channels to 2–98 percentiles

### line 4305  _(unsure)_

```python
merged_rgb[..., 0] = norm_intensity[0]  # red
```

The empty-channel case returned above before norm_intensity[0] was read.

### line 4306, trailing  _(unsure)_

```python
merged_rgb[..., 0] = norm_intensity[0]
```

red

### line 4308, trailing  _(unsure)_

```python
merged_rgb[..., 1] = norm_intensity[1]
```

green

### line 4310, trailing  _(unsure)_

```python
merged_rgb[..., 2] = norm_intensity[2]
```

blue

### line 4312  _(unsure)_

```python
combined_mask = None
```

Combined mask for overlay

### line 4327  _(unsure)_

```python
extra = 1 if combined_mask is not None else 0
```

Figure layout: channels + masks + merged overlay

### line 4343  _(unsure)_

```python
for ch_idx in range(n_channels):
```

Intensity channels

### line 4351  _(unsure)_

```python
for m_idx in range(n_masks):
```

Individual mask planes with random cmap

### line 4359  _(unsure)_

```python
unique_labels = np.unique(mask_plane)
```

Fallback: create a simple random colormap here

### line 4371  _(unsure)_

```python
if combined_mask is not None:
```

Merged channels + combined masks

## _infection_qc_pca_clustering

### line 4490  _(unsure)_

```python
try:
```

Optional imports for alternative embeddings

### line 4493, trailing  _(unsure)_

```python
except Exception:
```

optional

### lines 4497-4499

```python
from .utils import umap  # type: ignore
```

Through spacr.utils, never a bare `import umap`: umap's package init__ imports umap.parametric_umap -> tensorflow, and TF is not a spaCR dependency. The lazy wrapper blocks it for that import.

### lines 4501-4502

```python
umap.UMAP  # noqa: B018
```

Force the deferred import here, where this except clause can still turn a failure into umap = None.

### line 4504, trailing  _(unsure)_

```python
except Exception:
```

optional

### lines 4507-4510

```python
source_all_df = all_df
```

Keep the caller's measurements frame as the durable record.  Feature coercion below is deliberately limited to a candidate-only view, and the merge near the end writes to a shallow working copy before changing key dtypes or adding the adjusted call.

### lines 4754-4757  _(unsure)_

```python
strategy = str(settings.get("infection_intensity_strategy", "pca")).lower()
```

🔴 Key change is here 🔴

Use infection_intensity_strategy to define embedding method

### line 4764  _(unsure)_

```python
settings["infection_pca_method"] = embed_method
```

keep settings in sync so downstream code can use this if needed

### lines 4767-4770

```python
key_cols = ["plateID", "wellID", "fieldID", "cellID"]
```

A second `if embed_method not in {"pca", "umap", "tsne"}: embed_method "pca"` stood here and could not run: the branch above assigns either `strategy`, which is in that set, or the literal "pca". It was removed rather than excluded from coverage.

### line 4779

```python
cols_to_drop = [
```

Drop any existing adjusted_infected to avoid _x/_y columns on merge

### lines 4788-4791

```python
pathogen_token = f"ch{pathogen_chan}".lower()
```

The infection call cannot also be a grouping key or a feature. Both make the aggregation frame name one column twice.  Check the name-based candidate set before coercion so even a text-backed feature cannot evade this guard merely because pandas typed it ``object``.

### lines 4816-4823

```python
candidate_frame = schema.coerce_model_feature_types(
```

Build per-cell feature table

SQLite/pandas represents both numeric text and an all-NULL REAL column as object dtype.  Filtering on dtype first silently discarded the former and made the advertised PCA/UMAP/t-SNE QC a no-op.  Coerce only columns this model can actually use: unrelated metadata and other fluorescence channels must neither be converted nor turn into schema errors.

### lines 4840-4842

```python
tmp = all_df[key_cols + [infection_col]].copy()
```

Start from the durable columns and attach the coerced candidate values to this disposable aggregation frame.  The returned frame therefore keeps the database's original dtypes and metadata exactly as supplied.

### line 4851  _(unsure)_

```python
inf_any = group[infection_col].max().reset_index()
```

any cell that was ever infected in the time series is treated as infected

### lines 4853-4855

```python
cell_level = cell_level.merge(inf_any, on=key_cols, how="left", suffixes=("", "_y"), validate="on...
```

one_to_one: both sides come out of the same groupby(key_cols), so this widens the per-cell table by one column and must not add a row -- the PCA below assumes cell_level is row-aligned with the arrays it builds from it.

### lines 4858-4867

```python
cell_level[infection_col] = cell_level[infection_col].fillna(0).astype(bool)
```

A recovery block stood here -- `if infection_col not in cell_level.columns:` followed by a hunt through `<col>_y` and `<col>_x` and a second skip -- and no input could reach it. `inf_any` is `group[infection_col].max()`, so it always carries the column; the guard above has already refused the two cases where the left side could carry it as well and the suffix could move it; and `suffixes=("", "_y")` can never mint an `_x` at all. It was removed rather than excluded from coverage. `test_the_infection_call_cannot_be_a_feature_as_well` and its sibling assert the guard that makes it dead, so if the invariant stops holding a test says so.

### lines 4870-4872  _(unsure)_

```python
intensity_col = None
```

Decide pathogen-channel intensity column (needed for ground truth)

### lines 4892-4896

```python
morph_cols = [
```

Select morphology + pathogen-channel features morphology: cell_* columns without 'ch' (no per-channel intensity) pathogen:   cell_* columns that mention ch{pathogen_chan}

### line 4912  _(unsure)_

```python
clean_feature_cols = []
```

Drop degenerate features

### lines 4930-4933  _(unsure)_

```python
log_intensity = bool(settings.get("infection_pca_log_intensity", True))
```

Prepare feature matrix + ground-truth subsets

Optional log1p transform on intensity-like features to sharpen structure

### lines 4940-4942

```python
vals = cell_for_X[c].to_numpy(dtype=float, copy=True)
```

pandas 3 may expose an already-float column through a read-only view.  The log transform is deliberately local, so own the buffer before changing its finite entries.

### lines 4952-4962

```python
finite_counts = np.isfinite(X).sum(axis=1)
```

Remove rows with all NaNs.

Three skips stood here -- "no rows with finite features", "an all non-finite column is imputed as 0.0", and "fewer than 10 cells after filtering" -- and no input could reach any of them. `tmp.replace` above turned every infinity into NaN before the groupby, so in this table notna and isfinite are the same test; the degenerate-feature filter kept only columns with at least ten notna values, hence at least ten finite ones; and a row holding one of those has a positive finite count, so `mask_rows` keeps it. Every surviving column therefore still has ten finite values and the matrix still has ten rows.

### line 4970  _(unsure)_

```python
for j in range(X.shape[1]):
```

Median imputation per feature

### line 4978  _(unsure)_

```python
max_cells = int(settings.get("infection_pca_max_cells", 50000))
```

Optional subsampling for speed

### lines 5003-5005

```python
inf_vals = intens[y_int]
```

`inf_vals.size` is `np.sum(y_int)` and `uninf_vals.size` is

`np.sum(~y_int)`, both of which the guard above has already refused below 10, so no second per-class size check is possible here.

### line 5012  _(unsure)_

```python
intens_full = cell_level[intensity_col].to_numpy(dtype=float)
```

Boolean masks in the full (post-subsample) cell_level

### lines 5032-5034  _(unsure)_

```python
scaler = StandardScaler()
```

Embedding (PCA / UMAP / t-SNE) with optional hyperparameter search

### line 5038  _(unsure)_

```python
path_weight = float(settings.get("infection_pca_pathogen_weight", 1.0))
```

Optional: up-weight pathogen-channel features to emphasize infection signal

### line 5073  _(unsure)_

```python
if embed_method in {"umap", "tsne"}:
```

PCA (no hyperparameter search, but benefits from log/weighting above)

### line 5094  _(unsure)_

```python
infected_cluster = int(eval_stats["infected_cluster"])
```

Unpack evaluation stats

### line 5132  _(unsure)_

```python
adjusted = cluster_infected.astype(bool)
```

labels follow cluster

### line 5141, trailing  _(unsure)_

```python
else:
```

mode == "remove"

### lines 5161-5164  _(unsure)_

```python
if all_df is source_all_df:
```

Map adjusted infection back to all_df (frame level)

Ensure key dtypes match

### line 5185  _(unsure)_

```python
mask_missing = all_df["adjusted_infected"].isna()
```

Any rows that did not get an adjusted label inherit the original

### line 5202, trailing  _(unsure)_

```python
"method_label": method_label,
```

<- used for axis titles / panel labels

### line 5227  _(unsure)_

```python
mask_uninf_cluster_plot = cluster_labels == uninfected_cluster
```

Masks for remaining cells

### line 5231  _(unsure)_

```python
ax.scatter(
```

Plot clusters with transparency and filled markers

### line 5255  _(unsure)_

```python
ax.set_xlabel(f"{method_label} 1")
```

Axis titles and main title reflect method

## _infection_qc_pca_clustering._evaluate_embedding

### line 4543  _(unsure)_

```python
centroids = []
```

Centroid distance in embedding space

### line 4553  _(unsure)_

```python
sil = None
```

Silhouette in embedding space

### line 4571  _(unsure)_

```python
score = centroid_distance * gt_sep_score
```

Objective: distance * GT separation

## _infection_qc_pca_clustering._search_umap

### line 4593  _(unsure)_

```python
if not do_search:
```

No search: single run with configured/default params

### line 4611  _(unsure)_

```python
nn_grid = settings_local.get(
```

With search: small grid over n_neighbors and min_dist

## _infection_qc_pca_clustering._search_tsne._run_tsne

### line 4666  _(unsure)_

```python
def _run_tsne(perplexity, learning_rate):
```

Utility: run one t-SNE

## _infection_qc_pca_clustering._search_tsne

### line 4686  _(unsure)_

```python
if not do_search:
```

No search: single run with configured/default params

### line 4695  _(unsure)_

```python
perp_grid = settings_local.get(
```

With search: grid over perplexity and learning_rate

## _apply_infection_intensity_qc

### line 5337  _(unsure)_

```python
settings["infection_hist_data"] = None
```

Reset QC payloads by default; strategy helpers will overwrite if used

### line 5350  _(unsure)_

```python
os.makedirs(motility_dir, exist_ok=True)
```

Make sure output directory exists for plots

### line 5355  _(unsure)_

```python
if strategy in {"hist", "histogram", "histagram"}:
```

Strategy → QC helper

### line 5369  _(unsure)_

```python
scope = str(settings.get("infection_intensity_qc_scope", "combined") or "combined").lower()
```

Scope: combined (default), per-plate, per-well, or none

### line 5373  _(unsure)_

```python
return all_df, infection_col
```

Explicit request to skip QC

### line 5378, trailing  _(unsure)_

```python
local_settings = dict(settings)
```

shallow copy for QC helper

### line 5387  _(unsure)_

```python
settings["infection_hist_data"] = local_settings.get("infection_hist_data")
```

propagate QC payloads back

### line 5398  _(unsure)_

```python
if "adjusted_infected" in df_qc.columns:
```

normalise adjusted_infected if present

### line 5454  _(unsure)_

```python
if not set(group_cols).issubset(all_df.columns):
```

If requested group columns are missing, fall back to combined

### line 5521  _(unsure)_

```python
return all_df, infection_col
```

nothing processed → return original

## _compute_velocities_and_well_summary

### lines 5667-5668  _(unsure)_

```python
straightness_threshold = float(
```

Straightness-based artifact detection / filtering

Every track record above writes straightness before this frame is built.

### line 5760  _(unsure)_

```python
well_summary_df = pd.DataFrame(well_records)
```

A non-empty track_df groups into at least one well record.

## _validate_db_table_name

### lines 5795-5796

```python
if str(db_table_name).strip().lower() in RESERVED_DB_TABLE_NAMES:
```

SQLite table names are case-insensitive, so 'Cell' would replace 'cell' just as thoroughly.

## _feature_velocity_correlations

### lines 5893-5896

```python
track_features = track_df.merge(agg_features, on=group_cols, how="left",
```

many_to_one: agg_features is a groupby(group_cols) median, one row per track. track_df is not asserted unique because a caller may hand in a per-segment track table; what matters is that the feature side cannot duplicate a track and weight it twice in the correlations below.

## _make_motility_plots

### line 6138  _(unsure)_

```python
with figure_style(theme_target()):
```

Combined plot over all wells

### lines 6177-6181

```python
facecolor="none",
```

NO PANEL BEHIND THE NOTE: a rounded white box with a black edge is furniture the style has no other boxes to match, and on the dark theme it is a white rectangle laid over the tracks. The patch itself stays as an invisible spacer -- the four text lines below are positioned against its corner.

### line 6235  _(unsure)_

```python
well_summary_map = {}
```

Per-well plots

### line 6331  _(unsure)_

```python
if has_infected:
```

infected-only, re-centred to (0,0)

### line 6357  _(unsure)_

```python
if has_uninfected:
```

uninfected-only, re-centred to (0,0)

## _select_infection_feature_columns

### lines 6394-6395  _(unsure)_

```python
numeric_cols = schema.model_feature_columns(
```

This path accepts user-created tracking measurements, but still applies the shared provenance schema before its infection-specific filters.

### line 6409  _(unsure)_

```python
exclude |= {c for c in numeric_cols if c.endswith("_idx")}
```

drop any debug / temporary numeric cols if present

### line 6412  _(unsure)_

```python
exclude |= {c for c in numeric_cols if "centroid" in c.lower()}
```

drop centroid features (absolute coordinates)

### line 6415  _(unsure)_

```python
if pathogen_chan is not None:
```

exclude intensity columns for non-pathogen channels

### line 6426

```python
pass
```

if parsing fails, keep column

### lines 6434-6437

```python
agg_cols = list(feature_cols)
```

`numeric_cols` is a selection *from* `all_df.columns`, so every feature that survives the exclusions above is still a column of `all_df`: a membership filter here could not drop one, and could not empty a list the guard above has just refused when empty.

### line 6440  _(unsure)_

```python
cell_level = (
```

Build per-cell table to filter out useless columns

## _make_adjusted_qc_panel

### line 6573  _(unsure)_

```python
fig, ax_pca, ax_xgb, ax_hist = create_results_figure()
```

Create figure with desired layout

### lines 6603-6605

```python
ax_hist.axvline(
```

Thin, dashed and grey. It was 2 pt solid black -- heavier than either distribution it separates, and a reference is not a result.

## _infection_qc_histogram

### line 6790  _(unsure)_

```python
cand_cols = [
```

Prefer 95th percentile of pathogen channel; fall back to mean if needed

### line 6801  _(unsure)_

```python
settings["infection_hist_data"] = None
```

Initialize payload slot

### lines 6813-6815

```python
cols_to_drop = [
```

No second "column not found" skip: the loop above only assigns `intensity_col` from a candidate it has just found in `all_df.columns`, and the guard above has already returned for the one other outcome.

### lines 6817-6819

```python
cols_to_drop = [
```

IMPORTANT: drop any existing adjusted_infected when reusing DB This prevents merge from creating adjusted_infected_x / adjusted_infected_y and guarantees we recompute labels fresh each run.

### line 6851  _(unsure)_

```python
do_log = bool(settings.get("infection_intensity_log", False))
```

Optional log-transform to help separate populations

### line 6870  _(unsure)_

```python
hist_pct = float(settings.get("infection_hist_percentile", 25.0))
```

Fallback percentile (now default 25th)

### line 6874  _(unsure)_

```python
thresh_idx = None
```

First bin (low→high) where infected ≥ target_frac

### line 6882  _(unsure)_

```python
thr_val = float(np.nanpercentile(intensities, hist_pct))
```

fallback: hist_pct percentile of all cells (after optional log)

### lines 6896-6900

```python
cell_level["intensity_positive"] = intensities >= thr_val
```

Threshold in the space thr_val was derived in. `intensities` is the (optionally log10-transformed) array the histogram and thr_val came from; comparing the raw column against a log-space threshold used to call every cell positive (and, in mode='remove', silently delete the negatives). The ndarray is row-aligned with cell_level and assigns positionally.

### lines 6944-6947

```python
all_df = all_df.merge(
```

merge back. many_to_one, the same contract the PCA and XGBoost twins spell as validate="m:1": all_df is per frame, cell_level is one row per cell, and a second row for a cell would clone that cell's whole time series into the frame table.

### lines 6962-6963  _(unsure)_

```python
adjusted = all_df["adjusted_infected"].astype("boolean")
```

Now adjusted_infected definitely exists and comes from histogram QC; fill any NaNs (cells not in cell_level) from the original infection_col.

### line 6969  _(unsure)_

```python
make_graphs = bool(settings.get("infection_intensity_qc_graphs", True))
```

Decide whether to make / save QC graph

### line 6974

```python
vals_inf = intensities[mask_labels]
```

Prepare payload (even if we don't generate PNG)

### line 6989  _(unsure)_

```python
os.makedirs(motility_dir, exist_ok=True)
```

Plot histogram

### lines 7029-7035

```python
hist_path = save_figure_to_path(fig_h, hist_path, fmt="png")
```

fmt="png" on purpose, and it is the one exception in this module. This histogram is not only a figure the user keeps: it is read back by `mpimg.imread(qc_panel_path)` and drawn into the mask panel's QC axis. matplotlib cannot imread a PDF, so under the default "PDF" figure preference the read raised, the panel swallowed it, and the QC axis silently went blank. `save_figure`'s `fmt=` exists for exactly this: a raster something else consumes.

### line 7046  _(unsure)_

```python
settings["infection_intensity_qc_panel_type"] = "histogram"
```

Let the panel know what QC plot to embed (if present)

## _infection_qc_xgboost

### line 7114  _(unsure)_

```python
settings["infection_hist_data"] = None
```

init payload slots

### line 7121

```python
cols_to_drop = [
```

IMPORTANT: drop any existing adjusted_* / infection_prob* from DB reuse

### lines 7133-7134  _(unsure)_

```python
all_df = all_df.copy(deep=False)
```

Assignments below repair labels and key dtypes on this run's working frame; the database-shaped frame supplied by the caller is immutable.

### line 7149

```python
if "n_pathogens" in all_df.columns:
```

Ensure n_pathogens exists and has 0 instead of NaN

### line 7153  _(unsure)_

```python
if orig_infection_col not in all_df.columns:
```

Recover infection column if missing

### lines 7188-7190  _(unsure)_

```python
tracked_object = str(settings.get("tracked_object", "cell")).strip().lower()
```

Decide which object type's features to use (tracked_object)

### lines 7200-7204

```python
pattern_obj = re.compile(rf"^{re.escape(obj_prefix)}")
```

Name the real XGBoost candidates before aggregation.  Numeric text was previously discarded by ``median(numeric_only=True)`` and could never reach the schema validator below.  Restricting coercion to the chosen object, pathogen channel and non-centroid inputs also means an unrelated metadata column cannot abort a model that would never consume it.

### lines 7223-7225  _(unsure)_

```python
aggregation_df = all_df.copy(deep=False)
```

Aggregate to per-object level (median across frames)

### lines 7244-7247

```python
cell_level = cell_level.merge(
```

one_to_one: both sides are groupby(key_cols) reductions of all_df, so the per-cell training table gains a column and keeps its row count -- the feature matrix and label vector fed to XGBoost below are built from it by position.

### lines 7253-7259

```python
cell_level[orig_infection_col] = (
```

A recovery block stood here -- a hunt through `<col>_y` and `<col>_x` followed by a KeyError -- and no input could reach it. `agg_cols` excludes `orig_infection_col`, so the left side of the merge cannot carry it and the suffix cannot fire; `infection_any` is `groupby(key_cols)[orig_infection_col].max()`, so the right side always does; and `suffixes=("", "_y")` can never mint an `_x` at all. It was removed rather than excluded from coverage.

### lines 7269-7271  _(unsure)_

```python
intensity_candidates = [
```

Decide pathogen-channel intensity column for this tracked_object

### lines 7297-7300  _(unsure)_

```python
numeric_cols = schema.model_feature_columns(cell_level)
```

Build feature set: {tracked_object}_* only, excluding centroids, non-pathogen channels, degenerate features

### lines 7305-7310

```python
for c in numeric_cols:
```

`numeric_cols` cannot contain `orig_infection_col`, 'frame' or 'timeID': `agg_cols` above already excludes all three from the per-cell table, and `schema.model_feature_columns` would drop them anyway -- 'timeID' is provenance, 'frame' is not a declared measurement, and the infection call is cast to bool one block up, which that selector omits. So no skip for them is reachable here.

### line 7329  _(unsure)_

```python
clean_feature_cols = []
```

Drop degenerate features

### lines 7353-7355  _(unsure)_

```python
infected_cells = cell_level[cell_level[orig_infection_col]]
```

Global sanity check: do we even have enough infected/uninfected cells?

### lines 7372-7374  _(unsure)_

```python
inf_int = infected_cells[intensity_col].to_numpy(dtype=float)
```

Define confident training sets using intensity quartiles

### lines 7397-7401

```python
hi_inf = infected_cells[infected_cells[intensity_col] >= high_thr_inf].copy()
```

Neither quartile subset can come out empty: `inf_int` and `uninf_int` are the finite values of the same columns and the guard above has refused both when empty, so a percentile of them lies between their own min and max. The row holding the maximum therefore satisfies `>= high_thr_inf` and the row holding the minimum satisfies `<= low_thr_uninf`.

### lines 7412-7414  _(unsure)_

```python
hi_inf["xgb_label"] = 1
```

Curate XGBoost training data per well

### line 7440  _(unsure)_

```python
wells_single_class.append((plate_id, well_id))
```

wells with only one class → skip for training

### lines 7444-7445  _(unsure)_

```python
pos_idx = pos_df.index.to_numpy(copy=True)
```

``Generator.choice`` may shuffle its input while sampling.  pandas 3 exposes Index storage as read-only, so pass NumPy-owned buffers.

### line 7450  _(unsure)_

```python
n_per_class = min(n_pos, n_neg)
```

wells with enough data per class → balanced sampling within well

### line 7461  _(unsure)_

```python
pos_sel = pos_idx
```

small wells with both classes → keep all extreme examples

### lines 7465-7467

```python
train_idx_list.extend(pos_sel.tolist())
```

No second single-class skip: `n_pos` and `n_neg` are both non-zero by the guard above, so `pos_idx`/`neg_idx` are non-empty and `rng.choice` is asked for `min(n_pos, n_neg) >= 1` of them.

### lines 7503-7509

```python
X_all = cell_level[feature_cols].to_numpy(dtype=float, copy=True)
```

Build feature matrix + median imputation

Pandas 3 may expose an Arrow-backed, read-only ndarray here.  Median imputation below is deliberately in-place, so own the working buffer instead of depending on a writable view from a particular dataframe backend.

### lines 7538-7540

```python
removed = [f for f, k in zip(feature_cols, keep) if not k]
```

`keep[0]` is never cleared -- the inner loop only writes `keep[j]` for `j > i >= 0` -- so the filter always keeps at least the first feature and there is no "everything was correlated away" case.

### lines 7553-7554  _(unsure)_

```python
print(
```

The feature matrix always has a column here: `feature_cols` is non-empty by the guard above, and the correlation filter keeps `keep[0]`.

### lines 7650-7652  _(unsure)_

```python
drop_amb = bool(settings.get("infection_xgb_drop_ambiguous", True))
```

Drop ambiguous band (probability in [low, high])

### lines 7679-7681  _(unsure)_

```python
for col in key_cols:
```

Map adjusted calls back to all_df

### line 7685  _(unsure)_

```python
all_df = all_df.merge(
```

(any stale adjusted_infected/infection_prob already dropped above)

### line 7694  _(unsure)_

```python
ids_to_remove = set()
```

Combine removed_sets (disagreement + ambiguous)

### lines 7728-7733

```python
try:
```

Prepare QC payloads for the combined adjusted panel histogram of intensity (adjusted labels) PCA embedding of used features (adjusted labels) XGBoost feature importances (gain)

### lines 7735-7736  _(unsure)_

```python
intens = cell_level[intensity_col].to_numpy(dtype=float)
```

histogram payload intensity_col was selected from cell_level.columns above.

### line 7748  _(unsure)_

```python
thr_val = float(0.5 * (low_thr_uninf + high_thr_inf))
```

Use midpoint between training thresholds as a visual threshold

### lines 7766-7769

```python
X_panel = cell_level[used_feature_cols].to_numpy(
```

feature_cols is non-empty before training, and the correlation filter always retains its first entry. The imputation below is display-only.  Own the matrix because pandas 3 can return a read-only view for homogeneous columns.

### lines 7785-7790

```python
panel_components = min(2, X_scaled_panel.shape[1])
```

XGBoost can legitimately train on one surviving feature.  PCA cannot request two components from that matrix, but the panel contract is always a pair of plotting coordinates.  Fit the one available component and pad only the display coordinate with a zero axis; the fitted classifier and its feature set are unchanged.

### lines 7812-7814  _(unsure)_

```python
try:
```

Feature importance payload (no PNG; panels draw from this)

### line 7820  _(unsure)_

```python
sorted_pairs = sorted(
```

sort and truncate

### line 7846  _(unsure)_

```python
settings["infection_intensity_qc_panel_type"] = "xgboost"
```

Mark QC type for panels; no embedded PNG (mask panel draws nothing)

## automated_motility_assay

### lines 7890-7891

```python
db_table_name = _validate_db_table_name(settings["db_table_name"])
```

Checked here as well as at the write, so a run that would destroy the cell table stops before the hours of regionprops rather than after them.

### lines 7897-7900  _(unsure)_

```python
reuse_existing = settings.get("reuse_existing_measurements", True)
```

Optional reuse of existing measurements from SQLite → this table is treated as the ORIGINAL, pre-QC dataset.

### line 7917  _(unsure)_

```python
all_df = pd.read_sql_query(f"SELECT * FROM {db_table_name}", conn)
```

If the table does not exist, this will raise and fall back to recompute

### lines 7953-7955  _(unsure)_

```python
merged_dir = os.path.join(src, "merged")
```

Read merged files & basic metadata (if not reusing DB)

### line 7969  _(unsure)_

```python
groups = {}
```

group by (plateID, wellID, fieldID)

### line 8006  _(unsure)_

```python
sample_filename = sorted(all_files)[0]
```

Debug-plot one sample merged array with channel/mask labels

### lines 8021-8023  _(unsure)_

```python
if not loaded_from_db:
```

Build measurements if not reusing from DB

### lines 8041-8043

```python
non_empty = [df for df in dfs if not df.empty]
```

Guard on the filtered list, not on `dfs`: every group returning an empty frame (e.g. no cell masks) left pd.concat([]) to raise a cryptic ValueError("No objects to concatenate") before the intended RuntimeError.

### line 8065  _(unsure)_

```python
all_df = _smooth_tracks_and_features(
```

Clean tracks

### line 8083  _(unsure)_

```python
n_frames_db = all_df["frame"].nunique()
```

Already loaded smoothed measurements from DB (treated as ORIGINAL)

### lines 8096-8099  _(unsure)_

```python
if "infected" in all_df.columns:
```

Infection status per track (mask-based)

This is part of the ORIGINAL dataset.

### lines 8101-8102  _(unsure)_

```python
all_df["infected"] = all_df["infected"].fillna(False).astype(bool)
```

Reuse existing infection labels (DB-reused or previous run), but ensure no NaNs and correct dtype.

### lines 8116-8119

```python
all_df = all_df.merge(
```

many_to_one: `infected` is a groupby over the track key, one row per cell; all_df is per frame. The track counts printed straight after this are drop_duplicates() over the same key, so a fan-out here would not show up in them -- only in the frame counts nobody reads.

### lines 8145-8148

```python
all_df_original = all_df.copy(deep=True)
```

SNAPSHOT: ORIGINAL measurements (pre-QC) to be stored in SQLite. This copy is never overridden by adjusted/QC'd data.

### lines 8151-8154  _(unsure)_

```python
infection_col = "infected"
```

Optional infection-intensity QC (may create 'adjusted_infected') This operates on all_df only (not on all_df_original).

### lines 8164-8168

```python
print(
```

A weak XGBoost model can put every track inside the configured ambiguous band. Returning an empty successful assay then suppresses the well summary, correlations and every downstream plot. Preserve the mask labels as the conservative fallback while keeping the adjusted schema explicit for callers.

### lines 8191-8207

```python
xgb_proba_col = settings.get("infection_xgb_proba_column", None)
```

Try to locate a probability column created by the QC step.

THE FALLBACK BELOW WAS UNREACHABLE. This tested `is None`, and the setting's default is the non-empty string 'infection_xgb_proba' so auto-discovery was always skipped, and the classifier writes 'infection_prob', so the guard two blocks down never matched either. The whole track-level ambiguous-track filter silently did not run on any default configuration: not an error, not a warning, just an analysis step that never happened.

Discovery now runs when the column was NOT CHOSEN -- unset, or left at the shipped default -- and the name is not in the frame.

Not simply "the column is absent": a user who NAMES a column that does not exist must be told so, not silently given a different one. That distinction is the whole reason the default cannot be treated as a choice; it is what the caller gets for expressing no opinion.

### lines 8252-8256

```python
all_df = all_df.merge(
```

many_to_one: `ambiguous` is a filtered groupby(track_keys) mean, one row per track, used here only as a flag to drop rows by. A duplicate would multiply the frames of a track that is about to be dropped anyway -- and would make the "dropped N rows" message below a lie in the other direction.

### lines 8280-8283

```python
try:
```

Save ADJUSTED frame-level measurements to CSV ONLY. This NEVER overwrites the SQLite original table.

### lines 8303-8305  _(unsure)_

```python
(
```

Compute per-track velocities + per-well summary

### lines 8332-8337

```python
measurements_dir, db_path = _save_measurements_and_well_summary(
```

Save to DB:

all_df_original (pre-QC snapshot) is written to db_table_name well_summary_df is written as usual by _save_measurements_and_well_summary → adjusted labels NEVER touch the canonical measurements table.

### line 8345  _(unsure)_

```python
_feature_velocity_correlations(all_df, track_df, measurements_dir)
```

Feature–velocity correlation analysis (final labels, adjusted view)

### line 8353  _(unsure)_

```python
if settings.get("make_mask_panel", True):
```

Intensity + motility panel for mask-based labels

### line 8366  _(unsure)_

```python
label_tag=f"mask_{qc_strategy}",
```

encode both label type and QC strategy in the tag

### line 8370  _(unsure)_

```python
if (
```

Intensity + motility panel for adjusted labels (if distinct)

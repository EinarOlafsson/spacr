# Notes from `spacr/measure.py`

Prose lifted out of `spacr/measure.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (9 entries)
- [_pool_context](#_pool_context) (1 entry)
- [_start_manager](#_start_manager) (1 entry)
- [resolve_pool_size](#resolve_pool_size) (1 entry)
- [resolve_n_jobs](#resolve_n_jobs) (1 entry)
- [resolve_measurement_spacing](#resolve_measurement_spacing) (2 entries)
- [get_components](#get_components) (6 entries)
- [_calculate_zernike](#_calculate_zernike) (3 entries)
- [_analyze_cytoskeleton](#_analyze_cytoskeleton) (13 entries)
- [_safe_morphology_table](#_safe_morphology_table) (1 entry)
- [_join_child_to_parent_cell](#_join_child_to_parent_cell) (1 entry)
- [_spatial_adjacency](#_spatial_adjacency) (2 entries)
- [_spatial_measurements](#_spatial_measurements) (3 entries)
- [_morphology_of_organelle_type](#_morphology_of_organelle_type) (1 entry)
- [_morphological_measurements](#_morphological_measurements) (8 entries)
- [_morphological_measurements._with_distances](#_morphological_measurements_with_distances) (1 entry)
- [_morphological_measurements._with_spatial](#_morphological_measurements_with_spatial) (2 entries)
- [_morphological_measurements._with_bystanders](#_morphological_measurements_with_bystanders) (5 entries)
- [_summarize_organelles_per_parent](#_summarize_organelles_per_parent) (6 entries)
- [_intensity_measurements](#_intensity_measurements) (6 entries)
- [_create_dataframe](#_create_dataframe) (1 entry)
- [_extended_regionprops_table._gini](#_extended_regionprops_table_gini) (2 entries)
- [_extended_regionprops_table](#_extended_regionprops_table) (4 entries)
- [_extended_regionprops_table._masked_intensity](#_extended_regionprops_table_masked_intensity) (1 entry)
- [_calculate_homogeneity](#_calculate_homogeneity) (3 entries)
- [_periphery_intensity](#_periphery_intensity) (2 entries)
- [_outside_intensity](#_outside_intensity) (2 entries)
- [_calculate_radial_distribution._calculate_average_intensity](#_calculate_radial_distribution_calculate_average_intensity) (2 entries)
- [_calculate_radial_distribution](#_calculate_radial_distribution) (2 entries)
- [_calculate_correlation_object_level](#_calculate_correlation_object_level) (3 entries)
- [_estimate_blur](#_estimate_blur) (5 entries)
- [_measure_intensity_distance](#_measure_intensity_distance) (2 entries)
- [save_and_add_image_to_grid](#save_and_add_image_to_grid) (3 entries)
- [img_list_to_grid](#img_list_to_grid) (5 entries)
- [_per_crop_mode](#_per_crop_mode) (1 entry)
- [_measure_crop_core](#_measure_crop_core) (43 entries)
- [_record_organelle_caveats](#_record_organelle_caveats) (1 entry)
- [measure_crop](#measure_crop) (23 entries)
- [measure_crop.job_callback](#measure_cropjob_callback) (2 entries)
- [process_measure_crop_results](#process_measure_crop_results) (3 entries)
- [generate_cellpose_train_set](#generate_cellpose_train_set) (4 entries)
- [get_object_counts](#get_object_counts) (2 entries)
- [_crop_full_scale](#_crop_full_scale) (1 entry)
- [generate_object_dataset](#generate_object_dataset) (2 entries)
- [crop_objects_from_array](#crop_objects_from_array) (3 entries)

## Module level

### lines 76-80

```python
from .crops import (
```

The crop PNG format lives in spacr.crops: the writer's channel order (to_cv2_bgr), the folder marker (stamp_crop_folder) and the reader that undoes the legacy order all have to agree, so they live in one place. spacr.crops imports nothing from spacr and nothing heavy, so this costs the measure path nothing.

### line 91  _(unsure)_

```python
from . import settings as _settings_module
```

Backward-compatible public module binding used by downstream extensions.

### lines 94-95  _(unsure)_

```python
from . import measurement_schema as _measurement_schema
```

Public compatibility alias: measurement writers and downstream feature dictionaries historically import this constant from ``spacr.measure``.

### lines 98-100

```python
from .errors import RunLedger, ConfigurationError, raise_if_strict
```

Fail-loud accounting: a field that fails to measure is recorded, summarised at the end of the run, and stamped into measurements.db so a downstream regression cannot silently analyse 344 of 384 wells.

### lines 102-103  _(unsure)_

```python
from .runctx import run_context
```

One run id on every log line and every artifact, one seed reaching every RNG, one on_error policy at each batch boundary. See spacr.runctx.

### lines 106-112

```python
from .measure_hooks import (
```

Opt-in extension points, so illumination correction and a user-drawn ROI can change what is measured without editing this module. Both registries are empty by default and both entry points return their input object unchanged when they are, so an ordinary run is byte-identical to one from before they existed. spacr.measure_hooks imports numpy, the stdlib and spacr.errors only registering a hook must not drag in matplotlib/skimage/cv2. Re-exported here so `from spacr.measure import register_preprocessing_hook` also works.

### lines 136-138

```python
from .figures.style import figure_style, theme_target
```

THE HOUSE STYLE (136). `figures.style` imports matplotlib only inside its own functions, so naming it here costs nothing at import time.

### lines 142-164

```python
MORPHOLOGICAL_PROPS = [
```

3-D support: dimensionality, voxel spacing, and the units stamp

spaCR's mask generation can now emit (Z, Y, X) label volumes (see spacr.zstack, MODE_STITCH / MODE_VOLUMETRIC). Everything below exists so that such a volume is measured correctly *or refused*, and never measured wrongly.

Two invariants govern every change in this module:

1. **The 2-D path is bit-identical.** A 2-D field takes exactly the code it took before: ``spacing`` is None (skimage treats ``spacing=None`` as "omitted"), no property is dropped, no distance transform is sampled, and no column is renamed. Physical scaling is deliberately NOT applied in 2-D even when a voxel size is available, because that would turn every existing ``*_area`` column from px^2 into um^2 under an unchanged name and silently break every threshold ever written against a spaCR database.

2. **A row's units are never guessed at.** A 3-D run measures volumes where a 2-D run measures areas, so every row written to measurements.db carries :data:`MEASUREMENT_STAMP_COLUMNS`, and ``spacr.utils._merge_and_save_to_database`` refuses to append rows whose units differ from the ones already in the table.

### lines 921-950

```python
_SPATIAL_NO_NEIGHBOUR = -1.0
```

Spatial context: where an object sits relative to its own kind

Opt-in, default OFF (``spatial_measurements``), so an ordinary measure run does exactly zero extra work and cannot get slower.

The names below were chosen against the pipeline's silent column-deleters, not for prose. Four of them apply to every new measurement column:

``utils._check_integrity`` folds any column whose name contains the substring ``label`` into ``label_list`` and drops it with no warning -- and if it sorts before the real label column it *becomes* ``object_label`` and corrupts the merge key. No name here contains ``label``. ``utils.filter_dataframe_features`` does ``dropna(axis=1)``, so ONE NaN anywhere deletes the column from every model matrix. This emitter therefore never writes NaN -- see ``_SPATIAL_NO_NEIGHBOUR`` below. ``filter_dataframe_features`` / ``schema.model_feature_columns`` also strip names containing ``count`` and ``_id``. ``neighbor_count`` is BANNED for exactly that reason; ``touching_neighbors`` is the storable spelling. a column with no ``feature_dict`` entry reports as family "unknown" with a null description.

Neighbour *identities* are deliberately out of scope. Storing the label ids of adjacent objects was asked for, but ``validate_object_table_frame`` refuses any non-numeric column in the object namespace (a text column and a list column both raise ObjectTableSchemaError), and any name carrying ``label`` is either swallowed or becomes ``object_label``. ``touching_neighbors`` -- the number of distinct adjacent objects -- is the answer to the same question that the database can actually hold.

## _pool_context

### lines 238-239

```python
print(f"WARNING: {START_METHOD_ENV_VAR}={method!r} is not a "
```

An unusable name (``fork`` on Windows, a typo) must not take the run down: the platform default measures the same rows.

## _start_manager

### lines 360-362

```python
try:
```

get_start_method() is read here rather than passed in so the message reports what the *failed* Manager was actually using, even if the caller resolved a different name earlier.

## resolve_pool_size

### lines 389-390

```python
return max(1, min(n_jobs, int(n_files)))
```

max(1, ...) because Pool(0) raises ValueError, and a folder that turned out to hold no unmeasured fields must finish quietly rather than crash.

## resolve_n_jobs

### lines 412-413

```python
return max(1, cores - N_JOBS_HEADROOM)
```

Blank: leave headroom, but never drop below one worker -- a machine with four cores or fewer would otherwise get zero or a negative pool.

## resolve_measurement_spacing

### lines 486-488

```python
from .zstack import UnknownAnisotropyError
```

Imported here rather than at module scope: spacr.zstack is the authority on anisotropy and owns the error type, but measure.py must not pay for importing it on the overwhelmingly common 2-D path.

### lines 525-526  _(unsure)_

```python
stamp['measurement_units'] = UNITS_PX_XY
```

Geometry is correct, units are xy pixels. Recorded as such: a "volume" here is in cubic xy-pixels and is not a um^3.

## get_components

### line 598  _(unsure)_

```python
cell_to_nucleus = defaultdict(list)
```

Create mappings from each cell to its nucleus, pathogens, and cytoplasms

### line 603  _(unsure)_

```python
for cell_id in cell_labels:
```

Iterate over each cell label

### line 607  _(unsure)_

```python
nucleus_ids = np.unique(nucleus_mask[cell_mask == cell_id])
```

Find corresponding component labels

### line 610  _(unsure)_

```python
cell_to_nucleus[cell_id] = nucleus_ids[nucleus_ids != 0].tolist()
```

Update dictionaries, ignoring 0 (background) labels

### line 613  _(unsure)_

```python
nucleus_df = pd.DataFrame(list(cell_to_nucleus.items()), columns=['cell_id', 'nucleus'])
```

Convert dictionaries to dataframes

### lines 616-619

```python
nucleus_df = nucleus_df.explode('nucleus').dropna(
```

Explode lists

``explode`` turns an empty child list into a row whose child key is NaN. Those rows describe no relationship and duplicate the NaN merge key once per parent, which violates the one-child/one-parent contract downstream.

## _calculate_zernike

### lines 657-663

```python
coords = np.argwhere(region.image)
```

mahotas' signature is zernike_moments(im, radius, degree=8): the moments are computed on a disk of `radius` centred on the object's centre of mass, and pixels outside that disk are ignored. Passing `degree` positionally put the degree into `radius`, so every object 20 px or 2000 px -- was described on a fixed 8 px disk and the degree was always the default 8. Scaling the radius with the object is what makes the coefficients comparable across object sizes.

### lines 669-670  _(unsure)_

```python
radius = float(np.sqrt(((coords - centre) ** 2).sum(axis=1)).max())
```

Max distance from the centre of mass, which is exactly the centre mahotas uses by default, so the disk covers the whole object.

### lines 676-677

```python
feature_length = len(zernike_features[0])
```

``regions`` was checked non-empty above and one vector is appended for every region, so this list cannot be empty here.

## _analyze_cytoskeleton

### lines 747-749

```python
image = array[..., channel]
```

``[..., channel]`` rather than ``[:, :, channel]``: identical for the (Y, X, C) arrays this has always been given, and it selects the channel rather than a slab of X should a (Z, Y, X, C) array ever reach here.

### line 754  _(unsure)_

```python
for label in np.unique(mask):
```

Process each object in the mask based on its label

### line 757, trailing  _(unsure)_

```python
continue
```

Skip background

### line 759  _(unsure)_

```python
object_region = mask == label
```

Isolate the object using the label

### line 761, trailing  _(unsure)_

```python
region_intensity = np.where(object_region, image, 0)
```

Use np.where for more efficient masking

### line 763  _(unsure)_

```python
if np.any(region_intensity):
```

Ensure there are non-zero values to process

### line 765  _(unsure)_

```python
valid_pixels = region_intensity[region_intensity > 0]
```

Calculate adaptive offset based on intensity percentiles within the object

### line 767, trailing  _(unsure)_

```python
if len(valid_pixels) > 1:
```

Ensure there are enough pixels to compute percentiles

### line 769, trailing  _(unsure)_

```python
block_size = 35
```

Adjust this based on your object sizes and detail needs

### line 773  _(unsure)_

```python
skeleton = morphology.skeletonize(img_as_bool(cytoskeleton))
```

Skeletonize the thresholded cytoskeleton

### lines 779-783

```python
skel = skeleton.astype(np.uint8)
```

Branch points are skeleton pixels with >= 3 skeleton neighbours (the standard definition). The previous code called morphology.skeleton_branch_analysis, which does not exist in scikit-image and raised AttributeError whenever a region had enough pixels to reach this branch.

### line 790  _(unsure)_

```python
properties = {
```

Store properties

### line 798  _(unsure)_

```python
properties_list.append({
```

Handle cases with insufficient pixels

## _safe_morphology_table

### lines 842-843  _(unsure)_

```python
return frame[[prop for prop in requested if prop in frame.columns]]
```

Added guarded columns belong where callers requested them, not at the end.  (All morphology properties here are scalar columns.)

## _join_child_to_parent_cell

### lines 901-903

```python
raise
```

The duplicate is on the props side, which means regionprops_table emitted a label twice -- a different fault with a different fix, and one this message would misdescribe. Say nothing about cells.

## _spatial_adjacency

### lines 1021-1025

```python
if spacing is None:
```

scikit-image 0.22 is still the supported floor and predates the ``spacing`` keyword. Its implementation is a nearest-label EDT, so reproduce that small operation with scipy's long-standing ``sampling`` argument instead of silently growing anisotropic volumes in voxel units (or rejecting a documented feature).

### lines 1045-1046

```python
edge = [slice(None)] * ndim
```

np.roll wraps, which would make the first row a neighbour of the last. Blank the plane the wrap landed on.

## _spatial_measurements

### line 1138  _(unsure)_

```python
counts = np.asarray(
```

-1: query_ball_point counts the object itself.

### lines 1147-1148  _(unsure)_

```python
distances = distances.reshape(n, 1)
```

k=1 returns a 1-D array of self-distances; reshape so the column indexing below is uniform.

### lines 1170-1171  _(unsure)_

```python
frame[[near_col, second_col, pct_col]] = frame[
```

Belt and braces: a NaN reaching filter_dataframe_features deletes the whole column from every model matrix, so assert the contract here.

## _morphology_of_organelle_type

### lines 1204-1205

```python
return preset.morphology_for(settings.get('organelle_diameter'))
```

Size is half the mapping for 'vesicular' and 'spherical'; neither is a network at any size, but it is passed through rather than assumed.

## _morphological_measurements

### lines 1268-1269  _(unsure)_

```python
spatial_on = bool(settings.get('spatial_measurements', False))
```

Opt-in spatial context; default OFF, so the default measure run does exactly zero extra work here. See _spatial_measurements.

### lines 1276-1277

```python
distances_on = bool(settings.get('object_distances', False))
```

EVERY DISTANCE WORTH MEASURING, opt-in and off by default for the same reason `spatial_measurements` is: it is real time on a 3-D field.

### line 1280

```python
bystanders_on = bool(settings.get('bystander_measurements', False))
```

WHICH UNINFECTED CELLS ARE NEXT TO AN INFECTED ONE (instruction 388).

### lines 1285-1288

```python
bystander_reach = 0.0
```

A REACH THAT CANNOT BE READ MAKES EVERY UNINFECTED CELL DISTAL rather than guessing a distance: `classify` treats a non-positive reach that way already, and inventing bystanders from a mis-typed setting is the one failure that would look like a finding.

### lines 1465-1473

```python
nucleus_props = _join_child_to_parent_cell(
```

one_to_one; see _join_child_to_parent_cell for why, and for what was tried instead. Briefly: this was relaxed to one_to_many on the theory that a nucleus straddling two touching cells is a legitimate shape. It is not one measurements.db can store -- the nucleus table is keyed one row per object_label per field -- and relaxing it here only moved the same MergeError downstream into _merge_and_save_to_database, after the cell table for this field had already been committed. Backed out: fail before the first write, with a message that names the offending labels.

### lines 1497-1504

```python
pathogen_props = _join_child_to_parent_cell(
```

one_to_one, for the same reasons as the nucleus join above. This is the join the fan-out actually reaches, because the mask repair that prevents it is optional here: with merge_edge_pathogen_cells=False, a vacuole on the border between two host cells is listed under both cell_ids. That still cannot be stored -- the pathogen table is one row per object_label with one cell_id -- so it stops here, before the cell and nucleus tables for this field are written, rather than after.

### lines 1523-1524

```python
if spatial_on:
```

Type can warn that a family may be hard to interpret, but it never removes requested measurements or their output columns.

### lines 1549-1553

```python
cytoplasm_props = _props(cytoplasm_mask)
```

NEVER _with_spatial. The cytoplasm mask is built at measure.py:2662 as `np.where(interior, 0, cell_mask)` -- it carries the *cell's own* label, so it is one object per cell by construction and its "neighbours" would be the cell's neighbours restated under a second name, in a second table, as if they were independent measurements.

## _morphological_measurements._with_distances

### lines 1321-1322

```python
print(f"[measure] object distances for {name} were not "
```

A MEASUREMENT FAMILY THAT FAILS IS NOT A FAILED RUN. Every other measurement in this frame is still correct.

## _morphological_measurements._with_spatial

### lines 1337-1341

```python
merged = frame.merge(spatial, on='label', how='left',
```

Props on the LEFT, always: 'label' keeps column position 0, so utils._check_integrity still reads the *real* label into object_label. No spatial name contains 'label', so reversing this could not corrupt the merge key either -- but it would move the key column, and a test asserts the order.

### lines 1346-1349

```python
merged[[near_col, second_col, pct_col]] = merged[
```

how='left' could only introduce NaN if the two frames disagreed about the label set, which they cannot (both come from regionprops_table on the same mask). Filled rather than trusted: one NaN anywhere deletes the column from every model matrix.

## _morphological_measurements._with_bystanders

### lines 1379-1384

```python
print("[measure] bystanders were not measured: no median "
```

NO COLUMNS RATHER THAN WRONG ONES. A zero diameter means nothing measurable was found -- every cell clipped by the field edge, or a 3-D mask this planar measure does not cover. Emitting the block anyway would give a reach of zero, which marks every uninfected cell DISTAL: a confident wrong answer instead of a visible gap.

### lines 1388-1393

```python
if pathogen_links is None or not len(pathogen_links):
```

`get_components` returns an EXPLODED FRAME, not a dict:

one row per cell/pathogen pair, with cells holding no pathogen already dropped. So every cell_id present in it is infected, and no emptiness test is needed -- an earlier draft treated it as a mapping and asked `if found`, which is the ambiguous-truth-value error on a DataFrame.

### line 1405

```python
print(f"[measure] bystanders were not measured: "
```

A MEASUREMENT FAMILY THAT FAILS IS NOT A FAILED RUN.

### lines 1412-1416

```python
out = pd.DataFrame({
```

NUMERIC ONLY, because the object namespace refuses a non-numeric column -- the category itself cannot be stored, so it is carried as two flags. `is_infected` is not emitted: it is neither of these two, and a third column would be a third chance to disagree with the pathogen table about the same fact.

### lines 1422-1431

```python
distance = pd.to_numeric(block['distance_to_infected'],
```

-1.0 FOR "NOTHING INFECTED IN THIS FIELD", the same sentinel and for the same reason as `_SPATIAL_NO_NEIGHBOUR`: `classify` reports infinity there, and one non-finite value deletes the column from every model matrix. It is a sentinel, not a distance, and must not be averaged. BUILT RATHER THAN OVERWRITTEN. Under copy-on-write -- the default from pandas 3 -- `to_numpy` hands back a read-only view of the column, and assigning into it raises "assignment destination is read-only". `np.where` produces the sentinel-filled array in one step, so there is no in-place write to forbid.

## _summarize_organelles_per_parent

### lines 1629-1633

```python
organelle_df = pd.merge(
```

one_to_one: both frames are derived from the same organelle_mask and both carry one row per label -- regionprops_table on the left, _map_child_to_parent's single argmax-overlap parent on the right. A duplicate on either side means one of those two invariants broke, and the per-parent sums computed below would then double-count organelles.

### line 1643  _(unsure)_

```python
rows = []
```

No organelles — return empty summary for all parents

### line 1651  _(unsure)_

```python
for ch in range(channel_arrays.shape[-1]):
```

Per-channel intensity per organelle

### lines 1661-1663

```python
organelle_df[f'organelle_channel_{ch}_mean_intensity'] = intensities
```

'channel_<c>', not 'ch<c>'. Every other feature family in the database spells it out; this one did not, so the same idea had two names. utils.rename_columns_in_db migrates the old spelling.

### line 1666  _(unsure)_

```python
parent_props = pd.DataFrame(regionprops_table(parent_mask, properties=['label', 'area'], spacing=...
```

Get parent areas for fraction calculation

### line 1670  _(unsure)_

```python
summary_rows = []
```

Summarise per parent

## _intensity_measurements

### lines 1744-1747

```python
print("3-D mask: skipping GLCM homogeneity — "
```

Refused rather than approximated: graycomatrix takes a 2-D image only, and running it on one plane (or on a reshaped volume, which is what an unguarded call does) would report the texture of an arbitrary slice under a column name that promises the object's.

### lines 1754-1759

```python
col_lables = ['region_label', 'mean', 'percentile_5', 'percentile_10', 'percentile_25', 'percenti...
```

'percentile_<p>', not '<p>_percentile'. The object interior has always been written 'percentile_5' (_extended_regionprops_table); the periphery and outside rings used the reversed word order, so one database carried 'cell_channel_0_percentile_5' next to 'nucleus_channel_0_periphery_5_percentile' for the same statistic. utils.rename_columns_in_db migrates the old spelling on first read.

### lines 1770-1772

```python
channel_percentiles = _field_reference_percentiles(channel)
```

frac_high90 / frac_low10 are cut at the whole field's percentiles, so the pair belongs to the channel, not to the mask being measured. Computed once here instead of once per mask inside the call below.

### lines 1799-1809

```python
label_shape = np.asarray(label).shape
```

Measure focus on the object's 2-D patch, not on the 1-D vector of its pixels. The column is named 'blur' bare: the loop below adds the '<object>_channel_<i>_' prefix to every non-label column, and writing the prefix here too produced 'cell_channel_0_cell_channel_0_blur' in every database written before this fix. _estimate_blur cuts the object's bounding box grown by one pixel out of whatever it is handed, so hand it that patch rather than a whole-field boolean: the pixels it measures are the same ones, and the loop stops comparing the entire field once per object per channel.

### lines 1851-1857

```python
if settings.get('cell_mask_dim') is not None and np.max(cell_mask) != 0:
```

The parent-cell link must exist whether or not radial_dist ran. It used to arrive ONLY as a side effect of _create_dataframe, so with radial_dist=False the nucleus/pathogen/organelle tables lost cell_id entirely and _merge_and_save_to_database silently dropped it from the key columns. Build it from the masks instead, and strip the radial frame's copy so exactly one frame supplies it (two would collide as cell_id_x / cell_id_y in the morphology/intensity merge).

### lines 1871-1874

```python
parent_link['cell_id'] = parent_link['cell_id'].astype(float).replace(0.0, np.nan)
```

_map_child_to_parent uses 0 for "no overlapping cell". There is no cell 0, and the column this replaces was NaN in that case (an object outside every cell simply had no row in the radial frame), so keep NaN and keep the column's float dtype.

## _create_dataframe

### line 1900

```python
df = df.reset_index().rename(columns={'index': 'label'})
```

Reset the index and rename the column that was previously the index

## _extended_regionprops_table._gini

### line 2069  _(unsure)_

```python
array = np.abs(array[~np.isnan(array)])
```

Compute Gini coefficient (nan safe)

### line 2071  _(unsure)_

```python
n = array.size
```

Called only for the non-empty intensity branch below.

## _extended_regionprops_table

### lines 2082-2089

```python
if field_percentiles is None:
```

Reference thresholds for frac_high90 / frac_low10.

These used to be thresholded on the object's OWN 90th/10th percentile, which makes them 0.10 for any continuous distribution by construction — they reported quantisation and ties, not brightness. Thresholding on the whole field's percentiles instead gives what the names promise: the fraction of the object that is bright (or dim) relative to this field. A dim object scores near 0 for frac_high90, a bright one near 1.

### lines 2135-2138

```python
has_variation = not np.all(intens == intens[0])
```

scipy.stats deliberately returns NaN for a constant sample, but first emits a RuntimeWarning about catastrophic cancellation. Uniform segmented objects are ordinary image data, so take the mathematically-defined shortcut and keep measurement logs clean.

### lines 2147-2152

```python
mode_val = np.atleast_1d(np.asarray(mode(intens, nan_policy='omit').mode))
```

Mode (use the smallest mode value if multimodal). SciPy < 1.11 returned a 1-element array here, SciPy >= 1.11 returns a bare scalar. The old code did `mode_val[0]`, which raises IndexError on a scalar, and a bare `except` turned that into NaN — so on the installed SciPy (1.15) mode_intensity was NaN for every object in every database. atleast_1d handles both.

### lines 2178-2185

```python
percentiles = [5, 10, 25, 75, 85, 95]
```

One np.percentile call per region covering all six cut points, rather than one call per (region, cut point) that re-extracted and re-sorted the object's pixels each time. numpy takes a sequence for q and returns the same values.

The vector is the RAW masked intensity, deliberately not the NaN-filtered `intens` the loop above uses: filtering here would change these numbers on any region carrying a NaN.

## _extended_regionprops_table._masked_intensity

### line 2113, trailing

```python
except AttributeError:
```

scikit-image 0.22-0.25

## _calculate_homogeneity

### line 2213  _(unsure)_

```python
for region in regionprops(label):
```

Iterate through the regions in label_mask

### lines 2216-2220

```python
rescaled_image = rescale_intensity(
```

Hoisted out of the distance loop: the rescale depends only on the region, so at the six default distances five of the six calls were recomputing an identical array. Measured 0.039 ms per call against a 1.206 ms per-region budget, and the output is bit-identical -- the same array reaches graycomatrix either way.

### lines 2226-2231

```python
if not np.any(glcm):
```

No pixels in this bounding box are ``d`` columns apart. ``graycoprops`` deliberately turns that empty matrix into 0.0, which looks like a confidently heterogeneous object rather than an unobserved statistic.  Preserve the missing measurement honestly; the model boundary drops an entirely absent feature and median-imputes a partially observed one.

## _periphery_intensity

### lines 2252-2254

```python
boxes = _label_bounding_boxes(label_mask)
```

The boundary map is a single whole-field pass; the per-object work that follows is confined to each object's own bounding box (see _label_bounding_boxes) instead of comparing the whole field per object.

### line 2258, trailing  _(unsure)_

```python
for region in np.unique(label_mask)[1:]:
```

skip the background label

## _outside_intensity

### lines 2295-2299

```python
shape = np.asarray(label_mask).shape
```

The ring is at most `distance` xy pixels wide, so it lives inside the object's bounding box grown by that much (_ring_padding). Dilating and distance-transforming that box instead of the whole field is exact: the object is the only source in the map either way, and the box holds every voxel the ring can reach.

### line 2305, trailing  _(unsure)_

```python
for region in np.unique(label_mask)[1:]:
```

skip the background label

## _calculate_radial_distribution._calculate_average_intensity

### lines 2373-2374  _(unsure)_

```python
radial_distribution[0] = single_channel_image[region_mask].mean()
```

Degenerate: the cell is a single shell at distance 0. Everything belongs to the innermost bin.

### line 2381  _(unsure)_

```python
if i == num_bins - 1:
```

The final bin is closed so the farthest pixel is not dropped.

## _calculate_radial_distribution

### lines 2393-2404

```python
shape = np.asarray(cell_mask).shape
```

Every pair is measured inside the union of the cell's and the object's bounding boxes, grown by two voxels, rather than over the whole field. The bins are read only inside the cell, and an outer boundary lies at most one voxel outside its object, so that window holds every pixel the result depends on and every distance in it is the distance the whole field would have given: the object is the only source in the transform, and all of it is inside the window.

The window has to be the UNION and not the cell's box alone. Objects are selected by any overlap with the cell, so an object larger than its cell would otherwise contribute no boundary voxel to the crop and every distance in it would come back infinite.

### lines 2432-2434

```python
distance_map = distance_transform_edt(~object_boundary, sampling=spacing)
```

NOT multiplied by cell_region: that zeroed the distance of every pixel outside the cell and put the whole background in bin 0. The cell is applied as a mask when binning instead.

## _calculate_correlation_object_level

### lines 2499-2501

```python
boxes = _label_bounding_boxes(mask)
```

Each object's pixels are gathered from its own bounding box rather than by comparing the whole field once per object. The pixels and their order are the same, so every statistic below is unchanged.

### lines 2518-2522

```python
v1 = np.asarray(object_channel_image1, dtype=np.float64)
```

UNCONDITIONAL since 2026-09-02: the deprecated pair this used to sit beside is gone, so there is nothing left to choose between. Reuses the object_channel_image1/2 vectors the loop already extracted -- that reuse is what makes this ~+20% on the colocalisation block instead of ~+46%.

### line 2533

```python
M1_true = float(a[v2 > thr2].sum() / sa) if sa > 0 else 0.0
```

0.0, not NaN -- see the note in this function's docstring.

## _estimate_blur

### lines 2605-2606

```python
y_axis, x_axis = (mask.ndim - 2, mask.ndim - 1)
```

Bounding box grown by one pixel in y and x so the 3x3 kernel has real neighbours. Not grown in z: the kernel never reaches across planes.

### lines 2619-2622

```python
structure = np.zeros((3, 3, 3), dtype=bool)
```

In-plane erosion only, matching the in-plane kernel: a voxel is interior when its eight xy neighbours in its own plane are all in the object. A 3-D structuring element would additionally require the planes above and below, which no sample of the kernel reads.

### lines 2636-2638

```python
if image.dtype != np.float64:
```

cv2.Laplacian with CV_64F requires a float64 source: float32 (and any integer) inputs raise "Unsupported combination of source/destination format", so promote anything that isn't already float64.

### line 2642  _(unsure)_

```python
image_float = image
```

Already float64 — use as is.

### line 2644  _(unsure)_

```python
if volumetric:
```

Compute the Laplacian of the image

## _measure_intensity_distance

### lines 2679-2681

```python
physical = float(sigma) * float(spacing[-1])
```

sigma is quoted in xy pixels; convert to the same physical length on every axis. With spacing (dz, dxy, dxy) the z sigma is sigma*dxy/dz, i.e. fewer planes for the same distance.

### lines 2739-2744

```python
merged_df = dfs[0]
```

Merge all channel dataframes on label. one_to_one: every frame in `dfs` was built by walking the same `cell_labels` (np.unique of the cell mask) once, so each holds one row per cell and the same set of cells. This is a widening of one table across channels, not a relationship between two different object types -- if a label repeated, a channel's distances would be silently averaged over duplicate rows downstream.

## save_and_add_image_to_grid

### lines 2831-2834

```python
stamp_crop_folder(os.path.dirname(img_path))
```

Mark the folder BEFORE the first PNG lands: a run killed in between then leaves a marked folder holding fewer crops, never an unmarked folder of corrected ones, which is the single state that would be misread as legacy. Costs one stat per folder per process.

### line 2840  _(unsure)_

```python
if png_channels.dtype == np.uint16:
```

Ensure the image is in uint8 format for cv2 functions

### line 2844  _(unsure)_

```python
grid.append(png_channels)
```

Add the image to the diagnostic grid.

## img_list_to_grid

### lines 2863-2866

```python
with figure_style(theme_target()):
```

THE STYLE HAS TO BE ON BEFORE THE FIGURE EXISTS:

rcParams reach an artist when it is CREATED, so a context opened after `plt.subplots` would leave the spines, ticks and labels at the caller's globals.

### lines 2876-2877  _(unsure)_

```python
im = ax.imshow(image)
```

Grid entries are produced from ``png_dims`` in RGB order.  The OpenCV reversal belongs only at the PNG write boundary above.

### lines 2882-2883

```python
h, w = image.shape[:2]
```

Clip each crop to a rounded rectangle so the grid reads like the annotate view (soft corners) rather than hard square tiles.

### line 2894  _(unsure)_

```python
img_height, img_width = image.shape[:2]
```

Determine text size

### line 2901  _(unsure)_

```python
plt.subplots_adjust(wspace=0.08, hspace=0.08)
```

A little more breathing room between crops.

## _per_crop_mode

### line 2945  _(unsure)_

```python
return []
```

crop_mode is empty, so nothing is cropped and nothing indexes this.

## _measure_crop_core

### line 3101  _(unsure)_

```python
def _measure_crop_core(index, time_ls, file, settings):
```

@log_function_call

### lines 3122-3126

```python
from .utils import _merge_overlapping_objects, _filter_object, _relabel_parent_with_child_labels,...
```

spacr.plot is imported where it is used, not here: it is only reachable behind settings['plot'], and under a spawn/forkserver pool every worker pays for every import in this function from a cold interpreter. Measured on a developer box, spacr.plot alone is ~1.9 s and ~720 MB per worker for a default run that never draws anything.

### lines 3152-3153

```python
print(f"WARNING: {file_name} intensity values require x{factor:g} "
```

Deliberately independent of verbose: this conversion changes the unit represented by one stored intensity count.

### lines 3167-3173

```python
if data.ndim == 4 and data.shape[0] == 1:
```

A merged 2-D field is (Y, X, C); a merged z-stack is (Z, Y, X, C). Every slice below therefore indexes the LAST axis -- `data[..., k]` which is exactly `data[:, :, k]` for a 3-D array and the channel rather than a slab of X for a 4-D one. The old `data[:, :, channels]` on a (Z, Y, X, C) array returned shape (Z, Y, len(channels), C): it sliced X, kept every channel, and raised nothing, so the entire run was measuring an arbitrary three-pixel-wide strip of the field.

### lines 3175-3177

```python
data = data[0]
```

A one-plane "volume" is a 2-D field. Squeezing it here means it takes the ordinary 2-D path, so it measures identically to the same field saved without a z axis, and needs no anisotropy.

### lines 3181-3183

```python
spacing, units_stamp = resolve_measurement_spacing(
```

Raises when a 3-D field arrives without a voxel size or anisotropy, which the caller records on the run ledger. Done before any measurement so the run stops rather than half-filling a table.

### lines 3188-3190

```python
print(f"3-D field {file_name}: skipping the cropped-array plots "
```

spacr.plot._plot_cropped_arrays lays out one panel per slice of a (Y, X, C) array; handed a 4-D array it would either raise or plot a slice of X as if it were a channel.

### lines 3204-3215

```python
if preprocessing_hooks():
```

PREPROCESSING EXTENSION POINT. Registered hooks see exactly the array the intensity measurements see: the channels named by settings['channels'], already selected out of the merged stack, and not one feature computed yet. This is where a flat-field / illumination correction belongs. The PNG crops below are cut from `data` and are deliberately NOT rewritten, so the thumbnails stay a faithful record of what the microscope wrote while the numbers in measurements.db carry the correction.

The `if` is not just a micro-optimisation: with an empty registry nothing is allocated and channel_arrays is the identical object, so the default path cannot differ from the pre-hook one.

### lines 3229-3231

```python
cell_max = settings.get('cell_max_size')
```

AN UPPER BOUND TOO. A minimum removes debris; only a maximum removes a segmentation blow-up, which passes every minimum and carries its area into everything downstream.

### lines 3252-3254

```python
nucleus_max = settings.get('nucleus_max_size')
```

AN UPPER BOUND TOO. A minimum removes debris; only a maximum removes a segmentation blow-up, which passes every minimum and carries its area into everything downstream.

### lines 3281-3283

```python
pathogen_max = settings.get('pathogen_max_size')
```

AN UPPER BOUND TOO. A minimum removes debris; only a maximum removes a segmentation blow-up, which passes every minimum and carries its area into everything downstream.

### lines 3303-3308

```python
minimum = settings.get(f'{organelle_role}_min_area')
```

THE SURVIVING NAME. `_min_size` was retired in favour of `_min_area` because the two meant the same thing and were read by different code, so the preview and the run filtered differently with nothing saying so. A file still carrying `_min_size` is migrated by `RETIRED_SETTINGS` before it reaches here.

### lines 3313-3316

```python
current_mask = np.zeros_like(data[..., 0])
```

THE PRIMARY SLOT KEEPS ITS ZEROS FALLBACK, because

`organelle_mask` below is read unconditionally and the measurement path expects an array there whether or not a dimension was configured.

### lines 3319-3335

```python
continue
```

AN UNCONFIGURED SLOT ALLOCATES NOTHING, and this is the whole of the fix. Every role used to get a full-size zero array, so the allocation was a function of the VOCABULARY rather than of the experiment. 326 widened `ORGANELLE_ROLES` from four to 702 to close the untyped organelle collision, and that turned 8.4 MB per field into 1.47 GB at 1024x1024 uint16 -- 5.89 GB at 2048x2048, 2.94 GB at int32. A 175x regression in the measure loop, paid by every run whether or not it uses a single organelle.

Nothing downstream loses anything. Both consumers of these dicts iterate them and act only when `settings[f'{role}_mask_dim'] is not None` -- see `_measure_crop_core` at the `organelle_masks.update(...)` lines -- so the arrays being dropped here are exactly the ones that were allocated, copied, passed down, iterated and never read.

### line 3343  _(unsure)_

```python
if settings['cytoplasm']:
```

Create cytoplasm mask

### line 3346  _(unsure)_

```python
interior = np.zeros_like(cell_mask, dtype=bool)
```

Build a combined interior mask from all subcellular objects

### lines 3376-3388

```python
organelle_mask = organelle_masks['organelle']
```

YES, THIS RUNS TWICE, AND BOTH ARE WANTED. The organelle mask is already filtered where it is read, ~35 lines up, and that early pass is LOAD-BEARING: the cytoplasm mask is built as "cell minus every interior object" from the organelle mask between the two, so filtering only here would carve sub-threshold organelle debris out of the cytoplasm.

The second pass is harmless because `_filter_object` is idempotent -- measured: one pass and two give byte-identical masks -- and it keeps organelle in the same block as its four siblings, where someone looking for "where are the size filters" will find it. Removing either one is a behaviour change; removing the FIRST is a silent one.

### lines 3394-3417

```python
if region_filter_hooks():
```

REGION-FILTER EXTENSION POINT. Registered filters are handed the label ids of each object type (and, only if they ask, the centroids) and return a keep/drop boolean per object; a dropped label is zeroed out of its mask right here. This is where a user-drawn ROI belongs: "only measure inside this polygon".

The position is load-bearing in two directions.

Downstream: every size filter has already run, so a filter sees the objects that would actually have been measured -- and nothing has been measured yet, so keeping 5 of 500 objects costs 5 objects' worth of morphology, intensity, texture, radial-distribution and Zernike work rather than 500 followed by a DataFrame subset.

Upstream of _exclude_objects: culling a cell there propagates to its nucleus/pathogen/cytoplasm (they are multiplied by the surviving cell mask), which is what keeps the validate='one_to_one' parent joins in _morphological_measurements satisfiable. Filtering after it would let an ROI keep a nucleus whose cell it had just deleted.

It is also upstream of the `data[..., <mask>_dim] = ...` write-backs, so the PNG crops and region arrays cover the same objects the database does -- an object outside the ROI is not measured AND not cropped, rather than appearing in one output and not the other.

### lines 3449-3451

```python
for organelle_role, current_mask in organelle_masks.items():
```

Child tables must not keep objects whose parent cell was culled. The legacy helper knows only nucleus/pathogen/cytoplasm; apply its same pixel-level rule to every registered organelle slot.

### lines 3465-3474

```python
for organelle_role, current_mask in organelle_masks.items():
```

ORGANELLE WAS MISSING FROM THIS BLOCK. Cell, nucleus and pathogen were each written back; organelle was not, though its mask IS modified above -- `_filter_object` drops everything under `organelle_min_size` at the read. So the array kept the UNFILTERED organelle plane while the measurements used the filtered one.

The comment above these write-backs states the invariant they exist for: "the PNG crops and region arrays cover the same objects the measurements do". Organelle was outside that guarantee, so a crop could show debris the measurement table had already dropped.

### lines 3489-3507

```python
role_order = [
```

NAMED FOR WHAT WAS PASSED, NOT FOR THE WHOLE VOCABULARY. These names are zipped POSITIONALLY against the measurement lists below, and those lists carry one entry per mask that went in -- `_morphological_measurements` appends an empty frame for an unconfigured role rather than skipping it, which is what kept the old pairing aligned.

So `*ORGANELLE_ROLES` was only ever correct because

`organelle_masks` held EVERY role, configured or not, which is the 1.47 GB-per-field allocation this function no longer makes. With only the configured slots passed, a 705-name list zipped against a shorter result list silently truncates -- and `zip` drops from the END, so 'cytoplasm' was the entry lost and its morphology was looked up under an organelle's name.

Deriving the order from `extra_organelle_masks` keeps the two in step by construction. `_measure_crop_core` builds its dict as {'organelle': ...} then updates with the extras, so this is that same order.

### lines 3517-3520

```python
channel_arrays=channel_arrays)))
```

THE INTENSITY IMAGES, for the distance families that need them: local maxima and the intensity-centre offset. Optional, so a caller that only wants geometry passes nothing and pays for nothing.

### lines 3043-3044

```python
if frame.empty:
```

A parent mask with no objects left in this field -- every cell under `cell_min_size`, say -- makes `_summarize_organelles_per_parent` return a frame with NO COLUMNS, not a frame with no rows. One enabled slot passes that frame straight to `_merge_and_save_to_database`, which writes nothing for an empty frame. Two or more slots merged those frames on `label`, which an empty frame does not have, so the field raised `KeyError: 'label'` and was reported as failed (measured 2026-09-19 through External Masks with the default 8000 px cell filter; item 76). Skipping the empty frames makes two slots behave as one does. The frames are all empty or none are: every slot is summarised over the same parent labels.

### lines 3593-3596

```python
_write_intensity_rescale_record(
```

This is written after every requested measurement table has landed, so a worker that fails before producing measurements cannot leave a provenance row that makes the field look complete. The primary-key upsert keeps retries idempotent.

### lines 3601-3608

```python
print(f"3-D field {file_name}: measurements written, but no PNG "
```

Refused, not approximated. Every step of the crop path is irreducibly 2-D: _crop_center and _find_bounding_box take (row, col), cv2.imwrite writes an image, and the PNGs are the training input for spacr.deep_spacr, whose models take H x W x 3. Cropping a projection instead would silently substitute a different measurement of the object -- one where anything sitting above or below another object is merged into it -- under the same file names, and nothing downstream could tell it had happened.

### lines 3618-3622

```python
crop_ls = settings['crop_mode']
```

A bare string crop_mode used to be assigned to a local named

`crop_mode` and then thrown away: the very next line re-tested settings['crop_mode'] for list-ness, which a string never is, so the entire crop block was skipped. crop_mode='cell' wrote the measurements and NOT ONE PNG, without an error.

### lines 3633-3635

```python
if not isinstance(size_ls[0], (list, tuple)):
```

`isinstance(size_ls[0], int)` missed a float or a numpy int, and then `width, height = size_ls[crop_idx]` tried to unpack a scalar. Ask what it IS -- a pair, or a list of pairs.

### lines 3639-3641

```python
size_ls = _per_crop_mode(size_ls, len(crop_ls), 'png_size')
```

All three per-mode settings now broadcast the same way, so png_size no longer prints a mismatch warning and then raises IndexError on the very next line.

### lines 3650-3654

```python
if crop_mode not in CROP_MODES:
```

An unrecognised crop mode used to print and fall through, so crop_mask/dialate_png kept the PREVIOUS mode's values: crop_mode=['cell','banana'] cropped the cell mask a second time under the name 'banana'. Skip it instead, and name the modes that exist.

### lines 3675-3677

```python
dialate_png = False
```

Dilating a cytoplasm ring grows it into the nucleus it is defined as excluding, so the crop would no longer be cytoplasm. Not a user choice.

### lines 3679-3680

```python
dialate_png_ratio = dialate_png_ratios[crop_idx]
```

Assigned even though dilation is off, so the name never carries a previous crop mode's ratio into this one.

### line 3691  _(unsure)_

```python
region_cell_ids = np.atleast_1d(np.unique(cell_mask[region]))
```

Use the boolean mask to filter the cell_mask and then find unique IDs

### lines 3707-3712

```python
region_area = np.count_nonzero(region)
```

count_nonzero, not np.sum: when use_bounding_box is on, _find_bounding_box fills the box with the LABEL VALUE rather than with True, so np.sum gave pixels * label and object 100 dilated sqrt(100)=10x more than object 1 -- the crop depended on an arbitrary label id.

### lines 3714-3721

```python
approximate_diameter = np.sqrt(region_area)
```

The diameter of an object from its size is the ndim-th root of that size, not always the square root: a voxel count is a volume, and sqrt of a volume is not a length. Unreachable for a 3-D field today (the whole crop block is refused above), but wrong is wrong. Volumetric fields are refused before entering the crop block, so every region here is 2-D.

### lines 3724-3729

```python
if dialate_png_px > 0:
```

scipy reads iterations=0 as "repeat until nothing changes", NOT as "do nothing", so a radius that rounded down to 0 -- every object under 25 px at the default ratio 0.2 -- grew to fill the entire field. The crop then became an unmasked window centred on the middle of the field instead of on the object.

### lines 3731-3734

```python
struct = generate_binary_structure(region.ndim, region.ndim)
```

scipy requires the structuring element to have the same rank as the input; a fixed (2, 2) raises "structure and input must have same dimensionality" on a volume.

### lines 3744-3748

```python
png_channels = build_png_channels(
```

Assembled in FILE order -- red plane first -- from the declared mapping, so what the setting says is what the PNG's slots hold. `png_dims` still works and is translated to the mapping it always meant (entry 0 blue, 1 green, 2 red).

### lines 3768-3772

```python
grid = save_and_add_image_to_grid(
```

`build_png_channels` returns 1 plane (greyscale) or 3

(r, g, b) and never 2, so the pad-a-dummy-plane branch that used to live here is gone: a two-entry mapping already carries its empty plane, in the slot the user left blank rather than always the last one.

### lines 3784-3787

```python
from .normalization import apply_crop_dtype
```

`original` by default, so this stays the bare save it has always been -- uint16 in, uint16 out. The setting is a STORAGE choice; training precision is decided by ToTensor, which divides by 255 whatever is here.

### lines 3794-3797

```python
cells = np.unique(cell_mask)
```

Region arrays are independent of PNG output. In particular, save_arrays=True/save_png=False must not reference the PNG-only locals ``png_channels`` and ``img_path`` or register a .npy path in ``png_list``.

### lines 3802-3806

```python
cells = 0
```

`cells = 0` (a plain int) is the cross-process failure sentinel: the success path always assigns np.unique(...), an ndarray, so the parent's job_callback can tell the two apart and file this field on the run ledger. Without that the pool callback saw a normal result and the run reported as complete.

### lines 3808-3820

```python
error_text = "".join(
```

THE TRACEBACK GOES HOME WITH THE RESULT, because this runs in a multiprocessing.Pool worker and the parent's logging configuration is not this process's. `traceback.print_exc()` here writes to a worker stderr nobody is reading, and `RunLedger(...)` opened here is a second ledger in a second process -- so the parent could say only "worker traceback in ~/.spacr/logs/spacr.log", which was NOT TRUE: reported 2026-09-01 against plate1_E02_20_1.npy, where the named log held nothing about it and the one thing needed to fix the field was the one thing thrown away.

Returned as text rather than as the exception: an exception is not always picklable, and a field that fails with an unpicklable error would then fail again on the way back, losing the first failure.

### lines 3829-3831

```python
if settings['plot'] and grid:
```

A volumetric field deliberately skips every 2-D crop plot, leaving the grid empty. Matplotlib cannot build a zero-row subplot grid, so there is simply no ``__pngs`` figure in that case.

## _record_organelle_caveats

### line 3837  _(unsure)_

```python
def _record_organelle_caveats(settings, run):
```

@log_function_call

## measure_crop

### lines 3928-3930

```python
if settings.get('dry_run', False):
```

dry_run comes FIRST, before the local imports below: .io and .timelapse are heavy, and _save_settings_to_db writes to measurements.db as a side effect further down. A validate-only run must not reach either.

### lines 3962-3965

```python
with run_context('measure', settings) as run:
```

One run for the whole invocation: one id on every log line and every artifact it registers, one seed reaching numpy / random / torch, and one on_error policy honoured at the per-field boundary inside the pool loop. See spacr.runctx.

### lines 3970-3972

```python
settings = dict(base_settings)
```

Defaults and a previous source folder must not leak into the next one. In particular each merged folder may carry a different authoritative plane-layout manifest.

### lines 3993-4007

```python
from .database_concurrency import enable_wal_where_safe
```

Issue #15, "measurements sometimes hangs from completion". This run is about to start one worker per field, and every append issues pandas' `has_table` probe first -- a READ before writing. Under the rollback journal a writer cannot COMMIT until every reader's SHARED lock is gone, so with enough workers someone is always reading and the committing worker waits out its busy timeout: measured at 1.037 s blocked under DELETE against 0.002 s under WAL. "database is locked" then surfaces on whichever `has_table` loses, which is the reporter's traceback.

Once per source folder, not per write: the mode is a property of the file and persists. Silently declined on any filesystem not positively identified as local, which leaves the shipped DELETE behaviour exactly as it was.

### lines 4015-4027

```python
from .illumination import (
```

Illumination / flat-field correction, if the settings ask for it. Here, and not earlier: it estimates from the merged fields this loop is about to measure, so it needs `src` after the /merged normalisation above, and it is per source folder because illumination differs between acquisition sessions. It installs a preprocessing hook (and the env vars that carry it into every spawned worker), so it has to run before the pool below is built rather than beside it.

Off unless `illumination_correction` is True, in which case this call is the whole feature: without it the setting is a switch that does nothing and every intensity feature keeps its position-dependent bias. See spacr.illumination.

### lines 4063-4067

```python
if isinstance(settings['normalize'], bool) and settings['normalize']:
```

Category B, every one of these: the settings are wrong, so no field can be measured. Each historically printed a WARNING and returned None, which the caller cannot distinguish from a completed run that wrote no rows. SPACR_STRICT_ERRORS turns them into a ConfigurationError; the default stays as-is.

### lines 4085-4090

```python
if not all(isinstance(settings.get(key), int)
```

Secondary organelle slots beyond ``number_of_organelles`` are intentionally absent from the settings mapping. Missing therefore means the same thing as an explicit ``None``: this run has no mask/minimum for that optional slot. Direct indexing made every ordinary one-organelle Measure demo die on the first undeclared slot (``organelleb_mask_dim``).

### lines 4114-4116

```python
resume_plan = plan_measure_resume(settings)
```

MUST come before _save_settings_to_db: that writes the settings table with if_exists='replace', destroying the record of the run being resumed — which is what the settings comparison reads.

### lines 4122-4126

```python
_full_rescale_plan = build_plate_plan(
```

Scan the complete plate, not merely the fields left after a resume filter. Otherwise a resumed field could receive a different factor from fields already present in the same database. The plan is plain data and is copied into every worker with the settings.

### lines 4129-4132

```python
settings[PLAN_SETTINGS_KEY] = {
```

Do not pickle one per-field metadata record with every pool task. Workers recompute their own maximum from the array they already loaded; they need only the O(number of plates) maxima and the exceptional filenames.

### lines 4152-4156

```python
ledger = RunLedger('measure_crop')
```

One ledger per source folder. Both failure routes are covered: a worker that returned the cells==0 sentinel (it caught its own exception), and a worker that died outright — the latter used to be completely invisible, because apply_async stores the exception on an AsyncResult nobody ever read.

### lines 4158-4161

```python
run.adopt(ledger)
```

This folder's ledger joins the run: the ledger's run_id, every log line below and every artifact this run registers all carry one id, so the log of the run that produced a measurements.db can be pulled back with spacr.runctx.read_run_log().

### lines 4163-4170

```python
_record_organelle_caveats(settings, run)
```

WHAT THE ORGANELLE NUMBERS ABOUT TO BE WRITTEN WILL AND

WILL NOT MEAN, on the run's own journal. The caveat reached the console and stopped there, so a run read back later which is the only way anyone reads a batch -- carried the count-dependent columns with nothing beside them saying a reticular organelle is one connected object per cell and its neighbour count is therefore a fact about the segmentation.

### lines 4252-4254

```python
ctx = _pool_context()
```

One explicit context for both the Manager and the Pool. Mixing them -- a fork Manager serving a spawn Pool, say -- is how the shared time_ls proxy ends up unreachable from a worker.

### lines 4257-4260

```python
warn_if_hooks_will_not_reach_workers(start_method)
```

A spawn/forkserver worker is a fresh interpreter with empty hook registries, so a hook registered in *this* process would apply to nothing at all and the run would look completely normal. Say so before the pool starts rather than let it be invisible.

### lines 4265-4272

```python
try:
```

_start_manager, not ctx.Manager(), because the bare call fails as an EOFError from deep inside multiprocessing with no message at all. See ManagerStartError. try/finally, because on_error='stop' aborts here and an aborted run's evidence is exactly what a reader needs: without this the fields nobody heard from go uncounted and measurements.db is never stamped, so a half-written database reads as one nobody ever measured into.

### line 4276, trailing  _(unsure)_

```python
completed_jobs = set()
```

Set to keep track of completed jobs

### lines 4279-4281

```python
for offset in range(0, len(files), pool_jobs):
```

Bound outstanding work to one pool-width batch. Stop is checked only after all fields in that batch have completed their writes.

### lines 4294-4301

```python
for attempt in policy.attempts_for(
```

on_error, at the per-field boundary. The ledger entry is written either way by job_callback / make_error_callback, which is why the policy is bound with record=False; what on_error decides is whether the run survives the field. retry re-submits the field rather than re-reading the AsyncResult, which can only be got once.

### lines 4316-4319

```python
if attempt.last:
```

Only on the last attempt: the ledger counts fields, not tries, so a field that failed twice and then worked is one success.

### lines 4328-4331

```python
for job_file in files:
```

Fields the pool never reported on at all (killed worker, pool terminated before the task ran, or on_error='stop' ending the run at the first bad field). Counting them keeps n_attempted equal to the number of fields on disk.

### lines 4337-4343

```python
db_path = os.path.join(os.path.dirname(settings['src']),
```

Stamp measurements.db with the verdict, then print it last. This is the bit that turns "we printed a warning" into "the artifact knows it is suspect": spacr.errors.read_run_status() on this db tells a downstream reader how many fields are missing. In the finally, because an aborted run is exactly the one whose half-written database must not read as untouched.

### lines 4359-4363

```python
run.register_outputs(settings=settings, roots=source_folders)
```

Record what this run produced, stamped with the run id every log line above carries, so an artifact and its log can be joined: spacr.runctx.read_run_log(artifact.run_id). The canonicalized settings, not the ones handed in, so the hash recorded against each artifact covers the values actually used.

## measure_crop.job_callback

### lines 4201-4202

```python
if isinstance(result[2], int) and result[2] == 0:
```

cells is np.unique(cell_mask) on success and the int 0 when _measure_crop_core swallowed an exception for this field.

### lines 4204-4208

```python
detail = (result[4] if len(result) > 4 else "") or (
```

The worker's own traceback, carried back in the result. Recorded HERE, in the parent, whose logging configuration is the one writing spacr.log -- so the message that says the traceback is in the log is true.

## process_measure_crop_results

### lines 4377-4379

```python
index, avg_time, cells, figs = result[:4]
```

Five since the worker started carrying its traceback home; the four-tuple form is still accepted so a partial result saved by an older run can still be processed.

### lines 4387-4390

```python
from .plot import save_figure
```

Imported here, not at module scope: `spacr.plot` pulls in torch, cv2, seaborn, statsmodels and pingouin, and this module is on the cold measure-worker spawn path. See tests/test_measure_spawn.py.

### lines 4393-4396

```python
with figure_style(theme_target()):
```

THE STYLE HAS TO BE ON BEFORE THE FIGURE EXISTS:

rcParams reach an artist when it is CREATED, so a context opened after `plt.subplots` would leave the spines, ticks and labels at the caller's globals.

## generate_cellpose_train_set

### line 4445, trailing  _(unsure)_

```python
for filename in os.listdir(mask_folder):
```

List the contents of the directory

### lines 4454-4455

```python
ledger.record_failure(path, stage='read_mask',
```

cv2 signals failure by returning None rather than raising, so this needs recording explicitly.

### line 4461, trailing  _(unsure)_

```python
nr_of_objects = len(np.unique(mask)) - 1
```

Assuming 0 is background

### line 4462, trailing  _(unsure)_

```python
if nr_of_objects >= min_objects:
```

Use >= to include min_objects

## get_object_counts

### line 4484  _(unsure)_

```python
df = pd.read_sql_query("SELECT * FROM object_counts", conn)
```

Read the table into a pandas DataFrame

### line 4491  _(unsure)_

```python
conn.close()
```

Close the database connection

## _crop_full_scale

### lines 4498-4515

```python
def _crop_full_scale(dtype):
```

Object crops: the working dtype, and the one place it is left behind

A merged array is 16-bit (``uint16``, or ``int32`` once a cellpose label plane has been concatenated onto it). The crop path keeps that dtype from the ``.npy`` all the way to the writer, exactly as ``_measure_crop_core`` does: nothing in the middle of the pipeline is allowed to change it.

8-bit is genuinely required at exactly two places -- a PNG assembled by PIL (:func:`_save_object_crop`) and an RGB image handed to a GUI (``crop_objects_from_array(to_rgb=True)``). Both go through :func:`_crop_to_uint8`, which *rescales*. They used to go through ``np.clip(crop, 0, 255).astype(np.uint8)``, which does not: on a raw 16-bit crop every pixel above 255 -- i.e. every pixel of the object -- came out at exactly 255. Unnormalised 16-bit data shown as 8-bit has to look DARK; a clip is what turned it white, and those white crops were written to disk and trained on.

## generate_object_dataset

### lines 4778-4792

```python
stamp_crop_folder(output_dir)
```

Mark the folder BEFORE the first PNG lands, for the same reason `save_and_add_image_to_grid` does (crops.stamp_crop_folder): an unmarked folder means LEGACY to every reader, and these crops are not legacy.

`_save_object_crop` writes through PIL, which is already RGB -- so unlike the cv2 writer there is nothing to reverse here, and the bytes on disk were correct all along. What was missing was the marker saying so. Without it `crops.read_crop_png` resolved the folder to format 1 and reversed a correct file on load, so the annotator, the crop grid and the training loaders all showed channel 0 as blue and channel 2 as red while an external viewer showed them the right way round. Measured on a crop written with channel means (60000, 1200, 12000): PIL read (234, 4, 46) off the file, `read_crop_png` returned (46, 4, 234).

### lines 4871-4873

```python
raise ValueError(
```

A (Z, Y, X, C) volume. Everything below is 2-D indexing, and

`data[:, :, mask_dim]` on a 4-D array returns a slab of X, not a mask, without raising.

## crop_objects_from_array

### lines 4998-4999  _(unsure)_

```python
scored = []
```

Order by area (largest first) so the preview leads with the clearest objects; apply the area filter here too.

### lines 5014-5018

```python
ys, xs = np.where(mask == lbl)
```

No "label vanished" guard here, unlike generate_object_dataset: `scored` was built from np.unique of THIS plane a few lines up, so every label in it is in it. (In generate_object_dataset the label comes from the database and the plane from disk, which is a real chance to disagree, and that guard is exercised.)

### lines 5029-5031

```python
crop = _crop_to_uint8(crop)
```

Declared 8-bit boundary: a QImage/RGB888 wants uint8. Narrow by rescaling (_crop_to_uint8), then assemble -- so a raw 16-bit field previews dark rather than solid white.

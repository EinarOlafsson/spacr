# Notes from `spacr/submodules.py`

Prose lifted out of `spacr/submodules.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (5 entries)
- [display](#display) (1 entry)
- [train_cellpose](#train_cellpose) (5 entries)
- [test_cellpose_model.plot_cellpose_resilts](#test_cellpose_modelplot_cellpose_resilts) (3 entries)
- [test_cellpose_model](#test_cellpose_model) (12 entries)
- [apply_cellpose_model](#apply_cellpose_model) (2 entries)
- [plot_cellpose_batch](#plot_cellpose_batch) (1 entry)
- [analyze_percent_positive.translate_well_in_df](#analyze_percent_positivetranslate_well_in_df) (4 entries)
- [analyze_percent_positive.annotate_and_summarize](#analyze_percent_positiveannotate_and_summarize) (2 entries)
- [analyze_percent_positive](#analyze_percent_positive) (3 entries)
- [analyze_recruitment](#analyze_recruitment) (2 entries)
- [_resolve_plaque_model](#_resolve_plaque_model) (1 entry)
- [analyze_plaques](#analyze_plaques) (4 entries)
- [count_phenotypes](#count_phenotypes) (6 entries)
- [compare_reads_to_scores](#compare_reads_to_scores) (6 entries)
- [compare_reads_to_scores.calculate_well_score_fractions](#compare_reads_to_scorescalculate_well_score_fractions) (2 entries)
- [compare_reads_to_scores.plot_line](#compare_reads_to_scoresplot_line) (8 entries)
- [compare_reads_to_scores.calculate_grna_fraction_ratio](#compare_reads_to_scorescalculate_grna_fraction_ratio) (1 entry)
- [compare_reads_to_scores.calculate_well_read_fraction](#compare_reads_to_scorescalculate_well_read_fraction) (1 entry)
- [interpret_vision_model.generate_comparison_columns](#interpret_vision_modelgenerate_comparison_columns) (8 entries)
- [interpret_vision_model.group_feature_class](#interpret_vision_modelgroup_feature_class) (2 entries)
- [interpret_vision_model.create_extended_radar_plot](#interpret_vision_modelcreate_extended_radar_plot) (3 entries)
- [interpret_vision_model.extract_compartment_channel](#interpret_vision_modelextract_compartment_channel) (4 entries)
- [interpret_vision_model.read_and_preprocess_data](#interpret_vision_modelread_and_preprocess_data) (8 entries)
- [interpret_vision_model](#interpret_vision_model) (11 entries)
- [analyze_endodyogeny._calculate_volume_bins](#analyze_endodyogeny_calculate_volume_bins) (8 entries)
- [analyze_endodyogeny](#analyze_endodyogeny) (4 entries)
- [_compose_field_keys](#_compose_field_keys) (2 entries)
- [_ensure_field_key](#_ensure_field_key) (4 entries)
- [_set_analyze_replication_defaults](#_set_analyze_replication_defaults) (1 entry)
- [_assign_vacuole_ids](#_assign_vacuole_ids) (1 entry)
- [_replication_well_distribution](#_replication_well_distribution) (1 entry)
- [_replication_compare_conditions](#_replication_compare_conditions) (3 entries)
- [analyze_replication](#analyze_replication) (7 entries)
- [_set_analyze_invasion_defaults](#_set_analyze_invasion_defaults) (1 entry)
- [_resolve_invasion_background_column](#_resolve_invasion_background_column) (1 entry)
- [_bimodality_coefficient](#_bimodality_coefficient) (1 entry)
- [_invasion_threshold](#_invasion_threshold) (1 entry)
- [_invasion_field_thresholds](#_invasion_field_thresholds) (1 entry)
- [_invasion_classify](#_invasion_classify) (3 entries)
- [_invasion_well_table](#_invasion_well_table) (1 entry)
- [_invasion_stacked_bars](#_invasion_stacked_bars) (1 entry)
- [_invasion_threshold_panels](#_invasion_threshold_panels) (3 entries)
- [analyze_invasion](#analyze_invasion) (10 entries)
- [analyze_class_proportion](#analyze_class_proportion) (4 entries)
- [generate_score_heatmap.group_cv_score](#generate_score_heatmapgroup_cv_score) (1 entry)
- [generate_score_heatmap.calculate_fraction_mixed_condition](#generate_score_heatmapcalculate_fraction_mixed_condition) (4 entries)
- [generate_score_heatmap.plot_multi_channel_heatmap](#generate_score_heatmapplot_multi_channel_heatmap) (5 entries)
- [generate_score_heatmap.combine_classification_scores](#generate_score_heatmapcombine_classification_scores) (13 entries)
- [generate_score_heatmap.calculate_mae](#generate_score_heatmapcalculate_mae) (2 entries)
- [generate_score_heatmap](#generate_score_heatmap) (2 entries)
- [post_regression_analysis._analyze_and_visualize_grna_correlation](#post_regression_analysis_analyze_and_visualize_grna_correlation) (3 entries)
- [post_regression_analysis._compute_effect_sizes](#post_regression_analysis_compute_effect_sizes) (6 entries)
- [post_regression_analysis](#post_regression_analysis) (2 entries)

## Module level

### lines 72-75

```python
from .tabular import read_table
```

THE ONE READER (145). `spacr.tabular` imports pandas and nothing else, so naming it at module scope costs nothing -- and a local import in each of the eight functions that read a table here would be eight places for the next one to be forgotten.

### line 126, trailing  _(unsure)_

```python
from .plot import save_figure
```

every kept figure goes through the format/DPI preference

### line 2364  _(unsure)_

```python
interperate_vision_model = interpret_vision_model
```

Backward compatibility for the misspelling published in earlier releases.

### lines 3815-3824

```python
_INVASION_STATISTIC_TEMPLATES = {
```

Per-object statistics of the outside-stain channel, in the naming :func:`spacr.measure._intensity_measurements` actually writes.

Careful with the word "outside": measure.py's ``<object>_channel_<n>_outside_*`` columns are the intensity of a five-pixel ring *outside the object's own mask* (:func:`spacr.measure._outside_intensity`) in whatever channel is named. They are a local background estimate, and they have nothing to do with the outside/inside *stain* of this assay. The assay's outside stain is a channel, selected with ``outside_channel``; the statistics below read the parasite's own pixels in that channel.

### lines 3837-3838  _(unsure)_

```python
_INVASION_STATISTIC_AUTO_ORDER = ('periphery_95', 'percentile_95', 'mean')
```

Resolution order for intensity_statistic='auto'. The order is the argument in :func:`_resolve_invasion_intensity_column`.

## display

### lines 92-95

```python
def display(*args, **kwargs):
```

IPython may be mid-init (partially imported by another thread) — use a no-op fallback so importing this module never blocks. spaCR only calls display() from notebook contexts anyway; the Qt GUI ignores it.

## train_cellpose

### lines 344-355

```python
model_name = f"{settings['model_name']}_cpsam_e{settings['n_epochs']}_X{target_size}_Y{target_siz...
```

`_cyto_` was a Cellpose-3 leftover: it named the cyto model this function used to fine-tune. It fine-tunes 'cpsam' (below) and has done since the Cellpose 4 port, so the old infix stamped 'cyto' onto a CPSAM checkpoint and a user reading the filename was told the wrong architecture. New checkpoints say cpsam.

Names written before this change keep working: nothing parses the infix. spacr.model_zoo recognises a Cellpose checkpoint by its ``.CP_model`` / ``.CPmodel`` SUFFIX (model_zoo.CELLPOSE_SUFFIXES) or by the folder it sits in, and _resolve_cellpose_pretrained loads any existing path as given -- so `foo_cyto_e500_X1120_Y1120.CP_model` on disk still resolves, still loads, and still versions.

### line 372  _(unsure)_

```python
matched_filenames = sorted(image_filenames & label_filenames)
```

Only keep files that are present in both folders

### lines 383-394

```python
max_train_images = settings.get('max_train_images')
```

EVERY annotated field is training data. ``batch_size`` is the optimizer's minibatch size and is passed straight to train_seg below — it is not, and never was meant to be, a cap on the dataset.

This used to read ``n_base = min(settings['batch_size'], max_base_images)`` followed by ``unique_base_indices[:n_base]``, so a user who annotated 300 fields and left ``batch_size`` at its default of 8 trained on 8 images (2.7% of their work) and was told nothing.

``max_train_images`` is an opt-in ceiling for machines that cannot hold the whole set in RAM (the images are materialised as float32 arrays before train_seg sees them). Unset/None means "use everything".

### lines 417-419

```python
plot_cellpose_batch(images[:_TRAIN_PREVIEW_N], labels[:_TRAIN_PREVIEW_N])
```

Preview a handful only: plot_cellpose_batch lays out one column per image at 4 inches each, so handing it a full 300-image training set asks matplotlib for a 100-foot-wide figure.

### lines 428-430

```python
train_cp.train_seg(model.net,
```

Cellpose 4.x (SAM era) dropped the ``channels`` kwarg from train_seg — models are channel-agnostic now and take a ``channel_axis`` instead (None = greyscale / already-stacked).

## test_cellpose_model.plot_cellpose_resilts

### lines 473-475

```python
with figure_style(theme_target()):
```

THE STYLE OPENS BEFORE THE FIGURE EXISTS: rcParams reach an artist when it is CREATED, so a context entered after plt.subplots leaves the titles and the ground at whatever the session's globals are.

### lines 481-484

```python
axs[0].imshow(img, cmap='gray')
```

Greyscale per channel, the label maps in their random colours because a mask's colours are identities and not a quantity, and the column name as the header -- the micrograph row the style describes.

### lines 506-507  _(unsure)_

```python
save_path = save_figure(fig, save_path,
```

Saved inside the context: savefig.transparent and savefig.facecolor are read at write time.

## test_cellpose_model

### line 527  _(unsure)_

```python
matched_filenames = sorted(image_filenames & label_filenames)
```

Only keep files that are present in both folders

### lines 545-549

```python
n_objects_true_ls = []
```

These per-image metric lists used to be re-initialised INSIDE the batch loop, while names/scores accumulated across batches — so the df_results build below raised "All arrays must be of the same length" as soon as there was more than one batch. They belong here, next to names/scores, so every image contributes exactly one row.

### lines 557-559

```python
files_to_process = len(test_image_files)
```

test_image_folder is a path STRING (os.path.join), so len() of it measured the number of characters in the path, not the number of images — the progress line reported a nonsense total.

### lines 567-570

```python
masks_pred, flows, _ = model.eval(x=list(images),
```

Cellpose 4.x dropped ``interp`` and ``tile`` from eval; the tiling behaviour is now controlled by ``tile_overlap`` alone. It dropped ``channels`` too -- it logs "channels deprecated in v4.0.1+" and never reads the value, so [0, 0] configured nothing.

### lines 584-587

```python
aji = np.asarray(
```

Cellpose 4 returns one AJI value per mask as a 1-D ndarray; older releases returned a scalar. Normalise both contracts without averaging across images (this loop records one row per image).

### line 597  _(unsure)_

```python
lbl_lab = label(lbl)
```

Label masks

### line 601  _(unsure)_

```python
n_true = lbl_lab.max()
```

Count objects

### line 607  _(unsure)_

```python
area_true = [p.area for p in regionprops(lbl_lab)]
```

Mean object size (area)

### line 616  _(unsure)_

```python
ap, tp, fp, fn = average_precision([lbl], [pred], threshold=[0.5])
```

Compute object-level TP, FP, FN

### line 623  _(unsure)_

```python
prec = tp / (tp + fp) if (tp + fp) > 0 else 0
```

Precision, Recall, F1, Accuracy

### lines 634-636

```python
if settings['save']:
```

This block used to be duplicated verbatim, so every diagnostic figure was rendered and savefig'd twice to the same cellpose_result_{i+j:03d}.png path.

### lines 642-645

```python
files_processed = min(i + batch_size, len(test_dataset))
```

i already steps by batch_size, i.e. it IS the dataset index of the first image of this batch — (i+1)*batch_size overshot the number of images actually processed on every batch after the first. min() clamps the final, partial batch.

## apply_cellpose_model

### lines 769-772

```python
masks_pred, flows, _ = model.eval(x=list(images),
```

Cellpose 4.x dropped ``interp`` and ``tile`` from eval; the tiling behaviour is now controlled by ``tile_overlap`` alone. It dropped ``channels`` too -- it logs "channels deprecated in v4.0.1+" and never reads the value, so [0, 0] configured nothing.

### lines 814-819

```python
df_measurements = pd.DataFrame(measurements, columns=['image', 'object_id', 'area'])
```

Write after each batch. The columns must be declared: when a batch finds no objects (blank field, aggressive CP_probability, or circularize=True zeroing every peripheral object) `measurements` is still [] and pd.DataFrame([]) has NO columns, so the groupby below died with KeyError('image') and left measurements.csv as a bare newline that pd.read_csv rejects.

## plot_cellpose_batch

### lines 842-844

```python
with figure_style(theme_target()):
```

squeeze=False keeps axs 2-D for every batch size; with the default squeeze=True a single-image batch collapsed to a 1-D array and the axs[0, i] indexing below raised IndexError.

## analyze_percent_positive.translate_well_in_df

### lines 881-884

```python
df = read_table(csv_loc)
```

Load and extract metadata, THROUGH THE ONE READER (145): the plate and well columns are exactly what canonicalisation is for, and a file spelling them `Plate` / `Well` read back here as columns nothing downstream looks for.

### lines 886-893

```python
stems = df['Renamed TIFF'].map(
```

A renamed TIFF is '<plate>_<well>_<vendor token>.tif' (the convert_to_yokogawa contract). Taking the plate and the well from the FRONT was wrong in two ways that both end in a silently empty join: a plate id containing an underscore made the second token the plate's own tail rather than the well, and '.str.replace(".tif")' matched the substring anywhere in the name and missed '.tiff' entirely. Read the stem right to left instead — the vendor token never contains '_', so whatever is left over is the plate.

### lines 925-929

```python
wells = df_2['well'].map(lambda w: schema.parse_well(w))
```

Translate well to row and column. Through spacr.schema, so that a lowercase well, a 1536-plate row ('AA01') and a separator-bearing one ('A-01') get the same rowID here as they do in measurements.db. The hand-rolled version used string.ascii_uppercase.index, which raised ValueError on all three and took the whole CSV with it.

### line 935, trailing  _(unsure)_

```python
df_2['fieldID'] = schema.field_id(1)
```

default or extract from filename if needed

## analyze_percent_positive.annotate_and_summarize

### line 951  _(unsure)_

```python
df[annotation_col] = np.where(df[value_col] > threshold, 'above', 'below')
```

Annotate

### line 957  _(unsure)_

```python
count_df['total'] = count_df.sum(axis=1)
```

Calculate total and fractions

## analyze_percent_positive

### lines 981-989

```python
prc_parts = count_df['prc'].astype(str).map(
```

'prc' is plate_row_column, so it comes apart from the RIGHT: the plate is whatever is left once the row and the column have been taken off. The positional str.split('_') here assumed a plate with no underscore in it. Anything else raised the opaque "ValueError: Columns must be same length as key", which names neither the key nor the row that produced it; and a frame holding one SHORT key among good ones got that row's column_name filled with None, which merges to nothing without a word. schema has parse_prcf/parse_prcfo but no parse_prc; the rsplit below is the same right-to-left reading (see 'needs follow-up in spacr/schema.py').

### lines 1009-1018

```python
merged = pd.merge(count_df, translate_df, on=['rowID', 'column_name'], how='inner', validate='man...
```

many_to_many, and it genuinely is: the key deliberately omits the plate. count_df's plateID comes out of the measurements 'prc' while translate_df's is the plate written into the file name, and the two use different spellings ('plate1' vs the 'p1' this function rebuilds), so they cannot be joined on. That leaves one row per (row, column) per PLATE on both sides, and every plate's wells therefore match every other plate's. On a single-plate run — what analyze_percent_positive is written for — this is one-to-one; on a multi-plate rename_log it fans out, which is why 'plateID_y' is carried into the result below so the reader can see which plate each row came from.

### lines 1021-1025

```python
merged = merged[['plateID_y', 'well', 'plate_well','fieldID','rowID','column_name','prc_x','Origi...
```

'plateID_y' is the plate parsed from rename_log.csv's 'Renamed TIFF'. This used to read 'plate_y', a leftover from before the plate -> plateID rename: neither frame carries a 'plate' column any more, so pandas never synthesises a 'plate_y' suffix and the selection always raised KeyError "['plate_y'] not in index".

## analyze_recruitment

### lines 1094-1098

```python
settings['src'] = os.path.dirname(settings['src'])
```

The db already lives in the canonical <plate>/measurements/ folder, so src must go one more level up to the plate. The old code only skipped the move here and left src pointing at the measurements folder, which made the read below build <plate>/measurements/measurements/measurements.db.

### lines 1155-1160

```python
plot_image_mask_overlay(file_path,
```

`normalize=True` used to be passed here. There is no such parameter: every call raised TypeError, the bare except below swallowed it, and this branch has been printing a failure instead of drawing a single overlay. Percentile normalisation is what it wanted and `percentiles` is how it is asked for.

## _resolve_plaque_model

### lines 1370-1379

```python
local_dir = download_models()
```

NO 'cp' SUBFOLDER. This path carried one for as long as the plaque module has existed, and nothing lives there: `download_models` writes to `resources/models` and returns that, and no `cp` directory is created anywhere. So the DEFAULT plaque model resolved to a file that does not exist.

It went unnoticed because the tests around it assert the suffix (`.endswith('.CP_model')`) rather than that the file is there -- a path is a string until something opens it, and the thing that opens it is Cellpose, several steps later.

## analyze_plaques

### lines 1451-1455

```python
if settings.get('well_detection'):
```

WELL DETECTION FIRST, when the images hold more than one well. This rewrites `src` to the folder of per-well crops, so everything below segmentation, counting, the results rows -- is per WELL rather than per image. With detection off, src is unchanged and the old one-field-per- image behaviour is exactly what runs.

### lines 1485-1491

```python
scale = _plaque_scale_for(filename, settings)
```

THE WELL IS THE RULER. A pixel area is a property of the microscope; the same plaque at two magnifications gives two numbers, and pooling them compares optics rather than biology. The well is a manufactured object of known size present in the image, so dividing by its measured diameter puts every area into mm^2 and makes plates from different scopes comparable -- which is the whole reason the detector runs.

### line 1513  _(unsure)_

```python
summary_df = pd.DataFrame(summary_data)
```

Convert lists to pandas DataFrames

### line 1518  _(unsure)_

```python
db_name = os.path.join(folder, 'plaques_analysis.db')
```

Save DataFrames to a SQLite database

## count_phenotypes

### lines 1543-1546

```python
df = _read_db(settings['src'], tables=['png_list'])[0]
```

_read_db's signature is (db_loc, tables) and it returns a LIST of DataFrames (one per requested table) — the previous call used a non-existent `loc=` kwarg and treated the result as a single DataFrame, so count_phenotypes crashed for every caller.

### line 1552

```python
grouped_unique_count = df.groupby(['plateID', 'rowID', 'columnID'])[settings['annotation_column']...
```

Count unique values in 'value' column, grouped by 'plateID', 'rowID', 'columnID'

### line 1559

```python
pivot_df = grouped_counts.pivot_table(index=['plateID', 'rowID', 'columnID'], columns='value', va...
```

Pivot the DataFrame so that unique values are columns and their counts are in the rows

### line 1562  _(unsure)_

```python
pivot_df.columns = [f"value_{int(col)}" for col in pivot_df.columns]
```

Flatten the multi-level columns

### line 1565

```python
pivot_df.index = pivot_df.index.map(lambda x: f"{x[0]}_{x[1]}_{x[2]}")
```

Reset the index so that plate, row, and column form a combined index

### lines 1568-1572

```python
output_dir = os.path.dirname(settings['src'])
```

Save the pivoted counts next to the measurements database. The previous revision first did os.makedirs(os.path.join('src', 'results')) — a hard-coded RELATIVE path whose value was discarded on the very next line, so its only effect was littering the caller's cwd with a stray ./src/results directory.

## compare_reads_to_scores

### lines 1609-1612

```python
save_paths = [None, None]
```

save_paths is declared with a None default but was indexed unconditionally below, so the documented minimal call raised TypeError. plot_line already treats save_path=None as "don't save", so normalise to the two-element form here.

### lines 1807-1813

```python
if 'row' in reads_df_temp.columns and 'rowID' not in reads_df_temp.columns:
```

The reads-side row fixup used to test for 'row' but rename 'row_name', a pandas no-op, so neither legacy spelling was ever repaired. The "canonical not already present" guard mirrors utils' alias table and is load-bearing: without it a frame carrying both spellings ends up with two 'rowID' columns and dies later with "cannot reindex on an axis with duplicate labels".

### lines 1828-1831

```python
raise ValueError("reads_csv and scores_csv must contain the same number of elements if reads_csv ...
```

This branch used to only print: control then fell through to calculate_well_read_fraction(reads_df) with reads_df never bound, so the validation message was followed by a confusing UnboundLocalError. Raise so the branch actually terminates.

### lines 1846-1849

```python
df = pd.merge(reads_col_df, scores_col_df, on='prc', validate='one_to_one')
```

one_to_one: both sides have been reduced to one row per well — calculate_grna_fraction_ratio unstacks a groupby(['prc','grna_name']) and calculate_well_score_fractions groups on the well columns — and each point plotted below is one well, so neither side may contribute two.

### lines 1854-1858

```python
df = pd.merge(df, df_emp, left_on='rowID', right_on='key', validate='many_to_one')
```

many_to_one: df holds one row per well and empirical_dict is keyed by ROW (the mixing ratio that row was seeded with), so every well in a row picks up the same expected fraction. Built from a dict, the right side is unique by construction; the left is not, and must not be — a row has as many wells as the plate has columns kept by the `column`/`value` filter.

### lines 1861-1868

```python
if isinstance(y_columns, str):
```

`if any in y_columns not in df.columns` was a chained comparison, i.e. `(any in y_columns) and (y_columns not in df.columns)`, which is False for every realistic input — the guard was dead and an unknown y column reached seaborn as a cryptic ValueError instead. plot_line's else-branch also accepts a scalar column name and a bare y *vector* (Series/array), so only list/tuple/str forms name columns — iterating a Series here would test its VALUES against df.columns and bail out on a perfectly good call.

## compare_reads_to_scores.calculate_well_score_fractions

### lines 1632-1636

```python
for _cls in ('class_0', 'class_1'):
```

unstack(fill_value=0) only materialises columns for class labels that occur SOMEWHERE in the frame, so a scores table where every object got the same call yields just one class column and the fractions below raised KeyError. Backfill (never reindex — that would drop unexpected label columns that pass through today).

### lines 1640-1643

```python
summary_df = pd.merge(prc_summary, well_counts, on=['plateID', 'rowID', 'columnID', 'prc'], how='...
```

one_to_one: both sides are groupby reductions of the same frame over the same four columns (well_counts adds the class only to unstack it back into columns), so each well appears exactly once in each and the fractions below divide row-aligned counts.

## compare_reads_to_scores.plot_line

### line 1685  _(unsure)_

```python
df = df.loc[natsorted(df.index, key=lambda x: df.loc[x, x_column])]
```

Sort the DataFrame based on the x_column

### line 1691  _(unsure)_

```python
if isinstance(y_columns, list):
```

Handle multiple y-columns, each as a separate line

### lines 1699-1701

```python
sns.lineplot(
```

One hue per group, from the fixed palette and no longer than the number of groups -- seaborn raises when a palette is longer than the hue levels it is given.

### line 1708  _(unsure)_

```python
sns.lineplot(
```

A single series is the claim by default.

### line 1714  _(unsure)_

```python
ax.set_xlabel(xlabel if xlabel else x_column)
```

Set axis labels and title

### line 1719  _(unsure)_

```python
sns.despine(ax=ax)
```

Remove top and right spines

### lines 1722-1727

```python
if group_column or isinstance(y_columns, list):
```

Ensure legend only appears when needed and place it to the right. The frame comes off -- the style draws no boxes -- but the 'Legend' title stays, because tests/test_cov_submodules_reads_vs_scores.py reads it back as the marker that this figure is the line plot and not the scatter beside it.

### line 1734  _(unsure)_

```python
if save_path:
```

Save the plot if a save path is provided

## compare_reads_to_scores.calculate_grna_fraction_ratio

### line 1751  _(unsure)_

```python
grouped = df[df['grna_name'].isin([grna1, grna2])] \
```

Filter relevant grna_names within each prc and group them

## compare_reads_to_scores.calculate_well_read_fraction

### lines 1785-1788

```python
df = pd.merge(df, grouped_df, on='prc', validate='many_to_one')
```

many_to_one: one gRNA row per well on the left, one well total on the right. The fraction below is count/total, so a duplicated well total would repeat every gRNA of that well and make the fractions sum to more than 1 without changing any single value.

## interpret_vision_model.generate_comparison_columns

### line 1959, trailing  _(unsure)_

```python
base_col_name = comp0_col.replace(comp0, '')
```

Base feature name without compartment prefix

### line 1961  _(unsure)_

```python
for prefix, prefix_columns in compartment_columns.items():
```

Look for matching columns in other compartments

### line 1963, trailing  _(unsure)_

```python
if prefix == comp0:
```

Skip same-compartment comparisons

### line 1965  _(unsure)_

```python
related_col = prefix + base_col_name
```

Check if related column exists in other compartment

### line 1971  _(unsure)_

```python
ratio = (
```

Calculate ratio and handle infinite or NaN values

### line 1979  _(unsure)_

```python
if related_cols:
```

Generate all-to-all comparisons

### line 1984  _(unsure)_

```python
comp1, comp2 = rel_col_1.split('_')[0], rel_col_2.split('_')[0]
```

Create a new column name for each pairwise comparison

### line 1988  _(unsure)_

```python
ratio = (
```

Calculate pairwise ratio and handle infinite or NaN values

## interpret_vision_model.group_feature_class

### lines 2009-2014

```python
feature_groups = [g if isinstance(g, str) else f'channel_{g}'
```

spacr settings identify channels by integer id ([0, 1, 2, 3]), but the feature columns spell them 'channel_<n>'. The groups are fed straight to re.search below, which raises "first argument must be string or compiled pattern" on an int, so the documented channels=[0,1,2,3] crashed. String groups are left untouched (they are deliberately treated as regex patterns).

### line 2032  _(unsure)_

```python
importance_sum = df.groupby(name)['importance'].sum().reset_index(name=f'{name}_importance_sum')
```

Create new DataFrame with summed importance for each compartment and channel

## interpret_vision_model.create_extended_radar_plot

### line 2045  _(unsure)_

```python
def create_extended_radar_plot(values, labels, title):
```

Function to create radar plot for individual and combined values

### line 2053, trailing  _(unsure)_

```python
values = list(values) + [values[0]]
```

Close the loop for radar chart

### lines 2059-2061

```python
ax.plot(angles, values, linewidth=1.2, linestyle='solid',
```

One series, so it is the claim and takes the highlight; the fill is the same hue at the one alpha the published figures use under a curve.

## interpret_vision_model.extract_compartment_channel

### line 2078  _(unsure)_

```python
compartment = feature_name.split('_')[0]
```

Identify compartment as the first part before an underscore

### line 2084  _(unsure)_

```python
channels = []
```

Identify channels based on substring presence

### line 2095  _(unsure)_

```python
if channels:
```

If multiple channels are found, join them with a '+'

### line 2099, trailing  _(unsure)_

```python
channel = 'morphology'
```

Use 'morphology' if no channel identifier is found

## interpret_vision_model.read_and_preprocess_data

### line 2122  _(unsure)_

```python
df['object_label'] = df['object_label'].str.replace('o', '')
```

Clean and align columns for merging

### lines 2132-2136

```python
if 'column' in scores_df.columns:
```

Ordered so the more specific 'column_name' wins, mirroring the row branch above where 'row_name' wins over 'row'. The old order let a junk 'column' column override a good 'column_name'; that was invisible while the merge below still keyed on 'column_name', but now silently merges to zero rows.

### line 2145  _(unsure)_

```python
df['object_label'] = df['object_label'].str.replace('o', '').astype(str)
```

Remove the 'o' prefix from 'object_label' in df, ensuring it is a string type

### lines 2151-2157

```python
if 'columnID' not in df.columns and 'column_name' in df.columns:
```

The merge below used to key on the legacy 'column_name', but io._read_and_merge_data normalises every spelling to 'columnID' before it returns, so the merge raised KeyError "['column_name'] not in index" against any real measurements.db. Key on 'columnID' (matching the alias fixup above and the spacr.ml twin) while still accepting the legacy spelling, which older CSVs and hand-built frames in the wild still carry.

### line 2161  _(unsure)_

```python
df[['plateID', 'rowID', 'columnID', 'fieldID', 'object_label']] = df[['plateID', 'rowID', 'column...
```

Ensure all join columns have the same data type in both DataFrames

### line 2165  _(unsure)_

```python
scores_df = scores_df[['plateID', 'rowID', 'columnID', 'fieldID', 'object_label', settings['score...
```

Select only the necessary columns from scores_df for merging

### lines 2168-2205

```python
try:
```

Now merge DataFrames. one_to_one, because both sides are one row per SCORED CROP and this key is that crop's identity:

``_read_and_merge_data`` groups every object table on ``prcfo`` and joins them on that index under ``validate='one_to_one'``, so ``df`` is one row per ``prcfo`` — i.e. one row per object per field. the scores CSV is written per crop PNG by ``apply_model_to_tar`` / ``utils.process_vision_results``, which is the same one row per object per field.

An earlier comment here claimed many_to_one was right because "the scores CSV holds one score per object", so a timelapse database could legitimately match every frame of an object to its single score. That is wrong in both halves. A timelapse crop is ``plate_well_field_time_object`` and gets its own row in the scores CSV, so on a timelapse the RIGHT side repeats exactly as much as the left does — many_to_one does not tolerate the timelapse, it crashes on it with pandas' "Merge keys are not unique in right dataset", which names neither the timepoint nor the file. And a score that arrived per frame is not a per-object score to spread over frames.

The key here carries no timepoint, and this legacy copy has no way to add one (the newer explainer in spacr.ml does: it appends the timepoint column to the key and raises TimelapseKeyMismatch when only one side has it). So on a timelapse both sides repeat, the join would be a frames x frames fan-out per object, and one_to_one is what refuses it. It also refuses the two cases many_to_one let through silently: a measurements frame duplicated on the key, and a timelapse database scored by a non-timelapse run — where every frame would inherit one score and each object would enter the forest once per frame.

_merge_with_cardinality keeps pandas as the thing that detects the duplicate and reports it as a MergeCardinalityError naming the side, the key and the offending values. What it cannot know is WHY, so the timelapse — the one cause that is a property of the database rather than of a bad file — is named here when the frame actually has a timepoint column.

### lines 2230-2232

```python
X = schema.model_feature_frame(
```

Select measurements by schema role, not every numeric column. Object labels and acquisition provenance are numeric in many databases but must never be learned by the classifier.

## interpret_vision_model

### lines 2245-2253

```python
if settings['feature_importance'] or settings['permutation_importance'] or settings['shap']:
```

Step 1: Feature Importance using Random Forest

The outer guard used to read `feature_importance or feature_importance` — the same key OR'd with itself — so the forest was never fitted unless feature importance was explicitly requested. Permutation importance then hit UnboundLocalError on `model`, and SHAP on `feature_importance_df`, even though the docstring documents the three explainers as independent toggles. The forest and the importance frame are shared by all three; only the reporting, grouping and output writes belong to feature_importance itself.

### lines 2266-2269

```python
with figure_style(theme_target()):
```

Plot Feature Importance. ONE SERIES, SO IT IS GREY: the ranking is the claim and the bars carry it by length, so matplotlib's default saturated blue was decoration on every bar at once.

### line 2288  _(unsure)_

```python
if settings['permutation_importance']:
```

Step 2: Permutation Importance

### line 2296  _(unsure)_

```python
with figure_style(theme_target()):
```

Plot Permutation Importance

### line 2310  _(unsure)_

```python
if settings['shap']:
```

Step 3: SHAP Analysis

### line 2320  _(unsure)_

```python
model = RandomForestClassifier(random_state=42, n_jobs=settings['n_jobs'])
```

Refit the model on this subset of features

### line 2324  _(unsure)_

```python
if settings['shap_sample']:
```

Sample a smaller subset of rows to speed up SHAP

### lines 2326-2329

```python
sample = max(1, min(int(len(X_top) / 100), len(X_top)))
```

int(len/100) floors to 0 for any experiment with fewer than 100 objects, which handed shap an empty background AND an empty matrix to explain -> IndexError. Clamp to at least one row; for >=100 objects the clamp is a no-op.

### line 2335  _(unsure)_

```python
explainer = shap.Explainer(model.predict, X_sample)
```

Initialize SHAP explainer with the same subset of features

### line 2342  _(unsure)_

```python
shap_df = pd.DataFrame(shap_values.values, columns=X_sample.columns)
```

Convert SHAP values to a DataFrame for easier manipulation

### line 2345  _(unsure)_

```python
shap_df.columns = pd.MultiIndex.from_tuples(
```

Apply the function to create MultiIndex columns with compartment and channel

## analyze_endodyogeny._calculate_volume_bins

### lines 2432-2436

```python
edge_rtol = 1e-12
```

Python/NumPy versions can evaluate the vectorised ``area ** 1.5`` one ULP below the mathematically identical scalar bin edge.  Treat only machine-precision neighbours as the same boundary; otherwise an object exactly on a doubling edge changes bins across supported Python versions.

### lines 2439-2441

```python
if bins[-1] <= max_volume or np.isclose(
```

Ensure the last edge exceeds the data maximum so nothing is clipped. ``isclose`` matters when vectorised and scalar exponentiation land on opposite sides of the same representable edge.

### lines 2452-2454

```python
cut_values = df[volume_column].copy()
```

Snap numerical neighbours to the authoritative scalar edges before the left-closed cut.  Keep the reported volume untouched; this copy exists only to make boundary membership reproducible.

### line 2464  _(unsure)_

```python
df[bin_column] = pd.cut(
```

Cut into bins; values outside the range become NaN

### line 2472

```python
df['bin_index'] = pd.to_numeric(df['bin_index'], errors='coerce')
```

Coerce to float so NaN is preserved (int would raise)

### line 2475  _(unsure)_

```python
before = len(df)
```

Drop rows that fell outside all bins

### line 2489  _(unsure)_

```python
index_to_label = {i + 1: label for i, label in enumerate(capped_labels)}
```

Build the authoritative ordered mapping and apply it

### line 2493

```python
ordered_categories = [index_to_label[k] for k in sorted(index_to_label.keys())]
```

Convert to an ordered categorical so order is never ambiguous

## analyze_endodyogeny

### lines 2527-2529

```python
min_area_bin = settings['min_area_bin']
```

Local, not settings['min_area_bin']: the um scaling below is an internal unit change, and writing it back would mutate the caller's dict (and make a second call with the same dict scale the threshold twice).

### lines 2556-2558

```python
if settings['group_column'] not in df.columns:
```

This guard used to sit AFTER the dropna below. pandas' dropna raises a bare KeyError for exactly the condition tested here, so the informative "Available columns" message was unreachable dead code.

### lines 2582-2583  _(unsure)_

```python
df[bin_column] = df[bin_column].cat.remove_unused_categories()
```

Remove categories that have zero observations across the entire dataset so the contingency table passed to chi2_contingency has no all-zero columns

### line 2593  _(unsure)_

```python
legend_labels = [
```

Use the authoritative ordered list (no sorting, no dtype check needed)

## _compose_field_keys

### lines 2624-2626  _(unsure)_

```python
def _compose_field_keys(df, time_column, source):
```

The field key both object assays are built on

### lines 2669-2673

```python
composed[identity] = None
```

None marks "already tried, already recorded": a failing identity must not be re-composed once per row it appears on, and the raise below needs one entry per distinct identity to count. compose_prcf never returns None, so the sentinel cannot collide with a real key.

## _ensure_field_key

### lines 2746-2749

```python
if 'prcf' in df.columns and time_column is None:
```

Nothing to build and nothing to repair: a table that already carries a prcf and has no time axis is returned untouched, without composing a key it does not need. Composition can raise (see below), and it must only be able to do so for a frame this function is actually keying.

### lines 2753-2762

```python
keyed = _compose_field_keys(df, time_column, source)
```

Composed through schema, not concatenated, so the key it builds is one schema.parse_prcf can read back. Concatenation could not promise that: the timepoint went in exactly as the column stored it, and a table whose timeID is the integer 1 rather than 't1' produced 'plate1_r1_c1_f1_1' a key whose last element parse_prcf sees as a broken FIELD id, so every reader of that key raised instead of finding the timepoint. compose_prcf normalises each element ('1' -> 't1') and is a no-op on the canonical form utils._map_wells writes, which is what is already in the database. It raises on a plate id containing '_' -- deliberately, because such a prcf cannot be split back into its parts by anything downstream.

### lines 2773-2777

```python
blind = _compose_field_keys(df, None, source)
```

Either spelling of the time-blind key marks a stale row: the composed one for a table written by today's writers, the concatenated one for a table written by an older spacr (or by a foreign importer whose row ids are not canonical, where the two differ). The legacy spelling is only ever compared against, never written.

### lines 2789-2791

```python
df.loc[stale, 'prcf'] = keyed[stale].to_numpy()
```

.to_numpy(): assign positionally. `keyed` shares df's index, but a frame whose index carries repeated labels would make .loc align on the label and write the wrong rows.

## _set_analyze_replication_defaults

### lines 2796-2798  _(unsure)_

```python
def _set_analyze_replication_defaults(settings):
```

Replication assay (Toxoplasma endodyogeny) — parasites per vacuole

## _assign_vacuole_ids

### lines 3061-3062

```python
next_id = clusters.max() + 1 if len(clusters) else 1
```

Objects with a non-finite centroid cannot be clustered; each becomes its own vacuole rather than silently joining cluster 0 together.

## _replication_well_distribution

### lines 3127-3128

```python
row[f'frac_{suffix}'] = (count / n_vacuoles) if n_vacuoles else 0.0
```

No vacuoles is a real, reportable state (an uninfected well), not a divide-by-zero: every fraction is 0.0 and n_vacuoles says why.

## _replication_compare_conditions

### lines 3291-3293

```python
statistic = len(left_ladder) * len(right_ladder) / 2.0
```

SciPy 1.17 reports NaN when the pooled ranked outcome has zero variance. The two distributions are exactly identical, so the defined no-difference result is U=n1*n2/2, p=1.

### lines 3299-3300  _(unsure)_

```python
rank_biserial = 2.0 * statistic / (len(left_ladder) * len(right_ladder)) - 1.0
```

Rank-biserial correlation: +1 means every vacuole in group1 is further along the ladder than every vacuole in group2.

### lines 3305-3307

```python
pair_counts = counts.loc[[group1, group2]]
```

Dropping the buckets neither group occupies is what keeps scipy from rejecting the table over an all-zero column; every group has at least one vacuole, so no row can be empty.

## analyze_replication

### lines 3509-3512

```python
apply_defaults = getattr(settings_module, 'set_analyze_replication_defaults',
```

spacr.settings owns every pipeline's defaults and wins wherever it defines one; the local copy runs afterwards purely as a gap-filler, so the assay is callable before the GUI knobs are registered. Both use setdefault, so running settings.py first makes its values authoritative.

### lines 3527-3530

```python
parasite_frames, cell_frames = [], []
```

read one row per segmented parasite

Deliberately NOT _read_and_merge_data: that helper collapses the pathogen table onto the host cell (prcfo is built from cell_id), which destroys the per-vacuole identity this assay is built on.

### lines 3536-3538

```python
frame['plateID'] = f'plate{index + 1}'
```

prcf carries the ORIGINAL plate name and is what the vacuole id is built from, so relabelling plateID alone would let two plates that share a well/field/cell collapse into one vacuole.

### lines 3560-3562

```python
df = _ensure_field_key(df, source=f"table '{parasite_table}'",
```

The timepoint is part of this key on a timelapse; see _ensure_field_key. Every vacuole id below is built from prcf, so a time-blind one merges the same host cell across all of its frames into a single vacuole.

### lines 3577-3578

```python
host = pd.to_numeric(df['cell_id'], errors='coerce')
```

0 / NaN means the object overlapped no host cell — an extracellular parasite, which has no vacuole and cannot enter a replication count.

### line 3664  _(unsure)_

```python
seed_wells = None
```

wells that hold host cells but no vacuoles

### lines 3751-3752  _(unsure)_

```python
plt.close(well_fig)
```

The figures stay usable (savefig works on a closed figure); closing them keeps a batch run over many plates from accumulating open figures.

## _set_analyze_invasion_defaults

### lines 3759-3761  _(unsure)_

```python
def _set_analyze_invasion_defaults(settings):
```

Invasion assay (Toxoplasma) — two-colour outside/inside stain

## _resolve_invasion_background_column

### lines 3978-3980

```python
for suffix in (
```

Measurement columns are canonicalised on database read. Keep the legacy spelling as a fallback for direct DataFrame callers and old databases that have not passed through that normalisation yet.

## _bimodality_coefficient

### lines 4034-4035  _(unsure)_

```python
return 0.0
```

One value repeated is one population, but skew/kurtosis are 0/0 there; say "no evidence of two populations" explicitly.

## _invasion_threshold

### lines 4117-4120

```python
except (ValueError, RuntimeError, FloatingPointError,
```

Depending on NumPy/skimage versions an unrepresentable histogram range is rejected as ValueError, overflows during bin construction, or reaches the final integer-bin lookup as IndexError.  All three mean the sample cannot support a threshold; none should abort the rest of the well.

## _invasion_field_thresholds

### lines 4279-4280  _(unsure)_

```python
reference = control if np.isfinite(control) else automatic
```

A control-derived cut is the honest negative distribution, so it is what an automatic cut should be judged against when it exists.

## _invasion_classify

### lines 4342-4348

```python
merged = df.merge(fields[columns], on='prcf', how='left')
```

The contract is many_to_one: many parasites per field, one threshold row per field. It is enforced by _report_fan_out below rather than by validate='many_to_one', deliberately — pandas would raise its generic MergeError *before* the check ran, and io's helper says which assay failed, how many rows went in and came out, and what to do about the duplicated table. The left side is emphatically NOT unique: one row per parasite is the whole point of this frame.

### lines 4350-4353

```python
_report_fan_out(df, merged, ['prcf'], left_name='parasite',
```

``fields`` comes out of a groupby on prcf so it holds one row per key and this join cannot grow. Checked anyway: if a caller ever hands in a field table assembled some other way, a duplicated prcf would silently duplicate every parasite and inflate n_total.

### lines 4379-4380

```python
df['is_outside_low_threshold'] = np.where(usable, outside_low, np.nan)
```

The two sensitivity columns exist so a reader can see how much of the reported efficiency is the threshold rather than the biology.

## _invasion_well_table

### lines 4517-4519

```python
inflation = (row['invasion_efficiency_high_threshold']
```

Only the upward move counts. Raising the threshold turns attached into invaded and inflates the efficiency; lowering it can only do the opposite, which is the direction that never invents a result.

## _invasion_stacked_bars

### lines 4763-4765

```python
with figure_style(theme_target()):
```

Nothing was classifiable anywhere — no threshold existed. Say that on the axes rather than dying inside pandas' bar plot, because the unclassified count in the well table is the real answer here.

## _invasion_threshold_panels

### lines 4835-4840

```python
if cmap in (None, 'viridis'):
```

THE HISTOGRAM FILL IS FURNITURE, NOT A CLAIM: it is the same distribution in every panel, so it takes the style's one fill colour and `cmap` is only consulted when a caller has deliberately asked for something else. It used to be the middle of viridis -- a saturated green against which the crimson threshold line was the only louder thing on the panel.

### lines 4875-4878

```python
if np.isfinite(threshold):
```

THE APPLIED THRESHOLD IS THE CLAIM AND THE REFERENCE IS A

REFERENCE. They were crimson at 1.5 and steelblue at 1.2, two equally loud lines, so the panel did not say which one the classification actually used.

### lines 4884-4885  _(unsure)_

```python
axis.plot([], [], color=ROLES['reference'], linestyle=(0, (4, 3)),
```

A proxy handle, so the grey dashed reference is named in the legend without being drawn twice.

## analyze_invasion

### lines 5025-5028

```python
apply_defaults = getattr(settings_module, 'set_analyze_invasion_defaults',
```

spacr.settings owns every pipeline's defaults and wins wherever it defines one; the local copy runs afterwards purely as a gap-filler, so the assay is callable before the GUI knobs are registered. Both use setdefault, so running settings.py first makes its values authoritative.

### lines 5050-5054

```python
parasite_frames, cell_frames = [], []
```

read one row per segmented parasite

Deliberately NOT _read_and_merge_data: that helper collapses the pathogen table onto the host cell (prcfo is built from cell_id), which would sum several parasites' outside-stain intensities into one row and destroy the per-parasite call this assay exists to make.

### lines 5060-5062

```python
frame['plateID'] = f'plate{index + 1}'
```

prcf carries the ORIGINAL plate name and is what the per-field threshold is keyed on, so relabelling plateID alone would let two plates that share a well/field pool their fields.

### lines 5084-5086

```python
df = _ensure_field_key(df, source=f"table '{parasite_table}'",
```

The timepoint is part of this key on a timelapse; see _ensure_field_key. One outside-stain threshold is computed per prcf, so a time-blind one cuts every frame of a field on a single number.

### lines 5136-5140

```python
control_mask = _invasion_control_mask(df, settings['stain_baseline_wells'])
```

staining controls

Split them off before conditions are annotated: a no-primary or no-permeabilisation control is a staining control, not an experimental condition, so it has no entry in the well maps and must not appear in any efficiency.

### lines 5187-5188  _(unsure)_

```python
df['no_host_cell'] = False
```

No cell mask at all: nothing is known about host association, so nothing is forced and the stain decides every call.

### lines 5230-5232

```python
field_counts[name] = field_counts.get(name, 0)
```

Categorical value_counts currently emits every declared class, and ``get`` retains the defensive zero for a future plain-string input without adding a branch that the categorical path cannot take.

### lines 5234-5239

```python
fields = fields.merge(
```

one_to_one: `fields` is one row per prcf (built by a groupby in _invasion_field_thresholds) and field_counts is a value_counts over the same key unstacked into columns, so it is too. This is the row the invasion efficiency is computed on and reported per field, so a second row for a field would report that field twice with the same numbers and weight it double in every per-well and per-group mean below.

### line 5254  _(unsure)_

```python
seed_wells = None
```

wells that hold host cells but no parasites

### lines 5361-5362  _(unsure)_

```python
for figure in (well_fig, group_fig, threshold_fig):
```

The figures stay usable (savefig works on a closed figure); closing them keeps a batch run over many plates from accumulating open figures.

## analyze_class_proportion

### line 5391  _(unsure)_

```python
if not isinstance(settings['src'], list):
```

Process data

### lines 5426-5441

```python
_missing = int(df[settings['class_column']].isna().sum())
```

NaN -> class 0, and SAY SO. The fill is a deliberate choice, pinned by tests/test_cov_submodules_class_proportion.py: an object the classifier did not call counts as the negative class rather than vanishing from the contingency table.

It is the right answer when the column is a CLASSIFIER OUTPUT, where every object was scored and NaN means "below threshold". It is the wrong answer when the column is an ANNOTATION, where NaN means "nobody looked": annotate 500 of 40,000 cells as classes 1 and 2 and the other 39,500 arrive as a class-0 majority that decides the chi-squared on its own.

Not flipped here, because that would break the case it is right for. Reported instead, so the second case stops being silent -- a user who reads "39500 of 40000 objects have no value" knows at once which situation they are in.

### line 5452  _(unsure)_

```python
results_df, pairwise_results, fig = plot_proportion_stacked_bars(settings, df, settings['group_co...
```

Perform chi-squared test and plot

### line 5498  _(unsure)_

```python
if settings['save']:
```

Save additional results

## generate_score_heatmap.group_cv_score

### lines 5543-5546

```python
df = read_table(csv)
```

`read_table` canonicalises, so every spelling of the column arrives here as `columnID`. There used to be an `elif 'column'` fallback under this; it was unreachable rather than load-bearing, and an unreachable branch is one no test can ever justify.

## generate_score_heatmap.calculate_fraction_mixed_condition

### lines 5578-5580

```python
df = read_table(csv)
```

145, AND IT IS THE FIX FOR THE NOTE BELOW: canonicalisation is what makes the two spellings one, so the half-finished rename cannot bite again.

### lines 5582-5587

```python
if 'columnID' not in df.columns and 'column_name' in df.columns:
```

This helper was left half-way through the column_name -> columnID rename: it grouped by 'columnID' but filtered and merged on 'column_name', a key the grouped frame can never carry, so every call died with KeyError('column_name'). Key on 'columnID' like every sibling helper here, accepting the legacy spelling that older reads CSVs still use.

### lines 5591-5595

```python
if plate is not None:
```

`plate` is a plate NUMBER, not a column name, so `plate not in df.columns` was always True and the CSV's own plateID was always overwritten -- stamping the literal "plateNone" when plate is None. The prc keys then matched nothing downstream and the heatmap came back empty with no error. Guard the way both sibling helpers do.

### lines 5601-5605

```python
merged_df = pd.merge(df, grouped_df, on=['plateID', 'rowID', 'columnID'], validate='many_to_one')
```

many_to_one: the left side is one row per control sgRNA per well, the right side that well's total. 'fraction' below is count/total, so a duplicated total would repeat each sgRNA row and the fractions of a well would no longer sum to 1 — with every individual value still looking perfectly reasonable.

## generate_score_heatmap.plot_multi_channel_heatmap

### lines 5619-5622

```python
df = df.copy()
```

Copy first: this assignment used to mutate the CALLER's frame, so the temporary sort column survived in merged_df (the drop below only affects the local slice) and leaked into the returned frame, the saved *_data.csv and the MAE table as a bogus channel.

### line 5632  _(unsure)_

```python
df = df.drop('row_num', axis=1)
```

Drop temporary 'row_num' column after sorting

### line 5641  _(unsure)_

```python
heatmap_data = df.select_dtypes(include=[float, int])
```

Extract only numeric data for the heatmap

### line 5644  _(unsure)_

```python
with figure_style(theme_target(), frame='box'):
```

Plot heatmap with square boxes and no annotations

### lines 5658-5659

```python
rotate_ticks(axis, 45)
```

Long channel names rotate 45 and anchor right, as every categorical axis in the style does.

## generate_score_heatmap.combine_classification_scores

### line 5691  _(unsure)_

```python
if isinstance(folders, str):
```

Ensure `folders` is a list

### line 5695, trailing  _(unsure)_

```python
ls = []
```

Initialize ls to store found CSV file paths

### line 5697  _(unsure)_

```python
for folder in folders:
```

Iterate over the provided folders

### line 5700, trailing  _(unsure)_

```python
for sub_folder in sub_folders:
```

Iterate through sub-folders

### line 5701, trailing  _(unsure)_

```python
path = os.path.join(folder, sub_folder)
```

Join the full path

### line 5703, trailing  _(unsure)_

```python
if os.path.isdir(path):
```

Check if it’s a directory

### line 5705, trailing  _(unsure)_

```python
if os.path.exists(csv):
```

If CSV exists, add to list

### line 5710  _(unsure)_

```python
combined_df = None
```

Initialize combined DataFrame

### line 5714  _(unsure)_

```python
for csv_file in ls:
```

Loop through all collected CSV files and process them

### line 5716, trailing

```python
df = read_table(csv_file)
```

145: canonical column names

### line 5720

```python
grouped_df = df.groupby(['plateID', 'rowID', 'columnID'])[data_column].mean().reset_index()
```

Group the data by 'plateID', 'rowID', and 'columnID'

### line 5728  _(unsure)_

```python
if combined_df is None:
```

Merge into the combined DataFrame

### lines 5732-5736

```python
combined_df = pd.merge(combined_df, grouped_df, on=['plateID', 'rowID', 'columnID'], how='outer',...
```

one_to_one: each folder contributes a groupby mean, one row per well, and the accumulator stays one row per well because every merge into it is one_to_one. This frame is a well x channel matrix -- the heatmap plots it as one -- so a second row for a well would draw that well twice.

## generate_score_heatmap.calculate_mae

### line 5743  _(unsure)_

```python
channels = df.drop(columns=['fraction', 'prc']).select_dtypes(include=[float, int])
```

Extract numeric columns excluding 'fraction' and 'prc'

### line 5754  _(unsure)_

```python
mae_df = pd.DataFrame(mae_data)
```

Convert the list of dictionaries to a DataFrame

## generate_score_heatmap

### lines 5762-5769

```python
merged_df = pd.merge(fraction_df, result_df, on=['prc'], validate='many_to_one')
```

many_to_one on both joins below. The right sides are groupby reductions (one row per well) and that is the half that must hold: a well appearing twice in the score matrix or the CV table would repeat that well in the heatmap and count it twice in the MAE. The LEFT side is not asserted unique on purpose — fraction_df is the reads CSV filtered to one gRNA and a CSV that lists that gRNA twice for a well (two sequencing runs, say) is a legitimate input here; it stays two rows, visibly, instead of turning the whole call into a MergeError.

### lines 5776-5782

```python
if 'row_num' in merged_df.columns:
```

The guard used to test for 'row_number' while the helper adds 'row_num', so it never fired for the column it meant to drop and would KeyError on a frame that genuinely carries a 'row_number' data column. With the copy in the helper this is now a no-op kept as cheap defence. The matching mae_df guard was deleted: calculate_mae only ever emits Channel/MAE/Row, so it was dead and, if it had ever fired, would have dropped a differently-named column.

## post_regression_analysis._analyze_and_visualize_grna_correlation

### line 5808  _(unsure)_

```python
filtered_df = df[df['grna'].isin(grna_list)]
```

Filter the DataFrame to include only rows with gRNAs in the list

### line 5814  _(unsure)_

```python
correlation_matrix = pivot_df.corr()
```

Compute the correlation matrix

### lines 5821-5825

```python
with figure_style(theme_target(), frame='box'):
```

Visualize the correlation matrix as a heatmap. A CORRELATION IS SIGNED, which is the one case the style allows a diverging map for, so coolwarm stays and is centred on zero -- it was not, so an all-positive matrix came out red end to end and looked like a finding.

## post_regression_analysis._compute_effect_sizes

### line 5848  _(unsure)_

```python
corr_matrix = correlation_matrix.copy()
```

Ensure the matrix is symmetric and normalize values to 0-1

### line 5852  _(unsure)_

```python
effect_sizes = pd.Series(0.0, index=corr_matrix.index)
```

Initialize the effect sizes with dtype float

### line 5859  _(unsure)_

```python
for grna in corr_matrix.index:
```

Propagate the effect sizes

### line 5862  _(unsure)_

```python
effect_sizes[grna] = np.dot(corr_matrix.loc[grna], effect_sizes) / np.sum(corr_matrix.loc[grna])
```

Weighted sum of correlations with the fixed gRNAs

### lines 5869-5875

```python
with figure_style(theme_target()):
```

Visualization. GREY BARS, AND THE ANCHORS COLOURED: `hue` was the gRNA name and `palette='viridis'` gave every bar its own hue, so a 40-guide panel was a 40-colour ramp that encoded nothing the x axis did not already say. The gRNAs whose effect was FIXED by `grna_dict` -- the anchors the rest were propagated from -- are the ones a reader has to be able to pick out, so those are the coloured minority.

### lines 5885-5888

```python
saturation=1,
```

saturation=1: seaborn desaturates a bar fill to 0.75 by default, so the palette's #2E77BC reached the canvas as #4076AA. A fixed hue that arrives as a different hue is not a fixed hue.

## post_regression_analysis

### line 5912  _(unsure)_

```python
df = pd.read_csv(csv_file)
```

Load the data

### line 5915  _(unsure)_

```python
correlation_matrix = _analyze_and_visualize_grna_correlation(df, grna_list, save_folder, save)
```

Perform analysis

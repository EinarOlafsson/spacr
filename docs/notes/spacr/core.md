# Notes from `spacr/core.py`

Prose lifted out of `spacr/core.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [display](#display) (1 entry)
- [Module level](#module-level) (4 entries)
- [preprocess_generate_masks](#preprocess_generate_masks) (26 entries)
- [preprocess_generate_masks_timelapse](#preprocess_generate_masks_timelapse) (1 entry)
- [generate_image_umap](#generate_image_umap) (31 entries)
- [reducer_hyperparameter_search](#reducer_hyperparameter_search) (13 entries)
- [generate_screen_graphs](#generate_screen_graphs) (9 entries)

## display

### lines 70-73

```python
def display(*args, **kwargs):
```

IPython may be mid-init (partially imported by another thread) — use a no-op fallback so importing this module never blocks. spaCR only calls display() from notebook contexts anyway; the Qt GUI ignores it.

## Module level

### lines 79-81

```python
from .errors import RunLedger, raise_if_strict
```

Fail-loud accounting: per-folder / per-example failures are recorded on a RunLedger and reported in one block at the end of the run, and setup failures escalate to ConfigurationError under SPACR_STRICT_ERRORS.

### lines 84-85  _(unsure)_

```python
from . import artifacts as artifact_status
```

One run id on every log line and every artifact, one seed reaching every RNG, one on_error policy at each batch boundary. See spacr.runctx.

### line 88, trailing  _(unsure)_

```python
from .plot import save_figure
```

every kept figure goes through the format/DPI preference

### lines 92-94

```python
from .figures.style import figure_style, theme_target
```

THE HOUSE STYLE (136). `figures.style` imports matplotlib only inside its own functions, so naming it here costs nothing at import time.

## preprocess_generate_masks

### lines 215-218

```python
if settings.get('dry_run', False):
```

dry_run comes FIRST, before the local imports below: .object and .io pull in cellpose and the model machinery, which is exactly the cost a validate-only run exists to avoid. Nothing is read, written or loaded past this point when dry_run is set.

### lines 234-235

```python
reset_cellpose_model_reports()
```

A new run gets to state its model choice again; within one run each notice is printed once per object type rather than once per field.

### lines 238-240

```python
if 'src' in settings:
```

These previously *constructed* a ValueError without raising it (and then returned None), silently swallowing bad input despite the docstring promising a raise. Raise for real.

### lines 249-257

```python
if settings.get('pipeline_style', 'v1') == 'v2':
```

v2 streaming pipeline — OPT-IN flow that skips the rename/split/npz/npy multi-copy chain and goes straight from originals → merged/stack_<field>.npy with masks appended in-place. Roughly 60-80% less disk than v1 on typical plates. NOTE: v1 remains the DEFAULT — v2 does not yet reproduce v1's channel/stack/mask_stack folder layout (downstream tools + the e2e suite depend on it) and still parses channels 1-indexed on real CellVoyager data. Enable v2 explicitly with pipeline_style='v2'. See spacr.pipeline_v2 for design notes.

### lines 259-275

```python
settings = set_default_settings_preprocess_generate_masks(settings)
```

THE DEFAULTS, BEFORE ANYTHING READS THE DICT. This branch returns at the end, and `set_default_settings_preprocess_generate_masks` is only called further down inside the per-source loop -- which v2 never reaches. So every `settings.get(key, fallback)` below was answering with its own inline fallback rather than the declared default.

One of them changes segmentation: `cell_flow_threshold` is declared 1.0 and the fallback here was 0.4, and it goes straight to `model.eval(flow_threshold=...)`. Cellpose's remove_bad_flow_masks drops a mask whose flow error exceeds the threshold, so on a field with per-object flow errors {0.0, 0.12, 0.30, 0.75} the v1 pipeline keeps four cells and this branch kept three -- same plate, same settings dict, same weights.

Both helpers are setdefault-only and idempotent, so the later call on the v1 path is unaffected.

2026-09-19: the declared default is 0.4 now (428, GitHub #123), so the `0.4` fallback in `settings.get('cell_flow_threshold', 0.4)` and the declared default agree again. The fallback is still never reached, because the defaults are filled first.

### lines 298-300

```python
diameter=_eval_diameter(
```

COERCED, like every other numeric on this call. A diameter typed into the GUI or read from a CSV is a string, and Cellpose compares it with `> 0`.

### lines 319-321

```python
if settings.get('consolidate', False):
```

settings defaults (incl. 'consolidate') are only applied further down, inside the per-source loop; read defensively here so a settings dict without the key doesn't raise KeyError before that point.

### lines 335-337

```python
source_folders = settings['src']
```

Input validation admits only str/list, normalize_src_path preserves that contract, and the str arm above finishes the conversion.  Use the list directly: a second type check could only skip the run silently.

### lines 339-340

```python
ledger = RunLedger('preprocess_generate_masks')
```

One ledger for the whole invocation: a run over four plates that only managed three must not report as if it did four.

### lines 342-345

```python
module_key = 'timelapse' if settings.get('timelapse') else 'mask'
```

One run: one id on every log line and on every artifact this run registers (so the two can be joined), one seed reaching numpy / random / torch / cellpose, and one on_error policy honoured at the plate boundary below. See spacr.runctx.

### lines 349-353

```python
for attempt in run.policy.attempts_for(source_folder,
```

on_error, at the plate boundary. stop (default) lets the failure out and the run ends here; skip records the plate on the ledger and in run.policy.skips and moves to the next one; retry re-attempts this plate with a backoff and then behaves like stop. See spacr.runctx.

### lines 377-380

```python
print(f"Error: Tried to convert image files and image file name metadata without regex but failed.")
```

Category B: no file was renamed, so every step below would operate on an empty/unrecognised folder. Historically this printed and returned None, which reads exactly like success.

### line 322, 2026-09-19, the regex conversion no longer falls back

```python
except Exception as e:
    refusal = (
```

A failure in `convert_separate_files_to_yokogawa` used to be caught with a bare `except Exception:` and answered by running `convert_to_yokogawa` on the same folder, without printing why. That conversion does not read the regex. It gives every file the next free well in file order, so the plate's own wells (B03, C07) came out as A01, A02 and so on, one channel file per well, with only `rename_log.csv` to say so. With the sidecar cause removed (item 429), any other failure still reached it: a matched `.nd2` or a damaged TIFF that `tifffile` cannot read, two slices of different shapes, a full disk. It could also run over converted files the regex conversion had already written before it stopped, and convert those again as wells of their own.

There is no condition under which the fallback is provably safe. It never keeps the wells the file names carry, even when those names are not plate addresses, because the regex conversion groups a well's channels into one well and the fallback splits them. So the run now refuses: the ledger records the failure at stage `convert_metadata`, the error names the file (the converter now says which file and how many converted files it had written before it stopped), says that no fallback ran and why, and gives the two ways on: correct the file or the regex, or clear `custom_regex` so the wells are numbered deliberately.

The second way carries a condition the message now states, because the refusal leaves the plate part-converted: `convert_separate_files_to_yokogawa` writes each region as it goes and raises on the bad one, so the `plate*_*.tif` files already written stay in the folder and no `rename_log.csv` records them. `convert_to_yokogawa` reads every image in the folder, so clearing the regex and re-running over that folder converts those leftovers a second time, as wells of their own -- the renumbering this change exists to prevent, arrived at by following the advice. The message therefore says to move them out of the folder first. spaCR does not delete them itself: they are the user's images under a new name, and the converter cannot tell which of them this attempt wrote from which a previous good run did. `SPACR_STRICT_ERRORS` turns the refusal into a `ConfigurationError`, as for the other conversion failure. Pinned by `tests/test_a_failed_regex_conversion_never_renumbers_wells.py`.

### line 399, 2026-09-19, GitHub #124

```python
_check_archives_without_preprocessing(src)
```

With `preprocess` off, nothing below checked `masks/*.npz` before the segmenter opened them, so the #124 state (an archive a killed run cut short) still ended in `zipfile.BadZipFile` on this path after `spacr.io` learned to check them on the `preprocess` path. The check sets a damaged archive aside and stops with an error naming it, rather than normalising it again, because `preprocess` off is the user saying the normalised arrays already exist; the error says to turn `preprocess` on, which rebuilds the fields from `stack/`. It runs before the illumination resume below, whose `_normalized_npz_field_ids` opens every archive too. Reasons in `docs/notes/spacr/io.md`, "A re-run trusts nothing a killed run left".

### lines 409-411

```python
print('Error: At least one of the registered object channels must be defined')
```

Category B: with no object channel there is nothing to segment, so returning None here is indistinguishable from a successful run that produced no masks.

### lines 421-424

```python
if settings['timelapse']:
```

The bundled toxo_pv_lumen / toxo_cyto models were Cellpose-3 checkpoints and are gone: Cellpose 4 ships only cpsam, and their CPnet weights cannot load into its Transformer. This guard also never fired — it *constructed* a ValueError without raising it.

### lines 456-468

```python
os.makedirs(mask_src, exist_ok=True)
```

CREATE IT IF IT IS NOT THERE.

Only preprocess_img_data makes this folder, and

`preprocess` is exactly the box a user unticks when re-masking a plate that has already been measured. Delete masks/ first -- which is what re-masking means -- and the run died on a missing directory that it was about to fill anyway (issue #13). The reported workaround was to mkdir it by hand.

exist_ok, so the normal path where preprocessing just made it is unaffected.

### lines 473-477

```python
from .illumination import (
```

The normalized V1 NPZs are already on disk, so fitting or rewriting here would make the masks impossible to trace. Accept them only when the prior application record proves the same model, pipeline style, and exact field set completed.

### lines 576-579

```python
_load_and_concatenate_arrays(
```

resume (opt-in, default False): skip fields whose merged stack is already present and verified complete, so a crash at field 900 of 1000 does not cost the first 900. Validated rather than stat'ed — see spacr.resume.

### lines 587-602

```python
organelle_chann_dims={
```

THE SLOTS THIS RUN CONFIGURED, not every slot that can be named. 326 widened `ORGANELLE_ROLES` from four to 702, so this comprehension built a 701-entry dict of which all but a handful were None, on every mask run the same per-VOCABULARY shape 42417ea28 fixed one file over in the measure loop.

Nothing downstream loses anything:

`_load_and_concatenate_arrays` reads this with `extra_dims.get(role)` inside its own loop over the same roles, so an absent key and a key holding None are already the same answer. What changes is what lands in the settings record and the run manifest, where 701 nulls buried the slots a run actually used.

### lines 614-618

```python
merged_dir = os.path.join(src, 'merged')
```

Test mode plots every merged field. This used to take len() of the merged *path string*, i.e. a number that tracks how deeply the run folder is nested and has nothing to do with how many fields exist.

### lines 625-627

```python
plot_ledger = RunLedger('preprocess_generate_masks:overlay_plots')
```

A separate ledger: an overlay PDF that fails to render is cosmetic and must NOT brand the masks themselves as partial. It still gets accounted for.

### lines 647-650

```python
with plot_ledger.item(
```

Per example, not per batch: the old single try around the whole loop meant one unplottable field silently cancelled every remaining example.

### lines 683-685

```python
from .utils import cleanup_pipeline_folders
```

By default keep only merged/ (masks are embedded there + labels are in the database). keep_intermediate / keep_original_images opt out. The legacy delete_intermediate flag forces cleanup too.

### lines 695-696  _(unsure)_

```python
ledger.finalize()
```

Last thing on screen: a four-plate run that only completed three says so here, and the per-folder db carries the same verdict.

### lines 703-718

```python
try:
```

The `relationships` table -- which nucleus is in which cell, and so on. Rebuilt rather than topped up: the masks that define those relationships have just changed, so the previous answer is about objects that no longer exist.

Inside this loop and not after it. `db_path` is the loop variable, so a write placed after the loop would silently do one plate -- the last -- and leave every other plate in a multi-plate run without the table.

Never fatal, for the same reason the artifact registry above is not: masking succeeded, and a missing relationships table is rebuilt on demand by the Gate Editor anyway. Failing here would throw away hours of segmentation to protect a lookup that costs seconds.

### lines 727-736

```python
from .artifacts import register_run_outputs
```

Run completion hook: record what this run produced, and what it was produced from, in the project's artifact registry. strict=False — a registry that cannot be written is worth one printed line, never a lost run. See spacr/artifacts.py.

run_id is the same id every log line this run emitted carries, which is what makes "show me the log of the run that produced this file" answerable: spacr.runctx.read_run_log(artifact.run_id).

## preprocess_generate_masks_timelapse

### lines 775-776

```python
print("Timelapse module: settings['timelapse'] was False — forcing it "
```

Non-silent: the module *is* timelapse, so an incoming False (an old mask settings CSV, say) is overridden rather than quietly honoured.

## generate_image_umap

### lines 986-997

```python
if len(db_paths) > 1:
```

WARN ABOUT PLATE IDS THAT APPEAR IN MORE THAN ONE DATABASE

(instruction 109). This function has always accepted several sources and concatenated them, and has never checked that their plates are actually different plates. Two runs that both call a plate 'plate1' produce one key per object across both, so every per-well number computed downstream and every cluster the embedding shows -- is over two experiments at once, with nothing on screen to say so.

A warning rather than a refusal: unlike a fresh merge, this is an existing entry point with existing callers, and stopping a run that worked yesterday is a worse failure than telling the truth loudly. The source column below is what lets a user check the answer.

### lines 1013-1015

```python
pass
```

Never let the advisory check stop a run that would otherwise work -- a database missing a 'cell' table is a legitimate shape here, and this is only advice.

### lines 1017-1022

```python
from .io import open_crop_source, crop_refs_for_rows, CROP_REF_COLUMN
```

Where the thumbnails come from. 'png' (and 'auto' on any project that has a crop folder) reads the folder, exactly as before; 'merged' (and 'auto' with no folder) cuts each thumbnail out of merged/*.npy on demand, so the embedding can be drawn with no crop folder on disk. Either way the pixels arrive through spacr.crops, so a legacy folder is channel-corrected on load and the two sources show the same image.

### lines 1024-1027

```python
crop_object = 'cell'
```

'cell', not settings['visualize']: _read_and_join_tables anchors the join on the cell table, so every row's object_label IS a cell label. Cutting the nucleus plane with a cell label would return a different object or none -- for every point on the map.

### lines 1035-1036

```python
_validate_umap_source_db(db_path, tables,
```

Say which database is unusable, and why, instead of letting the join fail with a bare KeyError three modules away.

### lines 1039-1043

```python
df = _read_and_join_tables(db_path, table_names=tables,
```

require_crops=False: this embeds MEASUREMENTS. An object whose crop never wrote has valid measurements and belongs in the embedding it simply has no thumbnail to show when its point is hovered. Letting the png_list inner join drop it would silently shrink the embedding, which is a wrong number rather than a missing picture.

### lines 1046-1048

```python
df['_spacr_umap_db_path'] = db_path
```

Keep the exact update identities before correct_paths re-anchors png_path for display on this machine. These columns are removed before the result CSV/DataFrame leaves this function.

### lines 1050-1056

```python
df[UMAP_SOURCE_COLUMN] = _umap_source_label(settings['src'][i])
```

The SOURCE, as something a user can group and colour by

(instruction 109). The private column above is dropped before the result leaves this function; this one is not, because a merged embedding whose clusters turn out to be the source databases rather than biology is the single most important thing a multi-plate UMAP can show -- and it cannot show it if provenance never reaches the frame the user plots.

### lines 1062-1064

```python
df[CROP_REF_COLUMN] = crop_refs_for_rows(source, df,
```

No .copy(): _read_and_join_tables hands back a fresh frame that correct_paths already writes into, and copying a 200-column screen-sized frame to add one column is gigabytes for nothing.

### lines 1088-1091

```python
n_rows = min(int(settings['row_limit']), len(all_df))
```

row_limit is a cap, not a demand: asking for more rows than the screen contains used to abort the whole embedding with pandas' "Cannot take a larger sample than population", which names neither the setting nor the source.

### lines 1098-1101

```python
if CROP_REF_COLUMN in all_df.columns:
```

The handles are pulled off the frame here, after every row filter above and before the numeric preprocessing below, and the column is dropped in the same breath: it holds Python objects, so leaving it on the frame would put `<LazyCropPNG …>` into embedding_results.csv.

### lines 1108-1109  _(unsure)_

```python
print("No crop source and no 'png_path' column; plotting points only.")
```

No crop source and no png_path: the embedding still means something, the montage does not.

### line 1137  _(unsure)_

```python
numeric_data_df = pd.DataFrame(numeric_data)
```

Convert numeric_data back to a DataFrame to align with col_to_compare

### line 1140  _(unsure)_

```python
numeric_data_df = numeric_data_df.reset_index(drop=True)
```

Ensure numeric_data_df and col_to_compare are properly aligned

### line 1143  _(unsure)_

```python
numeric_data_df[settings['col_to_compare']] = col_to_compare
```

Assign the column back to numeric_data_df

### line 1146  _(unsure)_

```python
positive_control_df = numeric_data_df[numeric_data_df[settings['col_to_compare']] == settings['po...
```

Subset the dataframe based on specified column values for controls

### line 1155  _(unsure)_

```python
numeric_data = numeric_data_df.values
```

Convert numeric_data_df and control_numeric_data_df back to numpy arrays

### line 1162  _(unsure)_

```python
numeric_data = preprocess_data(
```

Apply the trained reducer to the entire dataset

### lines 1180-1183

```python
raise NotImplementedError(
```

Placeholder, never implemented. It used to `pass`, so the run fell through to `plot_embedding(embedding, ...)` with `embedding` unbound and died with an UnboundLocalError that pointed at the plotting code rather than at the setting the user turned on.

### line 1188  _(unsure)_

```python
else:
```

numeric_data, embedding, labels = generate_umap_from_images(image_paths, settings['n_neighbors'], settings['min_dist'], settings['metric'], settings['clustering'], settings['eps'], settings['min_samples'], settings['n_jobs'], settings['verbose'])

### line 1190  _(unsure)_

```python
numeric_data = preprocess_data(
```

Apply the trained reducer to the entire dataset

### lines 1211-1217

```python
keep = np.asarray(labels) != -1
```

Remove noise from the clusters (removes -1 labels from DBSCAN). The frame and the image paths must lose exactly the same rows: dropping points from the embedding alone left `labels` shorter than `all_df`, and the `all_df['cluster'] = labels` assignment at the end of this function then died with "Length of values (50) does not match length of index (60)" — i.e. this setting was unusable whenever DBSCAN called *some* (but not all) points noise.

### lines 1225-1227

```python
labels = np.ones(len(all_df), dtype=int)
```

Every point was noise (or the reducer returned no labels). Preserve the mutually-aligned rows, embedding and crop paths, and expose an explicit one-cluster fallback.

### lines 1232-1234

```python
records = []
```

Preserve the point → crop → original database-row identity for the Qt interactive explorer. The ordinary Matplotlib/PDF path simply ignores the attached payload.

### lines 1245-1248

```python
plot_labels = (
```

``color_by`` controls presentation only.  Keep algorithmic cluster labels as a separate data contract: exporting columnID (or any other metadata) under the name "cluster" made downstream cluster analysis silently analyse wells instead of the clusters the reducer found.

### lines 1279-1283

```python
'plot_labels': np.asarray(plot_labels),
```

What the STATIC figure was coloured by, which is not always the cluster label: `color_by` swaps in a metadata column. Carried separately from `labels` so a redraw of the finished figure reproduces the colours the run actually drew, while the explorer keeps clustering on the algorithmic labels.

### lines 1293-1297

```python
'settings': {key: value for key, value in settings.items()
```

The settings this embedding was produced with. The Qt figure settings window opens on these, so every Image UMAP knob it offers starts at the value the run used rather than at a package default the user never chose. Underscore-prefixed keys are internal plumbing (`_plot_theme`) and are not settings anyone edits.

### line 1320  _(unsure)_

```python
if settings['save_figure']:
```

Save figure as PDF if required

### line 1333

```python
all_df['cluster'] = cluster_labels
```

Export the clustering result, never the optional presentation labels.

### line 1342  _(unsure)_

```python
results_dir = os.path.join(settings['src'][0], 'results')
```

Save the results to a CSV file

### line 1363  _(unsure)_

```python
if return_fig:
```

(saving CSVs etc. unchanged)

## reducer_hyperparameter_search

### lines 1410-1412

```python
if not reduction_params:
```

Determine reduction method based on the keys in reduction_param. reduction_params is required: iterating None raised a TypeError from the generator expression below that never mentioned the argument's name.

### lines 1422-1424

```python
if wants_umap and wants_tsne:
```

The both-at-once check has to come FIRST. As a third `elif` after the UMAP branch it could never fire, so a mixed sweep silently ran as UMAP and every 'perplexity' value in it was ignored.

### lines 1432-1433

```python
raise ValueError(
```

Previously fell through with reduction_method unbound and died on the next line with an UnboundLocalError.

### lines 1439-1443

```python
if str(settings['reduction_method']).lower() not in ('umap', 'tsne'):
```

Validated here, once, before a single row is read: this check used to live inside the per-cell loop below, where it could never fire because the line under it had already overwritten reduction_method with 'umap' or 'tsne'. An unsupported method was therefore silently swapped out instead of reported.

### line 1480  _(unsure)_

```python
n_rows = min(int(settings['row_limit']), len(all_df))
```

Same cap semantics as generate_image_umap.

### line 1501  _(unsure)_

```python
clustering_params = []
```

Combine DBSCAN and KMeans parameters

### line 1515  _(unsure)_

```python
grid_rows = len(reduction_params)
```

Calculate the grid size

### lines 1522-1525

```python
with figure_style(theme_target()):
```

THE STYLE HAS TO BE ON BEFORE THE FIGURE EXISTS:

rcParams reach an artist when it is CREATED, so a context opened after `plt.subplots` would leave the spines, ticks and labels at the caller's globals.

### line 1529  _(unsure)_

```python
axs = np.atleast_1d(axs)
```

Make sure axs is always an array of axes

### line 1532  _(unsure)_

```python
for i, reduction_param in enumerate(reduction_params):
```

Iterate through the Cartesian product of reduction and clustering hyperparameters

### lines 1544-1546

```python
if settings['reduction_method'].lower() == 'umap':
```

Perform dimensionality reduction and clustering. The method is 'umap' or 'tsne' and nothing else — it is validated once, above, before any data is read.

### line 1558, trailing  _(unsure)_

```python
else:
```

'tsne'

### line 1568  _(unsure)_

```python
if settings['color_by']:
```

Plot the results

## generate_screen_graphs

### line 1649  _(unsure)_

```python
df = annotate_conditions(df, cells=settings['cells'], cell_loc=None, pathogens=settings['controls...
```

Annotate the data

### line 1652  _(unsure)_

```python
df['recruitment'] = _finite_ratio(
```

Calculate recruitment metric

### line 1657  _(unsure)_

```python
all_df = pd.concat([all_df, df], ignore_index=True)
```

Combine with the overall DataFrame

### line 1660  _(unsure)_

```python
plotter = spacrGraph(df,
```

Generate individual plot

### line 1675  _(unsure)_

```python
figs.append(fig)
```

Append to the lists

### line 1679  _(unsure)_

```python
plotter = spacrGraph(all_df,
```

Generate plot for the combined data (all_df)

### line 1697  _(unsure)_

```python
for i, fig in enumerate(figs):
```

Save figures and results

### line 1706  _(unsure)_

```python
dst = os.path.join(source, 'results')
```

Ensure the destination folder exists

### line 1711  _(unsure)_

```python
save_figure(fig, os.path.join(
```

Save the figure and results DataFrame

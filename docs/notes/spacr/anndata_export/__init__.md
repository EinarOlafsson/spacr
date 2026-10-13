# Notes from `spacr/anndata_export/__init__.py`

Prose lifted out of `spacr/anndata_export/__init__.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_read_frame](#_read_frame) (4 entries)
- [_attach_png_labels](#_attach_png_labels) (2 entries)
- [_label_columns](#_label_columns) (1 entry)
- [_build_obs](#_build_obs) (1 entry)
- [_apply_nan_policy](#_apply_nan_policy) (4 entries)
- [_align_embedding](#_align_embedding) (1 entry)
- [build_anndata](#build_anndata) (4 entries)
- [export_anndata](#export_anndata) (2 entries)
- [run_anndata_export](#run_anndata_export) (1 entry)
- [Module level](#module-level) (1 entry)

## _read_frame

### lines 526-530

```python
return with_object_type(frame, single_table), (single_table,)
```

The frame does not know which table it is; this function does, and `obs_names` is built from it. Without the stamp a per-table export of `nucleus` and one of `pathogen` index the same object two ways when the labels overlap -- and they always overlap, because each mask is labelled from 1 independently.

### lines 545-546

```python
from ..io import _read_and_join_tables
```

Imported here, not at module scope: spacr.io pulls torch and cellpose, and this module must stay importable on a machine that cannot segment.

### lines 549-553

```python
frame = _read_and_join_tables(
```

`drop_redundant_identity=False` is documented to KEEP the join's suffixed identity copies. The reader now collapses agreeing duplicates by default (instruction 79), which would have made that option a no-op the columns were gone before this module ever saw them. Passed through so the documented choice is the one that happens.

### lines 561-562

```python
return with_object_type(frame, "cell"), tuple(wanted)
```

One row per CELL -- the join is anchored there and the children arrive as columns, not rows -- so that is the type of every observation.

## _attach_png_labels

### lines 630-633

```python
keys = object_keys(png, timelapse=timelapse, object_type=anchor)
```

`id_column` is the anchor's own id column, so these crops are anchor-typed. Both sides of the reindex below have to be keyed the same way or every attached label lands as NaN -- which is silent, and costs exactly the annotation columns this function exists for.

### lines 636-640

```python
return frame, []
```

A timelapse database whose png_list still spells the timepoint `time_id` (see spacr.schema.TIME_COLUMN_ALIASES) cannot be keyed against a `timeID` object table without guessing which frame a crop belongs to. Attaching nothing loses the labels; attaching the wrong frame's label loses the experiment.

## _label_columns

### lines 708-710

```python
guessed = []
```

A database with no png_list, or an unreadable one. The export is still correct without the annotation hint; losing the whole export over a missing optional table would not be.

## _build_obs

### lines 892-894

```python
categorical = list(OBJECT_KEY_COLUMNS[:-1]) + [
```

Low-cardinality text is stored categorical: it is what scanpy's groupby/plotting expects, and on a million-object export it is the difference between a 40 MB obs and a 4 MB one.

## _apply_nan_policy

### lines 934-938

```python
"n_objects_counted": int(matrix.shape[0]),
```

The shape `n_missing` was counted over. Every count in this report is measured on the matrix as the policy received it, so the shape of that matrix has to be recorded with them: divide `n_missing` by the *written* shape instead and a dropping policy reports more than 100% missing.

### lines 967-968

```python
mask = missing
```

The two imputing policies. The mask is what keeps an imputed matrix distinguishable from a measured one.

### line 973, trailing  _(unsure)_

```python
else:
```

NAN_MEAN

### lines 975-977

```python
warnings.simplefilter("ignore", category=RuntimeWarning)
```

An all-NaN column has no mean; numpy says so and returns NaN, which is then filled with 0.0 below. The warning is expected here and would be noise on every export of a sparse feature.

## _align_embedding

### lines 1037-1040

```python
wanted = [k if k in indexed.index else untyped_object_key(k)
```

An embedding computed before object types existed, or by a caller that did not state one, keys its rows untyped. It still names the same objects, so it is resolved by dropping the type rather than reported as a population mismatch.

## build_anndata

### lines 1284-1287

```python
frame, read_tables = _read_frame(
```

`drop_redundant_identity=False` is documented to KEEP the join's suffixed identity copies. The reader collapses agreeing duplicates by default now (instruction 79), which would have made that option a no-op -- the columns were gone before this module saw them.

### lines 1296-1298

```python
if attach_labels and "png_list" in _available_tables(db_path):
```

Before filtering, so a filter may name an annotation column: "export the cells I called infected" is one of the two things this feature is for, and it cannot work if the label arrives after the mask.

### lines 1369-1373

```python
var = var.copy()
```

`var` was built from the frame, i.e. before the policy ran. Keep those counts under `n_missing_raw` -- for an imputing policy they are the only remaining record that a value was invented -- and let `n_missing` describe the matrix that was actually written, which is what a reader inspecting `var` is asking about.

### lines 1451-1455

```python
"artifact": {
```

NOT an artifact id. The id is a hash *of this file's bytes* (see spacr.artifacts._artifact_id), so writing it inside the file would change the bytes it was computed from. What is stored instead is everything needed to look the record up, which is what somebody holding an orphaned .h5ad actually needs.

## export_anndata

### lines 1577-1579

```python
os.makedirs(os.path.dirname(out_path), exist_ok=True)
```

`out_path` was made absolute just above, so its dirname is never empty -- at worst it is "/" -- and there is no falsy case for a guard to catch. `exist_ok` covers the directory already being there.

### lines 1589-1591

```python
return _dataclass_replace(result, path=out_path, artifact_id=artifact_id)
```

`replace` rather than a field-by-field copy: the copy silently dropped whichever field was added to ExportResult last, and a count that arrives as its default is indistinguishable from a real zero.

## run_anndata_export

### lines 1831-1833

```python
tables = [part.strip() for part in tables.split(",") if part.strip()]
```

A settings.csv round trip spells a list as one comma-separated cell; taking it apart here means `--set anndata_tables=cell,nucleus` works and does not silently export a table called "cell,nucleus".

## Module level

### lines 1940-1958

```python
register_anndata_settings()
```

THE QT ROW IS GONE, AND THAT IS THE FOLD.

This module used to register a sidebar tile of its own through `spacr.qt.app.register_app`, with the nine translations of its name, `entry=`, `defaults_module=` and `api_module=` riding along on the same call. An export is the sentence after "measure this plate" rather than a destination, so it is now a button on the Measure masthead (`spacr.qt.screens.measure`) and opens as a page beside the measure settings -- the same generic settings form, drawn from the defaults registered above, with the same Run button running `run_anndata_export`.

Nothing the row carried was dropped with it, only moved to a table that outlives the tile: the Run button's entry point to `spacr.qt.bridge.resolve_pipeline_entry`, the defaults module to `settings_model._FOLDED_DEFAULTS_MODULES`, the API link to `settings_model._APP_API_MODULE`, the header and blurb to `app_screen.APP_TITLES` / `APP_INTROS`, and the translated name to `spacr.qt.i18n`. `spacr-run anndata_export` never went through the row at all and is untouched.

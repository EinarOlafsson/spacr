# Notes from `spacr/filters.py`

Prose lifted out of `spacr/filters.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_png_paths](#_png_paths) (1 entry)
- [_attach_png_paths](#_attach_png_paths) (2 entries)
- [build_filters_frame](#build_filters_frame) (4 entries)
- [build_filters_from_relationships](#build_filters_from_relationships) (1 entry)
- [column_name_for](#column_name_for) (1 entry)
- [export_gate](#export_gate) (1 entry)
- [Module level](#module-level) (2 entries)
- [export_annotation](#export_annotation) (1 entry)

## _png_paths

### lines 381-382

```python
LOG.info("%d %s row(s) have no usable object id; those objects get "
```

Not an error: 'omulti'/'onone'/'error'/NULL are states real crops are in. The object keeps its measurements and simply has no path.

## _attach_png_paths

### lines 449-450

```python
LOG.info("crop paths not attached: png_list's object type is unknown")
```

No type axis to reason with. Matching labels blind is what caused this, so nothing is attached rather than something arbitrary.

### line 465  _(unsure)_

```python
if parent_column in frame.columns and parent_type_column in frame.columns:
```

The child side: join the child's PARENT label onto the crop's label.

## build_filters_frame

### lines 505-507

```python
LOG.info("could not build %s from %s; falling back to the object "
```

The relationships route is the intended one; this fallback keeps a database whose relationships cannot be built (an unreadable table, a schema nobody anticipated) gateable rather than blocked.

### lines 528-530

```python
keys = key_columns(frames[anchor])
```

The anchor decides the key set. A table with fewer identity columns is merged on what it shares, rather than being dropped for lacking a column the others happen to have.

### lines 557-562

```python
crop_type = png_crop_type(db_path)
```

This frame has no object_type axis -- it is one row per

(field, object_label) with `in_<table>` flags -- so a row that is NOT of the cropped type must not keep a path matched on the label alone. `in_cell = 0` with a cell crop attached is exactly the mismatch `_attach_png_paths` exists to prevent; here the flag is the type axis.

### lines 566-569

```python
out["png_path"] = out["png_path"].astype(object)
```

Pandas 3 may infer a nullable string dtype during the merge.  The public table distinguishes an absent crop as Python ``None``, so own an object-typed result before clearing paths for rows of another object type.

## build_filters_from_relationships

### lines 712-715

```python
for table in object_tables(db_path):
```

The `in_<table>` flags say the same thing `object_type` does, and are kept because a merge asks "is this object a nucleus" as a column test far more often than as a string comparison. Derived here rather than stored twice in the relationships table itself.

## column_name_for

### lines 775-776  _(unsure)_

```python
cleaned = f"g_{cleaned}"
```

A leading digit is legal in a quoted SQLite column but trips up every tool that reads the table afterwards, pandas query included.

## export_gate

### lines 838-840

```python
LOG.info("replacing existing filter column %r", column)
```

Re-exporting a gate REPLACES it. The alternative -- refusing, or suffixing -- leaves the user with filters_2, filters_3 and no way to tell which one is the gate currently on screen.

## Module level

### lines 891-893  _(unsure)_

```python
ANNOTATION_MODES: Tuple[str, ...] = ("binary", "multiclass")
```

Annotating from several gates at once

### lines 1027-1029

```python
ROWID_ALIASES: Tuple[str, ...] = ("_rowid_", "rowid", "oid")
```

Sampling -- the reason the module is laggy on a real dataset

## export_annotation

### lines 1021-1022

```python
write_filters_table(db_path, filters)
```

Unlabelled objects are left blank rather than filled: a multiclass annotation has no zero, and inventing one would create a class.

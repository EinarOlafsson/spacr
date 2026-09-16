# Notes from `spacr/selection.py`

Prose lifted out of `spacr/selection.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_object_prefixes](#_object_prefixes) (1 entry)
- [_key_columns](#_key_columns) (1 entry)
- [_compose](#_compose) (1 entry)
- [RangeFilter.mask](#rangefiltermask) (1 entry)
- [_match](#_match) (3 entries)

## _object_prefixes

### lines 252-255

```python
for value in stated.unique():
```

Validate the vocabulary once per call rather than once per row: an object table has a handful of distinct types and tens of millions of rows, and `object_type_prefix` raising per row would be the slow part of the only function on the lasso's hot path.

## _key_columns

### lines 268-269  _(unsure)_

```python
cols = list(schema.TIMEPOINT_KEY_COLUMNS) + [schema.OBJECT_LABEL_KEY]
```

Insert the timepoint before the object label, matching the order `ObjectTableSchema.row_key_columns(timelapse=True)` uses.

## _compose

### lines 283-287

```python
expected = len(cols) - 1
```

One pass over the composed key tells us whether any component smuggled a separator in: a well-formed key has exactly one per join. Checking here rather than per column keeps the common case — nothing to escape, and the key is byte for byte what it always was — at two extra passes instead of ten.

## RangeFilter.mask

### lines 450-453

```python
keep = values.notna().to_numpy(copy=True)
```

Pandas 3 may expose this boolean array as a read-only view.  The bounds below deliberately refine it in place, so ask pandas for an owned, writable buffer rather than relying on version-specific ``to_numpy`` ownership.

## _match

### lines 606-607  _(unsure)_

```python
mask = np.asarray(typed_rows.isin(keys), dtype=bool)
```

`Index.isin` already returns an ndarray — unlike `Series.isin`, which returns a Series. Calling `.to_numpy()` on it raises.

### line 620  _(unsure)_

```python
mask |= np.asarray(plain.isin(loose), dtype=bool)
```

A key naming no type names the object whatever its type.

### lines 623-626

```python
row_untyped = np.asarray(typed_rows) == np.asarray(plain)
```

A typed key still names a row that has not said what it is — but only such a row. Without the `row_untyped` guard a selection of `nucleus1` would light up `pathogen1`, which is the collapse rebuilt one level up.

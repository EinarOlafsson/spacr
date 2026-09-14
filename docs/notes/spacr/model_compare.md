# Notes from `spacr/model_compare.py`

Prose lifted out of `spacr/model_compare.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [object_overlap](#object_overlap) (2 entries)
- [adjusted_rand_index](#adjusted_rand_index) (2 entries)
- [_split_merge](#_split_merge) (1 entry)
- [compare_masks](#compare_masks) (2 entries)
- [load_fields](#load_fields) (2 entries)
- [compare_models](#compare_models) (1 entry)

## object_overlap

### lines 759-761  _(unsure)_

```python
def object_overlap(mask_a: Any, mask_b: Any) -> Dict[str, Any]:
```

the metric layer: label arrays in, numbers out

### line 791  _(unsure)_

```python
row0 = 1 if (n_rows and values_a[0] == 0) else 0
```

Drop the background row/column, keeping the marginals of what is left.

## adjusted_rand_index

### line 837  _(unsure)_

```python
return 1.0
```

Both masks empty: the two models made the same statement.

### lines 850-851  _(unsure)_

```python
return 1.0
```

Both partitions are structureless (every object one pixel): there is nothing to agree or disagree about, so this is agreement.

## _split_merge

### line 961  _(unsure)_

```python
with np.errstate(divide='ignore', invalid='ignore'):
```

in_a[i, j]: the share of B object j that lies inside A object i.

## compare_masks

### lines 1032-1035

```python
explained_a = events['split_parents'] | events['merged_children']
```

An object left over by the assignment is only a genuine difference when no split or merge already accounts for it: the pieces of a shattered A object are not new detections, and the A object they came from is not a missed one.

### line 1041, trailing  _(unsure)_

```python
matched_fraction = 1.0
```

both empty: trivially in agreement

## load_fields

### lines 1179-1182

```python
read = list(_read_field_file(path, filename,
```

Only the read is forgiven: one corrupt field must not cost the comparison the other two. A bad ``channel`` is a caller error and is raised below, outside the guard, so it cannot be swallowed and re-reported as "this folder has no images in it".

### lines 1186-1190

```python
LOG.warning("model comparison is skipping %s: it could not be "
```

Say which one. The comparison is then computed and drawn over fewer fields than the user asked for, and neither the figure nor the report carries "2 of 3" anywhere — so without this the only evidence that a field was dropped is that the caller counts the images and notices.

## compare_models

### lines 1290-1291

```python
import dataclasses
```

Two sides called the same thing make every message ambiguous. Copies, so renaming them never reaches back into the caller's own objects.

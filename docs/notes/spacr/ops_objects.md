# Notes from `spacr/ops_objects.py`

Prose lifted out of `spacr/ops_objects.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [objects_in_window](#objects_in_window) (1 entry)
- [number](#number) (1 entry)

## objects_in_window

### line 196  _(unsure)_

```python
touches = []
```

Touching an interior seam means this window cut the object short.

## number

### lines 342-343  _(unsure)_

```python
chosen.append((max(complete, key=lambda one: one.area), len(group)))
```

The largest complete observation: all of them saw the whole object, so they differ only by segmentation noise at its border.

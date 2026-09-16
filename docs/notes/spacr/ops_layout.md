# Notes from `spacr/ops_layout.py`

Prose lifted out of `spacr/ops_layout.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [WellLayout.pairs](#welllayoutpairs) (1 entry)
- [round_well_layout](#round_well_layout) (1 entry)

## WellLayout.pairs

### lines 224-225

```python
for name in ("down", "right"):
```

DOWN AND RIGHT ONLY, which is what makes each pair appear once AND puts the tiles in geometric order at the same time.

## round_well_layout

### lines 293-295

```python
step = 0.05
```

A circle of radius r spans 2r + 1 columns, so the radius is bounded by the count itself; step finely enough that no integer span is skipped between one radius and the next.

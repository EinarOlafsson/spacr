# Notes from `spacr/image_stitch.py`

Prose lifted out of `spacr/image_stitch.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [read_stage_positions](#read_stage_positions) (1 entry)
- [_sliding_ncc](#_sliding_ncc) (1 entry)
- [_pair_shift](#_pair_shift) (1 entry)
- [_read_tiles](#_read_tiles) (1 entry)
- [plan_mosaic](#plan_mosaic) (2 entries)
- [_mosaic_by_correlation](#_mosaic_by_correlation) (1 entry)
- [stitch_tiles](#stitch_tiles) (1 entry)

## read_stage_positions

### lines 186-188  _(unsure)_

```python
def read_stage_positions(paths: Sequence) -> Optional[List[Tuple[float, float]]]:
```

1. The file's own stage coordinates

## _sliding_ncc

### lines 303-305

```python
total_a = sum_a[width] - sum_a[displacements]
```

A contributes its RIGHT-hand columns and B its LEFT-hand ones: that is what "B sits d to the right" means, and getting it the other way round scores every pair against the wrong half of itself.

## _pair_shift

### lines 407-409

```python
found, score = (0, 0), 0.0
```

Exactly, at full resolution, around the coarse winner: the sample of lines cannot see a shift of three of them, and the placement is what this number becomes.

## _read_tiles

### lines 442-444

```python
return None
```

TILES OF TWO SIZES ARE NOT A GRID. A mosaic of mixed shapes needs per-tile placement, which needs stage coordinates; without them there is nothing to place them by.

## plan_mosaic

### lines 471-473

```python
return Mosaic(placements=(Placement(names[0], 0, 0, 0, 0),),
```

ONE TILE IS NOT A MOSAIC, and calling it a stage placement would claim evidence that was never read. It is placed because there is nowhere else for it to go.

### lines 495-496

```python
for arrangement in ARRANGEMENTS:
```

THE OTHER WAY ROUND IS A DIFFERENT GRID, not a different order: 6 tiles are 2x3 or 3x2 and the squarest shape cannot say which.

## _mosaic_by_correlation

### lines 576-578

```python
origin = (0, 0)
```

Walked out from the first tile along the edges that were believed; anything the walk cannot reach takes the average step, which is the best guess available for a seam that could not be measured.

## stitch_tiles

### lines 649-651

```python
assert mosaic is not None
```

A readable nonempty tile set always receives either a stage-derived, correlated, or explicitly assumed plan.  Keep that contract loud if a future planner change breaks it instead of disguising it as read failure.

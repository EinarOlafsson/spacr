# Notes from `spacr/align.py`

Prose lifted out of `spacr/align.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_well_ids](#_well_ids) (2 entries)
- [_TileReader.__init__](#_tilereader__init__) (1 entry)
- [_TileReader._open_tiff](#_tilereader_open_tiff) (1 entry)
- [scan_tiles](#scan_tiles) (2 entries)
- [_register_pair](#_register_pair) (3 entries)
- [_sequential_positions](#_sequential_positions) (3 entries)
- [_feather_width](#_feather_width) (1 entry)
- [plan_canvas](#plan_canvas) (1 entry)
- [write_stack](#write_stack) (2 entries)
- [save_coordinates](#save_coordinates) (1 entry)
- [align_folder](#align_folder) (1 entry)

## _well_ids

### lines 249-258

```python
def _well_ids(well: str) -> Tuple[str, str]:
```

Join keys — one definition, in spacr.schema

These used to be eight hand-rolled lines, copied here because spacr.utils imports torch and this module is on a GUI/thumbnail path. The copy did not agree with the original: 'AA01' came back as ('AA01', 'AA01') here and as ('error', 'error') there, so a stitched plate and a measured one disagreed about what a 1536 well is called. spacr.schema is stdlib-only, so there is no longer any reason to have a copy at all.

### lines 273-276

```python
text = str(well)
```

A well with no column at all ('A', ''). The old copy passed those through into both slots; keeping that here means a stitch of an oddly-named folder still produces *a* key rather than raising in the middle of a GUI thumbnail.

## _TileReader.__init__

### lines 761-762  _(unsure)_

```python
self._fields.append(open_merged_field(path, use_cache=False))
```

(H, W, C) — the merged-stack layout crops.MergedField was written for.

## _TileReader._open_tiff

### lines 784-785

```python
return tifffile.imread(path)
```

Compressed / tiled TIFFs cannot be mapped. Reading one tile whole is bounded by the tile, never by the canvas.

## scan_tiles

### lines 1144-1147

```python
if group_by_well:
```

The grid is laid out once per well when the caller groups by well, and once across the folder otherwise. `index` stays globally unique either way -- it keys the reader cache and names tiles in pair results -- so only the position WITHIN a grid is per group.

### line 1175, trailing  _(unsure)_

```python
except Exception as exc:
```

unreadable header

## _register_pair

### lines 1320-1323

```python
strip_a = _standardise(strip_a)
```

Zero-mean / unit-variance before the FFT. Removing DC is standard for phase correlation (the DC bin otherwise dominates the normalisation), and it keeps the |F|^2 products inside float32 — raw uint16 intensity over a 2048x205 strip overflows them.

### lines 1328-1340

```python
candidates: List[Tuple[float, float]] = []
```

Two normalisations, because neither is reliable alone.

'phase' whitens the spectrum, which is what makes phase correlation immune to illumination differences between fields — and also what makes it fail on smooth, low-texture images, where whitening amplifies quantisation noise until it outweighs the real signal. On a 64x50 strip of Gaussian-smoothed fluorescence it returns 0 px for a genuine 10 px offset. Plain cross-correlation gets that one right but is pulled around by intensity gradients.

So both are tried and each candidate is *scored* on the pixels it implies; the better score wins. One extra FFT over a strip is a rounding error next to being wrong.

### lines 1362-1363  _(unsure)_

```python
score, scored_px = _score_shift(
```

phase_cross_correlation returns the shift that maps the moving image onto the reference, so b's true position is nominal + it.

## _sequential_positions

### line 1544  _(unsure)_

```python
incoming: Dict[int, List[Tuple[int, float, float]]] = {}
```

incoming[k] = [(j, dy, dx)] meaning "p_k = p_j + (dy, dx)".

### lines 1551-1552

```python
options = [(abs(k - j), j, dy, dx)
```

The nearest already-placed tile in acquisition order: the predecessor a sequential stitcher would chain from.

### line 1557, trailing  _(unsure)_

```python
placed.add(k)
```

nothing to chain from; stage position

## _feather_width

### line 1834  _(unsure)_

```python
widths.append(int(span))
```

_overlap_windows returns None unless both dimensions are positive.

## plan_canvas

### lines 1878-1879

```python
origin_y = math.floor(round(top, 6))
```

Snap away solver noise before flooring: a solved 0.0 that came back as -3e-13 would otherwise cost the canvas a whole row of padding.

## write_stack

### line 2168  _(unsure)_

```python
boxes: List[Tuple[Placement, int, int, float, float]] = []
```

Integer canvas positions, plus the sub-pixel remainder.

### line 2263, trailing

```python
except Exception as exc:
```

stamping must not fail a run

## save_coordinates

### line 2408  _(unsure)_

```python
os.makedirs(parent, exist_ok=True)
```

dirname(abspath(...)) is always an absolute, non-empty directory.

## align_folder

### lines 2607-2609

```python
reference_channel=(
```

None, not 0: an unset setting must let the tiles' own channel stand, or align_folder defeats scan_tiles the same way the old estimate_offsets default did.

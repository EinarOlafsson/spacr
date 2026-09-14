# Notes from `spacr/qt/wand_rescue.py`

Prose lifted out of `spacr/qt/wand_rescue.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [flood_region](#flood_region) (2 entries)
- [trim_directional_runaway](#trim_directional_runaway) (1 entry)
- [taper_region_to_intensity](#taper_region_to_intensity) (1 entry)
- [wand_region](#wand_region) (3 entries)
- [magic_wand](#magic_wand) (1 entry)

## flood_region

### lines 91-93

```python
if values.ndim == 2:
```

Flooding the *distance from the seed* rather than the image itself collapses grey and multi-channel into one code path: the seed sits at distance 0, so a tolerance band around it is exactly the wand's rule.

### line 98  _(unsure)_

```python
return _sk_flood(distance, (int(seed_y), int(seed_x)),
```

connectivity=1 is the four-neighbourhood the BFS wand steps through.

## trim_directional_runaway

### lines 152-154

```python
baseline = float(profile[:i].max(initial=0))
```

The baseline is the established width BEFORE the candidate. Including the candidate would let a leak raise the bar it has to clear and hide itself.

## taper_region_to_intensity

### lines 264-266

```python
yy, xx = np.ogrid[:height, :width]
```

Erosion can erase a small object entirely. The click is trusted, so a few pixels around it are always foreground -- enough to seed the watershed without dictating the shape of its answer.

## wand_region

### lines 338-340

```python
lo, hi = 0.0, float(tolerance)
```

The detector proved this tolerance escapes. Rather than keep the half-plane cut, bisect for the highest tolerance whose whole flood stays put: that boundary is drawn by the image.

### lines 375-377

```python
report.update(rejected=True, capped=True, kept_px=0)
```

Refusing is a real answer: an object truncated at a budget is not the object, and a user who is tuning tolerance needs to see that the budget was the thing that stopped the flood.

### lines 384-386

```python
wide = max(1, int(s["gradient_margin"]))
```

A geodesic cap ends in an arc. Give it an intensity edge too, narrowing the band until the tapered result still fits the budget it was capped to.

## magic_wand

### lines 413-414  _(unsure)_

```python
return mask, {"flooded_px": 0, "kept_px": 0, "cuts": [],
```

Same report shape as a real click, so a caller writing it into a ledger does not have to special-case the nothing-to-do path.

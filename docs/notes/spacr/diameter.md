# Notes from `spacr/diameter.py`

Prose lifted out of `spacr/diameter.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_Source](#_source) (2 entries)
- [_analyse_plane](#_analyse_plane) (6 entries)
- [_aggregate](#_aggregate) (3 entries)
- [estimate_diameters](#estimate_diameters) (2 entries)

## _Source

### line 265, trailing

```python
kind: str = ""
```

'array' | 'raw' | ''

### line 266, trailing  _(unsure)_

```python
where: str = ""
```

human-readable location

## _analyse_plane

### lines 616-617  _(unsure)_

```python
smooth = gaussian_filter(img, 1.0)
```

1. flatten: denoise, then remove illumination with a sigma far larger than any plausible object so the objects survive the subtraction.

### line 622  _(unsure)_

```python
resid = img - smooth
```

2. reject planes whose structure is indistinguishable from pixel noise.

### lines 634-635  _(unsure)_

```python
thr = float(threshold_otsu(flat))
```

3. threshold, fill, label. threshold_otsu only raises on a single-valued image, which the amplitude check above has already rejected.

### lines 638-641

```python
fg_fraction = float(foreground.mean())
```

Otsu always leaves at least the brightest pixel above the threshold and at least the dimmest below it, so there is no degenerate all-or-nothing case to guard here; a field whose split is useless falls out downstream as "no object survived the border and size filters".

### lines 649-664

```python
padded = np.pad(foreground, 1)
```

4. distance-transform cross-check: seed one marker per inscribed-circle maximum and watershed the foreground apart again. Padding by one zero pixel keeps objects at the image edge bounded instead of letting the transform run off the array.

This runs on the UNFILLED foreground on purpose. Hole filling is right for the area measurement above -- a dark nucleus inside a cell is part of the cell -- but in a confluent packing the interstitial background between touching objects is also an enclosed hole, and filling it welds the whole field into one slab whose distance transform knows nothing about individual objects. Those interstices are precisely the signal that tells touching objects apart, so the transform keeps them. The cost is that a genuinely hollow object (a membrane-only ring) reads small here; when it does, the two measurements disagree, and aggregate downgrades the confidence and says so rather than picking a winner silently.

### lines 671-674

```python
r_coarse = float(np.median(edt[tuple(coarse.T)]))
```

Second pass: the coarse peaks set the suppression radius, so the refined pass keeps one seed per object instead of one per ripple. It cannot come back empty -- its threshold is no higher than the coarse pass's, so at least the global maximum survives.

## _aggregate

### lines 753-773

```python
confluent = fg >= fused_fraction
```

Fusion detection, i.e. when to stop believing the plain threshold.

A confluent monolayer merges into one component that touches the border, gets dropped as truncated, and leaves only debris behind -- so `thresh` collapses to nothing, or to a handful of specks that would be reported as a tiny diameter. Both halves of that signature are required here:

the threshold path kept nothing, or kept far fewer objects than the distance transform resolves, AND the field is dense enough for fusion to be the explanation.

Requiring both matters. The count disagreement ALONE is not evidence of fusion: a hollow, membrane-only object is one correct component by area but shatters into dozens of arc-shaped basins under the distance transform, so a ratio test on its own would throw away the right answer (60 px) in favour of the wall thickness (5 px) -- the same silent collapse this code exists to prevent, entered from the other side. A high foreground fraction alone is not evidence either: a dense but well-separated field can reach 30% foreground and still be measured correctly by thresholding. When only one signal fires, the threshold estimate is kept and the confidence is downgraded instead.

### line 807  _(unsure)_

```python
level = _HIGH
```

confidence, and the reasons for every downgrade

### lines 818-821

```python
spread = float((high - low) / diameter) if diameter > 0 else float("inf")
```

Spread is measured 10th-to-90th rather than by the IQR: a field holding two populations (debris plus cells, say) can have a razor-thin IQR around whichever one is more numerous while the reported range spans five-fold. The IQR version scored exactly that case 'high'.

## estimate_diameters

### lines 979-980

```python
per_object: Dict[str, List[_PlaneResult]] = {}
```

Channel-range check up front, so an out-of-range index is reported as a problem rather than raised as an IndexError halfway through a sample.

### line 1017, trailing  _(unsure)_

```python
except Exception as exc:
```

unreadable file, odd dtype

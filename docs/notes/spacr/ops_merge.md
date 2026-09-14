# Notes from `spacr/ops_merge.py`

Prose lifted out of `spacr/ops_merge.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [triangle_descriptors](#triangle_descriptors) (5 entries)
- [_similarity_from_pairs](#_similarity_from_pairs) (2 entries)
- [align_by_triangles](#align_by_triangles) (2 entries)
- [match_cells](#match_cells) (2 entries)

## triangle_descriptors

### line 81  _(unsure)_

```python
sides = np.array([
```

Side lengths, each labelled by the vertex OPPOSITE it.

### line 83, trailing  _(unsure)_

```python
np.linalg.norm(p1 - p2),
```

opposite vertex 0

### line 84, trailing  _(unsure)_

```python
np.linalg.norm(p0 - p2),
```

opposite vertex 1

### line 85, trailing  _(unsure)_

```python
np.linalg.norm(p0 - p1),
```

opposite vertex 2

### line 87, trailing  _(unsure)_

```python
order = np.argsort(sides)
```

shortest side first

## _similarity_from_pairs

### lines 121-123

```python
correction[1, 1] = -1.0
```

A reflection fits the points as well as a rotation but means the sample was flipped, which does not happen between two microscopes looking at the same well.

### lines 126-127  _(unsure)_

```python
scale = float((singular * np.diag(correction)).sum()
```

Umeyama's closed form: the scale is the trace of the singular values (sign-corrected) over the source's variance about its own centroid.

## align_by_triangles

### lines 186-188

```python
scale = float(np.median(scales))
```

The MEDIAN, not the mean: a handful of coincidental shape matches propose transforms that are arbitrarily wrong, and a mean would let one of them drag the answer.

### lines 194-203

```python
src_all = np.concatenate(matched_src, axis=0)
```

THE TRANSLATION IS SOLVED LAST, FROM EVERY CORRESPONDENCE AT ONCE, and this is not a refinement -- it is the difference between working and not. A per-triangle translation is `dst_mean - scale * R @ src_mean` using THAT triangle's own noisy scale and rotation, and any error in them is multiplied by the distance from the origin, which across a 500 px field is large. Measured with 1.5 px of centroid jitter: taking the median of per-triangle translations gave (146, -54) where the truth was (137.5, -92.25) and the scale and angle were already correct to 0.3 %. Re-solving here against the consensus scale and rotation, over every matched vertex, removes that lever entirely.

## match_cells

### line 254, trailing  _(unsure)_

```python
continue
```

not mutual: somebody else is closer

### lines 255-259

```python
if np.linalg.norm(src[i] - dst[j]) > threshold:
```

MEASURED IN FULL PRECISION, NOT FROM THE SEARCH. The search returns float32 distances -- a card's native width -- and a pair sitting exactly on the threshold would then be kept or dropped according to which backend ran. The decision is re-made here on the coordinates themselves so it cannot be.

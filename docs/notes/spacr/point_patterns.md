# Notes from `spacr/point_patterns.py`

Prose lifted out of `spacr/point_patterns.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [ripley_k](#ripley_k) (3 entries)
- [csr_envelope](#csr_envelope) (2 entries)
- [_distance_to_boundary](#_distance_to_boundary) (1 entry)

## ripley_k

### lines 114-116

```python
frame["k"] = np.nan
```

ONE POINT HAS NO PATTERN. NaN rather than zero: zero is a measured absence of clustering and this is the absence of a measurement.

### line 138

```python
counts = tree.query_ball_point(metric[keep], r, return_length=True)
```

`- 1` FOR THE POINT ITSELF, which every ball contains.

### lines 141-142  _(unsure)_

```python
values.append(area * pairs / (float(n) * kept))
```

lambda-hat is n / A from ALL the points -- the density is a property of the window, not of the subset the correction kept.

## csr_envelope

### lines 188-191

```python
chosen = inside[rng.choice(len(inside), size=int(n_points),
```

WITHOUT REPLACEMENT. Two cells cannot share a centroid pixel, and a duplicated point is a pair at distance zero -- which would put clustering into the null itself and raise the envelope at exactly the small radii the observation is being judged at.

### lines 200-202

```python
"lo": np.nanmin(stack, axis=0),
```

MIN AND MAX, NOT A QUANTILE. This is what makes the envelope an exact test at 2 / (simulations + 1); a percentile of a small sample has no such guarantee.

## _distance_to_boundary

### line 245  _(unsure)_

```python
def _distance_to_boundary(window: np.ndarray, scale: np.ndarray) -> np.ndarray:
```

the parts that decide the answer

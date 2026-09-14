# Notes from `spacr/ops_phenotype.py`

Prose lifted out of `spacr/ops_phenotype.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [refine_similarity](#refine_similarity) (1 entry)
- [seed_by_scaled_pairs](#seed_by_scaled_pairs) (3 entries)

## refine_similarity

### lines 308-314

```python
widths = [radius * 4.0, radius * 2.0]
```

WIDE FIRST, THEN TIGHT. The seed is poor by construction, so the first pass has to accept correspondences a long way out or there is nothing to re-solve from; the last pass has to accept only close ones or it re-solves from coincidences. One radius cannot be both, and using the scoring radius throughout refused every noisy case that this schedule now recovers. The final pass is always at `radius`, so what comes back is measured at the radius the caller asked for.

## seed_by_scaled_pairs

### lines 520-524

```python
seed_radius = radius * 4.0
```

A SEED IS SCORED LOOSELY, and this is not the same number the answer is scored with. Two points fix a transform exactly at those two points and approximately everywhere else, and the error grows with the distance from them -- so scoring a seed at the final radius asks it to already be the answer. It only has to be in the right basin.

### lines 526-536

```python
extent = float(max(np.ptp(src[:, 0]), np.ptp(src[:, 1])))
```

AND A SEED NEEDS A BASELINE. A pair of nuclei 10 px apart fixes the rotation to within the centroid noise, which is to say not at all; the same noise on a pair a third of the field apart is a fraction of a degree. Short pairs are also where the separation test stops discriminating, because almost any target pair passes it. `np.ptp(a)` AND NOT `a.ptp()`: NumPy 2.0 removed the ndarray METHOD and kept only the function. Every other `ptp` in this package was already written the surviving way, so this line was the one that raised AttributeError on any NumPy 2 install -- which is every install now, and is why this module measured 52% in the coverage sweep: seven tests could not reach past it.

### lines 553-563

```python
source_angle = np.arctan2(*(src[j] - src[i]))
```

Two points fix the rotation: the angle between the source separation and the target separation.

BOTH ANGLES ARE TAKEN THE SAME WAY ROUND, and it has to be the way the rotation matrix below reads. These are (row, column) points, so a separation is (dy, dx) and the angle that matches `[[cos, -sin], [sin, cos]]` acting on (row, column) is `arctan2(dy, dx)`. Written as `arctan2(dx, dy)` the two angles are each the complement of the right one, so their difference comes out NEGATED -- a seed rotated the wrong way, which the refinement then has to climb out of.

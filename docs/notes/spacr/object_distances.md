# Notes from `spacr/object_distances.py`

Prose lifted out of `spacr/object_distances.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [between_object_types](#between_object_types) (4 entries)
- [maxima_distances](#maxima_distances) (1 entry)

## between_object_types

### lines 256-258

```python
own = _sample(own_interior, centroids)
```

WHERE IT SITS IN ITSELF. The interior transform at the centroid is the centre's distance to its own rim; over the object's deepest point it is a shape-free 0-to-1 position that compares across sizes.

### lines 269-270

```python
edge = np.full(len(labels), np.inf, dtype=float)
```

HOW CLOSE TO THE EDGE OF THE FIELD. An object that touches it is clipped, and every measurement of it is of a fragment.

### lines 290-292

```python
other_labels, other_centroids = _centroids(other_mask)
```

CENTRE TO CENTRE, to the NEAREST object of the other type. The full N x M matrix is neither cheap nor something a one-row-per- object table can hold; the nearest is both.

### line 305  _(unsure)_

```python
overlap = []
```

OVERLAP, which is the answer when the distance is zero.

## maxima_distances

### lines 357-360

```python
for holder in (to_own, to_centre, *to_other.values()):
```

NaN, NOT ZERO. An object with no peak has no distance from one; zero would read as "the peak is right here", which is the opposite of what happened. The count column beside it says why the row is empty.

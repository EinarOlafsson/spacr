# Notes from `spacr/bystanders.py`

Prose lifted out of `spacr/bystanders.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_median_cell_diameter](#_median_cell_diameter) (1 entry)
- [distance_to_infected](#distance_to_infected) (1 entry)
- [neighbourhood](#neighbourhood) (1 entry)
- [compare](#compare) (1 entry)

## _median_cell_diameter

### lines 98-99

```python
pixel_area = float(scale[0]) * float(scale[1])
```

AREA IS A PRODUCT OF BOTH AXES, so the pixel area is the product of the two spacings; the diameter is its square root's companion.

## distance_to_infected

### lines 173-176

```python
from scipy.ndimage import labeled_comprehension
```

THE MINIMUM OVER THE CELL, NOT THE VALUE AT ITS CENTROID. A large or bent cell can have its centroid far from an infected neighbour while its membrane touches one, and it is the membrane that is exposed. `labeled_comprehension` walks each label once.

## neighbourhood

### lines 289-290

```python
take = min(int(k) + 1, len(labels)) if int(k) > 0 else 1
```

+1 BECAUSE THE FIRST HIT IS ALWAYS THE CELL ITSELF. Asking for k and dropping the nearest would silently return k-1 neighbours.

## compare

### lines 376-382

```python
"mean": float(values.mean()) if values.size else np.nan,
```

`ddof=1`, WHICH IS THE POINT OF THIS ITEM. The confound it exists to remove is an inflated control variance, so the spread reported here has to be the sample estimate a reader would compare against -- not the population one, which is smaller and would understate exactly the effect being looked for. NaN at n < 2 rather than 0: one cell has no spread, and 0 would read as a perfectly consistent group.

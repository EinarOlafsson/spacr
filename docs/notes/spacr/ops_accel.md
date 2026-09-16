# Notes from `spacr/ops_accel.py`

Prose lifted out of `spacr/ops_accel.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [nearest_neighbours](#nearest_neighbours) (1 entry)
- [nearest_neighbours.run](#nearest_neighboursrun) (1 entry)

## nearest_neighbours

### lines 254-256  _(unsure)_

```python
def nearest_neighbours(source: np.ndarray, target: np.ndarray, *,
```

The nearest neighbour: matching cells across two acquisitions

## nearest_neighbours.run

### lines 311-321

```python
gaps = torch.cdist(
```

NOT THE DEFAULT compute_mode. `cdist` switches to the ||a||^2 + ||b||^2 - 2ab expansion when the batch is big enough, so chunk=500 and chunk=7 run different arithmetic and the expansion's cancellation error is what differs which is a nearest-neighbour DISTANCE that changes with the working-set size, on a function whose whole contract is that it does not. Measured on 500 x 300 float32 points: default and `use_mm` are chunk-dependent at 4.14e-03 against the exact answer; this one is chunk-independent at 4.77e-07, which is float32's own floor and matches the numpy path.

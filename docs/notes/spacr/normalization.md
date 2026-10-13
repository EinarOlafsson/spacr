# Notes from `spacr/normalization.py`

Prose lifted out of `spacr/normalization.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [normalization_stats](#normalization_stats) (1 entry)
- [dataset_statistics](#dataset_statistics) (3 entries)
- [apply_crop_dtype](#apply_crop_dtype) (1 entry)

## normalization_stats

### lines 137-139

```python
raise ValueError(
```

Named, and NOT silently fallen back to: a run that asked for its own statistics and got ImageNet's would be a run whose model card says one thing and whose weights learned another.

## dataset_statistics

### lines 218-221

```python
source = loader if max_batches is None else islice(loader, max_batches)
```

islice rather than a counter and a break: a `for` pulls the next item BEFORE the body can stop, so a counter reads one batch more than was asked for -- which matters for a loader with side effects, one that reads from disk or advances a shuffle.

### line 229  _(unsure)_

```python
flat = values.transpose(1, 0, 2, 3).reshape(values.shape[1], -1)
```

(N, C, H, W) -> per channel over every pixel of every image.

### lines 243-244  _(unsure)_

```python
variance = np.maximum(total_sq / count - mean ** 2, 0.0)
```

max(0, ...): the identity can go a hair negative on a constant channel through floating-point cancellation, and sqrt of that is nan.

## apply_crop_dtype

### lines 305-307

```python
from .crops import narrow_to_uint8
```

A float crop has no declared range, so the only honest widening is through the same narrowing rule the PNG path uses and back up anything else invents a scale factor.

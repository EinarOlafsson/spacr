# Notes from `spacr/gpu_reduce.py`

Prose lifted out of `spacr/gpu_reduce.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [rapids_available](#rapids_available) (1 entry)
- [make_reducer](#make_reducer) (1 entry)
- [_cpu_estimator](#_cpu_estimator) (1 entry)

## rapids_available

### line 67  _(unsure)_

```python
return False
```

cuML present without a working cupy runtime is not a usable GPU.

## make_reducer

### lines 117-120

```python
LOG.info("cuML could not build a %s estimator; using the CPU "
```

A cuML that imports but cannot build the estimator -- a version skew, a CUDA mismatch -- falls back rather than taking the run down. The whole promise of an extra is that its absence, or its misbehaviour, costs nothing.

## _cpu_estimator

### lines 147-149

```python
from .utils import umap
```

The package-level ``umap`` import eagerly reaches parametric UMAP and TensorFlow. spaCR's lazy proxy loads only ``umap.umap_``, which is the CPU implementation this reducer actually needs.

# Notes from `spacr/measure_hooks.py`

Prose lifted out of `spacr/measure_hooks.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [RegionContext.centroids](#regioncontextcentroids) (1 entry)
- [_default_name](#_default_name) (1 entry)
- [_register](#_register) (1 entry)

## RegionContext.centroids

### lines 294-297

```python
from scipy.ndimage import center_of_mass
```

scipy is imported here rather than at module scope: this module is on the import path of anything that merely wants to *register* a hook, and most filters never ask for a centroid at all.

## _default_name

### line 351  _(unsure)_

```python
return base
```

Free, or already this exact callable: re-registering is idempotent.

## _register

### lines 374-375

```python
registry.pop(key, None)
```

Replace rather than append: an extension that re-installs itself on every GUI run must not end up applying its correction twice.

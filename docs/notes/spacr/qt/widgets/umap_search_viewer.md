# Notes from `spacr/qt/widgets/umap_search_viewer.py`

Prose lifted out of `spacr/qt/widgets/umap_search_viewer.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## axis_frame

### line 184  _(unsure)_

```python
high = np.where(np.isclose(high, low), low + 1.0, high)
```

Degenerate dimensions still receive a visible axis of finite length.

### lines 196-197  _(unsure)_

```python
for fraction in fractions:
```

A readable base-plane grid: X/Y in both 2D and 3D. The third axis rises from the same origin in 3D and rotates with the map.

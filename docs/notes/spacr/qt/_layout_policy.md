# Notes from `spacr/qt/_layout_policy.py`

Prose lifted out of `spacr/qt/_layout_policy.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## minimum_width_for

### lines 110-112

```python
return measured[max(measured)]
```

Above every measured scale: the largest requirement there is, which is the honest extrapolation -- it is a floor, and a floor from the widest thing measured is better than one invented above it.

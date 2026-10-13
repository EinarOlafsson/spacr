# Notes from `spacr/ops_compose.py`

Prose lifted out of `spacr/ops_compose.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## compose_window

### line 148

```python
if t_bottom <= window.top or t_top >= window.bottom:
```

Does this tile touch the window at all? Most do not.

### line 158  _(unsure)_

```python
wy0, wx0 = max(t_top, window.top), max(t_left, window.left)
```

The intersection, in window coordinates and in tile coordinates.

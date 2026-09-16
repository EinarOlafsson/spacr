# Notes from `spacr/style_base.py`

Prose lifted out of `spacr/style_base.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## apply_page

### lines 193-198

```python
wanted = bool(style.grid) and str(style.grid_axis) != "none"
```

THE GRID IS OFF WHEN IT IS OFF. matplotlib warns -- "First parameter to grid() is false, but line properties are supplied. The grid will be enabled." -- and then enables it, which is how a "draw a grid" tick box drew one whichever way it was set. The same fault was found and fixed in the save dialog; it is spelled once here so a third renderer cannot meet it again.

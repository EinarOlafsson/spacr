# Notes from `spacr/gate_library.py`

Prose lifted out of `spacr/gate_library.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## save

### lines 128-129

```python
temporary = target + ".part"
```

Written whole and then moved, so an interrupted save leaves the previous strategy intact rather than a truncated file the library still lists.

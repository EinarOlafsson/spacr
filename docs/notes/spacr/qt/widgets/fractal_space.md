# Notes from `spacr/qt/widgets/fractal_space.py`

Prose lifted out of `spacr/qt/widgets/fractal_space.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## SpaceEngine.__init__

### lines 474-475

```python
pass
```

A thread cap that cannot be set is a slower frame, never a reason for the backdrop not to draw.

# Notes from `spacr/qt/counting_tool.py`

Prose lifted out of `spacr/qt/counting_tool.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## CountingPanel._on_layers_changed

### lines 279-280  _(unsure)_

```python
"""Refresh the tally when the markers change.
```

Derived, not stored: a marker removed through the layer list changes the number here too.

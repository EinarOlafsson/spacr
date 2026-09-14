# Notes from `spacr/qt/gil_priority.py`

Prose lifted out of `spacr/qt/gil_priority.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## claim

### lines 53-54  _(unsure)_

```python
LOG.debug("could not lower the switch interval", exc_info=True)
```

An interpreter without the knob is not a reason to fail a run; it is a reason for the window to be less smooth.

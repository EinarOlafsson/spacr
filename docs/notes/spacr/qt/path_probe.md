# Notes from `spacr/qt/path_probe.py`

Prose lifted out of `spacr/qt/path_probe.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## _worker

### line 87

```python
probes.answered.emit(path, answer)
```

Queued to the GUI thread by Qt, because this is a worker.

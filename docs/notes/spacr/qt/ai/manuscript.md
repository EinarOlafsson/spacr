# Notes from `spacr/qt/ai/manuscript.py`

Prose lifted out of `spacr/qt/ai/manuscript.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## availability

### lines 102-104

```python
candidates = ()
```

The advice above this is still worth printing. A traceback here would replace a paragraph of help with nothing, and the run's own numbers do not depend on any provider being reachable.

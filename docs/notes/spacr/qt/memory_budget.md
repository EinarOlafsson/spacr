# Notes from `spacr/qt/memory_budget.py`

Prose lifted out of `spacr/qt/memory_budget.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## what_to_drop

### lines 152-153  _(unsure)_

```python
for key, size, _used in sorted(kept, key=lambda row: row[2]):
```

Least recently used first, which is the one least likely to be wanted next.

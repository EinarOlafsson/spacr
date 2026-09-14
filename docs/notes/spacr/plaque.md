# Notes from `spacr/plaque.py`

Prose lifted out of `spacr/plaque.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## detect_wells

### lines 218-220

```python
LOG.warning(
```

Reported, not silently dropped: a rejected well is a condition missing from the results, and a user who is not told will read that as "no plaques grew".

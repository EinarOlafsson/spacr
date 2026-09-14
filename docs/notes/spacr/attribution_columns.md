# Notes from `spacr/attribution_columns.py`

Prose lifted out of `spacr/attribution_columns.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## write

### lines 123-125

```python
cursor.execute(
```

SQLite has no ADD COLUMN IF NOT EXISTS, and re-running a write must not be an error -- an attribution is something a user redoes with a different threshold.

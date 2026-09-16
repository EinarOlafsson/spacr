# Notes from `spacr/qt/ingest_preview.py`

Prose lifted out of `spacr/qt/ingest_preview.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (1 entry)
- [plan_folder_extraction](#plan_folder_extraction) (1 entry)

## Module level

### line 33  _(unsure)_

```python
ROW_COLUMNS = ("original", "plate", "well", "field", "channel", "time", "canonical")
```

The columns every preview row carries, in table order.

## plan_folder_extraction

### lines 132-135

```python
files = (sorted(files) if limit is None
```

``nsmallest`` is ``sorted(...)[:limit]`` without ever holding the whole tree in memory: it keeps ``limit`` candidates and drops the rest as it goes. The previous line materialised every path under the folder — for a 100 000-image plate, to then take 200 of them.

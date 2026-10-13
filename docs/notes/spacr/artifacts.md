# Notes from `spacr/artifacts.py`

Prose lifted out of `spacr/artifacts.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (1 entry)
- [Registry._connect](#registry_connect) (1 entry)

## Module level

### line 167  _(unsure)_

```python
CAUSE_UNKNOWN = "unknown-artifact"
```

Staleness cause codes.

## Registry._connect

### lines 512-514

```python
pass
```

SQLite kept the old journal mode -- an older file, or a filesystem that will not do shared memory. DELETE mode is slower under contention but always correct.

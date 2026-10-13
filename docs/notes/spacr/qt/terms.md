# Notes from `spacr/qt/terms.py`

Prose lifted out of `spacr/qt/terms.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [agreed_version](#agreed_version) (1 entry)
- [record_agreement](#record_agreement) (1 entry)
- [register_translations](#register_translations) (2 entries)

## agreed_version

### lines 325-327

```python
return ""
```

A PROFILE THAT CANNOT BE READ HAS NOT AGREED. Answering "yes" when the store is unreachable would turn a broken settings file into a silent acceptance.

## record_agreement

### lines 369-371

```python
store.sync()
```

WRITTEN THROUGH IMMEDIATELY. QSettings flushes lazily, and an acceptance still sitting in a buffer when the process is killed is an acceptance the user gave and would be asked for again.

## register_translations

### lines 492-493

```python
return 0
```

A SCREEN WITH NO CATALOG IS STILL A SCREEN. Every caption falls back to the English it was written in.

### lines 500-501

```python
continue
```

A row that does not fit the catalog's shape is skipped rather than allowed to stop the rest of the screen being catalogued.

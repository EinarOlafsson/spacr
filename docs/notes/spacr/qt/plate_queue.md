# Notes from `spacr/qt/plate_queue.py`

Prose lifted out of `spacr/qt/plate_queue.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [import_plates_from_csv](#import_plates_from_csv) (2 entries)
- [Module level](#module-level) (1 entry)

## import_plates_from_csv

### line 313  _(unsure)_

```python
vv: Any = v
```

Try numeric coercion; fall back to raw string.

### line 322  _(unsure)_

```python
if v.lower() in ("true", "yes"):
```

Preserve booleans + None-ish tokens

## Module level

### lines 334-336  _(unsure)_

```python
RunnerFn = Callable[[QueueItem], None]
```

Runner — pure Python, injectable for testing

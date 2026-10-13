# Notes from `spacr/qt/laptop_mode.py`

Prose lifted out of `spacr/qt/laptop_mode.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [total_memory_gib](#total_memory_gib) (1 entry)
- [apply](#apply) (1 entry)

## total_memory_gib

### lines 56-58

```python
return None
```

os.sysconf is absent on Windows and refuses unknown names elsewhere. None means "could not be read", which every caller already treats as "do not decide laptop mode on memory".

## apply

### line 205  _(unsure)_

```python
os.environ.pop(_NO_BACKDROP, None)
```

Only ever our own suppression -- see `_suppressed_here`.

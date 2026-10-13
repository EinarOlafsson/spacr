# Notes from `spacr/columns.py`

Prose lifted out of `spacr/columns.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [ColumnNotFound.__str__](#columnnotfound__str__) (1 entry)
- [headers](#headers) (1 entry)

## ColumnNotFound.__str__

### line 71, trailing  _(unsure)_

```python
def __str__(self) -> str:
```

trivial

## headers

### lines 103-105

```python
continue
```

A file that is not a CSV, or is empty, or is being written. Not fatal: the other paths may answer the question, and the caller is about to be told which files it could read.

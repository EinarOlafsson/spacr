# Notes from `spacr/qc_quarantine.py`

Prose lifted out of `spacr/qc_quarantine.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_who](#_who) (1 entry)
- [_read_record](#_read_record) (1 entry)
- [_move_without_overwrite](#_move_without_overwrite) (1 entry)

## _who

### line 125, trailing

```python
except Exception:
```

a platform account lookup failure is not an error here

## _read_record

### lines 138-139

```python
return {"prior_record_error": f"{type(exc).__name__}: {exc}"}
```

A damaged old ledger must not make restoration impossible.  Keep the fact that it was damaged in the replacement audit record.

## _move_without_overwrite

### lines 182-187

```python
try:
```

A plain POSIX rename silently REPLACES a destination created between the check above and the syscall.  Link-then-unlink is the stdlib's atomic no-replace move for sibling regular files: link fails when the name already exists and both names address the same bytes until the source is removed.  The bounded copy fallback covers filesystems that do not support hard links while keeping O_EXCL's no-overwrite promise.

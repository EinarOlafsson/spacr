# Notes from `spacr/checkpoint.py`

Prose lifted out of `spacr/checkpoint.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [json_safe](#json_safe) (1 entry)
- [CheckpointStore.mark](#checkpointstoremark) (1 entry)

## json_safe

### line 91  _(unsure)_

```python
item = getattr(value, "item", None)
```

NumPy scalar types expose item(); using it keeps this module NumPy-free.

## CheckpointStore.mark

### lines 348-350

```python
cancellation_checkpoint()
```

The unit is now durable.  This is the earliest safe point at which a GUI Stop request may leave the workflow without losing or half-writing that unit.

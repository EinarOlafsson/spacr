# Notes from `spacr/frame_handoff.py`

Prose lifted out of `spacr/frame_handoff.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [held](#held) (1 entry)
- [stage](#stage) (1 entry)

## held

### lines 111-112  _(unsure)_

```python
return None
```

A caller that passed something with no path at all asked nothing of this module; it gets the same answer as a path nobody offered.

## stage

### lines 218-220

```python
if temporary_path is not None:
```

A failed stage did not publish the durable half of this contract. Roll back this offer exactly: a failed replacement must not erase a producer's pre-existing, successfully published handoff.

# Notes from `spacr/well_spec.py`

Prose lifted out of `spacr/well_spec.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (2 entries)
- [row_label](#row_label) (1 entry)

## Module level

### lines 23-27

```python
"positive_control_wells", "negative_control_wells",
```

The three control blocks (221). They were added to WELL_ONLY_SETTINGS which grants the plate button -- without being added here, and the audit caught it: `WELL_ONLY_SETTINGS` must be a SUBSET of this, since a setting cannot be "entirely wells" without being "may contain wells". That is the check working, not a formality.

### lines 39-40  _(unsure)_

```python
"positive_control_wells", "negative_control_wells",
```

Control-block settings contain only wells, so the plate-map picker can replace their complete value without discarding mixed metadata.

## row_label

### lines 135-136

```python
return "A" + string.ascii_uppercase[row - 27]
```

A 1536 has 32 rows, so the last six are AA..AF. Two letters is as far as any real plate goes.

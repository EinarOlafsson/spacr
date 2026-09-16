# Notes from `spacr/figures/summary.py`

Prose lifted out of `spacr/figures/summary.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## summarise

### line 104

```python
if called_and_big.any():
```

WHICH ONES. A count without names is not something anyone can act on.

### lines 115-116

```python
condition = None
```

DID THE ASSAY WORK. A screen whose controls do not separate has not measured anything, however many hits the correction reports.

### lines 145-146

```python
if p is not None:
```

CALIBRATION. On a real screen this is routinely off, and which way it is off changes whether the hit count is an over- or an undercount.

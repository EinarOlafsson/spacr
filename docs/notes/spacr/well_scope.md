# Notes from `spacr/well_scope.py`

Prose lifted out of `spacr/well_scope.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## select

### lines 127-130

```python
report["note"] = ("no gRNA was chosen, so there is no population to "
```

NOT AN ERROR AND NOT SILENTLY EVERYTHING. With no guide column or no selection there is no "the chosen guides", so the honest answer is an empty population and a report that says why -- drawing the whole table instead would look like a selection nobody made.

### line 142  _(unsure)_

```python
well = _column(frame, WELL_COLUMNS)
```

scope == "wells"

### line 155

```python
out[MATE_COLUMN] = [CHOSEN if flag else MATE
```

DISTINGUISHABLE ON THE PLOT, which is the whole point of this scope.

# Notes from `spacr/outlier_filter.py`

Prose lifted out of `spacr/outlier_filter.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [column_for](#column_for) (1 entry)
- [apply](#apply) (1 entry)

## column_for

### lines 56-58

```python
if "intensity" in str(criterion):
```

An intensity column carries its channel in its name, and the channel is the user's -- so accept the first that matches the shape rather than insisting on channel 1.

## apply

### lines 144-146

```python
report.append({"criterion": criterion, "caption": caption,
```

NOT SILENT. A filter the user switched on that found no column removed nothing, and a run that says nothing about it looks exactly like one where the filter worked and found nothing.

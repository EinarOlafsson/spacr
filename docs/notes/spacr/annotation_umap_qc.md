# Notes from `spacr/annotation_umap_qc.py`

Prose lifted out of `spacr/annotation_umap_qc.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [fit_on_controls](#fit_on_controls) (1 entry)
- [neighbour_purity](#neighbour_purity) (1 entry)

## fit_on_controls

### lines 109-112

```python
level = str(group_by) if groups is not None else "cell"
```

ONE SPLITTER FOR THE WHOLE PACKAGE. The grouped splitter refuses a design it cannot hold apart rather than falling back to a random one, which is the difference between an honest error and an optimistic score.

## neighbour_purity

### line 201  _(unsure)_

```python
neighbours = neighbours[neighbours != row][:wanted]
```

Drop self, which is only present for a control cell.

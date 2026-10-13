# Notes from `spacr/ops_solve.py`

Prose lifted out of `spacr/ops_solve.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## solve_placements

### lines 57-58  _(unsure)_

```python
parent = list(range(count))
```

Components first, so each gets exactly one pin. Without that the normal equations are singular for every component after the first.

# Notes from `spacr/localisation.py`

Prose lifted out of `spacr/localisation.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## table

### lines 78-79

```python
if gene_text.endswith(".0"):
```

`gene_nr` reads as a float when the column has a blank in it, so 244480 arrives as "244480.0" and joins to nothing at all.

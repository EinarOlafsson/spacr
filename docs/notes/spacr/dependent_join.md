# Notes from `spacr/dependent_join.py`

Prose lifted out of `spacr/dependent_join.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## join

### lines 170-173

```python
mine_frame = objects if _key(objects, columns) is not None \
```

THE OBJECT SIDE KEEPS ITS OWN COLUMNS WHEN IT HAS THEM. Only the dependent table is the one with something missing; making both sides go through the path would break a join that the ID columns would have made.

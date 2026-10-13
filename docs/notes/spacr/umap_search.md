# Notes from `spacr/umap_search.py`

Prose lifted out of `spacr/umap_search.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## walk_recipes

### lines 406-407  _(unsure)_

```python
seen, unique = set(), []
```

Deduplicated, keeping order: a walk that scores the same recipe twice spends the time and reports a second row that adds nothing.

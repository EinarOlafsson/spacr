# Notes from `spacr/example_data.py`

Prose lifted out of `spacr/example_data.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## fetch

### lines 249-251

```python
by_kind: Dict[str, List[str]] = {"counts": [], "scores": []}
```

REPORTED FOR THE REQUESTED KIND ONLY. Listing a path for a file that was never asked for -- and so may not be on disk -- would hand the caller a name it cannot open.

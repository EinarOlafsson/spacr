# Notes from `spacr/cli_workspace.py`

Prose lifted out of `spacr/cli_workspace.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## main

### lines 57-59

```python
print(f"{run_dir} carries no {DOC_NAME} — it was saved with "
```

NAMED, not "not found". A run saved with the feature off is a different thing from a run whose bundle failed to write, and the user's next step differs.

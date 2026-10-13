# Notes from `spacr/fit_resources.py`

Prose lifted out of `spacr/fit_resources.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Module level

### lines 51-55

```python
_PERFORMANCE_LOG_ENV = "SPACR_PERFORMANCE_LOG"
```

Resource accounting deliberately lives behind private names until the Qt preference and every worker entry point have one settled integration seam. Keeping these names private also means adding the recorder does not silently expand the translated public API.  The persisted schema, not a Python class, is the interface users and support tooling consume.

# Notes from `spacr/qt/maturity.py`

Prose lifted out of `spacr/qt/maturity.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## apply

### lines 299-304

```python
stages.pop(app_key, None)
```

Stable is the ABSENCE of a line, not a line reading "stable". ``APP_STAGE`` exists to record what is *not* signed off, and signing an app off is deleting its entry — writing the word in would give the table a second way to say the same thing, which `test_every_app_has_a_stage_and_it_is_written_down_once` exists to prevent.

### lines 317-319

```python
for app_key in unassessed_apps(stages, keys):
```

Phase 2 — the default, made explicit. `stages.get(key)` and not `key in stages` because a table that somehow holds an empty string or a None for a key has not said anything about it either.

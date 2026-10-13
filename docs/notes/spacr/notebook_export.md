# Notes from `spacr/notebook_export.py`

Prose lifted out of `spacr/notebook_export.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (1 entry)
- [export_run](#export_run) (1 entry)

## Module level

### lines 84-86  _(unsure)_

```python
_ENTRYPOINTS: Dict[str, str] = {
```

Entrypoint scaffolds — per pipeline app

## export_run

### line 192, trailing  _(unsure)_

```python
_read_settings(run_dir)
```

Validate that the recorded settings exist and parse.

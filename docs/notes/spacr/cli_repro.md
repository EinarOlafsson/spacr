# Notes from `spacr/cli_repro.py`

Prose lifted out of `spacr/cli_repro.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## main

### line 99  _(unsure)_

```python
candidate = runs_root() / args.run_dir
```

Try under runs_root() by basename


---

# Notes from `spacr/cli_repro.py`

Prose lifted out of `spacr/cli_repro.py` by `tools/extract_source_notes.py`.
Ordinary comments move here; tool directives and published attribute documentation stay in the module. The path mirrors the source path, which is how its reasons are found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## main

### lines 394-395

```python
with _open_sans_is_the_default(), open_run(app_key, settings) as run:
```

291: a replay is a pipeline run, so its figures are in Open Sans like the run it replays.

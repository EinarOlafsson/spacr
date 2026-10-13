# Notes from `spacr/qt/ask_for_the_path.py`

Prose lifted out of `spacr/qt/ask_for_the_path.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [somebody_is_there](#somebody_is_there) (1 entry)
- [ask_for_a_folder](#ask_for_a_folder) (2 entries)
- [ask_for_a_database_column](#ask_for_a_database_column) (3 entries)

## somebody_is_there

### line 68  _(unsure)_

```python
if os.environ.get("QT_QPA_PLATFORM", "").startswith("offscreen"):
```

An offscreen platform is a test or a render farm, not a person.

## ask_for_a_folder

### lines 106-107

```python
if chooser is None:
```

THE PRODUCTION PATH. `chooser` is injected by tests and by nothing else, so this is what the application always takes.

### line 124

```python
tried = complaint
```

Rejected IN the dialog rather than accepted and failed afterwards.

## ask_for_a_database_column

### lines 240-241

```python
if chooser is None:
```

THE PRODUCTION PATH. `chooser` is injected by tests and by nothing else, so this is what the application always takes.

### lines 268-269

```python
complaint = (f"{os.path.basename(database)} holds no tables. "
```

Rejected IN the form: a file that is not a database, or one with no tables, is the same failure one step later.

### line 276

```python
complaint = tried
```

Back out to the database rather than abandoning the form.

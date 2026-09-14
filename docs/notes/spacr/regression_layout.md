# Notes from `spacr/regression_layout.py`

Prose lifted out of `spacr/regression_layout.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (1 entry)
- [normalise_count_table_layout](#normalise_count_table_layout) (1 entry)

## Module level

### lines 20-21

```python
DEFAULT_ID_COLUMNS = (
```

Columns that describe the observation rather than one independent variable.  Callers can add project-specific columns through ``id_columns``.

## normalise_count_table_layout

### lines 222-224

```python
if (guide_column == "grna" and "grna" not in frame.columns
```

``process_reads`` has accepted the historical downloadable-data header ``grna_name`` for years.  Canonicalise it before layout inference so an otherwise valid long table is not misdiagnosed as a malformed wide one.

# Notes from `spacr/ports.py`

Prose lifted out of `spacr/ports.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Module level

### lines 433-437

```python
register_module_ports(ModulePorts(
```

THE MERGED CLASSIFY IS BOTH CLASSIFIERS, so it consumes and produces what either of them did. Without this the Core chain reads measure -> (nothing) -> regression: `chained_app_keys` is the registry intersected with the modules declared here, and the screen that took over from `classify` and `ml_analyze` declared nothing.

### lines 442-446

```python
consumes=(
```

THE UNION OF BOTH HALVES, because the screen fits either family. The image classifier needs `png_list` to find its crops; the gradient-boosting one needs only the feature table, so requiring png_list here would report the screen as blocked on a project where it can perfectly well run.

### lines 456-458

```python
register_module_ports(ModulePorts(
```

The timelapse module *is* the mask pipeline with tracking on — spacr.core.preprocess_generate_masks_timelapse calls preprocess_generate_masks — so it has the mask pipeline's ports.

### lines 464-467

```python
for _db_app in sorted(DB_APPS):
```

Every app spacr.validate already knows opens <src>/measurements/measurements.db gets a declaration derived from that fact rather than from invention: enough to answer "is there a database to read?", and no claim about outputs nobody has verified.

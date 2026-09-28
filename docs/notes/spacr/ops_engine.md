# Notes from `spacr/ops_engine.py`

Prose lifted out of `spacr/ops_engine.py` by `tools/extract_source_notes.py`.
Ordinary comments move here; tool directives and published attribute documentation stay in the module. The path mirrors the source path, which is how its reasons are found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (1 entry)
- [_cellpose_model](#_cellpose_model) (1 entry)
- [_load_library](#_load_library) (1 entry)
- [_decode](#_decode) (1 entry)

## Module level

### lines 31-33

```python
resource = None
```

Windows has no `resource`. The module still has to import there: the only thing it is used for is the peak-memory line in the report, and a missing number is not a reason to refuse to run the pipeline.

## _cellpose_model

### lines 1063-1070

```python
name = str(settings.get("cellpose_model") or "").strip()
```

372 PART 14-L built the model with no ``pretrained_model`` at all, so it ran the library's default -- ``cpsam_v2`` in the installed release, not the ``cpsam`` the OPS settings name. Passing that name explicitly would load different weights from the ones the well was validated with, so the default name means the default model. Another name goes in as a keyword rather than as a dict entry: the settings-flow analyser reads a string subscript as a settings key, and ``kwargs["pretrained_model"]`` published a setting nobody can set.

## _load_library

### lines 1500-1502

```python
import csv
```

The standard-library reader, not a DataFrame: a guide library is a list of sequences rather than a measurement table, so it has no column the tabular funnel's canonicalisation is there to repair.

## _decode

### lines 1681-1682  _(unsure)_

```python
context = multiprocessing.get_context("spawn")
```

A card shared between processes is a tenant nobody announced, so the fields decoded in worker processes stay on the CPU.

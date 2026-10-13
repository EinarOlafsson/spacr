# Notes from `spacr/training_basis.py`

Prose lifted out of `spacr/training_basis.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (3 entries)
- [resolve_basis](#resolve_basis) (1 entry)

## Module level

### lines 62-64

```python
"png_type": "path_string",
```

`png_type` was never a type. It is a substring that has to appear in a crop's path, so it is now called that. The old name is accepted because it is in every settings CSV a user has.

### lines 66-68

```python
"size": "image_size",
```

`size` and `image_size` were the same number under two names, set by different helpers in the same module. image_size wins: it says what it measures.

### lines 78-79  _(unsure)_

```python
"class_metadata", "metadata_rules",
```

`metadata_type_by` is gone: it named the column a class is defined by, which is the Classes editor's own column field.

## resolve_basis

### lines 113-115

```python
return RETIRED_BASES[basis]
```

A settings file older than the removal. It loads and runs; it does not raise, and it does not silently mean something else the mapping is recorded above with the reason.

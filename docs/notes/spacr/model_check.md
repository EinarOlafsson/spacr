# Notes from `spacr/model_check.py`

Prose lifted out of `spacr/model_check.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [resolve_model_source](#resolve_model_source) (1 entry)
- [_load_custom](#_load_custom) (1 entry)
- [check_model](#check_model) (2 entries)

## resolve_model_source

### lines 78-79

```python
LOG.info("custom_model_path %r does not exist; using model_type", path)
```

Named rather than ignored: a path that is set and missing is a mistake, not a preference for the built-in model.

## _load_custom

### lines 124-126

```python
if any(k in loaded for k in ("state_dict", "model_state_dict")):
```

A state dict is weights with no architecture. It can be loaded INTO a model but is not one, and the difference is worth saying plainly rather than failing later with a missing-attribute error.

## check_model

### lines 204-207

```python
wanted_classes = None
```

``class_names`` raises ClassDefinitionError (a ValueError) when the Classes editor has written an incomplete rule.  This checker is a click-time diagnostic, so that user error belongs in its report, not on the Qt event loop as an exception.

### lines 250-252

```python
problems.append(
```

The failure that silently half-works: a two-class head on a three-class problem trains happily and is wrong about every object of the third class.

# Notes from `spacr/classify.py`

Prose lifted out of `spacr/classify.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (2 entries)
- [classify](#classify) (4 entries)

## Module level

### lines 29-32

```python
ML_MODEL_TYPES: Tuple[str, ...] = (
```

Keep this list beside the family dispatcher: it is the boundary at which a merged settings payload becomes an ML run.  Rejecting a CV backbone here is both cheaper and more accurate than letting it survive database loading and fail deep inside ``ml_analysis``.

### lines 52-55

```python
"gradient_accumulation_steps",
```

`gradient_accumulation` retired 2026-09-09 (364): the step count alone says whether to accumulate, and `steps = 1` IS the off position. A greying table naming a key that no longer exists greys nothing.

## classify

### lines 201-209

```python
family = resolve_family(settings)
```

Two translations, both idempotent and both in one place: the shared vocabulary (names) and the class definition (what the names select). Anything downstream reads the current shape only. Resolve family-owned values before shared-vocabulary normalization. ``training_basis.normalize_settings`` deliberately treats model_type_ml as a legacy alias for model_type.  A merged payload has both keys, though, and the CV value wins that generic alias operation. Capturing the ML value here prevents maxvit_t (or any other CV backbone) from being sent to the classical estimator pipeline.

### lines 215-217

```python
try:
```

FlowView is optional observability.  Its complete setup boundary is failure-isolated so neither a renderer fault nor malformed trace state can replace a pipeline result or exception.

### lines 226-229

```python
from .crop_source import validate as validate_crops
```

Refuse a crop source that cannot produce images BEFORE training starts. Discovering that extract_channels was never set after an hour of dataset building is a worse failure than one at the door, and the message names the setting to change.

### lines 235-237

```python
resolved["model_type_ml"] = ml_model_type
```

The ML pipeline reads only its family-owned spelling.  Assignment, rather than setdefault, is intentional: ``model_type`` may contain the simultaneously visible CV choice in a merged settings file.

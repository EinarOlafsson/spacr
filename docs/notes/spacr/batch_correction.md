# Notes from `spacr/batch_correction.py`

Prose lifted out of `spacr/batch_correction.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_as_controls](#_as_controls) (1 entry)
- [_combat](#_combat) (2 entries)
- [correct_batch_effects](#correct_batch_effects) (2 entries)

## _as_controls

### lines 113-115

```python
out: List[Any] = []
```

The same rule one level in: a LIST holding one comma-separated string is what a settings CSV round-trip produces, and it fails exactly the same way.

## _combat

### line 435, trailing  _(unsure)_

```python
values = numeric.to_numpy(dtype=float).T
```

(features, rows)

### lines 491-494

```python
scale_floor = _COMBAT_MIN_VAR * np.maximum(
```

Relative, not `<= 0`. A column that is exactly constant comes out of `lstsq` with a residual around 1e-30 rather than 0, and dividing by its square root turns rounding noise into a feature with unit variance a dead channel that arrives in the corrected table looking alive.

## correct_batch_effects

### lines 651-654

```python
if _is_no_covariate(covariate):
```

Asked before anything is computed, and asked unconditionally: a single-batch frame short-circuits to a no-op below, and answering only for the runs that happen to have two plates would let the question go unasked in exactly the run that gets rerun on more data.

### lines 750-753

```python
try:
```

SAY WHAT IS ACTUALLY THERE. "matched nothing" with no sight of the column is a message that sends the user to the wrong place; the commonest cause is a control name that is not one of the values the column holds.

# Notes from `spacr/qt/widgets/outlier_model.py`

Prose lifted out of `spacr/qt/widgets/outlier_model.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (1 entry)
- [median_absolute_deviation](#median_absolute_deviation) (1 entry)
- [_reasons_per_feature](#_reasons_per_feature) (1 entry)

## Module level

### lines 288-290  _(unsure)_

```python
MAD_TO_SIGMA = 1.4826022185056018
```

The constants of the robust scale estimators

## median_absolute_deviation

### lines 372-374  _(unsure)_

```python
def median_absolute_deviation(values: Sequence[float]) -> float:
```

The robust estimators, on their own so they can be tested on a hand vector

## _reasons_per_feature

### lines 1234-1236

```python
if offending.size == 0:
```

A scan's own flags come from these scores, so this only trips for a caller that supplied its own. Such a row gets no sentence rather than an invented one.

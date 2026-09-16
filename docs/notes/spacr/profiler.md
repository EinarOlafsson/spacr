# Notes from `spacr/profiler.py`

Prose lifted out of `spacr/profiler.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [predict](#predict) (2 entries)
- [reference_row](#reference_row) (2 entries)
- [_sweep](#_sweep) (1 entry)

## predict

### lines 97-99  _(unsure)_

```python
def predict(model: Any, exog: pd.DataFrame, *,
```

One prediction call for seventeen backends

### lines 142-145

```python
pass
```

A statsmodels results object whose design does not line up raises from deep inside patsy; fall through to the linear predictor, which aligns by column name and can say what is missing.

## reference_row

### lines 344-346  _(unsure)_

```python
def reference_row(design: pd.DataFrame, *, method: str = "median",
```

Where the other inputs are held

### lines 375-376

```python
for name in row.index:
```

An intercept column is 1 by construction; holding it at its median is right by accident and at zero is wrong on purpose, so it is pinned.

## _sweep

### lines 403-406

```python
centre = low if math.isfinite(low) else 0.0
```

A column with a single observed value has no range to sweep. Widen it symmetrically rather than returning one point: "what if this were different" is the question, and a constant column is exactly when nobody knows the answer.

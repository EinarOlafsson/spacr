# Notes from `spacr/nonparametric_fits.py`

Prose lifted out of `spacr/nonparametric_fits.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Curve](#curve) (1 entry)
- [smooth](#smooth) (1 entry)
- [Agreement](#agreement) (1 entry)
- [agreement](#agreement) (1 entry)
- [Module level](#module-level) (1 entry)
- [spline_design](#spline_design) (1 entry)
- [report_agreement](#report_agreement) (1 entry)

## Curve

### lines 168-170

```python
@dataclass
```

B. A DIAGNOSTIC LAID OVER THE DATA

## smooth

### line 274

```python
return Curve(method, grid, mean, lower=mean - 1.96 * sd,
```

THE BAND IS THE POINT of choosing a Gaussian process at all.

## Agreement

### lines 283-285

```python
@dataclass
```

C. AN AGREEMENT CHECK

## agreement

### lines 380-382

```python
importance = permutation_importance(model, values, y, n_repeats=5,
```

PERMUTATION importance, not the tree's own impurity importance: the latter is biased toward columns with many distinct values, and a guide-abundance column has far more than a rare guide's does.

## Module level

### lines 410-412

```python
SPLINE_KNOTS = 4
```

A. A FIT THAT ANSWERS IN THE SAME CURRENCY

## spline_design

### lines 448-449

```python
continue
```

Too few distinct values to bend through; leave it linear rather than manufacturing a basis out of nothing.

## report_agreement

### line 510  _(unsure)_

```python
columns = {}
```

The design's guide columns carry the same names, so the two line up.
